"""
Step 7.4: Fixed 200 Test Set Evaluation (Layer 2)

This script is fully driven by `config/config.yaml`.

Workflow:
1. Fine-tune each enabled local model on the full ~780 SFT dataset
2. Run inference on the fixed 200-sample test set
3. Evaluate overall / per-class / pillar-level metrics
4. Save JSON, CSV, and Markdown reports

Run:
  uv run src/fine-tuning/run_fixed_test.py
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
REASONING_DIR = PROJECT_ROOT / "src" / "step5_reasoning"

for candidate in [PROJECT_ROOT, SCRIPT_DIR, REASONING_DIR]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


from run_cv_finetune import (  # noqa: E402
    LABELS,
    StopOnLabel,
    extract_label,
    oversample_dataset,
    setup_base_model,
    setup_model,
)
from reason import (  # noqa: E402
    CATEGORY_DEFINITIONS,
    CLASSIFY_SYSTEM_PROMPT,
    CLASSIFY_USER_TEMPLATE,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logging.getLogger("transformers").setLevel(logging.ERROR)


CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
LOCAL_MODELS = ["gemma", "llama", "qwen", "ministral"]
API_MODELS = ["gpt", "claude", "gemini"]
SUPPORTED_MODELS = LOCAL_MODELS + ["finbert"] + API_MODELS
INVALID_LABEL = "__INVALID__"
PILLARS = ["Environmental", "Social", "Governance", "Non-ESG"]
PILLAR_MAP = {
    "Climate Change": "Environmental",
    "Natural Capital": "Environmental",
    "Pollution & Waste": "Environmental",
    "Human Capital": "Social",
    "Product Liability": "Social",
    "Community Relations": "Social",
    "Corporate Governance": "Governance",
    "Business Ethics & Values": "Governance",
    "Non-ESG": "Non-ESG",
}


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {CONFIG_PATH}. "
            "Create config/config.yaml before running Step 7.4."
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    fixed_test_cfg = raw_config.get("step7", {}).get("fixed_test")
    if not isinstance(fixed_test_cfg, dict):
        raise ValueError("Missing `step7.fixed_test` section in config/config.yaml")

    models_cfg = fixed_test_cfg.get("models")
    if not isinstance(models_cfg, dict) or not models_cfg:
        raise ValueError("Missing `step7.fixed_test.models` in config/config.yaml")

    enabled_models = [
        model_key
        for model_key, model_cfg in models_cfg.items()
        if isinstance(model_cfg, dict) and model_cfg.get("enabled", False)
    ]
    if not enabled_models:
        raise ValueError("No enabled models found in `step7.fixed_test.models`.")

    unsupported = sorted(set(enabled_models) - set(SUPPORTED_MODELS))
    if unsupported:
        raise ValueError(f"Unsupported models in config: {unsupported}")

    return fixed_test_cfg


def format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if hasattr(value, "item"):
        try:
            scalar = value.item()
        except Exception:  # pragma: no cover - defensive fallback
            scalar = None
        if isinstance(scalar, float):
            return f"{scalar:.4f}"
        if scalar is not None:
            return str(scalar)
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_markdown(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        header = "| " + " | ".join(fieldnames) + " |"
        separator = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
        handle.write(header + "\n")
        handle.write(separator + "\n")
        for row in rows:
            values = [format_metric(row.get(field, "")) for field in fieldnames]
            handle.write("| " + " | ".join(values) + " |\n")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_test_messages(item: dict[str, Any], model_type: str) -> list[dict[str, Any]]:
    user_content = CLASSIFY_USER_TEMPLATE.format(
        definitions=CATEGORY_DEFINITIONS,
        text=item["combined_text"],
    )

    if model_type == "llama":
        return [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": CLASSIFY_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_content}],
        },
    ]


def train_full(
    model_type: str,
    dataset_sft,
    training_cfg: dict[str, Any],
    models_dir: Path,
) -> Path:
    logger.info("========== Training %s on full dataset ==========", model_type)
    logger.info("Original train size: %s", len(dataset_sft))

    train_ds = dataset_sft
    if training_cfg.get("oversample", True):
        train_ds = oversample_dataset(train_ds)
        logger.info("Oversampled train size: %s", len(train_ds))

    model, tokenizer, formatting_func, kwargs, api_class = setup_model(model_type)

    if api_class == "FastLanguageModel" and model_type == "llama":
        from unsloth.chat_templates import standardize_sharegpt

        train_ds = standardize_sharegpt(train_ds)

    adapter_dir = models_dir / model_type / "adapter"
    checkpoint_dir = models_dir / model_type / "checkpoints"

    if api_class in ["FastModel", "FastLanguageModel"]:
        train_ds = train_ds.map(formatting_func, batched=True)
        from transformers import DataCollatorForSeq2Seq
        from trl import SFTConfig, SFTTrainer

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            dataset_text_field="text",
            max_seq_length=training_cfg.get("max_seq_length", 2048),
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
            if model_type == "llama"
            else None,
            packing=False,
            args=SFTConfig(
                per_device_train_batch_size=training_cfg.get(
                    "per_device_train_batch_size", 4
                ),
                gradient_accumulation_steps=training_cfg.get(
                    "gradient_accumulation_steps", 4
                ),
                warmup_steps=training_cfg.get("warmup_steps", 5),
                num_train_epochs=training_cfg.get("num_train_epochs", 3),
                learning_rate=training_cfg.get("learning_rate", 2e-4),
                logging_steps=training_cfg.get("logging_steps", 1),
                optim=training_cfg.get("optim", "adamw_8bit"),
                weight_decay=training_cfg.get("weight_decay", 0.001),
                lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
                seed=training_cfg.get("seed", 3407),
                output_dir=str(checkpoint_dir),
                report_to="none",
            ),
        )

        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part=kwargs["instruction_part"],
            response_part=kwargs["response_part"],
        )
    else:
        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTConfig, SFTTrainer

        FastVisionModel.for_training(model)
        converted_dataset = [formatting_func(sample) for sample in train_ds]
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=converted_dataset,
            args=SFTConfig(
                per_device_train_batch_size=training_cfg.get(
                    "per_device_train_batch_size", 4
                ),
                gradient_accumulation_steps=training_cfg.get(
                    "gradient_accumulation_steps", 4
                ),
                warmup_steps=training_cfg.get("warmup_steps", 5),
                num_train_epochs=training_cfg.get("num_train_epochs", 3),
                learning_rate=training_cfg.get("learning_rate", 2e-4),
                logging_steps=training_cfg.get("logging_steps", 1),
                optim=training_cfg.get("optim", "adamw_8bit"),
                weight_decay=training_cfg.get("weight_decay", 0.001),
                lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
                seed=training_cfg.get("seed", 3407),
                output_dir=str(checkpoint_dir),
                report_to="none",
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_length=training_cfg.get("max_seq_length", 2048),
            ),
        )

    logger.info("Training %s...", model_type)
    trainer.train()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Saved adapter to %s", adapter_dir)

    del model, tokenizer, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return adapter_dir


def run_fixed_test_inference(
    model_type: str,
    adapter_dir: Path,
    test_data: list[dict[str, Any]],
    inference_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    import tqdm
    from peft import PeftModel
    from transformers import StoppingCriteriaList

    logger.info("Loading %s adapter from %s", model_type, adapter_dir)
    model, tokenizer, api_class = setup_base_model(model_type)
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    if api_class == "FastLanguageModel":
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
    elif api_class == "FastModel":
        from unsloth import FastModel

        FastModel.for_inference(model)
    else:
        from unsloth import FastVisionModel

        FastVisionModel.for_inference(model)

    stop_criteria = StoppingCriteriaList([StopOnLabel(tokenizer)])
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    results = []
    for item in tqdm.tqdm(test_data, desc=f"Fixed test inference ({model_type})"):
        messages = build_test_messages(item, model_type)

        if model_type == "llama":
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(tokenized, torch.Tensor):
                input_ids = tokenized.to("cuda")
                attention_mask = torch.ones_like(input_ids)
            else:
                input_ids = tokenized["input_ids"].to("cuda")
                attention_mask = tokenized.get(
                    "attention_mask",
                    torch.ones_like(input_ids),
                ).to("cuda")
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=inference_cfg.get("max_new_tokens", 256),
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=True,
                pad_token_id=pad_id,
                stopping_criteria=stop_criteria,
            )

        input_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        ).strip()

        results.append(
            {
                "paragraph_id": item["paragraph_id"],
                "ticker": item.get("ticker"),
                "filing_date": item.get("filing_date"),
                "raw_output": decoded,
                "parsed_label": extract_label(decoded),
                "ground_truth_label": item["label"],
            }
        )

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return results


def run_finbert_baseline(test_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for item in test_data:
        results.append(
            {
                "paragraph_id": item["paragraph_id"],
                "ticker": item.get("ticker"),
                "filing_date": item.get("filing_date"),
                "raw_output": item.get("finbert_label", ""),
                "parsed_label": item.get("finbert_label"),
                "ground_truth_label": item["label"],
            }
        )
    return results


def run_api_inference(
    model_type: str,
    test_data: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    inference_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    import tqdm

    from src.step7_cv.classify_with_llms import MODELS as API_MODEL_REGISTRY
    from src.step7_cv.classify_with_llms import normalize_label

    sdk_imports = {
        "gpt": "openai",
        "claude": "anthropic",
        "gemini": "google.generativeai",
    }

    registry = API_MODEL_REGISTRY[model_type]
    api_key = os.getenv(registry["api_key_env"])
    if not api_key:
        raise EnvironmentError(
            f"Missing environment variable: {registry['api_key_env']} for model `{model_type}`."
        )

    module_name = sdk_imports.get(model_type)
    if module_name is not None:
        try:
            __import__(module_name)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Missing Python package `{module_name}` required for model `{model_type}`. "
                "Install dependencies with `uv sync` before running Step 7.4."
            ) from exc

    caller = registry["caller"]
    model_name = model_cfg.get("api_model_name", registry["model_str"])
    retry_attempts = inference_cfg.get("api_retry_attempts", 3)
    retry_sleep = inference_cfg.get("api_retry_sleep_seconds", 2.0)
    delay_seconds = inference_cfg.get("api_delay_seconds", 0.5)

    logger.info("Running API model %s (%s)", model_type, model_name)

    results = []
    for item in tqdm.tqdm(test_data, desc=f"Fixed test inference ({model_type})"):
        last_error = None
        raw_output = ""
        for attempt in range(1, retry_attempts + 1):
            try:
                raw_output = caller(
                    item["combined_text"],
                    api_key=api_key,
                    model=model_name,
                )
                break
            except ModuleNotFoundError:
                raise
            except Exception as exc:  # pragma: no cover - network path
                last_error = str(exc)
                logger.warning(
                    "API error for %s on attempt %s/%s: %s",
                    item["paragraph_id"],
                    attempt,
                    retry_attempts,
                    exc,
                )
                if attempt < retry_attempts:
                    time.sleep(retry_sleep)
        parsed_label = normalize_label(raw_output) if raw_output else None
        results.append(
            {
                "paragraph_id": item["paragraph_id"],
                "ticker": item.get("ticker"),
                "filing_date": item.get("filing_date"),
                "raw_output": raw_output,
                "parsed_label": parsed_label,
                "ground_truth_label": item["label"],
                "error": last_error,
            }
        )
        time.sleep(delay_seconds)

    return results


def compute_metrics(
    results: list[dict[str, Any]],
    model_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        precision_recall_fscore_support,
    )

    y_true = [item["ground_truth_label"] for item in results]
    y_pred = [item["parsed_label"] if item["parsed_label"] in LABELS else INVALID_LABEL for item in results]

    accuracy = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="micro",
        zero_division=0,
    )
    kappa = cohen_kappa_score(
        y_true,
        y_pred,
        labels=LABELS + [INVALID_LABEL],
    )

    invalid_count = sum(label == INVALID_LABEL for label in y_pred)
    per_class_p, per_class_r, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            average=None,
            zero_division=0,
        )
    )

    y_true_pillar = [PILLAR_MAP.get(label, INVALID_LABEL) for label in y_true]
    y_pred_pillar = [PILLAR_MAP.get(label, INVALID_LABEL) for label in y_pred]
    pillar_p, pillar_r, pillar_f1, pillar_support = precision_recall_fscore_support(
        y_true_pillar,
        y_pred_pillar,
        labels=PILLARS,
        average=None,
        zero_division=0,
    )

    summary = {
        "Model": model_name,
        "Samples": len(results),
        "Accuracy": float(round(accuracy, 4)),
        "Macro P": float(round(macro_p, 4)),
        "Macro R": float(round(macro_r, 4)),
        "Macro F1": float(round(macro_f1, 4)),
        "Micro P": float(round(micro_p, 4)),
        "Micro R": float(round(micro_r, 4)),
        "Micro F1": float(round(micro_f1, 4)),
        "Kappa": float(round(kappa, 4)),
        "Invalid Count": invalid_count,
        "Invalid Rate": float(round(invalid_count / len(results), 4)),
    }

    per_class_rows = []
    for idx, label in enumerate(LABELS):
        per_class_rows.append(
            {
                "Model": model_name,
                "Category": label,
                "Precision": float(round(per_class_p[idx], 4)),
                "Recall": float(round(per_class_r[idx], 4)),
                "F1": float(round(per_class_f1[idx], 4)),
                "Support": int(per_class_support[idx]),
            }
        )

    pillar_rows = []
    for idx, pillar in enumerate(PILLARS):
        pillar_rows.append(
            {
                "Model": model_name,
                "Pillar": pillar,
                "Precision": float(round(pillar_p[idx], 4)),
                "Recall": float(round(pillar_r[idx], 4)),
                "F1": float(round(pillar_f1[idx], 4)),
                "Support": int(pillar_support[idx]),
            }
        )

    return summary, per_class_rows, pillar_rows


def report_metrics(summary: dict[str, Any]) -> None:
    logger.info("============================================================")
    logger.info("  %s - Fixed Test Set Results", summary["Model"])
    logger.info("============================================================")
    logger.info("  Samples:       %s", summary["Samples"])
    logger.info("  Accuracy:      %.4f", summary["Accuracy"])
    logger.info("  Macro F1:      %.4f", summary["Macro F1"])
    logger.info("  Micro F1:      %.4f", summary["Micro F1"])
    logger.info("  Cohen's Kappa: %.4f", summary["Kappa"])
    logger.info("  Invalid Count: %s", summary["Invalid Count"])


def save_reports(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    pillar_rows: list[dict[str, Any]],
) -> None:
    save_json(output_dir / "summary_metrics.json", summaries)

    overall_fields = [
        "Model",
        "Samples",
        "Accuracy",
        "Macro P",
        "Macro R",
        "Macro F1",
        "Micro P",
        "Micro R",
        "Micro F1",
        "Kappa",
        "Invalid Count",
        "Invalid Rate",
    ]
    per_class_fields = ["Model", "Category", "Precision", "Recall", "F1", "Support"]
    pillar_fields = ["Model", "Pillar", "Precision", "Recall", "F1", "Support"]

    write_csv(output_dir / "fixed_test_overall_metrics.csv", summaries, overall_fields)
    write_markdown(
        output_dir / "fixed_test_overall_metrics.md",
        summaries,
        overall_fields,
    )
    write_csv(
        output_dir / "fixed_test_per_class_metrics.csv",
        per_class_rows,
        per_class_fields,
    )
    write_markdown(
        output_dir / "fixed_test_per_class_metrics.md",
        per_class_rows,
        per_class_fields,
    )
    write_csv(
        output_dir / "fixed_test_pillar_metrics.csv",
        pillar_rows,
        pillar_fields,
    )
    write_markdown(
        output_dir / "fixed_test_pillar_metrics.md",
        pillar_rows,
        pillar_fields,
    )


def main() -> None:
    config = load_config()

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    inference_cfg = config.get("inference", {})
    execution_cfg = config.get("execution", {})
    models_cfg = config.get("models", {})

    train_sft_path = resolve_path(data_cfg["train_sft_path"])
    test_labeled_path = resolve_path(data_cfg["test_labeled_path"])
    output_dir = resolve_path(data_cfg["results_dir"])
    models_dir = resolve_path(data_cfg["models_dir"])

    if not test_labeled_path.exists():
        raise FileNotFoundError(f"Fixed test file not found: {test_labeled_path}")

    with test_labeled_path.open("r", encoding="utf-8") as handle:
        test_data = json.load(handle)

    logger.info("Loaded %s fixed-test samples from %s", len(test_data), test_labeled_path)
    logger.info("Test-set label distribution: %s", dict(Counter(item["label"] for item in test_data)))

    enabled_models = [
        model_key
        for model_key, model_cfg in models_cfg.items()
        if model_cfg.get("enabled", False)
    ]
    skip_train = execution_cfg.get("skip_train", False)
    overwrite_results = execution_cfg.get("overwrite_results", False)
    continue_on_error = execution_cfg.get("continue_on_error", True)

    dataset_sft = None
    if any(model_key in LOCAL_MODELS for model_key in enabled_models):
        if not train_sft_path.exists():
            raise FileNotFoundError(f"SFT training file not found: {train_sft_path}")
        dataset_sft = load_dataset("json", data_files=str(train_sft_path), split="train")
        logger.info("Loaded %s SFT samples from %s", len(dataset_sft), train_sft_path)

    summaries: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    pillar_rows: list[dict[str, Any]] = []

    manifest = {
        "config_path": str(CONFIG_PATH),
        "train_sft_path": str(train_sft_path),
        "test_labeled_path": str(test_labeled_path),
        "results_dir": str(output_dir),
        "models_dir": str(models_dir),
        "enabled_models": enabled_models,
        "skip_train": skip_train,
        "overwrite_results": overwrite_results,
        "test_label_distribution": dict(Counter(item["label"] for item in test_data)),
    }
    save_json(output_dir / "run_manifest.json", manifest)

    for model_key in enabled_models:
        result_path = output_dir / f"{model_key}_results.json"
        logger.info("Starting model: %s", model_key)

        try:
            if result_path.exists() and not overwrite_results:
                logger.info("Reusing cached results: %s", result_path)
                with result_path.open("r", encoding="utf-8") as handle:
                    results = json.load(handle)
            elif model_key == "finbert":
                results = run_finbert_baseline(test_data)
                save_json(result_path, results)
            elif model_key in LOCAL_MODELS:
                adapter_dir = models_dir / model_key / "adapter"
                if not skip_train:
                    if dataset_sft is None:
                        raise RuntimeError("Local-model training requested but SFT data was not loaded.")
                    adapter_dir = train_full(
                        model_type=model_key,
                        dataset_sft=dataset_sft,
                        training_cfg=training_cfg,
                        models_dir=models_dir,
                    )
                elif not adapter_dir.exists():
                    raise FileNotFoundError(
                        f"Adapter not found at {adapter_dir}. "
                        "Set `skip_train: false` or provide a trained adapter."
                    )

                results = run_fixed_test_inference(
                    model_type=model_key,
                    adapter_dir=adapter_dir,
                    test_data=test_data,
                    inference_cfg=inference_cfg,
                )
                save_json(result_path, results)
            elif model_key in API_MODELS:
                results = run_api_inference(
                    model_type=model_key,
                    test_data=test_data,
                    model_cfg=models_cfg[model_key],
                    inference_cfg=inference_cfg,
                )
                save_json(result_path, results)
            else:
                raise ValueError(f"Unsupported model: {model_key}")

            summary, model_per_class, model_pillar = compute_metrics(results, model_key)
            report_metrics(summary)
            summaries.append(summary)
            per_class_rows.extend(model_per_class)
            pillar_rows.extend(model_pillar)
        except Exception as exc:
            logger.exception("Failed while processing %s: %s", model_key, exc)
            if not continue_on_error:
                raise

    if not summaries:
        raise RuntimeError("No fixed-test results were produced.")

    save_reports(output_dir, summaries, per_class_rows, pillar_rows)
    logger.info("Saved fixed-test reports to %s", output_dir)


if __name__ == "__main__":
    main()
