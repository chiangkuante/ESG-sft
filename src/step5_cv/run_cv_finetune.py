from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import multiprocessing as mp
import re
import statistics
import sys
from pathlib import Path

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

import torch
from datasets import load_dataset
from transformers import StoppingCriteria, StoppingCriteriaList

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step5_cv.common import (
    LABELS,
    LABELS_SORTED,
    build_eval_messages,
    compute_overall_metrics,
    extract_label,
    load_json,
    load_step5_cv_config,
    resolve_path,
    save_json,
    write_csv,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_NAME_DEFAULTS = {
    "gemma": "unsloth/gemma-4-E4B-it",
    "gemma26b": "unsloth/Gemma-4-26B-A4B-it",
    "llama": "unsloth/Llama-3.2-3B-Instruct",
    "qwen": "unsloth/Qwen3.5-4B",
    "ministral": "unsloth/Ministral-3-3B-Instruct-2512",
}

REASONING_AND_LABEL_INSTRUCTION = (
    "- Return the final answer using exactly these two lines:\n"
    "  Reasoning: <brief explanation>\n"
    "  Label: <one valid category name>"
)
LABEL_ONLY_INSTRUCTION = "- Return the final answer using exactly one line:\n  Label: <one valid category name>"
SYSTEM_REASONING_SENTENCE = "Your job is to determine the single best label among the 9 ESG categories and explain the reasoning clearly."
SYSTEM_LABEL_ONLY_SENTENCE = "Your job is to determine the single best label among the 9 ESG categories and output only the final label."


class StopOnLabel(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        label_group = "|".join(re.escape(label) for label in LABELS_SORTED)
        self.pattern = re.compile(rf"(?is)label\s*:\s*(?:{label_group})")

    def __call__(self, input_ids, scores, **kwargs):
        for seq in input_ids:
            tail = self.tokenizer.decode(seq[-120:], skip_special_tokens=True)
            if self.pattern.search(tail) is None:
                return False
        return True


class Gemma4TextCollator:
    """Inject mm_token_type_ids for Gemma 4 text-only SFT."""

    def __init__(self, tokenizer):
        from transformers import DataCollatorForSeq2Seq

        base_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        self.inner = DataCollatorForSeq2Seq(tokenizer=base_tokenizer)

    def __call__(self, features):
        batch = self.inner(features)
        if "mm_token_type_ids" not in batch:
            batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch


def get_model_name(model_type: str, finetune_cfg: dict) -> str:
    model_registry = finetune_cfg.get("model_registry", {})
    configured = model_registry.get(model_type)
    if configured:
        return str(configured)
    return MODEL_NAME_DEFAULTS[model_type]


def apply_model_overrides(finetune_cfg: dict, model_type: str) -> dict:
    overrides = finetune_cfg.get("model_overrides", {})
    if not isinstance(overrides, dict):
        return finetune_cfg
    model_override = overrides.get(model_type)
    if not isinstance(model_override, dict):
        return finetune_cfg

    merged = dict(finetune_cfg)
    if "training" in model_override and isinstance(model_override["training"], dict):
        merged["training"] = {**finetune_cfg.get("training", {}), **model_override["training"]}
    if "runtime" in model_override and isinstance(model_override["runtime"], dict):
        merged["runtime"] = {**finetune_cfg.get("runtime", {}), **model_override["runtime"]}
    if "inference" in model_override and isinstance(model_override["inference"], dict):
        merged["inference"] = {**finetune_cfg.get("inference", {}), **model_override["inference"]}
    return merged


def resolve_train_batch_size(model_type: str, train_cfg: dict) -> int:
    return int(train_cfg["per_device_train_batch_size"])


def is_torchrun_distributed() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1 or "LOCAL_RANK" in os.environ or "RANK" in os.environ


def is_primary_process() -> bool:
    if not is_torchrun_distributed():
        return True
    return int(os.environ.get("RANK", "0")) == 0


def maybe_distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def maybe_destroy_process_group() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def resolve_runtime_dtype(finetune_cfg: dict):
    runtime_cfg = finetune_cfg.get("runtime", {})
    raw_dtype = runtime_cfg.get("dtype", "auto") if isinstance(runtime_cfg, dict) else "auto"
    value = str(raw_dtype).strip().lower()
    if value in {"", "auto", "none"}:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return None
        return torch.float16
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported finetune.runtime.dtype: {raw_dtype}")


def resolve_gradient_checkpointing(finetune_cfg: dict):
    runtime_cfg = finetune_cfg.get("runtime", {})
    raw_gc = runtime_cfg.get("use_gradient_checkpointing", "auto") if isinstance(runtime_cfg, dict) else "auto"
    if isinstance(raw_gc, bool) or raw_gc == "unsloth":
        return raw_gc
    value = str(raw_gc).strip().lower()
    if value in {"", "auto"}:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "unsloth"
        return True
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    if value == "unsloth":
        return "unsloth"
    raise ValueError(f"Unsupported finetune.runtime.use_gradient_checkpointing: {raw_gc}")


def resolve_vision_attn_implementation(model_type: str, finetune_cfg: dict) -> str | None:
    runtime_cfg = finetune_cfg.get("runtime", {})
    raw_attn = runtime_cfg.get("vision_attn_implementation", "auto") if isinstance(runtime_cfg, dict) else "auto"
    value = str(raw_attn).strip().lower()
    if value in {"", "auto", "none"}:
        dtype = resolve_runtime_dtype(finetune_cfg)
        if model_type == "ministral" and dtype == torch.float16:
            return "eager"
        return None
    if value in {"eager", "sdpa", "flash_attention_2"}:
        return value
    raise ValueError(f"Unsupported finetune.runtime.vision_attn_implementation: {raw_attn}")


def resolve_trainer_precision(finetune_cfg: dict) -> tuple[bool, bool]:
    dtype = resolve_runtime_dtype(finetune_cfg)
    if dtype == torch.float16:
        return True, False
    if dtype == torch.bfloat16 or dtype is None:
        return False, True
    return False, False


def resolve_inference_vram_fraction(inference_cfg: dict) -> float | None:
    raw_value = inference_cfg.get("vram_fraction")
    if raw_value is None:
        return None
    value = float(raw_value)
    if not 0 < value <= 1:
        raise ValueError(f"Unsupported finetune.inference.vram_fraction: {raw_value}")
    return value


def apply_inference_vram_limit(inference_cfg: dict) -> float | None:
    if not torch.cuda.is_available():
        return None
    fraction = resolve_inference_vram_fraction(inference_cfg)
    if fraction is None:
        return None
    torch.cuda.set_per_process_memory_fraction(fraction, device=torch.cuda.current_device())
    return fraction


def stabilize_vision_attention(model, attn_implementation: str | None) -> None:
    if not attn_implementation:
        return
    attr_paths = (
        "",
        "model",
        "model.model",
        "language_model",
        "model.language_model",
        "base_model",
        "base_model.model",
        "base_model.model.model",
        "base_model.model.language_model",
    )
    for attr_path in attr_paths:
        target = model
        if attr_path:
            for attr in attr_path.split("."):
                target = getattr(target, attr, None)
                if target is None:
                    break
        if target is None:
            continue
        config = getattr(target, "config", None)
        if config is None:
            continue
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = attn_implementation
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = attn_implementation


def get_ablation_cfg(finetune_cfg: dict) -> dict:
    raw = finetune_cfg.get("ablation", {})
    if not isinstance(raw, dict):
        return {
            "label_only": False,
            "human_only": False,
            "synthetic_only": False,
            "variant_name": "default",
        }
    return {
        "label_only": bool(raw.get("label_only", False)),
        "human_only": bool(raw.get("human_only", False)),
        "synthetic_only": bool(raw.get("synthetic_only", False)),
        "variant_name": str(raw.get("variant_name", "default")).strip() or "default",
    }


def resolve_variant_name(ablation_cfg: dict) -> str:
    requested = ablation_cfg["variant_name"]
    if requested != "default":
        return requested

    active_flags = []
    if ablation_cfg["label_only"]:
        active_flags.append("label_only")
    if ablation_cfg["human_only"]:
        active_flags.append("human_only")
    if ablation_cfg["synthetic_only"]:
        active_flags.append("synthetic_only")
    if not active_flags:
        return "default"
    return "_".join(active_flags)


def summarize_metric_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    metrics = ["accuracy", "macro_f1", "weighted_f1", "kappa"]
    summary = {}
    for metric in metrics:
        values = [row[metric] for row in rows]
        summary[f"{metric}_mean"] = round(float(statistics.mean(values)), 4)
        summary[f"{metric}_std"] = round(float(statistics.pstdev(values)), 4) if len(values) > 1 else 0.0
        best_row = max(rows, key=lambda row: row[metric])
        summary[f"{metric}_best_epoch"] = best_row["epoch_value"]
        summary[f"{metric}_best_value"] = best_row[metric]
    return summary


def summarize_fold_metrics(rows: list[dict]) -> dict:
    metrics = ["accuracy", "macro_f1", "weighted_f1", "kappa"]
    summary = {}
    for metric in metrics:
        values = [row[metric] for row in rows]
        summary[f"{metric}_mean"] = round(float(statistics.mean(values)), 4)
        summary[f"{metric}_std"] = round(float(statistics.pstdev(values)), 4) if len(values) > 1 else 0.0
    return summary


def build_cv_summary_markdown(model_type: str, variant_name: str, fold_rows: list[dict], summary: dict) -> str:
    fold_names = ", ".join(f"fold_{row['fold']}" for row in fold_rows)
    lines = [
        "# CV Evaluation Summary",
        "",
        f"- Model: `{model_type}`",
        f"- Variant: `{variant_name}`",
        f"- Folds: `{fold_names}`",
        "",
        "## Per-Fold Metrics",
        "",
        "| Fold | Samples | Accuracy | Macro F1 | Weighted F1 | Kappa |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in fold_rows:
        lines.append(
            f"| {row['fold']} | {row['samples']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['weighted_f1']:.4f} | {row['kappa']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Mean And Std",
            "",
            "| Metric | Mean | Std |",
            "| --- | ---: | ---: |",
            f"| Accuracy | {summary['accuracy_mean']:.4f} | {summary['accuracy_std']:.4f} |",
            f"| Macro F1 | {summary['macro_f1_mean']:.4f} | {summary['macro_f1_std']:.4f} |",
            f"| Weighted F1 | {summary['weighted_f1_mean']:.4f} | {summary['weighted_f1_std']:.4f} |",
            f"| Kappa | {summary['kappa_mean']:.4f} | {summary['kappa_std']:.4f} |",
            "",
        ]
    )
    return "\n".join(lines)


def get_xyz_plot_cfg(finetune_cfg: dict) -> dict:
    raw = finetune_cfg.get("xyz_plot", {})
    if not isinstance(raw, dict):
        return {"enabled": False, "epochs": []}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "epochs": parse_xyz_plot_epochs(raw.get("epochs", [])),
    }


def get_inference_cfg(finetune_cfg: dict) -> dict:
    raw = finetune_cfg.get("inference", {})
    if not isinstance(raw, dict):
        return {"checkpoint_strategy": "final", "checkpoint_epoch": None, "best_epoch_metric": "macro_f1"}
    return {
        "checkpoint_strategy": str(raw.get("checkpoint_strategy", "final")).strip() or "final",
        "checkpoint_epoch": raw.get("checkpoint_epoch"),
        "best_epoch_metric": str(raw.get("best_epoch_metric", "macro_f1")).strip() or "macro_f1",
        "max_new_tokens": raw.get("max_new_tokens"),
        "max_new_tokens_label_only": raw.get("max_new_tokens_label_only"),
    }


def resolve_inference_run_suffix(inference_cfg: dict) -> str:
    strategy = str(inference_cfg.get("checkpoint_strategy", "final")).strip().lower() or "final"
    if strategy in {"final", "last"}:
        return ""
    if strategy == "latest":
        return "checkpoint_latest"
    if strategy == "best_epoch":
        metric = str(inference_cfg.get("best_epoch_metric", "macro_f1")).strip() or "macro_f1"
        return f"checkpoint_best_epoch_{metric}"
    if strategy == "epoch":
        raw_epoch = inference_cfg.get("checkpoint_epoch")
        if raw_epoch is None:
            raise ValueError("inference.checkpoint_epoch must be set when checkpoint_strategy=epoch")
        return f"checkpoint_epoch_{int(raw_epoch)}"
    raise ValueError(f"Unsupported inference.checkpoint_strategy: {strategy}")


def log_effective_runtime_config(model_type: str, train_cfg: dict, finetune_cfg: dict, inference_cfg: dict) -> None:
    message = (
        f"Effective config for {model_type}: "
        f"batch={train_cfg.get('per_device_train_batch_size')} "
        f"per_device_train_batch_size={train_cfg.get('per_device_train_batch_size')} "
        f"grad_accum={train_cfg.get('gradient_accumulation_steps')} "
        f"epochs={train_cfg.get('num_train_epochs')} "
        f"max_seq_length={train_cfg.get('max_seq_length')} "
        f"dtype={finetune_cfg.get('runtime', {}).get('dtype')} "
        f"gc={finetune_cfg.get('runtime', {}).get('use_gradient_checkpointing')} "
        f"checkpoint_strategy={inference_cfg.get('checkpoint_strategy')} "
        f"checkpoint_epoch={inference_cfg.get('checkpoint_epoch')} "
        f"best_epoch_metric={inference_cfg.get('best_epoch_metric')} "
        f"inference_vram_fraction={inference_cfg.get('vram_fraction')}"
    )
    logger.info(message)
    print(message, flush=True)


def parse_xyz_plot_epochs(raw) -> list[int]:
    if raw is None:
        return []

    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        parts = [raw]

    values = []
    for part in parts:
        token = str(part).strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError("X/Y/Z plot epochs must be positive integers")
        values.append(value)
    if not values:
        return []
    return sorted(dict.fromkeys(values))


def find_epoch_checkpoints(checkpoints_root: Path) -> dict[int, Path]:
    epoch_map = {}
    for trainer_state_path in sorted(checkpoints_root.glob("checkpoint-*/trainer_state.json")):
        payload = load_json(trainer_state_path)
        epoch = payload.get("epoch")
        if epoch is None:
            continue
        rounded = int(round(float(epoch)))
        if abs(float(epoch) - rounded) <= 1e-3:
            epoch_map[rounded] = trainer_state_path.parent
    return epoch_map


def resolve_inference_checkpoint_dir(
    results_root: Path,
    models_root: Path,
    fold_idx: int,
    inference_cfg: dict,
) -> tuple[Path, dict]:
    strategy = str(inference_cfg.get("checkpoint_strategy", "final")).strip().lower() or "final"
    adapter_dir = models_root / f"fold_{fold_idx}" / "adapter"
    final_checkpoint_dir = models_root / f"fold_{fold_idx}" / "checkpoints" / "checkpoint-final"
    checkpoints_root = models_root / f"fold_{fold_idx}" / "checkpoints"
    checkpoint_info = {
        "checkpoint_strategy": strategy,
        "checkpoint_epoch": None,
        "checkpoint_dir": str(final_checkpoint_dir if final_checkpoint_dir.exists() else adapter_dir),
    }

    if strategy in {"final", "last"}:
        if final_checkpoint_dir.exists():
            checkpoint_info["checkpoint_dir"] = str(final_checkpoint_dir)
            return final_checkpoint_dir, checkpoint_info
        return adapter_dir, checkpoint_info

    epoch_map = find_epoch_checkpoints(checkpoints_root)
    if not epoch_map:
        logger.warning(
            "No epoch checkpoints found for fold %s under %s; falling back to final adapter",
            fold_idx,
            checkpoints_root,
        )
        return adapter_dir, checkpoint_info

    if strategy == "latest":
        epoch = max(epoch_map)
        checkpoint_dir = epoch_map[epoch]
        checkpoint_info.update({"checkpoint_epoch": epoch, "checkpoint_dir": str(checkpoint_dir)})
        return checkpoint_dir, checkpoint_info

    if strategy == "epoch":
        raw_epoch = inference_cfg.get("checkpoint_epoch")
        if raw_epoch is None:
            raise ValueError("inference.checkpoint_epoch must be set when checkpoint_strategy=epoch")
        epoch = int(raw_epoch)
        checkpoint_dir = epoch_map.get(epoch)
        if checkpoint_dir is None:
            raise ValueError(f"Checkpoint for epoch {epoch} not found under {checkpoints_root}")
        checkpoint_info.update({"checkpoint_epoch": epoch, "checkpoint_dir": str(checkpoint_dir)})
        return checkpoint_dir, checkpoint_info

    if strategy == "best_epoch":
        best_metric = str(inference_cfg.get("best_epoch_metric", "macro_f1")).strip() or "macro_f1"
        summary_candidates = [
            results_root / "xyz_plot_epochs" / f"fold_{fold_idx}" / "xyz_plot_summary.json",
            results_root.parent / "xyz_plot_epochs" / f"fold_{fold_idx}" / "xyz_plot_summary.json",
        ]
        summary_path = next((path for path in summary_candidates if path.exists()), None)
        if summary_path is None:
            logger.warning(
                "Best-epoch checkpoint requested but no xyz_plot_summary.json found for fold %s; falling back to final adapter",
                fold_idx,
            )
            return adapter_dir, checkpoint_info
        summary_payload = load_json(summary_path)
        best_epoch = summary_payload.get("summary", {}).get(f"{best_metric}_best_epoch")
        if best_epoch is None:
            raise ValueError(f"Metric `{best_metric}` not found in {summary_path}")
        epoch = int(best_epoch)
        checkpoint_dir = epoch_map.get(epoch)
        if checkpoint_dir is None:
            raise ValueError(f"Checkpoint for best epoch {epoch} not found under {checkpoints_root}")
        checkpoint_info.update({"checkpoint_epoch": epoch, "checkpoint_dir": str(checkpoint_dir)})
        return checkpoint_dir, checkpoint_info

    raise ValueError(f"Unsupported inference.checkpoint_strategy: {strategy}")


def validate_adapter_dir(adapter_dir: Path, fold_idx: int, strategy: str) -> None:
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Missing adapter/checkpoint directory for fold {fold_idx}: {adapter_dir} "
            f"(checkpoint_strategy={strategy})"
        )
    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"Adapter/checkpoint path is not a directory for fold {fold_idx}: {adapter_dir} "
            f"(checkpoint_strategy={strategy})"
        )
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"Missing adapter_config.json for fold {fold_idx}: {adapter_config} "
            f"(checkpoint_strategy={strategy})"
        )
    if not adapter_weights.exists():
        raise FileNotFoundError(
            f"Missing adapter_model.safetensors for fold {fold_idx}: {adapter_weights} "
            f"(checkpoint_strategy={strategy})"
        )


def save_final_checkpoint_snapshot(
    model,
    tokenizer,
    adapter_dir: Path,
    fold_idx: int,
    model_type: str,
) -> Path:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    final_checkpoint_dir = adapter_dir.parent / "checkpoints" / "checkpoint-final"
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_checkpoint_dir))
    tokenizer.save_pretrained(str(final_checkpoint_dir))
    save_json(
        final_checkpoint_dir / "final_checkpoint_metadata.json",
        {
            "fold": fold_idx,
            "model_type": model_type,
            "source_adapter_dir": str(adapter_dir),
            "checkpoint_dir": str(final_checkpoint_dir),
        },
    )
    return final_checkpoint_dir


def rewrite_prompt_for_label_only(content):
    if isinstance(content, str):
        return content.replace(REASONING_AND_LABEL_INSTRUCTION, LABEL_ONLY_INSTRUCTION)
    if isinstance(content, list):
        rewritten = []
        for part in content:
            updated = dict(part)
            if isinstance(updated.get("text"), str):
                updated["text"] = updated["text"].replace(REASONING_AND_LABEL_INSTRUCTION, LABEL_ONLY_INSTRUCTION)
            rewritten.append(updated)
        return rewritten
    return content


def rewrite_system_for_label_only(content):
    if isinstance(content, str):
        return content.replace(SYSTEM_REASONING_SENTENCE, SYSTEM_LABEL_ONLY_SENTENCE)
    if isinstance(content, list):
        rewritten = []
        for part in content:
            updated = dict(part)
            if isinstance(updated.get("text"), str):
                updated["text"] = updated["text"].replace(SYSTEM_REASONING_SENTENCE, SYSTEM_LABEL_ONLY_SENTENCE)
            rewritten.append(updated)
        return rewritten
    return content


def rewrite_conversations_for_ablation(sample: dict, ablation_cfg: dict) -> dict:
    if not ablation_cfg["label_only"]:
        return sample

    updated_messages = []
    for message in sample["conversations"]:
        updated = dict(message)
        if updated["role"] == "system":
            updated["content"] = rewrite_system_for_label_only(updated["content"])
        elif updated["role"] == "user":
            updated["content"] = rewrite_prompt_for_label_only(updated["content"])
        elif updated["role"] == "assistant":
            label = extract_label(updated["content"]) or sample.get("label")
            if label is None:
                raise ValueError(f"Failed to extract label for paragraph_id={sample.get('paragraph_id')}")
            updated["content"] = f"Label: {label}"
        updated_messages.append(updated)
    return {**sample, "conversations": updated_messages}


def filter_train_dataset_for_ablation(dataset_sft, ablation_cfg: dict, fold_idx: int):
    original_size = len(dataset_sft)
    if ablation_cfg["human_only"] and ablation_cfg["synthetic_only"]:
        filtered = dataset_sft.filter(lambda sample: sample["source"] in {"human", "synthetic"})
        logger.info(
            "Human+synthetic-only ablation active for fold %s: kept %s/%s training samples",
            fold_idx,
            len(filtered),
            original_size,
        )
        return filtered
    if ablation_cfg["human_only"]:
        filtered = dataset_sft.filter(lambda sample: sample["source"] == "human")
        logger.info(
            "Human-only ablation active for fold %s: kept %s/%s training samples",
            fold_idx,
            len(filtered),
            original_size,
        )
        return filtered
    if ablation_cfg["synthetic_only"]:
        filtered = dataset_sft.filter(lambda sample: sample["source"] in {"human", "synthetic"})
        logger.info(
            "Synthetic-only ablation active for fold %s: kept %s/%s training samples",
            fold_idx,
            len(filtered),
            original_size,
        )
        return filtered
    return dataset_sft


def build_eval_messages_for_objective(item: dict, model_type: str, ablation_cfg: dict):
    messages = build_eval_messages(item, model_type)
    if not ablation_cfg["label_only"]:
        return messages

    rewritten = []
    for message in messages:
        updated = dict(message)
        if updated["role"] == "system":
            updated["content"] = rewrite_system_for_label_only(updated["content"])
        elif updated["role"] == "user":
            updated["content"] = rewrite_prompt_for_label_only(updated["content"])
        rewritten.append(updated)
    return rewritten


def setup_base_model(model_type: str, train_cfg: dict, finetune_cfg: dict):
    model_name = get_model_name(model_type, finetune_cfg)
    max_seq_length = int(train_cfg["max_seq_length"])
    dtype = resolve_runtime_dtype(finetune_cfg)
    use_gc = resolve_gradient_checkpointing(finetune_cfg)
    attn_implementation = resolve_vision_attn_implementation(model_type, finetune_cfg)

    if model_type == "gemma":
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template

        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            dtype=dtype,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
        return model, tokenizer, "FastModel"

    if model_type == "gemma26b":
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template

        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            dtype=dtype,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
            use_gradient_checkpointing="unsloth",
        )
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
        return model, tokenizer, "FastModel"

    if model_type == "gemma26b":
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template

        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            dtype=dtype,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
            use_gradient_checkpointing="unsloth",
        )
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
        return model, tokenizer, "FastModel"

    if model_type == "llama":
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=True,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
        return model, tokenizer, "FastLanguageModel"

    if model_type == "qwen":
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=True,
            dtype=dtype,
            use_gradient_checkpointing=use_gc,
        )
        stabilize_vision_attention(model, attn_implementation)
        return model, tokenizer, "FastVisionModel"

    if model_type == "ministral":
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=True,
            dtype=dtype,
            use_gradient_checkpointing=use_gc,
        )
        stabilize_vision_attention(model, "eager")
        return model, tokenizer, "FastVisionModel"

    raise ValueError(f"Unknown model_type: {model_type}")


def setup_model(model_type: str, train_cfg: dict, finetune_cfg: dict):
    lora_r = int(train_cfg["lora_r"])
    lora_alpha = int(train_cfg["lora_alpha"])
    lora_dropout = float(train_cfg["lora_dropout"])
    random_state = int(train_cfg["random_state"])
    model_name = get_model_name(model_type, finetune_cfg)
    dtype = resolve_runtime_dtype(finetune_cfg)
    use_gc = resolve_gradient_checkpointing(finetune_cfg)
    attn_implementation = resolve_vision_attn_implementation(model_type, finetune_cfg)

    if model_type == "gemma":
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template

        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            dtype=dtype,
            max_seq_length=int(train_cfg["max_seq_length"]),
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            random_state=random_state,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

        def formatting_func(examples):
            texts = [
                tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
                for convo in examples["conversations"]
            ]
            return {"text": texts}

        return model, tokenizer, formatting_func, {"instruction_part": "<|turn>user\n", "response_part": "<|turn>model\n"}, "FastModel"

    if model_type == "gemma26b":
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template

        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            dtype=dtype,
            max_seq_length=int(train_cfg["max_seq_length"]),
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
            use_gradient_checkpointing="unsloth",
        )
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            random_state=random_state,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

        def formatting_func(examples):
            texts = [
                tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
                for convo in examples["conversations"]
            ]
            return {"text": texts}

        return model, tokenizer, formatting_func, {"instruction_part": "<|turn>user\n", "response_part": "<|turn>model\n"}, "FastModel"

    if model_type == "llama":
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=int(train_cfg["max_seq_length"]),
            dtype=dtype,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing=use_gc,
            random_state=random_state,
            use_rslora=False,
            loftq_config=None,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

        def formatting_func(examples):
            return {"text": [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in examples["conversations"]]}

        return model, tokenizer, formatting_func, {"instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n", "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n"}, "FastLanguageModel"

    if model_type in {"qwen", "ministral"}:
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=True,
            dtype=dtype,
            use_gradient_checkpointing=use_gc,
        )
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            random_state=random_state,
            use_rslora=False,
            loftq_config=None,
        )
        if model_type == "ministral":
            stabilize_vision_attention(model, "eager")
        elif attn_implementation:
            stabilize_vision_attention(model, attn_implementation)

        def convert_to_conversation(sample):
            return {
                "messages": [
                    {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}
                    for message in sample["conversations"]
                ]
            }

        return model, tokenizer, convert_to_conversation, None, "FastVisionModel"

    raise ValueError(f"Unknown model_type: {model_type}")


def build_inference_inputs(tokenizer, item: dict, model_type: str, ablation_cfg: dict):
    messages = build_eval_messages_for_objective(item, model_type, ablation_cfg)
    if model_type in {"gemma", "gemma26b"}:
        normalized_messages = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            normalized_messages.append({"role": message["role"], "content": content})
        prompt = tokenizer.apply_chat_template(normalized_messages, tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer(text=prompt, return_tensors="pt")
        return {key: value.to("cuda") for key, value in tokenized.items()}
    if model_type == "llama":
        tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if isinstance(tokenized, torch.Tensor):
            return {"input_ids": tokenized.to("cuda"), "attention_mask": torch.ones_like(tokenized).to("cuda")}
        return {key: value.to("cuda") for key, value in tokenized.items()}
    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    return {key: value.to("cuda") for key, value in tokenized.items()}


def run_fold_inference(model, tokenizer, val_items: list[dict], model_type: str, max_new_tokens: int, ablation_cfg: dict) -> list[dict]:
    stop_criteria = StoppingCriteriaList([StopOnLabel(tokenizer)])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    use_cache = model_type != "gemma26b"
    results = []
    total = len(val_items)
    for idx, item in enumerate(val_items, start=1):
        inputs = build_inference_inputs(tokenizer, item, model_type, ablation_cfg)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=use_cache,
                pad_token_id=pad_id,
                stopping_criteria=stop_criteria,
            )
        input_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        results.append(
            {
                "paragraph_id": item["paragraph_id"],
                "raw_output": decoded,
                "parsed_label": extract_label(decoded),
                "ground_truth_label": item["label"],
            }
        )
        if idx <= 3 or idx % 10 == 0 or idx == total:
            logger.info(
                "Inference sample %s/%s paragraph_id=%s pred=%s gold=%s",
                idx,
                total,
                item["paragraph_id"],
                results[-1]["parsed_label"],
                item["label"],
            )
    return results


def slice_json_dataset(dataset, limit: int | None):
    if limit is None or limit <= 0 or len(dataset) <= limit:
        return dataset
    return dataset.select(range(limit))


def train_single_fold(
    model_type: str,
    fold_idx: int,
    train_sft_path: Path,
    adapter_dir: Path,
    train_cfg: dict,
    finetune_cfg: dict,
    ablation_cfg: dict,
    dry_run_cfg: dict | None = None,
    keep_loaded: bool = False,
):
    dataset_sft = load_dataset("json", data_files=str(train_sft_path), split="train")
    dataset_sft = filter_train_dataset_for_ablation(dataset_sft, ablation_cfg, fold_idx)
    dataset_sft = dataset_sft.map(lambda sample: rewrite_conversations_for_ablation(sample, ablation_cfg))
    dry_run_enabled = bool(dry_run_cfg and dry_run_cfg.get("enabled"))
    if dry_run_enabled:
        dataset_sft = slice_json_dataset(dataset_sft, int(dry_run_cfg["train_samples"]))
        logger.info("Dry run enabled for fold %s: using %s train samples", fold_idx, len(dataset_sft))

    use_gc = resolve_gradient_checkpointing(finetune_cfg)
    model, tokenizer, formatting_func, response_cfg, api_class = setup_model(model_type, train_cfg, finetune_cfg)
    per_device_train_batch_size = resolve_train_batch_size(model_type, train_cfg)

    if api_class == "FastLanguageModel" and model_type == "llama":
        from unsloth.chat_templates import standardize_sharegpt
        dataset_sft = standardize_sharegpt(dataset_sft)

    if api_class in {"FastModel", "FastLanguageModel"}:
        dataset_sft = dataset_sft.map(formatting_func, batched=True)
        from trl import SFTConfig, SFTTrainer

        fp16, bf16 = resolve_trainer_precision(finetune_cfg)
        if "gemma" in model_type:
            data_collator = Gemma4TextCollator(tokenizer)
        elif model_type == "llama":
            from transformers import DataCollatorForSeq2Seq

            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        else:
            data_collator = None
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset_sft,
            dataset_text_field="text",
            max_seq_length=int(train_cfg["max_seq_length"]),
            data_collator=data_collator,
            packing=False,
            args=SFTConfig(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
                warmup_steps=int(train_cfg["warmup_steps"]),
                num_train_epochs=float(train_cfg["num_train_epochs"]),
                learning_rate=float(train_cfg["learning_rate"]),
                logging_steps=1,
                optim=str(train_cfg["optim"]),
                weight_decay=float(train_cfg["weight_decay"]),
                lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
                seed=int(train_cfg["random_state"]),
                max_steps=int(dry_run_cfg["max_steps"]) if dry_run_enabled else -1,
                output_dir=str(adapter_dir.parent / "checkpoints"),
                save_strategy="epoch",
                save_total_limit=20,
                report_to="none",
                gradient_checkpointing=bool(use_gc),
                fp16=fp16,
                bf16=bf16,
            ),
        )
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part=response_cfg["instruction_part"],
            response_part=response_cfg["response_part"],
        )
        if "gemma" in model_type:
            trainer.data_collator = Gemma4TextCollator(tokenizer)
    else:
        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTConfig, SFTTrainer

        FastVisionModel.for_training(model)
        converted_dataset = [formatting_func(sample) for sample in dataset_sft]
        fp16, bf16 = resolve_trainer_precision(finetune_cfg)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=converted_dataset,
            args=SFTConfig(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
                warmup_steps=int(train_cfg["warmup_steps"]),
                num_train_epochs=float(train_cfg["num_train_epochs"]),
                learning_rate=float(train_cfg["learning_rate"]),
                logging_steps=1,
                optim=str(train_cfg["optim"]),
                weight_decay=float(train_cfg["weight_decay"]),
                lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
                seed=int(train_cfg["random_state"]),
                max_steps=int(dry_run_cfg["max_steps"]) if dry_run_enabled else -1,
                output_dir=str(adapter_dir.parent / "checkpoints"),
                save_strategy="epoch",
                save_total_limit=20,
                report_to="none",
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_length=int(train_cfg["max_seq_length"]),
                gradient_checkpointing=bool(use_gc),
                fp16=fp16,
                bf16=bf16,
            ),
        )

    logger.info("Training %s fold %s", model_type, fold_idx)
    trainer.train()
    final_checkpoint_dir = adapter_dir.parent / "checkpoints" / "checkpoint-final"
    if is_primary_process():
        final_checkpoint_dir = save_final_checkpoint_snapshot(
            model=model,
            tokenizer=tokenizer,
            adapter_dir=adapter_dir,
            fold_idx=fold_idx,
            model_type=model_type,
        )
        logger.info(
            "Saved inference-ready weights for %s fold %s: adapter=%s final_checkpoint=%s",
            model_type,
            fold_idx,
            adapter_dir,
            final_checkpoint_dir,
        )
    maybe_distributed_barrier()

    del trainer
    if keep_loaded:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model, tokenizer, api_class

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def infer_single_fold(
    model_type: str,
    fold_idx: int,
    adapter_dir: Path,
    val_eval_path: Path,
    output_path: Path,
    max_new_tokens: int,
    train_cfg: dict,
    finetune_cfg: dict,
    ablation_cfg: dict,
    dry_run_cfg: dict | None = None,
) -> None:
    inference_cfg = finetune_cfg.get("inference", {})
    validate_adapter_dir(adapter_dir, fold_idx, str(inference_cfg.get("checkpoint_strategy", "final")))
    applied_vram_fraction = apply_inference_vram_limit(inference_cfg)
    if applied_vram_fraction is not None:
        logger.info(
            "Applied inference VRAM fraction %.3f for %s fold %s",
            applied_vram_fraction,
            model_type,
            fold_idx,
        )
    model, tokenizer, api_class = setup_base_model(model_type, train_cfg, finetune_cfg)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.to("cuda")
    if api_class == "FastLanguageModel":
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    elif api_class == "FastModel":
        from unsloth import FastModel
        FastModel.for_inference(model)
    else:
        from unsloth import FastVisionModel
        FastVisionModel.for_inference(model)

    val_items = load_json(val_eval_path)
    if dry_run_cfg and dry_run_cfg.get("enabled"):
        val_items = val_items[: int(dry_run_cfg["val_samples"])]
        logger.info("Dry run enabled for fold %s: using %s validation samples", fold_idx, len(val_items))
    logger.info("Starting inference for %s fold %s with %s validation samples", model_type, fold_idx, len(val_items))
    results = run_fold_inference(model, tokenizer, val_items, model_type, max_new_tokens, ablation_cfg)
    save_json(output_path, results)
    logger.info("Saved inference outputs for %s fold %s to %s", model_type, fold_idx, output_path)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def inference_worker(
    model_type: str,
    fold_idx: int,
    adapter_dir: str,
    val_eval_path: str,
    output_path: str,
    max_new_tokens: int,
    train_cfg: dict,
    finetune_cfg: dict,
    ablation_cfg: dict,
    dry_run_cfg: dict | None,
):
    infer_single_fold(
        model_type=model_type,
        fold_idx=fold_idx,
        adapter_dir=Path(adapter_dir),
        val_eval_path=Path(val_eval_path),
        output_path=Path(output_path),
        max_new_tokens=max_new_tokens,
        train_cfg=train_cfg,
        finetune_cfg=finetune_cfg,
        ablation_cfg=ablation_cfg,
        dry_run_cfg=dry_run_cfg,
    )


def training_worker(
    model_type: str,
    fold_idx: int,
    train_sft_path: str,
    adapter_dir: str,
    train_cfg: dict,
    finetune_cfg: dict,
    ablation_cfg: dict,
    dry_run_cfg: dict | None,
):
    train_single_fold(
        model_type=model_type,
        fold_idx=fold_idx,
        train_sft_path=Path(train_sft_path),
        adapter_dir=Path(adapter_dir),
        train_cfg=train_cfg,
        finetune_cfg=finetune_cfg,
        ablation_cfg=ablation_cfg,
        dry_run_cfg=dry_run_cfg,
        keep_loaded=False,
    )


def run_epoch_sweep(
    model_type: str,
    fold_entry: dict,
    train_cfg: dict,
    finetune_cfg: dict,
    ablation_cfg: dict,
    dry_run_cfg: dict,
    models_root: Path,
    results_root: Path,
    epoch_values: list[int],
) -> None:
    sweep_models_root = models_root / "xyz_plot_epochs" / "fold_0"
    sweep_results_root = results_root / "xyz_plot_epochs" / "fold_0"
    sweep_models_root.mkdir(parents=True, exist_ok=True)
    sweep_results_root.mkdir(parents=True, exist_ok=True)

    fold_idx = int(fold_entry["fold"])
    if fold_idx != 0:
        raise ValueError("X/Y/Z plot mode only supports fold_0")

    max_epoch = max(epoch_values)
    training_run_root = sweep_models_root / f"max_epoch_{max_epoch}"
    adapter_dir = training_run_root / "adapter"
    max_epoch_train_cfg = dict(train_cfg)
    max_epoch_train_cfg["num_train_epochs"] = max_epoch

    logger.info(
        "X/Y/Z plot training start: fold=0 epochs=%s label_only=%s",
        epoch_values,
        ablation_cfg["label_only"],
    )
    train_process = mp.Process(
        target=training_worker,
        args=(
            model_type,
            fold_idx,
            str(resolve_path(fold_entry["sft_path"])),
            str(adapter_dir),
            max_epoch_train_cfg,
            finetune_cfg,
            ablation_cfg,
            dry_run_cfg,
        ),
    )
    train_process.start()
    train_process.join()
    if train_process.exitcode != 0:
        raise RuntimeError(f"X/Y/Z training failed for max_epoch={max_epoch} with exit code {train_process.exitcode}")

    epoch_checkpoint_map = find_epoch_checkpoints(training_run_root / "checkpoints")
    missing_epochs = [epoch_value for epoch_value in epoch_values if epoch_value not in epoch_checkpoint_map]
    if missing_epochs:
        raise RuntimeError(f"Missing epoch checkpoints for X/Y/Z plot: {missing_epochs}")

    rows = []
    for epoch_value in epoch_values:
        run_name = f"epoch_{epoch_value}"
        checkpoint_dir = epoch_checkpoint_map[epoch_value]
        output_path = sweep_results_root / f"{run_name}_results.json"
        metric_path = sweep_results_root / f"{run_name}_metrics.json"

        logger.info("X/Y/Z plot inference start: fold=0 epoch=%s checkpoint=%s", epoch_value, checkpoint_dir.name)
        infer_process = mp.Process(
            target=inference_worker,
            args=(
                model_type,
                fold_idx,
                str(checkpoint_dir),
                str(resolve_path(fold_entry["val_eval_path"])),
                str(output_path),
                int(finetune_cfg["inference"]["max_new_tokens_label_only"]) if ablation_cfg["label_only"] else int(finetune_cfg["inference"]["max_new_tokens"]),
                max_epoch_train_cfg,
                finetune_cfg,
                ablation_cfg,
                dry_run_cfg,
            ),
        )
        infer_process.start()
        infer_process.join()
        if infer_process.exitcode != 0:
            raise RuntimeError(f"X/Y/Z inference failed for epoch={epoch_value} with exit code {infer_process.exitcode}")

        fold_results = load_json(output_path)
        metrics = compute_overall_metrics(fold_results)
        metrics["epoch_value"] = epoch_value
        save_json(metric_path, metrics)
        rows.append(metrics)
        logger.info(
            "X/Y/Z plot run complete: epoch=%s acc=%.4f macro_f1=%.4f",
            epoch_value,
            metrics["accuracy"],
            metrics["macro_f1"],
        )

    summary = {
        "mode": resolve_variant_name(ablation_cfg),
        "fold": 0,
        "epochs_tested": epoch_values,
        "runs": rows,
        "summary": summarize_metric_rows(rows),
        "sweet_spot_macro_f1": max(rows, key=lambda row: row["macro_f1"]),
    }
    save_json(sweep_results_root / "xyz_plot_summary.json", summary)
    write_csv(
        sweep_results_root / "xyz_plot_summary.csv",
        rows,
        ["epoch_value", "samples", "accuracy", "macro_f1", "weighted_f1", "kappa"],
    )


def write_full_cv_summary(results_root: Path, model_type: str, variant_name: str, fold_rows: list[dict]) -> None:
    ordered_rows = sorted(fold_rows, key=lambda row: row["fold"])
    summary = summarize_fold_metrics(ordered_rows)
    save_json(results_root / "overall_folds.json", ordered_rows)
    save_json(results_root / "overall_summary.json", {"folds": ordered_rows, "summary": summary})
    markdown = build_cv_summary_markdown(model_type, variant_name, ordered_rows, summary)
    (results_root / "overall_summary.md").write_text(markdown, encoding="utf-8")


def fold_completion_mtime(results_root: Path, models_root: Path, fold_idx: int) -> float | None:
    results_path = results_root / f"fold_{fold_idx}_results.json"
    metrics_path = results_root / f"fold_{fold_idx}_metrics.json"
    adapter_dir = models_root / f"fold_{fold_idx}" / "adapter"
    if not (results_path.exists() and metrics_path.exists() and adapter_dir.exists()):
        return None
    mtimes = [results_path.stat().st_mtime, metrics_path.stat().st_mtime, adapter_dir.stat().st_mtime]
    return max(mtimes)


def get_resume_anchor_mtime(results_root: Path, models_root: Path) -> float | None:
    mtimes: list[float] = []
    for fold_idx in range(3):
        fold_mtime = fold_completion_mtime(results_root, models_root, fold_idx)
        if fold_mtime is not None:
            mtimes.append(fold_mtime)
    if not mtimes:
        return None
    return max(mtimes)


def is_fold_complete(results_root: Path, models_root: Path, fold_idx: int, resume_anchor_mtime: float | None = None) -> bool:
    fold_mtime = fold_completion_mtime(results_root, models_root, fold_idx)
    if fold_mtime is None:
        return False
    if resume_anchor_mtime is None:
        return True
    return fold_mtime >= resume_anchor_mtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal Step 5 CV fine-tuning and inference")
    parser.add_argument("--model", required=True, choices=["gemma", "gemma26b", "llama", "qwen", "ministral"])
    args = parser.parse_args()

    if is_torchrun_distributed():
        raise RuntimeError(
            "run_cv_finetune.py 目前不支援直接用 torchrun 執行。"
            "這支腳本本身已用 multiprocessing 管理 fold 訓練與推論；再套 torchrun 會形成巢狀多進程/多卡。"
            "請改用 `uv run python src/step5_cv/run_cv_finetune.py --model ...`。"
        )

    config = load_step5_cv_config()
    cfg = config["finetune"]
    cfg = apply_model_overrides(cfg, args.model)
    train_cfg = cfg["training"]
    ablation_cfg = get_ablation_cfg(cfg)
    xyz_plot_cfg = get_xyz_plot_cfg(cfg)
    inference_cfg = get_inference_cfg(cfg)
    dry_run_cfg = cfg.get("dry_run", {})
    dry_run_enabled = bool(dry_run_cfg.get("enabled"))
    multi_gpu_cfg = cfg.get("multi_gpu", {})
    resume_enabled = bool(multi_gpu_cfg.get("resume", False))
    sft_output_dir = resolve_path(cfg["sft_output_dir"])
    results_root = resolve_path(cfg["results_root"]) / args.model
    models_root = resolve_path(cfg["models_root"]) / args.model
    variant_name = resolve_variant_name(ablation_cfg)
    if variant_name != "default":
        results_root = results_root / variant_name
        models_root = models_root / variant_name
    inference_suffix = resolve_inference_run_suffix(inference_cfg)
    if inference_suffix:
        results_root = results_root / inference_suffix
    if dry_run_enabled:
        results_root = results_root / str(dry_run_cfg.get("results_subdir", "dry_run"))
        models_root = models_root / str(dry_run_cfg.get("results_subdir", "dry_run"))
    results_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    resume_anchor_mtime = get_resume_anchor_mtime(results_root, models_root) if resume_enabled else None

    sft_manifest = load_json(sft_output_dir / "manifest.json")
    fold_entries = sorted(sft_manifest["folds"], key=lambda item: item["fold"])
    if xyz_plot_cfg["enabled"]:
        fold_entries = [entry for entry in fold_entries if int(entry["fold"]) == 0]
        if not fold_entries:
            raise ValueError("fold_0 not found in SFT manifest for X/Y/Z plot mode")
    if dry_run_enabled:
        target_fold = int(dry_run_cfg["fold"])
        fold_entries = [entry for entry in fold_entries if int(entry["fold"]) == target_fold]
        if not fold_entries:
            raise ValueError(f"Dry run fold {target_fold} not found in SFT manifest")
        logger.info("Dry run enabled: model=%s fold=%s", args.model, target_fold)
    logger.info(
        "Ablation mode: label_only=%s human_only=%s synthetic_only=%s variant=%s",
        ablation_cfg["label_only"],
        ablation_cfg["human_only"],
        ablation_cfg["synthetic_only"],
        variant_name,
    )
    log_effective_runtime_config(args.model, train_cfg, cfg, inference_cfg)
    if xyz_plot_cfg["enabled"]:
        epoch_values = xyz_plot_cfg["epochs"]
        if not epoch_values:
            raise ValueError("X/Y/Z plot is enabled but no epochs are configured in config.yaml")
        logger.info("X/Y/Z plot mode enabled: fold=0 epochs=%s", epoch_values)
        mp.set_start_method("spawn", force=True)
        run_epoch_sweep(
            model_type=args.model,
            fold_entry=fold_entries[0],
            train_cfg=train_cfg,
            finetune_cfg=cfg,
            ablation_cfg=ablation_cfg,
            dry_run_cfg=dry_run_cfg,
            models_root=models_root,
            results_root=results_root,
            epoch_values=epoch_values,
        )
        return

    if dry_run_enabled:
        fold_entry = fold_entries[0]
        adapter_dir = models_root / f"fold_{fold_entry['fold']}" / "adapter"
        fold_idx = fold_entry["fold"]
        model, tokenizer, api_class = train_single_fold(
            model_type=args.model,
            fold_idx=fold_idx,
            train_sft_path=resolve_path(fold_entry["sft_path"]),
            adapter_dir=adapter_dir,
            train_cfg=train_cfg,
            finetune_cfg=cfg,
            ablation_cfg=ablation_cfg,
            dry_run_cfg=dry_run_cfg,
            keep_loaded=True,
        )
        if api_class == "FastLanguageModel":
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        elif api_class == "FastModel":
            from unsloth import FastModel
            FastModel.for_inference(model)
        else:
            from unsloth import FastVisionModel
            FastVisionModel.for_inference(model)

        val_items = load_json(resolve_path(fold_entry["val_eval_path"]))[: int(dry_run_cfg["val_samples"])]
        output_path = results_root / f"fold_{fold_idx}_results.json"
        dry_run_max_new_tokens = int(cfg["inference"]["max_new_tokens_label_only"]) if ablation_cfg["label_only"] else int(cfg["inference"]["max_new_tokens"])
        rows = run_fold_inference(model, tokenizer, val_items, args.model, dry_run_max_new_tokens, ablation_cfg)
        save_json(output_path, rows)
        fold_metrics = compute_overall_metrics(rows)
        save_json(results_root / f"fold_{fold_idx}_metrics.json", fold_metrics)
        logger.info("%s dry run fold %s complete: acc=%.4f macro_f1=%.4f", args.model, fold_idx, fold_metrics["accuracy"], fold_metrics["macro_f1"])

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    mp.set_start_method("spawn", force=True)
    fold_summaries = []
    for fold_entry in fold_entries:
        fold_idx = fold_entry["fold"]
        adapter_dir = models_root / f"fold_{fold_idx}" / "adapter"
        if resume_enabled and is_fold_complete(results_root, models_root, fold_idx, resume_anchor_mtime):
            fold_metrics = load_json(results_root / f"fold_{fold_idx}_metrics.json")
            if "fold" not in fold_metrics:
                fold_metrics["fold"] = fold_idx
            fold_summaries.append(fold_metrics)
            logger.info("Skipping completed fold %s (resume=true)", fold_idx)
            continue

        train_process = mp.Process(
            target=training_worker,
            args=(
                args.model,
                fold_idx,
                str(resolve_path(fold_entry["sft_path"])),
                str(adapter_dir),
                train_cfg,
                cfg,
                ablation_cfg,
                dry_run_cfg,
            ),
        )
        train_process.start()
        train_process.join()
        if train_process.exitcode != 0:
            raise RuntimeError(f"Training process for fold {fold_idx} failed with exit code {train_process.exitcode}")

        output_path = results_root / f"fold_{fold_idx}_results.json"
        checkpoint_dir, checkpoint_info = resolve_inference_checkpoint_dir(
            results_root=results_root,
            models_root=models_root,
            fold_idx=fold_idx,
            inference_cfg=inference_cfg,
        )
        logger.info(
            "Inference checkpoint selected for fold %s: strategy=%s epoch=%s dir=%s",
            fold_idx,
            checkpoint_info["checkpoint_strategy"],
            checkpoint_info["checkpoint_epoch"],
            checkpoint_info["checkpoint_dir"],
        )
        infer_process = mp.Process(
            target=inference_worker,
            args=(
                args.model,
                fold_idx,
                str(checkpoint_dir),
                str(resolve_path(fold_entry["val_eval_path"])),
                str(output_path),
                int(inference_cfg["max_new_tokens_label_only"]) if ablation_cfg["label_only"] else int(inference_cfg["max_new_tokens"]),
                train_cfg,
                cfg,
                ablation_cfg,
                dry_run_cfg,
            ),
        )
        infer_process.start()
        infer_process.join()
        if infer_process.exitcode != 0:
            raise RuntimeError(f"Inference process for fold {fold_idx} failed with exit code {infer_process.exitcode}")
        rows = load_json(results_root / f"fold_{fold_idx}_results.json")
        fold_metrics = compute_overall_metrics(rows)
        fold_metrics["fold"] = fold_idx
        fold_metrics.update(checkpoint_info)
        save_json(results_root / f"fold_{fold_idx}_metrics.json", fold_metrics)
        fold_summaries.append(fold_metrics)
        logger.info("%s fold %s complete: acc=%.4f macro_f1=%.4f", args.model, fold_idx, fold_metrics["accuracy"], fold_metrics["macro_f1"])

    write_full_cv_summary(
        results_root=results_root,
        model_type=args.model,
        variant_name=variant_name,
        fold_rows=fold_summaries,
    )
    logger.info("Saved full CV summary to %s", results_root / "overall_summary.md")


if __name__ == "__main__":
    main()
