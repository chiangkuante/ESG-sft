from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

import unsloth  # noqa: F401  # Import before transformers-dependent modules.
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step4_cv.common import load_step4_config, resolve_annotator_name
from src.step5_cv.common import (
    compute_overall_metrics,
    load_json,
    load_step5_cv_config,
    resolve_path,
    save_json,
)
from src.step5_cv.run_cv_finetune import (
    MODEL_NAME_DEFAULTS,
    apply_inference_vram_limit,
    apply_model_overrides,
    build_cv_summary_markdown,
    get_ablation_cfg,
    get_inference_cfg,
    get_model_name,
    log_effective_runtime_config,
    run_fold_inference,
    setup_base_model,
    summarize_fold_metrics,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def resolve_experiment_names(baseline_cfg: dict[str, Any]) -> list[str]:
    raw = baseline_cfg.get("experiments")
    if raw is None:
        return [resolve_annotator_name(load_step4_config())]
    if isinstance(raw, str):
        names = [part.strip() for part in raw.split(",")]
    else:
        names = [str(part).strip() for part in raw]
    names = [name for name in names if name]
    if not names:
        raise ValueError("step5_cv.local_slm_baseline.experiments is empty")
    return names


def resolve_enabled_models(baseline_cfg: dict[str, Any]) -> list[str]:
    raw = baseline_cfg.get("models")
    if raw is None:
        return list(MODEL_NAME_DEFAULTS)
    if isinstance(raw, list):
        model_names = [str(item).strip() for item in raw if str(item).strip()]
    elif isinstance(raw, dict):
        model_names = [
            str(model_key)
            for model_key, model_cfg in raw.items()
            if not isinstance(model_cfg, dict) or bool(model_cfg.get("enabled", True))
        ]
    else:
        raise ValueError("step5_cv.local_slm_baseline.models must be a list or mapping")

    unknown = [model_name for model_name in model_names if model_name not in MODEL_NAME_DEFAULTS]
    if unknown:
        raise ValueError(f"Unknown local SLM baseline model(s): {unknown}")
    if not model_names:
        raise ValueError("No enabled models in step5_cv.local_slm_baseline.models")
    return model_names


def resolve_baseline_ablation_cfg(finetune_cfg: dict[str, Any], baseline_cfg: dict[str, Any]) -> dict[str, Any]:
    ablation_cfg = get_ablation_cfg(finetune_cfg)
    if "label_only" in baseline_cfg:
        ablation_cfg["label_only"] = bool(baseline_cfg["label_only"])
    ablation_cfg["human_only"] = False
    ablation_cfg["synthetic_only"] = False
    ablation_cfg["variant_name"] = "unfinetuned_local_slm"
    return ablation_cfg


def resolve_max_new_tokens(baseline_cfg: dict[str, Any], inference_cfg: dict[str, Any], label_only: bool) -> int:
    if baseline_cfg.get("max_new_tokens") is not None:
        return int(baseline_cfg["max_new_tokens"])
    if label_only and inference_cfg.get("max_new_tokens_label_only") is not None:
        return int(inference_cfg["max_new_tokens_label_only"])
    if inference_cfg.get("max_new_tokens") is not None:
        return int(inference_cfg["max_new_tokens"])
    return 16 if label_only else 512


def get_target_folds(baseline_cfg: dict[str, Any]) -> set[int] | None:
    raw = baseline_cfg.get("folds")
    if raw is None:
        return None
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    else:
        values = [str(part).strip() for part in raw]
    folds = {int(value) for value in values if value}
    if not folds:
        raise ValueError("step5_cv.local_slm_baseline.folds is empty")
    return folds


def write_full_cv_summary(results_root: Path, model_type: str, experiment_name: str, fold_rows: list[dict]) -> None:
    ordered_rows = sorted(fold_rows, key=lambda row: row["fold"])
    summary = summarize_fold_metrics(ordered_rows)
    save_json(results_root / "overall_folds.json", ordered_rows)
    save_json(results_root / "overall_summary.json", {"folds": ordered_rows, "summary": summary})
    markdown = build_cv_summary_markdown(
        model_type=model_type,
        variant_name=f"unfinetuned_local_slm/{experiment_name}",
        fold_rows=ordered_rows,
        summary=summary,
    )
    (results_root / "overall_summary.md").write_text(markdown, encoding="utf-8")


def prepare_base_model_for_inference(model_type: str, train_cfg: dict, finetune_cfg: dict):
    inference_cfg = finetune_cfg.get("inference", {})
    applied_vram_fraction = apply_inference_vram_limit(inference_cfg)
    if applied_vram_fraction is not None:
        logger.info("Applied inference VRAM fraction %.3f for %s", applied_vram_fraction, model_type)

    model, tokenizer, api_class = setup_base_model(model_type, train_cfg, finetune_cfg)
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
    return model, tokenizer


def run_model_baseline(
    model_type: str,
    experiment_names: list[str],
    finetune_cfg: dict[str, Any],
    baseline_cfg: dict[str, Any],
) -> None:
    cfg = apply_model_overrides(finetune_cfg, model_type)
    train_cfg = cfg["training"]
    inference_cfg = get_inference_cfg(cfg)
    ablation_cfg = resolve_baseline_ablation_cfg(cfg, baseline_cfg)
    max_new_tokens = resolve_max_new_tokens(baseline_cfg, inference_cfg, ablation_cfg["label_only"])
    target_folds = get_target_folds(baseline_cfg)
    resume_enabled = bool(baseline_cfg.get("resume", True))

    sft_output_dir = resolve_path(baseline_cfg.get("sft_output_dir", cfg["sft_output_dir"]))
    results_root_base = resolve_path(baseline_cfg.get("results_root", cfg["results_root"]))

    logger.info(
        "Local SLM baseline start: model=%s model_name=%s label_only=%s max_new_tokens=%s experiments=%s",
        model_type,
        get_model_name(model_type, cfg),
        ablation_cfg["label_only"],
        max_new_tokens,
        experiment_names,
    )
    log_effective_runtime_config(model_type, train_cfg, cfg, inference_cfg)

    model, tokenizer = prepare_base_model_for_inference(model_type, train_cfg, cfg)
    try:
        for experiment_name in experiment_names:
            manifest_path = sft_output_dir / experiment_name / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing SFT manifest for local SLM baseline: {manifest_path}")

            results_root = results_root_base / experiment_name / "local_slm_baseline" / model_type
            results_root.mkdir(parents=True, exist_ok=True)
            manifest = load_json(manifest_path)
            fold_entries = sorted(manifest["folds"], key=lambda item: item["fold"])
            if target_folds is not None:
                fold_entries = [entry for entry in fold_entries if int(entry["fold"]) in target_folds]
            if not fold_entries:
                raise ValueError(f"No target folds for local SLM baseline experiment={experiment_name}")

            fold_summaries: list[dict[str, Any]] = []
            for fold_entry in fold_entries:
                fold_idx = int(fold_entry["fold"])
                output_path = results_root / f"fold_{fold_idx}_results.json"
                metrics_path = results_root / f"fold_{fold_idx}_metrics.json"
                if resume_enabled and output_path.exists() and metrics_path.exists():
                    fold_metrics = load_json(metrics_path)
                    if "fold" not in fold_metrics:
                        fold_metrics["fold"] = fold_idx
                    fold_summaries.append(fold_metrics)
                    logger.info(
                        "Skipping completed local SLM baseline: experiment=%s model=%s fold=%s",
                        experiment_name,
                        model_type,
                        fold_idx,
                    )
                    continue

                val_eval_path = resolve_path(fold_entry["val_eval_path"])
                val_items = load_json(val_eval_path)
                logger.info(
                    "Running local SLM baseline: experiment=%s model=%s fold=%s samples=%s",
                    experiment_name,
                    model_type,
                    fold_idx,
                    len(val_items),
                )
                rows = run_fold_inference(model, tokenizer, val_items, model_type, max_new_tokens, ablation_cfg)
                for row in rows:
                    row["fold"] = fold_idx
                    row["baseline"] = "local_slm_unfinetuned"
                    row["model_type"] = model_type
                    row["model_name"] = get_model_name(model_type, cfg)
                    row["experiment"] = experiment_name
                save_json(output_path, rows)

                fold_metrics = compute_overall_metrics(rows)
                fold_metrics.update(
                    {
                        "fold": fold_idx,
                        "baseline": "local_slm_unfinetuned",
                        "model_type": model_type,
                        "model_name": get_model_name(model_type, cfg),
                        "experiment": experiment_name,
                    }
                )
                save_json(metrics_path, fold_metrics)
                fold_summaries.append(fold_metrics)
                logger.info(
                    "Local SLM baseline complete: experiment=%s model=%s fold=%s acc=%.4f macro_f1=%.4f",
                    experiment_name,
                    model_type,
                    fold_idx,
                    fold_metrics["accuracy"],
                    fold_metrics["macro_f1"],
                )

            write_full_cv_summary(results_root, model_type, experiment_name, fold_summaries)
            logger.info("Saved local SLM baseline summary to %s", results_root / "overall_summary.md")
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def model_worker(
    model_type: str,
    experiment_names: list[str],
    finetune_cfg: dict[str, Any],
    baseline_cfg: dict[str, Any],
) -> None:
    run_model_baseline(
        model_type=model_type,
        experiment_names=experiment_names,
        finetune_cfg=finetune_cfg,
        baseline_cfg=baseline_cfg,
    )


def main() -> None:
    config = load_step5_cv_config()
    baseline_cfg = config.get("local_slm_baseline", {})
    if not baseline_cfg.get("enabled", False):
        logger.info("step5_cv.local_slm_baseline.enabled is false; nothing to run.")
        return

    finetune_cfg = config["finetune"]
    experiment_names = resolve_experiment_names(baseline_cfg)
    model_names = resolve_enabled_models(baseline_cfg)

    mp.set_start_method("spawn", force=True)
    for model_type in model_names:
        process = mp.Process(
            target=model_worker,
            args=(model_type, experiment_names, finetune_cfg, baseline_cfg),
        )
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Local SLM baseline failed for model={model_type} with exit code {process.exitcode}")


if __name__ == "__main__":
    main()
