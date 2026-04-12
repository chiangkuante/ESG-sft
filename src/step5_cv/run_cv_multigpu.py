from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step5_cv.common import compute_overall_metrics, load_json, load_step5_cv_config, resolve_path, save_json
from src.step5_cv.run_cv_finetune import (
    apply_model_overrides,
    build_cv_summary_markdown,
    get_ablation_cfg,
    get_inference_cfg,
    get_xyz_plot_cfg,
    infer_single_fold,
    is_primary_process,
    log_effective_runtime_config,
    maybe_destroy_process_group,
    maybe_distributed_barrier,
    resolve_inference_checkpoint_dir,
    resolve_inference_run_suffix,
    resolve_variant_name,
    summarize_fold_metrics,
    train_single_fold,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def write_full_cv_summary(results_root: Path, model_type: str, variant_name: str, fold_rows: list[dict]) -> None:
    ordered_rows = sorted(fold_rows, key=lambda row: row["fold"])
    summary = summarize_fold_metrics(ordered_rows)
    save_json(results_root / "overall_folds.json", ordered_rows)
    save_json(results_root / "overall_summary.json", {"folds": ordered_rows, "summary": summary})
    markdown = build_cv_summary_markdown(model_type, variant_name, ordered_rows, summary)
    (results_root / "overall_summary.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="3-fold multi-GPU fine-tuning entry")
    parser.add_argument("--model", required=True, choices=["gemma", "gemma26b", "llama", "qwen", "ministral"])
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    config = load_step5_cv_config()
    cfg = apply_model_overrides(config["finetune"], args.model)
    train_cfg = cfg["training"]
    ablation_cfg = get_ablation_cfg(cfg)
    xyz_plot_cfg = get_xyz_plot_cfg(cfg)
    inference_cfg = get_inference_cfg(cfg)
    dry_run_cfg = cfg.get("dry_run", {})
    multi_gpu_cfg = cfg.get("multi_gpu", {})

    if bool(dry_run_cfg.get("enabled")):
        raise ValueError("run_cv_multigpu.py 不支援 dry_run")
    if bool(xyz_plot_cfg.get("enabled")):
        raise ValueError("run_cv_multigpu.py 不支援 xyz_plot")

    run_inference_after_training = bool(multi_gpu_cfg.get("run_inference_after_training", True))

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
    results_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    sft_manifest = load_json(sft_output_dir / "manifest.json")
    fold_entries = sorted(sft_manifest["folds"], key=lambda item: int(item["fold"]))

    logger.info(
        "Multi-GPU 3-fold mode: model=%s rank=%s local_rank=%s world_size=%s",
        args.model,
        os.environ.get("RANK", "0"),
        os.environ.get("LOCAL_RANK", "0"),
        os.environ.get("WORLD_SIZE", "1"),
    )
    logger.info(
        "Ablation mode: label_only=%s human_only=%s synthetic_only=%s variant=%s",
        ablation_cfg["label_only"],
        ablation_cfg["human_only"],
        ablation_cfg["synthetic_only"],
        variant_name,
    )
    log_effective_runtime_config(args.model, train_cfg, cfg, inference_cfg)

    fold_summaries: list[dict] = []
    for fold_entry in fold_entries:
        fold_idx = int(fold_entry["fold"])
        adapter_dir = models_root / f"fold_{fold_idx}" / "adapter"
        logger.info("Starting multi-GPU training for model=%s fold=%s", args.model, fold_idx)
        train_single_fold(
            model_type=args.model,
            fold_idx=fold_idx,
            train_sft_path=resolve_path(fold_entry["sft_path"]),
            adapter_dir=adapter_dir,
            train_cfg=train_cfg,
            finetune_cfg=cfg,
            ablation_cfg=ablation_cfg,
            dry_run_cfg=dry_run_cfg,
            keep_loaded=False,
        )
        maybe_distributed_barrier()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not run_inference_after_training:
            if is_primary_process():
                logger.info("Training complete for fold %s; inference skipped by config", fold_idx)
            continue

        if is_primary_process():
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
            output_path = results_root / f"fold_{fold_idx}_results.json"
            max_new_tokens = int(inference_cfg["max_new_tokens_label_only"]) if ablation_cfg["label_only"] else int(inference_cfg["max_new_tokens"])
            infer_single_fold(
                model_type=args.model,
                fold_idx=fold_idx,
                adapter_dir=checkpoint_dir,
                val_eval_path=resolve_path(fold_entry["val_eval_path"]),
                output_path=output_path,
                max_new_tokens=max_new_tokens,
                train_cfg=train_cfg,
                finetune_cfg=cfg,
                ablation_cfg=ablation_cfg,
                dry_run_cfg=dry_run_cfg,
            )
            rows = load_json(output_path)
            fold_metrics = compute_overall_metrics(rows)
            fold_metrics["fold"] = fold_idx
            fold_metrics.update(checkpoint_info)
            save_json(results_root / f"fold_{fold_idx}_metrics.json", fold_metrics)
            fold_summaries.append(fold_metrics)
            logger.info(
                "%s multi-GPU fold %s complete: acc=%.4f macro_f1=%.4f",
                args.model,
                fold_idx,
                fold_metrics["accuracy"],
                fold_metrics["macro_f1"],
            )
        maybe_distributed_barrier()

    if is_primary_process() and run_inference_after_training and fold_summaries:
        write_full_cv_summary(results_root=results_root, model_type=args.model, variant_name=variant_name, fold_rows=fold_summaries)
        logger.info("Saved multi-GPU full CV summary to %s", results_root / "overall_summary.md")

    maybe_destroy_process_group()


if __name__ == "__main__":
    main()
