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

from src.step5_cv.common import compute_overall_metrics, load_json, load_step5_cv_config, resolve_path, save_json, write_csv
from src.step5_cv.run_cv_finetune import (
    apply_model_overrides,
    find_epoch_checkpoints,
    get_ablation_cfg,
    get_inference_cfg,
    get_resume_anchor_mtime,
    get_xyz_plot_cfg,
    infer_single_fold,
    is_fold_complete,
    is_primary_process,
    log_effective_runtime_config,
    maybe_destroy_process_group,
    resolve_inference_checkpoint_dir,
    resolve_inference_run_suffix,
    resolve_variant_name,
    summarize_metric_rows,
    train_single_fold,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-fold multi-GPU fine-tuning entry")
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
        raise ValueError("run_fold_multigpu.py 不支援 dry_run")

    fold_idx = int(multi_gpu_cfg.get("fold", 0))
    run_inference_after_training = bool(multi_gpu_cfg.get("run_inference_after_training", True))
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
    results_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    sft_manifest = load_json(sft_output_dir / "manifest.json")
    fold_entries = {int(item["fold"]): item for item in sft_manifest["folds"]}

    # --- Resume logic: skip completed folds and auto-advance ---
    if resume_enabled:
        resume_anchor_mtime = get_resume_anchor_mtime(results_root, models_root)
        if is_fold_complete(results_root, models_root, fold_idx, resume_anchor_mtime):
            logger.info("Fold %s already complete, searching for next incomplete fold (resume=true)", fold_idx)
            all_fold_indices = sorted(fold_entries.keys())
            next_fold = None
            for candidate in all_fold_indices:
                if not is_fold_complete(results_root, models_root, candidate, resume_anchor_mtime):
                    next_fold = candidate
                    break
            if next_fold is None:
                logger.info("All folds are complete, nothing to run")
                return
            logger.info("Auto-advancing from fold %s to fold %s", fold_idx, next_fold)
            fold_idx = next_fold

    if fold_idx not in fold_entries:
        raise ValueError(f"Fold {fold_idx} not found in SFT manifest")
    fold_entry = fold_entries[fold_idx]

    logger.info(
        "Multi-GPU single-fold mode: model=%s fold=%s rank=%s local_rank=%s world_size=%s",
        args.model,
        fold_idx,
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

    if bool(xyz_plot_cfg.get("enabled")):
        epoch_values = xyz_plot_cfg["epochs"]
        if not epoch_values:
            raise ValueError("X/Y/Z plot is enabled but no epochs are configured in config.yaml")
        if fold_idx != 0:
            raise ValueError("run_fold_multigpu.py 的 xyz_plot 只支援 fold_0")

        sweep_models_root = models_root / "xyz_plot_epochs" / "fold_0"
        sweep_results_root = results_root / "xyz_plot_epochs" / "fold_0"
        sweep_models_root.mkdir(parents=True, exist_ok=True)
        sweep_results_root.mkdir(parents=True, exist_ok=True)

        max_epoch = max(epoch_values)
        training_run_root = sweep_models_root / f"max_epoch_{max_epoch}"
        adapter_dir = training_run_root / "adapter"
        max_epoch_train_cfg = dict(train_cfg)
        max_epoch_train_cfg["num_train_epochs"] = max_epoch

        logger.info(
            "Multi-GPU X/Y/Z plot start: model=%s fold=0 epochs=%s",
            args.model,
            epoch_values,
        )
        train_single_fold(
            model_type=args.model,
            fold_idx=fold_idx,
            train_sft_path=resolve_path(fold_entry["sft_path"]),
            adapter_dir=adapter_dir,
            train_cfg=max_epoch_train_cfg,
            finetune_cfg=cfg,
            ablation_cfg=ablation_cfg,
            dry_run_cfg=dry_run_cfg,
            keep_loaded=False,
        )

        maybe_destroy_process_group()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not is_primary_process():
            return

        epoch_checkpoint_map = find_epoch_checkpoints(training_run_root / "checkpoints")
        missing_epochs = [epoch_value for epoch_value in epoch_values if epoch_value not in epoch_checkpoint_map]
        if missing_epochs:
            raise RuntimeError(f"Missing epoch checkpoints for multi-GPU X/Y/Z plot: {missing_epochs}")

        rows = []
        max_new_tokens = int(inference_cfg["max_new_tokens_label_only"]) if ablation_cfg["label_only"] else int(inference_cfg["max_new_tokens"])
        for epoch_value in epoch_values:
            checkpoint_dir = epoch_checkpoint_map[epoch_value]
            output_path = sweep_results_root / f"epoch_{epoch_value}_results.json"
            metric_path = sweep_results_root / f"epoch_{epoch_value}_metrics.json"
            logger.info(
                "Multi-GPU X/Y/Z plot inference start: fold=0 epoch=%s checkpoint=%s",
                epoch_value,
                checkpoint_dir.name,
            )
            infer_single_fold(
                model_type=args.model,
                fold_idx=fold_idx,
                adapter_dir=checkpoint_dir,
                val_eval_path=resolve_path(fold_entry["val_eval_path"]),
                output_path=output_path,
                max_new_tokens=max_new_tokens,
                train_cfg=max_epoch_train_cfg,
                finetune_cfg=cfg,
                ablation_cfg=ablation_cfg,
                dry_run_cfg=dry_run_cfg,
            )
            fold_rows = load_json(output_path)
            metrics = compute_overall_metrics(fold_rows)
            metrics["epoch_value"] = epoch_value
            save_json(metric_path, metrics)
            rows.append(metrics)
            logger.info(
                "Multi-GPU X/Y/Z plot run complete: epoch=%s acc=%.4f macro_f1=%.4f",
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
        logger.info("Saved multi-GPU X/Y/Z plot summary to %s", sweep_results_root / "xyz_plot_summary.json")
        return

    adapter_dir = models_root / f"fold_{fold_idx}" / "adapter"
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

    maybe_destroy_process_group()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not run_inference_after_training:
        if is_primary_process():
            logger.info("Training complete for fold %s; inference skipped by config", fold_idx)
        return

    if not is_primary_process():
        return

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
    logger.info(
        "%s multi-GPU fold %s complete: acc=%.4f macro_f1=%.4f",
        args.model,
        fold_idx,
        fold_metrics["accuracy"],
        fold_metrics["macro_f1"],
    )


if __name__ == "__main__":
    main()
