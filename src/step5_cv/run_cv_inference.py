from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step4_cv.common import load_step4_config, resolve_annotator_name
from src.step5_cv.common import load_json, load_step5_cv_config, resolve_path, save_json
from src.step5_cv.run_cv_finetune import (
    apply_model_overrides,
    build_cv_summary_markdown,
    compute_overall_metrics,
    find_epoch_checkpoints,
    get_ablation_cfg,
    get_checkpointing_cfg,
    get_inference_cfg,
    get_xyz_plot_cfg,
    inference_worker,
    log_effective_runtime_config,
    resolve_inference_checkpoint_dir,
    resolve_inference_run_suffix,
    resolve_variant_name,
    summarize_fold_metrics,
    summarize_metric_rows,
    write_csv,
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


def run_inference_only(model_type: str) -> None:
    config = load_step5_cv_config()
    cfg = apply_model_overrides(config["finetune"], model_type)
    train_cfg = cfg["training"]
    ablation_cfg = get_ablation_cfg(cfg)
    xyz_plot_cfg = get_xyz_plot_cfg(cfg)
    inference_cfg = get_inference_cfg(cfg)
    checkpointing_cfg = get_checkpointing_cfg(cfg)
    dry_run_cfg = cfg.get("dry_run", {})
    dry_run_enabled = bool(dry_run_cfg.get("enabled"))
    xyz_plot_enabled = bool(xyz_plot_cfg.get("enabled"))

    annotator_name = resolve_annotator_name(load_step4_config())
    sft_output_dir = resolve_path(cfg["sft_output_dir"]) / annotator_name
    results_root = resolve_path(cfg["results_root"]) / annotator_name / model_type
    models_root = resolve_path(cfg["models_root"]) / annotator_name / model_type
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

    sft_manifest = load_json(sft_output_dir / "manifest.json")
    fold_entries = sorted(sft_manifest["folds"], key=lambda item: item["fold"])
    
    if xyz_plot_enabled:
        fold_entries = [entry for entry in fold_entries if int(entry["fold"]) == 0]
        if not fold_entries:
            raise ValueError("fold_0 not found in SFT manifest for X/Y/Z plot mode")
    if dry_run_enabled:
        target_fold = int(dry_run_cfg["fold"])
        fold_entries = [entry for entry in fold_entries if int(entry["fold"]) == target_fold]
        if not fold_entries:
            raise ValueError(f"Dry run fold {target_fold} not found in SFT manifest")
        logger.info("Dry run enabled: model=%s fold=%s", model_type, target_fold)

    logger.info(
        "Inference-only mode: model=%s variant=%s strategy=%s",
        model_type,
        variant_name,
        inference_cfg["checkpoint_strategy"],
    )
    logger.info(
        "Ablation mode: label_only=%s human_only=%s synthetic_only=%s",
        ablation_cfg["label_only"],
        ablation_cfg["human_only"],
        ablation_cfg["synthetic_only"],
    )
    log_effective_runtime_config(model_type, train_cfg, cfg, inference_cfg)

    mp.set_start_method("spawn", force=True)

    if xyz_plot_enabled:
        epoch_values = xyz_plot_cfg["epochs"]
        if not epoch_values:
            raise ValueError("X/Y/Z plot is enabled but no epochs are configured in config.yaml")
        logger.info("X/Y/Z plot mode enabled: fold=0 epochs=%s", epoch_values)
        
        sweep_models_root = models_root / "xyz_plot_epochs" / "fold_0"
        sweep_results_root = results_root / "xyz_plot_epochs" / "fold_0"
        sweep_results_root.mkdir(parents=True, exist_ok=True)
        max_epoch = max(epoch_values)
        training_run_root = sweep_models_root / f"max_epoch_{max_epoch}"
        max_epoch_train_cfg = dict(train_cfg)
        max_epoch_train_cfg["num_train_epochs"] = max_epoch
        
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
                    0,
                    str(checkpoint_dir),
                    str(resolve_path(fold_entries[0]["val_eval_path"])),
                    str(output_path),
                    int(inference_cfg["max_new_tokens_label_only"]) if ablation_cfg["label_only"] else int(inference_cfg["max_new_tokens"]),
                    max_epoch_train_cfg,
                    cfg,
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
        return

    fold_summaries: list[dict] = []
    for fold_entry in fold_entries:
        fold_idx = int(fold_entry["fold"])
        output_path = results_root / f"fold_{fold_idx}_results.json"
        
        if dry_run_enabled:
            # During a dry_run, we just assume the adapter is saved under the dry_run folder without strategy mapping
            checkpoint_dir = models_root / f"fold_{fold_idx}" / "adapter"
            checkpoint_info = {
                "checkpoint_strategy": "dry_run_adapter",
                "checkpoint_epoch": None,
                "checkpoint_dir": str(checkpoint_dir)
            }
        else:
            checkpoint_dir, checkpoint_info = resolve_inference_checkpoint_dir(
                results_root=results_root,
                models_root=models_root,
                fold_idx=fold_idx,
                inference_cfg=inference_cfg,
                checkpointing_cfg=checkpointing_cfg,
            )
            
        logger.info(
            "Inference checkpoint selected for fold %s: strategy=%s epoch=%s dir=%s",
            fold_idx,
            checkpoint_info["checkpoint_strategy"],
            checkpoint_info["checkpoint_epoch"],
            checkpoint_info["checkpoint_dir"],
        )
        
        val_eval_path = resolve_path(fold_entry["val_eval_path"])
        if dry_run_enabled:
            # Re-save val items truncated based on dry_run_cfg count if needed, or we rely on inference_worker loading limits
            # Actually, inference_worker loads the full file, but wait, dry_run handles val_samples internally?
            pass
            
        infer_process = mp.Process(
            target=inference_worker,
            args=(
                model_type,
                fold_idx,
                str(checkpoint_dir),
                str(val_eval_path),
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

        rows = load_json(output_path)
        if dry_run_enabled and dry_run_cfg.get("val_samples"):
            # The inference_worker will process all val samples unless we slice it in inference_worker 
            # (which we don't, inference_worker loads original dataset, wait! run_fold_inference uses val_items)
            # Let's slice the output rows just in case, or rather wait, inference_worker doesn't truncate! Let's handle it inside inference_worker in run_cv_finetune.py. I'll modify run_cv_finetune.py inference_worker.
            pass

        fold_metrics = compute_overall_metrics(rows)
        fold_metrics["fold"] = fold_idx
        fold_metrics.update(checkpoint_info)
        save_json(results_root / f"fold_{fold_idx}_metrics.json", fold_metrics)
        fold_summaries.append(fold_metrics)
        logger.info(
            "%s fold %s inference complete: acc=%.4f macro_f1=%.4f",
            model_type,
            fold_idx,
            fold_metrics["accuracy"],
            fold_metrics["macro_f1"],
        )

    if not dry_run_enabled:
        write_full_cv_summary(results_root=results_root, model_type=model_type, variant_name=variant_name, fold_rows=fold_summaries)
        logger.info("Saved inference-only summary to %s", results_root / "overall_summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5 CV inference-only entry")
    parser.add_argument("--model", required=True, choices=["gemma", "llama", "qwen", "ministral"])
    args = parser.parse_args()
    run_inference_only(args.model)


if __name__ == "__main__":
    main()
