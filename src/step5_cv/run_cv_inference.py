from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step5_cv.common import load_json, load_step5_cv_config, resolve_path, save_json
from src.step5_cv.run_cv_finetune import (
    apply_model_overrides,
    build_cv_summary_markdown,
    compute_overall_metrics,
    get_ablation_cfg,
    get_inference_cfg,
    get_xyz_plot_cfg,
    inference_worker,
    log_effective_runtime_config,
    resolve_inference_checkpoint_dir,
    resolve_inference_run_suffix,
    resolve_variant_name,
    summarize_fold_metrics,
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
    dry_run_cfg = cfg.get("dry_run", {})
    if bool(dry_run_cfg.get("enabled")):
        raise ValueError("run_cv_inference.py 不支援 dry_run；請直接使用 run_cv_finetune.py")
    if xyz_plot_cfg["enabled"]:
        raise ValueError("run_cv_inference.py 不支援 xyz_plot；請將 config.yaml 的 xyz_plot.enabled 設為 false")

    sft_output_dir = resolve_path(cfg["sft_output_dir"])
    results_root = resolve_path(cfg["results_root"]) / model_type
    models_root = resolve_path(cfg["models_root"]) / model_type
    variant_name = resolve_variant_name(ablation_cfg)
    if variant_name != "default":
        results_root = results_root / variant_name
        models_root = models_root / variant_name
    inference_suffix = resolve_inference_run_suffix(inference_cfg)
    if inference_suffix:
        results_root = results_root / inference_suffix
    results_root.mkdir(parents=True, exist_ok=True)

    sft_manifest = load_json(sft_output_dir / "manifest.json")
    fold_entries = sorted(sft_manifest["folds"], key=lambda item: item["fold"])
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
    fold_summaries: list[dict] = []
    for fold_entry in fold_entries:
        fold_idx = int(fold_entry["fold"])
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
                model_type,
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

        rows = load_json(output_path)
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

    write_full_cv_summary(results_root=results_root, model_type=model_type, variant_name=variant_name, fold_rows=fold_summaries)
    logger.info("Saved inference-only summary to %s", results_root / "overall_summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5 CV inference-only entry")
    parser.add_argument("--model", required=True, choices=["gemma", "gemma26b", "llama", "qwen", "ministral"])
    args = parser.parse_args()
    run_inference_only(args.model)


if __name__ == "__main__":
    main()
