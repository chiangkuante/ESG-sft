from __future__ import annotations

import logging
import statistics
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step4_cv.common import (
    load_step4_config,
    resolve_experiment_name,
    resolve_results_dir,
)
from src.step6_cv.common import (
    compute_overall_metrics,
    compute_per_class_metrics,
    compute_pillar_metrics,
    load_json,
    load_step6_cv_config,
    resolve_path,
    save_json,
    write_csv,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def summarize_overall(rows: list[dict]) -> dict:
    metrics = ["accuracy", "macro_f1", "weighted_f1", "kappa"]
    summary = {}
    for metric in metrics:
        values = [row[metric] for row in rows]
        summary[f"{metric}_mean"] = round(float(statistics.mean(values)), 4)
        summary[f"{metric}_std"] = round(float(statistics.pstdev(values)), 4)
    return summary


def main() -> None:
    config = load_step6_cv_config()
    eval_cfg = config["evaluation"]
    experiment_name = resolve_experiment_name(load_step4_config())
    results_root = resolve_results_dir(resolve_path(eval_cfg["results_root"]), experiment_name)

    model_dirs = [path for path in results_root.iterdir() if path.is_dir()]
    for model_dir in sorted(model_dirs):
        if model_dir.name == "xyz_plot_epochs":
            continue
        fold_files = sorted(model_dir.glob("fold_*_results.json"))
        if not fold_files:
            nested_variant_dirs = [path for path in model_dir.iterdir() if path.is_dir()]
            for variant_dir in sorted(nested_variant_dirs):
                if variant_dir.name == "xyz_plot_epochs":
                    continue
                fold_files = sorted(variant_dir.glob("fold_*_results.json"))
                if not fold_files:
                    continue

                fold_overall = []
                per_class_rows = []
                pillar_rows = []

                for fold_file in fold_files:
                    fold_results = load_json(fold_file)
                    fold_idx = int(fold_file.stem.split("_")[1])
                    overall = compute_overall_metrics(fold_results)
                    overall["fold"] = fold_idx
                    fold_overall.append(overall)

                    for row in compute_per_class_metrics(fold_results):
                        per_class_rows.append({"fold": fold_idx, **row})
                    for row in compute_pillar_metrics(fold_results):
                        pillar_rows.append({"fold": fold_idx, **row})

                summary = summarize_overall(fold_overall)
                save_json(variant_dir / "overall_summary.json", {"folds": fold_overall, "summary": summary})
                write_csv(
                    variant_dir / "overall_folds.csv",
                    fold_overall,
                    ["fold", "samples", "accuracy", "macro_f1", "weighted_f1", "kappa"],
                )
                write_csv(
                    variant_dir / "per_class_metrics.csv",
                    per_class_rows,
                    ["fold", "label", "precision", "recall", "f1", "support"],
                )
                write_csv(
                    variant_dir / "pillar_metrics.csv",
                    pillar_rows,
                    ["fold", "pillar", "precision", "recall", "f1", "support"],
                )
                logger.info("Evaluated %s/%s", model_dir.name, variant_dir.name)
            continue

        fold_overall = []
        per_class_rows = []
        pillar_rows = []

        for fold_file in fold_files:
            fold_results = load_json(fold_file)
            fold_idx = int(fold_file.stem.split("_")[1])
            overall = compute_overall_metrics(fold_results)
            overall["fold"] = fold_idx
            fold_overall.append(overall)

            for row in compute_per_class_metrics(fold_results):
                per_class_rows.append({"fold": fold_idx, **row})
            for row in compute_pillar_metrics(fold_results):
                pillar_rows.append({"fold": fold_idx, **row})

        summary = summarize_overall(fold_overall)
        save_json(model_dir / "overall_summary.json", {"folds": fold_overall, "summary": summary})
        write_csv(
            model_dir / "overall_folds.csv",
            fold_overall,
            ["fold", "samples", "accuracy", "macro_f1", "weighted_f1", "kappa"],
        )
        write_csv(
            model_dir / "per_class_metrics.csv",
            per_class_rows,
            ["fold", "label", "precision", "recall", "f1", "support"],
        )
        write_csv(
            model_dir / "pillar_metrics.csv",
            pillar_rows,
            ["fold", "pillar", "precision", "recall", "f1", "support"],
        )
        logger.info("Evaluated %s", model_dir.name)


if __name__ == "__main__":
    main()
