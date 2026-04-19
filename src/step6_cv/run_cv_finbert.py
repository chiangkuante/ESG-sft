from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step4_cv.common import load_step4_config, resolve_experiment_dir, resolve_experiment_name
from src.step6_cv.common import (
    compute_overall_metrics,
    load_json,
    load_step6_cv_config,
    resolve_path,
    save_json,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def main() -> None:
    config = load_step6_cv_config()
    cfg = config["finbert"]
    experiment_name = resolve_experiment_name(load_step4_config())
    step4_output_dir = resolve_experiment_dir(resolve_path(cfg["step4_output_dir"]), experiment_name)
    results_dir = resolve_experiment_dir(resolve_path(cfg["results_dir"]), experiment_name)

    manifest = load_json(step4_output_dir / "manifest.json")
    fold_summaries = []
    for fold_name in manifest["fold_directories"]:
        fold_idx = int(fold_name.split("_")[-1])
        human_val = load_json(step4_output_dir / fold_name / "human_val.json")
        results = []
        for item in human_val:
            results.append(
                {
                    "fold": fold_idx,
                    "paragraph_id": item["paragraph_id"],
                    "raw_output": item["finbert_label"],
                    "parsed_label": item["finbert_label"],
                    "ground_truth_label": item["label"],
                }
            )
        save_json(results_dir / f"fold_{fold_idx}_results.json", results)
        fold_metrics = compute_overall_metrics(results)
        fold_metrics["fold"] = fold_idx
        fold_summaries.append(fold_metrics)
        logger.info("FinBERT fold %s done: acc=%.4f macro_f1=%.4f", fold_idx, fold_metrics["accuracy"], fold_metrics["macro_f1"])

    save_json(results_dir / "overall_folds.json", fold_summaries)


if __name__ == "__main__":
    main()
