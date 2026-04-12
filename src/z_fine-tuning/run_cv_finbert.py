"""
Step 7.3 FinBERT baseline on 5-fold CV.

This script is fully driven by `config/config.yaml`.

Workflow:
1. Load the merged labeled training set with existing FinBERT predictions
2. Load shared 5-fold split indices
3. Export fold-wise FinBERT baseline predictions to `results/cv/finbert`
4. Optionally trigger the shared CV metrics aggregation script

Run:
  uv run src/fine-tuning/run_cv_finbert.py
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

LABELS = [
    "Climate Change",
    "Natural Capital",
    "Pollution & Waste",
    "Human Capital",
    "Product Liability",
    "Community Relations",
    "Corporate Governance",
    "Business Ethics & Values",
    "Non-ESG",
]


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    config = raw_config.get("step7", {}).get("cv_finbert")
    if not isinstance(config, dict):
        raise ValueError("Missing `step7.cv_finbert` section in config/config.yaml")

    return config


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def compute_fold_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [row["ground_truth_label"] for row in results]
    y_pred = [row["parsed_label"] for row in results]

    return {
        "samples": len(results),
        "accuracy": float(round(accuracy_score(y_true, y_pred), 4)),
        "macro_f1": float(
            round(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 4)
        ),
        "micro_f1": float(
            round(f1_score(y_true, y_pred, labels=LABELS, average="micro", zero_division=0), 4)
        ),
        "kappa": float(round(cohen_kappa_score(y_true, y_pred, labels=LABELS), 4)),
    }


def build_fold_results(
    train_full: list[dict[str, Any]],
    test_indices: list[int],
) -> list[dict[str, Any]]:
    results = []
    for idx in test_indices:
        item = train_full[idx]
        results.append(
            {
                "test_index": idx,
                "paragraph_id": item["paragraph_id"],
                "raw_output": item["finbert_label"],
                "parsed_label": item["finbert_label"],
                "ground_truth_label": item["label"],
            }
        )
    return results


def main() -> None:
    config = load_config()
    data_cfg = config.get("data", {})
    execution_cfg = config.get("execution", {})

    train_full_path = resolve_path(data_cfg["train_full_path"])
    cv_folds_path = resolve_path(data_cfg["cv_folds_path"])
    results_dir = resolve_path(data_cfg["results_dir"])

    overwrite_results = execution_cfg.get("overwrite_results", False)
    evaluate_after_run = execution_cfg.get("evaluate_after_run", True)

    if not train_full_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_full_path}")
    if not cv_folds_path.exists():
        raise FileNotFoundError(f"CV folds file not found: {cv_folds_path}")

    train_full = load_json(train_full_path)
    cv_folds = load_json(cv_folds_path)

    if not isinstance(train_full, list) or not train_full:
        raise ValueError("`train_full.json` is empty or invalid.")
    if not isinstance(cv_folds, list) or not cv_folds:
        raise ValueError("`cv_folds.json` is empty or invalid.")

    missing_finbert = [
        item["paragraph_id"]
        for item in train_full
        if not item.get("finbert_label")
    ]
    if missing_finbert:
        raise ValueError(
            f"Found {len(missing_finbert)} samples without `finbert_label`. "
            "Run Step 2 classification first."
        )

    logger.info("Loaded %s training items from %s", len(train_full), train_full_path)
    logger.info("Loaded %s CV folds from %s", len(cv_folds), cv_folds_path)

    fold_summaries = []
    for fold in cv_folds:
        fold_idx = fold["fold"]
        test_indices = fold["test_indices"]
        out_file = results_dir / f"fold_{fold_idx}_finbert.json"

        if out_file.exists() and not overwrite_results:
            logger.info("Reusing existing fold output: %s", out_file)
            results = load_json(out_file)
        else:
            results = build_fold_results(train_full, test_indices)
            save_json(out_file, results)
            logger.info("Saved fold %s results to %s", fold_idx, out_file)

        metrics = compute_fold_metrics(results)
        metrics["fold"] = fold_idx
        fold_summaries.append(metrics)
        logger.info(
            "Fold %s: acc=%.4f macro_f1=%.4f micro_f1=%.4f kappa=%.4f",
            fold_idx,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["micro_f1"],
            metrics["kappa"],
        )

    save_json(results_dir / "summary_metrics.json", fold_summaries)
    logger.info("Saved fold summary to %s", results_dir / "summary_metrics.json")

    if evaluate_after_run:
        eval_script = PROJECT_ROOT / "src" / "fine-tuning" / "evaluate_cv_metrics.py"
        logger.info("Running shared CV metrics aggregation: %s", eval_script)
        subprocess.run(
            ["uv", "run", str(eval_script)],
            cwd=str(PROJECT_ROOT),
            check=True,
        )


if __name__ == "__main__":
    main()
