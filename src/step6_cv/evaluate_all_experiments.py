"""final_v1：一次彙整 base / balance / combined 三個實驗的最終 evaluation。

對每個 (experiment, model) 讀 3 個 fold 的 metrics，輸出：
- overall Accuracy / Macro F1 / Weighted F1 / Kappa 的 mean/std
- per-class precision/recall/f1 mean/std
- pillar-level precision/recall/f1 mean/std

入口：
  uv run python src/step6_cv/evaluate_all_experiments.py
"""

from __future__ import annotations

import logging
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.fingerprint import atomic_write_json, get_run_id, load_config, load_json, now_iso, resolve_path
from src.step6_cv.common import LABELS, PILLARS, write_csv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

OVERALL = ["accuracy", "macro_f1", "weighted_f1", "kappa"]


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": round(float(statistics.mean(values)), 4),
        "std": round(float(statistics.pstdev(values)), 4) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def aggregate_model(metrics_dir: Path, model: str, folds: list[int]) -> dict[str, Any] | None:
    fold_metrics = []
    for fold in folds:
        path = metrics_dir / model / f"fold_{fold}_metrics.json"
        if not path.exists():
            logger.warning("Missing metrics: %s", path)
            return None
        fold_metrics.append(load_json(path))

    overall = {metric: mean_std([fm[metric] for fm in fold_metrics]) for metric in OVERALL}

    per_class = {}
    for label in LABELS:
        f1s, ps, rs = [], [], []
        for fm in fold_metrics:
            row = next((c for c in fm.get("per_class", []) if c["label"] == label), None)
            if row:
                f1s.append(row["f1"]); ps.append(row["precision"]); rs.append(row["recall"])
        per_class[label] = {"precision": mean_std(ps), "recall": mean_std(rs), "f1": mean_std(f1s)}

    pillar = {}
    for name in PILLARS:
        f1s, ps, rs = [], [], []
        for fm in fold_metrics:
            row = next((c for c in fm.get("pillar", []) if c["pillar"] == name), None)
            if row:
                f1s.append(row["f1"]); ps.append(row["precision"]); rs.append(row["recall"])
        pillar[name] = {"precision": mean_std(ps), "recall": mean_std(rs), "f1": mean_std(f1s)}

    return {"model": model, "folds": folds, "overall": overall, "per_class": per_class, "pillar": pillar}


def main() -> None:
    config = load_config()
    run_id = get_run_id(config)
    jm = config["step6_cv"]["job_matrix"]
    results_root = resolve_path(jm["results_root"])
    experiments = list(jm["experiments"])
    folds = [int(f) for f in jm["folds"]]
    models = sorted({m for spec in jm["machines"].values() for m in spec["models"]})

    summary: dict[str, Any] = {"stage": "step6_evaluation", "run_id": run_id, "created_at": now_iso(), "experiments": {}}
    overall_rows = []
    for experiment in experiments:
        metrics_dir = results_root / experiment
        exp_block = {}
        for model in models:
            agg = aggregate_model(metrics_dir, model, folds)
            if agg is None:
                logger.warning("Skip %s/%s (incomplete)", experiment, model)
                continue
            exp_block[model] = agg
            row = {"experiment": experiment, "model": model}
            for metric in OVERALL:
                row[f"{metric}_mean"] = agg["overall"][metric]["mean"]
                row[f"{metric}_std"] = agg["overall"][metric]["std"]
            overall_rows.append(row)
        summary["experiments"][experiment] = exp_block

    out_dir = results_root / "_evaluation"
    atomic_write_json(out_dir / "overall_summary.json", summary)
    write_csv(
        out_dir / "overall_folds.csv",
        overall_rows,
        ["experiment", "model"] + [f"{m}_{s}" for m in OVERALL for s in ("mean", "std")],
    )

    per_class_rows = []
    pillar_rows = []
    for experiment, block in summary["experiments"].items():
        for model, agg in block.items():
            for label, vals in agg["per_class"].items():
                per_class_rows.append(
                    {"experiment": experiment, "model": model, "label": label,
                     "precision_mean": vals["precision"]["mean"], "recall_mean": vals["recall"]["mean"],
                     "f1_mean": vals["f1"]["mean"], "f1_std": vals["f1"]["std"]}
                )
            for name, vals in agg["pillar"].items():
                pillar_rows.append(
                    {"experiment": experiment, "model": model, "pillar": name,
                     "precision_mean": vals["precision"]["mean"], "recall_mean": vals["recall"]["mean"],
                     "f1_mean": vals["f1"]["mean"], "f1_std": vals["f1"]["std"]}
                )
    write_csv(out_dir / "per_class_metrics.csv", per_class_rows,
              ["experiment", "model", "label", "precision_mean", "recall_mean", "f1_mean", "f1_std"])
    write_csv(out_dir / "pillar_metrics.csv", pillar_rows,
              ["experiment", "model", "pillar", "precision_mean", "recall_mean", "f1_mean", "f1_std"])
    logger.info("Evaluation written to %s (%s model summaries)", out_dir, len(overall_rows))


if __name__ == "__main__":
    main()
