from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step4_cv.common import ESG_CATEGORIES, load_json, load_step4_config, resolve_annotator_name, resolve_path, save_json
from src.step4_cv.assemble import (
    assemble_train_pool,
    build_synthetic_requests,
    build_train_pool_summary,
    ensure_placeholder_synthetic_file,
    load_synthetic_records,
)
from src.step4_cv.create_folds import create_and_save_folds
from src.step4_cv.data import count_by_label, load_balance_records, load_classified_pool, load_human_annotations
from src.step4_cv.prompts import build_synthetic_user_prompt

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)


def subset_by_indices(records: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    return [records[index] for index in indices]


def evaluate_threshold(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    accepted = [record for record in records if record["finbert_confidence"] >= threshold]
    if not accepted:
        return {
            "threshold": round(threshold, 4),
            "accepted": 0,
            "correct": 0,
            "accuracy": None,
            "coverage": 0.0,
        }

    correct = sum(1 for record in accepted if record["finbert_label"] == record["label"])
    return {
        "threshold": round(threshold, 4),
        "accepted": len(accepted),
        "correct": correct,
        "accuracy": round(correct / len(accepted), 4),
        "coverage": round(len(accepted) / len(records), 4),
    }


def calibrate_threshold(
    records: list[dict[str, Any]],
    target_accuracy: float,
    min_samples: int,
) -> dict[str, Any]:
    if not records:
        return {
            "threshold": 1.0001,
            "accepted": 0,
            "correct": 0,
            "accuracy": None,
            "coverage": 0.0,
            "meets_target": False,
            "candidate_count": 0,
        }

    candidate_thresholds = sorted({record["finbert_confidence"] for record in records}, reverse=True)

    qualified: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    for threshold in candidate_thresholds:
        metrics = evaluate_threshold(records, threshold)
        if metrics["accepted"] < min_samples:
            continue
        all_candidates.append(metrics)
        if metrics["accuracy"] is not None and metrics["accuracy"] >= target_accuracy:
            qualified.append(metrics)

    if qualified:
        best = max(qualified, key=lambda item: (item["accepted"], -item["threshold"]))
        best["meets_target"] = True
        best["candidate_count"] = len(qualified)
        return best

    if all_candidates:
        fallback = max(all_candidates, key=lambda item: (item["accuracy"], item["accepted"], -item["threshold"]))
        fallback["meets_target"] = False
        fallback["candidate_count"] = len(all_candidates)
        return fallback

    max_confidence = max(record["finbert_confidence"] for record in records)
    return {
        "threshold": round(min(1.0001, max_confidence + 0.0001), 4),
        "accepted": 0,
        "correct": 0,
        "accuracy": None,
        "coverage": 0.0,
        "meets_target": False,
        "candidate_count": 0,
    }


def calibrate_thresholds_by_class(
    train_records: list[dict[str, Any]],
    target_accuracy: float,
    min_samples_global: int,
    min_samples_per_class: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    global_threshold = calibrate_threshold(
        records=train_records,
        target_accuracy=target_accuracy,
        min_samples=min_samples_global,
    )

    thresholds: dict[str, dict[str, Any]] = {}
    for label in ESG_CATEGORIES:
        label_records = [record for record in train_records if record["finbert_label"] == label]
        if len(label_records) < min_samples_per_class:
            thresholds[label] = {
                **global_threshold,
                "source": "global_fallback",
                "label_support": len(label_records),
            }
            continue

        class_threshold = calibrate_threshold(
            records=label_records,
            target_accuracy=target_accuracy,
            min_samples=min_samples_per_class,
        )
        thresholds[label] = {
            **class_threshold,
            "source": "per_class" if class_threshold["meets_target"] else "per_class_relaxed",
            "label_support": len(label_records),
        }

    return global_threshold, thresholds


def select_pseudo_labels(
    unlabeled_pool: list[dict[str, Any]],
    thresholds_by_label: dict[str, dict[str, Any]],
    base_counts: dict[str, int],
    cap_by_label: dict[str, int],
    total_target: int | None,
    allocation_alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_counts = Counter()

    for record in unlabeled_pool:
        predicted_label = record["finbert_label"]
        threshold_info = thresholds_by_label[predicted_label]
        threshold = threshold_info["threshold"]

        if record["finbert_confidence"] < threshold:
            rejected_counts[predicted_label] += 1
            continue

        grouped[predicted_label].append(
            {
                **record,
                "label": predicted_label,
                "source": "pseudo",
                "pseudo_threshold": threshold,
                "threshold_source": threshold_info["source"],
            }
        )

    raw_counts: dict[str, int] = {}
    capacities: dict[str, int] = {}

    for label in ESG_CATEGORIES:
        candidates = sorted(
            grouped.get(label, []),
            key=lambda item: (-item["finbert_confidence"], item["paragraph_id"]),
        )
        raw_counts[label] = len(candidates)
        cap = cap_by_label.get(label)
        capacities[label] = min(len(candidates), cap) if cap is not None else len(candidates)

    selected_counts = allocate_balanced_pseudo_quota(
        base_counts=base_counts,
        capacities=capacities,
        total_target=total_target,
        alpha=allocation_alpha,
    )

    selected: list[dict[str, Any]] = []
    for label in ESG_CATEGORIES:
        candidates = sorted(
            grouped.get(label, []),
            key=lambda item: (-item["finbert_confidence"], item["paragraph_id"]),
        )
        selected.extend(candidates[: selected_counts[label]])

    summary = {
        "raw_selected_counts": raw_counts,
        "capacities_after_caps": capacities,
        "selected_counts": selected_counts,
        "rejected_below_threshold": {label: rejected_counts.get(label, 0) for label in ESG_CATEGORIES},
        "selected_total": len(selected),
        "selection_budget": total_target,
        "allocation_alpha": allocation_alpha,
    }
    return selected, summary


def allocate_balanced_pseudo_quota(
    base_counts: dict[str, int],
    capacities: dict[str, int],
    total_target: int | None,
    alpha: float,
) -> dict[str, int]:
    if total_target is None:
        return {label: capacities.get(label, 0) for label in ESG_CATEGORIES}

    total_capacity = sum(capacities.get(label, 0) for label in ESG_CATEGORIES)
    if total_capacity <= total_target:
        return {label: capacities.get(label, 0) for label in ESG_CATEGORIES}

    weights = {
        label: math.pow(max(base_counts.get(label, 0), 0), alpha) if capacities.get(label, 0) > 0 else 0.0
        for label in ESG_CATEGORIES
    }
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        raise ValueError("Pseudo-label allocation weights are all zero.")

    raw_targets = {
        label: total_target * weights[label] / weight_sum
        for label in ESG_CATEGORIES
    }
    quotas = {
        label: min(capacities.get(label, 0), int(math.floor(raw_targets[label])))
        for label in ESG_CATEGORIES
    }

    remaining = total_target - sum(quotas.values())
    while remaining > 0:
        candidates = [
            label
            for label in ESG_CATEGORIES
            if quotas[label] < capacities.get(label, 0)
        ]
        if not candidates:
            break

        candidates.sort(
            key=lambda label: (
                raw_targets[label] - quotas[label],
                weights[label],
                capacities[label] - quotas[label],
            ),
            reverse=True,
        )

        progress = False
        for label in candidates:
            if quotas[label] >= capacities[label]:
                continue
            quotas[label] += 1
            remaining -= 1
            progress = True
            if remaining == 0:
                break

        if not progress:
            break

    return quotas


def compute_balancing_plan(
    human_train_records: list[dict[str, Any]],
    pseudo_records: list[dict[str, Any]],
    alpha: float,
    total_budget: int,
) -> dict[str, Any]:
    merged_counts = Counter()
    human_counts = count_by_label(human_train_records)
    pseudo_counts = count_by_label(pseudo_records)

    for label in ESG_CATEGORIES:
        merged_counts[label] = human_counts[label] + pseudo_counts[label]

    denominator = sum(math.pow(max(merged_counts[label], 0), alpha) for label in ESG_CATEGORIES if merged_counts[label] > 0)
    if denominator == 0:
        raise ValueError("Merged training counts are all zero; cannot compute balancing plan.")

    plan: dict[str, Any] = {}
    total_target = 0
    total_needed = 0
    for label in ESG_CATEGORIES:
        current = merged_counts[label]
        if current <= 0:
            target = 0
        else:
            smoothed = total_budget * math.pow(current, alpha) / denominator
            target = max(current, int(round(smoothed)))
        synthetic_needed = max(0, target - current)
        total_target += target
        total_needed += synthetic_needed
        plan[label] = {
            "human_count": human_counts[label],
            "pseudo_count": pseudo_counts[label],
            "current_count": current,
            "target_count": target,
            "synthetic_needed": synthetic_needed,
        }

    return {
        "alpha": alpha,
        "total_budget": total_budget,
        "effective_target_total": total_target,
        "effective_synthetic_total": total_needed,
        "per_label": plan,
    }


def prepare_fold_outputs(
    output_dir: Path,
    fold: dict[str, Any],
    human_records: list[dict[str, Any]],
    unlabeled_pool: list[dict[str, Any]],
    config: dict[str, Any],
    balance_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    thresholds_cfg = config["thresholds"]
    pseudo_cfg = config["pseudo_labels"]
    balancing_cfg = config["balancing"]
    synthetic_cfg = config["synthetic_generation"]
    pseudo_enabled = bool(pseudo_cfg.get("enabled", True))
    synthetic_enabled = bool(synthetic_cfg.get("enabled", True))

    fold_idx = fold["fold"]
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    human_train = subset_by_indices(human_records, fold["train_indices"])
    human_val = subset_by_indices(human_records, fold["val_indices"])

    human_train_counts = count_by_label(human_train)

    if pseudo_enabled:
        global_threshold, thresholds_by_label = calibrate_thresholds_by_class(
            train_records=human_train,
            target_accuracy=float(thresholds_cfg["target_accuracy"]),
            min_samples_global=int(thresholds_cfg["min_samples_global"]),
            min_samples_per_class=int(thresholds_cfg["min_samples_per_class"]),
        )
        pseudo_records, pseudo_summary = select_pseudo_labels(
            unlabeled_pool=unlabeled_pool,
            thresholds_by_label=thresholds_by_label,
            base_counts=human_train_counts,
            cap_by_label={label: int(cap) for label, cap in pseudo_cfg.get("caps", {}).items()},
            total_target=(
                int(pseudo_cfg["total_target"])
                if pseudo_cfg.get("total_target") is not None
                else None
            ),
            allocation_alpha=float(pseudo_cfg.get("allocation_alpha", balancing_cfg["alpha"])),
        )
    else:
        global_threshold = None
        thresholds_by_label = {}
        pseudo_records = []
        pseudo_summary = {"selected_total": 0, "skipped": "pseudo_labels disabled"}

    if pseudo_enabled and synthetic_enabled:
        balancing_plan = compute_balancing_plan(
            human_train_records=human_train,
            pseudo_records=pseudo_records,
            alpha=float(balancing_cfg["alpha"]),
            total_budget=int(balancing_cfg["total_budget"]),
        )
        synthetic_requests = build_synthetic_requests(
            balancing_plan=balancing_plan,
            human_train=human_train,
            seed_examples_per_class=int(synthetic_cfg["seed_examples_per_class"]),
            boundary_focus_labels=list(synthetic_cfg.get("boundary_focus_labels", [])),
            random_state=int(config["folds"]["random_state"]) + fold_idx,
            prompt_builder=build_synthetic_user_prompt,
        )
        synthetic_data_path = fold_dir / str(synthetic_cfg["output_filename"])
        ensure_placeholder_synthetic_file(synthetic_data_path)
        synthetic_records = load_synthetic_records(synthetic_data_path)
    else:
        balancing_plan = {"effective_synthetic_total": 0, "skipped": "disabled"}
        synthetic_requests = []
        synthetic_records = []

    # balance records 作為獨立資料來源加入 train_pool（不參與 fold split）
    extra_balance = balance_records if balance_records else []

    train_pool = assemble_train_pool(
        human_train=human_train,
        pseudo_records=pseudo_records,
        synthetic_records=synthetic_records,
    )
    if extra_balance:
        train_pool = train_pool + extra_balance
        logger.info("Fold %s: added %s balance records to train_pool", fold_idx, len(extra_balance))

    train_pool_summary = build_train_pool_summary(
        train_pool=train_pool,
        requested_synthetic_total=int(balancing_plan.get("effective_synthetic_total", 0)),
    )

    save_json(fold_dir / "human_train.json", human_train)
    save_json(fold_dir / "human_val.json", human_val)
    save_json(fold_dir / "pseudo_labels.json", pseudo_records)
    save_json(fold_dir / "synthetic_requests.json", synthetic_requests)
    save_json(fold_dir / "train_pool.json", train_pool)
    save_json(fold_dir / "train_pool_summary.json", train_pool_summary)

    summary = {
        "fold": fold_idx,
        "human_train_size": len(human_train),
        "human_val_size": len(human_val),
        "balance_records_size": len(extra_balance),
        "human_train_distribution": human_train_counts,
        "human_val_distribution": count_by_label(human_val),
        "global_threshold": global_threshold,
        "thresholds_by_label": thresholds_by_label,
        "pseudo_label_summary": pseudo_summary,
        "balancing_plan": balancing_plan,
        "synthetic_request_count": len(synthetic_requests),
        "synthetic_data_path": synthetic_cfg.get("output_filename"),
        "train_pool_summary": train_pool_summary,
    }
    save_json(fold_dir / "summary.json", summary)
    return summary


def main() -> None:
    config = load_step4_config()

    paths_cfg = config["paths"]
    folds_cfg = config["folds"]
    annotator_name = resolve_annotator_name(config)
    balance_cfg = config.get("balance", {})
    pseudo_enabled = bool(config.get("pseudo_labels", {}).get("enabled", True))

    # 從 annotators 對照表取得 CSV 路徑
    annotator_key = config.get("annotator", "p1")
    annotators_cfg = config.get("annotators", {})
    if annotator_key not in annotators_cfg:
        raise ValueError(f"Unknown annotator '{annotator_key}' — check step4_cv.annotators in config.yaml")
    human_csv_path = resolve_path(annotators_cfg[annotator_key]["csv"])

    output_dir = resolve_path(paths_cfg["output_dir"]) / annotator_name
    classified_path = resolve_path(paths_cfg["classified_json"])
    folds_path = output_dir / "cv_folds.json"

    logger.info("Annotator: %s (annotator_name=%s)", annotator_key, annotator_name)

    # 載入人工標註
    human_records = load_human_annotations(human_csv_path)
    logger.info("Loaded %s human-labeled records from %s", len(human_records), human_csv_path)

    # 載入 balance 資料集
    balance_records: list[dict[str, Any]] = []
    balance_in_cv = False
    if balance_cfg.get("enabled", False):
        balance_csv_path = resolve_path(balance_cfg["csv"])
        balance_records = load_balance_records(balance_csv_path)
        balance_in_cv = bool(balance_cfg.get("include_in_cv", False))
        logger.info("Loaded %s balance records from %s (include_in_cv=%s)", len(balance_records), balance_csv_path, balance_in_cv)

    # 實驗3：合併後再 split
    fold_human_records = human_records
    if balance_in_cv and balance_records:
        fold_human_records = human_records + balance_records
        logger.info("Combined human + balance records for CV split: %s total", len(fold_human_records))

    # 載入 FinBERT pool（偽標籤啟用時才需要）
    if pseudo_enabled:
        classified_pool = load_classified_pool(load_json(classified_path))
        logger.info("Loaded %s FinBERT pool records from %s", len(classified_pool), classified_path)
    else:
        classified_pool = []
        logger.info("Pseudo-labels disabled, skipping FinBERT pool loading")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not folds_path.exists():
        folds = create_and_save_folds(
            human_records=fold_human_records,
            output_path=folds_path,
            n_splits=int(folds_cfg["n_splits"]),
            shuffle=bool(folds_cfg.get("shuffle", True)),
            random_state=int(folds_cfg["random_state"]),
        )
        logger.info("Created %s CV folds at %s", len(folds), folds_path)
    else:
        folds = load_json(folds_path)
        logger.info("Loaded %s existing CV folds from %s", len(folds), folds_path)

    human_paragraph_ids = {record["paragraph_id"] for record in fold_human_records}
    unlabeled_pool = [record for record in classified_pool if record["paragraph_id"] not in human_paragraph_ids]
    if pseudo_enabled:
        logger.info("Filtered unlabeled pool down to %s records", len(unlabeled_pool))

    # 實驗1：balance records 不參與 split，全部加入 train
    extra_balance = balance_records if (balance_cfg.get("enabled", False) and not balance_in_cv) else None

    fold_summaries = []
    for fold in folds:
        fold_summary = prepare_fold_outputs(
            output_dir=output_dir,
            fold=fold,
            human_records=fold_human_records,
            unlabeled_pool=unlabeled_pool,
            config=config,
            balance_records=extra_balance,
        )
        fold_summaries.append(fold_summary)
        logger.info(
            "Fold %s ready: human_train=%s human_val=%s balance=%s pseudo=%s synthetic_needed=%s",
            fold_summary["fold"],
            fold_summary["human_train_size"],
            fold_summary["human_val_size"],
            fold_summary.get("balance_records_size", 0),
            fold_summary["pseudo_label_summary"].get("selected_total", 0),
            fold_summary["balancing_plan"].get("effective_synthetic_total", 0),
        )

    manifest = {
        "annotator": annotator_key,
        "annotator_name": annotator_name,
        "folds_path": str(folds_path.relative_to(Path.cwd())) if folds_path.is_relative_to(Path.cwd()) else str(folds_path),
        "human_records": len(human_records),
        "balance_records": len(balance_records),
        "classified_pool_records": len(classified_pool),
        "unlabeled_pool_records": len(unlabeled_pool),
        "n_folds": len(fold_summaries),
        "fold_directories": [f"fold_{summary['fold']}" for summary in fold_summaries],
    }
    save_json(output_dir / "manifest.json", manifest)
    logger.info("Saved step4 manifest to %s", output_dir / "manifest.json")


if __name__ == "__main__":
    main()
