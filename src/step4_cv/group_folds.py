"""final_v1 group-aware stratified 3-fold。

以正規化 combined_text 的 fingerprint 作為 group，確保同一文本 group 不會
同時出現在 train 與 validation。同時提供 group label 衝突偵測。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from sklearn.model_selection import StratifiedGroupKFold

from src.common.fingerprint import text_fingerprint
from src.step4_cv.common import ESG_CATEGORIES


def attach_text_fingerprint(records: list[dict[str, Any]]) -> None:
    """就地為每筆 record 加上 text_fingerprint。"""
    for record in records:
        record["text_fingerprint"] = text_fingerprint(record.get("combined_text", ""))


def detect_label_conflicts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """相同 text group 卻有互相衝突 ground-truth label 時回傳衝突清單。"""
    by_group: dict[str, set[str]] = defaultdict(set)
    members: dict[str, list[str]] = defaultdict(list)
    for record in records:
        fp = record["text_fingerprint"]
        by_group[fp].add(record["label"])
        members[fp].append(record["paragraph_id"])

    conflicts: list[dict[str, Any]] = []
    for fp, labels in by_group.items():
        if len(labels) > 1:
            conflicts.append(
                {
                    "text_fingerprint": fp,
                    "labels": sorted(labels),
                    "paragraph_ids": sorted(members[fp]),
                }
            )
    return conflicts


def create_group_aware_folds(
    records: list[dict[str, Any]],
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> list[dict[str, Any]]:
    """回傳 fold 結構（index 對應傳入 records 的順序）。"""
    if not records:
        raise ValueError("Cannot build folds from empty record set.")

    labels = [record["label"] for record in records]
    groups = [record["text_fingerprint"] for record in records]

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    dummy_x = list(range(len(records)))

    folds: list[dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(dummy_x, labels, groups)):
        train_indices = sorted(int(i) for i in train_idx)
        val_indices = sorted(int(i) for i in val_idx)
        train_dist = Counter(labels[i] for i in train_indices)
        val_dist = Counter(labels[i] for i in val_indices)
        folds.append(
            {
                "fold": fold_idx,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "train_paragraph_ids": [records[i]["paragraph_id"] for i in train_indices],
                "val_paragraph_ids": [records[i]["paragraph_id"] for i in val_indices],
                "train_distribution": {label: train_dist.get(label, 0) for label in ESG_CATEGORIES},
                "val_distribution": {label: val_dist.get(label, 0) for label in ESG_CATEGORIES},
            }
        )
    return folds


def audit_cross_fold(folds: list[dict[str, Any]], population_size: int) -> dict[str, Any]:
    """驗證 validation 覆蓋整個 population 且每筆只出現一次。"""
    seen: Counter[int] = Counter()
    for fold in folds:
        for idx in fold["val_indices"]:
            seen[idx] += 1
    duplicated = sorted(idx for idx, count in seen.items() if count > 1)
    missing = sorted(set(range(population_size)) - set(seen))
    return {
        "population_size": population_size,
        "validation_covered": len(seen),
        "duplicated_validation_indices": duplicated,
        "missing_validation_indices": missing,
        "ok": not duplicated and not missing,
    }
