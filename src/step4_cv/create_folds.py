from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sklearn.model_selection import StratifiedKFold

from src.step4_cv.common import ESG_CATEGORIES, save_json


def create_cv_folds(
    human_records: list[dict[str, Any]],
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> list[dict[str, Any]]:
    labels = [record["label"] for record in human_records]
    counts = Counter(labels)
    frequent_indices = [index for index, label in enumerate(labels) if counts[label] >= n_splits]
    rare_indices = [index for index, label in enumerate(labels) if counts[label] < n_splits]

    fold_val_indices: list[list[int]] = [[] for _ in range(n_splits)]

    if frequent_indices:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        frequent_labels = [labels[index] for index in frequent_indices]
        dummy_x = list(range(len(frequent_indices)))

        for fold_idx, (_, val_idx) in enumerate(splitter.split(dummy_x, frequent_labels)):
            fold_val_indices[fold_idx].extend(frequent_indices[index] for index in val_idx.tolist())

    if rare_indices:
        rng = random.Random(random_state)
        grouped_rare: dict[str, list[int]] = {}
        for index in rare_indices:
            grouped_rare.setdefault(labels[index], []).append(index)

        for label in ESG_CATEGORIES:
            label_indices = grouped_rare.get(label, [])
            if not label_indices:
                continue
            if shuffle:
                rng.shuffle(label_indices)
            for offset, record_index in enumerate(label_indices):
                fold_idx = offset % n_splits
                fold_val_indices[fold_idx].append(record_index)

    all_indices = set(range(len(human_records)))
    folds: list[dict[str, Any]] = []
    for fold_idx, val_bucket in enumerate(fold_val_indices):
        val_indices = sorted(val_bucket)
        train_indices = sorted(all_indices - set(val_indices))
        train_dist = Counter(labels[index] for index in train_indices)
        val_dist = Counter(labels[index] for index in val_indices)

        folds.append(
            {
                "fold": fold_idx,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "test_indices": val_indices,
                "train_paragraph_ids": [human_records[index]["paragraph_id"] for index in train_indices],
                "val_paragraph_ids": [human_records[index]["paragraph_id"] for index in val_indices],
                "train_distribution": {label: train_dist.get(label, 0) for label in ESG_CATEGORIES},
                "val_distribution": {label: val_dist.get(label, 0) for label in ESG_CATEGORIES},
            }
        )

    return folds


def create_and_save_folds(
    human_records: list[dict[str, Any]],
    output_path,
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> list[dict[str, Any]]:
    folds = create_cv_folds(
        human_records=human_records,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    save_json(output_path, folds)
    return folds
