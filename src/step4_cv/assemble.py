from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from src.step4_cv.common import ESG_CATEGORIES, canonicalize_label, load_json, save_json


def select_seed_examples(
    human_train: list[dict[str, Any]],
    label: str,
    count: int,
    random_state: int,
) -> list[dict[str, Any]]:
    label_examples = [record for record in human_train if record["label"] == label]
    if not label_examples:
        return []

    rng = random.Random(f"{random_state}:{label}:{len(label_examples)}")
    ordered = sorted(label_examples, key=lambda item: (item["char_count"], item["paragraph_id"]))

    if len(ordered) <= count:
        chosen = ordered
    else:
        pivot = len(ordered) // 2
        candidates = ordered[max(0, pivot - count * 2): pivot + count * 2] or ordered
        chosen = rng.sample(candidates, k=count)
        chosen = sorted(chosen, key=lambda item: item["paragraph_id"])

    return [
        {
            "paragraph_id": item["paragraph_id"],
            "label": item["label"],
            "combined_text": item["combined_text"],
            "risk_heading": item.get("risk_heading", ""),
        }
        for item in chosen
    ]


def build_synthetic_requests(
    balancing_plan: dict[str, Any],
    human_train: list[dict[str, Any]],
    seed_examples_per_class: int,
    boundary_focus_labels: list[str],
    random_state: int,
    prompt_builder,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    per_label = balancing_plan["per_label"]

    for label in ESG_CATEGORIES:
        needed_count = int(per_label[label]["synthetic_needed"])
        if needed_count <= 0:
            continue

        seed_examples = select_seed_examples(
            human_train=human_train,
            label=label,
            count=seed_examples_per_class,
            random_state=random_state,
        )
        request = {
            "label": label,
            "needed_count": needed_count,
            "boundary_focus": label in set(boundary_focus_labels),
            "seed_examples": seed_examples,
        }
        request["user_prompt"] = prompt_builder(
            label=label,
            needed_count=needed_count,
            seed_examples=seed_examples,
            boundary_focus=request["boundary_focus"],
        )
        requests.append(request)

    return requests


def normalize_synthetic_record(item: dict[str, Any], fallback_label: str) -> dict[str, Any]:
    label = canonicalize_label(item.get("label")) or fallback_label
    if label is None:
        raise ValueError(f"Invalid synthetic label: {item.get('label')}")

    risk_heading = str(item.get("risk_heading") or "").strip()
    paragraph_text = str(item.get("paragraph_text") or "").strip()
    combined_text = str(item.get("combined_text") or "").strip()

    if not combined_text:
        if risk_heading and paragraph_text:
            combined_text = f"{risk_heading}\n{paragraph_text}"
        else:
            raise ValueError("Synthetic record missing combined_text and cannot be reconstructed.")

    if not paragraph_text:
        if "\n" in combined_text:
            risk_heading, paragraph_text = combined_text.split("\n", 1)
            risk_heading = risk_heading.strip()
            paragraph_text = paragraph_text.strip()
        else:
            paragraph_text = combined_text

    if not risk_heading:
        risk_heading = combined_text.split("\n", 1)[0].strip()

    return {
        "paragraph_id": str(item.get("paragraph_id") or ""),
        "risk_heading": risk_heading,
        "paragraph_text": paragraph_text,
        "combined_text": combined_text,
        "label": label,
        "char_count": len(combined_text),
        "source": "synthetic",
        "generation_notes": str(item.get("generation_notes") or "").strip(),
    }


def load_synthetic_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Synthetic data file must contain a JSON list: {path}")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Synthetic item #{index} is not an object.")
        fallback_label = canonicalize_label(item.get("label"))
        normalized_item = normalize_synthetic_record(item, fallback_label or "Non-ESG")
        if not normalized_item["paragraph_id"]:
            normalized_item["paragraph_id"] = f"SYNTH_{normalized_item['label'].replace(' ', '_')}_{index:04d}"
        normalized.append(normalized_item)

    return normalized


def ensure_placeholder_synthetic_file(path: Path) -> None:
    if not path.exists():
        save_json(path, [])


def assemble_train_pool(
    human_train: list[dict[str, Any]],
    pseudo_records: list[dict[str, Any]],
    synthetic_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    assembled: list[dict[str, Any]] = []

    for record in human_train:
        assembled.append({**record, "source_priority": 0})
    for record in pseudo_records:
        assembled.append({**record, "source_priority": 1})
    for record in synthetic_records:
        assembled.append({**record, "source_priority": 2})

    assembled.sort(key=lambda item: (item["source_priority"], item["label"], item["paragraph_id"]))
    return assembled


def build_train_pool_summary(
    train_pool: list[dict[str, Any]],
    requested_synthetic_total: int,
) -> dict[str, Any]:
    source_counts = Counter(record["source"] for record in train_pool)
    label_counts = Counter(record["label"] for record in train_pool)
    by_source_and_label: dict[str, dict[str, int]] = {}

    for source in ["human", "pseudo", "synthetic"]:
        source_records = [record for record in train_pool if record["source"] == source]
        source_label_counts = Counter(record["label"] for record in source_records)
        by_source_and_label[source] = {
            label: source_label_counts.get(label, 0) for label in ESG_CATEGORIES
        }

    actual_synthetic_total = source_counts.get("synthetic", 0)
    return {
        "train_pool_size": len(train_pool),
        "source_counts": dict(source_counts),
        "label_counts": {label: label_counts.get(label, 0) for label in ESG_CATEGORIES},
        "by_source_and_label": by_source_and_label,
        "requested_synthetic_total": requested_synthetic_total,
        "actual_synthetic_total": actual_synthetic_total,
        "synthetic_complete": actual_synthetic_total >= requested_synthetic_total,
    }
