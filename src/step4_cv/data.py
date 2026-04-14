from __future__ import annotations

import csv
from collections import Counter
import logging
from typing import Any

from src.step4_cv.common import ESG_CATEGORIES, canonicalize_label

logger = logging.getLogger(__name__)


def load_human_annotations(csv_path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped_unlabeled = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = canonicalize_label(row.get("label"))
            finbert_label = canonicalize_label(row.get("finbert_label"))
            if label is None:
                if not (row.get("label") or "").strip():
                    skipped_unlabeled += 1
                    continue
                raise ValueError(f"Invalid human label for paragraph_id={row.get('paragraph_id')}: {row.get('label')}")
            if finbert_label is None:
                raise ValueError(
                    f"Invalid FinBERT label for paragraph_id={row.get('paragraph_id')}: {row.get('finbert_label')}"
                )

            record = {
                "annotation_id": int(row["annotation_id"]) if row.get("annotation_id") else None,
                "paragraph_id": row["paragraph_id"],
                "ticker": row.get("ticker", ""),
                "filing_date": row.get("filing_date", ""),
                "combined_text": row.get("combined_text", ""),
                "risk_heading": row.get("risk_heading", ""),
                "risk_section": row.get("risk_section", ""),
                "label": label,
                "finbert_label": finbert_label,
                "finbert_confidence": float(row.get("finbert_confidence") or 0.0),
                "char_count": int(float(row["char_count"])) if row.get("char_count") else 0,
                "source": "human",
            }
            records.append(record)

    if skipped_unlabeled:
        logger.warning("Skipped %s unlabeled rows from %s", skipped_unlabeled, csv_path)

    validate_label_support(records)
    return records


def load_balance_records(csv_path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped_unlabeled = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = canonicalize_label(row.get("label"))
            finbert_label = canonicalize_label(row.get("finbert_label"))
            if label is None:
                if not (row.get("label") or "").strip():
                    skipped_unlabeled += 1
                    continue
                raise ValueError(f"Invalid balance label for paragraph_id={row.get('paragraph_id')}: {row.get('label')}")
            if finbert_label is None:
                raise ValueError(
                    f"Invalid FinBERT label for paragraph_id={row.get('paragraph_id')}: {row.get('finbert_label')}"
                )

            record = {
                "annotation_id": int(row["annotation_id"]) if row.get("annotation_id") else None,
                "paragraph_id": row["paragraph_id"],
                "ticker": row.get("ticker", ""),
                "filing_date": row.get("filing_date", ""),
                "combined_text": row.get("combined_text", ""),
                "risk_heading": row.get("risk_heading", ""),
                "risk_section": row.get("risk_section", ""),
                "label": label,
                "finbert_label": finbert_label,
                "finbert_confidence": float(row.get("finbert_confidence") or 0.0),
                "char_count": int(float(row["char_count"])) if row.get("char_count") else 0,
                "source": "balance",
            }
            records.append(record)

    if skipped_unlabeled:
        logger.warning("Skipped %s unlabeled rows from %s", skipped_unlabeled, csv_path)

    return records


def load_classified_pool(json_payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in json_payload:
        label = canonicalize_label(item.get("finbert_label"))
        if label is None:
            raise ValueError(f"Invalid FinBERT pool label for paragraph_id={item.get('paragraph_id')}: {item.get('finbert_label')}")

        cleaned.append(
            {
                "paragraph_id": item["paragraph_id"],
                "ticker": item.get("ticker", ""),
                "filing_date": item.get("filing_date", ""),
                "combined_text": item.get("combined_text", ""),
                "risk_heading": item.get("risk_heading", ""),
                "risk_section": item.get("risk_section", ""),
                "finbert_label": label,
                "finbert_confidence": float(item.get("finbert_confidence") or 0.0),
                "char_count": int(item.get("char_count") or 0),
            }
        )

    return cleaned


def validate_label_support(records: list[dict[str, Any]]) -> None:
    label_counts = Counter(record["label"] for record in records)
    missing = [label for label in ESG_CATEGORIES if label_counts.get(label, 0) == 0]
    if missing:
        raise ValueError(f"Missing human-labeled classes: {missing}")


def count_by_label(records: list[dict[str, Any]], label_key: str = "label") -> dict[str, int]:
    counter = Counter()
    for record in records:
        counter[record[label_key]] += 1
    return {label: counter.get(label, 0) for label in ESG_CATEGORIES}
