from __future__ import annotations

import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_recall_fscore_support

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step4_cv.common import load_yaml_config
from src.step5_reasoning.reason import CATEGORY_DEFINITIONS, CLASSIFY_SYSTEM_PROMPT, CLASSIFY_USER_TEMPLATE


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
LABELS_SORTED = sorted(LABELS, key=len, reverse=True)

PILLAR_MAP = {
    "Climate Change": "Environmental",
    "Natural Capital": "Environmental",
    "Pollution & Waste": "Environmental",
    "Human Capital": "Social",
    "Product Liability": "Social",
    "Community Relations": "Social",
    "Corporate Governance": "Governance",
    "Business Ethics & Values": "Governance",
    "Non-ESG": "Non-ESG",
}
PILLARS = ["Environmental", "Social", "Governance", "Non-ESG"]

logger = logging.getLogger(__name__)

INFERENCE_SYSTEM_PROMPT = CLASSIFY_SYSTEM_PROMPT


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_step5_cv_config() -> dict[str, Any]:
    raw = load_yaml_config()
    config = raw.get("step5_cv")
    if not isinstance(config, dict):
        raise ValueError("Missing `step5_cv` section in config/config.yaml")
    return config


def canonicalize_label(text: str | None) -> str | None:
    if text is None:
        return None
    text = re.sub(r"\s+", " ", str(text)).strip().strip("\"'")
    if not text:
        return None
    aliases = {
        "pollution and waste": "Pollution & Waste",
        "business ethics and values": "Business Ethics & Values",
        "non esg": "Non-ESG",
    }
    lower = text.lower()
    if lower in aliases:
        return aliases[lower]
    for label in LABELS_SORTED:
        if lower == label.lower():
            return label
    return None


def extract_label(text: str | None) -> str | None:
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None

    match = re.search(r"(?im)^\s*label\s*:\s*([^\n]+)", text)
    if match:
        return canonicalize_label(match.group(1).strip(" .:*_`"))

    structured_patterns = [
        r"(?i)\*\*\s*classification\s*:\s*([^\n*]+)",
        r"(?i)\bclassification\s*:\s*([^\n*]+)",
        r"(?i)\bthe best label is\s*\*\*([^*\n]+)\*\*",
        r"(?i)\bthe best classification is\s*\*\*([^*\n]+)\*\*",
        r"(?i)^\s*\*\*([^*\n]+)\*\*",
    ]
    for pattern in structured_patterns:
        match = re.search(pattern, text)
        if match:
            candidate = canonicalize_label(match.group(1).strip(" .:*_`"))
            if candidate is not None:
                return candidate

    for label in LABELS_SORTED:
        if text == label or text.startswith(label):
            return label

    patterns = [
        r"(?i)(?:answer|label|category)\s*:\s*(.+)$",
        r"(?i)(?:classified as|category is|label is)\s*:?\s*(.+)$",
        r"(?i)(?:the correct category is)\s*:?\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = canonicalize_label(match.group(1).strip())
            if candidate is not None:
                return candidate

    tail = text[-200:]
    matched = [label for label in LABELS_SORTED if re.search(rf"(?i)\b{re.escape(label)}\b", tail)]
    if len(matched) == 1:
        return matched[0]

    matched = [label for label in LABELS_SORTED if re.search(rf"(?i)\b{re.escape(label)}\b", text)]
    matched = list(dict.fromkeys(matched))
    if len(matched) == 1:
        return matched[0]
    return None


def build_eval_messages(item: dict[str, Any], model_type: str) -> list[dict[str, Any]]:
    user_content = CLASSIFY_USER_TEMPLATE.format(
        definitions=CATEGORY_DEFINITIONS,
        text=item["combined_text"],
    )
    if model_type in {"llama", "gemma", "gemma26b"}:
        return [
            {"role": "system", "content": INFERENCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    return [
        {"role": "system", "content": [{"type": "text", "text": INFERENCE_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_content}]},
    ]


def compute_overall_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [row["ground_truth_label"] for row in rows]
    y_pred = [row["parsed_label"] if row["parsed_label"] in LABELS else "Non-ESG" for row in rows]
    return {
        "samples": len(rows),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)), 4),
        "kappa": round(float(cohen_kappa_score(y_true, y_pred, labels=LABELS)), 4),
    }


def compute_per_class_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    y_true = [row["ground_truth_label"] for row in rows]
    y_pred = [row["parsed_label"] if row["parsed_label"] in LABELS else "Non-ESG" for row in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    results = []
    for label, p, r, f, s in zip(LABELS, precision, recall, f1, support):
        results.append(
            {
                "label": label,
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
        )
    return results


def compute_pillar_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    y_true = [PILLAR_MAP[row["ground_truth_label"]] for row in rows]
    y_pred = [PILLAR_MAP.get(row["parsed_label"], "Non-ESG") for row in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=PILLARS, zero_division=0
    )
    results = []
    for pillar, p, r, f, s in zip(PILLARS, precision, recall, f1, support):
        results.append(
            {
                "pillar": pillar,
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
        )
    return results


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
