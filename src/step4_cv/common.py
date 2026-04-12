from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

ESG_CATEGORIES = [
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

LABEL_ALIASES = {
    "pollution and waste": "Pollution & Waste",
    "business ethics and values": "Business Ethics & Values",
    "non esg": "Non-ESG",
}


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_step4_config() -> dict[str, Any]:
    raw_config = load_yaml_config()
    config = raw_config.get("step4_cv")
    if not isinstance(config, dict):
        raise ValueError("Missing `step4_cv` section in config/config.yaml")
    return config


def canonicalize_label(label: str | None) -> str | None:
    if label is None:
        return None

    text = " ".join(str(label).split()).strip().strip("\"'")
    if not text:
        return None

    lower = text.lower()
    if lower in LABEL_ALIASES:
        return LABEL_ALIASES[lower]

    for category in ESG_CATEGORIES:
        if lower == category.lower():
            return category

    return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
