from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
BASE_EXPERIMENT_NAME = "base"
BALANCE_EXPERIMENT_NAME = "balance"
COMBINED_EXPERIMENT_NAME = "combined"

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


def resolve_experiment_name(config: dict[str, Any]) -> str:
    balance_cfg = config.get("balance", {})
    if not balance_cfg.get("enabled", False):
        return BASE_EXPERIMENT_NAME
    if balance_cfg.get("include_in_cv", False):
        return COMBINED_EXPERIMENT_NAME
    return BALANCE_EXPERIMENT_NAME


def resolve_experiment_dir(base_dir: Path, experiment_name: str) -> Path:
    if experiment_name in {"", ".", BASE_EXPERIMENT_NAME}:
        return base_dir
    return base_dir / experiment_name


def resolve_results_dir(base_dir: Path, experiment_name: str) -> Path:
    """同 resolve_experiment_dir，但 base 實驗也會建立子目錄（results/base/）。
    用於所有 results/finbert 輸出路徑，確保三個實驗各自獨立。
    """
    if experiment_name in {"", "."}:
        return base_dir
    return base_dir / experiment_name


def resolve_human_csv_path(config: dict[str, Any]) -> Path:
    human_csv = config.get("human_csv")
    if not human_csv:
        raise ValueError("Missing `step4_cv.human_csv` in config/config.yaml")
    return resolve_path(str(human_csv))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
