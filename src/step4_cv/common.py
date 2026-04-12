from __future__ import annotations

from copy import deepcopy
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


def _apply_path_override(target: dict[str, Any], keys: list[str], value: Any) -> None:
    current = target
    for key in keys[:-1]:
        node = current.get(key)
        if not isinstance(node, dict):
            node = {}
            current[key] = node
        current = node
    current[keys[-1]] = value


def apply_active_data_profile(raw_config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(raw_config)
    active_profile = config.get("active_data_profile")
    profiles = config.get("data_profiles")
    if not active_profile or not isinstance(profiles, dict):
        return config

    profile = profiles.get(active_profile)
    if not isinstance(profile, dict):
        raise ValueError(f"Unknown active_data_profile: {active_profile}")

    override_map = {
        ("step4_cv", "paths", "human_annotations_csv"): profile.get("step4_cv", {}).get("human_annotations_csv"),
        ("step4_cv", "paths", "output_dir"): profile.get("step4_cv", {}).get("output_dir"),
        ("step4_cv", "paths", "folds_path"): profile.get("step4_cv", {}).get("folds_path"),
        ("step5_reasoning", "paths", "step4_output_dir"): profile.get("step5_reasoning", {}).get("step4_output_dir"),
        ("step5_reasoning", "paths", "output_dir"): profile.get("step5_reasoning", {}).get("output_dir"),
        ("step5_reasoning", "paths", "sft_output_dir"): profile.get("step5_reasoning", {}).get("sft_output_dir"),
        ("step5_cv", "finbert", "step4_output_dir"): profile.get("step5_cv", {}).get("finbert_step4_output_dir"),
        ("step5_cv", "finbert", "results_dir"): profile.get("step5_cv", {}).get("finbert_results_dir"),
        ("step5_cv", "finetune", "sft_output_dir"): profile.get("step5_cv", {}).get("finetune_sft_output_dir"),
        ("step5_cv", "finetune", "results_root"): profile.get("step5_cv", {}).get("finetune_results_root"),
        ("step5_cv", "finetune", "models_root"): profile.get("step5_cv", {}).get("finetune_models_root"),
        ("step5_cv", "api_llm", "step4_output_dir"): profile.get("step5_cv", {}).get("api_llm_step4_output_dir"),
        ("step5_cv", "api_llm", "results_root"): profile.get("step5_cv", {}).get("api_llm_results_root"),
        ("step5_cv", "evaluation", "results_root"): profile.get("step5_cv", {}).get("evaluation_results_root"),
    }

    for keys, value in override_map.items():
        if value is None:
            continue
        _apply_path_override(config, list(keys), value)

    return config


def load_yaml_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return apply_active_data_profile(raw)


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
