from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step4_cv.common import ESG_CATEGORIES, canonicalize_label, load_yaml_config, resolve_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_step5_config() -> dict[str, Any]:
    raw = load_yaml_config()
    config = raw.get("step5_reasoning")
    if not isinstance(config, dict):
        raise ValueError("Missing `step5_reasoning` section in config/config.yaml")
    return config


def normalize_reasoning_text(text: str) -> str:
    cleaned = " ".join(str(text).split()).strip()
    return cleaned.replace("```", "").strip()


def validate_reasoning(reasoning: str, min_chars: int, max_chars: int) -> bool:
    cleaned = normalize_reasoning_text(reasoning)
    if not cleaned:
        return False
    if len(cleaned) < min_chars or len(cleaned) > max_chars:
        return False
    if "<reasoning>" in cleaned or "<label>" in cleaned:
        return False
    if "```" in reasoning:
        return False
    return True
