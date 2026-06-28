"""final_v1 共用 fingerprint / manifest / atomic-write 工具。

所有 final_v1 stage 都用這裡的函式產生可追溯的 manifest：
ground-truth SHA-256、content fingerprint、config fingerprint、程式碼 fingerprint，
以及輸出檔 SHA-256。fingerprint 不符時上層必須停止或重建，不得因檔案存在就續跑。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# ------------------------------------------------------------
# 路徑 / config
# ------------------------------------------------------------
def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_run_id(config: dict[str, Any] | None = None) -> str:
    config = config or load_config()
    return str(config.get("pipeline", {}).get("run_id", "final_v1"))


def enforce_fingerprints(config: dict[str, Any] | None = None) -> bool:
    config = config or load_config()
    return bool(config.get("pipeline", {}).get("enforce_fingerprints", True))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


# ------------------------------------------------------------
# hashing
# ------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    """穩定排序的 JSON 字串，用於 content / config fingerprint。"""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint_obj(payload: Any) -> str:
    return sha256_text(canonical_json(payload))


# ------------------------------------------------------------
# 文本正規化（text_fingerprint 與 reasoning cache 共用同一套規則）
# ------------------------------------------------------------
def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    # 折疊所有空白（含換行）為單一空格，去頭尾，統一大小寫不變
    return re.sub(r"\s+", " ", str(text)).strip()


def text_fingerprint(text: str | None) -> str:
    return sha256_text(normalize_text(text))


# ------------------------------------------------------------
# config / code fingerprint
# ------------------------------------------------------------
def config_fingerprint(config_subset: Any) -> str:
    return fingerprint_obj(config_subset)


def code_fingerprint(paths: Iterable[str | Path]) -> dict[str, Any]:
    files: dict[str, str] = {}
    for raw in sorted({str(p) for p in paths}):
        path = resolve_path(raw)
        if not path.exists():
            files[raw] = "MISSING"
            continue
        rel = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
        files[rel] = sha256_file(path)
    combined = sha256_text(canonical_json(files))
    return {"combined": combined, "files": files}


# ------------------------------------------------------------
# atomic write
# ------------------------------------------------------------
def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    atomic_write_bytes(path, data)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel_to_root(path: str | Path) -> str:
    path = Path(path)
    if path.is_relative_to(PROJECT_ROOT):
        return str(path.relative_to(PROJECT_ROOT))
    return str(path)


# ------------------------------------------------------------
# ground truth manifest 讀取
# ------------------------------------------------------------
def load_ground_truth_manifest(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    manifest_path = resolve_path(config["pipeline"]["ground_truth"]["manifest"])
    return load_json(manifest_path)


def expected_gt_sha256(manifest: dict[str, Any], dataset: str) -> str:
    return manifest["datasets"][dataset]["file_sha256"]


# ------------------------------------------------------------
# manifest builder
# ------------------------------------------------------------
def build_stage_manifest(
    *,
    stage: str,
    run_id: str,
    ground_truth: dict[str, Any] | None,
    config_subset: Any,
    upstream: dict[str, Any] | None,
    code_paths: Iterable[str | Path],
    outputs: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "stage": stage,
        "run_id": run_id,
        "created_at": now_iso(),
        "ground_truth": ground_truth or {},
        "config_fingerprint": config_fingerprint(config_subset),
        "upstream": upstream or {},
        "code_fingerprint": code_fingerprint(code_paths),
        "outputs": outputs,
    }
    if extra:
        manifest.update(extra)
    return manifest


def output_file_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": rel_to_root(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


class FingerprintMismatch(RuntimeError):
    """fingerprint 不符且設定為 enforce 時拋出。"""


def _self_check() -> None:
    assert normalize_text("  a\n b\t c ") == "a b c"
    assert sha256_text("abc") == hashlib.sha256(b"abc").hexdigest()
    assert text_fingerprint("a  b") == text_fingerprint("a\nb")
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    print("fingerprint self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
