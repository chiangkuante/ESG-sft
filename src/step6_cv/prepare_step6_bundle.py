"""final_v1：由 Machine A 驗證 9 個 SFT fold 並打包 Machine B 所需的唯讀輸入 bundle。

輸出：
  artifacts/final_v1/step6_input_bundle.zip
  artifacts/final_v1/step6_input_bundle_manifest.json   （含每個檔的 SHA-256）

入口：
  uv run python src/step6_cv/prepare_step6_bundle.py
"""

from __future__ import annotations

import logging
import re
import sys
import zipfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.fingerprint import (
    atomic_write_json,
    get_run_id,
    load_config,
    load_json,
    now_iso,
    resolve_path,
    sha256_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

XML_RE = re.compile(r"(?is)<reasoning>\s*(.+?)\s*</reasoning>\s*<label>\s*(.+?)\s*</label>")


def validate_fold(sft_dir: Path, experiment: str, fold: int) -> dict:
    train_path = sft_dir / f"fold_{fold}" / "train_sft_text.json"
    val_path = sft_dir / f"fold_{fold}" / "val_eval.json"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing SFT files for {experiment} fold_{fold}: {train_path} / {val_path}")

    train = load_json(train_path)
    val = load_json(val_path)
    if not train:
        raise ValueError(f"Empty train_sft_text for {experiment} fold_{fold}")

    bad = 0
    for record in train:
        assistant = next((m for m in record["conversations"] if m["role"] == "assistant"), None)
        if not assistant or not XML_RE.search(assistant["content"]):
            bad += 1
    if bad:
        raise ValueError(f"{experiment} fold_{fold}: {bad} train rows missing valid <reasoning>/<label>")

    # val 不得含 training reasoning
    for record in val:
        if "reasoning" in record:
            raise ValueError(f"{experiment} fold_{fold}: val_eval contains 'reasoning' field")

    return {
        "experiment": experiment,
        "fold": fold,
        "train_sft_text": {"path": str(train_path), "sha256": sha256_file(train_path), "rows": len(train)},
        "val_eval": {"path": str(val_path), "sha256": sha256_file(val_path), "rows": len(val)},
    }


def main() -> None:
    config = load_config()
    run_id = get_run_id(config)
    jm_cfg = config["step6_cv"]["job_matrix"]
    sft_root = resolve_path(jm_cfg["sft_input_root"])
    bundle_path = resolve_path(jm_cfg["bundle_path"])
    manifest_path = resolve_path(jm_cfg["bundle_manifest"])

    experiments = list(jm_cfg["experiments"])
    folds = [int(f) for f in jm_cfg["folds"]]

    fold_records = []
    files_to_zip: list[Path] = []
    for experiment in experiments:
        sft_dir = sft_root / experiment
        exp_manifest = sft_dir / "manifest.json"
        if exp_manifest.exists():
            files_to_zip.append(exp_manifest)
        for fold in folds:
            record = validate_fold(sft_dir, experiment, fold)
            fold_records.append(record)
            files_to_zip.append(Path(record["train_sft_text"]["path"]))
            files_to_zip.append(Path(record["val_eval"]["path"]))

    expected = len(experiments) * len(folds)
    if len(fold_records) != expected:
        raise ValueError(f"Expected {expected} SFT folds, validated {len(fold_records)}")

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in files_to_zip:
            arcname = str(file_path.relative_to(resolve_path(".")))
            zf.write(file_path, arcname)

    manifest = {
        "stage": "step6_input_bundle",
        "run_id": run_id,
        "created_at": now_iso(),
        "experiments": experiments,
        "folds": folds,
        "fold_count": len(fold_records),
        "fold_files": fold_records,
        "bundle": {"path": str(jm_cfg["bundle_path"]), "sha256": sha256_file(bundle_path), "bytes": bundle_path.stat().st_size},
    }
    atomic_write_json(manifest_path, manifest)
    logger.info("Bundle ready: %s (%s folds) manifest=%s", bundle_path, len(fold_records), manifest_path)


if __name__ == "__main__":
    main()
