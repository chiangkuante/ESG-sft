"""Step 2（final_v1）：config-driven 完整 FinBERT classification。

讀 `config/config.yaml` 的 `step2_classification`，對 Step 1 完整 paragraphs 重新推論，
輸出到 final_v1 目錄並產生帶 fingerprint 的 manifest。本輪固定 full rerun，不讀舊 checkpoint。

入口（無 CLI 實驗參數）：
  uv run python src/step2_classification/run.py
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.fingerprint import (
    atomic_write_json,
    build_stage_manifest,
    expected_gt_sha256,
    get_run_id,
    load_config,
    load_ground_truth_manifest,
    output_file_record,
    resolve_path,
    sha256_file,
)
from src.step2_classification.classify import ESG_CATEGORIES, run_classification

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def validate_predictions(paragraphs: list[dict], expected_total: int) -> dict:
    if len(paragraphs) != expected_total:
        raise ValueError(
            f"Step 2 output count {len(paragraphs)} != Step 1 input count {expected_total}"
        )
    valid_labels = set(ESG_CATEGORIES)
    counts: Counter[str] = Counter()
    for record in paragraphs:
        label = record.get("finbert_label")
        conf = record.get("finbert_confidence")
        if label not in valid_labels:
            raise ValueError(f"Invalid finbert_label {label!r} for {record.get('paragraph_id')}")
        if conf is None or not (0.0 <= float(conf) <= 1.0):
            raise ValueError(
                f"finbert_confidence out of [0,1] ({conf}) for {record.get('paragraph_id')}"
            )
        counts[label] += 1
    return {label: counts.get(label, 0) for label in ESG_CATEGORIES}


def main() -> None:
    config = load_config()
    run_id = get_run_id(config)
    step2_cfg = config["step2_classification"]

    input_path = resolve_path(step2_cfg["input_path"])
    output_dir = resolve_path(step2_cfg["output_dir"])
    mode = str(step2_cfg.get("mode", "full_rerun"))
    resume = mode == "resume"

    if not input_path.exists():
        raise FileNotFoundError(f"Step 1 paragraphs not found: {input_path}")

    logger.info("Step 2 (%s) mode=%s input=%s output=%s", run_id, mode, input_path, output_dir)

    # 預先建立輸出目錄：classify 在分類迴圈中（首次 checkpoint）即會寫檔，
    # 不能等迴圈結束才 mkdir。
    output_dir.mkdir(parents=True, exist_ok=True)

    # full rerun：忽略並移除舊 checkpoint，避免續跑到舊版輸入
    if not resume:
        old_ckpt = output_dir / "classified_checkpoint.json"
        if old_ckpt.exists():
            old_ckpt.unlink()

    paragraphs = run_classification(
        input_path=input_path,
        output_dir=output_dir,
        batch_size=int(step2_cfg["batch_size"]),
        checkpoint_every=int(step2_cfg["checkpoint_interval"]),
        device=int(step2_cfg["device"]),
        resume=resume,
    )

    distribution = validate_predictions(paragraphs, expected_total=len(paragraphs))

    classified_path = output_dir / "classified.json"
    stats_path = output_dir / "classification_stats.json"

    gt_manifest = load_ground_truth_manifest(config)
    manifest = build_stage_manifest(
        stage="step2_classification",
        run_id=run_id,
        ground_truth={
            "base_file_sha256": expected_gt_sha256(gt_manifest, "base"),
            "blance_file_sha256": expected_gt_sha256(gt_manifest, "blance"),
        },
        config_subset=step2_cfg,
        upstream={
            "step1_paragraphs": {
                "path": str(step2_cfg["input_path"]),
                "sha256": sha256_file(input_path),
                "count": len(paragraphs),
            }
        },
        code_paths=[
            "src/step2_classification/run.py",
            "src/step2_classification/classify.py",
            "src/common/fingerprint.py",
        ],
        outputs={
            "classified": output_file_record(classified_path),
            "classification_stats": output_file_record(stats_path),
        },
        extra={
            "total": len(paragraphs),
            "label_distribution": distribution,
            "mode": mode,
        },
    )
    atomic_write_json(output_dir / "manifest.json", manifest)
    logger.info("Step 2 manifest -> %s", output_dir / "manifest.json")
    logger.info("Step 2 complete: %s records validated.", len(paragraphs))


if __name__ == "__main__":
    main()
