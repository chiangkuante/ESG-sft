"""final_v1：合併兩台機器的 Step 6 結果與 receipt。

- 驗證 36 個 job 全部有 completed receipt 且 fingerprint 無衝突。
- 不以較新檔案靜默覆蓋衝突結果；發現衝突即列出並以非零結束。
- 產生收斂報告供最終 evaluation 使用。

入口：
  uv run python src/step6_cv/collect_step6_results.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.fingerprint import atomic_write_json, get_run_id, load_config, load_json, now_iso, resolve_path
from src.step6_cv.job_matrix import build_jobs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def main() -> None:
    config = load_config()
    run_id = get_run_id(config)
    jm = config["step6_cv"]["job_matrix"]
    receipts_root = resolve_path(jm["receipts_root"])
    results_root = resolve_path(jm["results_root"])

    jobs = build_jobs(run_id, jm)
    completed, missing, failed, conflicts = [], [], [], []
    index = []

    for job in jobs:
        job_id = job["job_id"]
        receipt_path = receipts_root / f"{job_id}.json"
        conflict_path = receipts_root / f"{job_id}.conflict.json"
        if conflict_path.exists():
            conflicts.append(job_id)
            continue
        if not receipt_path.exists():
            missing.append(job_id)
            continue
        receipt = load_json(receipt_path)
        status = receipt.get("status")
        if status != "completed":
            failed.append(job_id)
            continue
        results_path = results_root / job["experiment"] / job["model"] / f"fold_{job['fold']}_results.json"
        metrics_path = results_root / job["experiment"] / job["model"] / f"fold_{job['fold']}_metrics.json"
        if not results_path.exists() or not metrics_path.exists():
            missing.append(job_id)
            continue
        index.append(
            {
                "job_id": job_id, "machine": receipt.get("machine"),
                "experiment": job["experiment"], "model": job["model"], "fold": job["fold"],
                "fingerprint": receipt.get("input_fingerprint", {}).get("combined"),
                "results_path": str(results_path), "metrics_path": str(metrics_path),
                "overall": receipt.get("outputs", {}).get("overall"),
            }
        )
        completed.append(job_id)

    report = {
        "stage": "step6_collect", "run_id": run_id, "created_at": now_iso(),
        "total_jobs": len(jobs), "completed": len(completed),
        "missing": missing, "failed": failed, "conflicts": conflicts,
        "index": index,
    }
    out_path = results_root / "_collect" / "collect_report.json"
    atomic_write_json(out_path, report)
    logger.info(
        "Collect: completed=%s missing=%s failed=%s conflicts=%s -> %s",
        len(completed), len(missing), len(failed), len(conflicts), out_path,
    )
    if missing or failed or conflicts:
        raise SystemExit(
            f"Incomplete: missing={len(missing)} failed={len(failed)} conflicts={len(conflicts)}"
        )


if __name__ == "__main__":
    main()
