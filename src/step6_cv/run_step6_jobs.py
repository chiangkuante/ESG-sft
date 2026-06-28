"""final_v1 Step 6 job runner（兩台機器共用同一無參數入口）。

  uv run python src/step6_cv/run_step6_jobs.py

- 只執行 config `step6_cv.job_matrix.active_machine` 指派給本機的 jobs。
- fingerprint-safe resume：已完成且 input/config fingerprint 相符才 skip；
  fingerprint 不同則寫 conflict report 並停止，不靜默覆蓋。
- job 失敗不標記完成；重跑時從 failed/pending 繼續。
- 每個 job 寫 receipt：GPU、套件版本、seed、有效超參數、checkpoint 路徑。
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.fingerprint import (
    atomic_write_json,
    canonical_json,
    get_run_id,
    load_config,
    load_json,
    now_iso,
    resolve_path,
    sha256_file,
    sha256_text,
)
from src.step6_cv.job_matrix import build_jobs, jobs_for_machine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def effective_training_subset(train_cfg: dict, ablation_cfg: dict, inference_cfg: dict, model_name: str) -> dict:
    return {
        "model_name": model_name,
        "lora_r": train_cfg["lora_r"],
        "lora_alpha": train_cfg["lora_alpha"],
        "lora_dropout": train_cfg["lora_dropout"],
        "num_train_epochs": train_cfg["num_train_epochs"],
        "learning_rate": train_cfg["learning_rate"],
        "weight_decay": train_cfg["weight_decay"],
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "max_seq_length": train_cfg["max_seq_length"],
        "random_state": train_cfg["random_state"],
        "ablation": ablation_cfg,
        "checkpoint_strategy": inference_cfg.get("checkpoint_strategy"),
    }


def job_input_fingerprint(train_path: Path, val_path: Path, config_subset: dict) -> dict:
    train_sha = sha256_file(train_path)
    val_sha = sha256_file(val_path)
    config_fp = sha256_text(canonical_json(config_subset))
    combined = sha256_text(canonical_json({"train": train_sha, "val": val_sha, "config": config_fp}))
    return {"train_sha256": train_sha, "val_sha256": val_sha, "config_fingerprint": config_fp, "combined": combined}


def collect_versions() -> dict:
    versions: dict[str, Any] = {}
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["cuda_available"] = torch.cuda.is_available()
        versions["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        versions["torch"] = None
    for pkg in ("transformers", "unsloth", "trl", "peft"):
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:  # noqa: BLE001
            versions[pkg] = None
    return versions


def run_one_job(
    job: dict,
    *,
    sft_root: Path,
    models_root: Path,
    results_root: Path,
    train_cfg: dict,
    finetune_cfg: dict,
    ablation_cfg: dict,
    inference_cfg: dict,
    model_name: str,
) -> dict:
    import multiprocessing as mp

    from src.step6_cv.common import compute_overall_metrics, compute_per_class_metrics, compute_pillar_metrics
    from src.step6_cv.run_cv_finetune import inference_worker, training_worker

    experiment, model, fold = job["experiment"], job["model"], job["fold"]
    fold_dir = sft_root / experiment / f"fold_{fold}"
    train_path = fold_dir / "train_sft_text.json"
    val_path = fold_dir / "val_eval.json"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing SFT input for {experiment} fold_{fold}")

    adapter_dir = models_root / experiment / model / f"fold_{fold}" / "adapter"
    exp_results = results_root / experiment / model
    results_path = exp_results / f"fold_{fold}_results.json"
    metrics_path = exp_results / f"fold_{fold}_metrics.json"

    max_new_tokens = int(inference_cfg.get("max_new_tokens") or 512)
    dry_run_cfg: dict[str, Any] = {}

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # train（子行程隔離 GPU 記憶體）
    train_proc = mp.Process(
        target=training_worker,
        args=(model, fold, str(train_path), str(adapter_dir), train_cfg, finetune_cfg, ablation_cfg, dry_run_cfg),
    )
    train_proc.start()
    train_proc.join()
    if train_proc.exitcode != 0:
        raise RuntimeError(f"training subprocess exitcode={train_proc.exitcode}")

    # inference（final checkpoint = adapter_dir）
    infer_proc = mp.Process(
        target=inference_worker,
        args=(model, fold, str(adapter_dir), str(val_path), str(results_path), max_new_tokens, train_cfg, finetune_cfg, ablation_cfg, dry_run_cfg),
    )
    infer_proc.start()
    infer_proc.join()
    if infer_proc.exitcode != 0:
        raise RuntimeError(f"inference subprocess exitcode={infer_proc.exitcode}")

    rows = load_json(results_path)
    metrics = compute_overall_metrics(rows)
    metrics["fold"] = fold
    metrics["per_class"] = compute_per_class_metrics(rows)
    metrics["pillar"] = compute_pillar_metrics(rows)
    atomic_write_json(metrics_path, metrics)
    return {
        "adapter_dir": str(adapter_dir),
        "results_path": str(results_path),
        "metrics_path": str(metrics_path),
        "overall": {k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1", "kappa")},
    }


def main() -> None:
    config = load_config()
    run_id = get_run_id(config)
    step6 = config["step6_cv"]
    jm = step6["job_matrix"]
    if not bool(jm.get("enabled", True)):
        logger.info("job_matrix disabled; nothing to do.")
        return

    active_machine = str(jm["active_machine"])
    sft_root = resolve_path(jm["sft_input_root"])
    models_root = resolve_path(jm["models_root"])
    results_root = resolve_path(jm["results_root"])
    receipts_root = resolve_path(jm["receipts_root"])

    finetune_cfg = step6["finetune"]
    from src.step6_cv.run_cv_finetune import apply_model_overrides, get_ablation_cfg, get_inference_cfg

    jobs = jobs_for_machine(build_jobs(run_id, jm), active_machine)
    logger.info("Machine %s has %s jobs", active_machine, len(jobs))

    failed: list[str] = []
    versions = collect_versions()
    for job in jobs:
        job_id = job["job_id"]
        model = job["model"]
        cfg = apply_model_overrides(finetune_cfg, model)
        train_cfg = cfg["training"]
        ablation_cfg = get_ablation_cfg(cfg)
        inference_cfg = get_inference_cfg(cfg)
        model_name = cfg.get("model_registry", {}).get(model, model)
        config_subset = effective_training_subset(train_cfg, ablation_cfg, inference_cfg, model_name)

        fold_dir = sft_root / job["experiment"] / f"fold_{job['fold']}"
        train_path = fold_dir / "train_sft_text.json"
        val_path = fold_dir / "val_eval.json"
        if not train_path.exists() or not val_path.exists():
            logger.error("Skipping %s: missing SFT input", job_id)
            failed.append(job_id)
            continue

        fp = job_input_fingerprint(train_path, val_path, config_subset)
        receipt_path = receipts_root / f"{job_id}.json"
        results_path = results_root / job["experiment"] / model / f"fold_{job['fold']}_results.json"
        metrics_path = results_root / job["experiment"] / model / f"fold_{job['fold']}_metrics.json"

        if receipt_path.exists():
            receipt = load_json(receipt_path)
            if receipt.get("status") == "completed":
                if receipt.get("input_fingerprint", {}).get("combined") == fp["combined"]:
                    if results_path.exists() and metrics_path.exists():
                        logger.info("Skip %s (completed, fingerprint match)", job_id)
                        continue
                    logger.warning("%s receipt completed but outputs missing; rerunning", job_id)
                else:
                    conflict = {
                        "job_id": job_id,
                        "reason": "input/config fingerprint changed vs completed receipt",
                        "receipt_fingerprint": receipt.get("input_fingerprint"),
                        "current_fingerprint": fp,
                        "detected_at": now_iso(),
                    }
                    atomic_write_json(receipts_root / f"{job_id}.conflict.json", conflict)
                    raise RuntimeError(
                        f"Fingerprint conflict for {job_id}; wrote conflict report and stopping."
                    )

        logger.info("Running %s", job_id)
        started = now_iso()
        try:
            outcome = run_one_job(
                job, sft_root=sft_root, models_root=models_root, results_root=results_root,
                train_cfg=train_cfg, finetune_cfg=cfg, ablation_cfg=ablation_cfg,
                inference_cfg=inference_cfg, model_name=model_name,
            )
            atomic_write_json(
                receipt_path,
                {
                    "job_id": job_id, "run_id": run_id, "machine": active_machine,
                    "experiment": job["experiment"], "model": model, "fold": job["fold"],
                    "status": "completed", "started_at": started, "finished_at": now_iso(),
                    "input_fingerprint": fp, "hyperparams": config_subset,
                    "versions": versions, "outputs": outcome,
                },
            )
            logger.info("Completed %s: %s", job_id, outcome["overall"])
        except Exception as exc:  # noqa: BLE001
            atomic_write_json(
                receipt_path,
                {
                    "job_id": job_id, "run_id": run_id, "machine": active_machine,
                    "experiment": job["experiment"], "model": model, "fold": job["fold"],
                    "status": "failed", "started_at": started, "finished_at": now_iso(),
                    "input_fingerprint": fp, "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            logger.error("Job %s failed: %s", job_id, exc)
            failed.append(job_id)

    if failed:
        raise SystemExit(f"{len(failed)} job(s) failed: {failed}")
    logger.info("All assigned jobs complete for machine %s", active_machine)


if __name__ == "__main__":
    main()
