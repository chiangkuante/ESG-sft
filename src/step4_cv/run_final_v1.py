"""Step 4（final_v1）：依 config matrix 一次建立 base / balance / combined 三個實驗。

- group-aware stratified 3-fold（同 text group 不會橫跨 train/val）
- text group label 衝突立即停止並輸出 conflict report
- 每個 fold 產生 leakage audit
- 每個實驗輸出帶 fingerprint 的 manifest

入口（無 CLI 實驗參數）：
  uv run python src/step4_cv/run_final_v1.py
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.fingerprint import (
    FingerprintMismatch,
    atomic_write_json,
    build_stage_manifest,
    enforce_fingerprints,
    expected_gt_sha256,
    get_run_id,
    load_config,
    load_ground_truth_manifest,
    output_file_record,
    resolve_path,
    sha256_file,
)
from src.step4_cv.common import ESG_CATEGORIES
from src.step4_cv.data import count_by_label, load_balance_records, load_human_annotations
from src.step4_cv.group_folds import (
    attach_text_fingerprint,
    audit_cross_fold,
    create_group_aware_folds,
    detect_label_conflicts,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

CODE_PATHS = [
    "src/step4_cv/run_final_v1.py",
    "src/step4_cv/group_folds.py",
    "src/step4_cv/data.py",
    "src/step4_cv/common.py",
    "src/common/fingerprint.py",
]


def _check_gt_sha(csv_path: Path, expected: str, dataset: str, enforce: bool) -> str:
    actual = sha256_file(csv_path)
    if actual != expected:
        message = (
            f"Ground-truth {dataset} SHA-256 mismatch:\n  file={csv_path}\n"
            f"  expected={expected}\n  actual  ={actual}"
        )
        if enforce:
            raise FingerprintMismatch(message)
        logger.warning(message)
    return actual


def _subset(records: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    return [records[i] for i in indices]


def build_fold_outputs(
    *,
    fold: dict[str, Any],
    cv_population: list[dict[str, Any]],
    balance_records: list[dict[str, Any]],
    add_balance_to_train: bool,
    fold_dir: Path,
) -> dict[str, Any]:
    fold_idx = fold["fold"]
    human_train = _subset(cv_population, fold["train_indices"])
    human_val = _subset(cv_population, fold["val_indices"])

    val_fps = {record["text_fingerprint"] for record in human_val}
    val_pids = {record["paragraph_id"] for record in human_val}

    train_pool = [dict(record) for record in human_train]
    excluded_balance: list[dict[str, Any]] = []
    if add_balance_to_train:
        for record in balance_records:
            if record["text_fingerprint"] in val_fps:
                excluded_balance.append(
                    {"paragraph_id": record["paragraph_id"], "reason": "val_group_leakage"}
                )
                continue
            train_pool.append(dict(record))

    # leakage audit
    train_pids = {record["paragraph_id"] for record in train_pool}
    train_fps = {record["text_fingerprint"] for record in train_pool}
    pid_overlap = sorted(train_pids & val_pids)
    fp_overlap = sorted(train_fps & val_fps)
    leakage_audit = {
        "fold": fold_idx,
        "train_pool_size": len(train_pool),
        "human_train_size": len(human_train),
        "human_val_size": len(human_val),
        "added_balance": len(train_pool) - len(human_train),
        "excluded_balance_val_leak": len(excluded_balance),
        "paragraph_id_overlap": pid_overlap,
        "text_fingerprint_overlap": fp_overlap,
        "train_label_distribution": count_by_label(train_pool),
        "val_label_distribution": count_by_label(human_val),
        "ok": not pid_overlap and not fp_overlap,
    }
    if not leakage_audit["ok"]:
        raise RuntimeError(
            f"Fold {fold_idx} leakage detected: "
            f"pid_overlap={pid_overlap[:5]} fp_overlap={fp_overlap[:5]}"
        )

    source_counts = Counter(record["source"] for record in train_pool)
    train_pool_summary = {
        "fold": fold_idx,
        "train_pool_size": len(train_pool),
        "source_counts": dict(source_counts),
        "label_counts": count_by_label(train_pool),
    }

    fold_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(fold_dir / "human_train.json", human_train)
    atomic_write_json(fold_dir / "human_val.json", human_val)
    atomic_write_json(fold_dir / "train_pool.json", train_pool)
    atomic_write_json(fold_dir / "train_pool_summary.json", train_pool_summary)
    atomic_write_json(fold_dir / "excluded_balance.json", excluded_balance)
    atomic_write_json(fold_dir / "leakage_audit.json", leakage_audit)

    summary = {
        "fold": fold_idx,
        "human_train_size": len(human_train),
        "human_val_size": len(human_val),
        "train_pool_size": len(train_pool),
        "leakage_ok": leakage_audit["ok"],
        "val_distribution": count_by_label(human_val),
    }
    atomic_write_json(fold_dir / "summary.json", summary)
    return summary


def run_experiment(
    *,
    experiment: dict[str, Any],
    base_records: list[dict[str, Any]],
    balance_records_all: list[dict[str, Any]],
    folds_cfg: dict[str, Any],
    output_root: Path,
    manifest_common: dict[str, Any],
) -> dict[str, Any]:
    name = experiment["name"]
    balance_enabled = bool(experiment["balance_enabled"])
    include_in_cv = bool(experiment["include_balance_in_cv"])

    exp_dir = output_root / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Experiment %s: balance_enabled=%s include_in_cv=%s", name, balance_enabled, include_in_cv
    )

    balance_records = balance_records_all if balance_enabled else []

    # 衝突偵測：本實驗實際參與的所有 record
    participating = base_records + balance_records
    conflicts = detect_label_conflicts(participating)
    if conflicts:
        atomic_write_json(exp_dir / "conflicts.json", conflicts)
        raise RuntimeError(
            f"Experiment {name}: {len(conflicts)} text group label conflict(s); "
            f"see {exp_dir / 'conflicts.json'}"
        )

    cv_population = base_records + (balance_records if include_in_cv else [])

    folds = create_group_aware_folds(
        records=cv_population,
        n_splits=int(folds_cfg["n_splits"]),
        shuffle=bool(folds_cfg.get("shuffle", True)),
        random_state=int(folds_cfg["random_state"]),
    )
    atomic_write_json(exp_dir / "cv_folds.json", folds)

    cross_fold = audit_cross_fold(folds, population_size=len(cv_population))
    atomic_write_json(exp_dir / "cross_fold_audit.json", cross_fold)
    if not cross_fold["ok"]:
        raise RuntimeError(f"Experiment {name}: cross-fold validation audit failed: {cross_fold}")

    add_balance_to_train = balance_enabled and not include_in_cv
    fold_summaries = []
    for fold in folds:
        fold_dir = exp_dir / f"fold_{fold['fold']}"
        fold_summaries.append(
            build_fold_outputs(
                fold=fold,
                cv_population=cv_population,
                balance_records=balance_records,
                add_balance_to_train=add_balance_to_train,
                fold_dir=fold_dir,
            )
        )

    fold_dirs = [f"fold_{summary['fold']}" for summary in fold_summaries]
    manifest = build_stage_manifest(
        stage="step4_cv",
        run_id=manifest_common["run_id"],
        ground_truth=manifest_common["ground_truth"],
        config_subset={
            "experiment": experiment,
            "folds": folds_cfg,
            "pseudo_labels_enabled": False,
            "synthetic_generation_enabled": False,
        },
        upstream=manifest_common["upstream"],
        code_paths=CODE_PATHS,
        outputs={
            "cv_folds": output_file_record(exp_dir / "cv_folds.json"),
            "cross_fold_audit": output_file_record(exp_dir / "cross_fold_audit.json"),
            "folds": {
                fold_dir: {
                    "train_pool": output_file_record(exp_dir / fold_dir / "train_pool.json"),
                    "human_val": output_file_record(exp_dir / fold_dir / "human_val.json"),
                    "leakage_audit": output_file_record(exp_dir / fold_dir / "leakage_audit.json"),
                }
                for fold_dir in fold_dirs
            },
        },
        extra={
            "experiment_name": name,
            "cv_population_size": len(cv_population),
            "cv_population_distribution": count_by_label(cv_population),
            "balance_records_total": len(balance_records),
            "n_folds": len(folds),
            "fold_directories": fold_dirs,
            "fold_summaries": fold_summaries,
        },
    )
    atomic_write_json(exp_dir / "manifest.json", manifest)
    logger.info("Experiment %s done: %s folds -> %s", name, len(folds), exp_dir)
    return manifest


def main() -> None:
    config = load_config()
    run_id = get_run_id(config)
    enforce = enforce_fingerprints(config)
    step4_cfg = config["step4_cv"]
    pipeline_cfg = config["pipeline"]
    gt_cfg = pipeline_cfg["ground_truth"]

    base_csv = resolve_path(gt_cfg["base_csv"])
    balance_csv = resolve_path(gt_cfg["balance_csv"])
    output_root = resolve_path(step4_cfg["output_root_final_v1"])

    gt_manifest = load_ground_truth_manifest(config)
    base_sha = _check_gt_sha(base_csv, expected_gt_sha256(gt_manifest, "base"), "base", enforce)
    balance_sha = _check_gt_sha(
        balance_csv, expected_gt_sha256(gt_manifest, "blance"), "blance", enforce
    )

    base_records = load_human_annotations(base_csv)
    balance_records_all = load_balance_records(balance_csv)
    if len(base_records) != int(gt_cfg["base_rows"]):
        raise ValueError(f"base rows {len(base_records)} != expected {gt_cfg['base_rows']}")
    if len(balance_records_all) != int(gt_cfg["balance_rows"]):
        raise ValueError(
            f"blance rows {len(balance_records_all)} != expected {gt_cfg['balance_rows']}"
        )

    attach_text_fingerprint(base_records)
    attach_text_fingerprint(balance_records_all)

    manifest_common = {
        "run_id": run_id,
        "ground_truth": {
            "base_path": gt_cfg["base_csv"],
            "base_file_sha256": base_sha,
            "base_content_fingerprint_sha256": gt_manifest["datasets"]["base"][
                "content_fingerprint_sha256"
            ],
            "blance_path": gt_cfg["balance_csv"],
            "blance_file_sha256": balance_sha,
            "blance_content_fingerprint_sha256": gt_manifest["datasets"]["blance"][
                "content_fingerprint_sha256"
            ],
        },
        "upstream": {
            "base_csv": {"path": gt_cfg["base_csv"], "sha256": base_sha},
            "balance_csv": {"path": gt_cfg["balance_csv"], "sha256": balance_sha},
        },
    }

    summaries = {}
    for experiment in step4_cfg["experiments"]:
        manifest = run_experiment(
            experiment=experiment,
            base_records=base_records,
            balance_records_all=balance_records_all,
            folds_cfg=step4_cfg["folds"],
            output_root=output_root,
            manifest_common=manifest_common,
        )
        summaries[experiment["name"]] = {
            "cv_population_size": manifest["cv_population_size"],
            "fold_directories": manifest["fold_directories"],
        }

    atomic_write_json(
        output_root / "manifest.json",
        {
            "stage": "step4_cv",
            "run_id": run_id,
            "ground_truth": manifest_common["ground_truth"],
            "experiments": summaries,
        },
    )
    logger.info("Step 4 final_v1 complete for experiments: %s", list(summaries))


if __name__ == "__main__":
    main()
