"""final_v1 Step 6 deterministic job matrix。

由 config 的 `step6_cv.job_matrix` 建立穩定 job 清單與 job ID。
Job ID 含 run_id / experiment / model / fold / task_type，供兩台機器分工與 resume。
此模組為純函式，可獨立測試（不載入模型）。
"""

from __future__ import annotations

from typing import Any

TASK_TYPE = "finetune_infer"


def make_job_id(run_id: str, experiment: str, model: str, fold: int, task_type: str = TASK_TYPE) -> str:
    return f"{run_id}__{experiment}__{model}__fold{fold}__{task_type}"


def model_to_machine(job_matrix_cfg: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for machine, spec in job_matrix_cfg.get("machines", {}).items():
        for model in spec.get("models", []):
            if model in mapping:
                raise ValueError(f"Model {model} assigned to multiple machines.")
            mapping[model] = machine
    return mapping


def build_jobs(run_id: str, job_matrix_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """建立全部 jobs（含機器指派），順序穩定。"""
    experiments = list(job_matrix_cfg["experiments"])
    folds = [int(f) for f in job_matrix_cfg["folds"]]
    assignment = model_to_machine(job_matrix_cfg)
    models = sorted(assignment)

    jobs: list[dict[str, Any]] = []
    for experiment in experiments:
        for model in models:
            for fold in folds:
                jobs.append(
                    {
                        "job_id": make_job_id(run_id, experiment, model, fold),
                        "run_id": run_id,
                        "experiment": experiment,
                        "model": model,
                        "fold": fold,
                        "task_type": TASK_TYPE,
                        "machine": assignment[model],
                    }
                )
    # job_id 必須唯一
    seen = set()
    for job in jobs:
        if job["job_id"] in seen:
            raise ValueError(f"Duplicate job_id: {job['job_id']}")
        seen.add(job["job_id"])
    return jobs


def jobs_for_machine(jobs: list[dict[str, Any]], machine: str) -> list[dict[str, Any]]:
    return [job for job in jobs if job["machine"] == machine]
