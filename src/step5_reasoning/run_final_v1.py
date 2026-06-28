"""Step 5（final_v1）：內容型 reasoning cache + XML SFT build。

- 共用 content cache：key = 正規化文本 + ground-truth label + prompt_version + provider/model + 生成參數
  相同文本/標籤/prompt/model 可跨 fold、跨實驗重用；文本相同但標籤不同不得共用。
- 只為 train pool 產生 reasoning；validation 不產生 reasoning，也不進入訓練資料。
- reasoning 僅以 ground-truth label 為目標；重試 failed/invalid 直到全部成功，否則停止。
- SFT assistant 固定 `<reasoning>...</reasoning>\n<label>...</label>`。

入口（無 CLI 實驗參數）：
  uv run python src/step5_reasoning/run_final_v1.py
"""

from __future__ import annotations

import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.fingerprint import (
    atomic_write_json,
    build_stage_manifest,
    canonical_json,
    get_run_id,
    load_config,
    load_json,
    normalize_text,
    now_iso,
    output_file_record,
    resolve_path,
    sha256_file,
    sha256_text,
)
from src.step4_cv.common import ESG_CATEGORIES, canonicalize_label
from src.step5_reasoning.common import normalize_reasoning_text, validate_reasoning
from src.step5_reasoning.generate_reasoning import batch_items, generate_batch_reasonings, get_caller
from src.step5_reasoning.reason import (
    CATEGORY_DEFINITIONS,
    CLASSIFY_SYSTEM_PROMPT,
    CLASSIFY_USER_TEMPLATE_XML,
    build_xml_answer,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

CODE_PATHS = [
    "src/step5_reasoning/run_final_v1.py",
    "src/step5_reasoning/generate_reasoning.py",
    "src/step5_reasoning/reason.py",
    "src/step5_reasoning/common.py",
    "src/common/fingerprint.py",
]


# ------------------------------------------------------------
# content cache
# ------------------------------------------------------------
def cache_key(text: str, label: str, *, prompt_version: str, provider: str, model: str, effort: str) -> str:
    payload = {
        "v": prompt_version,
        "provider": provider,
        "model": model,
        "effort": effort,
        "label": label,
        "text": normalize_text(text),
    }
    return sha256_text(canonical_json(payload))


def load_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        return load_json(path)
    return {}


def reasoning_is_valid(reasoning: str, label: str, min_chars: int, max_chars: int) -> bool:
    if not validate_reasoning(reasoning, min_chars=min_chars, max_chars=max_chars):
        return False
    # 不得含與 ground-truth 不一致的輸出 label
    label_line = re.search(r"(?im)^\s*label\s*:\s*([^\n]+)", reasoning)
    if label_line and canonicalize_label(label_line.group(1)) not in (None, label):
        return False
    return True


def generate_missing(
    *,
    missing: dict[str, dict[str, Any]],
    cache: dict[str, Any],
    cache_path: Path,
    gen_cfg: dict[str, Any],
    prompt_version: str,
    min_chars: int,
    max_chars: int,
) -> dict[str, int]:
    """對 cache miss 呼叫 API；持續重試直到全部成功，否則拋出。回傳統計。"""
    provider = str(gen_cfg["provider"])
    model = str(gen_cfg["model"])
    effort = str(gen_cfg.get("reasoning_effort", "low"))
    api_key = os.getenv(str(gen_cfg["api_key_env"]))
    if not api_key:
        raise ValueError(f"Missing API key env var: {gen_cfg['api_key_env']}")
    caller = get_caller(provider)
    batch_size = int(gen_cfg["batch_size"])
    max_output_tokens = int(gen_cfg["max_output_tokens"])
    sleep_seconds = float(gen_cfg["sleep_seconds"])
    retry_backoff = float(gen_cfg["retry_backoff_seconds"])
    max_passes = int(gen_cfg["max_retries"])

    stats = {"api_generated": 0, "invalid": 0}
    pending = dict(missing)
    for pass_idx in range(1, max_passes + 1):
        if not pending:
            break
        logger.info("Reasoning pass %s: %s items pending", pass_idx, len(pending))
        # batch 以 cache key 當作穩定 paragraph_id
        items = [
            {"paragraph_id": key, "combined_text": item["combined_text"], "label": item["label"], "source": item["source"]}
            for key, item in pending.items()
        ]
        still_pending: dict[str, dict[str, Any]] = {}
        for batch in batch_items(items, batch_size=batch_size):
            try:
                payload = generate_batch_reasonings(
                    batch=batch,
                    caller=caller,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=effort,
                )
                by_id = {
                    item["paragraph_id"]: item
                    for item in payload
                    if isinstance(item, dict) and item.get("paragraph_id")
                }
            except Exception as exc:  # noqa: BLE001 - batch 失敗整批重試
                logger.warning("Reasoning batch failed: %s", exc)
                for entry in batch:
                    still_pending[entry["paragraph_id"]] = pending[entry["paragraph_id"]]
                time.sleep(sleep_seconds + retry_backoff * pass_idx)
                continue

            for entry in batch:
                key = entry["paragraph_id"]
                candidate = by_id.get(key)
                reasoning = normalize_reasoning_text(candidate.get("reasoning", "")) if candidate else ""
                label = entry["label"]
                if reasoning_is_valid(reasoning, label, min_chars, max_chars):
                    cache[key] = {
                        "reasoning": reasoning,
                        "status": "success",
                        "label": label,
                        "provider": provider,
                        "model": model,
                        "prompt_version": prompt_version,
                        "retries": pass_idx,
                        "content_fingerprint": sha256_text(reasoning),
                        "created_at": now_iso(),
                    }
                    stats["api_generated"] += 1
                else:
                    stats["invalid"] += 1
                    still_pending[key] = pending[key]
            atomic_write_json(cache_path, cache)
            time.sleep(sleep_seconds)
        pending = still_pending

    if pending:
        raise RuntimeError(
            f"{len(pending)} reasoning items still failed after {max_passes} passes; "
            "not proceeding to SFT build."
        )
    return stats


# ------------------------------------------------------------
# per-experiment processing
# ------------------------------------------------------------
def process_experiment(
    *,
    name: str,
    step4_root: Path,
    reasoning_root: Path,
    sft_root: Path,
    cache: dict[str, Any],
    cache_path: Path,
    cfg: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    gen_cfg = cfg["generation"]
    cache_cfg = cfg["cache"]
    sft_cfg = cfg["sft"]
    prompt_version = str(cache_cfg["prompt_version"])
    provider = str(gen_cfg["provider"])
    model = str(gen_cfg["model"])
    effort = str(gen_cfg.get("reasoning_effort", "low"))
    min_chars = int(gen_cfg["min_reasoning_chars"])
    max_chars = int(gen_cfg["max_reasoning_chars"])
    shuffle_seed = int(sft_cfg["shuffle_seed"])

    step4_exp = step4_root / name
    step4_manifest = load_json(step4_exp / "manifest.json")
    fold_dirs = step4_manifest["fold_directories"]

    reasoning_exp = reasoning_root / name
    sft_exp = sft_root / name

    def key_for(item: dict[str, Any]) -> str:
        return cache_key(
            item["combined_text"], item["label"],
            prompt_version=prompt_version, provider=provider, model=model, effort=effort,
        )

    # 1) 收集全部 fold 的 cache miss，一次補齊
    fold_pools: dict[str, list[dict[str, Any]]] = {}
    missing: dict[str, dict[str, Any]] = {}
    for fold_dir in fold_dirs:
        train_pool = load_json(step4_exp / fold_dir / "train_pool.json")
        fold_pools[fold_dir] = train_pool
        atomic_write_json(reasoning_exp / fold_dir / "reasoning_input.json", train_pool)
        for item in train_pool:
            key = key_for(item)
            entry = cache.get(key)
            if not (entry and entry.get("status") == "success" and entry.get("reasoning")):
                missing[key] = item

    gen_stats = {"api_generated": 0, "invalid": 0}
    if missing:
        if not bool(gen_cfg.get("enabled", True)):
            raise RuntimeError(
                f"Experiment {name}: {len(missing)} reasoning cache miss but generation disabled."
            )
        gen_stats = generate_missing(
            missing=missing, cache=cache, cache_path=cache_path, gen_cfg=gen_cfg,
            prompt_version=prompt_version, min_chars=min_chars, max_chars=max_chars,
        )

    # 2) 由當下 train_pool 重新組裝 train_with_reasoning 與 SFT
    sft_manifest_folds = []
    total_failed = 0
    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.split("_")[-1])
        train_pool = fold_pools[fold_dir]
        cache_hit = 0
        with_reasoning = []
        sft_records = []
        for item in train_pool:
            key = key_for(item)
            entry = cache[key]
            cache_hit += 1
            reasoning = entry["reasoning"]
            with_reasoning.append({**item, "reasoning": reasoning, "failed": False})
            answer = build_xml_answer(reasoning, item["label"])
            _assert_sft_answer(answer, item["label"])
            sft_records.append(
                {
                    "paragraph_id": item["paragraph_id"],
                    "label": item["label"],
                    "source": item["source"],
                    "conversations": [
                        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": CLASSIFY_USER_TEMPLATE_XML.format(
                                definitions=CATEGORY_DEFINITIONS, text=item["combined_text"]
                            ),
                        },
                        {"role": "assistant", "content": answer},
                    ],
                }
            )

        rng = random.Random(shuffle_seed + fold_idx)
        rng.shuffle(sft_records)

        atomic_write_json(reasoning_exp / fold_dir / "train_with_reasoning.json", with_reasoning)
        report = {
            "fold": fold_idx,
            "train_pool_size": len(train_pool),
            "cache_hit": cache_hit,
            "cache_miss_at_start": sum(1 for it in train_pool if key_for(it) in missing),
            "api_generated_total": gen_stats["api_generated"],
            "invalid_total": gen_stats["invalid"],
            "failed": 0,
        }
        atomic_write_json(reasoning_exp / fold_dir / "reasoning_report.json", report)

        human_val = load_json(step4_exp / fold_dir / "human_val.json")
        atomic_write_json(sft_exp / fold_dir / "train_sft_text.json", sft_records)
        atomic_write_json(sft_exp / fold_dir / "val_eval.json", human_val)
        atomic_write_json(sft_exp / fold_dir / "excluded_items.json", [])

        sft_manifest_folds.append(
            {
                "fold": fold_idx,
                "sft_size": len(sft_records),
                "val_size": len(human_val),
                "reasoning_failed": 0,
                "train_sft_text": output_file_record(sft_exp / fold_dir / "train_sft_text.json"),
                "val_eval": output_file_record(sft_exp / fold_dir / "val_eval.json"),
            }
        )
        logger.info("Experiment %s %s: sft=%s val=%s", name, fold_dir, len(sft_records), len(human_val))

    manifest = build_stage_manifest(
        stage="step5_reasoning_sft",
        run_id=run_id,
        ground_truth=step4_manifest.get("ground_truth", {}),
        config_subset={"cache": cache_cfg, "generation_model": model, "sft": sft_cfg},
        upstream={"step4_manifest": output_file_record(step4_exp / "manifest.json")},
        code_paths=CODE_PATHS,
        outputs={"folds": sft_manifest_folds},
        extra={
            "experiment_name": name,
            "reasoning_failed_total": total_failed,
            "gen_stats": gen_stats,
        },
    )
    atomic_write_json(sft_exp / "manifest.json", manifest)
    return manifest


def _assert_sft_answer(answer: str, label: str) -> None:
    rmatch = re.search(r"(?is)<reasoning>\s*(.*?)\s*</reasoning>", answer)
    lmatch = re.search(r"(?is)<label>\s*(.*?)\s*</label>", answer)
    if not rmatch or not rmatch.group(1).strip():
        raise ValueError("SFT answer missing non-empty <reasoning>")
    if not lmatch or canonicalize_label(lmatch.group(1)) != label:
        raise ValueError(f"SFT answer <label> invalid or != ground-truth {label!r}")


def main() -> None:
    config = load_config()
    run_id = get_run_id(config)
    cfg = config["step5_reasoning"]
    paths = cfg["paths_final_v1"]

    step4_root = resolve_path(paths["step4_output_root"])
    reasoning_root = resolve_path(paths["reasoning_output_root"])
    sft_root = resolve_path(paths["sft_output_root"])
    cache_path = resolve_path(cfg["cache"]["dir"]) / "reasoning_cache.json"

    cache = load_cache(cache_path)
    logger.info("Loaded reasoning cache with %s entries from %s", len(cache), cache_path)

    summaries = {}
    for name in cfg["experiments"]:
        manifest = process_experiment(
            name=name, step4_root=step4_root, reasoning_root=reasoning_root, sft_root=sft_root,
            cache=cache, cache_path=cache_path, cfg=cfg, run_id=run_id,
        )
        summaries[name] = {
            "folds": len(manifest["outputs"]["folds"]),
            "reasoning_failed_total": manifest["reasoning_failed_total"],
        }

    atomic_write_json(
        sft_root / "manifest.json",
        {"stage": "step5_reasoning_sft", "run_id": run_id, "experiments": summaries},
    )
    logger.info("Step 5 final_v1 complete: %s", summaries)


if __name__ == "__main__":
    main()
