from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step4_cv.assemble import load_synthetic_records, normalize_synthetic_record
from src.step4_cv.common import (
    load_json,
    load_step4_config,
    resolve_experiment_dir,
    resolve_experiment_name,
    resolve_path,
    save_json,
)
from src.step4_cv.prompts import SYNTHETIC_SYSTEM_PROMPT, build_synthetic_user_prompt


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)

load_dotenv()


def extract_json_block(text: str) -> Any:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    for candidate in [cleaned]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise ValueError("Could not parse JSON from model response.")


def call_openai(
    prompt: str,
    model: str,
    api_key: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=SYNTHETIC_SYSTEM_PROMPT,
        input=prompt,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": reasoning_effort},
    )
    return (response.output_text or "").strip()


def call_anthropic(prompt: str, model: str, api_key: str, max_output_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        temperature=0.8,
        system=SYNTHETIC_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()


def call_gemini(prompt: str, model: str, api_key: str, max_output_tokens: int) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYNTHETIC_SYSTEM_PROMPT,
    )
    response = gen_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.8,
            max_output_tokens=max_output_tokens,
        ),
    )
    return response.text.strip()


def get_caller(provider: str):
    if provider == "openai":
        return call_openai
    if provider == "anthropic":
        return call_anthropic
    if provider == "gemini":
        return call_gemini
    raise ValueError(f"Unsupported synthetic provider: {provider}")


def assign_ids(
    fold_idx: int,
    label: str,
    batch_index: int,
    records: list[dict[str, Any]],
    offset: int,
) -> list[dict[str, Any]]:
    slug = (
        label.lower()
        .replace("&", "and")
        .replace(" ", "_")
        .replace("-", "_")
    )
    assigned: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        assigned.append(
            {
                **record,
                "paragraph_id": f"fold{fold_idx}_{slug}_syn_b{batch_index}_{offset + idx:04d}",
            }
        )
    return assigned


def save_generation_report(path: Path, report: dict[str, Any]) -> None:
    save_json(path, report)


def run_generation_batch(
    *,
    caller,
    provider: str,
    model: str,
    api_key: str,
    reasoning_effort: str,
    label: str,
    needed_count: int,
    seed_examples: list[dict[str, Any]],
    boundary_focus: bool,
    max_retries: int,
    sleep_seconds: float,
    retry_backoff_seconds: float,
    max_output_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt = build_synthetic_user_prompt(
        label=label,
        needed_count=needed_count,
        seed_examples=seed_examples,
        boundary_focus=boundary_focus,
    )

    attempts_log: list[dict[str, Any]] = []
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "openai":
                raw_text = caller(prompt, model, api_key, reasoning_effort, max_output_tokens)
            else:
                raw_text = caller(prompt, model, api_key, max_output_tokens)
            payload = extract_json_block(raw_text)
            if isinstance(payload, dict):
                payload = [payload]
            if not isinstance(payload, list):
                raise ValueError("Synthetic response JSON must be a list.")

            normalized = [
                normalize_synthetic_record(item, label)
                for item in payload
            ]
            normalized = normalized[:needed_count]
            attempts_log.append(
                {
                    "attempt": attempt,
                    "status": "success",
                    "requested_count": needed_count,
                    "returned_count": len(payload),
                    "accepted_count": len(normalized),
                }
            )
            time.sleep(sleep_seconds)
            return normalized, attempts_log
        except Exception as exc:
            last_error = exc
            attempts_log.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "requested_count": needed_count,
                    "error": str(exc),
                }
            )
            wait_seconds = sleep_seconds + retry_backoff_seconds * attempt
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Failed to generate synthetic data for label {label} after {max_retries} attempts."
    ) from last_error


def main() -> None:
    config = load_step4_config()
    paths_cfg = config["paths"]
    synthetic_cfg = config["synthetic_generation"]

    if not synthetic_cfg.get("enabled", False):
        logger.info("Synthetic API generation is disabled in config.step4_cv.synthetic_generation.enabled")
        return

    provider = str(synthetic_cfg["provider"])
    model = str(synthetic_cfg["model"])
    api_key_env = str(synthetic_cfg["api_key_env"])
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key env var: {api_key_env}")

    output_dir = resolve_experiment_dir(resolve_path(paths_cfg["output_dir"]), resolve_experiment_name(config))
    manifest = load_json(output_dir / "manifest.json")
    caller = get_caller(provider)
    batch_size = int(synthetic_cfg.get("max_samples_per_call", 20))
    max_retries = int(synthetic_cfg.get("max_retries", 3))
    reasoning_effort = str(synthetic_cfg.get("reasoning_effort", "low"))
    sleep_seconds = float(synthetic_cfg.get("sleep_seconds", 0.5))
    retry_backoff_seconds = float(synthetic_cfg.get("retry_backoff_seconds", 1.0))
    refresh_after_generation = bool(synthetic_cfg.get("refresh_after_generation", True))
    report_filename = str(synthetic_cfg.get("report_filename", "synthetic_generation_report.json"))
    max_output_tokens = int(synthetic_cfg.get("max_output_tokens", 18000))

    for fold_name in manifest["fold_directories"]:
        fold_dir = output_dir / fold_name
        fold_idx = int(fold_name.split("_")[-1])
        requests = load_json(fold_dir / "synthetic_requests.json")
        synthetic_path = fold_dir / str(synthetic_cfg["output_filename"])
        report_path = fold_dir / report_filename
        existing_records = load_synthetic_records(synthetic_path)
        existing_by_label = Counter(record["label"] for record in existing_records)
        all_records = list(existing_records)
        fold_report = {
            "fold": fold_idx,
            "provider": provider,
            "model": model,
            "existing_synthetic_total": len(existing_records),
            "labels": [],
            "generated_total": 0,
        }

        logger.info("Fold %s: existing synthetic=%s", fold_idx, len(existing_records))

        for request in requests:
            label = str(request["label"])
            requested_count = int(request["needed_count"])
            completed_count = existing_by_label.get(label, 0)
            remaining = requested_count - completed_count
            label_report = {
                "label": label,
                "requested_count": requested_count,
                "existing_count": completed_count,
                "generated_count": 0,
                "final_count": completed_count,
                "completed": remaining <= 0,
                "batches": [],
            }
            fold_report["labels"].append(label_report)
            save_generation_report(report_path, fold_report)

            if remaining <= 0:
                continue

            batch_index = 0
            while remaining > 0:
                current_batch_size = min(batch_size, remaining)
                normalized, attempts_log = run_generation_batch(
                    caller=caller,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    reasoning_effort=reasoning_effort,
                    label=label,
                    needed_count=current_batch_size,
                    seed_examples=request["seed_examples"],
                    boundary_focus=bool(request["boundary_focus"]),
                    max_retries=max_retries,
                    sleep_seconds=sleep_seconds,
                    retry_backoff_seconds=retry_backoff_seconds,
                    max_output_tokens=max_output_tokens,
                )
                normalized = assign_ids(
                    fold_idx=fold_idx,
                    label=label,
                    batch_index=batch_index,
                    records=normalized,
                    offset=existing_by_label.get(label, 0),
                )
                all_records.extend(normalized)
                existing_by_label[label] += len(normalized)
                remaining = requested_count - existing_by_label[label]
                save_json(synthetic_path, all_records)

                batch_report = {
                    "batch_index": batch_index,
                    "requested_count": current_batch_size,
                    "generated_count": len(normalized),
                    "remaining_after_batch": max(remaining, 0),
                    "attempts": attempts_log,
                }
                label_report["batches"].append(batch_report)
                label_report["generated_count"] += len(normalized)
                label_report["final_count"] = existing_by_label[label]
                label_report["completed"] = remaining <= 0
                fold_report["generated_total"] += len(normalized)
                save_generation_report(report_path, fold_report)

                logger.info(
                    "Fold %s label %s batch %s: generated %s records, remaining=%s",
                    fold_idx,
                    label,
                    batch_index,
                    len(normalized),
                    max(remaining, 0),
                )
                batch_index += 1

            save_generation_report(report_path, fold_report)

        save_generation_report(report_path, fold_report)
        logger.info(
            "Fold %s synthetic generation complete: total_generated=%s total_records=%s",
            fold_idx,
            fold_report["generated_total"],
            len(all_records),
        )

    if refresh_after_generation:
        run_script = Path(__file__).resolve().parent / "run.py"
        logger.info("Refreshing step4 train pools via %s", run_script)
        subprocess.run(
            ["uv", "run", "python", str(run_script)],
            cwd=str(Path(__file__).resolve().parents[2]),
            check=True,
        )


if __name__ == "__main__":
    main()
