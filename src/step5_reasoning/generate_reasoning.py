from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step5_reasoning.common import (
    ESG_CATEGORIES,
    load_json,
    load_step5_config,
    normalize_reasoning_text,
    resolve_path,
    save_json,
    validate_reasoning,
)
from src.step5_reasoning.reason import REASONING_SYSTEM_PROMPT, build_reasoning_batch_prompt


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def extract_json_block(text: str) -> Any:
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        import json

        return json.loads(cleaned)
    except Exception:
        pass

    left = cleaned.find("[")
    right = cleaned.rfind("]")
    if left >= 0 and right > left:
        import json

        return json.loads(cleaned[left : right + 1])

    raise ValueError("Could not parse JSON from reasoning response.")


def call_openai(prompt: str, model: str, api_key: str, max_output_tokens: int, reasoning_effort: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=REASONING_SYSTEM_PROMPT,
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
        temperature=0.2,
        system=REASONING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()


def call_gemini(prompt: str, model: str, api_key: str, max_output_tokens: int) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model_name=model, system_instruction=REASONING_SYSTEM_PROMPT)
    response = gen_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
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
    raise ValueError(f"Unsupported reasoning provider: {provider}")


def load_existing_reasonings(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = load_json(path)
    results: dict[str, dict[str, Any]] = {}
    for item in payload:
        paragraph_id = item["paragraph_id"]
        results[paragraph_id] = item
    return results


def build_failed_only_items(existing: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    failed_items = []
    for item in existing.values():
        if item.get("failed", False):
            failed_items.append(
                {
                    "paragraph_id": item["paragraph_id"],
                    "combined_text": item["combined_text"],
                    "label": item["label"],
                    "source": item["source"],
                }
            )
    return failed_items


def build_pending_items(train_pool: list[dict[str, Any]], existing: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pending = []
    for item in train_pool:
        existing_item = existing.get(item["paragraph_id"])
        if existing_item and not existing_item.get("failed", False) and existing_item.get("reasoning"):
            continue
        pending.append(
            {
                "paragraph_id": item["paragraph_id"],
                "combined_text": item["combined_text"],
                "label": item["label"],
                "source": item["source"],
            }
        )
    return pending


def batch_items(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def generate_batch_reasonings(
    batch: list[dict[str, Any]],
    caller,
    provider: str,
    model: str,
    api_key: str,
    max_output_tokens: int,
    reasoning_effort: str,
) -> list[dict[str, Any]]:
    prompt = build_reasoning_batch_prompt(batch)
    if provider == "openai":
        raw = caller(prompt, model, api_key, max_output_tokens, reasoning_effort)
    else:
        raw = caller(prompt, model, api_key, max_output_tokens)
    payload = extract_json_block(raw)
    if not isinstance(payload, list):
        raise ValueError("Reasoning response must be a JSON array.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reasoning for Step 5 train pools")
    parser.add_argument("--failed-only", action="store_true", help="Only retry items currently marked as failed.")
    parser.add_argument("--fold", type=int, choices=[0, 1, 2], default=None, help="Only run a specific fold.")
    args = parser.parse_args()

    config = load_step5_config()
    paths_cfg = config["paths"]
    generation_cfg = config["generation"]

    step4_output_dir = resolve_path(paths_cfg["step4_output_dir"])
    output_dir = resolve_path(paths_cfg["output_dir"])

    manifest = load_json(step4_output_dir / "manifest.json")
    batch_size = int(generation_cfg["batch_size"])
    min_chars = int(generation_cfg["min_reasoning_chars"])
    max_chars = int(generation_cfg["max_reasoning_chars"])

    fold_names = manifest["fold_directories"]
    if args.fold is not None:
        fold_names = [f"fold_{args.fold}"]

    for fold_name in fold_names:
        fold_idx = int(fold_name.split("_")[-1])
        fold_step4_dir = step4_output_dir / fold_name
        fold_output_dir = output_dir / fold_name
        train_pool = load_json(fold_step4_dir / "train_pool.json")

        reasoning_input_path = fold_output_dir / "reasoning_input.json"
        reasoning_output_path = fold_output_dir / "train_with_reasoning.json"
        reasoning_report_path = fold_output_dir / "reasoning_report.json"

        save_json(reasoning_input_path, train_pool)
        existing = load_existing_reasonings(reasoning_output_path)
        if args.failed_only:
            pending = build_failed_only_items(existing)
        else:
            pending = build_pending_items(train_pool, existing)

        report = {
            "fold": fold_idx,
            "train_pool_size": len(train_pool),
            "existing_complete": len(train_pool) - len(pending),
            "pending_count": len(pending),
            "generation_enabled": bool(generation_cfg.get("enabled", False)),
            "failed_only": bool(args.failed_only),
            "batches": [],
        }

        if not generation_cfg.get("enabled", False):
            save_json(reasoning_report_path, report)
            logger.info("Fold %s reasoning generation disabled; wrote input and report only.", fold_idx)
            continue

        provider = str(generation_cfg["provider"])
        model = str(generation_cfg["model"])
        api_key = os.getenv(str(generation_cfg["api_key_env"]))
        if not api_key:
            raise ValueError(f"Missing API key env var: {generation_cfg['api_key_env']}")

        caller = get_caller(provider)
        max_retries = int(generation_cfg["max_retries"])
        max_output_tokens = int(generation_cfg["max_output_tokens"])
        sleep_seconds = float(generation_cfg["sleep_seconds"])
        retry_backoff_seconds = float(generation_cfg["retry_backoff_seconds"])
        reasoning_effort = str(generation_cfg.get("reasoning_effort", "low"))

        batches = batch_items(pending, batch_size=batch_size)
        for batch_index, batch in enumerate(batches):
            last_error: Exception | None = None
            batch_report = {
                "batch_index": batch_index,
                "requested_count": len(batch),
                "status": "failed",
                "attempts": [],
            }
            for attempt in range(1, max_retries + 1):
                try:
                    payload = generate_batch_reasonings(
                        batch=batch,
                        caller=caller,
                        provider=provider,
                        model=model,
                        api_key=api_key,
                        max_output_tokens=max_output_tokens,
                        reasoning_effort=reasoning_effort,
                    )

                    by_id = {item["paragraph_id"]: item for item in payload if isinstance(item, dict) and item.get("paragraph_id")}
                    batch_results = []
                    for source_item in batch:
                        candidate = by_id.get(source_item["paragraph_id"])
                        reasoning = normalize_reasoning_text(candidate.get("reasoning", "")) if candidate else ""
                        failed = not validate_reasoning(reasoning, min_chars=min_chars, max_chars=max_chars)
                        batch_results.append(
                            {
                                **source_item,
                                "reasoning": reasoning,
                                "failed": failed,
                            }
                        )

                    for result in batch_results:
                        existing[result["paragraph_id"]] = result

                    save_json(reasoning_output_path, list(existing.values()))
                    batch_report["status"] = "success"
                    batch_report["accepted_count"] = sum(1 for item in batch_results if not item["failed"])
                    batch_report["failed_count"] = sum(1 for item in batch_results if item["failed"])
                    batch_report["attempts"].append(
                        {
                            "attempt": attempt,
                            "returned_count": len(payload),
                            "accepted_count": batch_report["accepted_count"],
                        }
                    )
                    time.sleep(sleep_seconds)
                    break
                except Exception as exc:
                    last_error = exc
                    batch_report["attempts"].append({"attempt": attempt, "error": str(exc)})
                    time.sleep(sleep_seconds + retry_backoff_seconds * attempt)
            else:
                raise RuntimeError(f"Failed reasoning batch {batch_index} in fold {fold_idx}") from last_error

            report["batches"].append(batch_report)
            save_json(reasoning_report_path, report)
            logger.info(
                "Fold %s reasoning batch %s complete: accepted=%s failed=%s",
                fold_idx,
                batch_index,
                batch_report.get("accepted_count", 0),
                batch_report.get("failed_count", 0),
            )

        final_items = list(existing.values())
        report["completed_count"] = sum(1 for item in final_items if not item.get("failed", False))
        report["failed_count"] = sum(1 for item in final_items if item.get("failed", False))
        save_json(reasoning_output_path, final_items)
        save_json(reasoning_report_path, report)


if __name__ == "__main__":
    main()
