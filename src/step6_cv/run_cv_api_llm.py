from __future__ import annotations

import logging
import os
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step4_cv.common import load_step4_config, resolve_experiment_dir, resolve_experiment_name, resolve_results_dir
from src.step6_cv.common import (
    build_eval_messages,
    compute_overall_metrics,
    extract_label,
    load_json,
    load_step6_cv_config,
    resolve_path,
    save_json,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

REASONING_AND_LABEL_INSTRUCTION = (
    "- Return the final answer using exactly these two lines:\n"
    "  Reasoning: <brief explanation>\n"
    "  Label: <one valid category name>"
)
LABEL_ONLY_INSTRUCTION = "- Return the final answer using exactly one line:\n  Label: <one valid category name>"
SYSTEM_REASONING_SENTENCE = "Your job is to determine the single best label among the 9 ESG categories and explain the reasoning clearly."
SYSTEM_LABEL_ONLY_SENTENCE = "Your job is to determine the single best label among the 9 ESG categories and output only the final label."


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


class RateLimiter:
    def __init__(self, limits: dict[str, Any]):
        self.rpm = int(limits.get("rpm", 0) or 0)
        self.rpd = int(limits.get("rpd", 0) or 0)
        self.tpm = int(limits.get("tpm", 0) or 0)
        self.tpd = int(limits.get("tpd", 0) or 0)
        self.input_tpm = int(limits.get("input_tpm", 0) or 0)
        self.output_tpm = int(limits.get("output_tpm", 0) or 0)
        self.minute_events: deque[tuple[float, int, int]] = deque()
        self.day_events: deque[tuple[float, int, int]] = deque()

    def _prune(self, now: float) -> None:
        while self.minute_events and now - self.minute_events[0][0] >= 60.0:
            self.minute_events.popleft()
        while self.day_events and now - self.day_events[0][0] >= 86400.0:
            self.day_events.popleft()

    def wait_for_slot(self, estimated_input_tokens: int, estimated_output_tokens: int) -> None:
        while True:
            now = time.time()
            self._prune(now)
            wait_seconds = 0.0

            if self.rpm and len(self.minute_events) >= self.rpm:
                wait_seconds = max(wait_seconds, 60.0 - (now - self.minute_events[0][0]))
            if self.rpd and len(self.day_events) >= self.rpd:
                wait_seconds = max(wait_seconds, 86400.0 - (now - self.day_events[0][0]))

            minute_input = sum(item[1] for item in self.minute_events)
            minute_output = sum(item[2] for item in self.minute_events)
            day_total = sum(item[1] + item[2] for item in self.day_events)

            if self.tpm and minute_input + minute_output + estimated_input_tokens + estimated_output_tokens > self.tpm:
                wait_seconds = max(wait_seconds, 60.0 - (now - self.minute_events[0][0]))
            if self.input_tpm and minute_input + estimated_input_tokens > self.input_tpm:
                wait_seconds = max(wait_seconds, 60.0 - (now - self.minute_events[0][0]))
            if self.output_tpm and minute_output + estimated_output_tokens > self.output_tpm:
                wait_seconds = max(wait_seconds, 60.0 - (now - self.minute_events[0][0]))
            if self.tpd and day_total + estimated_input_tokens + estimated_output_tokens > self.tpd:
                wait_seconds = max(wait_seconds, 86400.0 - (now - self.day_events[0][0]))

            if wait_seconds <= 0:
                return

            sleep_seconds = max(wait_seconds, 0.5)
            logger.info("Rate limiter sleeping for %.2f seconds", sleep_seconds)
            time.sleep(sleep_seconds)

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        now = time.time()
        event = (now, input_tokens, output_tokens)
        self.minute_events.append(event)
        self.day_events.append(event)


def rewrite_prompt_for_label_only(content):
    if isinstance(content, str):
        return content.replace(REASONING_AND_LABEL_INSTRUCTION, LABEL_ONLY_INSTRUCTION)
    return content


def rewrite_system_for_label_only(content):
    if isinstance(content, str):
        return content.replace(SYSTEM_REASONING_SENTENCE, SYSTEM_LABEL_ONLY_SENTENCE)
    return content


def build_api_messages(item: dict[str, Any], label_only: bool) -> list[dict[str, str]]:
    messages = build_eval_messages(item, "gemma")
    if not label_only:
        return messages

    rewritten = []
    for message in messages:
        updated = dict(message)
        if updated["role"] == "system":
            updated["content"] = rewrite_system_for_label_only(updated["content"])
        elif updated["role"] == "user":
            updated["content"] = rewrite_prompt_for_label_only(updated["content"])
        rewritten.append(updated)
    return rewritten


def call_openai(system_prompt: str, user_prompt: str, model: str, api_key: str, max_output_tokens: int, model_cfg: dict[str, Any] | None = None) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=max_output_tokens,
    )
    return (response.output_text or "").strip()


def call_anthropic(system_prompt: str, user_prompt: str, model: str, api_key: str, max_output_tokens: int, model_cfg: dict[str, Any] | None = None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        system=system_prompt,
        max_tokens=max_output_tokens,
        temperature=0,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()


def call_gemini(system_prompt: str, user_prompt: str, model: str, api_key: str, max_output_tokens: int, model_cfg: dict[str, Any] | None = None) -> str:
    thinking_level = str((model_cfg or {}).get("thinking_level", "")).strip()

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config_kwargs: dict[str, Any] = {
            "temperature": 0,
            "max_output_tokens": max_output_tokens,
            "system_instruction": system_prompt,
        }
        if thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = getattr(response, "text", None)
        if text:
            return str(text).strip()
        raise RuntimeError("Gemini google.genai response returned no text output.")
    except ImportError:
        import google.generativeai as genai

    def finish_reason_to_text(value: Any) -> str:
        if value is None:
            return "UNKNOWN"
        name = getattr(value, "name", None)
        if name:
            return str(name)
        return str(value)

    def extract_text_or_raise(response) -> str:
        try:
            text = (response.text or "").strip()
            if text:
                return text
        except Exception:
            pass

        candidates = getattr(response, "candidates", None) or []
        finish_reasons = []
        parts_text: list[str] = []
        for candidate in candidates:
            finish_reasons.append(finish_reason_to_text(getattr(candidate, "finish_reason", None)))
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    parts_text.append(str(text))

        if parts_text:
            return "\n".join(parts_text).strip()

        raise RuntimeError(f"Gemini returned no text output. finish_reasons={finish_reasons}")

    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model_name=model, system_instruction=system_prompt)
    if thinking_level:
        logger.warning(
            "google.genai is not installed; Gemini thinking_level=%s cannot be applied with deprecated google.generativeai fallback.",
            thinking_level,
        )
    candidate_budgets = [max_output_tokens]
    for fallback_budget in [128, 256]:
        if fallback_budget > max_output_tokens:
            candidate_budgets.append(fallback_budget)

    last_error: Exception | None = None
    for budget in candidate_budgets:
        try:
            response = gen_model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=budget,
                ),
            )
            return extract_text_or_raise(response)
        except Exception as exc:
            last_error = exc
            error_text = str(exc)
            is_max_tokens = "finish_reason" in error_text and ("2" in error_text or "MAX_TOKENS" in error_text)
            if is_max_tokens and budget != candidate_budgets[-1]:
                logger.warning(
                    "Gemini hit MAX_TOKENS with budget=%s. Retrying with larger max_output_tokens.",
                    budget,
                )
                continue
            raise

    raise RuntimeError(f"Gemini failed to return text output for model={model}") from last_error


def get_caller(provider: str):
    if provider == "openai":
        return call_openai
    if provider == "anthropic":
        return call_anthropic
    if provider == "gemini":
        return call_gemini
    raise ValueError(f"Unsupported provider: {provider}")


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_json(path)
    return {row["paragraph_id"]: row for row in rows}


def summarize_fold_metrics(rows: list[dict]) -> dict:
    metrics = ["accuracy", "macro_f1", "weighted_f1", "kappa"]
    summary = {}
    for metric in metrics:
        values = [row[metric] for row in rows]
        summary[f"{metric}_mean"] = round(float(statistics.mean(values)), 4)
        summary[f"{metric}_std"] = round(float(statistics.pstdev(values)), 4) if len(values) > 1 else 0.0
    return summary


def build_cv_summary_markdown(model_name: str, fold_rows: list[dict], summary: dict) -> str:
    fold_names = ", ".join(f"fold_{row['fold']}" for row in fold_rows)
    lines = [
        "# API LLM CV Evaluation Summary",
        "",
        f"- Model: `{model_name}`",
        f"- Folds: `{fold_names}`",
        "",
        "## Per-Fold Metrics",
        "",
        "| Fold | Samples | Accuracy | Macro F1 | Weighted F1 | Kappa |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in fold_rows:
        lines.append(
            f"| {row['fold']} | {row['samples']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['weighted_f1']:.4f} | {row['kappa']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Mean And Std",
            "",
            "| Metric | Mean | Std |",
            "| --- | ---: | ---: |",
            f"| Accuracy | {summary['accuracy_mean']:.4f} | {summary['accuracy_std']:.4f} |",
            f"| Macro F1 | {summary['macro_f1_mean']:.4f} | {summary['macro_f1_std']:.4f} |",
            f"| Weighted F1 | {summary['weighted_f1_mean']:.4f} | {summary['weighted_f1_std']:.4f} |",
            f"| Kappa | {summary['kappa_mean']:.4f} | {summary['kappa_std']:.4f} |",
            "",
        ]
    )
    return "\n".join(lines)


def write_model_summary(results_dir: Path, model_name: str, fold_rows: list[dict]) -> None:
    ordered = sorted(fold_rows, key=lambda row: row["fold"])
    summary = summarize_fold_metrics(ordered)
    save_json(results_dir / "overall_folds.json", ordered)
    save_json(results_dir / "overall_summary.json", {"folds": ordered, "summary": summary})
    markdown = build_cv_summary_markdown(model_name, ordered, summary)
    (results_dir / "overall_summary.md").write_text(markdown, encoding="utf-8")


def evaluate_model(model_key: str, model_cfg: dict[str, Any], api_cfg: dict[str, Any]) -> None:
    provider = str(model_cfg["provider"])
    model_name = str(model_cfg["model"])
    api_key = os.getenv(str(model_cfg["api_key_env"]))
    if not api_key:
        raise ValueError(f"Missing API key env var for {model_key}: {model_cfg['api_key_env']}")

    experiment_name = resolve_experiment_name(load_step4_config())
    step4_output_dir = resolve_experiment_dir(resolve_path(api_cfg["step4_output_dir"]), experiment_name)
    results_root = resolve_results_dir(resolve_path(api_cfg["results_root"]), experiment_name) / "api_llm" / model_key
    results_root.mkdir(parents=True, exist_ok=True)

    label_only = bool(api_cfg.get("label_only", True))
    max_output_tokens = int(model_cfg.get("max_output_tokens", api_cfg["max_output_tokens"]))
    max_retries = int(api_cfg["max_retries"])
    retry_backoff_seconds = float(api_cfg["retry_backoff_seconds"])
    caller = get_caller(provider)
    rate_limiter = RateLimiter(model_cfg.get("rate_limits", {}))

    manifest = load_json(step4_output_dir / "manifest.json")
    fold_summaries = []

    for fold_name in manifest["fold_directories"]:
        fold_idx = int(fold_name.split("_")[-1])
        human_val = load_json(step4_output_dir / fold_name / "human_val.json")
        output_path = results_root / f"fold_{fold_idx}_results.json"
        existing = load_existing_results(output_path)
        ordered_results: list[dict[str, Any]] = []

        for idx, item in enumerate(human_val, start=1):
            paragraph_id = item["paragraph_id"]
            cached = existing.get(paragraph_id)
            if cached is not None:
                ordered_results.append(cached)
                continue

            messages = build_api_messages(item, label_only=label_only)
            system_prompt = str(messages[0]["content"])
            user_prompt = str(messages[1]["content"])
            estimated_input_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
            last_error: Exception | None = None
            raw_output = ""

            for attempt in range(1, max_retries + 1):
                rate_limiter.wait_for_slot(estimated_input_tokens, max_output_tokens)
                try:
                    raw_output = caller(system_prompt, user_prompt, model_name, api_key, max_output_tokens, model_cfg)
                    rate_limiter.record_usage(estimated_input_tokens, estimate_tokens(raw_output))
                    break
                except Exception as exc:
                    rate_limiter.record_usage(estimated_input_tokens, 0)
                    last_error = exc
                    logger.warning(
                        "%s fold %s sample %s attempt %s failed: %s",
                        model_key,
                        fold_idx,
                        paragraph_id,
                        attempt,
                        exc,
                    )
                    time.sleep(retry_backoff_seconds * attempt)
            else:
                raise RuntimeError(f"Failed API inference for {model_key} fold {fold_idx} paragraph_id={paragraph_id}") from last_error

            row = {
                "fold": fold_idx,
                "paragraph_id": paragraph_id,
                "raw_output": raw_output,
                "parsed_label": extract_label(raw_output),
                "ground_truth_label": item["label"],
            }
            existing[paragraph_id] = row
            ordered_results.append(row)
            save_json(output_path, ordered_results)

            if idx <= 3 or idx % 10 == 0 or idx == len(human_val):
                logger.info(
                    "%s fold %s sample %s/%s pred=%s gold=%s",
                    model_key,
                    fold_idx,
                    idx,
                    len(human_val),
                    row["parsed_label"],
                    row["ground_truth_label"],
                )

        fold_metrics = compute_overall_metrics(ordered_results)
        fold_metrics["fold"] = fold_idx
        save_json(results_root / f"fold_{fold_idx}_metrics.json", fold_metrics)
        fold_summaries.append(fold_metrics)
        logger.info(
            "%s fold %s complete: acc=%.4f macro_f1=%.4f",
            model_key,
            fold_idx,
            fold_metrics["accuracy"],
            fold_metrics["macro_f1"],
        )

    write_model_summary(results_root, model_name, fold_summaries)
    logger.info("Saved API LLM summary to %s", results_root / "overall_summary.md")


def main() -> None:
    config = load_step6_cv_config()
    api_cfg = config.get("api_llm", {})
    if not api_cfg.get("enabled", False):
        logger.info("step6_cv.api_llm.enabled is false; nothing to run.")
        return

    models = api_cfg.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Missing step6_cv.api_llm.models in config/config.yaml")

    for model_key, model_cfg in models.items():
        if not isinstance(model_cfg, dict) or not model_cfg.get("enabled", False):
            continue
        evaluate_model(model_key=model_key, model_cfg=model_cfg, api_cfg=api_cfg)


if __name__ == "__main__":
    main()
