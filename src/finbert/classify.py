"""
Stage 2: FinBERT ESG 9-category classification.

Reads paragraphs from data/10k_1A/paragraphs.json, classifies each with
yiyanghkust/finbert-esg-9-categories, saves results with checkpointing.

Usage:
    python -m src.finbert.classify [--batch-size 32] [--checkpoint-every 5000]
"""

import argparse
import csv
import json
import os
from collections import Counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.logger import setup_logger

logger = setup_logger(__name__)

MODEL_NAME = "yiyanghkust/finbert-esg-9-categories"

ESG_CATEGORIES = [
    "Climate Change",
    "Natural Capital",
    "Pollution & Waste",
    "Human Capital",
    "Product Liability",
    "Community Relations",
    "Corporate Governance",
    "Business Ethics & Values",
    "Non-ESG",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: int = 0):
    """
    Load FinBERT-ESG model and tokenizer.
    Returns a HF text-classification pipeline.
    """
    logger.info("Loading model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device_arg = device if torch.cuda.is_available() else -1
    if device_arg >= 0:
        logger.info("Using GPU device %d (%s)", device_arg, torch.cuda.get_device_name(device_arg))
    else:
        logger.info("Using CPU (no CUDA available)")

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device_arg,
        truncation=True,
        max_length=512,
    )
    logger.info("Model loaded successfully")
    return pipe, tokenizer


# ---------------------------------------------------------------------------
# Token statistics
# ---------------------------------------------------------------------------

def compute_token_stats(tokenizer, texts: list[str], output_dir: str) -> list[bool]:
    """
    Compute token-length statistics for all texts.
    Returns a list of booleans indicating whether each text was truncated.

    Saves stats to data/10k_1A/token_stats.json.
    """
    logger.info("Computing token statistics for %d texts...", len(texts))

    token_lengths = []
    is_truncated = []

    # Process in chunks to avoid memory issues
    chunk_size = 1000
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        encoded = tokenizer(chunk, truncation=False, add_special_tokens=True)
        for ids in encoded["input_ids"]:
            length = len(ids)
            token_lengths.append(length)
            is_truncated.append(length > 512)

    # Statistics
    total = len(token_lengths)
    truncated_count = sum(is_truncated)
    truncated_pct = 100 * truncated_count / total if total else 0

    # Average truncation ratio for truncated paragraphs
    truncated_ratios = []
    for length, trunc in zip(token_lengths, is_truncated):
        if trunc:
            truncated_ratios.append((length - 512) / length)

    avg_trunc_ratio = (
        sum(truncated_ratios) / len(truncated_ratios) if truncated_ratios else 0
    )

    # Histogram buckets
    buckets = Counter()
    for length in token_lengths:
        bucket = (length // 100) * 100  # 0-99, 100-199, ...
        buckets[bucket] += 1

    stats = {
        "total_paragraphs": total,
        "truncated_count": truncated_count,
        "truncated_pct": round(truncated_pct, 2),
        "avg_truncation_ratio": round(avg_trunc_ratio, 4),
        "mean_token_length": round(sum(token_lengths) / total, 1) if total else 0,
        "max_token_length": max(token_lengths) if token_lengths else 0,
        "min_token_length": min(token_lengths) if token_lengths else 0,
        "token_length_histogram": {
            f"{k}-{k + 99}": v for k, v in sorted(buckets.items())
        },
    }

    stats_path = os.path.join(output_dir, "token_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("=== Token Statistics ===")
    logger.info("Total: %d, Truncated: %d (%.1f%%)", total, truncated_count, truncated_pct)
    logger.info("Avg truncation ratio (truncated only): %.2f%%", avg_trunc_ratio * 100)
    logger.info("Token length — mean: %.1f, min: %d, max: %d",
                stats["mean_token_length"], stats["min_token_length"], stats["max_token_length"])
    logger.info("Saved token stats to %s", stats_path)

    return is_truncated


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_batch(pipe, texts: list[str]) -> list[dict]:
    """
    Classify a list of texts using the FinBERT pipeline.
    Returns list of {'label': ..., 'score': ...} dicts.
    """
    results = pipe(texts)
    return results


def run_classification(
    input_path: str = "data/10k_1A/paragraphs.json",
    output_dir: str = "data/10k_1A",
    batch_size: int = 32,
    checkpoint_every: int = 5000,
    device: int = 0,
):
    """
    Run full classification pipeline (Steps 8-14).
    """
    logger.info("=" * 60)
    logger.info("Starting FinBERT classification")
    logger.info("=" * 60)

    # Load paragraphs
    logger.info("Loading paragraphs from %s", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        paragraphs = json.load(f)
    logger.info("Loaded %d paragraphs", len(paragraphs))

    # Load model
    pipe, tokenizer = load_model(device)

    # Token stats
    texts = [p["text"] for p in paragraphs]
    is_truncated_flags = compute_token_stats(tokenizer, texts, output_dir)

    # Mark truncation in records
    for p, trunc in zip(paragraphs, is_truncated_flags):
        p["is_truncated"] = trunc

    # Classify in batches
    checkpoint_path = os.path.join(output_dir, "classified_checkpoint.csv")
    fieldnames = [
        "paragraph_id", "ticker", "year", "text", "char_count",
        "predicted_category", "confidence", "is_truncated",
    ]

    classified = []
    total = len(paragraphs)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_results = classify_batch(pipe, batch_texts)

        for i, result in enumerate(batch_results):
            idx = start + i
            paragraphs[idx]["predicted_category"] = result["label"]
            paragraphs[idx]["confidence"] = round(result["score"], 4)
            classified.append(paragraphs[idx])

        processed = end
        if processed % 1000 < batch_size:
            logger.info("Classified %d / %d (%.1f%%)", processed, total, 100 * processed / total)

        # Checkpoint
        if processed % checkpoint_every < batch_size and processed > 0:
            _save_checkpoint(classified, checkpoint_path, fieldnames)
            logger.info("Checkpoint saved at %d paragraphs", processed)

    logger.info("Classification complete: %d paragraphs", len(classified))

    # Final export
    _export_results(classified, fieldnames, output_dir)

    # Statistics
    _compute_classification_stats(classified, output_dir)

    return classified


def _save_checkpoint(records: list[dict], path: str, fieldnames: list[str]):
    """Save intermediate results."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _export_results(records: list[dict], fieldnames: list[str], output_dir: str):
    """Export final classified data."""
    csv_path = os.path.join(output_dir, "classified.csv")
    json_path = os.path.join(output_dir, "classified.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info("Exported classified data -> %s, %s", csv_path, json_path)


def _compute_classification_stats(records: list[dict], output_dir: str):
    """Compute and log classification statistics."""
    total = len(records)
    cat_counts = Counter(r["predicted_category"] for r in records)

    esg_count = sum(v for k, v in cat_counts.items() if k != "Non-ESG")
    non_esg_count = cat_counts.get("Non-ESG", 0)

    # Per-year category distribution
    year_cat: dict[int, Counter] = {}
    for r in records:
        y = r["year"]
        if y not in year_cat:
            year_cat[y] = Counter()
        year_cat[y][r["predicted_category"]] += 1

    stats = {
        "total_classified": total,
        "category_distribution": {
            k: {"count": v, "pct": round(100 * v / total, 2)}
            for k, v in sorted(cat_counts.items(), key=lambda x: -x[1])
        },
        "esg_vs_non_esg": {
            "ESG": {"count": esg_count, "pct": round(100 * esg_count / total, 2)},
            "Non-ESG": {"count": non_esg_count, "pct": round(100 * non_esg_count / total, 2)},
        },
        "per_year": {
            y: dict(cats) for y, cats in sorted(year_cat.items())
        },
        "avg_confidence": round(
            sum(r["confidence"] for r in records) / total, 4
        ) if total else 0,
    }

    stats_path = os.path.join(output_dir, "classification_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("=== Classification Statistics ===")
    logger.info("Total classified: %d", total)
    logger.info("ESG: %d (%.1f%%) | Non-ESG: %d (%.1f%%)",
                esg_count, 100 * esg_count / total,
                non_esg_count, 100 * non_esg_count / total)
    for cat, info in stats["category_distribution"].items():
        logger.info("  %s: %d (%.1f%%)", cat, info["count"], info["pct"])
    logger.info("Average confidence: %.4f", stats["avg_confidence"])
    logger.info("Stats saved to %s", stats_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2: FinBERT ESG Classification")
    parser.add_argument("--input", default="data/10k_1A/paragraphs.json",
                        help="Input paragraphs JSON")
    parser.add_argument("--output-dir", default="data/10k_1A",
                        help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--checkpoint-every", type=int, default=5000,
                        help="Checkpoint interval (paragraphs)")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device index (-1 for CPU)")
    args = parser.parse_args()
    run_classification(args.input, args.output_dir, args.batch_size,
                       args.checkpoint_every, args.device)


if __name__ == "__main__":
    main()
