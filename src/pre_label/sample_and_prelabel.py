"""
Stage 3a: Stratified sampling & pre-annotation for Label Studio.

Reads classified paragraphs, performs stratified sampling, and exports
data in Label Studio JSON format with FinBERT pre-annotations.

Usage:
    python -m src.pre_label.sample_and_prelabel [--sample-pct 0.6] [--min-per-cat 20]
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

from src.logger import setup_logger

logger = setup_logger(__name__)

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


def stratified_sample(
    records: list[dict],
    sample_pct: float = 0.00055,

    min_per_category: int = 20,
    seed: int = 42,
) -> list[dict]:
    """
    Perform stratified sampling ensuring each category has at least
    *min_per_category* samples (if available).

    Args:
        records: Classified paragraph records.
        sample_pct: Proportion to sample (0.6% = 0.006).
        min_per_category: Minimum samples per category.
        seed: Random seed for reproducibility.

    Returns:
        Sampled records.
    """
    random.seed(seed)

    # Group by category
    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_category[r["predicted_category"]].append(r)

    sampled = []
    for cat in ESG_CATEGORIES:
        pool = by_category.get(cat, [])
        if not pool:
            logger.warning("Category '%s' has 0 paragraphs — skipped", cat)
            continue

        # Target count: max(min_per_category, pct * pool_size)
        target = max(min_per_category, int(len(pool) * sample_pct))
        target = min(target, len(pool))  # can't sample more than available

        selected = random.sample(pool, target)
        sampled.extend(selected)
        logger.info("Category '%s': %d / %d sampled", cat, len(selected), len(pool))

    # Shuffle to mix categories
    random.shuffle(sampled)
    logger.info("Total sampled: %d paragraphs", len(sampled))
    return sampled


def sort_by_confidence(records: list[dict]) -> list[dict]:
    """Sort records by confidence ascending (low-confidence first for review)."""
    return sorted(records, key=lambda r: r.get("confidence", 0))


def export_label_studio(records: list[dict], output_path: str):
    """
    Export in Label Studio JSON format with FinBERT pre-annotations.

    Requires for pre-annotations to display in Label Studio:
    - 'model_version' in predictions (identifies as AI prediction)
    - unique 'id' per result item
    """
    ls_data = []
    for i, r in enumerate(records):
        item = {
            "data": {
                "text": r["text"],
                "paragraph_id": r["paragraph_id"],
                "ticker": r["ticker"],
                "year": r["year"],
                "char_count": r["char_count"],
                "finbert_category": r["predicted_category"],
                "finbert_confidence": r["confidence"],
            },
            "predictions": [
                {
                    "model_version": "finbert-esg-9-categories",
                    "score": r["confidence"],
                    "result": [
                        {
                            "id": f"finbert_{i}",
                            "value": {"choices": [r["predicted_category"]]},
                            "from_name": "label",
                            "to_name": "text",
                            "type": "choices",
                        }
                    ],
                }
            ],
        }
        ls_data.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ls_data, f, ensure_ascii=False, indent=2)

    logger.info("Exported %d items for Label Studio -> %s", len(ls_data), output_path)



def export_sampled_csv(records: list[dict], output_path: str):
    """Export sampled paragraphs as CSV for reference."""
    fieldnames = [
        "paragraph_id", "ticker", "year", "text", "char_count",
        "predicted_category", "confidence", "is_truncated",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    logger.info("Exported sampled CSV -> %s", output_path)


def run_sampling(
    input_path: str = "data/10k_1A/classified.json",
    output_dir: str = "data/10k_1A",
    sample_pct: float = 0.00055,

    min_per_category: int = 20,
    seed: int = 42,
):
    """Run stratified sampling and export pipeline (Steps 15-19)."""
    logger.info("=" * 60)
    logger.info("Starting stratified sampling & pre-labeling")
    logger.info("=" * 60)

    # Load classified data
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded %d classified paragraphs", len(records))

    # Stratified sample
    sampled = stratified_sample(records, sample_pct, min_per_category, seed)

    # Sort by confidence (low first)
    sampled = sort_by_confidence(sampled)

    # Export
    os.makedirs(output_dir, exist_ok=True)
    export_label_studio(sampled, os.path.join(output_dir, "label_studio_import.json"))
    export_sampled_csv(sampled, os.path.join(output_dir, "sampled_paragraphs.csv"))

    # Summary stats
    cat_counts = Counter(r["predicted_category"] for r in sampled)
    logger.info("=== Sampling Summary ===")
    for cat in ESG_CATEGORIES:
        logger.info("  %s: %d", cat, cat_counts.get(cat, 0))

    return sampled


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Sampling & Pre-labeling")
    parser.add_argument("--input", default="data/10k_1A/classified.json")
    parser.add_argument("--output-dir", default="data/10k_1A")
    parser.add_argument("--sample-pct", type=float, default=0.00055,
                        help="Sampling proportion (default: 0.055%%)")  # ~500 samples

    parser.add_argument("--min-per-cat", type=int, default=20,
                        help="Minimum samples per category")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_sampling(args.input, args.output_dir, args.sample_pct, args.min_per_cat, args.seed)


if __name__ == "__main__":
    main()
