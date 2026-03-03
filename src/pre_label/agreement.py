"""
Stage 3c: Inter-annotator agreement calculation.

Reads completed human annotations (after Label Studio export),
computes Fleiss' Kappa and other agreement metrics.

This is a FRAMEWORK to be used after human annotation is complete.

Usage:
    python -m src.pre_label.agreement --annotations annotations.json
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

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


def fleiss_kappa(annotations: list[list[str]], categories: list[str]) -> float:
    """
    Compute Fleiss' Kappa for inter-annotator agreement.

    Args:
        annotations: List of lists. Each inner list contains the assigned
                     categories from each annotator for a single item.
                     E.g., [['Climate Change', 'Climate Change', 'Non-ESG'], ...]
        categories: List of all possible category labels.

    Returns:
        Fleiss' Kappa coefficient.
    """
    n_items = len(annotations)
    n_raters = len(annotations[0])
    n_categories = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Build count matrix: (n_items x n_categories)
    counts = np.zeros((n_items, n_categories), dtype=int)
    for i, item_annotations in enumerate(annotations):
        for label in item_annotations:
            if label in cat_to_idx:
                counts[i, cat_to_idx[label]] += 1

    # P_i: proportion of agreeing pairs for each item
    p_i = np.sum(counts ** 2, axis=1) - n_raters
    p_i = p_i / (n_raters * (n_raters - 1))

    # P_bar: mean of P_i
    p_bar = np.mean(p_i)

    # P_e: expected agreement by chance
    p_j = np.sum(counts, axis=0) / (n_items * n_raters)
    p_e = np.sum(p_j ** 2)

    # Kappa
    if p_e == 1:
        kappa = 1.0
    else:
        kappa = (p_bar - p_e) / (1 - p_e)

    return float(kappa)


def load_annotations(path: str) -> dict:
    """
    Load annotations from a JSON file.
    Expected format: list of {
        "paragraph_id": "...",
        "annotators": {
            "annotator_1": "category",
            "annotator_2": "category",
            ...
        }
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def compute_agreement(annotations_path: str, output_dir: str = "data/10k_1A"):
    """
    Compute inter-annotator agreement metrics.
    """
    logger.info("Loading annotations from %s", annotations_path)
    data = load_annotations(annotations_path)

    # Collect all annotator labels per item
    all_annotations = []
    for item in data:
        annotators = item.get("annotators", {})
        labels = list(annotators.values())
        if len(labels) >= 2:
            all_annotations.append(labels)

    if not all_annotations:
        logger.error("No items with 2+ annotations found")
        return

    # Ensure consistent number of raters (pad with None if needed)
    max_raters = max(len(a) for a in all_annotations)
    logger.info("Items: %d, Max raters: %d", len(all_annotations), max_raters)

    # Filter to items with consistent rater count
    consistent = [a for a in all_annotations if len(a) == max_raters]
    logger.info("Items with exactly %d raters: %d", max_raters, len(consistent))

    if len(consistent) < 2:
        logger.error("Not enough consistent annotations to compute Kappa")
        return

    kappa = fleiss_kappa(consistent, ESG_CATEGORIES)

    # Per-category agreement
    cat_agreement: dict[str, dict] = {}
    for cat in ESG_CATEGORIES:
        items_with_cat = [a for a in consistent if cat in a]
        if items_with_cat:
            agree = sum(1 for a in items_with_cat if len(set(a)) == 1)
            cat_agreement[cat] = {
                "items": len(items_with_cat),
                "full_agreement": agree,
                "agreement_pct": round(100 * agree / len(items_with_cat), 1),
            }

    results = {
        "fleiss_kappa": round(kappa, 4),
        "interpretation": _interpret_kappa(kappa),
        "n_items": len(consistent),
        "n_raters": max_raters,
        "per_category_agreement": cat_agreement,
    }

    output_path = os.path.join(output_dir, "agreement_metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("=== Agreement Metrics ===")
    logger.info("Fleiss' Kappa: %.4f (%s)", kappa, results["interpretation"])
    logger.info("Results saved to %s", output_path)

    return results


def _interpret_kappa(kappa: float) -> str:
    """Interpret Kappa value (Landis & Koch, 1977)."""
    if kappa < 0:
        return "Poor"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


def main():
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement (Fleiss' Kappa)"
    )
    parser.add_argument("--annotations", required=True,
                        help="Path to annotations JSON file")
    parser.add_argument("--output-dir", default="data/10k_1A")
    args = parser.parse_args()
    compute_agreement(args.annotations, args.output_dir)


if __name__ == "__main__":
    main()
