"""
Step 4: Balanced Sampling for ESG categories based on FinBERT predictions.

Usage:
  uv run src/step2_classification/sample_balanced.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CLASSIFIED = Path("data/processed/step2_classification/classified.json")
INPUT_SAMPLED = Path("data/processed/step2_classification/sampled_500.json")

OUTPUT_BALANCED_RAW = Path("data/processed/step2_classification/sampled_balanced.json")
OUTPUT_LABEL_STUDIO = Path("data/processed/step3_annotation/label_studio_balanced.json")

TARGET_SAMPLES_PER_CLASS = 60
RANDOM_SEED = 42

ESG_CATEGORIES = [
  "Climate Change",
  "Natural Capital",
  "Pollution & Waste",
  "Human Capital",
  "Product Liability",
  "Community Relations",
  "Corporate Governance",
  "Business Ethics & Values",
]


# ============================================================
# STRATIFIED SAMPLING
# ============================================================

def sample_by_confidence(paragraphs: list[dict], target_n: int) -> list[dict]:
  """Sample paragraphs stratified by FinBERT confidence (30% high, 40% mid, 30% low)."""
  if not paragraphs:
    return []

  # Sort by confidence descending
  sorted_paras = sorted(paragraphs, key=lambda x: x.get("finbert_confidence", 0.0), reverse=True)
  n = len(sorted_paras)

  # Split into High (top 30%), Mid (middle 40%), Low (bottom 30%)
  high_end = int(n * 0.3)
  mid_end = int(n * 0.7)

  high_conf = sorted_paras[:high_end]
  mid_conf = sorted_paras[high_end:mid_end]
  low_conf = sorted_paras[mid_end:]

  # Target allocations: 30%, 40%, 30% of target_n
  target_high = int(target_n * 0.3)
  target_mid = int(target_n * 0.4)
  target_low = target_n - target_high - target_mid

  # Adjust quotas if populations are smaller than targets
  high_actual = min(len(high_conf), target_high)
  mid_actual = min(len(mid_conf), target_mid)
  low_actual = min(len(low_conf), target_low)

  # Re-distribute missing quotas (simple greed fill)
  missing = target_n - (high_actual + mid_actual + low_actual)
  while missing > 0:
    added = False
    if high_actual < len(high_conf):
      high_actual += 1
      missing -= 1
      added = True
    elif mid_actual < len(mid_conf) and missing > 0:
      mid_actual += 1
      missing -= 1
      added = True
    elif low_actual < len(low_conf) and missing > 0:
      low_actual += 1
      missing -= 1
      added = True
    if not added:
      break

  # Randomly sample from groups
  sampled_high = random.sample(high_conf, high_actual)
  sampled_mid = random.sample(mid_conf, mid_actual)
  sampled_low = random.sample(low_conf, low_actual)

  sampled = sampled_high + sampled_mid + sampled_low
  return sampled


# ============================================================
# EXPORT
# ============================================================

def export_label_studio(records: list[dict], output_path: Path):
  """Export in Label Studio JSON format with FinBERT pre-annotations."""
  ls_data = []
  for r in records:
    finbert_label = r.get("finbert_label", "Non-ESG")
    finbert_conf = r.get("finbert_confidence", 0.0)

    item = {
      "data": {
        "combined_text": r["combined_text"],
        "paragraph_id": r["paragraph_id"],
        "ticker": r["ticker"],
        "filing_date": r.get("filing_date", ""),
        "risk_section": r.get("risk_section", ""),
        "risk_heading": r.get("risk_heading", ""),
        "finbert_label": finbert_label,
        "finbert_confidence": finbert_conf,
        "char_count": r["char_count"],
      },
      "predictions": [
        {
          "model_version": "finbert-esg-9-categories",
          "score": finbert_conf,
          "result": [
            {
              "from_name": "label",
              "to_name": "combined_text",
              "type": "choices",
              "value": {"choices": [finbert_label]},
            }
          ],
        }
      ],
    }
    ls_data.append(item)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(ls_data, f, ensure_ascii=False, indent=2)


# ============================================================
# MAIN
# ============================================================

def main():
  print("=" * 60)
  print(" Step 4: Balanced Sampling for ESG Categories")
  print("=" * 60)
  print(f" Input:     {INPUT_CLASSIFIED}")
  print(f" Exclude:    {INPUT_SAMPLED}")
  print(f" Target per:  {TARGET_SAMPLES_PER_CLASS}")
  print(f" Seed:     {RANDOM_SEED}")
  print("=" * 60)
  print()

  random.seed(RANDOM_SEED)

  # 1. Load data
  with open(INPUT_CLASSIFIED) as f:
    all_paras = json.load(f)
  print(f" Loaded {len(all_paras):,} paragraphs from pool")

  excluded_ids = set()
  if INPUT_SAMPLED.exists():
    with open(INPUT_SAMPLED) as f:
      sampled_paras = json.load(f)
      excluded_ids = {p["paragraph_id"] for p in sampled_paras}
  print(f" Excluding {len(excluded_ids)} previously sampled items")

  # 2. Filter available ESG paragraphs
  available_esg = defaultdict(list)
  for p in all_paras:
    if p["paragraph_id"] in excluded_ids:
      continue
    
    label = p.get("finbert_label", "Non-ESG")
    if label in ESG_CATEGORIES:
      available_esg[label].append(p)

  # 3. Sample for each category
  final_sampled = []
  print("\n --- Sampling by Category ---")
  for cat in ESG_CATEGORIES:
    candidates = available_esg[cat]
    sampled = sample_by_confidence(candidates, TARGET_SAMPLES_PER_CLASS)
    final_sampled.extend(sampled)
    print(f" {cat:<25}: Sampled {len(sampled):<3} / {len(candidates):,} available")

  print("\n --- Final Sample Statistics ---")
  print(f" Total balanced samples: {len(final_sampled)}")
  unique_tickers = len(set(p["ticker"] for p in final_sampled))
  print(f" Unique tickers:     {unique_tickers}")

  # 4. Export
  OUTPUT_BALANCED_RAW.parent.mkdir(parents=True, exist_ok=True)
  with open(OUTPUT_BALANCED_RAW, "w", encoding="utf-8") as f:
    json.dump(final_sampled, f, ensure_ascii=False, indent=2)

  export_label_studio(final_sampled, OUTPUT_LABEL_STUDIO)

  print("\n" + "=" * 60)
  print(" Sampling Complete")
  print("=" * 60)
  print(f" Raw output -> {OUTPUT_BALANCED_RAW}")
  print(f" Label Studio -> {OUTPUT_LABEL_STUDIO}")
  print("=" * 60)


if __name__ == "__main__":
  main()
