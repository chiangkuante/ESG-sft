"""
Step 2.3: Stratified Random Sampling & Annotation Preparation

Samples ~500 paragraphs from the FinBERT-classified pool (data/processed/step2_classification/classified.json),
stratified by ticker and year to ensure diversity. Splits into:
 - 200 fixed test set (original distribution, never used for training)
 - 300 training seed (for later merging with supplementary annotations)

Exports Label Studio JSON with FinBERT pre-annotations for accelerated human labeling.

Usage:
  uv run src/step2_classification/sample.py               # Default: 500 samples
  uv run src/step2_classification/sample.py --total 600 --test-size 200
  uv run src/step2_classification/sample.py --dry-run          # Preview without saving
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONSTANTS
# ============================================================

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

INPUT_PATH = Path("data/processed/step2_classification/classified.json")
OUTPUT_DIR = Path("data/processed/step2_classification")

# Label Studio labeling config with FinBERT info display
LABEL_CONFIG = """\
<View>
 <Header value="ESG Risk Factor Classification" />
 <View style="display:flex; gap:8px; margin-bottom:8px;">
  <View style="background:#e8f4fd; padding:6px 12px; border-radius:4px;">
   <Text name="ticker_info" value="$ticker" />
  </View>
  <View style="background:#f0f0f0; padding:6px 12px; border-radius:4px;">
   <Text name="date_info" value="$filing_date" />
  </View>
  <View style="background:#fff3cd; padding:6px 12px; border-radius:4px;">
   <Text name="finbert_info" value="FinBERT: $finbert_label ($finbert_confidence)" />
  </View>
 </View>
 <Text name="combined_text" value="$combined_text" />
 <Choices name="label" toName="combined_text" choice="single-required">
  <Choice value="Climate Change" />
  <Choice value="Natural Capital" />
  <Choice value="Pollution &amp; Waste" />
  <Choice value="Human Capital" />
  <Choice value="Product Liability" />
  <Choice value="Community Relations" />
  <Choice value="Corporate Governance" />
  <Choice value="Business Ethics &amp; Values" />
  <Choice value="Non-ESG" />
 </Choices>
</View>
"""


# ============================================================
# SAMPLING
# ============================================================

def stratified_sample(
  paragraphs: list[dict],
  total: int = 500,
  seed: int = 42,
) -> list[dict]:
  """Stratified random sampling by ticker, ensuring year diversity.

  Strategy:
  1. Group paragraphs by ticker
  2. Calculate per-ticker quota proportional to pool size
  3. Within each ticker, sample across available years
  4. Shuffle final result
  """
  random.seed(seed)

  # Group by ticker
  by_ticker: dict[str, list[dict]] = defaultdict(list)
  for p in paragraphs:
    by_ticker[p["ticker"]].append(p)

  n_tickers = len(by_ticker)
  total_pool = len(paragraphs)

  # Calculate per-ticker quotas (proportional, min 1)
  quotas = {}
  remaining = total
  for ticker, paras in sorted(by_ticker.items(), key=lambda x: len(x[1])):
    quota = max(1, round(len(paras) / total_pool * total))
    quota = min(quota, len(paras), remaining)
    quotas[ticker] = quota
    remaining -= quota
    if remaining <= 0:
      break

  # Distribute any remaining quota to largest tickers
  if remaining > 0:
    for ticker in sorted(by_ticker, key=lambda t: len(by_ticker[t]), reverse=True):
      if remaining <= 0:
        break
      available = len(by_ticker[ticker]) - quotas.get(ticker, 0)
      add = min(remaining, available)
      quotas[ticker] = quotas.get(ticker, 0) + add
      remaining -= add

  # Sample within each ticker, stratified by year
  sampled = []
  for ticker, quota in quotas.items():
    pool = by_ticker[ticker]

    # Group by year within this ticker
    by_year: dict[str, list[dict]] = defaultdict(list)
    for p in pool:
      year = p["filing_date"][:4] if p.get("filing_date") else "unknown"
      by_year[year].append(p)

    # Round-robin across years
    per_year = max(1, quota // len(by_year))
    selected = []
    for year, year_paras in sorted(by_year.items()):
      n = min(per_year, len(year_paras))
      selected.extend(random.sample(year_paras, n))

    # Fill up to quota if needed
    already = set(p["paragraph_id"] for p in selected)
    remaining_pool = [p for p in pool if p["paragraph_id"] not in already]
    if len(selected) < quota and remaining_pool:
      extra = random.sample(remaining_pool, min(quota - len(selected), len(remaining_pool)))
      selected.extend(extra)

    sampled.extend(selected[:quota])

  random.shuffle(sampled)
  return sampled


def split_train_test(
  sampled: list[dict],
  test_size: int = 200,
  seed: int = 42,
) -> tuple[list[dict], list[dict]]:
  """Split sampled data into test (fixed) and train (seed).

  Uses stratified split by ticker to maintain distribution in both sets.
  """
  random.seed(seed + 1) # different seed from sampling

  # Group by ticker
  by_ticker: dict[str, list[dict]] = defaultdict(list)
  for p in sampled:
    by_ticker[p["ticker"]].append(p)

  test_ratio = test_size / len(sampled)
  test_set = []
  train_set = []

  for ticker, paras in by_ticker.items():
    random.shuffle(paras)
    n_test = max(1, round(len(paras) * test_ratio))
    # Ensure at least 1 in train if we have 2+ paragraphs
    if len(paras) >= 2:
      n_test = min(n_test, len(paras) - 1)
    test_set.extend(paras[:n_test])
    train_set.extend(paras[n_test:])

  # Adjust if we over/under-sampled test
  random.shuffle(test_set)
  random.shuffle(train_set)

  if len(test_set) > test_size:
    overflow = test_set[test_size:]
    test_set = test_set[:test_size]
    train_set.extend(overflow)
  elif len(test_set) < test_size and len(train_set) > 0:
    deficit = test_size - len(test_set)
    move = train_set[:deficit]
    test_set.extend(move)
    train_set = train_set[deficit:]

  return train_set, test_set


# ============================================================
# EXPORT
# ============================================================

def export_label_studio(records: list[dict], output_path: Path, split_name: str = ""):
  """Export in Label Studio JSON format with FinBERT pre-annotations."""
  ls_data = []
  for i, r in enumerate(records):
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
        "split": split_name,
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

  print(f" Exported {len(ls_data)} items -> {output_path}")


def export_json(records: list[dict], output_path: Path):
  """Export raw records as JSON."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
  print(f" Exported {len(records)} records -> {output_path}")


# ============================================================
# MAIN
# ============================================================

def run(
  input_path: Path = INPUT_PATH,
  output_dir: Path = OUTPUT_DIR,
  total: int = 500,
  test_size: int = 200,
  seed: int = 42,
  dry_run: bool = False,
):
  """Run sampling, splitting, and export."""
  print(f"{'='*60}")
  print(f" Step 2.3: Stratified Sampling & Train/Test Split")
  print(f"{'='*60}")
  print(f" Input:   {input_path}")
  print(f" Total:   {total}")
  print(f" Test size: {test_size}")
  print(f" Train size: {total - test_size}")
  print(f" Seed:    {seed}")
  print(f" Pre-annot: FinBERT (auto-detected)")
  print(f"{'='*60}\n")

  # Load paragraphs
  with open(input_path) as f:
    paragraphs = json.load(f)
  print(f" Loaded {len(paragraphs):,} paragraphs from pool")

  # Stratified sample
  sampled = stratified_sample(paragraphs, total=total, seed=seed)
  print(f" Sampled {len(sampled)} paragraphs")

  # Split
  train_set, test_set = split_train_test(sampled, test_size=test_size, seed=seed)
  print(f" Split: {len(train_set)} train + {len(test_set)} test")

  # Statistics
  print(f"\n --- Sample Statistics ---")
  tickers_sampled = set(p["ticker"] for p in sampled)
  years_sampled = set(p["filing_date"][:4] for p in sampled if p.get("filing_date"))
  print(f" Unique tickers: {len(tickers_sampled)}")
  print(f" Years covered: {sorted(years_sampled)}")

  # Length distribution
  lengths = [p["char_count"] for p in sampled]
  print(f" Avg length:   {sum(lengths)/len(lengths):.0f} chars")
  print(f" Min/Max:    {min(lengths)}/{max(lengths)} chars")

  # FinBERT prediction distribution
  has_finbert = any(p.get("finbert_label") for p in sampled)
  if has_finbert:
    finbert_dist = Counter(p["finbert_label"] for p in sampled)
    esg_count = sum(v for k, v in finbert_dist.items() if k != "Non-ESG")
    print(f"\n --- FinBERT Distribution (sampled) ---")
    print(f" ESG:   {esg_count} ({100*esg_count/len(sampled):.1f}%)")
    print(f" Non-ESG: {finbert_dist.get('Non-ESG', 0)} ({100*finbert_dist.get('Non-ESG', 0)/len(sampled):.1f}%)")
    for cat in ESG_CATEGORIES:
      cnt = finbert_dist.get(cat, 0)
      print(f"  {cat:<30s} {cnt:>3}")

  # Ticker distribution in train vs test
  train_tickers = Counter(p["ticker"] for p in train_set)
  test_tickers = Counter(p["ticker"] for p in test_set)
  print(f"\n Train tickers: {len(train_tickers)} unique")
  print(f" Test tickers:  {len(test_tickers)} unique")

  if dry_run:
    print(f"\n DRY RUN — no files saved")
    return sampled, train_set, test_set

  # Export
  print(f"\n --- Exporting ---")

  # Full sampled set
  export_json(sampled, output_dir / "sampled_500.json")

  # Train/Test splits
  export_json(train_set, output_dir / "train_seed.json")
  export_json(test_set, output_dir / "test_fixed.json")

  # Label Studio JSON (combined for annotation, with split tag)
  export_label_studio(sampled, output_dir / "label_studio.json")

  # Separate Label Studio exports for convenience
  export_label_studio(train_set, output_dir / "label_studio_train.json", "train")
  export_label_studio(test_set, output_dir / "label_studio_test.json", "test")

  # Save label config for reference
  config_path = output_dir / "label_config.xml"
  with open(config_path, "w") as f:
    f.write(LABEL_CONFIG)
  print(f" Label config -> {config_path}")

  # Summary
  print(f"\n{'='*60}")
  print(f" Sampling Complete")
  print(f"{'='*60}")
  print(f" sampled_500.json    -> {len(sampled)} total samples")
  print(f" train_seed.json    -> {len(train_set)} training seed")
  print(f" test_fixed.json    -> {len(test_set)} fixed test set")
  print(f" label_studio.json  -> for Label Studio import (all)")
  print(f"{'='*60}")

  return sampled, train_set, test_set


def main():
  parser = argparse.ArgumentParser(
    description="Step 2: Stratified Sampling & Annotation Prep"
  )
  parser.add_argument(
    "--input", type=str, default=str(INPUT_PATH),
    help="Input classified JSON (default: data/processed/step2_classification/classified.json)",
  )
  parser.add_argument(
    "--output-dir", type=str, default=str(OUTPUT_DIR),
    help="Output directory (default: data/processed/step2_classification)",
  )
  parser.add_argument(
    "--total", type=int, default=500,
    help="Total samples to draw (default: 500)",
  )
  parser.add_argument(
    "--test-size", type=int, default=200,
    help="Fixed test set size (default: 200)",
  )
  parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed (default: 42)",
  )
  parser.add_argument(
    "--dry-run", action="store_true",
    help="Preview without saving",
  )
  args = parser.parse_args()

  run(
    input_path=Path(args.input),
    output_dir=Path(args.output_dir),
    total=args.total,
    test_size=args.test_size,
    seed=args.seed,
    dry_run=args.dry_run,
  )


if __name__ == "__main__":
  main()
