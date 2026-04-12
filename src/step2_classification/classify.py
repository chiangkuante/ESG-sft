"""
Step 2.1: FinBERT ESG 9-Category Full Prediction ()

Runs FinBERT-ESG prediction on the full paragraph pool.
Adds finbert_label and finbert_confidence to each record.

Usage:
  uv run src/step2_classification/classify.py               # Default settings
  uv run src/step2_classification/classify.py --batch-size 64       # Larger batches
  uv run src/step2_classification/classify.py --resume           # Resume from checkpoint
  uv run src/step2_classification/classify.py --input data/processed/step1_parsing/paragraphs.json
"""

import argparse
import json
import time
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# ============================================================
# CONSTANTS
# ============================================================

MODEL_NAME = "yiyanghkust/finbert-esg-9-categories"
INPUT_PATH = Path("data/processed/step1_parsing/paragraphs.json")
OUTPUT_DIR = Path("data/processed/step2_classification")

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


# ============================================================
# MODEL
# ============================================================

def load_model(device: int = 0):
  """Load FinBERT-ESG pipeline."""
  print(" Loading model:", MODEL_NAME)
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

  device_arg = device if torch.cuda.is_available() else -1
  if device_arg >= 0:
    print(f" Using GPU {device_arg}: {torch.cuda.get_device_name(device_arg)}")
  else:
    print(" Using CPU")

  pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device_arg,
    truncation=True,
    max_length=512,
    batch_size=1, # will be overridden per call
  )
  return pipe, tokenizer


# ============================================================
# CLASSIFICATION
# ============================================================

def run_classification(
  input_path: Path = INPUT_PATH,
  output_dir: Path = OUTPUT_DIR,
  batch_size: int = 32,
  checkpoint_every: int = 10000,
  device: int = 0,
  resume: bool = False,
):
  """Run FinBERT on all paragraphs."""

  print(f"{'='*60}")
  print(f" Step 2.1: FinBERT Full Prediction ()")
  print(f"{'='*60}")
  print(f" Input:   {input_path}")
  print(f" Output:   {output_dir}")
  print(f" Batch size: {batch_size}")
  print(f"{'='*60}\n")

  # Load paragraphs
  with open(input_path) as f:
    paragraphs = json.load(f)
  total = len(paragraphs)
  print(f" Loaded {total:,} paragraphs")

  # Check for checkpoint
  checkpoint_path = output_dir / "classified_checkpoint.json"
  start_idx = 0
  if resume and checkpoint_path.exists():
    with open(checkpoint_path) as f:
      checkpoint = json.load(f)
    start_idx = checkpoint.get("processed", 0)
    # Restore predictions
    for i in range(start_idx):
      paragraphs[i]["finbert_label"] = checkpoint["predictions"][i]["label"]
      paragraphs[i]["finbert_confidence"] = checkpoint["predictions"][i]["confidence"]
    print(f" Resuming from checkpoint: {start_idx:,} already processed")

  # Load model
  pipe, tokenizer = load_model(device)

  # Classify
  t0 = time.time()
  for start in range(start_idx, total, batch_size):
    end = min(start + batch_size, total)

    # Use combined_text (heading + paragraph) for classification
    batch_texts = [p["combined_text"] for p in paragraphs[start:end]]

    results = pipe(batch_texts, batch_size=batch_size)

    for i, result in enumerate(results):
      idx = start + i
      paragraphs[idx]["finbert_label"] = result["label"]
      paragraphs[idx]["finbert_confidence"] = round(result["score"], 4)

    # Progress
    processed = end
    if processed % 5000 < batch_size or processed == total:
      elapsed = time.time() - t0
      rate = processed / elapsed if elapsed > 0 else 0
      eta = (total - processed) / rate if rate > 0 else 0
      print(f" [{processed:>6,}/{total:,}] {100*processed/total:.1f}% "
         f"({rate:.0f}/s, ETA {eta/60:.1f}m)")

    # Checkpoint
    if processed % checkpoint_every < batch_size and processed > 0:
      _save_checkpoint(paragraphs, processed, checkpoint_path)

  elapsed = time.time() - t0
  print(f"\n Classification complete in {elapsed/60:.1f} minutes")

  # Save results
  output_dir.mkdir(parents=True, exist_ok=True)

  output_path = output_dir / "classified.json"
  with open(output_path, "w") as f:
    json.dump(paragraphs, f, ensure_ascii=False, indent=2)
  print(f" Saved -> {output_path}")

  # Statistics
  _print_stats(paragraphs, output_dir)

  # Clean up checkpoint
  if checkpoint_path.exists():
    checkpoint_path.unlink()

  return paragraphs


def _save_checkpoint(paragraphs: list[dict], processed: int, path: Path):
  """Save checkpoint for resume."""
  checkpoint = {
    "processed": processed,
    "predictions": [
      {"label": p.get("finbert_label", ""), "confidence": p.get("finbert_confidence", 0)}
      for p in paragraphs[:processed]
    ],
  }
  with open(path, "w") as f:
    json.dump(checkpoint, f)
  print(f" [checkpoint] Saved at {processed:,}")


def _print_stats(paragraphs: list[dict], output_dir: Path):
  """Print and save classification statistics."""
  total = len(paragraphs)
  cat_counts = Counter(p["finbert_label"] for p in paragraphs)

  esg_count = sum(v for k, v in cat_counts.items() if k != "Non-ESG")
  non_esg_count = cat_counts.get("Non-ESG", 0)

  print(f"\n {'='*60}")
  print(f" Classification Statistics")
  print(f" {'='*60}")
  print(f" Total:  {total:,}")
  print(f" ESG:   {esg_count:,} ({100*esg_count/total:.1f}%)")
  print(f" Non-ESG: {non_esg_count:,} ({100*non_esg_count/total:.1f}%)")
  print()
  for cat in ESG_CATEGORIES:
    cnt = cat_counts.get(cat, 0)
    pct = 100 * cnt / total
    bar = "█" * int(pct)
    print(f" {cat:<30s} {cnt:>6,} ({pct:>5.1f}%) {bar}")

  avg_conf = sum(p["finbert_confidence"] for p in paragraphs) / total
  print(f"\n Avg confidence: {avg_conf:.4f}")

  # Save
  stats = {
    "total": total,
    "category_distribution": {
      cat: {"count": cat_counts.get(cat, 0), "pct": round(100 * cat_counts.get(cat, 0) / total, 2)}
      for cat in ESG_CATEGORIES
    },
    "esg_count": esg_count,
    "non_esg_count": non_esg_count,
    "avg_confidence": round(avg_conf, 4),
  }
  stats_path = output_dir / "classification_stats.json"
  with open(stats_path, "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
  print(f" Stats -> {stats_path}")


# ============================================================
# MAIN
# ============================================================

def main():
  parser = argparse.ArgumentParser(description="Step 2.1: FinBERT Full Prediction ()")
  parser.add_argument("--input", type=str, default=str(INPUT_PATH))
  parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--checkpoint-every", type=int, default=10000)
  parser.add_argument("--device", type=int, default=0)
  parser.add_argument("--resume", action="store_true")
  args = parser.parse_args()

  run_classification(
    input_path=Path(args.input),
    output_dir=Path(args.output_dir),
    batch_size=args.batch_size,
    checkpoint_every=args.checkpoint_every,
    device=args.device,
    resume=args.resume,
  )


if __name__ == "__main__":
  main()
