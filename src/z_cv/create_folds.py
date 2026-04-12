"""
Step 7.1: Create 5-Fold Cross Validation Splits

This script reads the generated SFT training data and creates consistent
5-fold splits using StratifiedKFold to maintain label distribution.
All subsequent models will use these exact same train/test indices.

Usage:
  uv run src/step7_cv/create_folds.py
"""

import json
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_SFT = Path("data/processed/step6_sft/train_sft_xml.json")
OUTPUT_FOLDS = Path("data/processed/step7_cv/cv_folds.json")

N_SPLITS = 5
RANDOM_SEED = 42

# Define labels expected in the input data
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


import re

def extract_label_from_sft(item: dict) -> str:
  """Extract the label from the assistant's reasoning text."""
  text = item["conversations"][-1]["content"]
  
  # Try to extract from XML tag <label>...</label>
  match = re.search(r"<label>\s*(.*?)\s*</label>", text, re.IGNORECASE | re.DOTALL)
  if match:
    label = match.group(1).strip()
    for cat in ESG_CATEGORIES:
      if cat.lower() == label.lower():
        return cat
        
  # Fallback to simple matching if exact phrase gets garbled
  text_lower = text.lower()
  for cat in ESG_CATEGORIES:
    if cat.lower() in text_lower[-100:]: # Check the last 100 chars
      return cat
      
  print(f"Warning: Could not parse label cleanly from: {text[-150:]}")
  return "Non-ESG" # Conservative fallback


def print_dist(indices: list, labels: list, title: str):
  """Print label distribution for a subset of data."""
  subset_labels = [labels[i] for i in indices]
  counter = Counter(subset_labels)
  
  print(f"\n {title} ({len(indices)} items)")
  print(f" {'Label':<30} {'Count':>5} {'%':>6}")
  print(f" {'-'*43}")
  for label in ESG_CATEGORIES:
    count = counter.get(label, 0)
    pct = count / len(indices) * 100
    print(f" {label:<30} {count:>5} {pct:>5.1f}%")


def main():
  print("=" * 60)
  print(" Step 7.1: Create 5-Fold CV Splits")
  print("=" * 60)
  
  # 1. Load Data
  if not INPUT_SFT.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_SFT}")
    
  with open(INPUT_SFT) as f:
    sft_data = json.load(f)
  print(f" Loaded {len(sft_data)} training items")
  
  # 2. Extract Labels for Stratification
  labels = [extract_label_from_sft(item) for item in sft_data]
  print_dist(range(len(labels)), labels, "Full Dataset")
  
  # 3. Create Folds
  skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
  
  # We just need indices, so we pass a dummy X array [0, 1, 2...]
  X_dummy = list(range(len(sft_data)))
  
  folds = []
  print(f"\n Splitting into {N_SPLITS} folds (seed={RANDOM_SEED})...")
  
  for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_dummy, labels)):
    fold_data = {
      "fold": fold_idx,
      "train_indices": train_idx.tolist(),
      "test_indices": test_idx.tolist()
    }
    folds.append(fold_data)
    
    print(f"\n --- Fold {fold_idx} ---")
    print(f" Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Only print distribution for the first fold to avoid spam
    if fold_idx == 0:
      print_dist(train_idx, labels, f"Fold 0 Train Mix")
      print_dist(test_idx, labels, f"Fold 0 Test Mix")

  # 4. Save
  OUTPUT_FOLDS.parent.mkdir(parents=True, exist_ok=True)
  with open(OUTPUT_FOLDS, "w", encoding="utf-8") as f:
    json.dump(folds, f, separators=(',', ':')) # Compact JSON
    
  print("\n" + "=" * 60)
  print(f" ✓ Saved CV splits to {OUTPUT_FOLDS}")
  print("=" * 60)

if __name__ == "__main__":
  main()
