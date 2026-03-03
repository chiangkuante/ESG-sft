"""
將 project-9 重新標註的資料集分割成訓練集（80%）和測試集（20%）
採用 stratified split 保持各類別比例
輸出至 data/fine-tuning-data/
"""

import json
import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

random.seed(42)

INPUT_CSV = "data/origin_data/10k_1A/project-9-at-2026-03-02-01-50-d0b056e4.csv"
OUTPUT_DIR = Path("data/fine-tuning-data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 讀取資料
with open(INPUT_CSV, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"總筆數: {len(rows)}")

# 統一格式為後續使用的欄位
def row_to_record(r):
    return {
        "paragraph_id": r["paragraph_id"],
        "ticker": r["ticker"],
        "year": r["year"],
        "text": r["text"],
        "label": r["label"],
        "finbert_label": r["finbert_category"],
        "finbert_confidence": float(r["finbert_confidence"]) if r["finbert_confidence"] else 0.0,
    }

records = [row_to_record(r) for r in rows]

# Stratified split：每個類別 80% train, 20% test
by_label = defaultdict(list)
for rec in records:
    by_label[rec["label"]].append(rec)

train_records = []
test_records = []

print("\n各類別分割:")
print(f"{'類別':<30} {'總計':>5} {'Train':>6} {'Test':>5}")
print("-" * 48)

for label, items in sorted(by_label.items()):
    random.shuffle(items)
    n_test = max(1, round(len(items) * 0.2))
    n_train = len(items) - n_test
    train_records.extend(items[:n_train])
    test_records.extend(items[n_train:])
    print(f"  {label:<28} {len(items):>5} {n_train:>6} {n_test:>5}")

# 打亂順序
random.shuffle(train_records)
random.shuffle(test_records)

print(f"\n{'Train':>6}: {len(train_records)} 筆")
print(f"{'Test':>6}: {len(test_records)} 筆")

# 儲存 JSON
def save_json(records, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path} ({len(records)} records)")

def save_csv(records, path):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved: {path} ({len(records)} records)")

save_json(train_records, OUTPUT_DIR / "train.json")
save_csv(train_records, OUTPUT_DIR / "train.csv")
save_json(test_records, OUTPUT_DIR / "test.json")
save_csv(test_records, OUTPUT_DIR / "test.csv")
