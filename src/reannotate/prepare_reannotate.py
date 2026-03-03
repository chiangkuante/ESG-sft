"""
重新標註主資料集（不含補充資料集）的 Label Studio 匯入腳本

輸入：data/origin_data/10k_1A/train_annotated.json（995 筆）
輸出：data/origin_data/10k_1A/label_studio_reannotate.json

特色：
- 排除補充資料集（project-8: ALGN_2021_0004 等）
- 把當初的 human label 也放進 predictions，讓標注者可以看到舊標籤
- 同時保留 finbert 預測作為參考
"""

import json
import csv
from pathlib import Path

# 載入主標註資料（全部合併的 995 筆）
with open("data/origin_data/10k_1A/train_annotated.json") as f:
    all_annotated = {d["paragraph_id"]: d for d in json.load(f)}

print(f"train_annotated 總計: {len(all_annotated)} 筆")

# 主資料集的定義：project-6 匯入的 492 筆 paragraph_id
with open("data/origin_data/10k_1A/label_studio_import.json") as f:
    ls_import = json.load(f)
main_ids = set(d["data"]["paragraph_id"] for d in ls_import)
print(f"主資料集 (label_studio_import): {len(main_ids)} 筆")

# 只取主資料集中有完整標注的項目
main_only = [all_annotated[pid] for pid in main_ids if pid in all_annotated]
print(f"主資料集且有標注: {len(main_only)} 筆")

# 類別對應（用於構建 Label Studio 的 prediction 結構）
LABEL_MAP = {
    "Climate Change": "Climate Change",
    "Natural Capital": "Natural Capital",
    "Pollution & Waste": "Pollution & Waste",
    "Human Capital": "Human Capital",
    "Product Liability": "Product Liability",
    "Community Relations": "Community Relations",
    "Corporate Governance": "Corporate Governance",
    "Business Ethics & Values": "Business Ethics & Values",
    "Non-ESG": "Non-ESG",
}

# 構建 Label Studio 匯入格式
ls_items = []
for d in main_only:
    human_label = d.get("label", "")
    finbert_label = d.get("finbert_label", "")
    
    # 建立 prediction list（優先放人工標籤，次為 FinBERT）
    predictions = []
    
    # 人工標籤（舊標籤，分數設高讓它排第一）
    if human_label in LABEL_MAP:
        predictions.append({
            "model_version": "human_annotated_v1",
            "score": 0.99,
            "result": [
                {
                    "from_name": "label",
                    "to_name": "text",
                    "type": "choices",
                    "value": {"choices": [human_label]},
                }
            ],
        })
    
    # FinBERT 預測（次要參考）
    if finbert_label and finbert_label in LABEL_MAP and finbert_label != human_label:
        predictions.append({
            "model_version": "finbert-esg-9-categories",
            "score": float(d.get("finbert_confidence", 0.5)),
            "result": [
                {
                    "from_name": "label",
                    "to_name": "text",
                    "type": "choices",
                    "value": {"choices": [finbert_label]},
                }
            ],
        })
    
    ls_items.append({
        "data": {
            "text": d["text"],
            "paragraph_id": d["paragraph_id"],
            "ticker": d.get("ticker", ""),
            "year": d.get("year", ""),
            "char_count": len(d["text"]),
            "finbert_category": finbert_label,
            "finbert_confidence": d.get("finbert_confidence", 0.0),
            "previous_label": human_label,   # 顯示舊標籤供參考
        },
        "predictions": predictions,
    })

# 儲存
output_path = "data/origin_data/10k_1A/label_studio_reannotate.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(ls_items, f, ensure_ascii=False, indent=2)

print(f"\n輸出至: {output_path}")
print(f"總共: {len(ls_items)} 筆待重新標註")

# 統計舊標籤分布
from collections import Counter
old_labels = Counter(d["data"]["previous_label"] for d in ls_items)
print("\n舊標籤分布（供參考）：")
for label, cnt in sorted(old_labels.items(), key=lambda x: -x[1]):
    print(f"  {cnt:4d}  {label}")
