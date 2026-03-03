# Data Directory

## `fine-tuning-data/` -- 微調用的訓練/測試資料

| 檔案 | 大小 | 說明 |
|------|------|------|
| `train.json` / `train.csv` | 1.0 MB | 訓練集（896 筆），來自 project-9（490 筆重新標註）+ project-8（504 筆補充標註），80/20 分割 |
| `test.json` / `test.csv` | 115 KB | 測試集（98 筆），從 project-9 的 20% 分層抽樣 |
| `train_with_reasoning.json` | 1.7 MB | 訓練集 + GPT-5-mini 生成的 chain-of-thought 推理（896 筆） |
| `train_sft.json` | 7.3 MB | ShareGPT 格式的 SFT 訓練資料，含 system/user/assistant 對話，用於 Unsloth 微調 Gemma 3 |

## `origin_data/` -- 原始資料與標註過程產物

### `origin_data/10k_1A/` -- 10-K Item 1A 段落處理

> [!WARNING]
> 以下標記 **[LARGE]** 的檔案超過 100MB，不適合直接上傳 git，建議加入 `.gitignore`。
> 這些檔案可以由 `src/preprocess.py` 和 `src/finbert/classify.py` 從 `10k_raw/` 重新生成。

**大型中間產物（由 pipeline 生成，可重建）：**

| 檔案 | 大小 | 說明 |
|------|------|------|
| `paragraphs.csv` / `.json` | 727 / 811 MB | **[LARGE]** 從所有 10-K Item 1A 文件中提取的段落。由 `src/preprocess.py` 讀取 `10k_raw/` 原始 10-K 文件，經切割、清洗後產生。包含 paragraph_id, ticker, year, text 等欄位 |
| `classified.csv` / `.json` | 745 / 881 MB | **[LARGE]** FinBERT-ESG-9 對所有段落的分類結果。由 `src/finbert/classify.py` 對 `paragraphs.csv` 逐筆推論產生。包含 paragraph_id, text, finbert_label, finbert_confidence 等欄位 |
| `classified_checkpoint.csv` | 743 MB | **[LARGE]** FinBERT 分類的中間 checkpoint，防止中斷遺失進度。分類完成後可刪除 |

**小型資料檔（應上傳 git）：**

| 檔案 | 大小 | 說明 |
|------|------|------|
| `filtered_paragraphs.csv` | 6.8 MB | 過濾後的段落（移除過短/過長段落），用於抽樣標註 |
| `sampled_paragraphs.csv` | 491 KB | 從過濾段落中隨機抽樣，作為人工標註的候選集 |
| `short_paragraphs_sample.csv` | 6.8 MB | 短段落樣本（與 filtered_paragraphs.csv 相同內容） |
| `classification_stats.json` | 4 KB | FinBERT 分類統計（各類別數量分布） |
| `stats.json` | 1 KB | 段落統計（數量、長度分布等） |
| `token_stats.json` | 1 KB | Token 長度統計 |
| `label_studio_import.json` | 789 KB | 第一批匯入 Label Studio 的標註資料（492 筆） |
| `label_studio_reannotate.json` | 818 KB | 用於重新標註的 Label Studio 匯入格式 |
| `label_studio_supplement.json` | 801 KB | 補充標註用的 Label Studio 匯入格式（低頻類別） |
| `project-6-*.json` / `.csv` | 535 KB | Label Studio 第一次匯出的標註結果 |
| `project-8-*.csv` | 550 KB | Label Studio 補充標註匯出（504 筆低頻類別） |
| `project-9-*.csv` | 543 KB | Label Studio 重新標註匯出（490 筆，最終版本） |
| `train_annotated.csv` / `.json` | 1.0 MB | 合併後的標註資料集（995 筆，包含 project-6 + project-8） |
| `pipeline.log` | 938 KB | 資料處理 pipeline 日誌 |
| `classify_run.log` | 40 KB | FinBERT 分類執行日誌 |

### `origin_data/` 根目錄 -- 舊版訓練/測試分割

| 檔案 | 大小 | 說明 |
|------|------|------|
| `train.json` / `train.csv` | 917 KB | 舊版訓練集（重新標註前） |
| `test.json` / `test.csv` | 108 KB | 舊版測試集（重新標註前） |
| `val.json` / `val.csv` | 122 KB | 舊版驗證集（重新標註前） |

## 不追蹤的目錄

| 目錄 | 說明 |
|------|------|
| `origin_data/10k_raw/` | 原始 10-K 文件（已在 .gitignore） |
| `reason_data/` | 空目錄（已在 .gitignore） |
