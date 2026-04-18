# ESG 10-K 分類實驗：p2 目前架構

本文件只保留目前實際執行的 p2 實驗。所有流程設定以 `config/config.yaml` 為準，不在命令列傳遞實驗參數。偽標籤與 LLM 合成資料流程不列入本輪 todo。

---

## 目前目標

使用第二標註者 p2 的人工標註資料 `chiang_500.csv`，搭配額外平衡資料集 `blance_500.csv`，建立三種 3-fold CV 實驗，並比較：

- FinBERT baseline
- API LLM baseline
- 四個 local SLM 的 LoRA 微調結果
- 四個未微調 local SLM 的 zero-shot baseline

本輪三個實驗皆不使用偽標籤與合成資料：

```yaml
step4_cv:
  annotator: p2
  pseudo_labels:
    enabled: false
  synthetic_generation:
    enabled: false
```

---

## p2 三個實驗

`annotator_name` 由 `src/step4_cv/common.py::resolve_annotator_name` 依照 `step4_cv.balance` 自動決定。

| 實驗 | annotator_name | balance.enabled | balance.include_in_cv | 資料切法 | 訓練資料 |
|---|---|---:|---:|---|---|
| 實驗 1 | `p2_balance` | true | false | `chiang_500.csv` 做 3-fold | fold train + 全量 `blance_500.csv` |
| 實驗 2 | `p2` | false | false | `chiang_500.csv` 做 3-fold | fold train only |
| 實驗 3 | `p2_combined` | true | true | `chiang_500.csv` + `blance_500.csv` 合併後做 3-fold | 合併資料的 fold train |

目前 `config/config.yaml` 的 active 設定是實驗 3：

```yaml
step4_cv:
  annotator: p2
  balance:
    enabled: true
    include_in_cv: true
```

若要重建另外兩個實驗，僅調整 `step4_cv.balance`，再重跑 Step 4 與 Step 5 reasoning/SFT。

---

## 目錄與資料流

所有中間檔與結果依 `annotator_name` 分目錄保存：

```text
data/processed/step4_cv/{annotator_name}/
data/processed/step5_reasoning/{annotator_name}/
data/processed/step5_sft/{annotator_name}/
models/step5_cv/{annotator_name}/
results/step5_cv/{annotator_name}/
```

三個 p2 實驗對應：

```text
data/processed/step4_cv/p2_balance/
data/processed/step4_cv/p2/
data/processed/step4_cv/p2_combined/

data/processed/step5_sft/p2_balance/
data/processed/step5_sft/p2/
data/processed/step5_sft/p2_combined/

results/step5_cv/p2_balance/
results/step5_cv/p2/
results/step5_cv/p2_combined/
```

---

## Step 4：p2 CV Fold 與訓練池組裝

入口：`src/step4_cv/run.py`

設定來源：`config/config.yaml` 的 `step4_cv`

主要責任：

- 載入 p2 標註資料：`data/origin_data/10k_1A/chiang_500.csv`
- 視 `balance.enabled` 載入 `data/origin_data/10k_1A/blance_500.csv`
- 依 `balance.include_in_cv` 決定平衡資料是否參與 CV split
- 建立或讀取 `cv_folds.json`
- 產生每個 fold 的 `human_train.json`、`human_val.json`、`train_pool.json`
- 在目前 p2 設定下跳過 FinBERT pseudo-label pool 載入
- 在目前 p2 設定下不產生 synthetic requests

重要輸出：

```text
data/processed/step4_cv/{annotator_name}/manifest.json
data/processed/step4_cv/{annotator_name}/cv_folds.json
data/processed/step4_cv/{annotator_name}/fold_0/human_train.json
data/processed/step4_cv/{annotator_name}/fold_0/human_val.json
data/processed/step4_cv/{annotator_name}/fold_0/train_pool.json
data/processed/step4_cv/{annotator_name}/fold_0/train_pool_summary.json
```

`pseudo_labels.json` 會存在但在本輪 p2 實驗中為空；`synthetic_requests.json` 會存在但不使用。

---

## Step 5A：Reasoning 生成

入口：`src/step5_reasoning/run.py`

實際 reasoning 生成：`src/step5_reasoning/generate_reasoning.py`

設定來源：`config/config.yaml` 的 `step5_reasoning`

主要責任：

- 讀取 `data/processed/step4_cv/{annotator_name}/manifest.json`
- 對每個 fold 的 `train_pool.json` 產生 ESG 分類 reasoning
- 使用 `step5_reasoning.generation.model` 指定 API 模型
- 支援既有結果續跑，避免重複生成已完成樣本

重要輸出：

```text
data/processed/step5_reasoning/{annotator_name}/fold_0/reasoning_input.json
data/processed/step5_reasoning/{annotator_name}/fold_0/train_with_reasoning.json
data/processed/step5_reasoning/{annotator_name}/fold_0/reasoning_report.json
```

目前 config：

```yaml
step5_reasoning:
  generation:
    enabled: true
    run_reasoning_generation: true
    model: gpt-5.4-mini-2026-03-17
```

---

## Step 5B：SFT 資料組裝

入口：`src/step5_reasoning/build_sft.py`

也可由 `src/step5_reasoning/run.py` 呼叫，但需在 config 中開啟：

```yaml
step5_reasoning:
  sft:
    run_sft_build: true
```

主要責任：

- 讀取 `train_with_reasoning.json`
- 轉成 instruction-following SFT 格式
- assistant 回覆格式為 `<reasoning>...</reasoning>` 與 `<label>...</label>`
- 將 Step 4 的 `human_val.json` 複製為 `val_eval.json`
- 產生 `manifest.json`，供後續 fine-tune、inference 與 local baseline 使用

重要輸出：

```text
data/processed/step5_sft/{annotator_name}/manifest.json
data/processed/step5_sft/{annotator_name}/fold_0/train_sft_text.json
data/processed/step5_sft/{annotator_name}/fold_0/val_eval.json
data/processed/step5_sft/{annotator_name}/fold_0/excluded_items.json
```

---

## Step 5C：Baseline 與模型評估

### FinBERT Baseline

入口：`src/step5_cv/run_cv_finbert.py`

設定來源：

```yaml
step5_cv:
  finbert:
    step4_output_dir: data/processed/step4_cv
    results_dir: results/step5_cv/finbert
```

輸出：

```text
results/step5_cv/finbert/{annotator_name}/fold_0_results.json
results/step5_cv/finbert/{annotator_name}/fold_0_metrics.json
results/step5_cv/finbert/{annotator_name}/overall_folds.json
```

### API LLM Baseline

入口：`src/step5_cv/run_cv_api_llm.py`

設定來源：`step5_cv.api_llm`

目前啟用模型：

- `gemini_3_flash_preview`
- `gpt_5_4_mini_2026_03_17`
- `claude_sonnet_4_6`

輸出：

```text
results/step5_cv/{annotator_name}/api_llm/{model_key}/fold_0_results.json
results/step5_cv/{annotator_name}/api_llm/{model_key}/fold_0_metrics.json
results/step5_cv/{annotator_name}/api_llm/{model_key}/overall_summary.json
results/step5_cv/{annotator_name}/api_llm/{model_key}/overall_summary.md
```

### Local SLM LoRA 微調

入口：`src/step5_cv/run_cv_finetune.py`

設定來源：`step5_cv.finetune`

目前模型 registry：

```yaml
step5_cv:
  finetune:
    model_registry:
      gemma: unsloth/gemma-4-E4B-it
      llama: unsloth/Llama-3.2-3B-Instruct
      qwen: unsloth/Qwen3.5-4B
      ministral: unsloth/Ministral-3-3B-Instruct-2512
```

目前共同超參數：

```yaml
training:
  lora_r: 32
  lora_alpha: 32
  lora_dropout: 0
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  num_train_epochs: 15
  learning_rate: 2e-4
  random_state: 3407
  max_seq_length: 2048
```

輸出：

```text
models/step5_cv/{annotator_name}/{model_type}/fold_0/adapter/
models/step5_cv/{annotator_name}/{model_type}/fold_0/checkpoints/
results/step5_cv/{annotator_name}/{model_type}/fold_0_results.json
results/step5_cv/{annotator_name}/{model_type}/fold_0_metrics.json
results/step5_cv/{annotator_name}/{model_type}/overall_summary.json
results/step5_cv/{annotator_name}/{model_type}/overall_summary.md
```

### Local SLM 未微調 Baseline

入口：`src/step5_cv/run_cv_local_slm_baseline.py`

設定來源：`step5_cv.local_slm_baseline`

目前設定會一次讀取三個 p2 實驗的 SFT manifest：

```yaml
step5_cv:
  local_slm_baseline:
    enabled: true
    experiments:
      - p2_balance
      - p2
      - p2_combined
    label_only: false
    max_new_tokens: 512
```

此流程不讀取 LoRA adapter，不做訓練，只載入 base model 對 `val_eval.json` 推論。

輸出：

```text
results/step5_cv/{annotator_name}/local_slm_baseline/{model_type}/fold_0_results.json
results/step5_cv/{annotator_name}/local_slm_baseline/{model_type}/fold_0_metrics.json
results/step5_cv/{annotator_name}/local_slm_baseline/{model_type}/overall_summary.json
results/step5_cv/{annotator_name}/local_slm_baseline/{model_type}/overall_summary.md
```

---

## Step 5D：統一評估彙整

入口：`src/step5_cv/evaluate_cv_metrics.py`

設定來源：

```yaml
step5_cv:
  evaluation:
    results_root: results/step5_cv
```

主要責任：

- 讀取目前 active `annotator_name` 底下的 `fold_*_results.json`
- 重新計算 overall metrics
- 輸出 per-class 與 pillar-level metrics

輸出：

```text
overall_summary.json
overall_folds.csv
per_class_metrics.csv
pillar_metrics.csv
```

注意：此彙整器依目前 `step4_cv` 的 active `annotator_name` 掃描單一實驗。如果要彙整 `p2_balance`、`p2`、`p2_combined`，需分別切換 `step4_cv.balance` 後執行。

---

## 當前 Todo

### 已完成或已有產物

- `data/processed/step4_cv/p2/manifest.json`
- `data/processed/step4_cv/p2_balance/manifest.json`
- `data/processed/step4_cv/p2_combined/manifest.json`
- `data/processed/step5_sft/p2/manifest.json`
- `data/processed/step5_sft/p2_balance/manifest.json`
- `data/processed/step5_sft/p2_combined/manifest.json`
- `src/step5_cv/run_cv_local_slm_baseline.py`
- `step5_cv.local_slm_baseline` config

### 需要確認或執行

- 對三個 p2 實驗確認 `train_with_reasoning.json` 是否完整，缺漏時重跑 Step 5A。
- 若 reasoning 更新，將 `step5_reasoning.sft.run_sft_build` 設為 `true` 後重建 SFT manifest。
- 對四個 local SLM 分別完成 LoRA fine-tune 與 inference。
- 對三個 API LLM baseline 補齊 `p2_balance`、`p2`、`p2_combined` 結果。
- 執行未微調 local SLM baseline，補齊三個 p2 實驗與四個 local model 的 36 組 fold 推論。
- 最後分別對 `p2_balance`、`p2`、`p2_combined` 執行 metric 彙整。

---

## 評估指標

每個模型與每個 p2 實驗都輸出 3-fold mean/std：

- Accuracy
- Macro F1
- Weighted F1
- Cohen's Kappa
- Per-class Precision / Recall / F1
- Pillar-level F1：Environmental、Social、Governance、Non-ESG

九個分類標籤固定為：

- Climate Change
- Natural Capital
- Pollution & Waste
- Human Capital
- Product Liability
- Community Relations
- Corporate Governance
- Business Ethics & Values
- Non-ESG

---

## p2 實驗總量

| 類別 | 模型數 | 實驗數 | Fold | 說明 |
|---|---:|---:|---:|---|
| FinBERT | 1 | 3 | 3 | 9 組 fold 推論 |
| API LLM | 3 | 3 | 3 | 27 組 fold API 推論 |
| Local SLM LoRA | 4 | 3 | 3 | 36 組 fine-tune + inference |
| Local SLM 未微調 | 4 | 3 | 3 | 36 組 zero-shot local 推論 |

---

## 參考架構重點

- Step 4 只負責 fold 與 train pool，不在本輪 p2 實驗中做 pseudo-label 或 synthetic generation。
- Step 5 reasoning 與 SFT 依 `annotator_name` 讀寫，不跨實驗共用訓練資料。
- LoRA fine-tune 結果依 `{annotator_name}/{model_type}` 分開保存。
- 未微調 local baseline 依 `step5_cv.local_slm_baseline.experiments` 一次掃描三個 p2 SFT manifest。
- 所有模型推論使用 `src/step5_cv/common.py` 中相同的 label list、prompt builder、label parser 與 metric function。
