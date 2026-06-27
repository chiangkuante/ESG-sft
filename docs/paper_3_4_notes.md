# 3.4 模型選擇與配置 — 撰寫備註與規格

本檔依據 `config/config.yaml`、`src/step5_reasoning/`、`src/step6_cv/` 之實作，整理 3.4 章各節在論文中需備註之細節、規格與超參數。

---

## 3.4.1 教師模型

### GPT-5.4 mini 之選用理由
- **模型**：`gpt-5.4-mini-2026-03-17`（OpenAI Responses API）
- **配置位置**：`config/config.yaml::step5_reasoning.generation.model`
- 選用理由（建議寫入論文）：
  1. **推理能力**：屬支援 reasoning 控制（`reasoning_effort` 參數）之 mini 級模型，相對於非 reasoning 的同價位模型在多步分類解釋任務上具備穩定的逐步推導能力。
  2. **API 可及性**：OpenAI Responses API 提供穩定的 JSON 結構化輸出（搭配本研究之 batch JSON array 格式），且具備 mature 的 SDK 支援。
  3. **成本平衡**：相較 `gpt-5.4`（full）級別，mini 變體於 ~4,000 筆訓練資料 × 3 fold 之 reasoning 生成下成本可控；高 rate limit（10K rpm / 10M tpm / 1B tpd，見 `step6_cv.api_llm.models.gpt_5_4_mini_2026_03_17.rate_limits`）使批次生成可在合理時間完成。

### 教師模型之推論參數設定
（`src/step5_reasoning/generate_reasoning.py::call_openai`、`config/config.yaml::step5_reasoning.generation`）

| 參數 | 值 | 備註 |
|---|---|---|
| `model` | `gpt-5.4-mini-2026-03-17` | OpenAI Responses API |
| `reasoning_effort` | `low` | OpenAI reasoning 模型不接受 temperature，以 effort 控制 |
| `max_output_tokens` | 12000 | 單次回應 token 上限 |
| `batch_size` | 10 | 每次 API 呼叫內含 10 筆段落 |
| `max_retries` | 3 | 整批失敗時最多重試 3 次 |
| `sleep_seconds` | 0.05 | 批次間基礎延遲 |
| `retry_backoff_seconds` | 0.5 | 重試線性退避係數 |

**備援 provider 之溫度設定**（僅當切換 provider 時生效）：
- Anthropic（如 `claude-sonnet-4-6`）：`temperature=0.2`
- Gemini（如 `gemini-3-flash-preview`）：`temperature=0.2`

---

## 3.4.2 學生模型

### 四個開源 SLM 之選擇邏輯
（建議於論文中陳述）
1. **參數規模一致性**：四者皆落於 3B–4B 區間，於單張 24G 級 GPU（如 RTX 4090）可同時容納 base model 與 LoRA 適配器（r=32）並完成 5 epoch 微調，公平比較。
2. **架構多樣性**：涵蓋 Meta（Llama 3.2）、Google（Gemma 4）、Alibaba（Qwen 3.5）、Mistral AI（Ministral 3）四個主要家族，避免單一家族架構之偏差。
3. **Instruct 版本**：四者皆為已 instruction-tuned 之版本，與本研究 SFT 三段式對話格式（system / user / assistant）相容。
4. **Unsloth 生態相容**：四者皆有 Unsloth 量化版本，可使用 `FastLanguageModel` / `FastModel` / `FastVisionModel` 統一訓練介面，並啟用 4-bit 量化與 gradient checkpointing。
5. **開源授權**：四者皆採可商業使用之 open weight 授權，符合本研究本地部署、可重現之需求。

### 各模型版本與配置
（`config/config.yaml::step6_cv.finetune.model_registry`、`src/step6_cv/run_cv_finetune.py`）

| 學生模型 | HF / Unsloth 模型名稱 | Unsloth API | 備註 |
|---|---|---|---|
| Llama 3.2 3B-Instruct (Grattafiori et al., 2024) | `unsloth/Llama-3.2-3B-Instruct` | `FastLanguageModel` | `target_modules` 顯式指定；chat template `llama-3.1`；`DataCollatorForSeq2Seq`；`standardize_sharegpt` 預處理 |
| Gemma 4 4B-E4B-it (Gemma Team, 2024) | `unsloth/gemma-4-E4B-it` | `FastModel` | chat template `gemma-4`；移除 BOS 前綴；自訂 `Gemma4TextCollator` 注入 `mm_token_type_ids` |
| Qwen 3.5 4B (Yang et al., 2025) | `unsloth/Qwen3.5-4B` | `FastVisionModel` | text-only vision 格式；`UnslothVisionDataCollator` |
| Ministral 3-3B-Instruct-2512 (Liu et al., 2026) | `unsloth/Ministral-3-3B-Instruct-2512` | `FastVisionModel` | text-only vision 格式；`UnslothVisionDataCollator` |

### 學生模型於開源授權與本地部署上之共同特性
- 均提供 4-bit / 8-bit 量化權重，於 Unsloth 介面下可一鍵載入並啟用 LoRA。
- 均支援 `max_seq_length=2048` 設定。
- 均以 LoRA（`r=32, alpha=32, dropout=0, bias="none"`）作統一 PEFT 配置，避免不同模型超參數帶來之額外變因。

### 統一訓練超參數
（`config/config.yaml::step6_cv.finetune.training`）

| 類別 | 參數 | 值 |
|---|---|---|
| LoRA | r / alpha / dropout / bias | 32 / 32 / 0 / none |
| 訓練 | learning_rate | 2e-4 |
| 訓練 | per_device_train_batch_size | 2 |
| 訓練 | gradient_accumulation_steps | 4 |
| 訓練 | num_train_epochs | 5（X/Y/Z plot 模式：[7, 9, 11, 13, 15]，當前 `xyz_plot.epochs=[15]`） |
| 訓練 | warmup_steps | 5 |
| 優化器 | optim | adamw_8bit |
| 優化器 | weight_decay | 0.001 |
| 排程 | lr_scheduler_type | linear |
| 其他 | random_state / seed | 3407 |
| 其他 | max_seq_length | 2048 |

### 學生模型推論參數
（`src/step6_cv/run_cv_finetune.py::run_fold_inference`）
- `do_sample=False`（greedy decoding）
- `temperature=None`，`top_p=None`
- `max_new_tokens`：reasoning + label 模式為 512、`label_only` 模式為 16
- 自訂 `StopOnLabel` 停止條件：偵測到 `Label: <合法類別>` 即停止生成

### 建議表 3.3：學生模型特徵彙整表
建議欄位：模型 / 開發者 / 參數量 / Context 上限 / Unsloth API / chat template / Collator / 授權

---

## 3.4.3 基準模型

### A. ESG 領域專屬編碼器：FinBERT-ESG-9-Categories
（`src/step2_classification/classify.py::MODEL_NAME = "yiyanghkust/finbert-esg-9-categories"`）
- **模型**：`yiyanghkust/finbert-esg-9-categories`（Huang, 2023）
- **架構**：BERT-base sequence classification head，9 類 ESG 分類
- **角色**：
  1. Step 2 全資料池（~160K 筆）之偽標籤來源
  2. Step 6 驗證集直接推論之外部對照基線
- **推論流程**：HuggingFace `pipeline("text-classification")`，無 LoRA、無微調，直接 zero-shot 推論
- **驗證流程**：於 step6 採 3-fold validation 直接套用其預測（已於 step2 完成預測之結果），記錄 `accuracy`、`macro_f1`（`src/step6_cv/run_cv_finbert.py`）

### B. API LLM 三組（zero-shot / few-shot 推論基線）
（`config/config.yaml::step6_cv.api_llm.models`、`src/step6_cv/run_cv_api_llm.py`）

三組模型皆採與學生模型完全相同之 prompt（system + user，含類別定義與分類規則），直接於人工標註驗證集進行分類，未做任何 fine-tuning：

| 模型 | 配置鍵 | provider | API 鍵 | 推論參數 |
|---|---|---|---|---|
| GPT-5.4 mini | `gpt_5_4_mini_2026_03_17` | OpenAI Responses | `OPENAI_API_KEY` | `max_output_tokens=64`（共用），無 temperature 參數 |
| Gemini 3 Flash Preview | `gemini_3_flash_preview` | Google `google.genai` | `GEMINI_API_KEY` | `temperature=0`、`max_output_tokens=256`、`thinking_level=minimal` |
| Claude Sonnet 4.6 | `claude_sonnet_4_6` | Anthropic Messages | `ANTHROPIC_API_KEY` | `temperature=0`、`max_tokens=64`（共用） |

**Rate limits（步驟級）**：
- GPT-5.4 mini：1K rpm / 10M tpm / 1B tpd
- Gemini 3 Flash：1K rpm / 2M tpm / 10K rpd
- Claude Sonnet 4.6：1K rpm / 450K input_tpm / 90K output_tpm

**通用設定**：`max_retries=10`、`retry_backoff_seconds=10.0`、`label_only=true`（API LLM baseline 預設僅輸出標籤）

**備註**：教師生成（3.4.1）與 API LLM baseline（3.4.3）共用同一 GPT-5.4 mini 模型 ID，但用途不同：
- 教師：給定金標生成 reasoning（answer-conditioned），`reasoning_effort=low`、`max_output_tokens=12000`
- Baseline：未給金標直接分類（zero-shot），`max_output_tokens=64`

論文中應明確區分以避免混淆。

### C. 未微調 SLM zero-shot 基準
（`config/config.yaml::step6_cv.local_slm_baseline`、`src/step6_cv/run_cv_local_slm_baseline.py`）

採與學生模型**完全相同之四個開源 SLM**，但**不載入 LoRA 適配器**、**不進行任何 SFT**，直接於人工標註驗證集進行 zero-shot 推論：

| 模型 | 配置 | 推論 API |
|---|---|---|
| `unsloth/Llama-3.2-3B-Instruct` | `models.llama.enabled=true` | `FastLanguageModel.for_inference` |
| `unsloth/gemma-4-E4B-it` | `models.gemma.enabled=true` | `FastModel.for_inference` |
| `unsloth/Qwen3.5-4B` | `models.qwen.enabled=true` | `FastVisionModel.for_inference` |
| `unsloth/Ministral-3-3B-Instruct-2512` | `models.ministral.enabled=true` | `FastVisionModel.for_inference` |

**通用設定**：
- `max_new_tokens=512`（reasoning + label）／`16`（label_only）
- `label_only=false`（預設保留 reasoning + label 輸出，確保與微調後之輸出格式可比較）
- `experiments=[base, balance, combined]`：對應三種訓練資料設定下之零訓練對照
- `variant_name=unfinetuned_local_slm`：結果獨立存放於 `results/.../unfinetuned_local_slm/`
- 推論參數同 3.4.2：`do_sample=False`，`StopOnLabel` 停止條件

### 三組基準之角色定位（建議論文寫法）
| 基準 | 對應驗證之問題 |
|---|---|
| FinBERT-ESG | 領域專屬小模型 vs. 通用大模型 + 蒸餾的能力上界對照 |
| API LLM | 大型商用 LLM 之 zero-shot 上界（學生是否能藉蒸餾追平甚至超越） |
| 未微調 SLM | 同 base model 在無蒸餾下之能力下界（蒸餾增益純效應） |

---

## 附：相關程式碼與配置定位
- 教師模型生成：`src/step5_reasoning/generate_reasoning.py`、`config/config.yaml::step5_reasoning.generation`
- 學生模型訓練：`src/step6_cv/run_cv_finetune.py`、`config/config.yaml::step6_cv.finetune`
- 學生模型推論：`src/step6_cv/run_cv_finetune.py::run_fold_inference`
- FinBERT 推論：`src/step2_classification/classify.py`、`src/step6_cv/run_cv_finbert.py`
- API LLM baseline：`src/step6_cv/run_cv_api_llm.py`、`config/config.yaml::step6_cv.api_llm`
- 未微調 SLM baseline：`src/step6_cv/run_cv_local_slm_baseline.py`、`config/config.yaml::step6_cv.local_slm_baseline`
