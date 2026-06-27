# 3.5 訓練設定 — 撰寫備註與規格

本檔依據 `src/step6_cv/run_cv_finetune.py`、`config/config.yaml` 之實作，整理 3.5 章各節在論文中需備註之細節、規格與超參數。

---

## 3.5.1 LoRA 微調配置

### 超參數設定
（`config/config.yaml::step6_cv.finetune.training`）

| 參數 | 值 | 適用模型 |
|---|---|---|
| LoRA rank `r` | 32 | 全部 4 個學生模型 |
| `lora_alpha` | 32 | 全部（α/r = 1） |
| `lora_dropout` | 0 | 全部（依 Unsloth 官方建議，0 為最快實作） |
| `bias` | `"none"` | 全部（不訓練 bias） |
| `use_rslora` | `False` | 全部（Llama / Qwen / Ministral 顯式設為 False；Gemma 走 `FastModel.get_peft_model` 預設） |
| `loftq_config` | `None` | 全部（不使用 LoftQ 量化初始化） |
| `random_state` | 3407 | 全部 |

### 目標權重矩陣
所有四個學生模型之 LoRA 適配同時注入於 **attention 投影層** 與 **MLP 投影層**，共 7 個矩陣（per Transformer block）：

| 區塊 | 矩陣 |
|---|---|
| Attention | `q_proj`、`k_proj`、`v_proj`、`o_proj` |
| MLP（FFN/SwiGLU） | `gate_proj`、`up_proj`、`down_proj` |

**設定方式（依 Unsloth API 而異）**：
- **Llama 3.2 3B**（`FastLanguageModel.get_peft_model`）：以顯式 `target_modules` 參數逐一列出上述 7 個 projection
- **Gemma 4 4B-E4B**（`FastModel.get_peft_model`）：以 flag 控制
  - `finetune_vision_layers=False`（不訓練視覺層）
  - `finetune_language_layers=True`
  - `finetune_attention_modules=True`
  - `finetune_mlp_modules=True`
- **Qwen 3.5 4B / Ministral 3-3B**（`FastVisionModel.get_peft_model`）：與 Gemma 同樣以 flag 控制（vision off、language on、attention on、MLP on）

論文撰寫時可統一描述為：「LoRA 適配層覆蓋語言模型部分之所有 attention 與 MLP 投影矩陣（q/k/v/o 與 gate/up/down 共 7 個），不訓練視覺層、嵌入層、layer-norm 與 bias。」

### 可訓練參數量統計
程式碼未持久化此統計值。可由 LoRA 結構估算：

\[
N_{\text{LoRA}} \approx \sum_{l=1}^{L} \sum_{m \in \mathcal{M}} r \cdot (d_{m}^{\text{in}} + d_{m}^{\text{out}})
\]

其中 \(L\) 為 Transformer block 層數，\(\mathcal{M}\) 為 7 個 projection，\(r=32\)。

各模型之**總參數量**與**LoRA 可訓練參數量比例**約略落於 **0.5%–1.5%** 區間（典型 LoRA r=32, 3B–4B 模型範圍）。

> **論文撰寫備註**：實際數值請以 Unsloth 載入模型時於 stdout 輸出之 `trainable params: X || all params: Y || trainable%: Z` 為準。建議於微調首輪 log 中擷取四個模型對應數值，整理為**表 3.4** 之一欄。

### 量化載入
- 全部四個學生模型皆以 `load_in_4bit=True` 載入（4-bit NF4 量化），`load_in_8bit=False`、`full_finetuning=False`
- 4-bit 量化僅用於 base weight 載入，LoRA 權重以 fp16/bf16 訓練
- `dtype: auto`（依硬體決定 fp16 或 bf16；本研究 RTX 4090 → bf16）

---

## 3.5.2 訓練超參數
（`config/config.yaml::step6_cv.finetune.training`，由 `trl.SFTTrainer` 套用）

### 完整超參數表
| 類別 | 參數 | 值 | 備註 |
|---|---|---|---|
| 學習率 | `learning_rate` | 2e-4 | 全模型共用 |
| 批次 | `per_device_train_batch_size` | 2 | 受 24GB VRAM 限制 |
| 批次 | `gradient_accumulation_steps` | 4 | 等效 batch size = 8 |
| 訓練步數 | `num_train_epochs` | 5（X/Y/Z plot 模式：[7, 9, 11, 13, 15]） | 以 epoch 計，依資料量自動換算 step |
| Warmup | `warmup_steps` | 5 | **絕對步數**而非比例 |
| 優化器 | `optim` | `adamw_8bit` | bitsandbytes 8-bit AdamW（節省 optimizer state VRAM） |
| 權重衰減 | `weight_decay` | 0.01 | |
| 排程 | `lr_scheduler_type` | `linear` | 線性衰減至 0 |
| Loss masking | `train_on_responses_only` | 啟用 | 僅 assistant token 計入 loss |
| 序列長度 | `max_seq_length` | 2048 | tokenizer 截斷上限 |
| 隨機種子 | `seed` / `random_state` | 3407 | 全模型共用 |
| 精度 | `fp16` / `bf16` | bf16（RTX 4090 自動選用） | 由 `runtime.dtype: auto` 與 `resolve_trainer_precision` 決定 |
| Gradient checkpointing | `use_gradient_checkpointing` | `auto` | Unsloth 自動模式 |
| 報告 | `report_to` | `"none"` | 不外接 W&B / TB |

### 優化器選擇
採 `adamw_8bit`（bitsandbytes 實作之 8-bit AdamW）：
- 將 optimizer state（first/second moment）以 8-bit 儲存，於 24GB VRAM 上微調 3B–4B 模型時釋出顯著空間
- 與標準 fp32 AdamW 之數值差異於本任務尺度下可忽略

### Warmup 設定備註
本研究採**絕對步數** `warmup_steps=5` 而非比例。由於各 fold 訓練資料量約 3,500–4,000 筆、`per_device_train_batch_size=2` × `gradient_accumulation_steps=4` → 每 epoch 約 437–500 步、5 epoch 共約 2,200–2,500 步，warmup 比例約 **0.2%–0.23%**（極短 warmup，依 Unsloth 官方 notebook 範本）。

### 訓練硬體環境
| 項目 | 規格 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090（24 GB GDDR6X） |
| CPU | AMD Ryzen 9 9900X |
| 量化 / 精度 | 4-bit base + bf16 LoRA + 8-bit AdamW |
| 框架 | Unsloth + `trl.SFTTrainer` + PyTorch + Transformers |

---

## 3.5.3 訓練流程

### 三種資料組態（experiments）
（`config/config.yaml::step4_cv.balance`、`step6_cv.local_slm_baseline.experiments`）

| 實驗代號 | `balance.enabled` | `balance.include_in_cv` | 資料組態 |
|---|---|---|---|
| **base**（實驗 1） | `false` | — | 僅原始 500 筆人工標註 +（可選）偽標籤 + 合成資料 |
| **balance**（實驗 2） | `true` | `false` | 將 `blance_500.csv` **僅加入訓練集**（不參與 CV split），驗證集仍為原 500 筆人工標註 |
| **combined**（實驗 3） | `true` | `true` | `blance_500.csv` 與原 500 筆人工標註**合併後再 split**，訓練與驗證皆含平衡資料 |

每個學生模型會於上述三種資料組態各跑一次完整 3-fold 訓練 → 共 **4 模型 × 3 fold × 3 組態 = 36 次微調 run**。

### 各 fold 獨立訓練細節
（`src/step4_cv/create_folds.py`、`src/step6_cv/run_cv_finetune.py`）

1. **分層切分**：`StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`
   - 對於頻次 < 3 的稀有類別（Pollution & Waste 5、Community Relations 5、Natural Capital 3）採 round-robin 補齊，確保每個 validation fold 至少含 1 筆
2. **獨立資料管線**：每 fold 之 train pool 獨立通過 step5（reasoning 生成）→ step6_sft（SFT 組裝）
3. **獨立模型實體**：每 fold 重新從 base model 載入權重 → 套用 LoRA → 重新訓練（無跨 fold 權重共享）
4. **多進程隔離**：訓練與推論透過 `multiprocessing.Process(start_method="spawn")` 啟動子進程，避免 CUDA context / VRAM 殘留影響後續 fold
5. **隨機性**：訓練 seed `3407`（共用）；fold split seed `42`（資料層）；SFT shuffle seed `3407 + fold_idx`（每 fold 不同打散順序）

### 模型儲存與切換策略
（`config/config.yaml::step6_cv.finetune.checkpointing`、`inference`）

**儲存路徑結構**：
```
models/<experiment_name>/<variant_name>/<model_type>/fold_<k>/
├── adapter/                  # 推論用之最終 LoRA 權重副本
└── checkpoints/
    ├── checkpoint-<step>/    # 訓練過程之 epoch 檢查點
    └── ...
```

**Checkpointing 設定**：
- `save_strategy: auto`（依 epoch 數自動決定每 epoch 存一次）
- `save_total_limit: null`（不限制）
- `save_final_checkpoint_copy: false`（僅保留 trainer 自動產生之最終 checkpoint）
- `minimum_free_space_gb: 2`（磁碟保護）

**Checkpoint 切換策略**（`finetune.inference.checkpoint_strategy`，4 種模式）：
| 模式 | 行為 |
|---|---|
| `final` | 使用 `trainer.train()` 結束之最終 checkpoint（預設） |
| `latest` | 最後一個 epoch 之 checkpoint |
| `epoch` | 指定 `checkpoint_epoch` 值之 checkpoint |
| `best_epoch` | 由 X/Y/Z 掃描結果決定（依 `best_epoch_metric`，預設 `macro_f1`） |

**X/Y/Z plot（多 epoch 掃描）**：
- 配置：`xyz_plot.enabled` / `xyz_plot.epochs`（如 `[7, 9, 11, 13, 15]`）
- 僅於 `fold_0` 執行：先以 `max(epochs)` 訓練單次，再對中途各 epoch checkpoint 分別推論，找出 `sweet_spot_macro_f1`
- 輸出：`xyz_plot_summary.json` / `.csv`

### 變體（variant）資料夾
依 `finetune.ablation.variant_name` 區分輸出目錄，避免不同實驗條件互相覆蓋：
- `default`、`label_only`、`human_only`、`synthetic_only`、`label_only_human_only`…
- 同一變體下若 `resume: true`，已完成 fold 之 adapter 不會重訓

### 建議搭配：表 3.4 訓練超參數彙整表
建議欄位（一張表呈現所有四個學生模型）：
| 模型 | base weights | Unsloth API | LoRA targets | r/α/dropout | LR | Batch (per×accum=eff) | Epochs | Warmup | Optimizer | Weight decay | Scheduler | Seq len | Trainable params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

`Trainable params` 欄位請於微調首輪 log 中擷取（見 3.5.1 備註）。

---

## 附：相關程式碼與配置定位
- LoRA 套用：`src/step6_cv/run_cv_finetune.py::_load_model_and_apply_lora`（檢索 `get_peft_model` 區塊）
- SFT trainer 配置：`src/step6_cv/run_cv_finetune.py`（`SFTTrainer` + `train_on_responses_only`）
- Fold 切分：`src/step4_cv/create_folds.py::stratified_split_with_rare_class_protection`
- Checkpoint 策略：`src/step6_cv/run_cv_finetune.py`（搜尋 `checkpoint_strategy`）
- 配置：`config/config.yaml::step6_cv.finetune.training` / `checkpointing` / `inference` / `xyz_plot` / `ablation`
