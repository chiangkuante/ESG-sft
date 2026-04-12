# ESG 10-K Classification Project

## Rules
- Use traditional Chinese for all responses
- Don't use emoji
- Use `uv` to manage Python environment
- Don't use CLI tools; configure everything using `config.yaml` in `config/`
- Refer to `.todo/todo.md` for the full experiment pipeline
- Don't delete user-added comments

## Project Overview
This project classifies ESG (Environmental, Social, Governance) risks in U.S. 10-K filing Item 1A (Risk Factors) sections. Using FinBERT-ESG as the teacher model, we apply a knowledge distillation strategy to train LoRA fine-tuned LLMs (Gemma 3 4B, Qwen 3.5 4B, Llama 3.2 3B, Ministral 3B) that learn and surpass FinBERT-ESG's classification ability on 10-K Risk Factor paragraphs.

## Data Overview

- **Human-annotated dataset**: 500 samples (9 classes, from Project-16, single annotator)
- **Unlabeled dataset**: ~160,000 samples (same-source 10-K Risk Factor paragraphs)
- **FinBERT-Human agreement**: 95.6% (22/500 disagreements; 16/22 are ESG misclassified as Non-ESG)
- **Class distribution**: Non-ESG 297 (59.4%), Product Liability 59, Corporate Governance 40, Human Capital 37, Business Ethics & Values 29, Climate Change 25, Pollution & Waste 5, Community Relations 5, Natural Capital 3

## Data Pipeline
- **Step 0**: Download 10-K HTML from SEC EDGAR via `sec-downloader` (S&P 500, 2021-2025)
- **Step 1**: Parse Item 1A with `sec-parser` (heading-aware semantic parsing), output `combined_text` = heading + paragraph
- **Step 2**: FinBERT full prediction (~160,000 paragraphs) + random sampling 500 paragraphs
- **Step 3**: Human annotation in Label Studio (single annotator, Project-16, 500 samples with FinBERT pre-annotations)

## Class Balancing Strategy: Temperature-Based Smoothed Sampling

Formula from Conneau et al. (2020):

$$T_i = B \times \frac{n_i^\alpha}{\sum_j n_j^\alpha}$$

With lower bound: $T_i = \max(T_i, n_i)$ (no downsampling).

Parameters:
- $\alpha = 0.5$ (square-root smoothing)
- $B = 4000$ (total training budget, within LoRA effective range of 500-5,000)

Effect: compresses max/min class ratio from 83:1 to ~9:1 while keeping synthesis ratio below 90%.

## Validation Architecture: Stratified 3-Fold Cross-Validation

Using 3-Fold instead of 5-Fold because rare classes (Pollution & Waste 5, Community Relations 5, Natural Capital 3) may be completely absent in validation sets under 5-Fold. 3-Fold ensures each rare class appears at least 1 time per validation set.

### Per-Fold Pipeline (k = 1, 2, 3)

1. **Data split**: Stratified 3-Fold on 500 human-annotated samples (~333 train / ~167 val)
2. **Calibrate FinBERT confidence threshold**: Using only train split, find threshold for >=95% accuracy (per-class if possible, else global)
3. **Pseudo-labeling**: Filter high-confidence samples from 160K FinBERT predictions (cap Non-ESG at 800-1,000, ~2,000-3,000 total)
4. **Compute target counts**: Merge train + pseudo-labels, apply temperature smoothing formula to get per-class targets
5. **LLM synthetic data generation**: For classes needing augmentation ($S_i > 0$), use LLM API with few-shot seeds from train-only (no val leakage). Generate boundary cases targeting FinBERT's systematic weakness (ESG -> Non-ESG misclassification)
6. **LLM reasoning generation**: For all ~4,000 training samples, use LLM API to batch-generate chain-of-thought reasoning explaining ESG classification logic
7. **Assemble final training set**: Merge human annotations (~333) + pseudo-labels (~2,500) + synthetic (~1,500) = ~4,000 samples. Convert to SFT instruction-following format with XML tags: `<reasoning>...</reasoning>\n<label>...</label>`
8. **FinBERT inference**: Direct inference on validation set (no training)
9. **Local LLM LoRA fine-tuning + inference**: Fine-tune on training set (model learns both reasoning and classification), infer on validation set. 4 models x 3 folds = 12 fine-tuning runs

## Key Data Files
- `data/processed/step1_parsing/` — Parsed paragraphs from Item 1A
- `data/processed/step2_classification/classified.json` — FinBERT predictions on full ~160K pool
- `data/processed/step2_classification/classification_stats.json` — FinBERT prediction distribution stats
- `data/origin_data/10k_1A/project-16-at-2026-03-31-11-52-a8826fd3.csv` — 500-sample human-annotated dataset

## Model Fine-tuning (Unsloth)
- **Gemma 4 4B** (`unsloth/gemma-4-E4B-it`): `FastModel` API, chat template "gemma-4", remove BOS prefix
- **Llama 3.2 3B** (`unsloth/Llama-3.2-3B-Instruct`): `FastLanguageModel` API, `target_modules` list, chat template "llama-3.1", `DataCollatorForSeq2Seq`, `standardize_sharegpt` preprocessing
- **Qwen 3.5 4B** (`unsloth/Qwen3.5-4B`): `FastVisionModel` API, text-only vision format, `UnslothVisionDataCollator`
- **Ministral 3B** (`unsloth/Ministral-3-3B-Instruct-2512`): `FastVisionModel` API, text-only vision format, `UnslothVisionDataCollator`

### Unified Hyperparameters (per Unsloth official notebooks)
- LoRA: r=32, alpha=32, dropout=0, bias="none"
- Training: lr=2e-4, batch=2, grad_accum=4, num_train_epochs=5, warmup_steps=5
- Optimizer: adamw_8bit, weight_decay=0.001, lr_scheduler=linear
- random_state/seed: 3407, max_seq_length: 2048

### X/Y/Z Plot (Multi-Epoch Scan)
- 功能：在 fold_0 上對多個 epoch checkpoint 執行推論，找到最佳 epoch
- 配置：`finetune.xyz_plot.epochs: [7, 9, 11, 13, 15]`
- 輸出：`xyz_plot_summary.json`，記錄各 epoch 的 accuracy/macro_f1/weighted_f1/kappa 及 `sweet_spot_macro_f1`

### Checkpoint Strategy
`finetune.inference.checkpoint_strategy` 四種模式：
- `final`：使用最終 checkpoint（預設）
- `latest`：最後一個 epoch 的 checkpoint
- `epoch`：指定 `checkpoint_epoch` 值
- `best_epoch`：由 X/Y/Z 掃描結果決定（依 `best_epoch_metric` 指標，預設 macro_f1）

### API LLM Inference Models
用於推論對比（不做 LoRA 訓練）：
- **Gemini 3 Flash Preview**: 1K rpm, 2M tpm, 10K rpd
- **GPT-5.4-mini-2026-03-17**: 10K rpm, 10M tpm, 1B tpd
- **Claude Sonnet 4.6**: 1K rpm, 450K input_tpm, 90K output_tpm

## ESG Categories (9 classes)
Climate Change, Natural Capital, Pollution & Waste, Human Capital, Product Liability, Community Relations, Corporate Governance, Business Ethics & Values, Non-ESG

## Evaluation
- **Validation**: Stratified 3-Fold CV on 500 human-annotated samples (model comparison, report mean +/- std)
- **Metrics**: Accuracy, Macro/Weighted F1, per-class P/R/F1, Cohen's Kappa, Pillar-level F1
- **Pillar mapping**: Environmental (CC + NC + PW), Social (HC + PL + CR), Governance (CG + BEV), Non-ESG
- **Caveat**: Rare class per-class F1 may fluctuate due to small validation counts (1-2 per fold)

## Ablation Studies

透過 `config.yaml` 的 `finetune.ablation` 布林旗標與 `variant_name` 控制（對應輸出目錄名稱）：

- **`label_only`**: 移除訓練資料中的 `<reasoning>` 標籤，assistant 回應格式簡化為 `Label: {label}`，測試 chain-of-thought 推理的貢獻
- **`human_only`**: 只使用人工標註資料（~333 筆/fold），排除偽標籤與合成資料，對應「No pseudo-labels + No LLM synthetic」
- **`synthetic_only`**: 只保留 human + synthetic 資料，排除偽標籤，對應「No pseudo-labels」
- **不同 alpha 值**: 比較 alpha = 0.3, 0.5, 0.7 對類別平衡的影響
- **不同基礎模型**: 比較 Gemma 4 4B、Qwen 3.5 4B、Llama 3.2 3B、Ministral 3B

## References
- Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*.
- Nakada, R., et al. (2024). Synthetic Oversampling: Theory and A Practical Approach Using LLMs to Address Data Imbalance. *arXiv:2406.03628*.
