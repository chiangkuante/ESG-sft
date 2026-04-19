# ESG 10-K Classification

This project classifies ESG risks in Item 1A (Risk Factors) of U.S. 10-K filings into 9 categories: Climate Change, Natural Capital, Pollution & Waste, Human Capital, Product Liability, Community Relations, Corporate Governance, Business Ethics & Values, and Non-ESG. GPT-5.4 mini is used as the teacher model, and a rationale distillation strategy is applied to train LoRA fine-tuned SLMs (Gemma 4 E4B, Qwen 3.5 4B, Llama 3.2 3B, Ministral 3B), which are compared against the FinBERT baseline and API LLMs (Gemini, GPT, Claude).

本專案針對美國 10-K Filing 中 Item 1A (Risk Factors) 段落進行 ESG 風險分類，共 9 個類別：Climate Change、Natural Capital、Pollution & Waste、Human Capital、Product Liability、Community Relations、Corporate Governance、Business Ethics & Values、Non-ESG。以 GPT-5.4 mini 為教師模型，採推理蒸餾（Rationale Distillation）對 LoRA 微調的 SLM（Gemma 4 E4B、Qwen 3.5 4B、Llama 3.2 3B、Ministral 3B）進行訓練，並與 FinBERT baseline 及 API LLM（Gemini、GPT、Claude）比較。

## Data / 資料

Sampled from a pool of approximately 160,000 10-K Risk Factor paragraphs: 500 human-annotated samples (`chiang_500.csv`) and 480 class-balanced supplementary samples (`blance_500.csv`).

從約 160,000 段 10-K Risk Factor 段落抽取：人工標註 500 筆（`chiang_500.csv`）；類別平衡補充資料 480 筆（`blance_500.csv`）。

## Experimental Setup / 實驗設定

Stratified 3-Fold CV with three data compositions:

| Experiment | balance.enabled | balance.include_in_cv | Training Data |
|---|---|---|---|
| base | false | false | fold train of `chiang_500.csv` |
| balance | true | false | fold train + full `blance_500.csv` |
| combined | true | true | merge first, then fold split |

採 Stratified 3-Fold CV，三種資料組合：base 僅用 `chiang_500.csv`；balance 在 fold train 上追加全量 `blance_500.csv`；combined 將兩者合併後再做 fold split。

## Models / 模型

- FinBERT-ESG (zero-shot baseline)
- API LLMs: gemini-3-flash-preview, gpt-5.4-mini, claude-sonnet-4-6 (zero-shot baseline)
- Local SLMs (LoRA fine-tuned): Gemma 4 E4B, Qwen 3.5 4B, Llama 3.2 3B, Ministral 3B
- Local SLMs (zero-shot baseline): same four models without fine-tuning

包含 FinBERT-ESG 與三個 API LLM 的 zero-shot baseline；四個 local SLM 同時執行 LoRA 微調與未微調 baseline 兩種設定。

## Pipeline

```
step0_download       Download 10-K HTML from SEC EDGAR
step1_parsing        Parse Item 1A with sec-parser
step2_classification FinBERT full inference
step4_cv             CV fold and training pool assembly
step5_reasoning      LLM-generated chain-of-thought reasoning and SFT data assembly
step6_cv             FinBERT / API LLM / LoRA fine-tune / Local baseline inference and evaluation
```

Pipeline 分為：step0 下載 10-K HTML、step1 解析 Item 1A、step2 FinBERT 全量推論、step4 CV fold 組裝、step5 reasoning 與 SFT 資料組裝、step6 各模型推論與評估。

## Execution / 執行

Experiment parameters are primarily controlled via `config/config.yaml`; the environment is managed by `uv`. `run_cv_finetune.py` and `run_cv_inference.py` additionally require `--model` to specify the local SLM; other scripts take no CLI arguments.

```
uv run src/step4_cv/run.py
uv run src/step5_reasoning/run.py
uv run src/step6_cv/run_cv_finbert.py
uv run src/step6_cv/run_cv_api_llm.py
uv run src/step6_cv/run_cv_finetune.py --model {gemma|llama|qwen|ministral}
uv run src/step6_cv/run_cv_inference.py --model {gemma|llama|qwen|ministral}
uv run src/step6_cv/run_cv_local_slm_baseline.py
uv run src/step6_cv/evaluate_cv_metrics.py
```

To switch experiments, modify `step4_cv.balance.enabled` and `balance.include_in_cv` in `config.yaml`.

實驗參數以 `config/config.yaml` 為主，環境以 `uv` 管理。`run_cv_finetune.py` 與 `run_cv_inference.py` 另需 `--model` 指定 local SLM，其餘腳本不使用 CLI。切換實驗僅需修改 `config.yaml` 中 `step4_cv.balance.enabled` 與 `balance.include_in_cv`。

## Output Structure / 輸出結構

```
data/processed/step4_cv/{base|balance|combined}/       # CV fold data
data/processed/step5_reasoning/{...}/                  # Reasoning
data/processed/step6_sft/{...}/                        # SFT training data
models/{base|balance|combined}/{model}/fold_{k}/       # LoRA adapters
results/{base|balance|combined}/{...}/                 # Inference and evaluation results
```

依實驗名稱分目錄保存 CV 資料、reasoning、SFT 資料、LoRA adapter 與各模型推論結果。

## Evaluation Metrics / 評估指標

Each experiment and each model reports 3-fold mean/std: Accuracy, Macro F1, Weighted F1, Cohen's Kappa, per-class P/R/F1, and pillar-level F1 (Environmental / Social / Governance / Non-ESG).

每實驗每模型輸出 3-fold mean/std：Accuracy、Macro F1、Weighted F1、Cohen's Kappa、per-class P/R/F1、Pillar-level F1（Environmental / Social / Governance / Non-ESG）。

## Ablation

Switched via `human_only`, `label_only`, `synthetic_only`, and `variant_name` under `step6_cv.finetune.ablation`.

透過 `step6_cv.finetune.ablation` 的 `human_only`、`label_only`、`synthetic_only`、`variant_name` 切換。

## References

- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.*
- Huang, A. H. (2023). FinBERT-ESG-9-Categories [Model]. Hugging Face.
https://huggingface.co/yiyanghkust/finbert-esg-9-categories
- Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020.