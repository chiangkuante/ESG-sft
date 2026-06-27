# ESG 10-K 分類實驗：final_v1 重跑計畫與既有架構

## 2026-06-27 下一輪執行計畫

> 狀態：`WAITING_FOR_USER_APPROVAL`
>
> 本節是下一輪唯一有效的執行清單。目前只完成計畫撰寫；在使用者確認前，不執行 Step 2–5、不產生 Step 6 程式碼、不啟動訓練或推論。

### 0. 已鎖定的實驗條件

- Ground truth 版本固定為 `final_v1`：
  - Base：`data/origin_data/10k_1A/base_500_final_v1.csv`，500 筆。
  - Blance：`data/origin_data/10k_1A/blance_500_final_v1.csv`，480 筆。
  - Fingerprint manifest：`data/origin_data/10k_1A/ground_truth_final_v1_manifest.json`。
- Base 必須維持 500 筆，其中 Natural Capital 9 筆、Community Relations 6 筆。
- Label Studio 的 60 筆核定結果是最終標籤來源：base 39 筆、blance 21 筆。
- 三個實驗都只使用人工標注 ground truth，不使用 pseudo-label，不使用 synthetic data。
- Step 5 只針對 fold training pool 產生 reasoning；validation 不產生 reasoning，也不得進入訓練資料。
- SFT assistant 回覆固定包含：

```text
<reasoning>...</reasoning>
<label>...</label>
```

- Step 6 只執行 reasoning + label 正式版本：
  - `label_only: false`
  - `human_only: false`
  - `synthetic_only: false`
  - `non_esg_cap: null`
  - 不建立任何消融 variant
  - `xyz_plot.enabled: false`
  - `num_train_epochs: 15`
  - `weight_decay: 0.01`
  - `checkpoint_strategy: final`
- 所有 Python 環境與執行入口使用 `uv`。
- 所有實驗選項、模型、fold、機器分工都寫在 `config/config.yaml`；正式執行時不傳 `--model`、`--fold`、`--experiment` 等 CLI 實驗參數。
- Machine A 統一準備 Step 2–5 與 Step 6 輸入資料；Machine B 不自行重建 fold 或 reasoning。

### 1. 核准閘門

- [x] 產生 `base_500_final_v1.csv`。
- [x] 產生 `blance_500_final_v1.csv`。
- [x] 產生 ground-truth file SHA-256 與 content fingerprint。
- [x] `config/config.yaml` 的 ground-truth 路徑已指向 `final_v1`。
- [ ] 使用者確認本執行計畫。
- [ ] 確認後才開始下列程式修正、Step 2–5 重跑與 Step 6 自動化建置。

### 2. 重跑前必須完成的程式與 config 修正

#### 2.1 `final_v1` namespace 與 fingerprint 防呆

- [ ] 在 config 增加全域 `run_id: final_v1`。
- [ ] 所有新產物寫入獨立版本目錄，不覆寫或混用既有未版本化結果：

```text
data/processed/step2_classification/final_v1/
data/processed/step4_cv/final_v1/{base,balance,combined}/
data/processed/step5_reasoning/final_v1/{base,balance,combined}/
data/processed/step6_sft/final_v1/{base,balance,combined}/
models/final_v1/{base,balance,combined}/
results/final_v1/{base,balance,combined}/
```

- [ ] 每個 stage manifest 至少記錄：
  - stage 名稱與 `run_id`
  - ground-truth file SHA-256
  - ground-truth content fingerprint
  - config fingerprint
  - 直接上游檔案 fingerprint
  - 程式碼 fingerprint
  - 產生時間、筆數、類別分布與輸出檔 SHA-256
- [ ] 若 cache、fold、SFT 或 result 的 fingerprint 與目前輸入不一致，必須停止或重建；不得因檔案存在就直接續跑。
- [ ] 舊的 `data/processed/step4_cv/`、`data/processed/step5_reasoning/`、`data/processed/step6_sft/`、`models/`、`results/` 產物視為舊版，不得被 `final_v1` resume 邏輯採用。

#### 2.2 Step 2 改為 config-driven

- [ ] 在 `config/config.yaml` 新增 `step2_classification` 區段，至少包含：
  - input path
  - output path
  - model name
  - batch size
  - checkpoint interval
  - device
  - full rerun / resume policy
- [ ] 新增無 CLI 實驗參數的 Step 2 入口，或將既有入口改為直接讀 config。
- [ ] 本輪固定為 full rerun，不讀取舊 `classified_checkpoint.json`。
- [ ] Step 2 輸出必須寫到 `final_v1` 目錄並產生 manifest/fingerprint。

#### 2.3 Step 4 改為 group-aware 3-fold

- [ ] 不沿用舊 `cv_folds.json`；依 final_v1 ground truth 重建全部 fold。
- [ ] 對正規化後的 `combined_text` 計算 `text_fingerprint`，相同內容視為同一 group。
- [ ] 使用 stratified group-aware split，確保同 group 不會同時出現在 train 與 validation。
- [ ] 若同一 text group 出現互相衝突的 ground-truth label，立即停止並輸出 conflict report，不可靜默選標籤。
- [ ] 實驗 2 的 blance 資料雖只加入 training，但每個 fold 仍須排除與該 fold validation 相同 text group 的資料。
- [ ] 實驗 3 合併 base + blance 後再做 group-aware split。
- [ ] 每個 fold 產生 leakage audit，驗證：
  - train/validation 的 `paragraph_id` 無交集
  - train/validation 的 `text_fingerprint` 無交集
  - validation 每筆只出現於一個 fold
  - 每個 fold 的九類 support 與分布可追溯

#### 2.4 Step 5 改為內容型 reasoning cache

- [ ] 不再以 `paragraph_id` 作為唯一 cache key。
- [ ] Cache key 至少包含：
  - 正規化 `combined_text`
  - canonical ground-truth label
  - reasoning prompt version
  - provider 與 model
  - reasoning 生成參數
- [ ] 建立 final_v1 共用 cache，使相同文本、相同標籤、相同 prompt/model 可跨 fold 與跨實驗重用。
- [ ] 若文本相同但標籤不同，不得共用 reasoning。
- [ ] Cache 寫入需採原子更新，並保留 success/failed、重試次數、response metadata 與內容 fingerprint。
- [ ] 每個 fold 的 `train_with_reasoning.json` 仍須由當下 `train_pool.json` 重新組裝，不能直接把舊 fold 檔案當 cache。
- [ ] Reasoning report 必須列出 cache hit、cache miss、API generated、failed 與 invalid 數量。

#### 2.5 Config 正式值

目前 config 與目標值的差異如下；確認後才修改：

| Config key | 目前值 | final_v1 目標值 |
|---|---:|---:|
| `step5_reasoning.sft.run_sft_build` | `false` | `true` |
| `step6_cv.finetune.ablation.label_only` | `true` | `false` |
| `step6_cv.finetune.ablation.variant_name` | `label_only` | `reasoning_label` |
| `step6_cv.finetune.ablation.non_esg_cap` | `150` | `null` |
| `step6_cv.finetune.training.weight_decay` | `0.01` | `0.01` |
| `step6_cv.finetune.training.num_train_epochs` | `15` | `15` |
| `step6_cv.finetune.xyz_plot.enabled` | `false` | `false` |
| `step4_cv.pseudo_labels.enabled` | `false` | `false` |
| `step4_cv.synthetic_generation.enabled` | `false` | `false` |

- [ ] 移除或停用 `label_only` variant 的輸出路徑，避免與正式 reasoning + label 結果混淆。
- [ ] `xyz_plot.epochs` 即使保留 `[15]` 也不得觸發 sweep；正式執行只訓練一次 15 epochs。
- [ ] SFT build 驗證每筆 assistant 內容同時存在非空 `<reasoning>` 與合法 `<label>`。

### 3. 確認後由 Machine A 執行 Step 2–5

#### 3.1 Preflight

- [ ] 重新計算兩份 ground truth SHA-256，必須與 `ground_truth_final_v1_manifest.json` 相符。
- [ ] 驗證 base=500、blance=480、九類標籤合法、`paragraph_id` 不重複、文本不為空。
- [ ] 驗證 Step 1 `paragraphs.json` 與 `parsing_stats.json` 一致。
- [ ] 驗證 API key、GPU、磁碟空間與 `uv.lock` 環境，但不得將 secret 寫入 manifest。
- [ ] 執行 config schema 與 dry validation；只驗證，不先啟動昂貴工作。

#### 3.2 Step 2：完整 FinBERT classification

- [ ] 從 Step 1 完整 `paragraphs.json` 重新推論，不使用舊 checkpoint 或舊 classified output。
- [ ] 產生：

```text
data/processed/step2_classification/final_v1/classified.json
data/processed/step2_classification/final_v1/classification_stats.json
data/processed/step2_classification/final_v1/manifest.json
```

- [ ] 驗證輸出筆數與 Step 1 一致，每筆都有合法 `finbert_label` 與 `[0, 1]` 範圍的 confidence。
- [ ] 雖然本輪不使用 pseudo-label，仍保留完整 Step 2 作為 FinBERT 結果、統計與後續可追溯依據。

#### 3.3 Step 3

- [x] 不重跑。人工核定已整合到 final_v1 ground truth。

#### 3.4 Step 4：依序建立三個實驗

Job matrix 固定為：

| experiment | balance.enabled | balance.include_in_cv | CV population | fold training pool |
|---|---:|---:|---|---|
| `base` | false | false | base 500 | base fold train |
| `balance` | true | false | base 500 | base fold train + 無 validation group 洩漏的 blance |
| `combined` | true | true | base 500 + blance 480 | combined fold train |

- [ ] 用 config matrix 一次建立三個實驗，不手動反覆編輯 `balance.enabled/include_in_cv`。
- [ ] 所有實驗固定 3 folds、shuffle=true、random_state=42。
- [ ] `pseudo_labels.enabled=false`，不得讀取或加入 pseudo labels。
- [ ] `synthetic_generation.enabled=false`，不得建立或加入 synthetic samples。
- [ ] 每個實驗輸出 folds、train pool、validation、summary、leakage audit 與 manifest。
- [ ] 三個實驗全部通過 group leakage 與 fingerprint 驗證後才進入 Step 5。

#### 3.5 Step 5A：Reasoning generation

- [ ] 按 `base → balance → combined` 處理，以提高後續實驗的內容 cache hit。
- [ ] Reasoning 僅能以 train item 的 final ground-truth `label` 為目標產生。
- [ ] 禁止用 FinBERT prediction 取代 ground-truth label。
- [ ] 先讀共用 content cache，再只向 API 請求 cache miss。
- [ ] 重試 failed/invalid reasoning，直到每個 train pool item 都有有效 reasoning；若仍失敗則停止，不進入 SFT build。
- [ ] 驗證 reasoning 不含 Markdown code fence、XML wrapper 或與 ground-truth 不一致的輸出 label。

#### 3.6 Step 5B：SFT build

- [ ] 設定 `run_sft_build: true`，為三個實驗建立 SFT。
- [ ] 每筆 train sample 必須同時包含 reasoning 與 label。
- [ ] `val_eval.json` 只含 ground truth 與文本，不包含 training reasoning。
- [ ] 產生：

```text
data/processed/step6_sft/final_v1/{base,balance,combined}/manifest.json
data/processed/step6_sft/final_v1/{base,balance,combined}/fold_{0,1,2}/train_sft_text.json
data/processed/step6_sft/final_v1/{base,balance,combined}/fold_{0,1,2}/val_eval.json
```

- [ ] 驗證三個實驗共 9 個 fold 的 SFT/validation 檔完整、fingerprint 正確、reasoning failed=0。

### 4. 確認後產生 Step 6 所需程式碼、腳本與 config

#### 4.1 預定新增或重構的入口

- [ ] `src/step6_cv/job_matrix.py`
  - 從 config 建立 deterministic job matrix 與穩定 job ID。
  - Job ID 包含 `run_id/experiment/model/fold/task_type`。
- [ ] `src/step6_cv/prepare_step6_bundle.py`
  - 由 Machine A 驗證 9 個 SFT fold。
  - 建立 Machine B 所需的唯讀輸入 bundle 與 SHA-256 manifest。
- [ ] `src/step6_cv/run_step6_jobs.py`
  - 只讀 `config/config.yaml` 的 `active_machine` 與 job assignment。
  - 逐一執行 assigned pending jobs，支援 fingerprint-safe resume。
  - 不使用 `--model`、`--fold`、`--experiment`。
- [ ] `src/step6_cv/collect_step6_results.py`
  - 驗證兩台機器輸出 receipt 與 fingerprint。
  - 合併 results，不以較新檔案靜默覆蓋衝突結果。
- [ ] `src/step6_cv/evaluate_all_experiments.py`
  - 一次彙整 base、balance、combined。
  - 驗證 fold 完整性後輸出 overall、per-class、pillar metrics。
- [ ] 視需要把既有 `run_cv_finetune.py` 與 `run_cv_inference.py` 重構為可由 job runner 直接呼叫單一 model/fold 的函式；保留既有模型載入邏輯。
- [ ] 為 job matrix、fingerprint、resume、result merge 與 config validation 建立測試。

#### 4.2 Step 6 job matrix

正式 LoRA 工作總數：

```text
3 experiments × 4 models × 3 folds = 36 fine-tune + inference jobs
```

模型固定為：

- `gemma`
- `llama`
- `qwen`
- `ministral`

每個 job 固定：

- 使用對應 fold 的 `train_sft_text.json` 與 `val_eval.json`
- 15 epochs
- reasoning + label objective
- final checkpoint inference
- 不執行 X/Y/Z plot
- 不執行 label-only、human-only、synthetic-only 或 alpha ablation
- 產生 prediction、fold metrics、job receipt 與 output fingerprint

若保留完整既有比較範圍，另建立：

- FinBERT baseline：3 experiments × 3 folds = 9 fold evaluations。
- API LLM baseline：3 models × 3 experiments × 3 folds = 27 fold evaluations。
- 未微調 local SLM baseline：4 models × 3 experiments × 3 folds = 36 fold evaluations。

這些 baseline 不得產生額外 ablation variant。

#### 4.3 Config 預定結構

確認後在 `config/config.yaml` 加入下列概念；實際欄位以 schema/test 為準：

```yaml
pipeline:
  run_id: final_v1
  enforce_fingerprints: true

step4_cv:
  experiments:
    - name: base
      balance_enabled: false
      include_balance_in_cv: false
    - name: balance
      balance_enabled: true
      include_balance_in_cv: false
    - name: combined
      balance_enabled: true
      include_balance_in_cv: true

step5_reasoning:
  cache:
    strategy: content
    include_label: true
    include_prompt_and_model: true
  sft:
    run_sft_build: true

step6_cv:
  job_matrix:
    enabled: true
    active_machine: machine_a
    experiments: [base, balance, combined]
    folds: [0, 1, 2]
    machines:
      machine_a:
        models: [gemma, llama]
      machine_b:
        models: [qwen, ministral]
  finetune:
    training:
      num_train_epochs: 15
      weight_decay: 0.01
    ablation:
      label_only: false
      human_only: false
      synthetic_only: false
      non_esg_cap: null
      variant_name: reasoning_label
    xyz_plot:
      enabled: false
    inference:
      checkpoint_strategy: final
```

### 5. 兩台機器的實際分工

#### Machine A

- [ ] 完成第 2 節的程式/config 修正與測試。
- [ ] 執行 Step 2、三個 Step 4、三個 Step 5 reasoning/SFT。
- [ ] 產生 Step 6 input bundle：

```text
artifacts/final_v1/step6_input_bundle.zip
artifacts/final_v1/step6_input_bundle_manifest.json
```

- [ ] 執行 Gemma + Llama 的 18 個 LoRA jobs。
- [ ] 執行 Gemma + Llama 的未微調 baseline jobs。
- [ ] 統一執行 FinBERT 與 API LLM baseline。
- [ ] 收回 Machine B 結果並執行最終 evaluation。

#### Machine B

- [ ] 取得 Machine A 建立的同一份 Step 6 input bundle。
- [ ] 驗證 bundle SHA-256 與內含 manifest；不自行重建 fold、reasoning 或 SFT。
- [ ] 將 `step6_cv.job_matrix.active_machine` 設為 `machine_b`。
- [ ] 執行 Qwen + Ministral 的 18 個 LoRA jobs。
- [ ] 執行 Qwen + Ministral 的未微調 baseline jobs。
- [ ] 將 results、job receipts 與 output manifest 回傳 Machine A；模型 adapter 可另行保存，不強制放入小型結果 bundle。

兩台機器執行同一個無參數入口：

```bash
uv run python src/step6_cv/run_step6_jobs.py
```

Job runner 必須做到：

- 已完成且 fingerprint 相符才可 resume/skip。
- 同 job 若 input/config fingerprint 不同，必須建立衝突報告並停止。
- 每次只執行 config 指派給本機器的 jobs。
- job failure 不得標記完成；重新執行時從 failed/pending job 繼續。
- GPU 型號、套件版本、模型 revision、seed、有效超參數與 checkpoint 路徑寫入 receipt。

### 6. 最終驗收條件

- [ ] Step 2 輸出筆數等於 Step 1，所有 FinBERT prediction 合法。
- [ ] 三個實驗均為新建 final_v1 folds，沒有使用舊 fold。
- [ ] 9 個 fold 全部通過 paragraph ID 與 text group leakage audit。
- [ ] Reasoning cache key 包含文本、ground-truth label、prompt/model 設定。
- [ ] 9 個 `train_with_reasoning.json` 與 SFT fold reasoning failed=0。
- [ ] 所有 SFT assistant response 同時包含 reasoning 與 label。
- [ ] 36 個 LoRA jobs 全部使用 15 epochs、`label_only=false`、`non_esg_cap=null`。
- [ ] 沒有 X/Y/Z plot 或其他 ablation 輸出。
- [ ] 每個模型/實驗都有 3 個 fold predictions 與 metrics。
- [ ] Machine A/B job receipt 沒有重複、缺漏或 fingerprint 衝突。
- [ ] 最終 evaluation 產生 Accuracy、Macro/Weighted F1、Kappa、per-class 與 pillar-level mean/std。
- [ ] 最終報告明確記錄 `final_v1` ground-truth 與所有 input/config/output fingerprints。

---

## 既有架構參考

以下內容保留作為重構前的架構與舊產物參考。凡與上方 `final_v1` 計畫衝突者，以上方計畫為準；既有未版本化產物不得直接作為本輪正式結果。


本文件只保留目前實際執行的三個實驗。所有流程設定以 `config/config.yaml` 為準，不在命令列傳遞實驗參數。偽標籤與 LLM 合成資料流程不列入本輪 todo。

---

## 目前目標

使用目前的人工標註資料 `chiang_500.csv`，搭配額外平衡資料集 `blance_500.csv`，建立三種 3-fold CV 實驗，並比較：

- FinBERT baseline
- API LLM baseline
- 四個 local SLM 的 LoRA 微調結果
- 四個未微調 local SLM 的 zero-shot baseline

本輪三個實驗皆不使用偽標籤與合成資料：

```yaml
step4_cv:
  human_csv: data/origin_data/10k_1A/chiang_500.csv
  pseudo_labels:
    enabled: false
  synthetic_generation:
    enabled: false
```

---

## 三個實驗

`experiment_name` 由 `src/step4_cv/common.py::resolve_experiment_name` 依照 `step4_cv.balance` 自動決定。

| 實驗 | experiment_name | 路徑位置 | balance.enabled | balance.include_in_cv | 資料切法 | 訓練資料 |
|---|---|---:|---:|---|---|
| 實驗 1 | `base` | 根目錄 | false | false | `chiang_500.csv` 做 3-fold | fold train only |
| 實驗 2 | `balance` | `balance/` 子目錄 | true | false | `chiang_500.csv` 做 3-fold | fold train + 全量 `blance_500.csv` |
| 實驗 3 | `combined` | `combined/` 子目錄 | true | true | `chiang_500.csv` + `blance_500.csv` 合併後做 3-fold | 合併資料的 fold train |

目前 `config/config.yaml` 的 active 設定是實驗 2：

```yaml
step4_cv:
  human_csv: data/origin_data/10k_1A/chiang_500.csv
  balance:
    enabled: true
    include_in_cv: false
```

若要重建另外兩個實驗，僅調整 `step4_cv.balance`，再重跑 Step 4 與 Step 5 reasoning/SFT。

---

## 目錄與資料流

所有中間檔與結果依 `experiment_name` 分目錄保存；`base` 實驗直接放在根目錄，不另外建立子資料夾：

```text
data/processed/step4_cv/
data/processed/step5_reasoning/
data/processed/step6_sft/
models/
results/
```

三個實驗對應：

```text
data/processed/step4_cv/balance/
data/processed/step4_cv/
data/processed/step4_cv/combined/

data/processed/step6_sft/balance/
data/processed/step6_sft/
data/processed/step6_sft/combined/

results/balance/
results/
results/combined/
```

---

## Step 4：CV Fold 與訓練池組裝

入口：`src/step4_cv/run.py`

設定來源：`config/config.yaml` 的 `step4_cv`

主要責任：

- 載入人工標註資料：`data/origin_data/10k_1A/chiang_500.csv`
- 視 `balance.enabled` 載入 `data/origin_data/10k_1A/blance_500.csv`
- 依 `balance.include_in_cv` 決定平衡資料是否參與 CV split
- 建立或讀取 `cv_folds.json`
- 產生每個 fold 的 `human_train.json`、`human_val.json`、`train_pool.json`
- 在目前設定下跳過 FinBERT pseudo-label pool 載入
- 在目前設定下不產生 synthetic requests

重要輸出：

```text
data/processed/step4_cv/manifest.json
data/processed/step4_cv/cv_folds.json
data/processed/step4_cv/fold_0/human_train.json
data/processed/step4_cv/fold_0/human_val.json
data/processed/step4_cv/fold_0/train_pool.json
data/processed/step4_cv/fold_0/train_pool_summary.json
```

`pseudo_labels.json` 會存在但在本輪實驗中為空；`synthetic_requests.json` 會存在但不使用。

---

## Step 5A：Reasoning 生成

入口：`src/step5_reasoning/run.py`

實際 reasoning 生成：`src/step5_reasoning/generate_reasoning.py`

設定來源：`config/config.yaml` 的 `step5_reasoning`

主要責任：

- 讀取 `data/processed/step4_cv/{experiment_name}/manifest.json`
- 對每個 fold 的 `train_pool.json` 產生 ESG 分類 reasoning
- 使用 `step5_reasoning.generation.model` 指定 API 模型
- 支援既有結果續跑，避免重複生成已完成樣本

重要輸出：

```text
data/processed/step5_reasoning/{experiment_name}/fold_0/reasoning_input.json
data/processed/step5_reasoning/{experiment_name}/fold_0/train_with_reasoning.json
data/processed/step5_reasoning/{experiment_name}/fold_0/reasoning_report.json
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
data/processed/step6_sft/{experiment_name}/manifest.json
data/processed/step6_sft/{experiment_name}/fold_0/train_sft_text.json
data/processed/step6_sft/{experiment_name}/fold_0/val_eval.json
data/processed/step6_sft/{experiment_name}/fold_0/excluded_items.json
```

---

## Step 6A：Baseline 與模型評估

### FinBERT Baseline

入口：`src/step6_cv/run_cv_finbert.py`

設定來源：

```yaml
step6_cv:
  finbert:
    step4_output_dir: data/processed/step4_cv
    results_dir: results/finbert
```

輸出：

```text
results/finbert/{experiment_name}/fold_0_results.json
results/finbert/{experiment_name}/fold_0_metrics.json
results/finbert/{experiment_name}/overall_folds.json
```

### API LLM Baseline

入口：`src/step6_cv/run_cv_api_llm.py`

設定來源：`step6_cv.api_llm`

目前啟用模型：

- `gemini_3_flash_preview`
- `gpt_5_4_mini_2026_03_17`
- `claude_sonnet_4_6`

輸出：

```text
results/{experiment_name}/api_llm/{model_key}/fold_0_results.json
results/{experiment_name}/api_llm/{model_key}/fold_0_metrics.json
results/{experiment_name}/api_llm/{model_key}/overall_summary.json
results/{experiment_name}/api_llm/{model_key}/overall_summary.md
```

### Local SLM LoRA 微調

入口：`src/step6_cv/run_cv_finetune.py`

設定來源：`step6_cv.finetune`

目前模型 registry：

```yaml
step6_cv:
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
models/{experiment_name}/{model_type}/fold_0/adapter/
models/{experiment_name}/{model_type}/fold_0/checkpoints/
results/{experiment_name}/{model_type}/fold_0_results.json
results/{experiment_name}/{model_type}/fold_0_metrics.json
results/{experiment_name}/{model_type}/overall_summary.json
results/{experiment_name}/{model_type}/overall_summary.md
```

### Local SLM 未微調 Baseline

入口：`src/step6_cv/run_cv_local_slm_baseline.py`

設定來源：`step6_cv.local_slm_baseline`

目前設定會一次讀取三個實驗的 SFT manifest：

```yaml
step6_cv:
  local_slm_baseline:
    enabled: true
    experiments:
      - base
      - balance
      - combined
    label_only: false
    max_new_tokens: 512
```

此流程不讀取 LoRA adapter，不做訓練，只載入 base model 對 `val_eval.json` 推論。

輸出：

```text
results/{experiment_name}/local_slm_baseline/{model_type}/fold_0_results.json
results/{experiment_name}/local_slm_baseline/{model_type}/fold_0_metrics.json
results/{experiment_name}/local_slm_baseline/{model_type}/overall_summary.json
results/{experiment_name}/local_slm_baseline/{model_type}/overall_summary.md
```

---

## Step 6B：統一評估彙整

入口：`src/step6_cv/evaluate_cv_metrics.py`

設定來源：

```yaml
step6_cv:
  evaluation:
    results_root: results
```

主要責任：

- 讀取目前 active `experiment_name` 底下的 `fold_*_results.json`
- 重新計算 overall metrics
- 輸出 per-class 與 pillar-level metrics

輸出：

```text
overall_summary.json
overall_folds.csv
per_class_metrics.csv
pillar_metrics.csv
```

注意：此彙整器依目前 `step4_cv` 的 active `experiment_name` 掃描單一實驗。如果要彙整 `base`、`balance`、`combined`，需分別切換 `step4_cv.balance` 後執行。

---

## 當前 Todo

### 已完成或已有產物

- `data/processed/step4_cv/manifest.json`
- `data/processed/step4_cv/balance/manifest.json`
- `data/processed/step4_cv/combined/manifest.json`
- `data/processed/step6_sft/manifest.json`
- `data/processed/step6_sft/balance/manifest.json`
- `data/processed/step6_sft/combined/manifest.json`
- `src/step6_cv/run_cv_local_slm_baseline.py`
- `step6_cv.local_slm_baseline` config

### 需要確認或執行

- 對三個實驗確認 `train_with_reasoning.json` 是否完整，缺漏時重跑 Step 5A。
- 若 reasoning 更新，將 `step5_reasoning.sft.run_sft_build` 設為 `true` 後重建 SFT manifest。
- 對四個 local SLM 分別完成 LoRA fine-tune 與 inference。
- 對三個 API LLM baseline 補齊 `base`、`balance`、`combined` 結果。
- 執行未微調 local SLM baseline，補齊三個實驗與四個 local model 的 36 組 fold 推論。
- 最後分別對 `base`、`balance`、`combined` 執行 metric 彙整。

---

## 評估指標

每個模型與每個實驗都輸出 3-fold mean/std：

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

## 實驗總量

| 類別 | 模型數 | 實驗數 | Fold | 說明 |
|---|---:|---:|---:|---|
| FinBERT | 1 | 3 | 3 | 9 組 fold 推論 |
| API LLM | 3 | 3 | 3 | 27 組 fold API 推論 |
| Local SLM LoRA | 4 | 3 | 3 | 36 組 fine-tune + inference |
| Local SLM 未微調 | 4 | 3 | 3 | 36 組 zero-shot local 推論 |

---

## 參考架構重點

- Step 4 只負責 fold 與 train pool，不在本輪實驗中做 pseudo-label 或 synthetic generation。
- Step 5 reasoning 與 SFT 依 `experiment_name` 讀寫，不跨實驗共用訓練資料。
- LoRA fine-tune 結果依 `{experiment_name}/{model_type}` 分開保存。
- 未微調 local baseline 依 `step6_cv.local_slm_baseline.experiments` 一次掃描三個實驗的 SFT manifest。
- 所有模型推論使用 `src/step6_cv/common.py` 中相同的 label list、prompt builder、label parser 與 metric function。
