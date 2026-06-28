#!/usr/bin/env bash
#
# final_v1 Machine A：Step 2 -> Step 5 完整重跑（+ 打包 Step 6 輸入 bundle）。
#
# 全部選項皆讀 config/config.yaml，不傳任何 --model / --fold / --experiment 實驗參數。
#
# 需求：
#   - GPU（Step 2 FinBERT 推論、約 10-30 分鐘）
#   - .env 內 OPENAI_API_KEY（Step 5 reasoning 會呼叫付費 API，數千次呼叫）
#
# 用法：
#   bash scripts/run_machine_a_step2_5.sh
#
# 可用環境變數略過某些階段（除錯用）：
#   SKIP_STEP2=1  SKIP_STEP5=1  SKIP_BUNDLE=1

set -euo pipefail

# 切到專案根目錄（此腳本位於 scripts/）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/artifacts/final_v1/logs"
mkdir -p "$LOG_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

banner() {
  echo ""
  echo "============================================================"
  echo " $1  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "============================================================"
}

run_step() {
  local name="$1"; shift
  local log="$LOG_DIR/${RUN_TS}_${name}.log"
  banner "$name"
  echo " log -> $log"
  # 同時輸出到終端與 log
  "$@" 2>&1 | tee "$log"
}

# ---------------------------------------------------------------
# 0. Preflight（純驗證，不啟動昂貴工作；失敗即停止）
# ---------------------------------------------------------------
run_step "preflight" uv run python src/preflight_final_v1.py

# ---------------------------------------------------------------
# Step 2：完整 FinBERT classification（full rerun，寫入 final_v1）
# ---------------------------------------------------------------
if [[ "${SKIP_STEP2:-0}" == "1" ]]; then
  banner "Step 2 SKIPPED (SKIP_STEP2=1)"
else
  run_step "step2_classification" uv run python src/step2_classification/run.py
fi

# ---------------------------------------------------------------
# Step 3：不重跑（人工核定已整合進 final_v1 ground truth）
# ---------------------------------------------------------------
banner "Step 3 SKIPPED (ground truth 已含 Label Studio 60 筆核定結果)"

# ---------------------------------------------------------------
# Step 4：group-aware 3-fold，一次建立 base / balance / combined
# ---------------------------------------------------------------
run_step "step4_cv" uv run python src/step4_cv/run_final_v1.py

# ---------------------------------------------------------------
# Step 5：內容型 reasoning cache + XML SFT build（base -> balance -> combined）
# ---------------------------------------------------------------
if [[ "${SKIP_STEP5:-0}" == "1" ]]; then
  banner "Step 5 SKIPPED (SKIP_STEP5=1)"
else
  run_step "step5_reasoning_sft" uv run python src/step5_reasoning/run_final_v1.py
fi

# ---------------------------------------------------------------
# Step 6 輸入 bundle：驗證 9 個 SFT fold 並打包給 Machine B
# ---------------------------------------------------------------
if [[ "${SKIP_BUNDLE:-0}" == "1" ]]; then
  banner "Bundle SKIPPED (SKIP_BUNDLE=1)"
else
  run_step "prepare_step6_bundle" uv run python src/step6_cv/prepare_step6_bundle.py
fi

banner "DONE：Step 2-5 完成"
cat <<'EOF'

下一步（Step 6，需 GPU、可分兩台機器）：
  Machine A（本機，gemma + llama）：
    確認 config step6_cv.job_matrix.active_machine: machine_a
    uv run python src/step6_cv/run_step6_jobs.py

  Machine B（qwen + ministral）：
    取得 artifacts/final_v1/step6_input_bundle.zip 並驗證 SHA-256
    將 active_machine 改為 machine_b
    uv run python src/step6_cv/run_step6_jobs.py

  收回結果後（在 Machine A）：
    uv run python src/step6_cv/collect_step6_results.py
    uv run python src/step6_cv/evaluate_all_experiments.py
EOF
