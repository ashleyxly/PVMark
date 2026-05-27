#!/usr/bin/env bash
set -euo pipefail

PVMARK_ROOT="${PVMARK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_PREFIX="${ENV_PREFIX:-${PVMARK_ROOT}/.venv_wm_baseline}"
GPU="${GPU:-2}"
RESULTS_ROOT="${RESULTS_ROOT:-${PVMARK_ROOT}/reproduction_outputs/baseline_results}"
RUN_DIR="${RUN_DIR:-${RESULTS_ROOT}/kgw/opt1.3b_c4_num100_legacy_org}"
LEGACY_INPUT="${LEGACY_INPUT:-test_result/Org_scheme/Normal/C4_ModelName.OPT1_3B_Org_WM.json}"
BERT_MODEL="${BERT_MODEL:-bert-base-uncased}"
LOG_DIR="${LOG_DIR:-${RESULTS_ROOT}/logs/kgw_opt1.3b_c4_num100_legacy_org}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"

mkdir -p "$RUN_DIR" "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

PYTHON=(conda run --no-capture-output -p "$ENV_PREFIX" python -B)
RESUME_ARGS=()
if [[ "$FORCE_RECOMPUTE" == "1" ]]; then
  RESUME_ARGS+=(--no-resume)
fi

echo "KGW run dir: $RUN_DIR"
echo "Legacy input: $LEGACY_INPUT"
echo "GPU: $GPU"
echo "Logs: $LOG_DIR"
echo "Force recompute: $FORCE_RECOMPUTE"

"${PYTHON[@]}" -m baseline_eval.prepare_kgw_legacy \
  --input "$LEGACY_INPUT" \
  --output-dir "$RUN_DIR" \
  >"$LOG_DIR/prepare.log" 2>&1

"${PYTHON[@]}" -m baseline_eval.run_attacks \
  --input "$RUN_DIR/generations.json" \
  --output "$RUN_DIR/attacks.json" \
  --bert-model "$BERT_MODEL" \
  "${RESUME_ARGS[@]}" \
  >"$LOG_DIR/attacks.log" 2>&1

"${PYTHON[@]}" -m baseline_eval.run_robustness_detection \
  --generations "$RUN_DIR/generations.json" \
  --attacks "$RUN_DIR/attacks.json" \
  --output "$RUN_DIR/robustness_detection.json" \
  "${RESUME_ARGS[@]}" \
  >"$LOG_DIR/robustness.log" 2>&1

"${PYTHON[@]}" -m baseline_eval.summarize \
  --run-dir "$RUN_DIR" \
  --output-csv "$RUN_DIR/summary.csv" \
  >"$LOG_DIR/summary.log" 2>&1

echo "KGW robustness finished."
