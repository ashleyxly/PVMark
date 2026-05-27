#!/usr/bin/env bash
set -euo pipefail

PVMARK_ROOT="${PVMARK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_PREFIX="${ENV_PREFIX:-${PVMARK_ROOT}/.venv_wm_baseline}"
RESULTS_ROOT="${RESULTS_ROOT:-${PVMARK_ROOT}/reproduction_outputs/baseline_results}"
DATASET="${DATASET:-${PVMARK_ROOT}/experiment_data/prompts/num_100.json}"
GEN_MODEL="${GEN_MODEL:-facebook/opt-1.3b}"
PPL_MODEL="${PPL_MODEL:-facebook/opt-2.7b}"
BERT_MODEL="${BERT_MODEL:-bert-base-uncased}"
GPU="${GPU:-0}"
SEED="${SEED:-20242024}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RUN_TAG="${RUN_TAG:-opt1.3b_c4_num100}"
RUN_UPV="${RUN_UPV:-1}"
RUN_PDW="${RUN_PDW:-1}"
RUN_ATTACKS="${RUN_ATTACKS:-1}"
RUN_PPL="${RUN_PPL:-1}"

# Leave empty for formal experiments to use the original PDW default timeout.
# Set to 30 only for quick debugging:
#   PDW_MAX_TIME_BEFORE_PLANT_ERROR=30 MAX_SAMPLES=1 ...
PDW_MAX_TIME_BEFORE_PLANT_ERROR="${PDW_MAX_TIME_BEFORE_PLANT_ERROR:-}"
PDW_PROMPT_MAX_LENGTH="${PDW_PROMPT_MAX_LENGTH:-1024}"

export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

PYTHON=(conda run --no-capture-output -p "$ENV_PREFIX" python -B)

COMMON_ARGS=(
  --dataset "$DATASET"
  --model "$GEN_MODEL"
  --seed "$SEED"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  COMMON_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

UPV_DIR="$RESULTS_ROOT/unforgeable/$RUN_TAG"
PDW_DIR="$RESULTS_ROOT/publicly_detectable/$RUN_TAG"
LOG_ROOT="$RESULTS_ROOT/logs/$RUN_TAG"
mkdir -p "$UPV_DIR" "$PDW_DIR" "$LOG_ROOT"

run_step() {
  local name="$1"
  shift
  local log_file="$LOG_ROOT/${name}.log"
  echo "[$(date '+%F %T')] START $name"
  echo "log: $log_file"
  "$@" 2>&1 | tee "$log_file"
  echo "[$(date '+%F %T')] DONE  $name"
}

echo "Environment prefix: $ENV_PREFIX"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Dataset: $DATASET"
echo "Generation model: $GEN_MODEL"
echo "PPL model: $PPL_MODEL"
echo "Results root: $RESULTS_ROOT"
echo "Run tag: $RUN_TAG"
echo "MAX_SAMPLES: ${MAX_SAMPLES:-all}"
echo

run_step "00_gpu_check" "${PYTHON[@]}" -c \
  "import torch; print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

if [[ "$RUN_UPV" == "1" ]]; then
  run_step "01_unforgeable_generation" "${PYTHON[@]}" -m baseline_eval.run_unforgeable \
    "${COMMON_ARGS[@]}" \
    --output-dir "$UPV_DIR"

  if [[ "$RUN_ATTACKS" == "1" ]]; then
    run_step "02_unforgeable_attacks" "${PYTHON[@]}" -m baseline_eval.run_attacks \
      --input "$UPV_DIR/generations.json" \
      --bert-model "$BERT_MODEL" \
      --seed "$SEED"

    run_step "03_unforgeable_robustness_detection" "${PYTHON[@]}" -m baseline_eval.run_robustness_detection \
      --generations "$UPV_DIR/generations.json" \
      --attacks "$UPV_DIR/attacks.json"
  fi

  if [[ "$RUN_PPL" == "1" ]]; then
    PPL_ARGS=(--generations "$UPV_DIR/generations.json" --ppl-model "$PPL_MODEL")
    if [[ -f "$UPV_DIR/attacks.json" ]]; then
      PPL_ARGS+=(--attacks "$UPV_DIR/attacks.json")
    fi
    run_step "04_unforgeable_ppl" "${PYTHON[@]}" -m baseline_eval.run_ppl "${PPL_ARGS[@]}"
  fi

  run_step "05_unforgeable_summary" "${PYTHON[@]}" -m baseline_eval.summarize \
    --run-dir "$UPV_DIR" \
    --output-csv "$UPV_DIR/summary.csv"
fi

if [[ "$RUN_PDW" == "1" ]]; then
  PDW_ARGS=(
    "${COMMON_ARGS[@]}"
    --output-dir "$PDW_DIR"
    --prompt-max-length "$PDW_PROMPT_MAX_LENGTH"
  )
  if [[ -n "$PDW_MAX_TIME_BEFORE_PLANT_ERROR" ]]; then
    PDW_ARGS+=(--max-time-before-plant-error "$PDW_MAX_TIME_BEFORE_PLANT_ERROR")
  fi

  run_step "06_publicly_detectable_generation" "${PYTHON[@]}" -m baseline_eval.run_publicly_detectable \
    "${PDW_ARGS[@]}"

  if [[ "$RUN_ATTACKS" == "1" ]]; then
    run_step "07_publicly_detectable_attacks" "${PYTHON[@]}" -m baseline_eval.run_attacks \
      --input "$PDW_DIR/generations.json" \
      --bert-model "$BERT_MODEL" \
      --seed "$SEED"

    run_step "08_publicly_detectable_robustness_detection" "${PYTHON[@]}" -m baseline_eval.run_robustness_detection \
      --generations "$PDW_DIR/generations.json" \
      --attacks "$PDW_DIR/attacks.json"
  fi

  if [[ "$RUN_PPL" == "1" ]]; then
    PPL_ARGS=(--generations "$PDW_DIR/generations.json" --ppl-model "$PPL_MODEL")
    if [[ -f "$PDW_DIR/attacks.json" ]]; then
      PPL_ARGS+=(--attacks "$PDW_DIR/attacks.json")
    fi
    run_step "09_publicly_detectable_ppl" "${PYTHON[@]}" -m baseline_eval.run_ppl "${PPL_ARGS[@]}"
  fi

  run_step "10_publicly_detectable_summary" "${PYTHON[@]}" -m baseline_eval.summarize \
    --run-dir "$PDW_DIR" \
    --output-csv "$PDW_DIR/summary.csv"
fi

echo
echo "All requested steps finished."
echo "UPV dir: $UPV_DIR"
echo "PDW dir: $PDW_DIR"
echo "Logs:    $LOG_ROOT"
