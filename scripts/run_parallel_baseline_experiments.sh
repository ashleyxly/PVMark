#!/usr/bin/env bash
set -uo pipefail

PVMARK_ROOT="${PVMARK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_PREFIX="${ENV_PREFIX:-${PVMARK_ROOT}/.venv_wm_baseline}"
RESULTS_ROOT="${RESULTS_ROOT:-${PVMARK_ROOT}/reproduction_outputs/baseline_results}"
DATASET="${DATASET:-${PVMARK_ROOT}/experiment_data/prompts/num_100.json}"
GEN_MODEL="${GEN_MODEL:-facebook/opt-1.3b}"
PPL_MODEL="${PPL_MODEL:-facebook/opt-2.7b}"
BERT_MODEL="${BERT_MODEL:-bert-base-uncased}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6}"
SEED="${SEED:-20242024}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RUN_TAG="${RUN_TAG:-opt1.3b_c4_num100_parallel7}"
RUN_UPV="${RUN_UPV:-1}"
RUN_UPV_NETWORK="${RUN_UPV_NETWORK:-0}"
RUN_PDW="${RUN_PDW:-1}"
RUN_ATTACKS="${RUN_ATTACKS:-1}"
RUN_ROBUSTNESS="${RUN_ROBUSTNESS:-1}"
RUN_PPL="${RUN_PPL:-1}"
GEN_LOAD_FP16="${GEN_LOAD_FP16:-0}"
PPL_LOAD_FP16="${PPL_LOAD_FP16:-0}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-8}"
PDW_MAX_TIME_BEFORE_PLANT_ERROR="${PDW_MAX_TIME_BEFORE_PLANT_ERROR:-}"
PDW_PROMPT_MAX_LENGTH="${PDW_PROMPT_MAX_LENGTH:-1024}"
UPV_NETWORK_DETECTOR_MODEL="${UPV_NETWORK_DETECTOR_MODEL:-${RESULTS_ROOT}/unforgeable_network/trained_components/upv_network_opt13b_repo_detector_full/network_detector.pt}"
UPV_NETWORK_THRESHOLD="${UPV_NETWORK_THRESHOLD:-0.5}"
UPV_NETWORK_DETECTOR_FIXED_LENGTH="${UPV_NETWORK_DETECTOR_FIXED_LENGTH:-200}"
RETRY_FAILURES="${RETRY_FAILURES:-0}"
FAIL_FAST="${FAIL_FAST:-0}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi
NUM_SHARDS="${NUM_SHARDS:-${#GPUS[@]}}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

PYTHON=(conda run --no-capture-output -p "$ENV_PREFIX" python -B)

RESUME_ARGS=()
if [[ "$RETRY_FAILURES" == "1" ]]; then
  RESUME_ARGS+=(--retry-failures)
fi
if [[ "$FAIL_FAST" == "1" ]]; then
  RESUME_ARGS+=(--fail-fast)
fi

UPV_DIR="$RESULTS_ROOT/unforgeable/$RUN_TAG"
UPV_NETWORK_DIR="$RESULTS_ROOT/unforgeable_network/$RUN_TAG"
PDW_DIR="$RESULTS_ROOT/publicly_detectable/$RUN_TAG"
SHARD_ROOT="$RESULTS_ROOT/shards/$RUN_TAG"
LOG_ROOT="$RESULTS_ROOT/logs/$RUN_TAG"
mkdir -p "$UPV_DIR" "$UPV_NETWORK_DIR" "$PDW_DIR" "$SHARD_ROOT" "$LOG_ROOT"

COMMON_ARGS=(
  --dataset "$DATASET"
  --model "$GEN_MODEL"
  --seed "$SEED"
  --num-shards "$NUM_SHARDS"
  "${RESUME_ARGS[@]}"
)
if [[ -n "$MAX_SAMPLES" ]]; then
  COMMON_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "$GEN_LOAD_FP16" == "1" ]]; then
  COMMON_ARGS+=(--load-fp16)
fi

FAILED=0

gpu_for_shard() {
  local shard="$1"
  echo "${GPUS[$((shard % ${#GPUS[@]}))]}"
}

wait_for_stage() {
  local stage="$1"
  shift
  local pids=("$@")
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" == "1" ]]; then
    echo "[$(date '+%F %T')] WARN stage failed: $stage. Merging available checkpoints and continuing." >&2
    FAILED=1
  else
    echo "[$(date '+%F %T')] DONE $stage"
  fi
}

run_sharded_generation() {
  local scheme="$1"
  local module="$2"
  local output_dir="$3"
  local shard_dir="$SHARD_ROOT/$scheme/generation"
  mkdir -p "$shard_dir"

  local -a shard_dirs=()
  local -a pids=()
  echo "[$(date '+%F %T')] START ${scheme} generation with ${NUM_SHARDS} shards on GPUs ${GPU_LIST}"
  for ((i = 0; i < NUM_SHARDS; i++)); do
    local gpu
    gpu="$(gpu_for_shard "$i")"
    local one_shard="$shard_dir/shard_$i"
    local log_file="$LOG_ROOT/${scheme}_generation_shard_${i}_gpu_${gpu}.log"
    mkdir -p "$one_shard"
    shard_dirs+=("$one_shard")
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      args=("${COMMON_ARGS[@]}" --shard-index "$i" --output-dir "$one_shard")
      if [[ "$scheme" == "publicly_detectable" ]]; then
        args+=(--key-dir "$output_dir/keys" --prompt-max-length "$PDW_PROMPT_MAX_LENGTH")
        if [[ -n "$PDW_MAX_TIME_BEFORE_PLANT_ERROR" ]]; then
          args+=(--max-time-before-plant-error "$PDW_MAX_TIME_BEFORE_PLANT_ERROR")
        fi
      elif [[ "$scheme" == "unforgeable_network" ]]; then
        args+=(
          --network-detector-model "$UPV_NETWORK_DETECTOR_MODEL"
          --network-threshold "$UPV_NETWORK_THRESHOLD"
          --detector-fixed-length "$UPV_NETWORK_DETECTOR_FIXED_LENGTH"
          --device cuda
        )
      fi
      "${PYTHON[@]}" -m "$module" "${args[@]}"
    ) >"$log_file" 2>&1 &
    pids+=("$!")
    echo "  shard $i -> GPU $gpu, log $log_file"
  done
  wait_for_stage "${scheme} generation" "${pids[@]}"
  if ! "${PYTHON[@]}" -m baseline_eval.merge_shards generations \
    --scheme "$scheme" \
    --output-dir "$output_dir" \
    --inputs "${shard_dirs[@]}"; then
    echo "[$(date '+%F %T')] WARN ${scheme} generation merge failed" >&2
    FAILED=1
  fi
}

run_sharded_attacks() {
  local scheme="$1"
  local run_dir="$2"
  local shard_dir="$SHARD_ROOT/$scheme/attacks"
  mkdir -p "$shard_dir"

  local -a shard_dirs=()
  local -a pids=()
  echo "[$(date '+%F %T')] START ${scheme} attacks with ${NUM_SHARDS} shards"
  for ((i = 0; i < NUM_SHARDS; i++)); do
    local gpu
    gpu="$(gpu_for_shard "$i")"
    local one_shard="$shard_dir/shard_$i"
    local log_file="$LOG_ROOT/${scheme}_attacks_shard_${i}_gpu_${gpu}.log"
    mkdir -p "$one_shard"
    shard_dirs+=("$one_shard")
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "${PYTHON[@]}" -m baseline_eval.run_attacks \
        --input "$run_dir/generations.json" \
        --output "$one_shard/attacks.json" \
        --bert-model "$BERT_MODEL" \
        --seed "$SEED" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$i" \
        "${RESUME_ARGS[@]}"
    ) >"$log_file" 2>&1 &
    pids+=("$!")
    echo "  shard $i -> GPU $gpu, log $log_file"
  done
  wait_for_stage "${scheme} attacks" "${pids[@]}"
  if ! "${PYTHON[@]}" -m baseline_eval.merge_shards attacks \
    --output "$run_dir/attacks.json" \
    --inputs "${shard_dirs[@]}"; then
    echo "[$(date '+%F %T')] WARN ${scheme} attack merge failed" >&2
    FAILED=1
  fi
}

run_sharded_robustness() {
  local scheme="$1"
  local run_dir="$2"
  local shard_dir="$SHARD_ROOT/$scheme/robustness"
  mkdir -p "$shard_dir"

  local -a shard_dirs=()
  local -a pids=()
  echo "[$(date '+%F %T')] START ${scheme} robustness detection with ${NUM_SHARDS} shards"
  for ((i = 0; i < NUM_SHARDS; i++)); do
    local gpu
    gpu="$(gpu_for_shard "$i")"
    local one_shard="$shard_dir/shard_$i"
    local log_file="$LOG_ROOT/${scheme}_robustness_shard_${i}_gpu_${gpu}.log"
    mkdir -p "$one_shard"
    shard_dirs+=("$one_shard")
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "${PYTHON[@]}" -m baseline_eval.run_robustness_detection \
        --generations "$run_dir/generations.json" \
        --attacks "$run_dir/attacks.json" \
        --output "$one_shard/robustness_detection.json" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$i" \
        "${RESUME_ARGS[@]}"
    ) >"$log_file" 2>&1 &
    pids+=("$!")
    echo "  shard $i -> GPU $gpu, log $log_file"
  done
  wait_for_stage "${scheme} robustness detection" "${pids[@]}"
  if ! "${PYTHON[@]}" -m baseline_eval.merge_shards robustness \
    --output "$run_dir/robustness_detection.json" \
    --inputs "${shard_dirs[@]}"; then
    echo "[$(date '+%F %T')] WARN ${scheme} robustness merge failed" >&2
    FAILED=1
  fi
}

run_sharded_ppl() {
  local scheme="$1"
  local run_dir="$2"
  local shard_dir="$SHARD_ROOT/$scheme/ppl"
  mkdir -p "$shard_dir"

  local -a shard_dirs=()
  local -a pids=()
  echo "[$(date '+%F %T')] START ${scheme} PPL with ${NUM_SHARDS} shards"
  for ((i = 0; i < NUM_SHARDS; i++)); do
    local gpu
    gpu="$(gpu_for_shard "$i")"
    local one_shard="$shard_dir/shard_$i"
    local log_file="$LOG_ROOT/${scheme}_ppl_shard_${i}_gpu_${gpu}.log"
    mkdir -p "$one_shard"
    shard_dirs+=("$one_shard")
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      args=(
        -m baseline_eval.run_ppl
        --generations "$run_dir/generations.json"
        --output "$one_shard/ppl.json"
        --ppl-model "$PPL_MODEL"
        --batch-size "$PPL_BATCH_SIZE"
        --device cuda
        --num-shards "$NUM_SHARDS"
        --shard-index "$i"
        "${RESUME_ARGS[@]}"
      )
      if [[ -f "$run_dir/attacks.json" ]]; then
        args+=(--attacks "$run_dir/attacks.json")
      fi
      if [[ "$PPL_LOAD_FP16" == "1" ]]; then
        args+=(--load-fp16)
      fi
      "${PYTHON[@]}" "${args[@]}"
    ) >"$log_file" 2>&1 &
    pids+=("$!")
    echo "  shard $i -> GPU $gpu, log $log_file"
  done
  wait_for_stage "${scheme} PPL" "${pids[@]}"
  if ! "${PYTHON[@]}" -m baseline_eval.merge_shards ppl \
    --output "$run_dir/ppl.json" \
    --inputs "${shard_dirs[@]}"; then
    echo "[$(date '+%F %T')] WARN ${scheme} PPL merge failed" >&2
    FAILED=1
  fi
}

run_summary() {
  local scheme="$1"
  local run_dir="$2"
  local log_file="$LOG_ROOT/${scheme}_summary.log"
  echo "[$(date '+%F %T')] START ${scheme} summary"
  if ! "${PYTHON[@]}" -m baseline_eval.summarize \
    --run-dir "$run_dir" \
    --output-csv "$run_dir/summary.csv" >"$log_file" 2>&1; then
    echo "[$(date '+%F %T')] WARN ${scheme} summary failed; see $log_file" >&2
    FAILED=1
  fi
}

echo "Environment prefix: $ENV_PREFIX"
echo "GPU_LIST: $GPU_LIST"
echo "NUM_SHARDS: $NUM_SHARDS"
echo "Dataset: $DATASET"
echo "Generation model: $GEN_MODEL"
echo "PPL model: $PPL_MODEL"
echo "Results root: $RESULTS_ROOT"
echo "Run tag: $RUN_TAG"
echo "MAX_SAMPLES: ${MAX_SAMPLES:-all}"
echo "RUN_UPV: $RUN_UPV"
echo "RUN_UPV_NETWORK: $RUN_UPV_NETWORK"
echo "UPV_NETWORK_DETECTOR_MODEL: $UPV_NETWORK_DETECTOR_MODEL"
echo "UPV_NETWORK_THRESHOLD: $UPV_NETWORK_THRESHOLD"
echo "UPV_NETWORK_DETECTOR_FIXED_LENGTH: $UPV_NETWORK_DETECTOR_FIXED_LENGTH"
echo "RUN_PDW: $RUN_PDW"
echo "PDW_PROMPT_MAX_LENGTH: $PDW_PROMPT_MAX_LENGTH"
echo "PDW_MAX_TIME_BEFORE_PLANT_ERROR: ${PDW_MAX_TIME_BEFORE_PLANT_ERROR:-default}"
echo "RETRY_FAILURES: $RETRY_FAILURES"
echo "FAIL_FAST: $FAIL_FAST"
echo "Logs: $LOG_ROOT"
echo

if [[ "$RUN_UPV" == "1" ]]; then
  run_sharded_generation "unforgeable" "baseline_eval.run_unforgeable" "$UPV_DIR"
  if [[ "$RUN_ATTACKS" == "1" ]]; then
    run_sharded_attacks "unforgeable" "$UPV_DIR"
    if [[ "$RUN_ROBUSTNESS" == "1" ]]; then
      run_sharded_robustness "unforgeable" "$UPV_DIR"
    fi
  fi
  if [[ "$RUN_PPL" == "1" ]]; then
    run_sharded_ppl "unforgeable" "$UPV_DIR"
  fi
  run_summary "unforgeable" "$UPV_DIR"
fi

if [[ "$RUN_UPV_NETWORK" == "1" ]]; then
  run_sharded_generation "unforgeable_network" "baseline_eval.run_unforgeable_network" "$UPV_NETWORK_DIR"
  if [[ "$RUN_ATTACKS" == "1" ]]; then
    run_sharded_attacks "unforgeable_network" "$UPV_NETWORK_DIR"
    if [[ "$RUN_ROBUSTNESS" == "1" ]]; then
      run_sharded_robustness "unforgeable_network" "$UPV_NETWORK_DIR"
    fi
  fi
  if [[ "$RUN_PPL" == "1" ]]; then
    run_sharded_ppl "unforgeable_network" "$UPV_NETWORK_DIR"
  fi
  run_summary "unforgeable_network" "$UPV_NETWORK_DIR"
fi

if [[ "$RUN_PDW" == "1" ]]; then
  run_sharded_generation "publicly_detectable" "baseline_eval.run_publicly_detectable" "$PDW_DIR"
  if [[ "$RUN_ATTACKS" == "1" ]]; then
    run_sharded_attacks "publicly_detectable" "$PDW_DIR"
    if [[ "$RUN_ROBUSTNESS" == "1" ]]; then
      run_sharded_robustness "publicly_detectable" "$PDW_DIR"
    fi
  fi
  if [[ "$RUN_PPL" == "1" ]]; then
    run_sharded_ppl "publicly_detectable" "$PDW_DIR"
  fi
  run_summary "publicly_detectable" "$PDW_DIR"
fi

echo
echo "Requested parallel experiment finished."
echo "UPV legacy/key-based dir: $UPV_DIR"
echo "UPV network-based dir: $UPV_NETWORK_DIR"
echo "PDW dir: $PDW_DIR"
echo "Shard checkpoints: $SHARD_ROOT"
echo "Logs: $LOG_ROOT"

exit "$FAILED"
