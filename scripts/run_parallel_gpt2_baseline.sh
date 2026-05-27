#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?Usage: $0 <upv|pdw> [limit] [output_base] [gpus_csv]}"
LIMIT="${2:-1000}"
OUTPUT_BASE="${3:-${PVMark_RESULT_DIR:-tests/baseline_comparison}}"
GPUS_CSV="${4:-0,1,2,3}"
ENV_NAME="${ENV_NAME:-${PVMark_BASELINE_ENV:-pvmark_baseline}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
NUM_SHARDS="${#GPUS[@]}"

case "${METHOD}" in
  upv)
    METHOD_OUT="upv_gpt2"
    EXPERIMENT="${SCRIPT_DIR}/upv_experiment.py"
    ATTACK_DETECT="${SCRIPT_DIR}/upv_detect_attacks.py"
    ;;
  pdw)
    METHOD_OUT="pdw_gpt2"
    EXPERIMENT="${SCRIPT_DIR}/pdw_experiment.py"
    ATTACK_DETECT="${SCRIPT_DIR}/pdw_detect_attacks.py"
    ;;
  *)
    echo "Unknown method: ${METHOD}. Expected upv or pdw." >&2
    exit 2
    ;;
esac

FINAL_OUT="${OUTPUT_BASE}/${METHOD_OUT}"
SHARD_ROOT="${FINAL_OUT}_shards"
mkdir -p "${SHARD_ROOT}/logs"

EXTRA_EXPERIMENT_ARGS=()
if [[ "${METHOD}" == "pdw" ]]; then
  PDW_KEY_DIR="${FINAL_OUT}_pdw_shared_key"
  PYTHONDONTWRITEBYTECODE=1 conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/pdw_prepare_key.py" \
    --key-dir "${PDW_KEY_DIR}"
  EXTRA_EXPERIMENT_ARGS+=(--key-dir "${PDW_KEY_DIR}")
fi

run_shard() {
  local shard_index="$1"
  local gpu_id="$2"
  local shard_dir="${SHARD_ROOT}/shard_$(printf "%02d" "${shard_index}")"
  local log_file="${SHARD_ROOT}/logs/shard_$(printf "%02d" "${shard_index}").log"
  mkdir -p "${shard_dir}"
  {
    echo "[$(date -Is)] shard=${shard_index}/${NUM_SHARDS} gpu=${gpu_id} output=${shard_dir}"
    export PYTHONDONTWRITEBYTECODE=1
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    conda run -n "${ENV_NAME}" python "${EXPERIMENT}" \
      --mode full \
      --limit "${LIMIT}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard_index}" \
      --output-dir "${shard_dir}" \
      "${EXTRA_EXPERIMENT_ARGS[@]}"

    conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/run_attacks.py" \
      --input "${shard_dir}/generations.json" \
      --output "${shard_dir}/attacks.json"

    conda run -n "${ENV_NAME}" python "${ATTACK_DETECT}" \
      --input "${shard_dir}/attacks.json" \
      --output "${shard_dir}/attack_detection.json"

    conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/run_ppl.py" \
      --input "${shard_dir}/generations.json" \
      --output "${shard_dir}/ppl.json" \
      --text-key completion_text

    conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/run_ppl.py" \
      --input "${shard_dir}/attacks.json" \
      --output "${shard_dir}/attack_ppl.json" \
      --text-key missing

    echo "[$(date -Is)] shard=${shard_index} done"
  } >"${log_file}" 2>&1
}

pids=()
for shard_index in "${!GPUS[@]}"; do
  run_shard "${shard_index}" "${GPUS[$shard_index]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "One or more shards failed. Check logs under ${SHARD_ROOT}/logs. Re-run the same command to resume completed shard steps." >&2
  exit 1
fi

PYTHONDONTWRITEBYTECODE=1 conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/merge_shards.py" \
  --shard-root "${SHARD_ROOT}" \
  --output-dir "${FINAL_OUT}" \
  --num-shards "${NUM_SHARDS}"

echo "Finished parallel ${METHOD} GPT2 baseline run: ${FINAL_OUT}"
