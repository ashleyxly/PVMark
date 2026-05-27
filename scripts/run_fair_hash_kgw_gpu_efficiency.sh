#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-2}"
DEVICE="${DEVICE:-cuda:${GPU}}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PVMARK_ROOT="${PVMARK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PVMARK_ROOT}/reproduction_outputs/fair_hash_kgw_gpu_${STAMP}}"
MODEL="${MODEL:-${GEN_MODEL:-facebook/opt-1.3b}}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
WET_MAX_SAMPLES="${WET_MAX_SAMPLES:-${MAX_SAMPLES}}"
WDT_MAX_SAMPLES="${WDT_MAX_SAMPLES:-20}"
WET_TOKEN_COUNT="${WET_TOKEN_COUNT:-200}"
WDT_TOKEN_COUNTS="${WDT_TOKEN_COUNTS:-50,200}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-1}"
REPEAT="${REPEAT:-1}"
HASH_METHODS="${HASH_METHODS:-2,4}"
HASH_TYPES="${HASH_TYPES:-3,4,5}"
RUST_ID_HASH_TYPES="${RUST_ID_HASH_TYPES:-3,5}"
POSEIDON_GPU_HASH_TYPES="${POSEIDON_GPU_HASH_TYPES:-3}"
MIMC_GPU_HASH_TYPES="${MIMC_GPU_HASH_TYPES:-5}"
BASELINE_CONDA_PREFIX="${BASELINE_CONDA_PREFIX:-${ENV_PREFIX:-${PVMARK_ROOT}/.venv_wm_baseline}}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif [[ -x "${BASELINE_CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_CMD=("${BASELINE_CONDA_PREFIX}/bin/python")
elif [[ -d "${BASELINE_CONDA_PREFIX}" ]]; then
  PYTHON_CMD=(conda run --no-capture-output -p "${BASELINE_CONDA_PREFIX}" python)
else
  PYTHON_CMD=(python)
fi

COMMON_ARGS=(
  --model "${MODEL}"
  --device "${DEVICE}"
  --require-cuda
  --max-samples "${MAX_SAMPLES}"
  --wet-max-samples "${WET_MAX_SAMPLES}"
  --wdt-max-samples "${WDT_MAX_SAMPLES}"
  --wet-token-count "${WET_TOKEN_COUNT}"
  --wdt-token-counts "${WDT_TOKEN_COUNTS}"
  --warmup-samples "${WARMUP_SAMPLES}"
  --repeat "${REPEAT}"
)

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON_CMD[@]}" -B -m baseline_eval.benchmark_efficiency \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/original_kgw" \
  --schemes original

"${PYTHON_CMD[@]}" -B -m baseline_eval.benchmark_efficiency \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/hash_cpu_u32" \
  --schemes hash \
  --hash-types "${HASH_TYPES}" \
  --hash-methods "${HASH_METHODS}" \
  --hash-wet-backend cpu-u32 \
  --hash-result-label-suffix cpu-u32

"${PYTHON_CMD[@]}" -B -m baseline_eval.benchmark_efficiency \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/hash_poseidon2_gpu_native_fused" \
  --schemes hash \
  --hash-types 4 \
  --hash-methods "${HASH_METHODS}" \
  --hash-wet-backend poseidon2-gpu-native-fused \
  --hash-result-label-suffix gpu-native-fused

"${PYTHON_CMD[@]}" -B -m baseline_eval.benchmark_efficiency \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/hash_poseidon2_gpu_mask_prefill" \
  --schemes hash \
  --hash-types 4 \
  --hash-methods "${HASH_METHODS}" \
  --hash-wet-backend poseidon2-gpu-mask-cache \
  --hash-result-label-suffix gpu-mask-prefill \
  --prefill-hash-wet-cache

"${PYTHON_CMD[@]}" -B -m baseline_eval.benchmark_efficiency \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/hash_poseidon2_gpu_id_prefill" \
  --schemes hash \
  --hash-types 4 \
  --hash-methods "${HASH_METHODS}" \
  --hash-wet-backend poseidon2-gpu-id-cache \
  --hash-result-label-suffix gpu-id-prefill \
  --prefill-hash-wet-cache

"${PYTHON_CMD[@]}" -B -m baseline_eval.benchmark_efficiency \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/hash_poseidon_gpu_id_prefill" \
  --schemes hash \
  --hash-types "${POSEIDON_GPU_HASH_TYPES}" \
  --hash-methods "${HASH_METHODS}" \
  --hash-wet-backend poseidon-gpu-id-cache \
  --hash-result-label-suffix poseidon-gpu-id-prefill \
  --prefill-hash-wet-cache

"${PYTHON_CMD[@]}" -B -m baseline_eval.benchmark_efficiency \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/hash_mimc_gpu_id_prefill" \
  --schemes hash \
  --hash-types "${MIMC_GPU_HASH_TYPES}" \
  --hash-methods "${HASH_METHODS}" \
  --hash-wet-backend mimc-gpu-id-cache \
  --hash-result-label-suffix mimc-gpu-id-prefill \
  --prefill-hash-wet-cache

"${PYTHON_CMD[@]}" -B -m baseline_eval.benchmark_efficiency \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/hash_poseidon_mimc_rust_id_prefill" \
  --schemes hash \
  --hash-types "${RUST_ID_HASH_TYPES}" \
  --hash-methods "${HASH_METHODS}" \
  --hash-wet-backend rust-id-cache \
  --hash-result-label-suffix rust-id-prefill \
  --prefill-hash-wet-cache

"${PYTHON_CMD[@]}" -B -m baseline_eval.merge_efficiency_results \
  --inputs \
    "${OUTPUT_ROOT}/original_kgw" \
    "${OUTPUT_ROOT}/hash_cpu_u32" \
    "${OUTPUT_ROOT}/hash_poseidon2_gpu_native_fused" \
    "${OUTPUT_ROOT}/hash_poseidon2_gpu_mask_prefill" \
    "${OUTPUT_ROOT}/hash_poseidon2_gpu_id_prefill" \
    "${OUTPUT_ROOT}/hash_poseidon_gpu_id_prefill" \
    "${OUTPUT_ROOT}/hash_mimc_gpu_id_prefill" \
    "${OUTPUT_ROOT}/hash_poseidon_mimc_rust_id_prefill" \
  --output-dir "${OUTPUT_ROOT}/combined"

printf 'Wrote combined result: %s\n' "${OUTPUT_ROOT}/combined/efficiency_results.md"
