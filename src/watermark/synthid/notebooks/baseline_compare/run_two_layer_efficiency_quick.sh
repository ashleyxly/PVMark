#!/usr/bin/env bash
set -euo pipefail

ROOT="${PVMark_SYNTHID_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT_ROOT="${1:-tests/baseline_comparison/two_layer_efficiency_2026-05-22}"
GPU_ID="${GPU_ID:-2}"
CPUSET="${CPUSET:-36-71,108-143}"
ENV_NAME="${ENV_NAME:-${PVMark_BASELINE_ENV:-pvmark_baseline}}"
CORE_WET_RUNS="${CORE_WET_RUNS:-5}"
CORE_WDT_RUNS="${CORE_WDT_RUNS:-20}"
CORE_WARMUP_RUNS="${CORE_WARMUP_RUNS:-2}"
E2E_LIMIT="${E2E_LIMIT:-3}"
UPV_LIMIT="${UPV_LIMIT:-5}"
PDW_LIMIT="${PDW_LIMIT:-3}"

cd "$ROOT"
mkdir -p "$OUT_ROOT/logs"

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

run_step() {
  local name="$1"
  shift
  echo "[$(date -Is)] START $name"
  taskset -c "$CPUSET" "$@" 2>&1 | tee "$OUT_ROOT/logs/${name}.log"
  echo "[$(date -Is)] DONE $name"
}

run_step core_original_synthid \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_original_synthid_efficiency.py \
    --output-dir "$OUT_ROOT/core/original_synthid_lcg" \
    --backend original-python \
    --device cuda \
    --token-lengths 200 \
    --wet-runs "$CORE_WET_RUNS" \
    --wdt-runs "$CORE_WDT_RUNS" \
    --warmup-runs "$CORE_WARMUP_RUNS" \
    --progress-every 1 \
    --cache-mode warm \
    --score-type weighted_mean

run_step core_pvmark_poseidon_t3 \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_hash_synthid_efficiency.py \
    --output-dir "$OUT_ROOT/core/pvmark_poseidon_t3" \
    --hash-type 3 \
    --device cuda \
    --token-lengths 200 \
    --wet-runs "$CORE_WET_RUNS" \
    --wdt-runs "$CORE_WDT_RUNS" \
    --warmup-runs "$CORE_WARMUP_RUNS" \
    --progress-every 1 \
    --cache-mode warm \
    --score-type weighted_mean

run_step core_pvmark_poseidon2_t4 \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_hash_synthid_efficiency.py \
    --output-dir "$OUT_ROOT/core/pvmark_poseidon2_t4" \
    --hash-type 4 \
    --device cuda \
    --token-lengths 200 \
    --wet-runs "$CORE_WET_RUNS" \
    --wdt-runs "$CORE_WDT_RUNS" \
    --warmup-runs "$CORE_WARMUP_RUNS" \
    --progress-every 1 \
    --cache-mode warm \
    --score-type weighted_mean

run_step core_pvmark_mimc_t5 \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_hash_synthid_efficiency.py \
    --output-dir "$OUT_ROOT/core/pvmark_mimc_t5" \
    --hash-type 5 \
    --device cuda \
    --token-lengths 200 \
    --wet-runs "$CORE_WET_RUNS" \
    --wdt-runs "$CORE_WDT_RUNS" \
    --warmup-runs "$CORE_WARMUP_RUNS" \
    --progress-every 1 \
    --cache-mode warm \
    --score-type weighted_mean

run_step core_upv_strict_batch1 \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_upv_wet_wdt.py \
    --output-dir "$OUT_ROOT/core/upv_network_strict_batch1" \
    --checkpoint tests/baseline_comparison/upv_network_detector_gpt2_eli5/detector_z1.pt \
    --generations tests/baseline_comparison/upv_gpt2/generations.json \
    --attacks tests/baseline_comparison/upv_gpt2/attacks.json \
    --device cuda \
    --num-samples 20 \
    --wet-token-length 200 \
    --wdt-token-length 200 \
    --wet-mode strict_sequential \
    --wet-runs 20 \
    --wdt-runs 20 \
    --eval-batch-size 1

run_step e2e_original_synthid \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_synthid_e2e_generation_detection.py \
    --output-dir "$OUT_ROOT/e2e/original_synthid_lcg" \
    --scheme original \
    --device cuda \
    --limit "$E2E_LIMIT" \
    --max-new-tokens 200 \
    --score-type weighted_mean

run_step e2e_pvmark_poseidon_t3 \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_synthid_e2e_generation_detection.py \
    --output-dir "$OUT_ROOT/e2e/pvmark_poseidon_t3" \
    --scheme hash \
    --hash-type 3 \
    --device cuda \
    --limit "$E2E_LIMIT" \
    --max-new-tokens 200 \
    --score-type weighted_mean

run_step e2e_pvmark_poseidon2_t4 \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_synthid_e2e_generation_detection.py \
    --output-dir "$OUT_ROOT/e2e/pvmark_poseidon2_t4" \
    --scheme hash \
    --hash-type 4 \
    --device cuda \
    --limit "$E2E_LIMIT" \
    --max-new-tokens 200 \
    --score-type weighted_mean

run_step e2e_pvmark_mimc_t5 \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_synthid_e2e_generation_detection.py \
    --output-dir "$OUT_ROOT/e2e/pvmark_mimc_t5" \
    --scheme hash \
    --hash-type 5 \
    --device cuda \
    --limit "$E2E_LIMIT" \
    --max-new-tokens 200 \
    --score-type weighted_mean

run_step e2e_upv_public \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/upv_experiment.py \
    --mode full \
    --output-dir "$OUT_ROOT/e2e/upv_public" \
    --limit "$UPV_LIMIT" \
    --max-new-tokens 200 \
    --no-resume

run_step e2e_pdw \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/pdw_experiment.py \
    --mode full \
    --output-dir "$OUT_ROOT/e2e/pdw" \
    --limit "$PDW_LIMIT" \
    --num-tokens 200 \
    --key-mode shared \
    --no-resume

run_step e2e_pdw_summary \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_pdw_efficiency_from_records.py \
    --generations "$OUT_ROOT/e2e/pdw/generations.json" \
    --detection "$OUT_ROOT/e2e/pdw/detection.json" \
    --output "$OUT_ROOT/e2e/pdw/pdw_efficiency_from_records.json"

echo "[$(date -Is)] TWO_LAYER_EFFICIENCY_QUICK_COMPLETE $OUT_ROOT"
