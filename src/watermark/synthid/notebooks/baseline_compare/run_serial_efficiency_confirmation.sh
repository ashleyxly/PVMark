#!/usr/bin/env bash
set -euo pipefail

ROOT="${PVMark_SYNTHID_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT_ROOT="${1:-tests/baseline_comparison/serial_efficiency_confirmation_2026-05-21}"
GPU_ID="${GPU_ID:-0}"
CPUSET="${CPUSET:-0-35,72-107}"
ENV_NAME="${ENV_NAME:-${PVMark_BASELINE_ENV:-pvmark_baseline}}"

cd "$ROOT"
mkdir -p "$OUT_ROOT/logs"

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

run_step() {
  local name="$1"
  shift
  echo "[$(date -Is)] START $name"
  taskset -c "$CPUSET" "$@" 2>&1 | tee "$OUT_ROOT/logs/${name}.log"
  echo "[$(date -Is)] DONE $name"
}

run_step original_synthid \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_original_synthid_efficiency.py \
    --output-dir "$OUT_ROOT/original_synthid_lcg_warm_seq200" \
    --backend original-python \
    --device cuda \
    --token-lengths 200 \
    --wet-runs 100 \
    --wdt-runs 100 \
    --warmup-runs 10 \
    --progress-every 10 \
    --cache-mode warm \
    --score-type weighted_mean

run_step pvmark_poseidon_t3 \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_hash_synthid_efficiency.py \
    --output-dir "$OUT_ROOT/pvmark_poseidon_t3_warm_seq200" \
    --hash-type 3 \
    --device cuda \
    --token-lengths 200 \
    --wet-runs 100 \
    --wdt-runs 100 \
    --warmup-runs 10 \
    --progress-every 10 \
    --cache-mode warm \
    --score-type weighted_mean

run_step pvmark_poseidon2_t4 \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_hash_synthid_efficiency.py \
    --output-dir "$OUT_ROOT/pvmark_poseidon2_t4_warm_seq200" \
    --hash-type 4 \
    --device cuda \
    --token-lengths 200 \
    --wet-runs 100 \
    --wdt-runs 100 \
    --warmup-runs 10 \
    --progress-every 10 \
    --cache-mode warm \
    --score-type weighted_mean

run_step pvmark_mimc_t5 \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_hash_synthid_efficiency.py \
    --output-dir "$OUT_ROOT/pvmark_mimc_t5_warm_seq200" \
    --hash-type 5 \
    --device cuda \
    --token-lengths 200 \
    --wet-runs 100 \
    --wdt-runs 100 \
    --warmup-runs 10 \
    --progress-every 10 \
    --cache-mode warm \
    --score-type weighted_mean

run_step upv_network \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_upv_wet_wdt.py \
    --output-dir "$OUT_ROOT/upv_network_detector_gpt2_eli5_200" \
    --checkpoint tests/baseline_comparison/upv_network_detector_gpt2_eli5/detector_z1.pt \
    --generations tests/baseline_comparison/upv_gpt2/generations.json \
    --attacks tests/baseline_comparison/upv_gpt2/attacks.json \
    --device cuda \
    --wet-token-length 200 \
    --wdt-token-length 200 \
    --wet-runs 2000 \
    --wdt-runs 200

run_step pdw_from_records \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_pdw_efficiency_from_records.py \
    --generations tests/baseline_comparison/pdw_gpt2/generations.json \
    --detection tests/baseline_comparison/pdw_gpt2/detection.json \
    --output "$OUT_ROOT/pdw_gpt2/pdw_efficiency_from_records.json"

run_step audit \
  conda run -n "$ENV_NAME" python notebooks/baseline_compare/audit_efficiency_artifacts.py \
    --original "$OUT_ROOT/original_synthid_lcg_warm_seq200/efficiency_original_synthid_timing.json" \
    --hash "$OUT_ROOT/pvmark_poseidon2_t4_warm_seq200/efficiency_hash_synthid_timing.json" \
    --hash-type3 "$OUT_ROOT/pvmark_poseidon_t3_warm_seq200/efficiency_hash_synthid_timing.json" \
    --hash-type5 "$OUT_ROOT/pvmark_mimc_t5_warm_seq200/efficiency_hash_synthid_timing.json" \
    --upv "$OUT_ROOT/upv_network_detector_gpt2_eli5_200/wet_wdt_network_z1_wet200_wdt200.json" \
    --pdw "$OUT_ROOT/pdw_gpt2/pdw_efficiency_from_records.json" \
    --strict

echo "[$(date -Is)] SERIAL_CONFIRMATION_COMPLETE $OUT_ROOT"
