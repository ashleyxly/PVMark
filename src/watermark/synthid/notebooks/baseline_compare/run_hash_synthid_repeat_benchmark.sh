#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-tests/baseline_comparison/hash_synthid_repeat_fair_2026-05-25}"
REPEATS="${REPEATS:-5}"
ENV_NAME="${ENV_NAME:-${PVMark_BASELINE_ENV:-pvmark_baseline}}"
MODEL_PATH="${MODEL_PATH:-${PVMark_GPT2_MODEL:-gpt2}}"
WET_RUNS="${WET_RUNS:-30}"
WDT_RUNS="${WDT_RUNS:-300}"
WARMUP_RUNS="${WARMUP_RUNS:-5}"
TOKEN_LENGTH="${TOKEN_LENGTH:-200}"
TOP_K="${TOP_K:-40}"

mkdir -p "$OUT_ROOT/logs"

run_logged() {
  local name="$1"
  shift
  local log_path="$OUT_ROOT/logs/${name}.log"
  echo "[$(date -Is)] START $name"
  "$@" >"$log_path" 2>&1
  echo "[$(date -Is)] DONE $name"
}

for repeat in $(seq 1 "$REPEATS"); do
  repeat_label="$(printf "repeat_%02d" "$repeat")"
  repeat_dir="$OUT_ROOT/$repeat_label"
  mkdir -p "$repeat_dir"

  run_logged "${repeat_label}_original" \
    conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_original_synthid_efficiency.py \
      --output-dir "$repeat_dir/original_lcg" \
      --model-name-or-path "$MODEL_PATH" \
      --backend original-python \
      --device cuda \
      --token-lengths "$TOKEN_LENGTH" \
      --batch-size 1 \
      --wet-runs "$WET_RUNS" \
      --wdt-runs "$WDT_RUNS" \
      --warmup-runs "$WARMUP_RUNS" \
      --progress-every 0 \
      --cache-mode cold \
      --top-k "$TOP_K" \
      --score-type weighted_mean

  run_logged "${repeat_label}_poseidon2_t4" \
    conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_hash_synthid_efficiency.py \
      --output-dir "$repeat_dir/poseidon2_t4" \
      --model-name-or-path "$MODEL_PATH" \
      --device cuda \
      --hash-type 4 \
      --token-lengths "$TOKEN_LENGTH" \
      --batch-size 1 \
      --wet-runs "$WET_RUNS" \
      --wdt-runs "$WDT_RUNS" \
      --warmup-runs "$WARMUP_RUNS" \
      --progress-every 0 \
      --cache-mode cold \
      --top-k "$TOP_K" \
      --score-type weighted_mean \
      --fused-g-values \
      --fused-detect-g-values \
      --fast-context-mask \
      --fused-detector-score \
      --compile-update-scores

  run_logged "${repeat_label}_poseidon_t3" \
    conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_hash_synthid_efficiency.py \
      --output-dir "$repeat_dir/poseidon_t3" \
      --model-name-or-path "$MODEL_PATH" \
      --device cuda \
      --hash-type 3 \
      --token-lengths "$TOKEN_LENGTH" \
      --batch-size 1 \
      --wet-runs "$WET_RUNS" \
      --wdt-runs "$WDT_RUNS" \
      --warmup-runs "$WARMUP_RUNS" \
      --progress-every 0 \
      --cache-mode cold \
      --top-k "$TOP_K" \
      --score-type weighted_mean \
      --fused-g-values \
      --fused-detect-g-values \
      --fast-context-mask \
      --fused-detector-score \
      --compile-update-scores

  run_logged "${repeat_label}_mimc_t5" \
    conda run -n "$ENV_NAME" python notebooks/baseline_compare/time_hash_synthid_efficiency.py \
      --output-dir "$repeat_dir/mimc_t5" \
      --model-name-or-path "$MODEL_PATH" \
      --device cuda \
      --hash-type 5 \
      --token-lengths "$TOKEN_LENGTH" \
      --batch-size 1 \
      --wet-runs "$WET_RUNS" \
      --wdt-runs "$WDT_RUNS" \
      --warmup-runs "$WARMUP_RUNS" \
      --progress-every 0 \
      --cache-mode cold \
      --top-k "$TOP_K" \
      --score-type weighted_mean \
      --fused-g-values \
      --fused-detect-g-values \
      --fast-context-mask \
      --fused-detector-score \
      --compile-update-scores
done

conda run -n "$ENV_NAME" python notebooks/baseline_compare/summarize_hash_synthid_repeat_benchmark.py \
  --root "$OUT_ROOT" \
  --json "$OUT_ROOT/repeat_summary.json" \
  --csv "$OUT_ROOT/repeat_summary.csv" \
  --html docs/hash_synthid_repeat_fair_benchmark_2026-05-25.html

echo "[$(date -Is)] REPEAT_BENCHMARK_COMPLETE $OUT_ROOT"
