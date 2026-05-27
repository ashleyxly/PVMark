#!/usr/bin/env bash
# Run robustness experiments: Table 7 (attacks) + Table 9 (baselines) + Table 10 (UPV reverse)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PVMARK_ROOT="${PVMARK_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-$PVMARK_ROOT/reproduction_outputs}"
GPU="${GPU:-0}"
SMOKE="${SMOKE:-0}"

export PVMARK_ROOT
export PYTHONPATH="$PVMARK_ROOT:$SCRIPT_DIR:$PVMARK_ROOT/src/watermark/kgw:$PVMARK_ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== PVMark Robustness Experiments ==="
echo "PVMARK_ROOT: $PVMARK_ROOT"
echo "GPU: $GPU"
echo "Smoke mode: $SMOKE"
echo ""

mkdir -p "$RESULTS_ROOT"

if [ "$SMOKE" == "1" ]; then
    NUM_SAMPLES=5
    MAX_TOKENS=50
else
    NUM_SAMPLES=100
    MAX_TOKENS=200
fi

DATASET="$PVMARK_ROOT/experiment_data/prompts/num_100.json"

# --- Table 7: Robustness (MarkLLM attacks on KGW + SynthID) ---
echo "[1/3] Table 7: Robustness under attacks..."

echo "  [KGW] Generating watermarked text..."
python -B "$PVMARK_ROOT/src/watermark/kgw/run_watermark_test2.py" \
    --dataset "$DATASET" --num-samples "$NUM_SAMPLES" --max-new-tokens "$MAX_TOKENS" \
    --output-dir "$RESULTS_ROOT/table7_kgw_gen" \
    2>&1 | tee "$RESULTS_ROOT/table7_kgw_gen.log" || true

echo "  [KGW] Running MarkLLM attacks..."
python -B "$PVMARK_ROOT/src/baselines/markllm_attacks/run_attacks.py" \
    --input "$RESULTS_ROOT/table7_kgw_gen" \
    --output "$RESULTS_ROOT/table7_kgw_attacks" \
    2>&1 | tee "$RESULTS_ROOT/table7_kgw_attacks.log" || true

echo "  [KGW] Robustness detection..."
python -B "$PVMARK_ROOT/src/baselines/markllm_attacks/run_robustness_detection.py" \
    --generations "$RESULTS_ROOT/table7_kgw_gen" \
    --attacks "$RESULTS_ROOT/table7_kgw_attacks" \
    --output "$RESULTS_ROOT/table7_kgw_robustness.json" \
    2>&1 | tee "$RESULTS_ROOT/table7_kgw_robustness.log" || true

# --- Table 9: Baselines (UPV + PDW) ---
echo "[2/3] Table 9: Baseline comparisons..."

echo "  [UPV] Running UPV experiments..."
python -B "$PVMARK_ROOT/src/baselines/upv/run_unforgeable_network.py" \
    --dataset "$DATASET" --num-samples "$NUM_SAMPLES" --max-new-tokens "$MAX_TOKENS" \
    --output-dir "$RESULTS_ROOT/table9_upv" \
    2>&1 | tee "$RESULTS_ROOT/table9_upv.log" || true

echo "  [PDW] Running PDW experiments..."
python -B "$PVMARK_ROOT/src/baselines/pdw/run_publicly_detectable.py" \
    --dataset "$DATASET" --num-samples "$NUM_SAMPLES" --max-new-tokens "$MAX_TOKENS" \
    --output-dir "$RESULTS_ROOT/table9_pdw" \
    2>&1 | tee "$RESULTS_ROOT/table9_pdw.log" || true

# --- Table 10: UPV Reverse Training Attack ---
echo "[3/3] Table 10: UPV reverse training attack..."

echo "  [UPV Reverse] Training surrogate generator + attacking..."
python -B "$PVMARK_ROOT/src/baselines/upv/upv_reverse_training.py" \
    --dataset "$DATASET" --num-samples "$NUM_SAMPLES" \
    --output-dir "$RESULTS_ROOT/table10_upv_reverse" \
    2>&1 | tee "$RESULTS_ROOT/table10_upv_reverse.log" || true

echo ""
echo "=== Robustness Experiments Complete ==="
echo "Results in: $RESULTS_ROOT"
