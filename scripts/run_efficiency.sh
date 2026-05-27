#!/usr/bin/env bash
# Run efficiency experiments: Table 8 (WET/WDT benchmark)
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

echo "=== PVMark Efficiency Experiments (Table 8) ==="
echo "PVMARK_ROOT: $PVMARK_ROOT"
echo "GPU: $GPU"
echo "Smoke mode: $SMOKE"
echo ""

mkdir -p "$RESULTS_ROOT/table8_efficiency"

# --- KGW WET/WDT ---
echo "[1/3] KGW WET/WDT benchmark..."
python -B "$SCRIPT_DIR/benchmark_efficiency.py" \
    --output-dir "$RESULTS_ROOT/table8_efficiency" \
    --device "cuda:0" \
    2>&1 | tee "$RESULTS_ROOT/table8_efficiency/kgw_benchmark.log" || echo "  [WARN] KGW benchmark failed"

# --- SynthID WET/WDT ---
echo "[2/3] SynthID WET/WDT benchmark..."
cd "$PVMARK_ROOT/src/watermark/synthid"
python -B notebooks/test_generate_time.py \
    --output "$RESULTS_ROOT/table8_efficiency/synthid_wet.json" \
    2>&1 | tee "$RESULTS_ROOT/table8_efficiency/synthid_wet.log" || echo "  [WARN] SynthID WET failed"

python -B notebooks/test_detect_time.py \
    --output "$RESULTS_ROOT/table8_efficiency/synthid_wdt.json" \
    2>&1 | tee "$RESULTS_ROOT/table8_efficiency/synthid_wdt.log" || echo "  [WARN] SynthID WDT failed"
cd "$PVMARK_ROOT"

# --- Hash-based efficiency (if Rust lib is built) ---
echo "[3/3] Hash-based KGW efficiency..."
python -B "$SCRIPT_DIR/run_kgw_hash_end_to_end_efficiency.py" \
    --output-dir "$RESULTS_ROOT/table8_efficiency" \
    2>&1 | tee "$RESULTS_ROOT/table8_efficiency/hash_kgw.log" || echo "  [WARN] Hash KGW efficiency failed"

echo ""
echo "=== Efficiency Experiments Complete ==="
echo "Results in: $RESULTS_ROOT/table8_efficiency"
