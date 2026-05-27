#!/usr/bin/env bash
# Run effectiveness experiments: Table 4 (hash randomness) + Table 6 (watermark fidelity)
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

echo "=== PVMark Effectiveness Experiments ==="
echo "PVMARK_ROOT: $PVMARK_ROOT"
echo "GPU: $GPU"
echo "Smoke mode: $SMOKE"
echo ""

mkdir -p "$RESULTS_ROOT"

# --- Table 4: Hash Randomness (Rust) ---
echo "[1/2] Table 4: Hash randomness tests..."
HASH_DIR="$PVMARK_ROOT/src/hash_function/hash-function"
if [ -f "$HASH_DIR/Cargo.toml" ]; then
    cd "$HASH_DIR"
    echo "  Running hash uniformity test..."
    cargo run --release --bin test_hash_uniformity 2>&1 | tee "$RESULTS_ROOT/table4_hash_uniformity.txt" || true
    echo "  Running hash SAC test..."
    cargo run --release --bin test_hash_sac 2>&1 | tee "$RESULTS_ROOT/table4_hash_sac.txt" || true
    cd "$PVMARK_ROOT"
else
    echo "  SKIP: hash-function crate not found"
fi

# --- Table 6: Effectiveness (KGW + SynthID) ---
echo "[2/2] Table 6: Watermark effectiveness..."
DATASET="$PVMARK_ROOT/experiment_data/prompts/num_100.json"
if [ ! -f "$DATASET" ]; then
    echo "  ERROR: Dataset not found: $DATASET"
    echo "  Run 'bash download_data.sh' first."
    exit 1
fi

if [ "$SMOKE" == "1" ]; then
    INPUT_NUM=5
    MAX_TOKENS=50
else
    INPUT_NUM=100
    MAX_TOKENS=200
fi

# KGW effectiveness
echo "  Running KGW watermark embedding + detection..."
KGW_DIR="$PVMARK_ROOT/src/watermark/kgw"
python -B "$KGW_DIR/run_watermark_test2.py" \
    --input_num "$INPUT_NUM" \
    --max_new_tokens "$MAX_TOKENS" \
    --output-json "$RESULTS_ROOT/table6_kgw_results.json" \
    2>&1 | tee "$RESULTS_ROOT/table6_kgw.log" || echo "  [WARN] KGW effectiveness failed"

# SynthID effectiveness
echo "  Running SynthID watermark embedding + detection..."
cd "$PVMARK_ROOT/src/watermark/synthid"
python -B notebooks/run_wm.py \
    --input_num "$INPUT_NUM" \
    --max_new_tokens "$MAX_TOKENS" \
    2>&1 | tee "$RESULTS_ROOT/table6_synthid.log" || echo "  [WARN] SynthID effectiveness failed"
cd "$PVMARK_ROOT"

echo ""
echo "=== Effectiveness Experiments Complete ==="
echo "Results in: $RESULTS_ROOT"
