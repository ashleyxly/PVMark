#!/usr/bin/env bash
# Run ZKP benchmark: Figure ZKP costs (proof generation, verification, proof size)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PVMARK_ROOT="${PVMARK_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-$PVMARK_ROOT/reproduction_outputs}"
PTAU_DIR="${PTAU_DIR:-$PVMARK_ROOT/trusted_setup}"
BUILD_DIR="${BUILD_DIR:-$PVMARK_ROOT/build/circuits}"
SMOKE="${SMOKE:-0}"

export PVMARK_ROOT

echo "=== PVMark ZKP Benchmark ==="
echo "PVMARK_ROOT: $PVMARK_ROOT"
echo "PTAU_DIR: $PTAU_DIR"
echo ""

mkdir -p "$RESULTS_ROOT/zkp_benchmark"

# --- Groth16 (KGW circuits) ---
echo "[1/4] Groth16 benchmark (KGW)..."
KGW_CIRCOM="$PVMARK_ROOT/src/zkp/circom/kgw"
for hash_type in mimc poseidon poseidon2 blake2; do
    circuit_dir="$BUILD_DIR/kgw/$hash_type"
    [ -d "$circuit_dir" ] || continue
    for r1cs in "$circuit_dir"/*.r1cs; do
        [ -f "$r1cs" ] || continue
        name="$(basename "$r1cs" .r1cs)"
        echo "  Groth16: $hash_type/$name"
        # Use existing run_fixed_groth16.sh pattern
        bash "$KGW_CIRCOM/scripts/run_fixed_groth16.sh" 2>&1 | \
            tee "$RESULTS_ROOT/zkp_benchmark/groth16_${hash_type}_${name}.txt" || true
    done
done

# --- PlonK (KGW circuits) ---
echo "[2/4] PlonK benchmark (KGW)..."
for hash_type in mimc poseidon poseidon2; do
    circuit_dir="$BUILD_DIR/kgw/$hash_type"
    [ -d "$circuit_dir" ] || continue
    for r1cs in "$circuit_dir"/*.r1cs; do
        [ -f "$r1cs" ] || continue
        name="$(basename "$r1cs" .r1cs)"
        echo "  PlonK: $hash_type/$name"
        bash "$KGW_CIRCOM/scripts/run_fixed_plonk.sh" 2>&1 | \
            tee "$RESULTS_ROOT/zkp_benchmark/plonk_${hash_type}_${name}.txt" || true
    done
done

# --- halo2 (PLONKish) ---
echo "[3/4] halo2 benchmark..."
HALO2_DIR="$PVMARK_ROOT/src/zkp/halo2"
if [ -f "$HALO2_DIR/Cargo.toml" ]; then
    cd "$HALO2_DIR"
    for bin in "$HALO2_DIR"/src/bin/detect_watermark*.rs; do
        [ -f "$bin" ] || continue
        bin_name="$(basename "$bin" .rs)"
        echo "  halo2: $bin_name"
        cargo run --release --bin "$bin_name" 2>&1 | \
            tee "$RESULTS_ROOT/zkp_benchmark/halo2_${bin_name}.txt" || true
    done
    cd "$PVMARK_ROOT"
fi

# --- Nova (recursive ZKP) ---
echo "[4/4] Nova benchmark..."
NOVA_DIR="$PVMARK_ROOT/src/zkp/nova"
if [ -f "$NOVA_DIR/Cargo.toml" ]; then
    cd "$NOVA_DIR"
    echo "  Nova KGW recursive proof..."
    cargo run --release --bin mimc-detect -- --type-sampling kgw 2>&1 | \
        tee "$RESULTS_ROOT/zkp_benchmark/nova_kgw.txt" || true
    echo "  Nova SynthID recursive proof..."
    cargo run --release --bin mimc-detect -- --type-sampling synthid 2>&1 | \
        tee "$RESULTS_ROOT/zkp_benchmark/nova_synthid.txt" || true
    cd "$PVMARK_ROOT"
fi

echo ""
echo "=== ZKP Benchmark Complete ==="
echo "Results in: $RESULTS_ROOT/zkp_benchmark"
