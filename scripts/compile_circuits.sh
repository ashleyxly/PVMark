#!/usr/bin/env bash
# Compile all Circom circuits and generate witness generators.
# Prerequisites: circom, snarkjs installed (run setup_environment.sh first)
# Optional: set PTAU_DIR to skip ptau download prompt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PVMARK_ROOT="${PVMARK_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPT_DIR/os_detect.sh"
PTAU_DIR="${PTAU_DIR:-$PVMARK_ROOT/trusted_setup}"
BUILD_DIR="${BUILD_DIR:-$PVMARK_ROOT/build/circuits}"

if ! command -v circom >/dev/null 2>&1; then
    echo "ERROR: circom not found. Run scripts/setup_environment.sh first." >&2
    exit 127
fi

echo "=== Compiling PVMark Circom Circuits ==="
echo "PVMARK_ROOT: $PVMARK_ROOT"
echo "BUILD_DIR: $BUILD_DIR"
echo ""

mkdir -p "$BUILD_DIR"

COMPILED=0
FAILED=0

compile_circuit() {
    local circuit_path="$1"
    local circuit_dir="$(dirname "$circuit_path")"
    local circuit_name="$(basename "$circuit_path" .circom)"
    local out_dir="$BUILD_DIR/$2"
    mkdir -p "$out_dir"

    echo "  Compiling: $circuit_path"
    if circom "$circuit_path" --r1cs --wasm --sym --c --output "$out_dir" 2>&1 | tail -2; then
        COMPILED=$((COMPILED + 1))
    else
        echo "  FAILED: $circuit_path"
        FAILED=$((FAILED + 1))
    fi
}

# --- KGW circuits (Groth16/PlonK variants) ---
echo "[1/3] KGW circuits..."
KGW_CIRCOM="$PVMARK_ROOT/src/zkp/circom/kgw"
if [ -d "$KGW_CIRCOM" ]; then
    for hash_dir in blake2 keccak mimc pedersen poseidon poseidon2 sha256; do
        if [ -d "$KGW_CIRCOM/$hash_dir" ]; then
            for f in "$KGW_CIRCOM/$hash_dir"/*.circom; do
                [ -f "$f" ] && compile_circuit "$f" "kgw/$hash_dir"
            done
        fi
    done
    # Test circuits for different token counts
    for test_dir in "$KGW_CIRCOM"/test_zk_friendly_only_*_token*/; do
        [ -d "$test_dir" ] || continue
        local_name="$(basename "$test_dir")"
        for f in "$test_dir"/*.circom; do
            [ -f "$f" ] && compile_circuit "$f" "kgw/$local_name"
        done
    done
else
    echo "  SKIP: KGW circom directory not found"
fi

# --- SynthID circuits ---
echo "[2/3] SynthID circuits..."
SYNTHID_CIRCOM="$PVMARK_ROOT/src/zkp/circom/synthid/synthid_circom"
if [ -d "$SYNTHID_CIRCOM" ]; then
    # Non-Recursive-Hash
    for hash_dir in "$SYNTHID_CIRCOM"/Non-Recursive-Hash/*/; do
        [ -d "$hash_dir" ] || continue
        local_name="synthid/non-recursive/$(basename "$hash_dir")"
        for f in "$hash_dir"*.circom; do
            [ -f "$f" ] && compile_circuit "$f" "$local_name"
        done
    done
    # Recursive Hash
    for hash_dir in "$SYNTHID_CIRCOM"/Hash/*/; do
        [ -d "$hash_dir" ] || continue
        local_name="synthid/hash/$(basename "$hash_dir")"
        for f in "$hash_dir"*.circom; do
            [ -f "$f" ] && compile_circuit "$f" "$local_name"
        done
    done
    # LCG
    if [ -d "$SYNTHID_CIRCOM/LCG" ]; then
        for f in "$SYNTHID_CIRCOM/LCG"/*.circom; do
            [ -f "$f" ] && compile_circuit "$f" "synthid/lcg"
        done
    fi
else
    echo "  SKIP: SynthID circom directory not found"
fi

# --- Segment circuits ---
echo "[3/3] Segment circuits..."
SEGMENT_CIRCOM="$PVMARK_ROOT/src/zkp/circom/segment"
if [ -d "$SEGMENT_CIRCOM" ]; then
    shopt -s globstar
    for f in "$SEGMENT_CIRCOM"/**/*.circom; do
        [ -f "$f" ] && compile_circuit "$f" "segment"
    done
    shopt -u globstar
else
    echo "  SKIP: Segment circom directory not found"
fi

echo ""
echo "=== Compilation Complete ==="
echo "Compiled: $COMPILED, Failed: $FAILED"
echo "Output: $BUILD_DIR"
