#!/usr/bin/env bash
# Build all Rust components for PVMark.
# Components: hash-function, halo2, nova
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PVMARK_ROOT="${PVMARK_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: cargo not found. Install Rust first: https://rustup.rs/" >&2
    exit 127
fi

echo "=== Building PVMark Rust Components ==="
echo "PVMARK_ROOT: $PVMARK_ROOT"
echo ""

# --- 1. hash-function (MiMC/Poseidon/Poseidon2/Pedersen) ---
echo "[1/3] Building hash-function..."
HASH_DIR="$PVMARK_ROOT/src/hash_function/hash-function"
if [ -f "$HASH_DIR/Cargo.toml" ]; then
    cd "$HASH_DIR"
    cargo build --release 2>&1 | tail -3
    echo "  OK: hash-function"
else
    echo "  SKIP: $HASH_DIR/Cargo.toml not found"
fi
cd "$PVMARK_ROOT"

# --- 2. halo2 PLONKish circuits ---
echo "[2/3] Building halo2 circuits..."
HALO2_DIR="$PVMARK_ROOT/src/zkp/halo2"
if [ -f "$HALO2_DIR/Cargo.toml" ]; then
    cd "$HALO2_DIR"
    cargo build --release 2>&1 | tail -3
    echo "  OK: halo2"
else
    echo "  SKIP: $HALO2_DIR/Cargo.toml not found"
fi
cd "$PVMARK_ROOT"

# --- 3. Nova recursive ZKP (KGW + SynthID + Segment) ---
echo "[3/3] Building nova..."
NOVA_DIR="$PVMARK_ROOT/src/zkp/nova"
if [ -f "$NOVA_DIR/Cargo.toml" ]; then
    cd "$NOVA_DIR"
    cargo build --release 2>&1 | tail -3
    echo "  OK: nova"
else
    echo "  SKIP: $NOVA_DIR/Cargo.toml not found"
fi
cd "$PVMARK_ROOT"

echo ""
echo "=== Build Complete ==="
