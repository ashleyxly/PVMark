#!/usr/bin/env bash
# PVMark Smoke Test - Quick validation (15-30 min)
# Cross-platform: Linux and macOS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PVMARK_ROOT="${PVMARK_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPT_DIR/os_detect.sh"

PASS=0
FAIL=0
SKIP=0

export PVMARK_ROOT
export PYTHONPATH="$PVMARK_ROOT:$SCRIPT_DIR:$PVMARK_ROOT/src/watermark/kgw:$PVMARK_ROOT/src:${PYTHONPATH:-}"

echo "=== PVMark Smoke Test ==="
echo "OS: $OS_TYPE"
echo "PVMARK_ROOT: $PVMARK_ROOT"
echo ""

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }
skip() { echo "  SKIP: $1"; SKIP=$((SKIP + 1)); }

# --- 1. Python imports ---
echo "[1/8] Python imports..."
python -c "import torch; print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')" && pass "torch" || fail "torch"
python -c "import transformers; print(f'  transformers={transformers.__version__}')" && pass "transformers" || fail "transformers"
python -c "import numpy; print(f'  numpy={numpy.__version__}')" && pass "numpy" || fail "numpy"
python -c "import scipy; print(f'  scipy={scipy.__version__}')" && pass "scipy" || fail "scipy"
python -c "import nltk; print(f'  nltk={nltk.__version__}')" && pass "nltk" || fail "nltk"
python -c "import tokenizers; print(f'  tokenizers={tokenizers.__version__}')" && pass "tokenizers" || fail "tokenizers"
echo ""

# --- 2. Rust toolchain ---
echo "[2/8] Rust toolchain..."
command -v cargo >/dev/null 2>&1 && pass "cargo ($(cargo --version))" || fail "cargo not found"
command -v rustc >/dev/null 2>&1 && pass "rustc ($(rustc --version))" || fail "rustc not found"
echo ""

# --- 3. Circom toolchain ---
echo "[3/8] Circom toolchain..."
command -v circom >/dev/null 2>&1 && pass "circom" || skip "circom not installed"
command -v snarkjs >/dev/null 2>&1 && pass "snarkjs" || skip "snarkjs not installed"
echo ""

# --- 4. Hash function Rust build ---
echo "[4/8] Hash function Rust crate..."
HASH_DIR="$PVMARK_ROOT/src/hash_function/hash-function"
if [ -f "$HASH_DIR/Cargo.toml" ]; then
    cd "$HASH_DIR"
    cargo build --release --quiet 2>/dev/null && pass "hash-function build" || fail "hash-function build"
    cd "$PVMARK_ROOT"
else
    skip "hash-function crate not found"
fi
echo ""

# --- 5. halo2 build ---
echo "[5/8] halo2 Rust crate..."
HALO2_DIR="$PVMARK_ROOT/src/zkp/halo2"
if [ -f "$HALO2_DIR/Cargo.toml" ]; then
    cd "$HALO2_DIR"
    cargo build --release --quiet 2>/dev/null && pass "halo2 build" || fail "halo2 build"
    cd "$PVMARK_ROOT"
else
    skip "halo2 crate not found"
fi
echo ""

# --- 6. Nova build ---
echo "[6/8] Nova Rust crate..."
NOVA_DIR="$PVMARK_ROOT/src/zkp/nova"
if [ -f "$NOVA_DIR/Cargo.toml" ]; then
    cd "$NOVA_DIR"
    cargo build --release --quiet 2>/dev/null && pass "nova build" || fail "nova build"
    cd "$PVMARK_ROOT"
else
    skip "nova crate not found"
fi
echo ""

# --- 7. KGW watermark smoke test ---
echo "[7/8] KGW watermark smoke test..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    SMOKE=1 bash "$SCRIPT_DIR/run_effectiveness.sh" 2>/dev/null && pass "KGW smoke" || fail "KGW smoke"
else
    skip "No CUDA available, skipping GPU smoke test"
fi
echo ""

# --- 8. Hash uniformity test (CPU-only) ---
echo "[8/8] Hash uniformity test (CPU)..."
if [ -f "$HASH_DIR/Cargo.toml" ]; then
    cd "$HASH_DIR"
    run_with_timeout 60 cargo run --release --bin test_hash_uniformity 2>/dev/null && pass "hash uniformity" || fail "hash uniformity"
    cd "$PVMARK_ROOT"
else
    skip "hash-function crate not found"
fi

echo ""
echo "=== Smoke Test Results ==="
echo "PASS: $PASS  FAIL: $FAIL  SKIP: $SKIP"
if [ "$FAIL" -gt 0 ]; then
    echo "Some tests failed. Check the output above."
    exit 1
else
    echo "All tests passed (or skipped)."
fi
