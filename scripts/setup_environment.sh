#!/usr/bin/env bash
# PVMark Environment Setup Script
# Cross-platform: supports Linux (Ubuntu/Debian) and macOS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PVMARK_ROOT="${PVMARK_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPT_DIR/os_detect.sh"

echo "=== PVMark Environment Setup ==="
echo "OS: $OS_TYPE"
echo "PVMARK_ROOT: $PVMARK_ROOT"
echo ""

# --- Python dependencies ---
echo "[1/6] Installing Python dependencies..."
pip install -r "$PVMARK_ROOT/requirements.txt"
echo ""

# --- Rust toolchain ---
echo "[2/6] Checking Rust toolchain..."
if command -v cargo >/dev/null 2>&1; then
    echo "  Rust already installed: $(rustc --version)"
else
    echo "  Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "  Rust installed: $(rustc --version)"
fi
echo ""

# --- Node.js (for circom/snarkjs) ---
echo "[3/6] Checking Node.js..."
if command -v node >/dev/null 2>&1; then
    echo "  Node.js already installed: $(node --version)"
else
    echo "  Installing Node.js..."
    if has_package_manager brew; then
        brew install node
    elif has_package_manager apt-get; then
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
    elif has_package_manager dnf; then
        sudo dnf install -y nodejs
    else
        echo "  [WARN] Cannot auto-install Node.js. Install manually: https://nodejs.org/"
    fi
    echo "  Node.js: $(node --version 2>/dev/null || echo 'not installed')"
fi
echo ""

# --- Circom ---
echo "[4/6] Checking Circom..."
if command -v circom >/dev/null 2>&1; then
    echo "  Circom already installed"
else
    echo "  Installing Circom from source..."
    CIRCOM_DIR="$HOME/.local/bin"
    mkdir -p "$CIRCOM_DIR"
    CIRCOM_SRC="$(get_tmpdir)/circom_build"
    rm -rf "$CIRCOM_SRC"
    git clone https://github.com/iden3/circom.git "$CIRCOM_SRC"
    cd "$CIRCOM_SRC"
    cargo build --release
    cp target/release/circom "$CIRCOM_DIR/circom"
    cd "$PVMARK_ROOT"
    rm -rf "$CIRCOM_SRC"
    export PATH="$CIRCOM_DIR:$PATH"
    echo "  Circom installed to $CIRCOM_DIR/circom"
    echo "  NOTE: Add $CIRCOM_DIR to your PATH permanently:"
    echo "    echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
fi
echo ""

# --- snarkjs ---
echo "[5/6] Checking snarkjs..."
if command -v snarkjs >/dev/null 2>&1; then
    echo "  snarkjs already installed"
elif command -v npm >/dev/null 2>&1; then
    echo "  Installing snarkjs globally..."
    npm install -g snarkjs
    echo "  snarkjs installed"
else
    echo "  [WARN] npm not available, cannot install snarkjs"
fi
echo ""

# --- Build Rust components ---
echo "[6/6] Building Rust components..."
bash "$SCRIPT_DIR/build_rust.sh"
echo ""

echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Download data/models:  bash download_data.sh"
echo "  2. Download ptau files:   bash scripts/download_ptau.sh"
echo "  3. Compile circuits:      bash scripts/compile_circuits.sh"
echo "  4. Run smoke test:        bash scripts/run_smoke_test.sh"
