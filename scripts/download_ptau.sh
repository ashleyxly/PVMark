#!/usr/bin/env bash
# Download Circom trusted setup powers-of-tau files.
# Cross-platform: Linux and macOS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PVMARK_ROOT="${PVMARK_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPT_DIR/os_detect.sh"
PTAU_DIR="${PTAU_DIR:-$PVMARK_ROOT/trusted_setup}"

mkdir -p "$PTAU_DIR"

echo "=== Downloading Powers of Tau Files ==="
echo "PTAU_DIR: $PTAU_DIR"
echo ""

# Hermez Phase 1 ceremony files (BN254 curve)
# Sizes: p14~17MB, p15~34MB, p16~68MB, p17~136MB, p18~272MB, p19~544MB, p20~1.1GB
POWERS=(15 16 17 18 19 20)

for p in "${POWERS[@]}"; do
    FILE="powersOfTau28_hez_final_${p}.ptau"
    URL="https://hermez.s3-eu-west-1.amazonaws.com/$FILE"
    DEST="$PTAU_DIR/$FILE"

    if [ -f "$DEST" ]; then
        echo "[SKIP] $FILE already exists"
        continue
    fi

    echo "[DOWNLOAD] $FILE (power=$p)..."
    if curl -L --fail -o "$DEST" "$URL" 2>/dev/null; then
        echo "  OK: $FILE"
    else
        echo "  [WARN] Hermez download failed, trying alternative..."
        rm -f "$DEST"
        # Alternative: P0tion ceremony files from Ethereum community
        ALT_URL="https://ptau.hermez.io/$FILE"
        if curl -L --fail -o "$DEST" "$ALT_URL" 2>/dev/null; then
            echo "  OK: $FILE (from ptau.hermez.io)"
        else
            echo "  [WARN] Could not download $FILE"
            rm -f "$DEST"
            echo "  You may need to download ptau files manually."
            echo "  See: https://github.com/iden3/snarkjs#7-prepare-phase-2"
        fi
    fi
done

echo ""
echo "=== Verifying Downloads ==="
count=$(ls "$PTAU_DIR"/*.ptau 2>/dev/null | wc -l | tr -d ' ')
echo "Found $count ptau files in $PTAU_DIR"
ls -lh "$PTAU_DIR"/*.ptau 2>/dev/null || echo "No ptau files found"
echo ""
echo "=== Done ==="
echo "Set PTAU_DIR=$PTAU_DIR when running circuit compilation."
