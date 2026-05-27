#!/usr/bin/env bash
set -euo pipefail

PVMARK_ROOT="${PVMARK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
CIRCOM_ROOT="${CIRCOM_ROOT:-$PVMARK_ROOT/source_code/zkp/circom}"
PTAU_DIR="${PTAU_DIR:?set PTAU_DIR to the directory containing powersOfTau28_hez_final_*.ptau}"
CIRCUIT="${CIRCUIT:-test_zk_friendly_only_25_token/mimc_two_to_one_fixed_threshold.circom}"
INPUT_JSON="${INPUT_JSON:-$CIRCOM_ROOT/test_zk_friendly_only_25_token/input_fixed.json}"
BUILD_DIR="${BUILD_DIR:-$PVMARK_ROOT/reproduction_outputs/zkp/groth16_fixed}"
PTAU_POWER="${PTAU_POWER:-16}"

mkdir -p "$BUILD_DIR"
name="$(basename "$CIRCUIT" .circom)"
circom "$CIRCOM_ROOT/$CIRCUIT" --r1cs --wasm --sym --c --output "$BUILD_DIR"
"$BUILD_DIR/${name}_cpp/$name" "$INPUT_JSON" "$BUILD_DIR/${name}_witness.wtns"
snarkjs groth16 setup "$BUILD_DIR/${name}.r1cs" "$PTAU_DIR/powersOfTau28_hez_final_${PTAU_POWER}.ptau" "$BUILD_DIR/${name}_0.zkey"
snarkjs zkey contribute "$BUILD_DIR/${name}_0.zkey" "$BUILD_DIR/${name}_1.zkey" --name="artifact" -v -e="${ZKEY_ENTROPY:-pvmark}"
snarkjs zkey export verificationkey "$BUILD_DIR/${name}_1.zkey" "$BUILD_DIR/${name}_verification_key.json"
snarkjs groth16 prove "$BUILD_DIR/${name}_1.zkey" "$BUILD_DIR/${name}_witness.wtns" "$BUILD_DIR/${name}_proof.json" "$BUILD_DIR/${name}_public.json"
snarkjs groth16 verify "$BUILD_DIR/${name}_verification_key.json" "$BUILD_DIR/${name}_public.json" "$BUILD_DIR/${name}_proof.json"
