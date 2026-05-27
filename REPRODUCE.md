# PVMark Reproduction Guide

This guide explains how to reproduce the key results from the paper "PVMark: Enabling Public Verifiability for LLM Watermarking Schemes".

## Prerequisites

Before running experiments, complete these steps:

```bash
# 1. Install dependencies (Python, Rust, Circom, snarkjs)
bash scripts/setup_environment.sh

# 2. Download models and data
bash download_data.sh

# 3. Download Circom trusted setup ptau files (for ZKP experiments)
bash scripts/download_ptau.sh

# 4. Compile Circom circuits (for ZKP experiments)
bash scripts/compile_circuits.sh
```

Set environment variables:

```bash
export PVMARK_ROOT=/path/to/pvmark-artifact
export HF_MODEL_DIR=/path/to/huggingface/models   # where models are stored
export DATA_DIR=/path/to/datasets                   # where prompts are stored
export RESULT_DIR=/path/to/results                  # where outputs go
export CUDA_VISIBLE_DEVICES=0                       # GPU device (if available)
```

## Quick Start (15-30 minutes)

Run the smoke test to verify your setup:

```bash
bash scripts/run_smoke_test.sh
```

This verifies: Python imports, Rust builds, Circom toolchain, and basic watermark operations.

## Running Individual Watermark Schemes

### KGW Watermark (Embed + Detect)

Requires `hash_rustlib` (built from `src/hash_function/hash-function`):

```bash
cd src/watermark/kgw
python demo_watermark.py --model_name_or_path facebook/opt-1.3b
python demo_watermark.py --model_name_or_path facebook/opt-1.3b --gamma 0.25 --delta 2.0
```

### SynthID-Text Watermark (Embed + Detect)

Requires the `synthid-text` package installed (`pip install -e src/watermark/synthid`):

```bash
# Generate watermarked text
python src/watermark/synthid/notebooks/run_wm.py \
    --model_name_or_path openai-community/gpt2 \
    --data_path experiment_data/prompts/num_100.json \
    --output_path $RESULT_DIR/synthid_wm_output.json

# Generate non-watermarked text (baseline)
python src/watermark/synthid/notebooks/run_nonwm.py \
    --model_name_or_path openai-community/gpt2 \
    --data_path experiment_data/prompts/num_100.json \
    --output_path $RESULT_DIR/synthid_nonwm_output.json

# Test detection time
python src/watermark/synthid/notebooks/test_detect_time.py \
    --model_name_or_path openai-community/gpt2
```

## Full Reproduction

### Table 4: Hash Randomness and Uniformity

**Expected time**: 2-4 hours

```bash
cd src/hash_function/hash-function
cargo run --release --bin test_hash_uniformity
cargo run --release --bin test_hash_sac
```

**Expected output**: Avalanche effect coefficients close to 0.5, chi-square test pass rates >95%.

### Table 5: Implementation Matrix

**Expected time**: 1-2 hours (compile only)

```bash
bash scripts/compile_circuits.sh
```

**Expected output**: All scheme/hash/protocol combinations compile without errors. Each circuit produces a `.r1cs`, `.wasm`, and `.zkey` file.

### Table 6: Effectiveness and Fidelity

**Expected time**: 8-12 hours (full), 15-30 minutes (smoke test)

```bash
# Full experiment: KGW + SynthID watermark embedding and detection
bash scripts/run_effectiveness.sh

# Smoke test with minimal settings
SMOKE=1 bash scripts/run_effectiveness.sh
```

**Expected output**: Success rates >95% for OPT-1.3B/C4, >99% for GPT-2/ELI5, PPL values as reported in Table 6. Results saved to `$RESULT_DIR/effectiveness/`.

### Table 7: Robustness

**Expected time**: 12-18 hours (full)

```bash
# Run robustness experiments with all three attack types (deletion, wordnet, BERT)
bash scripts/run_robustness.sh

# Smoke test with minimal settings
SMOKE=1 bash scripts/run_robustness.sh
```

**Expected output**: Post-attack success rates and PPL values as reported in Table 7.

### Table 8: WET/WDT Efficiency

**Expected time**: 4-6 hours

```bash
bash scripts/run_efficiency.sh
```

**Expected output**: Watermark Embedding Time (WET) and Watermark Detection Time (WDT) in ms/token and s/sample, as reported in Table 8.

### Table 9: Baseline Comparisons

**Expected time**: 10-15 hours

```bash
# Included in the robustness script (runs UPV and PDW baselines alongside PVMark)
bash scripts/run_robustness.sh
```

**Expected output**: UPV and PDW effectiveness, robustness, and efficiency results as reported in Table 9.

### Table 10: UPV Reverse Training Attack

**Expected time**: 20-30 hours

```bash
# Included in the robustness script
bash scripts/run_robustness.sh
```

**Expected output**: Cracking metrics and forged detection rates as reported in Table 10.

### Figure: ZKP Costs (Groth16/PlonK/halo2/Nova)

**Expected time**: 24-48 hours (full)

```bash
# Run ZKP benchmark for all protocols
bash scripts/run_zkp_benchmark.sh
```

**Expected output**: Setup time, prove time, verification time, and proof size for each protocol and token count.

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce batch size or use smaller models
2. **CUDA errors**: Ensure CUDA toolkit is installed and GPU is available
3. **Missing data**: Run `bash download_data.sh` to download required data
4. **Circuit compilation errors**: Ensure Circom 2.0+ is installed and ptau files are downloaded
5. **Import errors**: Ensure `PVMARK_ROOT` is set and `PYTHONPATH` includes `$PVMARK_ROOT`

### Cross-Platform Notes

All shell scripts support both Linux and macOS. On macOS:
- Install `coreutils` via `brew install coreutils` for `gtimeout` support
- Circom and snarkjs work natively
- CUDA experiments require an NVIDIA GPU (skip on Apple Silicon)
