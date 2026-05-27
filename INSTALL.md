# PVMark Installation Guide

## System Requirements

### Hardware
- **CPU**: Intel Xeon Gold 6240C (or equivalent with 16+ cores)
- **RAM**: 256GB (minimum 64GB for smoke tests)
- **GPU**: NVIDIA GeForce RTX 3090 with 24GB VRAM (for CUDA optimizations)
- **Disk**: ~50GB free space

### Software
- **OS**: Ubuntu 20.04 LTS (other Linux distributions and macOS are also supported)
- **Python**: 3.9 or later
- **Rust**: 1.70 or later (with Cargo)
- **CUDA**: 11.8 or later (for GPU optimizations; CPU-only mode available)
- **Circom**: 2.0 or later (for ZKP circuits)
- **Node.js**: 16 or later (for snarkjs)

## One-Click Setup

```bash
# Full environment setup (Python deps, Rust, Circom, snarkjs)
bash scripts/setup_environment.sh
```

This script will:
1. Install Python dependencies from `requirements.txt`
2. Install Rust toolchain (if not present)
3. Install Node.js (if not present)
4. Build Circom from source (if not present)
5. Install snarkjs globally
6. Build all Rust components (hash-function, halo2, nova)

## Manual Installation Steps

If you prefer manual setup or the one-click script fails:

### 1. Clone the Repository

```bash
git clone <repository-url> pvmark-artifact
cd pvmark-artifact
```

### 2. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install all Python dependencies
pip install -r requirements.txt

# Install SynthID-Text as a package (required for SynthID experiments)
pip install -e src/watermark/synthid
```

The SynthID-Text package (`src/watermark/synthid/`) is based on Google DeepMind's
[synthid-text](https://github.com/google-deepmind/synthid-text) with modifications
to use ZKP-friendly hash functions. It provides:
- `synthid_text.logits_processing` — watermark embedding via g-value tournament sampling
- `synthid_text.detector_mean` — mean-based watermark detector (no training needed)
- `synthid_text.detector_bayesian` — Bayesian watermark detector (requires training)
- `synthid_text.synthid_mixin` — mixin for HuggingFace Transformers models (GPT-2, Gemma)

### 3. Install Rust Toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Build all Rust components at once
bash scripts/build_rust.sh
```

### 4. Install Circom and ZKP Tools

```bash
# Install Circom
git clone https://github.com/iden3/circom.git "$TMPDIR/circom_build"
cd "$TMPDIR/circom_build" && cargo build --release
mkdir -p ~/.local/bin && cp target/release/circom ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"

# Install snarkjs
npm install -g snarkjs
```

### 5. Download Models and Data

```bash
# Download required models and datasets
bash download_data.sh

# Download Circom trusted setup ptau files
bash scripts/download_ptau.sh
```

### 6. Compile Circom Circuits

```bash
bash scripts/compile_circuits.sh
```

### 7. Verify Installation

```bash
# Run smoke tests (15-30 minutes)
bash scripts/run_smoke_test.sh
```

## Environment Variables

Set the following environment variables:

```bash
export PVMARK_ROOT=/path/to/pvmark-artifact
export HF_MODEL_DIR=/path/to/huggingface/models
export DATA_DIR=/path/to/datasets
export RESULT_DIR=/path/to/results
export CUDA_VISIBLE_DEVICES=0  # GPU device ID
```

## Cross-Platform Notes

All shell scripts support both Linux and macOS. On macOS:
- Install `coreutils` via `brew install coreutils` for `gtimeout` support
- Circom and snarkjs work natively
- CUDA experiments require an NVIDIA GPU (skip on Apple Silicon)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU-only mode
2. **Circom compilation errors**: Ensure circom version is 2.0+
3. **Rust compilation errors**: Update Rust toolchain with `rustup update`
4. **Missing Python packages**: Run `pip install -r requirements.txt`

### Getting Help

For installation issues, please check the [GitHub Issues](https://github.com/ashleyxly/PVMark/issues) page.
