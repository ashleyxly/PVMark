# PVMark: Enabling Public Verifiability for LLM Watermarking Schemes

This artifact contains the source code, experiment scripts, and documentation for reproducing the results in the paper "PVMark: Enabling Public Verifiability for LLM Watermarking Schemes" (USENIX Security 2026).

## Overview

PVMark is a plugin based on zero-knowledge proof (ZKP) that enables the watermark detection process to be publicly verifiable by third parties without disclosing any secret key. This artifact includes:

- **Watermark implementations**: KGW and SynthID-Text watermarking schemes adapted for ZKP-friendly hash functions
- **Hash function library**: Rust implementations of MiMC, Poseidon, Poseidon2, Pedersen, BLAKE2, SHA256, Keccak
- **ZKP circuits**: Circom circuits (Groth16/PlonK), halo2 PLONKish circuits, and Nova recursive ZKP implementations
- **CUDA/GPU optimizations**: CUDA kernels for accelerated hash computation
- **Baseline comparisons**: UPV and PDW baseline implementations
- **Experiment scripts**: Scripts for reproducing effectiveness, robustness, and efficiency experiments

## Architecture

PVMark works by replacing standard hash functions (e.g., SHA-256) in watermark embedding/detection with ZKP-friendly alternatives (MiMC, Poseidon, Poseidon2), then proving the correctness of the detection algorithm via zero-knowledge proofs. The high-level data flow is:

```
Watermark Embedding (Python)          ZKP Verification (Rust/Circom)
  ┌──────────────────────┐            ┌──────────────────────────────┐
  │ KGW or SynthID-Text  │            │ Prover computes ZKP that:    │
  │ uses ZKP-friendly    │───────►    │ "I ran the detection algo    │
  │ hash functions       │            │  on (tokens, hash_key) and   │
  │ (MiMC/Poseidon/...)  │            │  got result R, without       │
  └──────────────────────┘            │  revealing hash_key"         │
                                      └──────────────────────────────┘
                                              │
                                              ▼
                                      ┌──────────────────────────────┐
                                      │ Verifier (public) checks:    │
                                      │ "The proof is valid, so R    │
                                      │  is trustworthy"             │
                                      └──────────────────────────────┘
```

**Three watermark schemes** are supported:
- **KGW** (`src/watermark/kgw/`): Hash-based green/red list watermarking. Embeds by biasing token selection toward "green list" tokens determined by a secret key and preceding context.
- **SynthID-Text** (`src/watermark/synthid/`): Tournament-sampling watermark using g-values. Embeds by seeding a PRNG with context-derived hashes and using tournament sampling to select tokens.
- **Segment-Watermark** (`src/watermark/segment/`): Multi-bit watermarking that encodes a payload message into the text.

**Four ZKP protocols** are implemented for the detection proof:
- **Groth16/PlonK** (`src/zkp/circom/`): Circom circuits compiled to R1CS, proved via snarkjs. Best for small token counts (<200).
- **halo2** (`src/zkp/halo2/`): PLONKish circuits in Rust. No trusted setup required.
- **Nova** (`src/zkp/nova/`): Recursive ZKP that processes tokens in batches. Most efficient for large token counts (200+), scales to 50,000+ tokens.

## Quick Start

```bash
# 1. Install dependencies (Python, Rust, Circom, snarkjs)
bash scripts/setup_environment.sh

# 2. Download models and data
bash download_data.sh

# 3. Download Circom trusted setup ptau files
bash scripts/download_ptau.sh

# 4. Compile Circom circuits (optional, for ZKP experiments)
bash scripts/compile_circuits.sh

# 5. Run smoke tests (15-30 minutes)
bash scripts/run_smoke_test.sh
```

See [INSTALL.md](INSTALL.md) for detailed installation and [REPRODUCE.md](REPRODUCE.md) for full reproduction instructions.

**Environment variables**: Scripts auto-detect `PVMARK_ROOT` from the script directory. To override, set `PVMARK_ROOT=/path/to/pvmark-artifact`. See [INSTALL.md](INSTALL.md) for `HF_MODEL_DIR`, `DATA_DIR`, and `RESULT_DIR`.

## Repository Structure

```
pvmark-artifact/
├── src/                                # Source code
│   ├── watermark/                      # Watermark implementations (Python)
│   │   ├── kgw/                        # KGW: hash-based green/red list watermark
│   │   │   ├── watermark_processor.py  #   Embedding (hash-based variant)
│   │   │   ├── detect_watermark.py     #   Detection (hash-based variant)
│   │   │   ├── watermark_processor_org_scheme.py  # Original KGW (benchmark)
│   │   │   ├── compute_text_PPL.py     #   Perplexity evaluation
│   │   │   └── normalizers.py          #   Text normalization utilities
│   │   ├── synthid/                    # SynthID-Text: tournament-sampling watermark
│   │   │   └── src/synthid_text/
│   │   │       ├── logits_processing.py    #   Embedding (g-value, tournament)
│   │   │       ├── detector_mean.py        #   Mean-based detector
│   │   │       ├── detector_bayesian.py    #   Bayesian detector
│   │   │       └── hashing_function.py     #   Hash function interface
│   │
│   ├── hash_function/                  # ZKP-friendly hash functions (Rust)
│   │   ├── hash-function/              # Main crate: MiMC, Poseidon, Poseidon2, Pedersen
│   │   │   └── src/bin/                #   Uniformity/SAC tests (Table 4)
│   │   ├── arkworks-mimc/              # Arkworks MiMC reference implementation
│   │   ├── mimc-rs/                    # Pure Rust MiMC implementation
│   │   ├── poseidon2/                  # Poseidon2 reference implementation
│   │   ├── hash_uniformity_test/       # Standalone uniformity test harness
│   │   └── circom_circuit/             # Circom hash circuit templates
│   │
│   ├── zkp/                            # Zero-knowledge proof implementations
│   │   ├── circom/                     # Circom circuits (Groth16/PlonK via snarkjs)
│   │   │   ├── kgw/                    #   KGW detection circuits (all hash variants)
│   │   │   │   ├── blake2/ poseidon/ poseidon2/ mimc/ keccak/ sha256/ pedersen/
│   │   │   │   └── test_zk_friendly_only_N_token_plonk/  # Per-token-count test circuits
│   │   │   ├── synthid/                #   SynthID detection circuits (recursive + non-recursive)
│   │   │   │   ├── Hash/ poseidon/ mimc/ poseidon2/  # Recursive hash detection
│   │   │   │   ├── LCG/                #     LCG generator + detection
│   │   │   │   └── Non-Recursive-Hash/ #     Direct full-proof
│   │   │   ├── segment/                #   Segment-Watermark detection circuits
│   │   │   └── circomlib/              #   circomlib dependency (comparators, mimc, poseidon)
│   │   ├── halo2/                      # halo2 PLONKish circuits (Rust)
│   │   │   └── src/bin/                #   Detection binaries for each scheme/hash combo
│   │   └── nova/                       # Nova recursive ZKP (Rust, KGK + SynthID + Segment)
│   │
│   ├── native_libraries/               # CUDA/GPU optimizations
│   │   ├── synthid_text/               # SynthID CUDA hash kernels
│   │   ├── kgw/                        # KGW CUDA hash kernels + lookup tables
│   │   └── scripts/                    # Rust native library build scripts
│   │
│   └── baselines/                      # Baseline comparison implementations
│       ├── upv/                        # UPV (Unforgeable Publicly-Verifiable) watermark
│       ├── pdw/                        # PDW (Publicly Detectable Watermark)
│       └── markllm_attacks/            # MarkLLM attack suite (deletion, wordnet, BERT)
│
├── baseline_eval/                      # Python bridge package (re-exports for import compatibility)
├── scripts/                            # Experiment runner scripts
│   ├── setup_environment.sh            # One-click environment setup
│   ├── download_ptau.sh                # Download Circom trusted setup (ptau) files
│   ├── build_rust.sh                   # Build all Rust components
│   ├── compile_circuits.sh             # Compile all Circom circuits
│   ├── run_effectiveness.sh            # Table 6: watermark effectiveness + fidelity
│   ├── run_robustness.sh               # Table 7+9+10: robustness + baselines + UPV reverse
│   ├── run_efficiency.sh               # Table 8: WET/WDT efficiency benchmarks
│   ├── run_zkp_benchmark.sh            # Figure: ZKP costs (all protocols)
│   ├── run_smoke_test.sh               # Quick validation (15-30 min)
│   ├── os_detect.sh                    # Cross-platform OS detection (Linux/macOS)
│   ├── common.py                       # Shared Python utilities
│   └── *.py                            # Individual benchmark/analysis scripts
│
├── experiment_data/                    # Experiment data
│   ├── prompts/                        # Prompt subsets (populated by download_data.sh)
│   └── README.md                       # Data format documentation
│
├── docs/                               # Additional documentation
│   ├── paper_result_map.md             # Paper table/figure → artifact path → command
│   ├── hardware_and_runtime.md         # Hardware and runtime environment details
│   └── known_limitations.md            # Known limitations and workarounds
│
├── download_data.sh                    # Download models and datasets
├── requirements.txt                    # Python dependencies
├── INSTALL.md                          # Detailed installation guide
├── REPRODUCE.md                        # Step-by-step reproduction guide
├── CITATION.cff                        # Citation metadata
├── LICENSE                             # MIT License
└── MANIFEST.sha256                     # File checksums (2122 files)
```

## Paper Result Mapping

| Paper Location | Artifact Path | Reproduction Command |
|----------------|---------------|----------------------|
| Table 4: Hash randomness | `src/hash_function/hash-function/` | `cargo run --release --bin test_hash_uniformity` |
| Table 5: Implementation matrix | `src/zkp/circom/` | `bash scripts/compile_circuits.sh` |
| Table 6: Effectiveness | `src/watermark/` | `bash scripts/run_effectiveness.sh` |
| Table 7: Robustness | `src/baselines/markllm_attacks/` | `bash scripts/run_robustness.sh` |
| Table 8: WET/WDT | `src/native_libraries/` | `bash scripts/run_efficiency.sh` |
| Table 9: Baselines | `src/baselines/upv/` + `src/baselines/pdw/` | `bash scripts/run_robustness.sh` |
| Table 10: UPV reverse | `src/baselines/upv/` | `bash scripts/run_robustness.sh` |
| Fig: ZKP costs | `src/zkp/` | `bash scripts/run_zkp_benchmark.sh` |

See [docs/paper_result_map.md](docs/paper_result_map.md) for detailed mapping including expected outputs and per-experiment commands.

## Hardware Requirements

- **CPU**: Intel Xeon Gold 6240C or equivalent (256GB RAM recommended)
- **GPU**: NVIDIA GeForce RTX 3090 or equivalent (for CUDA kernels)
- **OS**: Ubuntu 20.04 LTS (macOS also supported for CPU-only experiments)
- **Disk**: ~50GB for models and data

## Software Requirements

- Python 3.9+, PyTorch 2.0+, Transformers 4.30+
- Rust 1.70+ (with Cargo)
- CUDA 11.8+ (for GPU optimizations)
- Circom 2.0+ (for ZKP circuits)
- Node.js 16+ (for snarkjs)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
