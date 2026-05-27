# Paper Result to Artifact Mapping

This document maps each table and figure in the PVMark paper to the corresponding artifact paths and reproduction commands.

## Tables

### Table 4: Hash Randomness and Uniformity

**Paper claim**: MiMC, Poseidon, Poseidon2 exhibit sufficient randomness and uniformity for watermarking.

**Artifact location**: `src/hash_function/hash-function/`

**Key files**:
- `src/bin/test_hash_uniformity.rs` - Chi-square uniformity test
- `src/bin/test_hash_sac.rs` - Avalanche effect (SAC) test
- `src/lib.rs` - Hash function implementations

**Reproduction command**:
```bash
cd src/hash_function/hash-function
cargo run --release --bin test_hash_uniformity
cargo run --release --bin test_hash_sac
```

**Expected output**: Avalanche coefficients close to 0.5, chi-square pass rates >95%.

---

### Table 5: Implementation Matrix

**Paper claim**: All variants of watermark detection algorithm are implemented.

**Artifact location**: `src/zkp/`

**Key files**:
- `src/zkp/circom/kgw/` - KGW Circom circuits (all hash variants: blake2, mimc, poseidon, poseidon2, keccak, sha256, pedersen)
- `src/zkp/circom/synthid/` - SynthID Circom circuits (recursive + non-recursive)
- `src/zkp/circom/segment/` - Segment-Watermark Circom circuits
- `src/zkp/halo2/` - halo2 PLONKish circuits
- `src/zkp/nova/` - Nova recursive ZKP

**Reproduction command**:
```bash
bash scripts/download_ptau.sh
bash scripts/compile_circuits.sh
```

**Expected output**: All scheme/hash/protocol combinations compile successfully.

---

### Table 6: Effectiveness and Fidelity

**Paper claim**: PVMark preserves watermarking effectiveness (>95% SR for KGW, >99% for SynthID).

**Artifact location**: `src/watermark/`

**Key files**:
- `src/watermark/kgw/watermark_processor.py` - KGW embedding
- `src/watermark/kgw/detect_watermark.py` - KGW detection
- `src/watermark/synthid/src/synthid_text/logits_processing.py` - SynthID embedding
- `scripts/run_effectiveness.sh` - Experiment runner

**Reproduction command**:
```bash
bash scripts/run_effectiveness.sh

# Smoke test (minimal settings, 15-30 min)
SMOKE=1 bash scripts/run_effectiveness.sh
```

**Expected output**: Success rates and PPL values as reported in Table 6.

---

### Table 7: Robustness

**Paper claim**: PVMark maintains robustness against three attack types.

**Artifact location**: `src/baselines/markllm_attacks/`

**Key files**:
- `src/baselines/markllm_attacks/run_attacks.py` - Attack implementations
- `src/baselines/markllm_attacks/run_robustness_detection.py` - Detection after attacks
- `scripts/run_robustness.sh` - Experiment runner

**Reproduction command**:
```bash
bash scripts/run_robustness.sh

# Smoke test (minimal settings)
SMOKE=1 bash scripts/run_robustness.sh
```

**Expected output**: Post-attack success rates and PPL values as reported in Table 7.

---

### Table 8: WET/WDT Efficiency

**Paper claim**: PVMark's hash-based adaptation does not introduce bottlenecks.

**Artifact location**: `src/native_libraries/` + `scripts/`

**Key files**:
- `src/native_libraries/synthid_text/cuda_hash_ext_kernel.cu` - SynthID CUDA kernel
- `src/native_libraries/kgw/` - KGW CUDA kernel + lookup tables
- `scripts/benchmark_efficiency.py` - Benchmark runner

**Reproduction command**:
```bash
bash scripts/run_efficiency.sh
```

**Expected output**: WET and WDT values (ms/token and s/sample) as reported in Table 8.

---

### Table 9: Baseline Comparisons

**Paper claim**: PVMark outperforms UPV and PDW in robustness and efficiency.

**Artifact location**: `src/baselines/`

**Key files**:
- `src/baselines/upv/` - UPV implementation and experiments
- `src/baselines/pdw/` - PDW implementation and experiments
- `scripts/run_robustness.sh` - Runs baselines alongside PVMark

**Reproduction command**:
```bash
bash scripts/run_robustness.sh
```

**Expected output**: UPV and PDW results as reported in Table 9.

---

### Table 10: UPV Reverse Training Attack

**Paper claim**: UPV is vulnerable to reverse-training attacks.

**Artifact location**: `src/baselines/upv/`

**Key files**:
- `src/baselines/upv/` - UPV reverse training implementation
- `scripts/run_robustness.sh` - Runs reverse training attack

**Reproduction command**:
```bash
bash scripts/run_robustness.sh
```

**Expected output**: Cracking metrics and forged detection rates as reported in Table 10.

---

## Figures

### Figure: ZKP Costs (Groth16/PlonK/halo2/Nova)

**Paper claim**: Nova is most efficient for 200+ tokens.

**Artifact location**: `src/zkp/`

**Key files**:
- `src/zkp/circom/` - Circom circuits for Groth16/PlonK
- `src/zkp/halo2/` - halo2 circuits
- `src/zkp/nova/` - Nova recursive ZKP
- `scripts/run_zkp_benchmark.sh` - Benchmark runner

**Reproduction command**:
```bash
bash scripts/run_zkp_benchmark.sh
```

**Expected output**: Setup time, prove time, verification time, and proof size for each protocol.

---

### Figure: Nova Scalability

**Paper claim**: Nova scales to 50,000 tokens with reasonable overhead.

**Artifact location**: `src/zkp/nova/`

**Key files**:
- `src/zkp/nova/` - Nova implementation (KGW, SynthID, and Segment)

**Reproduction command**:
```bash
bash scripts/run_zkp_benchmark.sh
```

**Expected output**: Total computational overhead ~352s for KGW at 50,000 tokens.
