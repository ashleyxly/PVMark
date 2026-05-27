# KGW ZKP Source

This directory contains the KGW ZKP material recovered from the local `ZKLLMWatermark` and `Nova-Scotia` workspaces. It is fixed-variant only: hash-based KGW Sort variants are intentionally absent.

## Layout

| Path | Contents |
|---|---|
| `circom/` | Fixed-threshold KGW Circom circuits, `input_fixed.json` fixtures, local `circomlib/circuits/` dependencies, Groth16/PlonK helper scripts, and `process_res.py`. |
| `halo2/halo2-detection/` | Halo2 KGW detection crate with Fixed-only bins for Poseidon and MiMC. `Cargo.toml` exposes only `TwotoOneFixed` and `ThreetoOneFixed` binaries. |
| `nova_scotia/` | Source-only Nova-Scotia middleware plus a KGW MiMC fixed-threshold recursive example. Generated `_cpp`, `_js`, `.r1cs`, `.wtns`, `.wasm`, `.zkey`, and binary outputs are excluded. |
| `scripts/parse_zkp_outputs.py` | Parser for Circom timing logs, Halo2 timing logs, and proof/public/verifier JSON sizes. |

## Environment Paths To Fill

These variables point to machine-specific data or tools and must be set in the real experiment environment:

| Variable | Required by | Expected value |
|---|---|---|
| `PTAU_DIR` | `circom/scripts/run_fixed_groth16.sh`, `circom/scripts/run_fixed_plonk.sh` | Directory containing `powersOfTau28_hez_final_*.ptau`. |
| `HALO2_SRS_DIR` | Halo2 fixed detection binaries | Directory containing `perpetual-powers-of-tau-raw-<k>` files. Defaults to `external/srs_params` if unset. |
| `HALO2_RESULT_DIR` | `halo2/halo2-detection/run_test.sh` | Writable directory for Halo2 timing logs. |
| `PVMARK_ROOT` | ZKP shell scripts | Artifact root. Scripts infer it when run from this package. |
| `CIRCOM_ROOT` | Circom helper scripts | Override only if circuits are moved outside `source_code/zkp/circom`. |
| `BUILD_DIR` | Circom helper scripts | Writable output directory for generated proof artifacts. Defaults under `reproduction_outputs/zkp/`. |

Toolchain requirements are not vendored: Circom, snarkjs, Rust/Cargo, a C++ compiler for Circom witness generators, and any Halo2/Nova dependencies must be installed or available from local caches.

## Fixed-Only Commands

Circom Groth16 smoke run:

```bash
export PTAU_DIR=/path/to/ptau
bash source_code/zkp/circom/scripts/run_fixed_groth16.sh
```

Circom PlonK smoke run:

```bash
export PTAU_DIR=/path/to/ptau
bash source_code/zkp/circom/scripts/run_fixed_plonk.sh
```

Halo2 Fixed bins:

```bash
cd source_code/zkp/halo2/halo2-detection
export HALO2_SRS_DIR=/path/to/halo2_srs
bash compile.sh
bash run_test.sh
```

Nova-Scotia recursive MiMC Fixed example:

```bash
cd source_code/zkp/nova_scotia
bash compile_kgw_fixed.sh
ITERATION_COUNT=2 bash run_kgw_fixed_recursive.sh
```

## Included Results

Raw fixed-threshold timing logs and proof/public/verifier JSON snapshots are in:

```text
experiment_data/zkp_zero_bit_kgw/circom_timings/
experiment_data/zkp_zero_bit_kgw/halo2_results/
experiment_data/zkp_proof_sizes/fixed_threshold/
```
