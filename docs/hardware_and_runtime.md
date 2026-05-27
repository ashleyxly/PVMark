# Hardware and Runtime Environment

## Primary Server (KGW Experiments)

- **CPU**: Intel(R) Xeon(R) Gold 6240C @ 2.60GHz (256 cores)
- **RAM**: 256GB DDR4
- **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **OS**: Ubuntu 20.04 LTS
- **CUDA**: 11.8
- **Python**: 3.9
- **PyTorch**: 2.0
- **Rust**: 1.70
- **Circom**: 2.0

## Secondary Server (SynthID Experiments)

- **CPU**: Intel(R) Xeon(R) Gold 6240C @ 2.60GHz (256 cores)
- **RAM**: 256GB DDR4
- **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **OS**: Ubuntu 20.04 LTS
- **CUDA**: 11.8
- **Python**: 3.9
- **PyTorch**: 2.0
- **Rust**: 1.70

## Runtime Estimates

| Experiment | Time (Full) | Time (Smoke Test) |
|------------|-------------|-------------------|
| Table 4: Hash Randomness | 2-4 hours | 30 minutes |
| Table 5: Implementation Matrix | 1-2 hours | 30 minutes |
| Table 6: Effectiveness | 8-12 hours | 30 minutes |
| Table 7: Robustness | 12-18 hours | 2 hours |
| Table 8: WET/WDT | 4-6 hours | 1 hour |
| Table 9: Baselines | 10-15 hours | 2 hours |
| Table 10: UPV Reverse | 20-30 hours | 4 hours |
| Fig: ZKP Costs | 24-48 hours | 4 hours |

## Notes

- All times are approximate and depend on hardware performance
- Smoke tests use reduced datasets (10-100 samples)
- Full experiments use the complete datasets as described in the paper
- GPU is required for CUDA optimizations (WET/WDT benchmarks)
- CPU-only mode is available but significantly slower
