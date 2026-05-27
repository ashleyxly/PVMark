# Known Limitations

## Data and Model Availability

- **Models**: HuggingFace models are not included in the artifact due to size. Use `download_data.sh` to fetch them.
- **Datasets**: Full datasets (C4, Pile, ELI5) are not included. Only prompt subsets are provided.
- **Lookup tables**: KGW lookup tables (~2.35GB per hash) are not included. They can be regenerated using the provided scripts.

## Implementation Limitations

- **PDW variable length**: PDW's full-signature generation produces variable-length outputs, making direct WET comparison difficult.
- **UPV score comparability**: UPV network detector scores are not directly comparable to KGW z-scores or SynthID g-value scores.
- **PlonK memory**: PlonK circuits for SynthID are too large for full memory measurement on some systems.

## Reproduction Variations

- **Randomness**: Results may vary slightly due to random seed differences.
- **Hardware**: Performance numbers depend on specific hardware configurations.
- **Model versions**: HuggingFace model versions may change over time.

## Scope

- **Watermark robustness**: PVMark does not solve the robustness limits of watermarking schemes themselves.
- **ZKP soundness**: Implementation bugs in circuits or constraints may cause proofs to capture different statements than intended.
- **Deployment**: This artifact is for research purposes. Production deployment requires additional security review.
