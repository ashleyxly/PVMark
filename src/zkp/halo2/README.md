This is a PVMark implementation using halo2, with KGW as the underlying watermark scheme.

# PROG1: halo2-verify-detect-watermark-v2



## Compilation
Recommended to run in conda environment:
1. Install Python dependencies
2. Install rust-lib (ZKLLMWatermark_Codes-main/hash_function/hash-function) in the current Python environment, e.g.:
```
conda activate lm_watermark
maturin develop --release
```
3. Compile the Rust programs in this repo with the command:
```
cargo build --bin verify_detect_text_watermark_v2 --features dev-graph --release
```

## Execution
1. Run demo_watermark.py from llm-watermark to generate watermarked text:
```
python demo_watermark.py
```
  The public inputs are stored by default at ./test_result/halo2_circuit_public_inputs.txt

2. Run the Rust program from this repo to implement watermark detection and provide proof of detection correctness:

  Run with default parameters:
```
./target/release/verify_detect_text_watermark_v2
```

  View parameters and modify as needed:
```
./target/release/verify_detect_text_watermark_v2 -h
```

  Current default parameter settings:
```
Usage: verify_detect_text_watermark_v2 [OPTIONS]

Options:
  -i, --input-file-path <INPUT_FILE_PATH>
          The path of input file including public inputs [default: /mnt/disk2/username/kzg-halo2-merkle-tree/test_result/test_verify_detect_text_watermark/input.txt]
  -o, --output-file-path <OUTPUT_FILE_PATH>
          The path of output file including counting results [default: /mnt/disk2/username/kzg-halo2-merkle-tree/test_result/test_verify_detect_text_watermark/output.txt]
  -p, --proof-path <PROOF_PATH>
          The path of proof [default: /mnt/disk2/username/kzg-halo2-merkle-tree/test_result/test_verify_detect_text_watermark/proof.bin]
  -s, --secret-key <SECRET_KEY>
          The secret key value [default: 2023]
  -m, --max-token-num <MAX_TOKEN_NUM>
          The number of token [default: 200]
  -h, --help
          Print help
  -V, --version
          Print version
```

*This README was created with assistance from iflow cli and glm-4.7*
