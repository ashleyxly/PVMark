#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?Usage: $0 <upv|pdw> [limit] [output_base]}"
LIMIT="${2:-1000}"
OUTPUT_BASE="${3:-${PVMark_RESULT_DIR:-tests/baseline_comparison}}"
ENV_NAME="${ENV_NAME:-${PVMark_BASELINE_ENV:-pvmark_baseline}}"
GPU_ID="${GPU_ID:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

run_python() {
  conda run -n "${ENV_NAME}" python "$@"
}

case "${METHOD}" in
  upv)
    OUT_DIR="${OUTPUT_BASE}/upv_gpt2"
    run_python "${SCRIPT_DIR}/upv_experiment.py" \
      --mode full \
      --limit "${LIMIT}" \
      --output-dir "${OUT_DIR}"

    run_python "${SCRIPT_DIR}/run_attacks.py" \
      --input "${OUT_DIR}/generations.json" \
      --output "${OUT_DIR}/attacks.json"

    run_python "${SCRIPT_DIR}/upv_detect_attacks.py" \
      --input "${OUT_DIR}/attacks.json" \
      --output "${OUT_DIR}/attack_detection.json"
    ;;
  pdw)
    OUT_DIR="${OUTPUT_BASE}/pdw_gpt2"
    run_python "${SCRIPT_DIR}/pdw_experiment.py" \
      --mode full \
      --limit "${LIMIT}" \
      --output-dir "${OUT_DIR}"

    run_python "${SCRIPT_DIR}/run_attacks.py" \
      --input "${OUT_DIR}/generations.json" \
      --output "${OUT_DIR}/attacks.json"

    run_python "${SCRIPT_DIR}/pdw_detect_attacks.py" \
      --input "${OUT_DIR}/attacks.json" \
      --output "${OUT_DIR}/attack_detection.json"
    ;;
  *)
    echo "Unknown method: ${METHOD}. Expected upv or pdw." >&2
    exit 2
    ;;
esac

run_python "${SCRIPT_DIR}/run_ppl.py" \
  --input "${OUT_DIR}/generations.json" \
  --output "${OUT_DIR}/ppl.json" \
  --text-key completion_text

run_python "${SCRIPT_DIR}/run_ppl.py" \
  --input "${OUT_DIR}/attacks.json" \
  --output "${OUT_DIR}/attack_ppl.json" \
  --text-key missing

run_python "${SCRIPT_DIR}/summarize.py" \
  --generation "${OUT_DIR}/generations.json" \
  --detection "${OUT_DIR}/detection.json" \
  --output "${OUT_DIR}/summary.json"

run_python "${SCRIPT_DIR}/summarize_robustness.py" \
  --detection "${OUT_DIR}/attack_detection.json" \
  --ppl "${OUT_DIR}/attack_ppl.json" \
  --output-json "${OUT_DIR}/robustness_summary.json" \
  --output-csv "${OUT_DIR}/robustness_summary.csv"

echo "Finished ${METHOD} GPT2 baseline run: ${OUT_DIR}"
