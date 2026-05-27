#!/usr/bin/env bash
set -euo pipefail

WINDOW_SIZE="${1:?usage: run_upv_window_reverse_pipeline.sh WINDOW_SIZE GPU_ID}"
GPU_ID="${2:?usage: run_upv_window_reverse_pipeline.sh WINDOW_SIZE GPU_ID}"

ROOT="${PVMark_SYNTHID_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
UPV_ROOT="${PVMark_UPV_ROOT:-external/unforgeable_watermark}"
CONDA_ENV="baseline_wm"
DATASET_PATH="${PVMark_ELI5_SELECT_TEST:-experiment_data/prompts/select_test.json}"
GPT2_PATH="${PVMark_GPT2_MODEL:-gpt2}"

RUN_ROOT="${ROOT}/tests/baseline_comparison/upv_w${WINDOW_SIZE}_reverse_pipeline"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pipeline_w${WINDOW_SIZE}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "started_at=$(date -Is)"
echo "window_size=${WINDOW_SIZE}"
echo "gpu_id=${GPU_ID}"

GEN_ROOT="${RUN_ROOT}/generator_model"
GEN_TRAIN_DIR="${RUN_ROOT}/generator_train_data"
DETECTOR_DATA_DIR="${RUN_ROOT}/train_and_test_data"
DETECTOR_DIR="${RUN_ROOT}/network_detector"
REVERSE_DIR="${ROOT}/tests/baseline_comparison/upv_reverse_training_w${WINDOW_SIZE}_gpt2_eli5"

if [[ "${WINDOW_SIZE}" == "1" || "${WINDOW_SIZE}" == "2" ]]; then
  GEN_ROOT="${UPV_ROOT}/experiments/robustness/generator_model/gpt2/window_size_${WINDOW_SIZE}"
fi

COMBINE_MODEL="${GEN_ROOT}/combine_model.pt"
SUBNET_MODEL="${GEN_ROOT}/sub_net.pt"

if [[ ! -f "${COMBINE_MODEL}" || ! -f "${SUBNET_MODEL}" ]]; then
  echo "training_generator_for_w=${WINDOW_SIZE}"
  mkdir -p "${GEN_ROOT}" "${GEN_TRAIN_DIR}"
  env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    conda run -n "${CONDA_ENV}" python "${UPV_ROOT}/generate_data.py" \
      --bit_number 16 \
      --window_size "${WINDOW_SIZE}" \
      --sample_number 5000 \
      --output_file "${GEN_TRAIN_DIR}/train_generator_data.jsonl"

  env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    conda run -n "${CONDA_ENV}" python "${UPV_ROOT}/model_key.py" \
      --data_dir "${GEN_TRAIN_DIR}/train_generator_data.jsonl" \
      --bit_number 16 \
      --model_dir "${GEN_ROOT}/" \
      --window_size "${WINDOW_SIZE}" \
      --layers 5
fi

if [[ ! -f "${DETECTOR_DATA_DIR}/train_data.jsonl" ]]; then
  echo "generating_detector_train_data_for_w=${WINDOW_SIZE}"
  mkdir -p "${DETECTOR_DATA_DIR}"
  env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    conda run -n "${CONDA_ENV}" python -c "
import sys, time
from pathlib import Path
sys.path.insert(0, '${UPV_ROOT}')
from watermark_model import Watermark
out = Path('${DETECTOR_DATA_DIR}')
out.mkdir(parents=True, exist_ok=True)
start = time.time()
wm = Watermark(
    bit_number=16,
    window_size=${WINDOW_SIZE},
    layers=5,
    delta=2.0,
    model_dir='${COMBINE_MODEL}',
    beam_size=0,
)
wm.generate_and_save_train_data(10000, str(out))
print({'out': str(out / 'train_data.jsonl'), 'elapsed_sec': time.time() - start})
"
fi

if [[ ! -f "${DETECTOR_DIR}/detector_z1.pt" ]]; then
  echo "training_network_detector_for_w=${WINDOW_SIZE}"
  env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    conda run -n "${CONDA_ENV}" python "${ROOT}/notebooks/baseline_compare/upv_network_detector.py" \
      --mode train \
      --output-dir "${DETECTOR_DIR}" \
      --train-data "${DETECTOR_DATA_DIR}/train_data.jsonl" \
      --subnet "${SUBNET_MODEL}" \
      --model-name-or-path "${GPT2_PATH}" \
      --window-size "${WINDOW_SIZE}" \
      --layers 5 \
      --z-value 1 \
      --epochs 80 \
      --batch-size 64 \
      --lr 0.0005 \
      --seed 2026 \
      --device cuda
fi

CRACKING_JSON="${REVERSE_DIR}/cracking_eval/cracking_fixed-subnet_network_synthetic-green-ratio_hard_10k_seed2026.json"
if [[ ! -f "${CRACKING_JSON}" ]]; then
  echo "running_reverse_training_for_w=${WINDOW_SIZE}"
  env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    conda run -n "${CONDA_ENV}" python "${ROOT}/notebooks/baseline_compare/upv_reverse_training.py" \
      --mode full \
      --output-dir "${REVERSE_DIR}" \
      --dataset-path "${DATASET_PATH}" \
      --model-name-or-path "${GPT2_PATH}" \
      --upv-root "${UPV_ROOT}" \
      --detector-checkpoint "${DETECTOR_DIR}/detector_z1.pt" \
      --true-generator "${COMBINE_MODEL}" \
      --subnet "${SUBNET_MODEL}" \
      --query-source synthetic-green-ratio \
      --detector-source network \
      --label-mode hard \
      --attacker-mode fixed-subnet \
      --seed 2026 \
      --bit-number 16 \
      --window-size "${WINDOW_SIZE}" \
      --layers 5 \
      --num-query 10000 \
      --query-min-length 100 \
      --query-max-length 200 \
      --query-batch-size 512 \
      --synthetic-candidate-batch 128 \
      --epochs 30 \
      --batch-size 64 \
      --eval-batch-size 4096 \
      --lr 0.001 \
      --eval-windows 200000 \
      --max-new-tokens 200 \
      --generation-limit 1000 \
      --temperature 0.7 \
      --top-k 20 \
      --delta 2.0 \
      --beam-size 0 \
      --llm-name gpt2 \
      --device cuda
fi

FORGED_JSON="${REVERSE_DIR}/forged_detection/detection_fixed-subnet_network_synthetic-green-ratio_hard_10k_seed2026.json"
if [[ ! -f "${FORGED_JSON}" ]]; then
  echo "running_forgery_eval_for_w=${WINDOW_SIZE}"
  env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    conda run -n "${CONDA_ENV}" python "${ROOT}/notebooks/baseline_compare/upv_reverse_training.py" \
      --mode eval-forgery \
      --output-dir "${REVERSE_DIR}" \
      --dataset-path "${DATASET_PATH}" \
      --model-name-or-path "${GPT2_PATH}" \
      --upv-root "${UPV_ROOT}" \
      --detector-checkpoint "${DETECTOR_DIR}/detector_z1.pt" \
      --true-generator "${COMBINE_MODEL}" \
      --subnet "${SUBNET_MODEL}" \
      --query-source synthetic-green-ratio \
      --detector-source network \
      --label-mode hard \
      --attacker-mode fixed-subnet \
      --seed 2026 \
      --bit-number 16 \
      --window-size "${WINDOW_SIZE}" \
      --layers 5 \
      --num-query 10000 \
      --query-min-length 100 \
      --query-max-length 200 \
      --query-batch-size 512 \
      --synthetic-candidate-batch 128 \
      --epochs 30 \
      --batch-size 64 \
      --eval-batch-size 4096 \
      --lr 0.001 \
      --eval-windows 200000 \
      --max-new-tokens 200 \
      --generation-limit 100 \
      --temperature 0.7 \
      --top-k 20 \
      --delta 2.0 \
      --beam-size 0 \
      --llm-name gpt2 \
      --device cuda
fi

echo "finished_at=$(date -Is)"
