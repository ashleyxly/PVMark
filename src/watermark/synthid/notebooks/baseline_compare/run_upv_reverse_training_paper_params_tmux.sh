#!/usr/bin/env bash
set -euo pipefail

ROOT="${PVMark_SYNTHID_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT_DIR="${1:-$ROOT/tests/baseline_comparison/upv_reverse_training_paper_params_gpt2_eli5}"
ENV_NAME="${ENV_NAME:-${PVMark_BASELINE_ENV:-pvmark_baseline}}"
SCRIPT="$ROOT/notebooks/baseline_compare/upv_reverse_training.py"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$OUT_DIR/logs"

declare -a NUM_QUERY=("10000" "20000" "50000" "100000")
declare -a GPUS=("0" "1" "2" "3")

for i in "${!NUM_QUERY[@]}"; do
  n="${NUM_QUERY[$i]}"
  gpu="${GPUS[$i]}"
  session="upv_rev_paper_${n}_s2026"
  log="$OUT_DIR/logs/${session}.log"
  cmd="cd $ROOT && CUDA_VISIBLE_DEVICES=$gpu conda run -n $ENV_NAME python $SCRIPT --mode full --output-dir $OUT_DIR --num-query $n --seed 2026 --epochs 30 --batch-size 256 --query-batch-size 512 --eval-windows 200000 --eval-batch-size 4096 --detector-source network --query-source synthetic-green-ratio --attacker-mode fixed-subnet --label-mode hard --lr 0.01 --synthetic-candidate-batch 16 --device cuda > $log 2>&1"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session exists: $session"
  else
    tmux new-session -d -s "$session" "$cmd"
    echo "started $session on GPU $gpu, log: $log"
  fi
done
