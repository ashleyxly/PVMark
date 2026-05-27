#!/usr/bin/env bash
set -euo pipefail

ROOT="${PVMark_SYNTHID_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT_DIR="${1:-$ROOT/tests/baseline_comparison/upv_reverse_training_gpt2_eli5}"
ENV_NAME="${ENV_NAME:-${PVMark_BASELINE_ENV:-pvmark_baseline}}"
SCRIPT="$ROOT/notebooks/baseline_compare/upv_reverse_training.py"
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$OUT_DIR/logs"

declare -a NUM_QUERY=("10000" "20000" "50000" "100000")
declare -a GPUS=("0" "1" "2" "3")

for i in "${!NUM_QUERY[@]}"; do
  n="${NUM_QUERY[$i]}"
  gpu="${GPUS[$i]}"
  session="upv_rev_${n}_s2026"
  log="$OUT_DIR/logs/${session}.log"
  cmd="cd $ROOT && CUDA_VISIBLE_DEVICES=$gpu conda run -n $ENV_NAME python $SCRIPT --mode full --output-dir $OUT_DIR --num-query $n --seed 2026 --epochs 30 --attacker-mode fixed-subnet --detector-source network --query-source upv-train-data --label-mode hard --device cuda > $log 2>&1"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session exists: $session"
  else
    tmux new-session -d -s "$session" "$cmd"
    echo "started $session on GPU $gpu, log: $log"
  fi
done
