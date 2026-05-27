#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${PVMark_SYNTHID_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
SCRIPT_DIR="${REPO_ROOT}/notebooks/baseline_compare"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/tests/baseline_comparison}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs}"
LIMIT="${LIMIT:-1000}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3}"
ENV_NAME="${ENV_NAME:-${PVMark_BASELINE_ENV:-pvmark_baseline}}"
RUN_UPV="${RUN_UPV:-1}"
RUN_PDW="${RUN_PDW:-1}"

mkdir -p "${LOG_DIR}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_DIR}/baseline_full_${RUN_ID}.log"
LATEST_LOG="${LOG_DIR}/baseline_full_latest.log"

exec > >(tee -a "${MAIN_LOG}") 2>&1
ln -sfn "${MAIN_LOG}" "${LATEST_LOG}"

echo "[$(date -Is)] Baseline comparison run started"
echo "repo=${REPO_ROOT}"
echo "output_base=${OUTPUT_BASE}"
echo "limit=${LIMIT}"
echo "gpus=${GPUS_CSV}"
echo "env=${ENV_NAME}"
echo "run_upv=${RUN_UPV}"
echo "run_pdw=${RUN_PDW}"
echo "main_log=${MAIN_LOG}"

cd "${REPO_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH" >&2
  exit 127
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "conda env ${ENV_NAME} not found" >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[$(date -Is)] Initial GPU snapshot"
  nvidia-smi || true
fi

run_method() {
  local method="$1"
  local start_ts
  start_ts="$(date +%s)"
  echo "[$(date -Is)] Starting ${method}"
  PYTHONDONTWRITEBYTECODE=1 ENV_NAME="${ENV_NAME}" bash "${SCRIPT_DIR}/run_parallel_gpt2_baseline.sh" \
    "${method}" \
    "${LIMIT}" \
    "${OUTPUT_BASE}" \
    "${GPUS_CSV}"
  local end_ts
  end_ts="$(date +%s)"
  echo "[$(date -Is)] Finished ${method} in $((end_ts - start_ts)) sec"
}

if [[ "${RUN_UPV}" == "1" ]]; then
  run_method upv
fi

if [[ "${RUN_PDW}" == "1" ]]; then
  run_method pdw
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[$(date -Is)] Final GPU snapshot"
  nvidia-smi || true
fi

echo "[$(date -Is)] Baseline comparison run finished"
