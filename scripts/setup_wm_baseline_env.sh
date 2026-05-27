#!/usr/bin/env bash
set -euo pipefail

PVMARK_ROOT="${PVMARK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_PREFIX="${ENV_PREFIX:-${PVMARK_ROOT}/.venv_wm_baseline}"
SOURCE_ENV="${SOURCE_ENV:-}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${PVMARK_ROOT}/.conda_pkgs}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PVMARK_ROOT}/.pip_cache}"

mkdir -p "$(dirname "$ENV_PREFIX")" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  if [ -z "$SOURCE_ENV" ]; then
    echo "Set SOURCE_ENV to an existing conda environment to clone." >&2
    exit 2
  fi
  CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" PIP_CACHE_DIR="$PIP_CACHE_DIR" \
    conda create --prefix "$ENV_PREFIX" --clone "$SOURCE_ENV" -y
fi

PIP_CACHE_DIR="$PIP_CACHE_DIR" conda run -p "$ENV_PREFIX" pip install \
  msgpack==0.6.2 \
  bitstring==4.1.4 \
  reedsolo==1.7.0 \
  bls-lib==1.0.1 \
  "git+https://github.com/jfairoze/bplib.git#egg=bplib"

conda run -p "$ENV_PREFIX" python -c "import torch, transformers, evaluate; import bplib, bls, petlib, reedsolo, bitstring; print('wm_baseline env ok')"
