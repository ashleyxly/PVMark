#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-baseline_wm}"
BASE_ENV="${BASE_ENV:-synthid}"

if ! conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating $ENV_NAME by cloning $BASE_ENV. This reuses existing torch packages via conda links."
  conda create -y -n "$ENV_NAME" --clone "$BASE_ENV"
else
  echo "Conda env $ENV_NAME already exists."
fi

echo "Installing additional dependencies for publicly-detectable-watermark."
conda run -n "$ENV_NAME" pip install \
  msgpack==0.6.2 \
  bitstring==4.1.4 \
  reedsolo==1.7.0 \
  petlib \
  bls-lib==1.0.1 \
  "git+https://github.com/jfairoze/bplib.git#egg=bplib"

echo "Checking key imports."
conda run -n "$ENV_NAME" python -c "import torch, transformers, datasets, nltk, bitstring, reedsolo, bplib, bls, petlib; print('baseline env ok')"

