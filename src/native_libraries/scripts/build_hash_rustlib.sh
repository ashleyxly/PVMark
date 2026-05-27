#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUST_CRATE="${RUST_CRATE:-${REPO_ROOT}/artifact/third_party/hash_function/hash-function}"
ENV_NAME="${ENV_NAME:-synthid}"

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found. Install Rust first: https://rustup.rs/" >&2
  exit 127
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH" >&2
  exit 127
fi

if [[ ! -f "${RUST_CRATE}/Cargo.toml" ]]; then
  echo "Rust crate not found: ${RUST_CRATE}" >&2
  exit 2
fi

echo "[build_hash_rustlib] repo=${REPO_ROOT}"
echo "[build_hash_rustlib] rust_crate=${RUST_CRATE}"
echo "[build_hash_rustlib] conda_env=${ENV_NAME}"

cd "${RUST_CRATE}"
cargo build --release

SITE_PACKAGES="$(conda run -n "${ENV_NAME}" python -c 'import site; print(site.getsitepackages()[0])')"
EXT_SUFFIX="$(conda run -n "${ENV_NAME}" python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')"
PKG_DIR="${SITE_PACKAGES}/hash_rustlib"

mkdir -p "${PKG_DIR}"
cp "${RUST_CRATE}/target/release/libhash_rustlib.so" "${PKG_DIR}/hash_rustlib${EXT_SUFFIX}"
cat > "${PKG_DIR}/__init__.py" <<'PY'
from .hash_rustlib import *
PY

conda run -n "${ENV_NAME}" python -c "import hash_rustlib; print(hash_rustlib.__file__); print('hash_rustlib ok')"
