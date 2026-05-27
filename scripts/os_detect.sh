#!/usr/bin/env bash
# Common OS detection and cross-platform command aliases.
# Source this file in other scripts: source "$(dirname "${BASH_SOURCE[0]}")/os_detect.sh"

# Detect OS
OS_TYPE="$(uname -s)"
case "$OS_TYPE" in
    Linux*)  OS_LINUX=1; OS_MACOS=0 ;;
    Darwin*) OS_LINUX=0; OS_MACOS=1 ;;
    *)       OS_LINUX=0; OS_MACOS=0; echo "[WARN] Unsupported OS: $OS_TYPE" >&2 ;;
esac

# Cross-platform timeout command
# Linux: timeout (GNU coreutils)
# macOS: gtimeout (brew install coreutils) or fallback to perl
if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_CMD="gtimeout"
else
    # Fallback: no timeout support
    TIMEOUT_CMD=""
fi

run_with_timeout() {
    local seconds="$1"; shift
    if [ -n "$TIMEOUT_CMD" ]; then
        "$TIMEOUT_CMD" "$seconds" "$@"
    else
        "$@"
    fi
}

# Cross-platform sed -i (in-place edit)
# Linux: sed -i
# macOS: sed -i ''
sed_inplace() {
    if [ "$OS_MACOS" -eq 1 ]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

# Cross-platform sha256
# Linux: sha256sum
# macOS: shasum -a 256
sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$@"
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$@"
    else
        echo "ERROR: No sha256 tool found" >&2
        return 1
    fi
}

# Cross-platform nproc
get_nproc() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif [ "$OS_MACOS" -eq 1 ]; then
        sysctl -n hw.ncpu 2>/dev/null || echo 4
    else
        echo 4
    fi
}

# Cross-platform package manager check
has_package_manager() {
    local pm="$1"
    command -v "$pm" >/dev/null 2>&1
}

# Cross-platform temp directory
get_tmpdir() {
    echo "${TMPDIR:-/tmp}"
}
