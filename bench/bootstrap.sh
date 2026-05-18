#!/usr/bin/env bash
# One-shot bootstrap: clone erllama, build it, run the collect bench.
# Designed for operators benchmarking erllama on hardware that
# doesn't already have a checkout. Safe to re-run; reuses the clone.
#
# Usage (either form works):
#
#   curl -fsSL https://raw.githubusercontent.com/erllama/erllama/main/bench/bootstrap.sh | bash -s -- /path/to/model.gguf
#
#   bench/bootstrap.sh /path/to/model.gguf [out-json-path]
#
# Env knobs:
#   ERLLAMA_REF        git ref to check out (default: main)
#   ERLLAMA_DIR        clone target (default: $HOME/.erllama-bench/erllama)
#   N_GPU_LAYERS       passed through to collect.sh (default: 999)
#   SKIP_SHA256=1      skip the GGUF sha256 (faster, less identity info)
#
# Prerequisites on the host: git, cmake (>= 3.18), a C++17 compiler,
# rebar3 (>= 3.25), erlang/OTP 28. On CUDA hosts: nvidia-smi + CUDA
# toolkit available to llama.cpp's CMake. On Apple Silicon: nothing
# extra, Metal is built by default.

set -euo pipefail

MODEL="${1:-}"
OUT="${2:-}"
if [[ -z "$MODEL" ]]; then
    echo "usage: $0 <gguf-path> [out-json-path]" >&2
    exit 2
fi
if [[ ! -f "$MODEL" ]]; then
    echo "model not found: $MODEL" >&2
    exit 2
fi

REF="${ERLLAMA_REF:-main}"
DIR="${ERLLAMA_DIR:-$HOME/.erllama-bench/erllama}"
REPO="${ERLLAMA_REPO:-https://github.com/erllama/erllama.git}"

need() {
    command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 1; }
}
need git
need cmake
need rebar3
need erl

if [[ ! -d "$DIR/.git" ]]; then
    echo "cloning $REPO into $DIR..." >&2
    mkdir -p "$(dirname "$DIR")"
    git clone --depth 1 --branch "$REF" "$REPO" "$DIR"
else
    echo "updating existing clone at $DIR..." >&2
    git -C "$DIR" fetch --depth 1 origin "$REF"
    git -C "$DIR" checkout -q "$REF"
    git -C "$DIR" reset -q --hard "origin/$REF" || git -C "$DIR" reset -q --hard "$REF"
fi

# Re-exec the in-tree collect.sh with the same args. collect.sh
# handles `rebar3 compile` and host/GPU detection.
exec bash "$DIR/bench/collect.sh" "$MODEL" ${OUT:+"$OUT"}
