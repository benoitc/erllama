#!/usr/bin/env bash
# barrel_inference bench, collect mode.
#
# Runs a fixed set of workloads against one GGUF and writes a single
# JSON file describing host, GPU, model, and per-workload Stats. The
# file is suitable for off-line aggregation across machines and models.
#
# Usage:
#   bench/collect.sh <model-gguf-path> [out-json-path]
#
# Defaults output to:
#   bench/results/<gpu-slug>__<model-basename>__barrel_inference-<vsn>__<utc-ts>.json
#
# Tunables (env vars):
#   N_GPU_LAYERS          how many model layers to offload to GPU (default 999)
#   N_CTX                 llama_context size (default 4096)
#   N_BATCH               llama_decode batch size (default 4096)
#   N_SEQ_MAX             parallel sequence slots (default 1)
#   BENCH_SHORT_TOKENS    short-prompt target length (default 50)
#   BENCH_LONG_TOKENS     long-prompt target length (default 500)
#   BENCH_RESPONSE_TOKENS tokens to generate per workload (default 32)
#   SKIP_SHA256=1         skip sha256 of the GGUF (faster, but loses
#                         cross-machine model identity)

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
    echo "usage: bench/collect.sh <gguf-path> [out-json-path]" >&2
    exit 2
fi
if [[ ! -f "$MODEL" ]]; then
    echo "model not found: $MODEL" >&2
    exit 2
fi
OUT="${2:-}"

# ----- host detection -------------------------------------------------------

KERNEL="$(uname -s)"
ARCH="$(uname -m)"
HOST="$(hostname -s 2>/dev/null || hostname)"
case "$KERNEL" in
    Darwin)
        RELEASE="$(sw_vers -productVersion 2>/dev/null || uname -r)"
        CPU_BRAND="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
        PHYS_CORES="$(sysctl -n hw.physicalcpu 2>/dev/null || echo 0)"
        RAM_MB="$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1048576 ))"
        ;;
    Linux)
        RELEASE="$(uname -r)"
        CPU_BRAND="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//' || echo unknown)"
        PHYS_CORES="$(grep -c '^processor' /proc/cpuinfo 2>/dev/null || echo 0)"
        RAM_MB="$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0) / 1024 ))"
        ;;
    *)
        RELEASE="$(uname -r)"
        CPU_BRAND="unknown"
        PHYS_CORES=0
        RAM_MB=0
        ;;
esac

# ----- gpu detection --------------------------------------------------------

GPU_KIND="cpu"
GPU_NAME=""
GPU_MEMORY_MB=0
GPU_DRIVER=""

if command -v nvidia-smi >/dev/null 2>&1; then
    LINE="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
    if [[ -n "$LINE" ]]; then
        GPU_KIND="cuda"
        GPU_NAME="$(echo "$LINE" | cut -d, -f1 | sed 's/^ *//;s/ *$//')"
        GPU_MEMORY_MB="$(echo "$LINE" | cut -d, -f2 | sed 's/^ *//;s/ *$//')"
        GPU_DRIVER="$(echo "$LINE" | cut -d, -f3 | sed 's/^ *//;s/ *$//')"
    fi
fi

if [[ "$GPU_KIND" == "cpu" ]] && command -v rocm-smi >/dev/null 2>&1; then
    GPU_KIND="rocm"
    GPU_NAME="$(rocm-smi --showproductname 2>/dev/null | awk -F: '/Card series/ {gsub(/^ +| +$/, "", $2); print $2; exit}')"
    GPU_DRIVER="$(rocm-smi --showdriverversion 2>/dev/null | awk -F: '/Driver version/ {gsub(/^ +| +$/, "", $2); print $2; exit}')"
fi

if [[ "$GPU_KIND" == "cpu" && "$KERNEL" == "Darwin" ]]; then
    # Apple Silicon: GPU is integrated, name from chip brand
    if [[ -x /usr/sbin/system_profiler ]]; then
        CHIPSET="$(/usr/sbin/system_profiler SPDisplaysDataType 2>/dev/null | awk -F: '/Chipset Model/ {gsub(/^ +| +$/, "", $2); print $2; exit}')"
        if [[ -n "$CHIPSET" ]]; then
            GPU_KIND="metal"
            GPU_NAME="$CHIPSET"
            GPU_MEMORY_MB="$RAM_MB"  # unified memory
        fi
    fi
fi

if [[ -z "$GPU_NAME" ]]; then
    GPU_NAME="$CPU_BRAND"
fi

# ----- model metadata -------------------------------------------------------

MODEL_BASENAME="$(basename "$MODEL")"
MODEL_SLUG="$(echo "$MODEL_BASENAME" | sed 's/\.gguf$//' | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9.-]/-/g')"
MODEL_SHA256=""
if [[ "${SKIP_SHA256:-0}" != "1" ]]; then
    echo "computing sha256 of $MODEL_BASENAME (set SKIP_SHA256=1 to skip)..." >&2
    if command -v shasum >/dev/null 2>&1; then
        MODEL_SHA256="$(shasum -a 256 "$MODEL" | awk '{print $1}')"
    elif command -v sha256sum >/dev/null 2>&1; then
        MODEL_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')"
    fi
fi

# ----- output path ----------------------------------------------------------

GPU_SLUG="$(echo "$GPU_NAME" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g;s/^-//;s/-$//' | cut -c1-40)"
[[ -z "$GPU_SLUG" ]] && GPU_SLUG="cpu"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
VSN="$(awk -F'"' '/vsn,/ {print $2; exit}' src/barrel_inference.app.src 2>/dev/null || echo unknown)"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if git diff --quiet 2>/dev/null && git diff --cached --quiet 2>/dev/null; then
    GIT_DIRTY="clean"
else
    GIT_DIRTY="dirty"
fi

if [[ -z "$OUT" ]]; then
    mkdir -p bench/results
    OUT="bench/results/${GPU_SLUG}__${MODEL_SLUG}__barrel_inference-${VSN}__${TS}.json"
fi

# ----- build + run ----------------------------------------------------------

echo "building barrel_inference..." >&2
rebar3 compile >/dev/null

mkdir -p _build/bench/ebin
erlc -o _build/bench/ebin -I include -pa _build/default/lib/barrel_inference/ebin \
    bench/barrel_inference_bench_collect.erl

EBIN_PATHS=$(find _build/default/lib -maxdepth 2 -name ebin -type d | xargs -I {} echo "-pa {}")

echo "host=$HOST kernel=$KERNEL arch=$ARCH cpu=\"$CPU_BRAND\" cores=$PHYS_CORES ram_mb=$RAM_MB" >&2
echo "gpu=$GPU_KIND name=\"$GPU_NAME\" memory_mb=$GPU_MEMORY_MB driver=\"$GPU_DRIVER\"" >&2
echo "model=$MODEL_BASENAME sha256=${MODEL_SHA256:-<skipped>}" >&2
echo "writing to: $OUT" >&2

BARREL_INFERENCE_BENCH_OS_KERNEL="$KERNEL" \
BARREL_INFERENCE_BENCH_OS_RELEASE="$RELEASE" \
BARREL_INFERENCE_BENCH_ARCH="$ARCH" \
BARREL_INFERENCE_BENCH_CPU_BRAND="$CPU_BRAND" \
BARREL_INFERENCE_BENCH_PHYSICAL_CORES="$PHYS_CORES" \
BARREL_INFERENCE_BENCH_RAM_MB="$RAM_MB" \
BARREL_INFERENCE_BENCH_GPU_KIND="$GPU_KIND" \
BARREL_INFERENCE_BENCH_GPU_NAME="$GPU_NAME" \
BARREL_INFERENCE_BENCH_GPU_MEMORY_MB="$GPU_MEMORY_MB" \
BARREL_INFERENCE_BENCH_GPU_DRIVER="$GPU_DRIVER" \
BARREL_INFERENCE_BENCH_MODEL_SHA256="$MODEL_SHA256" \
BARREL_INFERENCE_BENCH_GIT_COMMIT="$GIT_COMMIT" \
BARREL_INFERENCE_BENCH_GIT_DIRTY="$GIT_DIRTY" \
HOSTNAME="$HOST" \
    erl -noinput -boot start_clean \
        $EBIN_PATHS -pa _build/bench/ebin \
        -run barrel_inference_bench_collect main "$MODEL" "$OUT" \
        -run init stop
