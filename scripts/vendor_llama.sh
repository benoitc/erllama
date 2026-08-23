#!/usr/bin/env bash
# Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
# See the LICENSE file at the project root.
#
# Vendor an upstream llama.cpp release tag into c_src/llama.cpp.
#
#   scripts/vendor_llama.sh b10068
#
# The vendored tree is upstream, unmodified. Only whole directories are
# removed (backends we never build, unused vendor libs, repo dotfiles);
# no file is edited. The script ends by diffing the result against the
# tarball and fails if any file differs from upstream.
set -euo pipefail

TAG="${1:-}"
if [ -z "$TAG" ]; then
    echo "usage: $0 <llama.cpp tag, e.g. b10068>" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/c_src/llama.cpp"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

URL="https://github.com/ggml-org/llama.cpp/archive/refs/tags/${TAG}.tar.gz"
echo "fetching $URL"
curl -fsSL "$URL" | tar -xz -C "$WORK"
SRC="$WORK/llama.cpp-${TAG}"

# Paths we vendor wholesale.
KEEP="cmake CMakeLists.txt common ggml include src LICENSE vendor"

# ggml backends we do not build (see c_src/CMakeLists.txt).
PRUNE_GGML="ggml-cann ggml-et ggml-hexagon ggml-hip ggml-musa ggml-opencl \
ggml-openvino ggml-rpc ggml-sycl ggml-virtgpu ggml-vulkan ggml-webgpu \
ggml-zdnn ggml-zendnn"

# vendor/ is kept whole: since b10593 upstream builds it as CMake targets
# (vendor/CMakeLists.txt) that common/ links against.
PRUNE_VENDOR=""

rm -rf "$DEST"
mkdir -p "$DEST"
for p in $KEEP; do
    cp -R "$SRC/$p" "$DEST/$p"
done
for d in $PRUNE_GGML; do
    rm -rf "$DEST/ggml/src/$d"
done
for d in $PRUNE_VENDOR; do
    rm -rf "$DEST/vendor/$d"
done
find "$DEST" -name .gitignore -delete
echo "$TAG" > "$DEST/.version"

# Verify: every vendored file must be byte-identical to upstream.
CHANGED="$(for p in $KEEP; do
    diff -rq "$SRC/$p" "$DEST/$p" 2>/dev/null | grep -v "^Only in $SRC" || true
done)"
if [ -n "$CHANGED" ]; then
    echo "vendored tree differs from upstream $TAG:" >&2
    echo "$CHANGED" >&2
    exit 1
fi
echo "vendored llama.cpp $TAG into c_src/llama.cpp (upstream, unmodified)"
du -sh "$DEST"
