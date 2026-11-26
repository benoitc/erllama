#!/bin/sh
# The packaging checkout may be newer than the application source tag.
set -eu
scripts=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
version=$1
platform=$2
out=$3
case "$platform:$(uname -sm)" in
    "macos-arm64:Darwin arm64"|"linux-arm64:Linux aarch64"|"linux-amd64:Linux x86_64"|"freebsd-amd64:FreeBSD amd64") ;;
    *) echo "Unexpected builder for $platform: $(uname -sm)" >&2; exit 1 ;;
esac
export ERLLAMA_OPTS='-DGGML_NATIVE=OFF'
rebar3 as prod release
rel=_build/prod/rel/erllama_server
grep -F "{vsn, \"$version\"}" apps/erllama/src/erllama.app.src >/dev/null
test -x "$rel/bin/erllama"
test -x "$rel/bin/erllama_server"
test -f "$rel/lib/erllama-$version/priv/erllama_nif.so"
mkdir -p "$out"
export COPYFILE_DISABLE=1
tar -czf "$out/erllama-$version-$platform.tar.gz" -C "$rel" .
"$scripts/check-archive.sh" "$version" "$out/erllama-$version-$platform.tar.gz" "$scripts/install.sh"
