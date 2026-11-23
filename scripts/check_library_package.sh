#!/bin/sh
# Prove that erllama builds and works as a standalone library from the
# Hex package this repository produces, without any server dependency.
#
#   1. Build the erllama package (`rebar3 hex build --app erllama`).
#   2. Use it from a throwaway consumer project (_checkouts), compile
#      the NIF there and run a test that starts erllama, calls the NIF
#      and runs a completion on the stub backend.
#   3. Fail if the consumer pulled in livery or barrel_mcp.
#
# Run from the repository root. Needs the same toolchain as the build
# (OTP 28, rebar3, CMake, a C/C++ compiler).
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$ROOT"

rebar3 hex build --app erllama --unpack >/dev/null
PKG=$(ls -d "$ROOT"/_build/default/lib/erllama/hex/erllama-* | head -n 1)
VSN=${PKG##*/erllama-}
echo "package: $PKG"

WORK=$(mktemp -d "${TMPDIR:-/tmp}/erllama-consumer.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/_checkouts" "$WORK/src" "$WORK/test"
cp -R "$PKG" "$WORK/_checkouts/erllama"

cat > "$WORK/rebar.config" <<EOF
{deps, [{erllama, "$VSN"}]}.
EOF

cat > "$WORK/src/consumer.app.src" <<'EOF'
{application, consumer, [
    {description, "erllama package consumer check"},
    {vsn, "0.0.1"},
    {applications, [kernel, stdlib, erllama]}
]}.
EOF

cat > "$WORK/test/consumer_tests.erl" <<'EOF'
-module(consumer_tests).
-include_lib("eunit/include/eunit.hrl").

library_works_standalone_test_() ->
    {timeout, 120, fun() ->
        {ok, _} = application:ensure_all_started(erllama),
        ?assert(is_binary(erllama_nif:kv_format())),
        {ok, M} = erllama:load_model(#{backend => erllama_model_stub}),
        ?assertMatch({ok, #{reply := _}}, erllama:complete(M, <<"one two three">>)),
        ok = erllama:unload(M),
        ok = application:stop(erllama)
    end}.
EOF

cd "$WORK"
rebar3 eunit

for dep in livery barrel_mcp; do
    if [ -e "_build/default/lib/$dep" ] || [ -e "_build/test/lib/$dep" ]; then
        echo "FAIL: the erllama package pulled in $dep" >&2
        exit 1
    fi
done
echo "erllama $VSN builds and runs standalone; no server dependency"
