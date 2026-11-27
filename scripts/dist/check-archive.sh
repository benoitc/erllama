#!/bin/sh
# Install from disk on a clean runner and exercise the bundled runtime and NIF.
set -eu
version=$1
file=$2
installer=$3
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export HOME="$tmp/home" ERLLAMA_DATA_DIR="$tmp/data" ERLLAMA_HTTP_PORT=18552 ERLLAMA_TIMEOUT=90000
mkdir -p "$HOME" "$tmp/download"
(umask 077 && head -c 24 /dev/urandom | base64 > "$HOME/.erlang.cookie")
cp "$file" "$tmp/download/"
name=$(basename "$file")
(cd "$tmp/download" && openssl dgst -sha256 -r "$name" | awk -v name="$name" '{print $1 "  " name}' > SHA256SUMS)
mv "$tmp/download/SHA256SUMS" "$tmp/good-sums"
printf '%064d  %s\n' 0 "$name" > "$tmp/download/SHA256SUMS"
if sh "$installer" --version "$version" --prefix "$tmp/prefix" --from "$tmp/download"; then
    echo 'Installer accepted a checksum mismatch' >&2
    exit 1
fi
test ! -e "$tmp/prefix/bin/erllama"
mv "$tmp/good-sums" "$tmp/download/SHA256SUMS"
sh "$installer" --version "$version" --prefix "$tmp/prefix" --from "$tmp/download"
bin=$tmp/prefix/bin
"$bin/erllama_server" foreground > "$tmp/server.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" || true; cat "$tmp/server.log"; rm -rf "$tmp"' EXIT
i=0
until curl -fsS "http://127.0.0.1:$ERLLAMA_HTTP_PORT/ready" >/dev/null 2>&1; do
    kill -0 "$server" 2>/dev/null || exit 1
    i=$((i+1))
    [ "$i" -lt 60 ] || exit 1
    sleep 1
done
"$bin/erllama" ping
"$bin/erllama_server" eval 'true = is_binary(erllama_nif:kv_format()).' | grep true
"$bin/erllama" status
"$bin/erllama_server" stop
wait "$server"
trap 'rm -rf "$tmp"' EXIT
