#!/bin/sh
set -eu
if [ ! -f "$HOME/.erlang.cookie" ]; then
    (umask 077 && openssl rand -hex 32 > "$HOME/.erlang.cookie")
fi
exec "$@"
