#!/usr/bin/env bash
# The walkthrough of guides/server/demo.md on a real model, through the
# `erllama' CLI: one release, several conversations, an agent session
# running a slow command, a client that leaves and comes back, the
# runtime view, one run cancelled, then a restart and a conversation
# resumed. erllama_server_demo_SUITE runs the same story on the stub.
#
#   rebar3 as prod release
#   scripts/demo.sh ~/Models/qwen2.5-1.5b-instruct-q4_k_m.gguf
#
# It starts its own release with a scratch data directory and stops it
# at the end. The model's answers vary; the release's behaviour does not.
set -euo pipefail

GGUF=${1:?usage: scripts/demo.sh MODEL.gguf}
R=${R:-_build/prod/rel/erllama_server}
E=$R/bin/erllama
DATA=$(mktemp -d)
export ERLLAMA_DATA_DIR=$DATA ERLLAMA_HTTP_PORT=${ERLLAMA_HTTP_PORT:-18100}
WS=$DATA/ws
mkdir -p "$WS"

say() { printf '\n== %s\n' "$*"; }
up() { for _ in $(seq 1 60); do "$R/bin/erllama_server" ping >/dev/null 2>&1 && return; sleep 0.5; done; exit 1; }
field() { "$E" inspect "$1" | awk -v k="$2" '$1 == k {print $2}'; }

if "$R/bin/erllama_server" ping >/dev/null 2>&1; then
    echo "a release is already running; stop it first" >&2
    exit 1
fi

say "1. one release, one model, four sequences"
"$R/bin/erllama_server" daemon
up
"$E" load "$GGUF" --name demo --n-ctx 8192 --parallel 4
trap '"$R/bin/erllama_server" stop >/dev/null 2>&1 || true' EXIT

say "2. two conversations, each with its own run"
A=$("$E" run demo --detach --max-tokens 2500 "List every number from 1 to 800, one per line.")
B=$("$E" run demo --detach --max-tokens 2500 "List every even number from 2 to 1600, one per line.")
echo "run A $A, run B $B"

say "3. an agent session asked to run a slow command"
printf '%s\n/quit\n' 'Use the shell tool to run exactly this command: sleep 20; echo slow done. Then tell me what it printed.' \
    | "$E" agent demo --workspace "$WS" --yes >"$DATA/agent.log" 2>&1 &
AGENT_CLI=$!
sleep 3

say "4. meanwhile, another conversation"
T0=$(date +%s)
"$E" run demo --max-tokens 20 "What is the capital of France? Answer in one word."
echo "[took $(( $(date +%s) - T0 )) s]"

say "5. a client follows run A, leaves, and another comes back to it"
"$E" attach "$A" >"$DATA/attach1.log" 2>&1 &
FOLLOWER=$!
sleep 1
# The client goes away (its process is killed), as a closed terminal.
kill -TERM $FOLLOWER 2>/dev/null || true
wait $FOLLOWER 2>/dev/null || true
echo "the first client saw $(wc -l <"$DATA/attach1.log" | tr -d ' ') lines, then went away"
"$E" attach "$A" --from 200 | head -3
echo "[the second client left too; run A goes on]"

say "6. what runs, what waits, what the agent does"
"$E" status
AGENT=$("$E" agents | awk '/^agent_/ {print $1; exit}')
[ -n "$AGENT" ] && "$E" inspect "$AGENT"

say "7. cancel run B; run A goes on to its end"
"$E" cancel "$B"
"$E" runs
while [ "$(field "$A" status)" = running ]; do sleep 1; done
"$E" result "$B" | tail -1
"$E" result "$A" | tail -1
wait $AGENT_CLI || true
echo "the agent session's log:"
tail -6 "$DATA/agent.log"

say "8. restart the release and go on with conversation A"
CONV=$(field "$A" conversation)
"$R/bin/erllama_server" stop >/dev/null
"$R/bin/erllama_server" daemon
up
"$E" conversations
"$E" run demo --conversation "$CONV" --max-tokens 40 "What was the last number you wrote? Answer with the number only."
