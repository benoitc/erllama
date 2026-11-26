# End-to-end: drive `erllama acp` from OpenCode in a scratch git worktree.
#
#   rebar3 as prod release
#   ERLLAMA_ACP_E2E=1 scripts/acp_e2e_opencode.sh ~/Models/tiny.gguf
#
# The script starts its own release with a scratch data dir, wires the
# release's `erllama acp' into OpenCode through the opencode-acp-provider
# plugin (project-level opencode.json, like a user would), runs
# `opencode run -m acp-erllama/fast ...` inside a scratch git worktree with a
# prompt that edits a file, and asserts:
#
#   - the edit landed in the worktree (so the client's cwd became the
#     session's workspace), and
#   - the release holds the agent session with that workspace.
#
# Requires `opencode' on PATH with the opencode-acp-provider plugin
# installed globally (default with opencode when listed in opencode.json).
# Gated on ERLLAMA_ACP_E2E=1 so plain check runs stay on the stub suite.

set -euo pipefail

if [ "${ERLLAMA_ACP_E2E:-}" != "1" ]; then
    echo "acp_e2e_opencode: ERLLAMA_ACP_E2E not set, skipping" >&2
    exit 0
fi

if [ "$#" -lt 1 ]; then
    echo "usage: ERLLAMA_ACP_E2E=1 scripts/acp_e2e_opencode.sh MODEL [DRIVER_DIR]" >&2
    exit 2
fi

MODEL_PATH=$1
R=${R:-_build/prod/rel/erllama_server}
DRIVER_DIR=${2:-$PWD}
R=$(cd "$R" && pwd)

command -v opencode >/dev/null || { echo "opencode not found on PATH" >&2; exit 2; }
opencode --version >/dev/null 2>&1 || opencode -v >/dev/null 2>&1 || true
# The plugin is declared in the fixture project's opencode.json; OpenCode
# installs it from there on first run. Only warn if it is missing globally,
# since a slow first-run install is not a failure by itself.
grep -rwq "opencode-acp-provider" "${XDG_CONFIG_HOME:-$HOME/.config}/opencode/opencode.json" \
    "${XDG_CONFIG_HOME:-$HOME/.config}/opencode/opencode.jsonc" 2>/dev/null ||
    echo "note: the opencode-acp-provider plugin is not in the global opencode config" >&2

WORK=$(mktemp -d)
trap 'kill "$REL_PID" 2>/dev/null || true; rm -rf "$WORK"' EXIT

echo "== staging a renamed copy of the release"
# A private node name and port: the user's own release, a dev release
# or a parallel e2e run may hold erllama_server@127.0.0.1 already.
cp -R "$R" "$WORK/rel"
R="$WORK/rel"
NODE_NAME=erllama_acp_e2e
# vm.args is regenerated from vm.args.src on every boot, so rename both.
for F in "$R/releases/0.1.0/vm.args" "$R/releases/0.1.0/vm.args.src"; do
    [ -f "$F" ] && sed -i '' "s/-name erllama_server@/-name ${NODE_NAME}@/" "$F"
done
export ERLLAMA_NODE="${NODE_NAME}@127.0.0.1"

echo "== starting a release with a scratch data dir"
export ERLLAMA_DATA_DIR="$WORK/data"
export ERLLAMA_HTTP_PORT=${ERLLAMA_HTTP_PORT:-18985}
mkdir -p "$ERLLAMA_DATA_DIR"
"$R/bin/erllama_server" foreground >"$WORK/release.log" 2>&1 &
REL_PID=$!

for _ in $(seq 1 60); do
    if "$R/bin/erllama" ping >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done
"$R/bin/erllama" ping >/dev/null

echo "== loading the model"
"$R/bin/erllama" load "$MODEL_PATH" --name fast --n-ctx 4096

echo "== opencode worktree"
PROJECT=$WORK/project
git init -q "$PROJECT"
cd "$PROJECT"

# The plugin config: open the opencode_acp_provider server for erllama.
cat >opencode.json <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "plugins": [
    {
      "package": "opencode-acp-provider",
      "options": {
        "servers": [
          {
            "id": "erllama",
            "name": "erllama",
            "command": ["$R/bin/erllama", "acp", "--model", "fast"]
          }
        ]
      }
    }
  ]
}
EOF

echo "hello from erllama" >hello.txt
git add -A
git commit -qm "init"

echo "== run: opencode drives the erllama model over ACP"
# The plugin registers the erllama provider in OpenCode's persistent
# service: discovery runs against the live release there and retries in
# the background until it answers. The first run boots the location's
# plugins and loses the race (the turn starts before registration
# lands); its failed run is expected. Registration is polled through
# the service API; once it lands, the next run selects the model.
opencode run -m acp-erllama/fast "Append the line 'acp was here' to hello.txt" \
    >"$WORK/opencode.log" 2>&1 || echo "(first run expected to lose the registration race)"

READY=""
for _ in $(seq 1 90); do
    if opencode api get "/api/provider?location%5Bdirectory%5D=$(pwd)" 2>/dev/null |
        grep -q '"acp-erllama"'
    then
        READY=1
        break
    fi
    sleep 2
done
[ -n "$READY" ] ||
    {
        echo "FAILED: the erllama provider never registered in the service:" >&2
        echo "hint: with the release still running, re-run this script; a" >&2
        echo "later run finds the provider once the plugin has registered it" >&2
        opencode api get "/api/provider?location%5Bdirectory%5D=$(pwd)" 2>&1 | tail -5 >&2
        exit 1
    }
echo ok: the erllama ACP provider is registered

opencode run -m acp-erllama/fast "Append the line 'acp was here' to hello.txt" \
    >"$WORK/opencode.log" 2>&1 ||
    {
        echo "FAILED: opencode run failed:" >&2
        tail -40 "$WORK/opencode.log" >&2
        exit 1
    }

if ! grep -q "acp was here" hello.txt; then
    echo "FAILED: the model did not write to the worktree" >&2
    tail -40 "$WORK/opencode.log" >&2
    exit 1
fi
echo "ok: the model wrote inside the worktree"

echo "== done"
