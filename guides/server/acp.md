# ACP: `erllama acp`

`erllama acp` runs the server's agent sessions as an ACP agent
(Agent Client Protocol) on stdio — JSON-RPC 2.0 messages one per line
(the ACP wire framing), protocol version 1; LSP-style `Content-Length`
frames are also accepted on input. Editors and agent clients that
embed ACP agents (Zed, OpenCode, ...) can then drive local models from
a running `erllama_server` release.

```bash
rebar3 as prod release
R=_build/prod/rel/erllama_server
$R/bin/erllama_server daemon
$R/bin/erllama load ~/Models/tiny.gguf --name fast --n-ctx 4096
```

The command the client spawns:

```bash
$R/bin/erllama acp [--model NAME] [--workspace PATH] [--yes | --read-only]
```

It only proxies to the running release (`ERLLAMA_NODE` and
`ERLLAMA_COOKIE`, like every other `erllama` command). Protocol
messages go over stdin/stdout; diagnostics go to standard error.

## Surface

- `initialize`, `authenticate`, `logout` — answered locally; local
  models need no authentication.
- `session/new` — starts one agent session on the release. The
  client's working directory (the project checkout, a git worktree)
  becomes the session workspace on the release machine, so the
  workspace tools resolve inside it. `--workspace` overrides it;
  `--model`, `--yes` and `--read-only` set the defaults for new
  sessions.
- `session/prompt` and `session/cancel`: a prompt runs one agent turn
  with `agent_send/2`; `session/cancel` interrupts the turn
  (`agent_interrupt/1`) and the prompt is answered with
  `"cancelled"`. Text chunks, tool calls (with their tool call ids and
  kinds), and thoughts stream as `session/update` notifications.
- Models are exposed as a `configOptions` select listing the release's
  models; `session/set_config_option` switches mid-session
  (`agent_set_model/2`; unknown model → `-32602`). The `default`, `yes`
  and `read_only` modes map onto the release permission policies with
  `session/set_mode` (`agent_set_policy/2`).
- Permission requests (`approval_request` job events) surface as
  `session/request_permission`; the selected option maps to allow or
  deny through `agent_approve/3`.
- `session/load`: the ACP session id is the release conversation id
  (stable across restarts). `session/load` reattaches to the live
  agent or resumes the conversation with its stored workspace and
  model, replaying the history as session updates.
- `session/list`, `session/close`, `session/delete`, plus errors:
  request before initialize `-32002`, unknown session `-32000`,
  bad params `-32602`, unknown method `-32601`.

## Call paths (source map)

- `apps/erllama_server/src/erllama_server_acp_stdio.erl` — framing.
  Reading happens on an `{fd, ...}` port so any kind of stdin (pipe,
  redirected file, tty) deliver messages with an explicit end-of-input
  instead of blocking in the io server; the buffer is drained eagerly
  after every frame so a burst all reaches the dispatcher before
  end of input. Writes go to the group leader device (`io:put_chars/2`)
  — the `standard_output` atom is refused when the VM runs spawned.
- `erllama_server_acp.erl` — the dispatcher: one spawned process
  owning the session table; each request handled in its own spawned
  process (monitored, so end of input drains answers still in flight
  before reporting `acp_done`); job events from the followers become
  `session/update` notifications, permission requests and prompt
  answers.
- `erllama_server_acp_map.erl` — the pure mapping layer: request
  params to release specs, job events to `session/update` payloads,
  turn reasons to stop reasons, JSON-RPC envelopes and error codes.
- `erllama_server_acp_session.erl` — one follower per session, attached
  with `erllama_server_client:follow/4` from cursor 0, forwarding job
  events to the dispatcher.
- `erllama_server_cli_acp.erl` — the `erllama acp` command, registered
  in `erllama_server_cli.erl` (`--model`, `--workspace`, `--yes`,
  `--read-only`).
- Server-side additions: `erllama_server_api.erl` (`agent_load/1`,
  `agent_set_model/2`, `agent_set_policy/2`, `agent_sessions/0`;
  `agent_start/1` resumes a conversation with its stored model),
  `erllama_server_agent.erl` (`set_model/2`, `set_policy/2`).

## Testing

- `apps/erllama_server/test/erllama_server_acp_tests.erl` — eunit over
  the framing and every mapping, including the JSON-RPC error codes.
- `apps/erllama_server/test/erllama_server_acp_SUITE.erl` (Common Test,
  stub models so it runs everywhere): a release peer plus a hidden
  client peer drive the dispatcher through the whole surface, plus one
  case spawning the CLI as an OS process over redirected stdio and
  asserting the framed answers and a clean exit at end of input.
- `scripts/acp_e2e_opencode.sh` — a real end-to-end run against
  OpenCode (see below), gated behind `ERLLAMA_ACP_E2E=1`.

## Trying it with OpenCode

With the `opencode-acp-provider` plugin, a project-level `opencode.json`
registers the agent:

```json
{
  "plugins": [
    {
      "package": "opencode-acp-provider",
      "options": {
        "servers": [
          {
            "id": "erllama",
            "name": "erllama",
            "command": ["/path/to/erllama_server/rel/bin/erllama", "acp"]
          }
        ]
      }
    }
  ]
}
```

`scripts/acp_e2e_opencode.sh` wires this up in a scratch git worktree,
stages a renamed copy of the release (a private node name, so it never
collides with a running release), starts it with a real model, and runs
`opencode run -m acp-erllama/<model>` with a prompt that edits a file,
asserting the edit landed in the worktree. The first run can lose the
provider-registration race inside the OpenCode service (its failed run
is expected); registration is polled through the service API and the
next run selects the model.

```bash
rebar3 as prod release
ERLLAMA_ACP_E2E=1 scripts/acp_e2e_opencode.sh ~/Models/tinyllama-1.1b-chat.gguf
```
