# Operating the release

The release is one long-running service that owns models, conversations,
runs and agent sessions, shared by every client. This page shows you
what it is doing, what a failure affects and what the release does
about it, and what survives a restart. You need it to watch a release
in use, to understand an error a client saw, and to know what to expect
when something crashes.

## What runs, what waits, what failed

```bash
R=_build/prod/rel/erllama_server
$R/bin/erllama status
```

```text
release   erllama_server@127.0.0.1
versions  erllama_server 0.2.0, erllama 0.14.0, api 2
memory    36401 of 49152 MiB used (system)
models
  demo:latest  running  ctx 8192  2 of 4 sequences busy
runs      2 running: 2 generating
agents
  agent_d65f1d43c1caaef7  tools  conv_5f134fc1dd11eca2, 1 tool call(s) running
mcp       (no server configured)
cache     0 hit(s), 4 miss(es), 1 save(s), 0 eviction(s); 29.4 MiB in RAM, 0.0 MiB on disk
failures  (none)
```

- `models`: each loaded model, how many of its sequences are busy, and
  how many requests wait for one (`N waiting for one`).
- `runs`: running runs by phase: `loading` the model, `compacting` the
  conversation, `waiting for a sequence`, `generating`.
- `agents`: each session's state (`idle`, `generating`, `tools`,
  `approving`, `compacting`), approvals pending, tool calls and
  background jobs running, messages queued.
- `mcp`: each MCP server, `ready` with its tools, or down with the
  reason.
- `cache`: hits, misses, saves and evictions since the release
  started, and the bytes each tier holds now.
- `failures`: the latest failures, each with what the release did next.

One job in detail:

```bash
$R/bin/erllama runs
$R/bin/erllama inspect agent_d65f1d43c1caaef7
```

`inspect` shows a run's phase and next event, or a session's state,
approvals with their arguments, tools, jobs, the turn's use of its
budgets and where its latest turn starts in the event stream.

Over HTTP (with an API key when keys are configured):

| Path | Gives |
|---|---|
| `GET /status` | the same as `erllama status`, as JSON |
| `GET /metrics` | Prometheus text: HTTP metrics, erllama's cache counters, `erllama_server_models_loaded`, `erllama_server_active_runs`, `erllama_server_waiting_runs`, `erllama_server_active_pulls`, `erllama_server_failures_total{kind}` |
| `GET /health`, `GET /ready` | the process is up; the engine and the model registry answer (no key needed) |

From Erlang, `erllama_server_api:status/0`, `inspect/1` and
`failures/1` return the same maps.

## What a failure affects

Every run, agent session, tool call, MCP connection and model is its
own process. A failure ends the process it happened in; the process
that owns it handles it, records it in `failures` and the others go on.

| What fails | What it affects | What happens next | Test |
|---|---|---|---|
| a run (bad option, model not loaded, a native error) | that run | it ends with an `error` event; its turn is not added to the conversation | `erllama_server_rpc_SUITE` |
| a model process (crash, kill) | the runs on that model | they end with `{model_down, Reason}`; runs on other models go on; the next request loads the model again | `erllama_server_ops_SUITE:model_crash_stays_with_its_model` |
| a request the model's native code rejects | that request | `{error, badarg}`; the model keeps serving | `erllama_real_model_SUITE:bad_request_leaves_the_model_running` |
| a tool call's process | that call | it gets an error result and the turn goes on | `erllama_server_ops_SUITE:tool_crash_stays_with_its_call` |
| a shell command past its timeout | that call | its process group is killed; the call gets an error result | `erllama_server_ops_SUITE:tool_timeout_is_recorded` |
| an MCP server (stopped, unreachable) | calls to its tools | they get error results after `mcp_timeout_ms`; a failed call makes the release check the server; once it is found down its tools are withdrawn and it is reconnected with backoff | `erllama_server_mcp_SUITE:server_down_and_back` |
| a CLI or Erlang client | nothing | the run or session goes on; attach again to follow it | `erllama_server_rpc_SUITE:run_survives_client_disconnect` |
| an HTTP client that disconnects | its request | the run is cancelled (an HTTP response has no one to go to) | `erllama_server_http_SUITE:disconnect_cancels_the_run` |
| an agent session's process | that session | its conversation is released and repaired; other sessions go on; a new session can resume the conversation | `erllama_server_ops_SUITE:session_crash_stays_with_its_session` |
| the release (stop, crash of the VM) | everything running | conversations are kept; on restart, a run or session that was cut is closed, its tool calls that may have run are marked unresolved and never run again | `erllama_server_convs_SUITE` |

Native code is the limit of this isolation. llama.cpp and the NIF run
inside the BEAM: a crash or abort there (a segfault, a failed GGML
assertion) ends the whole VM, with every model, run and session in it,
as a release crash does. Dirty schedulers keep long native calls off
the normal schedulers; they do not isolate crashes.

## Reattach, recover, restart

Three different things, often confused:

| | When | What you get | What you do not get |
|---|---|---|---|
| reattach | the run or session is still running | its events from where you choose, replayed from a bounded buffer (`run_buffer_events`, 1024), then live; a gap is reported when the start is gone | events older than the buffer |
| recover | the release restarted | every conversation, as it was at its last completed step; a cut turn is closed, unresolved tool calls are marked | the computation that was in flight: a cut run is not resumed |
| restart work | after recovery | a new run or session on the conversation (`--conversation ID`) | automatic retries: a tool call that may have run is never run again |

See [Driving the release from the CLI](rpc.md) for `attach`, [Agent
sessions](agent.md#come-back-to-a-session) for `agent --attach` and
[Conversations](conversations.md) for what is kept.

## Limits

- Requests on one model share its hardware: more of them at once means
  each goes slower. A model serves `parallel` requests at once; the
  release refuses more runs than `max_active_runs` (HTTP 429).
- The failure log keeps the last 200 entries, in memory: a restart
  clears it. The conversation logs keep what must survive.
- An MCP outage shows once a call fails or the periodic refresh runs,
  bounded by the MCP client's own timeouts (about 30 s for an HTTP
  server that stopped).
