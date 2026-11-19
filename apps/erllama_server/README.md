# erllama_server

The erllama_server release owns loaded models, KV state, conversations,
agent runs and MCP connections, and serves them to a terminal client
and to HTTP clients. You need this page if you want to run the server
rather than embed the [erllama library](../erllama/README.md) in your
own application.

This app is being built in phases; the current state is tracked in
[doc/features.md](../../doc/features.md).

## Build and run the release

From the repository root:

```bash
rebar3 release
_build/default/rel/erllama_server/bin/erllama_server daemon
_build/default/rel/erllama_server/bin/erllama_server ping
_build/default/rel/erllama_server/bin/erllama_server stop
```

The node is named `erllama_server@127.0.0.1` and Erlang distribution
listens on the loopback interface only. The cookie is not stored in
the release: the node and the control commands (`ping`, `stop`) use
`~/.erlang.cookie` of the user that runs them. `RELX_COOKIE` only
changes the cookie the control commands use, not the node's own.

For a self-contained release that carries its own Erlang runtime, use
`rebar3 as prod release` (output in `_build/prod/rel/erllama_server`).

## Talk to the release with the CLI

`bin/erllama` is the command-line client shipped beside the release.
It runs on the release's own runtime when the release carries one, and
connects to the running node from a temporary hidden node. It never
starts the inference application or loads the NIF itself.

```bash
R=_build/prod/rel/erllama_server
$R/bin/erllama status
$R/bin/erllama load ~/Models/tinyllama-1.1b-chat.gguf --id tiny --n-ctx 2048
$R/bin/erllama run tiny "Say hello in one short sentence."
RUN=$($R/bin/erllama run tiny --max-tokens 400 --detach "Count to a hundred.")
$R/bin/erllama attach "$RUN"
$R/bin/erllama cancel "$RUN"
$R/bin/erllama help
```

Runs belong to the release: they keep going when the CLI exits, and
you can attach to them again by id. See
[Driving the release from the CLI](../../guides/server/rpc.md) for
conversations, reattachment, the node and cookie setup, TLS for remote
administration, and what each error means.

## Configuration

The `erllama_server` application environment (`config/sys.config` in
the release). A bad value stops the release from starting.

| Key | Default | Meaning |
|---|---|---|
| `run_buffer_events` | `1024` | events kept per run for reattachment |
| `subscriber_max_unacked` | `256` | unread events after which a slow client is cut off (it reattaches) |
| `run_retention_ms` | `300000` | how long a finished run stays available to `attach` and `result` |
| `max_active_runs` | `64` | generating runs at once; more are refused |

The library's own settings (cache tiers, scheduler) go under `erllama`;
see the [configuration guide](../../guides/configuration.md).

## Where the code is

| Module | Role |
|---|---|
| `erllama_server_api` | the versioned service API every front end calls |
| `erllama_server_run` | one run: owns the erllama stream, numbers and buffers its events, serves subscribers |
| `erllama_server_runs` | the table of runs; starts them and enforces `max_active_runs` |
| `erllama_server_convs` | conversations: history, last cache key, one active run each |
| `erllama_server_generate` | renders chat or prompt requests into an erllama stream and parses the result |
| `erllama_server_client` | the client side, used by the CLI (no erllama module) |
| `erllama_server_cli` | the `erllama` command: argument parsing and output |

Tests: `test/erllama_server_rpc_SUITE.erl` drives the API from a
separate hidden node as the CLI does; `test/erllama_server_tls_SUITE.erl`
does the same over TLS distribution.
