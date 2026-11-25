# erllama_server

erllama_server is a local inference and agent service that keeps
running: one release loads your models and owns the conversations, runs
and agent sessions, and the `erllama` CLI, HTTP clients and other
Erlang nodes are clients of it. You need this page to run it. To call a
model from your own OTP application without a service, use the
[erllama library](../erllama/README.md) instead.

## What you get

- **Several clients share one loaded model.** A model serves `parallel`
  requests at once, batched into each decode step (default 4). Past the
  release's limits, work is refused (HTTP 429) instead of queued
  without end. See [Managing models](../../guides/server/models.md).
- **Clients come and go; the work stays.** A run or an agent session
  keeps going when its terminal closes. Another terminal can attach,
  replay what it missed and answer the approvals waiting for it.
  Conversations are kept on disk and survive a restart.
- **Failures stay with their owner.** Each run, agent session, tool
  call and MCP connection is its own process: a slow or crashed tool, a
  dead MCP server or a killed model affects only the work that used
  it, and `erllama status` shows what runs, what waits and what failed.
  See [Operating the release](../../guides/server/operations.md).
- **Computed prefixes are reused.** The KV cache is exact: a
  conversation's earlier turns and a long shared system prompt are
  restored instead of prefilled again.
- **The engine stays a library.** The same erllama runs inside your OTP
  application; the release is the service around it.

What it does not do: a crash in native code (llama.cpp, the NIF) ends
the whole VM; a restart keeps the conversations, not the computation
that was in flight; and requests on one model share its hardware, so
more at once means each is slower. The measurements behind these
points are in [the feature record](https://github.com/benoitc/erllama/blob/main/apps/erllama_server/doc/features.md), and
[A walkthrough](../../guides/server/demo.md) shows them on a real
model.

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
$R/bin/erllama pull hf://ggml-org/models/tinyllamas/stories260K.gguf
$R/bin/erllama load ~/Models/tinyllama-1.1b-chat.gguf --name tiny --n-ctx 2048
$R/bin/erllama run tiny "Say hello in one short sentence."
RUN=$($R/bin/erllama run tiny --max-tokens 400 --detach "Count to a hundred.")
$R/bin/erllama attach "$RUN"
$R/bin/erllama cancel "$RUN"
$R/bin/erllama help
```

Runs belong to the release: they keep going when the CLI exits, and
you can attach to them again by id. `erllama status` shows what runs,
waits and failed, and `erllama inspect JOB` one run or session in
detail. See
[Driving the release from the CLI](../../guides/server/rpc.md) for
conversations, reattachment, the node and cookie setup, TLS for remote
administration, and what each error means, and
[Managing models](../../guides/server/models.md) for downloads, the
registry, keep-alive and memory pressure.

## Serve the HTTP APIs

The release listens on `127.0.0.1:8080` and answers OpenAI
(`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
`/v1/models`, `/v1/responses`), Anthropic (`/v1/messages`) and Ollama
(`/api/...`) requests, plus `/health`, `/ready`, `/metrics` and
`/status`. See
[The HTTP APIs](../../guides/server/http.md) for every supported field,
API keys and limits.

## Work with an agent

```bash
$R/bin/erllama agent qwen --workspace /srv/projects/app
```

The model reads, searches and edits files in the workspace and runs
commands there, asking before it changes anything. `/quit` leaves the
session running; `erllama agent --attach AGENT` goes back to it from
any terminal. See [Agent sessions](../../guides/server/agent.md).

Editors and agent clients speaking the Agent Client Protocol (Zed,
OpenCode, ...) can drive the same sessions with `$R/bin/erllama acp`.
See [ACP](../../guides/server/acp.md).

Conversations, sessions' included, are kept on disk and survive a
restart. See [Conversations](../../guides/server/conversations.md).

Sessions can also use the tools of MCP servers, and the release can
serve its models at `/mcp`. See [MCP servers](../../guides/server/mcp.md).

## Configuration

The `erllama_server` application environment (`config/sys.config` in
the release). A bad value stops the release from starting.

| Key | Default | Meaning |
|---|---|---|
| `run_buffer_events` | `1024` | events kept per run for reattachment |
| `subscriber_max_unacked` | `256` | unread events after which a slow client is cut off (it reattaches) |
| `run_retention_ms` | `300000` | how long a finished run stays available to `attach` and `result` |
| `max_active_runs` | `64` | generating runs at once; more are refused |
| `data_dir` | `ERLLAMA_DATA_DIR`, else the platform user data directory | the model registry and downloaded files |
| `keep_alive_ms` | `300000` | idle time before a model unloads (`infinity` keeps it) |
| `default_n_ctx` | `4096` | context size when a model sets none (capped at its training context) |
| `default_parallel` | `4` | requests a model serves at once (its `parallel` load option overrides it) |
| `kv_unified` | `true` | those requests share one KV pool of `n_ctx` (`false`: each gets `n_ctx / parallel`) |
| `default_model` | none | the model a request without one uses |
| `models` | `[]` | local files registered at boot: `#{name, path}` plus `n_ctx`, `n_gpu_layers`, `parallel`, `kv_unified`, `mmproj`, `keep_alive` |
| `hf_token` | `HF_TOKEN` | Hugging Face token for gated repositories |
| `hf_base_url`, `ollama_base_url` | the public registries | where pulls go (a mirror) |
| `max_active_pulls` | `4` | downloads at once |
| `pull_idle_timeout_ms` | `60000` | a transfer silent for this long is resumed |
| `http` | off (`#{port => 8080}` in the release) | the HTTP listener; `ip` defaults to loopback |
| `api_keys` | `[]` | HTTP API keys (`Authorization: Bearer` or `x-api-key`) |
| `allow_unauthenticated` | `false` | allow a public `ip` without keys |
| `max_body_bytes` | `33554432` | HTTP request body limit |
| `max_http_requests` | `64` | HTTP requests in flight |
| `max_agents` | `16` | agent sessions at once |
| `mcp_servers` | `[]` | MCP servers whose tools agent sessions use (see [MCP servers](../../guides/server/mcp.md)) |
| `mcp_timeout_ms` | `60000` | the longest an MCP call may take |
| `mcp_endpoint` | `false` | serve the release's models at `/mcp` on the HTTP listener |
| `compact_threshold` | `0.75` | compact a conversation above this fraction of the model's context (`false`: never) |
| `compact_keep_messages` | `6` | recent messages a compaction keeps word for word |
| `compact_summary_tokens` | `512` | the longest compaction summary |

The library's own settings (cache tiers, scheduler) go under `erllama`;
see the [configuration guide](https://hexdocs.pm/erllama/configuration.html).

## Where the code is

Every front end goes through `erllama_server_api`: the CLI and other
Erlang nodes over distribution, the HTTP adapters in process (model
access, runs, embeddings, token counts). Why the release is shaped this
way is in [the server design record](../../internals/server-design.md).

Service API and jobs:

| Module | Role |
|---|---|
| `erllama_server_api` | the versioned service API every front end calls |
| `erllama_server_runs` | the table of live jobs (runs, downloads, agent sessions); starts them, enforces the limits, holds the embedding slots |
| `erllama_server_job` | the calls every job process answers: attach, detach, cancel, result, info |
| `erllama_server_events` | a job's numbered, buffered, flow-controlled event stream |
| `erllama_server_ring` | the bounded event buffer behind reattachment and gaps |
| `erllama_server_run_sup` | supervises the runs (temporary) |
| `erllama_server_run` | one run: owns the erllama stream, numbers and buffers its events, serves subscribers |
| `erllama_server_generate` | renders chat or prompt requests into an erllama stream and parses the result |
| `erllama_server_inference` | inference that is not a run: embeddings (they hold a run slot) and token counts |
| `erllama_server_ids` | the ids of runs, conversations, agent sessions, downloads, jobs and tool calls |

Release and operations:

| Module | Role |
|---|---|
| `erllama_server_app` | the application; a bad environment fails the start |
| `erllama_server_sup` | the top supervisor, `rest_for_one` in dependency order |
| `erllama_server_config` | the application environment: defaults, validation, reads |
| `erllama_server_status` | the runtime view behind `erllama status`, `inspect` and `GET /status` |
| `erllama_server_failures` | the bounded log of recent failures and what the release did next |

Models:

| Module | Role |
|---|---|
| `erllama_server_registry` | the persistent model registry (`registry.json`) |
| `erllama_server_models` | loads registry models on demand, keep-alive, the scheduler's model evictor |
| `erllama_server_pull_sup` | supervises the downloads (temporary) |
| `erllama_server_pull` | one download: plan, resumable transfer, integrity check, registration |
| `erllama_server_source` | what `hf://`, `ollama://` and `https://` sources mean and what to fetch |
| `erllama_server_gguf` | reads GGUF metadata without loading the model |

HTTP:

| Module | Role |
|---|---|
| `erllama_server_http` | the Livery listener: routes and the middleware stack |
| `erllama_server_http_auth` | API keys |
| `erllama_server_http_kit` | shared HTTP plumbing: JSON bodies, protocol error shapes, driving a run from a handler |
| `erllama_server_http_ops` | `/health`, `/ready`, `/metrics`, `/status` |
| `erllama_server_openai`, `erllama_server_openai_req` | the OpenAI chat, completion, embedding and model endpoints, and the request translation the other dialects share |
| `erllama_server_openai_responses`, `erllama_server_openai_responses_stream` | the OpenAI Responses API and its typed event stream |
| `erllama_server_responses` | the store of stored responses and their conversations |
| `erllama_server_anthropic` | the Anthropic Messages API |
| `erllama_server_ollama`, `erllama_server_ollama_models` | the Ollama generation and model-management endpoints |

Conversations:

| Module | Role |
|---|---|
| `erllama_server_convs` | conversations: history, last cache key, one active run each, persisted and recovered |
| `erllama_server_conv_log`, `erllama_server_conv_codec` | a conversation's log on disk: records, replay, repair after a stop; messages to JSON and back |
| `erllama_server_conv_export` | a conversation as one JSON or Markdown document |
| `erllama_server_compact` | context compaction: what to keep, the summary, when it is needed |

Agent sessions and tools:

| Module | Role |
|---|---|
| `erllama_server_agent_sup` | supervises the agent sessions (temporary) |
| `erllama_server_agent` | an agent session: the generate, approve, run tools, continue loop |
| `erllama_server_agent_gen` | one generation round of a session |
| `erllama_server_agent_policy` | which tools a session offers and the policy (`auto`, `ask`, `deny`) of each call |
| `erllama_server_tools`, `erllama_server_tool_fs`, `erllama_server_tool_shell` | the native tools (one definition each, with its class), file tools and shell jobs |
| `erllama_server_workspace` | keeps tool paths inside the workspace |

MCP:

| Module | Role |
|---|---|
| `erllama_server_mcp` | the MCP clients: connections, tool catalog, calls and their results |
| `erllama_server_mcp_endpoint` | the opt-in `/mcp` endpoint and its tools |

CLI (none of these loads an erllama module):

| Module | Role |
|---|---|
| `erllama_server_client` | the client side of the service API: connection, bounded calls, following a job |
| `erllama_server_cli` | the `erllama` command: argument parsing and output |
| `erllama_server_cli_chat` | `erllama chat`: the line-by-line conversation and its commands |
| `erllama_server_cli_agent`, `erllama_server_cli_mcp`, `erllama_server_cli_conv` | the agent, MCP and conversation commands |
| `erllama_server_cli_status` | how `erllama status` and `inspect` show the release |
| `erllama_server_cli_format` | output formatting |
| `erllama_server_cli_term` | the terminal: Ctrl-C as a key, cooked and raw modes, serialised output, the status line |
| `erllama_server_cli_input` | what is typed at a prompt: continued lines, blocks, commands |

## Tests

Common Test suites, from the repository root with
`rebar3 ct --dir apps/erllama_server/test` (most run on erllama's stub
backend; the real-model cases skip unless `LLAMA_TEST_MODEL` is set):

| Suite | Covers |
|---|---|
| `erllama_server_SUITE` | the application starts on top of erllama and stops without taking it down |
| `erllama_server_rpc_SUITE` | the service API from a separate hidden node, as the CLI uses it; the CLI under a terminal; `leaked/1` checks the client loaded no erllama module |
| `erllama_server_tls_SUITE` | the same over TLS distribution |
| `erllama_server_models_SUITE` | loading on demand, keep-alive, eviction, requests sharing a model |
| `erllama_server_pull_SUITE` | downloads from a local stand-in for Hugging Face and Ollama (`erllama_server_fake_registry`) |
| `erllama_server_http_SUITE` | the HTTP APIs over a socket: shapes, streaming, strict fields, errors, keys, limits |
| `erllama_server_agent_SUITE` | agent sessions with a scripted model (`erllama_server_agent_script`) |
| `erllama_server_mcp_SUITE` | sessions against real MCP servers over Streamable HTTP and stdio |
| `erllama_server_convs_SUITE` | restarts, forks, exports, and sessions or the store crashing mid-call |
| `erllama_server_compact_SUITE` | compaction of runs and sessions, automatic and on request |
| `erllama_server_ops_SUITE` | what an operator sees, and failures staying with their owner |
| `erllama_server_demo_SUITE` | the walkthrough of [guides/server/demo.md](../../guides/server/demo.md) |

EUnit and PropEr modules (`rebar3 eunit`, `rebar3 proper`):
`erllama_server_api_tests` (configuration, specs, the version check),
`erllama_server_cli_tests` and `erllama_server_cli_input_tests` (the
CLI and its input), `erllama_server_generate_tests` (option
conflicts), `erllama_server_openai_tests` and
`erllama_server_protocols_tests` (the HTTP dialects),
`erllama_server_registry_tests` (the registry and the GGUF reader),
`erllama_server_conv_tests` (the conversation log),
`erllama_server_compact_tests` (the compaction plan),
`erllama_server_tools_tests` (the native tools),
`erllama_server_mcp_tests` (MCP names, rendering, configuration),
`erllama_server_ring_tests` and `prop_erllama_server_ring` (the event
buffer).
