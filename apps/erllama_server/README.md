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
$R/bin/erllama pull hf://ggml-org/models/tinyllamas/stories260K.gguf
$R/bin/erllama load ~/Models/tinyllama-1.1b-chat.gguf --name tiny --n-ctx 2048
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
administration, and what each error means, and
[Managing models](../../guides/server/models.md) for downloads, the
registry, keep-alive and memory pressure.

## Serve the HTTP APIs

The release listens on `127.0.0.1:8080` and answers OpenAI
(`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
`/v1/models`, `/v1/responses`), Anthropic (`/v1/messages`) and Ollama
(`/api/...`) requests, plus `/health`, `/ready` and `/metrics`. See
[The HTTP APIs](../../guides/server/http.md) for every supported field,
API keys and limits.

## Work with an agent

```bash
$R/bin/erllama agent qwen --workspace /srv/projects/app
```

The model reads, searches and edits files in the workspace and runs
commands there, asking before it changes anything. See
[Agent sessions](../../guides/server/agent.md).

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
| `default_model` | none | the model a request without one uses |
| `models` | `[]` | local files registered at boot: `#{name, path}` plus `n_ctx`, `n_gpu_layers`, `mmproj`, `keep_alive` |
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

The library's own settings (cache tiers, scheduler) go under `erllama`;
see the [configuration guide](../../guides/configuration.md).

## Where the code is

| Module | Role |
|---|---|
| `erllama_server_api` | the versioned service API every front end calls |
| `erllama_server_run` | one run: owns the erllama stream, numbers and buffers its events, serves subscribers |
| `erllama_server_runs` | the table of jobs (runs and downloads); starts them and enforces the limits |
| `erllama_server_events` | a job's numbered, buffered, flow-controlled event stream |
| `erllama_server_registry` | the persistent model registry (`registry.json`) |
| `erllama_server_models` | loads registry models on demand, keep-alive, the scheduler's model evictor |
| `erllama_server_pull` | one download: plan, resumable transfer, integrity check, registration |
| `erllama_server_source` | what `hf://`, `ollama://` and `https://` sources mean and what to fetch |
| `erllama_server_gguf` | reads GGUF metadata without loading the model |
| `erllama_server_http` | the Livery listener: routes and the middleware stack |
| `erllama_server_http_kit` | shared HTTP plumbing: JSON bodies, protocol error shapes, driving a run from a handler |
| `erllama_server_http_auth` | API keys |
| `erllama_server_openai`, `erllama_server_openai_req` | the OpenAI chat, completion, embedding and model endpoints, and their request translation |
| `erllama_server_openai_responses`, `..._stream` | the OpenAI Responses API and its event stream |
| `erllama_server_responses` | stored responses and their conversations |
| `erllama_server_anthropic` | the Anthropic Messages API |
| `erllama_server_ollama`, `erllama_server_ollama_models` | the Ollama generation and model-management endpoints |
| `erllama_server_agent` | an agent session: the generate, approve, run tools, continue loop |
| `erllama_server_agent_gen` | one generation round of a session |
| `erllama_server_tools`, `_tool_fs`, `_tool_shell` | the native tools, their policies, file tools and shell jobs |
| `erllama_server_workspace` | keeps tool paths inside the workspace |
| `erllama_server_mcp` | the MCP clients: connections, tool catalog, calls and their results |
| `erllama_server_mcp_endpoint` | the opt-in `/mcp` endpoint and its tools |
| `erllama_server_cli_agent`, `erllama_server_cli_mcp`, `erllama_server_cli_format` | the CLI's agent and MCP commands and its output formatting |
| `erllama_server_http_ops` | health, readiness, metrics |
| `erllama_server_convs` | conversations: history, last cache key, one active run each |
| `erllama_server_generate` | renders chat or prompt requests into an erllama stream and parses the result |
| `erllama_server_client` | the client side, used by the CLI (no erllama module) |
| `erllama_server_cli` | the `erllama` command: argument parsing and output |

Tests: `test/erllama_server_rpc_SUITE.erl` drives the API from a
separate hidden node as the CLI does; `test/erllama_server_tls_SUITE.erl`
does the same over TLS distribution; `test/erllama_server_pull_SUITE.erl`
downloads from a local stand-in for Hugging Face and Ollama
(`test/erllama_server_fake_registry.erl`); and
`test/erllama_server_models_SUITE.erl` covers loading, keep-alive and
eviction; `test/erllama_server_http_SUITE.erl` drives the HTTP APIs over
a socket; `test/erllama_server_agent_SUITE.erl` drives agent sessions
with a scripted model (`test/erllama_server_agent_script.erl`);
`test/erllama_server_mcp_SUITE.erl` connects sessions to real MCP
servers over HTTP and stdio.
