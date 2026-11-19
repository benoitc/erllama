# erllama_server: progress record

The plan (7 phases) is in
[apps/erllama_server/PLAN.md](../apps/erllama_server/PLAN.md). This
page records what is implemented, what is outstanding and how it was
validated. A phase is only marked done when its required behaviour is
implemented and tested.

Branch: `feat/server-rpc` (Phases 1 and 2, on erllama 0.12.0,
`ad47520`), reviewed as one pull request.

## Phase 1: repository and build migration (implemented, awaiting review)

| Step | State | Notes |
|---|---|---|
| rebar3 umbrella behaviour checked | done | app hooks run from the app dir; `priv` is linked; PropEr finds `apps/*/test`; `hex build --app` works; empty fmt globs pass silently, so globs were fixed |
| move into `apps/erllama` | done | history kept; rebased onto 0.12.0 (the two files 0.12.0 added were placed under `apps/erllama`) |
| umbrella build | done | app `rebar.config` (NIF hooks, Hex, ex_doc), root tooling, elvis and erlfmt globs, Hex `files`, CI paths, bench, doc paths, root README |
| `apps/erllama_server` + release | done | app, sup, CT suite; relx release with `config/sys.config` and `config/vm.args.src`; node on loopback only |
| CLI escript | done (scaffold) | `erllama_server_cli`, built as the `erllama` escript before every release, shipped as `bin/erllama_cli` with the `bin/erllama` launcher; `--version` and `ping` only |
| standalone library check | done | `scripts/check_library_package.sh` and the `library-package` CI job |
| release check | done | `release` CI job: prod release, daemon, ping, CLI `--version` and `ping`, stop |

Validation (OTP 28.3.1, local):
- full gate: eunit 580/580, proper 13/13, ct 9 passed / 69 skipped, lint, xref and dialyzer clean, fmt clean;
- real-model suite via `--dir apps/erllama/test`: stories260K 40 passed / 4 skipped, TinyLlama 44/44;
- `scripts/check_library_package.sh`: the 0.12.0 package compiles its NIF and runs in a consumer project, with no livery or barrel_mcp;
- release: starts, `bin/erllama ping` answers `pong` with the right cookie and fails cleanly with a wrong one; the prod release runs the CLI on its bundled ERTS with no Erlang on `PATH`;
- the CLI loads no erllama module (checked in a fresh peer node).

Known limits and findings:
- rebar3's escriptize bundles the main app's dependencies, so the escript carries erllama's beam files (not its NIF library). The CLI never loads them; the peer-node test guards this;
- the relx start script uses `RELX_COOKIE` only for its control commands; the node itself uses `~/.erlang.cookie`. Phase 2 documents the cookie and node setup;
- stale generated HTML may remain in the root `doc/` of old checkouts (ignored).

## Phase 2: service API, runs and the RPC client (implemented, awaiting review)

| Feature | State | Notes |
|---|---|---|
| versioned service API | done | `erllama_server_api`, API version 1 in `include/erllama_server_api.hrl`; `hello/1` refuses other versions; every call returns `ok`, `{ok, _}` or `{error, _}` |
| runs | done | one `erllama_server_run` per run under `erllama_server_run_sup`; ids `run_<hex>`; events `started`, `token`, `thinking`, `done`, `cancelled`, `error`, numbered from 0 |
| replay and gaps | done | per-run ring of `run_buffer_events`; `attach(Run, Pid, Cursor)` returns the replay or `{gap, Oldest}` |
| bounded subscribers | done | acks every 16 events; more than `subscriber_max_unacked` unread events drops the subscriber with `{dropped, NextSeq}`; the client reattaches on its own |
| cancel, detach, result, retention | done | cancel ends with `cancelled`; results kept `run_retention_ms`; runs outlive their clients |
| conversations | done (in memory) | `erllama_server_convs`: history, one active run each, the last `finish_key` passed as `parent_key` to the next turn on the same model; cancelled or failed turns are not recorded |
| chat rendering | done | `erllama_server_generate`: `chat_apply` + `chat_parse` on llama.cpp models, the legacy template on backends without it (the stub); option whitelist keeps `parent_key`, `session_id`, `to` and middleware server-side |
| admission | done | `max_active_runs` checked and the run started in one step by `erllama_server_runs` |
| failure handling | done | unload or a model crash ends the run with an `error` event; a run that dies in any way (shutdown, crash, `kill`) has its stream cancelled by erllama, which now watches every stream's receiver (library fix, in the erllama CHANGELOG) |
| tool calls across turns | done | generated calls get ids and are stored in the OpenAI shape the template engine requires; incoming assistant `tool_calls` (OpenAI or flat) and `tool` messages with their `tool_call_id` are validated and kept |
| option conflicts | done | tools with a caller `grammar` or `json_schema` are refused as `erllama:chat/3` refuses them; a caller grammar replaces the template grammar with its triggers and prefill; chat-only options are refused on a raw prompt |
| client | done | `erllama_server_client`: hidden node named `erllama_cli_<os pid>_<random>` (unique across concurrent CLI processes), cookie, bounded `erpc` calls with clear errors (unreachable, no service API, incompatible version, timeout), `follow/4` with acks, reconnection with backoff, gap reporting and a 30 s idle check |
| CLI | done | `ping`, `status`, `models`, `load`, `unload`, `run` (`--conversation`, `--max-tokens`, `--detach`), `chat`, `runs`, `attach --from`, `cancel`, `result`, `--version`; exit codes 0, 1, 2 |
| TLS distribution | done | `ERLLAMA_DIST_OPTFILE` in the launcher; `ssl` is in the release; guide in `guides/server/rpc.md` |
| model loading | interim | `load PATH` only; the registry and downloads are Phase 3 |

Validation (OTP 28.3.1, local):
- full gate: eunit 599/599, proper 14/14, ct 25 passed / 70 skipped (gated real-model cases), lint, xref, dialyzer and fmt clean;
- `erllama_server_rpc_SUITE` (15 cases, release and client on separate peer nodes, three repeated runs green): streaming, result, reattach with a cursor, gap, dropped slow subscriber, cancel, run surviving the client node, conversation history, one run per conversation, `max_active_runs`, unload during a run, app stop cancelling streams, version mismatch, missing API, unreachable release; after every case the client node is checked to have loaded no erllama module;
- `erllama_server_tls_SUITE`: hello and a streamed run over `inet_tls` with a generated CA, the connection checked to be TLS;
- real model: `real_model_chat` ran on TinyLlama 1.1B (chat template and parser path); stories260K not run for this phase; the library's real-model suite was not rerun (no library change);
- prod release by hand: TinyLlama loaded with `--n-ctx 2048`, a streamed turn, `--detach` then `attach`, cancel after 123 tokens, the client killed mid-run and `attach --from 50` completing, `result`, clear errors for an unknown run and an unreachable node, `ERLLAMA_DIST_OPTFILE` reaching the client VM;
- `scripts/check_library_package.sh` still passes (no server dependency in the library);
- the `release` CI job now loads a stub model and runs, detaches, cancels and attaches through `bin/erllama`.

Outstanding and known limits:
- conversations and finished runs live in memory and are lost on restart (Phase 6);
- Ctrl-C quits the CLI without cancelling the run; keyboard interruption comes with the terminal work (Phase 7);
- the stub backend has no chat parser, so routine tests see unparsed replies; parsing is covered by the real-model case only.

## Phase 3a: model registry, downloads and lifetime (implemented)

| Feature | State | Notes |
|---|---|---|
| model registry | done | `erllama_server_registry`, `<data_dir>/registry.json` written atomically; names `family:tag`, aliases, copies, a default model; missing files reported; a corrupt file moved aside |
| GGUF metadata | done | `erllama_server_gguf` streams the header (large arrays skipped): architecture, context length, quantisation, parameters, chat template, projector modalities |
| downloads | done | `hf://` (file, or repository plus quantisation, with its mmproj), `ollama://` (model, projector and parameter layers), `https://` (optional sha256); one `erllama_server_pull` job each, followed like a run |
| resumable, verified transfers | done | `.part` files with a running sha256; `Range` resume after a stall or a cancel; a server ignoring ranges restarts the file; size and sha256 checked; redirects followed by hand, the Hugging Face token kept to its own origin |
| shared job events | done | the event ring and subscribers moved to `erllama_server_events`; runs and downloads answer the same attach, detach, cancel and result calls |
| load on demand and keep-alive | done | `erllama_server_models`: one shared load per model, held while used, unloaded after its keep-alive (`0`, ms, `infinity`, per request or per model) |
| eviction | done | `erllama_server_models` is erllama's `model_evictor`; the release turns the scheduler on |
| CLI | done | `pull`, `models`, `show`, `ps`, `load` (name or `.gguf` path), `unload`, `cp`, `rm`, `default`; download progress on the terminal |
| CLI escript | changed | built by `scripts/build_cli.escript` from erllama_server's modules only (22 files); rebar3 escriptize would have packed every server dependency |
| library fix | done | an unloaded model id can be loaded again (`erllama_registry` monitored the model from inside the model process, so the id was never freed) |

Validation is recorded with the Phase 3 commit.

Known limits:
- Livery writes a `500` response into an unfinished streamed body when a handler crashes mid-stream (found while testing interrupted downloads; reported upstream, not worked around in the server);
- split GGUF files are not supported;
- the registry entry's `backend` option (stub backend) exists for tests.

## Phase 3b: HTTP core and the OpenAI adapters (implemented)

| Feature | State | Notes |
|---|---|---|
| listener | done | `erllama_server_http` on Livery, off unless `http` is set (the release sets `127.0.0.1:8080`; `ERLLAMA_HTTP_PORT` overrides) |
| middleware | done | request ids, access log, API keys (`Bearer` or `x-api-key`, constant-time), body limit (413), concurrency limit (429 with `Retry-After`), metrics, crash handling; a public `ip` without keys refuses to start |
| one execution path | done | every HTTP request is a stateless run through `erllama_server_api:start_run/1` (`max_active_runs`, visible to `erllama runs`); the handler follows its events and cancels the run on disconnect |
| streaming parse | done | with a parser, the run re-parses after each token and emits content, reasoning and tool-call deltas (`tool_call` events); call syntax never reaches content |
| OpenAI | done | `/v1/models`, `/v1/chat/completions` (tools, named tool choice, `response_format`, logprobs, `stream_options.include_usage`, reasoning), `/v1/completions`, `/v1/embeddings` (float, base64); strict fields, 400 naming the field |
| multimodal | done | base64 `image_url` and `input_audio` for registry models whose projector covers the modality (400 otherwise); library option `media_add_special` added so the streamed chat prompt keeps a single BOS token |
| ops | done | `/health`, `/ready` (engine, registry, model manager), `/metrics` (Livery HTTP metrics, erllama counters, server gauges) |

## Phase 3c: Responses, Anthropic, Ollama (implemented)

| Feature | State | Notes |
|---|---|---|
| OpenAI Responses | done | `/v1/responses` (items, instructions, function tools and outputs, text formats, reasoning effort), typed SSE events with sequence numbers, `GET`/`DELETE`; stored responses continue their conversation through `previous_response_id` (in memory until Phase 6) |
| Anthropic Messages | done | `/v1/messages` (system, text, image, `tool_use`, `tool_result`, tools and tool choice, thinking), the typed SSE protocol, `count_tokens`, Anthropic error shape |
| Ollama | done | `/api/chat`, `/api/generate` (raw, load and unload with an empty prompt), `/api/embed`, `/api/embeddings`, `/api/tags`, `/api/show`, `/api/ps`, `/api/pull` (status lines from the download job), `/api/delete`, `/api/copy`, `/api/version`, `/`; `/api/create` and `/api/push` answer 501 |
| compatibility matrix | done | `guides/server/http.md`: every endpoint and field, supported, informational or refused |

Validation (OTP 28.3.1, local):
- full gate: eunit 646/646, proper 14/14, ct 76 passed / 77 skipped (gated real-model cases), lint, dialyzer, xref and fmt clean;
- `erllama_server_http_SUITE`: 26 cases on stub models over a real socket (every endpoint's shapes, streaming, strict fields, errors, keys, limits, disconnect cancelling the run, load on demand, unload during a stream, concurrent requests, ops endpoints, Ollama pull through a local registry stand-in);
- gated real-model cases run on Qwen2.5 1.5B: streamed tool calls with no call syntax in content, logprobs, an Anthropic `tool_use` round trip, a Responses function-call chain; on SmolVLM 256M: an image chat (streamed and not); TinyLlama skips the tool cases (its template has no tool grammar);
- official SDKs against the server (Python `openai`, `anthropic`, `ollama`, in a throwaway uv environment): chat, streaming with usage, a tool call and its answer, embeddings, a Responses chain and its streaming helper, Anthropic messages, its streaming helper and token counting, Ollama chat, streamed chat, list and embed.

## Phase 4: agent runtime and native tools (implemented)

| Feature | State | Notes |
|---|---|---|
| sessions | done | `erllama_server_agent`, one gen_statem per conversation (it holds the conversation for its life), a job like runs (attach, detach, reattach), concurrent across conversations, `max_agents` |
| loop | done | generate, approve, run tools, append results, generate again; a generator behaviour (`erllama_server_agent_gen`, scripted in tests); the model is held for each turn and loaded on demand |
| tools | done | `read_file` (paginated), `list_dir`, `search`, `write_file` (create only), `edit_file` (exact match), `shell` (timeout, head and tail output, background jobs), `shell_status`, `shell_kill`; paths confined to the workspace, symlinks included |
| execution order | done | reads run together; a write waits for the calls before it and holds back the ones after; results in call order; tool failures and crashes become error results |
| process groups | done | every command leads its own process group (the emulator starts port programs that way, on Linux and macOS); timeout, interrupt, kill or the session's end kill the whole group |
| policy and approvals | done | `auto`, `ask`, `deny` per tool, `yes` and `read_only` presets; `approval_request` events; allow, deny, always |
| budgets, queue, interrupt | done | rounds, tokens, time per turn; input during a turn queued; interrupt cancels generation and tools and records unfinished calls as interrupted |
| HTTP callers' tools | done | never executed: HTTP runs only stream calls back to the client |
| CLI | done | `agent MODEL --workspace PATH [--yes \| --read-only]`, `agents`, `agent-stop` |

Validation: `erllama_server_agent_SUITE` (9 cases with a scripted model: plain answers, ordering, approvals, unknown tools and outside paths, round budget, queued input, interrupt killing a shell and its descendants, background jobs, stopping and continuing a conversation), `erllama_server_tools_tests` (workspace confinement including symlinks, paging, listing, search, write and edit), and the gated real-model case on Qwen2.5 1.5B (the model reads a file with its tools to answer).

## Phase 5: barrel_mcp bridge (implemented)

| Feature | State | Notes |
|---|---|---|
| clients | done | `erllama_server_mcp`: one barrel_mcp client per configured server (stdio or Streamable HTTP), started in the background, reconnected with backoff, tool list refreshed every 30 s; a server that is down never blocks the release or a session |
| catalog | done | tools namespaced `server__tool` (sanitised, at most 64 bytes, suffixed on a collision); a snapshot per generation round; `readOnlyHint` tools run with reads, others are ordered like writes |
| results | done | text, structured content as JSON, media described, embedded text resources shown; tool errors, timeouts and lost servers become error results |
| policy | done | per server (`ask` by default, `auto`, `deny`); a denied server stays denied; `--yes` and `--read-only` apply to MCP tools; per-tool overrides through the service API |
| endpoint | done | opt-in `mcp_endpoint`: `/mcp` on the HTTP listener behind the API keys, tools `erllama_models` and `erllama_chat` (a stateless run, cancelled when the caller gives up) |
| API and CLI | done | `mcp/0`, `mcp_tools/0`, `mcp_resources/1`, `mcp_read/2`; `erllama mcp [tools \| resources SERVER \| read SERVER URI]` |

Validation: `erllama_server_mcp_SUITE` (6 cases against real servers: Streamable HTTP through livery_mcp in the test node and stdio through an `erl` subprocess; namespaced tools offered and called, media rendering, server and session policies with approvals, tool errors and timeouts, an unreachable server, a server going away and coming back, the release's own endpoint with API keys) and `erllama_server_mcp_tests` (names, rendering, configuration).

## Phases 6 and 7

Not started.
