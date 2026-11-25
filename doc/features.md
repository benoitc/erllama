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

Review fixes (2026-11-19): recursive search and listing no longer follow symbolic links out of the workspace; a model being evicted or unloaded is never handed to a new run (acquisitions wait and reload); a generation that dies ends its turn instead of leaving the session stuck; stopping a session or the release stops running and background commands through their process groups; the time budget ends a turn while it runs and the token budget caps each generation; two requests continuing the same stored response are independent branches; a blob already on disk is reused only after its sha256 checks out. Each has a regression case (`erllama_server_tools_tests`, `erllama_server_models_SUITE`, `erllama_server_agent_SUITE`, `erllama_server_http_SUITE`, `erllama_server_pull_SUITE`).

Validation: `erllama_server_agent_SUITE` (9 cases with a scripted model: plain answers, ordering, approvals, unknown tools and outside paths, round budget, queued input, interrupt killing a shell and its descendants, background jobs, stopping and continuing a conversation), `erllama_server_tools_tests` (workspace confinement including symlinks, paging, listing, search, write and edit), and the gated real-model case on Qwen2.5 1.5B (the model reads a file with its tools to answer).

## Phase 5: barrel_mcp bridge (implemented)

| Feature | State | Notes |
|---|---|---|
| clients | done | `erllama_server_mcp`: one barrel_mcp client per configured server (stdio or Streamable HTTP), started in the background, reconnected with backoff, tool list refreshed every 30 s; a server that is down never blocks the release or a session |
| catalog | done | tools namespaced `server__tool` (sanitised, at most 64 bytes, suffixed on a collision); a snapshot per generation round; `readOnlyHint` tools run with reads, others are ordered like writes |
| results | done | text, structured content as JSON, media described, embedded text resources shown; tool errors, timeouts and lost servers become error results |
| policy | done | per server (`ask` by default, `auto`, `deny`); a denied server stays denied; `--yes` and `--read-only` apply to MCP tools; per-tool overrides through the service API |
| endpoint | done | opt-in `mcp_endpoint`: `/mcp` on the HTTP listener behind the API keys, tools `erllama_models` and `erllama_chat` (a stateless run, cancelled when the caller gives up) |
| API and CLI | done | `mcp/0`, `mcp_tools/0`, `mcp_resources/1`, `mcp_read/2`, `mcp_prompts/1`, `mcp_prompt/3`, `agent_prompt/4`; `erllama mcp [tools \| resources SERVER \| read SERVER URI \| prompts SERVER \| prompt SERVER NAME ...]` |

Review fixes (2026-11-19):
- interrupting a session cancels its MCP requests at their servers (that request only; the shared connection stays);
- tool aliases are given once and never reassigned, and an `always` approval is bound to the server and remote tool, not the alias;
- every MCP server field is validated at startup (unknown keys refused), and a connection that cannot be set up stays with its server;
- resources (two tools per server that offers them) and prompts (API, `erllama mcp prompts`/`prompt`, `/prompt` in a session) are available to agent sessions.

The upstream gaps found here are fixed in barrel_mcp 4.7.0 and 4.8.0 and Livery 0.15.0, which the release now uses (barrel_mcp 4.8.1, Livery 0.15.1, hackney 4.12.4): the client cancels a request when the process that made it exits, so a session interrupts a call by stopping its worker; barrel_mcp's servers tell a cancelled tool over stdio and over Streamable HTTP; livery_mcp passes a closed request on to the engine; and a handler that crashes mid-stream gets its stream reset instead of a 500 written into it. The cancellation case runs against barrel_mcp's own servers on both transports, and a pull case covers a connection that breaks mid-blob. A cancel over Streamable HTTP on the 2026-07-28 revision used to reach the server about a second late, so a short tool was never told; barrel_mcp 4.8.1 (with hackney 4.12.4) fixes it.

Validation: `erllama_server_mcp_SUITE` (6 cases against real servers: Streamable HTTP through livery_mcp in the test node and stdio through an `erl` subprocess; namespaced tools offered and called, media rendering, server and session policies with approvals, tool errors and timeouts, an unreachable server, a server going away and coming back, the release's own endpoint with API keys) and `erllama_server_mcp_tests` (names, rendering, configuration).

## Phase 6a: durable conversations (implemented)

| Feature | State | Notes |
|---|---|---|
| persistence | done | one append-only, synced JSON log per conversation (`<data_dir>/conversations/<id>.jsonl`), apart from the KV cache; written before a change is visible; a torn last line is ignored |
| recovery | done | logs loaded at start; a run cut by a stop or a crash is released and its unanswered tool calls get a result: the recorded outcome, unresolved (may have run, never run again) or not run |
| tool records | done | intent before a call runs, outcome before the session generates again; a session that dies is repaired the same way at once, and its running tools go with it |
| KV | done | the finish key is persisted with its model and passed as `parent_key` on the next turn: warm when the cache holds it, transcript prefill otherwise |
| operations | done | list (snapshots hidden), show, fork, delete (refused while held), export (JSON, Markdown), attach; `agent --conversation` reuses the conversation's workspace |
| stored responses | done | one file each under `<data_dir>/responses/`; their snapshots are persisted conversations, so `previous_response_id` and branches survive a restart |

Validation: `erllama_server_conv_tests` (every message shape round trips, unknown keys stay binaries, torn tail, recovery of each tool state, repair durable) and `erllama_server_convs_SUITE` (restart keeps conversations and their continuation, fork, delete and export across a restart, stored responses and their branches across a restart, a killed session and a crashed store each leave the running call unresolved with its command stopped and a resumed session not running it again). Gated real-model case on Qwen2.5 1.5B: the turn after a restart is warm (`partial`).

## Phase 6b: compaction (implemented)

| Feature | State | Notes |
|---|---|---|
| automatic | done | before a conversation turn and at the start of a session's turn, the transcript is measured with the model's template; above `compact_threshold` of its context it is compacted first (`compacting`, `compacted` events) |
| on request | done | `conversation_compact/1` (idle conversation), `agent_compact/1` (idle session), `erllama conversation compact ID` |
| what is kept | done | system messages verbatim (one leading system message, so strict templates render); the first request verbatim; a summary of the middle by the conversation's model (goal, decisions, state of the work, what remains); recent messages verbatim from a user message, so tool calls keep their results |
| exact prompts | done | written as a snapshot with no finish key: the next turn is a new exact prompt; nothing about it in the cache |
| failure and interruption | done | a failed summary leaves the transcript unchanged; an interrupt, a stop or the time budget ends a compaction in progress |

Validation: `erllama_server_compact_tests` (system messages, first request, cuts at user messages, tool pairs, long turns, nothing to drop, rendering) and `erllama_server_compact_SUITE` (a run compacts automatically and goes on, the result survives a restart, on request with nothing to drop and with a middle, refused while a session holds the conversation, a session compacts at the start of a turn and on request, an interrupt stops a compaction in progress and leaves the conversation as it was). Gated real-model case on Qwen2.5 1.5B: six messages compacted to three, and the next turn still knows what was only in the summarised middle.

## Phase 7: terminal and release validation (implemented)

| Feature | State | Notes |
|---|---|---|
| Ctrl-C | done | in a terminal the CLI runs with `+Bc` and reads Ctrl-C as a key: `run` and a `chat` turn are cancelled, an `agent` turn interrupted, a `pull` cancelled (part file kept), `attach` detaches; a second Ctrl-C leaves; at a prompt it drops the line; with no terminal (`+Bd`) it ends the CLI at once |
| input | done | cooked mode with line editing at the prompt, raw mode only while a turn streams; a line ending in `\` continues, `"""` opens a block, `/` starts a command |
| commands | done | `chat`: `/new`, `/sessions`, `/switch`, `/fork`, `/compact`, `/help`, `/quit`; `agent`: the same on sessions, plus `/prompt` and `/stop` |
| rendering | done | content as it streams, reasoning dimmed, tool calls and results, approvals at a prompt, and a status line (loading, compacting, prefill, running tool) after a quiet second |
| closing the terminal | done | ends only the CLI: runs, sessions and loaded models stay on the release |
| resumed sessions | done | a session continuing a compacted conversation keeps its summary, in one leading system message with its own prompt once |

Terminal tests: `erllama_server_cli_input_tests` (continuation, blocks, commands, dropped lines, newline translation) and four cases in `erllama_server_rpc_SUITE` with the CLI under `script(1)`: Ctrl-C during a chat turn cancels the run and the prompt works after it; a continued line is one message; Ctrl-C at the prompt drops the line; `/switch` to another session whose approval is answered in the terminal. They run on macOS and on Linux.

### Release validation (2026-11-23, OTP 28.3.1)

- **Gate:** `rebar3 fmt --check`, compile, eunit, proper, ct, lint, dialyzer and xref, all clean; `scripts/check_library_package.sh` (the library builds standalone from its package, with no server dependency); the server suites on Linux (Docker, `erlang:28`).
- **Prod release, end to end** (`rebar3 as prod release`, a fresh data directory):
  - a real pull from Hugging Face (`hf://bartowski/SmolLM2-135M-Instruct-GGUF:Q4_K_M`, 100.6 MiB) and a run on it;
  - a local Qwen2.5 1.5B registered and run;
  - `chat` in a terminal: a turn, Ctrl-C on a long one (the run ends cancelled), `/sessions`, `/quit`;
  - `agent` in a terminal: the model reads a file with its tool and answers;
  - the official Python SDKs (`openai` chat, streaming and embeddings, `anthropic` messages);
  - a release restart: the conversations are back, an interrupted session's conversation is freed, and a turn on one answers from its history;
  - compaction on request: 6 messages to 3, with a summary that keeps the goal and the result;
  - `status` and `ps`.
- **Real models, server** (`rebar3 ct --dir apps/erllama_server/test`, `LLAMA_TEST_MODEL` Qwen2.5 1.5B, `LLAMA_TEST_VISION_MODEL` and `LLAMA_TEST_MMPROJ` SmolVLM 256M): 121 of 121 passed, none skipped. This covers chat, tool calls (streamed, with no call syntax in content), logprobs, the Responses chain, an Anthropic `tool_use` round trip, an image chat, an agent reading a file, a warm turn after a restart, and a compaction summary.
- **Real models, library:**
  - `erllama_real_model_SUITE` on TinyLlama 1.1B: 44 of 44;
  - `erllama_vision_SUITE` on SmolVLM 256M: 6 of 7, the audio case skipped (no audio model locally);
  - `erllama_family_SUITE` on Mamba 130M (recurrent) and Gemma 3 270M (sliding window): 5 of 5.
- **Not run:** the audio case (no audio GGUF and projector at hand); MCP against servers other than barrel_mcp's own (covered by `erllama_server_mcp_SUITE` against barrel_mcp over stdio and Streamable HTTP).

## Shared serving: several requests on one model (2026-11-23)

| Feature | State | Notes |
|---|---|---|
| finding | fixed | the server loaded every model with llama.cpp's default of one sequence (`n_seq_max = 1`), so a request on a busy model waited for the running generation to end: a short HTTP question during a long CLI reply got its first token only when the reply was done |
| parallel sequences | done | a model serves `parallel` requests at once (default `default_parallel`, 4), batched into one decode per step; `erllama load ... --parallel N`; `ps` shows how many are busy |
| KV pool | done | with `kv_unified` (default) the requests share one pool of `n_ctx` cells, each up to `n_ctx`, as llama-server with automatic slots; `kv_unified => false` gives each `n_ctx / parallel`; `parallel => 1` is the old behaviour |

Tests that fail before the fix and pass after it:
- `erllama_server_models_SUITE:second_request_is_not_queued`: a short request during a 40 s generation on the same model. Before, it was queued behind the long one; after, it is done in under 200 ms.
- `erllama_server_http_SUITE:http_request_during_a_long_run`: a streamed `/v1/chat/completions` during a long CLI-style run. Before, there was no token within 2 s; after, the first token comes in about 100 ms.
- Gated on Qwen2.5 1.5B, `real_model_concurrent_streams`: the short answer ("Paris.") in 238 ms while a 400-token reply goes on, and both end with text.

### Benchmark: what a second user waits for

`scripts/bench_shared_model.py`, measured on an Apple M-series machine with Metal: Qwen2.5 1.5B Q4_K_M, `n_ctx 4096`, greedy.
- A 1500-token streamed reply; at token 750, a short streamed question (16 tokens) on the same model.
- Medians of 3 runs. erllama_server is this branch (vendored llama.cpp v1.0.0); llama-server is MacPorts build b12110 with `-c 4096 -np 4 -kvu`.

| Setup | Short: first token | Short: total | Long tok/s before / during / after | Long: longest gap |
|---|---|---|---|---|
| erllama, `parallel 1` (as before the fix) | 5429 ms | 5436 ms | 150 / 140 / n/a | 15 ms |
| erllama, `parallel 4` | 67 ms | 76 ms | 149 / 67 / 139 | 39 ms |
| llama-server, 4 slots, unified, `cache_prompt: false` | 47 ms | 58 ms | 151 / 40 / 139 | 40 ms |
| llama-server, 4 slots, unified, prompt cache on | 19 ms (first run 84) | 28 ms | 146 / 96 / 134 | 19 ms |

- With the fix, the second user waits about 67 ms instead of the rest of the first user's reply.
- With both prefilling the short question from scratch, that is about 20 ms more than llama-server. The first user's longest pause (about 40 ms, the second request's prefill) is the same.
- llama-server's slot cache also reuses a short prompt it has seen. erllama's cache keeps only prompts of at least 512 tokens (`min_tokens`), so a repeated short question is always prefilled.
- Where the 20 ms went: the request rendered its chat template, tokenised its prompt and was admitted through three calls to the model process, one after another, and each waited for the decode step in progress (about 6.6 ms). Tokenising and rendering now run in the caller's process: the second user's first token comes in 54 ms (66 ms before, measured back to back), and the first user's speed is unchanged.
- The rest: the first token is sampled one step after the prompt's prefill, where llama-server samples right after it. Changing that touches the cache invariant that a token handed back is already in the KV, so it is left for review.
- Not measured here: a cache checkpoint of a long conversation pauses the model's other streams while it is saved (the save reads a paused context, an invariant of the cache). Its length depends on the conversation's size.

## Work that outlives its client, and what an operator sees (2026-11-23)

| Feature | State | Notes |
|---|---|---|
| runtime view | done | `erllama status`, `erllama inspect JOB`, `GET /status` (JSON) and `/metrics`: loaded models with busy sequences and requests waiting for one, runs by phase, agent sessions with pending approvals, running tools and background jobs, MCP servers, cache activity, memory, recent failures |
| waiting runs | fixed | a run waiting for a free sequence blocked in admission: it could not be inspected, and cancelling it timed out. It now starts its stream from a helper, answers at once, shows the phase `waiting` and cancels at once |
| failure log | done | a bounded log of the latest failures (run, model down, turn, tool crash or timeout, session down, MCP down or failed call, conversation repaired), each with what the release did next; `erllama_server_failures_total{kind}` in `/metrics` |
| reattach | done | `erllama agent --attach ID` from any terminal: the session's state, approvals and jobs, then its latest turn replayed from the event buffer (a gap is reported) and followed live; replayed approvals are asked only if still pending |
| MCP outages | done | a call that fails without an answer is recorded and makes the release check the server; an outage is recorded once. Status catches up within the client's own timeouts (about 30 s in the test) |
| malformed requests | fixed | a request the native code rejects crashed the shared model process; the caller now gets `{error, badarg}` |
| examples | done | `examples/embedded_app` (erllama in a supervised OTP application) and `examples/service_client` (a node using the release over distribution) |

Tests (stub models and scripted agents, repeatable):
- `erllama_server_models_SUITE:waiting_run_answers_and_cancels` (fails before the fix: `info` on a waiting run timed out).
- `erllama_server_ops_SUITE`: status shows running and waiting; a slow tool leaves another conversation alone (its turn took 21 ms while the tool ran); cancelling one run ends only that run; a tool timeout is recorded; a killed model ends its runs with `model_down` while a run on another model finishes and the next request reloads it; a killed session is recorded, its conversation freed and resumed, and another session goes on.
- `erllama_server_rpc_SUITE:tty_agent_attach_shows_what_happened`: a real terminal attaches to a session stopped at an approval, sees the output produced while nobody was attached, approves, and follows the turn to its end.
- `erllama_server_mcp_SUITE:server_down_and_back`: the failed call and the outage are each recorded, the outage once.
- Gated `erllama_real_model_SUITE:bad_request_leaves_the_model_running`.

### Benchmark: sharing one model and its cache

`scripts/bench_cache.py` over HTTP, Apple M4 Pro (48 GiB, Metal), Qwen2.5 1.5B Instruct Q4_K_M, `n_ctx 4096`, `parallel 4`, RAM cache tier only, greedy, one client at a time. Two runs on fresh releases agree within 30 ms; run 2:

| Step | First token | Total | Prompt tokens | Cache |
|---|---|---|---|---|
| 1. first request, the model loads on demand | 207 ms | 266 ms | 32 | miss |
| 2. same model, loaded | 30 ms | 88 ms | 32 | miss |
| 3. a 2737-token system prompt, cold | 1459 ms | 1517 ms | 2737 | miss, saves its first 2048 tokens |
| 4. same system prompt, another question | 429 ms | 487 ms | 2738 | prefix hit (2048 tokens) |
| 5. the same after the cache dropped its rows | 1454 ms | 1504 ms | 2738 | miss |
| 6. turn 2 sent whole by a stateless client | 50 ms | 101 ms | 2756 | prefix hit (turn 1 with its reply) |
| 7. turn 2 through the Responses API (`previous_response_id`) | n/a | 94 ms | 2756 | resumed from turn 1's key |

- The model is loaded once and every client uses it; step 1 includes its load (the file was already in the page cache).
- Unrelated requests share a prefix only at the cache's boundaries (multiples of `boundary_align_tokens`, 2048): a 1937-token system prompt in an earlier run was never reused.
- A later turn finds the previous turn's saved context whether its client links it or sends the whole history; linked turns (conversations, agent sessions, `previous_response_id`) resume from the exact key.
- After an eviction the request is prefilled again: slower, same answer.
- Admission under limited sequences is measured in the shared-serving benchmark above.

## Tensor parallelism, milestone 1: a reduction between hosts (2026-11-23)

Experimental. See [internals/tensor-parallel.md](../internals/tensor-parallel.md).

| Feature | State | Notes |
|---|---|---|
| rank-local API | done | `erllama_tp`: open this node's GPU as one rank, matvec, reduce, close; `erllama_tp_rank` holds a handle in a process for a remote coordinator |
| bootstrap bridge | done | the communicator's bootstrap callback (`ggml_backend_comm_init_rank`) goes through an Erlang agent and a coordinator, rank-ordered, bounded by the open's deadline |
| test coordinator | done | `test/erllama_tp_coordinator.erl`; group formation belongs to erllama_cluster |
| portable tests | done | `erllama_tp_bootstrap_SUITE`, in CI |
| hardware tests | done on two hosts | `erllama_tp_nccl_SUITE`, opt-in; hosts prepared with `scripts/tp_hosts.sh` |

Hardware run: two ASUS GX10 (NVIDIA GB10, compute capability 12.1, about 121 GB unified memory, driver 580.178.04, CUDA 13.0), one rank each; llama.cpp pin v1.0.0; NCCL 2.32.6 (pip wheel `nvidia-nccl-cu13`); OTP 28.3.1.

| Case | Result |
|---|---|
| fixture: rows `[1,2,3,4]`, `[4,3,2,1]`, input `[1,2,3,4]`, columns split | partials `[5,10]` and `[25,10]`; both ranks `[30,20]`, exact |
| 4096 values per rank (FP32 path) | tolerance 1e-6 of the summed magnitudes + 1e-6; worst error 0.02 of the bound |
| 2,097,152 values per rank (BF16 path) | tolerance 1.6e-2 of the summed magnitudes + 1e-6; worst error 0.29 of the bound |
| rank 0 alone | `{error, join_timeout}` after 2031 ms (coordinator deadline 2000 ms); the node stays usable |
| open, reduce, close three times | each round correct; free device memory flat (rank 0 about 5.2 GB free, rank 1 about 7.7 GB, another process holding the rest) |
| data path | NCCL `NET/IB` over RoCE, `rocep1s0f0` and `roceP2p1s0f0` at 200 Gb/s each (back-to-back links), out-of-band on `enP7s7`; GPU Direct RDMA disabled on GB10 |

The run is from a third machine:

```bash
scripts/tp_hosts.sh setup gx10-fa8e.local gx10-e25e.local
scripts/tp_hosts.sh sync  gx10-fa8e.local gx10-e25e.local
scripts/tp_hosts.sh build gx10-fa8e.local gx10-e25e.local
ERLLAMA_TP_HOSTS=gx10-fa8e.local,gx10-e25e.local \
    rebar3 ct --dir apps/erllama/test --suite erllama_tp_nccl_SUITE
```

Not done: a transformer block split across ranks (milestone 2), a full model (milestone 3), failure handling during generation and coordinated checkpoints (milestone 4), three or more hosts, Metal. No throughput or latency is claimed: nothing was measured beyond correctness.

## Tensor parallelism, milestone 2: a model split over the hosts (2026-11-23)

Experimental. A GGUF loaded with `tp_rank => Rank` is split with llama.cpp's own tensor split over one device per rank: this rank's GPU, wrapped, and placeholders for the others that hold and compute nothing. Collectives go through the rank's communicator; the logits are summed over the ranks exactly. No vendored file is changed. See [internals/tensor-parallel.md](../internals/tensor-parallel.md).

Run on the two GB10 hosts (pin v1.0.0, NCCL 2.32.6 over RoCE), TinyLlama 1.1B Chat (Llama architecture, Q4), `n_ctx 512`, flash attention on, prompt "The quick brown fox jumps over the lazy dog" (12 tokens), then one greedy token from the cache. Reference: the same file on one GB10.

| Case | Prompt | Next token from the cache | Reductions | Placeholders reserve (the other rank's share) | Free GPU memory used by the load |
|---|---|---|---|---|---|
| even split | max \|logit − one GPU\| 0.272 (largest \|logit\| 13.96), top 5 identical | 0.220 (largest 14.41), top 5 identical | 88 on each rank, same sizes in the same order | 385.3 MiB on each | 436 MiB and 412 MiB |
| uneven split (`tensor_split => [3.0, 1.0]`) | 0.246, top 5 identical | 0.213, top 5 identical | 88 on each, same order | 232.5 MiB on rank 0, 538.1 MiB on rank 1 | 605 MiB and 303 MiB |

- Tolerance, set before comparing: the same argmax, and a largest logit difference within 2% of the largest logit. Both ranks hold identical logits.
- 88 reductions: two per layer (attention output and feed-forward output) over 22 layers, for each of the two decodes.
- Memory is the free memory reported by the GPU before and after the load (unified memory shared with another process), so it is approximate; the placeholder figure is exact.
- The differences come from the split itself (matrix shapes change the CUDA kernels used); they are not shown to be rounding only. Throughput and latency were not measured.

Not done: serving through the normal erllama path with one sampling rank driving the others, failure handling during generation, coordinated checkpoints, a model larger than one GPU, more than two hosts, Metal.

## Tensor parallelism, milestone 3: serving a split model (2026-11-23)

Experimental. Rank 0 loads the split model through `erllama:load_model/2` (`tp_rank` and `tp_sink` options) and serves it with the normal API; it samples once per step. Every batch and KV change is sent, numbered, to the other ranks, which apply them in order. Ranks with a different file, options or engine build are refused. A split model saves no KV yet. See [internals/tensor-parallel.md](../internals/tensor-parallel.md).

Run on the two GB10 hosts over RoCE, TinyLlama 1.1B Chat Q4, `n_ctx 1024`, flash attention on, one sequence, greedy; reference: the same file loaded through erllama on one GB10.

| Check | Result |
|---|---|
| greedy completion of "The quick brown fox", 32 tokens | identical to one GPU, all 32 |
| stream of 128 tokens after a 242-token prompt | ends `length`, 128 tokens |
| cancel after 10 tokens | ends cancelled; the next request returns the same 32 greedy tokens |
| followers in step | 9196 reductions on each rank, same sizes in the same order |

| 242-token prompt, 128 tokens | First token | Prompt tokens/s | Decode tokens/s | Median gap |
|---|---|---|---|---|
| split over the two hosts (two runs) | 39.0 and 53.1 ms | 6202 and 4559 | 233.7 and 172.8 | 4.17 and 5.75 ms |
| one GB10 (two runs) | 22.7 and 22.1 ms | 10673 and 10935 | 288.9 and 290.8 | 3.41 ms |

- A 2048-value reduction (FP32), timed in each rank's process with the copy to and from the GPU: about 230 µs. Each token makes 44 reductions inside the model graph (two per layer), without the copies.
- For a model this small, splitting is slower than one GPU: the time saved computing half of each layer is smaller than the time spent reducing. The point of splitting is a model that does not fit one GPU; that run is still to be made (the test hosts' GPUs are shared with another workload).
- The split runs vary between runs (39 to 53 ms first token); the one-GPU runs do not. Only these configurations were measured.

Not done: failure handling during generation (a rank lost mid-stream, a stale or duplicate step, cancel during a collective), coordinated checkpoints and cache hits, a model larger than one GPU, more than two hosts, Metal.

## Tensor parallelism over plain TCP (2026-11-23)

Experimental. A group runs over ordinary Ethernet with NCCL's socket transport: set `NCCL_NET=Socket` and `NCCL_SOCKET_IFNAME` in each rank node's environment. Nothing in erllama changed; the launch recipe, the diagnostics and the checks are in [internals/tensor-parallel.md](../internals/tensor-parallel.md).

Same two GB10 hosts, same build, TinyLlama 1.1B Chat Q4, `n_ctx 1024`, a 242-token prompt and 128 streamed tokens, rank 0 serving (`erllama_tp_nccl_SUITE`, all 9 cases passed in each configuration). Three configurations: TCP over the two 200 Gb/s links (`NCCL_SOCKET_IFNAME==enp1s0f0np0,enP2p1s0f0np0`), TCP over the 2.5 Gb/s management port (`==enP7s7`), and RoCE (no NCCL variable set).

| What NCCL used | TCP, 200 Gb/s links | TCP, 2.5 Gb/s port | RoCE |
|---|---|---|---|
| network | `Using network Socket` | `Using network Socket` | `Using network IB` |
| devices | `[0]enp1s0f0np0 [1]enP2p1s0f0np0` | `[0]enP7s7` | `[0]rocep1s0f0 [1]roceP2p1s0f0` (RoCE) |
| channels | 8, `via NET/Socket/0` and `/1` alternating | 4, `via NET/Socket/0` | 8, `via NET/IB/0` and `/1` alternating |
| bytes sent per host by 17 reductions of 2M values (BF16) | 37.4 and 37.3 MB, nothing on the 2.5 Gb/s port | 74.7 MB | none in the interface counters (RDMA) |
| one such reduction | 7.2 ms | 18.7 ms | 4.5 ms |

| Two runs each | First token | Decode tokens/s | Median gap | 2048-value reduction |
|---|---|---|---|---|
| TCP, 200 Gb/s links | 132.8 and 126.8 ms | 65.6 and 55.8 | 14.90 and 18.18 ms | 342 and 581 µs |
| TCP, 2.5 Gb/s port | 197.5 and 213.8 ms | 43.3 and 47.5 | 24.24 and 21.54 ms | 421 and 492 µs |
| RoCE | 45.2 and 48.7 ms | 177.1 and 228.9 | 5.70 and 4.35 ms | 163 and 118 µs |
| one GB10, no split | 21.3 to 22.9 ms | 283.9 to 289.5 | 3.42 to 3.50 ms | |

- In every configuration the greedy completion was identical to one GPU for all 32 tokens, the 128-token stream finished, a cancel after 10 tokens ended cleanly and the next request gave the same tokens, and both ranks made 9196 reductions in the same order.
- Decoding is bound by the latency of the 44 small reductions per token, not by bandwidth: TCP over the 200 Gb/s links is only 1.2 to 1.5 times faster than over the 2.5 Gb/s port, and 3 to 4 times slower than RoCE.
- The NIC speeds are the hosts' reported link speeds (`/sys/class/net/*/speed`). The two hosts' GPUs are shared with another workload; the runs vary.

Not run over TCP: the rank-loss, stale-command and checkpoint cases, which come with milestone 4.

## Tensor parallelism, milestone 4a: failures while serving (2026-11-23)

Experimental. The split model is now served from a node that is no rank (`erllama_model_tp`): tokenizing and chat templates run there on the GGUF's vocabulary, and decoding, sampling and KV changes run on rank 0. Commands to the other ranks carry the group's generation; a stale or duplicate one is dropped before any GPU work, a gap stops the follower. A split model's decode is not interrupted: a cancel waits for the step in flight. When a rank goes, the coordinator kills the others and the model fails its requests once and stops. See [internals/tensor-parallel.md](../internals/tensor-parallel.md).

Run on the two GB10 hosts over RoCE, TinyLlama 1.1B Chat Q4, `n_ctx 2048`, `n_batch 512`, flash attention on, greedy; the serving node on the first host. Each case starts its own nodes (`erllama_tp_failure_SUITE`).

| Case | Result |
|---|---|
| rank 1's OS process killed after 10 streamed tokens | one terminal error, `{tp_rank_lost, {nodedown, rank 0}}`, 6.3 ms after the kill (the coordinator killed rank 0, held in the collective); the serving node lists its models in 8 ms; the model has stopped |
| a new group (new nodes, generation 2) | the same 32 greedy tokens as the first group |
| a command of the previous generation, then one already applied | both dropped (stale 1, duplicate 1); no reduction made; the next request gives the same tokens |
| a command past a gap | the follower stops (`{out_of_order, 70, 75}`); the next request fails once in 540 ms and the model stops |
| cancel right after a 962-token prompt is admitted | one terminal result, `cancelled`, after the first 128-token step; 2948 reductions on each rank, same order; the next request gives the same tokens |

Serving from the separate node, `erllama_tp_nccl_SUITE` (same model, `n_ctx 1024`, 242-token prompt, 128 tokens): greedy identical to one GPU for all 32 tokens; first token 64.3 ms, decode 150.6 tokens/s, median gap 6.57 ms (one GB10: 34.2 ms, 287.9 tokens/s, 3.42 ms); 9196 reductions on each rank. Each step now crosses the serving node and rank 0 once more than in milestone 3 (rank 0 serving: 173 to 234 tokens/s); one run each, so the share of that hop is not measured apart from the variation between runs.

Over TCP (`NCCL_NET=Socket` on the two 200 Gb/s links) the three cases pass too: the terminal error came 6.3 ms after the kill, the request after a gap failed once in 119 ms, and the cancel during prefill ended once, 68 ms after the cancel.

Not done: coordinated checkpoints and cache hits, a model larger than one GPU, more than two hosts, Metal.

## Tensor parallelism, milestone 4b: checkpoints of a split model (2026-11-23)

Experimental. A split model's cache entry holds every rank's share of the KV, packed at the same committed token: rank 0 puts the pack in the stream the other ranks follow, after every batch it decoded, and frames the shares (size and CRC32C each) into one payload, so the cache publishes all of them or none. A share that does not come within `tp_kv_timeout` fails the save and nothing is published. The world size and `tensor_split` are part of the KV identity. See [internals/tensor-parallel.md](../internals/tensor-parallel.md).

Run on the two GB10 hosts over RoCE, TinyLlama 1.1B Chat Q4, `n_ctx 2048`, a 330-token prompt, 32 greedy tokens; one cold save at the 320-token boundary and a finish save (`erllama_tp_checkpoint_SUITE`, two runs).

| Case | Result |
|---|---|
| a checkpoint, then the group unloaded and a new one formed on new nodes | the same prompt resumes from it (`partial`, 320 tokens read, restored on both ranks), same 32 tokens as the first group |
| rank 1 sends no share for the cold save | that save fails and the finish save is made (rank 0: 1 failed, 1 made); a new group misses on the same prompt (`cold`, nothing read) and gives the same tokens |

| | Time on rank 0 | Size |
|---|---|---|
| checkpoint (pack on both ranks, gather, frame) | 78 to 105 ms | 17.8 MB |
| restore (check, unpack on both ranks) | 49 and 68 ms | |

- Each rank's share is the size of the whole sequence's KV: the split backend reads the other ranks' slices as zeros. The checkpoint is twice the single-GPU size with two ranks.
- Timed apart from decoding, in rank 0's process; the cache's own framing and publication come on top.
- Over TCP (`NCCL_NET=Socket` on the two 200 Gb/s links) both cases pass too, with the same hits and misses (checkpoint 81 to 93 ms).

## Tensor parallelism: a model larger than one GPU (2026-11-23)

Experimental. Llama 3.3 70B Instruct in F16 (four GGUF parts, 131.4 GiB, 80 layers) on the two GB10 hosts, each with 121 GB of memory shared by its CPU and GPU, the hosts' other workload stopped for the run (`erllama_tp_capacity_SUITE`). `n_ctx 2048`, flash attention on, over RoCE, served from its own node through `erllama_model_tp`.

| | One GB10 | Split over the two |
|---|---|---|
| placement | auto-fit (8 GiB margin) chose 68 of 81 layers on the GPU, 13 on the CPU | 81 of 81 layers on the GPU of each rank, each holding its share; only the 2.0 GB token embedding table in CPU memory, as llama.cpp always places it |
| result | the load ran the host out of memory: the kernel killed the node (`Out of memory: Killed process ... beam.smp`, 144.6 GB of address space) | loaded and served |
| memory available before | 117 GB | 125.4 and 125.5 GB |
| peak use during load and generation | (killed) | 74.3 GB on rank 0, 73.9 GB on rank 1 |
| per rank | | 66,287 MiB of weights (its half), 320 MiB of KV, 317 MiB of compute buffers; the other rank's share (70.2 GB) reserved as address space only |

| Streamed completion, 64 tokens after a 220-token prompt (two runs) | First token | Decode | Median gap |
|---|---|---|---|
| split over the two GB10s | 1635 and 1582 ms | 3.34 and 3.33 tokens/s | 299.5 and 300.0 ms |

- Both streams ended `length` with 64 tokens. There is no one-GPU reference to compare the tokens with: no single host holds the model.
- Decoding reads each rank's 66 GB of F16 weights once per token, so the rate is bound by memory bandwidth, not by the reductions; a quantized file would be faster and smaller.
- The one-host case runs only with `ERLLAMA_TP_CAPACITY_ONE_HOST=1`: on hosts whose GPU shares the host's memory it ends with the node killed.

Not done: more than two hosts, Metal.

## Tensor parallelism: GLM-5.3-Flash over two GB10s (2026-11-23)

Experimental. GLM-5.3-Flash (`glm5-next`, 320B parameters with 18B active, 45 layers; MLA and KDA linear attention, a sparse-attention indexer) split over the two GB10 hosts with llama.cpp's own tensor split and served through `erllama_model_tp`, over RoCE, drafting off. Files: `ggml-org/GLM-5.3-Flash-GGUF` Q4_K (181 GB, revision `93ae818`) for speed and capacity, `unsloth/GLM-5.3-Flash-GGUF` UD-IQ2_XXS (102 GB, revision `a38483c`, small enough for one GB10) for correctness (`erllama_tp_glm_SUITE`). See [internals/tensor-parallel.md](../internals/tensor-parallel.md).

What it took:

- The other ranks now open the same context as rank 0: rank 0 adds recurrent-state snapshots for hybrid models, the others did not.
- Split loads keep no recurrent-state snapshots: with one, llama.cpp's split backend aborts every rank at the first decode (a view of the recurrent cache it cannot split). A hit on a cached prefix followed by new tokens still restores; a hit on a whole prompt is served cold.
- `load_mode => direct_io`: on a GB10 the page cache of a buffered load is not given back to CUDA, and a CUDA init failed with out of memory while the host showed 117 GB available.

| Correctness (UD-IQ2_XXS, split against one GB10) | Result |
|---|---|
| next token at the prompt and three decodes | identical at every step; top 5 identical or 4 of 5 |
| largest logit difference | 4.4% at the prompt (its reductions go through BF16), growing to 7.1% by the third decode |
| 32-token greedy replies to four prompts, alone | 3 of 4 byte-identical; the fourth is a near-tie that batching alone changes on one GB10 |
| reductions | 360 on each rank, same sizes, same order |
| four prompts at once | replies differ from the solo ones, and between two batched runs, on one GB10 as on the split model (when each request is admitted changes what a step computes); a cancel midway ends only that request |
| checkpoint, then a new group | the same prompt resumes from it (`partial`, 256 tokens restored on both ranks) with the same 32 tokens; 318 MB, 1.55 s to pack, 0.73 s to restore |

| Q4_K, 250 tokens, TensorFold's four prompts (`docs/dec.py`), 3 trials | erllama split, no drafts | TensorFold, no drafts | TensorFold, MTP | TensorFold, DFlash2 |
|---|---|---|---|---|
| 1 stream | 19.9 to 20.5 tokens/s (first token about 295 ms) | 28.8 | 41.3 | 40.9 |
| 4 streams, total | 39.7 to 40.6 tokens/s | 28.5 | 61.9 | 78.5 |

- Memory with Q4_K loaded, `n_ctx 8192`, 4 sequences: 30 to 32 GB left available on each GB10 (125 to 126 before).
- TensorFold runs a 4-bit EXL3 quant with FP8 KV and its own kernels; tokens are compared within one engine, never across.
- Not done: speculative decoding (TensorFold's single-stream lead is mostly drafts), a longer context, vision, and FP8 KV (refused by the pin for this model).
