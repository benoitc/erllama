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

## Phases 3c to 7

Not started.
