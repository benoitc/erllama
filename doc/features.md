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

## Phases 3 to 7

Not started.
