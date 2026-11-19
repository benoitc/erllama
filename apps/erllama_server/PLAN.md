# erllama umbrella + erllama_server: plan

## Context

The maintainer's implementation brief (kept outside the repository) asks for one repository with two OTP applications, one relx release, and a CLI connected to the release over Erlang RPC, built in seven phases.

Phase 1 was started on the local branch `refactor/umbrella-layout`, then parked so 0.12.0 could ship (now released, `v0.12.0` = `ad47520`). The branch is three commits ahead of the old main (`b9554ea`), and main has moved by #79, which touched 29 library files (src, test, guides, README, CHANGELOG, AGENTS).

This plan replaces the first version of this file. The changes the brief brought:

- **CLI** is an escript shipped beside the release and started by a launcher that uses the release's bundled runtime. There is no third OTP application, and the client never starts erllama or loads the NIF.
- **Livery handles both HTTP serving and outbound HTTP.** Downloads use `livery_client`, verified to exist: `request/3,4`, streamed bodies via `read/2` and `stream_next/1`, custom headers for `Range`, and timeout and retry layers. Hackney is not used directly.
- **barrel_mcp** handles MCP; only the bridge to the agent runtime is new.
- **No copying** from `erllama_server`, `barrel_inference_server` or ds4. They may only inform requirements and behaviour.
- **erllama keeps its package identity and public API.** Library consumers never pick up server dependencies or listeners: the server deps go only in `apps/erllama_server/rebar.config`.
- **The cache and NIF invariants in AGENTS.md stay intact.**

Kept from before:
- `guides/` and `internals/` stay at the root, and ex_doc runs from the root.
- The HTTP endpoint table (Phase 3).
- The progress record lives in `doc/features.md`.
- **One PR per phase:** stop for validation after each phase, with no commit or push without approval.

---

## Step 0: GitHub release 0.12.0 (done)

Published on 2026-11-19 at 09:54 UTC as "erllama 0.12.0", marked Latest: https://github.com/benoitc/erllama/releases/tag/v0.12.0
- The tag `v0.12.0` points at `ad47520`, the merge of #79.
- The notes are the CHANGELOG 0.12.0 section, as is.
- Hex publishing stays with you.

Phase 1 starts only after this.

## Phase 1: repository and build migration (finish on `refactor/umbrella-layout`)

### 1.A Rebase onto main (0.12.0)
- `git rebase main` on `refactor/umbrella-layout`. Git's rename detection should carry #79's changes to `src/` and `test/` into `apps/erllama/`.
- Resolve conflicts in the files both sides edited, keeping main's content and re-applying only the branch's path edits: `AGENTS.md` (layout tree and commands), `README.md` (it moves to `apps/erllama/README.md` and a new root README is added), `CHANGELOG.md` (moves), and the guides that #79 edited (`caching`, `configuration`, `examples`, `generation`, `loading`).
- Check that the files #79 added landed under `apps/erllama/`: `src/erllama_fingerprint.erl`, `test/erllama_fingerprint_tests.erl` and the others.
- Re-run the full gate before any new work.

### 1.B Finish the skeleton and release (turn the WIP commit into a clean one)
- **Keep** `apps/erllama_server` (app, sup, CT suite), the relx release in the root `rebar.config`, `config/sys.config` and `config/vm.args.src`. The node is `-name erllama_server@127.0.0.1` with distribution on loopback only.
- **CLI escript scaffold** (only `bin/erllama --version` and `ping` in this phase):
  - Main module `erllama_server_cli` in `apps/erllama_server/src`.
  - Root `rebar.config`: `{escript_main_app, erllama_server}`, `{escript_name, erllama}` and `{escript_incl_apps, [erllama_server]}`, so the erllama app and its NIF are not bundled.
  - The relx overlay ships the escript as `bin/erllama_cli` plus a `bin/erllama` shell launcher. The launcher runs `"$REL_ROOT"/erts-*/bin/escript` when ERTS is included (prod profile), otherwise the system `escript`.
- Update `apps/erllama_server/PLAN.md` on the branch to this plan, and `doc/features.md` to the current status.

### 1.C Prove erllama builds and is consumed on its own
- **`scripts/check_library_package.sh`:**
  - `rebar3 hex build --app erllama`, then unpack into a temporary consumer project's `_checkouts/erllama`.
  - `rebar3 compile` there, then an eunit test that starts erllama, calls `erllama_nif:kv_format/0`, and loads a stub model with `erllama:load_model/2` using `backend => erllama_model_stub`.
  - Assert that the consumer's lock has no `livery` or `barrel_mcp`.
- **CI:**
  - A `library-package` job running that script on ubuntu and macOS.
  - A `release` job: `rebar3 as prod release`, then `bin/erllama_server daemon`, `ping`, `bin/erllama --version` (with a check that the erllama app is not loaded in the escript VM), and `stop`.
  - The workflow path fixes from commit `c832a72` stay.

### Phase 1 verification
- **Full gate** from the root on OTP 28.3.1: fmt --check, compile, eunit, proper, ct, lint, dialyzer, xref. Deliberate fmt and lint violations must still fail.
- **Real-model suite** via `--dir apps/erllama/test --suite erllama_real_model_SUITE`, on stories260K and TinyLlama.
- **Fresh clone** build of the branch: `priv/erllama_nif.so` is in place.
- `rebar3 ex_doc` from the root: no new warnings, links resolve.
- `scripts/check_library_package.sh` passes.
- The release starts, pings and stops, and the escript runs on the release's ERTS.
- Green CI, then one PR for Phase 1. Stop for validation.

---

## Phases 2 to 7 (outline; each is planned in detail and shipped as its own PR)

**2. Release, service API, RPC client.**
- `erllama_server_api` is a versioned module (`?API_VSN`) shared by the RPC and HTTP adapters.
- Run and conversation ids, and events with sequence numbers.
- Each run keeps a bounded ring buffer for replay. Each subscriber gets a bounded queue; one that overflows is dropped with a gap marker. `attach(RunId, Cursor)` returns the replay or `{gap, OldestSeq}`.
- Cancel, attach, detach and fetching the result.
- The CLI escript starts a temporary hidden node named `erllama_cli_<random>@127.0.0.1`.
  - The cookie is documented: by default the release user's `~/.erlang.cookie`, with an override via env or flag. The relx `RELX_COOKIE` caveat found in Phase 1 is documented.
  - Control operations use `erpc` with timeouts; events stream as messages to a local subscriber process.
- Clear errors for an unreachable release and for incompatible protocol versions.
- TLS distribution for remote administration is documented and off by default.
- Workspaces always refer to the release machine.
- **Tests** run in CT with `peer`: streaming, reconnect with a cursor, a gap, cancel, a version mismatch, and an assertion that the client node never loads `erllama`.

**3. Model management and Livery API adapters.**
- **Model registry:** persistent, with aliases, defaults, local GGUF registration and metadata inspection.
- **Downloads** from `hf://`, `https://` and `ollama://` through `livery_client`: a streamed body into a `.part` file, `Range` resume, a sha256 check, progress events and cancellation.
- **Model lifetime:** keep-alive and capacity reporting go through erllama's existing admission, `erllama:pressure/0`, `erllama_model_evictor` and the scheduler (no second scheduler).
- Startup config validation.
- **HTTP:** one normalised request path through `erllama:chat/3`, `stream/3` and `embed_batch/2`, with thin protocol adapters on Livery (`livery_resp:sse/3` and `ndjson`; cancel on `livery_disconnect`).

  | Endpoint | Protocol | Streaming |
  |---|---|---|
  | `GET /v1/models`, `GET /v1/models/:id` | OpenAI | n/a |
  | `POST /v1/chat/completions` | OpenAI Chat Completions | SSE, `[DONE]` |
  | `POST /v1/responses`, `GET`/`DELETE /v1/responses/:id` | OpenAI Responses (tool calls, `previous_response_id`) | SSE typed events |
  | `POST /v1/completions` | OpenAI Completions | SSE |
  | `POST /v1/embeddings` | OpenAI | n/a |
  | `POST /v1/messages`, `/v1/messages/count_tokens` | Anthropic Messages | SSE |
  | `/api/chat`, `/api/generate`, `/api/embed[dings]`, `/api/tags`, `show`, `pull`, `delete`, `copy`, `ps`, `version` | Ollama | NDJSON |
  | `GET /health`, `/ready`, `/metrics` | ops | n/a |

- **Across every adapter:** request ids, bearer auth, body and concurrency limits from Livery middleware, and one error shape.
- A compatibility matrix lives at `guides/server/compatibility.md`; unsupported fields are rejected explicitly.
- **Reused from erllama:** chat templates, grammar/json_schema, stop sequences, and the chat parser for tool calls.
- Multimodal is exposed with its real limits.

**4. Agent runtime and native tools.**
- One supervised gen_statem per conversation, with at most one mutating run per conversation.
- Tools run in monitored workers: file read (paginated), list and search, create and exact edit, and shell with a timeout, bounded output, background status, and a process-group kill.
- **Run loop:**
  - budgets for tokens, rounds and elapsed time;
  - queued input and interruption;
  - concurrent tool calls with results kept in call order, and workspace mutations applied in order;
  - approval policy and requests;
  - errors returned as tool results when the model can recover.
- Tools declared by HTTP clients are executed by the client, never on the server.

**5. barrel_mcp bridge.**
- **Client side:**
  - stdio and Streamable HTTP clients (`barrel_mcp:start_client/2`);
  - namespaced binary tool names with collision handling;
  - a catalog fixed for each round and refreshed between rounds;
  - resources, prompts and structured content, with unsupported content reported rather than dropped;
  - calls wrapped in try/catch, because the client exits when the server closes, and every call has a timeout;
  - the same policy as native tools;
  - an optional server that is down never blocks inference.
- **Opt-in MCP server:** `livery_mcp:router/1` on the same Livery service. Its known gaps (no disconnect relay, weaker origin defaults) are fixed upstream or worked around, verified first.

**6. Durable conversations.**
- A transcript and run/tool log, stored apart from the KV cache files.
- List, attach, resume, fork, delete and export.
- A tool's intent is stored before it runs and its outcome before generation continues. An uncertain outcome is marked unresolved and never retried.
- On resume: restore exact KV when possible, otherwise prefill the transcript.
- Compaction, automatic and manual.
- The disconnect semantics as specified.

**7. Terminal, documentation, full release validation.**
- Multiline input, streamed rendering, prefill and tool progress, and approvals.
- `/new` `/sessions` `/switch` `/fork` `/compact` `/quit`.
- Closing the terminal never unloads models or stops the release.
- Server guides, and a final validation that lists which real-model checks ran and which were skipped.

**Out of scope for every phase:** pipeline or tensor parallelism and cluster routing; no new repository, package publication or deployment.
