# Phase 2: service API, runs and the RPC client

## Context

The maintainer's implementation brief (kept outside the repository) describes seven phases for turning erllama into a local inference and agent product. Phase 1 is committed on `refactor/umbrella-layout` (commits 38a45d7..8401bc3) and not pushed yet:
- the umbrella layout;
- the standalone Hex package check;
- an `erllama_server` app that is only a skeleton: an empty supervisor and a CLI with `--version` and `ping`.


Phase 2 is the first phase with real behaviour. The release owns generation runs. The `erllama` CLI drives them over Erlang distribution:
- it starts a run on a model;
- it streams the tokens as they arrive;
- it cancels, detaches and reattaches with a cursor;
- it gets the result.

It also gives clear errors when the release is down or speaks another API version. Everything goes through one small versioned service API that the Phase 3 HTTP adapters will reuse.

**Out of this phase:**
- the model registry and downloads (Phase 3);
- the agent loop and tools (Phase 4);
- MCP (Phase 5);
- persistence (Phase 6);
- terminal polish (Phase 7).

Loading a model in Phase 2 is an interim `load PATH`, which the Phase 3 registry replaces.

The branch is `feat/server-rpc`, cut from `refactor/umbrella-layout`. Its PR is rebased onto main after the Phase 1 PR merges. Each phase stops for review before it is committed.

## What erllama already gives us (reused, not reimplemented)

- **Streaming:** `erllama:stream/3` returns `{ok, Ref}` and sends `{erllama, Ref, Event}` to the `to` pid (`apps/erllama/src/erllama.erl:870`). The events are `{token, Bin}`, `{thinking, Bin}`, `{thinking_end, Bin}`, `{done, Stats}` and `{error, R}`. `Stats` holds `finish_reason`, `finish_key`, `generated`, the token counts and the timings.
- **Cancel:** `erllama:cancel(Ref)` is idempotent and still delivers `done` with `finish_reason => cancelled`.
- **Chat rendering on a real model:**
  - `erllama:chat_apply/3`, then `tokenize/3` with `#{add_special => false, parse_special => true}`;
  - then `stream/3` with `sampler_opts` and `stop_sequences` merged in;
  - then `chat_parse/3` on the final text.

  This is the same path `erllama_chat:chat/3` uses.
- **Chat on the stub:** the stub backend returns `{error, chat_not_supported}` from `chat_apply` (`erllama_model.erl:2451`). The fallback is `erllama:render_chat_template/2`, then `stream/3`, which the stub implements. This keeps routine tests deterministic and lets the stub's `step_delay_ms` make runs slow enough to test cancel.
- **Multi-turn warmth:** the run passes the previous turn's `finish_key` as `parent_key` (the AGENTS.md pattern). No sticky sessions in Phase 2.
- **Models:** `erllama:load_model/2`, `unload/1`, `list_models/0`, `model_info/1` and `pressure/0`.
- **Finding:** erllama does not monitor the stream's `to` pid (no monitor in `erllama_model.erl`). If the process that owns a stream dies, generation runs to its token limit. The run process therefore traps exits and cancels its ref in `terminate/2`. A brutal kill still leaks until the limit; this is documented, and it's not changed in the library this phase.

## Design

### Supervision (`apps/erllama_server/src`)

```
erllama_server_sup            rest_for_one; validates the app env in init/1
├── erllama_server_runs       owns the run table (ETS, public reads), monitors runs
├── erllama_server_convs      in-memory conversations (ETS), one active run each
└── erllama_server_run_sup    simple_one_for_one, temporary: one erllama_server_run per run
```

`rest_for_one`: if the run table restarts, the runs are restarted too, so the table never points at the wrong pids.

### Modules

- **`erllama_server_api`** is the versioned facade, `-define(API_VSN, 1)`. It is the only module the RPC client calls, and Phase 3 HTTP will call it too. Every function returns `{ok, _}` or `{error, Reason}`, never throws.
  - `hello(ClientApiVsn)`: `{ok, #{api, server_vsn, erllama_vsn, node}}` or `{error, {incompatible_api, #{server, client}}}`.
  - `status()`: versions, the loaded models with their phase, the active runs and `erllama:pressure/0`.
  - `models()`, `load_model(Path, Opts)`, `unload_model(Id)`. These are interim and are replaced by the Phase 3 registry.
  - `start_run(Spec)` returns `{ok, #{run_id, conversation_id}}`. The spec is `#{model, messages | prompt, conversation_id => _, opts => _}`. `opts` is a whitelist: `response_tokens`, the sampling keys, `stop_sequences` and the chat keys. Anything else returns `{error, {invalid_option, K}}`.
  - `attach(RunId, Pid, Cursor)`: `{ok, #{run_pid, replay => [Event], status}}` or `{error, {gap, OldestSeq}}` or `{error, not_found}`. `Cursor` is the next seq wanted (0 means from the start).
  - `detach(RunId, Pid)`, `cancel(RunId)`, `result(RunId)` (`{ok, Result}` or `{error, running | not_found}`) and `runs()`.
  - `conversation(Id)`: its messages and its active run.
- **`erllama_server_run`** is a gen_server, one per run, that traps exits.
  - **It owns the erllama stream ref.** It renders through `erllama_server_generate`.
  - **Events:** it turns each erllama event into a sequenced event `#{seq, type, data}`. The types are `started`, `token`, `thinking`, `done`, `cancelled` and `error`. It keeps them in an `erllama_server_ring`.
  - **Delivery:** it sends `{erllama_server_run, RunId, Event}` to each subscriber it monitors.
  - **Flow control:** subscribers ack with `RunPid ! {ack, self(), Seq}`. A subscriber with more than `subscriber_max_unacked` unacked events is dropped and sent `{erllama_server_run, RunId, {dropped, NextSeq}}`, so it can reattach with a cursor.
  - **Disconnect:** a subscriber that goes `DOWN` is removed, and the run continues. Runs survive CLI disconnect, as the prompt specifies.
  - **End of a run:** on its terminal event, it sets the result (content, reasoning, tool calls when parsed, stats, finish_reason). Then it tells `erllama_server_convs` the outcome:
    - on `done` with `stop` or `length`, the assistant message and the `finish_key` are appended;
    - on cancel or error, nothing is appended and only the conversation's lock is released.

    It stays alive for `run_retention_ms`, so `result` and `attach` still work, then stops.
  - **Shutdown:** `terminate/2` cancels a ref that is still running.
- **`erllama_server_generate`** holds the pure helpers that build the token stream for a spec:
  - the `chat_apply` path, with a fallback to `render_chat_template` on `chat_not_supported`;
  - the raw `prompt` path;
  - `finish/2`, which parses the final text with `chat_parse/3` when a parser exists.

  Phase 3 HTTP reuses it.
- **`erllama_server_ring`** is a pure bounded event buffer:
  - `new(Max)` and `push(Event, Ring)`;
  - `from(Cursor, Ring)` returns `{ok, Events}` or `{gap, Oldest}`;
  - `next_seq/1`.
- **`erllama_server_runs`**: the ETS table `{RunId, Pid, ConvId, Model, StartedAt}`, with lookups done in the caller. The owner monitors the runs and deletes their rows on `DOWN`.
- **`erllama_server_convs`**: an ETS table of `#{id, model, messages, last_finish_key, active_run}`.
  - `begin_run/3` returns `{error, conversation_busy}` when a run is active.
  - `end_run/3` appends the outcome.
  - Phase 6 makes it durable behind the same calls.
- **`erllama_server_ids`**: the `run_<base32 of 16 random bytes>` and `conv_...` ids.
- **`erllama_server_client`** is the client side, used by the escript.
  - **It never calls an `erllama*` library module**, which a test checks.
  - **Connecting:**
    - `connect/0` starts the hidden node, applies the cookie and calls `hello/1` with `?API_VSN` through `erpc:call(Node, erllama_server_api, hello, [V], Timeout)`;
    - an `{exception, undef, _}` is mapped to "this release has no service API";
    - a nodedown is mapped to "release not reachable".
  - **`call/3`:** bounded erpc with a timeout, `control_timeout_ms` (default 10 s, overridden by `ERLLAMA_TIMEOUT`).
  - **`follow(RunId, Cursor, Sink)`:**
    - a local receive loop that acks every N events;
    - on nodedown or `dropped` it reconnects, with backoff, from `last_seq + 1`;
    - on `{gap, Oldest}` it reports the gap and continues from `Oldest`;
    - it returns on a terminal event.
- **`erllama_server_cli`** handles parsing and printing. Commands:
  - `ping`, `status`, `models`;
  - `load PATH [--id ID] [--n-ctx N] [--gpu-layers N]` and `unload ID`;
  - `run MODEL [--conversation ID] [--detach] PROMPT...`, a single turn streamed to stdout (`--detach` prints the run id and exits);
  - `chat MODEL`, a minimal line loop where each line is a turn in one conversation;
  - `runs`, `attach RUN [--from SEQ]`, `cancel RUN`, `result RUN`, `--version`.

  The exit codes are 0 on success, 1 for a release or run error, and 2 for usage.

### Configuration (`erllama_server` app env, validated in `erllama_server_sup:init/1`)

A bad value makes `start_link` fail.
- `run_buffer_events`, default 1024;
- `subscriber_max_unacked`, default 256;
- `run_retention_ms`, default 300000;
- `max_active_runs`, default 64; beyond it, `start_run` returns `{error, too_many_runs}`.

### Node, cookie and TLS

- **Release node:** `vm.args.src` keeps `-name erllama_server@127.0.0.1` and loopback distribution.
- **Client node:** a hidden `erllama_cli_<unique>@<release host>`. The cookie is `ERLLAMA_COOKIE` or `~/.erlang.cookie`, and `ERLLAMA_NODE` sets the target.
- **Remote administration over TLS:** `rel/erllama` reads `ERLLAMA_DIST_OPTFILE`. When it is set, it exports `ERL_FLAGS="-proto_dist inet_tls -ssl_dist_optfile $F"` before running the escript. The release enables TLS through its own `vm.args`, as documented.
- **The guide** `guides/server/rpc.md` covers:
  - the node names and cookies, and the caveat that `RELX_COOKIE` affects only control commands;
  - the TLS setup for both sides;
  - the API version rule;
  - what a gap means;
  - that workspaces always refer to the release machine.

## Tests

- **eunit:**
  - `erllama_server_ring_tests`;
  - `erllama_server_api_tests` for the config validation, the spec and option validation, and the version check;
  - `erllama_server_cli_tests` (extended) for argument parsing and output formatting.
- **PropEr:** `prop_erllama_server_ring`. For any push sequence and cursor, `from/2` returns either a contiguous run starting at the cursor or `{gap, Oldest}` with `Oldest > Cursor`.
- **CT `erllama_server_rpc_SUITE`:** one peer runs the release apps and loads the stub with `step_delay_ms`. A second peer, hidden, is the client and uses only `erllama_server_client`. Cases:
  - streams tokens in order, ending with done;
  - the result matches the stream;
  - a cursor reattach is contiguous after the subscriber is killed;
  - a gap with a small buffer;
  - a slow subscriber is dropped and the run completes;
  - cancel gives `cancelled`, and the conversation gets nothing appended;
  - the run survives the client node stopping, and its result is fetched afterwards;
  - `conversation_busy`;
  - the second turn carries the history and the `parent_key`;
  - `max_active_runs`;
  - unload during a run gives an error event;
  - stopping the app cancels in-flight streams;
  - a version mismatch returns `incompatible_api`;
  - an unreachable release;
  - the client peer has never loaded `erllama`, `erllama_nif` or any `erllama_cache*` module (`code:all_loaded/0`).
- **CT `erllama_server_tls_SUITE`:** the two peers use `inet_tls` with certificates generated by `public_key:pkix_test_data/1`; `hello` and a streamed run work. It skips if `ssl` is unavailable.
- **Gated real model** (`LLAMA_TEST_MODEL`): a TinyLlama chat run goes through the `chat_apply` path and streams; the parsed message is non-empty. The final report says which real-model checks ran and which were skipped.
- **CI release job:** extended with `bin/erllama_server eval` to load a stub model, then `bin/erllama run stub hello`, `run --detach` plus `attach` and `cancel`, then `stop`.

## Docs and progress

- `apps/erllama_server/README.md`: the CLI commands and the configuration keys.
- `guides/server/rpc.md`: new.
- `doc/features.md`: implemented, outstanding (the registry is interim, and the terminal has no Ctrl-C cancel: Ctrl-C exits the client and the run keeps going) and the validation results.
- `apps/erllama_server/PLAN.md`: kept in sync.

## Verification

- **Full gate** from the root on OTP 28.3.1: fmt, compile, eunit, proper, ct, lint, dialyzer, xref.
- **Real-model case** on TinyLlama (`~/Models/tinyllama-1.1b-chat.gguf`).
- **The release end to end:**
  - `rebar3 as prod release`, then `bin/erllama_server daemon`;
  - `bin/erllama load ~/Models/tinyllama-1.1b-chat.gguf --id tiny`;
  - `bin/erllama run tiny "hello"`, which streams;
  - `run --detach`, `attach`, `cancel` and `result`;
  - kill the client mid-run, reattach, and the run is complete;
  - `bin/erllama_server stop`.
- Stop for review before any commit or push.

## Later phases (unchanged outline)

3. Model registry and downloads (`livery_client`); the Livery HTTP adapters for the OpenAI, Anthropic and Ollama endpoint table, through `erllama_server_api` and `erllama_server_generate`.
4. The agent runtime and native tools.
5. The barrel_mcp bridge.
6. Durable conversations behind `erllama_server_convs`.
7. Terminal polish (Ctrl-C cancel, slash commands), the guides and full release validation.

---

# Phase 4: agent runtime and native tools (design)

## Shape

- **`erllama_server_agent`**, one gen_statem per agent session, under
  `erllama_server_agent_sup` (simple_one_for_one, temporary). A session
  is a conversation plus a workspace (a directory on the release
  machine), a model, a tool policy and budgets. It holds its
  conversation's lock for its whole life, so only it mutates that
  transcript. Sessions run concurrently.
- **The session is a job** (`kind => agent` in `erllama_server_runs`):
  its events go through `erllama_server_events`, so a CLI attaches,
  detaches and reattaches as with runs. It outlives its clients.
  - Turn events: `turn_started`, `token`, `thinking`, `tool_call`
    (announced), `approval_request`, `tool_started`, `tool_output`
    (shell progress), `tool_result`, `round`, `turn_done` with a
    reason (`stop`, `cancelled`, `budget`, `error`).
  - The session's own end is the terminal event.
- **The loop, as states:**
  - `idle`: wait for input;
  - `generating`: stream the model's reply through
    `erllama_server_generate`, which holds the model for the turn;
  - `awaiting_approval`: if any tool call needs approval;
  - `running_tools`: execute the calls;
  - then back to `generating` with the results appended.
- **Generation goes through a generator callback.** The default
  renders the transcript and streams it. Tests supply a scripted one,
  so the loop is tested deterministically; a gated real-model case
  checks it end to end.
- **Input and interruption:**
  - input that arrives during a turn is queued and becomes the next
    turn;
  - `interrupt` cancels the generation or the running tools, kills
    shell process groups, and ends the turn as `cancelled`.
- **Budgets per turn:** `max_rounds` (20), `max_tokens` (32768
  generated tokens) and `max_elapsed_ms` (600000).
- **Policy:** each tool is `auto`, `ask` or `deny`.
  - Defaults: reads, listings, search and `shell_status` are `auto`;
    `write_file`, `edit_file`, `shell` and `shell_kill` are `ask`.
  - `ask` emits `approval_request` and waits, even with no client
    attached. `approve(Id, Req, allow | deny | always)` answers it.
  - A denied call returns an error result to the model.
- **Execution:**
  - read-only calls run concurrently in monitored workers;
  - mutating calls run one after another in call order;
  - results are appended in call order;
  - a worker crash or timeout becomes an `Error: ...` tool result the
    model can act on.
- **Tool sources:** HTTP adapters never execute tools (runs only
  stream calls back to the client). Only sessions run tools, and only
  the native tools and the MCP tools the release is configured with
  (Phase 5).

## Native tools (`erllama_server_tools`, `_fs`, `_shell`)

- **Paths:** every path resolves inside the workspace; `..` and
  symlinks that leave it are refused.
- **`read_file(path, offset, limit)`:** numbered lines, at most 400
  per call and 64 KiB, with the offset to continue from.
- **`list_dir(path, depth)`:** at most 500 entries.
- **`search(pattern, path, glob)`:** regular expression, binary and
  large files skipped, at most 200 matches.
- **`write_file(path, content)`:** creates a file; refuses to replace
  one (use `edit_file`).
- **`edit_file(path, old, new, replace_all)`:** exact match, refused
  when the text is absent or ambiguous.
- **`shell(command, timeout_ms, background)`:**
  - runs `/bin/sh` in the workspace, as the leader of its own process
    group (the emulator starts port programs that way; `set -m` was
    dropped because Linux `/bin/sh` turns job control off without a
    terminal);
  - keeps the first and last 16 KiB of output;
  - on timeout or interrupt, sends TERM to the group, then KILL;
  - `background` returns a job id at once.
- **`shell_status(job)` and `shell_kill(job)`:** for background jobs,
  which end with the session.

## Interfaces

- **Service API:**
  - `agent_start(#{model, workspace, conversation_id, policy, budgets})`
    returns `{ok, #{agent_id, conversation_id}}`;
  - `agent_send/2`, `agent_interrupt/1`, `agent_approve/3`,
    `agent_stop/1`, `agents/0`.
- **CLI:** `erllama agent MODEL --workspace PATH [--yes | --read-only]`
  runs a line loop. It prints tool activity, asks for approvals, and
  exits without stopping the session, which `erllama agents` lists.

---

# Phase 5: barrel_mcp bridge (design)

## Shape

- **`erllama_server_mcp`**, a gen_server after `erllama_server_agent_sup`
  in the rest_for_one tree (its restart never stops a session). One
  `barrel_mcp_client` per `mcp_servers` entry, linked (a restart never
  leaves a second stdio process), monitored, reconnected with backoff
  (1 s doubling to 60 s). The tool list is fetched from a helper
  process, never inside the gen_server, and refreshed every 30 s; a
  client still in its handshake (`not_ready`) is asked again shortly.
- **Catalog entries** `#{name, server, remote, description, parameters,
  class, policy}`. `name` is `server__tool` with anything outside
  `[A-Za-z0-9_-]` replaced, cut to 64 bytes, suffixed on a collision.
  `class` is `read` when the tool's `readOnlyHint` is true, else
  `write`. No atom is made from a server's names.
- **Calls** run in the session's tool worker: the client is looked up
  (an unavailable server is an error result), the call has
  `mcp_timeout_ms`, exits are caught. `render/1` turns a result into
  text: text blocks, structured content as JSON, media described,
  embedded text resources shown, `isError` as an error result.

## Sessions

- `generate/1` takes a catalog snapshot (`#d.mcp`), offers the native
  tools plus the snapshot's tools not denied, and the round's calls
  are planned and dispatched against that snapshot.
- Policy for an MCP tool: a server set to `deny` wins; then the
  session's per-tool policy; then the session mode (`yes`: `auto`,
  `read_only`: deny `write` tools); then the server's policy.
- Dispatch: the call's context carries the catalog entry, and
  `erllama_server_tools:execute/3` sends it to the server.

## Endpoint

- `mcp_endpoint => true` mounts `livery_mcp:handler/1` at `/mcp` on the
  HTTP listener, behind the same middleware (API keys included), and
  registers `erllama_models` and `erllama_chat` in barrel_mcp's
  registry. A chat is a stateless run through `start_run/1`; a guard
  process cancels it if the tool handler is stopped.

## API, CLI, config

- `mcp/0` (status), `mcp_tools/0`, `mcp_resources/1`, `mcp_read/2`;
  `erllama mcp [tools | resources SERVER | read SERVER URI]`.
- `mcp_servers` (validated: names, transports, policies, unique names),
  `mcp_timeout_ms`, `mcp_endpoint`.

## Tests

- CT `erllama_server_mcp_SUITE` against real servers: Streamable HTTP
  through livery_mcp in the test node, stdio through an `erl`
  subprocess running `barrel_mcp:start_stdio/0`.
- eunit `erllama_server_mcp_tests`: names, rendering, configuration.


---

# Phase 6: durable conversations (design)

Two steps: 6a persistence, recovery and the conversation operations;
6b compaction.

## 6a. Storage

- One append-only log per conversation,
  `<data_dir>/conversations/<id>.jsonl`, apart from the KV cache
  directories. One JSON record per line, written with `sync`, before
  the change is visible in memory. A torn last line (a crash mid-write)
  is ignored on load.
- Records (`"t"`):
  - `create`: id, model, kind (`conversation` or `snapshot`, the frozen
    copy behind a stored response), parent (fork source), time;
  - `msgs`: messages appended, with the finish key when it changed;
  - `snapshot`: the whole transcript and key (a fork's start; later a
    compaction);
  - `run`: `begin` (run or agent id, kind, model, workspace for an
    agent), `end`, `release`;
  - `tool`: `intent` before a call runs (call id, name, arguments),
    `outcome` once it finished (status, text), before generation goes
    on;
  - `recovered`: what a restart repaired.
- Messages are encoded by `erllama_server_conv_codec`: known keys
  become atoms again on load, never others; image and audio bytes are
  base64; the finish key (a 32-byte binary) is hex.
- `erllama_server_convs` keeps the in-memory table and is the only
  writer of the logs. It loads every log at start.

## Recovery

A log whose last `run` is a `begin` was cut by a stop or a crash.
On load:
- the run is released;
- if the last assistant message has tool calls without results, each
  gets one, so the transcript stays well formed:
  - an outcome was recorded: that outcome, exactly;
  - only an intent: marked unresolved ("the release stopped while
    this call ran; its outcome is unknown and it was not run again");
  - nothing: "the release stopped before this call ran";
- unresolved calls are listed on the conversation (`unresolved`) and
  never repeated.

## KV

The finish key is persisted with the model it belongs to. A turn on
the same model passes it as `parent_key`: erllama restores the exact
KV when its cache still holds it and prefills the transcript otherwise
(a miss is a cold start). Nothing about conversations goes into the
cache.

## Operations

- API: `conversations/0` (snapshots hidden), `conversation/1`,
  `conversation_fork/1`, `conversation_delete/1` (refused while a run
  holds it), `conversation_export/2` (`json` or `markdown`).
- Resume: `run`, `chat` and `agent --conversation ID`, also after a
  restart; an agent session reuses the workspace its conversation was
  last worked in when none is given.
- Attach: `erllama conversation attach ID` follows the run holding it.
- CLI: `erllama conversations`, `erllama conversation show | fork |
  rm | export | attach ID`.
- Stored responses (`/v1/responses`, `store: true`) are written to
  `<data_dir>/responses/<id>.json` and their snapshot conversations
  are persisted like any other, so `previous_response_id` works after
  a restart.

## Tests

- eunit: the codec (every message shape round trips), log replay with
  a torn tail, recovery of each tool state.
- CT `erllama_server_convs_SUITE`: a restart keeps conversations,
  stored responses and their branches; fork, delete, export; an agent
  stopped by a restart in the middle of a write tool gets an
  unresolved result and the call is not run again; a resumed session
  reuses its workspace.
- Gated real model: a turn after a restart of erllama_server is warm
  (the persisted key restores the KV).

## 6b. Compaction (design)

- **When.** Before a conversation turn, or at the start of an agent
  session's turn, the transcript is measured (`count_tokens/3` with
  the model's template). Above `compact_threshold` (default 0.75) of
  the model's `context_size` it is compacted first. Also on request:
  `conversation_compact/1` for an idle conversation,
  `agent_compact/1` for an idle session, and
  `erllama conversation compact ID`.
- **What is kept.**
  - Every system message verbatim. The first one stays at the head.
    Any others, and a summary section, are appended to it, so strict
    templates that accept a single leading system message still render.
  - The summary section holds the first user request verbatim and a
    summary of the middle, written by the conversation's own model.
    It is asked to keep the goal and constraints, decisions, the state
    of the work (files changed, commands run and their outcome) and
    what remains: the durable task state.
  - The recent messages verbatim, from a user message on. At least
    `compact_keep_messages` (default 6) are kept, and the cut is always
    at a user message, so tool calls and their results stay in pairs.
  - Nothing to drop (the only user message is the first, or the cut
    would keep everything): no compaction.
- **How.** `erllama_server_compact` does the plan (pure), the
  summary (a non-streamed generation of at most
  `compact_summary_tokens`, default 512, over the middle rendered as
  text, each message clipped to 2 KiB) and the measure. The
  conversation records a `snapshot` of the compacted transcript with
  no finish key, and a `compacted` record. The next turn is a new
  exact prompt: no parent key, nothing semantic in the cache.
- **Where.** A run compacts in a helper once the model is held, so it
  keeps answering attach and cancel. An agent session goes through a
  `compacting` state at the start of a turn, whose helper holds the
  model. Input that arrives meanwhile is queued, and interrupt, stop
  and the time budget apply. A failed summary leaves the transcript
  as it was; the turn goes on.
- **Events.** `compacting`, then `compacted` with the message counts
  before and after.
- **Tests.**
  - eunit for the plan: system messages, the first request, cuts at
    user messages, tool pairs, nothing to drop.
  - CT on the stub with a small context: a run compacts automatically
    and goes on; the snapshot survives a restart; the finish key is
    dropped. A session compacts at the start of a turn and on request,
    and an interrupt during compaction ends it.
  - Gated: a real model writes a summary and the next turn answers.
