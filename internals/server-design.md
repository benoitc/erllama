# Server design

This page records why erllama_server is shaped the way it is: the
decisions a change to the release must keep, or revisit on purpose.
You need it before you change the supervision tree, the service API,
how jobs reach the library, how tools run, or what the release writes
to disk. For how to use the release, see the
[server README](../apps/erllama_server/README.md) and the guides in
`guides/server/`.

## One facade: `erllama_server_api`

Every front end goes through one versioned module. The CLI and other
Erlang nodes call it over distribution; the HTTP adapters call it in
process for model access, runs, embeddings and token counts.

- A client calls `hello/1` first with the API version it speaks; any
  other version is refused with `{error, {incompatible_api, _}}`.
- Every function answers `ok`, `{ok, Value}` or `{error, Reason}`
  (plus `{queued, N}` from `agent_send/2` and `agent_prompt/4`). Called
  with the right argument types it does not raise, so a remote caller
  can tell a refusal from a crash.
- New behaviour gets an API function first; a front end never reaches
  past it into the job processes.

## Supervision: `rest_for_one`, in dependency order

```
erllama_server_sup              rest_for_one
├── erllama_server_failures     the failure log (every child records into it)
├── erllama_server_registry     the model registry
├── erllama_server_models       loaded models and their holders
├── erllama_server_runs         the job table and the run slots
├── erllama_server_convs        conversations and their locks
├── erllama_server_responses    stored responses
├── erllama_server_run_sup      runs (temporary)
├── erllama_server_pull_sup     downloads (temporary)
├── erllama_server_agent_sup    agent sessions (temporary)
├── erllama_server_mcp          MCP clients
└── erllama_server_http         the HTTP listener
```

- Each child depends only on those before it. When one restarts,
  everything after it restarts too, so no table keeps pointing at a
  stale pid or lock.
- The MCP clients come after the agent sessions: a restart of theirs
  never stops a session.
- The HTTP listener comes last: it takes requests only once everything
  it calls is up.
- Jobs are temporary: a crashed run, download or session is reported to
  its subscribers by its exit, not restarted.

## A run is its stream's receiver

`erllama_server_run` owns the `erllama:stream/3` ref and is the
process the stream sends to. erllama watches every stream's receiver
and cancels the request when it exits, however it exits (shutdown,
crash, kill), so a dead run never leaves its model generating. (The
first version trapped exits and cancelled in `terminate/2`, because
erllama did not watch the receiver then; a brutal kill leaked until the
token limit. The library fix made that unnecessary.)

## Admission by the work

- A generation is a run and takes one of the `max_active_runs` slots.
- An embedding is not a run, but it occupies the engine: it holds a
  slot from `erllama_server_runs` for as long as it runs
  (`erllama_server_inference`) and is refused with `too_many_runs`
  (HTTP 429) when they are full. The runs server monitors the holder,
  so a crashed holder frees its slot.
- Token counts and model lookups take no slot: they stay available
  when the release is full. A token count that needs its model loaded
  goes through the same load path as a run.

## The CLI never loads erllama

The `erllama` escript is built by `scripts/build_cli.escript` from
erllama_server's own modules, and `erllama_server_client` calls the
release only through `erpc`. The client node never loads `erllama`, its
NIF or a cache module: `erllama_server_cli_tests` checks it in a
fresh node, and `erllama_server_rpc_SUITE` (`leaked/1`) after every
case.

## Tools run only in agent sessions

- HTTP requests never execute tools: a run parses tool calls and
  streams them back to the client, which runs them.
- Only an agent session runs tools, and only the native tools and the
  tools of the MCP servers the release is configured with.
- A tool's class (`read` or `write`) lives with its definition in
  `erllama_server_tools`. Reads run together; a write waits for the
  calls before it and holds back the ones after. A name with no
  definition is treated as a write.
- The agent's generator (one generation round) is read from the
  application environment key `agent_generator`, a test hook for
  scripted models. It is not part of the remote API: `agent_start/1`
  keeps only `system` from the caller's spec.

## Shell commands own their process group

The emulator starts every port program as the leader of its own
process group, so the `/bin/sh` of a command leads a group holding
everything the command starts. A timeout, an interrupt or the end of
the session sends TERM to the group, then KILL after a grace period.
Commands do not rely on `set -m`: Linux `/bin/sh` turns job control
off when it has no terminal.

## MCP clients are linked

One `barrel_mcp_client` per configured server, linked to
`erllama_server_mcp`: when that process ends its clients end with it,
so a restart never leaves a second copy of a stdio server running. Tool
lists are fetched from a helper process, so the MCP process never waits
on a server; a server that is down is retried with backoff and never
blocks the release or a session.

## What is written, and when

- **Synced before visible.** Every conversation change is appended to
  its log (`<data_dir>/conversations/<id>.jsonl`) with `sync` before it
  is visible in memory. A torn last line is skipped on load.
- **Tool intent before the call, outcome after.** A session records a
  call's intent before it runs and its outcome before it generates
  again. After a stop or a crash, a call with an outcome gets that
  outcome; a call with only an intent is marked unresolved; a call with
  neither is marked not run. Unresolved calls are listed on the
  conversation and never run again.
- **Nothing about conversations goes into the KV cache.** The finish
  key of the last turn is persisted with its model and passed as
  `parent_key` on the next turn on that model: warm when the cache
  still holds it, a cold prefill of the transcript otherwise.
- **Compaction cuts only at a user message**, so a tool call and its
  result stay together. The compacted transcript is written as a
  snapshot with no finish key: the next turn is a new exact prompt.
