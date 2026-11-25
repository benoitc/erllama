# How erllama fits together

This page gives you the mental model of the repository: the words the
code uses, the two supervision trees, the path a request takes from an
HTTP client to a token, and which module holds what. You need it before
you change the code for the first time, or when a design record
assumes a term you do not know. How to make and check a change is in
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Two apps

```
apps/erllama          the library (Hex package erllama): models, requests,
                      the KV cache, one NIF over llama.cpp
apps/erllama_server   the release: runs, conversations, agent sessions,
                      tools, MCP, the HTTP APIs and the `erllama` CLI
```

The library knows models, requests and the cache; it has no notion of
users or conversations. The server is a client of the library like any
OTP application.

## Glossary

| Term | Meaning | Where |
|---|---|---|
| model process | one `gen_statem` per loaded model; it admits requests, batches them into one decode per tick and owns the llama.cpp context | `erllama_model` |
| backend | the module a model process calls for native work: llama.cpp, the stub for tests, or a split model | `erllama_model_backend` and its implementations |
| seq, sequence | one llama.cpp sequence id in the model's context; a model serves as many requests at once as it has sequences (`n_seq_max`) | `erllama_model` |
| sticky session | a `session_id` that keeps its sequence's KV cells in the context between turns, so the next turn prefills only its new tokens | `erllama_model`, [sessions guide](../guides/sessions.md) |
| cache row | one saved KV checkpoint, keyed by SHA-256 over the model fingerprint, context options and the rendered prompt bytes | `erllama_cache_meta_srv`, `erllama_cache_key` |
| tier | where a row's payload lives: RAM (ETS), `ram_file` or `disk` | `erllama_cache_ram`, `erllama_cache_disk_srv` |
| reservation | the meta server's grant to one writer to publish a row; checked again just before the file is linked | `erllama_cache_meta_srv`, [publish protocol](publish-protocol.md) |
| rank | one GPU of a tensor-parallel group, as a process on its node | `erllama_tp_rank`, `erllama_tp` |
| follower | a rank other than 0: it applies rank 0's command stream in order | `erllama_tp_rank` |
| run | one generation in the server: a process that owns an erllama stream and numbers its events | `erllama_server_run` |
| job | anything a client can attach to and follow by id: a run, a download or an agent session | `erllama_server_runs`, `erllama_server_job` |
| agent session | one `gen_statem` per conversation that generates, asks for approvals and runs tools | `erllama_server_agent` |

## The library's supervision tree

```
erllama_sup                         one_for_one
├── erllama_cache_sup               one_for_one
│   ├── erllama_cache_meta_srv      the only writer of row metadata and reservations
│   ├── erllama_cache_ram           the RAM tier
│   ├── erllama_cache_writer        save orchestration
│   ├── erllama_cache_tier_sup      one erllama_cache_disk_srv per disk or ram_file root
│   └── erllama_cache_tiers_boot    starts the tiers from the `tiers` env (transient)
├── erllama_registry                model id registry
├── erllama_inflight                streaming refs, model state, abort handles
├── erllama_chat_cache              per-model chat template handles
├── erllama_log                     llama.cpp logs into logger (before any model load)
├── erllama_model_sup               simple_one_for_one, temporary
│   └── erllama_model               one per loaded model
└── erllama_scheduler               cache eviction under memory pressure
```

- A model is `temporary`: a model that fails for good is not restarted
  and needs a fresh load. A decode that times out or fails is recovered
  in place and the model stays loaded.
- Tensor-parallel rank processes (`erllama_tp_rank`) are started with
  `gen_server:start` and are deliberately not supervised or linked: a
  rank lives until `stop/1` or until its node goes, and whoever formed
  the group decides what to do when one is lost.

## The server's supervision tree

```
erllama_server_sup                  rest_for_one
├── erllama_server_failures
├── erllama_server_registry
├── erllama_server_models
├── erllama_server_runs
├── erllama_server_convs
├── erllama_server_responses
├── erllama_server_run_sup          runs (temporary)
├── erllama_server_pull_sup         downloads (temporary)
├── erllama_server_agent_sup        agent sessions (temporary)
├── erllama_server_mcp
└── erllama_server_http
```

When a child restarts, every child after it restarts too, so nothing
points at a stale process. Why this order is in the
[server design record](https://github.com/benoitc/erllama/blob/main/internals/server-design.md).

## From an HTTP request to a token

```
client
  │  POST /v1/chat/completions
  ▼
erllama_server_http             route and middleware (keys, limits)
  ▼
erllama_server_openai           the dialect handler: validate, translate
  ▼
erllama_server_http_kit         start_run/2, then drive/3 follows the events
  ▼
erllama_server_api              start_run/1: the service API
  ▼
erllama_server_runs             max_active_runs check, start under run_sup
  ▼
erllama_server_run              owns the stream, numbers and buffers events
  ▼
erllama_server_generate         render the chat template, tokenise
  ▼
erllama:stream/3                the library's public API
  ▼
erllama_model                   admit into a sequence; each tick batches
                                every active request into one step
  ▼
erllama_model_llama:step/2  ->  erllama_nif:step/2   (llama.cpp decode, sample)
```

Each token goes back the same way: `{erllama, Ref, {token, Bin}}` to
the run, a numbered `token` event to the handler, a chunk in the
protocol's shape to the client. The Anthropic and Ollama handlers
follow the same path; the CLI starts its runs through the same
`erllama_server_api` over Erlang distribution.

## Module map

Library (`apps/erllama/src`):

| Area | Modules |
|---|---|
| public API | `erllama`, `erllama_cache`, `erllama_middleware` (with the hidden helpers `erllama_chat` and `erllama_stream`) |
| options | `erllama_opts` |
| model process and backends | `erllama_model`, `erllama_model_sup`, `erllama_model_backend`, `erllama_model_llama`, `erllama_model_stub`, `erllama_model_tp` |
| registry and state | `erllama_registry`, `erllama_inflight`, `erllama_chat_cache`, `erllama_log` |
| cache | `erllama_cache_*`, `erllama_fingerprint` |
| memory pressure | `erllama_scheduler`, `erllama_pressure`, `erllama_pressure_system`, `erllama_pressure_nvidia_smi`, `erllama_model_evictor` |
| tensor parallelism | `erllama_tp`, `erllama_tp_rank`, `erllama_tp_checkpoint` |
| NIF | `erllama_nif` and `c_src/` (`erllama_nif_*.c` by area, `erllama_safe.cpp`, the C++ shims for chat, media, ngram drafting, MTP and tensor parallelism) |

Server (`apps/erllama_server/src`): the table by area is in the
[server README](../apps/erllama_server/README.md#where-the-code-is).

## Where to read next

Design records, for a change to one subsystem:

- [Cache design](cache-design.md), [publish protocol](publish-protocol.md),
  [prefix checkpoint ladder](prefix-checkpoint-ladder.md)
- [Request lifecycle](request-lifecycle.md): admission, the tick, finish
- [NIF safety](nif-safety.md) and the [C safety audit](c-safety-audit.md)
- [Tensor parallelism](tensor-parallel-design.md)
- [Server design](https://github.com/benoitc/erllama/blob/main/internals/server-design.md)
- [Hardware test suites](hardware-suites.md): the opt-in suites and what they need

Guides, for using what is there: start with
[loading a model](../guides/loading.md) and
[generating text](../guides/generation.md) for the library, and the
[server walkthrough](https://github.com/benoitc/erllama/blob/main/guides/server/demo.md) for the release.
