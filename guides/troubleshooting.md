# Troubleshooting

Recipes for the two questions that come up most: why a prompt ran cold
when you expected a cache hit, and why a request failed. You need this
page when a turn is slower than it should be or a caller gets an
error. Each recipe links to the guide that explains the mechanism.

## Why is my prompt not warm?

### 1. See what the last admission did

```erlang
{ok, #{kind := Kind, prefix_len := N}} = erllama:last_cache_hit(<<"chat">>).
%% Kind: exact | partial | sticky | continuation | cold
```

`prefix_len` is the warm prefix in tokens. The same facts are on every
result: `cache_hit_kind` and `cache_delta => #{read, created}`.
`last_cache_hit/1` returns `{ok, undefined}` before the first request.

### 2. Ask the cache about this prompt

```erlang
{ok, Tokens} = erllama:tokenize(<<"chat">>, Prompt),
{ok, Bytes} = erllama:cached_prefix_len(<<"chat">>, Tokens).
```

`{ok, 0}` means no saved row matches any prefix of the prompt. A
non-zero value with a cold admission points at the request (media, or
the row was evicted in between), not at the save.

### 3. Read the counters

```erlang
C = erllama:counters(),
maps:with([misses, hits_exact, hits_resume, hits_longest_prefix,
           hits_sticky_partial, saves_cold, saves_finish, saves_dropped,
           duplicate_dropped, evictions, restore_failed], C).
```

| You see | Likely cause | Fix |
|---|---|---|
| `saves_cold` and `saves_finish` stay at 0 | The prompts are shorter than `min_tokens` / `cold_min_tokens` (512 by default), so nothing is saved; `finish_key` is `undefined`. | Lower them in the model's `policy` map ([caching](caching.md#save-policy-gates)). |
| `saves_dropped` rises | The RAM tier budget is too small for one row, or every writer slot of a file tier was busy. | Raise `ram_tier_budget` or `writer_max_concurrent` ([configuration](configuration.md#ram_tier_budget)). |
| `evictions` rises | Rows are evicted before the next turn: a small budget, the cache eviction scheduler, or a `gc/0` call. | Size the tier; check the scheduler watermarks ([caching](caching.md#memory-pressure-driven-eviction)). |
| `misses` rise while saves happen | Each prompt differs from the saved ones before the save point: the template re-renders the history, or the head is shared but the save is near the end. | Track `session_id` or `parent_key` ([sessions](sessions.md)); turn on the checkpoint ladder ([caching](caching.md#checkpoint-ladder)). |
| `restore_failed` rises | A recurrent or hybrid model refused the exact-hit restore; the request ran cold. | Expected on those families ([loading](loading.md#model-families)). |

### Common misconfigurations

- **RAM saves off.** `ram_tier_budget => 0` turns RAM saves off, and a
  model on the default `tier => ram` then always runs cold.
- **A different fingerprint.** Every cache key includes the model
  fingerprint. Replacing the GGUF, switching `fingerprint_mode`, or
  attaching or rescaling a LoRA adapter changes it, so the old rows no
  longer match ([loading](loading.md#fingerprint)).
- **A different KV layout.** Changing `type_k`, `type_v`, `flash_attn`,
  `n_seq_max`, `kv_unified` or `ctx_params_hash` changes the key; every
  prefix runs cold once ([loading](loading.md#common-pitfalls)).
- **Media requests.** A request with `media` never reads or writes the
  cache ([multimodal](multimodal.md)).

## Why did a request fail?

### 1. Find the request and its model

```erlang
[#{ref := Ref, model := Model} | _] = erllama:requests(),
{ok, #{model := Model, pid := Pid}} = erllama:request_info(Ref),
{ok, Phase} = erllama:phase(Model),        %% idle | prefilling | generating
{ok, Queued} = erllama:pending_len(Model). %% calls waiting behind it
```

A request that seems stuck with `pending_len/1` above 0 is waiting for
a free sequence, not failing; see `n_seq_max` in
[configuration](configuration.md#n_seq_max). `request_info/1` answers
`{error, not_found}` once the request has finished.

### 2. Match the error

| Error | Meaning | What to do |
|---|---|---|
| `{error, decode_timeout}` | A decode step ran past `decode_budget_ms` (30 s). | Retry. See below. |
| `{error, decode_aborted}` | `erllama:cancel/1` interrupted a decode. | Nothing; the cancel worked. |
| `{error, seq_capacity}` | `on_full => error` and no sequence was free. | Raise `n_seq_max` or retry later ([generation](generation.md#notes)). |
| `{error, sticky_busy}` | Another request on the same `session_id` is in flight. | Serialise the turns of one session ([sessions](sessions.md)). |
| `{error, no_session}` | `continue/3` or `fork_session/3` on an unknown session. | Start the session with `stream/3` or `complete/3`. |
| `{error, {transcript_mismatch, _}}` | `expect_committed` did not match the session. | Re-sync your transcript ([sessions](sessions.md)). |
| `{error, {invalid_option, K, V}}`, `{error, {unknown_option, K}}` | The option map failed validation; nothing ran. | Fix the key ([configuration](configuration.md)). |
| `{error, {unsupported_with_media, K}}`, `{error, no_mmproj}` | A media request with an option it cannot take, or a model without a projector. | See [multimodal](multimodal.md). |
| `{error, not_loaded}` | The model id is not loaded (or its process stopped). | Load it again ([loading](loading.md)). |

### `decode_timeout` and recovery

When a decode step runs past `decode_budget_ms`, erllama aborts it and
recovers the model in place: every request in flight or queued on that
model fails with `{error, decode_timeout}`, the context is recreated,
and the model stays loaded. Sticky-session pins are dropped; saved
cache rows are kept, so the next request still warm-restores. You
retry the request. If the context cannot be recreated, the model
process stops and you load the model again.

### 3. Read the native logs

llama.cpp's own messages (load errors, out-of-memory, failed
assertions) go to `logger` under the domain `[erllama, native]`. The
level is read once at application start, so set it in `sys.config`:

```erlang
{erllama, [{native_log_level, info}]}   %% none | error | warning | info | debug
```

See [observability](observability.md#native-llama-cpp-logs) for a
`logger` handler that keeps only those lines.

## On the server

With the `erllama_server` release, start from the status page:

```bash
R=_build/prod/rel/erllama_server
$R/bin/erllama status              # models, runs, cache counters, recent failures
$R/bin/erllama inspect RUN_OR_ID   # one run or agent session in detail
$R/bin/erllama attach RUN --from 0 # replay a run's events from the start
```

The `failures` line lists the latest failures and what the release did
next. `attach --from 0` replays the run's events, including its
`error` event, from the start of its buffer. See
[operating the release](https://github.com/benoitc/erllama/blob/main/guides/server/operations.md) and
[the CLI](https://github.com/benoitc/erllama/blob/main/guides/server/rpc.md#detach-reattach-cancel).

## See also

- [Caching](caching.md): lookup order, save reasons, eviction
- [Observability](observability.md): counters, accessors, logs
- [Sessions](sessions.md): `session_id`, `parent_key`, `continue/3`
- [Configuration](configuration.md): every knob with its default
