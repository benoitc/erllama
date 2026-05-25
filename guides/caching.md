# Caching guide

Barrel Inference's KV cache turns a multi-second prefill into a millisecond
restore. This guide is the operator's-eye view: what it does, when
it kicks in, and which knobs to touch.

## The mental model

A transformer's "KV state" is the per-layer key/value tensors
produced while reading the prompt. Once you have them, generating
the next token costs one forward pass. Without them, you have to
re-read every token of the prompt from scratch.

Barrel Inference's cache stores those tensors keyed on the **rendered
prompt bytes that produced them** (ds4-style, content-addressed):

```
key = sha256(model_fingerprint || quant || ctx_params || rendered_prompt_bytes)
```

where `rendered_prompt_bytes = detokenize(tokens)`. Keying on the
rendered bytes rather than the token-id list means the *same* logical
prompt still hits when it retokenises across turns (chat-template
wrapping, tool rendering). Same bytes → same key → guaranteed-correct
restore. There is no fuzzy matching layer; "close enough" is not
allowed at this level (the longest-byte-prefix lookup is still an
exact SHA-256 match). The exact tokens travel in the checkpoint
payload for KV resume.

## Three tiers

```
ram       ETS slabs in BEAM heap. Lowest latency, smallest budget.
ram_file  Files on /dev/shm. Fast, capped only by tmpfs size.
disk      Files on persistent storage. Survives restarts.
```

Each tier is an independently-supervised gen_server with its own
byte quota and its own LRU. A save is written to the tier you
configure on the model; reads consult an in-memory index that fans
out to the right tier.

The disk tier is **a first-class citizen**: large models that
wouldn't fit alongside a working set of warm KV state in RAM can
let the disk tier hold most of the cache, and warm-restore in
milliseconds when a hit comes in.

## When does a save happen?

The per-model `gen_statem` fires saves at five well-defined moments,
each with its own `save_reason`:

| Reason | When | Sync? |
|---|---|---|
| `cold` | Right after a cold prefill, at the trimmed-prefix boundary. Async — the writer pool does the work. |
| `continued` | Every `continued_interval` tokens during generation. Async. |
| `finish` | At the end of a completion, capturing prompt + reply. Async. |
| `evict` | When a holder is asked to release its slab. Sync (pause decode, pack, release). |
| `shutdown` | On `prep_stop` or `unload/1`. Sync, capped by `evict_save_timeout_ms`. |

Async saves go through `barrel_inference_cache_writer` — a pool of dirty-IO
workers. Sync saves block the calling process until the file is on
stable storage.

## When does a hit happen?

Three lookup paths, in order of preference:

1. **Exact key.** Caller passes the exact `parent_key` from the
   previous turn. Cheapest. Used by Erlang-native multi-turn flows.
2. **Resume.** Caller passes a `parent_key` from an earlier turn,
   and the new prompt strictly extends the cached prefix.
3. **Longest-prefix walk.** No `parent_key` supplied. The cache
   walks the new prompt's tokens backward by the configured stride
   (`boundary_align_tokens`) and probes the index for each
   alignment. The longest cached prefix wins.

For stateless callers — OpenAI/Anthropic-shaped HTTP APIs that
resend the full conversation each turn — option 3 is what you want.
You don't have to do anything; just call `barrel_inference:complete/2`.

## Save policy gates

Saving every prefix would flood the writer pool. Barrel Inference gates saves
behind a few thresholds, all overridable per-model.

| Gate | Default | What it does |
|---|---|---|
| `min_tokens` | 512 | Skip saves shorter than this. Prefills under 512 tokens are usually cheaper than the round-trip to disk. |
| `cold_min_tokens` | 512 | Don't fire a `cold` save for shorter prefills. |
| `cold_max_tokens` | 30 000 | Cap on cold-save size. Protects against pathological prompts. |
| `continued_interval` | 2048 | Fire a `continued` save every N generated tokens. |
| `boundary_trim_tokens` | 32 | Drop the last N tokens before saving. Mid-token, mid-sentence boundaries make poor resume points; trim to a safe alignment. |
| `boundary_align_tokens` | 2048 | Round trim down to a multiple of this. Sets the longest-prefix walk's stride. |
| `session_resume_wait_ms` | 500 | When a `parent_key` is supplied and the cache sees an in-flight finish save, wait up to this long for it to publish before falling through to a fresh prefill. |
| `prefill_chunk_size` | `max(64, n_batch div 4)` | Per-tick cap on how many tokens a single prefill row contributes. Not a save gate - lives in the same `policy` map but caps the scheduler's per-tick slice so a long prompt never monopolises the batch. `infinity` to disable. |

Bigger `boundary_align_tokens` = fewer probes per longest-prefix
walk but coarser hit alignment. 2048 is the default; 256 makes
hits more likely on shorter prompts at the cost of more probes.

## Memory-pressure-driven eviction

`barrel_inference_scheduler` is a polling gen_server that watches a pluggable
pressure source and evicts cache slabs when pressure crosses a
watermark. Off by default. Enable in `sys.config`:

```erlang
{barrel_inference, [
  {scheduler, #{
    enabled         => true,
    pressure_source => system,        %% portable, memsup-backed
    interval_ms     => 5000,
    high_watermark  => 0.85,
    low_watermark   => 0.75,
    evict_tiers     => [ram, ram_file] %% disk fills to its own quota
  }}
]}.
```

Sources shipped:

- `noop` — always reports zero pressure.
- `system` — OTP `memsup`. Linux, macOS, BSD, Windows.
- `nvidia_smi` — sums VRAM across all visible NVIDIA GPUs.
- `{module, M}` — calls `M:sample/0`. Implement
  `-behaviour(barrel_inference_pressure)` to write your own.

When the source reports `Used / Total >= high_watermark`, the
scheduler asks the cache to evict enough bytes to drop the ratio
below `low_watermark`, scoped to the configured tiers.

## Inspecting the cache

```erlang
%% Hit/miss/save counters and per-path latency totals.
barrel_inference_cache:get_counters().

%% Every row in the index, raw tuples:
%%   {Key, Tier, Size, LastUsedNs, Refcount, Status, HeaderBin,
%%    Location, TokensRef, Hits}
barrel_inference_cache_meta_srv:dump().

%% Synchronous full eviction pass: returns {evicted, N}.
barrel_inference_cache:gc().

%% Free at least N bytes, oldest LRU first: returns {evicted, N, BytesFreed}.
barrel_inference_cache:evict_bytes(256 * 1024 * 1024).
barrel_inference_cache:evict_bytes(256 * 1024 * 1024, [ram, ram_file]).
```

The counter map is documented inline on
`barrel_inference_cache:get_counters/0` — call it from a shell to see the
keys for your build.

## Disabling the cache

For benchmarking or sanity checks: load the model with `tier => ram`
and a tiny `min_tokens` to bypass saves entirely, or set the
application env to disable all saves at the policy level:

```erlang
{barrel_inference, [
  {min_tokens, infinity}       %% nothing ever clears the gate
]}.
```

There is no global "off switch" — disabling was an explicit
non-goal. The cache is the product.

## See also

- [Loading a model](loading.md) — option-by-option walkthrough.
- [Configuration reference](configuration.md) — every knob,
  with defaults.
- Internals: [cache design](../internals/cache-design.md) and
  [publish protocol](../internals/publish-protocol.md) for the
  reasons behind the choices.
