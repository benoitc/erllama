# Caching guide

erllama's KV cache turns a multi-second prefill into a millisecond
restore. This guide is the operator's-eye view: what it does, when
it kicks in, and which knobs to touch.

## The mental model

A transformer's "KV state" is the per-layer key/value tensors
produced while reading the prompt. Once you have them, generating
the next token costs one forward pass. Without them, you have to
re-read every token of the prompt from scratch.

erllama's cache stores those tensors keyed on the **rendered
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

The RAM tier is always on and is where models save by default
(`tier => ram`). Its size is bounded by `ram_tier_budget` (25% of
system memory by default; see the
[configuration reference](configuration.md#ram_tier_budget)); when a
new save does not fit, the least used RAM rows are evicted first.
Add `ram_file` and `disk` tiers with
`erllama_cache:add_tier/1` or the `tiers` application environment key;
each one is a supervised server rooted at a directory. A save is
written to the tier the model is configured with (`tier` + `tier_srv`
in its load config); reads consult an in-memory index that fans out to
the right tier.

```erlang
ok = erllama_cache:add_tier(#{name => kv_disk, backend => disk,
                              root => "/var/lib/erllama/kvc"}),
ok = erllama_cache:add_tier(#{name => kv_shm, backend => ram_file,
                              root => "/dev/shm/erllama"}),
erllama_cache:list_tiers().
%% [#{name => erllama_cache_ram, backend => ram, ...},
%%  #{name => kv_disk, backend => disk, root => "/var/lib/erllama/kvc", pid => <0.231.0>},
%%  #{name => kv_shm, backend => ram_file, ...}]
```

The disk tier is **a first-class citizen**: large models that
wouldn't fit alongside a working set of warm KV state in RAM can
let the disk tier hold most of the cache, and warm-restore in
milliseconds when a hit comes in.

## When does a save happen?

The model process saves at these moments, each with its own
`save_reason`:

| Reason | When |
|---|---|
| `cold` | Right after a cold prefill, at the trimmed-prefix boundary. |
| `continued` | Every `continued_interval` tokens during generation. |
| `finish` | At the end of a completion, capturing prompt + reply. |
| `agent_prefix` | When a request sets `prefix_checkpoint_len`, at that boundary. The row is pinned so eviction keeps it. |
| `evict` | When you call `erllama:evict/1`. |
| `shutdown` | When you call `erllama:shutdown/1` or `unload/1`, and when the erllama application stops. |

`evict` and `shutdown` save the live context of every in-flight
request and are capped by `evict_save_timeout_ms`; the others follow
the save policy gates below.

A save runs in the model process: it packs the KV state and writes it
to the model's tier before the model moves on. RAM saves are a memory
copy. File saves (`ram_file`, `disk`) also take a slot in the writer
pool (`writer_max_concurrent`); when every slot is busy, the save is
dropped and counted in `saves_dropped` rather than waited for.

## When does a hit happen?

Every request tries three lookups, in this order:

1. **Exact key.** The key of the whole rendered prompt. Hits when the
   same prompt was saved before.
2. **Resume.** Only when you pass `parent_key` (the `finish_key` of
   an earlier turn) and the new prompt extends that row's text. If
   the earlier turn's save is still being written, erllama waits up
   to `session_resume_wait_ms` for it.
3. **Longest saved prefix.** erllama checks the saved rows of this
   model from the longest prefix of the prompt down and resumes from
   the longest one it finds. Each check is an exact key match on that
   prefix.

Stateless callers (OpenAI/Anthropic-shaped HTTP APIs that resend the
full conversation each turn) are served by lookup 3 without doing
anything; `parent_key` only saves the prefix scan when you already
track turns.

A warm restore can still fail mid-flight: recurrent / hybrid models
(Mamba, RWKV, Jamba, ...) refuse the 1-token removal the exact-hit
primer needs unless the arch supports recurrent-state rollback, and
a stored payload can fail to unpack. Either way the engine drops the
sequence, bumps the `restore_failed` counter and admits the request
cold, so the reply stays correct; the hit is just not taken. Partial
hits (an extended prompt, the common agent case) need no primer and
stay warm on every family. See the model-families section of the
loading guide.

## Save policy gates

Saving every prefix would flood the writer pool. erllama gates saves
behind a few thresholds, all overridable per-model.

| Gate | Default | What it does |
|---|---|---|
| `min_tokens` | 512 | Skip saves shorter than this. Prefills under 512 tokens are usually cheaper than the round-trip to disk. |
| `cold_min_tokens` | 512 | Don't fire a `cold` save for shorter prefills. |
| `cold_max_tokens` | 30 000 | Cap on cold-save size. Protects against pathological prompts. |
| `continued_interval` | 2048 | Fire a `continued` save every N generated tokens. |
| `boundary_trim_tokens` | 32 | Drop the last N tokens before saving. Mid-token, mid-sentence boundaries make poor resume points; trim to a safe alignment. |
| `boundary_align_tokens` | 2048 | Round the cold-save point down to a multiple of this, so cold saves of similar prompts land on the same boundary. |
| `session_resume_wait_ms` | 500 | When a `parent_key` is supplied and the cache sees an in-flight finish save, wait up to this long for it to publish before falling through to a fresh prefill. |
| `prefill_chunk_size` | `max(64, n_batch div 4)` | Per-tick cap on how many tokens a single prefill row contributes. Not a save gate - lives in the same `policy` map but caps the scheduler's per-tick slice so a long prompt never monopolises the batch. `infinity` to disable. |

Bigger `boundary_align_tokens` = fewer probes per longest-prefix
walk but coarser hit alignment. 2048 is the default; 256 makes
hits more likely on shorter prompts at the cost of more probes.

## Memory-pressure-driven eviction

`erllama_scheduler` is a polling gen_server that watches a pluggable
pressure source and evicts cache slabs when pressure crosses a
watermark. Off by default. Enable in `sys.config`:

```erlang
{erllama, [
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

- `noop` - always reports zero pressure.
- `system` - OTP `memsup`. Linux, macOS, BSD, Windows.
- `nvidia_smi` - sums VRAM across all visible NVIDIA GPUs.
- `{module, M}` - calls `M:sample/0`. Implement
  `-behaviour(erllama_pressure)` to write your own.

When the source reports `Used / Total >= high_watermark`, the
scheduler asks the cache to evict enough bytes to drop the ratio
below `low_watermark`, scoped to the configured tiers.

## Inspecting the cache

```erlang
%% Hit/miss/save counters and per-path latency totals.
erllama_cache:get_counters().

%% Tiers, number of cached rows and bytes held per tier backend.
erllama_cache:info().
%% #{tiers => [...], entries => 42,
%%   bytes => #{ram => 12582912, ram_file => 0, disk => 805306368}}

%% Synchronous full eviction pass: returns {evicted, N}.
erllama_cache:gc().

%% Free at least N bytes, least used rows first: returns {evicted, N, BytesFreed}.
erllama_cache:evict_bytes(256 * 1024 * 1024).
erllama_cache:evict_bytes(256 * 1024 * 1024, [ram, ram_file]).
```

The counter keys are listed on `erllama_cache:get_counters/0`.

## TinyLlama on erllama and Ollama

We sent the same 44-token prompt to erllama and Ollama using TinyLlama
1.1B Chat Q4_K_M. Every reply contained 128 tokens. The figures are
medians of six requests per server on each machine, alternating which
server ran first. Client tokens/s covers the full request, from sending
the prompt on loopback to reading the final stream event.

| Machine | Server | Client tokens/s | First visible text | Full reply | Server-reported decode tokens/s |
|---|---|---:|---:|---:|---:|
| Apple M4 Pro | erllama | 223.3 | 30.3 ms | 573 ms | 225.0 |
| Apple M4 Pro | Ollama | 214.0 | 29.0 ms | 598 ms | 224.8 |
| NVIDIA GB10 | erllama | 259.7 | 12.3 ms | 493 ms | 262.6 |
| NVIDIA GB10 | Ollama | 240.2 | 62.9 ms | 533 ms | 273.3 |

On the **M4 Pro**, each timed request followed an unrelated prompt with
the model already loaded. Ollama reused one token on every run. The
first visible text arrived about 1.3 ms earlier with Ollama, and the
full reply finished about 25 ms earlier with erllama.

On the **GB10**, both servers were warmed once and then received the
measured prompt six times. Ollama reported faster decode, while erllama
delivered the first text about 51 ms earlier and finished about 40 ms
earlier. Compare the servers within each machine because the prompt
history and Ollama versions differ between machines.

### Resending the same prompt with both caches warm

On the M4 Pro, we enabled short-prompt checkpoints in erllama's RAM
tier and sent the 44-token prompt once to populate both caches. On all
six retries, both servers reused 43 tokens. Median time to first visible
text was **6.9 ms for erllama** and **8.5 ms for Ollama**.

The default cold-save minimum is 512 tokens. For this short-prompt test
we set `min_tokens` and `cold_min_tokens` to 32, and `boundary_trim_tokens`
and `boundary_align_tokens` to 1. That saves the first 43 tokens and leaves
one token to process on the retry.

[Exact prompt, build versions and raw measurements](https://github.com/benoitc/erllama/tree/8e54df7a45ac9cd217698162e128b097fa9bab84/bench/reports/2026-11-25-tinyllama-ollama)

## Disabling the cache

For benchmarking or sanity checks, turn RAM saves off and keep the
model on the `ram` tier:

```erlang
{erllama, [
  {ram_tier_budget, 0}       %% no RAM saves; ram-tier requests run cold
]}.
```

Models on a `disk` or `ram_file` tier still save; leave those tiers
out of the load config to keep the whole node cold.

## See also

- [Loading a model](loading.md) - option-by-option walkthrough.
- [Configuration reference](configuration.md) - every knob,
  with defaults.
- Internals: [cache design](../internals/cache-design.md) and
  [publish protocol](../internals/publish-protocol.md) for the
  reasons behind the choices.
