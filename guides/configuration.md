# Configuration reference

erllama configuration lives in two places: the OTP application
environment (`config/sys.config`) and the per-model option map
passed to `erllama:load_model/1,2`. This page is the full set.

## Application environment

Every key with its default; all of them are declared in
`src/erllama.app.src`.

```erlang
{erllama, [
  %% --------------- Cache tiers -----------------------------------
  %% Extra tiers started with the application (the RAM tier is always on).
  {tiers, [
    #{name => kv_disk, backend => disk,     root => "/var/lib/erllama/kvc"},
    #{name => kv_shm,  backend => ram_file, root => "/dev/shm/erllama"}
  ]},

  %% --------------- Save-policy gates -----------------------------
  {min_tokens,             512},
  {cold_min_tokens,        512},
  {cold_max_tokens,      30000},
  {continued_interval,    2048},
  {boundary_trim_tokens,    32},
  {boundary_align_tokens, 2048},

  %% --------------- Cache flow tunables ---------------------------
  {evict_save_timeout_ms,  30000},
  {session_resume_wait_ms,   500},
  {fingerprint_mode,         safe},   %% safe | gguf_chunked | fast_unsafe
  %% {writer_max_concurrent,   9},   %% default: 2 * schedulers + 1

  %% --------------- Chat -------------------------------------------
  {chat_cache_size, 64},              %% per-model template cache (LRU)
  %% {thinking_signing_key, <<"...">>},

  %% --------------- Observability ----------------------------------
  {middleware, []},                   %% see guides/middleware.md

  %% --------------- Memory-pressure scheduler ---------------------
  {scheduler, #{
    enabled         => false,
    pressure_source => noop,          %% noop | system | nvidia_smi | {module, M}
    interval_ms     => 5000,
    high_watermark  => 0.85,
    low_watermark   => 0.75,
    min_evict_bytes => 1048576,
    evict_tiers     => [ram, ram_file],
    unload_models_under_pressure => false,
    model_evictor   => undefined     %% module exporting evict_one/0
  }}
]}.
```

### `tiers`

Each entry is `#{name := atom(), backend := disk | ram_file, root :=
string()}`; the directory is created if missing and the tier is
registered under `name`, supervised by erllama. Models reference it
with `tier_srv => name` and `tier => backend`. The same can be done at
runtime with `erllama_cache:add_tier/1`; `erllama_cache:list_tiers/0`
shows what is running.

### Save-policy gates

See the [caching guide](caching.md#save-policy-gates) for what each
threshold does. All are overridable per-model via the `policy` map.

### `evict_save_timeout_ms`

How long synchronous `evict` and `shutdown` saves wait for the
writer to finish before giving up. Defaults to 30 s. Bump for
8B-class models on slow disks.

### `session_resume_wait_ms`

When a `parent_key` is supplied and the cache sees a matching
in-flight finish save, it waits up to this long for the save to
publish before falling through to a cold prefill. 500 ms is enough
for SSD-backed deployments; bump if you observe back-to-back
multi-turn cold misses on slow storage.

### `fingerprint_mode`

How to verify the model fingerprint at load:

- `safe` — full SHA-256 over the file. Slow on multi-GB GGUFs.
- `gguf_chunked` — fingerprint metadata + first weights tensor.
  Catches accidental corruption, not malicious tampering.
- `fast_unsafe` — trust the supplied fingerprint blindly. Use only
  when you fingerprint upstream and pass the digest through.

### `writer_max_concurrent`

Upper bound on cache saves written concurrently. Defaults to twice
the number of schedulers plus one.

### `chat_cache_size`

Number of per-model chat template handles kept in the LRU used by
`chat/3` and `chat_apply/3`. One entry per loaded model is enough;
the default 64 only matters with many models.

### `thinking_signing_key`

Binary key used to sign the signature attached to `{thinking_end,
Sig}` events so a client cannot forge a thinking block on a later
turn. Unset or `<<>>` disables signing.

### `middleware`

Global middleware chain applied to every API call; see the
[middleware guide](middleware.md).

### `scheduler`

See the [caching guide](caching.md#memory-pressure-driven-eviction).
`model_evictor` must name a loadable module exporting `evict_one/0`
(the `erllama_model_evictor` behaviour).

## Per-model options

Passed to `erllama:load_model/1,2`:

```erlang
#{
  backend           => erllama_model_llama,
  model_path        => "/path/to/x.gguf",
  model_opts        => #{n_gpu_layers => 99},
  context_opts      => #{
    n_ctx           => 4096,
    n_batch         => 512,
    n_seq_max       => 1,        %% > 1 enables the multi-tenant scheduler
    flash_attn      => auto,     %% boolean() | auto
    type_k          => f16,      %% KV element type for keys
    type_v          => f16,      %% KV element type for values
    decode_budget_ms => 30000    %% per-step decode budget; 0 disables
  },
  fingerprint       => <<32 bytes>>,
  fingerprint_mode  => safe,
  quant_type        => q4_k_m,
  quant_bits        => 4,
  ctx_params_hash   => <<32 bytes>>,
  context_size      => 4096,
  tier_srv          => my_disk,
  tier              => disk,
  policy            => #{
    min_tokens             => 256,
    cold_min_tokens        => 256,
    cold_max_tokens        => 8192,
    continued_interval     => 256,
    boundary_trim_tokens   => 32,
    boundary_align_tokens  => 256,
    session_resume_wait_ms => 500,
    prefill_chunk_size     => 1024   %% per-tick prefill slice cap
  }
}
```

### `prefill_chunk_size`

Per-tick prefill slice cap. Default `max(64, n_batch div 4)`; pass
`infinity` to disable. Lives in the `policy` map for parity with
the save-policy gates but does not gate saving - it caps how many
tokens a single prefill row contributes to one `step_tick`, so a
long prompt is sliced across multiple ticks and concurrent decoders
keep making progress between chunks.

### `n_seq_max`

Maximum concurrent sequences. Belongs under `context_opts`, not the
`policy` map - it is a context shape, not a save-policy gate.
Default `1` (single-tenant). Set higher to opt into multi-tenant
co-batched scheduling.

When `n_seq_max` is too low for the concurrent-session count, an
admission with no free seq blocks by default. Pass
`on_full => error` on `complete/3` / `stream/3` to fail fast with
`{error, seq_capacity}` instead, and size capacity up front with
`available_seqs` / `n_seq_max` from `model_info/1`.

### `decode_budget_ms`

Per-step wall-clock budget for a single `llama_decode`. Default
`30000`; `0` disables. A decode that exceeds it is aborted via the
context's ggml abort callback and returns `{error, decode_timeout}`,
so a wedged decode can never block the model process indefinitely.
The engine recovers in place (recreates the context, model stays
loaded) rather than cold-reloading. Recovery drops only the live
in-context KV cells and sticky-session pins - the tiered cache
(RAM/disk rows) persists, so a subsequent request still reuses
cached prefixes through the normal warm-restore lookup rather than
starting fully cold. The same callback honours `erllama:cancel/1`,
which interrupts an in-flight decode and surfaces
`{error, decode_aborted}`. A legitimate step is sub-second on a
loaded model, so the default only trips on a genuine wedge; lower it
if you want a tighter bound.

See [loading a model](loading.md) for the per-field walkthrough.

## Inspecting effective config

```erlang
1> application:get_env(erllama, scheduler).
{ok, #{enabled => true, ...}}

2> erllama_scheduler:status().
#{enabled => true, pressure_source => system, ...}

3> erllama_cache:info().
#{tiers => [#{name => erllama_cache_ram, backend => ram, ...}, ...],
  entries => 12,
  bytes => #{ram => 12582912, ram_file => 0, disk => 805306368}}

4> erllama:model_info(<<"chat">>).
{ok, #{id => <<"chat">>, status => idle, context_size => 8192,
       quant_type => q4_k_m, tier => disk, n_seq_max => 4, ...}}
```
