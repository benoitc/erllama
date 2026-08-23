# Loading a model

erllama serves one or more loaded models concurrently. Each loaded
model is a supervised `gen_statem` that owns a single
`llama_context*`, sits behind a registered name, and shares the
process-wide KV cache with every other model.

This guide walks through the load call and every option that
matters in practice.

## The minimal call

```erlang
1> {ok, _}  = application:ensure_all_started(erllama).
2> {ok, Bin} = file:read_file("/srv/models/tinyllama-1.1b-chat.Q4_K_M.gguf").
3> {ok, M} = erllama:load_model(#{
       model_path  => "/srv/models/tinyllama-1.1b-chat.Q4_K_M.gguf",
       fingerprint => crypto:hash(sha256, Bin)
   }).
{ok, <<"erllama_model_2375">>}
```

That is enough to run a completion. erllama fills in the cache
parameters from the application defaults; with no `tier`/`tier_srv`
override the model writes to the RAM tier (the only one started by
default).

`M` is a binary model id. Use it for every subsequent call:
`erllama:complete(M, ...)`, `erllama:unload(M)`, etc. Pick the id
yourself with `model_id => <<"chat">>` in the map or with
`load_model/2`.

The config is validated before the model process starts:

```erlang
4> erllama:load_model(#{}).
{error, {missing_config, model_path}}
5> erllama:load_model(#{model_path => "/no/such.gguf"}).
{error, {invalid_config, model_path, "/no/such.gguf"}}
6> erllama:load_model(#{model_path => Path, context_opts => #{n_ctxx => 1}}).
{error, {unknown_option, {context_opts, n_ctxx}}}
```

## The full option map

```erlang
#{
  model_id          => <<"chat">>,
  backend           => erllama_model_llama,
  model_path        => "/srv/models/llama-3.1-8b-instruct.Q4_K_M.gguf",
  model_opts        => #{n_gpu_layers => 99, use_mmap => true},
  context_opts      => #{n_ctx => 8192, n_batch => 4096, n_threads => 8},
  fingerprint       => Fp,
  fingerprint_mode  => safe,
  quant_type        => q4_k_m,
  quant_bits        => 4,
  ctx_params_hash   => HashOfCtxParams,
  context_size      => 8192,
  tier_srv          => my_disk,
  tier              => disk,
  policy            => #{ ... },
  thinking_markers  => #{start => <<"<think>">>, 'end' => <<"</think>">>}
}
```

### `backend`

Module implementing the `erllama_model_backend` behaviour. Two
shipped today:

- `erllama_model_llama` (default): the llama.cpp backend.
- `erllama_model_stub`: a deterministic backend with no NIF and no
  GGUF, for testing your own code against the API. It needs no
  `model_path`.

### `model_path`

Absolute path to a GGUF file. The model layer hands it to llama.cpp
verbatim; relative paths work too but are resolved against the BEAM's
current working directory, which is rarely what you want under a
release.

### `model_opts`

Pass-through to `llama_model_default_params()`. The fields that
matter day-to-day:

| Key | Default | Notes |
|---|---|---|
| `n_gpu_layers` | 0 | Number of transformer layers offloaded to GPU. Set high enough to cover the model on Metal/CUDA boxes. 99 effectively means "all". |
| `split_mode` | `layer` | Multi-GPU split policy. `none` keeps the model on `main_gpu`, `layer` slices by layer range, `row` slices each tensor row-wise. A bad atom raises `badarg`. |
| `main_gpu` | 0 | GPU index when `split_mode = none`, or the device that holds non-split tensors otherwise. |
| `tensor_split` | `[]` | Per-device proportions when splitting. Up to 16 floats (the vendored llama.cpp's `llama_max_devices()`); shorter lists zero-fill. |
| `use_mmap` | true | mmap the GGUF instead of copying into anon RAM. Leave on. |
| `use_mlock` | false | `mlock(2)` the model pages. Useful on workloads where `vm.swappiness` is non-zero and you can't afford to page out weights. |
| `vocab_only` | false | Open the file but skip weight loading. Tokenizer-only mode. |

### `context_opts`

Pass-through to `llama_context_default_params()`.

| Key | Default | Notes |
|---|---|---|
| `n_ctx` | 2048 | Maximum context length the model will accept. Caching is keyed on this. Setting it higher than the model trained on will silently degrade quality past the training horizon. |
| `n_batch` | 512 | Maximum tokens fed to a single `llama_decode` call. Bigger values prefill faster but use more VRAM/RAM. 4096 is a sane upper bound for 8B-class models on a 24 GB GPU. |
| `n_ubatch` | n_batch | Micro-batch size. Usually leave equal to `n_batch`. |
| `n_seq_max` | 1 | Maximum concurrent sequences. The default keeps single-tenant behaviour bit-for-bit; set `> 1` to opt into the multi-tenant scheduler so up to N requests prefill and decode concurrently through one `llama_decode` per tick. Capped at 256. |
| `n_threads` | hw_concurrency | CPU threads for prompt eval. |
| `n_threads_batch` | n_threads | CPU threads for batch eval. |
| `flash_attn` | `auto` | `true` enables, `false` disables, `auto` lets llama.cpp decide based on the build and model. |
| `type_k` | `f16` | KV cache element type for keys. One of `f16`, `f32`, `bf16`, `q4_0`, `q5_0`, `q5_1`, `q8_0`. Quantised KV trades a bit of quality for roughly 2x cache footprint reduction. |
| `type_v` | `f16` | KV cache element type for values. Same atom set as `type_k`. |
| `kv_unified` | false | Share one KV buffer across sequences. erllama sets it for `n_seq_max > 1` so every sequence can use the full context. |
| `embeddings` | false | Enable embedding output; required for `embed/2` and `embed_batch/2`. |
| `offload_kqv` | true | Keep the KV cache on the GPU when layers are offloaded. |
| `decode_budget_ms` | 30000 | Bound on one `llama_decode` call; a stalled decode fails the request with `decode_timeout` instead of hanging the model. |

### `fingerprint`

A 32-byte SHA-256 over the model file. The cache key includes this
fingerprint so a hit is bound to the exact GGUF that produced it; if
you replace the model on disk, old cache rows are no longer
addressable and will be evicted by LRU.

```erlang
{ok, Bin} = file:read_file(Path),
Fp = crypto:hash(sha256, Bin).
```

### `fingerprint_mode`

How aggressively the cache trusts the fingerprint:

- `safe` — recompute the fingerprint at load time. Slow on multi-GB
  files but ironclad.
- `gguf_chunked` — fingerprint the GGUF metadata chunk and the first
  weights tensor only. Order of magnitude faster; defeats accidental
  but not malicious tampering.
- `fast_unsafe` — trust whatever you pass in. Use only if you
  fingerprint upstream and pass the result through.

### `quant_type` and `quant_bits`

Identifies the quantisation byte-for-byte. Two models with the same
weights but different quant schemes have different cache rows.

### `ctx_params_hash`

A SHA-256 over the parts of `context_opts` that change KV layout —
typically `(n_ctx, n_batch)`. erllama treats two contexts with
different params as different cache namespaces.

```erlang
CtxHash = crypto:hash(sha256, term_to_binary({Nctx, Nbatch})).
```

### `context_size`

Plain integer copy of `n_ctx`. The cache uses it for bounds checks.

### `tier_srv` and `tier`

Where saves go. The RAM tier (`erllama_cache_ram`, `tier => ram`) is
always on and is the default. For `ram_file` or `disk`, add a tier and
reference it by name:

```erlang
ok = erllama_cache:add_tier(#{name => my_disk, backend => disk,
                              root => "/var/lib/erllama/kvc"}),
{ok, M} = erllama:load_model(Config#{tier_srv => my_disk, tier => disk}).
```

Tiers can also be declared once in the application environment
(`tiers`, see the [configuration guide](configuration.md)) so they
start with the application. `load_model` checks that `tier_srv` is
running and that `tier` matches its backend, so a mismatch fails at
load time (`{error, {invalid_config, tier, disk}}`), not at the first
save.

For production deployments use the disk tier: it survives restarts
and is the cheapest place to keep warm state.

### `policy`

Optional per-model overrides of the cache save-policy gates. Any
keys you omit fall back to the application defaults declared in
`erllama.app.src` (`min_tokens`, `cold_min_tokens`,
`cold_max_tokens`, `continued_interval`, `boundary_trim_tokens`,
`boundary_align_tokens`, `session_resume_wait_ms`). See the
[caching guide](caching.md) for what each gate means. Pass an empty
map (or omit the key entirely) to use the defaults.

### `thinking_markers`

`#{start => binary(), 'end' => binary()}`: the byte markers that open
and close an extended-thinking block for models that emit one (for
example `<think>` / `</think>`). With markers set and `thinking =>
enabled` on the request, the text between them arrives as
`{thinking, Bin}` stream events instead of `{token, Bin}`.

### `model_id`

Explicit id for `load_model/1`; the same as calling `load_model/2`.
Loading an id that is in use returns `{error, already_loaded}`.

## Loading multiple models

`load_model/2` takes an explicit binary id and is idempotent against
`{already_started, _}`: calling it twice with the same id returns
`{error, already_loaded}` the second time. To run two distinct models
concurrently:

```erlang
{ok, _} = erllama:load_model(<<"tiny">>, TinyConfig).
{ok, _} = erllama:load_model(<<"big">>,  BigConfig).
{ok, #{reply := R}}  = erllama:complete(<<"tiny">>, <<"hello">>).
{ok, #{reply := R2}} = erllama:complete(<<"big">>,  <<"hello">>).
```

Both share one `erllama_cache` instance — cache rows are scoped by
fingerprint, so they never collide.

## Unloading

```erlang
ok = erllama:unload(M).
```

Triggers a synchronous `shutdown` save (best-effort: capped by
`evict_save_timeout_ms`) and terminates the gen_statem. Any
in-flight cache writes are awaited up to that timeout.

## Common pitfalls

- **Forgetting the fingerprint.** Without it the cache key falls back
  to the path string, which means renaming the file invalidates the
  cache. Always pass an actual hash.
- **Wrong `n_ctx`.** The cache key includes `ctx_params_hash`. If you
  bump `n_ctx` for a tenant, expect a one-shot cold prefill across
  every cached prefix until the new rows accumulate.
