# Observability

Watching what erllama is doing: per-model state, cache behavior,
in-flight requests, memory, load progress, and llama.cpp's own logs.
You need this page when you operate erllama in production or debug
latency, and when you build a router in front of several models.

## Per-model state

```erlang
{ok, Info} = erllama:model_info(M),
#{status := Status,            %% idle | prefilling | generating
  n_seq_max := N, available_seqs := Avail, pinned_idle_seqs := Pins,
  arch := Arch, n_ctx_train := Train, recurrent := Rec,
  vram_estimate_b := Vram} = Info.
```

`model_info/1` is a synchronous snapshot: lifecycle status, sequence
pool balance, the model-family facts probed at load (`arch`,
`n_params`, `recurrent`, `hybrid`, `n_swa`, ...), quantisation, and
a VRAM estimate. `erllama:list_models/0` returns the same map for
every loaded model.

For hot paths (a router picking a model per request) use the
lock-free accessors instead; they read an ETS row without touching
the model process:

```erlang
{ok, Phase} = erllama:phase(M),          %% idle | prefilling | generating
{ok, Pending} = erllama:pending_len(M),  %% queued admissions
{ok, Hit} = erllama:last_cache_hit(M),   %% #{kind, prefix_len} | undefined
Total = erllama:queue_depth(),           %% node-wide queued admissions
{ok, D} = erllama:queue_depth(ModelId).  %% one model's queue depth
```

## In-flight requests

```erlang
Reqs = erllama:requests(),
%% [#{ref := Ref, pid := Pid, model := ModelId}, ...]
{ok, ReqInfo} = erllama:request_info(Ref).
```

Every admitted streaming request is registered until it finishes;
`request_info/1` returns `{error, not_found}` afterwards. Use this to
build cancellation UIs (`erllama:cancel(Ref)` works from any
process).

## Cache counters

```erlang
Counters = erllama:counters().
%% #{hits_exact, hits_resume, hits_longest_prefix, hits_sticky_partial,
%%   misses, saves_cold, saves_finish, evictions, restore_failed, ...}
```

Counters are node-wide, monotonic atomics; sample them periodically
into your metrics system (Prometheus, statsd, OpenTelemetry). Two to
alert on:

- a rising `misses` to hits ratio means your prompts stopped sharing
  prefixes (template drift, fingerprint churn);
- a non-zero `restore_failed` on a recurrent model means exact-hit
  reuse is degraded to cold fallbacks (correct, but slower).

Ask the cache what a prompt would hit before running it:

```erlang
{ok, Bytes} = erllama:cached_prefix_len(M, PromptTokens).
```

`erllama_cache:info/0` reports rows and bytes per tier, and
`erllama_cache:gc/0` / `evict_bytes/1,2` give you manual control
(see [caching](caching.md)).

## Memory

```erlang
{ok, #{total_b := T, free_b := F, used_b := U}} = erllama:vram_info(),
{ok, #{source := Src, used_b := U2, total_b := T2}} = erllama:pressure().
```

`vram_info/0` sums GPU memory across devices (`{error, no_gpu}` on
CPU-only builds). For the per-device breakdown use
`erllama:list_devices/0`: every registered backend device with its
name, type, free/total bytes, and capability flags (the same names
the `devices` load option accepts; see
[loading](loading.md#devices-moe-offload-and-auto-fit)).
`pressure/0` samples the source the background
evictor uses; `erllama:pressure_sources/0` lists what is available
on this host. The scheduler evicts cache tiers under pressure
automatically; see [configuration](configuration.md).

## Load progress

`load_model` blocks for the whole GGUF read. Get progress while it
runs:

```erlang
Self = self(),
spawn_link(fun() ->
    {ok, _} = erllama:load_model(#{model_path => Path, progress_to => Self, model_id => <<"big">>})
end),
receive {erllama_load_progress, <<"big">>, P} -> io:format("~.0f%~n", [P * 100]) end.
```

Messages are `{erllama_load_progress, ModelId, Float}`: values in
`[0.0, 1.0]`, non-decreasing, throttled to whole-percent steps,
ending with exactly `1.0`.

## Native llama.cpp logs

llama.cpp and ggml log to stderr by default. Route their lines into
Erlang's `logger` instead:

```erlang
%% in sys.config, before the application starts:
{erllama, [{native_log_level, info}]}
```

Lines arrive as `logger` events under the domain `[erllama, native]`
with levels mapped from ggml's (`debug | info | warning | error`).
The default level is `warning`, which keeps the load banner quiet;
`info` surfaces it. Filter or fan out with a standard `logger`
handler on the domain:

```erlang
logger:add_handler(llama_native, logger_std_h, #{
    filters => [{native, {fun logger_filters:domain/2, {log, sub, [erllama, native]}}}],
    filter_default => stop
}).
```

## Per-call tracing

Every API call runs through a hackney-style middleware chain: wrap
calls to time them, tag them, or short-circuit them. See
[middleware](middleware.md).

## See also

- [Caching](caching.md) - counter semantics, tiers, eviction
- [Configuration](configuration.md) - scheduler and pressure knobs
- [Middleware](middleware.md) - per-call instrumentation
