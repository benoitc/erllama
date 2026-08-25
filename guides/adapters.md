# LoRA adapters

Attaching low-rank adapters to a loaded model at runtime. You need
this page when you serve one base model with per-tenant or per-task
fine-tunes instead of shipping a merged GGUF per variant.

## Attach

```erlang
{ok, M} = erllama:load_model(#{model_path => BasePath}),
{ok, Adapter} = erllama:load_adapter(M, "/srv/adapters/support-tone.gguf"),
{ok, R} = erllama:complete(M, Prompt, #{response_tokens => 64}).
```

`load_adapter/2` reads a LoRA GGUF, attaches it to the model, and
applies it to every subsequent request. The handle is opaque; keep
it to detach or rescale later. Attach several adapters and they
compose.

## Scale and detach

```erlang
ok = erllama:set_adapter_scale(M, Adapter, 0.5),
Adapters = erllama:list_adapters(M),
ok = erllama:unload_adapter(M, Adapter).
```

Scale multiplies the adapter's contribution (1.0 is the trained
strength; 0.0 disables it without detaching). `unload_adapter/2` is
idempotent.

## Adapters and the cache

The prefix cache is keyed on an effective fingerprint that covers
the base model plus the attached adapter set and scales. That means:

- Changing the adapter set or a scale moves you to a fresh cache
  namespace: the next request prefixes cold, then warms as usual.
- Switching back to a previous set restores the previous namespace;
  its rows are still there (until evicted).
- Two models with the same base but different adapters never
  false-hit each other's rows.

So treat adapter flips like a model swap for latency purposes: cheap
to do, but the first request afterwards pays a cold prefill.

## Notes

- Adapters must target the loaded base model; llama.cpp rejects
  mismatched shapes at load with an error, not a crash.
- In-flight requests keep the adapter set they started with; changes
  apply from the next admission.
- The stub backend accepts the whole adapter API as no-ops with
  distinct handles, so supervisor and cleanup logic is unit-testable
  (see [testing](testing.md)).

## See also

- [Loading a model](loading.md) - fingerprints and cache identity
- [Caching](caching.md) - what a namespace switch means for hits
