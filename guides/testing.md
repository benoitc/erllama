# Testing your code

Running your application's tests against the full erllama API with no
GGUF file, no NIF, and no hardware. You need this page when you write
unit tests for code that calls erllama: agent loops, HTTP handlers,
routers, retrieval pipelines.

## The stub backend

`erllama_model_stub` is a deterministic in-memory backend. Load it by
setting `backend` and skipping `model_path`:

```erlang
{ok, _} = application:ensure_all_started(erllama),
{ok, M} = erllama:load_model(<<"test">>, #{backend => erllama_model_stub}),
{ok, #{reply := Reply}} = erllama:complete(M, <<"hello world">>, #{response_tokens => 4}).
```

What it gives you:

- **Determinism.** Tokenization hashes whitespace-delimited words;
  each generated token derives from the context hash. The same
  prompt always yields the same tokens, so assertions are stable.
- **A real cache.** `kv_pack`/`kv_unpack` serialize the token list,
  so cold saves, warm restores, sessions, forks, and `cache_delta`
  behave end to end exactly like the llama backend.
- **The full event protocol.** Streaming events, stop sequences,
  logprobs (synthetic), thinking events, cancellation, and the
  session API all work.

Things a stub cannot give you: meaningful text, chat templates
(`chat/3` returns `{error, chat_not_supported}`), embeddings that
mean anything (deterministic 16-dim vectors), and load progress.

## An eunit fixture

```erlang
with_erllama(Fun) ->
    {ok, Started} = application:ensure_all_started(erllama),
    {ok, M} = erllama:load_model(#{backend => erllama_model_stub}),
    try
        Fun(M)
    after
        erllama:unload(M),
        [application:stop(A) || A <- lists:reverse(Started)]
    end.

my_handler_streams_test() ->
    with_erllama(fun(M) ->
        {ok, Ref} = erllama:stream(M, <<"prompt words here">>, #{response_tokens => 3}),
        {ok, #{generated := Gen}} = erllama:collect(Ref, 5000),
        ?assertEqual(3, length(Gen))
    end).
```

## Behavior knobs

The stub accepts load-config keys that shape its behavior for
specific test scenarios:

| Key | What it does |
|---|---|
| `step_delay_ms => N` | Hold every decode step N ms, so you can keep a request in flight while another races it (queueing, `sticky_busy`, cancellation tests). |
| `thinking_capable => true` | Emit two `{thinking, _}` fragments and a `{thinking_end, _}` before the answer tokens. |
| `fail_seq_rm_last => true` | Partial KV removals fail, emulating a recurrent model: warm exact hits fall back to cold and bump `restore_failed`. |
| `fail_kv_unpack => true` | Cache restores fail: rows behave as misses. |
| `fail_seq_cp => true` | `fork_session/3` fails with `seq_cp_failed`. |

```erlang
{ok, M} = erllama:load_model(#{
    backend => erllama_model_stub,
    step_delay_ms => 50
}),
{ok, Ref} = erllama:stream(M, <<"slow one">>, #{response_tokens => 100}),
%% the request is reliably still running here:
ok = erllama:cancel(Ref).
```

Test helpers on the module itself let you assert what the engine did:
`erllama_model_stub:sampler_new_cfgs/0` records every sampler config
built, `seq_rm_last_calls/0` and `seq_cp_calls/0` record KV
operations, `wedge_next_step/1` makes the next decode fail with a
reason of your choice (recovery-path testing).

## Testing against a real model

Keep one small, fast integration suite gated on an environment
variable, following erllama's own pattern:

```erlang
init_per_suite(Config) ->
    case os:getenv("MY_TEST_MODEL") of
        false -> {skip, "set MY_TEST_MODEL to a GGUF path"};
        Path ->
            {ok, _} = application:ensure_all_started(erllama),
            [{model_path, Path} | Config]
    end.
```

A 1-2B instruct model (Qwen2.5-1.5B, TinyLlama) is enough to verify
prompts render and replies parse; use `temperature => 0.0` so runs
are reproducible. From escripts, exit with `init:stop()` rather than
`halt()` so the GPU backend tears down cleanly.

## Notes

- The stub needs no `model_path` but accepts every other load key,
  so your production config maps mostly reuse.
- Cache-dependent assertions want a permissive policy in tests:
  `policy => #{min_tokens => 4, cold_min_tokens => 4, boundary_trim_tokens => 0, boundary_align_tokens => 1}`
  makes even short prompts save and hit.

## See also

- [Examples](examples.md) - a 10-second smoke test, more recipes
- [Loading a model](loading.md) - the full config map
- `erllama_model_backend` - write your own backend behind the same API
