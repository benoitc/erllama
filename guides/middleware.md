# Middleware

erllama runs its main API calls (loading, generation, chat,
tokenization, embeddings) through a middleware chain, the same way
hackney wraps `hackney:request/1..5`. A middleware is a plain fun
`fun(Request, Next) -> Response` that can observe, rewrite,
short-circuit or wrap a call. There is no behaviour, no registry and
no dependency: metrics, logging, tracing and caching are recipes you
write against this one hook. Use it when you need to know what
erllama is doing, or to change it, without touching your call sites.

```erlang
-type request() :: #{op := atom(), model := erllama:model() | undefined, args := map()}.
-type next() :: fun((request()) -> term()).
-type middleware() :: fun((request(), next()) -> term()).
```

## Which calls go through the chain

| Intercepted (`op`) | Not intercepted |
|---|---|
| `load_model`, `unload` | `cancel/1` |
| `complete`, `stream`, `prefill_only`, `continue` | `end_session/2`, `reset_session/2` |
| `draft_tokens/3` (seen as `stream`) | `evict/1`, `shutdown/1` |
| `fork_session` | `render_chat_template/2` |
| `tokenize`, `detokenize`, `vocab_info` | `load_adapter/2`, `unload_adapter/2`, `set_adapter_scale/3`, `list_adapters/1` |
| `chat`, `chat_apply`, `chat_parse` | `verify/4`, `cached_prefix_len/2` |
| `embed`, `embed_batch` | `collect/2` and the status and observability getters (`model_info/1`, `status/1`, `phase/1`, `pending_len/1`, `last_cache_hit/1`, `requests/0`, `request_info/1`, `counters/0`, ...) |

A call in the right-hand column never reaches a middleware. Do not
build access control on middleware alone: a caller who can reach
`erllama` can cancel, end sessions, evict, change adapters or verify
without passing through your chain. Gate those calls in your own code.
`erllama_middleware_tests` (`facade_ops_all_wrapped_test`,
`facade_ops_not_wrapped_test`) checks both columns.

## Chain order

Outermost first. `[A, B, C]` means A wraps B wraps C: the request
flows `A -> B -> C -> erllama` and the response unwinds
`erllama -> C -> B -> A`.

## Install a chain

Global, applied to every call:

```erlang
application:set_env(erllama, middleware, [Log, Metrics]).
```

Per call, which replaces the global chain for that call:

```erlang
erllama:complete(Model, Prompt, #{middleware => [Log]}).
```

Per-call chains are accepted by every function that takes an option
map: `complete/3`, `prefill_only/3`, `stream/3`, `continue/3`,
`chat/3`, `chat_apply/3`. The other wrapped calls (`load_model`,
`unload`, `fork_session`, `tokenize`, `detokenize`, `vocab_info`,
`embed`, `embed_batch`, `chat_parse`) only see the global chain.

## What a middleware sees

| op | `args` |
|---|---|
| `load_model` | `#{config := map()}`; `model` is the id |
| `unload` | `#{}` |
| `complete` | `#{prompt := binary(), opts := map()}` |
| `prefill_only` | `#{tokens := [token_id()], opts := map()}` |
| `stream` | `#{prompt := binary() \| [token_id()], opts := map()}` |
| `continue` | `#{tokens := [token_id()], opts := map()}` |
| `chat` | `#{messages := [map()], opts := map()}` |
| `chat_apply` | `#{messages := [map()], opts := map()}` |
| `chat_parse` | `#{params := term(), input := binary(), partial := boolean()}` |
| `embed`, `embed_batch` | `#{input := term()}` |
| `fork_session` | `#{src := term(), new := term()}` |
| `tokenize` | `#{text := binary(), opts := map()}` |
| `detokenize` | `#{tokens := [token_id()]}`, plus `opts` from `detokenize/3` |
| `vocab_info` | `#{}` |

`opts` is the validated option map without the `middleware` key. The
response is whatever the function returns.

Validation runs before the chain: a bad option map is rejected without
calling any middleware.

Streaming calls (`stream`, `continue`) return `{ok, Ref}` to the chain;
the tokens arrive later at the `to` process. To time a whole streamed
request, hand erllama a proxy process as `to` and forward the events
from there (see the last recipe).

If a middleware raises, the exception propagates to the caller. erllama
does not wrap user code in `try`/`catch`.

## Recipes

### Log every call

```erlang
Log = fun(#{op := Op, model := Model} = Req, Next) ->
    T0 = erlang:monotonic_time(millisecond),
    Resp = Next(Req),
    Outcome = case Resp of
        {ok, _} -> ok;
        {ok, _, _} -> ok;
        ok -> ok;
        {error, Reason} -> {error, Reason}
    end,
    logger:info("erllama ~p ~p -> ~p (~p ms)",
                [Op, Model, Outcome, erlang:monotonic_time(millisecond) - T0]),
    Resp
end,
application:set_env(erllama, middleware, [Log]).
```

### Prometheus

```erlang
Prom = fun(#{op := Op} = Req, Next) ->
    T0 = erlang:monotonic_time(),
    prometheus_gauge:inc(erllama_calls_active, [Op]),
    Resp = Next(Req),
    prometheus_gauge:dec(erllama_calls_active, [Op]),
    prometheus_counter:inc(erllama_calls_total, [Op]),
    Dt = erlang:convert_time_unit(erlang:monotonic_time() - T0, native, microsecond) / 1_000_000,
    prometheus_histogram:observe(erllama_call_duration_seconds, [Op], Dt),
    Resp
end.
```

Declare the metrics at startup as usual; the middleware only emits.
The per-request timing of generation (`prefill_ms`, `generation_ms`,
`completion_tokens`, `cache_delta`) is in the `stats` map of every
`complete/3`, `collect/2` and `chat/3` result; cache-wide counters are
`erllama_cache:get_counters/0`, sampled from your collector at whatever
cadence you want.

### telemetry

```erlang
Tel = fun(#{op := Op, model := Model} = Req, Next) ->
    T0 = erlang:monotonic_time(),
    Meta = #{op => Op, model => Model},
    telemetry:execute([erllama, call, start], #{system_time => erlang:system_time()}, Meta),
    try
        Resp = Next(Req),
        telemetry:execute([erllama, call, stop],
                          #{duration => erlang:monotonic_time() - T0},
                          Meta#{result => Resp}),
        Resp
    catch Class:Reason:Stack ->
        telemetry:execute([erllama, call, exception],
                          #{duration => erlang:monotonic_time() - T0},
                          Meta#{kind => Class, reason => Reason, stacktrace => Stack}),
        erlang:raise(Class, Reason, Stack)
    end
end.
```

### Rewrite a request

```erlang
CapTokens = fun(#{op := complete, args := #{opts := O} = A} = Req, Next) ->
                    Next(Req#{args := A#{opts := O#{response_tokens => min(256, maps:get(response_tokens, O, 64))}}});
               (Req, Next) ->
                    Next(Req)
            end.
```

### Short-circuit

A middleware that does not call `Next` answers on its own and the
model is never called:

```erlang
PromptCache = fun(#{op := chat_apply, args := Args} = Req, Next) ->
                      Key = erlang:phash2(Args),
                      case ets:lookup(prompt_cache, Key) of
                          [{_, Resp}] -> Resp;
                          [] -> Resp = Next(Req), ets:insert(prompt_cache, {Key, Resp}), Resp
                      end;
                 (Req, Next) ->
                      Next(Req)
              end.
```

(`chat_apply` results carry a parser handle that is only valid for one
request; cache the `prompt` field, not the whole map, if you do this
for real.)

### Time a streamed request end to end

```erlang
Timed = fun(#{op := stream, args := #{opts := O} = A} = Req, Next) ->
    Caller = maps:get(to, O, self()),
    T0 = erlang:monotonic_time(millisecond),
    Proxy = spawn(fun() -> forward(Caller, T0) end),
    Next(Req#{args := A#{opts := O#{to => Proxy}}})
end.

forward(Caller, T0) ->
    receive
        {erllama, Ref, {done, Stats}} = Msg ->
            logger:info("stream ~p done in ~p ms", [Ref, erlang:monotonic_time(millisecond) - T0]),
            Caller ! Msg;
        {erllama, _, {error, _}} = Msg ->
            Caller ! Msg;
        Msg ->
            Caller ! Msg,
            forward(Caller, T0)
    end.
```
