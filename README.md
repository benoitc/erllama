# erllama

[![CI](https://github.com/benoitc/erllama/actions/workflows/ci.yml/badge.svg)](https://github.com/benoitc/erllama/actions/workflows/ci.yml)
[![Hex.pm](https://img.shields.io/hexpm/v/erllama.svg)](https://hex.pm/packages/erllama)

Run `llama.cpp` from Erlang. Keep prompts warm. Stay inside OTP.

erllama is a native Erlang/OTP runtime for `llama.cpp` with supervised
model processes, OpenAI-shaped completion APIs, and a byte-exact KV
cache that turns repeated prompt prefill from seconds into milliseconds.

If your app sends the same system prompt, agent scaffold, or conversation
prefix again and again, erllama saves the model state once and restores it
on the next request. No fuzzy matching. No hidden session server. Just
exact tokens, exact cache keys, and OTP supervision around the whole path.

## Why erllama?

- **Fast repeat prompts.** Cache hits restore KV state instead of
  recomputing prompt prefill.
- **Native OTP shape.** Each loaded model is a supervised process with a
  clear lifecycle: load, complete, stream, observe, unload.
- **Bigger-than-RAM warmth.** Hot prefixes can live in RAM, warm prefixes
  in tmpfs, and large working sets on disk.
- **Stateless-server friendly.** Resend the full conversation every turn
  and still get longest-prefix cache hits.
- **Multi-model safe.** Cache rows include the model fingerprint and
  context shape, so different models never collide on identical prompts.
- **Observable by default.** Hit/miss counters and per-model state probes
  are cheap enough to call from routers.
- **Built on `llama.cpp`.** Local GGUF inference with the platform support
  you expect: Metal, BLAS, CUDA toggles, and plain CPU fallback.

## Quick taste

```erlang
1> {ok, _} = application:ensure_all_started(erllama).
2> Path = "/srv/models/tinyllama-1.1b-chat.Q4_K_M.gguf".
3> {ok, Bin} = file:read_file(Path).
4> {ok, Model} = erllama:load_model(#{
       model_path => Path,
       fingerprint => crypto:hash(sha256, Bin)
   }).
{ok, <<"erllama_model_2375">>}

5> {ok, #{reply := Reply, finish_key := Key}} =
       erllama:complete(Model, <<"Once upon a time">>).
%% First call: cold prefill, async save.

6> {ok, #{reply := Reply2}} =
       erllama:complete(Model, <<"Once upon a time">>).
%% Same prompt: KV cache restore.

7> {ok, #{reply := Reply3}} =
       erllama:complete(Model,
                        <<"Once upon a time, in a quiet village">>).
%% Longer prompt: longest cached prefix wins.

8> {ok, #{reply := Reply4}} =
       erllama:complete(Model, <<"and they lived happily ever after">>,
                        #{parent_key => Key}).
%% Stateful resume from the previous finish save.
```

`load_model/1` returns a binary model id. Pass it to `complete/2,3`,
`stream/3`, `chat/3`, `tokenize/2`, `unload/1`, and the rest of the API.

## Install

erllama targets Erlang/OTP **28** and rebar3 **3.25+**.

Add it to `rebar.config`:

```erlang
{deps, [
    {erllama, "~> 0.10"}
]}.
```

Then start the application before loading models:

```erlang
{ok, _} = application:ensure_all_started(erllama).
```

The first compile builds the vendored `llama.cpp`. See
[Building](guides/building.md) for platform notes and CUDA/Metal options.

## API overview

Every call takes the model id (or pid) and returns `{ok, Result}` or
`{error, Reason}`; an unknown model is `{error, not_loaded}`, a bad
option is `{error, {unknown_option, Key}}`. `erllama:error_reason()`
lists every reason.

| Group | Functions |
|---|---|
| Lifecycle | `load_model/1,2`, `unload/1`, `whereis/1`, `list_models/0`, `model_info/1` |
| Completion | `complete/2,3`, `prefill_only/2,3` |
| Streaming | `stream/3`, `collect/2`, `continue/3`, `cancel/1`, `end_session/2`, `reset_session/2` |
| Chat | `chat/3`, `chat_apply/3`, `chat_parse/3`, `render_chat_template/2` |
| Tokens | `tokenize/2,3`, `detokenize/2` |
| Embeddings | `embed/2`, `embed_batch/2` |
| Adapters | `load_adapter/2`, `unload_adapter/2`, `set_adapter_scale/3`, `list_adapters/1` |
| Observability | `status/1`, `phase/1`, `pending_len/1`, `queue_depth/0,1`, `last_cache_hit/1`, `cached_prefix_len/2`, `counters/0`, `vram_info/0`, `pressure/0`, `requests/0` |
| Speculative | `draft_tokens/3`, `verify/4` |
| Memory control | `evict/1`, `shutdown/1` |

The cache has its own module, `erllama_cache` (`add_tier/1`, `info/0`,
`gc/0`, `evict_bytes/1,2`, `get_counters/0`), and every call can be
wrapped by a hackney-style middleware chain (`erllama_middleware`).

### Stream tokens

```erlang
{ok, Ref} = erllama:stream(Model, <<"Once upon a time">>, #{response_tokens => 200}),
loop(Ref).

loop(Ref) ->
    receive
        {erllama, Ref, {token, Bin}} -> io:put_chars(Bin), loop(Ref);
        {erllama, Ref, {done, Stats}} -> {ok, Stats};
        {erllama, Ref, {error, Reason}} -> {error, Reason}
    end.
```

Or let erllama write the loop: `{ok, #{reply := Reply}} = erllama:collect(Ref, 30000)`.

### Chat with tools

```erlang
Tools = [#{name => <<"weather">>,
           description => <<"Current weather for a city">>,
           parameters => #{type => object,
                           properties => #{city => #{type => string}},
                           required => [city]}}],
{ok, #{message := #{content := Text, tool_calls := Calls}}} =
    erllama:chat(Model,
                 [#{role => user, content => <<"Weather in Paris?">>}],
                 #{tools => Tools}).
%% Calls = [#{name => <<"weather">>, arguments => #{<<"city">> => <<"Paris">>}, id => _}]
```

### Test without a model

`erllama_model_stub` is a deterministic backend with no NIF and no
GGUF; load it with `backend => erllama_model_stub` to run your own
code against the whole API in unit tests.

## Common patterns

### Stateless HTTP completion

OpenAI/Anthropic-shaped servers usually resend the whole conversation on
each turn. That is fine. erllama walks the prompt backward and restores
the longest exact prefix it has already saved.

```erlang
handle_completion(ModelId, Prompt) ->
    {ok, #{reply := Reply}} =
        erllama:complete(ModelId, Prompt, #{response_tokens => 256}),
    Reply.
```

### Stateful Erlang session

If your session process already tracks turns, keep the returned
`finish_key` and pass it as `parent_key` on the next request. That skips
the longest-prefix walk and resumes directly from the saved row.

```erlang
{ok, #{reply := R1, finish_key := K1}} =
    erllama:complete(ModelId, Prompt1),

{ok, #{reply := R2, finish_key := K2}} =
    erllama:complete(ModelId, Prompt2, #{parent_key => K1}).
```

### Many models in one BEAM

Each loaded model is its own supervised process. The cache is shared, but
rows are fingerprint-segregated.

```erlang
{ok, _} = erllama:load_model(<<"tiny">>, TinyConfig),
{ok, _} = erllama:load_model(<<"big">>, BigConfig),

{ok, #{reply := R1}} = erllama:complete(<<"tiny">>, <<"summarise: ...">>),
{ok, #{reply := R2}} = erllama:complete(<<"big">>, <<"deep analysis: ...">>),

ok = erllama:unload(<<"tiny">>).
```

### Inspect live state

```erlang
1> erllama_cache:get_counters().
#{hits_exact => 142, hits_resume => 17, hits_longest_prefix => 89,
  misses => 12, saves_cold => 12, saves_finish => 31, ...}

2> erllama:phase(<<"big">>).
{ok, generating}
3> erllama:pending_len(<<"big">>).
{ok, 3}
4> erllama:last_cache_hit(<<"big">>).
{ok, #{kind => partial, prefix_len => 1024}}
```

## Documentation

| Need | Read |
|---|---|
| Load a model | [Loading a model](guides/loading.md) |
| Configure cache tiers and save policy | [Caching](guides/caching.md) |
| Configure `sys.config` and per-model options | [Configuration](guides/configuration.md) |
| Build from source | [Building](guides/building.md) |
| Copy working snippets | [Examples](guides/examples.md) |
| Run chat turns and tool calls | [Tool calls](guides/tool-calls.md) |
| Observe or wrap every call | [Middleware](guides/middleware.md) |
| Understand cache design tradeoffs | [Cache design](internals/cache-design.md) |
| Understand crash-safe save publication | [Publish protocol](internals/publish-protocol.md) |
| Understand request admission and decode flow | [Request lifecycle](internals/request-lifecycle.md) |
| Understand NIF lifetime safety | [NIF safety](internals/nif-safety.md) |

The API reference (`erllama`, `erllama_cache`, `erllama_middleware`,
`erllama_scheduler`, and the `erllama_model_backend` / `erllama_pressure`
behaviours for extensions) is published on
[HexDocs](https://hexdocs.pm/erllama). You can also build it locally:

```bash
rebar3 ex_doc
```

## Architecture

```text
erllama_sup
├── erllama_cache_sup
│   ├── erllama_cache_meta_srv
│   ├── erllama_cache_ram            the always-on RAM tier
│   ├── erllama_cache_writer
│   └── erllama_cache_tier_sup       disk / ram_file tiers (add_tier/1, `tiers` env)
├── erllama_registry
├── erllama_inflight
├── erllama_chat_cache
├── erllama_model_sup
│   └── erllama_model                one supervised gen_statem per loaded model
└── erllama_scheduler                memory-pressure poller, off by default
```

Models point at a tier with `tier` + `tier_srv` in their load config.

The important invariant is simple: cache hits are byte-exact. A key is
SHA-256 over the model fingerprint, quantization, context shape, and the
**rendered prompt bytes** (`detokenize(tokens)`), so a prompt that
retokenises across turns still hits. erllama may find a shorter saved
byte-prefix for a longer prompt, but it never returns an approximate match.

## Requirements

- Erlang/OTP **28**
- rebar3 **3.25+**
- C++17 toolchain
- `cmake` >= 3.20
- Apple Silicon: Metal + Accelerate are auto-detected
- Linux: BLAS is auto-detected; CUDA is enabled with
  `ERLLAMA_OPTS=-DGGML_CUDA=ON`
- FreeBSD: `erlang-runtime28` plus `cmake bash gmake`

## Status

erllama is pre-release. The cache, scheduler, and NIF safety wrappers have
unit, property, and Common Test coverage. The real-model Common Test suite
is gated by `LLAMA_TEST_MODEL` so normal CI can run without a GGUF file.

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Contributing

The contributor guide is [AGENTS.md](AGENTS.md). The short version:

```bash
rebar3 fmt
rebar3 compile
rebar3 eunit
rebar3 proper
rebar3 ct
rebar3 lint
rebar3 dialyzer
rebar3 xref
```

Run the real-model suite when you have a GGUF available:

```bash
LLAMA_TEST_MODEL=/path/to/tinyllama-1.1b-chat.Q4_K_M.gguf \
    rebar3 ct --suite=test/erllama_real_model_SUITE
```

Bumping the vendored `llama.cpp` is covered in
[UPDATE_LLAMA.md](UPDATE_LLAMA.md).

## Related projects

`erllama_cluster` is planned as a separate OTP application for routing,
cache-aware placement, speculative decoding, and distributed inference
across erllama nodes.

Repository: <https://github.com/benoitc/erllama>

## Acknowledgements

Same idea as [antirez/ds4](https://github.com/antirez/ds4).

## License

MIT. Copyright (c) 2026 Benoit Chesneau. See [LICENSE](LICENSE).

The vendored `c_src/llama.cpp/` retains its upstream MIT license; see
`c_src/llama.cpp/LICENSE`.
