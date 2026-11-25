# Examples

Copy-paste recipes that combine several erllama features. Each block
runs from `rebar3 shell` after
`{ok, _} = application:ensure_all_started(erllama)` (the boot is
omitted from the snippets). Topics a guide already covers are listed
at the end with a link to that guide.

## 10-second smoke test (no model required)

The stub backend runs the whole API without a GGUF or a NIF. From
`rebar3 shell`:

```erlang
1> {ok, _} = application:ensure_all_started(erllama).
2> {ok, M} = erllama:load_model(#{
       backend => erllama_model_stub,
       policy => #{min_tokens => 1, cold_min_tokens => 1,
                   boundary_trim_tokens => 0, boundary_align_tokens => 1}
   }).
3> {ok, #{cache_hit_kind := cold}} =
       erllama:complete(M, <<"one two three four">>, #{response_tokens => 4}).
4> {ok, #{cache_hit_kind := exact}} =
       erllama:complete(M, <<"one two three four">>, #{response_tokens => 4}).
5> erllama_cache:get_counters().
6> ok = erllama:unload(M).
```

The second completion is a byte-exact cache hit on the first one's
cold save, restored from the RAM tier. The small `policy` lets a
four-word prompt be saved; the defaults only save from 512 tokens.
Nothing touches llama.cpp: `llama_backend_init` never runs, so there
is no Metal/CUDA discovery cost.

## 1. Load a model and run a one-shot completion

```erlang
ok = erllama_cache:add_tier(#{name => my_disk, backend => disk,
                              root => "/var/lib/erllama/kvc"}),
{ok, Bin} = file:read_file("/srv/models/tinyllama-1.1b-chat.Q4_K_M.gguf"),
Fp = crypto:hash(sha256, Bin),

{ok, M} = erllama:load_model(#{
    backend          => erllama_model_llama,
    model_path       => "/srv/models/tinyllama-1.1b-chat.Q4_K_M.gguf",
    fingerprint      => Fp,
    fingerprint_mode => safe,
    quant_type       => q4_k_m,
    quant_bits       => 4,
    ctx_params_hash  => crypto:hash(sha256, term_to_binary({2048, 512})),
    context_size     => 2048,
    tier_srv         => my_disk,
    tier             => disk
}),

{ok, #{reply := Reply}} =
    erllama:complete(M, <<"Once upon a time, in a quiet village">>),

io:format("~s~n", [Reply]),
ok = erllama:unload(M).
```

First call cold-prefills the prompt and async-saves cold + finish
rows. Repeating the same call hits the cache via the exact-key path.

## 2. Stateless HTTP server (resends full conversation per turn)

```erlang
%% Inside your request handler. The cache walks the new prompt's
%% tokens backward by `boundary_align_tokens` and resumes from the
%% longest published prefix automatically: no parent_key needed.
handle_chat(ModelId, Prompt) ->
    {ok, #{reply := Reply}} =
        erllama:complete(ModelId, Prompt, #{response_tokens => 256}),
    {200, [{"Content-Type", "text/plain"}], Reply}.
```

Hits show up as `hits_longest_prefix` in `erllama_cache:get_counters/0`.

## 3. Concurrent agents on a shared system prompt

```erlang
ModelId = <<"assistant">>,
SharedPrefix = <<"You are a helpful assistant.\n">>,
Parent = self(),

%% Spawn N workers; each appends a different user query but they
%% all start with the same prefix. After the first agent's cold
%% prefill saves, every subsequent agent gets a longest-prefix hit
%% on the shared part and only prefills its tail.
Workers = [
    spawn(fun() ->
        Q = list_to_binary(io_lib:format("Worker ~p question.", [N])),
        Prompt = <<SharedPrefix/binary, Q/binary>>,
        {ok, #{reply := Reply}} = erllama:complete(ModelId, Prompt),
        Parent ! {N, Reply}
    end) || N <- lists:seq(1, 8)
],

%% Collect.
Replies = [receive {N, R} -> {N, R} end || N <- lists:seq(1, 8)],
Replies.
```

## 4. Streaming tokens (`stream/3`) with cancellation

```erlang
{ok, Tokens} = erllama:tokenize(ModelId, <<"Once upon a time">>),
{ok, Ref} = erllama:stream(ModelId, Tokens,
                           #{response_tokens => 200}),

loop(Ref) ->
    receive
        {erllama, Ref, {token, Fragment}} ->
            io:put_chars(Fragment),
            loop(Ref);
        {erllama, Ref, {done, _Stats}} ->
            io:nl(),
            ok;
        {erllama, Ref, {error, Reason}} ->
            {error, Reason}
    after 30000 ->
        erllama:cancel(Ref),
        %% Still drain the final done message after cancel.
        loop(Ref)
    end.
```

`cancel/1` is observed at the next inter-token boundary; the model
always emits a final `{erllama, Ref, {done, Stats}}` with
`#{cancelled => true}`.

## 5. Grammar-constrained sampling (GBNF)

```erlang
%% Force the model to emit a JSON-shaped string with a digit value.
Grammar = <<
    "root ::= \"{\" ws \"\\\"n\\\":\" ws digit \"}\"\n"
    "digit ::= [0-9]\n"
    "ws    ::= [ \\t\\n]*"
>>,
{ok, Toks} = erllama:tokenize(ModelId, <<"Reply with JSON:">>),
{ok, Ref}  = erllama:stream(ModelId, Toks,
                           #{response_tokens => 32, grammar => Grammar}).
```

The grammar is per request: it constrains every sampled token of this
request and no other.

## 6. Collect a thinking block and its answer

The model needs `thinking_markers` at load (see
[loading](loading.md#thinking_markers)); set `thinking_signing_key` in
the application environment to sign each block (see
[configuration](configuration.md#thinking_signing_key)).

```erlang
{ok, Tokens} = erllama:tokenize(<<"qwen3-thinking">>, <<"Solve: 23 * 17.">>),
{ok, Ref} = erllama:stream(<<"qwen3-thinking">>, Tokens,
                          #{response_tokens => 256,
                            thinking => enabled}),
Result = loop(Ref, <<>>, <<>>, <<>>).

loop(Ref, Thinking, Answer, Sig) ->
    receive
        {erllama, Ref, {thinking, Bin}} ->
            loop(Ref, <<Thinking/binary, Bin/binary>>, Answer, Sig);
        {erllama, Ref, {thinking_end, NewSig}} ->
            loop(Ref, Thinking, Answer, NewSig);
        {erllama, Ref, {token, Bin}} when is_binary(Bin) ->
            loop(Ref, Thinking, <<Answer/binary, Bin/binary>>, Sig);
        {erllama, Ref, {done, _Stats}} ->
            #{thinking => Thinking, answer => Answer, signature => Sig}
    end.
```

Add `thinking_budget_tokens => N` to cap the thinking phase: past `N`
thinking payloads erllama closes the block with
`{thinking_end, Sig}` and the rest of the output arrives as answer
tokens. See [generating text](generation.md#extended-thinking).

## 7. Warm a prompt without sampling (`prefill_only/2,3`)

Useful for priming the cache before a burst of short follow-ups,
or for holding a warm session across long pauses without consuming
generation budget.

```erlang
{ok, SysToks} = erllama:render_chat_template(ModelId, #{
    messages => [
        #{role => system, content => SystemPrompt},
        #{role => user,   content => FirstUserTurn}
    ]
}),
{ok, #{finish_key := FK,
       committed_tokens := N,
       cache_hit_kind := cold}} =
    erllama:prefill_only(ModelId, SysToks),

%% Later turn: pass FK as parent_key so the next complete/3 skips
%% the longest-prefix walk and resumes from the warm row.
{ok, #{reply := R}} =
    erllama:complete(ModelId, NextUserTurn, #{parent_key => FK}).
```

`finish_key` is `undefined` if the prompt was shorter than
`min_tokens` and the finish save was suppressed.

`prefill_only/3` accepts `Opts` with `parent_key`, useful when
chaining cache-warming calls across turns deterministically rather
than relying on the implicit longest-prefix walk:

```erlang
%% Materialise turn-1 bytes into cache.
{ok, #{finish_key := K1}} = erllama:prefill_only(ModelId, Turn1Tokens),

%% Extend the same prefix with turn 2's tokens; only the suffix
%% runs through prefill, returning K2 for the longer prompt.
{ok, #{finish_key := K2,
       cache_hit_kind := partial,
       cache_delta := #{read := R, created := C}}} =
    erllama:prefill_only(ModelId,
                         Turn1Tokens ++ Turn2Tokens,
                         #{parent_key => K1}).
```

## 8. Chat-template continuation (`continue/3`)

`stream/3` with `session_id` reuses the live KV cells across turns,
but only when the new prompt's tokens are a strict prefix extension
of what was stored on the prior turn. Production chat templates
typically render different role markers when the same first user
message is wrapped in a 1-message vs an N-message history, so the
retokenised prefix on turn N does not equal the stored tokens from
turn N-1: the sticky path keeps only the common prefix and
re-prefills the rest on every turn.

`continue/3` lets the caller assert "here is the new tokenised
tail" and the engine appends it directly on top of the resident KV,
with no prefix equality check and no cache lookup. The caller
takes the contract: if the asserted tail is wrong the generation is
garbage tokens, but engine state stays consistent.

```erlang
SessionId = ConvId,

%% Turn 1: render the full conversation as today, stream/3 pins the
%% session on finish.
{ok, Turn1Tokens} = erllama:render_chat_template(ModelId, #{
    messages => [#{role => <<"user">>, content => Q1}]
}),
{ok, Ref1} = erllama:stream(ModelId, Turn1Tokens,
                           #{session_id => SessionId,
                             response_tokens => 64}),
{Reply1, Stats1} = drain(Ref1),

%% After turn 1, the session holds prompt + reply on its pinned seq.
PriorTokenCount = maps:get(prompt_tokens, Stats1)
                + maps:get(completion_tokens, Stats1),

%% Turn 2: re-render the full conversation, then slice off the
%% tokens the engine already has and pass only the new tail to
%% continue/3. No need for the slice to match what the model
%% actually decoded: continue/3 trusts the caller.
{ok, Turn2Tokens} = erllama:render_chat_template(ModelId, #{
    messages => [
        #{role => <<"user">>, content => Q1},
        #{role => <<"assistant">>, content => Reply1},
        #{role => <<"user">>, content => Q2}
    ]
}),
Suffix = lists:nthtail(PriorTokenCount, Turn2Tokens),
{ok, Ref2} = erllama:continue(ModelId, Suffix,
                              #{session_id => SessionId,
                                to => self(),
                                response_tokens => 64}),
{Reply2, Stats2} = drain(Ref2).
%% Stats2.cache_hit_kind = continuation
%% Stats2.cache_delta.read = PriorTokenCount
%% Stats2.cache_delta.created = length(Suffix) + completion_tokens

ok = erllama:end_session(ModelId, SessionId).
```

During caller development it is worth running the slicing path in
debug mode and asserting `StoredTokens ++ Suffix == Turn2Tokens`:
when the chat template renders identically across turns the slice
will be a true tail, and the assertion validates the caller's
math. In production, drop the assertion: the whole point of
`continue/3` is that the slice need not match.

Errors:

- `{error, no_session}`: the session id is not known to the model.
  Falls back to the cold/warm path: rebuild the conversation and
  call `stream/3` instead.
- `{error, sticky_busy}`: an earlier request on the same session
  is still in flight. Serialise calls per session id (typical HTTP
  per-user pipelines do this naturally).
- `{error, {transcript_mismatch, Detail}}`: only when the optional
  `expect_committed` guard is set and diverges (see below).

`parent_key` is ignored on the `continue/3` path because no cache
lookup runs. The caller-asserted tail is the entire admission
contract.

### Guarding the splice (`expect_committed`)

`continue/3` trusts the tail, so a caller whose view of the session
drifted will silently produce garbage. To catch that, reconstruct
the prior turn's exact committed tokens from `Stats.generated` (the
`done` Stats now carry the exact generated token ids) and
pass them as `expect_committed`. The engine checks they equal the
session's stored context before prefilling:

```erlang
PriorCommitted = Turn1Tokens ++ maps:get(generated, Stats1),
{ok, Ref2} = erllama:continue(ModelId, Suffix,
                              #{session_id => SessionId,
                                to => self(),
                                response_tokens => 64,
                                expect_committed => PriorCommitted}),
```

On a mismatch the call returns
`{error, {transcript_mismatch, #{stored_len, expected_len, diverge_at}}}`
and prefills nothing; the seq stays pinned, so the caller can re-sync
its transcript and retry (or fall back to `stream/3`). Omit
`expect_committed` to keep the historical trust-the-tail behaviour.

## 9. Concurrent requests on one model (`n_seq_max`)

By default a model runs one request at a time. With
`context_opts.n_seq_max > 1` the request scheduler co-batches up to N
requests, prefilling and decoding them through one `llama_decode` per
tick.

```erlang
{ok, _} = erllama:load_model(<<"chat">>, Config#{
    context_opts => #{n_ctx => 8192, n_batch => 4096, n_seq_max => 4}
}),

Parent = self(),
Workers = [
    spawn(fun() ->
        Q = list_to_binary(io_lib:format("Worker ~p question.", [N])),
        {ok, #{reply := R}} = erllama:complete(<<"chat">>, Q),
        Parent ! {N, R}
    end) || N <- lists:seq(1, 4)
],
[receive {N, _R} -> ok end || N <- lists:seq(1, 4)].
```

Per-request samplers are isolated: each worker can pass its own
`temperature`, `seed`, or `grammar` in the `Opts` map without
spilling sampler state across the other in-flight requests.
Admissions past `n_seq_max` queue FIFO in the model's `pending`
list, observable via `erllama:pending_len/1`.

Long prompts are sliced by `prefill_chunk_size` (default
`max(64, n_batch div 4)`) so a single heavy prompt cannot
monopolise the batch. See [caching](caching.md) for warm-prefix
behaviour across concurrent workers.

## 10. Per-request cache delta (`cache_creation_input_tokens` / `cache_read_input_tokens`)

Every result and `Stats` map carries `cache_delta => #{read := N,
created := N}`. `read` counts tokens served from the warm prefix
at admission, `created` counts tokens this request added to the
cache beyond that prefix. Both default to `0`; together they
populate Anthropic's `cache_read_input_tokens` and
`cache_creation_input_tokens` accurately.

```erlang
{ok, #{cache_delta := #{read := R, created := C},
       cache_hit_kind := Hit}} =
    erllama:complete(ModelId, Prompt, #{response_tokens => 64}),
io:format("hit=~p read=~p created=~p~n", [Hit, R, C]).
```

What each combination tells you:

| `cache_hit_kind` | `read`         | `created`        |
| ---------------- | -------------- | ---------------- |
| `cold`           | `0`            | `prompt + generated` |
| `partial`        | warm prefix    | `committed - warm prefix` |
| `exact`          | full prompt    | `generated` |
| any, save below `min_tokens` | warm prefix | `0` (no save fired) |

Streaming consumers read the same map from the final
`{erllama, Ref, {done, Stats}}` message; `prefill_only/2` exposes it
on its result map too.

## 11. Microbench: cold vs. warm

```bash
bench/run.sh tiny    # TinyLlama 1.1B Q4_K_M
bench/run.sh large   # LLaMA-3 8B Q4_K_M (needs the file)
```

`bench/run.sh` drives a `cold_vs_warm` matrix plus a 4-agent
shared-prefix scenario; see `bench/README.md`.

## Covered in the guides

- Multi-turn with `parent_key` / `finish_key`, sticky sessions, forking:
  [sessions](sessions.md).
- Several models on one node: [loading](loading.md#loading-multiple-models).
- Load progress messages: [loading](loading.md#load-progress).
- Chat turns and tool calls: [tool calls](tool-calls.md); JSON-schema
  replies: [tool calls](tool-calls.md#forcing-a-schema-instead-of-tools).
- Embeddings: [embeddings](embeddings.md).
- Stop sequences, logprobs, extended thinking, speculative decoding:
  [generating text](generation.md).
- Fill-in-the-middle: [tokens](tokens.md#fill-in-the-middle-infill).
- Router accessors (`phase/1`, `pending_len/1`, `last_cache_hit/1`),
  cache counters, native logs: [observability](observability.md).
- Cache inspection, `gc/0`, `evict_bytes/1,2` and the cache eviction
  scheduler: [caching](caching.md).
- Tests with the stub backend or a real GGUF: [testing](testing.md).
- Something does not work: [troubleshooting](troubleshooting.md).

## See also

- [Configuration](configuration.md): full `sys.config` reference.
- [Building](building.md): platform-specific build notes.
