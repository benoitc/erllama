# Generating text

How you run inference: one-shot completions, token streams, and the
options that shape both. You need this page before any other API
guide, because every other entry point (`chat/3`, sessions,
embeddings) is built on the request model described here.

## The request model

Every generation request goes through the same pipeline inside the
model process:

1. your prompt is tokenized,
2. the tokens are prefilled into the model's KV memory (reusing any
   cached prefix, see [caching](caching.md)),
3. tokens are decoded one at a time until an end-of-generation token,
   a stop sequence, or the `response_tokens` budget.

`complete/2,3` runs the whole pipeline and returns the result;
`stream/3` returns immediately with a reference and sends you events
as tokens decode. Both take the same option map
(`erllama:request_opts()`).

## One-shot completion

```erlang
{ok, M} = erllama:load_model(#{model_path => "/srv/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"}),
{ok, R} = erllama:complete(M, <<"The capital of France is">>, #{
    response_tokens => 32,
    temperature => 0.0
}),
#{reply := Reply, finish_reason := Why, stats := Stats} = R.
```

`R` is a `completion_result()`: `reply` (detokenized text),
`generated` (the token ids), `finish_reason` (`stop | length |
cancelled`), `finish_key` (a cache key you can pass
as `parent_key` next turn), `cache_hit_kind`, `cache_delta`, and
`stats`.

## Streaming

```erlang
{ok, Ref} = erllama:stream(M, Tokens, #{response_tokens => 200, temperature => 0.7}),
loop(Ref).

loop(Ref) ->
    receive
        {erllama, Ref, {token, Bin}}   -> io:put_chars(Bin), loop(Ref);
        {erllama, Ref, {done, Stats}}  -> {ok, Stats};
        {erllama, Ref, {error, Reason}} -> {error, Reason}
    end.
```

`stream/3` takes a token list; tokenize your prompt first with
`erllama:tokenize/2,3`. Events arrive as `{erllama, Ref, Event}`
tuples at the `to` pid (default: the caller):

- `{token, Bin}` - a text fragment (omitted when empty)
- `{token_id, Id}` - every generated token id, in order
- `{logprobs, Entry}` - per-token logprobs when requested (below)
- `{thinking, Bin}` / `{thinking_end, Sig}` - extended-thinking track
- `{done, Stats}` - terminal success
- `{error, Reason}` - terminal failure; no `done` follows

When you do not want to write the receive loop, fold the stream:

```erlang
{ok, #{reply := Reply, stats := Stats}} = erllama:collect(Ref, 30000).
```

Cancel an in-flight stream with `erllama:cancel(Ref)`; you still
receive `{done, Stats}` with `finish_reason => cancelled`.

## Sampling options

With no sampling options the decode is greedy and deterministic. Set
`temperature > 0` to sample, then shape the distribution with the
other knobs:

```erlang
#{temperature => 0.8, top_p => 0.95, min_p => 0.05, seed => 42}
```

The full knob set mirrors llama.cpp: `top_k`, `typical_p`,
`top_n_sigma`, XTC, dynamic temperature, DRY, mirostat, penalties,
`logit_bias`, `ignore_eos`, and more. See the
[sampling reference](configuration.md#sampling-reference) for every
key, its default, and the chain order. A GBNF `grammar` constrains
every sampled token; see [tool calls](tool-calls.md) for the
grammar-enforced chat paths.

## Stop sequences

```erlang
{ok, R} = erllama:complete(M, Prompt, #{
    stop_sequences => [<<"\nUser:">>, <<"###">>]
}),
case R of
    #{stop_sequence := Matched} -> ...;   %% key present only on a match
    _ -> ...
end.
```

The first occurrence of any entry halts generation; the match is
trimmed from `reply`. In streaming mode erllama holds back the last
few bytes of output so a stop string spanning two tokens is still
caught before you see it.

## Logprobs

Ask for per-token probabilities with `logprobs => N` (1..32):

```erlang
{ok, #{logprobs := Lps}} = erllama:complete(M, Prompt, #{
    response_tokens => 8, temperature => 0.0, logprobs => 5
}).
%% Lps = [#{token_id := Id, logprob := Lp, top := [{AltId, AltLp}, ...]}, ...]
```

Each entry carries the sampled token's log probability plus the
model's top-N alternatives, descending. The values are full-vocab
log-softmax over the raw model distribution, before any sampler
stage (OpenAI semantics), so under greedy decoding the sampled token
always equals the top-1 entry. Streaming callers get the same maps
as `{logprobs, Entry}` events, each preceding its token's
`{token, _}` / `{token_id, _}` events; `collect/2` reassembles them
into the `logprobs` result key. Cost: one O(n_vocab) pass per token,
only when requested.

## Extended thinking

Models with a thinking phase (configured via `thinking_markers` at
load, see [loading](loading.md)) can route it to a separate track:

```erlang
{ok, Ref} = erllama:stream(M, Tokens, #{
    thinking => enabled,
    thinking_budget_tokens => 256
}),
%% {thinking, Bin} fragments, then {thinking_end, Signature},
%% then the normal {token, _} answer stream.
```

Thinking tokens do not land in `reply` or in the cache keys.
`chat/3` has its own template-level controls (`enable_thinking`,
`reasoning_format`); see [tool calls](tool-calls.md).

## Speculative decoding

Speed up generation without changing its output. The `speculative`
option enables draft-model-free ngram speculation: a per-model hash
table over previously seen tokens drafts likely continuations, one
batched decode verifies the whole draft against the model, and the
accepted prefix commits in a single tick. Rejected tokens are rolled
back, so the output is byte-identical to non-speculative decoding,
including seeded sampling at any temperature.

```erlang
{ok, R} = erllama:complete(M, Prompt, #{
    response_tokens => 256,
    speculative => true
}),
#{drafted := D, accepted := A} = maps:get(speculative, R).
```

It works on `stream/3`, `complete/3` and `continue/3`. It helps most
when the continuation repeats material the model has already seen:
agent loops re-sending transcripts, quoting or editing code,
structured output with recurring keys. The ngram table persists
across requests on the same model, so a re-sent transcript drafts
immediately on the next turn.

Tune it with a map instead of `true`:

```erlang
speculative => #{n_match => 24, n_max => 64, n_min => 48}
```

`n_match` is the window hashed for lookup, `n_max` caps the draft
length, and drafts shorter than `n_min` are discarded (defaults are
llama.cpp's). See [configuration](configuration.md#speculation) for
when the option is silently ignored.

For a two-model setup, the manual building blocks remain:

```erlang
{ok, Draft} = erllama:draft_tokens(SmallModel, PrefixTokens, #{max => 8}),
{ok, #{accepted := N, next := Next}} = erllama:verify(BigModel, PrefixTokens, Draft, 8).
```

`verify/4` runs one batched forward pass over the candidates,
accepts the longest prefix the verifier agrees with, and returns its
own next token (`eos` when the verifier would stop). The verifier's
KV state is restored before returning, so you can loop:
append the accepted tokens plus `Next`, draft again.

## Notes

- `response_tokens` defaults to a small budget; set it explicitly.
- Requests admit concurrently up to `context_opts.n_seq_max`
  sequences; beyond that they queue (`on_full => block`, the
  default) or fail fast (`on_full => error` gives
  `{error, seq_capacity}`).
- A wedged decode is bounded by `decode_budget_ms` (default 30 s)
  and surfaces `{error, decode_timeout}`; the engine recovers the
  context in place.
- `prefill_only/2,3` runs steps 1-2 without decoding: use it to warm
  the cache for a prefix you will complete later (see
  [caching](caching.md)).

## See also

- [Sessions](sessions.md) - keep KV state alive across turns
- [Tool calls](tool-calls.md) - chat templates, tools, JSON schemas
- [Configuration](configuration.md) - sampling reference, budgets
- [Examples](examples.md) - copy-paste recipes
