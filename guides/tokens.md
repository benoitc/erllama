# Tokens and vocabulary

Converting between text and token ids, inspecting the model's special
tokens, and assembling fill-in-the-middle prompts. You need this page
when you work below the text-level APIs: `stream/3` and `continue/3`
take token lists, and code-completion (FIM) prompts are built from
special token ids.

## Tokenize

```erlang
{ok, Tokens} = erllama:tokenize(M, <<"Hello world">>),
{ok, Tokens2} = erllama:tokenize(M, <<"<|im_start|>user">>, #{
    add_special => false,   %% no automatic BOS/EOS (default: true)
    parse_special => true   %% recognise special-token text (default: false)
}).
```

`add_special` controls whether the tokenizer prepends/appends the
model's configured BOS/EOS. `parse_special` lets marker text like
`<|im_start|>` map to its single special token instead of being
tokenized as plain characters; you want it when you hand-render chat
prompts (that is what `chat/3` uses internally).

## Detokenize

```erlang
{ok, Text} = erllama:detokenize(M, Tokens).
```

The arity-2 form drops special tokens from the output. It is also
the form the cache byte-keys are computed over, so its output is
stable across releases.

To see the special tokens, use the arity-3 form:

```erlang
{ok, WithMarkers} = erllama:detokenize(M, Tokens, #{unparse_special => true}),
{ok, Trimmed} = erllama:detokenize(M, Tokens, #{remove_special => true}).
```

`unparse_special` renders special/control tokens (`<|im_start|>`,
FIM markers, ...) into the text; `remove_special` strips a leading
BOS / trailing EOS on models configured to add them. Note the
arity-3 form uses llama.cpp's `llama_detokenize`, which on
SentencePiece models trims the synthetic leading space that the
arity-2 form preserves; do not expect byte equality between the two
forms.

## Vocabulary and special tokens

```erlang
{ok, V} = erllama:vocab_info(M),
#{n_vocab := N, bos := Bos, eos := Eos, eot := Eot,
  fim_pre := Pre, fim_suf := Suf, fim_mid := Mid} = V.
```

Every token key is an integer id or `undefined` when the model has
no such token. `add_bos` / `add_eos` tell you whether the tokenizer
inserts them automatically. Use `eot` (end-of-turn) when you build
multi-turn prompts by hand, and check `eos =/= undefined` before
relying on stop behavior.

## Fill-in-the-middle (infill)

Code models trained for FIM take the surrounding code as
`fim_pre ++ Prefix ++ fim_suf ++ Suffix ++ fim_mid` and generate the
middle:

```erlang
{ok, #{fim_pre := Pre, fim_suf := Suf, fim_mid := Mid}} = erllama:vocab_info(M),
{ok, PrefixT} = erllama:tokenize(M, <<"def add(a, b):\n    ">>, #{add_special => false}),
{ok, SuffixT} = erllama:tokenize(M, <<"\n\nprint(add(1, 2))">>, #{add_special => false}),
Prompt = [Pre | PrefixT] ++ [Suf | SuffixT] ++ [Mid],
{ok, Ref} = erllama:stream(M, Prompt, #{
    response_tokens => 64,
    infill => true,          %% llama.cpp's FIM-oriented final sampler
    stop_sequences => [<<"\n\n">>]
}),
{ok, #{reply := Middle}} = erllama:collect(Ref, 30000).
```

Skip the FIM path when `fim_pre` is `undefined`: the model was not
trained for it. `fim_rep` / `fim_sep` exist on models with
repository-level FIM formats; consult the model card for their
layout.

## Notes

- Token ids are model-specific; never mix ids across models.
- `tokenize/2` on a long input is bounded (inputs over the NIF's
  size cap return `{error, too_large}`).
- Round-tripping `tokenize -> detokenize` reproduces the text for
  plain content; special tokens and byte-fallback edge cases can
  differ by markers or a leading space, as described above.

## See also

- [Generating text](generation.md) - `stream/3` takes token lists
- [Sessions](sessions.md) - `continue/3` and transcript tokens
- [Configuration](configuration.md#sampling-reference) - the `infill` stage
