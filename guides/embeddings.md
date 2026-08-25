# Embeddings

Turning text into vectors with an embedding-capable model. You need
this page when you build retrieval, clustering, or semantic search on
top of erllama instead of a separate embedding service.

## Load with embeddings enabled

The context must be opened in embeddings mode:

```erlang
{ok, M} = erllama:load_model(#{
    model_path => "/srv/models/nomic-embed-text-v1.5.Q8_0.gguf",
    context_opts => #{embeddings => true, n_ctx => 2048}
}).
```

Use a model trained for embeddings. A chat model's vectors are not
meaningfully comparable; dedicated embedding GGUFs (nomic-embed,
bge, gte, e5, ...) are.

## Embed

```erlang
{ok, Vec} = erllama:embed(M, <<"the quick brown fox">>),
length(Vec).   %% the model's embedding dimension
```

`embed/2` accepts a binary (tokenized for you) or a token list. The
result is the sequence-pooled embedding as a float list.

Batch over several inputs in one call:

```erlang
{ok, Vecs} = erllama:embed_batch(M, [
    <<"first document">>,
    <<"second document">>,
    TokenList
]).
```

`embed_batch/2` preserves order, mixes binaries and token lists, and
stops at the first error.

## Cosine similarity

Vectors are not normalized for you:

```erlang
cosine(A, B) ->
    Dot = lists:sum(lists:zipwith(fun erlang:'*'/2, A, B)),
    NA = math:sqrt(lists:sum([X * X || X <- A])),
    NB = math:sqrt(lists:sum([X * X || X <- B])),
    Dot / (NA * NB).
```

## Notes

- Calling `embed/2` on a context without `embeddings => true`
  returns an error; generation calls on an embeddings context are
  equally not supported. Load two model instances when you need
  both (they share the cache infrastructure, not the weights).
- Inputs longer than `n_ctx` fail with `{error, context_overflow}`;
  chunk your documents first.
- The stub backend returns deterministic 16-dimensional vectors, so
  you can unit-test your retrieval plumbing without a model (see
  [testing](testing.md)).

## See also

- [Loading a model](loading.md) - context options
- [Testing](testing.md) - stub-backed embedding tests
