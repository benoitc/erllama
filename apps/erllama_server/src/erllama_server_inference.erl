%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_inference).
-moduledoc false.
%% Inference work that is not a run, for erllama_server_api: embeddings
%% and token counts. An embedding occupies inference capacity, so it
%% holds one of the `max_active_runs' slots (erllama_server_runs) while
%% it runs; a token count takes none. Both hold their model through
%% erllama_server_models, which loads it on demand as for a run.

-export([embed/3, count_tokens/3]).

-spec embed(binary() | undefined, [binary() | [non_neg_integer()]], map()) ->
    {ok, #{model := binary(), vectors := [[float()]], prompt_tokens := non_neg_integer()}}
    | {error, term()}.
embed(Model, Inputs, Opts) ->
    case erllama_server_runs:hold_slot(self()) of
        {ok, Slot} ->
            try
                with_model(Model, maps:get(keep_alive, Opts, undefined), fun(Id) ->
                    embed_with(Id, Inputs)
                end)
            after
                erllama_server_runs:release_slot(Slot)
            end;
        {error, _} = E ->
            E
    end.

-spec count_tokens(binary() | undefined, [map()], map()) ->
    {ok, non_neg_integer()} | {error, term()}.
count_tokens(Model, Messages, Opts) ->
    with_model(Model, undefined, fun(Id) ->
        erllama_server_generate:count_tokens(Id, Messages, Opts)
    end).

embed_with(Id, Inputs) ->
    case erllama:embed_batch(Id, Inputs) of
        {ok, Vectors} ->
            N = lists:sum([n_tokens(Id, I) || I <- Inputs]),
            {ok, #{model => Id, vectors => Vectors, prompt_tokens => N}};
        {error, R} ->
            {error, {embedding_failed, R}}
    end.

%% Hold `Model' (loading it if needed) while `Fun' runs with its id.
with_model(Model, KeepAlive, Fun) ->
    case erllama_server_models:acquire(Model, self(), KeepAlive) of
        {ok, Id} ->
            try
                Fun(Id)
            after
                erllama_server_models:release(Id, self())
            end;
        {error, _} = E ->
            E
    end.

n_tokens(_Id, Tokens) when is_list(Tokens) ->
    length(Tokens);
n_tokens(Id, Text) ->
    case erllama:tokenize(Id, Text) of
        {ok, T} -> length(T);
        _ -> 0
    end.
