%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Test helper for erllama_tp_glm_SUITE, loaded onto a node of one host:
%% the reference run of a model loaded whole on that host's GPU. It
%% makes the same sequence of decodes and KV removals the suite makes on
%% the split model, and returns the logits after each.
-module(erllama_tp_glm_ref).

-export([run/4, run/5, steps/0, steps/1, drive/4, argmax/1, tokenize/2, family/1]).

%% The sequence, on sequence 0: the prompt; one token; that token removed
%% (a 1-token removal, as a warm restore's primer or a rejected draft
%% does); the same token again; one more token.
steps() ->
    [prompt, next, {trim, 1}, again, next].

%% Without recurrent-state snapshots (n_rs_seq 0, which a split hybrid
%% model needs), a 1-token removal is refused: decodes only.
steps(no_rollback) ->
    [prompt, next, next, next].

%% The model's family (arch, n_layer, ...) from its GGUF header (no
%% weights loaded).
family(Path) ->
    {ok, V} = erllama_nif:load_model(Path, #{vocab_only => true}),
    try
        erllama_nif:model_family(V)
    after
        erllama_nif:free_model(V)
    end.

%% `Text' as token ids in the model's vocabulary (no weights loaded).
tokenize(Path, Text) ->
    {ok, V} = erllama_nif:load_model(Path, #{vocab_only => true}),
    try
        erllama_nif:tokenize(V, Text, #{add_special => true, parse_special => false})
    after
        erllama_nif:free_model(V)
    end.

%% Load `Path' whole (every layer on the GPU), run steps/0 from
%% `Prompt' (token ids), free the model. Returns the logits after each
%% decode step, as native F32 binaries, and the model's family.
run(Path, MOpts, COpts0, Prompt) ->
    run(Path, MOpts, COpts0, Prompt, steps()).

run(Path, MOpts, COpts0, Prompt, Steps) ->
    case erllama_nif:load_model(Path, MOpts) of
        {ok, M} -> run_loaded(M, COpts0, Prompt, Steps);
        {ok, M, _} -> run_loaded(M, COpts0, Prompt, Steps);
        {error, _} = E -> E
    end.

run_loaded(M, COpts0, Prompt, Steps) ->
    COpts = erllama_model_llama:family_context_opts(COpts0, M),
    Family = erllama_nif:model_family(M),
    try
        {ok, C} = erllama_nif:new_context(M, COpts),
        try
            {ok,
                drive(
                    Steps,
                    fun(T) -> erllama_nif:tp_decode(C, T) end,
                    fun(Keep) -> erllama_nif:kv_seq_rm(C, 0, Keep, -1) end,
                    Prompt
                ),
                Family}
        after
            erllama_nif:free_context(C)
        end
    after
        erllama_nif:free_model(M)
    end.

%% Run `Steps' with `Decode(Tokens) -> {ok, Logits}' and `Trim(Keep)':
%% the logits after each decode, in order. The split model's run in the
%% suite goes through the same function.
drive(Steps, Decode, Trim, Prompt) ->
    drive(
        Steps,
        Decode,
        Trim,
        #{prompt => Prompt, len => 0, logits => undefined, tok => undefined},
        []
    ).

drive([], _Decode, _Trim, _St, Acc) ->
    lists:reverse(Acc);
drive([prompt | Rest], Decode, Trim, #{prompt := P} = St, Acc) ->
    {ok, L} = Decode(P),
    drive(Rest, Decode, Trim, St#{len := length(P), logits := L}, [L | Acc]);
drive([next | Rest], Decode, Trim, #{len := N, logits := Last} = St, Acc) ->
    T = argmax(Last),
    {ok, L} = Decode([T]),
    drive(Rest, Decode, Trim, St#{len := N + 1, logits := L, tok := T}, [L | Acc]);
drive([{trim, K} | Rest], Decode, Trim, #{len := N} = St, Acc) ->
    ok = Trim(N - K),
    drive(Rest, Decode, Trim, St#{len := N - K}, Acc);
drive([again | Rest], Decode, Trim, #{len := N, tok := T} = St, Acc) ->
    {ok, L} = Decode([T]),
    drive(Rest, Decode, Trim, St#{len := N + 1, logits := L}, [L | Acc]).

argmax(Bin) ->
    {_, I, _} = lists:foldl(
        fun(V, {Best, BI, J}) ->
            case V > Best of
                true -> {V, J, J + 1};
                false -> {Best, BI, J + 1}
            end
        end,
        {-1.0e38, 0, 0},
        [V || <<V:32/float-native>> <= Bin]
    ),
    I.
