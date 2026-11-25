%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% A decode replaces every logits row of the context. A sequence the
%% decode did not hold keeps an index into rows that are gone: its next
%% sample answers `{error, stale_logits}' (before, it read another
%% sequence's row, or none and the process aborted). Re-decoding its
%% last token gives the row back, and the sequence then emits the token
%% it would have emitted had nothing else been decoded.
%%
%% Needs LLAMA_TEST_MODEL (a small GGUF, e.g. TinyLlama).
-module(erllama_stale_logits_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([a_decode_without_the_seq_leaves_its_row_stale/1]).
-export([another_seq_removed_keeps_the_row/1]).

all() ->
    [a_decode_without_the_seq_leaves_its_row_stale, another_seq_removed_keeps_the_row].

init_per_suite(Config) ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {skip, "LLAMA_TEST_MODEL is not set"};
        Path ->
            {ok, _} = application:ensure_all_started(erllama),
            [{model, Path} | Config]
    end.

end_per_suite(_Config) ->
    ok.

a_decode_without_the_seq_leaves_its_row_stale(Config) ->
    {ok, M} =
        case erllama_nif:load_model(?config(model, Config), #{n_gpu_layers => 0}) of
            {ok, X} -> {ok, X};
            {ok, X, _} -> {ok, X}
        end,
    try
        Tok = fun(T) ->
            erllama_nif:tokenize(M, T, #{add_special => true, parse_special => false})
        end,
        PA = Tok(<<"The history of ocean navigation begins with">>),
        PB = Tok(<<"A modern operating system schedules its threads with">>),
        %% Both sequences decoded together: the reference next tokens.
        {RefA, RefB} = with_context(M, fun(C, Sa, Sb) ->
            {ok, _} = erllama_nif:step(C, [{0, {prefill, PA}}, {1, {prefill, PB}}]),
            {ok, R} = erllama_nif:step(C, [{0, {decode, Sa}}, {1, {decode, Sb}}]),
            {token(0, R), token(1, R)}
        end),
        with_context(M, fun(C, Sa, Sb) ->
            {ok, _} = erllama_nif:step(C, [{0, {prefill, PA}}, {1, {prefill, PB}}]),
            %% Sequence 0 alone: sequence 1's row is gone.
            {ok, R0} = erllama_nif:step(C, [{0, {decode, Sa}}]),
            ?assertEqual(RefA, token(0, R0)),
            ?assertEqual({error, stale_logits}, erllama_nif:step(C, [{1, {decode, Sb}}])),
            %% Its last token decoded again gives the row back.
            ok = erllama_nif:kv_seq_rm(C, 1, length(PB) - 1, -1),
            {ok, _} = erllama_nif:step(C, [{1, {prefill, [lists:last(PB)]}}]),
            {ok, R1} = erllama_nif:step(C, [{1, {decode, Sb}}]),
            ?assertEqual(RefB, token(1, R1))
        end)
    after
        erllama_nif:free_model(M)
    end.

%% Two sequences decoded together, then the second one removed (its
%% request finished): the first one's row stays, and a speculative step
%% verifies a draft from it (before, the removal cleared the context's
%% decode_ready and the step answered no_logits, which stopped the
%% model).
another_seq_removed_keeps_the_row(Config) ->
    {ok, M} =
        case erllama_nif:load_model(?config(model, Config), #{n_gpu_layers => 0}) of
            {ok, X} -> {ok, X};
            {ok, X, _} -> {ok, X}
        end,
    try
        Tok = fun(T) ->
            erllama_nif:tokenize(M, T, #{add_special => true, parse_special => false})
        end,
        PA = Tok(<<"The history of ocean navigation begins with">>),
        PB = Tok(<<"A modern operating system schedules its threads with">>),
        with_context(M, fun(C, Sa, Sb) ->
            {ok, _} = erllama_nif:step(C, [{0, {prefill, PA}}, {1, {prefill, PB}}]),
            {ok, _} = erllama_nif:step(C, [{0, {decode, Sa}}, {1, {decode, Sb}}]),
            ok = erllama_nif:kv_seq_rm(C, 1, 0, -1),
            ?assertMatch(
                {ok, {spec, [_ | _], _, _}}, erllama_nif:spec_step(C, Sa, 0, [lists:last(PA)])
            )
        end)
    after
        erllama_nif:free_model(M)
    end.

with_context(M, Fun) ->
    {ok, C} = erllama_nif:new_context(M, #{n_ctx => 512, n_batch => 128, n_seq_max => 2}),
    try
        {ok, Sa} = erllama_nif:sampler_new(C, #{temperature => 0.0}),
        {ok, Sb} = erllama_nif:sampler_new(C, #{temperature => 0.0}),
        Fun(C, Sa, Sb)
    after
        erllama_nif:free_context(C)
    end.

token(Seq, Results) ->
    {Seq, {token, T, _}} = lists:keyfind(Seq, 1, Results),
    T.
