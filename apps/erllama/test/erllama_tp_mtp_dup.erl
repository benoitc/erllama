%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Test helper for erllama_tp_mtp_SUITE:mtp_batch_repeated_sampler,
%% loaded onto a node of one CUDA host: a model and its MTP head on that
%% host's GPU, two sequences prefilled, then an MTP round for both that
%% gives them the same sampler (refused: the round locks each item's
%% sampler, so a repeated one would lock its mutex twice while holding
%% the context's), then the same round with a sampler each.
-module(erllama_tp_mtp_dup).

-export([run/2]).

run(Path, MtpPath) ->
    MOpts = #{n_gpu_layers => 999, load_mode => direct_io},
    M = loaded(erllama_nif:load_model(Path, MOpts)),
    D = loaded(erllama_nif:load_model(MtpPath, MOpts#{load_mtp => true})),
    COpts = erllama_model_llama:family_context_opts(
        #{n_ctx => 1024, n_batch => 256, n_seq_max => 2, n_rs_seq => 2, flash_attn => enabled}, M
    ),
    {ok, C} = erllama_nif:new_context(M, COpts),
    {ok, Mtp} = erllama_nif:mtp_new(C, D, #{n_max => 1}),
    {ok, S0} = erllama_nif:sampler_new(C, #{temperature => 0.0}),
    {ok, S1} = erllama_nif:sampler_new(C, #{temperature => 0.0}),
    Tok = fun(T) -> erllama_nif:tokenize(M, T, #{add_special => true, parse_special => false}) end,
    {ok, _} = erllama_nif:step(C, [
        {0, {prefill, Tok(<<"The history of ocean navigation begins with">>)}},
        {1, {prefill, Tok(<<"A modern operating system schedules its threads with">>)}}
    ]),
    Repeated =
        try erllama_nif:mtp_step_batch(C, Mtp, [{S0, 0, auto}, {S0, 1, auto}], 1) of
            R -> {returned, R}
        catch
            error:badarg -> badarg
        end,
    Next = erllama_nif:mtp_step_batch(C, Mtp, [{S0, 0, auto}, {S1, 1, auto}], 1),
    #{repeated => Repeated, next => Next}.

loaded({ok, X}) -> X;
loaded({ok, X, _}) -> X.
