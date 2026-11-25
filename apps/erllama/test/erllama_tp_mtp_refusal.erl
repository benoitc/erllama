%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Test helper for erllama_tp_mtp_SUITE:mtp_batch_refusal_names_sequence,
%% loaded onto a node of one CUDA host: a model and its MTP head on that
%% host's GPU, two sequences prefilled, then batched MTP rounds with
%% mixed samplers. A drafting sequence with a sampler that is not greedy
%% is refused as `{not_greedy, Seq}', one with logprobs as
%% `{not_supported, Seq}', in either position; the refusals leave every
%% row as it was, so the next rounds, where only the greedy sequence
%% drafts, run and draft for it.
-module(erllama_tp_mtp_refusal).

-export([run/2]).

run(Path, MtpPath) ->
    MOpts = #{n_gpu_layers => 999, load_mode => direct_io},
    M = loaded(erllama_nif:load_model(Path, MOpts)),
    D = loaded(erllama_nif:load_model(MtpPath, MOpts#{load_mtp => true})),
    COpts = erllama_model_llama:family_context_opts(
        #{n_ctx => 1024, n_batch => 256, n_seq_max => 2, n_rs_seq => 2, flash_attn => enabled}, M
    ),
    {ok, C} = erllama_nif:new_context(M, COpts),
    {ok, Mtp} = erllama_nif:mtp_new(C, D, #{n_max => 2}),
    {ok, Greedy} = erllama_nif:sampler_new(C, #{temperature => 0.0}),
    {ok, Sampled} = erllama_nif:sampler_new(C, #{temperature => 0.8, seed => 7}),
    {ok, Logprobs} = erllama_nif:sampler_new(C, #{temperature => 0.0, logprobs => 4}),
    Tok = fun(T) -> erllama_nif:tokenize(M, T, #{add_special => true, parse_special => false}) end,
    {ok, _} = erllama_nif:step(C, [
        {0, {prefill, Tok(<<"The history of ocean navigation begins with">>)}},
        {1, {prefill, Tok(<<"A modern operating system schedules its threads with">>)}}
    ]),
    Round = fun(Items) -> erllama_nif:mtp_step_batch(C, Mtp, Items, 2) end,
    #{
        not_greedy_second => Round([{Greedy, 0, auto}, {Sampled, 1, auto}]),
        not_greedy_first => Round([{Sampled, 0, auto}, {Greedy, 1, auto}]),
        not_supported => Round([{Greedy, 0, auto}, {Logprobs, 1, []}]),
        next => Round([{Greedy, 0, auto}, {Sampled, 1, []}]),
        again => Round([{Greedy, 0, auto}, {Sampled, 1, []}])
    }.

loaded({ok, X}) -> X;
loaded({ok, X, _}) -> X.
