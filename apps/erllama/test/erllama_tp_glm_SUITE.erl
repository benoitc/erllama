%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Opt-in, on CUDA hosts (ERLLAMA_TP_HOSTS, see erllama_tp_nccl_SUITE),
%% with the hosts' GPUs free: GLM-5.3-Flash (a hybrid MoE with split
%% linear-attention state and mirrored MLA latent cache and indexer)
%% split over the group.
%%
%%   ERLLAMA_TP_GLM_REF   a GGUF of the model small enough for one host
%%                        (correctness: split against whole)
%%   ERLLAMA_TP_GLM       the GGUF to serve (four streams, the sweep)
%%
%% Paths are relative to the hosts' home (the first part of a split
%% GGUF). A case whose model is not set is skipped.
-module(erllama_tp_glm_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([split_matches_one_gpu/1, one_gpu_four_streams/1, four_streams/1, decode_sweep/1]).
-export([cache_after_reload/1]).

-define(NO_CACHE, #{
    min_tokens => infinity, cold_min_tokens => infinity, continued_interval => infinity
}).
%% Direct I/O: on a GB10 the page cache a buffered load leaves behind is
%% not reclaimed for CUDA, whose allocations can then fail although the
%% host shows the memory as available.
-define(MOPTS, #{n_gpu_layers => 999, load_mode => direct_io}).
-define(COPTS, #{n_ctx => 8192, n_batch => 512, flash_attn => enabled}).
%% The correctness case loads the small quant whole on one GB10 (about
%% 102 GB of the 117 GB it has free): a small context keeps it clear of
%% the host's memory limit. The split run uses the same.
%% n_rs_seq 0: the split backend cannot split the recurrent cache's
%% rollback snapshots (a permuted view of `cache_r_l*'), so a split
%% GLM keeps none; the reference keeps none either.
-define(REF_COPTS, #{n_ctx => 2048, n_batch => 512, flash_attn => enabled, n_rs_seq => 0}).

all() ->
    [split_matches_one_gpu, one_gpu_four_streams, four_streams, decode_sweep, cache_after_reload].

init_per_suite(Config) ->
    case erllama_tp_hosts:hosts() of
        false -> {skip, "set ERLLAMA_TP_HOSTS to the CUDA hosts"};
        Hosts -> [{hosts, Hosts} | Config]
    end.

end_per_suite(_Config) ->
    ok.

%% The small quant loaded whole on the first host, then split over the
%% group: the same decodes and the 1-token removal (steps/0 of
%% erllama_tp_glm_ref) give the same logits within the tolerance and
%% the same next token at every step; every rank makes the same
%% reductions.
split_matches_one_gpu(Config) ->
    Model = model("ERLLAMA_TP_GLM_REF"),
    [Host0 | _] = Hosts = ?config(hosts, Config),
    Cookie = cookie(),
    put(cookie, Cookie),
    {P, _} = erllama_tp_hosts:start_peer(name("erllama_glmref"), Host0, Cookie),
    ok = erllama_tp_hosts:load_module(P, erllama_tp_glm_ref),
    Prompt = peer:call(P, erllama_tp_glm_ref, tokenize, [
        erllama_tp_hosts:home(P, Model),
        <<"The quick brown fox jumps over the lazy dog. Then the dog">>
    ]),
    Ref =
        try
            {T, R} = timer:tc(
                fun() ->
                    peer:call(
                        P,
                        erllama_tp_glm_ref,
                        run,
                        [
                            erllama_tp_hosts:home(P, Model),
                            ?MOPTS,
                            ?REF_COPTS,
                            Prompt,
                            erllama_tp_glm_ref:steps(no_rollback)
                        ],
                        3600000
                    )
                end,
                millisecond
            ),
            {ok, Logits, Family} = R,
            ct:pal("one GB10, ~b ms: family ~p", [
                T, maps:with([arch, n_layer, recurrent, hybrid], Family)
            ]),
            Logits
        after
            peer:stop(P)
        end,
    erllama_tp_hosts:with_serving(Hosts, fun(S) ->
        #{rank_peers := Peers, rank_pids := Pids} = erllama_tp_hosts:form(Hosts, S, 1),
        Ranks = lists:zip(Peers, Pids),
        try
            split_against(Ranks, Model, Prompt, Ref)
        after
            erllama_tp_hosts:stop_ranks(Ranks),
            [catch peer:stop(Pr) || Pr <- Peers]
        end
    end).

split_against(Ranks, Model, Prompt, Ref) ->
    Loaded = erllama_tp_hosts:parallel([
        fun() ->
            peer:call(
                Pr,
                erllama_tp_rank,
                load,
                [R, erllama_tp_hosts:home(Pr, Model), {?MOPTS, ?REF_COPTS}],
                3600000
            )
        end
     || {Pr, R} <- Ranks
    ]),
    [ok = L || L <- Loaded],
    Decode = fun(Tokens) -> all_ranks(Ranks, decode, [Tokens]) end,
    %% Every rank's answer must agree; rank 0's is returned.
    Trim = fun(Keep) ->
        [First | Rest] = erllama_tp_hosts:parallel([
            fun() ->
                element(2, peer:call(Pr, erllama_tp_rank, seq_op, [R, {trim, 0, Keep}], 120000))
            end
         || {Pr, R} <- Ranks
        ]),
        [?assertEqual(First, X) || X <- Rest],
        First
    end,
    Split = erllama_tp_glm_ref:drive(
        erllama_tp_glm_ref:steps(no_rollback), Decode, Trim, Prompt
    ),
    %% The prompt's reductions are large enough to go through BF16;
    %% the decodes' (one token) are reduced in FP32.
    Names = [{prompt, bf16}, {next1, fp32}, {next2, fp32}, {next3, fp32}],
    [compare(N, A, B) || {N, A, B} <- lists:zip3(Names, Split, Ref)],
    %% Without snapshots a 1-token removal is refused, split or not.
    ?assertMatch({error, _}, Trim(length(Prompt) + 2)),
    Stats = [element(2, peer:call(Pr, erllama_tp_rank, stats, [R])) || {Pr, R} <- Ranks],
    [Sizes0 | Others] = [maps:get(sizes, St) || St <- Stats],
    ct:pal("reductions per rank: ~w; reserved for the other rank: ~w GB", [
        [maps:get(reductions, St) || St <- Stats],
        [maps:get(remote_bytes, St) / 1.0e9 || St <- Stats]
    ]),
    [?assertEqual(Sizes0, Sz) || Sz <- Others],
    ok.

%% The small quant served whole on one GB10 (no split): the same four
%% prompts alone and at once. Whether batching alone changes greedy
%% replies there tells what four_streams can demand of the split model.
one_gpu_four_streams(Config) ->
    Model = model("ERLLAMA_TP_GLM_REF"),
    [Host0 | _] = ?config(hosts, Config),
    {P, _} = erllama_tp_hosts:start_peer(name("erllama_glm1"), Host0, cookie()),
    try
        ok = erllama_tp_hosts:load_module(P, erllama_tp_stream),
        {ok, Id} = peer:call(
            P,
            erllama,
            load_model,
            [
                <<"glm_one_gpu">>,
                #{
                    backend => erllama_model_llama,
                    model_path => erllama_tp_hosts:home(P, Model),
                    model_opts => ?MOPTS,
                    context_opts => maps:merge(?REF_COPTS, #{n_seq_max => 4}),
                    policy => ?NO_CACHE
                }
            ],
            3600000
        ),
        G = #{id => Id},
        Prompts = four_prompts(P, G),
        Solo = [run(P, G, Pr, 32) || Pr <- Prompts],
        Many = [maps:get(runs, run_many(P, G, Prompts, 32, {none, 0})) || _ <- [1, 2]],
        report_batching("one GB10", Solo, Many)
    after
        catch peer:stop(P)
    end.

%% Four prompts of different lengths streamed at once through
%% erllama_model_tp: batched replies repeat exactly; against each prompt
%% streamed alone, where they first differ is reported (compare with
%% one_gpu_four_streams); a cancel midway ends that request only.
four_streams(Config) ->
    serve_glm(Config, fun(S, G) ->
        Prompts = four_prompts(S, G),
        Solo = [run(S, G, P, 32) || P <- Prompts],
        [?assertMatch(#{tokens := 32}, R) || R <- Solo],
        [M1, M2] = [run_many(S, G, Prompts, 32, {none, 0}) || _ <- [1, 2]],
        ct:pal("four at once: ~.1f tokens/s over ~.1f ms", [
            maps:get(tokens, M1) / maps:get(wall_ms, M1) * 1000, maps:get(wall_ms, M1)
        ]),
        report_batching("split over two GB10s", Solo, [maps:get(runs, M1), maps:get(runs, M2)]),
        %% A cancel midway: the request ends cancelled, the others finish.
        Cut = maps:get(runs, run_many(S, G, Prompts, 32, {2, 8})),
        ?assertMatch(#{ended := cancelled, tokens := 8}, lists:nth(2, Cut)),
        [?assertMatch(#{tokens := 32}, lists:nth(I, Cut)) || I <- [1, 3, 4]]
    end).

four_prompts(Node, G) ->
    [
        chat(Node, G, T)
     || T <- [
            <<"Name three colors.">>,
            <<"Explain in two sentences why the sky is blue.">>,
            erllama_tp_hosts:long_prompt(15),
            erllama_tp_hosts:long_prompt(40)
        ]
    ].

%% Two batched runs and the solo runs, compared: whether each batched
%% reply repeats, and how many bytes it shares with its solo reply. A
%% greedy reply follows the arithmetic, and the batch a step computes
%% depends on when each request was admitted, so batched replies need
%% not repeat; what matters is that the split model behaves like one GPU.
report_batching(Where, Solo, [Many1, Many2]) ->
    Rows = [
        {I, maps:get(text, A) =:= maps:get(text, B),
            common_prefix(maps:get(text, A), maps:get(text, B)),
            common_prefix(maps:get(text, A), maps:get(text, C)),
            byte_size(maps:get(text, C))}
     || {I, C, {A, B}} <- lists:zip3(lists:seq(1, 4), Solo, lists:zip(Many1, Many2))
    ],
    ct:pal(
        "~s, {stream, batched repeats, common bytes batched run 1 vs 2, common bytes batched vs solo, solo bytes}:~n~p",
        [Where, Rows]
    ),
    %% The solo replies, to compare one GPU with the split model.
    [
        ct:pal("~s, solo ~b: sha256 ~s ~p", [
            Where,
            I,
            binary:encode_hex(crypto:hash(sha256, maps:get(text, R))),
            binary:part(maps:get(text, R), 0, min(60, byte_size(maps:get(text, R))))
        ])
     || {I, R} <- lists:enumerate(Solo)
    ],
    Rows.

common_prefix(A, B) -> binary:longest_common_prefix([A, B]).

%% TensorFold's sweep (docs/dec.py of its recipe) on the same pair: its
%% four fixed prompts through the chat template, thinking off, greedy,
%% 250 tokens, 1 stream then 4, three trials each. Decode rate is taken
%% after the first token (dec.py removes prefill the same way).
decode_sweep(Config) ->
    serve_glm(Config, fun(S, G) ->
        Prompts = [chat(S, G, T) || T <- dec_prompts()],
        _ = run(S, G, hd(Prompts), 16),
        One = [run(S, G, hd(Prompts), 250) || _ <- lists:seq(1, 3)],
        [
            ct:pal(
                "1 stream, trial ~b: first token ~.1f ms, ~.1f tokens/s, median gap ~.2f ms, ~b tokens, ~p",
                [
                    I,
                    maps:get(ttft_ms, R),
                    rate(R),
                    median(maps:get(gaps_ms, R)),
                    maps:get(tokens, R),
                    maps:get(ended, R)
                ]
            )
         || {I, R} <- lists:enumerate(One)
        ],
        Four = [run_many(S, G, Prompts, 250, {none, 0}) || _ <- lists:seq(1, 3)],
        [
            ct:pal(
                "4 streams, trial ~b: ~.1f tokens/s total (wall), ~.1f per stream after the first token",
                [
                    I,
                    maps:get(tokens, M) / maps:get(wall_ms, M) * 1000,
                    lists:sum([rate(R) || R <- maps:get(runs, M)]) / 4
                ]
            )
         || {I, M} <- lists:enumerate(Four)
        ],
        [?assert(maps:get(tokens, R) > 0) || R <- One]
    end).

dec_prompts() ->
    [
        <<"Write five detailed paragraphs about the history of ocean navigation.">>,
        <<"Explain how a modern operating system schedules threads. Be thorough and use several paragraphs.">>,
        <<"Describe the engineering of suspension bridges in five paragraphs.">>,
        <<"Write five paragraphs on the development of the printing press and its consequences.">>
    ].

chat(S, #{id := Id}, Text) ->
    peer:call(S, erllama_tp_stream, chat_tokens, [Id, Text], 60000).

%% A request with the cache on, then the group reloaded and the same
%% request again: what a split GLM does with its checkpoint. Without
%% rollback snapshots the warm restore's 1-token removal is refused, so
%% the expected answer is a cold serve with the same tokens; the case
%% records the hit kind and the checkpoint stats either way.
cache_after_reload(Config) ->
    Hosts = ?config(hosts, Config),
    os:putenv("ERLLAMA_TP_MODEL", model("ERLLAMA_TP_GLM")),
    Policy = #{
        max_ladder_rows => 0,
        min_tokens => 64,
        cold_min_tokens => 128,
        cold_max_tokens => 1536,
        boundary_trim_tokens => 32,
        boundary_align_tokens => 64,
        continued_interval => infinity
    },
    Extra = #{model_opts => ?MOPTS, context_opts => serve_copts(), tp_call_timeout => 3600000},
    Prompt = erllama_tp_hosts:long_prompt(30),
    erllama_tp_hosts:with_serving(Hosts, fun(S) ->
        G1 = erllama_tp_hosts:serve(Hosts, S, 1, Policy, Extra),
        {T1, St1} = erllama_tp_hosts:complete_stats(S, G1, Prompt),
        Ck1 = peer:call(S, erllama_tp_rank, checkpoint_stats, [hd(maps:get(rank_pids, G1))]),
        ok = erllama_tp_hosts:stop_group(G1),
        G2 = erllama_tp_hosts:serve(Hosts, S, 2, Policy, Extra),
        {T2, St2} = erllama_tp_hosts:complete_stats(S, G2, Prompt),
        Ck2 = peer:call(S, erllama_tp_rank, checkpoint_stats, [hd(maps:get(rank_pids, G2))]),
        ok = erllama_tp_hosts:stop_group(G2),
        Keys = [cache_hit_kind, cache_delta, prompt_tokens, prefill_ms],
        ct:pal(
            "first group: ~p~ncheckpoints: ~p~nsecond group: ~p~nrestores: ~p~nsame tokens: ~p", [
                maps:with(Keys, St1), Ck1, maps:with(Keys, St2), Ck2, T1 =:= T2
            ]
        ),
        ?assertEqual(T1, T2)
    end).

%% -----------------------------------------------------------------------------

serve_glm(Config, Fun) ->
    Hosts = ?config(hosts, Config),
    os:putenv("ERLLAMA_TP_MODEL", model("ERLLAMA_TP_GLM")),
    erllama_tp_hosts:with_serving(Hosts, fun(S) ->
        G0 = erllama_tp_hosts:form(Hosts, S, 1),
        Peers = maps:get(rank_peers, G0),
        Before = [avail(P) || P <- Peers],
        G = erllama_tp_hosts:load(
            G0,
            ?NO_CACHE,
            #{model_opts => ?MOPTS, context_opts => serve_copts(), tp_call_timeout => 3600000},
            3600000
        ),
        After = [avail(P) || P <- Peers],
        ct:pal("loaded: memory available per rank ~w GB before, ~w GB after", [
            [round(B / 1.0e9) || B <- Before], [round(A / 1.0e9) || A <- After]
        ]),
        try
            Fun(S, G)
        after
            erllama_tp_hosts:stop_group(G)
        end
    end).

%% ERLLAMA_TP_GLM_SEQ (default 4) sequences; ERLLAMA_TP_GLM_UNIFIED=1
%% puts them in one KV stream.
serve_copts() ->
    Seq = list_to_integer(os:getenv("ERLLAMA_TP_GLM_SEQ", "4")),
    Unified = os:getenv("ERLLAMA_TP_GLM_UNIFIED") =:= "1",
    C0 = maps:put(n_seq_max, Seq, ?COPTS),
    %% ERLLAMA_TP_GLM_RS sets n_rs_seq (the recurrent-state rollback
    %% snapshots); unset, erllama's default for hybrid models applies.
    C =
        case os:getenv("ERLLAMA_TP_GLM_RS") of
            false -> C0;
            Rs -> maps:put(n_rs_seq, list_to_integer(Rs), C0)
        end,
    case Unified of
        true -> maps:put(kv_unified, true, C);
        false -> C
    end.

model(Var) ->
    case os:getenv(Var) of
        false -> exit({skip, Var ++ " is not set"});
        M -> M
    end.

run(S, #{id := Id}, Prompt, Max) ->
    peer:call(S, erllama_tp_stream, run, [Id, Prompt, Max, infinity], 3600000).

run_many(S, #{id := Id}, Prompts, Max, Cancel) ->
    peer:call(S, erllama_tp_stream, run_many, [Id, Prompts, Max, Cancel], 3600000).

%% Every rank makes the call; they must agree, and rank 0's is returned.
all_ranks(Ranks, Fn, Args) ->
    [First | Rest] = erllama_tp_hosts:parallel([
        fun() -> peer:call(Pr, erllama_tp_rank, Fn, [R | Args], 600000) end
     || {Pr, R} <- Ranks
    ]),
    [?assertEqual(First, X) || X <- Rest],
    First.

%% Same next token at every step, and at least 4 of the top 5. The
%% difference is reported, not bounded: the prompt's reductions go
%% through BF16, and every later step builds on that prompt's cache.
compare({What, Precision}, Split, Ref) ->
    S = f32s(Split),
    R = f32s(Ref),
    MaxDiff = lists:max([abs(A - B) || {A, B} <- lists:zip(S, R)]),
    Scale = lists:max([abs(B) || B <- R]),
    Top = fun(L) ->
        [
            I
         || {_, I} <- lists:sublist(
                lists:reverse(lists:sort(lists:zip(L, lists:seq(0, length(L) - 1)))), 5
            )
        ]
    end,
    {TopS, TopR} = {Top(S), Top(R)},
    ct:pal(
        "~p (~p reductions): max |split - one GPU| = ~.4g (~.2f% of the largest |logit| ~.4g); top 5 ~w vs ~w",
        [
            What, Precision, MaxDiff, MaxDiff / Scale * 100, Scale, TopS, TopR
        ]
    ),
    ?assertEqual(hd(TopR), hd(TopS), What),
    ?assert(length([I || I <- TopS, lists:member(I, TopR)]) >= 4, {What, TopS, TopR}).

f32s(Bin) -> [V || <<V:32/float-native>> <= Bin].

avail(P) ->
    {ok, Bin} = peer:call(P, file, read_file, ["/proc/meminfo"]),
    {match, [Kb]} = re:run(Bin, "MemAvailable:\\s+([0-9]+) kB", [{capture, all_but_first, binary}]),
    binary_to_integer(Kb) * 1024.

rate(#{gaps_ms := []}) -> 0.0;
rate(#{gaps_ms := Gaps}) -> length(Gaps) / lists:sum(Gaps) * 1000.

median([]) -> 0.0;
median(L) -> lists:nth((length(L) + 1) div 2, lists:sort(L)).

cookie() ->
    "erllama_tp_" ++ integer_to_list(erlang:unique_integer([positive])).

name(Base) ->
    list_to_atom(Base ++ "_" ++ integer_to_list(erlang:unique_integer([positive]))).
