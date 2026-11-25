%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Opt-in, on CUDA hosts (ERLLAMA_TP_HOSTS) with their GPUs free: MTP
%% speculative decoding of GLM-5.3-Flash split over the group. Rank 0
%% drafts two tokens a round with the model's MTP head on its own GPU;
%% the round is one 3-row batch on every rank, and a rejected draft is
%% taken back with the recurrent-state snapshots (n_rs_seq 2 on every
%% rank).
%%
%%   ERLLAMA_TP_GLM      the GGUF to serve (the first part of a split one)
%%   ERLLAMA_TP_GLM_MTP  the companion MTP GGUF, on the first host
%%   ERLLAMA_TP_MTP_SPLIT  optional shares of the model per rank (e.g.
%%                         "0.46,0.54"), to leave rank 0 room for the
%%                         MTP head
%%
%% Paths are relative to the hosts' home.
-module(erllama_tp_mtp_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([mtp_batch/1, mtp_scheduler/1, server_bench/1]).
-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([mtp_decode/1, divergence_margin/1, rollback_state_matches/1]).

-define(NO_CACHE, #{
    min_tokens => infinity, cold_min_tokens => infinity, continued_interval => infinity
}).
-define(MOPTS, #{n_gpu_layers => 999, load_mode => direct_io}).
-define(COPTS, #{
    n_ctx => 4096,
    n_batch => 1024,
    flash_attn => enabled,
    n_rs_seq => 2,
    n_seq_max => 2,
    kv_unified => true
}).
-define(MTP, #{speculative => #{type => mtp, n_max => 2}}).
%% mtp_scheduler: a token's lead (top logprob minus the next) under which
%% a difference is numerics; the tokens each request must agree on.
-define(CLEAR_LEAD, 1.0).
-define(MIN_SAME, 4).
-define(TOKENS, 250).

all() ->
    [mtp_decode, divergence_margin, rollback_state_matches, mtp_batch, mtp_scheduler, server_bench].

init_per_suite(Config) ->
    case {erllama_tp_hosts:hosts(), os:getenv("ERLLAMA_TP_GLM"), os:getenv("ERLLAMA_TP_GLM_MTP")} of
        {false, _, _} -> {skip, "set ERLLAMA_TP_HOSTS to the CUDA hosts"};
        {_, false, _} -> {skip, "set ERLLAMA_TP_GLM to a GLM-5.3-Flash GGUF"};
        {_, _, false} -> {skip, "set ERLLAMA_TP_GLM_MTP to its MTP GGUF"};
        {Hosts, Model, Mtp} -> [{hosts, Hosts}, {model, Model}, {mtp, Mtp} | Config]
    end.

end_per_suite(_Config) ->
    ok.

%% TensorFold's four fixed prompts (docs/dec.py of its recipe) through
%% the chat template, thinking off, greedy, 250 tokens, one stream:
%% each one without drafting, then with MTP, in the same context. The
%% decode rate is taken after the first token, as for the ordinary
%% decoding figures. Reports acceptance, draft, verify and rollback
%% time a round, and whether the drafter's context ended where the
%% target's does after every round (it resumes from the accepted
%% position). Then one erllama:complete with and without MTP.
mtp_decode(Config) ->
    serve(Config, fun(S, G, R0) ->
        Prompts = [chat(S, G, T) || T <- dec_prompts()],
        _ = run(S, G, hd(Prompts), 16, #{}),
        _ = run(S, G, hd(Prompts), 16, ?MTP),
        Rows = [prompt_case(S, G, R0, I, P) || {I, P} <- lists:enumerate(Prompts)],
        Plain = [maps:get(plain, R) || R <- Rows],
        Mtp = [maps:get(mtp, R) || R <- Rows],
        ct:pal(
            "over the four prompts: ordinary ~.1f tokens/s, MTP ~.1f tokens/s (mean of the per-prompt rates)",
            [mean(Plain), mean(Mtp)]
        ),
        [?assertEqual(0, maps:get(misaligned, R)) || R <- Rows],
        complete_case(S, G),
        ok
    end).

prompt_case(S, G, R0, I, Prompt) ->
    A = run(S, G, Prompt, ?TOKENS, #{}),
    K0 = mtp_stats(S, R0),
    B = run(S, G, Prompt, ?TOKENS, ?MTP),
    K = diff(mtp_stats(S, R0), K0),
    #{rounds := Rounds, drafted := Drafted, accepted := Acc} = K,
    Same = binary:longest_common_prefix([maps:get(text, A), maps:get(text, B)]),
    PerRound = fun(Key) -> maps:get(Key, K) / max(1, Rounds) / 1000 end,
    ct:pal(
        "prompt ~b: ordinary ~.1f tokens/s (first token ~.1f ms); MTP ~.1f tokens/s (first token ~.1f ms), ~b tokens, ~p~n"
        "  rounds ~b, drafted ~b, accepted ~b (~.3f a draft, ~.2f tokens a round); after a rejection ~b rounds accepted ~b (~.3f a draft)~n"
        "  a round: draft ~.2f ms, verify ~.2f ms (of which the drafter's catch-up ~.2f ms), rollback ~.2f ms; misaligned ~b, catch-up failures ~b~n"
        "  request stats ~p; same text for ~b of ~b bytes",
        [
            I,
            rate(A),
            maps:get(ttft_ms, A),
            rate(B),
            maps:get(ttft_ms, B),
            maps:get(tokens, B),
            maps:get(ended, B),
            Rounds,
            Drafted,
            Acc,
            Acc / max(1, Drafted),
            (Rounds + Acc) / max(1, Rounds),
            maps:get(after_reject_rounds, K),
            maps:get(after_reject_accepted, K),
            maps:get(after_reject_accepted, K) / max(1, 2 * maps:get(after_reject_rounds, K)),
            PerRound(draft_us),
            PerRound(verify_us),
            maps:get(process_us, K) / max(1, maps:get(process_calls, K)) / 1000,
            PerRound(rollback_us),
            maps:get(misaligned, K),
            maps:get(process_failures, K),
            maps:get(speculative, maps:get(stats, B, #{}), none),
            Same,
            byte_size(maps:get(text, A))
        ]
    ),
    ?assert(Rounds > 0),
    #{plain => rate(A), mtp => rate(B), misaligned => maps:get(misaligned, K)}.

%% erllama:complete, the plain-text path, with and without MTP.
complete_case(S, #{id := Id}) ->
    Prompt = <<"The history of ocean navigation begins with">>,
    Opts = #{response_tokens => 64, temperature => 0.0},
    {ok, #{generated := A}} = peer:call(S, erllama, complete, [Id, Prompt, Opts], 600000),
    {ok, #{generated := B} = St} = peer:call(
        S, erllama, complete, [Id, Prompt, maps:merge(Opts, ?MTP)], 600000
    ),
    Common = length(lists:takewhile(fun({X, Y}) -> X =:= Y end, lists:zip(A, B, trim))),
    ct:pal("erllama:complete, 64 tokens: ~b of ~b tokens the same with MTP; ~p", [
        Common, length(A), maps:get(speculative, St, none)
    ]),
    ?assertMatch(#{drafted := D} when D > 0, maps:get(speculative, St)).

%% Where the MTP output first differs from ordinary decoding, the top
%% two candidates of the ordinary run at that token: a small margin and
%% the MTP token being the runner-up says the 3-row verify batch only
%% rounds differently from a 1-row decode (as batching does on one GPU
%% too); a large margin says a rollback went wrong.
divergence_margin(Config) ->
    serve(Config, fun(S, #{id := Id}, _R0) ->
        Prompts = [<<"The history of ocean navigation begins with">> | dec_prompts()],
        Opts = #{response_tokens => 250, temperature => 0.0},
        Rows = [
            begin
                {ok, #{generated := A, logprobs := Lps}} =
                    peer:call(S, erllama, complete, [Id, P, Opts#{logprobs => 2}], 600000),
                {ok, #{generated := B}} =
                    peer:call(S, erllama, complete, [Id, P, maps:merge(Opts, ?MTP)], 600000),
                N = length(lists:takewhile(fun({X, Y}) -> X =:= Y end, lists:zip(A, B, trim))),
                case N < min(length(A), length(B)) of
                    true ->
                        #{top := Top, logprob := Lp} = lists:nth(N + 1, Lps),
                        {I, N, lists:nth(N + 1, A), lists:nth(N + 1, B), Lp, Top};
                    false ->
                        {I, N, same}
                end
            end
         || {I, P} <- lists:enumerate(Prompts)
        ],
        ct:pal(
            "{prompt, tokens the same, ordinary token, MTP token, its logprob, the ordinary run's top two}:~n~p",
            [Rows]
        )
    end).

%% Matching positions do not show that a rollback restored the right
%% state. For the round deciding each prompt's first divergence from
%% ordinary decoding, and the first rounds after a rejection: rebuild
%% the exact prefix the live MTP run had before the round in a fresh
%% group's context (one prefill), decode the same tokens in one batch
%% with every row out, and compare every row with the live run's (its
%% greedy choice and the logits of the live top 8). The batch shape and
%% row are the live ones, so only the state the rounds before left can
%% differ (and the prefix's own rounding: one prefill against rounds).
%% Tolerances, set before the run: a fresh greedy choice may differ
%% only where the live top two are within 0.5; the top-8 logits within
%% 2.0 (the largest difference seen between batch shapes was 1.62).
rollback_state_matches(Config) ->
    Live = serve(Config, fun(S, G, R0) ->
        Prompts = [chat(S, G, T) || T <- dec_prompts()],
        %% The drafter exists from the first MTP request.
        _ = run(S, G, hd(Prompts), 16, ?MTP),
        [live_rounds(S, G, R0, P) || P <- Prompts]
    end),
    Checks = with_group(Config, fun(G) ->
        lists:append([[replay(G, P, B, T) || T <- Targets] || {P, B, Targets} <- Live])
    end),
    ct:pal(
        "{why, after a rejection, position, row, live / fresh choice, live top-two margin, max |diff| over the live top 8}:~n~p",
        [Checks]
    ),
    [
        begin
            ?assert(Diff =< 2.0),
            ?assert(LiveTok =:= FreshTok orelse Margin =< 0.5)
        end
     || {_, _, _, _, {LiveTok, FreshTok}, Margin, Diff} <- Checks
    ],
    ?assert(lists:any(fun({_, After, _, _, _, _, _}) -> After end, Checks)).

%% The prompt, the MTP run's tokens and the rounds to check: the one
%% whose row chose the first token that differs from ordinary decoding,
%% and the first three that follow a rejected draft.
live_rounds(S, G, R0, P) ->
    #{stats := #{generated := A}} = run(S, G, P, ?TOKENS, #{}),
    [ok] = trace(S, R0, on),
    #{stats := #{generated := B}} = run(S, G, P, ?TOKENS, ?MTP),
    [Rounds] = trace(S, R0, read),
    [ok] = trace(S, R0, off),
    L = length(P),
    Indexed = lists:zip([undefined | lists:droplast(Rounds)], Rounds),
    Rejected = fun
        (undefined) -> false;
        (#{tokens := Ts, n_acc := Kept}) -> Kept < length(Ts) - 1
    end,
    N = length(lists:takewhile(fun({X, Y}) -> X =:= Y end, lists:zip(A, B, trim))),
    Div = [
        {divergence, Rejected(Prev), R, L + N - 1 - Pos}
     || N < min(length(A), length(B)),
        {Prev, #{pos := Pos, n_acc := Acc} = R} <- Indexed,
        Pos =< L + N - 1,
        L + N - 1 =< Pos + Acc
    ],
    After = lists:sublist(
        [{after_rejection, true, R, Acc} || {Prev, #{n_acc := Acc} = R} <- Indexed, Rejected(Prev)],
        3
    ),
    {P, B, Div ++ After}.

replay(
    G, P, B, {Why, AfterRej, #{pos := Pos, tokens := Toks, n_acc := Acc, rows := LiveRows}, Row}
) ->
    L = length(P),
    %% The round's kept tokens are the MTP run's at its position.
    ?assertEqual(lists:sublist(Toks, Acc + 1), lists:sublist(B, Pos - L + 1, Acc + 1)),
    {_, ok} = op(G, {drop, 0}),
    {_, {ok, _}} = op(G, {decode, 0, P ++ lists:sublist(B, Pos - L), false}),
    {_, {ok, Bin}} = op(G, {decode, 0, Toks, true}),
    Fresh = lists:nth(Row + 1, rows(Bin, length(Toks))),
    Live = lists:nth(Row + 1, LiveRows),
    [{LiveTok, V1}, {_, V2} | _] = Live,
    Diff = lists:max([abs(V - logit(Fresh, Id)) || {Id, V} <- Live]),
    {Why, AfterRej, Pos + Row, Row, {LiveTok, argmax(Fresh)}, V1 - V2, Diff}.

trace(S, R0, Op) ->
    peer:call(S, gen_server, call, [R0, {hosting, {mtp, Op}}, 60000], 70000).

with_group(Config, Fun) ->
    Hosts = ?config(hosts, Config),
    Model = ?config(model, Config),
    erllama_tp_hosts:with_serving(Hosts, fun(S) ->
        #{rank_peers := Peers, rank_pids := Pids} = erllama_tp_hosts:form(Hosts, S, 2),
        Ranks = lists:zip(Peers, Pids),
        try
            [
                ok = R
             || R <- erllama_tp_hosts:parallel([
                    fun() ->
                        peer:call(
                            P,
                            erllama_tp_rank,
                            load,
                            [R, erllama_tp_hosts:home(P, Model), {model_opts(), ?COPTS}],
                            3600000
                        )
                    end
                 || {P, R} <- Ranks
                ])
            ],
            Fun(#{ranks => Ranks})
        after
            erllama_tp_hosts:stop_ranks(Ranks),
            [catch peer:stop(P) || P <- Peers]
        end
    end).

%% Every rank makes the call and must answer the same.
op(#{ranks := Ranks}, Op) ->
    Res = erllama_tp_hosts:parallel([
        fun() -> peer:call(P, erllama_tp_rank, seq_op, [R, Op], 600000) end
     || {P, R} <- Ranks
    ]),
    Norm = [
        case X of
            {Ms, {ok, Bin, _, _}} -> {Ms, {ok, Bin}};
            Other -> Other
        end
     || X <- Res
    ],
    [{_, First} | _] = Norm,
    [?assertEqual(First, Y, Op) || {_, Y} <- Norm],
    {lists:max([Ms || {Ms, _} <- Norm]), First}.

rows(Bin, K) ->
    Sz = byte_size(Bin) div K,
    [binary:part(Bin, I * Sz, Sz) || I <- lists:seq(0, K - 1)].

logit(Row, Id) ->
    <<V:32/float-native>> = binary:part(Row, Id * 4, 4),
    V.

argmax(Bin) -> erllama_tp_glm_ref:argmax(Bin).

%% Prototype: four greedy requests, one MTP guess each, every request's
%% token and guess verified in one 8-row batch and kept or taken back on
%% its own (erllama_tp_mtp_bench, on the serving node, calling rank 0 as
%% the scheduler does). First correctness: one request forced to accept
%% and another to reject in the same round, against ordinary decoding;
%% the rejected one's next token must be the reference's; a request
%% cancelled mid-generation while the other three go on; the checkpoint
%% regression (regression/3), one request alone and four. Then speed:
%% ordinary co-batched steps against batched MTP rounds, 250 tokens a
%% request, three repeats in alternating order. Written as JSON to
%% ERLLAMA_TP_BENCH_OUT.
mtp_batch(Config) ->
    serve(Config, maps:put(n_seq_max, 4, ?COPTS), fun(S, G, R0) ->
        [
            ok = erllama_tp_hosts:load_module(S, M)
         || M <- [erllama_tp_mtp_bench, erllama_kv_seq_state]
        ],
        Seqs = [0, 1, 2, 3],
        Prompts = lists:zip(Seqs, [chat(S, G, T) || T <- dec_prompts()]),
        Ctx = peer:call(S, erllama_tp_mtp_bench, setup, [R0, Seqs], 600000),
        Call = fun(F, A) -> peer:call(S, erllama_tp_mtp_bench, F, [R0, Ctx | A], 3600000) end,
        _ = Call(ordinary, [Prompts, 16]),
        _ = Call(mtp, [Prompts, 16, #{}]),
        %% The checkpoint regression, one request alone in the cache: its
        %% restored cells land where the live ones were, so every check
        %% is byte for byte.
        [ok = peer:call(S, erllama_tp_rank, backend, [R0, seq_rm, [Q]]) || Q <- Seqs],
        #{resumed := At1, resume_info := One} = Call(mtp, [[hd(Prompts)], 64, #{resume => 3}]),
        ct:pal("checkpoint regression, one request, after round ~p: ~p", [At1, One]),
        ok = regression(One, [0], exact),
        os:getenv("ERLLAMA_TP_RESUME_ONE") =:= "1" andalso
            exit({skip, "one-request resume compare only"}),
        K0 = mtp_stats(S, R0),
        %% Correctness. Round 1 samples each request's second token K and
        %% guesses its third: the reference's third for request 0, a wrong
        %% one for request 1.
        #{tokens := Ref} = Call(ordinary, [Prompts, 64]),
        Third = fun(Q) -> lists:nth(3, maps:get(Q, Ref)) end,
        Forced = #{1 => #{0 => [Third(0)], 1 => [Third(1) + 1]}},
        #{tokens := A, rounds := [R1, R2 | _]} = Call(mtp, [Prompts, 64, #{forced => Forced}]),
        Acc = fun(Rd, Q) -> hd([N || {Q1, _, _, N} <- maps:get(results, Rd), Q1 =:= Q]) end,
        Head = fun(Rd, Q) -> hd(hd([C || {Q1, C, _, _} <- maps:get(results, Rd), Q1 =:= Q])) end,
        ct:pal(
            "forced round: request 0 accepted ~b (expect 1), request 1 accepted ~b (expect 0); request 1's next token ~b, reference ~b; tokens the same as ordinary decoding: ~p",
            [
                Acc(R1, 0),
                Acc(R1, 1),
                Head(R2, 1),
                Third(1),
                [{Q, common(maps:get(Q, A), maps:get(Q, Ref))} || Q <- Seqs]
            ]
        ),
        ?assertEqual(1, Acc(R1, 0)),
        ?assertEqual(0, Acc(R1, 1)),
        ?assertEqual(Third(1), Head(R2, 1)),
        #{tokens := U} = Call(mtp, [Prompts, 64, #{}]),
        #{tokens := Cn} = Call(mtp, [Prompts, 64, #{cancel => {10, 2}}]),
        ct:pal(
            "cancel request 2 after round 10: the others' tokens, the same as without the cancel: ~p",
            [
                [
                    {Q, length(maps:get(Q, Cn)), common(maps:get(Q, Cn), maps:get(Q, U))}
                 || Q <- [0, 1, 3]
                ]
            ]
        ),
        ?assertNot(maps:is_key(2, Cn)),
        [?assertEqual(64, length(maps:get(Q, Cn))) || Q <- [0, 1, 3]],
        %% The checkpoint regression, four requests: with a rollback
        %% pending, the restore lays the cells out otherwise, so the
        %% continuation is compared on tokens with a clear lead.
        #{resumed := At4, resume_info := Four} = Call(mtp, [Prompts, 64, #{resume => 5}]),
        ct:pal("checkpoint regression, four requests, after round ~p: ~p", [At4, Four]),
        ok = regression(Four, Seqs, placement_changes),
        ?assertEqual(0, maps:get(misaligned, diff(mtp_stats(S, R0), K0))),
        %% Speed (ERLLAMA_TP_BATCH_SPEED=0 skips it).
        Reps =
            case os:getenv("ERLLAMA_TP_BATCH_SPEED") of
                "0" -> [];
                _ -> [1, 2, 3]
            end,
        Legs = lists:append([
            [
                {Rep, Mode, Call(Mode, [Prompts, 250 | [#{} || Mode =:= mtp]])}
             || Mode <-
                    case Rep rem 2 of
                        1 -> [ordinary, mtp];
                        0 -> [mtp, ordinary]
                    end
            ]
         || Rep <- Reps
        ]),
        Rows = [batch_row(Rep, Mode, Run) || {Rep, Mode, Run} <- Legs],
        [ct:pal("~p", [R]) || R <- Rows],
        OutFile = filename:join(
            os:getenv("ERLLAMA_TP_BENCH_OUT", ?config(priv_dir, Config)), "mtp_batch.json"
        ),
        ok = file:write_file(OutFile, json:encode(#{rows => Rows})),
        ct:pal("results: ~s", [OutFile])
    end).

%% The checkpoint regression's results (erllama_tp_mtp_bench:resume/3):
%% every restore gives the checkpoint back (the same state by token
%% position, on every rank), with or without a rollback pending; with
%% none pending the first step leaves the same bytes; with one pending,
%% the same bytes too for a request alone (`exact'), and for several the
%% same greedy tokens while the live path's lead is clear.
regression(Results, Seqs, Mode) ->
    [
        begin
            #{pending := P, none_pending := NP} = maps:get(Q, Results),
            [?assertEqual(true, St) || #{state := St} <- maps:get(saved_restored, P)],
            [?assertEqual(true, St) || #{state := St} <- maps:get(saved_restored, NP)],
            [?assert(Bytes) || #{bytes := Bytes} <- maps:get(first_step, NP)],
            case Mode of
                exact -> [?assert(Bytes) || #{bytes := Bytes} <- maps:get(first_step, P)];
                placement_changes -> ok
            end,
            ?assertMatch({ok, _}, maps:get(continuation, P))
        end
     || Q <- Seqs
    ],
    ok.

batch_row(Rep, Mode, #{tokens := Toks, wall_ms := Wall} = Run) ->
    N = lists:sum([min(250, length(T)) || T <- maps:values(Toks)]),
    Base = #{
        repeat => Rep,
        mode => Mode,
        tokens => N,
        lengths => [length(T) || T <- maps:values(Toks)],
        tokens_per_s => N / Wall * 1000,
        wall_ms => Wall
    },
    case Mode of
        ordinary ->
            St = maps:get(steps_ms, Run),
            Base#{
                steps => length(St),
                step_ms => median(St),
                tokens_per_step => N / max(1, length(St))
            };
        mtp ->
            Rs = maps:get(rounds, Run),
            Committed = lists:sum([length(C) || R <- Rs, {_, C, _, _} <- maps:get(results, R)]),
            Times = [maps:get(times, R) || R <- Rs],
            Mean = fun(K) -> mean([maps:get(K, T) || T <- Times]) / 1000 end,
            Base#{
                rounds => length(Rs),
                round_ms => median([maps:get(wall_ms, R) || R <- Rs]),
                committed_per_round => Committed / max(1, length(Rs)),
                draft_rounds_per_request => maps:from_list([
                    {Q, length([1 || R <- Rs, lists:member(Q, maps:get(drafted, R))])}
                 || Q <- maps:keys(Toks)
                ]),
                accepted_per_draft =>
                    lists:sum([A || R <- Rs, {_, _, _, A} <- maps:get(results, R)]) /
                    max(1, lists:sum([length(maps:get(drafted, R)) || R <- Rs])),
                draft_ms => Mean(draft_us),
                verify_ms => Mean(verify_us),
                rollback_ms => Mean(rollback_us)
            }
    end.

median(L) -> lists:nth((length(L) + 1) div 2, lists:sort(L)).

common(A, B) -> length(lists:takewhile(fun({X, Y}) -> X =:= Y end, lists:zip(A, B, trim))).

%% Four concurrent requests through erllama's public API, three of them
%% drafting with MTP: a request with eight tokens to go (it has no
%% guess near its end while the others have one), one cancelled after
%% 12 tokens whose sequence a new request then takes, and one without
%% MTP (it joins every shared verify with no guess). Each request's
%% tokens against the same prompt decoded without MTP (four requests
%% co-batched, with logprobs): the same tokens, up to a first difference
%% that may only come where the reference's top token leads the next by
%% less than ?CLEAR_LEAD nats (a skipped or repeated token, or a wrong
%% row, differs where the lead is clear); every drafting request drafts;
%% every request ends as asked; the group serves another request after.
mtp_scheduler(Config) ->
    serve(Config, maps:put(n_seq_max, 4, ?COPTS), fun(S, #{id := Id} = G, _R0) ->
        ok = erllama_tp_hosts:load_module(S, erllama_tp_mtp_sched),
        [P0, P1, P2, P3] = [chat(S, G, T) || T <- dec_prompts()],
        Mtp = #{speculative => #{type => mtp, n_max => 1}},
        Run = fun(Specs) -> peer:call(S, erllama_tp_mtp_sched, run, [Id, Specs], 1800000) end,
        Ref = Run([
            {I, P, #{response_tokens => 48, logprobs => 2}, none}
         || {I, P} <- lists:enumerate(0, [P0, P1, P2, P3])
        ]),
        Got = Run([
            {a, P0, Mtp#{response_tokens => 48}, none},
            {b, P1, Mtp#{response_tokens => 8}, none},
            {c, P2, Mtp#{response_tokens => 48},
                {cancel_after, 12, {d, P0, Mtp#{response_tokens => 48}, none}}},
            {e, P3, #{response_tokens => 48}, none}
        ]),
        Against = #{a => 0, b => 1, c => 2, d => 0, e => 3},
        Rows = [
            begin
                #{tokens := Ts, ending := End} = R = maps:get(Name, Got),
                #{tokens := RefTs, logprobs := RefLps} = maps:get(maps:get(Name, Against), Ref),
                #{
                    request => Name,
                    ending => End,
                    tokens => length(Ts),
                    against_reference => against(Ts, RefTs, RefLps),
                    mtp_rounds => maps:get(rounds, speculative(R), 0),
                    accepted => maps:get(accepted, speculative(R), 0),
                    ttft_ms => maps:get(ttft_ms, R)
                }
            end
         || Name <- [a, b, c, d, e]
        ],
        [ct:pal("~p", [Row]) || Row <- Rows],
        [?assertMatch({ok, N} when N >= ?MIN_SAME, A) || #{against_reference := A} <- Rows],
        [?assertEqual(done, E) || #{request := Q, ending := E} <- Rows, Q =/= c],
        ?assertEqual(cancelled, maps:get(ending, maps:get(c, Got))),
        [?assert(N > 0) || #{request := Q, mtp_rounds := N} <- Rows, lists:member(Q, [a, b, d])],
        ?assertMatch(
            #{tokens := [_ | _], ending := done},
            maps:get(z, Run([{z, P0, #{response_tokens => 4}, none}]))
        )
    end).

%% `{ok, Same}': the tokens agree with the reference up to the end, or
%% up to a first difference where the reference's lead is not clear
%% (Same tokens before it); `{differs_at, Pos, Lead}' where it is.
against(Ts, RefTs, RefLps) ->
    Pairs = lists:zip3(
        lists:sublist(Ts, length(RefTs)),
        lists:sublist(RefTs, length(Ts)),
        lists:sublist(RefLps, min(length(Ts), length(RefTs)))
    ),
    against(Pairs, 0).

against([], Same) ->
    {ok, Same};
against([{T, T, _} | More], Same) ->
    against(More, Same + 1);
against([{_, _, #{top := Top}} | _], Same) ->
    case lead(Top) of
        L when L >= ?CLEAR_LEAD -> {differs_at, Same + 1, L};
        _ -> {ok, Same}
    end.

lead([{_, L1}, {_, L2} | _]) -> L1 - L2;
lead(_) -> 0.0.

speculative(#{stats := #{speculative := Sp}}) -> Sp;
speculative(_) -> #{}.

%% Four concurrent requests through erllama's normal HTTP server
%% (erllama_server on the serving node, against the split model loaded
%% there, data in /tmp), greedy, 250 tokens each, by
%% scripts/glm_serving/bench_client.py (ERLLAMA_TP_BENCH_CLIENT) run on
%% the first host: ordinary batching against one MTP guess a request.
%% A warm-up leg each, then three repeats in alternating order. Per leg:
%% each request's draft rounds (the response's `speculative' block),
%% the aggregate tokens/s and each request's time to its first visible
%% text. Written as JSON to ERLLAMA_TP_BENCH_OUT.
server_bench(Config) ->
    {ok, Client} = file:read_file(os:getenv("ERLLAMA_TP_BENCH_CLIENT")),
    serve(Config, maps:put(n_seq_max, 4, ?COPTS), fun(S, #{id := Id}, _R0) ->
        Url = start_server(S, 8090),
        ok = peer:call(S, file, write_file, ["/tmp/erllama_bench_client.py", Client]),
        Cfgs = [{ordinary, #{}}, {mtp1, #{speculative => #{type => mtp, n_max => 1}}}],
        Leg = fun(Name, Extra, Rep) ->
            Cmd = lists:flatten(
                io_lib:format(
                    "python3 -I /tmp/erllama_bench_client.py ~s ~s 4 reasoning_effort '~s'",
                    [Url, Id, json:encode(Extra)]
                )
            ),
            Out = peer:call(S, os, cmd, [Cmd], 1800000),
            L = json:decode(list_to_binary(lists:last(string:lexemes(Out, "\n")))),
            Rqs = maps:get(<<"requests">>, L),
            #{
                config => Name,
                repeat => Rep,
                aggregate_tps => maps:get(<<"aggregate_tps">>, L),
                ttft_ms => [ttft_ms(R) || R <- Rqs],
                tokens => [maps:get(<<"completion_tokens">>, R) || R <- Rqs],
                draft_rounds => [rounds(R) || R <- Rqs],
                reasoning_bytes => lists:sum([maps:get(<<"reasoning_bytes">>, R) || R <- Rqs]),
                statuses => lists:usort([maps:get(<<"status">>, R) || R <- Rqs]),
                leg => L
            }
        end,
        _ = [Leg(N, E, 0) || {N, E} <- Cfgs],
        Legs = [
            Leg(N, E, Rep)
         || Rep <- [1, 2, 3],
            {N, E} <-
                case Rep rem 2 of
                    1 -> Cfgs;
                    0 -> lists:reverse(Cfgs)
                end
        ],
        [ct:pal("~p", [maps:remove(leg, L)]) || L <- Legs],
        OutFile = filename:join(
            os:getenv("ERLLAMA_TP_BENCH_OUT", ?config(priv_dir, Config)), "erllama.bench.json"
        ),
        ok = file:write_file(OutFile, json:encode(#{legs => Legs})),
        [?assertEqual([200], St) || #{statuses := St} <- Legs],
        [?assertEqual(0, B) || #{reasoning_bytes := B} <- Legs],
        [
            [?assert(R > 0) || R <- Rounds]
         || #{config := mtp1, draft_rounds := Rounds} <- Legs
        ],
        ct:pal("results: ~s", [OutFile])
    end).

rounds(#{<<"speculative">> := #{<<"rounds">> := N}}) -> N;
rounds(_) -> 0.

ttft_ms(#{<<"ttft">> := null}) -> none;
ttft_ms(#{<<"ttft">> := T}) -> round(T * 10000) / 10.

%% erllama_server on the serving node, its OpenAI endpoint on
%% 127.0.0.1:`Port' of the first host, its data in /tmp (never the
%% user's data directory). It serves the models loaded on the node by
%% their id.
start_server(S, Port) ->
    Home = peer:call(S, os, getenv, ["HOME"]),
    Libs = peer:call(S, filelib, wildcard, [Home ++ "/erllama-tp/erllama/_build/test/lib/*/ebin"]),
    ok = peer:call(S, code, add_pathsa, [Libs]),
    Data = "/tmp/erllama_bench_data_" ++ integer_to_list(erlang:unique_integer([positive])),
    [
        ok = peer:call(S, application, set_env, [erllama_server, K, V])
     || {K, V} <- [{http, #{port => Port, ip => {127, 0, 0, 1}}}, {data_dir, Data}]
    ],
    {ok, _} = peer:call(S, application, ensure_all_started, [erllama_server], 60000),
    "http://127.0.0.1:" ++ integer_to_list(Port).

%% -----------------------------------------------------------------------------

serve(Config, Fun) ->
    serve(Config, ?COPTS, Fun).

serve(Config, COpts, Fun) ->
    Hosts = ?config(hosts, Config),
    os:putenv("ERLLAMA_TP_MODEL", ?config(model, Config)),
    erllama_tp_hosts:with_serving(Hosts, fun(S) ->
        G0 = erllama_tp_hosts:form(Hosts, S, 1),
        [P0 | _] = Peers = maps:get(rank_peers, G0),
        Before = [avail(P) || P <- Peers],
        G = erllama_tp_hosts:load(
            G0,
            ?NO_CACHE,
            #{
                model_opts => model_opts(),
                context_opts => COpts,
                tp_call_timeout => 3600000,
                draft_model => erllama_tp_hosts:home(P0, ?config(mtp, Config)),
                draft_model_opts => #{load_mode => direct_io}
            },
            3600000
        ),
        After = [avail(P) || P <- Peers],
        ct:pal(
            "loaded with the MTP head on rank 0: memory available per rank ~w GB before, ~w GB after",
            [
                [round(B / 1.0e9) || B <- Before], [round(A / 1.0e9) || A <- After]
            ]
        ),
        try
            Fun(S, G, hd(maps:get(rank_pids, G)))
        after
            erllama_tp_hosts:stop_group(G)
        end
    end).

model_opts() ->
    case os:getenv("ERLLAMA_TP_MTP_SPLIT") of
        false -> ?MOPTS;
        V -> maps:put(tensor_split, [list_to_float(F) || F <- string:lexemes(V, ",")], ?MOPTS)
    end.

run(S, #{id := Id}, Prompt, Max, Opts) ->
    peer:call(S, erllama_tp_stream, run, [Id, Prompt, Max, infinity, Opts], 600000).

mtp_stats(S, R0) ->
    case peer:call(S, gen_server, call, [R0, {hosting, {mtp, stats}}, 60000], 70000) of
        [K] -> K;
        [] -> #{}
    end.

diff(B, A) ->
    maps:map(fun(K, V) -> V - maps:get(K, A, 0) end, B).

dec_prompts() ->
    [
        <<"Write five detailed paragraphs about the history of ocean navigation.">>,
        <<"Explain how a modern operating system schedules threads. Be thorough and use several paragraphs.">>,
        <<"Describe the engineering of suspension bridges in five paragraphs.">>,
        <<"Write five paragraphs on the development of the printing press and its consequences.">>
    ].

chat(S, #{id := Id}, Text) ->
    peer:call(S, erllama_tp_stream, chat_tokens, [Id, Text], 60000).

rate(#{gaps_ms := []}) -> 0.0;
rate(#{gaps_ms := Gaps}) -> length(Gaps) / lists:sum(Gaps) * 1000.

mean(L) -> lists:sum(L) / max(1, length(L)).

avail(P) ->
    {ok, Bin} = peer:call(P, file, read_file, ["/proc/meminfo"]),
    {match, [Kb]} = re:run(Bin, "MemAvailable:\\s+([0-9]+) kB", [{capture, all_but_first, binary}]),
    binary_to_integer(Kb) * 1024.
