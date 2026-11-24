%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Opt-in, on CUDA hosts (ERLLAMA_TP_HOSTS) with their GPUs free:
%% rolling back the recurrent state of GLM-5.3-Flash split over the
%% group, with recurrent-state snapshots on (n_rs_seq 2). A batch of n
%% tokens keeps n - 1 snapshots, so a 3-token batch can take back at
%% most 2 tokens.
%%
%%   ERLLAMA_TP_GLM_REF  a GLM-5.3-Flash GGUF (rollback_matches)
%%   ERLLAMA_TP_GLM      the GGUF for the rounds (defaults to the above)
-module(erllama_tp_rollback_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([rollback_matches/1, known_plus_two_rounds/1]).

-define(MOPTS, #{n_gpu_layers => 999, load_mode => direct_io}).
-define(COPTS, #{
    n_ctx => 4096,
    n_batch => 1024,
    flash_attn => enabled,
    n_rs_seq => 2,
    n_seq_max => 2,
    kv_unified => true
}).
-define(P, 0.632).
-define(REF, 100).

all() ->
    [rollback_matches, known_plus_two_rounds].

init_per_suite(Config) ->
    case {erllama_tp_hosts:hosts(), os:getenv("ERLLAMA_TP_GLM_REF")} of
        {false, _} -> {skip, "set ERLLAMA_TP_HOSTS to the CUDA hosts"};
        {_, false} -> {skip, "set ERLLAMA_TP_GLM_REF to a GLM-5.3-Flash GGUF"};
        {Hosts, Model} -> [{hosts, Hosts}, {model, Model} | Config]
    end.

end_per_suite(_Config) ->
    ok.

%% Three tokens decoded in one batch, then rolled back by 2 (snapshot
%% slot 2) or by 1 (slot 1), then decoding goes on: the logits match a
%% drafting-off run on the same split, on an even split at sequence 0
%% and a 3:1 split at sequence 1 (the snapshots at a nonzero cell). The
%% logits are summed over the ranks, so a snapshot written into the
%% wrong history on either rank shows. Rolling back 3 is refused.
rollback_matches(Config) ->
    [
        with_group(Config, ?config(model, Config), maps:merge(?MOPTS, MExtra), ?COPTS, fun(G) ->
            rollback_case(G, Name, Seq)
        end)
     || {Name, Seq, MExtra} <- [{even, 0, #{}}, {'uneven 3:1', 1, #{tensor_split => [3.0, 1.0]}}]
    ],
    ok.

rollback_case(G, Name, S) ->
    P = prompt(G, 30),
    N = length(P),
    {_, {ok, L0}} = op(G, {decode, S, P, false}),
    %% Drafting off: four greedy tokens, one at a time.
    {Toks, Ref} = lists:foldl(
        fun(_, {Ts, [L | _] = Ls}) ->
            T = argmax(L),
            {_, {ok, L1}} = op(G, {decode, S, [T], false}),
            {Ts ++ [T], [L1 | Ls]}
        end,
        {[], [L0]},
        lists:seq(1, 4)
    ),
    [R4, R3, R2, _R1, _] = Ref,
    [T1, T2, T3, T4] = Toks,
    Again = fun() ->
        {_, ok} = op(G, {drop, S}),
        {_, {ok, _}} = op(G, {decode, S, P, false}),
        {_, {ok, _}} = op(G, {decode, S, [T1, T2, T3], true}),
        ?assertEqual(N + 2, element(2, op(G, {pos_max, S})))
    end,
    %% Back by 2: keep the prompt and T1.
    Again(),
    {_, ok} = op(G, {trim, S, N + 1}),
    ?assertEqual(N, element(2, op(G, {pos_max, S}))),
    {_, {ok, A2}} = op(G, {decode, S, [T2], false}),
    diff(io_lib:format("~p, seq ~b: back 2, then T2 vs drafting off", [Name, S]), A2, R2),
    {_, {ok, A3}} = op(G, {decode, S, [T3], false}),
    diff(io_lib:format("~p, seq ~b: back 2, then T3 vs drafting off", [Name, S]), A3, R3),
    %% Back by 1: keep the prompt, T1 and T2.
    Again(),
    {_, ok} = op(G, {trim, S, N + 2}),
    {_, {ok, B3}} = op(G, {decode, S, [T3], false}),
    diff(io_lib:format("~p, seq ~b: back 1, then T3 vs drafting off", [Name, S]), B3, R3),
    {_, {ok, B4}} = op(G, {decode, S, [T4], false}),
    diff(io_lib:format("~p, seq ~b: back 1, then T4 vs drafting off", [Name, S]), B4, R4),
    %% Back by 3: beyond what a 3-token batch keeps.
    Again(),
    {_, Refused} = op(G, {trim, S, N}),
    ct:pal("~p, seq ~b: back 3 answers ~p", [Name, S, Refused]),
    ?assertMatch({error, _}, Refused),
    {_, ok} = op(G, {drop, S}),
    ok.

%% Rounds of one known token and two guesses in one 3-row batch on the
%% live sequence, rolled back by 2 (first guess wrong), 1 (second wrong)
%% or 0. The correction or bonus token is the next round's known token.
%% Guesses are taken from a drafting-off reference (wrong ones made
%% wrong), each right with probability 0.632, so each round emits
%% 1 + p + p^2 tokens on average. Timed from the ranks; compared with
%% ordinary decoding in the same context. Then a cancelled round.
known_plus_two_rounds(Config) ->
    Model =
        case os:getenv("ERLLAMA_TP_GLM") of
            false -> ?config(model, Config);
            M -> M
        end,
    with_group(Config, Model, ?MOPTS, ?COPTS, fun(G) -> rounds_run(G) end).

rounds_run(G) ->
    P = prompt(G, 90),
    {_, {ok, L0}} = op(G, {decode, 0, P, false}),
    Pa = perf_ranks(G),
    {Ms, Ref} = lists:foldl(
        fun(_, {Acc, [L | _] = Ls}) ->
            {D, {ok, L1}} = op(G, {decode, 0, [argmax(L)], false}),
            {[D | Acc], [L1 | Ls]}
        end,
        {[], [L0]},
        lists:seq(1, ?REF)
    ),
    Pb = perf_ranks(G),
    RefLogits = lists:reverse(Ref),
    RefToks = [argmax(L) || L <- lists:droplast(RefLogits)],
    Plain = lists:sort(Ms),
    ct:pal(
        "ordinary decoding with snapshots on: median ~.1f ms, p95 ~.1f ms (~.1f tokens/s); graph counters per rank ~p",
        [
            pct(Plain, 50), pct(Plain, 95), 1000 / pct(Plain, 50), perf_delta(Pa, Pb)
        ]
    ),
    {_, ok} = op(G, {drop, 0}),
    {_, {ok, L1}} = op(G, {decode, 0, P, false}),
    rand:seed(exsss, {7, 11, 13}),
    Pc = perf_ranks(G),
    St = loop(G, RefToks, #{known => argmax(L1), committed => [], rounds => []}),
    Pd = perf_ranks(G),
    Rounds = lists:reverse(maps:get(rounds, St)),
    Committed = maps:get(committed, St),
    Total = lists:sum([R || {_, R, _} <- Rounds]),
    ByB = [{B, lists:sort([R || {B1, R, _} <- Rounds, B1 =:= B])} || B <- [2, 1, 0]],
    [
        ct:pal("rolled back ~b: ~b rounds, median ~.1f ms, p95 ~.1f ms", [
            B, length(L), pct(L, 50), pct(L, 95)
        ])
     || {B, L} <- ByB, L =/= []
    ],
    Emit = fun
        (2) -> 1;
        (1) -> 2;
        (0) -> 3
    end,
    W = [{2, 1 - ?P}, {1, ?P * (1 - ?P)}, {0, ?P * ?P}],
    ExpTok = lists:sum([Pr * Emit(B) || {B, Pr} <- W]),
    ExpMs = lists:sum([
        Pr * pct(proplists:get_value(B, ByB), 50)
     || {B, Pr} <- W, proplists:get_value(B, ByB) =/= []
    ]),
    ct:pal(
        "rounds: ~b, ~b tokens committed in ~.1f ms (~.1f tokens/s measured); weighted: ~.2f tokens a round, ~.1f ms a round, ~.1f tokens/s; ordinary ~.1f tokens/s; graph counters per rank ~p",
        [
            length(Rounds),
            length(Committed),
            Total,
            length(Committed) / Total * 1000,
            ExpTok,
            ExpMs,
            ExpTok / ExpMs * 1000,
            1000 / pct(Plain, 50),
            perf_delta(Pc, Pd)
        ]
    ),
    ?assertEqual(lists:sublist(RefToks, length(Committed)), Committed),
    %% A round cut after its verify: the known token was committed
    %% already, so the cancel takes back the two guesses (all a 3-token
    %% batch can take back); the slot is then dropped and serves the next
    %% request.
    {_, Pos} = op(G, {pos_max, 0}),
    Known = maps:get(known, St),
    {_, {ok, _}} = op(G, {decode, 0, [Known, 13, 13], true}),
    {_, ok} = op(G, {trim, 0, Pos + 2}),
    ?assertEqual(Pos + 1, element(2, op(G, {pos_max, 0}))),
    {_, ok} = op(G, {drop, 0}),
    {_, {ok, _}} = op(G, {decode, 0, P, false}),
    ?assertEqual(length(P) - 1, element(2, op(G, {pos_max, 0}))),
    {_, ok} = op(G, {drop, 0}),
    ok.

loop(G, RefToks, #{committed := C} = S) ->
    case length(RefToks) - length(C) >= 4 of
        false -> S;
        true -> loop(G, RefToks, round(G, RefToks, S))
    end.

round(G, RefToks, #{known := K, committed := C} = S) ->
    Next = lists:nthtail(length(C) + 1, RefToks),
    [N1, N2 | _] = Next,
    G1 = guess(N1),
    G2 =
        case G1 of
            N1 -> guess(N2);
            _ -> N2
        end,
    {_, Pos} = op(G, {pos_max, 0}),
    {VMs, {ok, V}} = op(G, {decode, 0, [K, G1, G2], true}),
    [Row0, Row1, Row2] = rows(V, 3),
    {Back, Emitted, NextKnown} =
        case {argmax(Row0) =:= G1, argmax(Row1) =:= G2} of
            {false, _} -> {2, [K], argmax(Row0)};
            {true, false} -> {1, [K, G1], argmax(Row1)};
            {true, true} -> {0, [K, G1, G2], argmax(Row2)}
        end,
    TMs =
        case Back of
            0 ->
                0;
            _ ->
                {T, ok} = op(G, {trim, 0, Pos + 1 + 3 - Back}),
                T
        end,
    S#{
        known := NextKnown,
        committed := C ++ Emitted,
        rounds := [{Back, VMs + TMs, length(Emitted)} | maps:get(rounds, S)]
    }.

%% Right with probability p, else made wrong.
guess(T) ->
    case rand:uniform() < ?P of
        true -> T;
        false -> T + 1
    end.

%% -----------------------------------------------------------------------------

with_group(Config, Model, MOpts, COpts, Fun) ->
    Hosts = ?config(hosts, Config),
    erllama_tp_hosts:with_serving(Hosts, fun(S) ->
        #{rank_peers := Peers, rank_pids := Pids} = erllama_tp_hosts:form(Hosts, S, 1),
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
                            [R, erllama_tp_hosts:home(P, Model), {MOpts, COpts}],
                            3600000
                        )
                    end
                 || {P, R} <- Ranks
                ])
            ],
            ok = erllama_tp_hosts:load_module(hd(Peers), erllama_tp_glm_ref),
            Fun(#{
                ranks => Ranks, path => erllama_tp_hosts:home(hd(Peers), Model), peer0 => hd(Peers)
            })
        after
            erllama_tp_hosts:stop_ranks(Ranks),
            [catch peer:stop(P) || P <- Peers]
        end
    end).

%% Every rank makes the call and must answer the same; the slowest
%% rank's time. A decode's answer is `{ok, Logits}'.
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

perf_ranks(#{ranks := Ranks}) ->
    [
        element(2, peer:call(P, erllama_tp_rank, seq_op, [R, {diag, perf}], 60000))
     || {P, R} <- Ranks
    ].

perf_delta(As, Bs) ->
    [
        maps:with(
            [n_reused, n_reallocs_same, n_reallocs_diff],
            maps:map(fun(K, V) -> V - maps:get(K, A) end, B)
        )
     || {A, B} <- lists:zip(As, Bs)
    ].

prompt(#{peer0 := P, path := Path}, N) ->
    Text = iolist_to_binary([
        <<"Write a short story about a lighthouse keeper. ">>,
        lists:duplicate(N - 1, <<"The quick brown fox jumps over the lazy dog. ">>)
    ]),
    peer:call(P, erllama_tp_glm_ref, tokenize, [Path, Text]).

rows(Bin, K) ->
    Sz = byte_size(Bin) div K,
    [binary:part(Bin, I * Sz, Sz) || I <- lists:seq(0, K - 1)].

argmax(Bin) -> erllama_tp_glm_ref:argmax(Bin).

%% Same top token; the largest difference and the top margin reported.
diff(What, A, B) ->
    FA = [V || <<V:32/float-native>> <= A],
    FB = [V || <<V:32/float-native>> <= B],
    Max = lists:max([abs(X - Y) || {X, Y} <- lists:zip(FA, FB)]),
    ct:pal("~ts: max |diff| ~.4g; top ~b vs ~b, margin ~.3f vs ~.3f", [
        What, Max, argmax(A), argmax(B), margin(FA), margin(FB)
    ]),
    ?assertEqual(argmax(B), argmax(A), lists:flatten(What)).

margin(L) ->
    [T1, T2 | _] = lists:reverse(lists:sort(L)),
    T1 - T2.

pct([], _) ->
    0.0;
pct(Sorted, P) ->
    lists:nth(max(1, min(length(Sorted), round(P / 100 * length(Sorted) + 0.5))), Sorted).
