%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Opt-in, on CUDA hosts (ERLLAMA_TP_HOSTS) with their GPUs free and
%% ERLLAMA_TP_GLM_REF naming a GLM-5.3-Flash GGUF: the spare-sequence
%% approach to speculative decoding on a split hybrid model, which
%% keeps no recurrent-state snapshot and so cannot remove a token.
%%
%% One request. Each round copies the live sequence to a spare, verifies
%% a draft there, and then adopts the spare (every draft token accepted)
%% or drops it and replays the accepted tokens and the correction once
%% on the live sequence. The drafts are set in advance from a
%% drafting-off reference run on the same split, so each round's
%% outcome (zero, partial, full acceptance) is chosen, not left to a
%% drafter. Greedy only.
-module(erllama_tp_spec_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([short_prefix/1, long_prefix/1, cancel_releases_the_spare/1, verify_cost/1]).

-define(MOPTS, #{n_gpu_layers => 999, load_mode => direct_io}).
-define(COPTS, #{
    n_ctx => 4096, n_batch => 1024, flash_attn => enabled, n_rs_seq => 0, n_seq_max => 2
}).
-define(K, 3).
-define(AFTER, 8).

all() ->
    [verify_cost, short_prefix, long_prefix, cancel_releases_the_spare].

init_per_suite(Config) ->
    case {erllama_tp_hosts:hosts(), os:getenv("ERLLAMA_TP_GLM_REF")} of
        {false, _} -> {skip, "set ERLLAMA_TP_HOSTS to the CUDA hosts"};
        {_, false} -> {skip, "set ERLLAMA_TP_GLM_REF to a GLM-5.3-Flash GGUF"};
        {Hosts, Model} -> [{hosts, Hosts}, {model, Model} | Config]
    end.

end_per_suite(_Config) ->
    ok.

short_prefix(Config) ->
    with_group(Config, fun(G) -> rounds(G, prompt(G, 1)) end).

long_prefix(Config) ->
    with_group(Config, fun(G) -> rounds(G, prompt(G, 90)) end).

%% Where a verify's time goes: the same 3-token decode on a fresh copy
%% of the live sequence, with every row's logits and with the last
%% row's, each three times, and a 1-token decode, at both prefixes.
verify_cost(Config) ->
    with_group(Config, fun(G) ->
        [
            begin
                P = prompt(G, N),
                {_, {ok, L}} = op(G, {decode, 0, P, false}),
                T = argmax(L),
                Time = fun(Toks, All) ->
                    {_, ok} = op(G, {cp, 0, 1}),
                    {Ms, {ok, _}} = op(G, {decode, 1, Toks, All}),
                    {_, ok} = op(G, {drop, 1}),
                    Ms
                end,
                Three = [T, 13, 13],
                Rows = [Time(Three, true) || _ <- [1, 2, 3]],
                Last = [Time(Three, false) || _ <- [1, 2, 3]],
                One = [Time([T], true) || _ <- [1, 2, 3]],
                ct:pal(
                    "~b-token prefix: 3 tokens, every row ~w ms; 3 tokens, last row ~w ms; 1 token ~w ms",
                    [
                        length(P),
                        [round(X) || X <- Rows],
                        [round(X) || X <- Last],
                        [round(X) || X <- One]
                    ]
                ),
                %% A copy's cost paid at the next decode, or a graph
                %% rebuilt at each change of sequence?
                {_, ok} = op(G, {cp, 0, 1}),
                Order = [
                    {Name, round(element(1, op(G, {decode, Sq, [13], false})))}
                 || {Name, Sq} <- [
                        {spare1, 1}, {spare2, 1}, {live1, 0}, {live2, 0}, {spare3, 1}, {live3, 0}
                    ]
                ],
                ct:pal("~b-token prefix, after one copy, 1-token decodes in order: ~p ms", [
                    length(P), Order
                ]),
                {_, ok} = op(G, {drop, 1}),
                {_, ok} = op(G, {drop, 0})
            end
         || N <- [1, 90]
        ],
        ok
    end).

%% A round cut after the verify (a cancel): the spare is dropped, the
%% live sequence is as before, and the spare serves the next request.
cancel_releases_the_spare(Config) ->
    with_group(Config, fun(G) ->
        P1 = prompt(G, 1),
        {_, {ok, L1}} = op(G, {decode, 0, P1, false}),
        Live = length(P1) - 1,
        {_, Live} = op(G, {pos_max, 0}),
        {_, ok} = op(G, {cp, 0, 1}),
        {_, {ok, _}} = op(G, {decode, 1, [argmax(L1), 13, 13], true}),
        %% Cancelled here.
        {_, ok} = op(G, {drop, 1}),
        ?assertEqual({0, -1}, setelement(1, op(G, {pos_max, 1}), 0)),
        ?assertEqual(Live, element(2, op(G, {pos_max, 0}))),
        %% The live sequence goes on as if no round had started.
        {_, {ok, A}} = op(G, {decode, 0, [argmax(L1)], false}),
        {_, ok} = op(G, {drop, 0}),
        {_, {ok, L1b}} = op(G, {decode, 0, P1, false}),
        {_, {ok, B}} = op(G, {decode, 0, [argmax(L1b)], false}),
        diff("after a cancelled round vs never started", A, B),
        %% The next request takes the spare again.
        {_, ok} = op(G, {drop, 0}),
        P2 = prompt(G, 3),
        {_, {ok, L2}} = op(G, {decode, 0, P2, false}),
        {_, ok} = op(G, {cp, 0, 1}),
        {_, {ok, V}} = op(G, {decode, 1, [argmax(L2)], true}),
        ?assertEqual(length(P2), element(2, op(G, {pos_max, 1}))),
        ?assert(byte_size(V) > 0)
    end).

%% -----------------------------------------------------------------------------

%% The reference (one token a step, drafting off), then the rounds:
%% full, partial and zero acceptance, then ordinary decoding, each
%% compared with the reference at the same committed position.
rounds(G, Prompt) ->
    N = 3 * ?K + 1 + ?AFTER,
    {DecMs, Ref} = reference(G, Prompt, N),
    RefToks = [T || {T, _} <- Ref],
    ct:pal("~b-token prefix; reference: ~b tokens, ~.1f ms a token (median)", [
        length(Prompt), N, median(DecMs)
    ]),
    {_, ok} = op(G, {drop, 0}),
    {_, {ok, L0}} = op(G, {decode, 0, Prompt, false}),
    St0 = #{live => 0, spare => 1, logits => L0, committed => [], times => []},
    Plan = [
        {full, fun(Next) -> lists:sublist(Next, ?K) end},
        {partial, fun(Next) -> lists:sublist(Next, ?K - 1) ++ [wrong(lists:nth(?K, Next))] end},
        {zero, fun(Next) -> [wrong(hd(Next)) | lists:sublist(tl(Next), ?K - 1)] end}
    ],
    St =
        try
            lists:foldl(
                fun({Kind, Draft}, S) ->
                    Next = lists:nthtail(length(maps:get(committed, S)), RefToks),
                    round(G, Kind, Draft(Next), Ref, S)
                end,
                St0,
                Plan
            )
        catch
            throw:{near_tie, Kind} -> {near_tie, Kind}
        end,
    rounds_end(G, St, Ref, RefToks, DecMs).

rounds_end(_G, {near_tie, Kind}, _Ref, _RefToks, _DecMs) ->
    ct:pal("stopped at the ~p round: the comparison with the reference ends at a near-tie", [Kind]),
    {comment, io_lib:format("near-tie in the ~p round", [Kind])};
rounds_end(G, St, Ref, RefToks, DecMs) ->
    %% Ordinary decoding after the rounds.
    Final = lists:foldl(fun(_, S) -> plain(G, Ref, S) end, St, lists:seq(1, ?AFTER)),
    Committed = maps:get(committed, Final),
    Common = length(
        lists:takewhile(
            fun({A, B}) -> A =:= B end,
            lists:zip(Committed, lists:sublist(RefToks, length(Committed)))
        )
    ),
    ct:pal("committed ~b tokens, the first ~b equal the reference", [length(Committed), Common]),
    report_times(lists:reverse(maps:get(times, Final)), median(DecMs)),
    _ = G,
    ?assertEqual(lists:sublist(RefToks, length(Committed)), Committed).

%% Drafting off: the prompt, then one greedy token a step on sequence 0.
reference(G, Prompt, N) ->
    {_, {ok, L0}} = op(G, {decode, 0, Prompt, false}),
    {Ms, Ref, _} = lists:foldl(
        fun(_, {Ms0, Acc, L}) ->
            T = argmax(L),
            {D, {ok, L1}} = op(G, {decode, 0, [T], false}),
            {[D | Ms0], [{T, L1} | Acc], L1}
        end,
        {[], [], L0},
        lists:seq(1, N)
    ),
    {lists:reverse(Ms), lists:reverse(Ref)}.

round(G, Kind, Draft, Ref, #{live := Live, spare := Spare, logits := L, committed := C} = S) ->
    {_, Pos} = op(G, {pos_max, Live}),
    {CpMs, ok} = op(G, {cp, Live, Spare}),
    {VMs, {ok, V}} = op(G, {decode, Spare, Draft, true}),
    Rows = rows(V),
    %% The verify ran on the spare: the live sequence is where it was.
    ?assertEqual(Pos, element(2, op(G, {pos_max, Live}))),
    ?assertEqual(Pos + length(Draft), element(2, op(G, {pos_max, Spare}))),
    A = accepted(Draft, [L | Rows]),
    Correction = argmax(lists:nth(A + 1, [L | Rows])),
    {Commit, Logits, Times, S1} =
        case A =:= length(Draft) of
            true ->
                %% Adopt: the spare becomes the live sequence; the old
                %% live sequence is dropped and is the next spare.
                {DropMs, ok} = op(G, {drop, Live}),
                {Draft, lists:last(Rows), #{copy => CpMs, verify => VMs, adopt => DropMs}, S#{
                    live := Spare, spare := Live
                }};
            false ->
                {DropMs, ok} = op(G, {drop, Spare}),
                ?assertEqual(-1, element(2, op(G, {pos_max, Spare}))),
                Replay = lists:sublist(Draft, A) ++ [Correction],
                {RMs, {ok, L1}} = op(G, {decode, Live, Replay, false}),
                {Replay, L1, #{copy => CpMs, verify => VMs, drop => DropMs, replay => RMs}, S}
        end,
    case A =:= expected(Kind) of
        true ->
            ok;
        false ->
            %% The batch of the verify computed a different top token
            %% than one-token decoding at a near-tie.
            Row = lists:nth(A + 1, [L | Rows]),
            %% The reference row predicting the same token: after the
            %% committed tokens and the accepted drafts.
            TieRef =
                case length(C) + A of
                    0 -> L;
                    Nth -> element(2, lists:nth(Nth, Ref))
                end,
            ct:pal(
                "~p round: ~b accepted, not ~b (draft ~w); verify row ~b top ~b margin ~.3f; reference top ~b margin ~.3f",
                [
                    Kind,
                    A,
                    expected(Kind),
                    Draft,
                    A + 1,
                    argmax(Row),
                    margin(f32s(Row)),
                    argmax(TieRef),
                    margin(f32s(TieRef))
                ]
            ),
            throw({near_tie, Kind})
    end,
    C1 = C ++ Commit,
    {_, RefL} = lists:nth(length(C1), Ref),
    diff(
        io_lib:format("~p round, ~b accepted, logits after ~b committed", [Kind, A, length(C1)]),
        Logits,
        RefL
    ),
    Total = lists:sum(maps:values(Times)),
    S1#{
        logits := Logits,
        committed := C1,
        times := [
            {Kind, length(Commit), Times#{total => Total, per_token => Total / length(Commit)}}
            | maps:get(times, S)
        ]
    }.

plain(G, Ref, #{live := Live, logits := L, committed := C} = S) ->
    T = argmax(L),
    {_, {ok, L1}} = op(G, {decode, Live, [T], false}),
    C1 = C ++ [T],
    {_, RefL} = lists:nth(length(C1), Ref),
    diff(io_lib:format("ordinary decode, logits after ~b committed", [length(C1)]), L1, RefL),
    S#{logits := L1, committed := C1}.

expected(full) -> ?K;
expected(partial) -> ?K - 1;
expected(zero) -> 0.

%% How many draft tokens the target agrees with, from its logits before
%% each one.
accepted(Draft, Logits) ->
    length(
        lists:takewhile(
            fun({D, L}) -> argmax(L) =:= D end,
            lists:zip(Draft, lists:sublist(Logits, length(Draft)))
        )
    ).

report_times(Rounds, DecMs) ->
    [
        ct:pal("~p round: ~b committed; ~s; ~.1f ms a committed token (ordinary decode ~.1f)", [
            Kind,
            N,
            lists:join(", ", [
                io_lib:format("~p ~.1f ms", [K, V])
             || {K, V} <- lists:sort(maps:to_list(maps:without([per_token], T)))
            ]),
            maps:get(per_token, T),
            DecMs
        ])
     || {Kind, N, T} <- Rounds
    ].

%% -----------------------------------------------------------------------------

with_group(Config, Fun) ->
    Hosts = ?config(hosts, Config),
    Model = ?config(model, Config),
    erllama_tp_hosts:with_serving(Hosts, fun(S) ->
        #{rank_peers := Peers, rank_pids := Pids} = erllama_tp_hosts:form(Hosts, S, 1),
        Ranks = lists:zip(Peers, Pids),
        try
            Before = [avail(P) || P <- Peers],
            [
                ok = R
             || R <- erllama_tp_hosts:parallel([
                    fun() ->
                        peer:call(
                            P,
                            erllama_tp_rank,
                            load,
                            [R, erllama_tp_hosts:home(P, Model), {?MOPTS, copts()}],
                            3600000
                        )
                    end
                 || {P, R} <- Ranks
                ])
            ],
            ok = erllama_tp_hosts:load_module(hd(Peers), erllama_tp_glm_ref),
            G = #{
                ranks => Ranks, path => erllama_tp_hosts:home(hd(Peers), Model), peer0 => hd(Peers)
            },
            R = Fun(G),
            After = [avail(P) || P <- Peers],
            ct:pal(
                "memory available per rank: ~w GB before the load, ~w GB at the end (2 sequence slots)",
                [
                    [round(B / 1.0e9) || B <- Before], [round(A / 1.0e9) || A <- After]
                ]
            ),
            R
        after
            erllama_tp_hosts:stop_ranks(Ranks),
            [catch peer:stop(P) || P <- Peers]
        end
    end).

%% ERLLAMA_TP_SPEC_UNIFIED=1: one KV stream for both sequences.
copts() ->
    case os:getenv("ERLLAMA_TP_SPEC_UNIFIED") of
        "1" -> maps:put(kv_unified, true, ?COPTS);
        _ -> ?COPTS
    end.

%% Every rank makes the call; the results must agree. The slowest rank's
%% time and the (common) result.
op(#{ranks := Ranks}, Op) ->
    Res = erllama_tp_hosts:parallel([
        fun() -> peer:call(P, erllama_tp_rank, seq_op, [R, Op], 600000) end
     || {P, R} <- Ranks
    ]),
    [{_, First} | _] = Res,
    [?assertEqual(First, X, Op) || {_, X} <- Res],
    {lists:max([Ms || {Ms, _} <- Res]), First}.

prompt(#{peer0 := P, path := Path}, N) ->
    Text = iolist_to_binary([
        <<"Write a short story about a lighthouse keeper. ">>,
        lists:duplicate(N - 1, <<"The quick brown fox jumps over the lazy dog. ">>)
    ]),
    peer:call(P, erllama_tp_glm_ref, tokenize, [Path, Text]).

rows(Bin) ->
    Sz = byte_size(Bin) div ?K,
    [binary:part(Bin, I * Sz, Sz) || I <- lists:seq(0, ?K - 1)].

wrong(T) -> T + 1.

argmax(Bin) -> erllama_tp_glm_ref:argmax(Bin).

%% The largest difference, and the top token's margin over the second in
%% both, where a near-tie can change a greedy token.
diff(What, A, B) ->
    FA = [V || <<V:32/float-native>> <= A],
    FB = [V || <<V:32/float-native>> <= B],
    Max = lists:max([abs(X - Y) || {X, Y} <- lists:zip(FA, FB)]),
    ct:pal("~ts: max |diff| ~.4g; top ~b vs ~b, margin ~.3f vs ~.3f", [
        What, Max, argmax(A), argmax(B), margin(FA), margin(FB)
    ]),
    ?assertEqual(argmax(B), argmax(A), lists:flatten(What)).

f32s(Bin) -> [V || <<V:32/float-native>> <= Bin].

margin(L) ->
    [T1, T2 | _] = lists:reverse(lists:sort(L)),
    T1 - T2.

avail(P) ->
    {ok, Bin} = peer:call(P, file, read_file, ["/proc/meminfo"]),
    {match, [Kb]} = re:run(Bin, "MemAvailable:\\s+([0-9]+) kB", [{capture, all_but_first, binary}]),
    binary_to_integer(Kb) * 1024.

median([]) -> 0.0;
median(L) -> lists:nth((length(L) + 1) div 2, lists:sort(L)).
