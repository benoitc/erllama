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
-export([profile_verify/1, kv_capacity_cost/1, kv_rows_reuse/1, spec_rounds/1]).

-define(MOPTS, #{n_gpu_layers => 999, load_mode => direct_io}).
-define(COPTS, #{
    n_ctx => 4096, n_batch => 1024, flash_attn => enabled, n_rs_seq => 0, n_seq_max => 2
}).
-define(K, 3).
-define(AFTER, 8).

all() ->
    [
        verify_cost,
        short_prefix,
        long_prefix,
        cancel_releases_the_spare,
        profile_verify,
        kv_capacity_cost,
        kv_rows_reuse,
        spec_rounds
    ].

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

%% Where a verify's time goes, under both KV layouts, on the live
%% sequence of a 901-token prompt with two sequence slots: one token
%% with one logit row, three tokens with the last row only (a
%% diagnostic), three tokens with every row. The three cases alternate
%% in blocks; the first run of each block (after a change of batch
%% shape) is reported apart, the others give the median and p95 of the
%% end-to-end time, of the decode (this rank's compute and the
%% collectives inside the graph) and of the host-buffer logits
%% reduction. The copy a per-sequence layout defers to the next decode
%% is measured on its own.
profile_verify(Config) ->
    [
        with_group(Config, COpts, fun(G) -> profile_layout(G, Layout) end)
     || {Layout, COpts} <- [{per_sequence, ?COPTS}, {unified, maps:put(kv_unified, true, ?COPTS)}]
    ],
    ok.

%% How much of the unified cache's cost on a single-token decode comes
%% from the layout and how much from the capacity of the active stream:
%%   A  one KV stream per sequence, n_ctx 4096 (2048 cells a stream)
%%   B  unified, n_ctx 2048 (2048 cells in the shared stream)
%%   C  unified, n_ctx 4096 (4096 cells in the shared stream)
%% and A again without the diagnostic callback, as a control. Two
%% configured sequence slots, one live sequence, the same 901-token
%% prompt and the same token ids; 8 warm decodes, then 40 timed ones at
%% positions 909 to 948, so the active KV extent stays in one padding
%% bucket. Then one armed decode for the graph's input shapes.
-define(KV_WARM, 8).
-define(KV_TIMED, 40).

kv_capacity_cost(Config) ->
    Base = maps:merge(?COPTS, #{n_ctx => 4096}),
    Cases = [
        {'A per-sequence 4096', maps:put(diag_inputs, true, Base)},
        {'B unified 2048',
            maps:merge(Base, #{diag_inputs => true, kv_unified => true, n_ctx => 2048})},
        {'C unified 4096', maps:merge(Base, #{diag_inputs => true, kv_unified => true})},
        {'A per-sequence 4096, no callback', Base}
    ],
    Res = [
        {Name, with_group(Config, COpts, fun(G) -> kv_case(G, Name) end)}
     || {Name, COpts} <- Cases
    ],
    Med = fun(Name) -> pct(maps:get(dec, proplists:get_value(Name, Res)), 50) end,
    ct:pal(
        "single-token decode medians: A ~.1f ms, B ~.1f ms, C ~.1f ms, A without callback ~.1f ms~n"
        "C - B (capacity within unified) ~.1f ms; B - A (layout at matched per-stream capacity) ~.1f ms",
        [
            Med('A per-sequence 4096'),
            Med('B unified 2048'),
            Med('C unified 4096'),
            Med('A per-sequence 4096, no callback'),
            Med('C unified 4096') - Med('B unified 2048'),
            Med('B unified 2048') - Med('A per-sequence 4096')
        ]
    ),
    ok.

kv_case(G, Name) ->
    P = prompt(G, 90),
    {_, {ok, _}} = op(G, {decode, 0, P, false}),
    Toks = [1000 + I || I <- lists:seq(1, ?KV_WARM + ?KV_TIMED + 1)],
    {Warm, Rest} = lists:split(?KV_WARM, Toks),
    {Timed, [Last]} = lists:split(?KV_TIMED, Rest),
    _ = [run_t(G, [T], false) || T <- Warm],
    {_, Perf0} = op(G, {diag, perf}),
    Runs = [run_t(G, [T], false) || T <- Timed],
    {_, Perf1} = op(G, {diag, perf}),
    {_, Pos} = op(G, {pos_max, 0}),
    Inputs =
        case op(G, {diag, arm}) of
            {_, ok} ->
                _ = run_t(G, [Last], false),
                element(2, op(G, {diag, read}));
            {_, {error, not_supported}} ->
                none
        end,
    S = summarize(Runs),
    Reuse = maps:map(fun(K, V) -> V - maps:get(K, Perf0) end, Perf1),
    ct:pal(
        "~s: positions ~b to ~b; decode ~s; end to end ~s; graph counters over the timed runs ~p", [
            Name, length(P) + ?KV_WARM, Pos, fmt(dec, S), fmt(e2e, S), Reuse
        ]
    ),
    case Inputs of
        none -> ok;
        _ -> ct:pal("~s: graph inputs (name, type, shape):~n~p", [Name, Inputs])
    end,
    {_, ok} = op(G, {drop, 0}),
    S.

%% Case B of kv_capacity_cost (unified, n_ctx 2048, no callback) on
%% fresh rank nodes, with LLAMA_KV_ROWS_DISABLE unset, then set on
%% both ranks (llama.cpp tests it by presence): the fused-KV-rows
%% setting each rank resolved, its graph counters, the decode times and
%% the argmax of each timed decode, compared between the two. Then,
%% unset only, reuse inside the padding bucket and across the 1024
%% boundary (a real change of extent must rebuild).
kv_rows_reuse(Config) ->
    COpts = maps:merge(?COPTS, #{kv_unified => true, n_ctx => 2048}),
    Saved = os:getenv("ERLLAMA_TP_NCCL_ENV", ""),
    Runs = [
        begin
            os:putenv("ERLLAMA_TP_NCCL_ENV", string:trim(Saved ++ " " ++ Env)),
            R = with_group(Config, COpts, fun(G) -> rows_case(G, Name, Name =:= unset) end),
            os:putenv("ERLLAMA_TP_NCCL_ENV", Saved),
            {Name, R}
        end
     || {Name, Env} <- [{unset, ""}, {'LLAMA_KV_ROWS_DISABLE=1', "LLAMA_KV_ROWS_DISABLE=1"}]
    ],
    [{_, #{trace := T1, dec := D1}}, {_, #{trace := T2, dec := D2}}] = Runs,
    Same = length([x || {{A, _}, {B, _}} <- lists:zip(T1, T2), A =:= B]),
    MaxDiff = lists:max([abs(X - Y) || {{_, X}, {_, Y}} <- lists:zip(T1, T2)]),
    ct:pal(
        "fixed trace: ~b of ~b argmax tokens agree between unset and set; largest difference of the top logit ~.4g~n"
        "decode median: unset ~.1f ms, set ~.1f ms (~.1f ms returned)",
        [Same, length(T1), MaxDiff, pct(D1, 50), pct(D2, 50), pct(D1, 50) - pct(D2, 50)]
    ),
    ok.

rows_case(G, Name, Extent) ->
    P = prompt(G, 90),
    {_, {ok, _}} = op(G, {decode, 0, P, false}),
    Toks = [1000 + I || I <- lists:seq(1, ?KV_WARM + ?KV_TIMED)],
    {Warm, Timed} = lists:split(?KV_WARM, Toks),
    _ = [run_t(G, [T], false) || T <- Warm],
    Perf0 = perf_ranks(G),
    Trace = [trace_t(G, T) || T <- Timed],
    Perf1 = perf_ranks(G),
    Dec = lists:sort([maps:get(dec, R) || {_, R} <- Trace]),
    ct:pal(
        "~p: fused KV rows as resolved on each rank: ~p~ndecode median ~.1f ms, p95 ~.1f ms; graph counters per rank over the 40 timed decodes: ~p",
        [
            Name,
            [
                lists:flatten(lists:join(" | ", [binary_to_list(L) || L <- Ls]))
             || Ls <- kv_rows_lines(G)
            ],
            pct(Dec, 50),
            pct(Dec, 95),
            perf_delta(Perf0, Perf1)
        ]
    ),
    case Extent of
        true -> extent_change(G);
        false -> ok
    end,
    {_, ok} = op(G, {drop, 0}),
    #{trace => [{Tok, Top} || {{Tok, Top}, _} <- Trace], dec => Dec}.

%% Decodes that stay below position 1024, then 8 that cross it: the
%% first part must reuse every graph, the crossing must rebuild.
extent_change(G) ->
    {_, Pos} = op(G, {pos_max, 0}),
    Below = 1023 - Pos,
    Pa = perf_ranks(G),
    _ = [run_t(G, [1500], false) || _ <- lists:seq(1, Below)],
    Pb = perf_ranks(G),
    _ = [run_t(G, [1500], false) || _ <- lists:seq(1, 8)],
    Pc = perf_ranks(G),
    ct:pal("extent: ~b decodes inside the bucket (to position 1023): ~p; 8 crossing 1024: ~p", [
        Below, perf_delta(Pa, Pb), perf_delta(Pb, Pc)
    ]).

%% The argmax token and the top logit of one timed decode, with its time.
trace_t(G, T) ->
    {Ms, {ok, L}, PerRank} = op_t(G, {decode, 0, [T], false}),
    F = [V || <<V:32/float-native>> <= L],
    Top = lists:max(F),
    Dec = lists:max([D || {_, D, _} <- PerRank]) / 1000,
    {{argmax(L), Top}, #{e2e => Ms, dec => Dec}}.

perf_ranks(#{ranks := Ranks}) ->
    [
        element(2, peer:call(P, erllama_tp_rank, seq_op, [R, {diag, perf}], 60000))
     || {P, R} <- Ranks
    ].

perf_delta(As, Bs) ->
    [
        maps:with(
            [n_reused, n_reallocs_same, n_reallocs_diff, n_eval],
            maps:map(fun(K, V) -> V - maps:get(K, A) end, B)
        )
     || {A, B} <- lists:zip(As, Bs)
    ].

kv_rows_lines(#{ranks := Ranks}) ->
    [
        case peer:call(P, file, read_file, ["/tmp/erllama_tp_spec.log"]) of
            {ok, Bin} ->
                [
                    L
                 || L <- binary:split(Bin, <<"\n">>, [global]),
                    binary:match(L, [<<"kv_rows">>, <<"K/V rows">>]) =/= nomatch
                ];
            _ ->
                []
        end
     || {P, _} <- Ranks
    ].

%% Native llama.cpp lines at info, to a file on the rank's host.
native_log(P) ->
    _ = peer:call(P, file, delete, ["/tmp/erllama_tp_spec.log"]),
    ok = peer:call(P, application, set_env, [erllama, native_log_level, info]),
    ok = peer:call(P, supervisor, terminate_child, [erllama_sup, erllama_log]),
    {ok, _} = peer:call(P, supervisor, restart_child, [erllama_sup, erllama_log]),
    _ = peer:call(P, logger, set_primary_config, [level, info]),
    _ = peer:call(P, logger, add_handler, [
        tp_spec,
        logger_std_h,
        #{
            level => info,
            config => #{file => "/tmp/erllama_tp_spec.log"},
            filter_default => stop,
            filters => [{native, {fun logger_filters:domain/2, {log, sub, [erllama, native]}}}]
        }
    ]),
    ok.

%% Whole speculative rounds against ordinary decoding, on the model in
%% ERLLAMA_TP_GLM (or ERLLAMA_TP_GLM_REF), unified KV, two slots, the
%% 901-token prefix. The reference: 100 greedy one-token decodes,
%% timed (ordinary decoding) and used as the drafts. Then rounds of 3
%% drafts; how many are accepted is drawn (seeded) from a geometric
%% distribution, each draft accepted with probability 0.632 (TensorFold's
%% MTP rate on the pair) until the first rejection. Each round is timed
%% whole: copy, verify, adopt or drop, replay, and the graph rebuilds a
%% change of batch shape brings. Committed tokens are checked against the
%% reference.
-define(SPEC_P, 0.632).
-define(SPEC_REF, 100).

spec_rounds(Config) ->
    Config1 =
        case os:getenv("ERLLAMA_TP_GLM") of
            false -> Config;
            M -> lists:keystore(model, 1, Config, {model, M})
        end,
    COpts = maps:merge(?COPTS, #{kv_unified => true}),
    with_group(Config1, COpts, fun(G) -> spec_rounds_run(G) end).

spec_rounds_run(G) ->
    Prompt = prompt(G, 90),
    {_, {ok, L0}} = op(G, {decode, 0, Prompt, false}),
    PerfA = perf_ranks(G),
    {RefMs, Ref} = lists:foldl(
        fun(_, {Ms, [{_, L} | _] = Acc}) ->
            T = argmax(L),
            {D, {ok, L1}} = op(G, {decode, 0, [T], false}),
            {[D | Ms], [{T, L1} | Acc]}
        end,
        {[], [{none, L0}]},
        lists:seq(1, ?SPEC_REF)
    ),
    PerfB = perf_ranks(G),
    RefList = tl(lists:reverse(Ref)),
    RefToks = [T || {T, _} <- RefList],
    Plain = lists:sort(RefMs),
    ct:pal(
        "ordinary decoding: ~b tokens, median ~.1f ms, p95 ~.1f ms (~.1f tokens/s); graph counters per rank ~p",
        [
            ?SPEC_REF,
            pct(Plain, 50),
            pct(Plain, 95),
            1000 / pct(Plain, 50),
            perf_delta(PerfA, PerfB)
        ]
    ),
    {_, ok} = op(G, {drop, 0}),
    {_, {ok, L1}} = op(G, {decode, 0, Prompt, false}),
    rand:seed(exsss, {7, 11, 13}),
    St0 = #{live => 0, spare => 1, logits => L1, committed => [], times => []},
    PerfC = perf_ranks(G),
    T0 = erlang:monotonic_time(microsecond),
    St = spec_loop(G, RefToks, RefList, St0),
    Wall = (erlang:monotonic_time(microsecond) - T0) / 1000,
    PerfD = perf_ranks(G),
    Rounds = lists:reverse(maps:get(times, St)),
    Committed = length(maps:get(committed, St)),
    ByA = [{A, lists:sort([Ms || {A1, Ms, _} <- Rounds, A1 =:= A])} || A <- [0, 1, 2, 3]],
    [
        ct:pal("~b of 3 accepted: ~b rounds, round median ~.1f ms, p95 ~.1f ms", [
            A, length(L), pct(L, 50), pct(L, 95)
        ])
     || {A, L} <- ByA, L =/= []
    ],
    %% The distribution's own weights, with the measured medians.
    W = [
        {0, 1 - ?SPEC_P},
        {1, ?SPEC_P * (1 - ?SPEC_P)},
        {2, ?SPEC_P * ?SPEC_P * (1 - ?SPEC_P)},
        {3, ?SPEC_P * ?SPEC_P * ?SPEC_P}
    ],
    Cmt = fun
        (3) -> 3;
        (A) -> A + 1
    end,
    ExpTok = lists:sum([P * Cmt(A) || {A, P} <- W]),
    ExpMs = lists:sum([
        P * pct(proplists:get_value(A, ByA), 50)
     || {A, P} <- W, proplists:get_value(A, ByA) =/= []
    ]),
    ct:pal(
        "speculative rounds: ~b rounds, ~b committed in ~.1f ms (~.1f tokens/s measured); weighted by the distribution: ~.2f tokens a round, ~.1f ms a round, ~.1f tokens/s; ordinary decoding ~.1f tokens/s; graph counters per rank ~p",
        [
            length(Rounds),
            Committed,
            Wall,
            Committed / Wall * 1000,
            ExpTok,
            ExpMs,
            ExpTok / ExpMs * 1000,
            1000 / pct(Plain, 50),
            perf_delta(PerfC, PerfD)
        ]
    ),
    ?assertEqual(lists:sublist(RefToks, Committed), maps:get(committed, St)).

%% Rounds until fewer than 4 reference tokens are left.
spec_loop(G, RefToks, RefList, #{committed := C} = S) ->
    Next = lists:nthtail(length(C), RefToks),
    case length(Next) >= ?K + 1 of
        false ->
            S;
        true ->
            A = draw_accepted(),
            Draft =
                lists:sublist(Next, A) ++ [wrong(lists:nth(A + 1, Next)) || A < ?K] ++
                    lists:sublist(lists:nthtail(min(A + 1, ?K), Next), max(0, ?K - A - 1)),
            spec_loop(G, RefToks, RefList, spec_round(G, A, lists:sublist(Draft, ?K), S))
    end.

draw_accepted() ->
    draw_accepted(0).
draw_accepted(?K) ->
    ?K;
draw_accepted(N) ->
    case rand:uniform() < ?SPEC_P of
        true -> draw_accepted(N + 1);
        false -> N
    end.

%% One timed round, as `round/5' without the reference comparisons.
spec_round(G, Want, Draft, #{live := Live, spare := Spare, logits := L, committed := C} = S) ->
    T0 = erlang:monotonic_time(microsecond),
    {_, ok} = op(G, {cp, Live, Spare}),
    {_, {ok, V}} = op(G, {decode, Spare, Draft, true}),
    Rows = rows(V),
    A = accepted(Draft, [L | Rows]),
    {Commit, Logits, S1} =
        case A =:= length(Draft) of
            true ->
                {_, ok} = op(G, {drop, Live}),
                {Draft, lists:last(Rows), S#{live := Spare, spare := Live}};
            false ->
                {_, ok} = op(G, {drop, Spare}),
                Replay = lists:sublist(Draft, A) ++ [argmax(lists:nth(A + 1, [L | Rows]))],
                {_, {ok, L1}} = op(G, {decode, Live, Replay, false}),
                {Replay, L1, S}
        end,
    Ms = (erlang:monotonic_time(microsecond) - T0) / 1000,
    case A =:= Want of
        true -> ok;
        false -> ct:pal("a round drawn for ~b accepted took ~b (a near-tie)", [Want, A])
    end,
    S1#{
        logits := Logits,
        committed := C ++ Commit,
        times := [{A, Ms, length(Commit)} | maps:get(times, S)]
    }.

-define(ROUNDS, 4).
-define(BLOCK, 6).

profile_layout(G, Layout) ->
    P = prompt(G, 90),
    {_, {ok, L0}} = op(G, {decode, 0, P, false}),
    T = argmax(L0),
    Cases = [
        {one_token, [T], false},
        {three_last_row, [T, 13, 13], false},
        {three_every_row, [T, 13, 13], true}
    ],
    Runs = lists:append([
        [{Name, I =:= 1, run_t(G, Toks, All)} || I <- lists:seq(1, ?BLOCK)]
     || _ <- lists:seq(1, ?ROUNDS), {Name, Toks, All} <- Cases
    ]),
    {_, End} = op(G, {pos_max, 0}),
    ct:pal("~p KV: prefix ~b to ~b tokens over the runs", [Layout, length(P), End + 1]),
    Stats = [
        {Name, summarize([X || {N, false, X} <- Runs, N =:= Name]),
            summarize([X || {N, true, X} <- Runs, N =:= Name])}
     || {Name, _, _} <- Cases
    ],
    [
        ct:pal(
            "~p KV, ~p (~b runs; first after a change of shape apart):~n  end to end ~s~n  decode ~s~n  logits reduction ~s~n  the rest (calls, Erlang) ~s~n  first run of a block: end to end ~s",
            [
                Layout,
                Name,
                maps:get(n, S),
                fmt(e2e, S),
                fmt(dec, S),
                fmt(red, S),
                fmt(rest, S),
                fmt(e2e, F)
            ]
        )
     || {Name, S, F} <- Stats
    ],
    [S1, S3L, S3A] = [S || {_, S, _} <- Stats],
    ct:pal(
        "~p KV, medians: 3 tokens last row vs 1 token: +~.1f ms decode; every row vs last row: +~.1f ms decode, +~.1f ms reduction, +~.1f ms end to end",
        [
            Layout,
            med(dec, S3L) - med(dec, S1),
            med(dec, S3A) - med(dec, S3L),
            med(red, S3A) - med(red, S3L),
            med(e2e, S3A) - med(e2e, S3L)
        ]
    ),
    Reductions = [{Name, collectives(G, Toks, All)} || {Name, Toks, All} <- Cases],
    ct:pal("~p KV, collectives inside the graph per decode: ~p", [Layout, Reductions]),
    Copies = [deferred_copy(G) || _ <- lists:seq(1, 5)],
    ct:pal("~p KV, a copy to the spare then two 1-token decodes on it, ms end to end: ~w", [
        Layout, Copies
    ]),
    {_, ok} = op(G, {drop, 0}),
    ok.

%% One decode on the live sequence: the slowest rank's times, in ms.
run_t(G, Toks, All) ->
    {Ms, {ok, _}, PerRank} = op_t(G, {decode, 0, Toks, All}),
    Dec = lists:max([D || {_, D, _} <- PerRank]) / 1000,
    Red = lists:max([R || {_, _, R} <- PerRank]) / 1000,
    #{e2e => Ms, dec => Dec, red => Red, rest => Ms - Dec - Red}.

collectives(#{ranks := [{P, R} | _]} = G, Toks, All) ->
    {ok, #{reductions := N0}} = peer:call(P, erllama_tp_rank, stats, [R]),
    _ = run_t(G, Toks, All),
    {ok, #{reductions := N1}} = peer:call(P, erllama_tp_rank, stats, [R]),
    N1 - N0.

deferred_copy(G) ->
    {_, ok} = op(G, {cp, 0, 1}),
    {First, {ok, _}} = op(G, {decode, 1, [13], false}),
    {Second, {ok, _}} = op(G, {decode, 1, [13], false}),
    {_, ok} = op(G, {drop, 1}),
    {round(First), round(Second)}.

summarize(Xs) ->
    maps:from_list(
        [{n, length(Xs)}] ++
            [{K, lists:sort([maps:get(K, X) || X <- Xs])} || K <- [e2e, dec, red, rest]]
    ).

med(K, S) -> pct(maps:get(K, S), 50).

fmt(K, S) ->
    L = maps:get(K, S),
    io_lib:format("median ~.1f ms, p95 ~.1f ms", [pct(L, 50), pct(L, 95)]).

pct([], _) ->
    0.0;
pct(Sorted, P) ->
    lists:nth(max(1, min(length(Sorted), round(P / 100 * length(Sorted) + 0.5))), Sorted).

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
    with_group(Config, copts(), Fun).

with_group(Config, COpts, Fun) ->
    Hosts = ?config(hosts, Config),
    Model = ?config(model, Config),
    erllama_tp_hosts:with_serving(Hosts, fun(S) ->
        #{rank_peers := Peers, rank_pids := Pids} = erllama_tp_hosts:form(Hosts, S, 1),
        Ranks = lists:zip(Peers, Pids),
        try
            _ = [native_log(P) || P <- Peers],
            Before = [avail(P) || P <- Peers],
            [
                ok = R
             || R <- erllama_tp_hosts:parallel([
                    fun() ->
                        peer:call(
                            P,
                            erllama_tp_rank,
                            load,
                            [R, erllama_tp_hosts:home(P, Model), {?MOPTS, COpts}],
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
op(G, Op) ->
    {Ms, Result, _PerRank} = op_t(G, Op),
    {Ms, Result}.

%% op/2 with each rank's own times: `{Ms, Result, [{EndToEndMs, DecodeUs,
%% ReduceUs}]}' for a decode (decode and reduction as the NIF measured
%% them), `{Ms, Result, [{EndToEndMs}]}' otherwise. A decode's result is
%% `{ok, Logits}'.
op_t(#{ranks := Ranks}, Op) ->
    Res = erllama_tp_hosts:parallel([
        fun() -> peer:call(P, erllama_tp_rank, seq_op, [R, Op], 600000) end
     || {P, R} <- Ranks
    ]),
    Split = [split_times(R) || R <- Res],
    [{First, _} | _] = Split,
    [?assertEqual(First, X, Op) || {X, _} <- Split],
    {lists:max([Ms || {Ms, _} <- Res]), First, [T || {_, T} <- Split]}.

split_times({Ms, {ok, Bin, DecUs, RedUs}}) -> {{ok, Bin}, {Ms, DecUs, RedUs}};
split_times({Ms, Other}) -> {Other, {Ms}}.

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
