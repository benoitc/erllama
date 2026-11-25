%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Opt-in, on CUDA hosts (ERLLAMA_TP_HOSTS, see erllama_tp_nccl_SUITE):
%% what a split model does when its group fails while serving.
%%
%% Every case starts its own nodes: a serving node on the first host
%% (erllama with erllama_model_tp, the coordinator, no GPU work) and
%% one rank node per host. The serving node must outlive any rank.
-module(erllama_tp_failure_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([
    rank_lost_mid_stream/1,
    stale_duplicate_and_gap/1,
    cancel_during_prefill/1
]).

-define(NO_CACHE, #{
    min_tokens => infinity, cold_min_tokens => infinity, continued_interval => infinity
}).
-define(PROMPT, <<"The quick brown fox">>).
%% A failure must end the request within this.
-define(BOUND_MS, 30000).

all() ->
    [rank_lost_mid_stream, stale_duplicate_and_gap, cancel_during_prefill].

init_per_suite(Config) ->
    case erllama_tp_hosts:hosts() of
        false -> {skip, "set ERLLAMA_TP_HOSTS to the CUDA hosts (see scripts/tp_hosts.sh)"};
        Hosts -> [{hosts, Hosts} | Config]
    end.

end_per_suite(_Config) ->
    ok.

%% -----------------------------------------------------------------------------

%% Rank 1's OS process killed while rank 0 streams: the request gets
%% one terminal error within the bound, the serving node still answers,
%% the model is gone, and a group formed again (new nodes, the next
%% generation) serves the same greedy completion as before.
rank_lost_mid_stream(Config) ->
    with_serving(Config, fun(S) ->
        G1 = serve(Config, S, 1),
        Before = complete(S, G1),
        Long = long_prompt(20),
        #{rank_pids := [_, R1 | _]} = G1,
        Run = peer:call(
            S,
            erllama_tp_stream,
            run_with,
            [model_id(G1), Long, 256, {10, {erllama_tp_stream, kill, [R1]}}, 10000],
            ?BOUND_MS * 3
        ),
        ct:pal("rank 1 killed after 10 tokens: ~p", [Run]),
        ?assertMatch([{error, _}], maps:get(terminals, Run)),
        ?assert(maps:get(to_terminal_ms, Run) < ?BOUND_MS),
        %% The serving node answers and the model has stopped.
        {Ms, Models} = timer:tc(
            fun() -> peer:call(S, erllama, list_models, [], 5000) end, millisecond
        ),
        Ids = [maps:get(id, M, undefined) || M <- Models],
        ct:pal("models after the loss (~b ms): ~p", [Ms, Ids]),
        ?assertNot(lists:member(model_id(G1), Ids)),
        ct:pal("coordinator: ~p", [peer:call(S, erllama_tp_coordinator, lost, [coord(G1)])]),
        stop_group(G1),
        G2 = serve(Config, S, 2),
        ?assertEqual(Before, complete(S, G2)),
        stop_group(G2)
    end).

%% Commands that do not fit the follower's sequence, sent while the
%% group serves: one of the previous generation and one applied already
%% are dropped before any GPU work (no reduction, the same answers);
%% one past a gap stops the follower, and the group fails as for a lost
%% rank.
stale_duplicate_and_gap(Config) ->
    with_serving(Config, fun(S) ->
        G = serve(Config, S, 3),
        Before = complete(S, G),
        #{rank_pids := [_, R1 | _], rank_peers := [_, P1 | _]} = G,
        #{applied := Applied} = peer:call(P1, erllama_tp_rank, follow_stats, [R1]),
        {ok, #{reductions := Red0}} = peer:call(P1, erllama_tp_rank, stats, [R1]),
        Bogus = {decode, <<1:32/signed-native, 0:32, 0:32, 0:32, 1:32>>},
        peer:call(P1, erlang, send, [R1, {erllama_tp, 2, Applied + 1, Bogus}]),
        peer:call(P1, erlang, send, [R1, {erllama_tp, 3, Applied, Bogus}]),
        Stats = peer:call(P1, erllama_tp_rank, follow_stats, [R1]),
        {ok, #{reductions := Red1}} = peer:call(P1, erllama_tp_rank, stats, [R1]),
        ct:pal("follower after a stale and a duplicate command: ~p", [Stats]),
        ?assertMatch(#{stale := 1, duplicates := 1}, Stats),
        ?assertEqual(Red0, Red1),
        ?assertEqual(Before, complete(S, G)),
        %% A gap: the follower stops, the coordinator ends the group, the
        %% next request fails once and the model stops.
        #{applied := Now} = peer:call(P1, erllama_tp_rank, follow_stats, [R1]),
        peer:call(P1, erlang, send, [R1, {erllama_tp, 3, Now + 5, Bogus}]),
        {Ms, Result} = timer:tc(
            fun() ->
                peer:call(
                    S,
                    erllama,
                    complete,
                    [model_id(G), ?PROMPT, #{response_tokens => 8, temperature => 0.0}],
                    ?BOUND_MS * 2
                )
            end,
            millisecond
        ),
        ct:pal("after a gap (~b ms): ~p; coordinator: ~p", [
            Ms, Result, peer:call(S, erllama_tp_coordinator, lost, [coord(G)])
        ]),
        ?assertMatch({error, _}, Result),
        ?assert(Ms < ?BOUND_MS),
        stop_group(G)
    end).

%% A cancel while a long prompt is being prefilled: the step in flight
%% completes on every rank (a split model's decode is not interrupted),
%% the request ends once, every command rank 0 sent was applied by the
%% follower, and the next request gives the same tokens as before.
cancel_during_prefill(Config) ->
    with_serving(Config, fun(S) ->
        G = serve(Config, S, 4),
        Before = complete(S, G),
        Run = peer:call(
            S,
            erllama_tp_stream,
            run_with,
            [model_id(G), long_prompt(80), 64, {0, {erllama, cancel, []}}, 5000],
            ?BOUND_MS * 3
        ),
        ct:pal("cancelled during prefill: ~p", [Run]),
        ?assertMatch([_], maps:get(terminals, Run)),
        ?assertEqual(Before, complete(S, G)),
        #{rank_pids := [R0, R1 | _], rank_peers := [P0, P1 | _]} = G,
        Stats = peer:call(P1, erllama_tp_rank, follow_stats, [R1]),
        {ok, #{sizes := Sizes0}} = peer:call(P0, erllama_tp_rank, stats, [R0]),
        {ok, #{sizes := Sizes1}} = peer:call(P1, erllama_tp_rank, stats, [R1]),
        ct:pal("follower: ~p; reductions ~b and ~b", [Stats, length(Sizes0), length(Sizes1)]),
        ?assertMatch(#{stale := 0, duplicates := 0}, Stats),
        ?assertEqual(Sizes0, Sizes1),
        stop_group(G)
    end).

%% -----------------------------------------------------------------------------

with_serving(Config, Fun) ->
    erllama_tp_hosts:with_serving(?config(hosts, Config), Fun).

serve(Config, S, Gen) ->
    erllama_tp_hosts:serve(?config(hosts, Config), S, Gen, ?NO_CACHE).

complete(S, G) ->
    erllama_tp_hosts:complete(S, G, ?PROMPT).

stop_group(G) ->
    erllama_tp_hosts:stop_group(G).

model_id(#{id := Id}) -> Id.
coord(#{coord := C}) -> C.

long_prompt(N) ->
    erllama_tp_hosts:long_prompt(N).
