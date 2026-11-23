%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% The bootstrap exchange of a tensor-parallel group, without a GPU:
%% the native callback a communicator's init calls, the agent that
%% serves it and the test coordinator, through
%% erllama_tp:bootstrap_selftest/5 (the same bridge erllama_tp:open/5
%% uses). Every rank is a process of this VM. The GPU side is
%% erllama_tp_nccl_SUITE (opt-in, on CUDA hosts).
-module(erllama_tp_bootstrap_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0]).
-export([
    ranks_gather_in_rank_order/1,
    missing_rank_fails_within_the_deadline/1,
    silent_coordinator_hits_the_native_deadline/1,
    bad_join_reply_fails_at_once/1,
    rank_lost_between_rounds/1,
    duplicate_rank_is_refused/1,
    no_communicator_in_this_build/1
]).

all() ->
    [
        ranks_gather_in_rank_order,
        missing_rank_fails_within_the_deadline,
        silent_coordinator_hits_the_native_deadline,
        bad_join_reply_fails_at_once,
        rank_lost_between_rounds,
        duplicate_rank_is_refused,
        no_communicator_in_this_build
    ].

%% Three ranks, four rounds: every rank gets every rank's record, in
%% rank order, each round.
ranks_gather_in_rank_order(_Config) ->
    {ok, C} = erllama_tp_coordinator:start(3, 5000),
    Results = run_ranks(C, 3, 4, 5000, fun(R) -> <<R, "rank">> end),
    Expected = lists:duplicate(4, <<0, "rank", 1, "rank", 2, "rank">>),
    ?assertEqual([{ok, Expected}, {ok, Expected}, {ok, Expected}], Results),
    erllama_tp_coordinator:stop(C).

%% Only one rank of two joins: it fails at the coordinator's deadline,
%% well before its own.
missing_rank_fails_within_the_deadline(_Config) ->
    {ok, C} = erllama_tp_coordinator:start(2, 300),
    {Ms, Result} = timed(fun() ->
        erllama_tp:bootstrap_selftest(2, <<"x">>, 1, erllama_tp_coordinator:join_fun(C, 0), 10000)
    end),
    ?assertEqual({error, join_timeout}, Result),
    ?assert(Ms < 2000, Ms),
    erllama_tp_coordinator:stop(C).

%% A coordinator that never answers: the native wait ends at the
%% open's own deadline.
silent_coordinator_hits_the_native_deadline(_Config) ->
    Never = fun(_Round, _Bytes) -> timer:sleep(infinity) end,
    {Ms, Result} = timed(fun() -> erllama_tp:bootstrap_selftest(2, <<"x">>, 1, Never, 300) end),
    ?assertEqual({error, timeout}, Result),
    ?assert(Ms < 3000, Ms).

%% A join that answers neither `{ok, _}' nor `{error, _}': the rank
%% fails at once with what it got, not at its deadline.
bad_join_reply_fails_at_once(_Config) ->
    Bad = fun(_Round, _Bytes) -> nonsense end,
    {Ms, Result} = timed(fun() -> erllama_tp:bootstrap_selftest(2, <<"x">>, 1, Bad, 10000) end),
    ?assertEqual({error, {bad_join_reply, nonsense}}, Result),
    ?assert(Ms < 2000, Ms).

%% Rank 1 completes round 1, then its agent dies: rank 0 fails at the
%% coordinator's deadline for round 2, rank 1 at its native deadline.
%% Both calls return.
rank_lost_between_rounds(_Config) ->
    {ok, C} = erllama_tp_coordinator:start(2, 500),
    Join1 = erllama_tp_coordinator:join_fun(C, 1),
    Dying = fun
        (1, Bytes) -> Join1(1, Bytes);
        (_, _) -> exit(self(), kill)
    end,
    Self = self(),
    spawn(fun() ->
        Self !
            {rank, 0,
                erllama_tp:bootstrap_selftest(
                    2, <<"a">>, 2, erllama_tp_coordinator:join_fun(C, 0), 5000
                )}
    end),
    spawn(fun() -> Self ! {rank, 1, erllama_tp:bootstrap_selftest(2, <<"b">>, 2, Dying, 1500)} end),
    R0 =
        receive
            {rank, 0, X} -> X
        after 10000 -> ct:fail(rank0_stuck)
        end,
    R1 =
        receive
            {rank, 1, Y} -> Y
        after 10000 -> ct:fail(rank1_stuck)
        end,
    ?assertEqual({error, join_timeout}, R0),
    ?assertEqual({error, timeout}, R1),
    erllama_tp_coordinator:stop(C).

%% Two ranks claiming the same rank: the second is refused, and the
%% refusal reaches the caller with the coordinator's reason.
duplicate_rank_is_refused(_Config) ->
    {ok, C} = erllama_tp_coordinator:start(2, 2000),
    Self = self(),
    spawn(fun() ->
        Self !
            {first,
                erllama_tp:bootstrap_selftest(
                    2, <<"a">>, 1, erllama_tp_coordinator:join_fun(C, 0), 5000
                )}
    end),
    timer:sleep(100),
    ?assertEqual(
        {error, duplicate_rank},
        erllama_tp:bootstrap_selftest(2, <<"b">>, 1, erllama_tp_coordinator:join_fun(C, 0), 5000)
    ),
    receive
        {first, _} -> ok
    after 10000 -> ct:fail(first_stuck)
    end,
    erllama_tp_coordinator:stop(C).

%% Without a GPU backend that has per-process communicators (no CUDA
%% with NCCL here), caps/0 says so and open/5 refuses.
no_communicator_in_this_build(_Config) ->
    {ok, Caps} = erllama_tp:caps(),
    ct:log("tp caps: ~p", [Caps]),
    case Caps of
        #{init_rank := true} ->
            {skip, "this build has per-process communicators; see erllama_tp_nccl_SUITE"};
        _ ->
            {ok, C} = erllama_tp_coordinator:start(1, 1000),
            ?assertEqual(
                {error, not_supported},
                erllama_tp:open(0, 0, 1, erllama_tp_coordinator:join_fun(C, 0), 1000)
            ),
            erllama_tp_coordinator:stop(C)
    end.

%% -----------------------------------------------------------------------------

run_ranks(C, World, Rounds, Timeout, Payload) ->
    Self = self(),
    [
        spawn(fun() ->
            Self !
                {rank, R,
                    erllama_tp:bootstrap_selftest(
                        World, Payload(R), Rounds, erllama_tp_coordinator:join_fun(C, R), Timeout
                    )}
        end)
     || R <- lists:seq(0, World - 1)
    ],
    [
        receive
            {rank, R, Res} -> Res
        after 20000 -> ct:fail({rank_stuck, R})
        end
     || R <- lists:seq(0, World - 1)
    ].

timed(Fun) ->
    T0 = erlang:monotonic_time(millisecond),
    R = Fun(),
    {erlang:monotonic_time(millisecond) - T0, R}.
