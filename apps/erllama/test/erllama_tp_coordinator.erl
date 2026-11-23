%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% A test coordinator for erllama_tp groups (group formation belongs to
%% erllama_cluster; this one is for the suites in this repository).
%%
%% One coordinator per group generation, for `World' ranks. Each
%% bootstrap round, it collects one record per rank and answers every
%% rank with the records concatenated in rank order. A round that is
%% not complete within `TimeoutMs' of its first record fails: every
%% rank that joined gets `{error, join_timeout}', and so does any later
%% call, because the group is then failed.
-module(erllama_tp_coordinator).
-behaviour(gen_server).

-export([start/2, stop/1, join_fun/2]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-record(state, {
    world :: pos_integer(),
    timeout :: non_neg_integer(),
    %% Round => #{Rank => {From, Bytes}}
    rounds = #{} :: #{pos_integer() => #{non_neg_integer() => {gen_server:from(), binary()}}},
    timers = #{} :: #{pos_integer() => reference()},
    failed :: term()
}).

start(World, TimeoutMs) ->
    gen_server:start(?MODULE, {World, TimeoutMs}, []).

stop(Coord) ->
    gen_server:stop(Coord).

%% The `Join' function erllama_tp:open/5 takes, for rank `Rank'. The
%% coordinator may be on another node.
join_fun(Coord, Rank) ->
    fun(Round, Bytes) -> gen_server:call(Coord, {exchange, Rank, Round, Bytes}, infinity) end.

init({World, TimeoutMs}) ->
    {ok, #state{world = World, timeout = TimeoutMs}}.

handle_call({exchange, _Rank, _Round, _Bytes}, _From, #state{failed = F} = S) when
    F =/= undefined
->
    {reply, {error, F}, S};
handle_call({exchange, Rank, Round, Bytes}, From, #state{world = W, rounds = Rs} = S) when
    Rank >= 0, Rank < W
->
    Recs = maps:get(Round, Rs, #{}),
    case is_map_key(Rank, Recs) of
        true ->
            {reply, {error, duplicate_rank}, S};
        false ->
            Recs1 = Recs#{Rank => {From, Bytes}},
            S1 = arm(Round, map_size(Recs) =:= 0, S),
            case map_size(Recs1) =:= W of
                true -> {noreply, complete(Round, Recs1, S1)};
                false -> {noreply, S1#state{rounds = Rs#{Round => Recs1}}}
            end
    end;
handle_call({exchange, _Rank, _Round, _Bytes}, _From, S) ->
    {reply, {error, bad_rank}, S}.

handle_cast(_Msg, S) ->
    {noreply, S}.

handle_info({round_timeout, Round}, #state{rounds = Rs} = S) ->
    [gen_server:reply(From, {error, join_timeout}) || _ := {From, _} <- maps:get(Round, Rs, #{})],
    {noreply, S#state{rounds = maps:remove(Round, Rs), failed = join_timeout}};
handle_info(_Msg, S) ->
    {noreply, S}.

arm(Round, true, #state{timers = T, timeout = Ms} = S) ->
    S#state{timers = T#{Round => erlang:send_after(Ms, self(), {round_timeout, Round})}};
arm(_Round, false, S) ->
    S.

%% Every rank's record is in: answer all of them, in rank order.
complete(Round, Recs, #state{rounds = Rs, timers = T} = S) ->
    _ = erlang:cancel_timer(maps:get(Round, T)),
    Ordered = [maps:get(R, Recs) || R <- lists:seq(0, S#state.world - 1)],
    Sizes = lists:usort([byte_size(B) || {_, B} <- Ordered]),
    Reply =
        case Sizes of
            [_] -> {ok, iolist_to_binary([B || {_, B} <- Ordered])};
            _ -> {error, size_mismatch}
        end,
    [gen_server:reply(From, Reply) || {From, _} <- Ordered],
    S#state{rounds = maps:remove(Round, Rs), timers = maps:remove(Round, T)}.
