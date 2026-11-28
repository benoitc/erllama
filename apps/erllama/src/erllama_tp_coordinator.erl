%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
-module(erllama_tp_coordinator).
-moduledoc """
Form and watch one generation of a tensor-parallel group.

Start the coordinator on the serving node. Pass `{call, Coord}` to
each rank and use `watch/2` once all ranks have joined. Give each rank
its own Erlang node: an abnormal rank exit terminates the remaining
rank nodes so that a blocked native collective can finish.

After a failure, read `lost/1` for the first watched rank that exited.
Stop the coordinator and create a new one for the next generation.
""".
-behaviour(gen_server).

-export([start/2, start/4, start_link/4, stop/1, join_fun/2, watch/2, members/2, snapshot/2, lost/1]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-record(state, {
    key :: {binary(), non_neg_integer()} | undefined,
    members = #{} :: #{non_neg_integer() => {reference(), pid(), pid()}},
    world :: pos_integer(),
    timeout :: non_neg_integer(),
    %% Round => #{Rank => {From, Bytes}}
    rounds = #{} :: #{
        pos_integer() | identity => #{non_neg_integer() => {gen_server:from(), binary()}}
    },
    timers = #{} :: #{pos_integer() | identity => reference()},
    ready = #{} :: #{non_neg_integer() => pos_integer()},
    failed :: term(),
    watched = #{} :: #{reference() => pid()},
    lost :: {pid(), term()} | undefined
}).

-type state() :: #state{}.

-doc "Start a coordinator. Each round has TimeoutMs from its first arrival.".
-spec start(pos_integer(), non_neg_integer()) -> {ok, pid()} | {error, term()}.
start(World, TimeoutMs) when is_integer(World), World > 0, is_integer(TimeoutMs), TimeoutMs >= 0 ->
    gen_server:start(?MODULE, {World, TimeoutMs}, []).

-doc "Start a named coordinator for configured ranks of one group and generation.".
-spec start(atom(), {binary(), non_neg_integer()}, pos_integer(), non_neg_integer()) ->
    {ok, pid()} | {error, term()}.
start(Name, {Id, Gen} = Key, World, Ms) when
    is_atom(Name),
    is_binary(Id),
    byte_size(Id) > 0,
    is_integer(Gen),
    Gen >= 0,
    is_integer(World),
    World > 0,
    is_integer(Ms),
    Ms >= 0
->
    gen_server:start({local, Name}, ?MODULE, {Key, World, Ms}, []).

-doc false.
-spec start_link(atom(), {binary(), non_neg_integer()}, pos_integer(), pos_integer()) ->
    gen_server:start_ret().
start_link(Name, Key, World, Ms) ->
    gen_server:start_link({local, Name}, ?MODULE, {Key, World, Ms}, []).

-doc "Stop this generation's coordinator.".
-spec stop(pid()) -> ok.
stop(Coord) ->
    gen_server:stop(Coord).

-doc "Return the bootstrap exchange function for Rank.".
-spec join_fun(pid(), non_neg_integer()) ->
    fun((pos_integer(), binary()) -> {ok, binary()} | {error, term()}).
join_fun(Coord, Rank) ->
    fun(Round, Bytes) -> gen_server:call(Coord, {exchange, Rank, Round, Bytes}, infinity) end.

-doc "Monitor the complete list of rank processes, one per dedicated rank node.".
-spec watch(pid(), [pid()]) -> ok | {error, term()}.
watch(Coord, RankPids) ->
    gen_server:call(Coord, {watch, RankPids}).

-doc "Read the pid and exit reason of the first watched rank lost, or undefined.".
-spec lost(pid()) -> {pid(), term()} | undefined.
lost(Coord) ->
    gen_server:call(Coord, lost).

-doc "Get the configured rank roles in rank order for this generation.".
-spec members(pid(), {binary(), non_neg_integer()}) -> {ok, [pid()]} | {error, term()}.
members(Coord, Key) ->
    gen_server:call(Coord, {members, Key}).

-doc "Read the loaded ranks, their sequence capacity and the first failure of this generation.".
-spec snapshot(pid(), {binary(), non_neg_integer()}) ->
    {ok, map(), term()} | {error, wrong_generation}.
snapshot(Coord, Key) ->
    gen_server:call(Coord, {snapshot, Key}).

-spec init({pos_integer(), non_neg_integer()} | {term(), pos_integer(), non_neg_integer()}) ->
    {ok, state()}.
init({World, TimeoutMs}) ->
    {ok, #state{world = World, timeout = TimeoutMs}};
init({Key, World, TimeoutMs}) ->
    {ok, #state{key = Key, world = World, timeout = TimeoutMs}}.

handle_call({snapshot, Key}, _, #state{key = Key, ready = Ready, failed = Why} = S) ->
    {reply, {ok, Ready, Why}, S};
handle_call({snapshot, _}, _, S) ->
    {reply, {error, wrong_generation}, S};
handle_call({resolve, Key}, _From, #state{key = Key} = S) when Key =/= undefined ->
    {reply, {ok, self()}, S};
handle_call({resolve, _}, _From, S) ->
    {reply, {error, wrong_generation}, S};
handle_call({members, Key}, _, #state{key = Key} = S) ->
    {reply, member_roles(S), S};
handle_call({members, _}, _, S) ->
    {reply, {error, wrong_generation}, S};
handle_call({enroll, Rank, World, Owner, Signature}, From, S) ->
    enroll(Rank, World, Owner, Signature, From, S);
handle_call({exchange, _Rank, _Round, _Bytes}, _From, #state{failed = F} = S) when
    F =/= undefined
->
    {reply, {error, F}, S};
handle_call({exchange, Rank, Round, Bytes}, From, #state{world = W} = S) when
    is_integer(Rank), Rank >= 0, Rank < W, is_integer(Round), Round > 0, is_binary(Bytes)
->
    exchange(Rank, Round, Bytes, From, S);
handle_call({exchange, _Rank, _Round, _Bytes}, _From, S) ->
    {reply, {error, bad_rank}, S};
handle_call({watch, Pids}, _From, S) ->
    watch_ranks(Pids, S);
handle_call(lost, _From, #state{lost = L} = S) ->
    {reply, L, S}.

member_roles(#state{failed = Why}) when Why =/= undefined ->
    {error, Why};
member_roles(#state{members = Ms, world = W}) when map_size(Ms) =:= W ->
    {ok, [Owner || _ := {_, _, Owner} <- maps:iterator(Ms, ordered)]};
member_roles(_) ->
    {error, group_forming}.

watch_ranks(_, #state{failed = F} = S) when F =/= undefined ->
    {reply, {error, F}, S};
watch_ranks(Pids, #state{watched = Ws} = S) when map_size(Ws) > 0 ->
    case is_list(Pids) andalso lists:sort(Pids) =:= lists:sort(maps:values(Ws)) of
        true -> {reply, ok, S};
        false -> {reply, {error, already_watching}, S}
    end;
watch_ranks(Pids, #state{world = W} = S) when length(Pids) =:= W ->
    case lists:all(fun is_pid/1, Pids) andalso length(lists:usort(Pids)) =:= W of
        true ->
            Monitors = maps:from_list([{monitor(process, P), P} || P <- Pids]),
            {reply, ok, S#state{watched = Monitors}};
        false ->
            {reply, {error, bad_ranks}, S}
    end;
watch_ranks(_, S) ->
    {reply, {error, bad_ranks}, S}.

enroll(_, _, _, _, _, #state{failed = Why} = S) when Why =/= undefined ->
    {reply, {error, Why}, S};
enroll(Rank, W, _, _, _, #state{world = World} = S) when W =/= World ->
    Why = {incompatible_ranks, [Rank]},
    {reply, {error, Why}, fail(Why, S)};
enroll(Rank, _, _, _, _, #state{members = Members} = S) when is_map_key(Rank, Members) ->
    {reply, {error, duplicate_rank}, S};
enroll(
    Rank, World, Owner, Signature, {Pid, _} = From, #state{key = Key, members = Members} = S
) when
    Key =/= undefined,
    is_integer(Rank),
    Rank >= 0,
    Rank < World,
    is_pid(Owner),
    node(Owner) =:= node(Pid),
    is_binary(Signature)
->
    Member = {monitor(process, Pid), Pid, Owner},
    exchange(Rank, identity, Signature, From, S#state{members = Members#{Rank => Member}});
enroll(_, _, _, _, _, S) ->
    {reply, {error, bad_rank}, S}.

handle_cast(
    {rank_ready, Rank, Owner, Slots}, #state{members = Ms, ready = Ready, failed = undefined} = S
) when
    element(3, map_get(Rank, Ms)) =:= Owner, is_integer(Slots), Slots > 0
->
    {noreply, S#state{ready = Ready#{Rank => Slots}}};
handle_cast(_Msg, S) ->
    {noreply, S}.

handle_info({timeout, Ref, Round}, #state{timers = T} = S) when map_get(Round, T) =:= Ref ->
    {noreply, fail(join_timeout, S)};
handle_info({'DOWN', Ref, process, Pid, Why}, #state{watched = Ws, lost = undefined} = S) when
    map_get(Ref, Ws) =:= Pid
->
    Others = maps:values(maps:remove(Ref, Ws)),
    _ = stop_remaining(Why, Others),
    {noreply, fail({tp_rank_lost, Why}, S#state{lost = {Pid, Why}})};
handle_info({'DOWN', Ref, process, Pid, Reason}, #state{members = Members, failed = undefined} = S) ->
    case [R || R := {M, P, _} <- Members, M =:= Ref, P =:= Pid] of
        [Rank] ->
            Why = erllama_tp_rank_role:loss_reason(Rank, Reason),
            {noreply, fail(Why, S#state{lost = {Pid, Reason}})};
        [] ->
            {noreply, S}
    end;
handle_info(_Msg, S) ->
    {noreply, S}.

stop_remaining(normal, Others) ->
    [
        spawn(fun() ->
            try
                erllama_tp_rank:stop(P)
            catch
                exit:_ -> ok
            end
        end)
     || P <- Others
    ];
stop_remaining(_, Others) ->
    [erllama_tp_rank:kill(P) || P <- Others].

fail(_, #state{failed = First} = S) when First =/= undefined -> S;
fail(Why, #state{rounds = Rounds, timers = Timers, members = Members} = S) ->
    _ = [Owner ! {tp_group_lost, self(), Why} || _ := {_, _, Owner} <- Members],
    _ = [erlang:cancel_timer(Ref) || _ := Ref <- Timers],
    [gen_server:reply(From, {error, Why}) || _ := Recs <- Rounds, _ := {From, _} <- Recs],
    S#state{rounds = #{}, timers = #{}, failed = Why}.

exchange(Rank, Round, Bytes, From, #state{world = W, rounds = Rs} = S) ->
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
    end.

arm(Round, true, #state{timers = T, timeout = Ms} = S) ->
    S#state{timers = T#{Round => erlang:start_timer(Ms, self(), Round)}};
arm(_Round, false, S) ->
    S.

%% Every rank's record is in: answer all of them, in rank order.
complete(identity, Recs, S) ->
    {_, Baseline} = maps:get(0, Recs),
    case lists:sort([Rank || Rank := {_, Signature} <- Recs, Signature =/= Baseline]) of
        [] ->
            [gen_server:reply(From, ok) || _ := {From, _} <- Recs],
            _ = erlang:cancel_timer(maps:get(identity, S#state.timers)),
            S#state{
                rounds = maps:remove(identity, S#state.rounds),
                timers = maps:remove(identity, S#state.timers)
            };
        Bad ->
            fail({incompatible_ranks, Bad}, S#state{rounds = (S#state.rounds)#{identity => Recs}})
    end;
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
