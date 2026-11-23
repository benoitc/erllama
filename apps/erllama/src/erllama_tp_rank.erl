%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
-module(erllama_tp_rank).
-moduledoc false.
%% Experimental: a process owning one rank's erllama_tp handle on its
%% node, so a coordinator elsewhere can drive the rank by pid (a NIF
%% resource cannot be used from another node). Not linked to whoever
%% starts it: the rank lives until stop/1, or until its node goes.
-behaviour(gen_server).

-export([start/1, matvec/3, reduce/2, info/1, stop/1]).
-export([init/1, handle_call/3, handle_cast/2, terminate/2]).

-record(state, {handle :: erllama_tp:rank(), rank :: non_neg_integer(), world :: pos_integer()}).
-type state() :: #state{}.

%% Joins the group before returning: `{ok, Pid}' once every rank joined,
%% `{error, Reason}' as erllama_tp:open/5 gives it. Opts: gpu (default
%% 0), rank, world, join, timeout (ms, default 60000).
-spec start(map()) -> {ok, pid()} | {error, term()}.
start(Opts) ->
    gen_server:start(?MODULE, Opts, []).

-spec matvec(pid(), [[float()]], [float()]) -> {ok, [float()], [float()]} | {error, term()}.
matvec(Pid, W, X) -> gen_server:call(Pid, {matvec, W, X}, infinity).

-spec reduce(pid(), [float()] | binary()) -> {ok, [float()] | binary()} | {error, term()}.
reduce(Pid, Values) -> gen_server:call(Pid, {reduce, Values}, infinity).

%% The rank, its OS process and its GPU's memory as the backend reports it.
-spec info(pid()) -> map().
info(Pid) -> gen_server:call(Pid, info, infinity).

-spec stop(pid()) -> ok.
stop(Pid) -> gen_server:stop(Pid).

-spec init(map()) -> {ok, state()} | {stop, term()}.
init(#{rank := Rank, world := World, join := Join} = Opts) ->
    Gpu = maps:get(gpu, Opts, 0),
    case erllama_tp:open(Gpu, Rank, World, Join, maps:get(timeout, Opts, 60000)) of
        {ok, H} -> {ok, #state{handle = H, rank = Rank, world = World}};
        {error, Why} -> {stop, Why}
    end.

handle_call({matvec, W, X}, _From, #state{handle = H} = S) ->
    {reply, erllama_tp:matvec(H, W, X), S};
handle_call({reduce, V}, _From, #state{handle = H} = S) ->
    {reply, erllama_tp:reduce(H, V), S};
handle_call(info, _From, #state{rank = R, world = W} = S) ->
    Gpus =
        case erllama:list_devices() of
            {ok, Ds} ->
                [maps:with([name, free_b, total_b], D) || #{type := T} = D <- Ds, T =/= cpu];
            _ ->
                []
        end,
    {reply, #{rank => R, world => W, os_pid => os:getpid(), node => node(), devices => Gpus}, S}.

handle_cast(_Msg, S) ->
    {noreply, S}.

terminate(_Reason, #state{handle = H}) ->
    erllama_tp:close(H).
