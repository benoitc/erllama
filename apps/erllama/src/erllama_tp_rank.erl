%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
-module(erllama_tp_rank).
-moduledoc false.
%% Experimental: a process owning one rank's erllama_tp handle on its
%% node, so a coordinator elsewhere can drive the rank by pid (a NIF
%% resource cannot be used from another node). Not linked to whoever
%% starts it: the rank lives until stop/1, or until its node goes.
%%
%% It can also hold a model split over the rank's group (load/3, then
%% decode/2 in lockstep with the other ranks) and, for comparisons, a
%% single-GPU reference copy of a model (ref_load/3, ref_decode/2).
-behaviour(gen_server).

-export([start/1, matvec/3, reduce/2, info/1, stop/1]).
-export([load/3, decode/2, stats/1, ref_load/3, ref_decode/2]).
-export([init/1, handle_call/3, handle_cast/2, terminate/2]).

-record(state, {
    handle :: erllama_tp:rank(),
    rank :: non_neg_integer(),
    world :: pos_integer(),
    model :: erllama_nif:model_ref() | undefined,
    ctx :: erllama_nif:context_ref() | undefined,
    ref_model :: erllama_nif:model_ref() | undefined,
    ref_ctx :: erllama_nif:context_ref() | undefined
}).
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

%% Load `Path' split over the group (every rank at once, same model and
%% options) and create its context. The options are those of
%% erllama_nif:load_model/2 and new_context/2.
-spec load(pid(), string(), {map(), map()}) -> ok | {error, term()}.
load(Pid, Path, Opts) -> gen_server:call(Pid, {load, Path, Opts}, infinity).

%% Decode `Tokens' after what the context holds; the last token's
%% logits, summed over the group (every rank makes the same call).
-spec decode(pid(), [non_neg_integer()]) -> {ok, binary()} | {error, term()}.
decode(Pid, Tokens) -> gen_server:call(Pid, {decode, Tokens}, infinity).

%% What the placeholders reserve and the reductions made so far.
-spec stats(pid()) -> {ok, map()}.
stats(Pid) -> gen_server:call(Pid, stats, infinity).

%% A single-device copy of `Path' on this rank's GPU, for comparisons.
-spec ref_load(pid(), string(), {map(), map()}) -> ok | {error, term()}.
ref_load(Pid, Path, Opts) -> gen_server:call(Pid, {ref_load, Path, Opts}, infinity).

-spec ref_decode(pid(), [non_neg_integer()]) -> {ok, binary()} | {error, term()}.
ref_decode(Pid, Tokens) -> gen_server:call(Pid, {ref_decode, Tokens}, infinity).

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
handle_call({load, Path, {MOpts, COpts}}, _From, #state{handle = H} = S) ->
    case load_with(Path, MOpts#{tp_rank => H}, COpts) of
        {ok, M, C} -> {reply, ok, S#state{model = M, ctx = C}};
        {error, _} = E -> {reply, E, S}
    end;
handle_call({decode, Tokens}, _From, #state{ctx = C} = S) when C =/= undefined ->
    {reply, erllama_nif:tp_decode(C, Tokens), S};
handle_call(stats, _From, #state{handle = H} = S) ->
    {reply, erllama_nif:tp_stats(H), S};
handle_call({ref_load, Path, {MOpts, COpts}}, _From, S) ->
    case load_with(Path, MOpts, COpts) of
        {ok, M, C} -> {reply, ok, S#state{ref_model = M, ref_ctx = C}};
        {error, _} = E -> {reply, E, S}
    end;
handle_call({ref_decode, Tokens}, _From, #state{ref_ctx = C} = S) when C =/= undefined ->
    {reply, erllama_nif:tp_decode(C, Tokens), S};
handle_call(info, _From, #state{rank = R, world = W} = S) ->
    {reply, #{rank => R, world => W, os_pid => os:getpid(), node => node(), devices => gpus()}, S};
handle_call(_Req, _From, S) ->
    {reply, {error, not_loaded}, S}.

handle_cast(_Msg, S) ->
    {noreply, S}.

terminate(_Reason, #state{handle = H} = S) ->
    _ = [erllama_nif:free_context(C) || C <- [S#state.ctx, S#state.ref_ctx], C =/= undefined],
    _ = [erllama_nif:free_model(M) || M <- [S#state.model, S#state.ref_model], M =/= undefined],
    erllama_tp:close(H).

gpus() ->
    case erllama:list_devices() of
        {ok, Ds} -> [maps:with([name, free_b, total_b], D) || #{type := T} = D <- Ds, T =/= cpu];
        _ -> []
    end.

load_with(Path, MOpts, COpts) ->
    case erllama_nif:load_model(Path, MOpts) of
        {ok, M} -> load_with_ctx(M, COpts);
        {ok, M, _Fit} -> load_with_ctx(M, COpts);
        {error, _} = E -> E
    end.

load_with_ctx(M, COpts) ->
    case erllama_nif:new_context(M, COpts) of
        {ok, C} ->
            {ok, M, C};
        {error, _} = E ->
            _ = erllama_nif:free_model(M),
            E
    end.
