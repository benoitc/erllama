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
%%
%% Serving: the serving rank loads the model through erllama
%% (serve/5), so it is used like any loaded model (complete, stream,
%% cancel), and samples. Its context sends every batch and KV change to
%% a forwarder that sends them, in order, to the other ranks, which have
%% loaded the same file with load/3 and apply them (tp_follow).
-behaviour(gen_server).

-export([start/1, matvec/3, reduce/2, info/1, stop/1]).
-export([load/3, decode/2, stats/1, ref_load/3, ref_decode/2]).
-export([serve/5, describe/3, time_reduce/3]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

%% A split model's KV is not saved or restored yet (checkpoints must be
%% coordinated across the ranks).
-define(NO_CACHE, #{
    min_tokens => infinity, cold_min_tokens => infinity, continued_interval => infinity
}).

-record(state, {
    handle :: erllama_tp:rank(),
    rank :: non_neg_integer(),
    world :: pos_integer(),
    model :: erllama_nif:model_ref() | undefined,
    ctx :: erllama_nif:context_ref() | undefined,
    ref_model :: erllama_nif:model_ref() | undefined,
    ref_ctx :: erllama_nif:context_ref() | undefined,
    %% The last command applied from the serving rank.
    seq = 0 :: non_neg_integer(),
    serving :: binary() | undefined
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

%% On the serving rank: load `Path' through erllama as `ModelId',
%% split over the group, once every rank in `Followers' (pids of the
%% other ranks' processes, which have run load/3 with the same file and
%% options) has been checked to hold the same model.
-spec serve(pid(), binary(), string(), {map(), map()}, [{pid(), map()}]) ->
    {ok, binary()} | {error, term()}.
serve(Pid, ModelId, Path, Opts, Followers) ->
    gen_server:call(Pid, {serve, ModelId, Path, Opts, Followers}, infinity).

%% `K' reductions of `N' values in a row, timed in this rank's process:
%% the mean in microseconds (every rank makes the same call).
-spec time_reduce(pid(), pos_integer(), pos_integer()) -> float().
time_reduce(Pid, N, K) -> gen_server:call(Pid, {time_reduce, N, K}, infinity).

%% What every rank must agree on before serving: the file (size and the
%% hash of its first MiB), the options and the engine build.
-spec describe(string(), {map(), map()}, pos_integer()) -> map().
describe(Path, {MOpts, COpts}, World) ->
    Head =
        case file:open(Path, [read, binary, raw]) of
            {ok, F} ->
                {ok, Bin} = file:read(F, 1048576),
                ok = file:close(F),
                crypto:hash(sha256, Bin);
            {error, _} = E ->
                E
        end,
    #{
        bytes => filelib:file_size(Path),
        head => Head,
        model_opts => MOpts,
        context_opts => COpts,
        world => World,
        kv_format => erllama_nif:kv_format()
    }.

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
    reply_loaded(load_with(Path, MOpts#{tp_rank => H}, COpts), split, S);
handle_call({decode, Tokens}, _From, #state{ctx = C} = S) when C =/= undefined ->
    {reply, erllama_nif:tp_decode(C, Tokens), S};
handle_call({serve, ModelId, Path, Opts, Followers}, _From, S) ->
    case serve_model(ModelId, Path, Opts, Followers, S) of
        {ok, Id} -> {reply, {ok, Id}, S#state{serving = Id}};
        {error, _} = E -> {reply, E, S}
    end;
handle_call({time_reduce, N, K}, _From, #state{handle = H} = S) ->
    Vals = <<<<1.0:32/float-native>> || _ <- lists:seq(1, N)>>,
    {ok, _} = erllama_tp:reduce(H, Vals),
    T0 = erlang:monotonic_time(microsecond),
    _ = [{ok, _} = erllama_tp:reduce(H, Vals) || _ <- lists:seq(1, K)],
    {reply, (erlang:monotonic_time(microsecond) - T0) / K, S};
handle_call(stats, _From, #state{handle = H} = S) ->
    {reply, erllama_nif:tp_stats(H), S};
handle_call({ref_load, Path, {MOpts, COpts}}, _From, S) ->
    reply_loaded(load_with(Path, MOpts, COpts), ref, S);
handle_call({ref_decode, Tokens}, _From, #state{ref_ctx = C} = S) when C =/= undefined ->
    {reply, erllama_nif:tp_decode(C, Tokens), S};
handle_call(info, _From, #state{rank = R, world = W} = S) ->
    {reply, #{rank => R, world => W, os_pid => os:getpid(), node => node(), devices => gpus()}, S};
handle_call(_Req, _From, S) ->
    {reply, {error, not_loaded}, S}.

handle_cast(_Msg, S) ->
    {noreply, S}.

%% A command from the serving rank: applied in order. A gap or a repeat
%% means this rank no longer has the serving rank's KV state.
handle_info({erllama_tp, Seq, Cmd}, #state{ctx = C, seq = Last} = S) when
    C =/= undefined, Seq =:= Last + 1
->
    case erllama_nif:tp_follow(C, Cmd) of
        ok -> {noreply, S#state{seq = Seq}};
        {error, Why} -> {stop, {follow_failed, Seq, Why}, S}
    end;
handle_info({erllama_tp, Seq, _Cmd}, #state{seq = Last} = S) ->
    {stop, {out_of_order, Last, Seq}, S};
handle_info(_Msg, S) ->
    {noreply, S}.

%% The serving rank's sink: every command to every other rank, in order.
forward(Followers) ->
    receive
        {erllama_tp, _Seq, _Cmd} = M ->
            _ = [F ! M || F <- Followers],
            forward(Followers)
    after 3600000 ->
        forward(Followers)
    end.

terminate(_Reason, #state{handle = H, serving = Id} = S) ->
    _ = Id =/= undefined andalso erllama:unload(Id),
    _ = [erllama_nif:free_context(C) || C <- [S#state.ctx, S#state.ref_ctx], C =/= undefined],
    _ = [erllama_nif:free_model(M) || M <- [S#state.model, S#state.ref_model], M =/= undefined],
    erllama_tp:close(H).

reply_loaded({ok, M, C}, split, S) -> {reply, ok, S#state{model = M, ctx = C}};
reply_loaded({ok, M, C}, ref, S) -> {reply, ok, S#state{ref_model = M, ref_ctx = C}};
reply_loaded({error, _} = E, _, S) -> {reply, E, S}.

%% Every follower must hold the same model as this rank; then this
%% rank's context sends to a forwarder that keeps them in step.
serve_model(ModelId, Path, {MOpts, COpts} = Opts, Followers, #state{handle = H, world = W}) ->
    Mine = describe(Path, Opts, W),
    case [F || {F, Desc} <- Followers, Desc =/= Mine] of
        [] ->
            Pids = [F || {F, _} <- Followers],
            Fwd = spawn_link(fun() -> forward(Pids) end),
            erllama:load_model(ModelId, #{
                backend => erllama_model_llama,
                model_path => Path,
                model_opts => MOpts#{tp_rank => H},
                context_opts => COpts#{tp_sink => Fwd},
                policy => ?NO_CACHE
            });
        Bad ->
            {error, {incompatible_ranks, Bad}}
    end.

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
