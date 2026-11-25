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
%% Serving: rank 0 hosts the model's backend (host/3) for
%% erllama_model_tp, the backend a serving VM loads the model with, so
%% it is used like any loaded model (complete, stream, cancel) while
%% the GPUs stay in other OS processes. Rank 0 samples. Its context
%% sends every batch and KV change to a forwarder that sends them, in
%% order and tagged with the group's generation, to the other ranks,
%% which have loaded the same file with load/3 and apply them
%% (tp_follow).
-behaviour(gen_server).

-export([start/1, matvec/3, reduce/2, info/1, stop/1]).
-export([load/3, decode/2, stats/1, ref_load/3, ref_decode/2]).
-export([seq_op/2]).
-export([host/3, backend/3, free_ref/2, follow_stats/1, describe/3, time_reduce/3, kill/1]).
-export([checkpoint_stats/1, fault/2]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-record(state, {
    handle :: erllama_tp:rank(),
    rank :: non_neg_integer(),
    world :: pos_integer(),
    model :: erllama_nif:model_ref() | undefined,
    ctx :: erllama_nif:context_ref() | undefined,
    ref_model :: erllama_nif:model_ref() | undefined,
    ref_ctx :: erllama_nif:context_ref() | undefined,
    %% The group's generation: commands of another one are stale.
    gen = 0 :: non_neg_integer(),
    %% The last command applied from the serving rank, and the commands
    %% dropped before any GPU work (another generation, or applied
    %% already).
    seq = 0 :: non_neg_integer(),
    stale = 0 :: non_neg_integer(),
    duplicates = 0 :: non_neg_integer(),
    %% On rank 0: the backend state hosted for erllama_model_tp, and the
    %% references (samplers) it handed out, by id.
    host :: term() | undefined,
    refs = #{} :: #{pos_integer() => reference()},
    next_ref = 1 :: pos_integer(),
    %% On rank 0: how long the last checkpoint and restore took (ms).
    ckpt = #{} :: #{atom() => number()},
    %% How long rank 0 waits for the other ranks' shares (ms).
    kv_timeout = 30000 :: timeout(),
    %% Tests only: drop this rank's share of the next checkpoint.
    drop_kv_pack = false :: boolean()
}).
-type state() :: #state{}.

%% Joins the group before returning: `{ok, Pid}' once every rank joined,
%% `{error, Reason}' as erllama_tp:open/5 gives it. Opts: gpu (default
%% 0), rank, world, join, timeout (ms, default 60000), generation (the
%% group's, default 0).
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

%% An operation on the split context's sequences (every rank makes the
%% same call): `{decode, Seq, Tokens, AllRows}' (logits summed over the
%% group), `{cp, Src, Dst}', `{drop, Seq}', `{trim, Seq, Keep}' (keep
%% the first Keep positions), `{pos_max, Seq}', `{diag, arm | read |
%% perf}' (erllama_nif:ctx_diag/2). Returns `{Ms, Result}'.
-spec seq_op(pid(), tuple()) -> {number(), term()}.
seq_op(Pid, Op) -> gen_server:call(Pid, {seq_op, Op}, infinity).

%% What the placeholders reserve and the reductions made so far.
-spec stats(pid()) -> {ok, map()}.
stats(Pid) -> gen_server:call(Pid, stats, infinity).

%% A single-device copy of `Path' on this rank's GPU, for comparisons.
-spec ref_load(pid(), string(), {map(), map()}) -> ok | {error, term()}.
ref_load(Pid, Path, Opts) -> gen_server:call(Pid, {ref_load, Path, Opts}, infinity).

-spec ref_decode(pid(), [non_neg_integer()]) -> {ok, binary()} | {error, term()}.
ref_decode(Pid, Tokens) -> gen_server:call(Pid, {ref_decode, Tokens}, infinity).

%% On rank 0: load the model of `Config' (erllama_model_llama's) split
%% over the group and host its backend, once every rank in `Followers'
%% (pids of the other ranks' processes, which have run load/3 with the
%% same file and options) has been checked to hold the same model.
-spec host(pid(), map(), [{pid(), map()}]) -> ok | {error, term()}.
host(Pid, Config, Followers) ->
    gen_server:call(Pid, {hosting, {host, Config, Followers}}, infinity).

%% Run backend callback `Fn' on the hosted state (references in `Args'
%% and in the result travel as `{erllama_tp_ref, Pid, Id}').
-spec backend(pid(), atom(), [term()]) -> term().
backend(Pid, Fn, Args) ->
    gen_server:call(Pid, {hosting, {backend, Fn, Args}}, infinity).

%% Release a reference handed out by backend/3 (a sampler or an MTP
%% drafter).
-spec free_ref(pid(), pos_integer()) -> ok | {error, term()}.
free_ref(Pid, Id) ->
    gen_server:call(Pid, {hosting, {free_ref, Id}}, infinity).

%% On rank 0: how long the last checkpoint and restore took, in ms.
-spec checkpoint_stats(pid()) -> #{atom() => number()}.
checkpoint_stats(Pid) -> gen_server:call(Pid, {hosting, checkpoint_stats}, infinity).

%% Tests only: `drop_kv_pack' makes this rank send no share for the next
%% checkpoint, as if the save were interrupted after the other shares.
-spec fault(pid(), drop_kv_pack) -> ok.
fault(Pid, drop_kv_pack) -> gen_server:call(Pid, {hosting, {fault, drop_kv_pack}}, infinity).

%% The serving rank's commands this rank applied and dropped.
-spec follow_stats(pid()) -> #{atom() => non_neg_integer()}.
follow_stats(Pid) -> gen_server:call(Pid, follow_stats, infinity).

%% End the rank's OS process at once, even if a native call holds it:
%% the way out of a collective that lost a peer.
-spec kill(pid() | node()) -> ok.
kill(Pid) when is_pid(Pid) -> kill(node(Pid));
kill(Node) when is_atom(Node) ->
    _ = erpc:cast(Node, fun() -> os:cmd("kill -9 " ++ os:getpid()) end),
    ok.

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
        {ok, H} ->
            {ok, #state{
                handle = H, rank = Rank, world = World, gen = maps:get(generation, Opts, 0)
            }};
        {error, Why} ->
            {stop, Why}
    end.

handle_call({matvec, W, X}, _From, #state{handle = H} = S) ->
    {reply, erllama_tp:matvec(H, W, X), S};
handle_call({reduce, V}, _From, #state{handle = H} = S) ->
    {reply, erllama_tp:reduce(H, V), S};
handle_call({load, Path, {MOpts, COpts}}, _From, #state{handle = H} = S) ->
    reply_loaded(load_with(Path, MOpts#{tp_rank => H}, split_context_opts(COpts)), split, S);
handle_call({decode, Tokens}, _From, #state{ctx = C} = S) when C =/= undefined ->
    {reply, erllama_nif:tp_decode(C, Tokens), S};
handle_call({seq_op, Op}, _From, #state{ctx = C} = S) when C =/= undefined ->
    {reply, timed(fun() -> seq_op_ctx(C, Op) end), S};
handle_call({hosting, Req}, _From, S) ->
    hosting(Req, S);
handle_call(follow_stats, _From, #state{seq = Seq, stale = St, duplicates = D} = S) ->
    {reply, #{applied => Seq, stale => St, duplicates => D}, S};
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

%% A command from the serving rank, decided before any GPU work: one of
%% another generation is stale and one applied already is a duplicate
%% (both dropped and counted); the next one is applied; a gap means
%% this rank no longer has the serving rank's KV state, so it stops.
handle_info({erllama_tp, Gen, _Seq, _Cmd}, #state{gen = G, stale = N} = S) when Gen =/= G ->
    {noreply, S#state{stale = N + 1}};
handle_info({erllama_tp, _Gen, Seq, _Cmd}, #state{seq = Last, duplicates = N} = S) when
    Seq =< Last
->
    {noreply, S#state{duplicates = N + 1}};
handle_info({erllama_tp, _Gen, Seq, Cmd}, #state{ctx = C, seq = Last} = S) when
    C =/= undefined, Seq =:= Last + 1, element(1, Cmd) =:= kv_pack;
    C =/= undefined, Seq =:= Last + 1, element(1, Cmd) =:= kv_unpack
->
    {noreply, follow_kv(Cmd, S#state{seq = Seq})};
handle_info({erllama_tp, _Gen, Seq, Cmd}, #state{ctx = C, seq = Last} = S) when
    C =/= undefined, Seq =:= Last + 1
->
    case erllama_nif:tp_follow(C, Cmd) of
        ok -> {noreply, S#state{seq = Seq}};
        {error, Why} -> {stop, {follow_failed, Seq, Why}, S}
    end;
handle_info({erllama_tp, _Gen, Seq, _Cmd}, #state{seq = Last} = S) ->
    {stop, {out_of_order, Last, Seq}, S};
handle_info(_Msg, S) ->
    {noreply, S}.

%% The serving rank's sink: every command to every other rank, in order,
%% tagged with the group's generation.
forward(Gen, Followers) ->
    receive
        {erllama_tp, Seq, Cmd} ->
            M = {erllama_tp, Gen, Seq, Cmd},
            _ = [F ! M || F <- Followers],
            forward(Gen, Followers)
    after 3600000 ->
        forward(Gen, Followers)
    end.

terminate(_Reason, #state{handle = H, host = B} = S) ->
    _ = B =/= undefined andalso erllama_model_llama:terminate(B),
    _ = [erllama_nif:free_context(C) || C <- [S#state.ctx, S#state.ref_ctx], C =/= undefined],
    _ = [erllama_nif:free_model(M) || M <- [S#state.model, S#state.ref_model], M =/= undefined],
    erllama_tp:close(H).

reply_loaded({ok, M, C}, split, S) -> {reply, ok, S#state{model = M, ctx = C}};
reply_loaded({ok, M, C}, ref, S) -> {reply, ok, S#state{ref_model = M, ref_ctx = C}};
reply_loaded({error, _} = E, _, S) -> {reply, E, S}.

%% A split model keeps no recurrent-state snapshots unless the caller
%% asks: llama.cpp's split backend cannot split the snapshot's view of a
%% hybrid model's recurrent cache (GLM-5.3-Flash: every rank aborts at
%% its first decode). Without snapshots a hybrid model cannot drop its
%% last token: a cache hit on the whole prompt is served cold (a hit on
%% a prefix, followed by new tokens, still restores), and a rejected
%% draft cannot be rolled back. Dense models ignore the option. Every
%% rank applies the same default.
split_context_opts(COpts) ->
    maps:merge(#{n_rs_seq => 0}, COpts).

%% Every follower must hold the same model as this rank; then this
%% rank's context sends to a forwarder that keeps them in step.
host_model(Config, Followers, #state{handle = H, world = W, gen = Gen}) ->
    #{model_path := Path} = Config,
    MOpts = maps:get(model_opts, Config, #{}),
    COpts = maps:get(context_opts, Config, #{}),
    Mine = describe(Path, {MOpts, COpts}, W),
    case [F || {F, Desc} <- Followers, Desc =/= Mine] of
        [] ->
            Pids = [F || {F, _} <- Followers],
            Fwd = spawn_link(fun() -> forward(Gen, Pids) end),
            erllama_model_llama:init(Config#{
                model_opts => MOpts#{tp_rank => H},
                context_opts => (split_context_opts(COpts))#{tp_sink => Fwd}
            });
        Bad ->
            {error, {incompatible_ranks, Bad}}
    end.

%% Rank 0 hosting erllama_model_tp's backend.
hosting({host, Config0, Followers}, #state{host = undefined} = S) ->
    Config = maps:remove(tp_kv_timeout, Config0),
    case host_model(Config, Followers, S) of
        {ok, B} ->
            {reply, ok, S#state{host = B, kv_timeout = maps:get(tp_kv_timeout, Config0, 30000)}};
        {error, _} = E ->
            {reply, E, S}
    end;
hosting({kv_pack, _} = Req, S) ->
    checkpointing(Req, S);
hosting({kv_unpack, _, _} = Req, S) ->
    checkpointing(Req, S);
hosting(checkpoint_stats, #state{ckpt = C} = S) ->
    {reply, C, S};
hosting({mtp, Op}, #state{refs = Refs} = S) ->
    {reply, drafters(Op, maps:values(Refs)), S};
hosting({fault, drop_kv_pack}, S) ->
    {reply, ok, S#state{drop_kv_pack = true}};
hosting({backend, Fn, Args}, #state{host = B} = S) when B =/= undefined ->
    {Args1, S1} = to_local(Args, S),
    hosted(apply(erllama_model_llama, Fn, [B | Args1]), Fn, S1);
hosting({free_ref, Id}, #state{refs = Refs} = S) ->
    case maps:take(Id, Refs) of
        {Ref, Rest} -> {reply, release(Ref), S#state{refs = Rest}};
        error -> {reply, {error, released}, S}
    end;
hosting(_Req, S) ->
    {reply, {error, not_loaded}, S}.

%% -----------------------------------------------------------------------------
%% Checkpoints
%%
%% Rank 0 puts the command in the stream the other ranks follow, after
%% every batch it decoded, then packs (or restores) its own share; each
%% other rank does the same with its share when it reaches the command,
%% and answers rank 0. The shares are framed into one binary
%% (erllama_tp_checkpoint), checked before any rank restores them.

checkpointing({kv_pack, SeqId}, #state{host = B, world = W, kv_timeout = T} = S) when
    B =/= undefined
->
    {Ms, R} = timed(fun() ->
        erllama_tp_checkpoint:pack(erllama_model_llama:get_context_ref(B), SeqId, W, T)
    end),
    {reply, R, S#state{ckpt = erllama_tp_checkpoint:record(pack, Ms, R, S#state.ckpt)}};
checkpointing({kv_unpack, Bin, SeqId}, #state{host = B, world = W, kv_timeout = T} = S) when
    B =/= undefined
->
    {Ms, R} = timed(fun() ->
        erllama_tp_checkpoint:restore(erllama_model_llama:get_context_ref(B), Bin, SeqId, W, T)
    end),
    {reply, R, S#state{ckpt = erllama_tp_checkpoint:record(unpack, Ms, R, S#state.ckpt)}};
checkpointing(_Req, S) ->
    {reply, {error, not_loaded}, S}.

%% A follower's share of a checkpoint or restore, answered to rank 0.
follow_kv({kv_pack, _SeqId, _Tag, _To}, #state{drop_kv_pack = true} = S) ->
    S#state{drop_kv_pack = false};
follow_kv(Cmd, #state{ctx = C, rank = R} = S) ->
    ok = erllama_tp_checkpoint:answer(Cmd, C, R),
    S.

seq_op_ctx(C, {decode, Seq, Tokens, All}) -> erllama_nif:tp_decode_seq(C, Seq, Tokens, All);
seq_op_ctx(C, {cp, Src, Dst}) -> erllama_nif:kv_seq_cp(C, Src, Dst);
seq_op_ctx(C, {drop, Seq}) -> erllama_nif:kv_seq_rm(C, Seq, 0, -1);
seq_op_ctx(C, {trim, Seq, Keep}) -> erllama_nif:kv_seq_rm(C, Seq, Keep, -1);
seq_op_ctx(C, {pos_max, Seq}) -> erllama_nif:kv_seq_pos_max(C, Seq);
seq_op_ctx(C, {diag, Op}) -> erllama_nif:ctx_diag(C, Op).

timed(Fun) ->
    {Us, R} = timer:tc(Fun),
    {Us / 1000, R}.

%% A backend callback's result: the callbacks that return a new state
%% keep it here and answer `ok'; references go out as handles.
hosted({ok, B}, Fn, S) when
    Fn =:= set_grammar; Fn =:= configure_sampler; Fn =:= clear_sampler; Fn =:= reset_context
->
    {reply, ok, S#state{host = B}};
hosted(Result, _Fn, S) ->
    {Out, S1} = to_remote(Result, S),
    {reply, Out, S1}.

to_remote(R, #state{refs = Refs, next_ref = N} = S) when is_reference(R) ->
    {{erllama_tp_ref, self(), N}, S#state{refs = Refs#{N => R}, next_ref = N + 1}};
to_remote(L, S) when is_list(L) ->
    lists:mapfoldl(fun to_remote/2, S, L);
to_remote(T, S) when is_tuple(T) ->
    {L, S1} = to_remote(tuple_to_list(T), S),
    {list_to_tuple(L), S1};
to_remote(X, S) ->
    {X, S}.

%% Of the MTP drafters rank 0 hosts (the handles are samplers and
%% drafters), one answer each: their counters (`stats') or their round
%% trace (erllama_nif:mtp_trace/2).
drafters(Op, Refs) ->
    lists:filtermap(
        fun(R) ->
            try
                {true,
                    case Op of
                        stats -> erllama_nif:mtp_stats(R);
                        _ -> erllama_nif:mtp_trace(R, Op)
                    end}
            catch
                error:badarg -> false
            end
        end,
        Refs
    ).

release(R) ->
    try
        erllama_model_llama:sampler_free(R)
    catch
        error:badarg -> erllama_nif:mtp_free(R)
    end.

to_local({erllama_tp_ref, Pid, Id}, #state{refs = Refs} = S) when Pid =:= self() ->
    {maps:get(Id, Refs, released), S};
to_local(L, S) when is_list(L) ->
    lists:mapfoldl(fun to_local/2, S, L);
to_local(T, S) when is_tuple(T) ->
    {L, S1} = to_local(tuple_to_list(T), S),
    {list_to_tuple(L), S1};
to_local(X, S) ->
    {X, S}.

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

%% The context options erllama_model_llama derives from the model's
%% family (rank 0 opens its context through it), so every rank holds the
%% same kind of state.
load_with_ctx(M, COpts0) ->
    COpts = erllama_model_llama:family_context_opts(COpts0, M),
    case erllama_nif:new_context(M, COpts) of
        {ok, C} ->
            {ok, M, C};
        {error, _} = E ->
            _ = erllama_nif:free_model(M),
            E
    end.
