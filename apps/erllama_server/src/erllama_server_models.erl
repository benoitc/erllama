%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_models).
-moduledoc false.
%% Which registry models are loaded, and for how long. A request that
%% names a registry model loads it on demand; concurrent requests for a
%% model that is loading share that one load. A request holds the model
%% while it runs (`acquire/3' ... `release/2', or the holder exiting).
%% When the last holder lets go, the model stays loaded for its
%% keep-alive (the request's, the entry's or `keep_alive_ms'), then is
%% unloaded if nothing used it meanwhile.
%%
%% Under memory pressure erllama's scheduler asks `evict_one/0' (this
%% module is its `model_evictor'): the least recently used model that
%% nothing holds is unloaded. There is no second scheduler here.
%%
%% Models loaded directly through erllama (not from the registry) can
%% be used by name but are never unloaded by this module.

-behaviour(gen_server).
-behaviour(erllama_model_evictor).

-export([start_link/0, resolve/1, acquire/3, release/2, load/2, unload/1, ps/0, evict_one/0]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-define(LOAD_TIMEOUT, infinity).

-record(model, {
    entry :: map(),
    holders = #{} :: #{pid() => reference()},
    keep_alive :: non_neg_integer() | infinity,
    last_used :: integer(),
    timer :: reference() | undefined,
    loaded_at :: integer()
}).

-record(state, {
    models = #{} :: #{binary() => #model{}},
    %% A load in progress: the loader's monitor, and who waits for it.
    loading = #{} :: #{binary() => {reference(), [{gen_server:from(), pid(), term()}]}},
    %% An unload in progress, and the acquisitions waiting for it to
    %% finish: they then load the model again.
    unloading = #{} :: #{
        binary() => [{gen_server:from(), {acquire, map(), pid(), term()} | unloaded}]
    }
}).
-type state() :: #state{}.

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

%% What a request's `model' refers to: a registry entry or a model
%% loaded directly through erllama.
-spec resolve(binary() | undefined) ->
    {ok, {registry, map()} | {loaded, binary()}} | {error, term()}.
resolve(Name) ->
    case erllama_server_registry:resolve(Name) of
        {ok, #{status := missing, name := N}} ->
            {error, {model_file_missing, N}};
        {ok, Entry} ->
            {ok, {registry, Entry}};
        {error, _} = E when is_binary(Name) ->
            case erllama:whereis(Name) of
                {ok, _} -> {ok, {loaded, Name}};
                {error, not_loaded} -> E
            end;
        {error, _} = E ->
            E
    end.

%% Load `Name' if needed and hold it for `Holder' until `release/2'
%% or until `Holder' exits. Returns the erllama model id.
-spec acquire(binary() | undefined, pid(), non_neg_integer() | infinity | undefined) ->
    {ok, binary()} | {error, term()}.
acquire(Name, Holder, KeepAlive) ->
    case resolve(Name) of
        {ok, {loaded, Id}} ->
            {ok, Id};
        {ok, {registry, Entry}} ->
            gen_server:call(?MODULE, {acquire, Entry, Holder, KeepAlive}, ?LOAD_TIMEOUT);
        {error, _} = E ->
            E
    end.

-spec release(binary(), pid()) -> ok.
release(Id, Holder) ->
    gen_server:cast(?MODULE, {release, Id, Holder}).

%% Load without holding: the keep-alive starts at once.
-spec load(binary(), non_neg_integer() | infinity | undefined) -> {ok, binary()} | {error, term()}.
load(Name, KeepAlive) ->
    case acquire(Name, self(), KeepAlive) of
        {ok, Id} ->
            ok = release(Id, self()),
            {ok, Id};
        {error, _} = E ->
            E
    end.

-spec unload(binary()) -> ok | {error, term()}.
unload(Name) ->
    Id =
        case resolve(Name) of
            {ok, {registry, #{name := N}}} -> N;
            {ok, {loaded, N}} -> N;
            {error, _} -> Name
        end,
    gen_server:call(?MODULE, {unload, Id}, infinity).

%% Loaded models with what this module knows of them.
-spec ps() -> [map()].
ps() ->
    Managed = gen_server:call(?MODULE, managed),
    [
        maps:merge(
            maps:with(
                [id, status, backend, context_size, quant_tag, vram_estimate_b, n_seq_max], Info
            ),
            maps:get(maps:get(id, Info), Managed, #{managed => false})
        )
     || Info <- erllama:list_models()
    ].

%% erllama_model_evictor: called by erllama's scheduler process.
-spec evict_one() -> {unloaded, binary()} | none.
evict_one() ->
    case gen_server:call(?MODULE, pick_evictable, 30000) of
        {ok, Id} ->
            _ = erllama:unload(Id),
            gen_server:cast(?MODULE, {unloaded, Id}),
            {unloaded, Id};
        none ->
            none
    end.

%% -----------------------------------------------------------------------------

-spec init([]) -> {ok, state()}.
init([]) ->
    {ok, #state{}}.

handle_call({acquire, #{name := Id}, _, _} = Req, From, #state{unloading = U} = St) when
    is_map_key(Id, U)
->
    %% Being unloaded: granting it now would hand out a model about to
    %% go. Wait, then load it again.
    {noreply, St#state{unloading = U#{Id := [{From, Req} | maps:get(Id, U)]}}};
handle_call(
    {acquire, #{name := Id} = Entry, Holder, KA}, From, #state{models = Ms, loading = L} = St
) ->
    case {Ms, L} of
        {#{Id := M}, _} ->
            case erllama:whereis(Id) of
                {ok, _} ->
                    {reply, {ok, Id}, St#state{models = Ms#{Id := hold(Holder, KA, M)}}};
                {error, not_loaded} ->
                    %% Unloaded behind our back (an explicit erllama:unload).
                    handle_call({acquire, Entry, Holder, KA}, From, St#state{
                        models = maps:remove(Id, Ms)
                    })
            end;
        {_, #{Id := {Mon, Waiters}}} ->
            {noreply, St#state{loading = L#{Id := {Mon, [{From, Holder, KA} | Waiters]}}}};
        _ ->
            Self = self(),
            {_, Mon} = spawn_monitor(fun() -> Self ! {load_result, Id, load_model(Entry)} end),
            {noreply, St#state{loading = L#{Id => {Mon, [{From, Holder, KA}]}}}}
    end;
handle_call(managed, _From, #state{models = Ms} = St) ->
    {reply, maps:map(fun(_, M) -> describe(M) end, Ms), St};
handle_call({unload, Id}, From, #state{unloading = U} = St) when is_map_key(Id, U) ->
    %% Already going: answer when it is gone.
    {noreply, St#state{unloading = U#{Id := [{From, unloaded} | maps:get(Id, U)]}}};
handle_call({unload, Id}, From, #state{models = Ms} = St) ->
    case maps:take(Id, Ms) of
        {M, Ms1} ->
            maps:foreach(fun(_, Mon) -> erlang:demonitor(Mon, [flush]) end, M#model.holders);
        error ->
            Ms1 = Ms
    end,
    {noreply, begin_unload(Id, From, St#state{models = Ms1})};
handle_call(pick_evictable, _From, #state{models = Ms, unloading = U} = St) ->
    Idle = [
        {M#model.last_used, Id}
     || Id := M <- Ms, map_size(M#model.holders) =:= 0, not is_map_key(Id, U)
    ],
    case lists:sort(Idle) of
        [{_, Id} | _] ->
            cancel_timer(maps:get(Id, Ms)),
            {reply, {ok, Id}, St#state{unloading = U#{Id => []}}};
        [] ->
            {reply, none, St}
    end;
handle_call(_Req, _From, St) ->
    {reply, {error, unknown_call}, St}.

handle_cast({release, Id, Holder}, #state{models = Ms} = St) ->
    case Ms of
        #{Id := M} -> {noreply, St#state{models = Ms#{Id := let_go(Id, Holder, M)}}};
        _ -> {noreply, St}
    end;
handle_cast({unloaded, Id}, #state{models = Ms, unloading = U} = St) ->
    Waiting = lists:reverse(maps:get(Id, U, [])),
    St1 = St#state{models = maps:remove(Id, Ms), unloading = maps:remove(Id, U)},
    {noreply, lists:foldl(fun retry_acquire/2, St1, Waiting)};
handle_cast(_Msg, St) ->
    {noreply, St}.

handle_info({load_result, Id, Result}, #state{loading = L} = St) when is_map_key(Id, L) ->
    {Mon, _} = maps:get(Id, L),
    erlang:demonitor(Mon, [flush]),
    {noreply, loaded(Id, {loaded, Id, Result}, St)};
handle_info({'DOWN', Mon, process, _, Reason}, #state{loading = L} = St) ->
    %% A loader that died before reporting crashed in the load.
    case [Id || Id := {M, _} <- L, M =:= Mon] of
        [Id] -> {noreply, loaded(Id, Reason, St)};
        [] -> {noreply, holder_down(Mon, St)}
    end;
handle_info({keep_alive, Id, _Ref}, #state{unloading = U} = St) when is_map_key(Id, U) ->
    {noreply, St};
handle_info({keep_alive, Id, Ref}, #state{models = Ms} = St) ->
    case Ms of
        #{Id := #model{timer = Ref, holders = H} = M} when map_size(H) =:= 0 ->
            St1 = St#state{models = Ms#{Id := M#model{timer = undefined}}},
            {noreply, begin_unload(Id, undefined, St1)};
        _ ->
            {noreply, St}
    end;
handle_info(_Msg, St) ->
    {noreply, St}.

%% -----------------------------------------------------------------------------

%% The loader exited with the load's outcome; answer everyone waiting.
loaded(Id, {loaded, Id, Result}, #state{loading = L, models = Ms} = St) ->
    {{_, Waiters}, L1} = maps:take(Id, L),
    case Result of
        {ok, Id, Entry} ->
            Now = erlang:monotonic_time(millisecond),
            M0 = #model{
                entry = Entry,
                keep_alive = default_keep_alive(Entry),
                last_used = Now,
                loaded_at = Now
            },
            M = lists:foldl(
                fun({From, Holder, KA}, Acc) ->
                    gen_server:reply(From, {ok, Id}),
                    hold(Holder, KA, Acc)
                end,
                M0,
                Waiters
            ),
            St#state{loading = L1, models = Ms#{Id => M}};
        {error, _} = E ->
            [gen_server:reply(From, E) || {From, _, _} <- Waiters],
            St#state{loading = L1}
    end;
loaded(Id, Crash, #state{loading = L} = St) ->
    {{_, Waiters}, L1} = maps:take(Id, L),
    [gen_server:reply(From, {error, {load_failed, Crash}}) || {From, _, _} <- Waiters],
    St#state{loading = L1}.

load_model(#{name := Id} = Entry) ->
    case erllama:load_model(Id, load_config(Entry)) of
        {ok, Id} -> {ok, Id, Entry};
        {error, already_loaded} -> {ok, Id, Entry};
        {error, _} = E -> E
    end.

%% The erllama load config of a registry entry. `context_size' is the
%% cache's copy of `n_ctx' and must match it.
load_config(#{path := Path} = Entry) ->
    Load = maps:get(load, Entry, #{}),
    NCtx = n_ctx(Entry, Load),
    Base = #{
        model_path => Path,
        model_opts => maps:with([n_gpu_layers], Load),
        context_opts => #{n_ctx => NCtx},
        context_size => NCtx
    },
    with_extras(Base, Entry, Load).

with_extras(_Base, _Entry, #{backend := erllama_model_stub} = Load) ->
    %% Tests: the same entry on erllama's stub backend.
    #{backend => erllama_model_stub, step_delay_ms => maps:get(step_delay_ms, Load, 0)};
with_extras(Base, Entry, Load) ->
    WithProj =
        case Entry of
            #{mmproj := MM} when is_binary(MM) -> Base#{mmproj_path => MM};
            _ -> Base
        end,
    case Load of
        #{chat_template := T} -> WithProj#{chat_template => T};
        _ -> WithProj
    end.

%% The entry's `n_ctx', else `default_n_ctx' capped at what the model
%% was trained for.
n_ctx(_Entry, #{n_ctx := N}) ->
    N;
n_ctx(#{path := Path}, _) ->
    Default = erllama_server_config:get(default_n_ctx),
    case erllama_server_gguf:read(Path) of
        {ok, H} ->
            case erllama_server_gguf:summary(H) of
                #{context_length := Train} when is_integer(Train), Train > 0 -> min(Train, Default);
                _ -> Default
            end;
        {error, _} ->
            Default
    end.

default_keep_alive(Entry) ->
    case maps:get(keep_alive, Entry, undefined) of
        undefined -> erllama_server_config:get(keep_alive_ms);
        KA -> KA
    end.

hold(Holder, KA, #model{holders = H} = M) ->
    cancel_timer(M),
    M1 = M#model{timer = undefined, last_used = erlang:monotonic_time(millisecond)},
    M2 =
        case KA of
            undefined -> M1;
            _ -> M1#model{keep_alive = KA}
        end,
    case is_map_key(Holder, H) of
        true -> M2;
        false -> M2#model{holders = H#{Holder => erlang:monitor(process, Holder)}}
    end.

let_go(Id, Holder, #model{holders = H} = M) ->
    case maps:take(Holder, H) of
        {Mon, H1} ->
            erlang:demonitor(Mon, [flush]),
            idle_check(Id, M#model{holders = H1, last_used = erlang:monotonic_time(millisecond)});
        error ->
            M
    end.

holder_down(Mon, #state{models = Ms} = St) ->
    Ms1 = maps:map(
        fun(Id, #model{holders = H} = M) ->
            case [P || P := R <- H, R =:= Mon] of
                [Pid] -> idle_check(Id, M#model{holders = maps:remove(Pid, H)});
                [] -> M
            end
        end,
        Ms
    ),
    St#state{models = Ms1}.

%% Nothing holds the model: start its keep-alive.
idle_check(Id, #model{holders = H, keep_alive = KA} = M) when map_size(H) =:= 0 ->
    case KA of
        infinity ->
            M;
        0 ->
            %% Through the timer path, so the unload is marked in the state.
            Ref = make_ref(),
            self() ! {keep_alive, Id, Ref},
            M#model{timer = Ref};
        Ms when is_integer(Ms) ->
            Ref = make_ref(),
            _ = erlang:send_after(Ms, self(), {keep_alive, Id, Ref}),
            M#model{timer = Ref}
    end;
idle_check(_Id, M) ->
    M.

%% Unload in a helper (it can take a while) and mark it, so
%% acquisitions wait for it; `From', if any, gets the unload's result.
begin_unload(Id, From, #state{unloading = U} = St) ->
    Self = self(),
    _ = spawn(fun() ->
        R = erllama:unload(Id),
        _ = From =/= undefined andalso gen_server:reply(From, R),
        gen_server:cast(Self, {unloaded, Id})
    end),
    St#state{unloading = U#{Id => maps:get(Id, U, [])}}.

retry_acquire({From, unloaded}, St) ->
    gen_server:reply(From, ok),
    St;
retry_acquire({From, Req}, St) ->
    case handle_call(Req, From, St) of
        {reply, R, St1} ->
            gen_server:reply(From, R),
            St1;
        {noreply, St1} ->
            St1
    end.

cancel_timer(_M) ->
    %% Timers are matched by reference on arrival; a stale one is
    %% ignored, so nothing needs cancelling.
    ok.

describe(#model{holders = H, keep_alive = KA, last_used = LU, timer = T, entry = E}) ->
    Expires =
        case {T, KA} of
            {undefined, _} ->
                undefined;
            {_, infinity} ->
                undefined;
            {_, Ms} ->
                erlang:system_time(millisecond) + Ms - (erlang:monotonic_time(millisecond) - LU)
        end,
    #{
        managed => true,
        holders => map_size(H),
        keep_alive => KA,
        expires_at => Expires,
        mmproj => maps:get(mmproj, E, undefined) =/= undefined
    }.
