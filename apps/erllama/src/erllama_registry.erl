%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_registry).
-moduledoc false.
%% ETS-backed `via` callback for naming `erllama_model` gen_statems by
%% binary `model_id()`.
%%
%% Atoms are unbounded global table entries; user-supplied model
%% identifiers (e.g. coming from an HTTP request body in a front-end
%% server) cannot safely be `binary_to_atom/1`'d. This registry lets a
%% gen_statem be registered as
%%
%% ```
%% gen_statem:start_link({via, erllama_registry, ModelId}, ...)
%% ```
%%
%% with `ModelId :: binary()`, and looked up via `whereis_name/1`.
%%
%% The registry is a tiny gen_server that owns a public ETS table.
%% Lookups happen straight from ETS in the caller process, so the
%% gen_server is never on the hot path.
%%
%% A row also carries the model's shared state (`set_shared/2`): what
%% a caller needs to tokenise or render a chat template without
%% waiting for the model process, which is busy in its decode step
%% while it generates. The row, and so the shared state, goes away
%% when the model process exits.
-behaviour(gen_server).

-export([
    start_link/0,
    register_name/2,
    unregister_name/1,
    whereis_name/1,
    send/2,
    all/0,
    set_shared/2,
    shared/1
]).

-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-define(TABLE, ?MODULE).

-type model_id() :: erllama:model_id().
-export_type([model_id/0]).

%% =============================================================================
%% via API
%% =============================================================================

%% Called by gen from inside the process being registered, so the
%% monitor that removes the name when the process dies has to be taken
%% by this module's server (a process monitoring itself never gets its
%% own DOWN). A name still held by a dead process is taken over.
-spec register_name(model_id(), pid()) -> yes | no.
register_name(Name, Pid) when is_binary(Name), is_pid(Pid) ->
    case ets:insert_new(?TABLE, {Name, Pid, undefined, undefined}) of
        true ->
            gen_server:cast(?MODULE, {monitor, Name, Pid}),
            yes;
        false ->
            case ets:lookup(?TABLE, Name) of
                [{Name, Old, _, _} = Stale] when Old =/= Pid ->
                    case is_process_alive(Old) of
                        true ->
                            no;
                        false ->
                            true = ets:delete_object(?TABLE, Stale),
                            register_name(Name, Pid)
                    end;
                _ ->
                    no
            end
    end.

-spec unregister_name(model_id()) -> model_id().
unregister_name(Name) when is_binary(Name) ->
    _ = ets:delete(?TABLE, Name),
    Name.

-spec whereis_name(model_id()) -> pid() | undefined.
whereis_name(Name) when is_binary(Name) ->
    case ets:lookup(?TABLE, Name) of
        [{Name, Pid, _, _}] when is_pid(Pid) ->
            case is_process_alive(Pid) of
                true -> Pid;
                false -> undefined
            end;
        [] ->
            undefined
    end.

-spec send(model_id(), term()) -> pid().
send(Name, Msg) when is_binary(Name) ->
    case whereis_name(Name) of
        Pid when is_pid(Pid) ->
            Pid ! Msg,
            Pid;
        undefined ->
            error({badarg, {Name, Msg}})
    end.

%% Snapshot of all currently-registered models. Used by
%% erllama:list_models/0.
-spec all() -> [{model_id(), pid()}].
all() ->
    [
        {Name, Pid}
     || {Name, Pid, _Mon, _Shared} <- ets:tab2list(?TABLE),
        is_process_alive(Pid)
    ].

%% Called by the registered model process itself once its backend is
%% up. A no-op when `Name` is not registered to the caller.
-spec set_shared(model_id(), term()) -> ok.
set_shared(Name, Shared) when is_binary(Name) ->
    Self = self(),
    case ets:lookup(?TABLE, Name) of
        [{Name, Self, _, _}] -> true = ets:update_element(?TABLE, Name, {4, Shared});
        _ -> ok
    end,
    ok.

%% The shared state of `Name` and the pid that published it, or
%% `undefined` when the model is not registered or published none.
-spec shared(model_id()) -> {pid(), term()} | undefined.
shared(Name) when is_binary(Name) ->
    case ets:lookup(?TABLE, Name) of
        [{Name, Pid, _, Shared}] when Shared =/= undefined -> {Pid, Shared};
        _ -> undefined
    end.

%% =============================================================================
%% gen_server
%% =============================================================================

start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

init([]) ->
    _ = ets:new(?TABLE, [
        named_table,
        public,
        set,
        {read_concurrency, true},
        {write_concurrency, true}
    ]),
    {ok, #{}}.

handle_call(_, _, S) -> {reply, {error, unknown_call}, S}.
handle_cast({monitor, Name, Pid}, S) ->
    Ref = monitor(process, Pid),
    case ets:lookup(?TABLE, Name) of
        [{Name, Pid, _, _}] -> true = ets:update_element(?TABLE, Name, {3, Ref});
        _ -> demonitor(Ref, [flush])
    end,
    {noreply, S};
handle_cast(_, S) ->
    {noreply, S}.

handle_info({'DOWN', _MonRef, process, Pid, _Reason}, S) ->
    _ = ets:match_delete(?TABLE, {'_', Pid, '_', '_'}),
    {noreply, S};
handle_info(_, S) ->
    {noreply, S}.

terminate(_, _) -> ok.
