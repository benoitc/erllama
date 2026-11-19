%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_convs).
-moduledoc false.
%% Conversations held by the release: the message history, the model
%% of the last turn, the cache key that turn finished on (passed as
%% `parent_key' to the next turn on the same model) and the run that
%% currently owns the conversation. At most one run is active per
%% conversation. In memory for now; persistence keeps this interface.

-behaviour(gen_server).

-export([start_link/0, begin_run/3, end_run/4, append/4, release/2, get/1, list/0]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-define(TAB, ?MODULE).

%% The table is the state; the process exists to own it and to
%% serialise the reservations.
-record(state, {}).
-type state() :: #state{}.

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

%% Reserve `ConvId' (created when `undefined' or unknown) for `RunId'.
%% Returns the history and the parent key to continue from.
-spec begin_run(binary() | undefined, term(), binary()) ->
    {ok, #{id := binary(), messages := [map()], parent_key := term()}}
    | {error, conversation_busy}.
begin_run(ConvId, Model, RunId) ->
    gen_server:call(?MODULE, {begin_run, ConvId, Model, RunId}).

%% The run finished a turn: append its messages and remember where the
%% KV cache stands, then free the conversation.
-spec end_run(binary(), binary(), [map()], term()) -> ok.
end_run(ConvId, RunId, Messages, FinishKey) ->
    gen_server:call(?MODULE, {end_run, ConvId, RunId, Messages, FinishKey}).

%% Record messages while keeping the conversation (an agent session
%% holds it across turns). A no-op unless `Owner' holds it.
-spec append(binary(), binary(), [map()], term()) -> ok.
append(ConvId, Owner, Messages, FinishKey) ->
    gen_server:call(?MODULE, {append, ConvId, Owner, Messages, FinishKey}).

%% Free the conversation without appending (cancel, error, crash).
%% A no-op unless `RunId' is the active run.
-spec release(binary(), binary()) -> ok.
release(ConvId, RunId) ->
    gen_server:call(?MODULE, {release, ConvId, RunId}).

-spec get(binary()) -> {ok, map()} | {error, not_found}.
get(ConvId) ->
    case ets:lookup(?TAB, ConvId) of
        [{_, Conv}] -> {ok, Conv};
        [] -> {error, not_found}
    end.

-spec list() -> [map()].
list() ->
    [
        maps:remove(messages, C#{turns => length(maps:get(messages, C))})
     || {_, C} <- ets:tab2list(?TAB)
    ].

-spec init([]) -> {ok, state()}.
init([]) ->
    _ = ets:new(?TAB, [named_table, set, protected, {read_concurrency, true}]),
    {ok, #state{}}.

handle_call({begin_run, ConvId0, Model, RunId}, _From, S) ->
    {reply, reserve(ConvId0, Model, RunId), S};
handle_call({end_run, ConvId, RunId, New, FinishKey}, _From, S) ->
    case ets:lookup(?TAB, ConvId) of
        [{_, #{active_run := RunId, messages := Msgs} = C}] ->
            C1 = C#{
                active_run := undefined,
                messages := Msgs ++ New,
                last_finish_key := FinishKey,
                updated_at := now_ms()
            },
            true = ets:insert(?TAB, {ConvId, C1});
        _ ->
            ok
    end,
    {reply, ok, S};
handle_call({append, ConvId, Owner, New, FinishKey}, _From, S) ->
    ok = do_append(ConvId, Owner, New, FinishKey),
    {reply, ok, S};
handle_call({release, ConvId, RunId}, _From, S) ->
    case ets:lookup(?TAB, ConvId) of
        [{_, #{active_run := RunId} = C}] ->
            true = ets:insert(?TAB, {ConvId, C#{active_run := undefined}});
        _ ->
            ok
    end,
    {reply, ok, S};
handle_call(_Req, _From, S) ->
    {reply, {error, unknown_call}, S}.

handle_cast(_Msg, S) ->
    {noreply, S}.

handle_info(_Msg, S) ->
    {noreply, S}.

reserve(undefined, Model, RunId) ->
    reserve(erllama_server_ids:conversation(), Model, RunId);
reserve(ConvId, Model, RunId) ->
    Conv =
        case ets:lookup(?TAB, ConvId) of
            [{_, C}] -> C;
            [] -> new(ConvId, Model)
        end,
    case Conv of
        #{active_run := Active} when Active =/= undefined ->
            {error, conversation_busy};
        #{messages := Msgs} ->
            %% The last turn's cache key only helps the same model.
            ParentKey =
                case Conv of
                    #{model := Model, last_finish_key := K} -> K;
                    _ -> undefined
                end,
            true = ets:insert(?TAB, {ConvId, Conv#{active_run := RunId, model := Model}}),
            {ok, #{id => ConvId, messages => Msgs, parent_key => ParentKey}}
    end.

do_append(ConvId, Owner, New, FinishKey) ->
    case ets:lookup(?TAB, ConvId) of
        [{_, #{active_run := Owner, messages := Msgs} = C}] ->
            Key =
                case FinishKey of
                    undefined -> maps:get(last_finish_key, C);
                    K -> K
                end,
            true = ets:insert(
                ?TAB,
                {ConvId, C#{
                    messages := Msgs ++ New, last_finish_key := Key, updated_at := now_ms()
                }}
            ),
            ok;
        _ ->
            ok
    end.

new(ConvId, Model) ->
    Now = now_ms(),
    #{
        id => ConvId,
        model => Model,
        messages => [],
        last_finish_key => undefined,
        active_run => undefined,
        created_at => Now,
        updated_at => Now
    }.

now_ms() -> erlang:system_time(millisecond).
