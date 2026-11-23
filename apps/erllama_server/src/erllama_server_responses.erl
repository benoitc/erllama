%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_responses).
-moduledoc false.
%% Stored OpenAI Responses (`store: true'): the response object a client
%% can read back, and a snapshot of its conversation as it stood when
%% the response was made (history and the KV cache key). A request with
%% `previous_response_id' continues from a copy of that snapshot, so
%% two requests continuing the same response are independent branches.
%%
%% Each stored response is a file, `<data_dir>/responses/<id>.json',
%% written before it is visible and loaded again when the release
%% starts; its snapshot is a persisted conversation (kind `snapshot').

-behaviour(gen_server).

-export([start_link/0, put/3, get/1, continue/1, delete/1]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).
-compile({no_auto_import, [put/2, get/1]}).

-define(TAB, ?MODULE).

-record(state, {}).
-type state() :: #state{}.

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

%% Store `Response' with a snapshot of `Conv' as it stands now.
-spec put(binary(), map(), binary()) -> ok | {error, term()}.
put(Id, Response, Conv) ->
    case erllama_server_convs:fork(Conv, snapshot) of
        {ok, Snapshot} -> gen_server:call(?MODULE, {put, Id, Response, Snapshot});
        {error, _} = E -> E
    end.

%% A new conversation to continue response `Id' from.
-spec continue(binary()) -> {ok, binary()} | {error, not_found}.
continue(Id) ->
    case get(Id) of
        {ok, _, Snapshot} -> erllama_server_convs:fork(Snapshot);
        {error, _} = E -> E
    end.

-spec get(binary()) -> {ok, map(), binary()} | {error, not_found}.
get(Id) ->
    case ets:lookup(?TAB, Id) of
        [{Id, Response, Conv}] -> {ok, Response, Conv};
        [] -> {error, not_found}
    end.

-spec delete(binary()) -> ok | {error, not_found}.
delete(Id) ->
    gen_server:call(?MODULE, {delete, Id}).

-spec init([]) -> {ok, state()}.
init([]) ->
    _ = ets:new(?TAB, [named_table, set, protected, {read_concurrency, true}]),
    _ = [load(F) || F <- filelib:wildcard("*.json", dir())],
    {ok, #state{}}.

dir() -> filename:join(erllama_server_config:data_dir(), "responses").

path(Id) -> filename:join(dir(), <<Id/binary, ".json">>).

load(File) ->
    try json:decode(element(2, {ok, _} = file:read_file(filename:join(dir(), File)))) of
        #{<<"id">> := Id, <<"response">> := R, <<"snapshot">> := Snap} ->
            true = ets:insert(?TAB, {Id, R, Snap});
        _ ->
            logger:warning(#{msg => "unreadable stored response", file => File})
    catch
        _:_ -> logger:warning(#{msg => "unreadable stored response", file => File})
    end.

handle_call({put, Id, Response, Conv}, _From, S) ->
    P = path(Id),
    ok = filelib:ensure_dir(P),
    Doc = json:encode(#{<<"id">> => Id, <<"response">> => Response, <<"snapshot">> => Conv}),
    case file:write_file(P, Doc, [raw, binary, sync]) of
        ok ->
            true = ets:insert(?TAB, {Id, Response, Conv}),
            {reply, ok, S};
        {error, _} = E ->
            {reply, E, S}
    end;
handle_call({delete, Id}, _From, S) ->
    case ets:take(?TAB, Id) of
        [{_, _, Snapshot}] ->
            _ = file:delete(path(Id)),
            _ = erllama_server_convs:delete(Snapshot),
            {reply, ok, S};
        [] ->
            {reply, {error, not_found}, S}
    end;
handle_call(_Req, _From, S) ->
    {reply, {error, unknown_call}, S}.

handle_cast(_Msg, S) ->
    {noreply, S}.

handle_info(_Msg, S) ->
    {noreply, S}.
