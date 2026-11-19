%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_responses).
-moduledoc false.
%% Stored OpenAI Responses (`store: true'): the response object a client
%% can read back, and the conversation it belongs to, so a request with
%% `previous_response_id' continues that conversation (history and KV
%% cache warmth). Kept in memory; durable storage comes with the
%% persistent conversations.

-behaviour(gen_server).

-export([start_link/0, put/3, get/1, delete/1]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).
-compile({no_auto_import, [put/2, get/1]}).

-define(TAB, ?MODULE).

-record(state, {}).
-type state() :: #state{}.

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

-spec put(binary(), map(), binary() | none) -> ok.
put(Id, Response, Conv) ->
    gen_server:call(?MODULE, {put, Id, Response, Conv}).

-spec get(binary()) -> {ok, map(), binary() | none} | {error, not_found}.
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
    {ok, #state{}}.

handle_call({put, Id, Response, Conv}, _From, S) ->
    true = ets:insert(?TAB, {Id, Response, Conv}),
    {reply, ok, S};
handle_call({delete, Id}, _From, S) ->
    case ets:take(?TAB, Id) of
        [_] -> {reply, ok, S};
        [] -> {reply, {error, not_found}, S}
    end;
handle_call(_Req, _From, S) ->
    {reply, {error, unknown_call}, S}.

handle_cast(_Msg, S) ->
    {noreply, S}.

handle_info(_Msg, S) ->
    {noreply, S}.
