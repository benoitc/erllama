%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_convs).
-moduledoc false.
%% Conversations held by the release: the message history, the model
%% of the last turn, the cache key that turn finished on (passed as
%% `parent_key' to the next turn on the same model) and the run that
%% currently owns the conversation. At most one run is active per
%% conversation.
%%
%% Every change is written to the conversation's log
%% (erllama_server_conv_log) before it is visible here, and the logs are
%% loaded again when the release starts. A run that a stop or a crash
%% cut is repaired then: its conversation is freed, and tool calls left
%% without a result get one, an unresolved one when the call may have
%% run (never run again). A tool's intent is recorded before it runs and
%% its outcome before the session generates again (tool_intent/5,
%% tool_outcome/5).
%%
%% Conversations are `conversation' or `snapshot' (the frozen copy
%% behind a stored response, hidden from list/0).

-behaviour(gen_server).

-export([start_link/0, begin_run/3, begin_run/4, end_run/4, append/4, release/2, get/1, list/0]).
-export([fork/1, fork/2, delete/1, tool_intent/5, tool_outcome/5, compact/4]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-define(TAB, ?MODULE).

%% The table is the state; the process exists to own it, to serialise
%% the reservations and to be the only writer of the logs.
-record(state, {}).
-type state() :: #state{}.

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

%% Reserve `ConvId' (created when `undefined' or unknown) for `RunId'.
%% Returns the history and the parent key to continue from.
-spec begin_run(binary() | undefined, term(), binary()) ->
    {ok, #{id := binary(), messages := [map()], parent_key := term(), workspace := term()}}
    | {error, conversation_busy}.
begin_run(ConvId, Model, RunId) ->
    begin_run(ConvId, Model, RunId, #{kind => run}).

%% `Info': `kind' (run | agent) and, for an agent, `workspace'.
-spec begin_run(binary() | undefined, term(), binary(), map()) ->
    {ok, #{id := binary(), messages := [map()], parent_key := term(), workspace := term()}}
    | {error, conversation_busy}.
begin_run(ConvId, Model, RunId, Info) ->
    gen_server:call(?MODULE, {begin_run, ConvId, Model, RunId, Info}).

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
%% A no-op unless `RunId' is the active run. A run that left tool calls
%% without results (a session that died) is repaired as a restart would
%% repair it: recorded outcomes are used, calls that may have run are
%% marked unresolved.
-spec release(binary(), binary()) -> ok.
release(ConvId, RunId) ->
    gen_server:call(?MODULE, {release, ConvId, RunId}).

%% A tool call is about to run (durable when this returns).
-spec tool_intent(binary(), binary(), binary(), binary(), map()) -> ok.
tool_intent(ConvId, Owner, CallId, Name, Args) ->
    gen_server:call(?MODULE, {tool, ConvId, Owner, {intent, CallId, Name, Args}}).

%% A tool call finished (durable when this returns).
-spec tool_outcome(binary(), binary(), binary(), ok | error, binary()) -> ok.
tool_outcome(ConvId, Owner, CallId, Status, Text) ->
    gen_server:call(?MODULE, {tool, ConvId, Owner, {outcome, CallId, Status, Text}}).

%% Replace the transcript with its compacted form (erllama_server_compact).
%% The finish key goes: the compacted transcript is a new prompt. A
%% no-op unless `Owner' holds the conversation.
-spec compact(binary(), binary(), [map()], map()) -> ok.
compact(ConvId, Owner, Msgs, Info) ->
    gen_server:call(?MODULE, {compact, ConvId, Owner, Msgs, Info}).

%% A new conversation with a copy of `ConvId''s history and cache key:
%% what is later added to either one never shows in the other.
-spec fork(binary()) -> {ok, binary()} | {error, not_found}.
fork(ConvId) ->
    fork(ConvId, conversation).

-spec fork(binary(), conversation | snapshot) -> {ok, binary()} | {error, not_found}.
fork(ConvId, Kind) ->
    gen_server:call(?MODULE, {fork, ConvId, Kind}).

%% Forget a conversation and its log. Refused while a run holds it.
-spec delete(binary()) -> ok | {error, conversation_busy}.
delete(ConvId) ->
    gen_server:call(?MODULE, {delete, ConvId}).

-spec get(binary()) -> {ok, map()} | {error, not_found}.
get(ConvId) ->
    case ets:lookup(?TAB, ConvId) of
        [{_, Conv}] -> {ok, maps:remove(tools, Conv)};
        [] -> {error, not_found}
    end.

%% The conversations (not the snapshots), without their messages.
-spec list() -> [map()].
list() ->
    [
        maps:without([messages, tools], C#{turns => length(maps:get(messages, C))})
     || {_, #{kind := conversation} = C} <- ets:tab2list(?TAB)
    ].

-spec init([]) -> {ok, state()}.
init([]) ->
    _ = ets:new(?TAB, [named_table, set, protected, {read_concurrency, true}]),
    _ = [load(Id) || Id <- erllama_server_conv_log:ids()],
    {ok, #state{}}.

%% Load one log, repairing what a stop left half done.
load(Id) ->
    case erllama_server_conv_log:read(Id) of
        {ok, Records} ->
            case erllama_server_conv_log:replay(Records) of
                {ok, Conv0} ->
                    {Conv, Repairs} = erllama_server_conv_log:recover(Conv0),
                    ok = erllama_server_conv_log:append(Id, Repairs),
                    ok = recovered(Id, Repairs, <<"a run cut by the stop">>),
                    true = ets:insert(?TAB, {Id, Conv#{tools => #{}}});
                {error, Why} ->
                    logger:warning(#{msg => "unreadable conversation log", id => Id, why => Why})
            end;
        {error, Why} ->
            logger:warning(#{msg => "unreadable conversation log", id => Id, why => Why})
    end.

handle_call({begin_run, ConvId0, Model, RunId, Info}, _From, S) ->
    {reply, reserve(ConvId0, Model, RunId, Info), S};
handle_call({end_run, ConvId, RunId, New, FinishKey}, _From, S) ->
    ok = do_end_run(ConvId, RunId, New, FinishKey),
    {reply, ok, S};
handle_call({append, ConvId, Owner, New, FinishKey}, _From, S) ->
    ok = do_append(ConvId, Owner, New, FinishKey),
    {reply, ok, S};
handle_call({release, ConvId, RunId}, _From, S) ->
    {reply, do_release(ConvId, RunId), S};
handle_call({tool, ConvId, Owner, What}, _From, S) ->
    {reply, do_tool(ConvId, Owner, What), S};
handle_call({compact, ConvId, Owner, Msgs, Info}, _From, S) ->
    {reply, do_compact(ConvId, Owner, Msgs, Info), S};
handle_call({fork, ConvId, Kind}, _From, S) ->
    {reply, do_fork(ConvId, Kind), S};
handle_call({delete, ConvId}, _From, S) ->
    {reply, do_delete(ConvId), S};
handle_call(_Req, _From, S) ->
    {reply, {error, unknown_call}, S}.

handle_cast(_Msg, S) ->
    {noreply, S}.

handle_info(_Msg, S) ->
    {noreply, S}.

do_release(ConvId, RunId) ->
    owned(ConvId, RunId, fun(#{tools := Tools} = C) ->
        {Repaired, Records} = erllama_server_conv_log:recover(
            C#{open_run => RunId, tools := Tools}
        ),
        ok = log(ConvId, Records),
        ok = recovered(ConvId, Records, <<"a run that ended without finishing its turn">>),
        Repaired#{active_run := undefined, tools => #{}}
    end).

%% What a repair did, for operators. Only repairs that changed the
%% transcript (calls given a result) or freed a restart's cut run.
recovered(ConvId, Records, What) ->
    case [R || #{t := recovered} = R <- Records] of
        [#{run := Run, unresolved := U}] ->
            erllama_server_failures:record(
                recovered,
                ConvId,
                #{run => Run, unresolved => U},
                <<What/binary,
                    ": calls left without a result got one; unresolved ones are never run again">>
            );
        [] when Records =/= [], What =:= <<"a run cut by the stop">> ->
            erllama_server_failures:record(
                recovered,
                ConvId,
                #{run => open_run(Records)},
                <<What/binary, ": its conversation was freed">>
            );
        [] ->
            ok
    end.

open_run(Records) ->
    case [R || #{t := run, run := R} <- Records] of
        [R | _] -> R;
        [] -> undefined
    end.

do_tool(ConvId, Owner, What) ->
    owned(ConvId, Owner, fun(#{tools := Tools} = C) ->
        Rec = tool_record(Owner, What),
        ok = log(ConvId, [Rec]),
        C#{tools := track(What, Rec, Tools)}
    end).

do_compact(ConvId, Owner, Msgs, Info) ->
    owned(ConvId, Owner, fun(C) ->
        ok = log(ConvId, [
            erllama_server_conv_log:snapshot(Msgs, undefined),
            erllama_server_conv_log:compacted(Owner, Info)
        ]),
        C#{
            messages := Msgs,
            last_finish_key := undefined,
            compactions => maps:get(compactions, C, 0) + 1,
            updated_at := now_ms()
        }
    end).

%% Kept as they read back from the log, so a live repair and a repair
%% at load see the same thing.
track(What, Rec, Tools) ->
    CallId = element(2, What),
    Phase = element(1, What),
    Prev = maps:get(CallId, Tools, #{}),
    Tools#{CallId => Prev#{Phase => json:decode(iolist_to_binary(json:encode(Rec)))}}.

tool_record(Owner, {intent, CallId, Name, Args}) ->
    erllama_server_conv_log:tool_intent(Owner, CallId, Name, Args);
tool_record(Owner, {outcome, CallId, Status, Text}) ->
    erllama_server_conv_log:tool_outcome(Owner, CallId, Status, Text).

%% Apply `Fun' to the conversation when `Owner' holds it.
owned(ConvId, Owner, Fun) ->
    case ets:lookup(?TAB, ConvId) of
        [{_, #{active_run := Owner} = C}] ->
            true = ets:insert(?TAB, {ConvId, Fun(C)}),
            ok;
        _ ->
            ok
    end.

log(ConvId, Records) ->
    case erllama_server_conv_log:append(ConvId, Records) of
        ok ->
            ok;
        {error, Why} ->
            %% The conversation goes on in memory; the operator is told.
            logger:error(#{msg => "conversation log write failed", id => ConvId, why => Why}),
            ok
    end.

do_end_run(ConvId, RunId, New, FinishKey) ->
    owned(ConvId, RunId, fun(#{messages := Msgs} = C) ->
        ok = log(ConvId, [
            erllama_server_conv_log:msgs(New, FinishKey),
            erllama_server_conv_log:run(RunId, #{phase => 'end'})
        ]),
        C#{
            active_run := undefined,
            messages := Msgs ++ New,
            last_finish_key := FinishKey,
            updated_at := now_ms(),
            tools := #{}
        }
    end).

do_append(ConvId, Owner, New, FinishKey) ->
    owned(ConvId, Owner, fun(#{messages := Msgs} = C) ->
        {Key, KeyRec} =
            case FinishKey of
                undefined -> {maps:get(last_finish_key, C), keep};
                K -> {K, K}
            end,
        ok = log(ConvId, [erllama_server_conv_log:msgs(New, KeyRec)]),
        C#{messages := Msgs ++ New, last_finish_key := Key, updated_at := now_ms()}
    end).

do_fork(ConvId, Kind) ->
    case ets:lookup(?TAB, ConvId) of
        [{_, #{messages := Msgs, last_finish_key := Key, model := Model} = C}] ->
            New = erllama_server_ids:conversation(),
            Now = now_ms(),
            ok = log(New, [
                erllama_server_conv_log:create(New, Model, Kind, ConvId, Now),
                erllama_server_conv_log:snapshot(Msgs, Key)
            ]),
            true = ets:insert(
                ?TAB,
                {New, C#{
                    id := New,
                    kind := Kind,
                    parent := ConvId,
                    active_run := undefined,
                    tools := #{},
                    created_at := Now,
                    updated_at := Now
                }}
            ),
            {ok, New};
        [] ->
            {error, not_found}
    end.

do_delete(ConvId) ->
    case ets:lookup(?TAB, ConvId) of
        [{_, #{active_run := Active}}] when Active =/= undefined ->
            {error, conversation_busy};
        _ ->
            ok = erllama_server_conv_log:delete(ConvId),
            true = ets:delete(?TAB, ConvId),
            ok
    end.

reserve(undefined, Model, RunId, Info) ->
    reserve(erllama_server_ids:conversation(), Model, RunId, Info);
reserve(ConvId, Model, RunId, Info) ->
    {Conv, Created} =
        case ets:lookup(?TAB, ConvId) of
            [{_, C}] -> {C, []};
            [] -> {new(ConvId, Model), [create_record(ConvId, Model)]}
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
            Workspace = maps:get(workspace, Info, maps:get(workspace, Conv)),
            ok = log(ConvId, Created ++ [begin_record(RunId, Info, Model, Workspace)]),
            true = ets:insert(
                ?TAB,
                {ConvId, Conv#{
                    active_run := RunId, model := Model, workspace := Workspace, tools => #{}
                }}
            ),
            {ok, #{
                id => ConvId,
                messages => Msgs,
                parent_key => ParentKey,
                workspace => Workspace
            }}
    end.

begin_record(RunId, Info, Model, Workspace) ->
    erllama_server_conv_log:run(RunId, #{
        phase => 'begin',
        kind => maps:get(kind, Info, run),
        model => Model,
        workspace => Workspace
    }).

create_record(ConvId, Model) ->
    erllama_server_conv_log:create(ConvId, Model, conversation, undefined, now_ms()).

new(ConvId, Model) ->
    Now = now_ms(),
    #{
        id => ConvId,
        model => Model,
        kind => conversation,
        parent => undefined,
        messages => [],
        last_finish_key => undefined,
        active_run => undefined,
        workspace => undefined,
        unresolved => [],
        tools => #{},
        created_at => Now,
        updated_at => Now
    }.

now_ms() -> erlang:system_time(millisecond).
