%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_run).
-moduledoc false.
%% One generation run. The process owns the erllama stream, numbers
%% its events, keeps the last ones in a ring for reattachment and
%% sends them to its subscribers as
%% `{erllama_server_run, RunId, Event}'.
%%
%% Flow control: a subscriber acks with `RunPid ! {ack, self(), Seq}'.
%% One that falls more than `subscriber_max_unacked' events behind is
%% dropped with `{erllama_server_run, RunId, {dropped, NextSeq}}' and
%% can reattach from `NextSeq'. A subscriber that exits is forgotten;
%% the run goes on (runs outlive the clients that started them).
%%
%% This process is the stream's receiver: erllama cancels the stream
%% when it exits, however it exits (shutdown, crash or kill), so a
%% dead run never leaves its model generating.

-behaviour(gen_server).

-export([start_link/1, go/1, attach/3, detach/2, cancel/1, result/1, info/1]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-define(GO_TIMEOUT, 60000).

-record(state, {
    id :: binary(),
    conv :: binary() | undefined,
    model :: erllama:model_id(),
    input :: erllama_server_generate:input(),
    opts :: map(),
    new_msgs :: [map()],
    parent_key :: term(),
    ref :: reference() | undefined,
    model_mon :: reference() | undefined,
    parser :: erllama_server_generate:parser(),
    ring :: erllama_server_ring:ring(),
    subs = #{} :: #{pid() => {reference(), integer()}},
    status = pending :: pending | erllama_server_runs:status(),
    result :: map() | undefined,
    text = [] :: [binary()],
    thinking = [] :: [binary()],
    max_unacked :: pos_integer(),
    retention :: non_neg_integer()
}).
-type state() :: #state{}.

-spec start_link(map()) -> {ok, pid()} | {error, term()}.
start_link(Spec) ->
    gen_server:start_link(?MODULE, Spec, []).

%% Start generating. Errors that the request itself causes (unknown
%% model, bad option, template failure) come back here and the run
%% stops; later failures are `error' events.
-spec go(pid()) -> ok | {error, term()}.
go(Pid) -> call(Pid, go, ?GO_TIMEOUT).

-spec attach(pid(), pid(), non_neg_integer()) ->
    {ok, map()} | {error, {gap, non_neg_integer()} | not_found}.
attach(Pid, Sub, Cursor) -> call(Pid, {attach, Sub, Cursor}, 5000).

-spec detach(pid(), pid()) -> ok | {error, not_found}.
detach(Pid, Sub) -> call(Pid, {detach, Sub}, 5000).

-spec cancel(pid()) -> ok | {error, not_found}.
cancel(Pid) -> call(Pid, cancel, 5000).

-spec result(pid()) -> {ok, map()} | {error, running | not_found}.
result(Pid) -> call(Pid, result, 5000).

-spec info(pid()) -> {ok, map()} | {error, not_found}.
info(Pid) -> call(Pid, info, 5000).

%% A run that exits between the table lookup and the call is reported
%% as gone rather than crashing the caller.
call(Pid, Req, Timeout) ->
    try
        gen_server:call(Pid, Req, Timeout)
    catch
        exit:{noproc, _} -> {error, not_found};
        exit:{normal, _} -> {error, not_found}
    end.

-spec init(map()) -> {ok, state()}.
init(#{run_id := Id, model := Model, input := Input} = Spec) ->
    {ok, #state{
        id = Id,
        conv = maps:get(conversation_id, Spec, undefined),
        model = Model,
        input = Input,
        opts = maps:get(opts, Spec, #{}),
        new_msgs = maps:get(new_messages, Spec, []),
        parent_key = maps:get(parent_key, Spec, undefined),
        parser = undefined,
        ring = erllama_server_ring:new(erllama_server_config:get(run_buffer_events)),
        max_unacked = erllama_server_config:get(subscriber_max_unacked),
        retention = erllama_server_config:get(run_retention_ms)
    }}.

handle_call(go, _From, #state{status = pending} = St) ->
    case start_stream(St) of
        {ok, St1} -> {reply, ok, St1};
        {error, _} = E -> {stop, normal, E, St}
    end;
handle_call(go, _From, St) ->
    {reply, {error, already_started}, St};
handle_call({attach, Sub, Cursor}, _From, St) ->
    attach_sub(Sub, Cursor, St);
handle_call({detach, Sub}, _From, #state{subs = Subs} = St) ->
    {reply, ok, St#state{subs = drop_sub(Sub, Subs)}};
handle_call(cancel, _From, #state{status = running, ref = Ref} = St) ->
    ok = erllama:cancel(Ref),
    {reply, ok, St};
handle_call(cancel, _From, St) ->
    {reply, ok, St};
handle_call(result, _From, #state{status = running} = St) ->
    {reply, {error, running}, St};
handle_call(result, _From, #state{result = Result} = St) ->
    {reply, {ok, Result}, St};
handle_call(info, _From, St) ->
    {reply, {ok, summary(St)}, St};
handle_call(_Req, _From, St) ->
    {reply, {error, unknown_call}, St}.

handle_cast(_Msg, St) ->
    {noreply, St}.

handle_info({erllama, Ref, Event}, #state{ref = Ref, status = running} = St) ->
    {noreply, stream_event(Event, St)};
handle_info({ack, Sub, Seq}, #state{subs = Subs} = St) when is_integer(Seq) ->
    case Subs of
        #{Sub := {Mon, Acked}} when Seq > Acked ->
            {noreply, St#state{subs = Subs#{Sub := {Mon, Seq}}}};
        _ ->
            {noreply, St}
    end;
handle_info({'DOWN', Mon, process, _, Reason}, #state{model_mon = Mon, status = running} = St) ->
    {noreply, fail({model_down, Reason}, St#state{model_mon = undefined})};
handle_info({'DOWN', _Mon, process, Sub, _}, #state{subs = Subs} = St) ->
    {noreply, St#state{subs = maps:remove(Sub, Subs)}};
handle_info(expire, St) ->
    {stop, normal, St};
handle_info(_Msg, St) ->
    {noreply, St}.

%% -----------------------------------------------------------------------------

start_stream(#state{model = Model, input = Input, opts = Opts, parent_key = PK} = St) ->
    case erllama:whereis(Model) of
        {ok, MPid} ->
            Opts1 =
                case PK of
                    undefined -> Opts;
                    _ -> Opts#{parent_key => PK}
                end,
            case erllama_server_generate:start(Model, Input, Opts1, self()) of
                {ok, Ref, Parser} ->
                    St1 = St#state{
                        ref = Ref,
                        parser = Parser,
                        model_mon = erlang:monitor(process, MPid),
                        status = running
                    },
                    {ok, emit(started, started_data(St1), St1)};
                {error, _} = E ->
                    E
            end;
        {error, not_loaded} ->
            {error, {model_not_loaded, Model}}
    end.

started_data(#state{id = Id, conv = Conv, model = Model}) ->
    #{run_id => Id, conversation_id => Conv, model => Model}.

stream_event({token, Bin}, #state{text = T} = St) ->
    emit(token, Bin, St#state{text = [Bin | T]});
stream_event({thinking, Bin}, #state{thinking = T} = St) ->
    emit(thinking, Bin, St#state{thinking = [Bin | T]});
stream_event({done, Stats}, St) ->
    complete(Stats, St);
stream_event({error, Reason}, St) ->
    fail(Reason, St);
stream_event(_Other, St) ->
    St.

complete(Stats, #state{parser = Parser, text = T, thinking = Th} = St) ->
    Text = iolist_to_binary(lists:reverse(T)),
    Thinking = iolist_to_binary(lists:reverse(Th)),
    Result = erllama_server_generate:finish(Parser, Text, Thinking, Stats),
    case maps:get(finish_reason, Result) of
        cancelled ->
            release_conv(St),
            terminal(cancelled, Result, St);
        _ ->
            record_turn(Result, St),
            terminal(done, Result, St)
    end.

fail(Reason, St) ->
    release_conv(St),
    terminal(error, #{error => Reason}, St).

%% Emit the last event, keep the result for `result/1', stop sending
%% to subscribers and stay around for `run_retention_ms'.
terminal(Type, Data, #state{id = Id, model_mon = Mon, retention = Retention} = St) ->
    St1 = emit(Type, Data, St),
    _ = is_reference(Mon) andalso erlang:demonitor(Mon, [flush]),
    ok = erllama_server_runs:set_status(Id, Type),
    _ = erlang:send_after(Retention, self(), expire),
    maps:foreach(fun(_, {M, _}) -> erlang:demonitor(M, [flush]) end, St1#state.subs),
    St1#state{
        status = Type,
        result = Data#{status => Type},
        subs = #{},
        model_mon = undefined,
        text = [],
        thinking = []
    }.

record_turn(_Result, #state{conv = undefined}) ->
    ok;
record_turn(Result, #state{conv = Conv, id = Id, new_msgs = New}) ->
    Reply = erllama_server_generate:reply_message(Result),
    erllama_server_convs:end_run(Conv, Id, New ++ [Reply], maps:get(finish_key, Result)).

release_conv(#state{conv = undefined}) -> ok;
release_conv(#state{conv = Conv, id = Id}) -> erllama_server_convs:release(Conv, Id).

emit(Type, Data, #state{id = Id, ring = Ring, subs = Subs, max_unacked = Max} = St) ->
    {Event, Ring1} = erllama_server_ring:push(#{type => Type, data => Data}, Ring),
    St#state{ring = Ring1, subs = deliver(Id, Event, Subs, Max)}.

deliver(Id, #{seq := Seq} = Event, Subs, Max) ->
    maps:filtermap(
        fun(Sub, {Mon, Acked}) ->
            case Seq - Acked > Max of
                true ->
                    erlang:demonitor(Mon, [flush]),
                    Sub ! {erllama_server_run, Id, {dropped, Seq}},
                    false;
                false ->
                    Sub ! {erllama_server_run, Id, Event},
                    true
            end
        end,
        Subs
    ).

%% The replay travels in the reply, so it counts as delivered: the
%% subscriber is acked up to the last event it contains.
attach_sub(Sub, Cursor, #state{ring = Ring, status = Status, subs = Subs} = St) ->
    case erllama_server_ring:from(Cursor, Ring) of
        {gap, Oldest} ->
            {reply, {error, {gap, Oldest}}, St};
        {ok, Replay} ->
            Reply = #{run_pid => self(), replay => Replay, status => Status},
            case Status of
                running ->
                    Subs1 = drop_sub(Sub, Subs),
                    Acked = erllama_server_ring:next_seq(Ring) - 1,
                    Mon = erlang:monitor(process, Sub),
                    {reply, {ok, Reply}, St#state{subs = Subs1#{Sub => {Mon, Acked}}}};
                _ ->
                    {reply, {ok, Reply}, St}
            end
    end.

drop_sub(Sub, Subs) ->
    case maps:take(Sub, Subs) of
        {{Mon, _}, Subs1} ->
            erlang:demonitor(Mon, [flush]),
            Subs1;
        error ->
            Subs
    end.

summary(#state{id = Id, conv = Conv, model = Model, status = Status, ring = Ring}) ->
    #{
        run_id => Id,
        conversation_id => Conv,
        model => Model,
        status => Status,
        next_seq => erllama_server_ring:next_seq(Ring)
    }.
