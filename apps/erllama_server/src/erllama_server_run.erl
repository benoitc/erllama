%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_run).
-moduledoc false.
%% One generation run. The process owns the erllama stream and turns
%% it into the job event stream of erllama_server_events (numbered,
%% buffered for reattachment, flow-controlled). A subscriber that exits
%% is forgotten; the run goes on (runs outlive the clients that started
%% them).
%%
%% This process is the stream's receiver: erllama cancels the stream
%% when it exits, however it exits (shutdown, crash or kill), so a
%% dead run never leaves its model generating.
%%
%% A conversation turn whose transcript nears the model's context is
%% compacted first (erllama_server_compact), in a helper, once the
%% model is held: `compacting', then `compacted' events.
%%
%% Admission waits for a free sequence when the model serves as many
%% requests as it has; the stream is started from a helper, so the run
%% keeps answering (info, attach, cancel) while it waits. Its phase
%% (`loading', `compacting', `waiting', `generating') is kept in the
%% job table for listings.

-behaviour(gen_server).

-export([start_link/1, go/1]).
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
    %% With a parser, tokens go through it and only parsed deltas are
    %% emitted (content, reasoning, tool calls).
    sp :: erllama_server_generate:stream_parser() | undefined,
    events :: erllama_server_events:events(),
    status = pending :: pending | erllama_server_runs:status(),
    result :: map() | undefined,
    text = [] :: [binary()],
    thinking = [] :: [binary()],
    retention :: non_neg_integer(),
    keep_alive :: non_neg_integer() | infinity | undefined,
    %% The model is held (erllama_server_models) from acquisition to the
    %% terminal event; a cancel that arrives while it loads is applied
    %% once it is loaded.
    acquired = false :: boolean(),
    cancel_requested = false :: boolean(),
    %% The helper compacting the conversation before the turn.
    compactor :: pid() | undefined,
    %% The helper starting the stream (admission may wait for a free
    %% sequence).
    admitting :: pid() | undefined,
    %% Stream events that arrived before the helper said which stream
    %% is ours (newest first).
    early = [] :: [{reference(), term()}],
    phase = loading :: erllama_server_runs:phase()
}).
-type state() :: #state{}.

-spec start_link(map()) -> {ok, pid()} | {error, term()}.
start_link(Spec) ->
    gen_server:start_link(?MODULE, Spec, []).

%% Start the run: it answers at once, then loads the model if needed
%% (a `loading' event) and starts generating; a failure from then on
%% is an `error' event. The other calls (attach, detach, cancel,
%% result, info) go through erllama_server_job.
-spec go(pid()) -> ok | {error, term()}.
go(Pid) -> erllama_server_job:call(Pid, go, ?GO_TIMEOUT).

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
        events = erllama_server_events:new(Id),
        retention = erllama_server_config:get(run_retention_ms),
        keep_alive = maps:get(keep_alive, Spec, undefined)
    }}.

handle_call(go, _From, #state{status = pending, model = Model, keep_alive = KA} = St) ->
    St1 = emit(started, started_data(St), St#state{status = running}),
    St2 =
        case erllama:whereis(Model) of
            {ok, _} -> St1;
            {error, not_loaded} -> emit(loading, #{model => Model}, St1)
        end,
    Self = self(),
    _ = spawn_link(fun() -> Self ! {acquired, erllama_server_models:acquire(Model, Self, KA)} end),
    {reply, ok, set_phase(loading, St2)};
handle_call(go, _From, St) ->
    {reply, {error, already_started}, St};
handle_call({attach, Sub, Cursor}, _From, #state{events = Ev, status = Status} = St) ->
    case erllama_server_events:attach(Sub, Cursor, Status =:= running, Ev) of
        {{ok, Replay}, Ev1} ->
            Reply = #{run_pid => self(), replay => Replay, status => Status},
            {reply, {ok, Reply}, St#state{events = Ev1}};
        {{error, _} = E, Ev1} ->
            {reply, E, St#state{events = Ev1}}
    end;
handle_call({detach, Sub}, _From, #state{events = Ev} = St) ->
    {reply, ok, St#state{events = erllama_server_events:detach(Sub, Ev)}};
handle_call(cancel, _From, #state{status = running, compactor = C} = St) when is_pid(C) ->
    unlink(C),
    exit(C, kill),
    Result = erllama_server_generate:finish(undefined, <<>>, <<>>, #{finish_reason => cancelled}),
    release_conv(St),
    {reply, ok, terminal(cancelled, Result, St#state{compactor = undefined})};
handle_call(cancel, _From, #state{status = running, admitting = H} = St) when is_pid(H) ->
    %% Still waiting for a sequence: end now. A stream admitted later
    %% is cancelled when it arrives (`stream_started').
    Result = erllama_server_generate:finish(undefined, <<>>, <<>>, #{finish_reason => cancelled}),
    release_conv(St),
    {reply, ok, terminal(cancelled, Result, St)};
handle_call(cancel, _From, #state{status = running, ref = undefined} = St) ->
    {reply, ok, St#state{cancel_requested = true}};
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

handle_info({acquired, {ok, _Id}}, #state{status = running, cancel_requested = true} = St) ->
    Result = erllama_server_generate:finish(undefined, <<>>, <<>>, #{finish_reason => cancelled}),
    release_conv(St),
    {noreply, terminal(cancelled, Result, St#state{acquired = true})};
handle_info({acquired, {ok, _Id}}, #state{status = running, conv = Conv} = St) when
    is_binary(Conv)
->
    {noreply, maybe_compact(St#state{acquired = true})};
handle_info({acquired, {ok, _Id}}, #state{status = running} = St) ->
    {noreply, stream_or_fail(St#state{acquired = true})};
handle_info({compacting, C}, #state{compactor = C, status = running} = St) ->
    {noreply, set_phase(compacting, emit(compacting, #{}, St))};
handle_info({compaction, C, Outcome}, #state{compactor = C, status = running} = St) ->
    {noreply, stream_or_fail(compacted(Outcome, St#state{compactor = undefined}))};
handle_info({acquired, {error, Reason}}, #state{status = running} = St) ->
    {noreply, fail(Reason, St)};
handle_info({acquired, {ok, Id}}, St) ->
    erllama_server_models:release(Id, self()),
    {noreply, St};
handle_info({stream_started, H, {ok, Ref, Parser}}, #state{admitting = H, status = running} = St) ->
    St1 = St#state{
        admitting = undefined,
        early = [],
        ref = Ref,
        parser = Parser,
        sp = erllama_server_generate:stream_parser(Parser)
    },
    Early = [E || {R, E} <- lists:reverse(St#state.early), R =:= Ref],
    {noreply, replay_early(Early, set_phase(generating, St1))};
handle_info({stream_started, H, {error, Reason}}, #state{admitting = H, status = running} = St) ->
    {noreply, fail(Reason, St#state{admitting = undefined})};
handle_info({stream_started, _H, {ok, Ref, _Parser}}, St) ->
    %% Admitted after the run ended (cancelled while waiting, or its
    %% model went down): nobody wants this stream.
    _ = erllama:cancel(Ref),
    {noreply, St#state{admitting = undefined, early = []}};
handle_info({stream_started, _H, _}, St) ->
    {noreply, St#state{admitting = undefined, early = []}};
handle_info({'EXIT', _, normal}, St) ->
    {noreply, St};
handle_info({erllama, Ref, Event}, #state{ref = Ref, status = running} = St) ->
    {noreply, stream_event(Event, St)};
handle_info({erllama, Ref, Event}, #state{admitting = H, early = Early} = St) when is_pid(H) ->
    %% The stream sends to this process as soon as it is admitted,
    %% which can be before the helper's `stream_started'.
    {noreply, St#state{early = [{Ref, Event} | Early]}};
handle_info({ack, Sub, Seq}, #state{events = Ev} = St) when is_integer(Seq) ->
    {noreply, St#state{events = erllama_server_events:ack(Sub, Seq, Ev)}};
handle_info({'DOWN', Mon, process, _, Reason}, #state{model_mon = Mon, status = running} = St) ->
    ok = erllama_server_failures:record(model_down, St#state.model, #{
        run => St#state.id, reason => Reason
    }),
    {noreply, fail({model_down, Reason}, St#state{model_mon = undefined})};
handle_info({'DOWN', Mon, process, Pid, _}, #state{events = Ev} = St) ->
    case erllama_server_events:down(Mon, Pid, Ev) of
        {ok, Ev1} -> {noreply, St#state{events = Ev1}};
        ignore -> {noreply, St}
    end;
handle_info(expire, St) ->
    {stop, normal, St};
handle_info(_Msg, St) ->
    {noreply, St}.

%% -----------------------------------------------------------------------------

stream_or_fail(St) ->
    case start_stream(St) of
        {ok, St1} -> St1;
        {error, Reason} -> fail(Reason, St)
    end.

%% Measure, and compact if needed, in a helper: the run keeps answering
%% meanwhile.
maybe_compact(#state{model = Model, input = {messages, All}} = St) ->
    Self = self(),
    Pid = spawn_link(fun() ->
        Outcome = erllama_server_compact:run(Model, All, #{
            on_start => fun() -> Self ! {compacting, self()} end
        }),
        Self ! {compaction, self(), Outcome}
    end),
    St#state{compactor = Pid};
maybe_compact(St) ->
    stream_or_fail(St).

%% The compacted history replaces the conversation's (without this
%% turn's new messages, which end_run adds as usual); the turn starts
%% from no cache key.
compacted({compacted, NewAll, Info}, #state{conv = Conv, id = Id, new_msgs = New} = St) ->
    History = lists:sublist(NewAll, length(NewAll) - length(New)),
    ok = erllama_server_convs:compact(Conv, Id, History, Info),
    emit(compacted, Info, St#state{input = {messages, NewAll}, parent_key = undefined});
compacted({error, Reason}, St) ->
    logger:warning(#{msg => "compaction failed; the turn goes on", reason => Reason}),
    St;
compacted(none, St) ->
    St.

start_stream(#state{model = Model, input = Input, opts = Opts, parent_key = PK} = St) ->
    case erllama:whereis(Model) of
        {ok, MPid} ->
            Opts1 =
                case PK of
                    undefined -> Opts;
                    _ -> Opts#{parent_key => PK}
                end,
            Self = self(),
            Helper = spawn_link(fun() ->
                Self ! {stream_started, self(), start_generation(Model, Input, Opts1, Self)}
            end),
            St1 = St#state{admitting = Helper, model_mon = erlang:monitor(process, MPid)},
            {ok, set_phase(waiting, St1)};
        {error, not_loaded} ->
            {error, {model_not_loaded, Model}}
    end.

replay_early([], St) ->
    St;
replay_early([E | Rest], #state{status = running} = St) ->
    replay_early(Rest, stream_event(E, St));
replay_early(_, St) ->
    St.

%% In the helper: a crash in it is this run's error, not the run's
%% own crash.
start_generation(Model, Input, Opts, To) ->
    try
        erllama_server_generate:start(Model, Input, Opts, To)
    catch
        Class:Reason -> {error, {Class, Reason}}
    end.

set_phase(Phase, #state{id = Id} = St) ->
    ok = erllama_server_runs:set_phase(Id, Phase),
    St#state{phase = Phase}.

started_data(#state{id = Id, conv = Conv, model = Model}) ->
    #{run_id => Id, conversation_id => Conv, model => Model}.

stream_event({token, Bin}, #state{text = T, sp = undefined} = St) ->
    emit(token, Bin, St#state{text = [Bin | T]});
stream_event({token, Bin}, #state{text = T, sp = Sp} = St) ->
    {Deltas, Sp1} = erllama_server_generate:feed(Bin, Sp),
    lists:foldl(
        fun({Type, Data}, Acc) -> emit(Type, Data, Acc) end,
        St#state{text = [Bin | T], sp = Sp1},
        Deltas
    );
stream_event({logprobs, Entry}, St) ->
    emit(logprobs, Entry, St);
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

fail(Reason, #state{id = Id} = St) ->
    ok = erllama_server_failures:record(run_failed, Id, Reason),
    release_conv(St),
    terminal(error, #{error => Reason}, St).

%% Emit the last event, keep the result for `result/1', stop sending
%% to subscribers and stay around for `run_retention_ms'.
terminal(Type, Data, #state{id = Id, model_mon = Mon, retention = Retention} = St) ->
    _ = St#state.acquired andalso erllama_server_models:release(St#state.model, self()),
    St1 = emit(Type, Data, St),
    _ = is_reference(Mon) andalso erlang:demonitor(Mon, [flush]),
    ok = erllama_server_runs:set_status(Id, Type),
    _ = erlang:send_after(Retention, self(), expire),
    St1#state{
        status = Type,
        result = Data#{status => Type},
        events = erllama_server_events:close(St1#state.events),
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

emit(Type, Data, #state{events = Ev} = St) ->
    St#state{events = erllama_server_events:emit(Type, Data, Ev)}.

summary(#state{id = Id, conv = Conv, model = Model, status = Status, phase = Phase, events = Ev}) ->
    Base = #{
        run_id => Id,
        kind => run,
        conversation_id => Conv,
        model => Model,
        status => Status,
        next_seq => erllama_server_events:next_seq(Ev)
    },
    case Status of
        running -> Base#{phase => Phase};
        _ -> Base
    end.
