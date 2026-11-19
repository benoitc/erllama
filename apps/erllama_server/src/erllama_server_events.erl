%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_events).
-moduledoc false.
%% The event stream of a job (a run or a model download), as pure state
%% kept by the job's process. Every emitted event is numbered, kept in a
%% bounded ring for reattachment and sent to each subscriber as
%% `{erllama_server_run, JobId, Event}'.
%%
%% Flow control: a subscriber acks with `JobPid ! {ack, self(), Seq}'.
%% One that falls more than `max_unacked' events behind is dropped with
%% `{erllama_server_run, JobId, {dropped, NextSeq}}' and reattaches
%% from `NextSeq'. The owning process passes subscriber `DOWN' messages
%% to `down/3'; a subscriber that exits is simply forgotten.

-export([new/1, emit/3, attach/4, detach/2, ack/3, down/3, close/1, next_seq/1]).
-export_type([events/0]).

-record(events, {
    id :: binary(),
    ring :: erllama_server_ring:ring(),
    subs = #{} :: #{pid() => {reference(), integer()}},
    max_unacked :: pos_integer()
}).
-opaque events() :: #events{}.

%% A stream for job `Id', sized from the application environment.
-spec new(binary()) -> events().
new(Id) ->
    #events{
        id = Id,
        ring = erllama_server_ring:new(erllama_server_config:get(run_buffer_events)),
        max_unacked = erllama_server_config:get(subscriber_max_unacked)
    }.

%% Number, keep and deliver one event.
-spec emit(atom(), term(), events()) -> events().
emit(Type, Data, #events{id = Id, ring = Ring, subs = Subs, max_unacked = Max} = Ev) ->
    {Event, Ring1} = erllama_server_ring:push(#{type => Type, data => Data}, Ring),
    Ev#events{ring = Ring1, subs = deliver(Id, Event, Subs, Max)}.

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

%% The events from `Cursor' on, or `{gap, Oldest}'. `Live' says whether
%% the job still emits: only then is `Sub' subscribed. The replay
%% travels in the caller's reply, so it counts as delivered.
-spec attach(pid(), non_neg_integer(), boolean(), events()) ->
    {{ok, [map()]} | {error, {gap, non_neg_integer()}}, events()}.
attach(Sub, Cursor, Live, #events{ring = Ring, subs = Subs} = Ev) ->
    case erllama_server_ring:from(Cursor, Ring) of
        {gap, Oldest} ->
            {{error, {gap, Oldest}}, Ev};
        {ok, Replay} when Live ->
            Subs1 = drop(Sub, Subs),
            Acked = erllama_server_ring:next_seq(Ring) - 1,
            Mon = erlang:monitor(process, Sub),
            {{ok, Replay}, Ev#events{subs = Subs1#{Sub => {Mon, Acked}}}};
        {ok, Replay} ->
            {{ok, Replay}, Ev}
    end.

-spec detach(pid(), events()) -> events().
detach(Sub, #events{subs = Subs} = Ev) ->
    Ev#events{subs = drop(Sub, Subs)}.

-spec ack(pid(), integer(), events()) -> events().
ack(Sub, Seq, #events{subs = Subs} = Ev) ->
    case Subs of
        #{Sub := {Mon, Acked}} when Seq > Acked -> Ev#events{subs = Subs#{Sub := {Mon, Seq}}};
        _ -> Ev
    end.

%% A `DOWN' the owner received: `{ok, Ev}' when it was a subscriber's,
%% `ignore' otherwise (the owner's own monitors).
-spec down(reference(), pid(), events()) -> {ok, events()} | ignore.
down(Mon, Pid, #events{subs = Subs} = Ev) ->
    case Subs of
        #{Pid := {Mon, _}} -> {ok, Ev#events{subs = maps:remove(Pid, Subs)}};
        _ -> ignore
    end.

%% After the terminal event: no more deliveries, no more monitors.
-spec close(events()) -> events().
close(#events{subs = Subs} = Ev) ->
    maps:foreach(fun(_, {M, _}) -> erlang:demonitor(M, [flush]) end, Subs),
    Ev#events{subs = #{}}.

-spec next_seq(events()) -> non_neg_integer().
next_seq(#events{ring = Ring}) -> erllama_server_ring:next_seq(Ring).

drop(Sub, Subs) ->
    case maps:take(Sub, Subs) of
        {{Mon, _}, Subs1} ->
            erlang:demonitor(Mon, [flush]),
            Subs1;
        error ->
            Subs
    end.
