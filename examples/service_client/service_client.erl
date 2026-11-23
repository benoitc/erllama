%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%%
%% Use a running erllama_server release from your own Erlang node, over
%% distribution, through its service API (erllama_server_api). The
%% release owns the model, the conversation and the run; this node only
%% starts work and reads events, so it can leave and come back.
%%
%%   {ok, Run} = service_client:start(Node, <<"qwen">>, <<"Count to fifty.">>),
%%   service_client:follow(Node, Run, 0).
%%
%% Events arrive as `{erllama_server_run, RunId, #{seq, type, data}}'.
%% A subscriber acks with `RunPid ! {ack, self(), Seq}'; one that falls
%% too far behind is sent `{dropped, NextSeq}' and attaches again.
-module(service_client).

-export([connect/1, start/3, start/4, follow/3, status/1, inspect/2, cancel/2]).

-define(ACK_EVERY, 32).

%% Connect this node to the release (`Node' such as
%% 'erllama_server@127.0.0.1'; same cookie).
-spec connect(node()) -> ok | {error, term()}.
connect(Node) ->
    case net_kernel:connect_node(Node) of
        true -> ok;
        false -> {error, {unreachable, Node}}
    end.

-spec start(node(), binary(), binary()) -> {ok, binary()} | {error, term()}.
start(Node, Model, Prompt) ->
    start(Node, Model, Prompt, #{}).

%% One chat turn, in a new conversation unless `conversation_id' is in
%% `Opts'. The run goes on whatever happens to this node.
-spec start(node(), binary(), binary(), map()) -> {ok, binary()} | {error, term()}.
start(Node, Model, Prompt, Opts) ->
    Spec = maps:merge(
        #{model => Model, messages => [#{role => <<"user">>, content => Prompt}]},
        Opts
    ),
    case erpc:call(Node, erllama_server_api, start_run, [Spec]) of
        {ok, #{run_id := Run}} -> {ok, Run};
        {error, _} = E -> E
    end.

%% Print a run from event `Cursor' to its end, and return that event.
%% A cursor the release no longer holds is reported and the replay
%% goes on from the oldest event kept.
-spec follow(node(), binary(), non_neg_integer()) -> {ok, map()} | {error, term()}.
follow(Node, Run, Cursor) ->
    case erpc:call(Node, erllama_server_api, attach, [Run, self(), Cursor]) of
        {ok, #{run_pid := Pid, replay := Replay}} ->
            case events(Replay, Cursor) of
                {done, E} -> {ok, E};
                {next, Next} -> live(Node, Run, Pid, Next, 0)
            end;
        {error, {gap, Oldest}} ->
            io:format("[events ~b to ~b are no longer kept]~n", [Cursor, Oldest - 1]),
            follow(Node, Run, Oldest);
        {error, _} = E ->
            E
    end.

-spec status(node()) -> {ok, map()}.
status(Node) ->
    erpc:call(Node, erllama_server_api, status, []).

-spec inspect(node(), binary()) -> {ok, map()} | {error, term()}.
inspect(Node, Run) ->
    erpc:call(Node, erllama_server_api, inspect, [Run]).

-spec cancel(node(), binary()) -> ok | {error, term()}.
cancel(Node, Run) ->
    erpc:call(Node, erllama_server_api, cancel, [Run]).

events([], Next) ->
    {next, Next};
events([#{seq := Seq} = E | Rest], Next) when Seq >= Next ->
    case show(E) of
        done -> {done, E};
        more -> events(Rest, Seq + 1)
    end;
events([_ | Rest], Next) ->
    events(Rest, Next).

live(Node, Run, Pid, Next, Unacked) ->
    receive
        {erllama_server_run, Run, #{seq := Next} = E} ->
            case show(E) of
                done ->
                    {ok, E};
                more when Unacked + 1 >= ?ACK_EVERY ->
                    Pid ! {ack, self(), Next},
                    live(Node, Run, Pid, Next + 1, 0);
                more ->
                    live(Node, Run, Pid, Next + 1, Unacked + 1)
            end;
        {erllama_server_run, Run, {dropped, From}} ->
            follow(Node, Run, From)
    after 30000 ->
        {error, timeout}
    end.

show(#{type := token, data := Bin}) ->
    io:put_chars(Bin),
    more;
show(#{type := done, data := #{finish_reason := Why}}) ->
    io:format("~n[done: ~p]~n", [Why]),
    done;
show(#{type := T} = E) when T =:= done; T =:= cancelled; T =:= error ->
    io:format("~n[~p ~0p]~n", [T, maps:get(data, E, undefined)]),
    done;
show(_) ->
    more.
