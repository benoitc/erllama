%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% A scripted stand-in for erllama_server_agent_gen: each generation
%% round takes the next reply of the session's script and sends it as
%% the real generator would (text deltas, announced tool calls, the
%% final result). Every round's input messages and offered tool names
%% are kept, so tests can check what the model was shown.
%%
%% A reply is `#{text => Bin, calls => [{Id, Name, ArgsMap}]}', or
%% `{hang, Text}' for a round that streams `Text' and never ends.
%% @end
-module(erllama_server_agent_script).

-export([start/4, set/2, rounds/1, tools/1]).

-define(TAB, ?MODULE).

set(AgentId, Replies) ->
    ensure_table(),
    true = ets:insert(?TAB, [
        {{script, AgentId}, Replies}, {{rounds, AgentId}, []}, {{tools, AgentId}, []}
    ]),
    ok.

%% The messages each round was given, oldest first.
rounds(AgentId) ->
    [{_, R}] = ets:lookup(?TAB, {rounds, AgentId}),
    lists:reverse(R).

%% The tool names each round was offered, oldest first.
tools(AgentId) ->
    [{_, T}] = ets:lookup(?TAB, {tools, AgentId}),
    lists:reverse(T).

ensure_table() ->
    case ets:whereis(?TAB) of
        undefined ->
            Self = self(),
            spawn(fun() ->
                _ = ets:new(?TAB, [named_table, public, set]),
                Self ! table_ready,
                receive
                    never -> ok
                end
            end),
            receive
                table_ready -> ok
            end;
        _ ->
            ok
    end.

start(Session, #{agent_id := Id}, Messages, Opts) ->
    spawn_link(fun() -> round(Session, Id, Messages, Opts) end).

round(Session, Id, Messages, Opts) ->
    [{_, Rounds}] = ets:lookup(?TAB, {rounds, Id}),
    [{_, Tools}] = ets:lookup(?TAB, {tools, Id}),
    Names = [N || #{name := N} <- maps:get(tools, Opts, [])],
    true = ets:insert(?TAB, [{{rounds, Id}, [Messages | Rounds]}, {{tools, Id}, [Names | Tools]}]),
    case ets:lookup(?TAB, {script, Id}) of
        [{_, [Reply | Rest]}] ->
            true = ets:insert(?TAB, {{script, Id}, Rest}),
            reply(Session, Reply);
        _ ->
            reply(Session, #{text => <<"(script exhausted)">>})
    end.

reply(Session, {hang, Text}) ->
    Session ! {agent_gen, self(), {delta, token, Text}},
    receive
        never -> ok
    end;
reply(Session, Reply) ->
    Text = maps:get(text, Reply, <<>>),
    Calls = [#{id => Id, name => N, arguments => A} || {Id, N, A} <- maps:get(calls, Reply, [])],
    [Session ! {agent_gen, self(), {delta, token, Text}} || Text =/= <<>>],
    [
        Session ! {agent_gen, self(), {delta, tool_call, #{index => I, id => Id, name => N}}}
     || {I, #{id := Id, name := N}} <- lists:zip(lists:seq(0, length(Calls) - 1), Calls)
    ],
    Result = #{
        content => Text,
        reasoning => <<>>,
        tool_calls => Calls,
        finish_reason => stop,
        finish_key => undefined,
        usage => #{prompt_tokens => 10, completion_tokens => 5},
        timings => #{prefill_ms => 1, generation_ms => 1},
        cache_hit_kind => cold,
        stop_sequence => undefined
    },
    Session ! {agent_gen, self(), {done, Result}}.
