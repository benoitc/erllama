%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_agent_gen).
-moduledoc false.
%% One generation round of an agent session, in a process linked to the
%% session: it renders the transcript, streams the model's reply through
%% erllama_server_generate (parsed deltas, then the final result) and
%% reports to the session as `{agent_gen, Pid, Msg}':
%%
%%   {delta, token | thinking | tool_call, Data}
%%   {done, Result}       Result as erllama_server_generate:finish/4
%%   {error, Reason}
%%
%% The session cancels a round by killing this process: it is the
%% stream's receiver, so erllama cancels the stream with it.
%%
%% Tests replace this module (the session's `generator' option) with one
%% that sends the same messages from a script.

-behaviour(erllama_server_agent).

-export([start/4]).

%% A round with no word from the engine for this long has stalled.
-define(STALL_MS, 300000).

%% `Ctx': `model' (an erllama model id the session holds).
-spec start(pid(), map(), [map()], map()) -> pid().
start(Session, #{model := Model}, Messages, Opts) ->
    spawn_link(fun() -> run(Session, Model, Messages, Opts) end).

%% The model is held for the session (released when its turn ends) and
%% loaded here if needed, so the session itself never blocks on a load.
run(Session, Model0, Messages, Opts) ->
    case erllama_server_models:acquire(Model0, Session, undefined) of
        {ok, Model} -> generate(Session, Model, Messages, Opts);
        {error, Reason} -> Session ! {agent_gen, self(), {error, Reason}}
    end.

generate(Session, Model, Messages, Opts) ->
    case erllama_server_generate:start(Model, {messages, Messages}, Opts, self()) of
        {ok, Ref, Parser} ->
            loop(Session, Ref, Parser, erllama_server_generate:stream_parser(Parser), [], []);
        {error, Reason} ->
            Session ! {agent_gen, self(), {error, Reason}}
    end.

loop(Session, Ref, Parser, Sp, Text, Thinking) ->
    receive
        {erllama, Ref, {token, Bin}} when Sp =:= undefined ->
            Session ! {agent_gen, self(), {delta, token, Bin}},
            loop(Session, Ref, Parser, Sp, [Bin | Text], Thinking);
        {erllama, Ref, {token, Bin}} ->
            {Deltas, Sp1} = erllama_server_generate:feed(Bin, Sp),
            _ = [Session ! {agent_gen, self(), {delta, T, D}} || {T, D} <- Deltas],
            loop(Session, Ref, Parser, Sp1, [Bin | Text], Thinking);
        {erllama, Ref, {thinking, Bin}} ->
            Session ! {agent_gen, self(), {delta, thinking, Bin}},
            loop(Session, Ref, Parser, Sp, Text, [Bin | Thinking]);
        {erllama, Ref, {done, Stats}} ->
            T = iolist_to_binary(lists:reverse(Text)),
            Th = iolist_to_binary(lists:reverse(Thinking)),
            Session !
                {agent_gen, self(), {done, erllama_server_generate:finish(Parser, T, Th, Stats)}};
        {erllama, Ref, {error, Reason}} ->
            Session ! {agent_gen, self(), {error, Reason}};
        {erllama, Ref, _Other} ->
            loop(Session, Ref, Parser, Sp, Text, Thinking)
    after ?STALL_MS ->
        erllama:cancel(Ref),
        Session ! {agent_gen, self(), {error, stalled}}
    end.
