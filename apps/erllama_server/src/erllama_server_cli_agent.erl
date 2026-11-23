%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_cli_agent).
-moduledoc false.
%% The CLI's agent commands: `agent' (a session driven line by line:
%% the model's text, its tool activity and the approval prompts),
%% `agents' and `agent-stop'. Leaving the loop leaves the session
%% running on the release. Ctrl-C interrupts the turn; a second one
%% leaves the CLI. Commands: /new, /sessions, /switch ID, /fork,
%% /compact, /prompt, /help, /quit (the session goes on), /stop.

-export([prompt_args/1]).
-export([agent/4, agents/4, stop/4]).

-define(HELP, [
    "/new           a new session, same model and workspace\n",
    "/sessions      list the agent sessions\n",
    "/switch ID     go to session ID, or start one on conversation ID\n",
    "/fork          a new session on a copy of this conversation\n",
    "/compact       shorten this session's conversation now\n",
    "/prompt SERVER NAME [KEY=VALUE ...]  send an MCP prompt\n",
    "/quit          leave; the session keeps running\n",
    "/stop          end the session\n",
    "a line ending in \\ goes on; \"\"\" opens a block up to \"\"\"\n"
]).

agent(Node, _Info, Opts, [Model]) when
    is_map_key(workspace, Opts); is_map_key(conversation_id, Opts)
->
    ok = erllama_server_cli_term:interactive(),
    case start_session(Node, spec(Model, maps:get(workspace, Opts, undefined), Opts)) of
        {ok, S} ->
            say("[/quit leaves the session running, /stop ends it, /help lists commands]\n"),
            loop(S#{model => Model, opts => Opts});
        {error, Reason} ->
            erllama_server_cli:fail(Reason)
    end;
agent(_, _, _, _) ->
    erllama_server_cli:usage("agent").

%% A new session: its id, conversation, and where its events start.
start_session(Node, Spec) ->
    case erllama_server_client:call(Node, agent_start, [Spec]) of
        {ok, #{agent_id := Id, conversation_id := Conv}} ->
            announce(Id, Conv),
            {ok, #{node => Node, id => Id, conv => Conv, cursor => 1}};
        {error, _} = E ->
            E
    end.

spec(Model, Ws, Opts) ->
    Policy =
        case Opts of
            #{yes := true} -> yes;
            #{read_only := true} -> read_only;
            _ -> #{}
        end,
    Base0 = #{model => list_to_binary(Model), policy => Policy},
    Base =
        case Ws of
            undefined -> Base0;
            _ -> Base0#{workspace => unicode:characters_to_binary(Ws)}
        end,
    Conv = [{conversation_id, erllama_server_cli:to_bin(C)} || #{conversation_id := C} <- [Opts]],
    Budgets = [{budgets, #{max_rounds => R}} || #{max_rounds := R} <- [Opts]],
    maps:merge(Base, maps:from_list(Conv ++ Budgets)).

loop(#{id := Id} = S) ->
    case erllama_server_cli_input:read("you> ") of
        eof ->
            {0, ["\n[session ", Id, " keeps running]\n"]};
        {command, "quit", _} ->
            {0, ["[session ", Id, " keeps running; erllama agent-stop ", Id, " ends it]\n"]};
        {command, "stop", _} ->
            erllama_server_cli:reply(
                erllama_server_client:call(maps:get(node, S), agent_stop, [Id]), fun(_) ->
                    "[session ended]\n"
                end
            );
        {command, Name, Args} ->
            loop(command(Name, Args, S));
        {text, Text} ->
            turn(agent_send, [Text], S);
        _ ->
            loop(S)
    end.

command("new", [], #{node := Node, model := Model, opts := Opts} = S) ->
    Ws = maps:get(workspace, Opts, undefined),
    Spec = spec(Model, Ws, maps:remove(conversation_id, Opts)),
    switch_to(start_session(Node, Spec#{conversation_id => undefined}), S);
command("sessions", [], #{node := Node} = S) ->
    {_, Out} = agents(Node, #{}, #{}, []),
    ok = erllama_server_cli_term:out(Out),
    S;
command("switch", ["agent_" ++ _ = Id], S) ->
    to_session(bin(Id), S);
command("switch", [Conv], #{node := Node, model := Model, opts := Opts} = S) ->
    switch_to(start_session(Node, spec(Model, undefined, Opts#{conversation_id => Conv})), S);
command("fork", [], S) ->
    fork(S);
command("compact", [], #{node := Node, id := Id} = S) ->
    ok = compact(Node, Id),
    S;
command("prompt", Args, S) ->
    mcp_prompt(Args, S);
command("help", _, S) ->
    ok = erllama_server_cli_term:out(?HELP),
    S;
command(Name, _, S) ->
    say(["[unknown command /", Name, "; /help lists them]\n"]),
    S.

%% An MCP server's prompt as this turn's input.
mcp_prompt(Args, S) ->
    case prompt_args(Args) of
        {ok, Server, Name, PArgs} ->
            turn(agent_prompt, [Server, Name, PArgs], S);
        error ->
            say("usage: /prompt SERVER NAME [KEY=VALUE ...]\n"),
            S
    end.

%% Go to a running session, from the next event it will send.
to_session(Id, #{node := Node} = S) ->
    case erllama_server_client:call(Node, run, [Id]) of
        {ok, #{kind := agent, conversation_id := Conv, next_seq := Next}} ->
            announce(Id, Conv),
            S#{id := Id, conv := Conv, cursor := Next};
        {ok, _} ->
            say(["[", Id, " is not an agent session]\n"]),
            S;
        {error, Reason} ->
            say_error(Reason),
            S
    end.

%% A new session on a copy of this session's conversation.
fork(#{node := Node, conv := Conv, model := Model, opts := Opts} = S) ->
    case erllama_server_client:call(Node, conversation_fork, [Conv]) of
        {ok, #{conversation_id := New}} ->
            say(["[a copy of ", Conv, ": ", New, "]\n"]),
            switch_to(
                start_session(Node, spec(Model, undefined, Opts#{conversation_id => New})), S
            );
        {error, Reason} ->
            say_error(Reason),
            S
    end.

compact(Node, Id) ->
    case erllama_server_client:call(Node, agent_compact, [Id]) of
        {ok, #{before := B, 'after' := A}} -> say(io_lib:format("[~b messages, now ~b]~n", [B, A]));
        {ok, _} -> say("[nothing to compact]\n");
        {error, Reason} -> say_error(Reason)
    end.

announce(Id, Conv) -> say(["[agent ", Id, ", conversation ", Conv, "]\n"]).

switch_to({ok, New}, S) ->
    maps:merge(S, New);
switch_to({error, Reason}, S) ->
    say_error(Reason),
    S.

%% `SERVER NAME KEY=VALUE...' as binaries.
-spec prompt_args([string()]) -> {ok, binary(), binary(), #{binary() => binary()}} | error.
prompt_args([Server, Name | Pairs]) ->
    Split = [string:split(P, "=") || P <- Pairs],
    case lists:all(fun is_pair/1, Split) of
        true ->
            Args = maps:from_list([{bin(K), bin(V)} || [K, V] <- Split]),
            {ok, bin(Server), bin(Name), Args};
        false ->
            error
    end;
prompt_args(_) ->
    error.

is_pair([_, _]) -> true;
is_pair(_) -> false.

bin(S) -> unicode:characters_to_binary(S).

%% Send input (or a prompt) and follow the turn. Ctrl-C interrupts it.
turn(Fun, Args, #{node := Node, id := Id, cursor := Cursor} = S) ->
    case erllama_server_client:call(Node, Fun, [Id | Args]) of
        Sent when Sent =:= ok; element(1, Sent) =:= queued ->
            Sink = fun(E) -> event(Node, Id, E) end,
            Stop = fun(#{type := T}) -> T =:= turn_done end,
            Result = erllama_server_cli_term:interrupt(
                fun() -> erllama_server_client:call(Node, agent_interrupt, [Id]) end,
                fun() -> erllama_server_client:follow(Node, Id, Cursor, Sink, Stop) end
            ),
            case Result of
                {ok, #{type := turn_done, seq := Seq}} -> loop(S#{cursor := Seq + 1});
                {ok, #{type := done}} -> {0, "[session ended]\n"};
                {ok, _} -> {1, "[session ended unexpectedly]\n"};
                {error, Reason} -> erllama_server_cli:fail(Reason)
            end;
        {error, Reason} ->
            say_error(Reason),
            loop(S)
    end.

event(_Node, _Id, #{type := token, data := Bin}) ->
    erllama_server_cli_term:out(Bin);
event(_Node, _Id, #{type := thinking, data := Bin}) ->
    erllama_server_cli_term:dim(Bin);
event(_Node, _Id, #{type := tool_call}) ->
    %% Shown once it starts, with its arguments (tool_started).
    ok;
event(_Node, _Id, #{type := T}) when T =:= turn_started; T =:= round ->
    erllama_server_cli_term:status("thinking");
event(_Node, _Id, #{type := tool_started, data := #{name := N, arguments := A}}) ->
    ok = erllama_server_cli_term:status(N),
    say(io_lib:format("~n[tool] ~ts ~ts~n", [N, args_text(A)]));
event(_Node, _Id, #{type := tool_result, data := #{ok := Ok, output := Out}}) ->
    Mark =
        case Ok of
            true -> "ok";
            false -> "error"
        end,
    ok = erllama_server_cli_term:status("thinking"),
    say(io_lib:format("[~s] ~ts~n", [Mark, first_line(Out)]));
event(Node, Id, #{type := approval_request, data := Req}) ->
    ok = erllama_server_cli_term:status(undefined),
    ask_approval(Node, Id, Req);
event(_Node, _Id, #{type := turn_done, data := #{reason := stop}}) ->
    ok = erllama_server_cli_term:status(undefined),
    erllama_server_cli_term:out("\n");
event(_Node, _Id, #{type := turn_done, data := #{reason := R}}) ->
    ok = erllama_server_cli_term:status(undefined),
    say(io_lib:format("~n[turn ended: ~0p]~n", [R]));
event(_Node, _Id, E) ->
    erllama_server_cli:print_event(E).

ask_approval(Node, Id, #{call_id := C, name := N, arguments := A}) ->
    Answer = erllama_server_cli_term:prompt(
        io_lib:format("~nallow ~ts ~ts? [y]es / [n]o / [a]lways: ", [N, args_text(A)])
    ),
    Decision =
        case Answer of
            L when is_list(L) ->
                case string:lowercase(string:trim(L)) of
                    "a" ++ _ -> always;
                    "y" ++ _ -> allow;
                    _ -> deny
                end;
            _ ->
                deny
        end,
    _ = erllama_server_client:call(Node, agent_approve, [Id, C, Decision]),
    ok.

say(Text) -> erllama_server_cli_term:err(Text).

say_error(Reason) -> erllama_server_cli_term:err(erllama_server_cli:format_error(Reason)).

args_text(A) ->
    T = iolist_to_binary(json:encode(A)),
    case byte_size(T) > 200 of
        true -> <<(binary:part(T, 0, 200))/binary, "...">>;
        false -> T
    end.

first_line(Out) ->
    hd(binary:split(Out, <<"\n">>) ++ [<<>>]).

agents(Node, _Info, _Opts, []) ->
    erllama_server_cli:reply(erllama_server_client:call(Node, agents, []), fun
        ([]) ->
            "no agent sessions\n";
        (L) ->
            [
                io_lib:format("~ts  ~p  ~ts  ~ts  ~ts~n", [Id, St, M, Conv, Ws])
             || #{run_id := Id, state := St, model := M, conversation_id := Conv, workspace := Ws} <-
                    L
            ]
    end);
agents(_, _, _, _) ->
    erllama_server_cli:usage("agents").

stop(Node, _Info, _Opts, [Id]) ->
    erllama_server_cli:reply(
        erllama_server_client:call(Node, agent_stop, [list_to_binary(Id)]), fun(_) -> "" end
    );
stop(_, _, _, _) ->
    erllama_server_cli:usage("agent-stop").
