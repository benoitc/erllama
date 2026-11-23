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
%%
%% Going back to a running session (`agent --attach ID', /switch ID)
%% shows what it is doing, then replays its latest turn from the
%% session's event buffer and follows it live: output produced while
%% no client was attached is shown, and a part of the turn the bounded
%% buffer no longer holds is reported as a gap.

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

agent(Node, _Info, #{attach := Id} = Opts, []) ->
    ok = erllama_server_cli_term:interactive(),
    S0 = #{node => Node, id => undefined, conv => undefined, cursor => 0, opts => Opts},
    case to_session(bin(Id), S0) of
        #{id := undefined} -> {1, ""};
        S -> loop(S)
    end;
agent(Node, _Info, Opts, [Model]) when
    is_map_key(workspace, Opts); is_map_key(conversation_id, Opts)
->
    ok = erllama_server_cli_term:interactive(),
    case start_session(Node, spec(Model, maps:get(workspace, Opts, undefined), Opts)) of
        {ok, S} ->
            say("[/quit leaves the session running, /stop ends it, /help lists commands]\n"),
            loop(S#{model => Model, opts => Opts});
        {error, conversation_busy} ->
            {1, busy_text(Node, maps:get(conversation_id, Opts))};
        {error, Reason} ->
            erllama_server_cli:fail(Reason)
    end;
agent(_, _, _, _) ->
    erllama_server_cli:usage("agent").

%% The conversation is held by a running job: name it, and how to go
%% back to it when it is a session.
busy_text(Node, Conv) ->
    case erllama_server_client:call(Node, conversation, [erllama_server_cli:to_bin(Conv)]) of
        {ok, #{active_run := <<"agent_", _/binary>> = Id}} ->
            [
                "erllama: session ",
                Id,
                " is working in this conversation; erllama agent --attach ",
                Id,
                "\n"
            ];
        {ok, #{active_run := Id}} when is_binary(Id) ->
            ["erllama: run ", Id, " is working in this conversation; erllama attach ", Id, "\n"];
        _ ->
            "erllama: the conversation already has a run in progress\n"
    end.

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

%% Go to a running session: what it is doing, then its latest turn
%% (replayed, then followed live while it runs).
to_session(Id, #{node := Node} = S) ->
    case erllama_server_client:call(Node, inspect, [Id]) of
        {ok, #{kind := agent, conversation_id := Conv, next_seq := Next, model := M} = Info} ->
            announce(Id, Conv),
            say(arrival_text(Info)),
            S1 = S#{id := Id, conv := Conv, cursor := Next, model => binary_to_list(M)},
            Pending = [C || #{call_id := C} <- maps:get(approvals, Info, [])],
            catch_up(maps:get(turn_seq, Info, undefined), Next, Pending, S1);
        {ok, _} ->
            say(["[", Id, " is not an agent session]\n"]),
            S;
        {error, Reason} ->
            say_error(Reason),
            S
    end.

arrival_text(#{state := State} = I) ->
    Parts =
        [
            T
         || {K, What} <- [{approvals, "approval(s) pending"}, {tools, "tool call(s) running"}],
            N <- [length(maps:get(K, I, []))],
            N > 0,
            T <- [io_lib:format("~b ~s", [N, What])]
        ] ++
            [
                io_lib:format("~b background job(s)", [N])
             || N <- [length([J || #{running := true} = J <- maps:get(jobs, I, [])])],
                N > 0
            ],
    ["[", atom_to_list(State), [[", ", P] || P <- Parts], "]\n"].

%% Replay the latest turn from `TurnSeq' up to its end (following it
%% live if it still runs). A replayed approval request is asked only
%% if that call is still pending (`Pending', as of event `Next').
catch_up(undefined, _Next, _Pending, S) ->
    S;
catch_up(TurnSeq, Next, Pending, #{node := Node, id := Id} = S) ->
    say(io_lib:format("[the latest turn, from event ~b]~n", [TurnSeq])),
    Sink = fun
        (#{type := approval_request, seq := Seq, data := #{call_id := C, name := N}} = E) when
            Seq < Next
        ->
            case lists:member(C, Pending) of
                true -> event(Node, Id, E);
                false -> say(["[", N, " ", C, " was already answered]\n"])
            end;
        (E) ->
            event(Node, Id, E)
    end,
    case follow_turn(Node, Id, TurnSeq, Sink) of
        {ok, #{type := turn_done, seq := Seq}} -> S#{cursor := Seq + 1};
        _ -> S
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

%% Follow a session from `Cursor' to the end of a turn; Ctrl-C
%% interrupts the turn.
follow_turn(Node, Id, Cursor, Sink) ->
    erllama_server_cli_term:interrupt(
        fun() -> erllama_server_client:call(Node, agent_interrupt, [Id]) end,
        fun() ->
            erllama_server_client:follow(Node, Id, Cursor, Sink, fun(#{type := T}) ->
                T =:= turn_done
            end)
        end
    ).

%% Send input (or a prompt) and follow the turn. Ctrl-C interrupts it.
turn(Fun, Args, #{node := Node, id := Id, cursor := Cursor} = S) ->
    case erllama_server_client:call(Node, Fun, [Id | Args]) of
        Sent when Sent =:= ok; element(1, Sent) =:= queued ->
            case follow_turn(Node, Id, Cursor, fun(E) -> event(Node, Id, E) end) of
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
