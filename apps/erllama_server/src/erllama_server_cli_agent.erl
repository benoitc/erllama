%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_cli_agent).
-moduledoc false.
%% The CLI's agent commands: `agent' (a session driven line by line:
%% the model's text, its tool activity and the approval prompts),
%% `agents' and `agent-stop'. Leaving the loop leaves the session
%% running on the release.

-export([agent/4, agents/4, stop/4]).

agent(Node, _Info, #{workspace := Ws} = Opts, [Model]) ->
    case erllama_server_client:call(Node, agent_start, [spec(Model, Ws, Opts)]) of
        {ok, #{agent_id := Id, conversation_id := Conv}} ->
            io:format(
                standard_error,
                "[agent ~ts, conversation ~ts; /quit leaves it running, /stop ends it]~n",
                [Id, Conv]
            ),
            agent_loop(Node, Id, 1);
        {error, Reason} ->
            erllama_server_cli:fail(Reason)
    end;
agent(_, _, _, _) ->
    erllama_server_cli:usage("agent").

spec(Model, Ws, Opts) ->
    Policy =
        case Opts of
            #{yes := true} -> yes;
            #{read_only := true} -> read_only;
            _ -> #{}
        end,
    Base = #{
        model => list_to_binary(Model),
        workspace => unicode:characters_to_binary(Ws),
        policy => Policy
    },
    Conv = [{conversation_id, erllama_server_cli:to_bin(C)} || #{conversation_id := C} <- [Opts]],
    Budgets = [{budgets, #{max_rounds => R}} || #{max_rounds := R} <- [Opts]],
    maps:merge(Base, maps:from_list(Conv ++ Budgets)).

agent_loop(Node, Id, Cursor) ->
    case io:get_line("you> ") of
        eof ->
            {0, ["\n[session ", Id, " keeps running]\n"]};
        {error, _} ->
            {0, ""};
        Line ->
            case string:trim(Line) of
                "/quit" ->
                    {0, ["[session ", Id, " keeps running; erllama agent-stop ", Id, " ends it]\n"]};
                "/stop" ->
                    erllama_server_cli:reply(
                        erllama_server_client:call(Node, agent_stop, [Id]), fun(_) ->
                            "[session ended]\n"
                        end
                    );
                "" ->
                    agent_loop(Node, Id, Cursor);
                Text ->
                    agent_turn(Node, Id, Cursor, unicode:characters_to_binary(Text))
            end
    end.

agent_turn(Node, Id, Cursor, Text) ->
    case erllama_server_client:call(Node, agent_send, [Id, Text]) of
        Sent when Sent =:= ok; element(1, Sent) =:= queued ->
            Sink = fun(E) -> agent_event(Node, Id, E) end,
            Stop = fun(#{type := T}) -> T =:= turn_done end,
            case erllama_server_client:follow(Node, Id, Cursor, Sink, Stop) of
                {ok, #{type := turn_done, seq := Seq}} -> agent_loop(Node, Id, Seq + 1);
                {ok, #{type := done}} -> {0, "[session ended]\n"};
                {ok, _} -> {1, "[session ended unexpectedly]\n"};
                {error, Reason} -> erllama_server_cli:fail(Reason)
            end;
        {error, Reason} ->
            erllama_server_cli:fail(Reason)
    end.

agent_event(_Node, _Id, #{type := token, data := Bin}) ->
    io:put_chars(Bin);
agent_event(_Node, _Id, #{type := tool_started, data := #{name := N, arguments := A}}) ->
    io:format(standard_error, "~n[tool] ~ts ~ts~n", [N, args_text(A)]);
agent_event(_Node, _Id, #{type := tool_result, data := #{ok := Ok, output := Out}}) ->
    Mark =
        case Ok of
            true -> "ok";
            false -> "error"
        end,
    io:format(standard_error, "[~s] ~ts~n", [Mark, first_line(Out)]);
agent_event(Node, Id, #{type := approval_request, data := Req}) ->
    ask_approval(Node, Id, Req);
agent_event(_Node, _Id, #{type := turn_done, data := #{reason := stop}}) ->
    io:put_chars("\n");
agent_event(_Node, _Id, #{type := turn_done, data := #{reason := R}}) ->
    io:format(standard_error, "~n[turn ended: ~0p]~n", [R]);
agent_event(_Node, _Id, E) ->
    erllama_server_cli:print_event(E).

ask_approval(Node, Id, #{call_id := C, name := N, arguments := A}) ->
    Answer = io:get_line(
        io_lib:format("~nallow ~ts ~ts? [y]es / [n]o / [a]lways: ", [N, args_text(A)])
    ),
    Decision =
        case string:lowercase(string:trim(Answer)) of
            "a" ++ _ -> always;
            "y" ++ _ -> allow;
            _ -> deny
        end,
    _ = erllama_server_client:call(Node, agent_approve, [Id, C, Decision]),
    ok.

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
