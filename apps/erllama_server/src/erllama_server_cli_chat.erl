%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_cli_chat).
-moduledoc false.
%% `erllama chat MODEL': a conversation typed line by line (see
%% erllama_server_cli_input for multiline input), its replies streamed.
%% Ctrl-C cancels the turn being generated; the conversation stays.
%% Commands work on conversations: /new, /sessions, /switch ID, /fork,
%% /compact, /help, /quit.

-export([chat/4]).

-define(HELP, [
    "/new           start a new conversation\n",
    "/sessions      list the conversations\n",
    "/switch ID     continue conversation ID\n",
    "/fork          go on in a copy of this conversation\n",
    "/compact       shorten this conversation now\n",
    "/quit          leave (end of input does too)\n",
    "a line ending in \\ goes on; \"\"\" opens a block up to \"\"\"\n"
]).

-spec chat(node(), map(), map(), [string()]) -> {non_neg_integer(), iodata()}.
chat(Node, _Info, Opts, [Model]) ->
    ok = erllama_server_cli_term:interactive(),
    loop(#{node => Node, model => Model, opts => Opts});
chat(_, _, _, _) ->
    erllama_server_cli:usage("chat").

loop(S) ->
    case erllama_server_cli_input:read("> ") of
        eof -> {0, ""};
        {command, "quit", _} -> {0, ""};
        {command, Name, Args} -> loop(command(Name, Args, S));
        {text, Text} -> loop(turn(Text, S));
        _ -> loop(S)
    end.

turn(Text, #{node := Node, model := Model, opts := Opts} = S) ->
    case start(Node, Model, Text, Opts) of
        {ok, #{run_id := Run, conversation_id := Conv}} ->
            {_, Out} = erllama_server_cli:follow_cancelling(Node, Run),
            ok = erllama_server_cli_term:out(Out),
            S#{opts := Opts#{conversation_id => Conv}};
        {error, Reason} ->
            say_error(Reason),
            S
    end.

start(Node, Model, Text, Opts) ->
    Spec0 = #{
        model => list_to_binary(Model),
        messages => [#{role => user, content => Text}],
        opts => maps:with([response_tokens], Opts)
    },
    Spec =
        case Opts of
            #{conversation_id := C} -> Spec0#{conversation_id => erllama_server_cli:to_bin(C)};
            _ -> Spec0
        end,
    erllama_server_client:call(Node, start_run, [Spec]).

command("new", [], #{opts := Opts} = S) ->
    say("[a new conversation starts with the next message]\n"),
    S#{opts := maps:remove(conversation_id, Opts)};
command("sessions", [], #{node := Node} = S) ->
    {_, Out} = erllama_server_cli_conv:list(Node, #{}, #{}, []),
    ok = erllama_server_cli_term:out(Out),
    S;
command("switch", [Id], #{node := Node, opts := Opts} = S) ->
    case erllama_server_client:call(Node, conversation, [bin(Id)]) of
        {ok, #{id := C}} ->
            say(["[conversation ", C, "]\n"]),
            S#{opts := Opts#{conversation_id => C}};
        {error, Reason} ->
            say_error(Reason),
            S
    end;
command("fork", [], #{opts := #{conversation_id := C}} = S) ->
    fork(C, S);
command("compact", [], #{node := Node, opts := #{conversation_id := C}} = S) ->
    ok = compact(Node, C),
    S;
command(Name, [], S) when Name =:= "fork"; Name =:= "compact" ->
    say("[no conversation yet: send a message first]\n"),
    S;
command("help", _, S) ->
    ok = erllama_server_cli_term:out(?HELP),
    S;
command(Name, _, S) ->
    say(["[unknown command /", Name, "; /help lists them]\n"]),
    S.

fork(C, #{node := Node, opts := Opts} = S) ->
    case erllama_server_client:call(Node, conversation_fork, [bin(C)]) of
        {ok, #{conversation_id := New}} ->
            say(["[going on in ", New, ", a copy of ", C, "]\n"]),
            S#{opts := Opts#{conversation_id => New}};
        {error, Reason} ->
            say_error(Reason),
            S
    end.

compact(Node, C) ->
    case erllama_server_client:call(Node, conversation_compact, [bin(C)]) of
        {ok, #{before := B, 'after' := A}} -> say(io_lib:format("[~b messages, now ~b]~n", [B, A]));
        {ok, _} -> say("[nothing to compact]\n");
        {error, Reason} -> say_error(Reason)
    end.

say(Text) -> erllama_server_cli_term:err(Text).

say_error(Reason) -> erllama_server_cli_term:err(erllama_server_cli:format_error(Reason)).

bin(S) -> unicode:characters_to_binary(S).
