%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_cli_input).
-moduledoc false.
%% What the user types at a `chat' or `agent' prompt:
%%
%%   text           a message (one line)
%%   text \         continues on the next line
%%   """            starts a block, up to a line with """
%%   /name args     a command
%%
%% A line on which Ctrl-C was pressed (in cooked mode the key is part of
%% the line) is dropped.

-export([read/1, read/2]).

-type input() ::
    {text, binary()}
    | {command, string(), [string()]}
    | dropped
    | empty
    | eof.
-export_type([input/0]).

-define(MORE, "... ").

-spec read(iodata()) -> input().
read(Prompt) ->
    read(Prompt, fun erllama_server_cli_term:prompt/1).

%% With the line reader given (tests).
-spec read(iodata(), fun((iodata()) -> string() | eof | {error, term()})) -> input().
read(Prompt, Read) ->
    case line(Read(Prompt)) of
        {ok, L} -> first(L, Read);
        Other -> Other
    end.

line(eof) ->
    eof;
line({error, _}) ->
    eof;
line(L) when is_list(L); is_binary(L) ->
    S = unicode:characters_to_list(L),
    case lists:member(3, S) of
        true -> dropped;
        false -> {ok, string:trim(S, trailing, "\r\n")}
    end.

first(L, Read) ->
    case string:trim(L) of
        "" ->
            empty;
        "\"\"\"" ->
            block(Read, []);
        "/" ++ Cmd ->
            case string:lexemes(Cmd, " \t") of
                [Name | Args] -> {command, string:lowercase(Name), Args};
                [] -> empty
            end;
        _ ->
            continued(L, Read, [])
    end.

%% A line ending in `\' goes on.
continued(L, Read, Acc) ->
    case lists:suffix("\\", L) of
        true ->
            Part = lists:droplast(L),
            case line(Read(?MORE)) of
                {ok, Next} -> continued(Next, Read, [Part | Acc]);
                eof -> text(lists:reverse([Part | Acc]));
                Other -> Other
            end;
        false ->
            text(lists:reverse([L | Acc]))
    end.

block(Read, Acc) ->
    case line(Read(?MORE)) of
        {ok, L} ->
            case string:trim(L) of
                "\"\"\"" -> text(lists:reverse(Acc));
                _ -> block(Read, [L | Acc])
            end;
        eof ->
            text(lists:reverse(Acc));
        Other ->
            Other
    end.

text(Lines) ->
    case string:trim(lists:join("\n", Lines)) of
        "" -> empty;
        T -> {text, unicode:characters_to_binary(T)}
    end.
