%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_cli_input_tests).
-include_lib("eunit/include/eunit.hrl").

-define(IN, erllama_server_cli_input).

%% A reader over canned lines; also records the prompts it was shown.
reader(Lines) ->
    Tab = ets:new(lines, [public]),
    true = ets:insert(Tab, {lines, Lines}),
    fun(Prompt) ->
        [{_, L}] = ets:lookup(Tab, lines),
        put(prompts, [lists:flatten(io_lib:format("~ts", [Prompt])) | get_prompts()]),
        case L of
            [] ->
                eof;
            [H | T] ->
                true = ets:insert(Tab, {lines, T}),
                H
        end
    end.

get_prompts() ->
    case get(prompts) of
        undefined -> [];
        P -> P
    end.

read(Lines) ->
    erase(prompts),
    ?IN:read("> ", reader(Lines)).

text_test() ->
    ?assertEqual({text, <<"hello there">>}, read(["hello there\n"])),
    ?assertEqual({text, <<"é"/utf8>>}, read([[233, $\n]])).

continued_test() ->
    ?assertEqual({text, <<"first\nsecond\nthird">>}, read(["first\\\n", "second\\\n", "third\n"])),
    ?assertEqual(["... ", "... ", "> "], get_prompts()).

block_test() ->
    ?assertEqual(
        {text, <<"def f():\n    return 1">>},
        read(["\"\"\"\n", "def f():\n", "    return 1\n", "\"\"\"\n"])
    ),
    %% End of input closes the block.
    ?assertEqual({text, <<"a\nb">>}, read(["\"\"\"\n", "a\n", "b\n"])).

command_test() ->
    ?assertEqual({command, "switch", ["conv_1"]}, read(["/switch conv_1\n"])),
    ?assertEqual({command, "quit", []}, read(["  /QUIT \n"])),
    ?assertEqual(empty, read(["/\n"])).

dropped_and_empty_test() ->
    ?assertEqual(dropped, read(["half typed" ++ [3] ++ "\n"])),
    ?assertEqual(dropped, read(["more\\\n", [3, $\n]])),
    ?assertEqual(empty, read(["   \n"])),
    ?assertEqual(eof, read([])).

crlf_test() ->
    ?assertEqual(<<"a\r\nb\r\n">>, erllama_server_cli_term:crlf(["a\n", <<"b\n">>])).
