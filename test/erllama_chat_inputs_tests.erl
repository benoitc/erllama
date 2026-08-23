%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% erllama_chat:inputs/2: the Erlang-term -> NIF inputs boundary.
-module(erllama_chat_inputs_tests).
-include_lib("eunit/include/eunit.hrl").

tools_are_wrapped_in_the_openai_shape_test() ->
    Tool = #{
        name => <<"calc">>,
        description => <<"Arithmetic">>,
        parameters => #{type => object, properties => #{e => #{type => string}}}
    },
    #{tools := Json} = erllama_chat:inputs([], #{tools => [Tool]}),
    [Decoded] = json:decode(Json),
    ?assertEqual(<<"function">>, maps:get(<<"type">>, Decoded)),
    Fun = maps:get(<<"function">>, Decoded),
    ?assertEqual(<<"calc">>, maps:get(<<"name">>, Fun)),
    ?assertEqual(<<"Arithmetic">>, maps:get(<<"description">>, Fun)),
    ?assertEqual(<<"object">>, maps:get(<<"type">>, maps:get(<<"parameters">>, Fun))).

already_wrapped_tools_pass_through_test() ->
    Wrapped = #{type => function, function => #{name => <<"x">>, parameters => #{}}},
    #{tools := Json} = erllama_chat:inputs([], #{tools => [Wrapped]}),
    [Decoded] = json:decode(Json),
    ?assertEqual(<<"x">>, maps:get(<<"name">>, maps:get(<<"function">>, Decoded))).

tool_without_parameters_gets_an_empty_object_schema_test() ->
    #{tools := Json} = erllama_chat:inputs([], #{tools => [#{name => <<"now">>}]}),
    [#{<<"function">> := #{<<"parameters">> := Params}}] = json:decode(Json),
    ?assertEqual(<<"object">>, maps:get(<<"type">>, Params)).

messages_and_defaults_test() ->
    Msgs = [#{role => user, content => <<"hi">>}],
    Inputs = erllama_chat:inputs(Msgs, #{}),
    ?assertEqual(
        [#{<<"role">> => <<"user">>, <<"content">> => <<"hi">>}],
        json:decode(maps:get(messages, Inputs))
    ),
    ?assertEqual(<<"[]">>, maps:get(tools, Inputs)),
    ?assertEqual(auto, maps:get(tool_choice, Inputs)),
    ?assertEqual(false, maps:get(parallel_tool_calls, Inputs)).
