%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Thinking off for a template that always opens its thinking block.
-module(erllama_chat_thinking_tests).
-include_lib("eunit/include/eunit.hrl").

-define(C, erllama_chat).

%% GLM-5.3-Flash's render: the generation prompt opens `<think>' whatever
%% enable_thinking says.
open() ->
    #{
        prompt => <<"<|user|>Hi<|assistant|><think>">>,
        generation_prompt => <<"<|assistant|><think>">>,
        supports_thinking => true,
        thinking_start_tag => <<"<think>">>,
        thinking_end_tags => [<<"</think>">>]
    }.

closes_an_open_block_when_thinking_is_off_test() ->
    {ok, Params, R} = ?C:close_thinking(false, p, open()),
    ?assertEqual({thinking_closed, p, <<"</think>">>}, Params),
    ?assertEqual(<<"<|user|>Hi<|assistant|><think></think>">>, maps:get(prompt, R)),
    ?assertEqual(<<"<|assistant|><think></think>">>, maps:get(generation_prompt, R)).

leaves_thinking_on_alone_test() ->
    ?assertEqual({ok, p, open()}, ?C:close_thinking(true, p, open())).

leaves_a_template_that_closes_it_itself_alone_test() ->
    %% e.g. Qwen3 renders an empty block for enable_thinking false.
    Closed = (open())#{prompt => <<"<|assistant|><think>\n\n</think>\n\n">>},
    ?assertEqual({ok, p, Closed}, ?C:close_thinking(false, p, Closed)).

leaves_a_template_without_thinking_alone_test() ->
    None = #{prompt => <<"<|assistant|>">>, supports_thinking => false},
    ?assertEqual({ok, p, None}, ?C:close_thinking(false, p, None)),
    NoTags = (open())#{thinking_end_tags => []},
    ?assertEqual({ok, p, NoTags}, ?C:close_thinking(false, p, NoTags)).
