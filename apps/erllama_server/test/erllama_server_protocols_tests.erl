%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_protocols_tests).
-include_lib("eunit/include/eunit.hrl").

%% -----------------------------------------------------------------------------
%% Anthropic
%% -----------------------------------------------------------------------------

anthropic_maps_blocks_test() ->
    Body = #{
        <<"model">> => <<"m">>,
        <<"max_tokens">> => 50,
        <<"system">> => [#{<<"type">> => <<"text">>, <<"text">> => <<"be brief">>}],
        <<"stop_sequences">> => [<<"END">>],
        <<"top_k">> => 20,
        <<"thinking">> => #{<<"type">> => <<"enabled">>, <<"budget_tokens">> => 1024},
        <<"tools">> => [
            #{<<"name">> => <<"w">>, <<"input_schema">> => #{<<"type">> => <<"object">>}}
        ],
        <<"tool_choice">> => #{<<"type">> => <<"any">>, <<"disable_parallel_tool_use">> => true},
        <<"messages">> => [
            #{<<"role">> => <<"user">>, <<"content">> => <<"weather?">>},
            #{
                <<"role">> => <<"assistant">>,
                <<"content">> => [
                    #{<<"type">> => <<"text">>, <<"text">> => <<"checking">>},
                    #{
                        <<"type">> => <<"tool_use">>,
                        <<"id">> => <<"tu_1">>,
                        <<"name">> => <<"w">>,
                        <<"input">> => #{<<"city">> => <<"Paris">>}
                    }
                ]
            },
            #{
                <<"role">> => <<"user">>,
                <<"content">> => [
                    #{
                        <<"type">> => <<"tool_result">>,
                        <<"tool_use_id">> => <<"tu_1">>,
                        <<"content">> => <<"sunny">>
                    },
                    #{<<"type">> => <<"text">>, <<"text">> => <<"thanks">>}
                ]
            }
        ]
    },
    {ok, #{model := <<"m">>, messages := Msgs, opts := Opts}, #{stream := false}} = erllama_server_anthropic:request(
        Body
    ),
    ?assertEqual(
        [<<"system">>, <<"user">>, <<"assistant">>, <<"tool">>, <<"user">>],
        [R || #{role := R} <- Msgs]
    ),
    [_, _, Asst, Tool, _] = Msgs,
    ?assertMatch(
        #{
            content := <<"checking">>,
            tool_calls := [#{id := <<"tu_1">>, function := #{name := <<"w">>}}]
        },
        Asst
    ),
    ?assertEqual(#{role => <<"tool">>, tool_call_id => <<"tu_1">>, content => <<"sunny">>}, Tool),
    ?assertMatch(
        #{
            response_tokens := 50,
            stop_sequences := [<<"END">>],
            top_k := 20,
            enable_thinking := true,
            tool_choice := required,
            parallel_tool_calls := false
        },
        Opts
    ),
    [?assertMatch({ok, _}, erllama_server_generate:message(M)) || M <- Msgs].

anthropic_refuses_test() ->
    Base = #{
        <<"model">> => <<"m">>,
        <<"max_tokens">> => 5,
        <<"messages">> => [#{<<"role">> => <<"user">>, <<"content">> => <<"x">>}]
    },
    ?assertMatch({error, _}, erllama_server_anthropic:request(maps:remove(<<"max_tokens">>, Base))),
    ?assertMatch({error, _}, erllama_server_anthropic:request(Base#{<<"top_logprobs">> => 1})),
    Url = #{
        <<"type">> => <<"image">>,
        <<"source">> => #{<<"type">> => <<"url">>, <<"url">> => <<"https://x/y.png">>}
    },
    ?assertMatch(
        {error, _},
        erllama_server_anthropic:request(Base#{
            <<"messages">> => [#{<<"role">> => <<"user">>, <<"content">> => [Url]}]
        })
    ),
    ?assertMatch(
        {ok, _, _},
        erllama_server_anthropic:request(Base#{<<"metadata">> => #{<<"user_id">> => <<"u">>}})
    ).

%% -----------------------------------------------------------------------------
%% Ollama
%% -----------------------------------------------------------------------------

ollama_tool_results_answer_calls_in_order_test() ->
    Body = #{
        <<"model">> => <<"m">>,
        <<"messages">> => [
            #{<<"role">> => <<"user">>, <<"content">> => <<"weather in two cities">>},
            #{
                <<"role">> => <<"assistant">>,
                <<"content">> => <<>>,
                <<"tool_calls">> => [
                    #{
                        <<"function">> => #{
                            <<"name">> => <<"w">>, <<"arguments">> => #{<<"city">> => <<"Paris">>}
                        }
                    },
                    #{
                        <<"function">> => #{
                            <<"name">> => <<"w">>, <<"arguments">> => #{<<"city">> => <<"Rome">>}
                        }
                    }
                ]
            },
            #{<<"role">> => <<"tool">>, <<"content">> => <<"sunny">>},
            #{<<"role">> => <<"tool">>, <<"content">> => <<"rain">>, <<"tool_name">> => <<"w">>}
        ],
        <<"options">> => #{
            <<"num_predict">> => 8,
            <<"stop">> => <<"END">>,
            <<"repeat_penalty">> => 1.1,
            <<"num_ctx">> => 4096
        },
        <<"format">> => <<"json">>,
        <<"think">> => false,
        <<"keep_alive">> => <<"10m">>
    },
    {ok, #{messages := Msgs, opts := Opts, keep_alive := 600000}, #{stream := true}} = erllama_server_ollama:chat_request(
        Body
    ),
    [
        _,
        #{tool_calls := [#{id := A}, #{id := B}]},
        #{tool_call_id := TA},
        #{tool_call_id := TB, name := <<"w">>}
    ] = Msgs,
    ?assertEqual({A, B}, {TA, TB}),
    ?assertEqual(
        #{
            response_tokens => 8,
            stop_sequences => [<<"END">>],
            repetition_penalty => 1.1,
            json_schema => #{<<"type">> => <<"object">>},
            enable_thinking => false
        },
        Opts
    ),
    [?assertMatch({ok, _}, erllama_server_generate:message(M)) || M <- Msgs].

ollama_generate_test() ->
    {ok, #{messages := [#{role := <<"system">>}, #{role := <<"user">>, content := <<"p">>}]}, _} =
        erllama_server_ollama:generate_request(#{
            <<"model">> => <<"m">>, <<"prompt">> => <<"p">>, <<"system">> => <<"s">>
        }),
    {ok, #{prompt := <<"raw">>}, _} = erllama_server_ollama:generate_request(#{
        <<"model">> => <<"m">>, <<"prompt">> => <<"raw">>, <<"raw">> => true
    }),
    ?assertMatch(
        {error, _},
        erllama_server_ollama:generate_request(#{
            <<"model">> => <<"m">>, <<"prompt">> => <<"p">>, <<"suffix">> => <<"x">>
        })
    ),
    ?assertMatch({error, _}, erllama_server_ollama:options(#{<<"num_gpu">> => 2})),
    ?assertEqual({ok, #{}}, erllama_server_ollama:options(#{<<"num_predict">> => -1})).

%% -----------------------------------------------------------------------------
%% Responses
%% -----------------------------------------------------------------------------

responses_items_test() ->
    Body = #{
        <<"model">> => <<"m">>,
        <<"store">> => false,
        <<"instructions">> => <<"be brief">>,
        <<"input">> => [
            #{<<"role">> => <<"developer">>, <<"content">> => <<"house rules">>},
            #{
                <<"type">> => <<"message">>,
                <<"role">> => <<"user">>,
                <<"content">> => [#{<<"type">> => <<"input_text">>, <<"text">> => <<"hi">>}]
            },
            #{
                <<"type">> => <<"function_call">>,
                <<"call_id">> => <<"c1">>,
                <<"name">> => <<"w">>,
                <<"arguments">> => <<"{}">>
            },
            #{
                <<"type">> => <<"function_call">>,
                <<"call_id">> => <<"c2">>,
                <<"name">> => <<"w">>,
                <<"arguments">> => <<"{}">>
            },
            #{
                <<"type">> => <<"function_call_output">>,
                <<"call_id">> => <<"c1">>,
                <<"output">> => <<"sunny">>
            },
            #{<<"type">> => <<"reasoning">>, <<"summary">> => []}
        ],
        <<"text">> => #{<<"format">> => #{<<"type">> => <<"json_object">>}},
        <<"max_output_tokens">> => 30
    },
    {ok, #{messages := Msgs, opts := Opts, conversation_id := none}, #{store := false}} = erllama_server_openai_responses:request(
        Body
    ),
    ?assertEqual([<<"system">>, <<"system">>, <<"user">>, <<"assistant">>, <<"tool">>], [
        R
     || #{role := R} <- Msgs
    ]),
    [_, _, _, #{tool_calls := Calls}, _] = Msgs,
    ?assertEqual([<<"c1">>, <<"c2">>], [Id || #{id := Id} <- Calls]),
    ?assertEqual(#{response_tokens => 30, json_schema => #{<<"type">> => <<"object">>}}, Opts),
    [?assertMatch({ok, _}, erllama_server_generate:message(M)) || M <- Msgs].

responses_refuses_test() ->
    Base = #{<<"model">> => <<"m">>, <<"input">> => <<"x">>},
    ?assertMatch(
        {error, _}, erllama_server_openai_responses:request(Base#{<<"truncation">> => <<"auto">>})
    ),
    ?assertMatch(
        {error, _}, erllama_server_openai_responses:request(Base#{<<"background">> => true})
    ),
    ?assertMatch(
        {error, _},
        erllama_server_openai_responses:request(Base#{
            <<"tools">> => [#{<<"type">> => <<"web_search">>}]
        })
    ),
    ?assertMatch(
        {error, _},
        erllama_server_openai_responses:request(Base#{
            <<"input">> => [#{<<"type">> => <<"computer_call">>}]
        })
    ).
