%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_openai_tests).
-include_lib("eunit/include/eunit.hrl").

-define(O, erllama_server_openai_req).

user(C) -> #{<<"role">> => <<"user">>, <<"content">> => C}.

chat(Extra) -> maps:merge(#{<<"model">> => <<"m">>, <<"messages">> => [user(<<"hi">>)]}, Extra).

opts(Extra) ->
    {ok, #{opts := Opts}, _} = ?O:chat_request(chat(Extra)),
    Opts.

maps_sampling_and_length_test() ->
    ?assertEqual(
        #{
            response_tokens => 9,
            temperature => 0.2,
            top_p => 0.9,
            seed => 7,
            stop_sequences => [<<"END">>]
        },
        opts(#{
            <<"max_tokens">> => 9,
            <<"temperature">> => 0.2,
            <<"top_p">> => 0.9,
            <<"seed">> => 7,
            <<"stop">> => <<"END">>
        })
    ),
    ?assertEqual(#{response_tokens => 5}, opts(#{<<"max_completion_tokens">> => 5})).

maps_model_stream_and_keep_alive_test() ->
    {ok, Spec, Meta} = ?O:chat_request(
        chat(#{
            <<"stream">> => true,
            <<"stream_options">> => #{<<"include_usage">> => true},
            <<"keep_alive">> => <<"5m">>
        })
    ),
    ?assertMatch(
        #{
            model := <<"m">>,
            keep_alive := 300000,
            messages := [#{role := <<"user">>, content := <<"hi">>}]
        },
        Spec
    ),
    ?assertMatch(#{stream := true, include_usage := true, logprobs := false}, Meta).

maps_tools_test() ->
    Tool = #{
        <<"type">> => <<"function">>,
        <<"function">> => #{
            <<"name">> => <<"w">>,
            <<"description">> => <<"weather">>,
            <<"parameters">> => #{<<"type">> => <<"object">>}
        }
    },
    Other = #{<<"type">> => <<"function">>, <<"function">> => #{<<"name">> => <<"x">>}},
    ?assertMatch(
        #{
            tools := [#{name := <<"w">>, description := <<"weather">>}, #{name := <<"x">>}],
            tool_choice := auto
        },
        opts(#{<<"tools">> => [Tool, Other]})
    ),
    %% A named function: required, and only that tool is offered.
    ?assertMatch(
        #{tools := [#{name := <<"x">>}], tool_choice := required},
        opts(#{
            <<"tools">> => [Tool, Other],
            <<"tool_choice">> => #{
                <<"type">> => <<"function">>, <<"function">> => #{<<"name">> => <<"x">>}
            }
        })
    ),
    ?assertMatch(
        {error, {bad_request, _, <<"tool_choice">>}},
        ?O:chat_request(
            chat(#{
                <<"tools">> => [Tool],
                <<"tool_choice">> => #{
                    <<"type">> => <<"function">>, <<"function">> => #{<<"name">> => <<"nope">>}
                }
            })
        )
    ).

maps_response_format_test() ->
    ?assertEqual(
        #{json_schema => #{<<"type">> => <<"object">>}},
        opts(#{<<"response_format">> => #{<<"type">> => <<"json_object">>}})
    ),
    S = #{
        <<"type">> => <<"object">>, <<"properties">> => #{<<"a">> => #{<<"type">> => <<"integer">>}}
    },
    ?assertEqual(
        #{json_schema => S},
        opts(#{
            <<"response_format">> => #{
                <<"type">> => <<"json_schema">>,
                <<"json_schema">> => #{<<"name">> => <<"x">>, <<"schema">> => S}
            }
        })
    ),
    ?assertEqual(#{}, opts(#{<<"response_format">> => #{<<"type">> => <<"text">>}})).

maps_messages_test() ->
    Img = base64:encode(<<1, 2, 3>>),
    {ok, Msgs} = ?O:messages([
        #{<<"role">> => <<"system">>, <<"content">> => <<"be brief">>},
        user([
            #{<<"type">> => <<"text">>, <<"text">> => <<"look">>},
            #{
                <<"type">> => <<"image_url">>,
                <<"image_url">> => #{<<"url">> => <<"data:image/png;base64,", Img/binary>>}
            }
        ]),
        #{
            <<"role">> => <<"assistant">>,
            <<"content">> => null,
            <<"tool_calls">> => [
                #{
                    <<"id">> => <<"call_1">>,
                    <<"type">> => <<"function">>,
                    <<"function">> => #{<<"name">> => <<"w">>, <<"arguments">> => <<"{}">>}
                }
            ]
        },
        #{
            <<"role">> => <<"tool">>,
            <<"tool_call_id">> => <<"call_1">>,
            <<"content">> => <<"sunny">>
        }
    ]),
    [Sys, User, Asst, ToolMsg] = Msgs,
    ?assertEqual(#{role => <<"system">>, content => <<"be brief">>}, Sys),
    ?assertMatch(
        #{content := [#{type := text, text := <<"look">>}, #{type := image, data := <<1, 2, 3>>}]},
        User
    ),
    ?assertMatch(
        #{
            content := null,
            tool_calls := [#{id := <<"call_1">>, type := function, function := #{name := <<"w">>}}]
        },
        Asst
    ),
    ?assertEqual(
        #{role => <<"tool">>, tool_call_id => <<"call_1">>, content => <<"sunny">>}, ToolMsg
    ),
    %% They are valid erllama_server_generate messages.
    [?assertMatch({ok, _}, erllama_server_generate:message(M)) || M <- Msgs].

refuses_test() ->
    Bad = fun(Body) -> element(1, ?O:chat_request(Body)) =:= error end,
    ?assert(Bad(chat(#{<<"logit_bias">> => #{}}))),
    ?assert(Bad(chat(#{<<"n">> => 3}))),
    ?assert(Bad(#{<<"model">> => <<"m">>, <<"messages">> => []})),
    ?assert(Bad(chat(#{<<"stop">> => 3}))),
    ?assert(Bad(chat(#{<<"top_logprobs">> => 50, <<"logprobs">> => true}))),
    ?assert(Bad(chat(#{<<"response_format">> => #{<<"type">> => <<"xml">>}}))),
    ?assert(
        Bad(#{
            <<"messages">> => [
                user([
                    #{
                        <<"type">> => <<"image_url">>,
                        <<"image_url">> => #{<<"url">> => <<"http://x/y.png">>}
                    }
                ])
            ]
        })
    ),
    ?assert(Bad(#{<<"messages">> => [user([#{<<"type">> => <<"video">>}])]})),
    ?assertMatch(
        {ok, _, _}, ?O:chat_request(chat(#{<<"user">> => <<"me">>, <<"metadata">> => #{}}))
    ).

completion_and_embedding_requests_test() ->
    ?assertMatch(
        {ok, #{prompt := <<"p">>, opts := #{response_tokens := 3}}, _},
        ?O:completion_request(#{<<"prompt">> => <<"p">>, <<"max_tokens">> => 3})
    ),
    ?assertMatch({error, _}, ?O:completion_request(#{<<"prompt">> => <<"p">>, <<"echo">> => true})),
    ?assertMatch(
        {ok, #{inputs := [<<"a">>, <<"b">>]}, #{format := <<"float">>}},
        ?O:embedding_request(#{<<"input">> => [<<"a">>, <<"b">>]})
    ),
    ?assertMatch(
        {ok, #{inputs := [[1, 2, 3]]}, _}, ?O:embedding_request(#{<<"input">> => [1, 2, 3]})
    ),
    ?assertMatch(
        {error, _}, ?O:embedding_request(#{<<"input">> => <<"a">>, <<"dimensions">> => 8})
    ),
    ?assertMatch(
        {error, _},
        ?O:embedding_request(#{<<"input">> => <<"a">>, <<"encoding_format">> => <<"hex">>})
    ).

durations_test() ->
    ?assertEqual({ok, 30000}, ?O:duration_ms(30)),
    ?assertEqual({ok, 30000}, ?O:duration_ms(<<"30s">>)),
    ?assertEqual({ok, 300000}, ?O:duration_ms(<<"5m">>)),
    ?assertEqual({ok, 3600000}, ?O:duration_ms(<<"1h">>)),
    ?assertEqual({ok, 0}, ?O:duration_ms(0)),
    ?assertEqual({ok, infinity}, ?O:duration_ms(-1)),
    ?assertEqual({ok, infinity}, ?O:duration_ms(<<"-1">>)),
    ?assertEqual(error, ?O:duration_ms(<<"soon">>)).

%% -----------------------------------------------------------------------------
%% API keys
%% -----------------------------------------------------------------------------

auth(Keys, Req) ->
    Cap = livery_test_adapter:run(
        [{erllama_server_http_auth, Keys}], fun(_) -> livery_resp:text(200, <<"ok">>) end, Req
    ),
    livery_test_adapter:status(Cap).

api_keys_test() ->
    Path = #{method => <<"POST">>, path => <<"/v1/chat/completions">>},
    ?assertEqual(200, auth([], Path)),
    ?assertEqual(401, auth([<<"k">>], Path)),
    ?assertEqual(200, auth([<<"k">>], Path#{headers => [{<<"authorization">>, <<"Bearer k">>}]})),
    ?assertEqual(200, auth([<<"k">>], Path#{headers => [{<<"x-api-key">>, <<"k">>}]})),
    ?assertEqual(401, auth([<<"k">>], Path#{headers => [{<<"authorization">>, <<"Bearer kk">>}]})),
    ?assertEqual(200, auth([<<"k">>], #{method => <<"GET">>, path => <<"/health">>})).

public_listener_needs_keys_test() ->
    V = fun erllama_server_config:validate/1,
    ?assertMatch(
        {error, {invalid_config, http, _}}, V(#{http => #{port => 8080, ip => {0, 0, 0, 0}}})
    ),
    ?assertEqual(ok, V(#{http => #{port => 8080, ip => {0, 0, 0, 0}}, api_keys => [<<"k">>]})),
    ?assertEqual(
        ok, V(#{http => #{port => 8080, ip => {0, 0, 0, 0}}, allow_unauthenticated => true})
    ),
    ?assertEqual(ok, V(#{http => #{port => 8080}})),
    ?assertMatch({error, {invalid_config, http, _}}, V(#{http => #{port => 99999}})).
