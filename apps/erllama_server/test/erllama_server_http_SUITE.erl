%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% The HTTP APIs over a real socket: OpenAI response shapes, streamed
%% and not; strict fields; errors; API keys; body and concurrency
%% limits; a client that disconnects mid-stream cancelling its run;
%% load on demand with keep-alive; the ops endpoints. Stub models, so
%% the suite runs anywhere; the real-model cases are gated on
%% LLAMA_TEST_MODEL.
%% @end
-module(erllama_server_http_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1, init_per_testcase/2, end_per_testcase/2]).
-export([
    lists_models/1,
    chat_completion/1,
    chat_completion_stream/1,
    text_completion/1,
    text_completion_stream/1,
    embeddings_float_and_base64/1,
    strict_fields/1,
    unknown_model_is_404/1,
    api_keys/1,
    body_limit/1,
    concurrency_limit/1,
    run_limit_is_429/1,
    disconnect_cancels_the_run/1,
    load_on_demand_and_keep_alive/1,
    unload_during_stream/1,
    concurrent_requests_on_two_models/1,
    ops_endpoints/1,
    real_model_tool_call/1,
    real_model_logprobs/1,
    image_needs_a_projector/1,
    real_model_image_chat/1,
    anthropic_message/1,
    anthropic_stream/1,
    anthropic_errors_and_count/1,
    responses_chain/1,
    responses_branch/1,
    responses_stream/1,
    ollama_chat_and_generate/1,
    ollama_models/1,
    ollama_pull/1,
    real_model_anthropic_tools/1,
    real_model_responses_tools/1
]).

all() ->
    [
        lists_models,
        chat_completion,
        chat_completion_stream,
        text_completion,
        text_completion_stream,
        embeddings_float_and_base64,
        strict_fields,
        unknown_model_is_404,
        api_keys,
        body_limit,
        concurrency_limit,
        run_limit_is_429,
        disconnect_cancels_the_run,
        load_on_demand_and_keep_alive,
        unload_during_stream,
        concurrent_requests_on_two_models,
        ops_endpoints,
        real_model_tool_call,
        real_model_logprobs,
        image_needs_a_projector,
        real_model_image_chat,
        anthropic_message,
        anthropic_stream,
        anthropic_errors_and_count,
        responses_chain,
        responses_branch,
        responses_stream,
        ollama_chat_and_generate,
        ollama_models,
        ollama_pull,
        real_model_anthropic_tools,
        real_model_responses_tools
    ].

-define(ENV, [data_dir, http, api_keys, max_body_bytes, max_http_requests, max_active_runs]).

init_per_suite(Config) ->
    {ok, _} = application:ensure_all_started(livery),
    Config.

end_per_suite(_Config) ->
    ok.

init_per_testcase(TC, Config) ->
    Dir = filename:join(?config(priv_dir, Config), atom_to_list(TC)),
    ok = filelib:ensure_path(Dir),
    ok = application:set_env(erllama_server, data_dir, Dir),
    ok = application:set_env(erllama_server, http, #{port => 0}),
    [ok = application:set_env(erllama_server, K, V) || {K, V} <- env(TC)],
    {ok, _} = application:ensure_all_started(erllama_server),
    {ok, _} = erllama:load_model(<<"stub">>, #{backend => erllama_model_stub, step_delay_ms => 1}),
    {ok, _} = erllama:load_model(<<"slow">>, #{backend => erllama_model_stub, step_delay_ms => 20}),
    [
        {base, <<"http://127.0.0.1:", (integer_to_binary(erllama_server_http:port()))/binary>>},
        {dir, Dir}
        | Config
    ].

end_per_testcase(_TC, _Config) ->
    _ = application:stop(erllama_server),
    [erllama:unload(Id) || #{id := Id} <- erllama:list_models()],
    [application:unset_env(erllama_server, K) || K <- ?ENV],
    ok.

env(api_keys) -> [{api_keys, [<<"k1">>, <<"k2">>]}];
env(body_limit) -> [{max_body_bytes, 256}];
env(concurrency_limit) -> [{max_http_requests, 1}];
env(run_limit_is_429) -> [{max_active_runs, 1}];
env(_) -> [].

%% -----------------------------------------------------------------------------
%% Client helpers
%% -----------------------------------------------------------------------------

client() -> livery_client:new(#{}).

post(Config, Path, Body) ->
    post(Config, Path, Body, []).

post(Config, Path, Body, Headers) ->
    Url = <<(?config(base, Config))/binary, Path/binary>>,
    Bin =
        case Body of
            B when is_binary(B) -> B;
            M -> iolist_to_binary(json:encode(M))
        end,
    {ok, Resp} = livery_client:request(client(), post, Url, #{
        body => Bin,
        headers => [{<<"content-type">>, <<"application/json">>} | Headers],
        timeout => 30000
    }),
    {full, RespBody} = livery_client:body(Resp),
    {livery_client:status(Resp), RespBody}.

get(Config, Path) ->
    get(Config, Path, []).

get(Config, Path, Headers) ->
    {ok, Resp} = livery_client:request(
        client(), get, <<(?config(base, Config))/binary, Path/binary>>, #{headers => Headers}
    ),
    {full, Body} = livery_client:body(Resp),
    {livery_client:status(Resp), Body}.

decode(Bin) -> json:decode(Bin).

%% The `data:' payloads of a server-sent events body.
sse(Body) ->
    [D || Frame <- binary:split(Body, <<"\n\n">>, [global]), <<"data: ", D/binary>> <- [Frame]].

chunks(Body) ->
    [decode(D) || D <- sse(Body), D =/= <<"[DONE]">>].

chat(Content) -> chat(Content, #{}).

chat(Content, Extra) ->
    maps:merge(
        #{
            <<"model">> => <<"stub">>,
            <<"messages">> => [#{<<"role">> => <<"user">>, <<"content">> => Content}],
            <<"max_tokens">> => 4
        },
        Extra
    ).

wait(Fun) -> wait(Fun, 250).
wait(_Fun, 0) ->
    ct:fail(condition_not_reached);
wait(Fun, N) ->
    case Fun() of
        true ->
            ok;
        false ->
            timer:sleep(20),
            wait(Fun, N - 1)
    end.

%% -----------------------------------------------------------------------------
%% Cases
%% -----------------------------------------------------------------------------

lists_models(Config) ->
    {200, Body} = get(Config, <<"/v1/models">>),
    #{<<"object">> := <<"list">>, <<"data">> := Data} = decode(Body),
    ?assertEqual(
        [<<"slow">>, <<"stub">>],
        lists:sort([Id || #{<<"id">> := Id, <<"object">> := <<"model">>} <- Data])
    ),
    {200, One} = get(Config, <<"/v1/models/stub">>),
    ?assertMatch(#{<<"id">> := <<"stub">>}, decode(One)),
    {404, _} = get(Config, <<"/v1/models/none">>).

chat_completion(Config) ->
    {200, Body} = post(Config, <<"/v1/chat/completions">>, chat(<<"hello">>)),
    #{
        <<"id">> := <<"chatcmpl-", _/binary>>,
        <<"object">> := <<"chat.completion">>,
        <<"model">> := <<"stub">>,
        <<"choices">> := [
            #{<<"index">> := 0, <<"message">> := Msg, <<"finish_reason">> := <<"length">>}
        ],
        <<"usage">> := #{
            <<"completion_tokens">> := 4, <<"prompt_tokens">> := P, <<"total_tokens">> := T
        }
    } = decode(Body),
    ?assertMatch(#{<<"role">> := <<"assistant">>, <<"content">> := C} when byte_size(C) > 0, Msg),
    ?assertEqual(P + 4, T).

chat_completion_stream(Config) ->
    {200, Body} = post(
        Config,
        <<"/v1/chat/completions">>,
        chat(<<"hello">>, #{
            <<"stream">> => true, <<"stream_options">> => #{<<"include_usage">> => true}
        })
    ),
    ?assertEqual(<<"[DONE]">>, lists:last(sse(Body))),
    [First | Rest] = chunks(Body),
    ?assertMatch(
        #{
            <<"object">> := <<"chat.completion.chunk">>,
            <<"choices">> := [#{<<"delta">> := #{<<"role">> := <<"assistant">>}}]
        },
        First
    ),
    Usage = lists:last(Rest),
    ?assertMatch(#{<<"choices">> := [], <<"usage">> := #{<<"completion_tokens">> := 4}}, Usage),
    Finish = lists:nth(length(Rest) - 1, Rest),
    ?assertMatch(#{<<"choices">> := [#{<<"finish_reason">> := <<"length">>}]}, Finish),
    Text = iolist_to_binary([
        C
     || #{<<"choices">> := [#{<<"delta">> := #{<<"content">> := C}}]} <- Rest
    ]),
    ?assert(byte_size(Text) > 0),
    %% One id across the stream.
    ?assertEqual(1, length(lists:usort([Id || #{<<"id">> := Id} <- [First | Rest]]))).

text_completion(Config) ->
    {200, Body} = post(Config, <<"/v1/completions">>, #{
        <<"model">> => <<"stub">>, <<"prompt">> => <<"once upon">>, <<"max_tokens">> => 3
    }),
    #{
        <<"object">> := <<"text_completion">>,
        <<"choices">> := [#{<<"text">> := T, <<"finish_reason">> := <<"length">>}]
    } =
        decode(Body),
    ?assert(byte_size(T) > 0).

text_completion_stream(Config) ->
    {200, Body} = post(Config, <<"/v1/completions">>, #{
        <<"model">> => <<"stub">>,
        <<"prompt">> => <<"once upon">>,
        <<"max_tokens">> => 3,
        <<"stream">> => true
    }),
    Chunks = chunks(Body),
    ?assert(lists:all(fun(#{<<"object">> := O}) -> O =:= <<"text_completion">> end, Chunks)),
    ?assertMatch(#{<<"choices">> := [#{<<"finish_reason">> := <<"length">>}]}, lists:last(Chunks)).

embeddings_float_and_base64(Config) ->
    {200, F} = post(Config, <<"/v1/embeddings">>, #{
        <<"model">> => <<"stub">>, <<"input">> => [<<"a b">>, <<"c">>]
    }),
    #{
        <<"data">> := [#{<<"index">> := 0, <<"embedding">> := V0}, #{<<"index">> := 1}],
        <<"usage">> := #{<<"prompt_tokens">> := N}
    } =
        decode(F),
    ?assert(N > 0),
    {200, B} = post(Config, <<"/v1/embeddings">>, #{
        <<"model">> => <<"stub">>, <<"input">> => <<"a b">>, <<"encoding_format">> => <<"base64">>
    }),
    #{<<"data">> := [#{<<"embedding">> := B64}]} = decode(B),
    Floats = [X || <<X:32/float-little>> <= base64:decode(B64)],
    ?assertEqual(length(V0), length(Floats)),
    [?assert(abs(X - Y) < 1.0e-6) || {X, Y} <- lists:zip(V0, Floats)].

strict_fields(Config) ->
    Expect = fun(Body, Param) ->
        {400, Resp} = post(Config, <<"/v1/chat/completions">>, Body),
        #{<<"error">> := #{<<"type">> := <<"invalid_request_error">>, <<"param">> := P}} = decode(
            Resp
        ),
        ?assertEqual(Param, P)
    end,
    Expect(chat(<<"x">>, #{<<"logit_bias">> => #{}}), <<"logit_bias">>),
    Expect(chat(<<"x">>, #{<<"n">> => 2}), <<"n">>),
    Expect(
        #{
            <<"model">> => <<"stub">>,
            <<"messages">> => [
                #{
                    <<"role">> => <<"user">>,
                    <<"content">> => [
                        #{
                            <<"type">> => <<"image_url">>,
                            <<"image_url">> => #{<<"url">> => <<"https://x/y.png">>}
                        }
                    ]
                }
            ]
        },
        <<"messages">>
    ),
    Expect(chat(<<"x">>, #{<<"tools">> => [#{<<"type">> => <<"code_interpreter">>}]}), <<"tools">>),
    %% Informational fields pass.
    {200, _} = post(Config, <<"/v1/chat/completions">>, chat(<<"x">>, #{<<"user">> => <<"u1">>})),
    {400, _} = post(Config, <<"/v1/chat/completions">>, <<"not json">>),
    {400, _} = post(Config, <<"/v1/chat/completions">>, <<"[1,2]">>).

unknown_model_is_404(Config) ->
    {404, Body} = post(
        Config, <<"/v1/chat/completions">>, chat(<<"x">>, #{<<"model">> => <<"nope">>})
    ),
    ?assertMatch(
        #{<<"error">> := #{<<"type">> := <<"not_found_error">>, <<"param">> := <<"model">>}},
        decode(Body)
    ).

api_keys(Config) ->
    {401, Denied} = post(Config, <<"/v1/chat/completions">>, chat(<<"x">>)),
    ?assertMatch(#{<<"error">> := #{<<"type">> := <<"authentication_error">>}}, decode(Denied)),
    {401, _} = post(Config, <<"/v1/chat/completions">>, chat(<<"x">>), [
        {<<"authorization">>, <<"Bearer wrong">>}
    ]),
    {200, _} = post(Config, <<"/v1/chat/completions">>, chat(<<"x">>), [
        {<<"authorization">>, <<"Bearer k2">>}
    ]),
    {200, _} = post(Config, <<"/v1/chat/completions">>, chat(<<"x">>), [{<<"x-api-key">>, <<"k1">>}]),
    %% Health probes need no key.
    {200, _} = get(Config, <<"/health">>),
    {200, _} = get(Config, <<"/ready">>),
    {401, _} = get(Config, <<"/metrics">>).

body_limit(Config) ->
    Big = binary:copy(<<"word ">>, 200),
    {413, _} = post(Config, <<"/v1/chat/completions">>, chat(Big)).

concurrency_limit(Config) ->
    Self = self(),
    _ = spawn(fun() ->
        Self !
            {first,
                post(
                    Config,
                    <<"/v1/chat/completions">>,
                    chat(<<"x">>, #{<<"model">> => <<"slow">>, <<"max_tokens">> => 40})
                )}
    end),
    ok = wait(fun() -> erllama:status(<<"slow">>) =:= {ok, generating} end),
    {429, Body} = post(Config, <<"/v1/chat/completions">>, chat(<<"x">>)),
    ?assertMatch(#{<<"error">> := #{<<"type">> := <<"rate_limit_error">>}}, decode(Body)),
    receive
        {first, {200, _}} -> ok
    after 30000 -> ct:fail(first_request_did_not_finish)
    end.

run_limit_is_429(Config) ->
    Self = self(),
    _ = spawn(fun() ->
        Self !
            {first,
                post(
                    Config,
                    <<"/v1/chat/completions">>,
                    chat(<<"x">>, #{<<"model">> => <<"slow">>, <<"max_tokens">> => 40})
                )}
    end),
    ok = wait(fun() -> erllama:status(<<"slow">>) =:= {ok, generating} end),
    {429, _} = post(Config, <<"/v1/chat/completions">>, chat(<<"x">>)),
    receive
        {first, {200, _}} -> ok
    after 30000 -> ct:fail(first_request_did_not_finish)
    end.

%% A client that goes away mid-stream: its run is cancelled and the
%% model is free again at once (500 tokens at 20 ms would take 10 s).
disconnect_cancels_the_run(_Config) ->
    Port = erllama_server_http:port(),
    {ok, S} = gen_tcp:connect({127, 0, 0, 1}, Port, [binary, {active, false}]),
    Body = iolist_to_binary(
        json:encode(
            chat(<<"x">>, #{
                <<"model">> => <<"slow">>, <<"max_tokens">> => 500, <<"stream">> => true
            })
        )
    ),
    Req = [
        <<"POST /v1/chat/completions HTTP/1.1\r\nhost: 127.0.0.1\r\ncontent-type: application/json\r\ncontent-length: ">>,
        integer_to_binary(byte_size(Body)),
        <<"\r\n\r\n">>,
        Body
    ],
    ok = gen_tcp:send(S, Req),
    {ok, First} = gen_tcp:recv(S, 0, 10000),
    ?assertMatch(<<"HTTP/1.1 200", _/binary>>, First),
    ok = wait(fun() -> erllama:status(<<"slow">>) =:= {ok, generating} end),
    ok = gen_tcp:close(S),
    ok = wait(fun() -> erllama:requests() =:= [] end),
    ok = wait(fun() -> erllama:status(<<"slow">>) =:= {ok, idle} end),
    ok = wait(fun() -> [R || #{status := cancelled} = R <- erllama_server_runs:list()] =/= [] end).

load_on_demand_and_keep_alive(Config) ->
    Path = filename:join(?config(dir, Config), "lazy.gguf"),
    ok = file:write_file(Path, erllama_server_fake_registry:gguf(4096)),
    {ok, #{name := Id}} = erllama_server_api:register_model(Path, #{
        name => <<"lazy">>, backend => erllama_model_stub
    }),
    ?assertEqual({error, not_loaded}, erllama:whereis(Id)),
    {200, Body} = post(
        Config,
        <<"/v1/chat/completions">>,
        chat(<<"x">>, #{<<"model">> => <<"lazy">>, <<"keep_alive">> => 0})
    ),
    ?assertMatch(#{<<"model">> := <<"lazy">>}, decode(Body)),
    ok = wait(fun() -> erllama:whereis(Id) =:= {error, not_loaded} end).

unload_during_stream(Config) ->
    Self = self(),
    _ = spawn(fun() ->
        Self !
            {stream,
                post(
                    Config,
                    <<"/v1/chat/completions">>,
                    chat(<<"x">>, #{
                        <<"model">> => <<"slow">>, <<"max_tokens">> => 500, <<"stream">> => true
                    })
                )}
    end),
    ok = wait(fun() -> erllama:status(<<"slow">>) =:= {ok, generating} end),
    ok = erllama:unload(<<"slow">>),
    receive
        {stream, {200, Body}} ->
            ?assertEqual(<<"[DONE]">>, lists:last(sse(Body))),
            ?assertMatch(#{<<"error">> := _}, lists:last(chunks(Body)))
    after 30000 -> ct:fail(stream_did_not_end)
    end.

concurrent_requests_on_two_models(Config) ->
    Self = self(),
    Models = [<<"stub">>, <<"slow">>, <<"stub">>, <<"slow">>, <<"stub">>, <<"slow">>],
    [
        spawn(fun() ->
            Self !
                {done, M,
                    post(Config, <<"/v1/chat/completions">>, chat(<<"x">>, #{<<"model">> => M}))}
        end)
     || M <- Models
    ],
    Results = [
        receive
            {done, _, R} -> R
        after 30000 -> timeout
        end
     || _ <- Models
    ],
    ?assert(lists:all(fun({S, _}) -> S =:= 200 end, Results)).

ops_endpoints(Config) ->
    {200, H} = get(Config, <<"/health">>),
    ?assertMatch(#{<<"status">> := <<"ok">>}, decode(H)),
    {200, R} = get(Config, <<"/ready">>),
    ?assertMatch(#{<<"status">> := <<"ok">>}, decode(R)),
    {200, _} = post(Config, <<"/v1/chat/completions">>, chat(<<"x">>)),
    {200, M} = get(Config, <<"/metrics">>),
    ?assertNotEqual(nomatch, binary:match(M, <<"erllama_server_models_loaded 2">>)),
    ?assertNotEqual(nomatch, binary:match(M, <<"http_server_request_duration">>)).

%% -----------------------------------------------------------------------------
%% Gated: a real model
%% -----------------------------------------------------------------------------

with_real_model(Name, Fun) ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {skip, "set LLAMA_TEST_MODEL to a GGUF path to run the real-model case"};
        Path ->
            {ok, Id} = erllama_server_api:load_model(Path, #{name => Name, n_ctx => 4096}),
            try
                Fun(Id)
            after
                erllama_server_api:unload(Id)
            end
    end.

weather_tool() ->
    #{
        <<"type">> => <<"function">>,
        <<"function">> => #{
            <<"name">> => <<"get_weather">>,
            <<"description">> => <<"Current weather for a city">>,
            <<"parameters">> => #{
                <<"type">> => <<"object">>,
                <<"properties">> => #{<<"city">> => #{<<"type">> => <<"string">>}},
                <<"required">> => [<<"city">>]
            }
        }
    }.

%% A tool-capable model (Qwen2.5, Llama 3.1, ...): the call comes back
%% as tool_calls, streamed as announced calls plus their arguments, and
%% no call syntax leaks into the content.
real_model_tool_call(Config) ->
    with_real_model(<<"real_tools">>, fun(Id) ->
        Req = chat(<<"What is the weather in Paris? Use the tool.">>, #{
            <<"model">> => Id,
            <<"max_tokens">> => 128,
            <<"tools">> => [weather_tool()],
            <<"tool_choice">> => <<"required">>
        }),
        {200, Body} = post(Config, <<"/v1/chat/completions">>, Req),
        case decode(Body) of
            #{
                <<"choices">> := [
                    #{
                        <<"message">> := #{<<"tool_calls">> := [Call | _]},
                        <<"finish_reason">> := <<"tool_calls">>
                    }
                ]
            } ->
                #{
                    <<"id">> := _,
                    <<"function">> := #{<<"name">> := <<"get_weather">>, <<"arguments">> := Args}
                } = Call,
                ?assertMatch(#{<<"city">> := _}, decode(Args)),
                {200, SBody} = post(Config, <<"/v1/chat/completions">>, Req#{<<"stream">> => true}),
                Deltas = [D || #{<<"choices">> := [#{<<"delta">> := D}]} <- chunks(SBody)],
                Names = [
                    N
                 || #{<<"tool_calls">> := [#{<<"function">> := #{<<"name">> := N}}]} <- Deltas
                ],
                ?assertEqual([<<"get_weather">>], Names),
                Content = iolist_to_binary([C || #{<<"content">> := C} <- Deltas, is_binary(C)]),
                ?assertEqual(nomatch, binary:match(Content, <<"get_weather">>));
            Other ->
                ct:log("no tool call: ~p", [Other]),
                {skip, "the model's template synthesizes no tool grammar"}
        end
    end).

real_model_logprobs(Config) ->
    with_real_model(<<"real_logprobs">>, fun(Id) ->
        Req = chat(<<"Say hi.">>, #{
            <<"model">> => Id,
            <<"max_tokens">> => 6,
            <<"logprobs">> => true,
            <<"top_logprobs">> => 2
        }),
        {200, Body} = post(Config, <<"/v1/chat/completions">>, Req),
        #{<<"choices">> := [#{<<"logprobs">> := #{<<"content">> := Entries}}]} = decode(Body),
        ?assert(length(Entries) > 0),
        [#{<<"token">> := T, <<"logprob">> := L, <<"top_logprobs">> := Top} | _] = Entries,
        ?assert(is_binary(T) andalso is_number(L)),
        ?assertEqual(2, length(Top))
    end).

%% A 32x32 BMP gradient (the library's vision suite uses the same).
test_image() ->
    W = 32,
    H = 32,
    Row = fun(Y) ->
        list_to_binary(lists:duplicate(W, <<(min(255, Y * 8)):8, 40:8, (max(0, 255 - Y * 8)):8>>))
    end,
    Pixels = <<<<(Row(Y))/binary>> || Y <- lists:seq(0, H - 1)>>,
    Size = byte_size(Pixels),
    Header = <<"BM", (54 + Size):32/little, 0:32, 54:32/little>>,
    Dib =
        <<40:32/little, W:32/little-signed, H:32/little-signed, 1:16/little, 24:16/little, 0:32,
            Size:32/little, 2835:32/little, 2835:32/little, 0:32, 0:32>>,
    <<Header/binary, Dib/binary, Pixels/binary>>.

image_chat(Model) ->
    Url = <<"data:image/bmp;base64,", (base64:encode(test_image()))/binary>>,
    #{
        <<"model">> => Model,
        <<"max_tokens">> => 24,
        <<"messages">> => [
            #{
                <<"role">> => <<"user">>,
                <<"content">> => [
                    #{
                        <<"type">> => <<"text">>,
                        <<"text">> => <<"Describe the colours of this image in one sentence.">>
                    },
                    #{<<"type">> => <<"image_url">>, <<"image_url">> => #{<<"url">> => Url}}
                ]
            }
        ]
    }.

image_needs_a_projector(Config) ->
    Path = filename:join(?config(dir, Config), "blind.gguf"),
    ok = file:write_file(Path, erllama_server_fake_registry:gguf(4096)),
    {ok, _} = erllama_server_api:register_model(Path, #{
        name => <<"blind">>, backend => erllama_model_stub
    }),
    {400, Body} = post(Config, <<"/v1/chat/completions">>, image_chat(<<"blind">>)),
    #{<<"error">> := #{<<"message">> := Msg}} = decode(Body),
    ?assertNotEqual(nomatch, binary:match(Msg, <<"no multimodal projector">>)).

real_model_image_chat(Config) ->
    case {os:getenv("LLAMA_TEST_VISION_MODEL"), os:getenv("LLAMA_TEST_MMPROJ")} of
        {M, P} when is_list(M), is_list(P) ->
            {ok, Id} = erllama_server_api:load_model(M, #{
                name => <<"vision">>, mmproj => list_to_binary(P), n_ctx => 4096
            }),
            try
                {200, Body} = post(Config, <<"/v1/chat/completions">>, image_chat(Id)),
                #{<<"choices">> := [#{<<"message">> := #{<<"content">> := Text}}]} = decode(Body),
                ct:log("image reply: ~ts", [Text]),
                ?assert(byte_size(Text) > 0),
                {200, SBody} = post(Config, <<"/v1/chat/completions">>, (image_chat(Id))#{
                    <<"stream">> => true
                }),
                ?assertEqual(<<"[DONE]">>, lists:last(sse(SBody)))
            after
                erllama_server_api:unload(Id)
            end;
        _ ->
            {skip, "set LLAMA_TEST_VISION_MODEL and LLAMA_TEST_MMPROJ to run the image case"}
    end.

%% -----------------------------------------------------------------------------
%% Anthropic Messages
%% -----------------------------------------------------------------------------

anthropic(Extra) ->
    maps:merge(
        #{
            <<"model">> => <<"stub">>,
            <<"max_tokens">> => 4,
            <<"messages">> => [#{<<"role">> => <<"user">>, <<"content">> => <<"hi">>}]
        },
        Extra
    ).

%% Named SSE events: [{Event, Data}].
events(Body) ->
    [
        {E, decode(D)}
     || Frame <- binary:split(Body, <<"\n\n">>, [global]),
        [<<"event: ", E/binary>>, <<"data: ", D/binary>>] <- [binary:split(Frame, <<"\n">>)]
    ].

anthropic_message(Config) ->
    {200, Body} = post(Config, <<"/v1/messages">>, anthropic(#{<<"system">> => <<"be brief">>})),
    #{
        <<"id">> := <<"msg_", _/binary>>,
        <<"type">> := <<"message">>,
        <<"role">> := <<"assistant">>,
        <<"content">> := [#{<<"type">> := <<"text">>, <<"text">> := T}],
        <<"stop_reason">> := <<"max_tokens">>,
        <<"usage">> := #{<<"input_tokens">> := In, <<"output_tokens">> := 4}
    } = decode(Body),
    ?assert(byte_size(T) > 0 andalso In > 0).

anthropic_stream(Config) ->
    {200, Body} = post(Config, <<"/v1/messages">>, anthropic(#{<<"stream">> => true})),
    Names = [E || {E, _} <- events(Body)],
    ?assertMatch([<<"message_start">>, <<"content_block_start">> | _], Names),
    ?assertEqual(
        [<<"content_block_stop">>, <<"message_delta">>, <<"message_stop">>],
        lists:nthtail(length(Names) - 3, Names)
    ),
    [{_, #{<<"delta">> := #{<<"stop_reason">> := <<"max_tokens">>}}}] = [
        X
     || {<<"message_delta">>, _} = X <- events(Body)
    ],
    ?assertEqual(4, length([D || {<<"content_block_delta">>, D} <- events(Body)])).

anthropic_errors_and_count(Config) ->
    {400, Body} = post(Config, <<"/v1/messages">>, maps:remove(<<"max_tokens">>, anthropic(#{}))),
    ?assertMatch(
        #{<<"type">> := <<"error">>, <<"error">> := #{<<"type">> := <<"invalid_request_error">>}},
        decode(Body)
    ),
    {404, NotFound} = post(Config, <<"/v1/messages">>, anthropic(#{<<"model">> => <<"nope">>})),
    ?assertMatch(#{<<"error">> := #{<<"type">> := <<"not_found_error">>}}, decode(NotFound)),
    {200, Count} = post(
        Config, <<"/v1/messages/count_tokens">>, maps:remove(<<"max_tokens">>, anthropic(#{}))
    ),
    ?assertMatch(#{<<"input_tokens">> := N} when N > 0, decode(Count)).

%% -----------------------------------------------------------------------------
%% OpenAI Responses
%% -----------------------------------------------------------------------------

responses_chain(Config) ->
    {200, B1} = post(Config, <<"/v1/responses">>, #{
        <<"model">> => <<"stub">>, <<"input">> => <<"hi">>, <<"max_output_tokens">> => 3
    }),
    #{
        <<"id">> := Id1,
        <<"object">> := <<"response">>,
        <<"output">> := [#{<<"type">> := <<"message">>}],
        <<"usage">> := #{<<"input_tokens">> := In1}
    } =
        decode(B1),
    {200, B2} = post(Config, <<"/v1/responses">>, #{
        <<"model">> => <<"stub">>,
        <<"input">> => <<"again">>,
        <<"previous_response_id">> => Id1,
        <<"max_output_tokens">> => 3
    }),
    #{<<"previous_response_id">> := Id1, <<"usage">> := #{<<"input_tokens">> := In2}} = decode(B2),
    %% The second turn is prompted with the first.
    ?assert(In2 > In1),
    {200, Got} = get(Config, <<"/v1/responses/", Id1/binary>>),
    ?assertMatch(#{<<"id">> := Id1}, decode(Got)),
    {ok, Resp} = livery_client:request(
        client(), delete, <<(?config(base, Config))/binary, "/v1/responses/", Id1/binary>>, #{}
    ),
    ?assertEqual(200, livery_client:status(Resp)),
    {404, _} = get(Config, <<"/v1/responses/", Id1/binary>>),
    {400, _} = post(Config, <<"/v1/responses">>, #{
        <<"model">> => <<"stub">>,
        <<"input">> => <<"x">>,
        <<"previous_response_id">> => <<"resp_none">>
    }),
    %% Not stored: nothing to read back.
    {200, B3} = post(Config, <<"/v1/responses">>, #{
        <<"model">> => <<"stub">>,
        <<"input">> => <<"x">>,
        <<"store">> => false,
        <<"max_output_tokens">> => 2
    }),
    #{<<"id">> := Id3} = decode(B3),
    {404, _} = get(Config, <<"/v1/responses/", Id3/binary>>).

%% Two requests continuing the same response are separate branches.
responses_branch(Config) ->
    Post = fun(Input, Prev) ->
        Base = #{<<"model">> => <<"stub">>, <<"input">> => Input, <<"max_output_tokens">> => 2},
        Body =
            case Prev of
                undefined -> Base;
                _ -> Base#{<<"previous_response_id">> => Prev}
            end,
        {200, B} = post(Config, <<"/v1/responses">>, Body),
        #{<<"id">> := Id} = decode(B),
        Id
    end,
    UserInputs = fun(RespId) ->
        <<"resp_", Hex/binary>> = RespId,
        {ok, #{conversation_id := Conv}} = erllama_server_api:run(<<"run_", Hex/binary>>),
        {ok, #{messages := Msgs}} = erllama_server_api:conversation(Conv),
        [C || #{role := <<"user">>, content := C} <- Msgs]
    end,
    R = Post(<<"FIRST">>, undefined),
    A = Post(<<"BRANCH_A">>, R),
    B = Post(<<"BRANCH_B">>, R),
    ?assertEqual([<<"FIRST">>, <<"BRANCH_A">>], UserInputs(A)),
    ?assertEqual([<<"FIRST">>, <<"BRANCH_B">>], UserInputs(B)),
    %% Continuing a branch again extends that branch only.
    A2 = Post(<<"A_AGAIN">>, A),
    ?assertEqual([<<"FIRST">>, <<"BRANCH_A">>, <<"A_AGAIN">>], UserInputs(A2)),
    ?assertEqual([<<"FIRST">>, <<"BRANCH_B">>], UserInputs(B)).

responses_stream(Config) ->
    {200, Body} = post(Config, <<"/v1/responses">>, #{
        <<"model">> => <<"stub">>,
        <<"input">> => <<"hi">>,
        <<"max_output_tokens">> => 2,
        <<"stream">> => true
    }),
    Evs = events(Body),
    ?assertEqual(
        [
            <<"response.created">>,
            <<"response.in_progress">>,
            <<"response.output_item.added">>,
            <<"response.content_part.added">>,
            <<"response.output_text.delta">>,
            <<"response.output_text.delta">>,
            <<"response.output_text.done">>,
            <<"response.content_part.done">>,
            <<"response.output_item.done">>,
            <<"response.completed">>
        ],
        [E || {E, _} <- Evs]
    ),
    ?assertEqual(lists:seq(0, length(Evs) - 1), [N || {_, #{<<"sequence_number">> := N}} <- Evs]).

%% -----------------------------------------------------------------------------
%% Ollama
%% -----------------------------------------------------------------------------

ndjson(Body) -> [decode(L) || L <- binary:split(Body, <<"\n">>, [global, trim_all])].

ollama_chat_and_generate(Config) ->
    Chat = #{
        <<"model">> => <<"stub">>,
        <<"messages">> => [#{<<"role">> => <<"user">>, <<"content">> => <<"hi">>}],
        <<"options">> => #{<<"num_predict">> => 3}
    },
    {200, Stream} = post(Config, <<"/api/chat">>, Chat),
    Lines = ndjson(Stream),
    ?assertEqual(4, length(Lines)),
    ?assertMatch(
        #{<<"done">> := true, <<"done_reason">> := <<"length">>, <<"eval_count">> := 3},
        lists:last(Lines)
    ),
    {200, Whole} = post(Config, <<"/api/chat">>, Chat#{<<"stream">> => false}),
    ?assertMatch(
        #{<<"done">> := true, <<"message">> := #{<<"role">> := <<"assistant">>, <<"content">> := C}} when
            byte_size(C) > 0,
        decode(Whole)
    ),
    {200, Gen} = post(Config, <<"/api/generate">>, #{
        <<"model">> => <<"stub">>,
        <<"prompt">> => <<"once">>,
        <<"stream">> => false,
        <<"options">> => #{<<"num_predict">> => 2}
    }),
    ?assertMatch(#{<<"done">> := true, <<"response">> := R} when byte_size(R) > 0, decode(Gen)),
    {400, Bad} = post(Config, <<"/api/chat">>, Chat#{<<"options">> => #{<<"num_gpu">> => 1}}),
    ?assertMatch(#{<<"error">> := _}, decode(Bad)),
    {200, Emb} = post(Config, <<"/api/embed">>, #{
        <<"model">> => <<"stub">>, <<"input">> => [<<"a">>, <<"b">>]
    }),
    ?assertMatch(#{<<"embeddings">> := [_, _]}, decode(Emb)).

ollama_models(Config) ->
    Path = filename:join(?config(dir, Config), "local.gguf"),
    ok = file:write_file(Path, erllama_server_fake_registry:gguf(4096)),
    {ok, _} = erllama_server_api:register_model(Path, #{
        name => <<"local">>, backend => erllama_model_stub
    }),
    {200, Tags} = get(Config, <<"/api/tags">>),
    ?assertMatch(
        #{
            <<"models">> := [
                #{
                    <<"name">> := <<"local:latest">>,
                    <<"details">> := #{<<"format">> := <<"gguf">>, <<"family">> := <<"llama">>}
                }
            ]
        },
        decode(Tags)
    ),
    {200, Show} = post(Config, <<"/api/show">>, #{<<"model">> => <<"local">>}),
    ?assertMatch(
        #{
            <<"capabilities">> := [<<"completion">>],
            <<"model_info">> := #{<<"general.architecture">> := <<"llama">>}
        },
        decode(Show)
    ),
    {200, _} = post(Config, <<"/api/copy">>, #{
        <<"source">> => <<"local">>, <<"destination">> => <<"local:two">>
    }),
    {ok, Del} = livery_client:request(
        client(), delete, <<(?config(base, Config))/binary, "/api/delete">>, #{
            body => json:encode(#{<<"model">> => <<"local:two">>})
        }
    ),
    ?assertEqual(200, livery_client:status(Del)),
    {200, PsBody} = get(Config, <<"/api/ps">>),
    ?assertMatch(#{<<"models">> := [_ | _]}, decode(PsBody)),
    {200, V} = get(Config, <<"/api/version">>),
    ?assertMatch(#{<<"version">> := _}, decode(V)),
    {501, _} = post(Config, <<"/api/create">>, #{}).

ollama_pull(Config) ->
    {ok, Fake} = erllama_server_fake_registry:start(),
    try
        ok = application:set_env(erllama_server, ollama_base_url, maps:get(base, Fake)),
        {200, Body} = post(Config, <<"/api/pull">>, #{<<"model">> => <<"tiny:1b">>}),
        Lines = ndjson(Body),
        ?assertMatch(#{<<"status">> := <<"pulling manifest">>}, hd(Lines)),
        ?assertMatch(#{<<"status">> := <<"success">>}, lists:last(Lines)),
        ?assert(lists:any(fun(#{<<"status">> := S}) -> S =:= <<"pulling model">> end, Lines)),
        {ok, #{name := <<"tiny:1b">>}} = erllama_server_registry:resolve(<<"tiny:1b">>)
    after
        application:unset_env(erllama_server, ollama_base_url),
        erllama_server_fake_registry:stop(Fake)
    end.

%% -----------------------------------------------------------------------------
%% Gated: tool round trips on a tool-capable model
%% -----------------------------------------------------------------------------

real_model_anthropic_tools(Config) ->
    with_real_model(<<"real_anthropic">>, fun(Id) ->
        Tool = #{
            <<"name">> => <<"get_weather">>,
            <<"description">> => <<"Current weather for a city">>,
            <<"input_schema">> => #{
                <<"type">> => <<"object">>,
                <<"properties">> => #{<<"city">> => #{<<"type">> => <<"string">>}},
                <<"required">> => [<<"city">>]
            }
        },
        Ask = #{
            <<"role">> => <<"user">>,
            <<"content">> => <<"What is the weather in Paris? Use the tool.">>
        },
        Req = #{
            <<"model">> => Id,
            <<"max_tokens">> => 128,
            <<"tools">> => [Tool],
            <<"tool_choice">> => #{<<"type">> => <<"any">>},
            <<"messages">> => [Ask]
        },
        {200, B1} = post(Config, <<"/v1/messages">>, Req),
        case decode(B1) of
            #{<<"stop_reason">> := <<"tool_use">>, <<"content">> := Content} ->
                [#{<<"id">> := UseId, <<"input">> := #{<<"city">> := _}} = Use] = [
                    C
                 || #{<<"type">> := <<"tool_use">>} = C <- Content
                ],
                Result = #{
                    <<"role">> => <<"user">>,
                    <<"content">> => [
                        #{
                            <<"type">> => <<"tool_result">>,
                            <<"tool_use_id">> => UseId,
                            <<"content">> => <<"21 C and sunny">>
                        }
                    ]
                },
                Req2 = Req#{
                    <<"tool_choice">> => #{<<"type">> => <<"none">>},
                    <<"messages">> => [
                        Ask, #{<<"role">> => <<"assistant">>, <<"content">> => [Use]}, Result
                    ]
                },
                {200, B2} = post(Config, <<"/v1/messages">>, Req2),
                #{<<"content">> := [#{<<"type">> := <<"text">>, <<"text">> := Answer} | _]} = decode(
                    B2
                ),
                ct:log("answer: ~ts", [Answer]),
                ?assert(byte_size(Answer) > 0);
            Other ->
                ct:log("no tool use: ~p", [Other]),
                {skip, "the model's template synthesizes no tool grammar"}
        end
    end).

real_model_responses_tools(Config) ->
    with_real_model(<<"real_responses">>, fun(Id) ->
        Tool = #{
            <<"type">> => <<"function">>,
            <<"name">> => <<"get_weather">>,
            <<"parameters">> => #{
                <<"type">> => <<"object">>,
                <<"properties">> => #{<<"city">> => #{<<"type">> => <<"string">>}},
                <<"required">> => [<<"city">>]
            }
        },
        Req = #{
            <<"model">> => Id,
            <<"input">> => <<"What is the weather in Paris? Use the tool.">>,
            <<"tools">> => [Tool],
            <<"tool_choice">> => <<"required">>,
            <<"max_output_tokens">> => 128
        },
        {200, B1} = post(Config, <<"/v1/responses">>, Req),
        #{<<"id">> := Id1, <<"output">> := Output} = decode(B1),
        case [O || #{<<"type">> := <<"function_call">>} = O <- Output] of
            [#{<<"call_id">> := CallId, <<"name">> := <<"get_weather">>} | _] ->
                Next = #{
                    <<"model">> => Id,
                    <<"previous_response_id">> => Id1,
                    <<"input">> => [
                        #{
                            <<"type">> => <<"function_call_output">>,
                            <<"call_id">> => CallId,
                            <<"output">> => <<"21 C and sunny">>
                        }
                    ],
                    <<"tools">> => [Tool],
                    <<"tool_choice">> => <<"none">>,
                    <<"max_output_tokens">> => 64
                },
                {200, B2} = post(Config, <<"/v1/responses">>, Next),
                #{
                    <<"output">> := [
                        #{<<"type">> := <<"message">>, <<"content">> := [#{<<"text">> := Answer}]}
                        | _
                    ]
                } = decode(B2),
                ct:log("answer: ~ts", [Answer]),
                ?assert(byte_size(Answer) > 0);
            [] ->
                {skip, "the model's template synthesizes no tool grammar"}
        end
    end).
