%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_api_tests).
-include_lib("eunit/include/eunit.hrl").
-include("erllama_server_api.hrl").

hello_accepts_own_version_test() ->
    {ok, #{api := V}} = erllama_server_api:hello(?ERLLAMA_SERVER_API_VSN),
    ?assertEqual(?ERLLAMA_SERVER_API_VSN, V),
    ?assertEqual(V, erllama_server_api:api_version()).

hello_refuses_other_versions_test() ->
    ?assertEqual(
        {error, {incompatible_api, #{server => ?ERLLAMA_SERVER_API_VSN, client => 99}}},
        erllama_server_api:hello(99)
    ).

start_run_needs_a_map_test() ->
    ?assertEqual({error, invalid_spec}, erllama_server_api:start_run(prompt)).

server_controlled_options_are_refused_test() ->
    [
        ?assertEqual(
            {error, {invalid_option, K}},
            erllama_server_api:start_run(#{model => <<"m">>, prompt => <<"x">>, opts => #{K => 1}})
        )
     || K <- [parent_key, session_id, to, middleware, media]
    ].

generation_options_are_accepted_test() ->
    ?assertEqual(
        ok,
        erllama_server_generate:check_opts(#{response_tokens => 4, temperature => 0.2, tools => []})
    ).

config_defaults_are_valid_test() ->
    ?assertEqual(ok, erllama_server_config:validate(erllama_server_config:defaults())).

config_rejects_bad_values_test() ->
    ?assertEqual(
        {error, {invalid_config, run_buffer_events, 0}},
        erllama_server_config:validate(#{run_buffer_events => 0})
    ),
    ?assertEqual(
        {error, {invalid_config, max_active_runs, many}},
        erllama_server_config:validate(#{max_active_runs => many})
    ),
    ?assertEqual(ok, erllama_server_config:validate(#{run_retention_ms => 0})).

bad_config_fails_the_start_test() ->
    application:set_env(erllama_server, run_buffer_events, -1),
    try
        ?assertEqual(
            {error, {invalid_config, run_buffer_events, -1}},
            erllama_server_app:start(normal, [])
        )
    after
        application:unset_env(erllama_server, run_buffer_events)
    end.

conversations_test_() ->
    {setup,
        fun() ->
            {ok, Pid} = gen_server:start(
                {local, erllama_server_convs}, erllama_server_convs, [], []
            ),
            Pid
        end,
        fun(Pid) -> gen_server:stop(Pid) end, [
            fun one_run_at_a_time/0,
            fun turn_records_messages_and_parent_key/0,
            fun parent_key_is_per_model/0,
            fun release_only_frees_the_owner/0
        ]}.

one_run_at_a_time() ->
    {ok, #{id := C}} = erllama_server_convs:begin_run(undefined, m, <<"r1">>),
    ?assertEqual({error, conversation_busy}, erllama_server_convs:begin_run(C, m, <<"r2">>)),
    ok = erllama_server_convs:release(C, <<"r1">>),
    ?assertMatch({ok, _}, erllama_server_convs:begin_run(C, m, <<"r2">>)).

turn_records_messages_and_parent_key() ->
    {ok, #{id := C, messages := [], parent_key := undefined}} =
        erllama_server_convs:begin_run(<<"conv_t">>, m, <<"r1">>),
    Turn = [
        #{role => <<"user">>, content => <<"hi">>}, #{role => <<"assistant">>, content => <<"yo">>}
    ],
    ok = erllama_server_convs:end_run(C, <<"r1">>, Turn, <<"key1">>),
    ?assertMatch(
        {ok, #{messages := Turn, parent_key := <<"key1">>}},
        erllama_server_convs:begin_run(C, m, <<"r2">>)
    ).

parent_key_is_per_model() ->
    {ok, #{id := C}} = erllama_server_convs:begin_run(<<"conv_m">>, m, <<"r1">>),
    ok = erllama_server_convs:end_run(C, <<"r1">>, [], <<"key1">>),
    ?assertMatch(
        {ok, #{parent_key := undefined}}, erllama_server_convs:begin_run(C, other, <<"r2">>)
    ).

release_only_frees_the_owner() ->
    {ok, #{id := C}} = erllama_server_convs:begin_run(<<"conv_o">>, m, <<"r1">>),
    ok = erllama_server_convs:release(C, <<"someone_else">>),
    ?assertEqual({error, conversation_busy}, erllama_server_convs:begin_run(C, m, <<"r2">>)),
    ok = erllama_server_convs:end_run(
        C, <<"someone_else">>, [#{role => <<"user">>, content => <<"x">>}], k
    ),
    ?assertMatch({ok, #{messages := []}}, erllama_server_convs:get(C)).
