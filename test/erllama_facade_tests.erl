%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Tests for the public erllama module façade. Exercises the
%% bucket-A additions: list_models/0, model_info/1, tokenize/2,
%% detokenize/2, unload/1, whereis/1, validation and not_loaded shapes.
-module(erllama_facade_tests).
-include_lib("eunit/include/eunit.hrl").

%% =============================================================================
%% Fixtures
%% =============================================================================

with_app(Body) ->
    {ok, Started} = application:ensure_all_started(erllama),
    try
        Body()
    after
        [application:stop(A) || A <- lists:reverse(Started)],
        ok
    end.

minimal_config() ->
    #{
        backend => erllama_model_stub,
        fingerprint => binary:copy(<<16#11>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        quant_bits => 16,
        ctx_params_hash => binary:copy(<<16#22>>, 32),
        context_size => 1024,
        policy => #{
            min_tokens => 4,
            cold_min_tokens => 4,
            cold_max_tokens => 1000,
            continued_interval => 2048,
            boundary_trim_tokens => 0,
            boundary_align_tokens => 1,
            session_resume_wait_ms => 50
        }
    }.

%% =============================================================================
%% list_models / load_model / unload
%% =============================================================================

empty_list_models_test() ->
    with_app(fun() ->
        ?assertEqual([], erllama:list_models())
    end).

load_unload_model_roundtrip_test() ->
    with_app(fun() ->
        {ok, Id} = erllama:load_model(<<"facade_a">>, minimal_config()),
        ?assertEqual(<<"facade_a">>, Id),
        ?assertMatch([_], erllama:list_models()),
        ok = erllama:unload(Id),
        ?assertEqual([], erllama:list_models())
    end).

load_model_auto_id_returns_binary_test() ->
    with_app(fun() ->
        {ok, Id} = erllama:load_model(minimal_config()),
        try
            ?assert(is_binary(Id)),
            ?assertMatch(<<"erllama_model_", _/binary>>, Id)
        after
            erllama:unload(Id)
        end
    end).

list_models_includes_metadata_test() ->
    with_app(fun() ->
        {ok, Id} = erllama:load_model(<<"facade_b">>, minimal_config()),
        try
            [Info] = erllama:list_models(),
            ?assertEqual(Id, maps:get(id, Info)),
            ?assertEqual(idle, maps:get(status, Info)),
            ?assertEqual(erllama_model_stub, maps:get(backend, Info)),
            ?assertEqual(1024, maps:get(context_size, Info))
        after
            erllama:unload(Id)
        end
    end).

load_same_id_returns_already_loaded_test() ->
    with_app(fun() ->
        {ok, _} = erllama:load_model(<<"dup">>, minimal_config()),
        try
            ?assertEqual(
                {error, already_loaded},
                erllama:load_model(<<"dup">>, minimal_config())
            )
        after
            erllama:unload(<<"dup">>)
        end
    end).

%% =============================================================================
%% model_info / tokenize / detokenize
%% =============================================================================

model_info_test() ->
    with_app(fun() ->
        {ok, Id} = erllama:load_model(<<"facade_info">>, minimal_config()),
        try
            {ok, Info} = erllama:model_info(Id),
            ?assertEqual(Id, maps:get(id, Info)),
            ?assertEqual(idle, maps:get(status, Info))
        after
            erllama:unload(Id)
        end
    end).

tokenize_then_detokenize_test() ->
    with_app(fun() ->
        {ok, Id} = erllama:load_model(<<"facade_tok">>, minimal_config()),
        try
            {ok, Tokens} = erllama:tokenize(Id, <<"hello world">>),
            ?assert(is_list(Tokens)),
            {ok, Bin} = erllama:detokenize(Id, Tokens),
            ?assert(is_binary(Bin))
        after
            erllama:unload(Id)
        end
    end).

unload_unknown_returns_not_loaded_test() ->
    with_app(fun() ->
        ?assertEqual({error, not_loaded}, erllama:unload(<<"absent">>))
    end).

%% =============================================================================
%% not_loaded everywhere, validation, model_id, whereis
%% =============================================================================

not_loaded_shapes_test() ->
    with_app(fun() ->
        Id = <<"facade_absent">>,
        ?assertEqual({error, not_loaded}, erllama:model_info(Id)),
        ?assertEqual({error, not_loaded}, erllama:status(Id)),
        ?assertEqual({error, not_loaded}, erllama:phase(Id)),
        ?assertEqual({error, not_loaded}, erllama:pending_len(Id)),
        ?assertEqual({error, not_loaded}, erllama:queue_depth(Id)),
        ?assertEqual({error, not_loaded}, erllama:last_cache_hit(Id)),
        ?assertEqual({error, not_loaded}, erllama:list_adapters(Id)),
        ?assertEqual({error, not_loaded}, erllama:whereis(Id)),
        ?assertEqual({error, not_loaded}, erllama:tokenize(Id, <<"x">>)),
        ?assertEqual({error, not_loaded}, erllama:detokenize(Id, [1])),
        ?assertEqual({error, not_loaded}, erllama:complete(Id, <<"x">>)),
        ?assertEqual({error, not_loaded}, erllama:prefill_only(Id, [1])),
        ?assertEqual({error, not_loaded}, erllama:stream(Id, [1], #{})),
        ?assertEqual({error, not_loaded}, erllama:end_session(Id, s)),
        ?assertEqual({error, not_loaded}, erllama:reset_session(Id, s)),
        ?assertEqual({error, not_loaded}, erllama:evict(Id)),
        ?assertEqual({error, not_loaded}, erllama:shutdown(Id)),
        ?assertEqual({error, not_loaded}, erllama:cached_prefix_len(Id, [1, 2])),
        ?assertEqual({ok, 0}, erllama:cached_prefix_len(Id, []))
    end).

loaded_shapes_test() ->
    with_app(fun() ->
        {ok, Id} = erllama:load_model(<<"facade_shapes">>, minimal_config()),
        try
            {ok, Pid} = erllama:whereis(Id),
            ?assert(is_pid(Pid)),
            ?assertEqual({ok, Pid}, erllama:whereis(Pid)),
            ?assertEqual({ok, idle}, erllama:status(Id)),
            ?assertEqual({ok, idle}, erllama:phase(Id)),
            ?assertEqual({ok, 0}, erllama:pending_len(Id)),
            ?assertEqual({ok, 0}, erllama:queue_depth(Id)),
            ?assertEqual({ok, undefined}, erllama:last_cache_hit(Id)),
            ?assertEqual({ok, []}, erllama:list_adapters(Id)),
            ?assertEqual(ok, erllama:evict(Id)),
            {ok, #{reply := _}} = erllama:complete(Id, <<"hello world">>, #{response_tokens => 2}),
            ?assertMatch({ok, #{kind := _, prefix_len := _}}, erllama:last_cache_hit(Id))
        after
            erllama:unload(Id)
        end
    end).

model_id_from_config_test() ->
    with_app(fun() ->
        Config = (minimal_config())#{model_id => <<"facade_named">>},
        {ok, <<"facade_named">>} = erllama:load_model(Config),
        ?assertEqual({error, already_loaded}, erllama:load_model(Config)),
        ok = erllama:unload(<<"facade_named">>)
    end).

load_validation_test() ->
    with_app(fun() ->
        ?assertEqual({error, {missing_config, model_path}}, erllama:load_model(#{})),
        ?assertEqual(
            {error, {invalid_config, model_path, "/no/such.gguf"}},
            erllama:load_model(#{model_path => "/no/such.gguf"})
        ),
        ?assertEqual(
            {error, {unknown_option, bogus}},
            erllama:load_model((minimal_config())#{bogus => 1})
        ),
        ?assertEqual([], erllama:list_models())
    end).

request_validation_test() ->
    with_app(fun() ->
        {ok, Id} = erllama:load_model(<<"facade_reqv">>, minimal_config()),
        try
            ?assertEqual(
                {error, {unknown_option, max_tokens}},
                erllama:complete(Id, <<"x">>, #{max_tokens => 1})
            ),
            ?assertEqual(
                {error, {invalid_option, response_tokens, 0}},
                erllama:complete(Id, <<"x">>, #{response_tokens => 0})
            ),
            ?assertEqual(
                {error, {unknown_option, response_tokens}},
                erllama:prefill_only(Id, [1, 2], #{response_tokens => 1})
            ),
            ?assertEqual(
                {error, {unknown_option, nope}},
                erllama:stream(Id, [1, 2], #{nope => 1})
            )
        after
            erllama:unload(Id)
        end
    end).
