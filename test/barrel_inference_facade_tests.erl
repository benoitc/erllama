%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Tests for the public barrel_inference module façade. Exercises the
%% bucket-A additions: list_models/0, model_info/1, tokenize/2,
%% detokenize/2, unload_model/1.
-module(barrel_inference_facade_tests).
-include_lib("eunit/include/eunit.hrl").

%% =============================================================================
%% Fixtures
%% =============================================================================

with_app(Body) ->
    {ok, Started} = application:ensure_all_started(barrel_inference),
    try
        Body()
    after
        [application:stop(A) || A <- lists:reverse(Started)],
        ok
    end.

minimal_config() ->
    #{
        backend => barrel_inference_model_stub,
        tier_srv => barrel_inference_cache_disk_default,
        tier => disk,
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
%% list_models / load_model / unload_model
%% =============================================================================

empty_list_models_test() ->
    with_app(fun() ->
        ?assertEqual([], barrel_inference:list_models())
    end).

load_unload_model_roundtrip_test() ->
    with_app(fun() ->
        {ok, Id} = barrel_inference:load_model(<<"facade_a">>, minimal_config()),
        ?assertEqual(<<"facade_a">>, Id),
        ?assertMatch([_], barrel_inference:list_models()),
        ok = barrel_inference:unload_model(Id),
        ?assertEqual([], barrel_inference:list_models())
    end).

load_model_auto_id_returns_binary_test() ->
    with_app(fun() ->
        {ok, Id} = barrel_inference:load_model(minimal_config()),
        try
            ?assert(is_binary(Id)),
            ?assertMatch(<<"barrel_inference_model_", _/binary>>, Id)
        after
            barrel_inference:unload(Id)
        end
    end).

list_models_includes_metadata_test() ->
    with_app(fun() ->
        {ok, Id} = barrel_inference:load_model(<<"facade_b">>, minimal_config()),
        try
            [Info] = barrel_inference:list_models(),
            ?assertEqual(Id, maps:get(id, Info)),
            ?assertEqual(idle, maps:get(status, Info)),
            ?assertEqual(barrel_inference_model_stub, maps:get(backend, Info)),
            ?assertEqual(1024, maps:get(context_size, Info))
        after
            barrel_inference:unload_model(Id)
        end
    end).

load_same_id_returns_already_loaded_test() ->
    with_app(fun() ->
        {ok, _} = barrel_inference:load_model(<<"dup">>, minimal_config()),
        try
            ?assertEqual(
                {error, already_loaded},
                barrel_inference:load_model(<<"dup">>, minimal_config())
            )
        after
            barrel_inference:unload(<<"dup">>)
        end
    end).

%% =============================================================================
%% model_info / tokenize / detokenize
%% =============================================================================

model_info_test() ->
    with_app(fun() ->
        {ok, Id} = barrel_inference:load_model(<<"facade_info">>, minimal_config()),
        try
            Info = barrel_inference:model_info(Id),
            ?assertEqual(Id, maps:get(id, Info)),
            ?assertEqual(idle, maps:get(status, Info))
        after
            barrel_inference:unload(Id)
        end
    end).

tokenize_then_detokenize_test() ->
    with_app(fun() ->
        {ok, Id} = barrel_inference:load_model(<<"facade_tok">>, minimal_config()),
        try
            {ok, Tokens} = barrel_inference:tokenize(Id, <<"hello world">>),
            ?assert(is_list(Tokens)),
            {ok, Bin} = barrel_inference:detokenize(Id, Tokens),
            ?assert(is_binary(Bin))
        after
            barrel_inference:unload(Id)
        end
    end).

unload_unknown_returns_not_found_test() ->
    with_app(fun() ->
        ?assertEqual({error, not_found}, barrel_inference:unload_model(<<"absent">>))
    end).
