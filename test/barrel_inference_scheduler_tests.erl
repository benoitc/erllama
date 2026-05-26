%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(barrel_inference_scheduler_tests).
-behaviour(barrel_inference_pressure).
-include_lib("eunit/include/eunit.hrl").
-include("barrel_inference_cache.hrl").

-export([sample/0, evict_one/0]).

%% Model-evictor callback for the proactive-eviction tests. Returns
%% whatever the test stashed (default `none`), so a test can drive
%% {unloaded, _} / none / garbage returns deterministically.
-define(EVICT_RET_KEY, {?MODULE, evict_ret}).

evict_one() ->
    persistent_term:get(?EVICT_RET_KEY, none).

evict_ret_set(V) -> persistent_term:put(?EVICT_RET_KEY, V).
evict_ret_clear() -> catch persistent_term:erase(?EVICT_RET_KEY).

%% =============================================================================
%% Fixtures
%% =============================================================================

%% A pluggable pressure source that reads from persistent_term so tests
%% can drive the scheduler deterministically without exec'ing real
%% commands.
-define(STUB_KEY, {?MODULE, stub_pressure}).

stub_set(Used, Total) ->
    persistent_term:put(?STUB_KEY, {Used, Total}).

stub_clear() ->
    catch persistent_term:erase(?STUB_KEY).

with_subsystem(Body) ->
    ok = barrel_inference_cache_counters:init(),
    barrel_inference_cache_counters:reset(),
    {ok, _Meta} = barrel_inference_cache_meta_srv:start_link(),
    {ok, _Ram} = barrel_inference_cache_ram:start_link(),
    try
        Body()
    after
        catch gen_server:stop(barrel_inference_cache_ram),
        catch gen_server:stop(barrel_inference_cache_meta_srv),
        stub_clear()
    end.

with_scheduler(Config, Body) ->
    with_subsystem(fun() ->
        {ok, _Sch} = barrel_inference_scheduler:start_link(Config),
        try
            Body()
        after
            catch gen_server:stop(barrel_inference_scheduler)
        end
    end).

key(N) ->
    crypto:hash(sha256, <<"sched-test-", (integer_to_binary(N))/binary>>).

insert_slab(N, Size) ->
    K = key(N),
    ok = barrel_inference_cache_meta_srv:insert_available(K, ram, Size, <<"H">>, {ram}),
    K.

%% =============================================================================
%% Pressure source dispatch
%% =============================================================================

sample_noop_test() ->
    ?assertEqual({0, 1}, barrel_inference_pressure:sample(noop)).

sample_module_test() ->
    stub_set(75, 100),
    try
        ?assertEqual(
            {75, 100},
            barrel_inference_pressure:sample({module, ?MODULE})
        )
    after
        stub_clear()
    end.

%% Behaviour callback so {module, ?MODULE} works in dispatch tests.
sample() ->
    persistent_term:get(?STUB_KEY).

%% =============================================================================
%% Scheduler basics
%% =============================================================================

starts_disabled_test() ->
    with_scheduler(#{}, fun() ->
        Status = barrel_inference_scheduler:status(),
        ?assertEqual(false, maps:get(enabled, Status)),
        ?assertEqual(noop, maps:get(pressure_source, Status))
    end).

force_check_disabled_returns_skipped_test() ->
    with_scheduler(#{}, fun() ->
        ?assertEqual({skipped, disabled}, barrel_inference_scheduler:force_check())
    end).

force_check_below_watermark_test() ->
    stub_set(10, 100),
    with_scheduler(
        #{
            enabled => true,
            pressure_source => {module, ?MODULE},
            high_watermark => 0.9,
            low_watermark => 0.7
        },
        fun() ->
            ?assertEqual(
                {skipped, below_watermark},
                barrel_inference_scheduler:force_check()
            )
        end
    ).

force_check_evicts_when_above_watermark_test() ->
    stub_set(95, 100),
    with_scheduler(
        #{
            enabled => true,
            pressure_source => {module, ?MODULE},
            high_watermark => 0.85,
            low_watermark => 0.75,
            min_evict_bytes => 1
        },
        fun() ->
            _K1 = insert_slab(1, 5),
            _K2 = insert_slab(2, 7),
            Result = barrel_inference_scheduler:force_check(),
            case Result of
                {evicted, N, Bytes} ->
                    ?assert(N >= 1),
                    ?assert(Bytes >= 5);
                Other ->
                    ?assertMatch({evicted, _, _}, Other)
            end
        end
    ).

force_check_above_watermark_no_slabs_test() ->
    stub_set(95, 100),
    with_scheduler(
        #{
            enabled => true,
            pressure_source => {module, ?MODULE},
            high_watermark => 0.85,
            low_watermark => 0.75
        },
        fun() ->
            ?assertEqual(
                {skipped, nothing_to_evict},
                barrel_inference_scheduler:force_check()
            )
        end
    ).

enable_disable_test() ->
    with_scheduler(#{}, fun() ->
        ok = barrel_inference_scheduler:enable(true),
        #{enabled := true} = barrel_inference_scheduler:status(),
        ok = barrel_inference_scheduler:enable(false),
        #{enabled := false} = barrel_inference_scheduler:status()
    end).

set_thresholds_test() ->
    with_scheduler(#{}, fun() ->
        ok = barrel_inference_scheduler:set_thresholds(0.92, 0.5),
        #{high_watermark := 0.92, low_watermark := 0.5} =
            barrel_inference_scheduler:status(),
        ?assertMatch({error, _}, barrel_inference_scheduler:set_thresholds(0.5, 0.9))
    end).

%% Config validation is exercised directly through validate_config/1
%% rather than spawning a gen_server with bad config: the latter would
%% emit a SASL =CRASH REPORT= for every case and pollute every CI log.

invalid_watermarks_at_init_test() ->
    Cfg = #{high_watermark => 0.5, low_watermark => 0.9},
    ?assertMatch(
        {error, {invalid_config, {watermarks, _}}},
        barrel_inference_scheduler:validate_config(Cfg)
    ).

invalid_interval_zero_at_init_test() ->
    Cfg = #{interval_ms => 0},
    ?assertMatch(
        {error, {invalid_config, {interval_ms, _}}},
        barrel_inference_scheduler:validate_config(Cfg)
    ).

invalid_interval_negative_at_init_test() ->
    Cfg = #{interval_ms => -10},
    ?assertMatch(
        {error, {invalid_config, {interval_ms, _}}},
        barrel_inference_scheduler:validate_config(Cfg)
    ).

disk_tier_skipped_by_default_test() ->
    stub_set(95, 100),
    with_scheduler(
        #{
            enabled => true,
            pressure_source => {module, ?MODULE},
            high_watermark => 0.85,
            low_watermark => 0.75,
            min_evict_bytes => 1
        },
        fun() ->
            DiskKey = crypto:hash(sha256, <<"sched-disk-only">>),
            ok = barrel_inference_cache_meta_srv:insert_available(
                DiskKey, disk, 1024, <<"H">>, {disk, "/tmp/x"}
            ),
            ?assertEqual(
                {skipped, nothing_to_evict},
                barrel_inference_scheduler:force_check()
            ),
            {ok, _Row} = barrel_inference_cache_meta_srv:lookup_exact(DiskKey)
        end
    ).

disk_tier_evicted_when_explicit_test() ->
    stub_set(95, 100),
    with_scheduler(
        #{
            enabled => true,
            pressure_source => {module, ?MODULE},
            high_watermark => 0.85,
            low_watermark => 0.75,
            min_evict_bytes => 1,
            evict_tiers => all
        },
        fun() ->
            DiskKey = crypto:hash(sha256, <<"sched-disk-evict">>),
            ok = barrel_inference_cache_meta_srv:insert_available(
                DiskKey, disk, 1024, <<"H">>, {disk, "/tmp/y"}
            ),
            {evicted, 1, 1024} = barrel_inference_scheduler:force_check()
        end
    ).

%% =============================================================================
%% Proactive model eviction (escalation when cache can't relieve pressure)
%% =============================================================================

model_unload_config() ->
    #{
        enabled => true,
        pressure_source => {module, ?MODULE},
        high_watermark => 0.85,
        low_watermark => 0.75,
        min_evict_bytes => 1,
        unload_models_under_pressure => true,
        model_evictor => ?MODULE
    }.

model_unload_escalates_when_cache_empty_test() ->
    stub_set(95, 100),
    evict_ret_set({unloaded, <<"m1">>}),
    try
        with_scheduler(model_unload_config(), fun() ->
            %% Empty cache -> evict_bytes returns {evicted,0,0} -> escalate.
            ?assertEqual({unloaded_model, <<"m1">>}, barrel_inference_scheduler:force_check()),
            S = barrel_inference_scheduler:status(),
            ?assertEqual(1, maps:get(models_unloaded_total, S)),
            ?assertEqual(<<"m1">>, maps:get(last_model_unloaded, S))
        end)
    after
        evict_ret_clear()
    end.

model_unload_none_when_no_idle_test() ->
    stub_set(95, 100),
    evict_ret_set(none),
    try
        with_scheduler(model_unload_config(), fun() ->
            ?assertEqual({skipped, nothing_to_evict}, barrel_inference_scheduler:force_check()),
            ?assertEqual(0, maps:get(models_unloaded_total, barrel_inference_scheduler:status()))
        end)
    after
        evict_ret_clear()
    end.

model_unload_garbage_return_falls_back_test() ->
    stub_set(95, 100),
    evict_ret_set({unloaded, not_a_binary}),
    try
        with_scheduler(model_unload_config(), fun() ->
            %% Garbage return is normalised to none; scheduler must not crash.
            ?assertEqual({skipped, nothing_to_evict}, barrel_inference_scheduler:force_check()),
            ?assertEqual(0, maps:get(models_unloaded_total, barrel_inference_scheduler:status()))
        end)
    after
        evict_ret_clear()
    end.

model_unload_disabled_by_default_test() ->
    stub_set(95, 100),
    evict_ret_set({unloaded, <<"m1">>}),
    %% model_evictor set but unload_models_under_pressure absent (default false).
    Cfg = #{
        enabled => true,
        pressure_source => {module, ?MODULE},
        high_watermark => 0.85,
        low_watermark => 0.75,
        min_evict_bytes => 1,
        model_evictor => ?MODULE
    },
    try
        with_scheduler(Cfg, fun() ->
            ?assertEqual({skipped, nothing_to_evict}, barrel_inference_scheduler:force_check())
        end)
    after
        evict_ret_clear()
    end.

cache_relief_skips_model_unload_test() ->
    stub_set(95, 100),
    evict_ret_set({unloaded, <<"m1">>}),
    try
        with_scheduler(model_unload_config(), fun() ->
            %% A slab larger than the target lets the cache relieve pressure,
            %% so the evictor must not be consulted.
            _K = insert_slab(1, 1000),
            ?assertMatch({evicted, _, _}, barrel_inference_scheduler:force_check()),
            ?assertEqual(0, maps:get(models_unloaded_total, barrel_inference_scheduler:status()))
        end)
    after
        evict_ret_clear()
    end.

invalid_unload_flag_at_init_test() ->
    ?assertMatch(
        {error, {invalid_config, {unload_models_under_pressure, _}}},
        barrel_inference_scheduler:validate_config(#{unload_models_under_pressure => yes})
    ).

invalid_model_evictor_at_init_test() ->
    ?assertMatch(
        {error, {invalid_config, {model_evictor, _}}},
        barrel_inference_scheduler:validate_config(#{model_evictor => "not_an_atom"})
    ).

valid_model_unload_config_test() ->
    ?assertEqual(
        ok,
        barrel_inference_scheduler:validate_config(#{
            unload_models_under_pressure => true, model_evictor => some_module
        })
    ).

sample_records_reading_test() ->
    stub_set(42, 100),
    with_scheduler(
        #{pressure_source => {module, ?MODULE}},
        fun() ->
            {42, 100} = barrel_inference_scheduler:sample(),
            #{
                last_used := 42,
                last_total := 100,
                last_ratio := R
            } = barrel_inference_scheduler:status(),
            ?assert(R > 0.41 andalso R < 0.43)
        end
    ).
