%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(barrel_inference_scheduler_tests).
-behaviour(barrel_inference_pressure).
-include_lib("eunit/include/eunit.hrl").
-include("barrel_inference_cache.hrl").

-export([sample/0]).

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
