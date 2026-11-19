%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_cache_ram_tests).
-include_lib("eunit/include/eunit.hrl").
-include("erllama_cache.hrl").

with_cache(Body) ->
    with_cache(1024 * 1024, Body).

%% Run Body against a RAM tier with the given `ram_tier_budget`.
with_cache(Budget, Body) ->
    ok = erllama_cache_counters:init(),
    application:set_env(erllama, ram_tier_budget, Budget),
    {ok, _M} = erllama_cache_meta_srv:start_link(),
    {ok, _R} = erllama_cache_ram:start_link(),
    try
        Body()
    after
        erllama_test_helpers:stop_quiet(erllama_cache_ram),
        erllama_test_helpers:stop_quiet(erllama_cache_meta_srv),
        application:unset_env(erllama, ram_tier_budget)
    end.

key(N) ->
    crypto:hash(sha256, integer_to_binary(N)).

%% Store a slab and register its row, as the writer does.
store_row(Key, Slab) ->
    ok = erllama_cache_ram:store(Key, Slab),
    erllama_cache_meta_srv:insert_available(Key, ram, byte_size(Slab), <<>>, {ram}).

store_then_load_returns_slab_test() ->
    with_cache(fun() ->
        Slab = <<"hello slab">>,
        ok = store_row(key(1), Slab),
        ?assertEqual({ok, Slab}, erllama_cache_ram:load(key(1)))
    end).

row_registered_for_stored_slab_test() ->
    with_cache(fun() ->
        Slab = <<"x">>,
        ok = store_row(key(1), Slab),
        {ok, Row} = erllama_cache_meta_srv:lookup_exact(key(1)),
        ?assertEqual(ram, element(?POS_TIER, Row)),
        ?assertEqual(1, element(?POS_SIZE, Row))
    end).

load_miss_unknown_key_test() ->
    with_cache(fun() ->
        ?assertEqual(miss, erllama_cache_ram:load(key(99)))
    end).

delete_removes_slab_test() ->
    with_cache(fun() ->
        ok = store_row(key(1), <<"x">>),
        ok = erllama_cache_ram:delete(key(1)),
        ?assertEqual(miss, erllama_cache_ram:load(key(1)))
    end).

size_bytes_aggregates_test() ->
    with_cache(fun() ->
        ok = store_row(key(1), <<"abc">>),
        ok = store_row(key(2), <<"defgh">>),
        ?assertEqual(8, erllama_cache_ram:size_bytes())
    end).

size_bytes_decrements_after_delete_test() ->
    with_cache(fun() ->
        ok = store_row(key(1), <<"abc">>),
        ok = store_row(key(2), <<"defgh">>),
        ok = erllama_cache_ram:delete(key(1)),
        ?assertEqual(5, erllama_cache_ram:size_bytes())
    end).

%% =============================================================================
%% End-to-end: meta_srv eviction also clears the slab from RAM
%% =============================================================================

eviction_removes_slab_too_test() ->
    with_cache(fun() ->
        ok = store_row(key(1), <<"abc">>),
        ok = store_row(key(2), <<"defgh">>),
        {evicted, 2} = erllama_cache_meta_srv:gc(),
        ?assertEqual(miss, erllama_cache_ram:load(key(1))),
        ?assertEqual(miss, erllama_cache_ram:load(key(2))),
        ?assertEqual(0, erllama_cache_ram:size_bytes())
    end).

eviction_skips_referenced_slab_test() ->
    with_cache(fun() ->
        ok = store_row(key(1), <<"abc">>),
        ok = store_row(key(2), <<"defgh">>),
        {ok, _Ref, _, _, _, _} = erllama_cache_meta_srv:checkout(key(1), self()),
        {evicted, 1} = erllama_cache_meta_srv:gc(),
        ?assertMatch({ok, _}, erllama_cache_ram:load(key(1))),
        ?assertEqual(miss, erllama_cache_ram:load(key(2)))
    end).

%% =============================================================================
%% Budget
%% =============================================================================

store_evicts_oldest_rows_to_fit_test() ->
    with_cache(8, fun() ->
        ok = store_row(key(1), <<"abcde">>),
        ok = store_row(key(2), <<"fgh">>),
        %% 8 of 8 bytes used: the next 2 bytes evict the oldest row.
        ok = store_row(key(3), <<"ij">>),
        ?assertEqual(miss, erllama_cache_ram:load(key(1))),
        ?assertEqual(miss, erllama_cache_meta_srv:lookup_exact(key(1))),
        ?assertMatch({ok, _}, erllama_cache_ram:load(key(2))),
        ?assertMatch({ok, _}, erllama_cache_ram:load(key(3))),
        ?assert(erllama_cache_ram:size_bytes() =< 8)
    end).

store_rejects_slab_larger_than_budget_test() ->
    with_cache(4, fun() ->
        ?assertEqual({error, too_large}, erllama_cache_ram:store(key(1), <<"abcde">>)),
        ?assertEqual(miss, erllama_cache_ram:load(key(1)))
    end).

zero_budget_disables_ram_saves_test() ->
    with_cache(0, fun() ->
        ?assertEqual({error, disabled}, erllama_cache_ram:store(key(1), <<"a">>)),
        ?assertEqual(0, erllama_cache_ram:budget())
    end).

store_keeps_checked_out_rows_test() ->
    with_cache(5, fun() ->
        ok = store_row(key(1), <<"abcde">>),
        {ok, Ref, _, _, _, _} = erllama_cache_meta_srv:checkout(key(1), self()),
        %% The only row is pinned by a reader: nothing can be evicted.
        ?assertEqual({error, ram_full}, erllama_cache_ram:store(key(2), <<"x">>)),
        ?assertMatch({ok, _}, erllama_cache_ram:load(key(1))),
        ok = erllama_cache_meta_srv:checkin(Ref),
        ?assertEqual(ok, erllama_cache_ram:store(key(2), <<"x">>))
    end).

resolve_budget_test() ->
    ?assertEqual({ok, 0}, erllama_cache_ram:resolve_budget(0)),
    ?assertEqual({ok, 4096}, erllama_cache_ram:resolve_budget(4096)),
    {ok, Frac} = erllama_cache_ram:resolve_budget({fraction, 0.25}),
    ?assert(Frac > 0),
    [
        ?assertEqual(
            {error, {invalid_config, ram_tier_budget, Bad}},
            erllama_cache_ram:resolve_budget(Bad)
        )
     || Bad <- [-1, {fraction, 0}, {fraction, 1.5}, {fraction, x}, big]
    ].

invalid_budget_fails_start_test() ->
    application:set_env(erllama, ram_tier_budget, {fraction, 2}),
    process_flag(trap_exit, true),
    try
        ?assertEqual(
            {error, {invalid_config, ram_tier_budget, {fraction, 2}}},
            erllama_cache_ram:start_link()
        )
    after
        process_flag(trap_exit, false),
        application:unset_env(erllama, ram_tier_budget)
    end.
