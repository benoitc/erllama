%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Supervised cache tiers: erllama_cache:add_tier/1, remove_tier/1,
%% list_tiers/0, info/0, the `tiers' application environment key and
%% the load-time tier validation.
-module(erllama_cache_tier_tests).
-include_lib("eunit/include/eunit.hrl").

tmp_dir(Tag) ->
    Dir = filename:join(
        os:getenv("TMPDIR", "/tmp"),
        "erllama_tier_" ++ Tag ++ "_" ++ integer_to_list(erlang:unique_integer([positive]))
    ),
    ok = filelib:ensure_path(Dir),
    Dir.

add_list_remove_test() ->
    erllama_test_helpers:with_app(fun() ->
        Root = tmp_dir("disk"),
        ?assertMatch([#{name := erllama_cache_ram, backend := ram}], erllama_cache:list_tiers()),
        ok = erllama_cache:add_tier(#{name => tier_t1, backend => disk, root => Root}),
        ?assertEqual(
            {error, already_started},
            erllama_cache:add_tier(#{name => tier_t1, backend => disk, root => Root})
        ),
        [_, #{name := tier_t1, backend := disk, root := Root, pid := Pid}] = erllama_cache:list_tiers(),
        ?assert(is_pid(Pid)),
        ?assertEqual(Pid, whereis(tier_t1)),
        ok = erllama_cache:remove_tier(tier_t1),
        ?assertEqual({error, not_found}, erllama_cache:remove_tier(tier_t1)),
        ?assertEqual(undefined, whereis(tier_t1))
    end).

ram_file_tier_test() ->
    erllama_test_helpers:with_app(fun() ->
        Root = tmp_dir("ramfile"),
        ok = erllama_cache:add_tier(#{name => tier_rf, backend => ram_file, root => Root}),
        ?assertMatch(
            [_, #{name := tier_rf, backend := ram_file}],
            erllama_cache:list_tiers()
        ),
        ok = erllama_cache:remove_tier(tier_rf)
    end).

invalid_tier_spec_test() ->
    erllama_test_helpers:with_app(fun() ->
        ?assertMatch({error, {invalid_tier, _}}, erllama_cache:add_tier(#{name => x})),
        ?assertMatch(
            {error, {invalid_tier, _}},
            erllama_cache:add_tier(#{name => x, backend => tape, root => "/tmp"})
        )
    end).

env_tiers_start_with_app_test() ->
    Root = tmp_dir("env"),
    application:set_env(erllama, tiers, [#{name => tier_env, backend => disk, root => Root}]),
    try
        erllama_test_helpers:with_app(fun() ->
            ?assertMatch([_, #{name := tier_env, backend := disk}], erllama_cache:list_tiers())
        end)
    after
        application:unset_env(erllama, tiers)
    end.

model_saves_to_added_tier_test() ->
    erllama_test_helpers:with_app(fun() ->
        Root = tmp_dir("model"),
        ok = erllama_cache:add_tier(#{name => tier_m, backend => disk, root => Root}),
        {ok, Id} = erllama:load_model(#{
            backend => erllama_model_stub,
            tier => disk,
            tier_srv => tier_m,
            fingerprint => crypto:strong_rand_bytes(32),
            policy => #{
                min_tokens => 1,
                cold_min_tokens => 1,
                boundary_trim_tokens => 0,
                boundary_align_tokens => 1
            }
        }),
        try
            {ok, #{finish_key := Key}} = erllama:complete(Id, <<"one two three four">>, #{
                response_tokens => 2
            }),
            ?assert(is_binary(Key)),
            ok = wait_for_disk_row(Root, 2000),
            #{entries := N, bytes := #{disk := Disk}, tiers := Tiers} = erllama_cache:info(),
            ?assert(N >= 1),
            ?assert(Disk > 0),
            ?assertMatch([_, #{name := tier_m}], Tiers)
        after
            erllama:unload(Id)
        end
    end).

wait_for_disk_row(Root, Timeout) ->
    Deadline = erlang:monotonic_time(millisecond) + Timeout,
    wait_for_disk_row_loop(Root, Deadline).

wait_for_disk_row_loop(Root, Deadline) ->
    case filelib:wildcard(filename:join(Root, "*.kvc")) of
        [_ | _] ->
            ok;
        [] ->
            case erlang:monotonic_time(millisecond) > Deadline of
                true ->
                    erlang:error(no_disk_row);
                false ->
                    timer:sleep(20),
                    wait_for_disk_row_loop(Root, Deadline)
            end
    end.

load_validates_tier_test() ->
    erllama_test_helpers:with_app(fun() ->
        Base = #{backend => erllama_model_stub},
        ?assertEqual(
            {error, {invalid_config, tier_srv, nope}},
            erllama:load_model(Base#{tier => disk, tier_srv => nope})
        ),
        ?assertEqual(
            {error, {invalid_config, tier, disk}},
            erllama:load_model(Base#{tier => disk})
        ),
        Root = tmp_dir("validate"),
        ok = erllama_cache:add_tier(#{name => tier_v, backend => ram_file, root => Root}),
        ?assertEqual(
            {error, {invalid_config, tier, disk}},
            erllama:load_model(Base#{tier => disk, tier_srv => tier_v})
        ),
        {ok, Id} = erllama:load_model(Base#{tier => ram_file, tier_srv => tier_v}),
        ok = erllama:unload(Id)
    end).
