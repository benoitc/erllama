%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% Model lifetime: registry models load on demand, concurrent requests
%% share one load, a model stays loaded for its keep-alive once nothing
%% holds it, erllama's scheduler can evict the least recently used idle
%% one, and an unload followed by a run reloads cleanly. The models are
%% registered on erllama's stub backend so the suite runs anywhere.
%% @end
-module(erllama_server_models_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_testcase/2, end_per_testcase/2]).
-export([
    loads_on_demand/1,
    concurrent_requests_share_one_load/1,
    keep_alive_unloads_idle_model/1,
    keep_alive_zero_and_infinity/1,
    held_model_outlives_keep_alive/1,
    evictor_unloads_least_recently_used/1,
    unload_then_run_reloads/1,
    two_models_at_once/1,
    load_failure_reaches_every_waiter/1
]).

all() ->
    [
        loads_on_demand,
        concurrent_requests_share_one_load,
        keep_alive_unloads_idle_model,
        keep_alive_zero_and_infinity,
        held_model_outlives_keep_alive,
        evictor_unloads_least_recently_used,
        unload_then_run_reloads,
        two_models_at_once,
        load_failure_reaches_every_waiter
    ].

init_per_testcase(TC, Config) ->
    Dir = filename:join(?config(priv_dir, Config), atom_to_list(TC)),
    ok = filelib:ensure_path(Dir),
    ok = application:set_env(erllama_server, data_dir, Dir),
    ok = application:set_env(erllama_server, keep_alive_ms, 60000),
    {ok, _} = application:ensure_all_started(erllama_server),
    [{dir, Dir} | Config].

end_per_testcase(_TC, _Config) ->
    _ = application:stop(erllama_server),
    [erllama:unload(Id) || #{id := Id} <- erllama:list_models()],
    application:unset_env(erllama_server, data_dir),
    application:unset_env(erllama_server, keep_alive_ms),
    ok.

%% A stub-backed registry model named `Name'.
add_model(Config, Name) ->
    add_model(Config, Name, #{}).

add_model(Config, Name, Extra) ->
    Path = filename:join(?config(dir, Config), <<Name/binary, ".gguf">>),
    ok = file:write_file(Path, erllama_server_fake_registry:gguf(4096)),
    {ok, #{name := Id}} = erllama_server_api:register_model(
        Path, maps:merge(#{name => Name, backend => erllama_model_stub, step_delay_ms => 1}, Extra)
    ),
    Id.

loaded(Id) -> erllama:whereis(Id) =/= {error, not_loaded}.

run(Model, Opts) ->
    Spec = maps:merge(
        #{
            model => Model,
            messages => [#{role => user, content => <<"hi">>}],
            opts => #{response_tokens => 3}
        },
        Opts
    ),
    {ok, #{run_id := Run}} = erllama_server_api:start_run(Spec),
    Run.

wait_done(Run) ->
    wait(fun() ->
        case erllama_server_api:result(Run) of
            {ok, #{status := S}} -> S =/= running;
            _ -> false
        end
    end),
    {ok, R} = erllama_server_api:result(Run),
    R.

wait(Fun) -> wait(Fun, 300).
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

events(Run) ->
    {ok, #{replay := Replay}} = erllama_server_api:attach(Run, self(), 0),
    [T || #{type := T} <- Replay].

%% -----------------------------------------------------------------------------

loads_on_demand(Config) ->
    Id = add_model(Config, <<"alpha">>),
    ?assertNot(loaded(Id)),
    Run = run(<<"alpha">>, #{}),
    ?assertMatch(#{status := done}, wait_done(Run)),
    ?assert(loaded(Id)),
    ?assertMatch([started, loading | _], events(Run)),
    [#{id := Id, managed := true, holders := 0}] = erllama_server_models:ps().

concurrent_requests_share_one_load(Config) ->
    Id = add_model(Config, <<"beta">>),
    Self = self(),
    [
        spawn(fun() -> Self ! {got, erllama_server_models:acquire(<<"beta">>, Self, undefined)} end)
     || _ <- lists:seq(1, 5)
    ],
    Results = [
        receive
            {got, R} -> R
        after 30000 -> timeout
        end
     || _ <- lists:seq(1, 5)
    ],
    ?assertEqual(lists:duplicate(5, {ok, Id}), Results),
    ?assertEqual(1, length([I || #{id := I} <- erllama:list_models(), I =:= Id])),
    [#{holders := 1}] = erllama_server_models:ps(),
    ok = erllama_server_models:release(Id, self()).

keep_alive_unloads_idle_model(Config) ->
    Id = add_model(Config, <<"gamma">>, #{keep_alive => 200}),
    #{status := done} = wait_done(run(<<"gamma">>, #{})),
    ok = wait(fun() -> not loaded(Id) end),
    ?assertEqual([], erllama_server_models:ps()).

keep_alive_zero_and_infinity(Config) ->
    Zero = add_model(Config, <<"zero">>),
    #{status := done} = wait_done(run(<<"zero">>, #{keep_alive => 0})),
    ok = wait(fun() -> not loaded(Zero) end),
    Forever = add_model(Config, <<"forever">>, #{keep_alive => 1}),
    #{status := done} = wait_done(run(<<"forever">>, #{keep_alive => infinity})),
    timer:sleep(300),
    ?assert(loaded(Forever)).

%% A model in use is not unloaded when its keep-alive runs out.
held_model_outlives_keep_alive(Config) ->
    Id = add_model(Config, <<"held">>, #{keep_alive => 50, step_delay_ms => 20}),
    Run = run(<<"held">>, #{opts => #{response_tokens => 40}}),
    timer:sleep(400),
    ?assert(loaded(Id)),
    ?assertMatch(#{status := done}, wait_done(Run)),
    ok = wait(fun() -> not loaded(Id) end).

evictor_unloads_least_recently_used(Config) ->
    Old = add_model(Config, <<"old">>),
    New = add_model(Config, <<"new">>),
    {ok, Old} = erllama_server_api:load(<<"old">>, #{}),
    timer:sleep(20),
    {ok, New} = erllama_server_api:load(<<"new">>, #{}),
    ?assertEqual({unloaded, Old}, erllama_server_models:evict_one()),
    ?assertNot(loaded(Old)),
    ?assert(loaded(New)),
    %% A held model is never evicted.
    {ok, New} = erllama_server_models:acquire(<<"new">>, self(), undefined),
    ?assertEqual(none, erllama_server_models:evict_one()),
    ok = erllama_server_models:release(New, self()).

unload_then_run_reloads(Config) ->
    Id = add_model(Config, <<"again">>),
    #{status := done} = wait_done(run(<<"again">>, #{})),
    ok = erllama_server_api:unload(<<"again">>),
    ?assertMatch(#{status := done}, wait_done(run(<<"again">>, #{}))),
    ?assert(loaded(Id)).

two_models_at_once(Config) ->
    A = add_model(Config, <<"one">>, #{step_delay_ms => 10}),
    B = add_model(Config, <<"two">>, #{step_delay_ms => 10}),
    Runs = [
        run(M, #{opts => #{response_tokens => 20}})
     || M <- [<<"one">>, <<"two">>, <<"one">>, <<"two">>]
    ],
    [?assertMatch(#{status := done}, wait_done(R)) || R <- Runs],
    ?assert(loaded(A) andalso loaded(B)).

load_failure_reaches_every_waiter(Config) ->
    %% A registry entry whose file is not a model llama.cpp can load.
    Path = filename:join(?config(dir, Config), "broken.gguf"),
    ok = file:write_file(Path, erllama_server_fake_registry:gguf(4096)),
    {ok, #{name := Id}} = erllama_server_api:register_model(Path, #{name => <<"broken">>}),
    Self = self(),
    [
        spawn(fun() -> Self ! {got, erllama_server_models:acquire(Id, Self, undefined)} end)
     || _ <- lists:seq(1, 3)
    ],
    Results = [
        receive
            {got, R} -> R
        after 60000 -> timeout
        end
     || _ <- lists:seq(1, 3)
    ],
    ?assert(lists:all(fun(R) -> element(1, R) =:= error end, Results)),
    Run = run(<<"broken">>, #{}),
    ?assertMatch(#{status := error}, wait_done(Run)).
