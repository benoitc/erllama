%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
-module(erllama_tp_rank_role_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").
-behaviour(gen_server).

-export([all/0, init_per_testcase/2, end_per_testcase/2]).
-export([
    configured_ranks/1,
    coordinator_starts_late/1,
    different_path/1,
    different_options/1,
    different_world/1,
    different_version/1,
    stale_generation/1,
    failed_load/1
]).
-export([start_link/1, load/3, init/1, handle_call/3, handle_cast/2]).

all() ->
    [
        configured_ranks,
        coordinator_starts_late,
        different_path,
        different_options,
        different_world,
        different_version,
        stale_generation,
        failed_load
    ].

init_per_testcase(Case, Config) ->
    Dir = ?config(priv_dir, Config),
    Path = filename:join(Dir, "rank_model.gguf"),
    ok = file:write_file(Path, <<"rank model fixture">>),
    {ok, S, Node} = peer_node("serve", []),
    {ok, C} = peer:call(S, erllama_tp_coordinator, start, [
        erllama_tp_coordinator, {<<"g">>, 1}, 2, 10000
    ]),
    Base = #{
        group_id => <<"g">>,
        generation => 1,
        world => 2,
        gpu => 0,
        coordinator => Node,
        model_path => Path,
        model_opts => #{n_gpu_layers => 1},
        context_opts => #{n_ctx => 128}
    },
    R1 =
        case Case of
            different_path ->
                Other = Path ++ ".other",
                {ok, _} = file:copy(Path, Other),
                Base#{model_path => Other};
            different_options ->
                Base#{context_opts => #{n_ctx => 256}};
            different_world ->
                Base#{world => 3};
            stale_generation ->
                Base#{generation => 2};
            _ ->
                Base
        end,
    {ok, P0, _} = rank_peer(Dir, Base#{rank => 0}, false),
    {ok, P1, _} = rank_peer(Dir, R1#{rank => 1}, Case =:= failed_load),
    [{serving, S}, {coord, C}, {ranks, [P0, P1]}, {base, Base} | Config].

end_per_testcase(_, Config) ->
    [catch peer:stop(P) || P <- ?config(ranks, Config) ++ [?config(serving, Config)]],
    ok.

rank_peer(Dir, C, Fail) ->
    Name = "rank" ++ integer_to_list(maps:get(rank, C)),
    File = filename:join(Dir, Name ++ ".config"),
    Settings = [{erllama, [{tp_rank, C}, {tp_rank_driver, ?MODULE}, {fake_load_error, Fail}]}],
    ok = file:write_file(File, io_lib:format("~p.~n", [Settings])),
    peer_node(Name, ["-config", File]).

peer_node(Prefix, Args) ->
    peer:start(#{
        name => peer:random_name(Prefix),
        host => "127.0.0.1",
        longnames => true,
        connection => standard_io,
        args => [
            "+S",
            "2",
            "-pa",
            filename:join(code:lib_dir(erllama), "ebin"),
            filename:dirname(code:which(?MODULE))
            | Args
        ]
    }).

boot(P) ->
    {ok, _} = peer:call(P, application, ensure_all_started, [erllama], 20000).

coordinator_starts_late(Config) ->
    S = ?config(serving, Config),
    ok = peer:call(S, gen_server, stop, [?config(coord, Config)]),
    [P0, P1] = ?config(ranks, Config),
    boot(P0),
    boot(P1),
    #{state := forming} = status(P0),
    {ok, _} = peer:call(S, erllama_tp_coordinator, start, [
        erllama_tp_coordinator, {<<"g">>, 1}, 2, 10000
    ]),
    [#{state := ready} = await(P, ready) || P <- [P0, P1]],
    ok.

configured_ranks(Config) ->
    [P0, P1] = ?config(ranks, Config),
    boot(P0),
    ?assertMatch(
        #{state := forming, group_id := <<"g">>, world := 2, generation := 1, rank := 0}, status(P0)
    ),
    boot(P1),
    Base = ?config(base, Config),
    Load = {maps:get(model_path, Base), {maps:get(model_opts, Base), maps:get(context_opts, Base)}},
    [
        begin
            ?assertMatch(#{state := ready}, await(P, ready)),
            ?assertEqual(Load, peer:call(P, sys, get_state, [?MODULE]))
        end
     || P <- [P0, P1]
    ],
    ok = peer:stop(P1),
    ?assertMatch(#{state := lost, reason := {tp_rank_lost, 1, _}}, await(P0, lost)),
    ?assertEqual(undefined, peer:call(P0, erlang, whereis, [?MODULE])).

different_path(Config) -> mismatch(Config).
different_options(Config) -> mismatch(Config).
different_world(Config) -> mismatch(Config).
different_version(Config) ->
    [_, P1] = ?config(ranks, Config),
    {ok, [{application, erllama, Spec}]} = file:consult(
        filename:join([code:lib_dir(erllama), "ebin", "erllama.app"])
    ),
    ok = peer:call(P1, application, load, [
        {application, erllama, lists:keyreplace(vsn, 1, Spec, {vsn, "other"})}
    ]),
    mismatch(Config).

mismatch(Config) ->
    [P0, P1] = ?config(ranks, Config),
    boot(P0),
    boot(P1),
    [
        begin
            ?assertMatch(#{state := lost, reason := {incompatible_ranks, [1]}}, await(P, lost)),
            ?assertEqual(undefined, peer:call(P, erlang, whereis, [?MODULE]))
        end
     || P <- [P0, P1]
    ].

stale_generation(Config) ->
    [_, P1] = ?config(ranks, Config),
    boot(P1),
    ?assertMatch(#{state := lost, reason := wrong_generation}, await(P1, lost)).

failed_load(Config) ->
    [P0, P1] = ?config(ranks, Config),
    boot(P0),
    boot(P1),
    [?assertMatch(#{state := lost, reason := load_failed}, await(P, lost)) || P <- [P0, P1]].

status(P) -> peer:call(P, erllama_tp_rank_role, status, []).
await(P, State) -> await(P, State, 500).
await(P, State, 0) ->
    ct:fail({expected, State, status(P)});
await(P, State, N) ->
    case status(P) of
        #{state := State} = S ->
            S;
        _ ->
            timer:sleep(20),
            await(P, State, N - 1)
    end.

%% Exercise the native bootstrap bridge, replacing GPU allocation and model load only.
start_link(C) -> gen_server:start_link({local, ?MODULE}, ?MODULE, C, []).
load(P, Path, Opts) -> gen_server:call(P, {load, Path, Opts}).
init(#{join := {call, Coord}, rank := R, world := W, timeout := Ms}) ->
    Join = erllama_tp_coordinator:join_fun(Coord, R),
    {ok, [<<0, 1>>]} = erllama_tp:bootstrap_selftest(W, <<R>>, 1, Join, Ms),
    {ok, undefined}.
handle_call({load, Path, Opts}, _, _) ->
    case application:get_env(erllama, fake_load_error, false) of
        true -> {reply, {error, load_failed}, undefined};
        false -> {reply, ok, {Path, Opts}}
    end.
handle_cast(_, S) -> {noreply, S}.
