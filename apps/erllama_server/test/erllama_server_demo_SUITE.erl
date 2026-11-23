%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% The walkthrough of guides/server/demo.md, deterministic: one release,
%% one model (the stub), several conversations, an agent session with a
%% slow and a failing tool (scripted), a client that leaves and comes
%% back, the runtime view, one run cancelled, then a restart and a
%% conversation resumed. scripts/demo.sh runs the same story on a real
%% model through the CLI.
-module(erllama_server_demo_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_testcase/2, end_per_testcase/2]).
-export([walkthrough/1]).

all() -> [walkthrough].

init_per_testcase(TC, Config) ->
    Base = filename:join(?config(priv_dir, Config), atom_to_list(TC)),
    Ws = filename:join(Base, "ws"),
    ok = filelib:ensure_path(Ws),
    ok = application:set_env(erllama_server, data_dir, filename:join(Base, "data")),
    ok = application:set_env(erllama_server, keep_alive_ms, 60000),
    {ok, _} = application:ensure_all_started(erllama_server),
    [{ws, list_to_binary(Ws)}, {dir, Base} | Config].

end_per_testcase(_TC, _Config) ->
    _ = application:stop(erllama_server),
    [erllama:unload(Id) || #{id := Id} <- erllama:list_models()],
    application:unset_env(erllama_server, data_dir),
    application:unset_env(erllama_server, keep_alive_ms),
    ok.

walkthrough(Config) ->
    %% 1. One release, one model (four sequences; a run takes about 5 s).
    M = add_model(Config, <<"demo">>, #{step_delay_ms => 10}),

    %% 2. Independent conversations, each with its own run.
    {A, ConvA} = run(M, undefined, <<"tell a long story">>, 500),
    {B, _ConvB} = run(M, undefined, <<"tell another">>, 500),
    ok = wait(fun() -> phase(A) =:= generating andalso phase(B) =:= generating end),

    %% 3. An agent session whose turn runs a slow command, then a
    %%    failing one.
    Agent = agent(Config, M, [
        #{calls => [{<<"s1">>, <<"shell">>, #{<<"command">> => <<"sleep 2; echo slow">>}}]},
        #{calls => [{<<"f1">>, <<"shell">>, #{<<"command">> => <<"echo broken >&2; exit 3">>}}]},
        #{text => <<"the second command failed with exit 3">>}
    ]),
    ok = erllama_server_api:agent_send(Agent, <<"run the checks">>),
    ok = wait(fun() -> running_tools(Agent) =:= [<<"shell">>] end),

    %% 4. Another conversation goes on meanwhile.
    T0 = erlang:monotonic_time(millisecond),
    {C, _} = run(M, undefined, <<"a quick question">>, 3),
    ?assertMatch(#{status := done}, wait_done(C)),
    ?assert(erlang:monotonic_time(millisecond) - T0 < 1000),

    %% 5. A client follows run A, goes away, and another picks up where
    %%    it stopped: no event lost, none repeated.
    Self = self(),
    First = spawn(fun() -> follower(Self, A, 0, 20) end),
    Seen =
        receive
            {seen, First, S0} -> S0
        after 10000 -> ct:fail(no_events)
        end,
    exit(First, kill),
    Next = lists:last(Seen) + 1,
    Second = spawn(fun() -> follower(Self, A, Next, 20) end),
    Seen2 =
        receive
            {seen, Second, S1} -> S1
        after 10000 -> ct:fail(no_events)
        end,
    ?assertEqual(lists:seq(Next, Next + 19), Seen2),

    %% 6. What runs, what waits, what the agent does, the cache.
    {ok, Status} = erllama_server_api:status(),
    ct:log("~ts", [erllama_server_cli_status:format_status(Status)]),
    ?assertMatch(#{runs := #{generating := 2}}, Status),
    ?assertMatch([#{run_id := Agent, tools := 1}], maps:get(agents, Status)),
    ?assertMatch(#{misses := _}, maps:get(cache, Status)),

    %% 7. Cancel B: A goes on to its end.
    ok = erllama_server_api:cancel(B),
    ?assertMatch(#{status := cancelled}, wait_done(B)),
    ?assertMatch(#{status := done, finish_reason := length}, wait_done(A)),
    ok = wait(fun() -> agent_state(Agent) =:= idle end, 600),
    {ok, Failures} = erllama_server_api:failures(10),
    ct:log("failures: ~p", [Failures]),

    %% 8. Restart the release: conversation A is still there and takes
    %%    a new turn on top of its history.
    ok = application:stop(erllama_server),
    {ok, _} = application:ensure_all_started(erllama_server),
    {ok, #{messages := History}} = erllama_server_api:conversation(ConvA),
    ?assertMatch([#{role := <<"user">>}, #{role := <<"assistant">>}], History),
    {Again, ConvA} = run(M, ConvA, <<"go on">>, 3),
    ?assertMatch(#{status := done}, wait_done(Again)),
    {ok, #{messages := After}} = erllama_server_api:conversation(ConvA),
    ?assertEqual(4, length(After)).

%% -----------------------------------------------------------------------------

follower(Parent, Run, Cursor, N) ->
    {ok, #{replay := Replay}} = erllama_server_api:attach(Run, self(), Cursor),
    Seqs = collect(Run, [Seq || #{seq := Seq} <- Replay], N),
    Parent ! {seen, self(), lists:sublist(Seqs, N)},
    receive
        stop -> ok
    end.

collect(_Run, Seqs, N) when length(Seqs) >= N ->
    Seqs;
collect(Run, Seqs, N) ->
    receive
        {erllama_server_run, Run, #{seq := Seq}} -> collect(Run, Seqs ++ [Seq], N)
    after 10000 -> Seqs
    end.

add_model(Config, Name, Extra) ->
    Path = filename:join(?config(dir, Config), <<Name/binary, ".gguf">>),
    ok = file:write_file(Path, erllama_server_fake_registry:gguf(4096)),
    {ok, #{name := Id}} = erllama_server_api:register_model(
        Path, maps:merge(#{name => Name, backend => erllama_model_stub}, Extra)
    ),
    Id.

run(Model, Conv, Text, Tokens) ->
    Spec0 = #{
        model => Model,
        messages => [#{role => <<"user">>, content => Text}],
        opts => #{response_tokens => Tokens}
    },
    Spec =
        case Conv of
            undefined -> Spec0;
            _ -> Spec0#{conversation_id => Conv}
        end,
    {ok, #{run_id := Run, conversation_id := C}} = erllama_server_api:start_run(Spec),
    {Run, C}.

agent(Config, Model, Script) ->
    {ok, #{agent_id := Id}} = erllama_server_api:agent_start(#{
        model => Model,
        workspace => ?config(ws, Config),
        generator => erllama_server_agent_script,
        policy => yes
    }),
    ok = erllama_server_agent_script:set(Id, Script),
    Id.

phase(Run) ->
    case erllama_server_api:inspect(Run) of
        {ok, #{phase := P}} -> P;
        _ -> undefined
    end.

agent_state(Id) ->
    case erllama_server_api:inspect(Id) of
        {ok, #{state := S}} -> S;
        _ -> undefined
    end.

running_tools(Id) ->
    {ok, #{tools := Tools}} = erllama_server_api:inspect(Id),
    [N || #{name := N, status := running} <- Tools].

wait_done(Run) ->
    ok = wait(
        fun() ->
            case erllama_server_api:result(Run) of
                {ok, #{status := _}} -> true;
                _ -> false
            end
        end,
        1500
    ),
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
