%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Several conversations on shared models: what an operator sees
%% (status, inspect, recent failures), that a slow or failing piece of
%% work stays with its owner, and what keeps working when a model, a
%% tool or a session process fails. Stub models and scripted agents,
%% so every case is repeatable.
-module(erllama_server_ops_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_testcase/2, end_per_testcase/2]).
-export([
    status_shows_running_and_waiting/1,
    slow_tool_leaves_other_conversations_alone/1,
    cancel_ends_only_its_run/1,
    tool_timeout_is_recorded/1,
    tool_crash_stays_with_its_call/1,
    model_crash_stays_with_its_model/1,
    session_crash_stays_with_its_session/1
]).

all() ->
    [
        status_shows_running_and_waiting,
        slow_tool_leaves_other_conversations_alone,
        cancel_ends_only_its_run,
        tool_timeout_is_recorded,
        tool_crash_stays_with_its_call,
        model_crash_stays_with_its_model,
        session_crash_stays_with_its_session
    ].

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

%% -----------------------------------------------------------------------------

%% One sequence, two runs: the second waits, and status, inspect and
%% the CLI say so.
status_shows_running_and_waiting(Config) ->
    M = add_model(Config, <<"single">>, #{step_delay_ms => 20, parallel => 1}),
    Long = run(M, #{opts => #{response_tokens => 2000}}),
    ok = wait(fun() -> phase(Long) =:= generating end),
    Waiting = run(M, #{conversation_id => none}),
    ok = wait(fun() -> phase(Waiting) =:= waiting end),
    %% The model counts it once its next step has run.
    ok = wait(fun() -> model_waiting(M) =:= 1 end),
    {ok, S} = erllama_server_api:status(),
    ?assertMatch(#{runs := #{generating := 1, waiting := 1}}, S),
    ?assertMatch(
        [#{id := M, waiting := 1, n_seq_max := 1, available_seqs := 0}], maps:get(models, S)
    ),
    ?assertMatch({ok, #{phase := waiting, status := running}}, erllama_server_api:inspect(Waiting)),
    Text = iolist_to_binary(erllama_server_cli_status:format_status(S)),
    ct:log("~ts", [Text]),
    ?assertNotEqual(nomatch, binary:match(Text, <<"1 waiting for one">>)),
    ?assertNotEqual(nomatch, binary:match(Text, <<"1 waiting for a sequence">>)),
    ok = erllama_server_api:cancel(Waiting),
    ok = erllama_server_api:cancel(Long).

%% A session waits on a slow tool; another conversation on the same
%% model runs a turn meanwhile, and inspect shows the tool running.
slow_tool_leaves_other_conversations_alone(Config) ->
    M = add_model(Config, <<"shared">>, #{}),
    Agent = agent(Config, M, [
        #{calls => [{<<"s1">>, <<"shell">>, #{<<"command">> => <<"sleep 3; echo slept">>}}]},
        #{text => <<"done waiting">>}
    ]),
    ok = erllama_server_api:agent_send(Agent, <<"wait for it">>),
    ok = wait(fun() -> running_tools(Agent) =:= [<<"shell">>] end),
    T0 = erlang:monotonic_time(millisecond),
    Other = run(M, #{}),
    ?assertMatch(#{status := done}, wait_done(Other)),
    Took = erlang:monotonic_time(millisecond) - T0,
    ct:log("the other conversation's turn took ~b ms while the tool ran", [Took]),
    ?assert(Took < 2000),
    ?assertEqual([<<"shell">>], running_tools(Agent)),
    {ok, S} = erllama_server_api:status(),
    ?assertMatch([#{run_id := Agent, tools := 1}], maps:get(agents, S)),
    ok = wait(fun() -> agent_state(Agent) =:= idle end, 600),
    ok = erllama_server_api:agent_stop(Agent).

%% Cancelling one run ends that run; the others go on to the end.
cancel_ends_only_its_run(Config) ->
    M = add_model(Config, <<"three">>, #{step_delay_ms => 5, parallel => 3}),
    [A, B, C] = [
        run(M, #{conversation_id => none, opts => #{response_tokens => 80}})
     || _ <- [1, 2, 3]
    ],
    ok = wait(fun() -> lists:all(fun(R) -> phase(R) =:= generating end, [A, B, C]) end),
    ok = erllama_server_api:cancel(B),
    ?assertMatch(#{status := cancelled}, wait_done(B)),
    ?assertMatch(#{status := done, finish_reason := length}, wait_done(A)),
    ?assertMatch(#{status := done, finish_reason := length}, wait_done(C)).

%% A shell call that outlives its timeout is killed; the turn goes on
%% and the failure is recorded with what happened.
tool_timeout_is_recorded(Config) ->
    M = add_model(Config, <<"timeouts">>, #{}),
    Agent = agent(Config, M, [
        #{
            calls => [
                {<<"t1">>, <<"shell">>, #{<<"command">> => <<"sleep 5">>, <<"timeout_ms">> => 300}}
            ]
        },
        #{text => <<"it timed out">>}
    ]),
    ok = erllama_server_api:agent_send(Agent, <<"go">>),
    ok = wait(fun() -> agent_state(Agent) =:= idle end, 400),
    ?assertMatch(
        [#{kind := tool_timed_out, subject := Agent, outcome := _}],
        [F || #{kind := tool_timed_out} = F <- failures()]
    ),
    ok = erllama_server_api:agent_stop(Agent).

%% The process running a tool call dies: that call gets an error
%% result, the turn goes on to its end, and the failure is recorded.
tool_crash_stays_with_its_call(Config) ->
    M = add_model(Config, <<"crashes">>, #{}),
    Agent = agent(Config, M, [
        #{calls => [{<<"c1">>, <<"shell">>, #{<<"command">> => <<"sleep 3">>}}]},
        #{text => <<"the tool crashed">>}
    ]),
    ok = erllama_server_api:agent_send(Agent, <<"go">>),
    ok = wait(fun() -> running_tools(Agent) =:= [<<"shell">>] end),
    {ok, #{pid := Pid}} = erllama_server_runs:lookup(Agent),
    [Worker] = tool_workers(Pid),
    exit(Worker, kill),
    ok = wait(fun() -> agent_state(Agent) =:= idle end, 400),
    ?assertMatch(
        [#{subject := Agent, reason := <<"#{reason => killed", _/binary>>}],
        [F || #{kind := tool_crashed} = F <- failures()]
    ),
    ok = erllama_server_api:agent_stop(Agent).

%% The session's tool workers: linked processes running a tool.
tool_workers(Session) ->
    {links, Links} = erlang:process_info(Session, links),
    [
        P
     || P <- Links,
        is_pid(P),
        {current_stacktrace, Stack} <- [erlang:process_info(P, current_stacktrace)],
        lists:any(fun({Mod, _, _, _}) -> Mod =:= erllama_server_tools end, Stack)
    ].

%% A model process that dies ends the runs on it, with an error; a run
%% on another model goes on; the next request loads the model again.
model_crash_stays_with_its_model(Config) ->
    A = add_model(Config, <<"model-a">>, #{step_delay_ms => 20}),
    B = add_model(Config, <<"model-b">>, #{step_delay_ms => 5}),
    RunA = run(A, #{opts => #{response_tokens => 2000}}),
    RunB = run(B, #{opts => #{response_tokens => 200}}),
    ok = wait(fun() -> phase(RunA) =:= generating andalso phase(RunB) =:= generating end),
    {ok, PidA} = erllama:whereis(A),
    exit(PidA, kill),
    ?assertMatch(#{status := error, error := {model_down, killed}}, wait_done(RunA)),
    ?assertMatch(#{status := done}, wait_done(RunB)),
    ?assertMatch([#{subject := A} | _], [F || #{kind := model_down} = F <- failures()]),
    ?assertMatch(#{status := done}, wait_done(run(A, #{}))).

%% A session process that dies is recorded; its conversation is freed
%% and can be resumed; another session goes on.
session_crash_stays_with_its_session(Config) ->
    M = add_model(Config, <<"sessions">>, #{}),
    Doomed = agent(Config, M, [#{text => <<"one">>}]),
    Survivor = agent(Config, M, [#{text => <<"still here">>}, #{text => <<"and again">>}]),
    {ok, #{conversation_id := Conv}} = erllama_server_api:inspect(Doomed),
    {ok, #{pid := Pid}} = erllama_server_runs:lookup(Doomed),
    exit(Pid, kill),
    ok = wait(fun() -> erllama_server_runs:lookup(Doomed) =:= {error, not_found} end),
    ?assertMatch([#{subject := Doomed}], [F || #{kind := agent_down} = F <- failures()]),
    ok = erllama_server_api:agent_send(Survivor, <<"hello">>),
    ok = wait(fun() -> agent_state(Survivor) =:= idle andalso turns(Survivor) =:= 1 end),
    {ok, #{agent_id := Resumed}} = erllama_server_api:agent_start(#{
        model => M, conversation_id => Conv, generator => erllama_server_agent_script
    }),
    ok = erllama_server_agent_script:set(Resumed, [#{text => <<"back">>}]),
    ok = erllama_server_api:agent_send(Resumed, <<"are you there">>),
    ok = wait(fun() -> agent_state(Resumed) =:= idle andalso turns(Resumed) =:= 1 end),
    [ok = erllama_server_api:agent_stop(Id) || Id <- [Survivor, Resumed]].

%% -----------------------------------------------------------------------------

add_model(Config, Name, Extra) ->
    Path = filename:join(?config(dir, Config), <<Name/binary, ".gguf">>),
    ok = file:write_file(Path, erllama_server_fake_registry:gguf(4096)),
    {ok, #{name := Id}} = erllama_server_api:register_model(
        Path, maps:merge(#{name => Name, backend => erllama_model_stub, step_delay_ms => 1}, Extra)
    ),
    Id.

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

model_waiting(M) ->
    {ok, #{models := Ms}} = erllama_server_api:status(),
    hd([W || #{id := Id, waiting := W} <- Ms, Id =:= M] ++ [none]).

agent_state(Id) ->
    case erllama_server_api:inspect(Id) of
        {ok, #{state := S}} -> S;
        _ -> undefined
    end.

running_tools(Id) ->
    {ok, #{tools := Tools}} = erllama_server_api:inspect(Id),
    [N || #{name := N, status := running} <- Tools].

turns(Id) ->
    length(erllama_server_agent_script:rounds(Id)).

failures() ->
    {ok, Fs} = erllama_server_api:failures(50),
    Fs.

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
