%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% Agent sessions, with a scripted model (erllama_server_agent_script)
%% so every case is deterministic: answers without tools, reads running
%% together and writes in call order, results in call order, approvals,
%% unknown tools and paths outside the workspace, budgets, queued input,
%% interruption killing a running shell and its descendants, background
%% jobs, and stopping a session. One gated case drives a real model.
%% @end
-module(erllama_server_agent_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_testcase/2, end_per_testcase/2]).
-export([
    answers_without_tools/1,
    reads_together_writes_in_order/1,
    approval_allow_deny_always/1,
    unknown_tool_and_outside_path/1,
    round_budget/1,
    queued_input_runs_next/1,
    interrupt_kills_the_shell/1,
    background_job/1,
    stop_frees_the_conversation/1,
    generator_crash_fails_the_turn/1,
    stop_kills_background_jobs/1,
    time_budget_interrupts/1,
    token_budget_caps_generation/1,
    real_model_reads_a_file/1
]).

all() ->
    [
        answers_without_tools,
        reads_together_writes_in_order,
        approval_allow_deny_always,
        unknown_tool_and_outside_path,
        round_budget,
        queued_input_runs_next,
        interrupt_kills_the_shell,
        background_job,
        stop_frees_the_conversation,
        generator_crash_fails_the_turn,
        stop_kills_background_jobs,
        time_budget_interrupts,
        token_budget_caps_generation,
        real_model_reads_a_file
    ].

init_per_testcase(TC, Config) ->
    Base = filename:join(?config(priv_dir, Config), atom_to_list(TC)),
    Ws = filename:join(Base, "ws"),
    ok = filelib:ensure_path(Ws),
    ok = application:set_env(erllama_server, data_dir, filename:join(Base, "data")),
    {ok, _} = application:ensure_all_started(erllama_server),
    {ok, _} = erllama:load_model(<<"stub">>, #{backend => erllama_model_stub}),
    [{ws, list_to_binary(Ws)} | Config].

end_per_testcase(_TC, _Config) ->
    _ = application:stop(erllama_server),
    [erllama:unload(Id) || #{id := Id} <- erllama:list_models()],
    application:unset_env(erllama_server, data_dir),
    ok.

%% A session on the scripted model.
start(Config, Script) ->
    start(Config, Script, #{}).

start(Config, Script, Extra) ->
    Spec = maps:merge(
        #{
            model => <<"stub">>,
            workspace => ?config(ws, Config),
            generator => erllama_server_agent_script
        },
        Extra
    ),
    {ok, #{agent_id := Id, conversation_id := Conv}} = erllama_server_api:agent_start(Spec),
    ok = erllama_server_agent_script:set(Id, Script),
    {Id, Conv}.

%% Send input and collect the session's events up to the turn's end.
turn(Id, Text) ->
    ok = erllama_server_api:agent_send(Id, Text),
    collect_turn(Id).

collect_turn(Id) ->
    {ok, #{replay := Replay}} = erllama_server_api:attach(Id, self(), cursor(Id)),
    Events = until_turn_done(Id, Replay),
    ok = erllama_server_api:detach(Id, self()),
    flush(Id),
    put({cursor, Id}, maps:get(seq, lists:last(Events)) + 1),
    Events.

cursor(Id) ->
    case get({cursor, Id}) of
        undefined -> 0;
        C -> C
    end.

until_turn_done(Id, Acc) ->
    case [E || #{type := turn_done} = E <- Acc] of
        [_ | _] ->
            {Before, [Done | _]} = lists:splitwith(fun(#{type := T}) -> T =/= turn_done end, Acc),
            Before ++ [Done];
        [] ->
            receive
                {erllama_server_run, Id, #{} = E} -> until_turn_done(Id, Acc ++ [E])
            after 30000 -> ct:fail({no_turn_done, Acc})
            end
    end.

flush(Id) ->
    receive
        {erllama_server_run, Id, _} -> flush(Id)
    after 0 -> ok
    end.

types(Events) -> [T || #{type := T} <- Events].

done_reason(Events) ->
    #{data := #{reason := R}} = lists:last(Events),
    R.

conv_messages(Conv) ->
    {ok, #{messages := Msgs}} = erllama_server_api:conversation(Conv),
    Msgs.

%% The tool results the model was shown in its last round.
last_tool_results(Id) ->
    Last = lists:last(erllama_server_agent_script:rounds(Id)),
    [{N, C} || #{role := <<"tool">>, name := N, content := C} <- Last].

wait(Fun) -> wait(Fun, 250).
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

%% -----------------------------------------------------------------------------

answers_without_tools(Config) ->
    {Id, Conv} = start(Config, [#{text => <<"hello there">>}]),
    Events = turn(Id, <<"hi">>),
    ?assertEqual(stop, done_reason(Events)),
    ?assertEqual(<<"hello there">>, iolist_to_binary([D || #{type := token, data := D} <- Events])),
    ?assertMatch(
        [
            #{role := <<"user">>, content := <<"hi">>},
            #{role := <<"assistant">>, content := <<"hello there">>}
        ],
        conv_messages(Conv)
    ).

%% Reads before a write run, the write waits for them, the read after
%% the write sees what it wrote, and results come back in call order.
reads_together_writes_in_order(Config) ->
    Ws = ?config(ws, Config),
    ok = file:write_file(filename:join(Ws, "a.txt"), <<"alpha\nbeta\n">>),
    Calls = [
        {<<"c1">>, <<"read_file">>, #{<<"path">> => <<"a.txt">>}},
        {<<"c2">>, <<"list_dir">>, #{}},
        {<<"c3">>, <<"write_file">>, #{<<"path">> => <<"new/b.txt">>, <<"content">> => <<"one\n">>}},
        {<<"c4">>, <<"read_file">>, #{<<"path">> => <<"new/b.txt">>}},
        {<<"c5">>, <<"edit_file">>, #{
            <<"path">> => <<"new/b.txt">>, <<"old">> => <<"one">>, <<"new">> => <<"two">>
        }},
        {<<"c6">>, <<"search">>, #{<<"pattern">> => <<"two|beta">>}}
    ],
    {Id, Conv} = start(Config, [#{calls => Calls}, #{text => <<"done">>}], #{policy => yes}),
    Events = turn(Id, <<"work">>),
    ?assertEqual(stop, done_reason(Events)),
    Results = last_tool_results(Id),
    ?assertEqual(
        [
            <<"read_file">>,
            <<"list_dir">>,
            <<"write_file">>,
            <<"read_file">>,
            <<"edit_file">>,
            <<"search">>
        ],
        [N || {N, _} <- Results]
    ),
    [{_, R1}, {_, R2}, {_, R3}, {_, R4}, {_, R5}, {_, R6}] = Results,
    ?assertNotEqual(nomatch, binary:match(R1, <<"1\talpha">>)),
    ?assertNotEqual(nomatch, binary:match(R2, <<"a.txt">>)),
    ?assertNotEqual(nomatch, binary:match(R3, <<"created">>)),
    ?assertNotEqual(nomatch, binary:match(R4, <<"1\tone">>)),
    ?assertNotEqual(nomatch, binary:match(R5, <<"replaced 1">>)),
    ?assertNotEqual(nomatch, binary:match(R6, <<"new/b.txt:1: two">>)),
    ?assertEqual({ok, <<"two\n">>}, file:read_file(filename:join([Ws, "new", "b.txt"]))),
    %% The write started only after the reads before it had finished.
    Seq = [
        {T, C}
     || #{type := T, data := #{call_id := C}} <- Events, T =:= tool_started orelse T =:= tool_result
    ],
    Index = fun(X) -> length(lists:takewhile(fun(Y) -> Y =/= X end, Seq)) end,
    ?assert(Index({tool_result, <<"c1">>}) < Index({tool_started, <<"c3">>})),
    ?assert(Index({tool_result, <<"c2">>}) < Index({tool_started, <<"c3">>})),
    ?assert(Index({tool_result, <<"c3">>}) < Index({tool_started, <<"c4">>})),
    %% The transcript holds the call and its results, in order.
    Roles = [R || #{role := R} <- conv_messages(Conv)],
    ?assertEqual(
        [<<"user">>, <<"assistant">>] ++ lists:duplicate(6, <<"tool">>) ++ [<<"assistant">>], Roles
    ).

approval_allow_deny_always(Config) ->
    Script = [
        #{
            calls => [
                {<<"w1">>, <<"write_file">>, #{<<"path">> => <<"x.txt">>, <<"content">> => <<"x">>}}
            ]
        },
        #{
            calls => [
                {<<"w2">>, <<"write_file">>, #{<<"path">> => <<"y.txt">>, <<"content">> => <<"y">>}}
            ]
        },
        #{
            calls => [
                {<<"w3">>, <<"write_file">>, #{<<"path">> => <<"z.txt">>, <<"content">> => <<"z">>}}
            ]
        },
        #{
            calls => [
                {<<"w4">>, <<"write_file">>, #{<<"path">> => <<"q.txt">>, <<"content">> => <<"q">>}}
            ]
        },
        #{text => <<"ok">>}
    ],
    {Id, _} = start(Config, Script),
    ok = erllama_server_api:agent_send(Id, <<"write files">>),
    Approve = fun(CallId, Decision) ->
        ok = wait(fun() -> lists:member(CallId, pending_requests(Id)) end),
        ok = erllama_server_api:agent_approve(Id, CallId, Decision)
    end,
    Approve(<<"w1">>, allow),
    Approve(<<"w2">>, deny),
    Approve(<<"w3">>, always),
    %% `always' covers write_file for the rest of the session: no request for w4.
    Events = collect_turn(Id),
    ?assertEqual(stop, done_reason(Events)),
    ?assertNot(
        lists:member(<<"w4">>, [C || #{type := approval_request, data := #{call_id := C}} <- Events])
    ),
    Ws = ?config(ws, Config),
    ?assert(filelib:is_file(filename:join(Ws, "x.txt"))),
    ?assertNot(filelib:is_file(filename:join(Ws, "y.txt"))),
    ?assert(filelib:is_file(filename:join(Ws, "z.txt"))),
    ?assert(filelib:is_file(filename:join(Ws, "q.txt"))),
    Shown = lists:append([
        [C || #{role := <<"tool">>, content := C} <- R]
     || R <- erllama_server_agent_script:rounds(Id)
    ]),
    ?assert(lists:any(fun(C) -> binary:match(C, <<"declined">>) =/= nomatch end, Shown)).

pending_requests(Id) ->
    {ok, #{replay := Replay}} = erllama_server_api:attach(Id, self(), 0),
    ok = erllama_server_api:detach(Id, self()),
    flush(Id),
    Asked = [C || #{type := approval_request, data := #{call_id := C}} <- Replay],
    Started = [C || #{type := tool_started, data := #{call_id := C}} <- Replay],
    Asked -- Started.

unknown_tool_and_outside_path(Config) ->
    Calls = [
        {<<"u1">>, <<"rm_rf">>, #{}},
        {<<"u2">>, <<"read_file">>, #{<<"path">> => <<"../../etc/passwd">>}},
        {<<"u3">>, <<"read_file">>, #{<<"path">> => <<"/etc/hosts">>}}
    ],
    {Id, _} = start(Config, [#{calls => Calls}, #{text => <<"ok">>}], #{policy => yes}),
    Events = turn(Id, <<"go">>),
    ?assertEqual(stop, done_reason(Events)),
    [{<<"rm_rf">>, R1}, {<<"read_file">>, R2}, {<<"read_file">>, R3}] = last_tool_results(Id),
    ?assertMatch(<<"Error: unknown tool", _/binary>>, R1),
    ?assertMatch(<<"Error: the path is outside the workspace">>, R2),
    ?assertMatch(<<"Error: the path is outside the workspace">>, R3).

round_budget(Config) ->
    Loop = #{calls => [{<<"l">>, <<"list_dir">>, #{}}]},
    {Id, _} = start(Config, lists:duplicate(10, Loop), #{
        policy => yes, budgets => #{max_rounds => 3}
    }),
    Events = turn(Id, <<"loop">>),
    ?assertEqual(#{budget => rounds}, done_reason(Events)),
    ?assertEqual(3, length([R || #{type := round} = R <- Events])).

queued_input_runs_next(Config) ->
    Ws = ?config(ws, Config),
    Slow = #{calls => [{<<"s">>, <<"shell">>, #{<<"command">> => <<"sleep 0.5">>}}]},
    {Id, Conv} = start(
        Config, [Slow, #{text => <<"first done">>}, #{text => <<"second done">>}], #{policy => yes}
    ),
    ok = erllama_server_api:agent_send(Id, <<"first">>),
    {queued, 1} = erllama_server_api:agent_send(Id, <<"second">>),
    First = collect_turn(Id),
    ?assertEqual(stop, done_reason(First)),
    Second = collect_turn(Id),
    ?assertEqual(stop, done_reason(Second)),
    ?assertMatch(
        #{data := #{input := <<"second">>}}, hd([E || #{type := turn_started} = E <- Second])
    ),
    ?assertEqual([<<"first">>, <<"second">>], [
        C
     || #{role := <<"user">>, content := C} <- conv_messages(Conv)
    ]),
    _ = Ws.

interrupt_kills_the_shell(Config) ->
    Marker = integer_to_binary(erlang:unique_integer([positive])),
    Cmd = <<"sh -c 'sleep 4", Marker/binary, "' & sleep 5", Marker/binary>>,
    {Id, Conv} = start(
        Config,
        [#{calls => [{<<"k">>, <<"shell">>, #{<<"command">> => Cmd}}]}, #{text => <<"after">>}],
        #{policy => yes}
    ),
    ok = erllama_server_api:agent_send(Id, <<"run it">>),
    Running = fun() ->
        string:trim(
            os:cmd("ps -A -o command | grep -c '^sleep [45]" ++ binary_to_list(Marker) ++ "'")
        )
    end,
    ok = wait(fun() -> Running() =:= "2" end),
    ok = erllama_server_api:agent_interrupt(Id),
    Events = collect_turn(Id),
    ?assertEqual(cancelled, done_reason(Events)),
    ok = wait(fun() -> Running() =:= "0" end),
    ?assertMatch(
        #{content := <<"Error: interrupted by the user">>}, lists:last(conv_messages(Conv))
    ),
    %% The session goes on.
    Next = turn(Id, <<"still there?">>),
    ?assertEqual(stop, done_reason(Next)).

background_job(Config) ->
    Script = [
        #{
            calls => [
                {<<"b1">>, <<"shell">>, #{
                    <<"command">> => <<"echo started; sleep 30">>, <<"background">> => true
                }}
            ]
        },
        #{text => <<"started it">>}
    ],
    {Id, _} = start(Config, Script, #{policy => yes}),
    _ = turn(Id, <<"start a job">>),
    [{<<"shell">>, Started}] = last_tool_results(Id),
    {match, [Job]} = re:run(Started, <<"(job_[0-9a-f]+)">>, [{capture, all_but_first, binary}]),
    ok = erllama_server_agent_script:set(Id, [
        #{
            calls => [
                {<<"b2">>, <<"shell_status">>, #{<<"job">> => Job}},
                {<<"b3">>, <<"shell_kill">>, #{<<"job">> => Job}}
            ]
        },
        #{text => <<"stopped">>}
    ]),
    %% Let the job print before asking about it.
    timer:sleep(300),
    _ = turn(Id, <<"check and stop it">>),
    [{<<"shell_status">>, Status}, {<<"shell_kill">>, Killed}] = lists:nthtail(
        1, last_tool_results(Id)
    ),
    ?assertMatch(<<"running\n", _/binary>>, Status),
    ?assertNotEqual(nomatch, binary:match(Status, <<"started">>)),
    ?assertMatch(<<"killed", _/binary>>, Killed).

stop_frees_the_conversation(Config) ->
    {Id, Conv} = start(Config, [#{text => <<"one">>}]),
    _ = turn(Id, <<"hi">>),
    ok = erllama_server_api:agent_stop(Id),
    {ok, #{replay := Replay}} = erllama_server_api:attach(Id, self(), 0),
    ?assertEqual(done, maps:get(type, lists:last(Replay))),
    ?assertMatch({error, _}, erllama_server_api:agent_send(Id, <<"more">>)),
    %% The conversation is free: a new session continues it.
    {Id2, Conv} = start(Config, [#{text => <<"two">>}], #{conversation_id => Conv}),
    _ = turn(Id2, <<"again">>),
    [First | _] = erllama_server_agent_script:rounds(Id2),
    ?assertEqual([<<"system">>, <<"user">>, <<"assistant">>, <<"user">>], [
        R
     || #{role := R} <- First
    ]).

%% A round that dies without reporting fails its turn; the session
%% stays usable.
generator_crash_fails_the_turn(Config) ->
    {Id, _} = start(Config, [{hang, <<"thinking">>}, #{text => <<"back">>}]),
    ok = erllama_server_api:agent_send(Id, <<"go">>),
    ok = wait(fun() -> is_pid(erllama_server_agent_script:generator(Id)) end),
    exit(erllama_server_agent_script:generator(Id), kill),
    Events = collect_turn(Id),
    ?assertEqual(#{error => {generator_exited, killed}}, done_reason(Events)),
    ?assertEqual(stop, done_reason(turn(Id, <<"again">>))).

%% Ending a session stops its background commands, whole process group
%% included, before they can do anything more.
stop_kills_background_jobs(Config) ->
    Ws = ?config(ws, Config),
    Late = filename:join(Ws, "late.txt"),
    Script = [
        #{
            calls => [
                {<<"j1">>, <<"shell">>, #{
                    <<"command">> =>
                        <<"(sleep 1; echo child > child.txt) & sleep 1; echo late > late.txt">>,
                    <<"background">> => true
                }}
            ]
        },
        #{text => <<"started">>}
    ],
    {Id, _} = start(Config, Script, #{policy => yes}),
    _ = turn(Id, <<"start it">>),
    [{<<"shell">>, Started}] = last_tool_results(Id),
    ?assertMatch(<<"started background job ", _/binary>>, Started),
    ok = erllama_server_api:agent_stop(Id),
    timer:sleep(2000),
    ct:log("files: ~p", [file:list_dir(Ws)]),
    ?assertNot(filelib:is_file(Late)),
    ?assertNot(filelib:is_file(filename:join(Ws, "child.txt"))).

%% The time budget ends a turn while it runs, not only between rounds.
time_budget_interrupts(Config) ->
    {Id, _} = start(Config, [{hang, <<"slow">>}], #{budgets => #{max_elapsed_ms => 50}}),
    T0 = erlang:monotonic_time(millisecond),
    Events = turn(Id, <<"go">>),
    ?assertEqual(#{budget => time}, done_reason(Events)),
    ?assert(erlang:monotonic_time(millisecond) - T0 < 1000),
    %% While a tool runs, too: the running command is stopped.
    {Id2, _} = start(
        Config,
        [#{calls => [{<<"t1">>, <<"shell">>, #{<<"command">> => <<"sleep 5">>}}]}],
        #{policy => yes, budgets => #{max_elapsed_ms => 300}}
    ),
    T1 = erlang:monotonic_time(millisecond),
    ?assertEqual(#{budget => time}, done_reason(turn(Id2, <<"go">>))),
    ?assert(erlang:monotonic_time(millisecond) - T1 < 2000).

%% The token budget bounds what the model may generate (the real
%% generator on erllama's stub backend).
token_budget_caps_generation(_Config) ->
    {ok, #{agent_id := Id}} = erllama_server_api:agent_start(#{
        model => <<"stub">>,
        workspace => ?config(ws, _Config),
        budgets => #{max_tokens => 1}
    }),
    Events = turn(Id, <<"tell me a long story">>),
    #{data := #{tokens := Tokens} = Done} = lists:last(Events),
    ?assert(Tokens =< 1),
    ?assertEqual(#{budget => tokens}, maps:get(reason, Done)).

%% Gated: a real tool-capable model reads a file to answer.
real_model_reads_a_file(Config) ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {skip, "set LLAMA_TEST_MODEL to a tool-capable GGUF to run the real-model case"};
        Path ->
            Ws = ?config(ws, Config),
            ok = file:write_file(
                filename:join(Ws, "secret.txt"), <<"The code word is PELICAN.\n">>
            ),
            {ok, Model} = erllama_server_api:load_model(Path, #{
                name => <<"agent_real">>, n_ctx => 8192
            }),
            {ok, #{agent_id := Id}} = erllama_server_api:agent_start(#{
                model => Model,
                workspace => Ws,
                policy => yes,
                %% A model with no tool template only writes text: keep its
                %% turn short (it then skips below).
                budgets => #{max_tokens => 256}
            }),
            Events = turn(
                Id,
                <<"What is the code word in secret.txt? Read the file and answer with the word only.">>
            ),
            Answer = iolist_to_binary([D || #{type := token, data := D} <- Events]),
            ct:log("events: ~p~nanswer: ~ts", [types(Events), Answer]),
            case lists:member(tool_started, types(Events)) of
                true ->
                    ?assertNotEqual(
                        nomatch, binary:match(string:uppercase(Answer), <<"PELICAN">>)
                    );
                false ->
                    {skip, "the model's template synthesizes no tool grammar"}
            end
    end.
