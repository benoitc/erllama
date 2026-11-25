%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% Durable conversations: they survive a restart of the release, with
%% their stored responses; fork, delete and export; and a session that
%% dies, or a store that crashes, in the middle of a tool call leaves
%% the call marked unresolved, never run again, with the transcript
%% still well formed for the next turn. Runs use erllama's stub
%% backend; sessions use the scripted model.
%% @end
-module(erllama_server_convs_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_testcase/2, end_per_testcase/2]).
-export([
    restart_keeps_conversations/1,
    fork_delete_export/1,
    stored_responses_survive_restart/1,
    dead_session_leaves_unresolved/1,
    store_crash_recovers/1,
    real_model_warm_after_restart/1
]).

all() ->
    [
        restart_keeps_conversations,
        fork_delete_export,
        stored_responses_survive_restart,
        dead_session_leaves_unresolved,
        store_crash_recovers,
        real_model_warm_after_restart
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
    _ = application:unset_env(erllama_server, agent_generator),
    _ = application:stop(erllama_server),
    [erllama:unload(Id) || #{id := Id} <- erllama:list_models()],
    application:unset_env(erllama_server, data_dir),
    ok.

restart() ->
    ok = application:stop(erllama_server),
    {ok, _} = application:ensure_all_started(erllama_server),
    ok.

%% One turn of a conversation on the stub; returns its conversation.
turn(Conv, Text) ->
    Spec0 = #{
        model => <<"stub">>,
        messages => [#{role => <<"user">>, content => Text}],
        opts => #{response_tokens => 3}
    },
    Spec =
        case Conv of
            undefined -> Spec0;
            _ -> Spec0#{conversation_id => Conv}
        end,
    {ok, #{run_id := Run, conversation_id := C}} = erllama_server_api:start_run(Spec),
    ok = wait(fun() ->
        case erllama_server_api:result(Run) of
            {ok, #{status := done}} -> true;
            _ -> false
        end
    end),
    C.

users(Conv) ->
    {ok, #{messages := Msgs}} = erllama_server_api:conversation(Conv),
    [C || #{role := <<"user">>, content := C} <- Msgs].

wait(Fun) -> wait(Fun, 500).
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

restart_keeps_conversations(_Config) ->
    Conv = turn(undefined, <<"first">>),
    _ = turn(Conv, <<"second">>),
    {ok, #{messages := Before}} = erllama_server_api:conversation(Conv),
    ok = restart(),
    {ok, #{messages := After, active_run := undefined, unresolved := []}} =
        erllama_server_api:conversation(Conv),
    ?assertEqual(Before, After),
    ?assertMatch({ok, [#{id := Conv}]}, erllama_server_api:conversations()),
    %% It goes on where it was.
    _ = turn(Conv, <<"third">>),
    ?assertEqual([<<"first">>, <<"second">>, <<"third">>], users(Conv)).

fork_delete_export(_Config) ->
    Conv = turn(undefined, <<"shared">>),
    {ok, #{conversation_id := Fork}} = erllama_server_api:conversation_fork(Conv),
    _ = turn(Fork, <<"only in the fork">>),
    ?assertEqual([<<"shared">>], users(Conv)),
    ?assertEqual([<<"shared">>, <<"only in the fork">>], users(Fork)),
    {ok, Json} = erllama_server_api:conversation_export(Fork, json),
    #{<<"id">> := Fork, <<"parent">> := Conv, <<"messages">> := [_, _, _, _]} = json:decode(Json),
    {ok, Md} = erllama_server_api:conversation_export(Fork, markdown),
    ?assertNotEqual(nomatch, binary:match(Md, <<"only in the fork">>)),
    %% Both survive a restart; a deleted one does not.
    ok = erllama_server_api:conversation_delete(Conv),
    ok = restart(),
    ?assertEqual({error, not_found}, erllama_server_api:conversation(Conv)),
    ?assertEqual([<<"shared">>, <<"only in the fork">>], users(Fork)).

stored_responses_survive_restart(_Config) ->
    Conv = turn(undefined, <<"FIRST">>),
    ok = erllama_server_responses:put(<<"resp_1">>, #{<<"id">> => <<"resp_1">>}, Conv),
    %% The snapshot behind it is not listed as a conversation.
    {ok, Listed} = erllama_server_api:conversations(),
    ?assertEqual([Conv], [Id || #{id := Id} <- Listed]),
    ok = restart(),
    ?assertMatch({ok, #{<<"id">> := <<"resp_1">>}, _}, erllama_server_responses:get(<<"resp_1">>)),
    {ok, A} = erllama_server_responses:continue(<<"resp_1">>),
    {ok, B} = erllama_server_responses:continue(<<"resp_1">>),
    _ = turn(A, <<"BRANCH_A">>),
    _ = turn(B, <<"BRANCH_B">>),
    ?assertEqual([<<"FIRST">>, <<"BRANCH_A">>], users(A)),
    ?assertEqual([<<"FIRST">>, <<"BRANCH_B">>], users(B)),
    ok = erllama_server_responses:delete(<<"resp_1">>),
    ok = restart(),
    ?assertEqual({error, not_found}, erllama_server_responses:get(<<"resp_1">>)).

%% A session killed while a command runs: the command is stopped, the
%% call is unresolved, and resuming the conversation (in the same
%% workspace, found from the conversation) does not run it again.
dead_session_leaves_unresolved(Config) ->
    Ws = ?config(ws, Config),
    {Id, Conv} = slow_call(Config),
    {ok, #{pid := Pid}} = erllama_server_runs:lookup(Id),
    exit(Pid, kill),
    ok = wait(fun() -> unresolved(Conv) =:= [<<"c1">>] end),
    check_repaired(Conv, Ws),
    resume_does_not_repeat(Conv, Ws).

%% The conversation store crashes while a command runs: it comes back
%% from its logs with the call unresolved.
store_crash_recovers(Config) ->
    Ws = ?config(ws, Config),
    {_Id, Conv} = slow_call(Config),
    exit(whereis(erllama_server_convs), kill),
    ok = wait(fun() ->
        is_pid(whereis(erllama_server_convs)) andalso
            (catch unresolved(Conv)) =:= [<<"c1">>]
    end),
    check_repaired(Conv, Ws),
    %% And again after a full restart: the repair was written down.
    ok = restart(),
    ?assertEqual([<<"c1">>], unresolved(Conv)),
    check_repaired(Conv, Ws).

slow_call(Config) ->
    ok = application:set_env(erllama_server, agent_generator, erllama_server_agent_script),
    Script = [
        #{
            calls => [
                {<<"c1">>, <<"shell">>, #{<<"command">> => <<"sleep 1; echo done > out.txt">>}}
            ]
        },
        #{text => <<"never reached">>}
    ],
    {ok, #{agent_id := Id, conversation_id := Conv}} = erllama_server_api:agent_start(#{
        model => <<"stub">>,
        workspace => ?config(ws, Config),
        policy => yes
    }),
    ok = erllama_server_agent_script:set(Id, Script),
    ok = erllama_server_api:agent_send(Id, <<"run it">>),
    ok = wait(fun() -> started(Id) end),
    {Id, Conv}.

started(Id) ->
    {ok, #{replay := Replay}} = erllama_server_api:attach(Id, self(), 0),
    ok = erllama_server_api:detach(Id, self()),
    flush(Id),
    lists:member(tool_started, [T || #{type := T} <- Replay]).

flush(Id) ->
    receive
        {erllama_server_run, Id, _} -> flush(Id)
    after 0 -> ok
    end.

unresolved(Conv) ->
    {ok, #{unresolved := U}} = erllama_server_api:conversation(Conv),
    U.

check_repaired(Conv, Ws) ->
    {ok, #{messages := Msgs, active_run := undefined}} = erllama_server_api:conversation(Conv),
    [#{role := <<"tool">>, tool_call_id := <<"c1">>, content := Content}] =
        [M || #{role := <<"tool">>} = M <- Msgs],
    ?assertMatch(<<"Error: the release stopped while this call was running", _/binary>>, Content),
    %% The command went with its owner.
    timer:sleep(1500),
    ?assertNot(filelib:is_file(filename:join(Ws, "out.txt"))).

resume_does_not_repeat(Conv, Ws) ->
    ok = application:set_env(erllama_server, agent_generator, erllama_server_agent_script),
    {ok, #{agent_id := Id2}} = erllama_server_api:agent_start(#{
        model => <<"stub">>,
        conversation_id => Conv,
        policy => yes
    }),
    {ok, #{workspace := Ws}} = erllama_server_api:conversation(Conv),
    ok = erllama_server_agent_script:set(Id2, [#{text => <<"noted">>}]),
    ok = erllama_server_api:agent_send(Id2, <<"what happened?">>),
    ok = wait(fun() -> length(erllama_server_agent_script:rounds(Id2)) =:= 1 end),
    [Shown] = erllama_server_agent_script:rounds(Id2),
    ?assertMatch(
        [#{tool_call_id := <<"c1">>, content := <<"Error: the release stopped", _/binary>>}],
        [M || #{role := <<"tool">>} = M <- Shown]
    ),
    timer:sleep(1500),
    ?assertNot(filelib:is_file(filename:join(Ws, "out.txt"))).

%% Gated: after a restart of the release (erllama, and its cache, keep
%% running), the next turn starts from the persisted finish key: the KV
%% is restored, not prefilled from scratch.
real_model_warm_after_restart(_Config) ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {skip, "set LLAMA_TEST_MODEL to a GGUF to run the real-model case"};
        Path ->
            {ok, Model} = erllama_server_api:load_model(Path, #{name => <<"conv_real">>}),
            Turn = fun(Conv, Text) ->
                Spec0 = #{
                    model => Model,
                    messages => [#{role => <<"user">>, content => Text}],
                    opts => #{response_tokens => 8}
                },
                Spec =
                    case Conv of
                        undefined -> Spec0;
                        _ -> Spec0#{conversation_id => Conv}
                    end,
                {ok, #{run_id := Run, conversation_id := C}} = erllama_server_api:start_run(Spec),
                ok = wait(fun() ->
                    case erllama_server_api:result(Run) of
                        {ok, #{status := S}} -> S =/= running;
                        _ -> false
                    end
                end),
                {ok, R} = erllama_server_api:result(Run),
                {C, R}
            end,
            %% Long enough to be saved (erllama keeps prompts of at least
            %% `min_tokens', 512 by default).
            Context = iolist_to_binary([
                <<"Notes for the record. ">>,
                [
                    <<"The river runs north past the old mill and the stone bridge. ">>
                 || _ <- lists:seq(1, 60)
                ],
                <<"Now name a colour.">>
            ]),
            {Conv, #{status := done}} = Turn(undefined, Context),
            ok = restart(),
            {Conv, #{status := done, cache_hit_kind := Kind}} = Turn(Conv, <<"And another one?">>),
            ct:log("second turn after restart: ~p", [Kind]),
            ?assertNotEqual(cold, Kind)
    end.
