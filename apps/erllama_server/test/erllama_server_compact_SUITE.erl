%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% Context compaction on erllama's stub backend, with a threshold low
%% enough that every eligible transcript is compacted: a conversation
%% turn compacts automatically and goes on (and the result survives a
%% restart), a conversation compacts on request, an agent session
%% compacts at the start of a turn and on request, and an interrupt
%% stops a compaction in progress. A gated case has a real model write
%% the summary.
%% @end
-module(erllama_server_compact_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_testcase/2, end_per_testcase/2]).
-export([
    run_compacts_automatically/1,
    conversation_compacts_on_request/1,
    session_compacts_at_turn_start/1,
    session_compacts_on_request/1,
    interrupt_stops_compaction/1,
    real_model_summary/1
]).

all() ->
    [
        run_compacts_automatically,
        conversation_compacts_on_request,
        session_compacts_at_turn_start,
        session_compacts_on_request,
        interrupt_stops_compaction,
        real_model_summary
    ].

init_per_testcase(TC, Config) ->
    Base = filename:join(?config(priv_dir, Config), atom_to_list(TC)),
    Ws = filename:join(Base, "ws"),
    ok = filelib:ensure_path(Ws),
    Threshold =
        case TC of
            conversation_compacts_on_request -> false;
            session_compacts_on_request -> false;
            interrupt_stops_compaction -> false;
            real_model_summary -> false;
            _ -> 0.001
        end,
    Env = [
        {data_dir, filename:join(Base, "data")},
        {compact_threshold, Threshold},
        {compact_keep_messages, 2},
        {compact_summary_tokens, 8}
    ],
    [ok = application:set_env(erllama_server, K, V) || {K, V} <- Env],
    {ok, _} = application:ensure_all_started(erllama_server),
    {ok, _} = erllama:load_model(<<"stub">>, #{backend => erllama_model_stub}),
    [{ws, list_to_binary(Ws)} | Config].

end_per_testcase(_TC, _Config) ->
    _ = application:stop(erllama_server),
    [erllama:unload(Id) || #{id := Id} <- erllama:list_models()],
    [
        application:unset_env(erllama_server, K)
     || K <- [data_dir, compact_threshold, compact_keep_messages, compact_summary_tokens]
    ],
    ok.

%% One conversation turn; returns its conversation and run.
turn(Conv, Text) ->
    turn(<<"stub">>, Conv, Text, 3).

turn(Model, Conv, Text, Tokens) ->
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
    ok = wait(fun() ->
        case erllama_server_api:result(Run) of
            {ok, #{status := S}} -> S =/= running;
            _ -> false
        end
    end),
    {ok, #{status := done}} = erllama_server_api:result(Run),
    {C, Run}.

events(Job) ->
    {ok, #{replay := Replay}} = erllama_server_api:attach(Job, self(), 0),
    ok = erllama_server_api:detach(Job, self()),
    flush(Job),
    [T || #{type := T} <- Replay].

flush(Job) ->
    receive
        {erllama_server_run, Job, _} -> flush(Job)
    after 0 -> ok
    end.

messages(Conv) ->
    {ok, #{messages := M}} = erllama_server_api:conversation(Conv),
    M.

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

%% The compacted form: one system message with the section, then a tail
%% that starts at a user message.
check_compacted(Msgs, First) ->
    [#{role := <<"system">>, content := Head}, #{role := <<"user">>} | _] = Msgs,
    ?assertNotEqual(nomatch, binary:match(Head, <<"The first request was:\n", First/binary>>)),
    ?assertNotEqual(nomatch, binary:match(Head, <<"Summary of what happened since:">>)).

%% -----------------------------------------------------------------------------

run_compacts_automatically(_Config) ->
    {Conv, R1} = turn(undefined, <<"first task">>),
    ?assertNot(lists:member(compacted, events(R1))),
    {Conv, _} = turn(Conv, <<"second">>),
    {Conv, R3} = turn(Conv, <<"third">>),
    %% By the third turn there is a middle to summarise.
    ?assert(lists:member(compacted, events(R3))),
    Msgs = messages(Conv),
    check_compacted(Msgs, <<"first task">>),
    ?assertMatch(#{role := <<"user">>, content := <<"third">>}, lists:nth(length(Msgs) - 1, Msgs)),
    %% The compacted transcript is what a restart brings back.
    ok = application:stop(erllama_server),
    {ok, _} = application:ensure_all_started(erllama_server),
    ?assertEqual(Msgs, messages(Conv)),
    {Conv, _} = turn(Conv, <<"fourth">>).

conversation_compacts_on_request(_Config) ->
    {Conv, _} = turn(undefined, <<"first task">>),
    ?assertEqual({ok, #{compacted => false}}, erllama_server_api:conversation_compact(Conv)),
    {Conv, _} = turn(Conv, <<"second">>),
    {Conv, _} = turn(Conv, <<"third">>),
    ?assertEqual(6, length(messages(Conv))),
    {ok, #{before := 6, 'after' := 3}} = erllama_server_api:conversation_compact(Conv),
    check_compacted(messages(Conv), <<"first task">>),
    %% A conversation a session holds is refused.
    {ok, #{agent_id := Agent}} = erllama_server_api:agent_start(#{
        model => <<"stub">>,
        workspace => ?config(ws, [{ws, <<"/tmp">>}]),
        conversation_id => Conv,
        generator => erllama_server_agent_script
    }),
    ?assertEqual({error, conversation_busy}, erllama_server_api:conversation_compact(Conv)),
    ok = erllama_server_api:agent_stop(Agent).

session(Config, Script) ->
    {ok, #{agent_id := Id, conversation_id := Conv}} = erllama_server_api:agent_start(#{
        model => <<"stub">>,
        workspace => ?config(ws, Config),
        generator => erllama_server_agent_script
    }),
    ok = erllama_server_agent_script:set(Id, Script),
    {Id, Conv}.

agent_turn(Id, Text) ->
    ok = erllama_server_api:agent_send(Id, Text),
    ok = wait(fun() -> idle(Id) end).

idle(Id) ->
    {ok, Info} = erllama_server_api:run(Id),
    maps:get(state, Info, undefined) =:= idle.

session_compacts_at_turn_start(Config) ->
    {Id, Conv} = session(Config, [
        #{text => <<"one">>}, #{text => <<"two">>}, #{text => <<"three">>}
    ]),
    ok = agent_turn(Id, <<"first task">>),
    ok = agent_turn(Id, <<"second">>),
    ok = agent_turn(Id, <<"third">>),
    ?assert(lists:member(compacted, events(Id))),
    %% The round after the compaction was given the compacted transcript.
    Last = lists:last(erllama_server_agent_script:rounds(Id)),
    check_compacted(Last, <<"first task">>),
    check_compacted(messages(Conv), <<"first task">>).

session_compacts_on_request(Config) ->
    {Id, Conv} = session(Config, [#{text => <<"one">>}, #{text => <<"two">>}]),
    ok = agent_turn(Id, <<"first task">>),
    ok = agent_turn(Id, <<"second">>),
    ok = agent_turn(Id, <<"third">>),
    {ok, #{before := B, 'after' := A}} = erllama_server_api:agent_compact(Id),
    ?assert(A < B),
    check_compacted(messages(Conv), <<"first task">>),
    %% The session goes on.
    ok = agent_turn(Id, <<"fourth">>).

%% A slow model (each token takes a while) so the compaction can be
%% caught in progress.
interrupt_stops_compaction(Config) ->
    {ok, _} = erllama:load_model(<<"slow">>, #{backend => erllama_model_stub, step_delay_ms => 200}),
    ok = application:set_env(erllama_server, compact_summary_tokens, 200),
    {ok, #{agent_id := Id, conversation_id := Conv}} = erllama_server_api:agent_start(#{
        model => <<"slow">>,
        workspace => ?config(ws, Config),
        generator => erllama_server_agent_script
    }),
    ok = erllama_server_agent_script:set(Id, [
        #{text => <<"a">>}, #{text => <<"b">>}, #{text => <<"c">>}
    ]),
    ok = agent_turn(Id, <<"first task">>),
    ok = agent_turn(Id, <<"second">>),
    ok = agent_turn(Id, <<"third">>),
    Before = messages(Conv),
    Self = self(),
    _ = spawn(fun() -> Self ! {compact, erllama_server_api:agent_compact(Id)} end),
    ok = wait(fun() -> lists:member(compacting, events(Id)) end),
    ok = erllama_server_api:agent_interrupt(Id),
    receive
        {compact, R} -> ?assertEqual({error, interrupted}, R)
    after 5000 -> ct:fail(compact_never_answered)
    end,
    ?assertEqual(Before, messages(Conv)),
    ok = wait(fun() -> idle(Id) end).

%% Gated: a real model writes the summary, and the next turn answers.
real_model_summary(_Config) ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {skip, "set LLAMA_TEST_MODEL to a GGUF to run the real-model case"};
        Path ->
            ok = application:set_env(erllama_server, compact_summary_tokens, 160),
            {ok, Model} = erllama_server_api:load_model(Path, #{name => <<"compact_real">>}),
            {Conv, _} = turn(
                Model, undefined, <<"Plan a picnic for four people on Saturday.">>, 48
            ),
            {Conv, _} = turn(Model, Conv, <<"We will go to the lake. What should we bring?">>, 48),
            {Conv, _} = turn(Model, Conv, <<"Add something for rain.">>, 48),
            {ok, #{before := B, 'after' := A}} = erllama_server_api:conversation_compact(Conv),
            ct:log("compacted ~b -> ~b messages: ~p", [B, A, messages(Conv)]),
            [#{content := Head} | _] = messages(Conv),
            [_, Summary] = binary:split(Head, <<"Summary of what happened since:\n">>),
            ?assert(byte_size(string:trim(Summary)) > 20),
            {Conv, Run} = turn(Model, Conv, <<"Remind me where we are going.">>, 32),
            {ok, #{content := Answer}} = erllama_server_api:result(Run),
            ct:log("after compaction: ~ts", [Answer]),
            ?assert(byte_size(Answer) > 0)
    end.
