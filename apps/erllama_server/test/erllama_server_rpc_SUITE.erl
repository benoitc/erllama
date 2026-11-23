%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% The service API over Erlang distribution, as the CLI uses it: a
%% release node runs erllama and erllama_server with stub models, and
%% a separate hidden client node drives runs through
%% erllama_server_client only. The client node has every application
%% on its code path but must never load an erllama module.
%% @end
-module(erllama_server_rpc_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1, init_per_testcase/2, end_per_testcase/2]).
-export([
    streams_a_run_to_the_end/1,
    result_matches_the_stream/1,
    reattach_with_cursor_is_contiguous/1,
    old_cursor_reports_a_gap/1,
    lagging_subscriber_is_dropped/1,
    cancel_ends_with_cancelled/1,
    run_survives_client_disconnect/1,
    conversation_accumulates_turns/1,
    conversation_runs_one_at_a_time/1,
    max_active_runs_is_enforced/1,
    unload_during_run_is_an_error_event/1,
    killed_run_stops_inference_and_frees_capacity/1,
    app_stop_cancels_streams/1,
    incompatible_api_version/1,
    missing_service_api/1,
    unreachable_release/1,
    concurrent_cli_processes/1,
    cli_cancel_while_another_follows/1,
    tty_ctrl_c_cancels_a_chat_turn/1,
    tty_multiline_is_one_message/1,
    tty_ctrl_c_at_the_prompt_drops_the_line/1,
    tty_agent_switch_and_approval/1,
    tty_agent_attach_shows_what_happened/1,
    tool_exchange_round_trips_through_the_conversation/1,
    real_model_chat/1,
    real_model_accepts_tool_history/1,
    real_model_tool_exchange/1
]).
%% Run inside the client peer.
-export([c_collect/3, c_partial/3, c_lagging/2]).

-define(COOKIE, "erllama_rpc_suite").

all() ->
    [
        streams_a_run_to_the_end,
        result_matches_the_stream,
        reattach_with_cursor_is_contiguous,
        old_cursor_reports_a_gap,
        lagging_subscriber_is_dropped,
        cancel_ends_with_cancelled,
        run_survives_client_disconnect,
        conversation_accumulates_turns,
        conversation_runs_one_at_a_time,
        max_active_runs_is_enforced,
        unload_during_run_is_an_error_event,
        killed_run_stops_inference_and_frees_capacity,
        app_stop_cancels_streams,
        incompatible_api_version,
        missing_service_api,
        unreachable_release,
        concurrent_cli_processes,
        cli_cancel_while_another_follows,
        tty_ctrl_c_cancels_a_chat_turn,
        tty_multiline_is_one_message,
        tty_ctrl_c_at_the_prompt_drops_the_line,
        tty_agent_switch_and_approval,
        tty_agent_attach_shows_what_happened,
        tool_exchange_round_trips_through_the_conversation,
        real_model_chat,
        real_model_accepts_tool_history,
        real_model_tool_exchange
    ].

%% -----------------------------------------------------------------------------
%% Setup
%% -----------------------------------------------------------------------------

init_per_suite(Config) ->
    {ok, Rel, RelNode} = start_peer(erllama_server_rpc_rel, []),
    ok = peer:call(Rel, application, set_env, [erllama_server, data_dir, ?config(priv_dir, Config)]),
    {ok, _} = peer:call(Rel, application, ensure_all_started, [erllama_server]),
    {ok, _} = rcall(Rel, erllama, load_model, [<<"fast">>, stub(1)]),
    {ok, _} = rcall(Rel, erllama, load_model, [<<"slow">>, stub(20)]),
    [{rel, Rel}, {rel_node, RelNode} | Config].

end_per_suite(Config) ->
    peer:stop(?config(rel, Config)),
    ok.

init_per_testcase(_TC, Config) ->
    {ok, Cli, _} = start_peer(erllama_server_rpc_cli, ["-hidden"]),
    {ok, _, #{api := _}} = peer:call(Cli, erllama_server_client, connect, [
        ?config(rel_node, Config)
    ]),
    [{cli, Cli} | Config].

end_per_testcase(_TC, Config) ->
    Cli = ?config(cli, Config),
    Leaked =
        case is_process_alive(Cli) of
            true ->
                L = leaked(Cli),
                peer:stop(Cli),
                L;
            false ->
                []
        end,
    Rel = ?config(rel, Config),
    [ok = peer:call(Rel, application, unset_env, [erllama_server, K]) || K <- config_keys()],
    ?assertEqual([], Leaked),
    ok.

%% erllama modules the client node loaded, other than the client and
%% this suite's helpers.
leaked(Cli) ->
    Allowed = ["erllama_server_client", atom_to_list(?MODULE)],
    Loaded = [atom_to_list(M) || {M, _} <- peer:call(Cli, code, all_loaded, [])],
    [M || "erllama" ++ _ = M <- Loaded, not lists:member(M, Allowed)].

config_keys() ->
    [run_buffer_events, subscriber_max_unacked, run_retention_ms, max_active_runs].

start_peer(Prefix, Extra) ->
    Paths = [P || P <- code:get_path(), string:find(P, "_build") =/= nomatch],
    SuiteDir = filename:dirname(code:which(?MODULE)),
    peer:start(#{
        name => peer:random_name(Prefix),
        host => "127.0.0.1",
        longnames => true,
        connection => standard_io,
        args => ["-setcookie", ?COOKIE, "-pa", SuiteDir | lists:append([["-pa", P] || P <- Paths])] ++
            Extra
    }).

stub(DelayMs) ->
    #{backend => erllama_model_stub, step_delay_ms => DelayMs}.

rcall(Peer, M, F, A) -> peer:call(Peer, M, F, A, 30000).

%% A control call made by the client node, as the CLI makes it.
ccall(Config, F, A) ->
    peer:call(
        ?config(cli, Config), erllama_server_client, call, [?config(rel_node, Config), F, A], 30000
    ).

set_env(Config, K, V) ->
    ok = peer:call(?config(rel, Config), application, set_env, [erllama_server, K, V]).

turn(Model, Text) ->
    turn(Model, Text, #{response_tokens => 8}).

turn(Model, Text, Opts) ->
    #{model => Model, messages => [#{role => user, content => Text}], opts => Opts}.

start(Config, Spec) ->
    {ok, #{run_id := Run}} = ccall(Config, start_run, [Spec]),
    Run.

collect(Config, Run, Cursor) ->
    peer:call(
        ?config(cli, Config), ?MODULE, c_collect, [?config(rel_node, Config), Run, Cursor], 60000
    ).

wait_status(Config, Run, Want) ->
    wait(fun() ->
        case rcall(?config(rel, Config), erllama_server_api, run, [Run]) of
            {ok, #{status := Want}} -> true;
            _ -> false
        end
    end).

wait(Fun) -> wait(Fun, 200).
wait(_Fun, 0) ->
    ct:fail(condition_not_reached);
wait(Fun, N) ->
    case Fun() of
        true ->
            ok;
        false ->
            timer:sleep(25),
            wait(Fun, N - 1)
    end.

types(Events) -> [T || #{type := T} <- Events].
seqs(Events) -> [S || #{seq := S} <- Events].
tokens(Events) -> iolist_to_binary([D || #{type := token, data := D} <- Events]).

%% -----------------------------------------------------------------------------
%% Client-side helpers (executed in the client peer)
%% -----------------------------------------------------------------------------

%% Follow a run, keeping every event the sink sees.
c_collect(Node, Run, Cursor) ->
    Sink = fun(E) -> put(events, [E | get_events()]) end,
    Res = erllama_server_client:follow(Node, Run, Cursor, Sink),
    {Res, lists:reverse(get_events())}.

get_events() ->
    case get(events) of
        undefined -> [];
        L -> L
    end.

%% A subscriber that reads `N' events, then dies. Returns the last seq.
c_partial(Node, Run, N) ->
    Self = self(),
    {Pid, Mon} = spawn_monitor(fun() ->
        {ok, #{replay := Replay}} = erllama_server_client:call(Node, attach, [Run, self(), 0]),
        Self ! {last, read(N - length(Replay), lists:last([-1 | seqs(Replay)]))}
    end),
    receive
        {last, Seq} ->
            receive
                {'DOWN', Mon, process, Pid, _} -> Seq
            end
    after 10000 -> exit(partial_timeout)
    end.

read(N, Last) when N =< 0 -> Last;
read(N, _Last) ->
    receive
        {erllama_server_run, _, #{seq := S}} -> read(N - 1, S)
    end.

%% A subscriber that never acks: returns the drop notice it gets.
c_lagging(Node, Run) ->
    {ok, _} = erllama_server_client:call(Node, attach, [Run, self(), 0]),
    lag().

lag() ->
    receive
        {erllama_server_run, _, {dropped, Seq}} ->
            {dropped, Seq};
        {erllama_server_run, _, #{type := T}} when T =:= done; T =:= cancelled; T =:= error ->
            not_dropped;
        {erllama_server_run, _, _} ->
            lag()
    after 20000 -> timeout
    end.

%% -----------------------------------------------------------------------------
%% Cases
%% -----------------------------------------------------------------------------

streams_a_run_to_the_end(Config) ->
    Run = start(Config, turn(<<"fast">>, <<"hello there">>)),
    {{ok, #{type := done}}, Events} = collect(Config, Run, 0),
    ?assertMatch([started | _], types(Events)),
    ?assertEqual(lists:seq(0, length(Events) - 1), seqs(Events)),
    ?assertEqual(8, length([token || token <- types(Events)])),
    ?assertEqual(done, lists:last(types(Events))).

result_matches_the_stream(Config) ->
    Run = start(Config, turn(<<"fast">>, <<"one two">>)),
    {{ok, #{data := Done}}, Events} = collect(Config, Run, 0),
    {ok, Result} = ccall(Config, result, [Run]),
    ?assertEqual(tokens(Events), maps:get(content, Result)),
    ?assertEqual(maps:get(content, Done), maps:get(content, Result)),
    ?assertMatch(
        #{status := done, finish_reason := length, usage := #{completion_tokens := 8}}, Result
    ).

reattach_with_cursor_is_contiguous(Config) ->
    Run = start(Config, turn(<<"slow">>, <<"keep going">>, #{response_tokens => 40})),
    Last = peer:call(?config(cli, Config), ?MODULE, c_partial, [?config(rel_node, Config), Run, 5]),
    ?assert(Last >= 4),
    {{ok, #{type := done}}, Events} = collect(Config, Run, Last + 1),
    ?assertEqual(lists:seq(Last + 1, Last + length(Events)), seqs(Events)),
    ?assertEqual(done, lists:last(types(Events))).

old_cursor_reports_a_gap(Config) ->
    set_env(Config, run_buffer_events, 4),
    Run = start(Config, turn(<<"fast">>, <<"short buffer">>)),
    ok = wait_status(Config, Run, done),
    {{ok, #{type := done}}, Events} = collect(Config, Run, 0),
    [{gap, 0, Oldest} | Rest] = Events,
    ?assert(Oldest > 0),
    ?assertEqual(lists:seq(Oldest, Oldest + 3), seqs(Rest)).

lagging_subscriber_is_dropped(Config) ->
    set_env(Config, subscriber_max_unacked, 3),
    Run = start(Config, turn(<<"slow">>, <<"lagging">>, #{response_tokens => 20})),
    Res = peer:call(
        ?config(cli, Config), ?MODULE, c_lagging, [?config(rel_node, Config), Run], 30000
    ),
    ?assertMatch({dropped, _}, Res),
    ok = wait_status(Config, Run, done),
    %% The dropped subscriber picks up where it was cut off.
    {dropped, From} = Res,
    {{ok, #{type := done}}, Events} = collect(Config, Run, From),
    ?assertEqual(From, hd(seqs(Events))).

cancel_ends_with_cancelled(Config) ->
    Spec = (turn(<<"slow">>, <<"never mind">>, #{response_tokens => 500}))#{
        conversation_id => <<"conv_cancel">>
    },
    {ok, #{run_id := Run}} = ccall(Config, start_run, [Spec]),
    timer:sleep(100),
    ok = ccall(Config, cancel, [Run]),
    {{ok, #{type := cancelled}}, _} = collect(Config, Run, 0),
    {ok, #{finish_reason := cancelled}} = ccall(Config, result, [Run]),
    %% A cancelled turn is not recorded, and the conversation is free.
    {ok, #{messages := [], active_run := undefined}} = ccall(Config, conversation, [
        <<"conv_cancel">>
    ]).

run_survives_client_disconnect(Config) ->
    Cli = ?config(cli, Config),
    Node = ?config(rel_node, Config),
    Run = start(Config, turn(<<"slow">>, <<"walk away">>, #{response_tokens => 15})),
    _ = peer:call(Cli, erlang, spawn, [?MODULE, c_collect, [Node, Run, 0]]),
    timer:sleep(100),
    peer:stop(Cli),
    ok = wait_status(Config, Run, done),
    {ok, Cli2, _} = start_peer(erllama_server_rpc_cli, ["-hidden"]),
    Config2 = lists:keystore(cli, 1, Config, {cli, Cli2}),
    {ok, #{status := done, usage := #{completion_tokens := 15}}} = ccall(Config2, result, [Run]),
    ?assertEqual([], leaked(Cli2)),
    peer:stop(Cli2).

conversation_accumulates_turns(Config) ->
    Spec1 = turn(<<"fast">>, <<"first">>),
    {ok, #{run_id := R1, conversation_id := Conv}} = ccall(Config, start_run, [Spec1]),
    {{ok, #{type := done, data := #{usage := #{prompt_tokens := P1}}}}, _} = collect(Config, R1, 0),
    Spec2 = (turn(<<"fast">>, <<"second">>))#{conversation_id => Conv},
    {ok, #{run_id := R2, conversation_id := Conv}} = ccall(Config, start_run, [Spec2]),
    {{ok, #{type := done, data := #{usage := #{prompt_tokens := P2}}}}, _} = collect(Config, R2, 0),
    %% The second turn is prompted with the whole history.
    ?assert(P2 > P1),
    {ok, #{messages := Msgs}} = ccall(Config, conversation, [Conv]),
    ?assertEqual(
        [<<"user">>, <<"assistant">>, <<"user">>, <<"assistant">>],
        [R || #{role := R} <- Msgs]
    ),
    ?assertMatch([#{content := <<"first">>}, _, #{content := <<"second">>}, _], Msgs).

conversation_runs_one_at_a_time(Config) ->
    Spec = (turn(<<"slow">>, <<"busy">>, #{response_tokens => 500}))#{
        conversation_id => <<"conv_busy">>
    },
    {ok, #{run_id := Run}} = ccall(Config, start_run, [Spec]),
    ?assertEqual({error, conversation_busy}, ccall(Config, start_run, [Spec])),
    ok = ccall(Config, cancel, [Run]),
    ok = wait_status(Config, Run, cancelled),
    {ok, #{run_id := Run2}} = ccall(Config, start_run, [Spec#{opts => #{response_tokens => 2}}]),
    {{ok, #{type := done}}, _} = collect(Config, Run2, 0).

max_active_runs_is_enforced(Config) ->
    set_env(Config, max_active_runs, 1),
    Run = start(Config, turn(<<"slow">>, <<"only one">>, #{response_tokens => 500})),
    ?assertEqual({error, too_many_runs}, ccall(Config, start_run, [turn(<<"fast">>, <<"two">>)])),
    ok = ccall(Config, cancel, [Run]),
    ok = wait_status(Config, Run, cancelled),
    ?assertMatch({ok, _}, ccall(Config, start_run, [turn(<<"fast">>, <<"two">>)])).

unload_during_run_is_an_error_event(Config) ->
    Rel = ?config(rel, Config),
    {ok, _} = rcall(Rel, erllama, load_model, [<<"victim">>, stub(20)]),
    Run = start(Config, turn(<<"victim">>, <<"pull the rug">>, #{response_tokens => 500})),
    timer:sleep(100),
    ok = rcall(Rel, erllama, unload, [<<"victim">>]),
    {{ok, #{type := error}}, _} = collect(Config, Run, 0),
    {ok, #{status := error}} = ccall(Config, result, [Run]).

%% A run killed outright (no terminate/2) must not leave its model
%% generating while its slot and conversation are handed out again.
killed_run_stops_inference_and_frees_capacity(Config) ->
    Rel = ?config(rel, Config),
    set_env(Config, max_active_runs, 1),
    Spec = (turn(<<"slow">>, <<"killed">>, #{response_tokens => 500}))#{
        conversation_id => <<"conv_kill">>
    },
    {ok, #{run_id := Run}} = ccall(Config, start_run, [Spec]),
    ok = wait(fun() -> rcall(Rel, erllama, status, [<<"slow">>]) =:= {ok, generating} end),
    {ok, #{pid := Pid}} = rcall(Rel, erllama_server_runs, lookup, [Run]),
    true = rcall(Rel, erlang, exit, [Pid, kill]),
    %% 500 tokens at 20 ms would run for 10 s; the stream is cancelled
    %% within a few steps and the model goes idle.
    ok = wait(fun() -> rcall(Rel, erllama, requests, []) =:= [] end, 40),
    ok = wait(fun() -> rcall(Rel, erllama, status, [<<"slow">>]) =:= {ok, idle} end, 40),
    ?assertEqual({error, not_found}, ccall(Config, run, [Run])),
    %% The slot and the conversation are free again.
    {ok, #{run_id := Run2}} = ccall(Config, start_run, [Spec#{opts => #{response_tokens => 2}}]),
    {{ok, #{type := done}}, _} = collect(Config, Run2, 0).

app_stop_cancels_streams(Config) ->
    Rel = ?config(rel, Config),
    Spec = turn(<<"slow">>, <<"stop the world">>, #{response_tokens => 500}),
    {ok, _} = rcall(Rel, erllama_server_api, start_run, [Spec]),
    ok = wait(fun() -> rcall(Rel, erllama, status, [<<"slow">>]) =:= {ok, generating} end),
    ok = rcall(Rel, application, stop, [erllama_server]),
    %% 500 tokens at 20 ms would take 10 s; the stream is cancelled
    %% when its run stops, so the model goes idle almost at once.
    ok = wait(fun() -> rcall(Rel, erllama, status, [<<"slow">>]) =:= {ok, idle} end, 40),
    {ok, _} = rcall(Rel, application, ensure_all_started, [erllama_server]).

incompatible_api_version(Config) ->
    ?assertMatch({error, {incompatible_api, #{client := 999}}}, ccall(Config, hello, [999])).

missing_service_api(Config) ->
    ?assertEqual(
        {error, {no_service_api, ?config(rel_node, Config)}}, ccall(Config, no_such_function, [])
    ).

unreachable_release(Config) ->
    ?assertEqual(
        {error, {unreachable, 'erllama_absent@127.0.0.1'}},
        peer:call(
            ?config(cli, Config),
            erllama_server_client,
            connect,
            ['erllama_absent@127.0.0.1'],
            30000
        )
    ).

%% -----------------------------------------------------------------------------
%% The CLI as separate OS processes, as several terminals run it
%% -----------------------------------------------------------------------------

%% Start `erllama Args' in a fresh VM with only erllama_server on its
%% code path, as the escript has, and the release node and cookie in
%% its environment.
cli_os(Config, Args) ->
    Erl = filename:join([code:root_dir(), "bin", "erl"]),
    Ebin = filename:dirname(code:which(erllama_server_cli)),
    Expr = lists:flatten(io_lib:format("erllama_server_cli:main(~p).", [Args])),
    open_port({spawn_executable, Erl}, [
        {args, ["-noshell", "-pa", Ebin, "-eval", Expr]},
        {env, [
            {"ERLLAMA_NODE", atom_to_list(?config(rel_node, Config))},
            {"ERLLAMA_COOKIE", ?COOKIE}
        ]},
        exit_status,
        binary,
        stderr_to_stdout
    ]).

cli_wait(Port) -> cli_wait(Port, []).
cli_wait(Port, Acc) ->
    receive
        {Port, {data, D}} -> cli_wait(Port, [Acc, D]);
        {Port, {exit_status, S}} -> {S, iolist_to_binary(Acc)}
    after 60000 -> ct:fail({cli_timeout, iolist_to_binary(Acc)})
    end.

%% Fresh VMs started together must not pick the same client node name.
concurrent_cli_processes(Config) ->
    Ports = [cli_os(Config, ["ping"]) || _ <- lists:seq(1, 8)],
    Results = [cli_wait(P) || P <- Ports],
    [?assertMatch({0, <<"pong ", _/binary>>}, R) || R <- Results].

cli_cancel_while_another_follows(Config) ->
    Run = start(Config, turn(<<"slow">>, <<"two terminals">>, #{response_tokens => 500})),
    Follower = cli_os(Config, ["attach", binary_to_list(Run)]),
    %% Wait until the follower is streaming before cancelling.
    First =
        receive
            {Follower, {data, D}} -> D
        after 30000 -> ct:fail(follower_printed_nothing)
        end,
    ?assertMatch({0, _}, cli_wait(cli_os(Config, ["cancel", binary_to_list(Run)]))),
    {Code, Out} = cli_wait(Follower, [First]),
    ?assertEqual(1, Code),
    ?assertNotEqual(nomatch, binary:match(Out, <<"[cancelled]">>)).

%% -----------------------------------------------------------------------------
%% The CLI in a terminal (a pseudo-terminal from script(1)), typed to.
%% -----------------------------------------------------------------------------

%% The CLI under a pseudo-terminal, as `bin/erllama' runs it in one
%% (`+Bc'). Keys go in with tty_type/2; the output is collected.
tty_cli(Config, Args) ->
    Erl = filename:join([code:root_dir(), "bin", "erl"]),
    Ebin = filename:dirname(code:which(erllama_server_cli)),
    Expr = lists:flatten(io_lib:format("erllama_server_cli:main(~p).", [Args])),
    Cmd = [Erl, "+Bc", "-noshell", "-pa", Ebin, "-eval", Expr],
    ScriptArgs =
        case os:type() of
            {unix, darwin} -> ["-q", "/dev/null" | Cmd];
            _ -> ["-q", "-e", "-c", lists:join(" ", [shell_quote(A) || A <- Cmd]), "/dev/null"]
        end,
    Port = open_port({spawn_executable, os:find_executable("script")}, [
        {args, ScriptArgs},
        {env, [
            {"ERLLAMA_NODE", atom_to_list(?config(rel_node, Config))},
            {"ERLLAMA_COOKIE", ?COOKIE},
            {"TERM", "xterm"}
        ]},
        exit_status,
        binary,
        stderr_to_stdout
    ]),
    put({tty_out, Port}, <<>>),
    Port.

shell_quote(A) -> "'" ++ string:replace(A, "'", "'\\''", all) ++ "'".

tty_type(Port, Keys) ->
    true = port_command(Port, Keys),
    ok.

%% Wait until the output holds `Text'.
tty_wait(Port, Text) ->
    tty_wait(Port, Text, 300).
tty_wait(Port, Text, 0) ->
    ct:fail({tty_never_showed, Text, get({tty_out, Port})});
tty_wait(Port, Text, N) ->
    case binary:match(get({tty_out, Port}), Text) of
        nomatch ->
            receive
                {Port, {data, D}} ->
                    put({tty_out, Port}, <<(get({tty_out, Port}))/binary, D/binary>>),
                    tty_wait(Port, Text, N);
                {Port, {exit_status, S}} ->
                    ct:fail({tty_exited, S, Text, get({tty_out, Port})})
            after 100 -> tty_wait(Port, Text, N - 1)
            end;
        _ ->
            ok
    end.

%% Wait until the output matches the regular expression `Re'.
tty_wait_re(Port, Re) ->
    tty_wait_re(Port, Re, 300).
tty_wait_re(Port, Re, 0) ->
    ct:fail({tty_never_matched, Re, get({tty_out, Port})});
tty_wait_re(Port, Re, N) ->
    case re:run(get({tty_out, Port}), Re) of
        {match, _} ->
            ok;
        nomatch ->
            receive
                {Port, {data, D}} ->
                    put({tty_out, Port}, <<(get({tty_out, Port}))/binary, D/binary>>),
                    tty_wait_re(Port, Re, N)
            after 100 -> tty_wait_re(Port, Re, N - 1)
            end
    end.

tty_exit(Port) ->
    receive
        {Port, {data, D}} ->
            put({tty_out, Port}, <<(get({tty_out, Port}))/binary, D/binary>>),
            tty_exit(Port);
        {Port, {exit_status, _}} ->
            get({tty_out, Port})
    after 30000 -> ct:fail({tty_did_not_exit, get({tty_out, Port})})
    end.

skip_without_script(Fun) ->
    case os:find_executable("script") of
        false -> {skip, "script(1) is needed for the terminal cases"};
        _ -> Fun()
    end.

runs_of(Config) ->
    {ok, Runs} = ccall(Config, runs, []),
    Runs.

%% Ctrl-C while a reply streams cancels that turn; the prompt comes back
%% and the commands work.
tty_ctrl_c_cancels_a_chat_turn(Config) ->
    skip_without_script(fun() ->
        Before = length(runs_of(Config)),
        P = tty_cli(Config, ["chat", "slow", "--max-tokens", "2000"]),
        ok = tty_wait(P, <<"> ">>),
        ok = tty_type(P, "a long story\r"),
        %% Streaming: the stub's tokens are digits.
        ok = tty_wait_re(P, "story\r\n[0-9]{4}"),
        ok = tty_type(P, [3]),
        ok = tty_wait(P, <<"[cancelled]">>),
        ok = tty_wait(P, <<"[cancelled]\r\n> ">>),
        ok = tty_type(P, "/sessions\r"),
        ok = tty_wait(P, <<" messages ">>),
        ok = tty_type(P, "/quit\r"),
        _ = tty_exit(P),
        New = lists:sublist(lists:reverse(runs_of(Config)), length(runs_of(Config)) - Before),
        ?assertMatch([#{status := cancelled}], New)
    end).

%% A line ending in `\' goes on: the two lines are one message.
tty_multiline_is_one_message(Config) ->
    skip_without_script(fun() ->
        P = tty_cli(Config, ["chat", "fast", "--max-tokens", "4"]),
        ok = tty_wait(P, <<"> ">>),
        ok = tty_type(P, "first part\\\r"),
        ok = tty_wait(P, <<"... ">>),
        ok = tty_type(P, "second part\r"),
        ok = wait(fun() ->
            lists:any(
                fun
                    (#{conversation_id := C}) when is_binary(C) ->
                        case ccall(Config, conversation, [C]) of
                            {ok, #{messages := [#{content := <<"first part\nsecond part">>} | _]}} ->
                                true;
                            _ ->
                                false
                        end;
                    (_) ->
                        false
                end,
                runs_of(Config)
            )
        end),
        ok = tty_type(P, "/quit\r"),
        _ = tty_exit(P)
    end).

%% Ctrl-C at the prompt drops what was typed: no run starts.
tty_ctrl_c_at_the_prompt_drops_the_line(Config) ->
    skip_without_script(fun() ->
        Before = length(runs_of(Config)),
        P = tty_cli(Config, ["chat", "fast"]),
        ok = tty_wait(P, <<"> ">>),
        %% Typed as a person types: the keys, Ctrl-C, Enter, then the
        %% next prompt before anything else.
        ok = tty_type(P, "nope"),
        timer:sleep(200),
        ok = tty_type(P, [3]),
        timer:sleep(200),
        ok = tty_type(P, "\r"),
        ok = tty_wait_re(P, "nope.*\r\n> "),
        ?assertEqual(Before, length(runs_of(Config))),
        ok = tty_type(P, "/quit\r"),
        _ = tty_exit(P),
        ?assertEqual(Before, length(runs_of(Config)))
    end).

%% /switch to another session, whose turn asks for an approval answered
%% in the terminal.
tty_agent_switch_and_approval(Config) ->
    skip_without_script(fun() ->
        Ws = filename:join(?config(priv_dir, Config), "tty_ws"),
        ok = filelib:ensure_path(Ws),
        {ok, #{agent_id := Agent}} = ccall(Config, agent_start, [
            #{
                model => <<"fast">>,
                workspace => list_to_binary(Ws),
                generator => erllama_server_agent_script
            }
        ]),
        ok = rcall(?config(rel, Config), erllama_server_agent_script, set, [
            Agent,
            [
                #{
                    calls => [
                        {<<"w1">>, <<"write_file">>, #{
                            <<"path">> => <<"note.txt">>, <<"content">> => <<"hi">>
                        }}
                    ]
                },
                #{text => <<"written">>}
            ]
        ]),
        P = tty_cli(Config, ["agent", "fast", "--workspace", Ws]),
        ok = tty_wait(P, <<"you> ">>),
        ok = tty_type(P, "/switch " ++ binary_to_list(Agent) ++ "\r"),
        ok = tty_wait(P, <<"[agent ", Agent/binary>>),
        ok = tty_type(P, "write the note\r"),
        ok = tty_wait(P, <<"allow write_file">>),
        ok = tty_type(P, "y\r"),
        ok = tty_wait(P, <<"written\r\nyou> ">>),
        ok = tty_type(P, "/quit\r"),
        _ = tty_exit(P),
        ?assertEqual({ok, <<"hi">>}, file:read_file(filename:join(Ws, "note.txt")))
    end).

%% A session works while no client is attached and stops at an
%% approval. `agent --attach' from a new terminal shows it waiting,
%% replays the turn's output, asks for the approval and follows the
%% rest of the turn.
tty_agent_attach_shows_what_happened(Config) ->
    skip_without_script(fun() ->
        Ws = filename:join(?config(priv_dir, Config), "tty_attach_ws"),
        ok = filelib:ensure_path(Ws),
        {ok, #{agent_id := Agent}} = ccall(Config, agent_start, [
            #{
                model => <<"fast">>,
                workspace => list_to_binary(Ws),
                generator => erllama_server_agent_script
            }
        ]),
        ok = rcall(?config(rel, Config), erllama_server_agent_script, set, [
            Agent,
            [
                #{
                    text => <<"looking first">>,
                    calls => [
                        {<<"w1">>, <<"write_file">>, #{
                            <<"path">> => <<"away.txt">>, <<"content">> => <<"done">>
                        }}
                    ]
                },
                #{text => <<"written while you were away">>}
            ]
        ]),
        ok = ccall(Config, agent_send, [Agent, <<"write the note">>]),
        ok = wait(fun() ->
            case ccall(Config, inspect, [Agent]) of
                {ok, #{state := approving, approvals := [_]}} -> true;
                _ -> false
            end
        end),
        P = tty_cli(Config, ["agent", "--attach", binary_to_list(Agent)]),
        ok = tty_wait(P, <<"[approving, 1 approval(s) pending]">>),
        ok = tty_wait(P, <<"looking first">>),
        ok = tty_wait(P, <<"allow write_file">>),
        ok = tty_type(P, "y\r"),
        ok = tty_wait(P, <<"written while you were away">>),
        ok = tty_wait(P, <<"you> ">>),
        ok = tty_type(P, "/quit\r"),
        _ = tty_exit(P),
        ?assertEqual({ok, <<"done">>}, file:read_file(filename:join(Ws, "away.txt")))
    end).

%% Gated: a real GGUF through llama.cpp's chat template and parser.
real_model_chat(Config) ->
    with_real_model(Config, <<"real_chat">>, fun(Id) ->
        Run = start(
            Config, turn(Id, <<"Say hello in one short sentence.">>, #{response_tokens => 32})
        ),
        {{ok, #{type := done}}, Events} = collect(Config, Run, 0),
        ?assert(byte_size(tokens(Events)) > 0),
        {ok, #{content := Content}} = ccall(Config, result, [Run]),
        ?assert(byte_size(Content) > 0)
    end).

%% -----------------------------------------------------------------------------
%% Tool calls across turns
%% -----------------------------------------------------------------------------

weather_tool() ->
    #{
        name => <<"get_weather">>,
        description => <<"Current weather for a city">>,
        parameters => #{
            type => object,
            properties => #{city => #{type => string}},
            required => [city]
        }
    }.

%% The assistant's call and the tool's answer, as a client sends them
%% back: the call flat, as result/1 returns it.
tool_turn(CallId) ->
    [
        #{
            role => assistant,
            content => null,
            tool_calls => [
                #{name => <<"get_weather">>, arguments => #{city => <<"Paris">>}, id => CallId}
            ]
        },
        #{
            role => tool,
            tool_call_id => CallId,
            name => <<"get_weather">>,
            content => <<"{\"temp_c\": 21}">>
        }
    ].

tool_exchange_round_trips_through_the_conversation(Config) ->
    Opts = #{tools => [weather_tool()], response_tokens => 4},
    {ok, #{run_id := R1, conversation_id := Conv}} =
        ccall(Config, start_run, [
            #{
                model => <<"fast">>,
                messages => [#{role => user, content => <<"Weather in Paris?">>}],
                opts => Opts
            }
        ]),
    {{ok, #{type := done}}, _} = collect(Config, R1, 0),
    {ok, #{run_id := R2}} =
        ccall(Config, start_run, [
            #{
                model => <<"fast">>,
                conversation_id => Conv,
                messages => tool_turn(<<"call_p1">>),
                opts => Opts
            }
        ]),
    {{ok, #{type := done}}, _} = collect(Config, R2, 0),
    {ok, #{messages := Msgs}} = ccall(Config, conversation, [Conv]),
    ?assertEqual(
        [<<"user">>, <<"assistant">>, <<"assistant">>, <<"tool">>, <<"assistant">>],
        [R || #{role := R} <- Msgs]
    ),
    [_, _, Call, Result, _] = Msgs,
    ?assertMatch(
        #{
            content := null,
            tool_calls := [
                #{
                    id := <<"call_p1">>,
                    type := <<"function">>,
                    function := #{name := <<"get_weather">>}
                }
            ]
        },
        Call
    ),
    ?assertMatch(#{tool_call_id := <<"call_p1">>, name := <<"get_weather">>}, Result).

%% Each case loads under its own id: an unloaded id is freed
%% asynchronously, so reusing it right away can see `already_loaded'.
with_real_model(Config, Name, Fun) ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {skip, "set LLAMA_TEST_MODEL to a GGUF path to run the real-model case"};
        Path ->
            Rel = ?config(rel, Config),
            {ok, Id} = rcall(Rel, erllama_server_api, load_model, [
                Path, #{id => Name, n_ctx => 4096}
            ]),
            try
                Fun(Id)
            after
                rcall(Rel, erllama, unload, [Id])
            end
    end.

%% Any model: a stored tool exchange must pass llama.cpp's own message
%% parser (it refuses tool calls without `type' and `function').
real_model_accepts_tool_history(Config) ->
    with_real_model(Config, <<"real_history">>, fun(Id) ->
        Msgs = [#{role => user, content => <<"Weather in Paris?">>} | tool_turn(<<"call_r1">>)],
        Run = start(Config, #{
            model => Id,
            messages => Msgs,
            opts => #{tools => [weather_tool()], response_tokens => 24}
        }),
        {{ok, #{type := done}}, _} = collect(Config, Run, 0)
    end).

%% A tool-capable model (Qwen2.5, Llama 3.1, ...): the model calls the
%% tool, the client answers with the call's id, the model goes on.
real_model_tool_exchange(Config) ->
    with_real_model(Config, <<"real_tools">>, fun(Id) ->
        Opts = #{tools => [weather_tool()], tool_choice => required, response_tokens => 128},
        Ask = #{role => user, content => <<"What is the weather in Paris? Use the tool.">>},
        {ok, #{run_id := R1, conversation_id := Conv}} =
            ccall(Config, start_run, [#{model => Id, messages => [Ask], opts => Opts}]),
        {{ok, #{type := done}}, _} = collect(Config, R1, 0),
        case ccall(Config, result, [R1]) of
            {ok, #{tool_calls := []}} ->
                {skip, "the model's template synthesizes no tool grammar"};
            {ok, #{tool_calls := [#{id := CallId, name := <<"get_weather">>} | _]}} ->
                {ok, #{messages := [_, #{tool_calls := [#{id := CallId}]}]}} =
                    ccall(Config, conversation, [Conv]),
                Answer = #{role => tool, tool_call_id => CallId, content => <<"{\"temp_c\": 21}">>},
                {ok, #{run_id := R2}} =
                    ccall(Config, start_run, [
                        #{
                            model => Id,
                            conversation_id => Conv,
                            messages => [Answer],
                            opts => Opts#{tool_choice => none}
                        }
                    ]),
                {{ok, #{type := done}}, _} = collect(Config, R2, 0),
                {ok, #{content := Text}} = ccall(Config, result, [R2]),
                ct:log("reply after the tool result: ~ts", [Text]),
                ?assert(byte_size(Text) > 0)
        end
    end).
