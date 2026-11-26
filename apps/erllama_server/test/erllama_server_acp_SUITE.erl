%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% The ACP agent (`erllama acp') end to end against a stub-model
%% release: a release peer runs erllama_server with scripted agents, a
%% hidden client peer runs `erllama_server_acp' with a sink in place of
%% stdio and drives it frame by frame (the CT node is nonode@nohost,
%% so the scenarios run inside the client peer). One case spawns the
%% CLI as an OS process over real stdio pipes.
%% @end
-module(erllama_server_acp_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1, init_per_testcase/2, end_per_testcase/2]).
-export([
    initialize_handshake/1,
    session_new_exposes_models_and_modes/1,
    prompt_streams_updates_and_answers/1,
    permission_request_is_answered/1,
    cancel_answers_the_prompt_as_cancelled/1,
    set_config_option_switches_model/1,
    set_mode_switches_policy/1,
    list_load_and_unknown_session/1,
    delete_removes_the_session/1,
    cli_over_stdio_e2e/1
]).
%% Run inside the client peer.
-export([
    tc_initialize/2,
    tc_new_session/2,
    tc_prompt/3,
    tc_permission/3,
    tc_cancel/3,
    tc_set_config/2,
    tc_set_mode/2,
    tc_list_load/2,
    tc_delete/2
]).

-define(COOKIE, "erllama_acp_suite").
-define(TIMEOUT, 30000).

all() ->
    [
        initialize_handshake,
        session_new_exposes_models_and_modes,
        prompt_streams_updates_and_answers,
        permission_request_is_answered,
        cancel_answers_the_prompt_as_cancelled,
        set_config_option_switches_model,
        set_mode_switches_policy,
        list_load_and_unknown_session,
        delete_removes_the_session,
        cli_over_stdio_e2e
    ].

%% -----------------------------------------------------------------------------
%% Setup
%% -----------------------------------------------------------------------------

init_per_suite(Config) ->
    {ok, Rel, RelNode} = start_peer(erllama_server_acp_rel, []),
    ok = peer:call(Rel, application, set_env, [erllama_server, data_dir, ?config(priv_dir, Config)]),
    {ok, _} = peer:call(Rel, application, ensure_all_started, [erllama_server]),
    {ok, _} = rcall(Rel, erllama, load_model, [<<"fast">>, stub(1)]),
    {ok, _} = rcall(Rel, erllama, load_model, [<<"slow">>, stub(20)]),
    ok = peer:call(Rel, application, set_env, [
        erllama_server, agent_generator, erllama_server_agent_script
    ]),
    Ws = filename:join(?config(priv_dir, Config), "ws"),
    ok = filelib:ensure_path(Ws),
    [{rel, Rel}, {rel_node, RelNode}, {ws, Ws} | Config].

end_per_suite(Config) ->
    peer:stop(?config(rel, Config)),
    ok.

init_per_testcase(_TC, Config) ->
    {ok, Cli, _} = start_peer(erllama_server_acp_cli, ["-hidden"]),
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
    ?assertEqual([], Leaked),
    ok.

%% erllama library modules the client node must never load.
leaked(Cli) ->
    Loaded = [atom_to_list(M) || {M, _} <- peer:call(Cli, code, all_loaded, [])],
    [
        M
     || "erllama" ++ _ = M <- Loaded,
        not lists:member(M, [
            "erllama_server_client",
            "erllama_server_acp",
            "erllama_server_acp_map",
            "erllama_server_acp_stdio",
            "erllama_server_acp_session",
            atom_to_list(?MODULE)
        ])
    ].

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

run_tc(Config, Fun, Args) ->
    peer:call(
        ?config(cli, Config),
        ?MODULE,
        Fun,
        [?config(rel_node, Config), ?config(ws, Config) | Args],
        60000
    ).

%% -----------------------------------------------------------------------------
%% Cases (thin wrappers running the scenario inside the client peer)
%% -----------------------------------------------------------------------------

initialize_handshake(Config) -> run_tc(Config, tc_initialize, []).
session_new_exposes_models_and_modes(Config) -> run_tc(Config, tc_new_session, []).
prompt_streams_updates_and_answers(Config) ->
    run_tc(Config, tc_prompt, [#{text => <<"hello there">>}]).
permission_request_is_answered(Config) ->
    run_tc(Config, tc_permission, [
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
    ]).
cancel_answers_the_prompt_as_cancelled(Config) ->
    run_tc(Config, tc_cancel, [{hang, <<"working">>}]).
set_config_option_switches_model(Config) -> run_tc(Config, tc_set_config, []).
set_mode_switches_policy(Config) -> run_tc(Config, tc_set_mode, []).
list_load_and_unknown_session(Config) -> run_tc(Config, tc_list_load, []).
delete_removes_the_session(Config) -> run_tc(Config, tc_delete, []).

%% -----------------------------------------------------------------------------
%% Scenarios, run inside the client peer (sink = this process)
%% -----------------------------------------------------------------------------

req(Id, Method, Params) ->
    #{<<"jsonrpc">> => <<"2.0">>, <<"id">> => Id, <<"method">> => Method, <<"params">> => Params}.

start_acp(RelNode) ->
    %% Like `erllama acp --model fast': the release has no registry
    %% default, so new sessions name their model.
    start_acp(RelNode, #{model => <<"fast">>}).

start_acp(RelNode, Defaults) ->
    erllama_server_acp:start(RelNode, Defaults, #{caller => self(), sink => self()}).

send(Acp, Frame) ->
    erllama_server_acp:handle_frame(Acp, Frame).

wait_frame(Pred) ->
    wait_frame(Pred, [], ?TIMEOUT).

wait_frame(Pred, Seen, Timeout) ->
    receive
        {acp_out, _Pid, F} ->
            case Pred(F) of
                true -> {F, lists:reverse(Seen)};
                false -> wait_frame(Pred, [F | Seen], Timeout)
            end
    after Timeout ->
        ct:fail({no_frame, Seen})
    end.

is_response(Id) ->
    fun(F) -> maps:get(<<"id">>, F, undefined) =:= Id andalso is_map_key(<<"result">>, F) end.

is_error(Id) ->
    fun(F) -> maps:get(<<"id">>, F, undefined) =:= Id andalso is_map_key(<<"error">>, F) end.

err_code(F) ->
    maps:get(<<"code">>, maps:get(<<"error">>, F)).

%% Wait for the response frames of each id in `Ids', in any order
%% (cancellation answers both the prompt and the cancel request).
wait_ids(Ids) ->
    wait_ids(Ids, #{}, []).

wait_ids(Want, Found, Seen) ->
    case Want -- maps:keys(Found) of
        [] ->
            {Found, Seen};
        Missing ->
            receive
                {acp_out, _Pid, F} ->
                    Id = maps:get(<<"id">>, F, undefined),
                    case lists:member(Id, Want) of
                        true -> wait_ids(Want, Found#{Id => F}, Seen);
                        false -> wait_ids(Want, Found, [F | Seen])
                    end
            after ?TIMEOUT ->
                ct:fail({no_frame, Missing, Seen})
            end
    end.

initialize(Acp, Id) ->
    send(Acp, req(Id, <<"initialize">>, #{<<"protocolVersion">> => 1})),
    {F, _} = wait_frame(is_response(Id)),
    maps:get(<<"result">>, F).

new_session(Acp, Id, Ws) ->
    send(Acp, req(Id, <<"session/new">>, #{<<"cwd">> => unicode:characters_to_binary(Ws)})),
    {F, _} = wait_frame(is_response(Id)),
    maps:get(<<"result">>, F).

prompt(Acp, Id, Sid, Text) ->
    send(
        Acp,
        req(Id, <<"session/prompt">>, #{
            <<"sessionId">> => Sid,
            <<"prompt">> => [#{<<"type">> => <<"text">>, <<"text">> => Text}]
        })
    ).

agent_for(RelNode, Sid) ->
    {ok, Agents} = erllama_server_client:call(RelNode, agents, []),
    [A] = [A || A <- Agents, maps:get(conversation_id, A, undefined) =:= Sid],
    maps:get(run_id, A).

set_script(RelNode, AgentId, Script) ->
    erpc:call(RelNode, erllama_server_agent_script, set, [AgentId, Script]).

stop_acp(Acp) ->
    erllama_server_acp:stop(Acp),
    receive
        {acp_done, Acp} -> ok
    after 5000 ->
        ct:fail(acp_did_not_stop)
    end.

tc_initialize(RelNode, _Ws) ->
    Acp = start_acp(RelNode),
    send(Acp, req(1, <<"session/list">>, #{})),
    {F1, _} = wait_frame(is_error(1)),
    ?assertEqual(-32002, err_code(F1)),
    Result = initialize(Acp, 2),
    ?assertEqual(1, maps:get(<<"protocolVersion">>, Result)),
    ?assertEqual(true, maps:get(<<"loadSession">>, maps:get(<<"agentCapabilities">>, Result))),
    send(Acp, req(3, <<"bogus/method">>, #{})),
    {F3, _} = wait_frame(is_error(3)),
    ?assertEqual(-32601, err_code(F3)),
    stop_acp(Acp),
    ok.

tc_new_session(RelNode, Ws) ->
    Acp = start_acp(RelNode),
    initialize(Acp, 1),
    Result = new_session(Acp, 2, Ws),
    Sid = maps:get(<<"sessionId">>, Result),
    ?assert(is_binary(Sid)),
    [ModelOpt] = maps:get(<<"configOptions">>, Result),
    ?assertEqual(<<"model">>, maps:get(<<"id">>, ModelOpt)),
    Values = [maps:get(<<"value">>, O) || O <- maps:get(<<"options">>, ModelOpt)],
    ?assert(lists:member(<<"fast">>, Values)),
    ?assert(lists:member(<<"slow">>, Values)),
    Modes = maps:get(<<"modes">>, Result),
    ?assertEqual(<<"default">>, maps:get(<<"currentModeId">>, Modes)),
    send(Acp, req(3, <<"session/close">>, #{<<"sessionId">> => Sid})),
    {F3, _} = wait_frame(is_response(3)),
    ?assertEqual(#{}, maps:get(<<"result">>, F3)),
    stop_acp(Acp),
    ok.

tc_prompt(RelNode, Ws, Script) ->
    Acp = start_acp(RelNode),
    initialize(Acp, 1),
    Result = new_session(Acp, 2, Ws),
    Sid = maps:get(<<"sessionId">>, Result),
    AgentId = agent_for(RelNode, Sid),
    ok = set_script(RelNode, AgentId, [Script]),
    prompt(Acp, 3, Sid, <<"hi">>),
    {Resp, Seen} = wait_frame(is_response(3)),
    ?assertEqual(<<"end_turn">>, maps:get(<<"stopReason">>, maps:get(<<"result">>, Resp))),
    Updates = [
        maps:get(<<"update">>, maps:get(<<"params">>, F))
     || F <- Seen, maps:get(<<"method">>, F, <<>>) =:= <<"session/update">>
    ],
    Kinds = [maps:get(<<"sessionUpdate">>, U) || U <- Updates],
    ?assert(lists:member(<<"user_message_chunk">>, Kinds)),
    ?assert(lists:member(<<"agent_message_chunk">>, Kinds)),
    stop_acp(Acp),
    ok.

tc_permission(RelNode, Ws, Script) ->
    Acp = start_acp(RelNode),
    initialize(Acp, 1),
    Result = new_session(Acp, 2, Ws),
    Sid = maps:get(<<"sessionId">>, Result),
    AgentId = agent_for(RelNode, Sid),
    ok = set_script(RelNode, AgentId, Script),
    prompt(Acp, 3, Sid, <<"write the note">>),
    {PermReq, _} = wait_frame(
        fun(F) -> maps:get(<<"method">>, F, <<>>) =:= <<"session/request_permission">> end
    ),
    ReqId = maps:get(<<"id">>, PermReq),
    Params = maps:get(<<"params">>, PermReq),
    ?assertEqual(Sid, maps:get(<<"sessionId">>, Params)),
    ToolCall = maps:get(<<"toolCall">>, Params),
    ?assertEqual(<<"w1">>, maps:get(<<"toolCallId">>, ToolCall)),
    ?assertEqual(<<"edit">>, maps:get(<<"kind">>, ToolCall)),
    send(
        Acp,
        #{
            <<"jsonrpc">> => <<"2.0">>,
            <<"id">> => ReqId,
            <<"result">> => #{
                <<"outcome">> => #{
                    <<"outcome">> => <<"selected">>, <<"optionId">> => <<"allow_once">>
                }
            }
        }
    ),
    {Resp, Seen} = wait_frame(is_response(3)),
    ?assertEqual(<<"end_turn">>, maps:get(<<"stopReason">>, maps:get(<<"result">>, Resp))),
    Updates = [
        maps:get(<<"update">>, maps:get(<<"params">>, F))
     || F <- Seen, maps:get(<<"method">>, F, <<>>) =:= <<"session/update">>
    ],
    Kinds = [maps:get(<<"sessionUpdate">>, U) || U <- Updates],
    ?assert(lists:member(<<"tool_call">>, Kinds)),
    ?assert(lists:member(<<"tool_call_update">>, Kinds)),
    ?assertEqual({ok, <<"hi">>}, file:read_file(filename:join(Ws, "note.txt"))),
    stop_acp(Acp),
    ok.

tc_cancel(RelNode, Ws, Script) ->
    Acp = start_acp(RelNode),
    initialize(Acp, 1),
    Result = new_session(Acp, 2, Ws),
    Sid = maps:get(<<"sessionId">>, Result),
    AgentId = agent_for(RelNode, Sid),
    ok = set_script(RelNode, AgentId, [Script]),
    prompt(Acp, 3, Sid, <<"hi">>),
    %% Wait for the turn to start before cancelling: the cancel must
    %% target an in-flight prompt.
    {_, _} = wait_frame(fun(F) -> maps:get(<<"method">>, F, <<>>) =:= <<"session/update">> end),
    send(Acp, req(4, <<"session/cancel">>, #{<<"sessionId">> => Sid})),
    {Found, _} = wait_ids([3, 4]),
    ?assertEqual(2, maps:size(Found)),
    ?assertEqual(
        <<"cancelled">>,
        maps:get(<<"stopReason">>, maps:get(<<"result">>, maps:get(3, Found)))
    ),
    ?assertEqual(#{}, maps:get(<<"result">>, maps:get(4, Found))),
    stop_acp(Acp),
    ok.

tc_set_config(RelNode, Ws) ->
    Acp = start_acp(RelNode),
    initialize(Acp, 1),
    Result = new_session(Acp, 2, Ws),
    Sid = maps:get(<<"sessionId">>, Result),
    AgentId = agent_for(RelNode, Sid),
    send(
        Acp,
        req(3, <<"session/set_config_option">>, #{
            <<"sessionId">> => Sid, <<"configId">> => <<"model">>, <<"value">> => <<"slow">>
        })
    ),
    {F3, _} = wait_frame(is_response(3)),
    ?assertEqual(#{}, maps:get(<<"result">>, F3)),
    {ok, Agents} = erllama_server_client:call(RelNode, agents, []),
    [A] = [A || A <- Agents, maps:get(run_id, A) =:= AgentId],
    ?assertEqual(<<"slow">>, maps:get(model, A)),
    send(
        Acp,
        req(4, <<"session/set_config_option">>, #{
            <<"sessionId">> => Sid, <<"configId">> => <<"model">>, <<"value">> => <<"nope">>
        })
    ),
    {F4, _} = wait_frame(is_error(4)),
    ?assertEqual(-32602, err_code(F4)),
    stop_acp(Acp),
    ok.

tc_set_mode(RelNode, Ws) ->
    Acp = start_acp(RelNode),
    initialize(Acp, 1),
    Result = new_session(Acp, 2, Ws),
    Sid = maps:get(<<"sessionId">>, Result),
    send(Acp, req(3, <<"session/set_mode">>, #{<<"sessionId">> => Sid, <<"modeId">> => <<"yes">>})),
    {F3, Seen} = wait_frame(is_response(3)),
    ?assertEqual(#{}, maps:get(<<"result">>, F3)),
    ModeUpdates = [
        maps:get(<<"update">>, maps:get(<<"params">>, F))
     || F <- Seen,
        maps:get(<<"method">>, F, <<>>) =:= <<"session/update">>,
        maps:get(<<"sessionUpdate">>, maps:get(<<"update">>, maps:get(<<"params">>, F)), <<>>) =:=
            <<"current_mode_update">>
    ],
    ?assertMatch([#{<<"currentModeId">> := <<"yes">>} | _], ModeUpdates),
    send(
        Acp, req(4, <<"session/set_mode">>, #{<<"sessionId">> => Sid, <<"modeId">> => <<"bogus">>})
    ),
    {F4, _} = wait_frame(is_error(4)),
    ?assertEqual(-32602, err_code(F4)),
    stop_acp(Acp),
    ok.

tc_list_load(RelNode, Ws) ->
    Acp = start_acp(RelNode),
    initialize(Acp, 1),
    Result = new_session(Acp, 2, Ws),
    Sid = maps:get(<<"sessionId">>, Result),
    send(Acp, req(3, <<"session/list">>, #{})),
    {F3, _} = wait_frame(is_response(3)),
    Sessions = maps:get(<<"sessions">>, maps:get(<<"result">>, F3)),
    [S] = [S || S <- Sessions, maps:get(<<"sessionId">>, S) =:= Sid],
    ?assertEqual(unicode:characters_to_binary(Ws), maps:get(<<"cwd">>, S)),
    ?assert(is_binary(maps:get(<<"updatedAt">>, S))),
    send(
        Acp,
        req(4, <<"session/load">>, #{<<"sessionId">> => <<"conv_nope">>, <<"cwd">> => <<"/tmp">>})
    ),
    {F4, _} = wait_frame(is_error(4)),
    ?assertEqual(-32000, err_code(F4)),
    prompt(Acp, 5, <<"conv_nope">>, <<"hi">>),
    {F5, _} = wait_frame(is_error(5)),
    ?assertEqual(-32000, err_code(F5)),
    %% Close, then load: the conversation resumes with its workspace.
    send(Acp, req(6, <<"session/close">>, #{<<"sessionId">> => Sid})),
    {_, _} = wait_frame(is_response(6)),
    send(Acp, req(7, <<"session/load">>, #{<<"sessionId">> => Sid, <<"cwd">> => <<"/tmp">>})),
    {F7, _} = wait_frame(is_response(7)),
    Loaded = maps:get(<<"result">>, F7),
    ?assertEqual(Sid, maps:get(<<"sessionId">>, Loaded)),
    ?assert(is_map_key(<<"configOptions">>, Loaded)),
    AgentId = agent_for(RelNode, Sid),
    {ok, Agents} = erllama_server_client:call(RelNode, agents, []),
    [A] = [A || A <- Agents, maps:get(run_id, A) =:= AgentId],
    ?assertEqual(unicode:characters_to_binary(Ws), maps:get(workspace, A)),
    stop_acp(Acp),
    ok.

tc_delete(RelNode, Ws) ->
    Acp = start_acp(RelNode),
    initialize(Acp, 1),
    Result = new_session(Acp, 2, Ws),
    Sid = maps:get(<<"sessionId">>, Result),
    send(Acp, req(3, <<"session/delete">>, #{<<"sessionId">> => Sid})),
    {F3, _} = wait_frame(is_response(3)),
    ?assertEqual(#{}, maps:get(<<"result">>, F3)),
    send(Acp, req(4, <<"session/list">>, #{})),
    {F4, _} = wait_frame(is_response(4)),
    Sessions = maps:get(<<"sessions">>, maps:get(<<"result">>, F4)),
    ?assertEqual([], [S || S <- Sessions, maps:get(<<"sessionId">>, S) =:= Sid]),
    send(Acp, req(5, <<"session/load">>, #{<<"sessionId">> => Sid, <<"cwd">> => <<"/tmp">>})),
    {F5, _} = wait_frame(is_error(5)),
    ?assertEqual(-32000, err_code(F5)),
    stop_acp(Acp),
    ok.

%% -----------------------------------------------------------------------------
%% The CLI over real stdio pipes (runs on the CT node: port IO is local)
%% -----------------------------------------------------------------------------

cli_over_stdio_e2e(Config) ->
    RelNode = ?config(rel_node, Config),
    Ws = ?config(ws, Config),
    Priv = ?config(priv_dir, Config),
    Erl = filename:join([code:root_dir(), "bin", "erl"]),
    Ebin = filename:dirname(code:which(erllama_server_cli)),
    %% Framed input, precomputed: initialize, then session/new whose
    %% cwd is the workspace. stdin ends after it, as when an ACP
    %% client goes away, so the CLI must exit 0 having answered both.
    In = filename:join(Priv, "acp_in.framed"),
    Out = filename:join(Priv, "acp_out.framed"),
    Err = filename:join(Priv, "acp_err.txt"),
    Frames = [
        req(1, <<"initialize">>, #{<<"protocolVersion">> => 1}),
        req(2, <<"session/new">>, #{<<"cwd">> => unicode:characters_to_binary(Ws)})
    ],
    ok = file:write_file(In, iolist_to_binary([erllama_server_acp_stdio:frame(F) || F <- Frames])),
    %% Production `main' reroutes logger to standard error so stdout
    %% only ever carries framed messages; the spawned VM does the same
    %% (the fd port's boot report would otherwise land on stdout).
    Expr =
        "ok = logger:remove_handler(default), "
        "ok = logger:add_handler(default, logger_std_h, #{config => #{type => standard_error}}), "
        "R = (catch erllama_server_cli:run([\"acp\", \"--model\", \"fast\"])), "
        "case R of {0, _} -> halt(0); _ -> halt(1) end.",
    Cmd = lists:flatten(
        io_lib:format(
            "'~s' -noshell -pa '~s' -eval '~s' < '~s' > '~s' 2> '~s'",
            [Erl, Ebin, Expr, In, Out, Err]
        )
    ),
    Port = open_port(
        {spawn_executable, os:find_executable("bash")},
        [
            {args, ["-c", Cmd]},
            {env, [
                {"ERLLAMA_NODE", atom_to_list(RelNode)},
                {"ERLLAMA_COOKIE", ?COOKIE},
                {"ERLLAMA_TIMEOUT", "15000"}
            ]},
            exit_status,
            binary,
            hide
        ]
    ),
    0 = wait_exit(Port),
    {ok, OutData} = file:read_file(Out),
    ?assertEqual(2, count_frames(OutData)),
    {ok, F1, Rest} = erllama_server_acp_stdio:parse(OutData),
    {ok, R1} = erllama_server_acp_stdio:decode(F1),
    ?assertEqual(1, maps:get(<<"protocolVersion">>, maps:get(<<"result">>, R1))),
    {ok, F2, <<>>} = erllama_server_acp_stdio:parse(Rest),
    {ok, R2} = erllama_server_acp_stdio:decode(F2),
    Result = maps:get(<<"result">>, R2),
    ?assert(is_binary(maps:get(<<"sessionId">>, Result))),
    [ModelOpt] = maps:get(<<"configOptions">>, Result),
    Values = [maps:get(<<"value">>, O) || O <- maps:get(<<"options">>, ModelOpt)],
    ?assert(lists:member(<<"fast">>, Values)),
    ?assertEqual(<<"default">>, maps:get(<<"currentModeId">>, maps:get(<<"modes">>, Result))),
    ok.

count_frames(Bin) -> count_frames(Bin, 0).
count_frames(<<>>, N) ->
    N;
count_frames(Bin, N) ->
    case erllama_server_acp_stdio:parse(Bin) of
        {ok, _, Rest} -> count_frames(Rest, N + 1);
        {error, _} when byte_size(Bin) > 0 -> count_frames(skip_byte(Bin), N);
        _ -> N
    end.

skip_byte(<<_, Rest/binary>>) -> Rest;
skip_byte(Bin) -> Bin.

wait_exit(Port) ->
    receive
        {Port, {exit_status, S}} -> S
    after 60000 ->
        ct:fail(port_did_not_exit)
    end.
