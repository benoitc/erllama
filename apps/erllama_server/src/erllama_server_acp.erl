%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_acp).
-moduledoc false.
%% The ACP agent dispatcher for `erllama acp': one process owning the
%% session table and the pending permission answers. Frames from the
%% stdio reader (or `handle_frame/2' in tests) arrive here; each
%% request runs in its own spawned handler so a prompt never blocks a
%% cancel; per-session follower processes stream the release's job
%% events back as `{acp_event, Sid, Event}'. The release is only ever
%% talked to through `erllama_server_client' (erpc + follow), so this
%% module never loads an erllama library module.
%%
%% An ACP session id is the release's conversation id: stable across
%% restarts, which is what makes `session/load' work.

-export([start/3, handle_frame/2, stop/1]).

-record(st, {
    node :: node(),
    caller :: pid(),
    defaults = #{} :: map(),
    writer :: pid() | undefined,
    sink :: pid() | undefined,
    sessions = #{} :: #{binary() => sess()},
    pending = #{} :: #{term() => {binary(), binary()}},
    next_req = 1 :: pos_integer(),
    initialized = false :: boolean(),
    handlers = #{} :: #{pid() => reference()}
}).

-record(sess, {
    agent_id :: binary(),
    cwd :: binary(),
    model :: binary(),
    mode = default :: erllama_server_acp_map:mode(),
    follower :: pid() | undefined,
    prompts = [] :: [term()]
}).

-type sess() :: #sess{}.

-define(CALL_TIMEOUT, 30000).
-define(SESSION_NOT_FOUND, -32000).
-define(NOT_INITIALIZED, -32002).

%% Start the agent. `Opts': `caller' (gets `{acp_done, Pid}' at end of
%% input), and for tests `sink' (receives `{acp_out, Pid, Frame}'
%% instead of writing to standard output).
-spec start(node(), map(), map()) -> pid().
start(Node, Defaults, Opts) ->
    Caller = maps:get(caller, Opts),
    %% Not linked: the caller (the CLI process, or a test) outlives the
    %% agent and watches it with a monitor instead.
    spawn(fun() -> init(Node, Defaults, Caller, Opts) end).

init(Node, Defaults, Caller, Opts) ->
    St0 = #st{node = Node, caller = Caller, defaults = Defaults},
    St =
        case maps:get(sink, Opts, undefined) of
            undefined ->
                %% The group leader is the stdio device even when the
                %% VM was spawned on a port (`io:put_chars' on the
                %% `standard_output' atom is refused there).
                Writer = erllama_server_acp_stdio:start_writer(group_leader()),
                _ = erllama_server_acp_stdio:start_reader(self()),
                St0#st{writer = Writer};
            Sink when is_pid(Sink) ->
                St0#st{sink = Sink}
        end,
    loop(St).

%% Feed one decoded frame (tests drive the agent this way).
-spec handle_frame(pid(), map()) -> ok.
handle_frame(Pid, Frame) ->
    Pid ! {frame, Frame},
    ok.

-spec stop(pid()) -> ok.
stop(Pid) ->
    Pid ! stop,
    ok.

%% -----------------------------------------------------------------------------
%% The loop
%% -----------------------------------------------------------------------------

loop(#st{} = St) ->
    receive
        {frame, Frame} ->
            loop(handle_incoming(self(), St, Frame));
        {out, Term} ->
            loop(emit(St, Term));
        {call, {From, Ref}, Req} ->
            {Reply, St1} = handle_call(Req, St),
            From ! {acp_reply, Ref, Reply},
            loop(St1);
        {acp_event, Sid, Event} ->
            loop(effects(St, Sid, erllama_server_acp_map:event_updates(Event)));
        eof ->
            end_of_input(St);
        {reader_error, _R} ->
            end_of_input(St);
        stop ->
            quit(St)
        %% The loop exits through `stop', `eof' or a reader error; it only
        %% ever blocks waiting for one of those messages.
    after infinity -> ok
    end.

%% stdin closed: burst input often leaves request handlers with
%% answers still in flight, so keep serving frames already accepted
%% until every handler is done, then leave.
end_of_input(St) ->
    receive
        {out, Term} ->
            end_of_input(emit(St, Term));
        {acp_event, Sid, Event} ->
            end_of_input(effects(St, Sid, erllama_server_acp_map:event_updates(Event)));
        {call, {From, Ref}, Req} ->
            {Reply, St1} = handle_call(Req, St),
            From ! {acp_reply, Ref, Reply},
            end_of_input(St1);
        {frame, Frame} ->
            end_of_input(handle_incoming(self(), St, Frame));
        {'DOWN', _Ref, process, Pid, _Reason} ->
            end_of_input(St#st{handlers = maps:remove(Pid, St#st.handlers)})
    after 5000 ->
        end_of_input(quit(St))
    end.

quit(#st{caller = Caller} = St) ->
    [
        exit(S#sess.follower, kill)
     || {_Sid, #sess{follower = F} = S} <- maps:to_list(St#st.sessions), is_pid(F)
    ],
    Caller ! {acp_done, self()},
    ok.

emit(#st{sink = Sink} = St, Term) when is_pid(Sink) ->
    Sink ! {acp_out, self(), Term},
    St;
emit(#st{writer = Writer} = St, Term) ->
    _ = erllama_server_acp_stdio:send(Writer, Term),
    St.

%% Handlers and followers report frames through the dispatcher, so
%% output stays serialized.
out(D, Term) ->
    D ! {out, Term},
    ok.

handle_incoming(_D, St, {invalid_json, _Body}) ->
    emit(St, erllama_server_acp_map:error(null, -32700, invalid_json));
handle_incoming(D, St, Frame) ->
    case erllama_server_acp_map:incoming(Frame) of
        {request, Id, Method, Params} ->
            handle_request(D, St, Id, Method, Params);
        {notification, Method, Params} ->
            handle_notification(D, St, Method, Params);
        {response, Id, Result} ->
            %% Inline: this runs in the dispatcher itself, so state is
            %% read directly (call/2 to self would deadlock).
            handle_response(St, Id, Result);
        {error_response, Id, _Code, _E} ->
            St#st{pending = maps:remove(Id, St#st.pending)};
        invalid ->
            emit(St, erllama_server_acp_map:error(null, -32600, invalid_request))
    end.

handle_request(D, St, Id, <<"initialize">> = Method, Params) ->
    (spawn_handler(D, St, Id, Method, Params))#st{initialized = true};
handle_request(_D, #st{initialized = false} = St, Id, _Method, _Params) ->
    _ = emit(St, erllama_server_acp_map:error(Id, ?NOT_INITIALIZED, not_initialized)),
    St;
handle_request(D, St, Id, Method, Params) ->
    spawn_handler(D, St, Id, Method, Params).

spawn_handler(D, St, Id, Method, Params) ->
    Pid = spawn(fun() -> request(D, Id, Method, Params, St) end),
    Ref = erlang:monitor(process, Pid),
    St#st{handlers = maps:put(Pid, Ref, St#st.handlers)}.

handle_notification(D, St, <<"session/cancel">>, Params) ->
    _ = method(D, St#st.node, St#st.defaults, undefined, <<"session/cancel">>, Params),
    St;
handle_notification(_D, St, _Method, _Params) ->
    St.

handle_response(St, Id, Result) ->
    case maps:get(Id, St#st.pending, undefined) of
        {AgentId, CallId} ->
            Decision = erllama_server_acp_map:permission_decision(Result),
            _ = erllama_server_client:call(St#st.node, agent_approve, [AgentId, CallId, Decision]),
            St#st{pending = maps:remove(Id, St#st.pending)};
        undefined ->
            St
    end.

%% -----------------------------------------------------------------------------
%% Requests, each in its own process
%% -----------------------------------------------------------------------------

request(D, Id, Method, Params, St) ->
    Node = St#st.node,
    Defaults = St#st.defaults,
    try
        case method(D, Node, Defaults, Id, Method, Params) of
            ok ->
                ok;
            {reply, Result} when Id =/= undefined ->
                out(D, erllama_server_acp_map:response(Id, Result));
            {reply, _} ->
                ok;
            {error, Code, R} when Id =/= undefined ->
                out(D, erllama_server_acp_map:error(Id, Code, fmt(R)));
            {error, _, _} ->
                ok
        end
    catch
        C:Err:Stack when Id =/= undefined ->
            out(D, erllama_server_acp_map:error(Id, -32603, fmt({C, Err, Stack})));
        _C:_Err:_Stack ->
            ok
    end.

method(_D, _Node, _Defaults, _Id, <<"initialize">>, _Params) ->
    {reply, erllama_server_acp_map:initialize_result()};
method(_D, _Node, _Defaults, _Id, <<"authenticate">>, _Params) ->
    %% Local models: nothing to authenticate.
    {reply, #{}};
method(_D, _Node, _Defaults, _Id, <<"logout">>, _Params) ->
    {reply, #{}};
method(D, Node, Defaults, _Id, <<"session/new">>, Params) ->
    case erllama_server_acp_map:new_session_spec(Params, Defaults) of
        {ok, Spec} -> start_session(D, Node, Spec);
        {error, R} -> {error, -32602, R}
    end;
method(D, Node, _Defaults, _Id, <<"session/load">>, Params) ->
    case erllama_server_acp_map:load_session_spec(Params) of
        {ok, #{conversation_id := Sid}} ->
            case call(D, {session, Sid}) of
                {ok, #sess{} = Sess} ->
                    {reply, session_response(Node, Sid, Sess)};
                undefined ->
                    %% Attach to the live agent holding the
                    %% conversation, or resume the conversation.
                    case erllama_server_client:call(Node, agent_load, [Sid]) of
                        {ok, #{agent_id := AgentId, model := Model, workspace := Ws}} ->
                            register_loaded(D, Node, Sid, AgentId, Model, Ws);
                        {error, _} ->
                            {error, ?SESSION_NOT_FOUND, {session_not_found, Sid}}
                    end
            end;
        {error, R} ->
            {error, -32602, R}
    end;
method(_D, Node, _Defaults, _Id, <<"session/list">>, _Params) ->
    case erllama_server_client:call(Node, agent_sessions, []) of
        {ok, Sessions} ->
            {reply, #{<<"sessions">> => [erllama_server_acp_map:session_info(S) || S <- Sessions]}};
        {error, R} ->
            {error, -32603, R}
    end;
method(D, Node, _Defaults, Id, <<"session/prompt">>, Params) ->
    case maps:get(<<"sessionId">>, Params, undefined) of
        Sid when is_binary(Sid) ->
            case call(D, {session, Sid}) of
                {ok, #sess{agent_id = AgentId}} ->
                    prompt(D, Node, Id, Sid, AgentId, Params);
                undefined ->
                    {error, ?SESSION_NOT_FOUND, {session_not_found, Sid}}
            end;
        _ ->
            {error, -32602, missing_session_id}
    end;
method(D, Node, _Defaults, _Id, <<"session/cancel">>, Params) ->
    with_session(D, Params, fun(AgentId) ->
        Sid = maps:get(<<"sessionId">>, Params),
        _ = erllama_server_client:call(Node, agent_interrupt, [AgentId]),
        %% The interrupted turn answers the oldest queued prompt on its
        %% `turn_done'; when the turn had not started yet there is no
        %% `turn_done' coming, so answer it here.
        case call(D, {pop_prompt, Sid}) of
            undefined ->
                ok;
            PromptId ->
                out(
                    D,
                    erllama_server_acp_map:response(PromptId, #{
                        <<"stopReason">> => <<"cancelled">>
                    })
                )
        end,
        {reply, #{}}
    end);
method(D, Node, _Defaults, _Id, <<"session/set_mode">>, Params) ->
    with_session(D, Params, fun(AgentId) ->
        Sid = maps:get(<<"sessionId">>, Params),
        case erllama_server_acp_map:mode_from_id(maps:get(<<"modeId">>, Params, <<>>)) of
            {ok, Mode} ->
                case erllama_server_client:call(Node, agent_set_policy, [AgentId, Mode]) of
                    ok ->
                        _ = call(D, {set_mode, Sid, Mode}),
                        out(
                            D,
                            erllama_server_acp_map:notification(<<"session/update">>, #{
                                <<"sessionId">> => Sid,
                                <<"update">> => #{
                                    <<"sessionUpdate">> => <<"current_mode_update">>,
                                    <<"currentModeId">> => erllama_server_acp_map:mode_id(Mode)
                                }
                            })
                        ),
                        {reply, #{}};
                    {error, R} ->
                        {error, -32603, R}
                end;
            {error, R} ->
                {error, -32602, R}
        end
    end);
method(D, Node, _Defaults, _Id, <<"session/set_config_option">>, Params) ->
    with_session(D, Params, fun(AgentId) ->
        Sid = maps:get(<<"sessionId">>, Params),
        ConfigId = maps:get(<<"configId">>, Params, <<>>),
        Value = maps:get(<<"value">>, Params, undefined),
        case {ConfigId, Value} of
            {<<"model">>, V} when is_binary(V) ->
                set_config_model(D, Node, Sid, AgentId, V);
            {<<"model">>, _} ->
                {error, -32602, invalid_model_value};
            _ ->
                {error, -32602, {unknown_config_option, ConfigId}}
        end
    end);
method(D, Node, _Defaults, _Id, <<"session/close">>, Params) ->
    with_session(D, Params, fun(AgentId) ->
        _ = erllama_server_client:call(Node, agent_stop, [AgentId]),
        Sid = maps:get(<<"sessionId">>, Params),
        _ = call(D, {remove_session, Sid}),
        {reply, #{}}
    end);
method(D, Node, _Defaults, _Id, <<"session/delete">>, Params) ->
    with_session(D, Params, fun(AgentId) ->
        Sid = maps:get(<<"sessionId">>, Params),
        _ = erllama_server_client:call(Node, agent_stop, [AgentId]),
        _ = erllama_server_client:call(Node, conversation_delete, [Sid]),
        _ = call(D, {remove_session, Sid}),
        {reply, #{}}
    end);
method(_D, _Node, _Defaults, _Id, _Method, _Params) ->
    {error, -32601, method_not_found}.

with_session(D, Params, Fun) ->
    case maps:get(<<"sessionId">>, Params, undefined) of
        Sid when is_binary(Sid) ->
            case call(D, {session, Sid}) of
                {ok, #sess{agent_id = AgentId}} -> Fun(AgentId);
                undefined -> {error, ?SESSION_NOT_FOUND, {session_not_found, Sid}}
            end;
        _ ->
            {error, -32602, missing_session_id}
    end.

%% Queue the prompt before sending: the turn may end before
%% `agent_send' returns, and a `turn_done' that finds no queued prompt
%% is dropped.
prompt(D, Node, Id, Sid, AgentId, Params) ->
    case erllama_server_acp_map:prompt_text(Params) of
        {ok, Text} ->
            _ = call(D, {add_prompt, Sid, Id}),
            case erllama_server_client:call(Node, agent_send, [AgentId, Text]) of
                ok ->
                    ok;
                {queued, _} ->
                    ok;
                {error, R} ->
                    _ = call(D, {remove_prompt, Sid, Id}),
                    {error, -32603, R}
            end;
        {error, R} ->
            {error, -32602, R}
    end.

%% `session/set_config_option' for the `model' option: the model must
%% be known to the server before it can drive the session.
set_config_model(D, Node, Sid, AgentId, V) ->
    case erllama_server_client:call(Node, model, [V]) of
        {ok, _} ->
            case erllama_server_client:call(Node, agent_set_model, [AgentId, V]) of
                ok ->
                    _ = call(D, {set_model, Sid, V}),
                    {reply, #{}};
                {error, R} ->
                    {error, -32603, R}
            end;
        {error, _} ->
            {error, -32602, {unknown_model, V}}
    end.

start_session(D, Node, Spec) ->
    case erllama_server_client:call(Node, agent_start, [Spec]) of
        {ok, #{agent_id := AgentId, conversation_id := Conv} = R} ->
            Mode = maps:get(policy, Spec, default),
            Model = maps:get(model, R, default_model(Node)),
            Ws = maps:get(workspace, R, maps:get(workspace, Spec)),
            register_loaded(D, Node, Conv, AgentId, Model, Ws, Mode);
        {error, {missing, workspace}} ->
            {error, -32602, missing_workspace};
        {error, {workspace, R}} ->
            {error, -32602, {workspace, R}};
        {error, R} ->
            {error, -32603, R}
    end.

register_loaded(D, Node, Conv, AgentId, Model, Ws) ->
    register_loaded(D, Node, Conv, AgentId, Model, Ws, default).

register_loaded(D, Node, Conv, AgentId, Model, Ws, Mode) ->
    Sess = #sess{agent_id = AgentId, cwd = Ws, model = Model, mode = Mode},
    Follower = erllama_server_acp_session:start(D, Node, Conv, AgentId, 0),
    _ = call(D, {register_session, Conv, Sess#sess{follower = Follower}}),
    {reply, session_response(Node, Conv, Sess#sess{follower = Follower})}.

session_response(Node, Sid, #sess{mode = Mode, model = Model}) ->
    {ok, Models} = erllama_server_client:call(Node, models, []),
    #{
        <<"sessionId">> => Sid,
        <<"modes">> => erllama_server_acp_map:modes(Mode),
        <<"configOptions">> => erllama_server_acp_map:config_options(Models, Model)
    }.

default_model(Node) ->
    case erllama_server_client:call(Node, models, []) of
        {ok, Models} ->
            case [N || #{name := N, default := true} <- Models] of
                [N | _] -> N;
                [] -> <<>>
            end;
        _ ->
            <<>>
    end.

%% -----------------------------------------------------------------------------
%% Events from a session's follower
%% -----------------------------------------------------------------------------

effects(St, _Sid, []) ->
    St;
effects(St, Sid, [Update | Rest]) when is_map(Update) ->
    effects(
        emit(
            St,
            erllama_server_acp_map:notification(<<"session/update">>, #{
                <<"sessionId">> => Sid, <<"update">> => Update
            })
        ),
        Sid,
        Rest
    );
effects(St, Sid, [{permission, Req} | Rest]) ->
    {ReqId, St1} = new_req_id(St),
    St2 = add_pending(ReqId, {agent_of(Sid, St1), maps:get(call_id, Req)}, St1),
    effects(
        emit(St2, erllama_server_acp_map:permission_request(ReqId, Sid, Req)),
        Sid,
        Rest
    );
effects(St, Sid, [{turn_done, Data} | Rest]) ->
    case pop_prompt(Sid, St) of
        {PromptId, St1} when PromptId =/= undefined ->
            effects(
                emit(
                    St1,
                    erllama_server_acp_map:response(PromptId, #{
                        <<"stopReason">> => erllama_server_acp_map:stop_reason(Data)
                    })
                ),
                Sid,
                Rest
            );
        {undefined, St1} ->
            effects(St1, Sid, Rest)
    end;
effects(St, Sid, [session_ended | Rest]) ->
    effects(St, Sid, Rest).

agent_of(Sid, St) ->
    case maps:get(Sid, St#st.sessions, undefined) of
        #sess{agent_id = AgentId} -> AgentId;
        undefined -> <<>>
    end.

%% -----------------------------------------------------------------------------
%% State access (called from handler processes)
%% -----------------------------------------------------------------------------

call(D, Req) ->
    call(D, Req, ?CALL_TIMEOUT).

call(D, Req, Timeout) ->
    Ref = make_ref(),
    D ! {call, {self(), Ref}, Req},
    receive
        {acp_reply, Ref, R} -> R
    after Timeout ->
        {error, timeout}
    end.

handle_call(get_node, St) ->
    {St#st.node, St};
handle_call(get_defaults, St) ->
    {St#st.defaults, St};
handle_call({session, Sid}, St) ->
    case maps:get(Sid, St#st.sessions, undefined) of
        undefined -> {undefined, St};
        Sess -> {{ok, Sess}, St}
    end;
handle_call({register_session, Sid, Sess}, St) ->
    {ok, put_session(Sid, Sess, St)};
handle_call({remove_session, Sid}, St) ->
    {ok, St#st{sessions = maps:remove(Sid, St#st.sessions)}};
handle_call({add_prompt, Sid, Id}, St) ->
    {ok, update_session(Sid, fun(S) -> S#sess{prompts = S#sess.prompts ++ [Id]} end, St)};
handle_call({remove_prompt, Sid, Id}, St) ->
    {ok, update_session(Sid, fun(S) -> S#sess{prompts = lists:delete(Id, S#sess.prompts)} end, St)};
handle_call({pop_prompt, Sid}, St) ->
    pop_prompt(Sid, St);
handle_call({set_mode, Sid, Mode}, St) ->
    {ok, update_session(Sid, fun(S) -> S#sess{mode = Mode} end, St)};
handle_call({set_model, Sid, Model}, St) ->
    {ok, update_session(Sid, fun(S) -> S#sess{model = Model} end, St)};
handle_call({new_req_id}, St) ->
    {St#st.next_req, St#st{next_req = St#st.next_req + 1}};
handle_call({add_pending, ReqId, Pend}, St) ->
    {ok, add_pending(ReqId, Pend, St)};
handle_call(_Other, St) ->
    {{error, unknown_call}, St}.

put_session(Sid, Sess, St) ->
    Sessions = St#st.sessions,
    St#st{sessions = Sessions#{Sid => Sess}}.

update_session(Sid, Fun, St) ->
    case maps:get(Sid, St#st.sessions, undefined) of
        undefined -> St;
        Sess -> put_session(Sid, Fun(Sess), St)
    end.

pop_prompt(Sid, St) ->
    case maps:get(Sid, St#st.sessions, undefined) of
        undefined ->
            {undefined, St};
        #sess{prompts = []} = Sess ->
            {undefined, put_session(Sid, Sess, St)};
        #sess{prompts = [Id | Rest]} = Sess ->
            {Id, put_session(Sid, Sess#sess{prompts = Rest}, St)}
    end.

new_req_id(St) ->
    {St#st.next_req, St#st{next_req = St#st.next_req + 1}}.

add_pending(ReqId, Pend, St) ->
    Pending = St#st.pending,
    St#st{pending = Pending#{ReqId => Pend}}.

fmt(R) when is_binary(R) -> R;
fmt(R) when is_atom(R) -> atom_to_binary(R);
fmt(R) -> unicode:characters_to_binary(io_lib:format("~p", [R])).
