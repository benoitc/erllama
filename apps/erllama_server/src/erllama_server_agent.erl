%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_agent).
-moduledoc false.
%% An agent session: a conversation worked on by a model with tools, in
%% a workspace on the release machine. The session holds its
%% conversation for its whole life and is the only writer of its
%% transcript; sessions on other conversations run concurrently.
%%
%% A turn starts with user input and loops: generate (the model may call
%% tools), ask for approval of the calls that need it, run the calls,
%% append their results, generate again, until the model answers
%% without calling a tool, a budget runs out, or the turn is
%% interrupted. Input that arrives during a turn waits for the next one.
%%
%%   idle -> generating -> (approving) -> tools -> generating ... -> idle
%%
%% Tool calls run in monitored worker processes. Calls that only read
%% run together; a call that changes the workspace waits for the calls
%% before it and holds back the ones after it, so mutations happen in
%% call order. Results are appended in call order. A failing or
%% crashing tool becomes an error result the model can act on.
%%
%% The tools of the configured MCP servers (erllama_server_mcp) join
%% the native ones. Each generation round takes a snapshot of their
%% catalog: the model is offered, and its calls resolve against, the
%% tools of that snapshot, so a server coming or going mid-round
%% changes nothing until the next one.
%%
%% The session is a job (erllama_server_events): clients attach,
%% detach and reattach to its events; it outlives them.

-behaviour(gen_statem).

-export([start_link/1, send/2, approve/3, interrupt/1, stop/1]).
-export([callback_mode/0, init/1, handle_event/4, terminate/3]).

%% How a session generates (erllama_server_agent_gen; tests script it):
%% start a round for `Session' over `Messages' and report as
%% `{agent_gen, Pid, {delta, Type, Data} | {done, Result} | {error, R}}'.
-callback start(Session :: pid(), Ctx :: map(), Messages :: [map()], Opts :: map()) -> pid().

-define(SYSTEM, <<
    "You are an agent working in the directory WORKSPACE on the user's machine. "
    "Use the tools to read, search and change files there and to run commands. "
    "Paths are relative to that directory. Read before you edit, make the smallest "
    "change that works, and check your work when you can. When you are done, answer "
    "the user briefly."
>>).

-record(call, {
    index :: non_neg_integer(),
    id :: binary(),
    name :: binary(),
    args :: map(),
    class :: erllama_server_tools:class(),
    status = queued :: queued | pending | running | done,
    result :: {ok | error, binary()} | undefined,
    worker :: {pid(), reference()} | undefined
}).

-record(d, {
    id :: binary(),
    conv :: binary(),
    model :: binary(),
    workspace :: binary(),
    policy :: #{binary() => erllama_server_tools:policy()},
    %% `yes' and `read_only' also decide for MCP tools not in `policy'.
    mode = default :: default | yes | read_only,
    %% This round's MCP tools, by namespaced name.
    mcp = #{} :: #{binary() => map()},
    budgets :: map(),
    generator :: module(),
    events :: erllama_server_events:events(),
    transcript = [] :: [map()],
    queue = [] :: [binary()],
    turn :: map() | undefined,
    gen :: pid() | undefined,
    calls = [] :: [#call{}],
    jobs = #{} :: #{binary() => pid()},
    finish_key :: term(),
    retention :: non_neg_integer(),
    closed = false :: boolean()
}).

-type data() :: #d{}.

%% -----------------------------------------------------------------------------
%% API (attach, detach, cancel, result and info go through
%% erllama_server_job; cancel interrupts the current turn)
%% -----------------------------------------------------------------------------

-spec start_link(map()) -> {ok, pid()} | {error, term()}.
start_link(Spec) ->
    gen_statem:start_link(?MODULE, Spec, []).

%% Input for the session: starts a turn, or waits for the current one.
-spec send(pid(), binary()) -> ok | {queued, pos_integer()} | {error, term()}.
send(Pid, Text) -> erllama_server_job:call(Pid, {send, Text}, 5000).

-spec approve(pid(), binary(), allow | deny | always) -> ok | {error, term()}.
approve(Pid, CallId, Decision) -> erllama_server_job:call(Pid, {approve, CallId, Decision}, 5000).

-spec interrupt(pid()) -> ok | {error, term()}.
interrupt(Pid) -> erllama_server_job:call(Pid, cancel, 5000).

%% End the session (its conversation stays).
-spec stop(pid()) -> ok | {error, term()}.
stop(Pid) -> erllama_server_job:call(Pid, stop, 10000).

%% -----------------------------------------------------------------------------

callback_mode() -> handle_event_function.

-spec init(map()) -> {ok, idle, data()}.
init(#{run_id := Id, model := Model, workspace := Ws, conversation_id := Conv} = Spec) ->
    process_flag(trap_exit, true),
    System = binary:replace(maps:get(system, Spec, ?SYSTEM), <<"WORKSPACE">>, Ws, [global]),
    D0 = #d{
        id = Id,
        conv = Conv,
        model = Model,
        workspace = Ws,
        policy = initial_policy(Spec),
        mode = maps:get(mode, Spec, default),
        budgets = maps:merge(
            #{max_rounds => 20, max_tokens => 32768, max_elapsed_ms => 600000},
            maps:get(budgets, Spec, #{})
        ),
        generator = maps:get(generator, Spec, erllama_server_agent_gen),
        events = erllama_server_events:new(Id),
        transcript = [#{role => <<"system">>, content => System} | maps:get(history, Spec, [])],
        finish_key = maps:get(parent_key, Spec, undefined),
        retention = erllama_server_config:get(run_retention_ms)
    },
    D = emit(
        started, #{agent_id => Id, conversation_id => Conv, model => Model, workspace => Ws}, D0
    ),
    {ok, idle, D}.

%% The native tools' defaults, overridden by the session's choices.
initial_policy(Spec) ->
    maps:merge(
        maps:from_list([
            {N, erllama_server_tools:default_policy(N)}
         || N <- erllama_server_tools:names()
        ]),
        maps:get(policy, Spec, #{})
    ).

%% --- calls every state answers ---------------------------------------------

handle_event({call, From}, {attach, Sub, Cursor}, State, #d{events = Ev} = D) ->
    case erllama_server_events:attach(Sub, Cursor, State =/= stopped, Ev) of
        {{ok, Replay}, Ev1} ->
            {keep_state, D#d{events = Ev1}, [
                {reply, From, {ok, #{run_pid => self(), replay => Replay, status => status(State)}}}
            ]};
        {{error, _} = E, Ev1} ->
            {keep_state, D#d{events = Ev1}, [{reply, From, E}]}
    end;
handle_event({call, From}, {detach, Sub}, _State, #d{events = Ev} = D) ->
    {keep_state, D#d{events = erllama_server_events:detach(Sub, Ev)}, [{reply, From, ok}]};
handle_event({call, From}, Req, State, D) when Req =:= info; Req =:= result ->
    {keep_state_and_data, [{reply, From, {ok, summary(State, D)}}]};
handle_event(info, {ack, Sub, Seq}, _State, #d{events = Ev} = D) when is_integer(Seq) ->
    {keep_state, D#d{events = erllama_server_events:ack(Sub, Seq, Ev)}};
handle_event(info, {'DOWN', Mon, process, Pid, Reason}, State, D) ->
    down(Mon, Pid, Reason, State, D);
handle_event(info, {'EXIT', Gen, Reason}, generating, #d{gen = Gen} = D) when is_pid(Gen) ->
    %% The round died without reporting (a crash, or killed from
    %% outside): the turn fails, and its model hold is released.
    end_turn({error, {generator_exited, Reason}}, D#d{gen = undefined}, []);
handle_event(info, {'EXIT', _Pid, _Reason}, _State, _D) ->
    %% Linked generators, workers and shell jobs: their outcome comes
    %% through their own messages or monitors.
    keep_state_and_data;
handle_event(info, {turn_deadline, Ref}, State, #d{turn = #{deadline_ref := Ref}} = D) when
    State =:= generating; State =:= tools; State =:= approving
->
    D1 = interrupt(State, <<"the turn ran out of time">>, D),
    end_turn({budget, time}, D1, []);
handle_event(info, expire, stopped, _D) ->
    stop;
%% --- stopped ----------------------------------------------------------------
handle_event({call, From}, _Req, stopped, _D) ->
    {keep_state_and_data, [{reply, From, {error, stopped}}]};
%% --- input --------------------------------------------------------------------
handle_event({call, From}, {send, Text}, idle, D) when is_binary(Text), Text =/= <<>> ->
    {next_state, generating, start_turn(Text, D), [{reply, From, ok}]};
handle_event({call, From}, {send, Text}, _State, #d{queue = Q} = D) when
    is_binary(Text), Text =/= <<>>
->
    {keep_state, D#d{queue = Q ++ [Text]}, [{reply, From, {queued, length(Q) + 1}}]};
handle_event({call, From}, {send, _}, _State, _D) ->
    {keep_state_and_data, [{reply, From, {error, empty_input}}]};
handle_event({call, From}, cancel, idle, _D) ->
    {keep_state_and_data, [{reply, From, ok}]};
handle_event({call, From}, cancel, State, D) ->
    end_turn(cancelled, interrupt(State, D), [{reply, From, ok}]);
handle_event({call, From}, stop, State, D) ->
    D1 =
        case State of
            idle -> D;
            _ -> close_turn(cancelled, interrupt(State, D))
        end,
    {next_state, stopped, shutdown(D1), [{reply, From, ok}]};
handle_event({call, From}, {approve, CallId, Decision}, approving, D) ->
    approve_call(CallId, Decision, From, D);
handle_event({call, From}, {approve, _, _}, _State, _D) ->
    {keep_state_and_data, [{reply, From, {error, nothing_to_approve}}]};
%% --- generation ---------------------------------------------------------------
handle_event(info, {agent_gen, Gen, {delta, Type, Data}}, generating, #d{gen = Gen} = D) ->
    {keep_state, emit(Type, Data, D)};
handle_event(info, {agent_gen, Gen, {done, Result}}, generating, #d{gen = Gen} = D) ->
    after_generation(Result, D#d{gen = undefined});
handle_event(info, {agent_gen, Gen, {error, Reason}}, generating, #d{gen = Gen} = D) ->
    end_turn({error, Reason}, D#d{gen = undefined}, []);
%% --- tools ----------------------------------------------------------------------
handle_event(info, {tool_done, Worker, Index, Result}, tools, D) ->
    tool_finished(Worker, Index, Result, D);
handle_event(_Type, _Msg, _State, _D) ->
    keep_state_and_data.

terminate(_Reason, State, #d{} = D) ->
    %% Running tools are stopped here, not left to whoever kills this
    %% process's workers next.
    _ = shutdown(interrupt(State, D)),
    ok.

status(stopped) -> done;
status(_) -> running.

%% -----------------------------------------------------------------------------
%% Turns
%% -----------------------------------------------------------------------------

start_turn(Text, #d{budgets = #{max_elapsed_ms := Ms}} = D) ->
    User = #{role => <<"user">>, content => Text},
    D1 = record([User], D),
    Ref = make_ref(),
    _ = erlang:send_after(Ms, self(), {turn_deadline, Ref}),
    Turn = #{rounds => 0, tokens => 0, deadline => now_ms() + Ms, deadline_ref => Ref},
    generate(emit(turn_started, #{input => Text}, D1#d{turn = Turn})).

generate(#d{generator = G, model = M, transcript = T, turn = Turn, id = Id} = D0) ->
    D = D0#d{mcp = mcp_snapshot()},
    Native = [
        Def
     || #{name := N} = Def <- erllama_server_tools:definitions(),
        maps:get(N, D#d.policy) =/= deny
    ],
    Mcp = [
        maps:with([name, description, parameters], E)
     || #{name := N} = E <- lists:sort(maps:values(D#d.mcp)),
        policy(N, D) =/= deny
    ],
    %% The round may generate at most what is left of the turn's tokens.
    Left = max(1, maps:get(max_tokens, D#d.budgets) - maps:get(tokens, Turn)),
    Opts = #{tools => Native ++ Mcp, response_tokens => Left},
    Opts1 =
        case D#d.finish_key of
            undefined -> Opts;
            K -> Opts#{parent_key => K}
        end,
    Ctx = #{model => M, agent_id => Id},
    Gen = G:start(self(), Ctx, T, Opts1),
    emit(round, #{round => maps:get(rounds, Turn) + 1}, D#d{gen = Gen}).

after_generation(Result, #d{turn = Turn} = D) ->
    #{completion_tokens := G} = maps:get(usage, Result, #{completion_tokens => 0}),
    Turn1 = Turn#{rounds := maps:get(rounds, Turn) + 1, tokens := maps:get(tokens, Turn) + G},
    Reply = erllama_server_generate:reply_message(Result),
    D1 = record([Reply], D#d{turn = Turn1, finish_key = maps:get(finish_key, Result, undefined)}),
    case maps:get(tool_calls, Reply, []) of
        [] ->
            case budget_left(D1) of
                {exceeded, tokens} -> end_turn({budget, tokens}, D1, []);
                _ -> end_turn(stop, D1, [])
            end;
        Calls ->
            plan_calls(Calls, D1)
    end.

mcp_snapshot() ->
    case erllama_server_config:get(mcp_servers) of
        [] -> #{};
        _ -> maps:from_list([{N, E} || #{name := N} = E <- erllama_server_mcp:catalog()])
    end.

%% The policy of a tool in this session. A native tool's is in `policy'
%% (filled from the defaults at start). For an MCP tool, a server the
%% operator set to `deny' stays denied; otherwise the session's choice
%% for that name, then its mode, then the server's policy decide.
policy(Name, #d{policy = Policy, mcp = Mcp, mode = Mode}) ->
    case {Mcp, Policy} of
        {#{Name := #{policy := deny}}, _} -> deny;
        {_, #{Name := P}} -> P;
        {#{Name := #{class := Class, policy := P}}, _} -> mode_policy(Mode, Class, P);
        _ -> deny
    end.

mode_policy(yes, _Class, _P) -> auto;
mode_policy(read_only, write, _P) -> deny;
mode_policy(_Mode, _Class, P) -> P.

class(Name, #d{mcp = Mcp}) ->
    case Mcp of
        #{Name := #{class := C}} -> C;
        _ -> erllama_server_tools:class(Name)
    end.

known(Name, #d{mcp = Mcp}) ->
    is_map_key(Name, Mcp) orelse lists:member(Name, erllama_server_tools:names()).

%% Each call is denied, waits for approval, or is ready to run.
plan_calls(Calls, D) ->
    Planned = [
        plan_call(I, C, D)
     || {I, C} <- lists:zip(lists:seq(0, length(Calls) - 1), Calls)
    ],
    D1 = D#d{calls = Planned},
    Pending = [C || #call{status = pending} = C <- Planned],
    case Pending of
        [] ->
            run_calls(D1);
        _ ->
            D2 = lists:foldl(
                fun(#call{id = Id, name = N, args = A}, Acc) ->
                    emit(approval_request, #{call_id => Id, name => N, arguments => A}, Acc)
                end,
                D1,
                Pending
            ),
            {next_state, approving, D2}
    end.

plan_call(I, #{id := Id, function := #{name := Name, arguments := ArgsJson}}, D) ->
    Args = decode_args(ArgsJson),
    Base = #call{index = I, id = Id, name = Name, args = Args, class = class(Name, D)},
    case {known(Name, D), policy(Name, D)} of
        {false, _} ->
            Base#call{status = done, result = {error, <<"unknown tool: ", Name/binary>>}};
        {true, deny} ->
            Base#call{
                status = done, result = {error, <<"this tool is not allowed in this session">>}
            };
        {true, ask} ->
            Base#call{status = pending};
        {true, auto} ->
            Base
    end.

decode_args(A) when is_map(A) -> A;
decode_args(A) when is_binary(A) ->
    try json:decode(A) of
        M when is_map(M) -> M;
        _ -> #{}
    catch
        error:_ -> #{}
    end.

approve_call(CallId, Decision, From, #d{calls = Calls, policy = Policy} = D) ->
    case [C || #call{id = Id, status = pending} = C <- Calls, Id =:= CallId] of
        [#call{name = Name} = C] ->
            {Updated, Policy1} =
                case Decision of
                    deny ->
                        {
                            C#call{
                                status = done, result = {error, <<"the user declined this call">>}
                            },
                            Policy
                        };
                    allow ->
                        {C#call{status = queued}, Policy};
                    always ->
                        {C#call{status = queued}, Policy#{Name => auto}}
                end,
            D1 = D#d{calls = replace(Updated, Calls), policy = Policy1},
            case [P || #call{status = pending} = P <- D1#d.calls] of
                [] -> with_reply(run_calls(D1), From);
                _ -> {keep_state, D1, [{reply, From, ok}]}
            end;
        [] ->
            {keep_state_and_data, [{reply, From, {error, {no_pending_call, CallId}}}]}
    end.

with_reply({next_state, S, D}, From) -> {next_state, S, D, [{reply, From, ok}]};
with_reply({next_state, S, D, Actions}, From) -> {next_state, S, D, [{reply, From, ok} | Actions]}.

run_calls(D) ->
    D1 = schedule(D),
    case all_done(D1) of
        true -> next_round(D1);
        false -> {next_state, tools, D1}
    end.

%% Start what may run now: a read when no earlier write is unfinished,
%% a write when every earlier call is finished.
schedule(#d{calls = Calls} = D0) ->
    {Started, {_, _, D}} = lists:mapfoldl(
        fun
            (#call{status = queued, class = read} = C, {false = W, _Any, D1}) ->
                %% Started, it is now unfinished for the writes after it.
                {C1, D2} = start_call(C, D1),
                {C1, {W, true, D2}};
            (#call{status = queued, class = write} = C, {false, false, D1}) ->
                {C1, D2} = start_call(C, D1),
                {C1, {true, true, D2}};
            (#call{status = done} = C, Acc) ->
                {C, Acc};
            (#call{class = write} = C, {_, _, D1}) ->
                {C, {true, true, D1}};
            (C, {W, _, D1}) ->
                {C, {W, true, D1}}
        end,
        {false, false, D0},
        Calls
    ),
    D#d{calls = Started}.

start_call(
    #call{index = I, name = Name, args = Args, id = Id} = C,
    #d{workspace = Ws, jobs = Jobs, mcp = Mcp} = D
) ->
    Self = self(),
    Ctx0 = #{workspace => Ws, owner => Self, jobs => Jobs},
    Ctx =
        case Mcp of
            #{Name := Entry} -> Ctx0#{mcp => Entry};
            _ -> Ctx0
        end,
    Worker = spawn_monitor(fun() ->
        Self ! {tool_done, self(), I, erllama_server_tools:execute(Name, Args, Ctx)}
    end),
    {
        C#call{status = running, worker = Worker},
        emit(tool_started, #{call_id => Id, name => Name, arguments => Args}, D)
    }.

tool_finished(Worker, Index, Result, #d{calls = Calls, jobs = Jobs} = D) ->
    case lists:keyfind(Index, #call.index, Calls) of
        #call{worker = {Worker, Mon}, id = Id, name = Name} = C ->
            erlang:demonitor(Mon, [flush]),
            {Res, Jobs1} =
                case Result of
                    {ok, Text, {job, JobId, Pid}} -> {{ok, Text}, Jobs#{JobId => Pid}};
                    _ -> {Result, Jobs}
                end,
            D1 = emit(
                tool_result,
                #{
                    call_id => Id,
                    name => Name,
                    ok => element(1, Res) =:= ok,
                    output => preview(element(2, Res))
                },
                D
            ),
            D2 = D1#d{
                calls = replace(C#call{status = done, result = Res, worker = undefined}, Calls),
                jobs = Jobs1
            },
            continue_tools(D2);
        _ ->
            keep_state_and_data
    end.

continue_tools(D) ->
    D1 = schedule(D),
    case all_done(D1) of
        true -> next_round(D1);
        false -> {keep_state, D1}
    end.

all_done(#d{calls = Calls}) ->
    lists:all(fun(#call{status = S}) -> S =:= done end, Calls).

%% The results, in call order, then the next generation unless a budget
%% is spent.
next_round(#d{calls = Calls} = D) ->
    Results = [tool_message(C) || C <- lists:keysort(#call.index, Calls)],
    D1 = record(Results, D#d{calls = []}),
    case budget_left(D1) of
        ok -> {next_state, generating, generate(D1)};
        {exceeded, What} -> end_turn({budget, What}, D1, [])
    end.

tool_message(#call{id = Id, name = Name, result = {Status, Text}}) ->
    Content =
        case Status of
            ok -> Text;
            error -> <<"Error: ", Text/binary>>
        end,
    #{role => <<"tool">>, tool_call_id => Id, name => Name, content => Content}.

budget_left(#d{turn = #{rounds := R, tokens := T, deadline := Dl}, budgets = B}) ->
    case {R >= maps:get(max_rounds, B), T >= maps:get(max_tokens, B), now_ms() >= Dl} of
        {true, _, _} -> {exceeded, rounds};
        {_, true, _} -> {exceeded, tokens};
        {_, _, true} -> {exceeded, time};
        _ -> ok
    end.

%% -----------------------------------------------------------------------------
%% Ending a turn
%% -----------------------------------------------------------------------------

end_turn(Reason, D, Actions) ->
    D1 = close_turn(Reason, D),
    case D1#d.queue of
        [Next | Rest] -> {next_state, generating, start_turn(Next, D1#d{queue = Rest}), Actions};
        [] -> {next_state, idle, D1, Actions}
    end.

close_turn(Reason, #d{turn = Turn, model = M} = D) ->
    erllama_server_models:release(M, self()),
    Info = maps:with([rounds, tokens], Turn),
    emit(turn_done, Info#{reason => reason(Reason)}, D#d{turn = undefined}).

reason({error, R}) -> #{error => R};
reason({budget, What}) -> #{budget => What};
reason(R) -> R.

%% Stop what the turn is doing. Unfinished tool calls get an
%% "interrupted" result, so the transcript stays well formed.
interrupt(State, D) ->
    interrupt(State, <<"interrupted by the user">>, D).

interrupt(generating, _Why, #d{gen = Gen} = D) when is_pid(Gen) ->
    unlink(Gen),
    exit(Gen, kill),
    D#d{gen = undefined};
interrupt(State, Why, #d{calls = Calls} = D) when State =:= tools; State =:= approving ->
    [kill_worker(C) || #call{worker = W} = C <- Calls, W =/= undefined],
    Closed = [
        case C of
            #call{status = done} ->
                C;
            _ ->
                C#call{
                    status = done,
                    result = {error, Why},
                    worker = undefined
                }
        end
     || C <- Calls
    ],
    Results = [tool_message(C) || C <- lists:keysort(#call.index, Closed)],
    record(Results, D#d{calls = []});
interrupt(_State, _Why, D) ->
    D.

%% A running command is stopped through its job first: killing the
%% worker alone leaves the group's cleanup to happen later, or never if
%% the release is stopping.
kill_worker(#call{name = <<"shell">>, worker = {Pid, Mon}}) ->
    erlang:demonitor(Mon, [flush]),
    ok = erllama_server_tool_shell:stop_run(Pid),
    exit(Pid, kill);
kill_worker(#call{worker = {Pid, Mon}}) ->
    erlang:demonitor(Mon, [flush]),
    exit(Pid, kill).

%% End the session: background jobs stop, the conversation is released.
shutdown(#d{closed = true} = D) ->
    D;
shutdown(#d{jobs = Jobs, conv = Conv, id = Id} = D) ->
    ok = erllama_server_tool_shell:stop_jobs(maps:values(Jobs)),
    _ = erllama_server_convs:release(Conv, Id),
    ok = erllama_server_runs:set_status(Id, done),
    _ = erlang:send_after(D#d.retention, self(), expire),
    D1 = emit(done, #{agent_id => Id, conversation_id => Conv}, D#d{jobs = #{}, closed = true}),
    D1#d{events = erllama_server_events:close(D1#d.events)}.

%% -----------------------------------------------------------------------------

down(Mon, Pid, Reason, tools, #d{calls = Calls} = D) ->
    case [C || #call{worker = {W, M}} = C <- Calls, W =:= Pid, M =:= Mon] of
        [#call{index = I}] ->
            Text = iolist_to_binary(io_lib:format("the tool crashed: ~0p", [Reason])),
            tool_finished(Pid, I, {error, Text}, D);
        [] ->
            subscriber_down(Mon, Pid, D)
    end;
down(Mon, Pid, _Reason, _State, D) ->
    subscriber_down(Mon, Pid, D).

subscriber_down(Mon, Pid, #d{events = Ev} = D) ->
    case erllama_server_events:down(Mon, Pid, Ev) of
        {ok, Ev1} -> {keep_state, D#d{events = Ev1}};
        ignore -> keep_state_and_data
    end.

%% Append to the transcript and to the conversation.
record(Msgs, #d{transcript = T, conv = Conv, id = Id, finish_key = K} = D) ->
    ok = erllama_server_convs:append(Conv, Id, Msgs, K),
    D#d{transcript = T ++ Msgs}.

replace(#call{index = I} = C, Calls) -> lists:keyreplace(I, #call.index, Calls, C).

emit(Type, Data, #d{events = Ev} = D) ->
    D#d{events = erllama_server_events:emit(Type, Data, Ev)}.

preview(Text) when byte_size(Text) > 400 -> <<(binary:part(Text, 0, 400))/binary, "...">>;
preview(Text) -> Text.

summary(State, #d{id = Id, conv = Conv, model = M, workspace = Ws, queue = Q, events = Ev}) ->
    #{
        run_id => Id,
        kind => agent,
        conversation_id => Conv,
        model => M,
        workspace => Ws,
        state => State,
        queued => length(Q),
        status => status(State),
        next_seq => erllama_server_events:next_seq(Ev)
    }.

now_ms() -> erlang:monotonic_time(millisecond).
