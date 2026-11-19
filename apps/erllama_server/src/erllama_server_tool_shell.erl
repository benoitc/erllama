%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_tool_shell).
-moduledoc false.
%% Shell commands for agent sessions. Each command runs in a job
%% process that owns the port: `/bin/sh' in the workspace, with the
%% command in its own process group (`set -m'), so ending the job ends
%% every process the command started. The job keeps the first and last
%% 16 KiB of output, enforces the timeout (TERM to the group, then KILL),
%% and is linked to its owner: when the owner goes (an interrupted tool
%% worker, a stopped session), the group is killed.
%%
%% `run/3' runs a command to its end; `start/3' starts a background job
%% that `status/1' and `kill/1' address.

-export([run/3, start/4, status/1, kill/1, format/1]).
-export([init/4]).

-define(HEAD, 16384).
-define(TAIL, 16384).
-define(GRACE_MS, 2000).
-define(MARK, "__ERLLAMA_PGID ").

-type result() :: #{
    exit := integer() | timeout | killed,
    output := binary(),
    truncated := non_neg_integer(),
    ms := non_neg_integer()
}.
-export_type([result/0]).

%% Run `Cmd' in `Dir' and wait for it (at most `TimeoutMs').
-spec run(binary(), binary(), pos_integer()) -> result().
run(Dir, Cmd, TimeoutMs) ->
    {Pid, Mon} = spawn_monitor(?MODULE, init, [self(), Dir, Cmd, TimeoutMs]),
    link(Pid),
    receive
        {shell_done, Pid, Result} ->
            erlang:demonitor(Mon, [flush]),
            unlink(Pid),
            Result;
        {'DOWN', Mon, process, Pid, Reason} ->
            died(Reason)
    after TimeoutMs + 3 * ?GRACE_MS ->
        %% The job enforces the timeout; this only guards against a job
        %% that stopped answering.
        exit(Pid, kill),
        died(no_answer)
    end.

died(Reason) ->
    #{
        exit => killed,
        output => iolist_to_binary(io_lib:format("the shell job died: ~p", [Reason])),
        truncated => 0,
        ms => 0
    }.

%% A background job owned by `Owner' (an agent session): linked to it,
%% killed with it.
-spec start(pid(), binary(), binary(), pos_integer()) -> pid().
start(Owner, Dir, Cmd, TimeoutMs) ->
    spawn(fun() ->
        link(Owner),
        init(Owner, Dir, Cmd, TimeoutMs)
    end).

-spec status(pid()) -> {running, result()} | {done, result()} | {error, gone}.
status(Job) -> ask(Job, status).

-spec kill(pid()) -> {done, result()} | {error, gone}.
kill(Job) -> ask(Job, kill).

ask(Job, What) ->
    Mon = erlang:monitor(process, Job),
    Job ! {What, self(), Mon},
    receive
        {Mon, Reply} ->
            erlang:demonitor(Mon, [flush]),
            Reply;
        {'DOWN', Mon, process, Job, _} ->
            {error, gone}
    after 10000 ->
        erlang:demonitor(Mon, [flush]),
        {error, gone}
    end.

%% The model-facing text of a result.
-spec format(result()) -> binary().
format(#{exit := Exit, output := Out, truncated := T, ms := Ms}) ->
    Status =
        case Exit of
            0 -> <<"exit 0">>;
            running -> <<"still running">>;
            timeout -> iolist_to_binary(io_lib:format("timed out after ~b ms (killed)", [Ms]));
            killed -> <<"killed">>;
            N -> iolist_to_binary(io_lib:format("exit ~b", [N]))
        end,
    Note =
        case T of
            0 ->
                <<>>;
            _ ->
                iolist_to_binary(
                    io_lib:format("~n(~b bytes of output omitted from the middle)", [T])
                )
        end,
    iolist_to_binary([Status, <<"\n">>, Out, Note]).

%% -----------------------------------------------------------------------------
%% The job process
%% -----------------------------------------------------------------------------

-record(job, {
    owner :: pid(),
    port :: port(),
    pgid :: string() | undefined,
    pending = <<>> :: binary(),
    head = <<>> :: binary(),
    tail = <<>> :: binary(),
    dropped = 0 :: non_neg_integer(),
    started :: integer(),
    result :: result() | undefined,
    waiting = [] :: [{pid(), reference()}],
    %% `timeout' or `killed' asked for before the shell said where the
    %% command runs; applied as soon as it does.
    stop :: timeout | killed | undefined,
    exit_as :: timeout | killed | undefined
}).

-spec init(pid(), binary(), binary(), pos_integer()) -> no_return().
init(Owner, Dir, Cmd, TimeoutMs) ->
    process_flag(trap_exit, true),
    %% The command is passed as a positional parameter, never spliced
    %% into the script. Monitor mode is on only to give it its own
    %% process group. The shell's own notices go nowhere.
    Script =
        "set -m\n"
        "/bin/sh -c \"$1\" </dev/null 2>&1 &\n"
        "pid=$!\n"
        "set +m\n"
        "echo \""
        ?MARK
        "$pid\"\n"
        "exec 2>/dev/null\n"
        "wait $pid\n",
    Port = open_port(
        {spawn_executable, "/bin/sh"},
        [
            {args, ["-c", Script, "sh", Cmd]},
            {cd, Dir},
            binary,
            exit_status,
            stderr_to_stdout,
            use_stdio
        ]
    ),
    _ = erlang:send_after(TimeoutMs, self(), timeout),
    loop(#job{owner = Owner, port = Port, started = erlang:monotonic_time(millisecond)}).

loop(#job{port = Port, owner = Owner} = J) ->
    %% A finished background job nobody asks about goes away after a while.
    Idle =
        case J#job.result of
            undefined -> infinity;
            _ -> 600000
        end,
    receive
        {Port, {data, Data}} ->
            loop(output(Data, J));
        {Port, {exit_status, Status}} ->
            finish(Status, J);
        timeout when J#job.result =:= undefined ->
            loop(stop(timeout, J));
        give_up when J#job.result =:= undefined, J#job.pgid =:= undefined ->
            %% The shell never said where the command runs: close it.
            catch_port_close(Port),
            finish(killed, J);
        {Req, From, Ref} when Req =:= status; Req =:= kill ->
            loop(request(Req, From, Ref, J));
        {'EXIT', Owner, _} ->
            _ = stop(killed, J),
            exit(normal);
        _ ->
            loop(J)
    after Idle ->
        exit(normal)
    end.

request(status, From, Ref, #job{result = undefined} = J) ->
    From ! {Ref, {running, snapshot(running, J)}},
    J;
request(kill, From, Ref, #job{result = undefined} = J) ->
    stop(killed, J#job{waiting = [{From, Ref} | J#job.waiting]});
request(_Req, From, Ref, #job{result = R} = J) ->
    From ! {Ref, {done, R}},
    J.

%% Stop the command's process group now, or as soon as it is known.
stop(Why, #job{pgid = undefined} = J) ->
    _ = erlang:send_after(?GRACE_MS, self(), give_up),
    J#job{stop = Why, exit_as = Why};
stop(Why, J) ->
    stop_group(J),
    J#job{exit_as = Why}.

%% The first line carries the command's process group.
output(Data, #job{pgid = undefined, pending = P} = J) ->
    Buf = <<P/binary, Data/binary>>,
    case binary:split(Buf, <<"\n">>) of
        [<<?MARK, Pgid/binary>>, Rest] ->
            stop_pending(keep(Rest, J#job{pgid = binary_to_list(Pgid), pending = <<>>}));
        [_Incomplete] ->
            J#job{pending = Buf};
        [Other, Rest] ->
            output(Rest, keep(<<Other/binary, "\n">>, J#job{pending = <<>>}))
    end;
output(Data, J) ->
    keep(Data, J).

stop_pending(#job{stop = undefined} = J) -> J;
stop_pending(#job{stop = Why} = J) -> stop(Why, J#job{stop = undefined}).

keep(<<>>, J) ->
    J;
keep(Data, #job{head = H} = J) when byte_size(H) < ?HEAD ->
    Room = ?HEAD - byte_size(H),
    case Data of
        <<Take:Room/binary, Rest/binary>> ->
            keep_tail(Rest, J#job{head = <<H/binary, Take/binary>>});
        _ ->
            J#job{head = <<H/binary, Data/binary>>}
    end;
keep(Data, J) ->
    keep_tail(Data, J).

keep_tail(<<>>, J) ->
    J;
keep_tail(Data, #job{tail = T, dropped = D} = J) ->
    New = <<T/binary, Data/binary>>,
    case byte_size(New) - ?TAIL of
        Over when Over > 0 -> J#job{tail = binary:part(New, Over, ?TAIL), dropped = D + Over};
        _ -> J#job{tail = New}
    end.

snapshot(Exit, #job{head = H, tail = T, dropped = D, started = S}) ->
    #{
        exit => Exit,
        output => <<H/binary, T/binary>>,
        truncated => D,
        ms => erlang:monotonic_time(millisecond) - S
    }.

finish(Status, #job{owner = Owner, waiting = W, exit_as = As} = J) ->
    Exit =
        case As of
            undefined -> Status;
            _ -> As
        end,
    Result = snapshot(Exit, J),
    Owner ! {shell_done, self(), Result},
    _ = [From ! {Ref, {done, Result}} || {From, Ref} <- W],
    loop(J#job{result = Result, waiting = []}).

%% TERM to the whole group, then KILL after a grace period.
stop_group(#job{pgid = Pgid}) ->
    _ = os:cmd("kill -TERM -" ++ Pgid ++ " 2>/dev/null"),
    _ = spawn(fun() ->
        timer:sleep(?GRACE_MS),
        os:cmd("kill -KILL -" ++ Pgid ++ " 2>/dev/null")
    end),
    ok.

catch_port_close(Port) ->
    try
        port_close(Port)
    catch
        error:badarg -> ok
    end,
    ok.
