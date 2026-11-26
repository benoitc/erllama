%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_tool_shell).
-moduledoc false.
%% Shell commands for agent sessions. Each command runs in a job
%% process that owns the port: `/bin/sh' in the workspace. The emulator
%% starts every port program as the leader of its own process group, so
%% signalling that group ends every process the command started. The job keeps the first and last
%% 16 KiB of output, enforces the timeout (TERM to the group, then KILL),
%% and is linked to its owner: when the owner goes (an interrupted tool
%% worker, a stopped session), the group is killed.
%%
%% `run/4' runs a command to its end; `start/4' starts a background job
%% that `status/1' and `kill/1' address.

-export([
    run/3, run/4, start/4, status/1, kill/1, stop_jobs/1, stop_run/1, format/1, display_output/1
]).

-define(HEAD, 16384).
-define(TAIL, 16384).
-define(GRACE_MS, 2000).

-type result() :: #{
    exit := integer() | timeout | killed,
    output := binary(),
    truncated := non_neg_integer(),
    ms := non_neg_integer()
}.
-export_type([result/0, progress/0]).

%% Run `Cmd' in `Dir' and wait for it (at most `TimeoutMs').
-type progress() :: fun((binary(), non_neg_integer()) -> term()) | undefined.
-spec run(binary(), binary(), pos_integer()) -> result().
run(Dir, Cmd, TimeoutMs) ->
    run(Dir, Cmd, TimeoutMs, undefined).

-spec run(binary(), binary(), pos_integer(), progress()) -> result().
run(Dir, Cmd, TimeoutMs, Progress) ->
    Owner = self(),
    {Pid, Mon} = spawn_opt(fun() -> init(Owner, Dir, Cmd, TimeoutMs, Progress =/= undefined) end, [
        link, monitor
    ]),
    Deadline = erlang:monotonic_time(millisecond) + TimeoutMs + 3 * ?GRACE_MS,
    await_job(Pid, Mon, Deadline, Progress).

await_job(Pid, Mon, Deadline, Progress) ->
    Left = max(0, Deadline - erlang:monotonic_time(millisecond)),
    receive
        {shell_output, Pid, Bytes, Omitted} when is_function(Progress, 2) ->
            Progress(Bytes, Omitted),
            await_job(Pid, Mon, Deadline, Progress);
        {shell_done, Pid, Result} ->
            erlang:demonitor(Mon, [flush]),
            unlink(Pid),
            Result;
        {stop_shell, From, Ref} ->
            %% The session interrupts this call: stop the command's group
            %% before this process is killed, so it cannot outlive it.
            Result = stopped(kill(Pid)),
            From ! {Ref, stopped},
            Result;
        {'DOWN', Mon, process, Pid, Reason} ->
            died(Reason)
    after Left ->
        %% The job enforces the timeout; this only guards against a job
        %% that stopped answering.
        exit(Pid, kill),
        died(no_answer)
    end.

stopped({done, R}) -> R;
stopped({error, gone}) -> died(gone).

%% Stop the command a `run/4' in process `Runner' is waiting for, and
%% wait (bounded) until its process group has been dealt with.
-spec stop_run(pid()) -> ok.
stop_run(Runner) ->
    Mon = erlang:monitor(process, Runner),
    Runner ! {stop_shell, self(), Mon},
    receive
        {Mon, stopped} -> ok;
        {'DOWN', Mon, process, Runner, _} -> ok
    after 2 * ?GRACE_MS + 1000 -> ok
    end,
    erlang:demonitor(Mon, [flush]),
    ok.

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
        init(Owner, Dir, Cmd, TimeoutMs, false)
    end).

-spec status(pid()) -> {running, result()} | {done, result()} | {error, gone}.
status(Job) -> ask(Job, status).

-spec kill(pid()) -> {done, result()} | {error, gone}.
kill(Job) -> ask(Job, kill).

%% Stop background jobs (a session ending): each kills its command's
%% process group, all at once. Never `exit(Job, kill)' first: that
%% would skip the group, and the command would outlive the session.
-spec stop_jobs([pid()]) -> ok.
stop_jobs(Jobs) ->
    Asked = [ask_kill(J) || J <- Jobs],
    Deadline = erlang:monotonic_time(millisecond) + 2 * ?GRACE_MS,
    lists:foreach(fun({J, Mon}) -> await_stop(J, Mon, Deadline) end, Asked).

ask_kill(Job) ->
    Mon = erlang:monitor(process, Job),
    Job ! {kill, self(), Mon},
    {Job, Mon}.

await_stop(Job, Mon, Deadline) ->
    Left = max(0, Deadline - erlang:monotonic_time(millisecond)),
    receive
        {Mon, _} -> erlang:demonitor(Mon, [flush]);
        {'DOWN', Mon, process, Job, _} -> ok
    after Left ->
        %% The group has had TERM; the job itself stopped answering.
        erlang:demonitor(Mon, [flush]),
        exit(Job, kill)
    end.

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
    progress = off :: off | ready | {binary(), non_neg_integer()},
    partial = <<>> :: binary(),
    port :: port(),
    pgid :: string(),
    head = <<>> :: binary(),
    tail = <<>> :: binary(),
    dropped = 0 :: non_neg_integer(),
    started :: integer(),
    result :: result() | undefined,
    waiting = [] :: [{pid(), reference()}],
    exit_as :: timeout | killed | undefined
}).

init(Owner, Dir, Cmd, TimeoutMs, Live) ->
    process_flag(trap_exit, true),
    %% The command is passed as a positional parameter, never spliced
    %% into the script.
    Port = open_port(
        {spawn_executable, "/bin/sh"},
        [
            {args, ["-c", "exec /bin/sh -c \"$1\" </dev/null 2>&1", "sh", Cmd]},
            {cd, Dir},
            binary,
            exit_status,
            stderr_to_stdout,
            use_stdio
        ]
    ),
    {os_pid, OsPid} = erlang:port_info(Port, os_pid),
    _ = erlang:send_after(TimeoutMs, self(), timeout),
    loop(#job{
        owner = Owner,
        progress =
            case Live of
                true -> ready;
                false -> off
            end,
        port = Port,
        pgid = integer_to_list(OsPid),
        started = erlang:monotonic_time(millisecond)
    }).

loop(#job{port = Port, owner = Owner} = J) ->
    %% A finished background job nobody asks about goes away after a while.
    Idle =
        case J#job.result of
            undefined -> infinity;
            _ -> 600000
        end,
    receive
        {Port, {data, Data}} ->
            loop(keep(Data, buffer_output(Data, J)));
        flush_output ->
            loop(flush_output(J, false));
        {Port, {exit_status, Status}} ->
            finish(Status, J);
        timeout when J#job.result =:= undefined ->
            loop(stop(timeout, J));
        {Req, From, Ref} when Req =:= status; Req =:= kill ->
            loop(request(Req, From, Ref, J));
        {'EXIT', Owner, _} when J#job.result =/= undefined ->
            exit(normal);
        {'EXIT', Owner, _} ->
            _ = stop(killed, J),
            await_exit(Port);
        _ ->
            loop(J)
    after Idle ->
        exit(normal)
    end.

%% The owner is gone and the group signalled: wait (bounded) for the
%% shell to exit, so its port is not closed under a dying command.
await_exit(Port) ->
    receive
        {Port, {exit_status, _}} -> exit(normal);
        {Port, _} -> await_exit(Port)
    after 2 * ?GRACE_MS + 1000 ->
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

%% A finished job's group is never signalled again.
stop(_Why, #job{result = R} = J) when R =/= undefined ->
    J;
stop(Why, J) ->
    stop_group(J),
    J#job{exit_as = Why}.

%% Batch noisy commands into one bounded preview per 100 ms. Keeping the most
%% recent bytes lets a subscriber see later progress even during an output flood.
buffer_output(_Data, #job{progress = off} = J) ->
    J;
buffer_output(Data, #job{progress = ready} = J) ->
    erlang:send_after(100, self(), flush_output),
    buffer_output(Data, J#job{progress = {<<>>, 0}});
buffer_output(Data, #job{progress = {Prev, Omitted}} = J) ->
    J#job{progress = append_tail(Data, Prev, Omitted)}.

flush_output(#job{progress = {Bytes, Omitted}, partial = Partial} = J, Final) ->
    {Input, Skipped} =
        case Omitted of
            0 -> {<<Partial/binary, Bytes/binary>>, 0};
            _ -> {Bytes, Omitted + byte_size(Partial)}
        end,
    {Text, Rest} = output_text(Input, Final, []),
    send_output(Text, Skipped, J),
    J#job{progress = ready, partial = Rest};
flush_output(#job{partial = Partial} = J, true) when Partial =/= <<>> ->
    send_output(<<16#FFFD/utf8>>, 0, J),
    J#job{partial = <<>>};
flush_output(J, _) ->
    J.

send_output(<<>>, 0, _) ->
    ok;
send_output(Text, Omitted, #job{owner = Owner}) ->
    Owner ! {shell_output, self(), Text, Omitted},
    ok.

output_text(Bytes, Final, Acc) ->
    case unicode:characters_to_binary(Bytes) of
        Text when is_binary(Text) -> {iolist_to_binary(lists:reverse([Text | Acc])), <<>>};
        {incomplete, Text, _} when Final ->
            {iolist_to_binary(lists:reverse([<<16#FFFD/utf8>>, Text | Acc])), <<>>};
        {incomplete, Text, Rest} ->
            {iolist_to_binary(lists:reverse([Text | Acc])), binary:copy(Rest)};
        {error, Text, <<_, Rest/binary>>} ->
            output_text(Rest, Final, [<<16#FFFD/utf8>>, Text | Acc])
    end.

display_output(#{output := Text, omitted := 0}) ->
    Text;
display_output(#{output := Text, omitted := N}) ->
    iolist_to_binary([<<"[">>, integer_to_binary(N), <<" bytes omitted]\n">>, Text]).

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
    {Tail, Dropped} = append_tail(Data, T, D),
    J#job{tail = Tail, dropped = Dropped}.

append_tail(Data, Prev, Dropped) ->
    Combined = <<Prev/binary, Data/binary>>,
    case byte_size(Combined) - ?TAIL of
        Over when Over > 0 -> {binary:copy(binary:part(Combined, Over, ?TAIL)), Dropped + Over};
        _ -> {Combined, Dropped}
    end.

snapshot(Exit, #job{head = H, tail = T, dropped = D, started = S}) ->
    #{
        exit => Exit,
        output => <<H/binary, T/binary>>,
        truncated => D,
        ms => erlang:monotonic_time(millisecond) - S
    }.

finish(Status, #job{owner = Owner, waiting = W, exit_as = As} = J0) ->
    J = flush_output(J0, true),
    Exit =
        case As of
            undefined -> Status;
            _ -> As
        end,
    Result = snapshot(Exit, J),
    Owner ! {shell_done, self(), Result},
    _ = [From ! {Ref, {done, Result}} || {From, Ref} <- W],
    loop(J#job{result = Result, waiting = []}).

%% TERM to the whole group, then KILL after a grace period. The KILL
%% is left to a detached shell, not an Erlang process: one that would
%% be stopped with the application (its processes are killed when it
%% stops) and leave the group running.
stop_group(#job{pgid = Pgid}) ->
    G = "-" ++ Pgid,
    Grace = integer_to_list(max(1, ?GRACE_MS div 1000)),
    _ = os:cmd(
        "kill -TERM " ++ G ++ " 2>/dev/null; (sleep " ++ Grace ++ "; kill -KILL " ++ G ++
            " 2>/dev/null) </dev/null >/dev/null 2>&1 &"
    ),
    ok.
