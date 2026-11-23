%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%%
%% What the summarizer does inside an application, step by step: run
%% it from `rebar3 shell' with the application started.
-module(summarizer_demo).

-export([run/0]).

-define(TEXT, <<
    "Erlang processes are isolated: each has its own heap and mailbox, and a crash in one "
    "does not corrupt another. Supervisors restart failed processes according to a "
    "strategy, so a system recovers from faults without stopping."
>>).

run() ->
    %% 1. A summary streams to the caller as messages.
    T0 = erlang:monotonic_time(millisecond),
    {ok, Job} = summarizer:summarize(?TEXT),
    N = print(Job, 0),
    io:format("~n[~b messages, ~b ms]~n", [N, erlang:monotonic_time(millisecond) - T0]),

    %% 2. As many jobs as the model has sequences run at once; one more
    %%    is refused, not queued.
    #{sequences := #{n_seq_max := Seqs}} = summarizer:info(),
    Jobs = [element(2, summarizer:summarize(?TEXT)) || _ <- lists:seq(1, Seqs)],
    io:format("one more while ~b run: ~p~n", [Seqs, summarizer:summarize(?TEXT)]),
    io:format("info: ~p~n", [summarizer:info()]),
    [ok = summarizer:cancel(J) || J <- Jobs],

    %% 3. A caller that exits: its job is cancelled.
    Self = self(),
    _ = spawn(fun() ->
        {ok, J} = summarizer:summarize(?TEXT),
        Self ! {job, J},
        receive
            {summarizer, J, {text, _}} -> exit(gone)
        end
    end),
    Dead = receive {job, X} -> X end,
    timer:sleep(500),
    io:format("its caller exited; erllama's request: ~p~n", [erllama:request_info(Dead)]),

    %% 4. The summarizer crashes: its supervisor restarts it, the model
    %%    stays loaded, and the stream that went to it is cancelled.
    {ok, Orphan} = summarizer:summarize(?TEXT),
    exit(whereis(summarizer), kill),
    timer:sleep(300),
    io:format("restarted: ~p, model: ~p~n", [is_pid(whereis(summarizer)), erllama:whereis(<<"summarizer">>)]),
    io:format("the stream to the killed process: ~p~n", [erllama:request_info(Orphan)]),
    {ok, Again} = summarizer:summarize_sync(?TEXT, 60000),
    io:format("after the restart: ~ts~n", [Again]),
    ok.

print(Job, N) ->
    receive
        {summarizer, Job, {text, Bin}} ->
            io:put_chars(Bin),
            print(Job, N + 1);
        {summarizer, Job, {done, _}} ->
            N;
        {summarizer, Job, {error, Reason}} ->
            io:format("~n[error ~p]~n", [Reason]),
            N
    after 60000 ->
        N
    end.
