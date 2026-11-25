%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Test helper for erllama_tp_mtp_SUITE:mtp_scheduler, loaded onto the
%% serving node: concurrent greedy requests through erllama's public
%% API, each with its own options, an optional cancel after some tokens
%% and an optional request started once it ended (which takes its
%% sequence).
-module(erllama_tp_mtp_sched).

-export([run/2]).

%% `Specs': [{Name, Prompt, Opts, Then}], Then = none | {cancel_after,
%% N, NextSpec}. Returns #{Name => #{tokens, logprobs, ending, stats,
%% ttft_ms}} for every request started, the follow-ups included.
run(Model, Specs) ->
    Self = self(),
    T0 = now_us(),
    Pids = [spawn_link(fun() -> Self ! {self(), one(Model, S, T0)} end) || S <- Specs],
    maps:from_list(
        lists:append([
            receive
                {Pid, Rs} -> Rs
            end
         || Pid <- Pids
        ])
    ).

one(Model, {Name, Prompt, Opts, Then}, T0) ->
    {ok, Ref} = erllama:stream(Model, Prompt, Opts#{temperature => 0.0}),
    CancelAt =
        case Then of
            {cancel_after, N, _} -> N;
            none -> infinity
        end,
    R = collect(Ref, CancelAt, T0, #{tokens => [], logprobs => [], ttft_ms => undefined}),
    case Then of
        {cancel_after, _, Next} -> [{Name, R} | one(Model, Next, now_us())];
        none -> [{Name, R}]
    end.

collect(Ref, CancelAt, _T0, #{tokens := Ts} = Acc) when length(Ts) =:= CancelAt ->
    ok = erllama:cancel(Ref),
    drain(Ref, Acc#{ending => cancelled});
collect(Ref, CancelAt, T0, #{tokens := Ts, logprobs := Lps} = Acc) ->
    receive
        {erllama, Ref, {token, Bin}} ->
            Acc1 =
                case Ts of
                    [] -> Acc#{ttft_ms => (now_us() - T0) / 1000};
                    _ -> Acc
                end,
            collect(Ref, CancelAt, T0, Acc1#{tokens => Ts ++ [Bin]});
        {erllama, Ref, {logprobs, Lp}} ->
            collect(Ref, CancelAt, T0, Acc#{logprobs => Lps ++ [Lp]});
        {erllama, Ref, {done, Stats}} ->
            Acc#{ending => done, stats => Stats};
        {erllama, Ref, {error, Why}} ->
            Acc#{ending => {error, Why}}
    after 300000 ->
        Acc#{ending => timeout}
    end.

%% After a cancel: whatever the request still sends until it ends.
drain(Ref, Acc) ->
    receive
        {erllama, Ref, {done, Stats}} -> Acc#{stats => Stats};
        {erllama, Ref, {error, _}} -> Acc;
        {erllama, Ref, _} -> drain(Ref, Acc)
    after 10000 ->
        Acc
    end.

now_us() -> erlang:monotonic_time(microsecond).
