%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Test helper for erllama_tp_nccl_SUITE, loaded onto the serving
%% rank's node: stream one request through erllama and time it.
-module(erllama_tp_stream).

-export([run/4]).

%% Stream `Prompt' on `Model' (greedy, at most `Max' tokens); cancel
%% after `CancelAfter' tokens (`infinity' to let it finish). Returns the
%% tokens, the time to the first one, the gaps between the others, and
%% how the request ended.
run(Model, Prompt, Max, CancelAfter) ->
    T0 = erlang:monotonic_time(microsecond),
    {ok, Ref} = erllama:stream(Model, Prompt, #{response_tokens => Max, temperature => 0.0}),
    collect(Ref, T0, CancelAfter, [], []).

collect(Ref, T0, CancelAfter, Times, Text) when length(Times) =:= CancelAfter ->
    ok = erllama:cancel(Ref),
    finish(Ref, T0, Times, Text, cancelled);
collect(Ref, T0, CancelAfter, Times, Text) ->
    receive
        {erllama, Ref, {token, Bin}} ->
            collect(Ref, T0, CancelAfter, [erlang:monotonic_time(microsecond) | Times], [Bin | Text]);
        {erllama, Ref, {done, Stats}} ->
            result(T0, Times, Text, {done, maps:get(finish_reason, Stats, done)});
        {erllama, Ref, {error, Why}} ->
            result(T0, Times, Text, {error, Why})
    after 120000 ->
        result(T0, Times, Text, timeout)
    end.

%% Drain to the request's end after a cancel.
finish(Ref, T0, Times, Text, How) ->
    receive
        {erllama, Ref, {token, _}} -> finish(Ref, T0, Times, Text, How);
        {erllama, Ref, {done, _}} -> result(T0, Times, Text, How);
        {erllama, Ref, {error, _}} -> result(T0, Times, Text, How)
    after 30000 ->
        result(T0, Times, Text, {How, not_ended})
    end.

result(T0, Times0, Text, How) ->
    Times = lists:reverse(Times0),
    Gaps =
        case Times of
            [] -> [];
            _ -> [(B - A) / 1000 || {A, B} <- lists:zip(lists:droplast(Times), tl(Times))]
        end,
    #{
        ended => How,
        tokens => length(Times),
        text => iolist_to_binary(lists:reverse(Text)),
        ttft_ms =>
            case Times of
                [F | _] -> (F - T0) / 1000;
                [] -> undefined
            end,
        gaps_ms => Gaps
    }.
