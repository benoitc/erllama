%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Test helper for the tensor-parallel suites, loaded onto the serving
%% node: stream one request through erllama and time it.
-module(erllama_tp_stream).

-export([run/4, run_with/5, run_many/4, chat_tokens/2, kill/2]).

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

%% Stream `Prompt' and call `{M, F, A}' with the request's ref appended
%% once `At' tokens arrived (0: right after the request started); then collect every message the request
%% sends until `QuietMs' pass without one. Returns the tokens, the
%% terminal messages (there must be one), and the time from `Fun' to
%% the first terminal message.
run_with(Model, Prompt, Max, {At, Fun}, QuietMs) ->
    {ok, Ref} = erllama:stream(Model, Prompt, #{response_tokens => Max, temperature => 0.0}),
    T = maybe_fire(0, At, Fun, Ref, undefined),
    drain(Ref, At, Fun, QuietMs, 0, T, []).

drain(Ref, At, Fun, QuietMs, N, T, Terminals) ->
    receive
        {erllama, Ref, {token, _}} ->
            drain(Ref, At, Fun, QuietMs, N + 1, maybe_fire(N + 1, At, Fun, Ref, T), Terminals);
        {erllama, Ref, {done, Stats}} ->
            drain(Ref, At, Fun, QuietMs, N, T, [{done, Stats, now_us()} | Terminals]);
        {erllama, Ref, {error, Why}} ->
            drain(Ref, At, Fun, QuietMs, N, T, [{error, Why, now_us()} | Terminals])
    after QuietMs ->
        Ts = lists:reverse(Terminals),
        #{
            tokens => N,
            terminals => [{K, V} || {K, V, _} <- Ts],
            to_terminal_ms =>
                case {T, Ts} of
                    {undefined, _} -> undefined;
                    {_, []} -> undefined;
                    {_, [{_, _, At1} | _]} -> (At1 - T) / 1000
                end
        }
    end.

maybe_fire(N, N, {M, F, A}, Ref, undefined) ->
    _ = apply(M, F, A ++ [Ref]),
    now_us();
maybe_fire(_, _, _, _, T) ->
    T.

now_us() -> erlang:monotonic_time(microsecond).

%% For run_with/5: kill a rank's OS process.
kill(RankPid, _Ref) -> erllama_tp_rank:kill(RankPid).

%% Stream every prompt of `Prompts' at once (greedy, at most `Max'
%% tokens each); the one at index `CancelIdx' (1-based, or `none') is
%% cancelled after `CancelAfter' tokens. Returns each stream's result as
%% run/4 does, in order, and the wall time and total tokens.
run_many(Model, Prompts, Max, {CancelIdx, CancelAfter}) ->
    Self = self(),
    T0 = erlang:monotonic_time(microsecond),
    Pids = [
        spawn_link(fun() ->
            After =
                case I of
                    CancelIdx -> CancelAfter;
                    _ -> infinity
                end,
            Self ! {self(), run(Model, P, Max, After)}
        end)
     || {I, P} <- lists:enumerate(Prompts)
    ],
    Runs = [
        receive
            {Pid, R} -> R
        end
     || Pid <- Pids
    ],
    Ms = (erlang:monotonic_time(microsecond) - T0) / 1000,
    #{runs => Runs, wall_ms => Ms, tokens => lists:sum([maps:get(tokens, R) || R <- Runs])}.

%% `Text' as one user turn through the model's chat template, thinking
%% off, as token ids (the template's markers as special tokens).
chat_tokens(Model, Text) ->
    {ok, #{prompt := P}} = erllama:chat_apply(
        Model, [#{role => user, content => Text}], #{enable_thinking => false}
    ),
    {ok, Toks} = erllama:tokenize(Model, P, #{parse_special => true, add_special => false}),
    Toks.

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
