%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_stream).
-moduledoc false.
%% Receive-loop helpers over the `{erllama, Ref, Event}' stream
%% protocol. `collect/2' is the loop a caller of `erllama:stream/3'
%% would otherwise write; `drain/2' empties the mailbox of a
%% cancelled request.

-export([collect/2, collect_token_ids/2, drain/2]).

-doc """
Wait for the request `Ref` to finish and fold its events into a
`completion_result()`-shaped map: `reply`, `generated`, `thinking`,
`stats`, `finish_reason`, `finish_key`, `cache_hit_kind`,
`cache_delta`, `committed_tokens`, `context_tokens` and
`stop_sequence` when one fired. `{error, timeout}` after `Timeout`
milliseconds without any event; the request is cancelled and its
remaining events drained.
""".
-spec collect(reference(), timeout()) -> {ok, map()} | {error, term()}.
collect(Ref, Timeout) when is_reference(Ref) ->
    Init = #{text => [], thinking => [], token_ids => []},
    case fold(Ref, Timeout, fun collect_event/2, Init) of
        {ok, Acc, Stats} -> {ok, result(Acc, Stats)};
        {error, _} = E -> E
    end.

collect_event({token, Bin}, Acc) ->
    Acc#{text := [maps:get(text, Acc), Bin]};
collect_event({token_id, Id}, Acc) ->
    Acc#{token_ids := [Id | maps:get(token_ids, Acc)]};
collect_event({thinking, Bin}, Acc) ->
    Acc#{thinking := [maps:get(thinking, Acc), Bin]};
collect_event(_Other, Acc) ->
    Acc.

result(#{text := Text, thinking := Thinking}, Stats) ->
    Generated = maps:get(generated, Stats, []),
    Base = #{
        reply => iolist_to_binary(Text),
        thinking => iolist_to_binary(Thinking),
        generated => Generated,
        committed_tokens => maps:get(committed_tokens, Stats, 0),
        finish_key => maps:get(finish_key, Stats, undefined),
        cache_hit_kind => maps:get(cache_hit_kind, Stats, cold),
        finish_reason => maps:get(finish_reason, Stats, stop),
        cache_delta => maps:get(cache_delta, Stats, #{read => 0, created => 0}),
        stats => Stats
    },
    case maps:find(stop_sequence, Stats) of
        {ok, Stop} -> Base#{stop_sequence => Stop};
        error -> Base
    end.

-doc """
Collect only the generated token ids of `Ref`, in order. Used by
`erllama:draft_tokens/3`. `{error, timeout}` after `Timeout` ms of
silence (request cancelled, mailbox drained).
""".
-spec collect_token_ids(reference(), timeout()) -> {ok, [non_neg_integer()]} | {error, term()}.
collect_token_ids(Ref, Timeout) ->
    Fun = fun
        ({token_id, Id}, Acc) -> [Id | Acc];
        (_Other, Acc) -> Acc
    end,
    case fold(Ref, Timeout, Fun, []) of
        {ok, Acc, _Stats} -> {ok, lists:reverse(Acc)};
        {error, _} = E -> E
    end.

%% Fold the non-terminal events of `Ref' with `Fun'; returns the
%% accumulator together with the `done' stats, the `error' reason, or
%% `{error, timeout}' after `Timeout' ms of silence (the request is
%% cancelled and its remaining events drained).
fold(Ref, Timeout, Fun, Acc) ->
    receive
        {erllama, Ref, {done, Stats}} -> {ok, Acc, Stats};
        {erllama, Ref, {error, Reason}} -> {error, Reason};
        {erllama, Ref, Event} -> fold(Ref, Timeout, Fun, Fun(Event, Acc))
    after Timeout ->
        ok = erllama_model:cancel(Ref),
        ok = drain(Ref, 100),
        {error, timeout}
    end.

-doc """
Discard every event of `Ref` still in the mailbox. Returns once the
terminal event has been seen or after `Tail` milliseconds of silence.
""".
-spec drain(reference(), timeout()) -> ok.
drain(Ref, Tail) ->
    receive
        {erllama, Ref, {done, _}} -> ok;
        {erllama, Ref, {error, _}} -> ok;
        {erllama, Ref, _} -> drain(Ref, Tail)
    after Tail -> ok
    end.
