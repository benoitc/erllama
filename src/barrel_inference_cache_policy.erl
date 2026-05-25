%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% Pure-Erlang policy decisions for the barrel_inference_cache subsystem.
%%
%% Two responsibilities:
%%
%%   1. Boundary trim: cold saves persist a *trimmed-aligned prefix* of
%%      the prompt rather than the full live token list, so the next
%%      request whose prompt is a textual extension of this one still
%%      lands on the saved cache key after BPE retokenisation. The
%%      algorithm trims a fixed number of tokens off the tail and
%%      aligns the result down to a multiple of a configured chunk.
%%
%%   2. Save-reason gating: cold/continued/finish saves each have a
%%      simple guard (token-count thresholds and intervals). Eviction
%%      and shutdown saves are unconditional and do not pass through
%%      this module.
%%
%% This module has no side effects; everything is testable as plain
%% data transformations.
%% @end
-module(barrel_inference_cache_policy).

-export([
    trim_boundary/3,
    cold_save_split/2,
    cold_save_segments/2,
    should_continued_save/3,
    should_finish_save/2,
    validate_config/1
]).

-export_type([config/0, token/0]).

-type token() :: non_neg_integer().

-type config() :: #{
    min_tokens := non_neg_integer(),
    cold_min_tokens := non_neg_integer(),
    cold_max_tokens := non_neg_integer(),
    continued_interval := pos_integer(),
    boundary_trim_tokens := non_neg_integer(),
    boundary_align_tokens := pos_integer(),
    %% Cold-save checkpoint ladder. Optional: when absent, no ladder
    %% checkpoints are written (`cold_save_segments/2` yields one remainder
    %% segment). `max_ladder_rows` is the maximum number of cold-save
    %% checkpoints; `ladder_interval` is their spacing (aligned down to
    %% boundary_align_tokens).
    ladder_interval => pos_integer(),
    max_ladder_rows => non_neg_integer(),
    session_resume_wait_ms => non_neg_integer(),
    prefill_chunk_size => pos_integer() | infinity
}.

-define(DEFAULT_LADDER_INTERVAL, 16384).
-define(DEFAULT_MAX_LADDER_ROWS, 0).

%% =============================================================================
%% Boundary trim
%% =============================================================================

-spec trim_boundary([token()], non_neg_integer(), pos_integer()) ->
    {ok, [token()]} | {skip, too_short}.
trim_boundary(Tokens, Trim, Align) when
    is_list(Tokens), is_integer(Trim), Trim >= 0, is_integer(Align), Align > 0
->
    Len = length(Tokens),
    case trim_count(Len, Trim, Align) of
        {ok, N} -> {ok, lists:sublist(Tokens, N)};
        {skip, Reason} -> {skip, Reason}
    end.

-spec trim_count(non_neg_integer(), non_neg_integer(), pos_integer()) ->
    {ok, non_neg_integer()} | {skip, too_short}.
trim_count(Len, Trim, Align) ->
    AfterTrim = Len - Trim,
    case AfterTrim < Align of
        true -> {skip, too_short};
        false -> {ok, (AfterTrim div Align) * Align}
    end.

%% =============================================================================
%% Save-reason gating
%% =============================================================================

%% Decide whether a cold save fires, and if so, return both the trimmed
%% prefix to pack/save and the remaining tokens still to be prefilled
%% into the live context.
-spec cold_save_split([token()], config()) ->
    {trim, [token()], [token()]} | no_save.
cold_save_split(Tokens, Cfg) ->
    Len = length(Tokens),
    Min = maps:get(cold_min_tokens, Cfg),
    Max = maps:get(cold_max_tokens, Cfg),
    Trim = maps:get(boundary_trim_tokens, Cfg),
    Align = maps:get(boundary_align_tokens, Cfg),
    case Len < Min orelse Len > Max of
        true ->
            no_save;
        false ->
            case trim_count(Len, Trim, Align) of
                {ok, N} ->
                    {Prefix, Rest} = lists:split(N, Tokens),
                    {trim, Prefix, Rest};
                {skip, _} ->
                    no_save
            end
    end.

%% Split a cold prompt into prefill SEGMENTS for the checkpoint ladder. A
%% cold save fires at each internal segment boundary; the final segment is
%% the non-saving sub-align remainder (covered by the finish save). The
%% boundaries are: up to `max_ladder_rows` stride-aligned ladder points
%% spaced ~`ladder_interval` apart, PLUS the legacy trim boundary (the
%% largest aligned prefix, `cold_save_split`'s boundary). So with
%% `max_ladder_rows = 0` (or the key absent) the result is exactly the
%% legacy single-checkpoint behaviour; a positive value adds that many
%% head checkpoints below the trim boundary. Every boundary is a multiple
%% of `boundary_align_tokens`, so the longest-prefix walk (which probes
%% only stride-aligned lengths) can hit every persisted row. Returns
%% `[Tokens]` (one remainder, no cold save) when the prompt is out of the
%% [cold_min_tokens, cold_max_tokens] band or too short to align.
-spec cold_save_segments([token()], config()) -> [[token()]].
cold_save_segments(Tokens, Cfg) ->
    Len = length(Tokens),
    Min = maps:get(cold_min_tokens, Cfg),
    Max = maps:get(cold_max_tokens, Cfg),
    Trim = maps:get(boundary_trim_tokens, Cfg),
    Align = maps:get(boundary_align_tokens, Cfg),
    case Len < Min orelse Len > Max of
        true ->
            [Tokens];
        false ->
            case trim_count(Len, Trim, Align) of
                {skip, _} ->
                    [Tokens];
                {ok, TrimBoundary} ->
                    Ladder = ladder_boundaries(TrimBoundary, Min, Align, Cfg),
                    split_at(Ladder ++ [TrimBoundary], Tokens)
            end
    end.

%% Stride-aligned ladder points strictly below the trim boundary, spaced
%% ~ladder_interval apart, at most max_ladder_rows of them (ascending).
-spec ladder_boundaries(non_neg_integer(), non_neg_integer(), pos_integer(), config()) ->
    [pos_integer()].
ladder_boundaries(TrimBoundary, Min, Align, Cfg) ->
    MaxRows = maps:get(max_ladder_rows, Cfg, ?DEFAULT_MAX_LADDER_ROWS),
    Interval = maps:get(ladder_interval, Cfg, ?DEFAULT_LADDER_INTERVAL),
    Step = max(Align, (Interval div Align) * Align),
    [B || K <- lists:seq(1, MaxRows), B <- [K * Step], B >= Min, B < TrimBoundary].

%% Cut Tokens at the given absolute cumulative positions, returning the
%% segments plus the trailing remainder (which may be empty when the last
%% boundary is the full length).
-spec split_at([pos_integer()], [token()]) -> [[token()]].
split_at(Boundaries, Tokens) -> split_at(Boundaries, 0, Tokens, []).

split_at([], _Pos, Rest, Acc) ->
    lists:reverse([Rest | Acc]);
split_at([B | Bs], Pos, Tokens, Acc) ->
    {Seg, Rest} = lists:split(B - Pos, Tokens),
    split_at(Bs, B, Rest, [Seg | Acc]).

%% Continued saves fire every `continued_interval` tokens of *new*
%% generation (i.e. live token count minus the count at the last save).
-spec should_continued_save(non_neg_integer(), non_neg_integer(), config()) ->
    boolean().
should_continued_save(LiveCount, LastSavedAtCount, Cfg) when
    is_integer(LiveCount),
    LiveCount >= 0,
    is_integer(LastSavedAtCount),
    LastSavedAtCount >= 0
->
    Interval = maps:get(continued_interval, Cfg),
    Min = maps:get(min_tokens, Cfg),
    LiveCount - LastSavedAtCount >= Interval andalso LiveCount >= Min.

%% Finish saves fire at successful end-of-stream provided the live
%% sequence is at or above the global minimum.
-spec should_finish_save(non_neg_integer(), config()) -> boolean().
should_finish_save(LiveCount, Cfg) when is_integer(LiveCount), LiveCount >= 0 ->
    LiveCount >= maps:get(min_tokens, Cfg).

%% =============================================================================
%% Config validation
%% =============================================================================

-spec validate_config(map()) -> ok | {error, term()}.
validate_config(Cfg) ->
    Required = [
        min_tokens,
        cold_min_tokens,
        cold_max_tokens,
        continued_interval,
        boundary_trim_tokens,
        boundary_align_tokens
    ],
    case [K || K <- Required, not maps:is_key(K, Cfg)] of
        [] -> check_invariants(Cfg);
        Missing -> {error, {missing_keys, Missing}}
    end.

-spec check_invariants(config()) -> ok | {error, term()}.
check_invariants(Cfg) ->
    Min = maps:get(min_tokens, Cfg),
    ColdMin = maps:get(cold_min_tokens, Cfg),
    ColdMax = maps:get(cold_max_tokens, Cfg),
    Interval = maps:get(continued_interval, Cfg),
    Trim = maps:get(boundary_trim_tokens, Cfg),
    Align = maps:get(boundary_align_tokens, Cfg),
    %% Ladder keys are optional; validate the effective value.
    LadderInterval = maps:get(ladder_interval, Cfg, ?DEFAULT_LADDER_INTERVAL),
    MaxRows = maps:get(max_ladder_rows, Cfg, ?DEFAULT_MAX_LADDER_ROWS),
    Checks = [
        {is_integer(Min) andalso Min >= 0, {invalid, min_tokens, Min}},
        {is_integer(ColdMin) andalso ColdMin >= Min, {ordering, cold_min_tokens_lt_min_tokens}},
        {
            is_integer(ColdMax) andalso ColdMax >= ColdMin,
            {ordering, cold_max_tokens_lt_cold_min_tokens}
        },
        {is_integer(Interval) andalso Interval > 0, {invalid, continued_interval, Interval}},
        {is_integer(Trim) andalso Trim >= 0, {invalid, boundary_trim_tokens, Trim}},
        {is_integer(Align) andalso Align > 0, {invalid, boundary_align_tokens, Align}},
        {
            is_integer(LadderInterval) andalso LadderInterval > 0,
            {invalid, ladder_interval, LadderInterval}
        },
        {is_integer(MaxRows) andalso MaxRows >= 0, {invalid, max_ladder_rows, MaxRows}}
    ],
    case lists:dropwhile(fun({Pass, _}) -> Pass end, Checks) of
        [] -> ok;
        [{_, Reason} | _] -> {error, Reason}
    end.
