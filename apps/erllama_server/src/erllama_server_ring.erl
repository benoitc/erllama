%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_ring).
-moduledoc false.
%% Bounded buffer of a run's events. Every pushed event gets the next
%% sequence number (from 0). Only the last `Max' events are kept, so a
%% subscriber that reattaches with a cursor either gets a contiguous
%% replay or is told the oldest sequence still held.

-export([new/1, push/2, from/2, next_seq/1, oldest_seq/1]).
-export_type([ring/0]).

-record(ring, {
    max :: pos_integer(),
    next = 0 :: non_neg_integer(),
    len = 0 :: non_neg_integer(),
    q = queue:new() :: queue:queue(map())
}).
-opaque ring() :: #ring{}.

-spec new(pos_integer()) -> ring().
new(Max) when is_integer(Max), Max > 0 ->
    #ring{max = Max}.

%% Stamp `Event' with the next sequence number and keep it.
-spec push(map(), ring()) -> {map(), ring()}.
push(Event, #ring{max = Max, next = Next, len = Len, q = Q} = R) when is_map(Event) ->
    E = Event#{seq => Next},
    Q1 = queue:in(E, Q),
    R1 =
        case Len < Max of
            true -> R#ring{next = Next + 1, len = Len + 1, q = Q1};
            false -> R#ring{next = Next + 1, q = queue:drop(Q1)}
        end,
    {E, R1}.

%% Events from sequence `Cursor' on. A cursor older than the oldest
%% kept event is a gap: the caller decides whether to continue from
%% `Oldest'. A cursor at or past the end returns nothing.
-spec from(non_neg_integer(), ring()) -> {ok, [map()]} | {gap, non_neg_integer()}.
from(Cursor, #ring{next = Next, len = Len, q = Q}) when is_integer(Cursor), Cursor >= 0 ->
    Oldest = Next - Len,
    case Cursor of
        _ when Cursor < Oldest -> {gap, Oldest};
        _ when Cursor >= Next -> {ok, []};
        _ -> {ok, lists:nthtail(Cursor - Oldest, queue:to_list(Q))}
    end.

-spec next_seq(ring()) -> non_neg_integer().
next_seq(#ring{next = Next}) -> Next.

-spec oldest_seq(ring()) -> non_neg_integer().
oldest_seq(#ring{next = Next, len = Len}) -> Next - Len.
