%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_ring_tests).
-include_lib("eunit/include/eunit.hrl").

push_n(N, Max) ->
    lists:foldl(
        fun(I, R) ->
            {_, R1} = erllama_server_ring:push(#{i => I}, R),
            R1
        end,
        erllama_server_ring:new(Max),
        lists:seq(0, N - 1)
    ).

seqs({ok, Es}) -> [S || #{seq := S} <- Es].

push_stamps_sequence_test() ->
    {E0, R} = erllama_server_ring:push(#{type => a}, erllama_server_ring:new(3)),
    {E1, _} = erllama_server_ring:push(#{type => b}, R),
    ?assertEqual(#{type => a, seq => 0}, E0),
    ?assertEqual(#{type => b, seq => 1}, E1).

replay_from_cursor_test() ->
    R = push_n(5, 10),
    ?assertEqual([0, 1, 2, 3, 4], seqs(erllama_server_ring:from(0, R))),
    ?assertEqual([3, 4], seqs(erllama_server_ring:from(3, R))),
    ?assertEqual({ok, []}, erllama_server_ring:from(5, R)),
    ?assertEqual({ok, []}, erllama_server_ring:from(99, R)).

overflow_keeps_the_last_events_test() ->
    R = push_n(10, 4),
    ?assertEqual(6, erllama_server_ring:oldest_seq(R)),
    ?assertEqual(10, erllama_server_ring:next_seq(R)),
    ?assertEqual([6, 7, 8, 9], seqs(erllama_server_ring:from(6, R))).

cursor_older_than_buffer_is_a_gap_test() ->
    R = push_n(10, 4),
    ?assertEqual({gap, 6}, erllama_server_ring:from(0, R)),
    ?assertEqual({gap, 6}, erllama_server_ring:from(5, R)).
