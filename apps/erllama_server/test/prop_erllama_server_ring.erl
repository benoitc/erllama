%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(prop_erllama_server_ring).
-include_lib("proper/include/proper.hrl").

%% Whatever was pushed and whatever cursor a subscriber comes back
%% with, it gets either the events from exactly that cursor to the
%% newest, contiguous, or a gap naming the oldest event still held,
%% which is newer than the cursor.
prop_replay_is_contiguous_or_a_gap() ->
    ?FORALL(
        {Max, N, Cursor},
        {range(1, 16), range(0, 64), range(0, 80)},
        begin
            R = lists:foldl(
                fun(I, Acc) ->
                    {_, Acc1} = erllama_server_ring:push(#{i => I}, Acc),
                    Acc1
                end,
                erllama_server_ring:new(Max),
                lists:seq(1, N)
            ),
            Oldest = max(0, N - Max),
            case erllama_server_ring:from(Cursor, R) of
                {gap, O} ->
                    O =:= Oldest andalso O > Cursor;
                {ok, Es} ->
                    Seqs = [S || #{seq := S} <- Es],
                    Cursor >= Oldest andalso
                        Seqs =:= lists:seq(Cursor, max(Cursor, N) - 1) andalso
                        [I || #{i := I} <- Es] =:= [S + 1 || S <- Seqs]
            end
        end
    ).
