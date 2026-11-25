%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Test helper: one sequence's KV state as llama.cpp writes it
%% (llama_state_seq_get_data) for a hybrid model: the attention cache,
%% the recurrent state, then (GLM-5.3-Flash) the indexer cache. The format
%% holds each cell's position and rows in the order of the cells in the
%% cache, and no cell address; equal/2 compares two states by token
%% position, so the same state laid out in other cells compares equal.
%% Values are not transposed (flash attention).
-module(erllama_kv_seq_state).

-export([parse/1, equal/2]).

%% {ok, #{seq => Seq, attn => KV, recr => Binary, idx => KV}} (after the
%% state header: magic and sequence id), KV = #{Pos => {SeqIds,
%% [Row]}} (a row per layer, keys then values); idx is #{} for a model
%% without an indexer.
parse(Bin) ->
    first([fun() -> parse(Bin, V) end || V <- [keys_only, keys_values]], no_layout).

%% true, or where two states first differ.
equal(A, B) ->
    case {parse(A), parse(B)} of
        {{ok, SA}, {ok, SB}} -> compare(SA, SB);
        {E, {ok, _}} -> {unparsable, a, E};
        {_, E} -> {unparsable, b, E}
    end.

compare(#{seq := S1}, #{seq := S2}) when S1 =/= S2 ->
    {seq, S1, S2};
compare(#{attn := A1, recr := R1, idx := I1}, #{attn := A2, recr := R2, idx := I2}) ->
    case {cells(A1, A2), R1 =:= R2, cells(I1, I2)} of
        {true, true, true} -> true;
        {true, true, D} -> {idx, D};
        {true, false, _} -> recr;
        {D, _, _} -> {attn, D}
    end.

cells(K1, K2) when map_size(K1) =/= map_size(K2) ->
    {cell_count, map_size(K1), map_size(K2)};
cells(K1, K2) ->
    case [P || P <- lists:sort(maps:keys(K1)), maps:get(P, K1) =/= maps:get(P, K2, missing)] of
        [] -> true;
        [P | _] -> {pos, P}
    end.

parse(<<16#af143cd8:32/little, Seq:32/little-signed, Bin/binary>>, V) ->
    try
        {Attn, Rest} = kv(Bin, V),
        {ok, State} = recr_then_idx(Rest, Attn, V),
        {ok, State#{seq => Seq}}
    catch
        throw:Why -> {error, Why};
        error:{badmatch, _} -> {error, truncated}
    end;
parse(_Bin, _V) ->
    {error, no_state_header}.

first([], Why) ->
    {error, Why};
first([F | Fs], _Why) ->
    case F() of
        {ok, _} = Ok -> Ok;
        {error, Why} -> first(Fs, Why)
    end.

%% The recurrent section: its cells (one per sequence), s_trans,
%% n_layer, then a row entry per recurrent layer of each state. The
%% number of entries is not written, so the section ends where the rest
%% reads as a whole indexer section (or is empty).
recr_then_idx(Bin, Attn, V) ->
    <<NCells:32/little, R0/binary>> = Bin,
    {_, R1} = meta(NCells, R0, []),
    <<_STrans:32/little, _NLayer:32/little, Entries/binary>> = R1,
    Head = byte_size(Bin) - byte_size(Entries),
    split([Head + E || E <- entry_ends(NCells, Entries, 0, [])], Bin, Attn, V).

split([], _Bin, _Attn, _V) ->
    throw(no_section_boundary);
split([End | Ends], Bin, Attn, V) ->
    <<Recr:End/binary, Tail/binary>> = Bin,
    case Tail of
        <<>> ->
            {ok, #{attn => Attn, recr => Recr, idx => #{}}};
        _ ->
            case catch kv(Tail, V) of
                {Idx, <<>>} when is_map(Idx) -> {ok, #{attn => Attn, recr => Recr, idx => Idx}};
                _ -> split(Ends, Bin, Attn, V)
            end
    end.

%% Where each plausible row entry ends, longest first.
entry_ends(NCells, <<Type:32/little-signed, Row:64/little, More/binary>>, Off, Acc) when
    Type >= 0, Type < 64, Row * NCells =< byte_size(More)
->
    N = Row * NCells,
    <<_:N/binary, Next/binary>> = More,
    End = Off + 12 + N,
    entry_ends(NCells, Next, End, [End | Acc]);
entry_ends(_NCells, _Bin, _Off, Acc) ->
    Acc.

%% A KV cache section: its cells by position.
kv(<<NStream:32/little, Rest/binary>>, V) when NStream >= 1, NStream =< 256 ->
    streams(NStream, Rest, V, #{});
kv(_, _V) ->
    throw(bad_kv_section).

streams(0, Rest, _V, Acc) ->
    {Acc, Rest};
streams(N, <<0:32/little, Rest/binary>>, V, Acc) ->
    streams(N - 1, Rest, V, Acc);
streams(N, <<NCells:32/little, Rest/binary>>, V, Acc) ->
    {Meta, R1} = meta(NCells, Rest, []),
    <<VTrans:32/little, NLayer:32/little, _RotK:32/little, _RotV:32/little, R2/binary>> = R1,
    VTrans =:= 0 orelse throw(transposed_values),
    {Keys, R3} = layers(NLayer, NCells, R2, []),
    {Values, R4} =
        case V of
            keys_only -> {[], R3};
            keys_values -> layers(NLayer, NCells, R3, [])
        end,
    Rows = Keys ++ Values,
    Cells = maps:from_list([
        {Pos, {Seqs, [lists:nth(I, L) || L <- Rows]}}
     || {I, {Pos, Seqs}} <- lists:enumerate(Meta)
    ]),
    streams(N - 1, R4, V, maps:merge(Acc, Cells)).

meta(0, Rest, Acc) ->
    {lists:reverse(Acc), Rest};
meta(N, <<Pos:32/little-signed, NSeq:32/little, Rest/binary>>, Acc) when NSeq =< 256 ->
    Bytes = NSeq * 4,
    <<Seqs:Bytes/binary, More/binary>> = Rest,
    meta(N - 1, More, [{Pos, Seqs} | Acc]);
meta(_N, _Bin, _Acc) ->
    throw(bad_cell).

%% A row entry per layer: type, row size, a row per cell.
layers(0, _NCells, Rest, Acc) ->
    {lists:reverse(Acc), Rest};
layers(N, NCells, <<_Type:32/little, Row:64/little, Rest/binary>>, Acc) when Row > 0 ->
    Bytes = Row * NCells,
    <<Data:Bytes/binary, More/binary>> = Rest,
    layers(N - 1, NCells, More, [[R || <<R:Row/binary>> <= Data] | Acc]);
layers(_N, _NCells, _Bin, _Acc) ->
    throw(bad_layer).
