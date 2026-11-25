%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
-module(erllama_tp_checkpoint).
-moduledoc false.
%% Experimental: a KV checkpoint of a model split over a tensor-parallel
%% group, as one binary holding every rank's share (rank order) with its
%% size and CRC32C:
%%
%%   <<"ETPK", 2:8, World:16, (Size:64, Crc:32) x World, Share x World>>
%%
%% Version 1 frames were saved before the split backend read mirrored
%% tensors from the rank's own device: their shares hold zeros for those
%% tensors on every rank but the first, so they are refused.
%%
%% One binary, so the cache publishes every share or none; a restore
%% checks the whole frame before any rank applies its share.
%%
%% Rank 0 puts the command in the stream the other ranks follow, after
%% every batch it decoded (erllama_nif:tp_send/2), then packs or restores
%% its own share; each other rank does the same with its share when it
%% reaches the command (answer/3), so every share is at the same
%% committed token.

-export([pack/4, restore/5, answer/3]).
-export([frame/1, unframe/2, gather/3, first_error/1, record/4]).

%% On rank 0, for a group of `World' ranks: every rank's share of
%% `SeqId''s KV, framed; `{error, {incomplete_checkpoint, Ranks}}' when
%% a share did not come within `Timeout' ms.
-spec pack(erllama_nif:context_ref(), non_neg_integer(), pos_integer(), timeout()) ->
    binary() | {error, term()}.
pack(Ctx, SeqId, World, Timeout) ->
    Tag = make_ref(),
    ok = erllama_nif:tp_send(Ctx, {kv_pack, SeqId, Tag, self()}),
    Mine = erllama_nif:tp_kv_pack(Ctx, SeqId),
    case collect(Mine, Tag, World, Timeout, incomplete_checkpoint) of
        {ok, Bin, Others} -> frame([Bin | Others]);
        {error, _} = E -> E
    end.

%% On rank 0: restore a checkpoint on every rank of the group.
-spec restore(erllama_nif:context_ref(), binary(), non_neg_integer(), pos_integer(), timeout()) ->
    ok | {error, term()}.
restore(Ctx, Frame, SeqId, World, Timeout) ->
    case unframe(Frame, World) of
        {ok, [Mine | Others]} ->
            Tag = make_ref(),
            ok = erllama_nif:tp_send(Ctx, {kv_unpack, SeqId, Others, Tag, self()}),
            Local = erllama_nif:tp_kv_unpack(Ctx, Mine, SeqId),
            case collect(Local, Tag, World, Timeout, incomplete_restore) of
                {ok, ok, Acks} -> first_error(Acks);
                {error, _} = E -> E
            end;
        {error, _} = E ->
            E
    end.

%% Rank 0's own result with the other ranks' answers.
collect({error, _} = E, _Tag, _World, _Timeout, _What) ->
    E;
collect(Mine, Tag, World, Timeout, What) ->
    case gather(Tag, lists:seq(1, World - 1), Timeout) of
        {ok, Others} -> {ok, Mine, Others};
        {missing, Ranks} -> {error, {What, Ranks}}
    end.

%% On rank `Rank' (not 0), reaching a checkpoint command: its share, or
%% its restore, answered to rank 0.
-spec answer(tuple(), erllama_nif:context_ref(), pos_integer()) -> ok.
answer({kv_pack, SeqId, Tag, To}, Ctx, Rank) ->
    To ! {erllama_tp_kv, Tag, Rank, erllama_nif:tp_kv_pack(Ctx, SeqId)},
    ok;
answer({kv_unpack, SeqId, Shares, Tag, To}, Ctx, Rank) ->
    To ! {erllama_tp_kv, Tag, Rank, erllama_nif:tp_kv_unpack(Ctx, lists:nth(Rank, Shares), SeqId)},
    ok.

%% The shares in rank order, or the first error among them.
-spec frame([binary() | {error, term()}]) -> binary() | {error, term()}.
frame(Shares) ->
    case [E || {error, _} = E <- Shares] of
        [] ->
            Index = [<<(byte_size(B)):64, (erllama_nif:crc32c(B)):32>> || B <- Shares],
            iolist_to_binary([<<"ETPK", 2:8, (length(Shares)):16>>, Index, Shares]);
        [E | _] ->
            E
    end.

%% The shares of a frame made for `World' ranks, each checked.
-spec unframe(binary(), pos_integer()) -> {ok, [binary()]} | {error, bad_checkpoint}.
unframe(<<"ETPK", 2:8, W:16, Rest/binary>>, W) ->
    IndexBytes = W * 12,
    case Rest of
        <<Index:IndexBytes/binary, Data/binary>> ->
            shares([{Sz, Crc} || <<Sz:64, Crc:32>> <= Index], Data, []);
        _ ->
            {error, bad_checkpoint}
    end;
unframe(_, _W) ->
    {error, bad_checkpoint}.

shares([], <<>>, Acc) ->
    {ok, lists:reverse(Acc)};
shares([{Sz, Crc} | Rest], Data, Acc) ->
    case Data of
        <<B:Sz/binary, More/binary>> ->
            case erllama_nif:crc32c(B) of
                Crc -> shares(Rest, More, [B | Acc]);
                _ -> {error, bad_checkpoint}
            end;
        _ ->
            {error, bad_checkpoint}
    end;
shares(_, _, _) ->
    {error, bad_checkpoint}.

%% Every listed rank's `{erllama_tp_kv, Tag, Rank, Result}', in rank
%% order, or the ranks still missing after `Timeout' ms.
-spec gather(reference(), [pos_integer()], timeout()) ->
    {ok, [term()]} | {missing, [pos_integer()]}.
gather(Tag, Ranks, Timeout) ->
    gather(Tag, Ranks, Timeout, #{}).

gather(_Tag, [], _T, Got) ->
    {ok, [V || {_, V} <- lists:sort(maps:to_list(Got))]};
gather(Tag, Waiting, T, Got) ->
    receive
        {erllama_tp_kv, Tag, Rank, Result} ->
            gather(Tag, lists:delete(Rank, Waiting), T, Got#{Rank => Result})
    after T ->
        {missing, Waiting}
    end.

-spec first_error([term()]) -> ok | {error, term()}.
first_error(Results) ->
    case [E || {error, _} = E <- Results] of
        [] -> ok;
        [E | _] -> E
    end.

%% Rank 0's checkpoint stats after a pack or a restore (`What'): how
%% many were made and failed, and the last one's time, size and result.
-spec record(pack | unpack, number(), term(), map()) -> map().
record(What, Ms, R, Stats) ->
    {Result, Bytes} =
        case R of
            Bin when is_binary(Bin) -> {ok, byte_size(Bin)};
            ok -> {ok, 0};
            Err -> {Err, 0}
        end,
    Key = fun(Suffix) -> list_to_atom(atom_to_list(What) ++ Suffix) end,
    Count = Key(
        case Result of
            ok -> "_ok";
            _ -> "_failed"
        end
    ),
    Stats#{
        What => Result,
        Key("_ms") => Ms,
        Key("_bytes") => Bytes,
        Count => maps:get(Count, Stats, 0) + 1
    }.
