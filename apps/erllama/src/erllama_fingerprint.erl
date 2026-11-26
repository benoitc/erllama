%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_fingerprint).
-moduledoc false.
%% Model fingerprint derived from the GGUF file, used when a model is
%% loaded without an explicit `fingerprint`. The fingerprint is the
%% namespace of every cache key, so two different models must never
%% share one.
%%
%% Modes (`fingerprint_mode`):
%%   safe          SHA-256 of the whole file. The same value you get
%%                 from `crypto:hash(sha256, FileBytes)`, so it matches
%%                 a fingerprint computed by hand. Costs a full read.
%%   gguf_chunked  SHA-256 of the size, the first 64 MiB (GGUF header,
%%                 metadata, start of the weights) and the last 4 MiB.
%%                 Constant cost whatever the file size. The default.
%%   fast_unsafe   SHA-256 of the absolute path, size and mtime. No
%%                 read at all; a file replaced in place with the same
%%                 size and mtime keeps its fingerprint.
%% gguf_chunked and fast_unsafe hash a mode tag first, so no mode can
%% produce another mode's value.

-include_lib("kernel/include/file.hrl").

-export([derive/2]).

-export_type([mode/0]).

-type mode() :: safe | gguf_chunked | fast_unsafe.

-define(HEAD_BYTES, 64 * 1024 * 1024).
-define(TAIL_BYTES, 4 * 1024 * 1024).
-define(READ_CHUNK, 8 * 1024 * 1024).

-spec derive(file:filename_all(), mode()) -> {ok, <<_:256>>} | {error, term()}.
derive(Path, Mode) ->
    case erllama_gguf_shards:paths(Path) of
        {ok, [Only]} ->
            derive_file(Only, Mode);
        {ok, Paths} ->
            derive_shards(
                Paths,
                Mode,
                crypto:hash_update(crypto:hash_init(sha256), <<"erllama:gguf-shards:v1">>)
            );
        {error, _} = E ->
            E
    end.

derive_shards([], _Mode, Hash) ->
    {ok, crypto:hash_final(Hash)};
derive_shards([Path | Rest], Mode, Hash) ->
    case derive_file(Path, Mode) of
        {ok, Digest} -> derive_shards(Rest, Mode, crypto:hash_update(Hash, Digest));
        {error, Reason} -> {error, {shard, Path, Reason}}
    end.

derive_file(Path, fast_unsafe) ->
    case file:read_file_info(Path, [{time, posix}]) of
        {ok, #file_info{size = Size, mtime = MTime}} ->
            Abs = unicode:characters_to_binary(filename:absname(Path)),
            {ok,
                crypto:hash(sha256, [
                    <<"erllama:fast_unsafe">>, <<Size:64, MTime:64/signed>>, Abs
                ])};
        {error, _} = E ->
            E
    end;
derive_file(Path, gguf_chunked) ->
    with_file(Path, fun(Fd, Size) ->
        Ctx0 = crypto:hash_update(crypto:hash_init(sha256), [
            <<"erllama:gguf_chunked">>, <<Size:64>>
        ]),
        HeadLen = min(Size, ?HEAD_BYTES),
        TailStart = max(HeadLen, Size - ?TAIL_BYTES),
        case hash_range(Fd, 0, HeadLen, Ctx0) of
            {ok, Ctx1} -> hash_range(Fd, TailStart, Size - TailStart, Ctx1);
            {error, _} = E -> E
        end
    end);
derive_file(Path, safe) ->
    with_file(Path, fun(Fd, Size) ->
        hash_range(Fd, 0, Size, crypto:hash_init(sha256))
    end).

with_file(Path, Fun) ->
    case file:read_file_info(Path) of
        {ok, #file_info{size = Size}} ->
            case file:open(Path, [read, raw, binary]) of
                {ok, Fd} ->
                    try Fun(Fd, Size) of
                        {ok, Ctx} -> {ok, crypto:hash_final(Ctx)};
                        {error, _} = E -> E
                    after
                        ok = file:close(Fd)
                    end;
                {error, _} = E ->
                    E
            end;
        {error, _} = E ->
            E
    end.

hash_range(_Fd, _Offset, 0, Ctx) ->
    {ok, Ctx};
hash_range(Fd, Offset, Len, Ctx) ->
    case file:pread(Fd, Offset, min(Len, ?READ_CHUNK)) of
        {ok, Bin} ->
            N = byte_size(Bin),
            hash_range(Fd, Offset + N, Len - N, crypto:hash_update(Ctx, Bin));
        eof ->
            {error, truncated};
        {error, _} = E ->
            E
    end.
