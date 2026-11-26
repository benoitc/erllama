%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_gguf_shards).
-moduledoc false.
-export([parts/1, paths/2, paths/1]).

-spec parts(file:filename_all()) -> single | {binary(), non_neg_integer(), non_neg_integer()}.
parts(Path) ->
    case
        re:run(Path, <<"^(.+)-([0-9]{5})-of-([0-9]{5})[.]gguf$">>, [
            {capture, all_but_first, binary}
        ])
    of
        {match, [Prefix, Index, Total]} ->
            {Prefix, binary_to_integer(Index), binary_to_integer(Total)};
        nomatch ->
            single
    end.

-spec paths(file:filename_all()) ->
    {ok, [binary()]} | {error, {invalid_shard, file:filename_all()}}.
paths(Path) ->
    case parts(Path) of
        single -> {ok, [unicode:characters_to_binary(Path)]};
        {Prefix, I, N} when I >= 1, I =< N -> {ok, paths(Prefix, N)};
        _ -> {error, {invalid_shard, Path}}
    end.

-spec paths(binary(), pos_integer()) -> [binary()].
paths(Prefix, N) ->
    [
        iolist_to_binary([Prefix, io_lib:format("-~5..0B-of-~5..0B.gguf", [I, N])])
     || I <- lists:seq(1, N)
    ].
