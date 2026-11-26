%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_shards).
-moduledoc false.

-export([inspect/1, link_set/1, blob/1]).

-spec inspect([binary(), ...]) -> {ok, non_neg_integer()} | {error, term()}.
inspect(Paths) -> inspect(Paths, 0, length(Paths), 0).

inspect([], _, _, Size) ->
    {ok, Size};
inspect([Path | Rest], I, Count, Size) ->
    case erllama_server_gguf:read(Path) of
        {ok, #{kv := KV}} ->
            case {maps:get(<<"split.no">>, KV, 0), maps:get(<<"split.count">>, KV, 1)} of
                {I, Count} -> inspect(Rest, I + 1, Count, Size + filelib:file_size(Path));
                Actual -> {error, {shard_metadata, Path, {I, Count}, Actual}}
            end;
        {error, enoent} when Count > 1 -> {error, {missing_shard, Path}};
        {error, enoent} ->
            {error, {enoent, Path}};
        {error, Reason} ->
            {error, {not_a_gguf, Path, Reason}}
    end.

-spec blob(binary()) -> binary().
blob(Sha) ->
    filename:join(erllama_server_registry:blobs_dir(), <<"sha256-", Sha/binary, ".gguf">>).

%% llama.cpp opens siblings by name. Hard links keep the verified download
%% reusable across sets and cost no additional weight storage.
-spec link_set([binary(), ...]) -> {ok, binary(), binary()} | {error, term()}.
link_set(Digests) ->
    Set = binary:encode_hex(crypto:hash(sha256, Digests), lowercase),
    Prefix = filename:join(erllama_server_registry:blobs_dir(), <<"set-", Set/binary>>),
    [First | _] = Paths = erllama_gguf_shards:paths(Prefix, length(Digests)),
    case links(lists:zip(Digests, Paths)) of
        ok -> {ok, First, Set};
        Error -> Error
    end.

links([]) ->
    ok;
links([{Sha, Path} | Rest]) ->
    %% The registry serializes publication. Replace a link if a damaged
    %% content blob was downloaded again under a new inode.
    Tmp = <<Path/binary, ".link">>,
    _ = file:delete(Tmp),
    case file:make_link(blob(Sha), Tmp) of
        ok ->
            case file:rename(Tmp, Path) of
                ok ->
                    _ = file:delete(Tmp),
                    links(Rest);
                {error, _} = E ->
                    _ = file:delete(Tmp),
                    E
            end;
        {error, _} = E ->
            E
    end.
