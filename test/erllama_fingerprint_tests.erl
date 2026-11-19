%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_fingerprint_tests).
-include_lib("eunit/include/eunit.hrl").

%% A file with the given contents in a fresh temp directory; Body gets
%% its path. The directory is removed afterwards.
with_file(Contents, Body) ->
    Dir = erllama_test_helpers:tmp_dir("erllama_fingerprint"),
    Path = filename:join(Dir, "model.gguf"),
    ok = file:write_file(Path, Contents),
    try
        Body(Path)
    after
        erllama_test_helpers:rm_rf(Dir)
    end.

contents() ->
    <<"GGUF", (crypto:strong_rand_bytes(4096))/binary>>.

safe_is_the_plain_file_sha256_test() ->
    C = contents(),
    with_file(C, fun(P) ->
        ?assertEqual({ok, crypto:hash(sha256, C)}, erllama_fingerprint:derive(P, safe))
    end).

derivation_is_deterministic_test() ->
    with_file(contents(), fun(P) ->
        [
            ?assertEqual(erllama_fingerprint:derive(P, M), erllama_fingerprint:derive(P, M))
         || M <- [safe, gguf_chunked, fast_unsafe]
        ]
    end).

modes_never_share_a_value_test() ->
    with_file(contents(), fun(P) ->
        Fps = [
            element(2, erllama_fingerprint:derive(P, M))
         || M <- [safe, gguf_chunked, fast_unsafe]
        ],
        ?assertEqual(3, length(lists:usort(Fps))),
        [?assertEqual(32, byte_size(Fp)) || Fp <- Fps]
    end).

gguf_chunked_changes_with_head_tail_and_size_test() ->
    C = contents(),
    <<First:8, Rest/binary>> = C,
    Size = byte_size(C),
    Mid = binary:part(C, 0, Size - 1),
    <<Last:8>> = binary:part(C, Size - 1, 1),
    Variants = [
        <<(First bxor 1):8, Rest/binary>>,
        <<Mid/binary, (Last bxor 1):8>>,
        <<C/binary, 0>>
    ],
    {ok, Base} = with_file(C, fun(P) -> erllama_fingerprint:derive(P, gguf_chunked) end),
    [
        ?assertNotEqual(
            {ok, Base}, with_file(V, fun(P) -> erllama_fingerprint:derive(P, gguf_chunked) end)
        )
     || V <- Variants
    ].

fast_unsafe_changes_with_mtime_test() ->
    with_file(contents(), fun(P) ->
        {ok, A} = erllama_fingerprint:derive(P, fast_unsafe),
        ok = file:change_time(P, {{2020, 1, 1}, {0, 0, 0}}),
        {ok, B} = erllama_fingerprint:derive(P, fast_unsafe),
        ?assertNotEqual(A, B)
    end).

missing_file_is_an_error_test() ->
    [
        ?assertMatch({error, enoent}, erllama_fingerprint:derive("/nonexistent/x.gguf", M))
     || M <- [safe, gguf_chunked, fast_unsafe]
    ].
