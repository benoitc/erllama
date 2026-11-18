%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Pure-Erlang unit tests for erllama_model_llama. The NIF-backed
%% paths (init/1, step/2, etc.) require a real GGUF and live in
%% erllama_real_model_SUITE; this module covers the small helpers
%% that have no native dependency.
-module(erllama_model_llama_tests).
-include_lib("eunit/include/eunit.hrl").

%% thinking_signature/3 reads the node-wide signing key from the
%% application environment and HMAC-SHA256s the supplied bytes.
%% Without a key, the function falls back to <<>> -- the documented
%% "no signature available" path.

thinking_signature_returns_empty_without_key_test() ->
    application:unset_env(erllama, thinking_signing_key),
    ?assertEqual(
        <<>>,
        erllama_model_llama:thinking_signature(undefined, 0, <<"some thinking text">>)
    ).

thinking_signature_returns_hmac_with_key_test() ->
    Key = <<"unit-test-key">>,
    Bytes = <<"some thinking text">>,
    application:set_env(erllama, thinking_signing_key, Key),
    try
        Sig = erllama_model_llama:thinking_signature(undefined, 0, Bytes),
        ?assert(is_binary(Sig)),
        ?assertEqual(32, byte_size(Sig)),
        ?assertEqual(crypto:mac(hmac, sha256, Key, Bytes), Sig)
    after
        application:unset_env(erllama, thinking_signing_key)
    end.

thinking_signature_is_deterministic_test() ->
    Key = <<"unit-test-key">>,
    application:set_env(erllama, thinking_signing_key, Key),
    try
        Bytes = <<"identical input">>,
        S1 = erllama_model_llama:thinking_signature(undefined, 0, Bytes),
        S2 = erllama_model_llama:thinking_signature(undefined, 0, Bytes),
        ?assertEqual(S1, S2)
    after
        application:unset_env(erllama, thinking_signing_key)
    end.

thinking_signature_empty_key_treated_as_unset_test() ->
    application:set_env(erllama, thinking_signing_key, <<>>),
    try
        ?assertEqual(
            <<>>,
            erllama_model_llama:thinking_signature(undefined, 0, <<"bytes">>)
        )
    after
        application:unset_env(erllama, thinking_signing_key)
    end.

%% kv_format/1 decides which cache rows a context may restore: it must
%% change with every option that changes the KV payload layout, ignore
%% the others, and not depend on map construction order.

kv_format_changes_with_kv_layout_test() ->
    Base = #{type_k => f16, type_v => f16, flash_attn => auto, n_seq_max => 1},
    Fmt = erllama_model_llama:kv_format(Base),
    [
        ?assertNotEqual(Fmt, erllama_model_llama:kv_format(maps:put(K, V, Base)))
     || {K, V} <- [
            {type_k, q8_0},
            {type_v, q8_0},
            {flash_attn, enabled},
            {n_seq_max, 4},
            {kv_unified, true}
        ]
    ].

kv_format_ignores_other_opts_test() ->
    Base = #{type_k => q8_0},
    ?assertEqual(
        erllama_model_llama:kv_format(Base),
        erllama_model_llama:kv_format(Base#{n_ctx => 8192, n_threads => 8, decode_budget_ms => 100})
    ).

kv_format_is_order_independent_test() ->
    A = maps:from_list([{type_k, q8_0}, {n_seq_max, 2}, {flash_attn, auto}]),
    B = maps:from_list([{flash_attn, auto}, {n_seq_max, 2}, {type_k, q8_0}]),
    ?assertEqual(erllama_model_llama:kv_format(A), erllama_model_llama:kv_format(B)).

kv_format_flash_attn_aliases_name_one_layout_test() ->
    ?assertEqual(
        erllama_model_llama:kv_format(#{flash_attn => true}),
        erllama_model_llama:kv_format(#{flash_attn => enabled})
    ),
    ?assertEqual(
        erllama_model_llama:kv_format(#{flash_attn => false}),
        erllama_model_llama:kv_format(#{flash_attn => disabled})
    ).

kv_format_auto_is_the_default_layout_test() ->
    Default = erllama_model_llama:kv_format(#{}),
    ?assertEqual(Default, erllama_model_llama:kv_format(#{type_k => auto, type_v => auto})),
    ?assertEqual(Default, erllama_model_llama:kv_format(#{flash_attn => auto})).

kv_format_starts_with_engine_build_test() ->
    Engine = erllama_nif:kv_format(),
    Fmt = erllama_model_llama:kv_format(#{}),
    ?assertEqual(
        <<Engine/binary, "/k=default,v=default,fa=default,seq=default,unified=default">>, Fmt
    ).
