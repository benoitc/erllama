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

%% kv_namespace/1 decides which cache rows a context may restore, so
%% it must change with every option that changes the KV payload
%% layout and must not depend on map construction order.

kv_namespace_changes_with_kv_layout_test() ->
    Base = #{n_ctx => 4096, n_seq_max => 1, type_k => f16, type_v => f16},
    NsBase = erllama_model_llama:kv_namespace(Base),
    [
        ?assertNotEqual(NsBase, erllama_model_llama:kv_namespace(maps:put(K, V, Base)))
     || {K, V} <- [
            {n_ctx, 8192},
            {n_seq_max, 4},
            {type_k, q8_0},
            {type_v, q8_0},
            {flash_attn, true}
        ]
    ].

kv_namespace_ignores_non_layout_opts_test() ->
    Base = #{n_ctx => 4096},
    ?assertEqual(
        erllama_model_llama:kv_namespace(Base),
        erllama_model_llama:kv_namespace(Base#{n_threads => 8, decode_budget_ms => 100})
    ).

kv_namespace_is_order_independent_test() ->
    A = maps:from_list([{n_ctx, 4096}, {type_k, q8_0}, {n_seq_max, 2}]),
    B = maps:from_list([{n_seq_max, 2}, {type_k, q8_0}, {n_ctx, 4096}]),
    ?assertEqual(erllama_model_llama:kv_namespace(A), erllama_model_llama:kv_namespace(B)).

kv_namespace_carries_engine_info_test() ->
    Ns = erllama_model_llama:kv_namespace(#{}),
    {Engine, _Shape} = binary_to_term(Ns),
    ?assertEqual(erllama_nif:engine_info(), Engine).
