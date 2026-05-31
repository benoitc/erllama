%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Pure-Erlang unit tests for barrel_inference_model_llama. The NIF-backed
%% paths (init/1, step/2, etc.) require a real GGUF and live in
%% barrel_inference_real_model_SUITE; this module covers the small helpers
%% that have no native dependency.
-module(barrel_inference_model_llama_tests).
-include_lib("eunit/include/eunit.hrl").

%% thinking_signature/3 reads the node-wide signing key from the
%% application environment and HMAC-SHA256s the supplied bytes.
%% Without a key, the function falls back to <<>> -- the documented
%% "no signature available" path.

thinking_signature_returns_empty_without_key_test() ->
    application:unset_env(barrel_inference, thinking_signing_key),
    ?assertEqual(
        <<>>,
        barrel_inference_model_llama:thinking_signature(undefined, 0, <<"some thinking text">>)
    ).

thinking_signature_returns_hmac_with_key_test() ->
    Key = <<"unit-test-key">>,
    Bytes = <<"some thinking text">>,
    application:set_env(barrel_inference, thinking_signing_key, Key),
    try
        Sig = barrel_inference_model_llama:thinking_signature(undefined, 0, Bytes),
        ?assert(is_binary(Sig)),
        ?assertEqual(32, byte_size(Sig)),
        ?assertEqual(crypto:mac(hmac, sha256, Key, Bytes), Sig)
    after
        application:unset_env(barrel_inference, thinking_signing_key)
    end.

thinking_signature_is_deterministic_test() ->
    Key = <<"unit-test-key">>,
    application:set_env(barrel_inference, thinking_signing_key, Key),
    try
        Bytes = <<"identical input">>,
        S1 = barrel_inference_model_llama:thinking_signature(undefined, 0, Bytes),
        S2 = barrel_inference_model_llama:thinking_signature(undefined, 0, Bytes),
        ?assertEqual(S1, S2)
    after
        application:unset_env(barrel_inference, thinking_signing_key)
    end.

thinking_signature_empty_key_treated_as_unset_test() ->
    application:set_env(barrel_inference, thinking_signing_key, <<>>),
    try
        ?assertEqual(
            <<>>,
            barrel_inference_model_llama:thinking_signature(undefined, 0, <<"bytes">>)
        )
    after
        application:unset_env(barrel_inference, thinking_signing_key)
    end.

%% `tool_call_end_is_eos/1' is the contract the scheduler reads to
%% decide whether to flush the in-span tool_call_bytes buffer on EOS
%% (the EOS-bounded end-marker path, opt-in via the `<<"$eos">>'
%% sentinel on `tool_call_markers.end'). Both shipped backends MUST
%% export it - the llama backend returns true when the model was
%% initialised with the sentinel; the stub backend returns false
%% (no marker plumbing). Functional verification on a real
%% backend is in `barrel_inference_real_model_SUITE' once the
%% Granite family lands and exercises the path.

llama_backend_exports_tool_call_end_is_eos_test() ->
    ?assert(
        erlang:function_exported(barrel_inference_model_llama, tool_call_end_is_eos, 1)
    ).

stub_backend_tool_call_end_is_eos_returns_false_test() ->
    ?assert(
        erlang:function_exported(barrel_inference_model_stub, tool_call_end_is_eos, 1)
    ),
    ?assertNot(barrel_inference_model_stub:tool_call_end_is_eos(any_state)).
