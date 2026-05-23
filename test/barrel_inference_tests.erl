%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Unit tests for top-level barrel_inference:* helpers that do not need a
%% loaded model. Functions that exercise the full inference path
%% live in barrel_inference_streaming_tests, barrel_inference_lora_tests, etc.
-module(barrel_inference_tests).
-include_lib("eunit/include/eunit.hrl").

%% =============================================================================
%% list_cached_prefixes/2
%% =============================================================================

list_cached_prefixes_empty_prompt_test() ->
    %% Short-circuits before touching the registry, so no app start
    %% required.
    ?assertEqual({ok, 0}, barrel_inference:list_cached_prefixes(<<"any">>, [])).

list_cached_prefixes_unloaded_model_test() ->
    %% Same: the registry lookup is the only side-effect, and it
    %% returns undefined for a model that was never loaded.
    {ok, _} = application:ensure_all_started(barrel_inference),
    try
        ?assertEqual(
            {error, model_not_loaded},
            barrel_inference:list_cached_prefixes(<<"never-loaded-model">>, [1, 2, 3])
        )
    after
        ok = application:stop(barrel_inference)
    end.

%% =============================================================================
%% draft_tokens/3
%% =============================================================================

draft_tokens_empty_prefix_test() ->
    %% Short-circuits before infer/4, so no app required.
    ?assertEqual(
        {error, empty_prefix},
        barrel_inference:draft_tokens(<<"any">>, [], #{max => 4})
    ).

%% =============================================================================
%% verify/4 unit tests (the snapshot/restore proof lives in the
%% barrel_inference_real_model_SUITE since it needs a loaded GGUF + the
%% existing test infrastructure for cache + fingerprints).
%% =============================================================================

verify_rejects_non_binary_model_id_test() ->
    ?assertError(function_clause, barrel_inference:verify(not_a_binary, [1], [2], 1)).
