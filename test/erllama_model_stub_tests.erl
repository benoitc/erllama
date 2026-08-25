%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Tests for the multi-sequence backend callbacks on the stub
%% backend. These exercise the behaviour surface directly (no
%% gen_statem in the loop) so the round-trips are observable
%% deterministically — the stub never has to allocate a real
%% llama_context.
-module(erllama_model_stub_tests).
-include_lib("eunit/include/eunit.hrl").

stub_step_prefill_returns_prefilled_test() ->
    {ok, S} = erllama_model_stub:init(#{}),
    {ok, [{0, prefilled}, {1, prefilled}]} =
        erllama_model_stub:step(S, [
            {0, {prefill, [1, 2, 3]}},
            {1, {prefill, [4, 5, 6]}}
        ]).

stub_step_decode_is_deterministic_per_seq_and_sampler_test() ->
    {ok, S} = erllama_model_stub:init(#{}),
    {ok, Sampler0} = erllama_model_stub:sampler_new(S, #{}),
    {ok, [{0, {token, T1, 0}}]} =
        erllama_model_stub:step(S, [{0, {decode, Sampler0}}]),
    %% Same seq, same sampler -> same token. The stub is deterministic
    %% by design so cache-integration tests can pin outputs.
    {ok, [{0, {token, T2, 0}}]} =
        erllama_model_stub:step(S, [{0, {decode, Sampler0}}]),
    ?assertEqual(T1, T2).

stub_step_decode_differs_across_seqs_test() ->
    {ok, S} = erllama_model_stub:init(#{}),
    {ok, Sampler} = erllama_model_stub:sampler_new(S, #{}),
    %% A scheduler bug that pairs the wrong sampler with the wrong
    %% seq would break this — different seqs must produce different
    %% tokens for the same sampler.
    {ok, Results} =
        erllama_model_stub:step(S, [
            {0, {decode, Sampler}},
            {1, {decode, Sampler}}
        ]),
    [{0, {token, T0, _}}, {1, {token, T1, _}}] = Results,
    ?assertNotEqual(T0, T1).

stub_step_co_batches_prefill_and_decode_test() ->
    {ok, S} = erllama_model_stub:init(#{}),
    {ok, Sampler} = erllama_model_stub:sampler_new(S, #{}),
    {ok, [{0, {token, T, 0}}, {1, prefilled}]} =
        erllama_model_stub:step(S, [
            {0, {decode, Sampler}},
            {1, {prefill, [10, 20]}}
        ]),
    ?assert(is_integer(T)).

stub_sampler_new_returns_unique_refs_test() ->
    {ok, S} = erllama_model_stub:init(#{}),
    {ok, A} = erllama_model_stub:sampler_new(S, #{}),
    {ok, B} = erllama_model_stub:sampler_new(S, #{}),
    ?assertNotEqual(A, B),
    ok = erllama_model_stub:sampler_free(A),
    ok = erllama_model_stub:sampler_free(B).

stub_kv_pack_seq_id_matches_legacy_test() ->
    %% The stub doesn't track per-seq state separately — the binary
    %% comes from encoding the token list — so the seq-aware and
    %% legacy arities should produce identical bytes for the same
    %% token list. Guards against an accidental divergence in the
    %% stub that could mask scheduler bugs.
    {ok, S} = erllama_model_stub:init(#{}),
    Tokens = [11, 22, 33, 44],
    Legacy = erllama_model_stub:kv_pack(S, Tokens),
    SeqAware = erllama_model_stub:kv_pack(S, Tokens, 1),
    ?assertEqual(Legacy, SeqAware).

stub_kv_unpack_seq_id_is_noop_test() ->
    {ok, S} = erllama_model_stub:init(#{}),
    ?assertEqual(ok, erllama_model_stub:kv_unpack(S, <<>>, 0)),
    ?assertEqual(ok, erllama_model_stub:kv_unpack(S, <<>>, 1)).

stub_seq_rm_is_noop_test() ->
    {ok, S} = erllama_model_stub:init(#{}),
    ?assertEqual(ok, erllama_model_stub:seq_rm(S, 0)),
    ?assertEqual(ok, erllama_model_stub:seq_rm(S, 7)).

%% step_delay_ms knob: each `step/2' call holds for at least the
%% configured number of ms before returning. Used by server-side
%% concurrency tests (e.g. `chat_busy_returns_429') to keep a holder
%% request deterministically in-flight while other requests race for
%% the queue, instead of relying on the test sleeping long enough.
stub_step_delay_ms_holds_step_for_at_least_that_long_test() ->
    {ok, S} = erllama_model_stub:init(#{step_delay_ms => 50}),
    T0 = erlang:monotonic_time(millisecond),
    {ok, _} =
        erllama_model_stub:step(S, [{0, {prefill, [1, 2, 3]}}]),
    T1 = erlang:monotonic_time(millisecond),
    ?assert(T1 - T0 >= 50).

stub_step_delay_ms_default_is_zero_test() ->
    %% Default (knob unset) must add no measurable delay so the cache
    %% / integration tests using the stub keep their hot path.
    {ok, S} = erllama_model_stub:init(#{}),
    T0 = erlang:monotonic_time(millisecond),
    {ok, _} =
        erllama_model_stub:step(S, [{0, {prefill, [1, 2, 3]}}]),
    T1 = erlang:monotonic_time(millisecond),
    ?assert(T1 - T0 < 20).

stub_step_delay_ms_rejects_non_neg_integer_test() ->
    %% Invalid configs (non-integer, negative) coerce to 0, matching
    %% the surrounding `bool_opt/2' tolerance pattern. A typo can't
    %% wedge a test by silently turning into a giant sleep.
    {ok, S} = erllama_model_stub:init(#{step_delay_ms => bad}),
    T0 = erlang:monotonic_time(millisecond),
    {ok, _} =
        erllama_model_stub:step(S, [{0, {prefill, [1, 2, 3]}}]),
    T1 = erlang:monotonic_time(millisecond),
    ?assert(T1 - T0 < 20).

stub_detokenize_opts_matches_plain_test() ->
    %% The stub has no special tokens; the arity-3 form accepts the
    %% options and produces the same bytes as the plain form.
    {ok, S} = erllama_model_stub:init(#{}),
    Tokens = [1, 22, 333],
    ?assertEqual(
        erllama_model_stub:detokenize(S, Tokens),
        erllama_model_stub:detokenize(S, Tokens, #{unparse_special => true})
    ).

stub_step_logprobs_shape_test() ->
    %% A sampler created with `logprobs => N` turns decode results
    %% into the 4-tuple shape with N descending top entries.
    {ok, S} = erllama_model_stub:init(#{}),
    {ok, Sampler} = erllama_model_stub:sampler_new(S, #{logprobs => 2}),
    {ok, [{0, {token, T, 0, {Lp, Top}}}]} =
        erllama_model_stub:step(S, [{0, {decode, Sampler}}]),
    ?assert(is_integer(T)),
    ?assert(is_float(Lp)),
    ?assertEqual(2, length(Top)),
    ok = erllama_model_stub:sampler_free(Sampler),
    %% After free the knob is gone: plain 3-tuples again.
    {ok, Sampler2} = erllama_model_stub:sampler_new(S, #{}),
    {ok, [{0, {token, _, 0}}]} =
        erllama_model_stub:step(S, [{0, {decode, Sampler2}}]).
