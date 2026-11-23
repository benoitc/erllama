%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_cache_key_tests).
-include_lib("eunit/include/eunit.hrl").

fp() -> binary:copy(<<16#AA>>, 32).
ctx() -> binary:copy(<<16#BB>>, 32).

make_returns_32_bytes_test() ->
    Key = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => f16,
        ctx_params_hash => ctx(),
        text => <<"1 2 3">>
    }),
    ?assertEqual(32, byte_size(Key)).

make_is_deterministic_test() ->
    Components = #{
        fingerprint => fp(),
        quant_type => q4_k_m,
        ctx_params_hash => ctx(),
        text => <<"1 2 3 4">>
    },
    ?assertEqual(
        erllama_cache_key:make(Components),
        erllama_cache_key:make(Components)
    ).

%% The key is over the rendered bytes, not the token-id list: two
%% token sequences that detokenise to the same bytes produce the
%% same key. make/1 and make_text/4 must agree.
make_text_matches_make_test() ->
    Key1 = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => f16,
        ctx_params_hash => ctx(),
        text => <<"hello world">>
    }),
    Key2 = erllama_cache_key:make_text(fp(), f16, ctx(), <<"hello world">>),
    ?assertEqual(Key1, Key2).

make_distinct_for_text_change_test() ->
    A = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => f16,
        ctx_params_hash => ctx(),
        text => <<"1 2 3">>
    }),
    B = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => f16,
        ctx_params_hash => ctx(),
        text => <<"1 2 4">>
    }),
    ?assertNotEqual(A, B).

%% A byte-prefix has a distinct key from the full bytes: the
%% longest-prefix scan relies on each stored byte length keying
%% uniquely.
make_distinct_for_prefix_test() ->
    Full = erllama_cache_key:make_text(fp(), f16, ctx(), <<"abcde">>),
    Prefix = erllama_cache_key:make_text(fp(), f16, ctx(), <<"abc">>),
    ?assertNotEqual(Full, Prefix).

namespace_is_32_bytes_and_text_independent_test() ->
    %% The namespace folds in fp/quant/ctx but NOT the prompt bytes, so
    %% it groups all keys produced by the same model/quant/context.
    Ns = erllama_cache_key:namespace(fp(), q4_k_m, ctx()),
    ?assertEqual(32, byte_size(Ns)),
    ?assertEqual(Ns, erllama_cache_key:namespace(fp(), q4_k_m, ctx())).

namespace_distinct_for_quant_and_fp_test() ->
    A = erllama_cache_key:namespace(fp(), f16, ctx()),
    B = erllama_cache_key:namespace(fp(), q4_k_m, ctx()),
    C = erllama_cache_key:namespace(binary:copy(<<16#AB>>, 32), f16, ctx()),
    ?assertNotEqual(A, B),
    ?assertNotEqual(A, C).

with_kv_format_undefined_is_identity_test() ->
    ?assertEqual(ctx(), erllama_cache_key:with_kv_format(ctx(), undefined)).

with_kv_format_separates_layouts_test() ->
    A = erllama_cache_key:with_kv_format(ctx(), <<"llama.cpp/b10593/seq-v2">>),
    B = erllama_cache_key:with_kv_format(ctx(), <<"llama.cpp/v1.0.0/seq-v4">>),
    ?assertEqual(32, byte_size(A)),
    ?assertNotEqual(ctx(), A),
    ?assertNotEqual(A, B),
    ?assertEqual(A, erllama_cache_key:with_kv_format(ctx(), <<"llama.cpp/b10593/seq-v2">>)).

make_distinct_for_quant_change_test() ->
    A = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => f16,
        ctx_params_hash => ctx(),
        text => <<"1">>
    }),
    B = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => q4_k_m,
        ctx_params_hash => ctx(),
        text => <<"1">>
    }),
    ?assertNotEqual(A, B).

make_distinct_for_fingerprint_change_test() ->
    A = erllama_cache_key:make(#{
        fingerprint => binary:copy(<<16#AA>>, 32),
        quant_type => f16,
        ctx_params_hash => ctx(),
        text => <<"1">>
    }),
    B = erllama_cache_key:make(#{
        fingerprint => binary:copy(<<16#AB>>, 32),
        quant_type => f16,
        ctx_params_hash => ctx(),
        text => <<"1">>
    }),
    ?assertNotEqual(A, B).

make_distinct_for_ctx_change_test() ->
    A = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => f16,
        ctx_params_hash => binary:copy(<<16#BB>>, 32),
        text => <<"1">>
    }),
    B = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => f16,
        ctx_params_hash => binary:copy(<<16#BC>>, 32),
        text => <<"1">>
    }),
    ?assertNotEqual(A, B).

make_badarg_short_fingerprint_test() ->
    ?assertError(
        function_clause,
        erllama_cache_key:make(#{
            fingerprint => <<1, 2, 3>>,
            quant_type => f16,
            ctx_params_hash => ctx(),
            text => <<>>
        })
    ).

quant_byte_round_trip_test() ->
    Quants = [
        f32,
        f16,
        bf16,
        q4_0,
        q4_1,
        q5_0,
        q5_1,
        q8_0,
        q2_k,
        q3_k_s,
        q3_k_m,
        q3_k_l,
        q4_k_m,
        q4_k_s,
        q5_k_m,
        q5_k_s,
        q6_k,
        q8_k,
        iq1_s,
        iq1_m,
        iq2_xxs,
        iq2_xs,
        iq2_s,
        iq2_m,
        iq3_xxs,
        iq3_xs,
        iq3_s,
        iq3_m,
        iq4_nl,
        iq4_xs,
        q2_k_s,
        tq1_0,
        tq2_0,
        mxfp4_moe,
        nvfp4,
        q1_0,
        q2_0,
        f8_e4m3
    ],
    [
        ?assertEqual(
            {ok, Q}, erllama_cache_key:quant_atom(erllama_cache_key:quant_byte(Q))
        )
     || Q <- Quants
    ],
    %% Distinct bytes, none of them the 255 catch-all.
    Bytes = [erllama_cache_key:quant_byte(Q) || Q <- Quants],
    ?assertEqual(length(Bytes), length(lists:usort(Bytes))),
    ?assertNot(lists:member(255, Bytes)).

quant_atom_unknown_test() ->
    ?assertEqual({error, unknown_quant}, erllama_cache_key:quant_atom(254)).

quant_atom_reads_back_the_catch_all_byte_test() ->
    %% quant_byte/1 writes 255 for a label it does not know; a file
    %% carrying that byte must stay readable.
    ?assertEqual(
        {ok, unknown}, erllama_cache_key:quant_atom(erllama_cache_key:quant_byte(future_q9_x))
    ).

from_ftype_maps_llama_ftypes_test() ->
    ?assertEqual({f32, 32}, erllama_cache_key:from_ftype(0)),
    ?assertEqual({f16, 16}, erllama_cache_key:from_ftype(1)),
    ?assertEqual({q8_0, 8}, erllama_cache_key:from_ftype(7)),
    ?assertEqual({q4_k_m, 4}, erllama_cache_key:from_ftype(15)),
    ?assertEqual({bf16, 16}, erllama_cache_key:from_ftype(32)),
    ?assertEqual({f8_e4m3, 8}, erllama_cache_key:from_ftype(42)),
    %% The GUESSED flag does not change the type.
    ?assertEqual({q4_k_m, 4}, erllama_cache_key:from_ftype(15 bor 1024)),
    %% Removed upstream values, out of range and "unavailable" (-1).
    [?assertEqual(undefined, erllama_cache_key:from_ftype(F)) || F <- [4, 5, 6, 33, 43, -1]].

from_ftype_types_have_a_quant_byte_test() ->
    Detected = [erllama_cache_key:from_ftype(F) || F <- lists:seq(0, 42)],
    [
        ?assertNotEqual(255, erllama_cache_key:quant_byte(QT))
     || {QT, _Bits} <- Detected
    ].

quant_byte_unknown_atom_test() ->
    %% Catch-all clause maps any unknown atom to byte 255 so cache
    %% key derivation never crashes the model gen_statem.
    ?assertEqual(255, erllama_cache_key:quant_byte(future_q9_x)).

encode_decode_tokens_round_trip_test() ->
    Tokens = [0, 1, 16#FFFFFFFF, 12345, 67890],
    Bin = erllama_cache_key:encode_tokens(Tokens),
    ?assertEqual(20, byte_size(Bin)),
    ?assertEqual(Tokens, erllama_cache_key:decode_tokens(Bin)).

encode_empty_tokens_test() ->
    ?assertEqual(<<>>, erllama_cache_key:encode_tokens([])),
    ?assertEqual([], erllama_cache_key:decode_tokens(<<>>)).
