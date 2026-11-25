%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
-module(erllama_tp_checkpoint_tests).

-include_lib("eunit/include/eunit.hrl").

round_trip_test() ->
    Shares = [<<"rank zero">>, <<>>, <<"rank two">>],
    Frame = erllama_tp_checkpoint:frame(Shares),
    ?assertEqual({ok, Shares}, erllama_tp_checkpoint:unframe(Frame, 3)).

a_share_missing_is_no_checkpoint_test() ->
    ?assertEqual(
        {error, timeout}, erllama_tp_checkpoint:frame([<<"a">>, {error, timeout}])
    ).

another_world_is_refused_test() ->
    Frame = erllama_tp_checkpoint:frame([<<"a">>, <<"b">>]),
    ?assertEqual({error, bad_checkpoint}, erllama_tp_checkpoint:unframe(Frame, 3)).

a_damaged_share_is_refused_test() ->
    Frame = erllama_tp_checkpoint:frame([<<"aaaa">>, <<"bbbb">>]),
    Damaged = binary:replace(Frame, <<"bbbb">>, <<"bbbc">>),
    ?assertEqual({error, bad_checkpoint}, erllama_tp_checkpoint:unframe(Damaged, 2)).

a_truncated_frame_is_refused_test() ->
    Frame = erllama_tp_checkpoint:frame([<<"aaaa">>, <<"bbbb">>]),
    Short = binary:part(Frame, 0, byte_size(Frame) - 1),
    ?assertEqual({error, bad_checkpoint}, erllama_tp_checkpoint:unframe(Short, 2)),
    ?assertEqual({error, bad_checkpoint}, erllama_tp_checkpoint:unframe(<<Frame/binary, 0>>, 2)).

gather_in_rank_order_test() ->
    Tag = make_ref(),
    self() ! {erllama_tp_kv, Tag, 2, two},
    self() ! {erllama_tp_kv, Tag, 1, one},
    ?assertEqual({ok, [one, two]}, erllama_tp_checkpoint:gather(Tag, [1, 2], 1000)).

gather_reports_the_missing_ranks_test() ->
    Tag = make_ref(),
    self() ! {erllama_tp_kv, Tag, 1, one},
    ?assertEqual({missing, [2]}, erllama_tp_checkpoint:gather(Tag, [1, 2], 50)).

%% Frames saved before the mirrored-tensor fix (version 1) are refused.
a_version_1_frame_is_refused_test() ->
    <<"ETPK", 2:8, Rest/binary>> = erllama_tp_checkpoint:frame([<<"aaaa">>, <<"bbbb">>]),
    ?assertEqual(
        {error, bad_checkpoint}, erllama_tp_checkpoint:unframe(<<"ETPK", 1:8, Rest/binary>>, 2)
    ).

%% A split checkpoint saved before the fix (`/tp=' in the KV format) is
%% under another cache key than the same prompt after it, so it misses.
checkpoints_from_before_the_fix_miss_test() ->
    Fmt = <<"llama.cpp v1.0.0/state 2/n_ctx=4096">>,
    Layout = <<"2:even">>,
    Before = <<Fmt/binary, "/tp=", Layout/binary>>,
    After = erllama_model_tp:kv_format(Fmt, Layout),
    ?assertNotEqual(Before, After),
    Ctx = crypto:hash(sha256, <<"context params">>),
    Key = fun(F) ->
        erllama_cache_key:make(#{
            fingerprint => crypto:hash(sha256, <<"model">>),
            quant_type => q4_k_m,
            ctx_params_hash => erllama_cache_key:with_kv_format(Ctx, F),
            text => <<"the same prompt">>
        })
    end,
    ?assertNotEqual(Key(Before), Key(After)).
