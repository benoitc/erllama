%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Multimodal requests over the stub backend: the scheduler's media
%% path (sole-request prefill, admission gates, cache bypass, stats)
%% without a GGUF or mmproj. The stub enables media with the
%% `media_caps' load knob and prefills deterministically
%% (`media_n_pos' positions per item). Real projector coverage lives
%% in erllama_vision_SUITE.
-module(erllama_media_tests).
-include_lib("eunit/include/eunit.hrl").

%% =============================================================================
%% Fixtures
%% =============================================================================

with_app(Body) ->
    {ok, Started} = application:ensure_all_started(erllama),
    Dir = make_tmp_dir(),
    DiskSrv = list_to_atom(
        "media_disk_" ++ integer_to_list(erlang:unique_integer([positive]))
    ),
    {ok, _} = erllama_cache_disk_srv:start_link(DiskSrv, Dir),
    try
        Body(DiskSrv)
    after
        erllama_test_helpers:stop_quiet(DiskSrv),
        erllama_test_helpers:rm_rf(Dir),
        [application:stop(A) || A <- lists:reverse(Started)],
        ok
    end.

minimal_config(DiskSrv) ->
    #{
        backend => erllama_model_stub,
        tier_srv => DiskSrv,
        tier => disk,
        fingerprint => binary:copy(<<16#77>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        quant_bits => 16,
        ctx_params_hash => binary:copy(<<16#78>>, 32),
        context_size => 1024,
        media_caps => #{vision => true, audio => false},
        policy => #{
            min_tokens => 4,
            cold_min_tokens => 4,
            cold_max_tokens => 1000,
            continued_interval => 2048,
            boundary_trim_tokens => 0,
            boundary_align_tokens => 1,
            session_resume_wait_ms => 50
        }
    }.

with_model(Overrides, Body) ->
    with_app(fun(DiskSrv) ->
        Id = iolist_to_binary([
            "media_", integer_to_binary(erlang:unique_integer([positive]))
        ]),
        Cfg = maps:merge(minimal_config(DiskSrv), Overrides),
        {ok, _} = erllama:load_model(Id, Cfg),
        try
            Body(Id)
        after
            erllama:unload(Id)
        end
    end).

make_tmp_dir() ->
    Base = os:getenv("TMPDIR", "/tmp"),
    Dir = filename:join(
        Base,
        "erllama_media_tests_" ++
            integer_to_list(erlang:unique_integer([positive]))
    ),
    ok = file:make_dir(Dir),
    Dir.

collect_stream(Ref, TimeoutMs) ->
    collect_stream(Ref, TimeoutMs, []).

collect_stream(Ref, TimeoutMs, Acc) ->
    receive
        {erllama, Ref, {token, Bin}} ->
            collect_stream(Ref, TimeoutMs, [Bin | Acc]);
        {erllama, Ref, {done, Stats}} ->
            {lists:reverse(Acc), Stats};
        {erllama, Ref, {error, Reason}} ->
            {error, Reason}
    after TimeoutMs ->
        {timeout, lists:reverse(Acc)}
    end.

img() -> #{type => image, data => <<"stub image bytes">>}.

%% =============================================================================
%% Happy paths
%% =============================================================================

stream_with_media_generates_test() ->
    with_model(#{media_n_pos => 7}, fun(Id) ->
        {ok, Ref} = erllama:stream(Id, <<"look at <__media__> now">>, #{
            response_tokens => 5,
            media => [img()]
        }),
        {Out, Stats} = collect_stream(Ref, 5000),
        ?assertEqual(5, length(Out)),
        %% 4 prompt words + 7 positions for the one item.
        ?assertEqual(
            #{items => 1, n_tokens => 11, n_pos => 11},
            maps:get(media, Stats)
        ),
        %% Media requests bypass the cache: no finish checkpoint.
        ?assertEqual(undefined, maps:get(finish_key, Stats))
    end).

complete_with_media_generates_test() ->
    with_model(#{}, fun(Id) ->
        {ok, R} = erllama:complete(Id, <<"see <__media__>">>, #{
            response_tokens => 3,
            media => [img()]
        }),
        ?assertEqual(3, length(maps:get(generated, R))),
        ?assertMatch(#{items := 1}, maps:get(media, R)),
        ?assertEqual(undefined, maps:get(finish_key, R))
    end).

two_media_items_counted_test() ->
    with_model(#{media_n_pos => 3}, fun(Id) ->
        {ok, R} = erllama:complete(Id, <<"a <__media__> b <__media__>">>, #{
            response_tokens => 2,
            media => [img(), img()]
        }),
        %% 4 prompt words + 2 * 3 positions.
        ?assertEqual(
            #{items => 2, n_tokens => 10, n_pos => 10},
            maps:get(media, R)
        )
    end).

%% =============================================================================
%% Gates
%% =============================================================================

audio_unsupported_by_projector_test() ->
    with_model(#{}, fun(Id) ->
        ?assertEqual(
            {error, {unsupported_media, audio}},
            erllama:stream(Id, <<"a <__media__>">>, #{
                media => [#{type => audio, data => <<"wav">>}]
            })
        )
    end).

no_mmproj_rejected_test() ->
    with_app(fun(DiskSrv) ->
        Cfg = maps:remove(media_caps, minimal_config(DiskSrv)),
        {ok, _} = erllama:load_model(<<"media_none">>, Cfg),
        try
            ?assertEqual(
                {error, no_mmproj},
                erllama:stream(<<"media_none">>, <<"n <__media__>">>, #{
                    media => [img()]
                })
            )
        after
            erllama:unload(<<"media_none">>)
        end
    end).

session_with_media_rejected_test() ->
    with_model(#{}, fun(Id) ->
        ?assertEqual(
            {error, {unsupported_with_media, session_id}},
            erllama:stream(Id, <<"s <__media__>">>, #{
                media => [img()], session_id => sess
            })
        )
    end).

speculative_with_media_rejected_test() ->
    with_model(#{}, fun(Id) ->
        ?assertEqual(
            {error, {unsupported_with_media, speculative}},
            erllama:stream(Id, <<"s <__media__>">>, #{
                media => [img()], speculative => true
            })
        )
    end).

token_prompt_with_media_rejected_test() ->
    with_model(#{}, fun(Id) ->
        ?assertEqual(
            {error, {invalid_option, media, requires_binary_prompt}},
            erllama:stream(Id, [1, 2, 3], #{media => [img()]})
        )
    end).

continue_with_media_rejected_test() ->
    with_model(#{}, fun(Id) ->
        ?assertEqual(
            {error, {unsupported_with_media, continue}},
            erllama:continue(Id, [1, 2], #{
                session_id => s, media => [img()]
            })
        )
    end).

%% =============================================================================
%% Scheduling and failures
%% =============================================================================

media_waits_for_sole_active_test() ->
    with_model(
        #{context_opts => #{n_seq_max => 2}, step_delay_ms => 20},
        fun(Id) ->
            %% A short text request is in flight; the media request
            %% must wait for it and still complete afterwards.
            {ok, HolderRef} = erllama:stream(Id, <<"holder one">>, #{
                response_tokens => 4
            }),
            {ok, MediaRef} = erllama:stream(Id, <<"m <__media__>">>, #{
                response_tokens => 3, media => [img()]
            }),
            {_HOut, _} = collect_stream(HolderRef, 30000),
            {Out, Stats} = collect_stream(MediaRef, 30000),
            ?assertEqual(3, length(Out)),
            ?assertMatch(#{items := 1}, maps:get(media, Stats))
        end
    ).

cancel_pending_media_test() ->
    with_model(
        #{context_opts => #{n_seq_max => 2}, step_delay_ms => 50},
        fun(Id) ->
            %% Keep the media request waiting behind a long holder,
            %% cancel it before its prefill ever runs.
            {ok, HolderRef} = erllama:stream(Id, <<"holder one">>, #{
                response_tokens => 500
            }),
            {ok, MediaRef} = erllama:stream(Id, <<"m <__media__>">>, #{
                response_tokens => 3, media => [img()]
            }),
            ok = erllama:cancel(MediaRef),
            {_Out, Stats} = collect_stream(MediaRef, 5000),
            ?assertEqual(cancelled, maps:get(finish_reason, Stats)),
            ok = erllama:cancel(HolderRef)
        end
    ).

media_prefill_failure_fails_request_test() ->
    with_model(#{fail_media_prefill => true}, fun(Id) ->
        {ok, Ref} = erllama:stream(Id, <<"m <__media__>">>, #{
            response_tokens => 3, media => [img()]
        }),
        ?assertEqual(
            {error, {media_failed, media_decode_failed}},
            collect_stream(Ref, 5000)
        ),
        %% The model recovers: a plain request still works.
        {ok, R} = erllama:complete(Id, <<"plain text">>, #{response_tokens => 2}),
        ?assertEqual(2, length(maps:get(generated, R)))
    end).
