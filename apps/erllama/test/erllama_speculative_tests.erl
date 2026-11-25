%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Ngram speculation over the stub backend: the scheduler's spec
%% ticks (multi-token commit, target capping, fallback on draft
%% misses, pause under concurrency, cancellation) without a GGUF.
%% The stub drafts perfectly by default (its decode stream is one
%% constant token per request), `spec_draft => wrong` misses every
%% draft, `none` never drafts. The mid-batch stop-string KV trim is
%% covered by the real-model CT case (the stub's token values are
%% not predictable across requests, so a stop string cannot be
%% aimed at a mid-batch position here).
-module(erllama_speculative_tests).
-include_lib("eunit/include/eunit.hrl").

%% =============================================================================
%% Fixtures
%% =============================================================================

with_app(Body) ->
    {ok, Started} = application:ensure_all_started(erllama),
    Dir = make_tmp_dir(),
    DiskSrv = list_to_atom(
        "spec_disk_" ++ integer_to_list(erlang:unique_integer([positive]))
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
        fingerprint => binary:copy(<<16#55>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        quant_bits => 16,
        ctx_params_hash => binary:copy(<<16#66>>, 32),
        context_size => 1024,
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
            "spec_", integer_to_binary(erlang:unique_integer([positive]))
        ]),
        Cfg = maps:merge(minimal_config(DiskSrv), Overrides),
        {ok, _} = erllama:load_model(Id, Cfg),
        erllama_model_stub:reset_spec_accept_calls(),
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
        "erllama_spec_tests_" ++
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

%% =============================================================================
%% Perfect drafts
%% =============================================================================

perfect_drafts_hit_target_exactly_test() ->
    with_model(#{}, fun(Id) ->
        {ok, R} = erllama:complete(Id, <<"hello world foo">>, #{
            response_tokens => 10, speculative => true
        }),
        Gen = maps:get(generated, R),
        ?assertEqual(10, length(Gen)),
        %% The stub stream is one constant token per request.
        ?assertEqual(1, length(lists:usort(Gen))),
        %% KV == context: prompt (3 words) + generated.
        ?assertEqual(13, maps:get(committed_tokens, R)),
        #{drafted := D, accepted := A} = maps:get(speculative, R),
        ?assert(D > 0),
        ?assertEqual(D, A)
    end).

speculative_stats_absent_when_off_test() ->
    with_model(#{}, fun(Id) ->
        {ok, R} = erllama:complete(Id, <<"hello world">>, #{response_tokens => 4}),
        ?assertNot(maps:is_key(speculative, R))
    end).

streaming_events_and_stats_test() ->
    with_model(#{}, fun(Id) ->
        {ok, Ref} = erllama:stream(Id, <<"stream me now">>, #{
            response_tokens => 8, speculative => true
        }),
        {Out, Stats} = collect_stream(Ref, 5000),
        ?assertEqual(8, length(Out)),
        ?assertEqual(length, maps:get(finish_reason, Stats)),
        #{drafted := D, accepted := A} = maps:get(speculative, Stats),
        ?assert(D > 0),
        ?assertEqual(D, A)
    end).

accept_feedback_reported_test() ->
    with_model(#{spec_draft_len => 3}, fun(Id) ->
        {ok, _} = erllama:complete(Id, <<"a b c">>, #{
            response_tokens => 9, speculative => true
        }),
        Calls = erllama_model_stub:spec_accept_calls(),
        ?assert(length(Calls) >= 1),
        %% Perfect drafts: every reported acceptance equals a full
        %% (possibly target-truncated) draft, never more than the
        %% configured draft length.
        [?assert(N >= 0 andalso N =< 3) || {_Seq, N} <- Calls]
    end).

%% =============================================================================
%% Fallbacks
%% =============================================================================

wrong_drafts_fall_back_cleanly_test() ->
    with_model(#{spec_draft => wrong}, fun(Id) ->
        {ok, R} = erllama:complete(Id, <<"hello world foo">>, #{
            response_tokens => 10, speculative => true
        }),
        Gen = maps:get(generated, R),
        ?assertEqual(10, length(Gen)),
        ?assertEqual(1, length(lists:usort(Gen))),
        #{drafted := D, accepted := A} = maps:get(speculative, R),
        ?assert(D > 0),
        ?assertEqual(0, A)
    end).

no_draft_available_falls_back_test() ->
    with_model(#{spec_draft => none}, fun(Id) ->
        {ok, R} = erllama:complete(Id, <<"hello world foo">>, #{
            response_tokens => 6, speculative => true
        }),
        ?assertEqual(6, length(maps:get(generated, R))),
        %% Option accepted (stats key present) but nothing drafted.
        ?assertEqual(
            #{drafted => 0, accepted => 0, rounds => 0}, maps:get(speculative, R)
        )
    end).

logprobs_disable_speculation_test() ->
    with_model(#{}, fun(Id) ->
        {ok, R} = erllama:complete(Id, <<"hello world foo">>, #{
            response_tokens => 4, speculative => true, logprobs => 2
        }),
        ?assertEqual(4, length(maps:get(generated, R))),
        %% Gated off: no speculative key at all.
        ?assertNot(maps:is_key(speculative, R)),
        ?assertEqual(4, length(maps:get(logprobs, R)))
    end).

thinking_disables_speculation_test() ->
    with_model(#{thinking_capable => true}, fun(Id) ->
        {ok, Ref} = erllama:stream(Id, <<"think about it">>, #{
            response_tokens => 4, speculative => true, thinking => enabled
        }),
        {_Out, Stats} = collect_stream(Ref, 5000),
        ?assertNot(maps:is_key(speculative, Stats))
    end).

%% =============================================================================
%% Concurrency: speculation pauses while another request is active
%% =============================================================================

pauses_under_concurrency_test() ->
    with_model(
        #{context_opts => #{n_seq_max => 2}, step_delay_ms => 20},
        fun(Id) ->
            %% Long-running holder outlives the whole speculative
            %% request, so the speculative one is never alone and
            %% must not draft once.
            {ok, HolderRef} = erllama:stream(Id, <<"holder one">>, #{
                response_tokens => 500
            }),
            {ok, SpecRef} = erllama:stream(Id, <<"speculate me">>, #{
                response_tokens => 5, speculative => true
            }),
            {Out, Stats} = collect_stream(SpecRef, 30000),
            ?assertEqual(5, length(Out)),
            ?assertEqual(
                #{drafted => 0, accepted => 0, rounds => 0},
                maps:get(speculative, Stats)
            ),
            ok = erllama:cancel(HolderRef)
        end
    ).

resumes_when_alone_test() ->
    with_model(
        #{context_opts => #{n_seq_max => 2}, step_delay_ms => 5},
        fun(Id) ->
            %% Short holder finishes early; the speculative request
            %% resumes drafting once alone again.
            {ok, HolderRef} = erllama:stream(Id, <<"holder one">>, #{
                response_tokens => 2
            }),
            {ok, SpecRef} = erllama:stream(Id, <<"speculate me">>, #{
                response_tokens => 40, speculative => true
            }),
            {_HOut, _HStats} = collect_stream(HolderRef, 30000),
            {Out, Stats} = collect_stream(SpecRef, 30000),
            ?assertEqual(40, length(Out)),
            #{drafted := D, accepted := A} = maps:get(speculative, Stats),
            ?assert(D > 0),
            ?assertEqual(D, A)
        end
    ).

%% =============================================================================
%% Cancellation and sessions
%% =============================================================================

cancel_mid_speculation_test() ->
    with_model(#{step_delay_ms => 30}, fun(Id) ->
        {ok, Ref} = erllama:stream(Id, <<"cancel me later">>, #{
            response_tokens => 500, speculative => true
        }),
        %% Let a few ticks (incl. spec ticks) run, then cancel.
        timer:sleep(150),
        ok = erllama:cancel(Ref),
        {_Out, Stats} = collect_stream(Ref, 5000),
        ?assertEqual(true, maps:get(cancelled, Stats)),
        ?assertEqual(cancelled, maps:get(finish_reason, Stats))
    end).

session_turns_keep_speculating_test() ->
    with_model(#{}, fun(Id) ->
        {ok, R1} = erllama:complete(Id, <<"turn one text">>, #{
            response_tokens => 6, speculative => true, session_id => spec_sess
        }),
        ?assert(maps:is_key(speculative, R1)),
        Transcript = <<"turn one text ", (maps:get(reply, R1))/binary>>,
        {ok, R2} = erllama:complete(Id, <<Transcript/binary, " and more">>, #{
            response_tokens => 6, speculative => true, session_id => spec_sess
        }),
        ?assertEqual(6, length(maps:get(generated, R2))),
        #{drafted := D2, accepted := A2} = maps:get(speculative, R2),
        ?assert(D2 > 0),
        ?assertEqual(D2, A2)
    end).

%% =============================================================================
%% Recovery
%% =============================================================================

%% A speculative step that times out fails its caller with the bare
%% reason, wrapped once, and the recovered model serves the next one.
spec_step_timeout_reaches_callers_test() ->
    with_model(#{}, fun(Id) ->
        ok = erllama_model_stub:wedge_next_spec_step(decode_timeout),
        {ok, Ref} = erllama:stream(Id, <<"stream me now">>, #{
            response_tokens => 8, speculative => true
        }),
        ?assertEqual({error, decode_timeout}, collect_stream(Ref, 5000)),
        ok = erllama_model_stub:wedge_next_spec_step(decode_timeout),
        ?assertEqual(
            {error, decode_timeout},
            erllama:complete(Id, <<"hello world foo">>, #{
                response_tokens => 8, speculative => true
            })
        ),
        {ok, R} = erllama:complete(Id, <<"hello world foo">>, #{
            response_tokens => 8, speculative => true
        }),
        ?assertEqual(8, length(maps:get(generated, R)))
    end).

%% =============================================================================
%% MTP rounds for several requests
%% =============================================================================

mtp_opts(N) ->
    #{response_tokens => N, speculative => #{type => mtp, n_max => 2}}.

with_mtp_model(Body) ->
    with_model(#{context_opts => #{n_seq_max => 2}, step_delay_ms => 20}, fun(Id) ->
        ok = erllama_model_stub:reset_mtp_calls(),
        try
            Body(Id)
        after
            erllama_model_stub:reset_mtp_calls()
        end
    end).

%% The backend refuses the neighbour's sampler in a batched round; only
%% the neighbour loses drafting. The first request (seq 0) still drafts,
%% and drafts alone once the neighbour (seq 1) has finished.
mtp_refusal_not_greedy_keeps_the_other_drafting_test() ->
    mtp_refusal_keeps_the_other_drafting(not_greedy).

mtp_refusal_not_supported_keeps_the_other_drafting_test() ->
    mtp_refusal_keeps_the_other_drafting(not_supported).

mtp_refusal_keeps_the_other_drafting(Reason) ->
    with_mtp_model(fun(Id) ->
        ok = erllama_model_stub:refuse_mtp_seq(1, Reason),
        {ok, A} = erllama:stream(Id, <<"the first and longer request">>, mtp_opts(40)),
        {ok, B} = erllama:stream(Id, <<"its neighbour">>, mtp_opts(8)),
        {OutB, _StatsB} = collect_stream(B, 10000),
        {OutA, StatsA} = collect_stream(A, 10000),
        ?assertEqual(8, length(OutB)),
        ?assertEqual(40, length(OutA)),
        ?assertEqual(length, maps:get(finish_reason, StatsA)),
        Calls = erllama_model_stub:mtp_calls(),
        %% Refused once: the neighbour is not asked again.
        ?assertEqual([{refused, {Reason, 1}}], [C || {refused, _} = C <- Calls]),
        After = tl(lists:dropwhile(fun(C) -> C =/= {refused, {Reason, 1}} end, Calls)),
        ?assert(lists:any(fun(C) -> element(1, C) =:= lone andalso element(2, C) =:= 0 end, After)),
        Batches = [L || {batch, L} <- After],
        [?assertEqual(0, proplists:get_value(1, L, 0)) || L <- Batches],
        case Reason of
            not_greedy ->
                %% The neighbour still joins, without guesses, while the
                %% first request drafts in the same round.
                ?assert(lists:any(fun(L) -> proplists:get_value(0, L, 0) > 0 end, Batches));
            not_supported ->
                %% The neighbour cannot join, so no round is batched.
                ?assertEqual([], Batches)
        end,
        #{rounds := Rounds} = maps:get(speculative, StatsA),
        ?assert(Rounds > 0)
    end).

%% A batched round that times out fails both requests with the bare
%% reason, and the recovered model serves the next request.
mtp_batch_timeout_reaches_both_callers_test() ->
    with_mtp_model(fun(Id) ->
        {ok, A} = erllama:stream(Id, <<"the first request">>, mtp_opts(200)),
        {ok, B} = erllama:stream(Id, <<"the second request">>, mtp_opts(200)),
        ok = wait_for_batch(5000),
        ok = erllama_model_stub:wedge_next_spec_step(decode_timeout),
        ?assertEqual({error, decode_timeout}, collect_stream(A, 5000)),
        ?assertEqual({error, decode_timeout}, collect_stream(B, 5000)),
        {ok, R} = erllama:complete(Id, <<"after the recovery">>, mtp_opts(6)),
        ?assertEqual(6, length(maps:get(generated, R)))
    end).

wait_for_batch(Timeout) when Timeout =< 0 ->
    {timeout, erllama_model_stub:mtp_calls()};
wait_for_batch(Timeout) ->
    case [L || {batch, L} <- erllama_model_stub:mtp_calls()] of
        [_ | _] ->
            ok;
        [] ->
            timer:sleep(10),
            wait_for_batch(Timeout - 10)
    end.
