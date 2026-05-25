%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(barrel_inference_model_tests).
-include_lib("eunit/include/eunit.hrl").
-include("barrel_inference_cache.hrl").

%% =============================================================================
%% Fixtures
%% =============================================================================

with_model(PolicyOverrides, Body) ->
    with_model(PolicyOverrides, #{}, Body).

with_model(PolicyOverrides, ConfigOverrides, Body) ->
    ok = barrel_inference_cache_counters:init(),
    barrel_inference_cache_counters:reset(),
    {ok, _} = barrel_inference_registry:start_link(),
    {ok, _} = barrel_inference_inflight:start_link(),
    {ok, _} = barrel_inference_cache_meta_srv:start_link(),
    {ok, _} = barrel_inference_cache_ram:start_link(),
    {ok, _} = barrel_inference_cache_writer:start_link(2),
    Dir = make_tmp_dir(),
    {ok, _} = barrel_inference_cache_disk_srv:start_link(test_disk, Dir),
    Policy = maps:merge(default_policy(), PolicyOverrides),
    BaseConfig = #{
        tier_srv => test_disk,
        tier => disk,
        fingerprint => binary:copy(<<16#AA>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        quant_bits => 16,
        ctx_params_hash => binary:copy(<<16#BB>>, 32),
        context_size => 4096,
        policy => Policy
    },
    Config = maps:merge(BaseConfig, ConfigOverrides),
    {ok, _} = barrel_inference_model:start_link(<<"test_model">>, Config),
    try
        Body(Config)
    after
        catch barrel_inference_model:stop(<<"test_model">>),
        catch gen_server:stop(test_disk),
        catch gen_server:stop(barrel_inference_cache_writer),
        catch gen_server:stop(barrel_inference_cache_ram),
        catch gen_server:stop(barrel_inference_cache_meta_srv),
        catch gen_server:stop(barrel_inference_inflight),
        catch gen_server:stop(barrel_inference_registry),
        rm_rf(Dir)
    end.

default_policy() ->
    #{
        min_tokens => 4,
        cold_min_tokens => 4,
        cold_max_tokens => 1000,
        continued_interval => 2048,
        boundary_trim_tokens => 0,
        boundary_align_tokens => 1,
        session_resume_wait_ms => 50
    }.

short_prompt() -> <<"hi">>.

long_prompt() ->
    list_to_binary(string:join([integer_to_list(N) || N <- lists:seq(1, 12)], " ")).

key_for_tokens(Tokens, Cfg) ->
    barrel_inference_cache_key:make(#{
        fingerprint => maps:get(fingerprint, Cfg),
        quant_type => maps:get(quant_type, Cfg),
        ctx_params_hash => maps:get(ctx_params_hash, Cfg),
        tokens => Tokens
    }).

prompt_tokens(Prompt) ->
    [
        erlang:phash2(W) rem (1 bsl 32)
     || W <- binary:split(Prompt, <<" ">>, [global, trim_all]),
        W =/= <<>>
    ].

wait_for_key(Key, TimeoutMs) ->
    Deadline = erlang:monotonic_time(millisecond) + TimeoutMs,
    wait_for_key_loop(Key, Deadline).

wait_for_key_loop(Key, Deadline) ->
    case barrel_inference_cache_meta_srv:lookup_exact(Key) of
        {ok, _} = R ->
            R;
        miss ->
            case erlang:monotonic_time(millisecond) > Deadline of
                true ->
                    miss;
                false ->
                    timer:sleep(10),
                    wait_for_key_loop(Key, Deadline)
            end
    end.

%% =============================================================================
%% Lifecycle
%% =============================================================================

starts_in_idle_test() ->
    with_model(#{}, fun(_) ->
        ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>))
    end).

model_info_returns_map_test() ->
    with_model(#{}, fun(_) ->
        Info = barrel_inference_model:model_info(<<"test_model">>),
        ?assertEqual(<<"test_model">>, maps:get(id, Info)),
        ?assertEqual(idle, maps:get(status, Info)),
        ?assert(is_pid(maps:get(pid, Info))),
        ?assertEqual(barrel_inference_model_stub, maps:get(backend, Info)),
        ?assertEqual(4096, maps:get(context_size, Info)),
        ?assertEqual(f16, maps:get(quant_type, Info)),
        ?assertEqual(16, maps:get(quant_bits, Info)),
        ?assertEqual(disk, maps:get(tier, Info)),
        ?assertEqual(32, byte_size(maps:get(fingerprint, Info))),
        ?assertEqual(1, maps:get(n_seq_max, Info)),
        ?assertEqual(1, maps:get(available_seqs, Info))
    end).

model_info_via_pid_test() ->
    with_model(#{}, fun(_) ->
        Pid = barrel_inference_registry:whereis_name(<<"test_model">>),
        Info = barrel_inference_model:model_info(Pid),
        ?assertEqual(<<"test_model">>, maps:get(id, Info))
    end).

tokenize_returns_list_test() ->
    with_model(#{}, fun(_) ->
        {ok, Tokens} = barrel_inference_model:tokenize(<<"test_model">>, <<"hello world">>),
        ?assert(is_list(Tokens)),
        ?assert(lists:all(fun is_integer/1, Tokens))
    end).

tokenize_empty_string_test() ->
    with_model(#{}, fun(_) ->
        {ok, Tokens} = barrel_inference_model:tokenize(<<"test_model">>, <<>>),
        ?assertEqual([], Tokens)
    end).

detokenize_roundtrip_test() ->
    %% The stub backend is not roundtrippable (phash2-based), but the
    %% types should line up: tokenize -> [int], detokenize -> binary.
    with_model(#{}, fun(_) ->
        {ok, Tokens} = barrel_inference_model:tokenize(<<"test_model">>, <<"hi there">>),
        {ok, Bin} = barrel_inference_model:detokenize(<<"test_model">>, Tokens),
        ?assert(is_binary(Bin))
    end).

tokenize_concurrent_with_idle_test() ->
    with_model(#{}, fun(_) ->
        ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>)),
        {ok, _} = barrel_inference_model:tokenize(<<"test_model">>, <<"x">>),
        ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>))
    end).

via_unknown_model_crashes_test() ->
    with_model(#{}, fun(_) ->
        ?assertExit(
            {noproc, {barrel_inference_model, not_found, <<"unknown">>}},
            barrel_inference_model:status(<<"unknown">>)
        )
    end).

complete_returns_response_test() ->
    with_model(#{}, fun(_) ->
        {ok, Result} = barrel_inference_model:complete(<<"test_model">>, short_prompt()),
        ?assert(is_map(Result)),
        ?assert(is_binary(maps:get(reply, Result))),
        Generated = maps:get(generated, Result),
        ?assert(length(Generated) > 0),
        ?assertEqual(length(Generated), maps:get(completion_tokens, maps:get(stats, Result))),
        ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>))
    end).

%% =============================================================================
%% Cold save fires for prompts in [cold_min, cold_max]
%% =============================================================================

short_prompt_does_not_cold_save_test() ->
    with_model(
        #{min_tokens => 4, cold_min_tokens => 4, cold_max_tokens => 1000},
        fun(Cfg) ->
            {ok, _} = barrel_inference_model:complete(<<"test_model">>, short_prompt()),
            %% Short prompt has 1 token; min is 4 -> no cold save.
            Tokens = prompt_tokens(short_prompt()),
            ColdKey = key_for_tokens(Tokens, Cfg),
            ?assertEqual(miss, barrel_inference_cache_meta_srv:lookup_exact(ColdKey))
        end
    ).

long_prompt_fires_cold_save_test() ->
    with_model(#{}, fun(Cfg) ->
        {ok, _} = barrel_inference_model:complete(<<"test_model">>, long_prompt()),
        Tokens = prompt_tokens(long_prompt()),
        ColdKey = key_for_tokens(Tokens, Cfg),
        ?assertMatch({ok, _Row}, wait_for_key(ColdKey, 1000))
    end).

%% =============================================================================
%% Finish save fires at end-of-stream when total is above min
%% =============================================================================

finish_save_fires_for_long_prompt_test() ->
    with_model(#{}, fun(Cfg) ->
        {ok, #{generated := Generated, finish_key := ReportedKey}} =
            barrel_inference_model:complete(<<"test_model">>, long_prompt(), #{response_tokens => 6}),
        FullTokens = prompt_tokens(long_prompt()) ++ Generated,
        FinishKey = key_for_tokens(FullTokens, Cfg),
        ?assertEqual(FinishKey, ReportedKey),
        ?assertMatch({ok, _Row}, wait_for_key(FinishKey, 1000))
    end).

%% =============================================================================
%% Cache hit on repeat
%% =============================================================================

repeat_prompt_hits_finish_save_path_test() ->
    with_model(#{}, fun(Cfg) ->
        {ok, #{generated := Gen1}} =
            barrel_inference_model:complete(<<"test_model">>, long_prompt(), #{response_tokens => 4}),
        FullKey1 = key_for_tokens(prompt_tokens(long_prompt()) ++ Gen1, Cfg),
        {ok, _} = wait_for_key(FullKey1, 1000),
        %% Second complete with the *same prompt + response continuation*:
        %% the cache row keyed on prompt-only doesn't yet exist (the
        %% first run only persisted the cold/finish keys), but a third
        %% turn that uses parent_key=FullKey1 will hit session resume.
        ?assertMatch({ok, _Row}, barrel_inference_cache_meta_srv:lookup_exact(FullKey1))
    end).

%% =============================================================================
%% parent_key (session resume)
%% =============================================================================

parent_key_session_resume_test() ->
    with_model(#{}, fun(Cfg) ->
        %% Turn 1: prompt + response.
        {ok, #{generated := Gen1, finish_key := ReportedKey1}} =
            barrel_inference_model:complete(<<"test_model">>, long_prompt(), #{response_tokens => 4}),
        FullTokens1 = prompt_tokens(long_prompt()) ++ Gen1,
        FullKey1 = key_for_tokens(FullTokens1, Cfg),
        ?assertEqual(FullKey1, ReportedKey1),
        {ok, _} = wait_for_key(FullKey1, 1000),
        %% Turn 2: a longer prompt that strictly prefix-extends turn 1's
        %% live tokens. Use parent_key = FullKey1 to take the session
        %% resume path.
        Extension = list_to_binary(
            stub_detokenize_decimal(FullTokens1) ++ " more tokens for turn two"
        ),
        {ok, _} =
            barrel_inference_model:complete(<<"test_model">>, Extension, #{
                parent_key => FullKey1,
                response_tokens => 2
            }),
        ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>))
    end).

%% =============================================================================
%% Continued saves during generation
%% =============================================================================

continued_save_fires_during_long_generation_test() ->
    %% Lower continued_interval so the fixture's prompt + response
    %% crosses the boundary at least once.
    with_model(
        #{continued_interval => 4, response_target => 8},
        fun(Cfg) ->
            PromptTokens = prompt_tokens(long_prompt()),
            {ok, #{generated := Generated}} = barrel_inference_model:complete(
                <<"test_model">>, long_prompt(), #{response_tokens => 8}
            ),
            %% A continued save fires when LiveTokens - LastSavedAt
            %% reaches continued_interval. With the cold save firing
            %% at 12 prompt tokens, last_save_at = 12. After 4 more
            %% generated tokens, a continued save fires for the
            %% first 16 tokens. Those tokens are
            %% PromptTokens ++ first 4 generated.
            FirstContinuedTokens =
                PromptTokens ++ lists:sublist(Generated, 4),
            FirstContinuedKey = key_for_tokens(FirstContinuedTokens, Cfg),
            ?assertMatch({ok, _Row}, wait_for_key(FirstContinuedKey, 1000))
        end
    ).

%% =============================================================================
%% Evict save
%% =============================================================================

evict_idle_with_no_context_is_noop_test() ->
    with_model(#{}, fun(_Cfg) ->
        ok = barrel_inference_model:evict(<<"test_model">>),
        ?assertEqual(0, length(barrel_inference_cache_meta_srv:dump())),
        ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>))
    end).

evict_during_generation_persists_live_state_test() ->
    %% Fire evict in the middle of a long generation. The model
    %% intercepts the call between decode_step events and writes
    %% an evict save with whatever tokens are live.
    with_model(#{response_target => 200, continued_interval => 100000}, fun(_Cfg) ->
        Parent = self(),
        spawn(fun() ->
            Parent !
                {done,
                    barrel_inference_model:complete(
                        <<"test_model">>, long_prompt(), #{response_tokens => 200}
                    )}
        end),
        %% The reply binding (above) is now a map; downstream just
        %% receives the whole {ok, Map} tuple as `_` since this test
        %% only cares about save side-effects.
        %% Give the gen_statem a beat to enter generating.
        timer:sleep(0),
        ok = barrel_inference_model:evict(<<"test_model">>),
        %% Wait for the request to complete (the finish save fires too,
        %% so we expect at least an evict row plus cold + finish).
        receive
            {done, _} -> ok
        after 5000 -> erlang:error(timeout)
        end,
        timer:sleep(50),
        ?assert(length(barrel_inference_cache_meta_srv:dump()) >= 2)
    end).

%% =============================================================================
%% Shutdown save
%% =============================================================================

shutdown_idle_with_no_context_is_noop_test() ->
    with_model(#{}, fun(_Cfg) ->
        ok = barrel_inference_model:shutdown(<<"test_model">>),
        ?assertEqual(0, length(barrel_inference_cache_meta_srv:dump())),
        ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>))
    end).

%% =============================================================================
%% Counters move under traffic
%% =============================================================================

counters_track_misses_and_saves_test() ->
    with_model(#{}, fun(_Cfg) ->
        Before = barrel_inference_cache:get_counters(),
        {ok, _} =
            barrel_inference_model:complete(<<"test_model">>, long_prompt(), #{response_tokens => 4}),
        timer:sleep(50),
        After = barrel_inference_cache:get_counters(),
        %% A fresh prompt is a miss, fires cold + finish saves.
        ?assert(maps:get(misses, After) > maps:get(misses, Before)),
        ?assert(maps:get(saves_cold, After) > maps:get(saves_cold, Before)),
        ?assert(maps:get(saves_finish, After) > maps:get(saves_finish, Before))
    end).

%% =============================================================================
%% Concurrency
%% =============================================================================

concurrent_complete_rejects_with_busy_test() ->
    with_model(#{}, fun(_Cfg) ->
        Parent = self(),
        %% Spawn a slow caller that keeps the gen_statem busy.
        spawn(fun() ->
            Parent ! {first, barrel_inference_model:complete(<<"test_model">>, long_prompt())}
        end),
        timer:sleep(0),
        %% Try a second complete; the gen_statem is in prefilling/
        %% generating and rejects.
        Result = barrel_inference_model:complete(<<"test_model">>, short_prompt()),
        receive
            {first, _} -> ok
        after 5000 -> erlang:error(first_caller_timeout)
        end,
        case Result of
            %% raced and got there first/after
            {ok, _} -> ok;
            %% busy as expected
            {error, busy} -> ok
        end
    end).

%% =============================================================================
%% PR1: completion_result map shape
%% =============================================================================

complete_returns_finish_key_matching_full_tokens_test() ->
    with_model(#{}, fun(Cfg) ->
        {ok, #{
            generated := Generated,
            context_tokens := ContextTokens,
            finish_key := FinishKey,
            cache_hit_kind := HitKind
        }} =
            barrel_inference_model:complete(<<"test_model">>, long_prompt(), #{response_tokens => 4}),
        ExpectedTokens = prompt_tokens(long_prompt()) ++ Generated,
        ?assertEqual(ExpectedTokens, ContextTokens),
        ?assertEqual(key_for_tokens(ContextTokens, Cfg), FinishKey),
        ?assertEqual(cold, HitKind)
    end).

complete_finish_key_undefined_when_below_min_tokens_test() ->
    %% min_tokens default 4; the short prompt produces 1 + a small
    %% generation. Force a higher threshold so the finish save is
    %% suppressed and finish_key comes back as undefined.
    with_model(#{min_tokens => 10_000}, fun(_Cfg) ->
        {ok, #{finish_key := FinishKey}} =
            barrel_inference_model:complete(<<"test_model">>, short_prompt(), #{
                response_tokens => 2
            }),
        ?assertEqual(undefined, FinishKey)
    end).

complete_committed_tokens_equals_context_tokens_length_test() ->
    with_model(#{}, fun(_Cfg) ->
        {ok, #{context_tokens := Ctx, committed_tokens := N}} =
            barrel_inference_model:complete(<<"test_model">>, long_prompt(), #{response_tokens => 4}),
        ?assertEqual(length(Ctx), N)
    end).

complete_stop_sequence_trims_reply_and_reports_match_test() ->
    %% Stub backend's detokenize produces space-separated decimals.
    %% Stopping on any digit guarantees an early match.
    AllDigits = [
        <<"0">>,
        <<"1">>,
        <<"2">>,
        <<"3">>,
        <<"4">>,
        <<"5">>,
        <<"6">>,
        <<"7">>,
        <<"8">>,
        <<"9">>
    ],
    with_model(#{}, fun(_Cfg) ->
        {ok, Result} = barrel_inference_model:complete(
            <<"test_model">>,
            long_prompt(),
            #{response_tokens => 8, stop_sequences => AllDigits}
        ),
        Match = maps:get(stop_sequence, Result),
        ?assert(is_binary(Match)),
        ?assert(lists:member(Match, AllDigits)),
        ?assertEqual(stop, maps:get(finish_reason, Result)),
        ?assertEqual(stop, maps:get(finish_reason, maps:get(stats, Result))),
        ?assertEqual(Match, maps:get(stop_sequence, maps:get(stats, Result))),
        %% reply is trimmed at the first occurrence of the matched stop.
        Reply = maps:get(reply, Result),
        ?assertEqual(nomatch, binary:match(Reply, Match))
    end).

complete_stop_sequence_absent_when_no_match_test() ->
    with_model(#{}, fun(_Cfg) ->
        {ok, Result} = barrel_inference_model:complete(
            <<"test_model">>,
            long_prompt(),
            #{response_tokens => 1, stop_sequences => [<<"unmatchable-xyz">>]}
        ),
        ?assertNot(maps:is_key(stop_sequence, Result)),
        ?assertNot(maps:is_key(stop_sequence, maps:get(stats, Result)))
    end).

%% =============================================================================
%% cache_delta accounting (Anthropic-style)
%% =============================================================================

complete_cache_delta_cold_test() ->
    with_model(#{}, fun(_Cfg) ->
        {ok, Result} = barrel_inference_model:complete(
            <<"test_model">>, long_prompt(), #{response_tokens => 4}
        ),
        Delta = maps:get(cache_delta, Result),
        ?assertEqual(0, maps:get(read, Delta)),
        Committed = maps:get(committed_tokens, Result),
        ?assertEqual(Committed, maps:get(created, Delta)),
        %% Stats carry the same delta for streaming callers.
        ?assertEqual(Delta, maps:get(cache_delta, maps:get(stats, Result)))
    end).

complete_cache_delta_exact_warm_test() ->
    with_model(#{}, fun(Cfg) ->
        Tokens = prompt_tokens(long_prompt()),
        %% First call: cold; cold save fires on the prompt prefix.
        {ok, _R1} = barrel_inference_model:complete(
            <<"test_model">>, long_prompt(), #{response_tokens => 4}
        ),
        ColdKey = key_for_tokens(Tokens, Cfg),
        {ok, _} = wait_for_key(ColdKey, 1000),
        %% Second call with the same prompt: exact hit on the cold-save key.
        {ok, R2} = barrel_inference_model:complete(
            <<"test_model">>, long_prompt(), #{response_tokens => 4}
        ),
        ?assertEqual(exact, maps:get(cache_hit_kind, R2)),
        Delta = maps:get(cache_delta, R2),
        ?assertEqual(length(Tokens), maps:get(read, Delta)),
        Generated = length(maps:get(generated, R2)),
        ?assertEqual(Generated, maps:get(created, Delta))
    end).

complete_cache_delta_finish_suppressed_test() ->
    with_model(#{min_tokens => 10_000}, fun(_Cfg) ->
        {ok, Result} = barrel_inference_model:complete(
            <<"test_model">>, long_prompt(), #{response_tokens => 2}
        ),
        Delta = maps:get(cache_delta, Result),
        ?assertEqual(0, maps:get(read, Delta)),
        ?assertEqual(0, maps:get(created, Delta))
    end).

prefill_only_cache_delta_test() ->
    with_model(#{}, fun(_Cfg) ->
        Tokens = prompt_tokens(long_prompt()),
        {ok, Result} = barrel_inference_model:prefill_only(<<"test_model">>, Tokens),
        Delta = maps:get(cache_delta, Result),
        ?assertEqual(0, maps:get(read, Delta)),
        ?assertEqual(length(Tokens), maps:get(created, Delta))
    end).

prefill_only_returns_finish_key_and_warm_resumes_test() ->
    with_model(#{}, fun(Cfg) ->
        Tokens = prompt_tokens(long_prompt()),
        {ok, #{
            context_tokens := Ctx,
            committed_tokens := N,
            finish_key := FinishKey,
            cache_hit_kind := HitKind
        }} = barrel_inference_model:prefill_only(<<"test_model">>, Tokens),
        ?assertEqual(Tokens, Ctx),
        ?assertEqual(length(Tokens), N),
        ?assertEqual(cold, HitKind),
        ?assertEqual(key_for_tokens(Tokens, Cfg), FinishKey),
        ?assertMatch({ok, _Row}, wait_for_key(FinishKey, 1000)),
        %% Now resume from FinishKey; the cache should report exact hit.
        {ok, #{cache_hit_kind := exact}} =
            barrel_inference_model:complete(
                <<"test_model">>, long_prompt(), #{
                    parent_key => FinishKey,
                    response_tokens => 2
                }
            )
    end).

prefill_only_with_parent_key_chains_warm_contexts_test() ->
    %% Warm a prefix via prefill_only/2, then extend it via
    %% prefill_only/3 with parent_key. The second call must take the
    %% exact warm path (cache_hit_kind = exact), prefill only the
    %% suffix tokens, and surface a fresh finish_key for the new
    %% prefix-plus-suffix row.
    with_model(#{}, fun(Cfg) ->
        Prefix = prompt_tokens(long_prompt()),
        Suffix = prompt_tokens(short_prompt()),
        Extended = Prefix ++ Suffix,
        {ok, #{finish_key := PrefixKey}} =
            barrel_inference_model:prefill_only(<<"test_model">>, Prefix),
        ?assertEqual(key_for_tokens(Prefix, Cfg), PrefixKey),
        {ok, _} = wait_for_key(PrefixKey, 1000),
        {ok, #{
            cache_hit_kind := Kind,
            context_tokens := ExtCtx,
            committed_tokens := ExtN,
            finish_key := ExtendedKey,
            cache_delta := #{read := Read, created := Created}
        }} = barrel_inference_model:prefill_only(
            <<"test_model">>, Extended, #{parent_key => PrefixKey}
        ),
        %% Session resume from parent_key is reported as `partial`
        %% (only a prefix of the prompt was in cache); only an
        %% identical prompt produces `exact`.
        ?assertEqual(partial, Kind),
        ?assertEqual(Extended, ExtCtx),
        ?assertEqual(length(Extended), ExtN),
        ?assertEqual(key_for_tokens(Extended, Cfg), ExtendedKey),
        %% Read = prefix length restored from cache; Created = the
        %% suffix tokens added by this call.
        ?assertEqual(length(Prefix), Read),
        ?assertEqual(length(Suffix), Created)
    end).

%% =============================================================================
%% sticky-seq (session_id) admission
%% =============================================================================

sticky_seq_continues_on_same_session_test() ->
    %% Two requests with the same session_id. The second sees
    %% cache_hit_kind = sticky: the stored prefix is taken from
    %% live KV (no warm-restore) and only the new suffix runs
    %% through the prefill pipeline.
    SessionId = make_ref(),
    with_model(#{}, fun(_Cfg) ->
        {ok, #{generated := Gen1}} = barrel_inference_model:complete(
            <<"test_model">>,
            long_prompt(),
            #{response_tokens => 3, session_id => SessionId}
        ),
        Turn1Tokens = prompt_tokens(long_prompt()) ++ Gen1,
        NewSuffix = prompt_tokens(<<"continue please">>),
        FullTokens = Turn1Tokens ++ NewSuffix,
        {ok, Ref} = barrel_inference_model:infer(
            <<"test_model">>,
            FullTokens,
            #{response_tokens => 2, session_id => SessionId},
            self()
        ),
        Stats = drain_done(Ref, 5000),
        ?assertEqual(sticky, maps:get(cache_hit_kind, Stats)),
        Delta = maps:get(cache_delta, Stats),
        ?assertEqual(length(Turn1Tokens), maps:get(read, Delta)),
        ?assert(maps:get(created, Delta) >= length(NewSuffix)),
        ?assertEqual(ok, barrel_inference:end_session(<<"test_model">>, SessionId)),
        %% Idempotent: a second end_session on the same id is a no-op.
        ?assertEqual(ok, barrel_inference:end_session(<<"test_model">>, SessionId))
    end).

sticky_seq_does_not_affect_non_session_callers_test() ->
    %% Sessions only kick in when session_id is set; callers that
    %% omit it see the existing one-shot path bit-identically.
    with_model(#{}, fun(_Cfg) ->
        {ok, #{cache_hit_kind := HitKind1}} = barrel_inference_model:complete(
            <<"test_model">>, long_prompt(), #{response_tokens => 2}
        ),
        ?assert(HitKind1 =:= cold orelse HitKind1 =:= exact)
    end).

end_session_unknown_is_noop_test() ->
    with_model(#{}, fun(_Cfg) ->
        ?assertEqual(ok, barrel_inference:end_session(<<"test_model">>, make_ref()))
    end).

%% available_seqs reflects the live free-list head. With n_seq_max=1
%% (the default in with_model/2), a single in-flight infer drives it
%% to 0 and a cancel restores it to 1.
model_info_available_seqs_decrements_with_inflight_test_() ->
    {timeout, 30, fun() ->
        with_model(#{}, fun(_Cfg) ->
            Info0 = barrel_inference_model:model_info(<<"test_model">>),
            ?assertEqual(1, maps:get(available_seqs, Info0)),
            {ok, Ref} = barrel_inference_model:infer(
                <<"test_model">>,
                prompt_tokens(long_prompt()),
                #{response_tokens => 10000},
                self()
            ),
            %% Wait for first token so we know the seq has been taken
            %% off the idle list and is in req_table.
            receive
                {barrel_inference_token, Ref, _} -> ok;
                {barrel_inference_token_id, Ref, _} -> ok
            after 2000 -> erlang:error(no_first_token)
            end,
            Info1 = barrel_inference_model:model_info(<<"test_model">>),
            ?assertEqual(0, maps:get(available_seqs, Info1)),
            ok = barrel_inference_model:cancel(Ref),
            receive
                {barrel_inference_done, Ref, _} -> ok
            after 5000 -> erlang:error(timeout_drain)
            end,
            Info2 = barrel_inference_model:model_info(<<"test_model">>),
            ?assertEqual(1, maps:get(available_seqs, Info2))
        end)
    end}.

reset_session_unknown_returns_not_found_test() ->
    with_model(#{}, fun(_Cfg) ->
        ?assertEqual(
            {ok, not_found},
            barrel_inference:reset_session(<<"test_model">>, make_ref())
        )
    end).

reset_session_idle_session_returns_recovered_test() ->
    %% Pin a session with a completed turn so session_seq holds the
    %% mapping but req_table is empty. reset_session must reclaim
    %% the seq cleanly and the next session lookup must miss.
    SessionId = make_ref(),
    with_model(#{}, fun(_Cfg) ->
        {ok, _} = barrel_inference_model:complete(
            <<"test_model">>,
            long_prompt(),
            #{response_tokens => 2, session_id => SessionId}
        ),
        ?assertEqual(
            {ok, recovered},
            barrel_inference:reset_session(<<"test_model">>, SessionId)
        ),
        %% Second call: session has been dropped, so not_found.
        ?assertEqual(
            {ok, not_found},
            barrel_inference:reset_session(<<"test_model">>, SessionId)
        )
    end).

reset_session_in_flight_signals_caller_and_recovers_test() ->
    %% Mirror the wedge: a streaming infer is in req_table when the
    %% recovery call arrives. The caller must be signalled with
    %% {barrel_inference_error, Ref, engine_reset}, the seq must return to
    %% idle, and a fresh infer on the same session_id must admit.
    SessionId = make_ref(),
    with_model(#{}, fun(_Cfg) ->
        {ok, #{generated := Gen1}} = barrel_inference_model:complete(
            <<"test_model">>,
            long_prompt(),
            #{response_tokens => 3, session_id => SessionId}
        ),
        Turn1Tokens = prompt_tokens(long_prompt()) ++ Gen1,
        NewSuffix = prompt_tokens(<<"continue please">>),
        Turn2Tokens = Turn1Tokens ++ NewSuffix,
        {ok, Ref1} = barrel_inference_model:infer(
            <<"test_model">>,
            Turn2Tokens,
            #{response_tokens => 10000, session_id => SessionId},
            self()
        ),
        %% Wait for the first token so we know the seq is in req_table.
        receive
            {barrel_inference_token, Ref1, _} -> ok;
            {barrel_inference_token_id, Ref1, _} -> ok
        after 2000 -> erlang:error(no_first_token)
        end,
        ?assertEqual(
            {ok, recovered},
            barrel_inference:reset_session(<<"test_model">>, SessionId)
        ),
        receive
            {barrel_inference_error, Ref1, engine_reset} -> ok
        after 2000 -> erlang:error(no_engine_reset)
        end,
        %% Drain any tokens emitted between the reset call and the
        %% reset taking effect; they're harmless.
        drain_stream_messages(Ref1),
        %% Seq returned to idle pool: a fresh infer on the same
        %% session_id admits without sticky_busy and without queueing.
        {ok, Ref2} = barrel_inference_model:infer(
            <<"test_model">>,
            Turn2Tokens,
            #{response_tokens => 2, session_id => SessionId},
            self()
        ),
        _Stats = drain_done(Ref2, 5000),
        ?assertEqual(ok, barrel_inference:end_session(<<"test_model">>, SessionId))
    end).

drain_stream_messages(Ref) ->
    receive
        {barrel_inference_token, Ref, _} -> drain_stream_messages(Ref);
        {barrel_inference_token_id, Ref, _} -> drain_stream_messages(Ref);
        {barrel_inference_thinking_end, Ref, _} -> drain_stream_messages(Ref);
        {barrel_inference_tool_call_end, Ref, _} -> drain_stream_messages(Ref);
        {barrel_inference_done, Ref, _} -> drain_stream_messages(Ref);
        {barrel_inference_error, Ref, _} -> drain_stream_messages(Ref)
    after 50 -> ok
    end.

%% Drain a stream collecting the {barrel_inference_token_id, _, _} ids in
%% order; returns {Stats, CollectedIds} at the done message.
drain_done_collecting_ids(Ref, TimeoutMs) ->
    drain_done_collecting_ids(Ref, TimeoutMs, []).

drain_done_collecting_ids(Ref, TimeoutMs, Acc) ->
    receive
        {barrel_inference_done, Ref, Stats} ->
            {Stats, lists:reverse(Acc)};
        {barrel_inference_token_id, Ref, Id} ->
            drain_done_collecting_ids(Ref, TimeoutMs, [Id | Acc]);
        {barrel_inference_token, Ref, _} ->
            drain_done_collecting_ids(Ref, TimeoutMs, Acc);
        {barrel_inference_thinking_end, Ref, _} ->
            drain_done_collecting_ids(Ref, TimeoutMs, Acc);
        {barrel_inference_tool_call_end, Ref, _} ->
            drain_done_collecting_ids(Ref, TimeoutMs, Acc)
    after TimeoutMs ->
        erlang:error({timeout, drain_done_collecting_ids})
    end.

%% #3: the barrel_inference_done Stats map carries the exact generated token
%% ids, in order, matching the {barrel_inference_token_id, _, _} stream.
infer_stats_carries_generated_token_ids_test() ->
    with_model(#{}, fun(_Cfg) ->
        {ok, Ref} = barrel_inference_model:infer(
            <<"test_model">>,
            prompt_tokens(long_prompt()),
            #{response_tokens => 4},
            self()
        ),
        {Stats, StreamedIds} = drain_done_collecting_ids(Ref, 5000),
        Generated = maps:get(generated, Stats),
        ?assert(is_list(Generated)),
        ?assertEqual(length(Generated), maps:get(completion_tokens, Stats)),
        ?assertEqual(StreamedIds, Generated)
    end).

%% #5: on_full => error fails fast with {error, seq_capacity} when no
%% seq is free, instead of blocking. Default still queues.
admit_on_full_error_returns_seq_capacity_test_() ->
    {timeout, 30, fun() ->
        with_model(#{}, fun(_Cfg) ->
            %% Take the only seq (n_seq_max defaults to 1) with a
            %% long-running streaming infer.
            {ok, Ref1} = barrel_inference_model:infer(
                <<"test_model">>,
                prompt_tokens(long_prompt()),
                #{response_tokens => 10000},
                self()
            ),
            receive
                {barrel_inference_token, Ref1, _} -> ok;
                {barrel_inference_token_id, Ref1, _} -> ok
            after 2000 -> erlang:error(no_first_token)
            end,
            %% Second admit with on_full => error must fail fast.
            ?assertEqual(
                {error, seq_capacity},
                barrel_inference_model:complete(
                    <<"test_model">>,
                    long_prompt(),
                    #{response_tokens => 2, on_full => error}
                )
            ),
            ok = barrel_inference_model:cancel(Ref1),
            receive
                {barrel_inference_done, Ref1, _} -> ok
            after 5000 -> erlang:error(timeout_drain)
            end,
            drain_stream_messages(Ref1)
        end)
    end}.

%% #5: default on_full (block) still queues — the second admit
%% completes once the first frees its seq.
admit_default_blocks_and_queues_test_() ->
    {timeout, 30, fun() ->
        with_model(#{}, fun(_Cfg) ->
            {ok, Ref1} = barrel_inference_model:infer(
                <<"test_model">>,
                prompt_tokens(long_prompt()),
                #{response_tokens => 10000},
                self()
            ),
            receive
                {barrel_inference_token, Ref1, _} -> ok;
                {barrel_inference_token_id, Ref1, _} -> ok
            after 2000 -> erlang:error(no_first_token)
            end,
            %% Queue a second infer (default block) from a helper proc
            %% so we don't deadlock the test process on gen_statem:call.
            Parent = self(),
            _ = spawn(fun() ->
                R = barrel_inference_model:infer(
                    <<"test_model">>,
                    prompt_tokens(long_prompt()),
                    #{response_tokens => 2},
                    Parent
                ),
                Parent ! {second_admit, R}
            end),
            %% Free the first seq so the queued admit dispatches.
            ok = barrel_inference_model:cancel(Ref1),
            receive
                {second_admit, {ok, Ref2}} ->
                    _ = drain_stream_messages(Ref2),
                    ok;
                {second_admit, Other} ->
                    erlang:error({unexpected_admit, Other})
            after 5000 -> erlang:error(queued_admit_never_dispatched)
            end,
            drain_stream_messages(Ref1)
        end)
    end}.

%% =============================================================================
%% #1: bounded/interruptible decode + in-place recovery
%% =============================================================================

%% A wedged/aborted decode (forced via the stub) recovers the context
%% in place: the in-flight caller is failed, the gen_statem stays
%% alive (no supervisor restart), and a fresh infer admits and runs.
decode_timeout_recovers_in_place_test_() ->
    {timeout, 30, fun() ->
        with_model(#{}, fun(_Cfg) ->
            Pid = barrel_inference_registry:whereis_name(<<"test_model">>),
            ok = barrel_inference_model_stub:wedge_next_step(decode_timeout),
            {ok, Ref1} = barrel_inference_model:infer(
                <<"test_model">>,
                prompt_tokens(long_prompt()),
                #{response_tokens => 4},
                self()
            ),
            receive
                {barrel_inference_error, Ref1, _Reason} -> ok
            after 5000 -> erlang:error(no_error_on_wedge)
            end,
            %% Same gen_statem pid: recovered in place, not restarted.
            ?assertEqual(Pid, barrel_inference_registry:whereis_name(<<"test_model">>)),
            ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>)),
            %% Fresh infer works (wedge already consumed by the stub).
            {ok, Ref2} = barrel_inference_model:infer(
                <<"test_model">>,
                prompt_tokens(long_prompt()),
                #{response_tokens => 2},
                self()
            ),
            _ = drain_done(Ref2, 5000),
            ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>))
        end)
    end}.

%% A hard llama_decode failure (e.g. "no KV slot" from an over-long
%% prompt) recovers in place like a timeout, rather than crash-looping
%% the model.
decode_failed_recovers_in_place_test_() ->
    {timeout, 30, fun() ->
        with_model(#{}, fun(_Cfg) ->
            Pid = barrel_inference_registry:whereis_name(<<"test_model">>),
            ok = barrel_inference_model_stub:wedge_next_step({decode_failed, 1}),
            {ok, Ref1} = barrel_inference_model:infer(
                <<"test_model">>,
                prompt_tokens(long_prompt()),
                #{response_tokens => 4},
                self()
            ),
            receive
                {barrel_inference_error, Ref1, _Reason} -> ok
            after 5000 -> erlang:error(no_error_on_decode_failed)
            end,
            %% Same pid: recovered in place, not restarted.
            ?assertEqual(Pid, barrel_inference_registry:whereis_name(<<"test_model">>)),
            ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>)),
            {ok, Ref2} = barrel_inference_model:infer(
                <<"test_model">>,
                prompt_tokens(long_prompt()),
                #{response_tokens => 2},
                self()
            ),
            _ = drain_done(Ref2, 5000),
            ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>))
        end)
    end}.

%% A synchronous complete/3 caller in flight when the decode fails must
%% get an {error,_} reply. Recovery keeps the gen_statem alive, so without
%% the notify_failure/2 sync-reply fix the call would hang to its timeout.
decode_failed_sync_caller_replies_test_() ->
    {timeout, 30, fun() ->
        with_model(#{}, fun(_Cfg) ->
            ok = barrel_inference_model_stub:wedge_next_step({decode_failed, 1}),
            ?assertMatch(
                {error, _},
                barrel_inference_model:complete(
                    <<"test_model">>, long_prompt(), #{response_tokens => 4}
                )
            ),
            ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>))
        end)
    end}.

%% Recovery clears sticky sessions: after a wedge the seq pool is fully
%% available again and the old session is gone.
recover_in_place_clears_sessions_test_() ->
    {timeout, 30, fun() ->
        SessionId = make_ref(),
        with_model(#{}, fun(_Cfg) ->
            {ok, _} = barrel_inference_model:complete(
                <<"test_model">>,
                long_prompt(),
                #{response_tokens => 2, session_id => SessionId}
            ),
            Info0 = barrel_inference_model:model_info(<<"test_model">>),
            ?assertEqual(0, maps:get(available_seqs, Info0)),
            ok = barrel_inference_model_stub:wedge_next_step(decode_aborted),
            {ok, Ref} = barrel_inference_model:infer(
                <<"test_model">>,
                prompt_tokens(long_prompt()) ++ prompt_tokens(<<"more">>),
                #{response_tokens => 2, session_id => SessionId},
                self()
            ),
            receive
                {barrel_inference_error, Ref, _} -> ok
            after 5000 -> erlang:error(no_error_on_wedge)
            end,
            Info1 = barrel_inference_model:model_info(<<"test_model">>),
            ?assertEqual(
                maps:get(n_seq_max, Info1), maps:get(available_seqs, Info1)
            ),
            ?assertEqual(
                {ok, not_found},
                barrel_inference:reset_session(<<"test_model">>, SessionId)
            )
        end)
    end}.

%% cancel/1 on an in-flight request lands within a bounded time and the
%% caller receives a terminal message.
cancel_bounded_test_() ->
    {timeout, 30, fun() ->
        with_model(#{}, fun(_Cfg) ->
            {ok, Ref} = barrel_inference_model:infer(
                <<"test_model">>,
                prompt_tokens(long_prompt()),
                #{response_tokens => 10000},
                self()
            ),
            receive
                {barrel_inference_token, Ref, _} -> ok;
                {barrel_inference_token_id, Ref, _} -> ok
            after 2000 -> erlang:error(no_first_token)
            end,
            ok = barrel_inference_model:cancel(Ref),
            receive
                {barrel_inference_done, Ref, _} -> ok;
                {barrel_inference_error, Ref, _} -> ok
            after 5000 -> erlang:error(cancel_did_not_land)
            end,
            drain_stream_messages(Ref)
        end)
    end}.

%% =============================================================================
%% continue/3 — caller-asserted continuation on a pinned sticky session
%% =============================================================================

continue_extends_session_without_prefix_check_test() ->
    %% Turn 1 pins a session. Turn 2 calls continue/3 with a Suffix
    %% that has NO byte-for-byte relationship with what the chat
    %% template would have rendered — exactly the case where the
    %% sticky path in infer/4 would have evicted the seq. continue/3
    %% must accept the tail anyway, prefill it on the live KV, and
    %% report cache_hit_kind = continuation with the correct delta.
    SessionId = make_ref(),
    with_model(#{}, fun(_Cfg) ->
        {ok, #{generated := Gen1}} = barrel_inference_model:complete(
            <<"test_model">>,
            long_prompt(),
            #{response_tokens => 3, session_id => SessionId}
        ),
        Turn1Tokens = prompt_tokens(long_prompt()) ++ Gen1,
        %% Caller-asserted tail: deliberately not derivable from any
        %% re-rendering of the prior prompt + reply.
        Suffix = [12345, 67890, 11111],
        {ok, Ref} = barrel_inference:continue(
            <<"test_model">>,
            Suffix,
            #{
                session_id => SessionId,
                caller_pid => self(),
                response_tokens => 2
            }
        ),
        Stats = drain_done(Ref, 5000),
        ?assertEqual(continuation, maps:get(cache_hit_kind, Stats)),
        Delta = maps:get(cache_delta, Stats),
        ?assertEqual(length(Turn1Tokens), maps:get(read, Delta)),
        ?assert(maps:get(created, Delta) >= length(Suffix)),
        %% prompt_tokens reflects the logical input (stored prefix +
        %% new tail), matching infer/4 semantics for HTTP-layer
        %% input-token reporting.
        ?assertEqual(
            length(Turn1Tokens) + length(Suffix),
            maps:get(prompt_tokens, Stats)
        ),
        ?assertEqual(ok, barrel_inference:end_session(<<"test_model">>, SessionId))
    end).

%% expect_committed matching the session's stored tokens admits as
%% usual.
continue_with_matching_committed_admits_test() ->
    SessionId = make_ref(),
    with_model(#{}, fun(_Cfg) ->
        {ok, #{generated := Gen1}} = barrel_inference_model:complete(
            <<"test_model">>,
            long_prompt(),
            #{response_tokens => 3, session_id => SessionId}
        ),
        Turn1Tokens = prompt_tokens(long_prompt()) ++ Gen1,
        {ok, Ref} = barrel_inference:continue(
            <<"test_model">>,
            [12345, 67890],
            #{
                session_id => SessionId,
                caller_pid => self(),
                response_tokens => 2,
                expect_committed => Turn1Tokens
            }
        ),
        Stats = drain_done(Ref, 5000),
        ?assertEqual(continuation, maps:get(cache_hit_kind, Stats)),
        ?assertEqual(ok, barrel_inference:end_session(<<"test_model">>, SessionId))
    end).

%% A divergent expect_committed fails fast without prefilling and
%% leaves the session pinned so a follow-up continue still admits.
continue_with_wrong_committed_rejects_test() ->
    SessionId = make_ref(),
    with_model(#{}, fun(_Cfg) ->
        {ok, #{generated := Gen1}} = barrel_inference_model:complete(
            <<"test_model">>,
            long_prompt(),
            #{response_tokens => 3, session_id => SessionId}
        ),
        Turn1Tokens = prompt_tokens(long_prompt()) ++ Gen1,
        Wrong = Turn1Tokens ++ [999999],
        Result = barrel_inference:continue(
            <<"test_model">>,
            [12345],
            #{
                session_id => SessionId,
                caller_pid => self(),
                response_tokens => 2,
                expect_committed => Wrong
            }
        ),
        ?assertMatch({error, {transcript_mismatch, _}}, Result),
        {error, {transcript_mismatch, Detail}} = Result,
        ?assertEqual(length(Turn1Tokens), maps:get(stored_len, Detail)),
        ?assertEqual(length(Wrong), maps:get(expected_len, Detail)),
        ?assertEqual(length(Turn1Tokens), maps:get(diverge_at, Detail)),
        %% Seq not consumed: a follow-up continue without the guard
        %% still admits and completes.
        {ok, Ref} = barrel_inference:continue(
            <<"test_model">>,
            [12345],
            #{session_id => SessionId, caller_pid => self(), response_tokens => 2}
        ),
        _ = drain_done(Ref, 5000),
        ?assertEqual(ok, barrel_inference:end_session(<<"test_model">>, SessionId))
    end).

%% Without expect_committed the historical "trust the tail" path is
%% unchanged.
continue_without_committed_unchanged_test() ->
    SessionId = make_ref(),
    with_model(#{}, fun(_Cfg) ->
        {ok, _} = barrel_inference_model:complete(
            <<"test_model">>,
            long_prompt(),
            #{response_tokens => 3, session_id => SessionId}
        ),
        {ok, Ref} = barrel_inference:continue(
            <<"test_model">>,
            [12345, 67890],
            #{session_id => SessionId, caller_pid => self(), response_tokens => 2}
        ),
        Stats = drain_done(Ref, 5000),
        ?assertEqual(continuation, maps:get(cache_hit_kind, Stats)),
        ?assertEqual(ok, barrel_inference:end_session(<<"test_model">>, SessionId))
    end).

continue_returns_no_session_for_unknown_session_test() ->
    with_model(#{}, fun(_Cfg) ->
        ?assertEqual(
            {error, no_session},
            barrel_inference:continue(
                <<"test_model">>,
                [1, 2, 3],
                #{session_id => make_ref(), caller_pid => self()}
            )
        )
    end).

continue_returns_no_session_when_session_id_missing_test() ->
    %% Opts must carry session_id explicitly. An Opts map without it
    %% is a malformed call; reject with no_session before reaching
    %% the gen_statem so a stray caller can't bounce off the model.
    with_model(#{}, fun(_Cfg) ->
        ?assertEqual(
            {error, no_session},
            barrel_inference:continue(
                <<"test_model">>,
                [1, 2, 3],
                #{caller_pid => self()}
            )
        )
    end).

continue_returns_sticky_busy_when_seq_in_flight_test() ->
    %% Establish the session with a finished turn so session_seq is
    %% populated, then launch a long-running streaming infer that
    %% extends the stored prefix (so the sticky path keeps the
    %% session alive and the seq lands in req_table). The third
    %% call — continue/3 on the same session — must reject with
    %% sticky_busy without enqueueing.
    SessionId = make_ref(),
    with_model(#{}, fun(_Cfg) ->
        {ok, #{generated := Gen1}} = barrel_inference_model:complete(
            <<"test_model">>,
            long_prompt(),
            #{response_tokens => 3, session_id => SessionId}
        ),
        Turn1Tokens = prompt_tokens(long_prompt()) ++ Gen1,
        NewSuffix = prompt_tokens(<<"continue please">>),
        Turn2Tokens = Turn1Tokens ++ NewSuffix,
        {ok, Ref1} = barrel_inference_model:infer(
            <<"test_model">>,
            Turn2Tokens,
            #{response_tokens => 10000, session_id => SessionId},
            self()
        ),
        receive
            {barrel_inference_token, Ref1, _} -> ok;
            {barrel_inference_token_id, Ref1, _} -> ok
        after 2000 -> erlang:error(no_first_token)
        end,
        ?assertEqual(
            {error, sticky_busy},
            barrel_inference:continue(
                <<"test_model">>,
                [99],
                #{session_id => SessionId, caller_pid => self()}
            )
        ),
        ok = barrel_inference_model:cancel(Ref1),
        receive
            {barrel_inference_done, Ref1, _} -> ok
        after 5000 -> erlang:error(timeout_drain)
        end,
        ?assertEqual(ok, barrel_inference:end_session(<<"test_model">>, SessionId))
    end).

continue_with_empty_suffix_generates_from_current_kv_test() ->
    %% Edge case: empty Suffix means prefill_cursor=undefined and
    %% generation runs immediately from the existing KV. The path
    %% must not blow up on `lists:nthtail` style assumptions.
    SessionId = make_ref(),
    with_model(#{}, fun(_Cfg) ->
        {ok, _} = barrel_inference_model:complete(
            <<"test_model">>,
            long_prompt(),
            #{response_tokens => 3, session_id => SessionId}
        ),
        {ok, Ref} = barrel_inference:continue(
            <<"test_model">>,
            [],
            #{
                session_id => SessionId,
                caller_pid => self(),
                response_tokens => 1
            }
        ),
        Stats = drain_done(Ref, 5000),
        ?assertEqual(continuation, maps:get(cache_hit_kind, Stats)),
        ?assertEqual(ok, barrel_inference:end_session(<<"test_model">>, SessionId))
    end).

%% Drain streaming until the done message and return its Stats map.
drain_done(Ref, TimeoutMs) ->
    receive
        {barrel_inference_done, Ref, Stats} -> Stats;
        {barrel_inference_token, Ref, _} -> drain_done(Ref, TimeoutMs);
        {barrel_inference_token_id, Ref, _} -> drain_done(Ref, TimeoutMs);
        {barrel_inference_thinking_end, Ref, _} -> drain_done(Ref, TimeoutMs);
        {barrel_inference_tool_call_end, Ref, _} -> drain_done(Ref, TimeoutMs)
    after TimeoutMs ->
        erlang:error({timeout, drain_done})
    end.

%% =============================================================================
%% PR2: per-model observability snapshot (phase / pending_len /
%% last_cache_hit) readable lock-free from outside the gen_statem
%% =============================================================================

phase_starts_idle_and_reflects_state_test() ->
    with_model(#{}, fun(_Cfg) ->
        ?assertEqual(idle, barrel_inference:phase(<<"test_model">>)),
        {ok, _} = barrel_inference_model:complete(<<"test_model">>, short_prompt()),
        %% Back to idle after the synchronous complete returns.
        ?assertEqual(idle, barrel_inference:phase(<<"test_model">>))
    end).

phase_unknown_model_returns_idle_test() ->
    with_model(#{}, fun(_Cfg) ->
        ?assertEqual(idle, barrel_inference:phase(<<"never_loaded">>))
    end).

pending_len_zero_when_idle_test() ->
    with_model(#{}, fun(_Cfg) ->
        ?assertEqual(0, barrel_inference:pending_len(<<"test_model">>))
    end).

pending_len_increments_when_queued_test() ->
    %% Streaming infer with a huge response_tokens keeps the
    %% gen_statem in `generating` long enough to admit a queued
    %% second request. The pending_len read must come back
    %% instantly without serialising behind the in-flight decode —
    %% that's the whole point of the obs ETS table.
    with_model(#{}, fun(_Cfg) ->
        {ok, PromptTokens} = barrel_inference_model:tokenize(<<"test_model">>, <<"hi">>),
        {ok, Ref1} = barrel_inference_model:infer(
            <<"test_model">>, PromptTokens, #{response_tokens => 10000}, self()
        ),
        %% Wait for the first token so we know the model is past
        %% prefill and actively decoding.
        receive
            {barrel_inference_token, Ref1, _} -> ok;
            {barrel_inference_token_id, Ref1, _} -> ok
        after 2000 -> erlang:error(no_first_token)
        end,
        %% Issue a second infer from a separate process — the
        %% gen_statem:call for a queued request blocks until that
        %% request is dispatched, which only happens after Ref1
        %% finishes. From a worker we can fire the call and read
        %% pending_len on the test process while the worker waits.
        Parent = self(),
        spawn_link(fun() ->
            R2 =
                barrel_inference_model:infer(
                    <<"test_model">>, PromptTokens, #{response_tokens => 1}, Parent
                ),
            Parent ! {worker_returned, R2}
        end),
        %% Give the second call a beat to enter the gen_statem
        %% mailbox and land in `handle_common -> enqueue`.
        ok = wait_for_pending_len(<<"test_model">>, 1, 2000),
        %% Drain: cancel Ref1, then expect Ref2 to dispatch.
        ok = barrel_inference_model:cancel(Ref1),
        receive
            {barrel_inference_done, Ref1, _} -> ok
        after 5000 -> erlang:error(timeout_first)
        end,
        Ref2 =
            receive
                {worker_returned, {ok, R}} -> R
            after 5000 -> erlang:error(worker_timeout)
            end,
        receive
            {barrel_inference_done, Ref2, _} -> ok
        after 5000 -> erlang:error(timeout_second)
        end,
        ?assertEqual(0, barrel_inference:pending_len(<<"test_model">>))
    end).

%% Poll pending_len until it reaches Expected or Timeout expires.
%% Returns ok on success, raises on timeout. Used by the queue
%% observability test where the second infer call blocks until the
%% first finishes, so we cannot assert pending_len synchronously
%% after a `{ok, Ref}` return.
wait_for_pending_len(ModelId, Expected, TimeoutMs) ->
    Deadline = erlang:monotonic_time(millisecond) + TimeoutMs,
    wait_for_pending_len_loop(ModelId, Expected, Deadline).

wait_for_pending_len_loop(ModelId, Expected, Deadline) ->
    case barrel_inference:pending_len(ModelId) of
        Expected ->
            ok;
        _ ->
            case erlang:monotonic_time(millisecond) > Deadline of
                true ->
                    erlang:error({pending_len_timeout, ModelId, Expected});
                false ->
                    timer:sleep(10),
                    wait_for_pending_len_loop(ModelId, Expected, Deadline)
            end
    end.

last_cache_hit_undefined_before_any_request_test() ->
    with_model(#{}, fun(_Cfg) ->
        ?assertEqual(undefined, barrel_inference:last_cache_hit(<<"test_model">>))
    end).

last_cache_hit_after_cold_admission_reports_cold_test() ->
    %% Cold admission populates the obs row with kind=cold and
    %% prefix_len=0. Distinct from `undefined` (model never
    %% admitted anything) so external routers can tell them apart.
    with_model(#{}, fun(_Cfg) ->
        {ok, _} = barrel_inference_model:complete(<<"test_model">>, long_prompt(), #{
            response_tokens => 2
        }),
        ?assertEqual(
            #{kind => cold, prefix_len => 0},
            barrel_inference:last_cache_hit(<<"test_model">>)
        )
    end).

last_cache_hit_after_warm_resume_test() ->
    with_model(#{}, fun(_Cfg) ->
        Tokens = prompt_tokens(long_prompt()),
        {ok, #{finish_key := FinishKey}} =
            barrel_inference_model:prefill_only(<<"test_model">>, Tokens),
        ?assertMatch({ok, _Row}, wait_for_key(FinishKey, 1000)),
        %% Resume from FinishKey: the exact key for `Tokens` is now
        %% in the cache, so lookup_or_resume hits the exact path
        %% before consulting parent_key.
        {ok, _} = barrel_inference_model:complete(
            <<"test_model">>, long_prompt(), #{
                parent_key => FinishKey,
                response_tokens => 2
            }
        ),
        Hit = barrel_inference:last_cache_hit(<<"test_model">>),
        ?assertMatch(#{kind := exact, prefix_len := _}, Hit),
        #{prefix_len := PrefixLen} = Hit,
        ?assertEqual(length(Tokens), PrefixLen)
    end).

model_info_carries_phase_and_pending_len_test() ->
    with_model(#{}, fun(_Cfg) ->
        Info = barrel_inference_model:model_info(<<"test_model">>),
        ?assertEqual(idle, maps:get(phase, Info)),
        ?assertEqual(0, maps:get(pending_len, Info)),
        ?assertEqual(undefined, maps:get(last_cache_hit, Info)),
        %% Existing keys preserved.
        ?assertEqual(idle, maps:get(status, Info))
    end).

%% =============================================================================
%% Helpers
%% =============================================================================

%% =============================================================================
%% Multi-sequence scheduler (n_seq_max > 1)
%% =============================================================================

%% Two concurrent complete/3 calls with n_seq_max=2 should both
%% return their own results. The gen_statem dispatches them to
%% seq_ids 0 and 1, co-batches their decode through one step/2
%% call per tick, and replies to each caller at its own finish.
two_concurrent_completes_each_return_own_result_test_() ->
    {timeout, 10, fun two_concurrent_completes_each_return_own_result_/0}.

two_concurrent_completes_each_return_own_result_() ->
    ConfigOverrides = #{
        context_opts => #{n_seq_max => 2, n_batch => 64}
    },
    with_model(#{}, ConfigOverrides, fun(_Cfg) ->
        Parent = self(),
        Pid1 = spawn_link(fun() ->
            Parent !
                {one,
                    barrel_inference_model:complete(
                        <<"test_model">>, <<"hi">>, #{response_tokens => 2}
                    )}
        end),
        Pid2 = spawn_link(fun() ->
            Parent !
                {two,
                    barrel_inference_model:complete(
                        <<"test_model">>, <<"yo">>, #{response_tokens => 2}
                    )}
        end),
        Reply1 =
            receive
                {one, R} -> R
            after 5000 -> erlang:error(timeout_one)
            end,
        Reply2 =
            receive
                {two, R2} -> R2
            after 5000 -> erlang:error(timeout_two)
            end,
        ?assertMatch({ok, #{reply := _}}, Reply1),
        ?assertMatch({ok, #{reply := _}}, Reply2),
        %% Drain link signals.
        _ = Pid1,
        _ = Pid2,
        ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>))
    end).

%% Once a request finishes its seq_id must return to the idle pool
%% so a subsequent admission can reuse it.
seq_id_freed_on_finish_test_() ->
    {timeout, 10, fun seq_id_freed_on_finish_/0}.

seq_id_freed_on_finish_() ->
    ConfigOverrides = #{
        context_opts => #{n_seq_max => 2, n_batch => 64}
    },
    with_model(#{}, ConfigOverrides, fun(_Cfg) ->
        %% Run three sequential completes against an n_seq_max=2
        %% model. If finish doesn't recycle seq_ids, the third call
        %% would block forever.
        lists:foreach(
            fun(_) ->
                {ok, _} = barrel_inference_model:complete(
                    <<"test_model">>, <<"hi">>, #{response_tokens => 2}
                )
            end,
            lists:seq(1, 3)
        ),
        ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>))
    end).

%% Once n_seq_max admits are in flight, the next admit queues in
%% pending. Fire 3 concurrent infers against an n_seq_max=2 model;
%% the third must queue and only dispatch after one of the first
%% two finishes.
pending_fifo_fills_when_seq_ids_exhausted_test_() ->
    {timeout, 10, fun pending_fifo_fills_when_seq_ids_exhausted_/0}.

pending_fifo_fills_when_seq_ids_exhausted_() ->
    ConfigOverrides = #{
        context_opts => #{n_seq_max => 2, n_batch => 64}
    },
    with_model(#{}, ConfigOverrides, fun(_Cfg) ->
        {ok, Tokens} = barrel_inference_model:tokenize(<<"test_model">>, <<"hi">>),
        %% Three concurrent admits. With n_seq_max=2 the third is
        %% queued. All three should ultimately return their refs
        %% and emit a done message.
        Parent = self(),
        Spawn = fun(N) ->
            spawn_link(fun() ->
                R = barrel_inference_model:infer(
                    <<"test_model">>, Tokens, #{response_tokens => 2}, Parent
                ),
                Parent ! {ref, N, R}
            end)
        end,
        _ = [Spawn(N) || N <- [a, b, c]],
        Refs = lists:map(
            fun(_N) ->
                receive
                    {ref, _, {ok, Ref}} -> Ref
                after 5000 -> erlang:error(no_ref)
                end
            end,
            [a, b, c]
        ),
        %% Each Ref should emit at least one barrel_inference_done.
        lists:foreach(
            fun(Ref) ->
                receive
                    {barrel_inference_done, Ref, _} -> ok
                after 5000 -> erlang:error({no_done, Ref})
                end
            end,
            Refs
        ),
        %% Drain any stray token messages.
        drain_messages()
    end).

%% =============================================================================
%% Chunked prefill (PR5)
%% =============================================================================

%% A small prefill_chunk_size forces the slicer to split a 12-token
%% prompt across multiple ticks. The cursor advance must track the
%% actual slice length sent each tick: if it over- or under-counts,
%% the final context_tokens diverges from prompt ++ generated, the
%% finish key changes, and this test fails.
prefill_cursor_advances_in_chunks_test() ->
    with_model(#{prefill_chunk_size => 2}, fun(Cfg) ->
        {ok, #{generated := Gen, finish_key := FinishKey}} =
            barrel_inference_model:complete(<<"test_model">>, long_prompt(), #{
                response_tokens => 3
            }),
        FullTokens = prompt_tokens(long_prompt()) ++ Gen,
        ExpectedKey = key_for_tokens(FullTokens, Cfg),
        ?assertEqual(ExpectedKey, FinishKey),
        ?assertEqual(3, length(Gen)),
        ?assertEqual(idle, barrel_inference_model:status(<<"test_model">>))
    end).

%% Cold save must fire only when the trimmed prefix is fully
%% prefilled, not after each chunk. With prefill_chunk_size=2 the
%% trim is split across several ticks; the cold save fires once at
%% the end, capturing exactly the trim tokens.
prefill_chunks_cold_save_at_trim_boundary_test() ->
    with_model(#{prefill_chunk_size => 2}, fun(Cfg) ->
        {ok, _} = barrel_inference_model:complete(<<"test_model">>, long_prompt(), #{
            response_tokens => 2
        }),
        Tokens = prompt_tokens(long_prompt()),
        ColdKey = key_for_tokens(Tokens, Cfg),
        ?assertMatch({ok, _Row}, wait_for_key(ColdKey, 1000))
    end).

%% Default prefill_chunk_size is max(64, n_batch div 4).
prefill_chunk_size_default_test() ->
    ConfigOverrides = #{context_opts => #{n_batch => 1024}},
    with_model(#{}, ConfigOverrides, fun(_Cfg) ->
        Policy = barrel_inference_model:get_policy(<<"test_model">>),
        ?assertEqual(256, maps:get(prefill_chunk_size, Policy))
    end).

prefill_chunk_size_default_floor_test() ->
    ConfigOverrides = #{context_opts => #{n_batch => 64}},
    with_model(#{}, ConfigOverrides, fun(_Cfg) ->
        Policy = barrel_inference_model:get_policy(<<"test_model">>),
        ?assertEqual(64, maps:get(prefill_chunk_size, Policy))
    end).

drain_messages() ->
    receive
        _ -> drain_messages()
    after 0 -> ok
    end.

stub_detokenize_decimal(Tokens) ->
    string:join([integer_to_list(T) || T <- Tokens], " ").

make_tmp_dir() ->
    Base = os:getenv("TMPDIR", "/tmp"),
    Dir = filename:join(
        Base,
        "barrel_inference_model_tests_" ++
            integer_to_list(erlang:unique_integer([positive]))
    ),
    ok = file:make_dir(Dir),
    Dir.

rm_rf(Dir) ->
    case file:list_dir(Dir) of
        {ok, Entries} -> [file:delete(filename:join(Dir, E)) || E <- Entries];
        _ -> ok
    end,
    file:del_dir(Dir).
