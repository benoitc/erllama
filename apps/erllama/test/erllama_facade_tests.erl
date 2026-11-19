%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Tests for the public erllama module façade. Exercises the
%% bucket-A additions: list_models/0, model_info/1, tokenize/2,
%% detokenize/2, unload/1, whereis/1, validation and not_loaded shapes.
-module(erllama_facade_tests).
-include_lib("eunit/include/eunit.hrl").

%% =============================================================================
%% Fixtures
%% =============================================================================

minimal_config() ->
    #{
        backend => erllama_model_stub,
        fingerprint => binary:copy(<<16#11>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        quant_bits => 16,
        ctx_params_hash => binary:copy(<<16#22>>, 32),
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

%% =============================================================================
%% list_models / load_model / unload
%% =============================================================================

empty_list_models_test() ->
    erllama_test_helpers:with_app(fun() ->
        ?assertEqual([], erllama:list_models())
    end).

load_unload_model_roundtrip_test() ->
    erllama_test_helpers:with_app(fun() ->
        {ok, Id} = erllama:load_model(<<"facade_a">>, minimal_config()),
        ?assertEqual(<<"facade_a">>, Id),
        ?assertMatch([_], erllama:list_models()),
        ok = erllama:unload(Id),
        ?assertEqual([], erllama:list_models())
    end).

load_model_auto_id_returns_binary_test() ->
    erllama_test_helpers:with_app(fun() ->
        {ok, Id} = erllama:load_model(minimal_config()),
        try
            ?assert(is_binary(Id)),
            ?assertMatch(<<"erllama_model_", _/binary>>, Id)
        after
            erllama:unload(Id)
        end
    end).

list_models_includes_metadata_test() ->
    erllama_test_helpers:with_app(fun() ->
        {ok, Id} = erllama:load_model(<<"facade_b">>, minimal_config()),
        try
            [Info] = erllama:list_models(),
            ?assertEqual(Id, maps:get(id, Info)),
            ?assertEqual(idle, maps:get(status, Info)),
            ?assertEqual(erllama_model_stub, maps:get(backend, Info)),
            ?assertEqual(1024, maps:get(context_size, Info))
        after
            erllama:unload(Id)
        end
    end).

load_same_id_returns_already_loaded_test() ->
    erllama_test_helpers:with_app(fun() ->
        {ok, _} = erllama:load_model(<<"dup">>, minimal_config()),
        try
            ?assertEqual(
                {error, already_loaded},
                erllama:load_model(<<"dup">>, minimal_config())
            )
        after
            erllama:unload(<<"dup">>)
        end
    end).

%% =============================================================================
%% model_info / tokenize / detokenize
%% =============================================================================

model_info_test() ->
    erllama_test_helpers:with_app(fun() ->
        {ok, Id} = erllama:load_model(<<"facade_info">>, minimal_config()),
        try
            {ok, Info} = erllama:model_info(Id),
            ?assertEqual(Id, maps:get(id, Info)),
            ?assertEqual(idle, maps:get(status, Info))
        after
            erllama:unload(Id)
        end
    end).

tokenize_then_detokenize_test() ->
    erllama_test_helpers:with_app(fun() ->
        {ok, Id} = erllama:load_model(<<"facade_tok">>, minimal_config()),
        try
            {ok, Tokens} = erllama:tokenize(Id, <<"hello world">>),
            ?assert(is_list(Tokens)),
            {ok, Bin} = erllama:detokenize(Id, Tokens),
            ?assert(is_binary(Bin))
        after
            erllama:unload(Id)
        end
    end).

unload_unknown_returns_not_loaded_test() ->
    erllama_test_helpers:with_app(fun() ->
        ?assertEqual({error, not_loaded}, erllama:unload(<<"absent">>))
    end).

%% =============================================================================
%% not_loaded everywhere, validation, model_id, whereis
%% =============================================================================

not_loaded_shapes_test() ->
    erllama_test_helpers:with_app(fun() ->
        Id = <<"facade_absent">>,
        ?assertEqual({error, not_loaded}, erllama:model_info(Id)),
        ?assertEqual({error, not_loaded}, erllama:status(Id)),
        ?assertEqual({error, not_loaded}, erllama:phase(Id)),
        ?assertEqual({error, not_loaded}, erllama:pending_len(Id)),
        ?assertEqual({error, not_loaded}, erllama:queue_depth(Id)),
        ?assertEqual({error, not_loaded}, erllama:last_cache_hit(Id)),
        ?assertEqual({error, not_loaded}, erllama:list_adapters(Id)),
        ?assertEqual({error, not_loaded}, erllama:whereis(Id)),
        ?assertEqual({error, not_loaded}, erllama:tokenize(Id, <<"x">>)),
        ?assertEqual({error, not_loaded}, erllama:detokenize(Id, [1])),
        ?assertEqual({error, not_loaded}, erllama:complete(Id, <<"x">>)),
        ?assertEqual({error, not_loaded}, erllama:prefill_only(Id, [1])),
        ?assertEqual({error, not_loaded}, erllama:stream(Id, [1], #{})),
        ?assertEqual({error, not_loaded}, erllama:end_session(Id, s)),
        ?assertEqual({error, not_loaded}, erllama:reset_session(Id, s)),
        ?assertEqual({error, not_loaded}, erllama:evict(Id)),
        ?assertEqual({error, not_loaded}, erllama:shutdown(Id)),
        ?assertEqual({error, not_loaded}, erllama:cached_prefix_len(Id, [1, 2])),
        ?assertEqual({ok, 0}, erllama:cached_prefix_len(Id, []))
    end).

loaded_shapes_test() ->
    erllama_test_helpers:with_app(fun() ->
        {ok, Id} = erllama:load_model(<<"facade_shapes">>, minimal_config()),
        try
            {ok, Pid} = erllama:whereis(Id),
            ?assert(is_pid(Pid)),
            ?assertEqual({ok, Pid}, erllama:whereis(Pid)),
            ?assertEqual({ok, idle}, erllama:status(Id)),
            ?assertEqual({ok, idle}, erllama:phase(Id)),
            ?assertEqual({ok, 0}, erllama:pending_len(Id)),
            ?assertEqual({ok, 0}, erllama:queue_depth(Id)),
            ?assertEqual({ok, undefined}, erllama:last_cache_hit(Id)),
            ?assertEqual({ok, []}, erllama:list_adapters(Id)),
            ?assertEqual(ok, erllama:evict(Id)),
            {ok, #{reply := _}} = erllama:complete(Id, <<"hello world">>, #{response_tokens => 2}),
            ?assertMatch({ok, #{kind := _, prefix_len := _}}, erllama:last_cache_hit(Id))
        after
            erllama:unload(Id)
        end
    end).

model_id_from_config_test() ->
    erllama_test_helpers:with_app(fun() ->
        Config = (minimal_config())#{model_id => <<"facade_named">>},
        {ok, <<"facade_named">>} = erllama:load_model(Config),
        ?assertEqual({error, already_loaded}, erllama:load_model(Config)),
        ok = erllama:unload(<<"facade_named">>)
    end).

load_validation_test() ->
    erllama_test_helpers:with_app(fun() ->
        ?assertEqual({error, {missing_config, model_path}}, erllama:load_model(#{})),
        ?assertEqual(
            {error, {invalid_config, model_path, "/no/such.gguf"}},
            erllama:load_model(#{model_path => "/no/such.gguf"})
        ),
        ?assertEqual(
            {error, {unknown_option, bogus}},
            erllama:load_model((minimal_config())#{bogus => 1})
        ),
        ?assertEqual([], erllama:list_models())
    end).

request_validation_test() ->
    erllama_test_helpers:with_app(fun() ->
        {ok, Id} = erllama:load_model(<<"facade_reqv">>, minimal_config()),
        try
            ?assertEqual(
                {error, {unknown_option, max_tokens}},
                erllama:complete(Id, <<"x">>, #{max_tokens => 1})
            ),
            ?assertEqual(
                {error, {invalid_option, response_tokens, 0}},
                erllama:complete(Id, <<"x">>, #{response_tokens => 0})
            ),
            ?assertEqual(
                {error, {unknown_option, response_tokens}},
                erllama:prefill_only(Id, [1, 2], #{response_tokens => 1})
            ),
            ?assertEqual(
                {error, {unknown_option, nope}},
                erllama:stream(Id, [1, 2], #{nope => 1})
            )
        after
            erllama:unload(Id)
        end
    end).

%% =============================================================================
%% Saves on unload and application stop
%% =============================================================================

%% Start a slow stub stream and wait for its first token, so the
%% request has a live context to save.
in_flight_stream(Model) ->
    {ok, Ref} = erllama:stream(Model, <<"one two three four five six">>, #{
        response_tokens => 500
    }),
    receive
        {erllama, Ref, {token, _}} -> Ref
    after 5000 -> error(no_first_token)
    end.

slow_model() ->
    {ok, M} = erllama:load_model((minimal_config())#{step_delay_ms => 20}),
    M.

shutdown_saves() ->
    maps:get(saves_shutdown, erllama_cache:get_counters()).

%% Consume the stream's events up to its terminal one and return it,
%% so no late message leaks into a later test sharing this process.
await_end(Ref) ->
    receive
        {erllama, Ref, {error, Reason}} -> {error, Reason};
        {erllama, Ref, {done, _}} -> done;
        {erllama, Ref, _} -> await_end(Ref)
    after 5000 -> error({stream_did_not_end, Ref})
    end.

unload_saves_in_flight_context_test_() ->
    {timeout, 30, fun() ->
        erllama_test_helpers:with_app(fun() ->
            M = slow_model(),
            Ref = in_flight_stream(M),
            Before = shutdown_saves(),
            ?assertEqual(ok, erllama:unload(M)),
            %% The interrupted stream ends with an error, not silence.
            ?assertEqual({error, not_loaded}, await_end(Ref)),
            ?assertEqual(Before + 1, shutdown_saves()),
            ?assertEqual({error, not_loaded}, erllama:unload(M))
        end)
    end}.

unload_idle_model_saves_nothing_test() ->
    erllama_test_helpers:with_app(fun() ->
        M = slow_model(),
        Before = shutdown_saves(),
        ?assertEqual(ok, erllama:unload(M)),
        ?assertEqual(Before, shutdown_saves())
    end).

app_stop_saves_in_flight_context_test_() ->
    {timeout, 30, fun() ->
        {ok, Started} = application:ensure_all_started(erllama),
        try
            M = slow_model(),
            Ref = in_flight_stream(M),
            Before = shutdown_saves(),
            ok = application:stop(erllama),
            ?assertEqual({error, not_loaded}, await_end(Ref)),
            %% The counters outlive the application.
            ?assertEqual(Before + 1, shutdown_saves())
        after
            [application:stop(A) || A <- lists:reverse(Started)]
        end
    end}.

%% =============================================================================
%% A stream whose receiver exits is cancelled
%% =============================================================================

wait_until(Fun) -> wait_until(Fun, 200).
wait_until(_Fun, 0) ->
    error(condition_not_reached);
wait_until(Fun, N) ->
    case Fun() of
        true ->
            ok;
        false ->
            timer:sleep(10),
            wait_until(Fun, N - 1)
    end.

in_flight(Ref) -> erllama:request_info(Ref) =/= {error, not_found}.

%% 500 tokens at 20 ms would take 10 s; the cancel lands in a few steps.
receiver_exit_cancels_the_stream_test_() ->
    {timeout, 30, fun() ->
        erllama_test_helpers:with_app(fun() ->
            M = slow_model(),
            Receiver = spawn(fun() ->
                receive
                    stop -> ok
                end
            end),
            {ok, Ref} = erllama:stream(M, <<"one two three">>, #{
                response_tokens => 500, to => Receiver
            }),
            ok = wait_until(fun() -> erllama:status(M) =:= {ok, generating} end),
            exit(Receiver, kill),
            ok = wait_until(fun() -> not in_flight(Ref) end),
            ok = wait_until(fun() -> erllama:status(M) =:= {ok, idle} end),
            ?assertEqual(#{}, maps:get(receivers, sys:get_state(erllama_inflight)))
        end)
    end}.

dead_receiver_is_cancelled_at_admission_test_() ->
    {timeout, 30, fun() ->
        erllama_test_helpers:with_app(fun() ->
            M = slow_model(),
            {Dead, Mon} = spawn_monitor(fun() -> ok end),
            receive
                {'DOWN', Mon, process, Dead, _} -> ok
            end,
            {ok, Ref} = erllama:stream(M, <<"one two three">>, #{response_tokens => 500, to => Dead}),
            ok = wait_until(fun() -> not in_flight(Ref) end),
            ok = wait_until(fun() -> erllama:status(M) =:= {ok, idle} end)
        end)
    end}.

%% A stream that ends normally leaves no receiver watch behind.
finished_stream_drops_its_receiver_watch_test() ->
    erllama_test_helpers:with_app(fun() ->
        {ok, M} = erllama:load_model(minimal_config()),
        {ok, Ref} = erllama:stream(M, <<"one two three">>, #{response_tokens => 3}),
        ?assertEqual(done, await_end(Ref)),
        ok = wait_until(fun() ->
            #{receivers := R, watched := W} = sys:get_state(erllama_inflight),
            R =:= #{} andalso W =:= #{}
        end)
    end).

%% =============================================================================
%% An unloaded id can be loaded again
%% =============================================================================

reload_same_id_after_unload_test() ->
    erllama_test_helpers:with_app(fun() ->
        {ok, <<"again">>} = erllama:load_model(<<"again">>, minimal_config()),
        ok = erllama:unload(<<"again">>),
        ?assertEqual({error, not_loaded}, erllama:whereis(<<"again">>)),
        ?assertEqual({ok, <<"again">>}, erllama:load_model(<<"again">>, minimal_config())),
        ?assertMatch({ok, _}, erllama:whereis(<<"again">>)),
        ok = erllama:unload(<<"again">>)
    end).

%% A crashed model releases its id too.
reload_same_id_after_crash_test() ->
    erllama_test_helpers:with_app(fun() ->
        {ok, Id} = erllama:load_model(<<"crashy">>, minimal_config()),
        {ok, Pid} = erllama:whereis(Id),
        Mon = monitor(process, Pid),
        exit(Pid, kill),
        receive
            {'DOWN', Mon, process, Pid, _} -> ok
        end,
        ?assertEqual({ok, Id}, erllama:load_model(Id, minimal_config())),
        ok = erllama:unload(Id)
    end).
