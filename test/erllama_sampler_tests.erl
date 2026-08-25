%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Stub-backend coverage for the sampler-params plumbing added in
%% Phase 2: every sampler key supplied via complete/3's Opts (or
%% infer/4's Params) must land on the backend's configure_sampler/2
%% callback verbatim. Non-sampler keys must be stripped.
-module(erllama_sampler_tests).
-include_lib("eunit/include/eunit.hrl").

with_app(Body) ->
    {ok, Started} = application:ensure_all_started(erllama),
    Dir = make_tmp_dir(),
    {ok, _} = erllama_cache_disk_srv:start_link(sampler_disk, Dir),
    try
        Body()
    after
        erllama_test_helpers:stop_quiet(sampler_disk),
        erllama_test_helpers:rm_rf(Dir),
        [application:stop(A) || A <- lists:reverse(Started)],
        ok
    end.

make_tmp_dir() ->
    Base = filename:basedir(user_cache, "erllama-sampler-tests"),
    Dir = filename:join(Base, integer_to_list(erlang:unique_integer([positive]))),
    ok = filelib:ensure_path(Dir),
    Dir.

minimal_config() ->
    #{
        backend => erllama_model_stub,
        tier_srv => sampler_disk,
        tier => disk,
        fingerprint => binary:copy(<<16#33>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        quant_bits => 16,
        ctx_params_hash => binary:copy(<<16#44>>, 32),
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
%% Tests
%% =============================================================================

empty_opts_no_sampler_cfg_test() ->
    %% No sampler keys in Opts -> stub receives configure_sampler(#{}).
    with_app(fun() ->
        Id = <<"sampler_a">>,
        {ok, _} = erllama:load_model(Id, minimal_config()),
        try
            {ok, _} = erllama:complete(Id, <<"hello world">>),
            ?assertEqual(#{}, get_stub_cfg(Id))
        after
            erllama:unload(Id)
        end
    end).

full_sampler_set_lands_on_backend_test() ->
    with_app(fun() ->
        Id = <<"sampler_b">>,
        {ok, _} = erllama:load_model(Id, minimal_config()),
        Opts = #{
            response_tokens => 4,
            parent_key => undefined,
            temperature => 0.7,
            top_k => 40,
            top_p => 0.9,
            min_p => 0.05,
            repetition_penalty => 1.1,
            frequency_penalty => 0.2,
            presence_penalty => 0.3,
            penalty_last_n => 128,
            typical_p => 0.95,
            top_n_sigma => 2.0,
            xtc_probability => 0.2,
            xtc_threshold => 0.15,
            dynatemp_range => 0.4,
            dynatemp_exponent => 1.5,
            min_keep => 2,
            dry_multiplier => 0.8,
            dry_base => 1.75,
            dry_allowed_length => 3,
            dry_penalty_last_n => 256,
            dry_sequence_breakers => [<<"\n">>, <<":">>],
            mirostat => 0,
            mirostat_tau => 4.0,
            mirostat_eta => 0.2,
            logit_bias => [{7, -5.0}],
            ignore_eos => true,
            infill => false,
            logprobs => 0,
            seed => 42
        },
        try
            {ok, _} = erllama:complete(Id, <<"hello">>, Opts),
            Cfg = get_stub_cfg(Id),
            ?assertEqual(0.7, maps:get(temperature, Cfg)),
            ?assertEqual(40, maps:get(top_k, Cfg)),
            ?assertEqual(0.9, maps:get(top_p, Cfg)),
            ?assertEqual(0.05, maps:get(min_p, Cfg)),
            ?assertEqual(1.1, maps:get(repetition_penalty, Cfg)),
            ?assertEqual(0.2, maps:get(frequency_penalty, Cfg)),
            ?assertEqual(0.3, maps:get(presence_penalty, Cfg)),
            ?assertEqual(128, maps:get(penalty_last_n, Cfg)),
            ?assertEqual(0.95, maps:get(typical_p, Cfg)),
            ?assertEqual(2.0, maps:get(top_n_sigma, Cfg)),
            ?assertEqual(0.2, maps:get(xtc_probability, Cfg)),
            ?assertEqual(0.15, maps:get(xtc_threshold, Cfg)),
            ?assertEqual(0.4, maps:get(dynatemp_range, Cfg)),
            ?assertEqual(1.5, maps:get(dynatemp_exponent, Cfg)),
            ?assertEqual(2, maps:get(min_keep, Cfg)),
            ?assertEqual(0.8, maps:get(dry_multiplier, Cfg)),
            ?assertEqual([<<"\n">>, <<":">>], maps:get(dry_sequence_breakers, Cfg)),
            ?assertEqual(0, maps:get(mirostat, Cfg)),
            ?assertEqual(4.0, maps:get(mirostat_tau, Cfg)),
            ?assertEqual([{7, -5.0}], maps:get(logit_bias, Cfg)),
            ?assertEqual(true, maps:get(ignore_eos, Cfg)),
            ?assertEqual(false, maps:get(infill, Cfg)),
            ?assertEqual(0, maps:get(logprobs, Cfg)),
            ?assertEqual(42, maps:get(seed, Cfg)),
            %% Non-sampler keys must not leak in.
            ?assertNot(maps:is_key(response_tokens, Cfg)),
            ?assertNot(maps:is_key(parent_key, Cfg))
        after
            erllama:unload(Id)
        end
    end).

grammar_opt_lands_on_backend_test() ->
    with_app(fun() ->
        Id = <<"sampler_c">>,
        {ok, _} = erllama:load_model(Id, minimal_config()),
        try
            {ok, _} = erllama:complete(
                Id, <<"hi">>, #{grammar => <<"root ::= [01]+">>}
            ),
            Cfg = get_stub_cfg(Id),
            ?assertEqual(<<"root ::= [01]+">>, maps:get(grammar, Cfg))
        after
            erllama:unload(Id)
        end
    end).

lazy_grammar_opts_land_on_backend_test() ->
    %% The template-grammar keys chat/3 injects (grammar_lazy,
    %% trigger_patterns, trigger_tokens, grammar_prefill) flow through
    %% request validation and sampler_cfg_from to the backend's
    %% sampler_new.
    with_app(fun() ->
        Id = <<"sampler_lazy">>,
        {ok, _} = erllama:load_model(Id, minimal_config()),
        try
            {ok, _} = erllama:complete(
                Id, <<"hi">>, #{
                    grammar => <<"root ::= call">>,
                    grammar_lazy => true,
                    trigger_patterns => [<<"<tool_call>">>],
                    trigger_tokens => [42]
                }
            ),
            Cfg = get_stub_cfg(Id),
            ?assertEqual(true, maps:get(grammar_lazy, Cfg)),
            ?assertEqual([<<"<tool_call>">>], maps:get(trigger_patterns, Cfg)),
            ?assertEqual([42], maps:get(trigger_tokens, Cfg)),
            {ok, _} = erllama:complete(
                Id, <<"hi again">>, #{
                    grammar => <<"root ::= call">>,
                    grammar_prefill => <<"<|assistant|>">>
                }
            ),
            Cfg2 = get_stub_cfg(Id),
            ?assertEqual(<<"<|assistant|>">>, maps:get(grammar_prefill, Cfg2))
        after
            erllama:unload(Id)
        end
    end).

infer_path_also_configures_sampler_test() ->
    %% Same plumbing must work via the streaming infer/4 entry.
    with_app(fun() ->
        Id = <<"sampler_d">>,
        {ok, _} = erllama:load_model(Id, minimal_config()),
        try
            {ok, Tokens} = erllama:tokenize(Id, <<"a b c">>),
            {ok, Ref} = erllama:stream(Id, Tokens, #{
                response_tokens => 2, temperature => 0.5, seed => 7
            }),
            drain(Ref),
            Cfg = get_stub_cfg(Id),
            ?assertEqual(0.5, maps:get(temperature, Cfg)),
            ?assertEqual(7, maps:get(seed, Cfg))
        after
            erllama:unload(Id)
        end
    end).

drain(Ref) ->
    receive
        {erllama, Ref, {token, _}} -> drain(Ref);
        {erllama, Ref, {done, _}} -> ok;
        {erllama, Ref, {error, R}} -> ?assert({unexpected_error, R} =:= ok)
    after 5000 ->
        ?assert({timeout, drain} =:= ok)
    end.

%% =============================================================================
%% Helpers — read the stub's recorded config via the model's test
%% accessor.
%% =============================================================================

get_stub_cfg(ModelId) ->
    %% After the scheduler moved to step/2 + per-request samplers,
    %% the cfg is no longer kept on the backend state — it's
    %% snapshotted on the model gen_statem's #data so tests can
    %% read it back without poking at the opaque sampler_ref.
    case erllama_model:get_last_sampler_cfg(ModelId) of
        undefined -> #{};
        Cfg -> Cfg
    end.
