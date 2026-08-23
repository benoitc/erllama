%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_opts_tests).

-include_lib("eunit/include/eunit.hrl").

-define(M, erllama_opts).

%% ---------------------------------------------------------------------------
%% load_config/1
%% ---------------------------------------------------------------------------

load_defaults_backend_to_llama_test() ->
    Path = existing_file(),
    {ok, C} = ?M:load_config(#{model_path => Path}),
    ?assertEqual(erllama_model_llama, maps:get(backend, C)).

load_requires_model_path_for_llama_test() ->
    ?assertEqual({error, {missing_config, model_path}}, ?M:load_config(#{})).

load_stub_needs_no_path_test() ->
    ?assertMatch({ok, _}, ?M:load_config(#{backend => erllama_model_stub})).

load_rejects_missing_file_test() ->
    ?assertEqual(
        {error, {invalid_config, model_path, "/nonexistent/model.gguf"}},
        ?M:load_config(#{model_path => "/nonexistent/model.gguf"})
    ).

load_rejects_unknown_backend_test() ->
    ?assertEqual(
        {error, {invalid_config, backend, no_such_backend}},
        ?M:load_config(#{backend => no_such_backend})
    ).

load_rejects_unknown_key_test() ->
    ?assertEqual(
        {error, {unknown_option, bogus}},
        ?M:load_config(#{backend => erllama_model_stub, bogus => 1})
    ).

load_rejects_unknown_sub_key_test() ->
    ?assertEqual(
        {error, {unknown_option, {context_opts, n_foo}}},
        ?M:load_config(#{backend => erllama_model_stub, context_opts => #{n_foo => 1}})
    ),
    ?assertEqual(
        {error, {unknown_option, {model_opts, prefetch}}},
        ?M:load_config(#{backend => erllama_model_stub, model_opts => #{prefetch => true}})
    ),
    ?assertEqual(
        {error, {unknown_option, {policy, nope}}},
        ?M:load_config(#{backend => erllama_model_stub, policy => #{nope => 1}})
    ).

load_type_checks_test_() ->
    Base = #{backend => erllama_model_stub},
    Bad = [
        {fingerprint, <<1, 2, 3>>},
        {fingerprint_mode, fastest},
        {quant_bits, 0},
        {ctx_params_hash, <<>>},
        {context_size, -1},
        {tier, tape},
        {tier_srv, "disk"},
        {model_id, not_a_binary},
        {step_delay_ms, -5},
        {context_opts, []},
        {model_opts, []},
        {policy, []}
    ],
    [
        ?_assertEqual({error, {invalid_config, K, V}}, ?M:load_config(maps:put(K, V, Base)))
     || {K, V} <- Bad
    ].

load_sub_type_checks_test_() ->
    Base = #{backend => erllama_model_stub},
    Cases = [
        {{error, {invalid_config, n_ctx, 0}}, Base#{context_opts => #{n_ctx => 0}}},
        {{error, {invalid_config, flash_attn, 'maybe'}}, Base#{
            context_opts => #{flash_attn => 'maybe'}
        }},
        {{error, {invalid_config, type_k, q9_0}}, Base#{context_opts => #{type_k => q9_0}}},
        {{error, {invalid_config, split_mode, diagonal}}, Base#{
            model_opts => #{split_mode => diagonal}
        }},
        {{error, {invalid_config, use_mmap, 1}}, Base#{model_opts => #{use_mmap => 1}}},
        {{error, {invalid_config, min_tokens, -1}}, Base#{policy => #{min_tokens => -1}}}
    ],
    Good = Base#{
        context_opts => #{n_ctx => 4096, kv_unified => true, type_k => q8_0},
        model_opts => #{n_gpu_layers => -1, split_mode => layer, tensor_split => [0.5, 0.5]},
        policy => #{min_tokens => 4, prefill_chunk_size => infinity}
    },
    [?_assertEqual(Expected, ?M:load_config(Config)) || {Expected, Config} <- Cases] ++
        [?_assertMatch({ok, _}, ?M:load_config(Good))].

%% ---------------------------------------------------------------------------
%% request_opts/1 and prefill_opts/1
%% ---------------------------------------------------------------------------

request_accepts_known_keys_test() ->
    Opts = #{
        response_tokens => 8,
        parent_key => binary:copy(<<1>>, 32),
        session_id => s1,
        on_full => error,
        stop_sequences => [<<"\n">>],
        thinking => enabled,
        thinking_budget_tokens => 10,
        temperature => 0.2,
        top_p => 0.9,
        top_k => 40,
        min_p => 0.05,
        repetition_penalty => 1.1,
        seed => 7,
        grammar => <<"root ::= \"a\"">>,
        prefix_checkpoint_len => 3,
        to => self(),
        expect_committed => [1, 2, 3],
        middleware => [fun(R, Next) -> Next(R) end]
    },
    ?assertEqual({ok, Opts}, ?M:request_opts(Opts)).

request_rejects_unknown_key_test() ->
    ?assertEqual({error, {unknown_option, max_tokens}}, ?M:request_opts(#{max_tokens => 5})).

request_type_checks_test_() ->
    Bad = [
        {response_tokens, 0},
        {parent_key, <<"short">>},
        {on_full, retry},
        {stop_sequences, [stop]},
        {thinking, yes},
        {thinking_budget_tokens, 0},
        {prefix_checkpoint_len, -1},
        {to, not_a_pid},
        {expect_committed, [a]},
        {middleware, [not_a_fun]},
        {grammar, "string"},
        {temperature, hot},
        {seed, -1}
    ],
    [
        ?_assertEqual({error, {invalid_option, K, V}}, ?M:request_opts(maps:put(K, V, #{})))
     || {K, V} <- Bad
    ].

prefill_accepts_subset_only_test() ->
    ?assertMatch({ok, _}, ?M:prefill_opts(#{parent_key => undefined, on_full => block})),
    ?assertEqual(
        {error, {unknown_option, response_tokens}},
        ?M:prefill_opts(#{response_tokens => 4})
    ).

non_map_opts_test() ->
    ?assertEqual({error, {invalid_option, opts, []}}, ?M:request_opts([])),
    ?assertEqual({error, {invalid_config, config, nope}}, ?M:load_config(nope)).

%% ---------------------------------------------------------------------------

existing_file() ->
    Dir = filename:join(
        os:getenv("TMPDIR", "/tmp"),
        "erllama_opts_tests_" ++ integer_to_list(erlang:unique_integer([positive]))
    ),
    ok = filelib:ensure_path(Dir),
    Path = filename:join(Dir, "model.gguf"),
    ok = file:write_file(Path, <<"GGUF">>),
    Path.
