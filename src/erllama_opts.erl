%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_opts).
-moduledoc false.
%% Validation of the maps the `erllama` façade accepts: the
%% `load_model` config and the per-request option maps. Pure
%% functions; every rejection is `{error, {missing_config, Key}}`,
%% `{error, {invalid_config, Key, Value}}`, `{error, {unknown_option,
%% Key}}` or `{error, {invalid_option, Key, Value}}` so callers get a
%% typed reason instead of a crash inside the model process.

-export([load_config/1, request_opts/1, prefill_opts/1]).

-define(LOAD_KEYS, [
    backend,
    model_path,
    model_opts,
    context_opts,
    fingerprint,
    fingerprint_mode,
    quant_type,
    quant_bits,
    ctx_params_hash,
    context_size,
    tier,
    tier_srv,
    policy,
    thinking_markers,
    chat_template,
    model_id,
    %% erllama_model_stub only (test backend)
    step_delay_ms,
    thinking_capable
]).

-define(MODEL_OPT_KEYS, [
    n_gpu_layers, main_gpu, use_mmap, use_mlock, load_mode, vocab_only, split_mode, tensor_split
]).

-define(CONTEXT_OPT_KEYS, [
    n_ctx,
    n_batch,
    n_ubatch,
    n_seq_max,
    n_threads,
    n_threads_batch,
    embeddings,
    offload_kqv,
    kv_unified,
    flash_attn,
    type_k,
    type_v,
    decode_budget_ms
]).

-define(POLICY_KEYS, [
    min_tokens,
    cold_min_tokens,
    cold_max_tokens,
    continued_interval,
    boundary_trim_tokens,
    boundary_align_tokens,
    session_resume_wait_ms,
    prefill_chunk_size,
    ladder_interval,
    max_ladder_rows
]).

-define(SAMPLER_KEYS, [grammar, repetition_penalty, top_k, top_p, min_p, temperature, seed]).

-define(REQUEST_KEYS,
    ?SAMPLER_KEYS ++
        [
            response_tokens,
            parent_key,
            session_id,
            on_full,
            stop_sequences,
            thinking,
            thinking_budget_tokens,
            prefix_checkpoint_len,
            to,
            expect_committed,
            middleware
        ]
).

-define(PREFILL_KEYS, [parent_key, session_id, on_full, prefix_checkpoint_len, middleware]).

-define(KV_TYPES, [auto, f16, f32, bf16, q4_0, q5_0, q5_1, q8_0]).

%% -----------------------------------------------------------------------------
%% load_model config
%% -----------------------------------------------------------------------------

%% Returns the config with `backend` defaulted. `model_id` is left in
%% place; the façade strips it.
-spec load_config(map()) -> {ok, map()} | {error, term()}.
load_config(Config) when is_map(Config) ->
    Backend = maps:get(backend, Config, erllama_model_llama),
    Config1 = Config#{backend => Backend},
    steps(Config1, [
        fun(C) -> unknown(C, ?LOAD_KEYS) end,
        fun check_backend/1,
        fun check_model_path/1,
        fun(C) -> sub_keys(C, model_opts, ?MODEL_OPT_KEYS) end,
        fun(C) -> sub_keys(C, context_opts, ?CONTEXT_OPT_KEYS) end,
        fun(C) -> sub_keys(C, policy, ?POLICY_KEYS) end,
        fun check_load_types/1,
        fun check_tier/1
    ]);
load_config(Other) ->
    {error, {invalid_config, config, Other}}.

check_backend(#{backend := B} = C) when is_atom(B) ->
    case code:ensure_loaded(B) of
        {module, B} -> {ok, C};
        {error, _} -> {error, {invalid_config, backend, B}}
    end;
check_backend(#{backend := B}) ->
    {error, {invalid_config, backend, B}}.

check_model_path(#{backend := erllama_model_stub} = C) ->
    {ok, C};
check_model_path(#{model_path := Path} = C) ->
    case is_path(Path) andalso filelib:is_regular(Path) of
        true -> {ok, C};
        false -> {error, {invalid_config, model_path, Path}}
    end;
check_model_path(_C) ->
    {error, {missing_config, model_path}}.

is_path(P) when is_binary(P) -> true;
is_path(P) when is_list(P) -> io_lib:printable_unicode_list(P) orelse P =:= [];
is_path(_) -> false.

check_load_types(C) ->
    Checks = [
        {fingerprint, fun(V) -> is_binary(V) andalso byte_size(V) =:= 32 end},
        {fingerprint_mode, fun(V) -> lists:member(V, [safe, gguf_chunked, fast_unsafe]) end},
        {quant_type, fun is_atom/1},
        {quant_bits, fun pos_int/1},
        {ctx_params_hash, fun(V) -> is_binary(V) andalso byte_size(V) =:= 32 end},
        {context_size, fun pos_int/1},
        {tier, fun(V) -> lists:member(V, [ram, ram_file, disk]) end},
        {tier_srv, fun is_atom/1},
        {thinking_markers, fun is_map/1},
        {chat_template, fun is_binary/1},
        {model_id, fun is_binary/1},
        {step_delay_ms, fun non_neg_int/1},
        {thinking_capable, fun is_boolean/1},
        {model_opts, fun is_map/1},
        {context_opts, fun is_map/1},
        {policy, fun is_map/1}
    ],
    case check_types(C, Checks, invalid_config) of
        ok -> check_sub_types(C);
        Err -> Err
    end.

check_sub_types(C) ->
    case check_types(maps:get(model_opts, C, #{}), model_opt_checks(), invalid_config) of
        ok ->
            case
                check_types(maps:get(context_opts, C, #{}), context_opt_checks(), invalid_config)
            of
                ok ->
                    case check_types(maps:get(policy, C, #{}), policy_checks(), invalid_config) of
                        ok -> {ok, C};
                        Err -> Err
                    end;
                Err ->
                    Err
            end;
        Err ->
            Err
    end.

model_opt_checks() ->
    [
        {n_gpu_layers, fun is_integer/1},
        {main_gpu, fun non_neg_int/1},
        {use_mmap, fun is_boolean/1},
        {use_mlock, fun is_boolean/1},
        {load_mode, fun(V) -> lists:member(V, [auto, none, mmap, mlock, mmap_mlock, direct_io]) end},
        {vocab_only, fun is_boolean/1},
        {split_mode, fun(V) -> lists:member(V, [none, layer, row]) end},
        {tensor_split, fun(V) -> is_list(V) andalso lists:all(fun is_number/1, V) end}
    ].

context_opt_checks() ->
    [
        {n_ctx, fun pos_int/1},
        {n_batch, fun pos_int/1},
        {n_ubatch, fun pos_int/1},
        {n_seq_max, fun pos_int/1},
        {n_threads, fun pos_int/1},
        {n_threads_batch, fun pos_int/1},
        {embeddings, fun is_boolean/1},
        {offload_kqv, fun is_boolean/1},
        {kv_unified, fun is_boolean/1},
        {flash_attn, fun(V) -> lists:member(V, [auto, true, false, enabled, disabled]) end},
        {type_k, fun(V) -> lists:member(V, ?KV_TYPES) end},
        {type_v, fun(V) -> lists:member(V, ?KV_TYPES) end},
        {decode_budget_ms, fun pos_int/1}
    ].

policy_checks() ->
    [{K, fun(V) -> non_neg_int(V) orelse V =:= infinity end} || K <- ?POLICY_KEYS].

%% `tier_srv' must name a running tier server and `tier' must match
%% its backend, so a mismatch fails at load time instead of at the
%% first save.
check_tier(C) ->
    Tier = maps:get(tier, C, ram),
    Srv = maps:get(tier_srv, C, erllama_cache_ram),
    case {Tier, Srv} of
        {ram, erllama_cache_ram} ->
            {ok, C};
        {ram, Other} ->
            {error, {invalid_config, tier_srv, Other}};
        {_, erllama_cache_ram} ->
            {error, {invalid_config, tier, Tier}};
        _ ->
            case whereis(Srv) of
                undefined ->
                    {error, {invalid_config, tier_srv, Srv}};
                _Pid ->
                    case tier_backend(Srv) of
                        Tier -> {ok, C};
                        _ -> {error, {invalid_config, tier, Tier}}
                    end
            end
    end.

tier_backend(Srv) ->
    try
        erllama_cache_disk_srv:tier_of(Srv)
    catch
        exit:_ -> undefined
    end.

%% -----------------------------------------------------------------------------
%% per-request options (complete/3, stream/3, continue/3)
%% -----------------------------------------------------------------------------

-spec request_opts(map()) -> {ok, map()} | {error, term()}.
request_opts(Opts) when is_map(Opts) ->
    steps(Opts, [
        fun(O) -> unknown(O, ?REQUEST_KEYS) end,
        fun(O) -> check_types_ok(O, request_checks()) end
    ]);
request_opts(Other) ->
    {error, {invalid_option, opts, Other}}.

-spec prefill_opts(map()) -> {ok, map()} | {error, term()}.
prefill_opts(Opts) when is_map(Opts) ->
    steps(Opts, [
        fun(O) -> unknown(O, ?PREFILL_KEYS) end,
        fun(O) -> check_types_ok(O, request_checks()) end
    ]);
prefill_opts(Other) ->
    {error, {invalid_option, opts, Other}}.

request_checks() ->
    [
        {response_tokens, fun pos_int/1},
        {parent_key, fun(V) ->
            V =:= undefined orelse (is_binary(V) andalso byte_size(V) =:= 32)
        end},
        {on_full, fun(V) -> V =:= block orelse V =:= error end},
        {stop_sequences, fun(V) -> is_list(V) andalso lists:all(fun is_binary/1, V) end},
        {thinking, fun(V) -> V =:= enabled orelse V =:= disabled end},
        {thinking_budget_tokens, fun pos_int/1},
        {prefix_checkpoint_len, fun non_neg_int/1},
        {to, fun is_pid/1},
        {expect_committed, fun(V) -> is_list(V) andalso lists:all(fun non_neg_int/1, V) end},
        {middleware, fun(V) -> is_list(V) andalso lists:all(fun is_function/1, V) end},
        {grammar, fun is_binary/1},
        {repetition_penalty, fun is_number/1},
        {top_k, fun is_integer/1},
        {top_p, fun is_number/1},
        {min_p, fun is_number/1},
        {temperature, fun is_number/1},
        {seed, fun non_neg_int/1}
    ].

check_types_ok(O, Checks) ->
    case check_types(O, Checks, invalid_option) of
        ok -> {ok, O};
        Err -> Err
    end.

%% -----------------------------------------------------------------------------
%% helpers
%% -----------------------------------------------------------------------------

steps(Value, []) ->
    {ok, Value};
steps(Value, [F | Rest]) ->
    case F(Value) of
        {ok, V1} -> steps(V1, Rest);
        {error, _} = E -> E
    end.

unknown(Map, Known) ->
    case [K || K <- maps:keys(Map), not lists:member(K, Known)] of
        [] -> {ok, Map};
        [K | _] -> {error, {unknown_option, K}}
    end.

sub_keys(C, Key, Known) ->
    case maps:get(Key, C, #{}) of
        Sub when is_map(Sub) ->
            case [K || K <- maps:keys(Sub), not lists:member(K, Known)] of
                [] -> {ok, C};
                [K | _] -> {error, {unknown_option, {Key, K}}}
            end;
        Other ->
            {error, {invalid_config, Key, Other}}
    end.

check_types(_Map, [], _Tag) ->
    ok;
check_types(Map, [{Key, Pred} | Rest], Tag) ->
    case maps:find(Key, Map) of
        error ->
            check_types(Map, Rest, Tag);
        {ok, V} ->
            case Pred(V) of
                true -> check_types(Map, Rest, Tag);
                false -> {error, {Tag, Key, V}}
            end
    end.

pos_int(V) -> is_integer(V) andalso V > 0.
non_neg_int(V) -> is_integer(V) andalso V >= 0.
