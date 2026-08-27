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

-export([load_config/1, request_opts/1, prefill_opts/1, chat_opts/1]).

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
    progress_to,
    %% erllama_model_stub only (test backend)
    step_delay_ms,
    thinking_capable,
    fail_seq_rm_last,
    fail_kv_unpack,
    fail_seq_cp,
    spec_draft,
    spec_draft_len,
    media_caps,
    media_n_pos,
    fail_media_prefill,
    %% multimodal projector (libmtmd)
    mmproj_path,
    mmproj_opts
]).

-define(MODEL_OPT_KEYS, [
    n_gpu_layers,
    main_gpu,
    use_mmap,
    use_mlock,
    load_mode,
    vocab_only,
    split_mode,
    tensor_split,
    devices,
    cpu_moe,
    tensor_buft_overrides,
    fit
]).

-define(CONTEXT_OPT_KEYS, [
    n_ctx,
    n_batch,
    n_ubatch,
    n_seq_max,
    n_rs_seq,
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

-define(SAMPLER_KEYS, [
    grammar,
    grammar_lazy,
    trigger_patterns,
    trigger_tokens,
    grammar_prefill,
    repetition_penalty,
    frequency_penalty,
    presence_penalty,
    penalty_last_n,
    top_k,
    top_p,
    min_p,
    typical_p,
    top_n_sigma,
    xtc_probability,
    xtc_threshold,
    dynatemp_range,
    dynatemp_exponent,
    min_keep,
    dry_multiplier,
    dry_base,
    dry_allowed_length,
    dry_penalty_last_n,
    dry_sequence_breakers,
    mirostat,
    mirostat_tau,
    mirostat_eta,
    logit_bias,
    ignore_eos,
    infill,
    logprobs,
    temperature,
    seed
]).

-define(CHAT_OPT_KEYS, [
    tools,
    tool_choice,
    parallel_tool_calls,
    json_schema,
    enable_thinking,
    reasoning_format,
    continue_final_message
]).

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
            middleware,
            speculative,
            media
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
        fun check_fit_exclusive/1,
        fun check_tier/1
    ]);
load_config(Other) ->
    {error, {invalid_config, config, Other}}.

%% The auto-fit only touches parameters still at their llama
%% defaults, so combining it with manual placement would fail only
%% when memory is tight. Reject the combinations loudly instead.
%% split_mode row is also rejected under fit (upstream only refuses
%% multi-device row; the blanket rule is simpler and documented).
check_fit_exclusive(#{model_opts := MOpts} = C) when is_map(MOpts) ->
    case maps:is_key(fit, MOpts) of
        false ->
            check_devices_none_split(C, MOpts);
        true ->
            Conflicts = [n_gpu_layers, tensor_split, cpu_moe, tensor_buft_overrides, vocab_only],
            case [K || K <- Conflicts, maps:is_key(K, MOpts)] of
                [] ->
                    case lists:member(maps:get(split_mode, MOpts, layer), [tensor, row]) of
                        true ->
                            {error, {invalid_config, fit, {incompatible, split_mode}}};
                        false ->
                            check_devices_none_split(C, MOpts)
                    end;
                [K | _] ->
                    {error, {invalid_config, fit, {incompatible, K}}}
            end
    end;
check_fit_exclusive(C) ->
    {ok, C}.

%% `devices => none` + `split_mode => tensor` fails inside llama
%% (the tensor meta device needs at least one device); reject early.
check_devices_none_split(C, MOpts) ->
    case
        maps:get(devices, MOpts, undefined) =:= none andalso
            maps:get(split_mode, MOpts, layer) =:= tensor
    of
        true -> {error, {invalid_config, devices, {incompatible, split_mode}}};
        false -> {ok, C}
    end.

%% Media requests bypass the KV cache and the flat-token-list
%% machinery; the session / cache-resume / speculation options make
%% no sense with them and are rejected loudly.
check_media_exclusive(#{media := _} = O) ->
    Conflicts = [session_id, parent_key, expect_committed, speculative, prefix_checkpoint_len],
    case [K || K <- Conflicts, maps:is_key(K, O)] of
        [] -> {ok, O};
        [K | _] -> {error, {unsupported_with_media, K}}
    end;
check_media_exclusive(O) ->
    {ok, O}.

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
    case check_types(C, base_load_checks() ++ backend_load_checks(), invalid_config) of
        ok -> check_sub_types(C);
        Err -> Err
    end.

base_load_checks() ->
    [
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
        {progress_to, fun is_pid/1},
        {model_opts, fun is_map/1},
        {context_opts, fun is_map/1},
        {policy, fun is_map/1},
        {mmproj_path, fun is_path/1},
        {mmproj_opts, fun mmproj_opts_check/1}
    ].

%% Stub-backend test knobs.
backend_load_checks() ->
    [
        {step_delay_ms, fun non_neg_int/1},
        {thinking_capable, fun is_boolean/1},
        {fail_seq_rm_last, fun is_boolean/1},
        {fail_kv_unpack, fun is_boolean/1},
        {fail_seq_cp, fun is_boolean/1},
        {spec_draft, fun(V) -> lists:member(V, [perfect, wrong, none]) end},
        {spec_draft_len, fun pos_int/1},
        {media_caps, fun is_map/1},
        {media_n_pos, fun pos_int/1},
        {fail_media_prefill, fun is_boolean/1}
    ].

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
        {split_mode, fun(V) -> lists:member(V, [none, layer, row, tensor]) end},
        {tensor_split, fun(V) -> is_list(V) andalso lists:all(fun is_number/1, V) end},
        {devices, fun devices_opt/1},
        {cpu_moe, fun cpu_moe_opt/1},
        {tensor_buft_overrides, fun tbo_opt/1},
        {fit, fun fit_opt/1}
    ].

%% `cpu_moe => true | N`: all expert tensors to CPU, or the first N
%% blocks.
cpu_moe_opt(true) -> true;
cpu_moe_opt(V) -> pos_int(V).

%% `devices => [NameBin] | none`. Names come from
%% erllama:list_devices/0; availability is checked at load.
devices_opt(none) ->
    true;
devices_opt(V) when is_list(V), V =/= [] ->
    length(V) =< 16 andalso
        lists:all(fun(D) -> is_binary(D) andalso D =/= <<>> end, V);
devices_opt(_) ->
    false.

%% `tensor_buft_overrides => [{PatternBin, cpu | DeviceNameBin}]`.
tbo_opt(V) when is_list(V) ->
    length(V) =< 4096 andalso
        lists:all(
            fun
                ({P, cpu}) ->
                    is_binary(P) andalso P =/= <<>>;
                ({P, D}) ->
                    is_binary(P) andalso P =/= <<>> andalso
                        is_binary(D) andalso D =/= <<>>;
                (_) ->
                    false
            end,
            V
        );
tbo_opt(_) ->
    false.

%% `fit => true | #{margin_mib => pos_int | [pos_int], n_ctx => auto,
%% min_ctx => pos_int}`. min_ctx only means anything with
%% `n_ctx => auto` (the auto branch is where fit may reduce the
%% context, bounded below by min_ctx).
fit_opt(true) ->
    true;
fit_opt(V) when is_map(V) ->
    Known = [margin_mib, n_ctx, min_ctx],
    KeysOk = lists:all(fun(K) -> lists:member(K, Known) end, maps:keys(V)),
    KeysOk andalso
        margin_ok(maps:get(margin_mib, V, 1)) andalso
        maps:get(n_ctx, V, auto) =:= auto andalso
        pos_int(maps:get(min_ctx, V, 1)) andalso
        (not maps:is_key(min_ctx, V) orelse maps:is_key(n_ctx, V));
fit_opt(_) ->
    false.

margin_ok(N) when is_integer(N) ->
    N > 0;
margin_ok(L) when is_list(L), L =/= [] ->
    length(L) =< 16 andalso lists:all(fun pos_int/1, L);
margin_ok(_) ->
    false.

context_opt_checks() ->
    [
        {n_ctx, fun pos_int/1},
        {n_batch, fun pos_int/1},
        {n_ubatch, fun pos_int/1},
        {n_seq_max, fun pos_int/1},
        {n_rs_seq, fun non_neg_int/1},
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
        fun(O) -> check_types_ok(O, request_checks()) end,
        fun check_media_exclusive/1
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

%% Chat-level option keys of `erllama:chat/3` / `chat_apply/3` (the
%% facade validates them separately from the request opts).
-spec chat_opts(map()) -> {ok, map()} | {error, term()}.
chat_opts(Opts) when is_map(Opts) ->
    steps(Opts, [
        fun(O) -> unknown(O, ?CHAT_OPT_KEYS) end,
        fun(O) -> check_types_ok(O, chat_checks()) end
    ]);
chat_opts(Other) ->
    {error, {invalid_option, opts, Other}}.

chat_checks() ->
    [
        {tools, fun(V) -> is_list(V) andalso lists:all(fun is_map/1, V) end},
        {tool_choice, fun(V) -> lists:member(V, [auto, required, none]) end},
        {parallel_tool_calls, fun is_boolean/1},
        {json_schema, fun(V) -> is_map(V) orelse is_binary(V) end},
        {enable_thinking, fun is_boolean/1},
        {reasoning_format, fun(V) -> lists:member(V, [deepseek, none]) end},
        {continue_final_message, fun(V) ->
            lists:member(V, [none, auto, content, reasoning])
        end}
    ].

%% `mmproj_opts => #{use_gpu, n_threads, image_min_tokens,
%% image_max_tokens}` passthrough to mtmd_context_params.
mmproj_opts_check(V) when is_map(V) ->
    Known = [use_gpu, n_threads, image_min_tokens, image_max_tokens],
    lists:all(fun(K) -> lists:member(K, Known) end, maps:keys(V)) andalso
        is_boolean(maps:get(use_gpu, V, true)) andalso
        pos_int(maps:get(n_threads, V, 1)) andalso
        pos_int(maps:get(image_min_tokens, V, 1)) andalso
        pos_int(maps:get(image_max_tokens, V, 1));
mmproj_opts_check(_) ->
    false.

request_checks() ->
    base_request_checks() ++ sampler_checks().

base_request_checks() ->
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
        {grammar_lazy, fun is_boolean/1},
        {trigger_patterns, fun(V) -> is_list(V) andalso lists:all(fun is_binary/1, V) end},
        {trigger_tokens, fun(V) -> is_list(V) andalso lists:all(fun non_neg_int/1, V) end},
        {grammar_prefill, fun is_binary/1},
        {speculative, fun speculative_opt/1},
        {media, fun media_opt/1}
    ].

%% `media => [#{type := image | audio, data := binary()}]` - encoded
%% image (jpg/png/bmp/gif) or audio (wav/mp3/flac) bytes; the prompt
%% must carry one media marker per item. Media requests bypass the
%% KV cache and are incompatible with the session / cache / spec
%% options (checked in check_media_exclusive).
media_opt(V) when is_list(V), V =/= [] ->
    length(V) =< 64 andalso
        lists:all(
            fun
                (#{type := T, data := D}) ->
                    (T =:= image orelse T =:= audio) andalso
                        is_binary(D) andalso D =/= <<>>;
                (_) ->
                    false
            end,
            V
        );
media_opt(_) ->
    false.

%% `speculative => true | false | #{type => ngram_mod, n_match | n_max
%% | n_min => pos_int()}`. `true` = ngram_mod with upstream defaults.
speculative_opt(V) when is_boolean(V) ->
    true;
speculative_opt(V) when is_map(V) ->
    Known = [type, n_match, n_max, n_min],
    maps:fold(
        fun
            (type, T, Acc) -> Acc andalso T =:= ngram_mod;
            (K, N, Acc) -> Acc andalso lists:member(K, Known) andalso pos_int(N)
        end,
        true,
        V
    );
speculative_opt(_) ->
    false.

sampler_checks() ->
    [
        {repetition_penalty, fun is_number/1},
        {frequency_penalty, fun is_number/1},
        {presence_penalty, fun is_number/1},
        {penalty_last_n, fun is_integer/1},
        {top_k, fun is_integer/1},
        {top_p, fun is_number/1},
        {min_p, fun is_number/1},
        {typical_p, fun is_number/1},
        {top_n_sigma, fun is_number/1},
        {xtc_probability, fun is_number/1},
        {xtc_threshold, fun is_number/1},
        {dynatemp_range, fun is_number/1},
        {dynatemp_exponent, fun is_number/1},
        {min_keep, fun pos_int/1},
        {dry_multiplier, fun is_number/1},
        {dry_base, fun is_number/1},
        {dry_allowed_length, fun is_integer/1},
        {dry_penalty_last_n, fun is_integer/1},
        {dry_sequence_breakers, fun(V) -> is_list(V) andalso lists:all(fun is_binary/1, V) end},
        {mirostat, fun(V) -> lists:member(V, [0, 1, 2]) end},
        {mirostat_tau, fun is_number/1},
        {mirostat_eta, fun is_number/1},
        {logit_bias, fun logit_bias_list/1},
        {ignore_eos, fun is_boolean/1},
        {infill, fun is_boolean/1},
        {logprobs, fun(V) -> is_integer(V) andalso V >= 0 andalso V =< 32 end},
        {temperature, fun is_number/1},
        {seed, fun non_neg_int/1}
    ].

logit_bias_list(V) when is_list(V) ->
    lists:all(
        fun
            ({Tok, Bias}) -> non_neg_int(Tok) andalso is_number(Bias);
            (_) -> false
        end,
        V
    );
logit_bias_list(_) ->
    false.

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
