%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Collect-mode bench: runs a fixed set of workloads against a single
%% GGUF and writes a JSON document describing host, GPU, model, and
%% per-workload Stats. Companion of `bench/collect.sh` which detects
%% host/GPU metadata before invoking this module.
%%
%% Output is a single JSON object on disk. Schema is documented in
%% `bench/README.md` under the "Collect mode" section. Operators run
%% this on each machine they care about; aggregation is done off-line
%% by feeding the resulting files back into a separate summariser.
-module(erllama_bench_collect).

-include_lib("kernel/include/file.hrl").

-export([main/1, run/2]).

-define(SEED_SENTENCE,
    "The quick brown fox jumps over the lazy dog while the curious cat watches "
    "from the windowsill, contemplating the geometry of the afternoon light. "
).

%% ----- escript entry ---------------------------------------------------------

main([ModelPath, OutPath]) ->
    case run(ModelPath, OutPath) of
        ok ->
            io:format("wrote ~ts~n", [OutPath]),
            halt(0);
        {error, Reason} ->
            io:format(standard_error, "bench failed: ~p~n", [Reason]),
            halt(1)
    end;
main(_) ->
    io:format(standard_error, "usage: erllama_bench_collect <model-path> <out-json>~n", []),
    halt(2).

%% ----- public --------------------------------------------------------------

run(ModelPath, OutPath) ->
    case filelib:is_regular(ModelPath) of
        false -> {error, {model_not_found, ModelPath}};
        true ->
            T0 = erlang:monotonic_time(millisecond),
            ensure_app_started(),
            Tag = "bench_collect_" ++ integer_to_list(erlang:unique_integer([positive])),
            Dir = make_tmp_dir(Tag),
            DiskSrv = list_to_atom("disk_" ++ Tag),
            {ok, _} = erllama_cache_disk_srv:start_link(DiskSrv, Dir),
            ModelId = iolist_to_binary(["bench_", Tag]),
            Config = model_config(ModelPath, DiskSrv),
            LoadT0 = erlang:monotonic_time(millisecond),
            {ok, _} = erllama:load_model(ModelId, Config),
            LoadMs = erlang:monotonic_time(millisecond) - LoadT0,
            Workloads =
                try
                    run_workloads(ModelId)
                after
                    catch erllama:unload(ModelId),
                    catch gen_server:stop(DiskSrv),
                    catch rm_rf(Dir)
                end,
            ElapsedMs = erlang:monotonic_time(millisecond) - T0,
            Doc = #{
                schema_version => 1,
                captured_at => iso8601_now(),
                host => host_meta(),
                gpu => gpu_meta(),
                erllama => erllama_meta(),
                model => model_meta(ModelPath),
                config => config_meta(Config),
                model_load_ms => LoadMs,
                workloads => Workloads,
                elapsed_total_ms => ElapsedMs
            },
            ok = filelib:ensure_dir(OutPath),
            ok = file:write_file(OutPath, json:encode(Doc)),
            ok
    end.

%% ----- workloads -----------------------------------------------------------

run_workloads(Model) ->
    %% Each measurement starts with a freshly cleared cache so cross-
    %% workload prefix overlap doesn't accidentally warm a "cold"
    %% number. warm_long is the deliberate exception: it shares cache
    %% state with cold_long so the second call hits the row the first
    %% one wrote.
    ShortTokens = env_int("BENCH_SHORT_TOKENS", 200),
    LongTokens = env_int("BENCH_LONG_TOKENS", 500),
    RespTokens = env_int("BENCH_RESPONSE_TOKENS", 32),
    ShortPrompt = generate_prompt(ShortTokens),
    LongPrompt = generate_prompt(LongTokens),
    %% Warmup discard: pay the first-llama_decode JIT cost on a
    %% prompt large enough to compile the same prefill kernel the
    %% measured workloads will hit. Without this Metal compiles the
    %% big-batch kernel during the FIRST measured workload, which
    %% then reports several extra ms of "prefill time" that is
    %% really one-time kernel compilation. Result discarded.
    warmup(Model, LongTokens),
    reset_cache(),
    ColdShort = measure_complete(Model, <<"cold_short">>, ShortPrompt, RespTokens),
    reset_cache(),
    ColdLong = measure_complete(Model, <<"cold_long">>, LongPrompt, RespTokens),
    WarmLong = measure_complete(Model, <<"warm_long">>, LongPrompt, RespTokens),
    reset_cache(),
    ContinueTurns = measure_continue_3turn(Model, ShortPrompt, RespTokens),
    [
        ColdShort,
        ColdLong,
        WarmLong,
        #{
            name => <<"continue_3turn">>,
            turns => ContinueTurns
        }
    ].

warmup(Model, TargetTokens) ->
    %% Use a distinct seed so the warmup prompt does not share a
    %% prefix with the measured prompts (which would warm the cache
    %% beyond the kernel layer and contaminate cold_short / cold_long).
    Reps = max(1, (TargetTokens * 5) div 80 + 1),
    Prompt = list_to_binary(lists:duplicate(Reps, "Warmup pass content; ignore. ")),
    _ = erllama_model:complete(Model, Prompt, #{response_tokens => 4}),
    ok.

%% Clear the meta server's index and reset cache counters so the next
%% workload starts with a clean cache for its admission decision.
reset_cache() ->
    catch erllama_cache_meta_srv:gc(),
    catch erllama_cache_counters:reset(),
    ok.

measure_complete(Model, Name, Prompt, RespTokens) ->
    {ok, Result} = erllama_model:complete(Model, Prompt, #{response_tokens => RespTokens}),
    Stats = maps:get(stats, Result),
    Delta = maps:get(cache_delta, Stats),
    PromptTokens = maps:get(prompt_tokens, Stats),
    Completion = maps:get(completion_tokens, Stats),
    PrefillMs = maps:get(prefill_ms, Stats),
    GenMs = maps:get(generation_ms, Stats),
    #{
        name => Name,
        prompt_tokens => PromptTokens,
        completion_tokens => Completion,
        prefill_ms => PrefillMs,
        generation_ms => GenMs,
        tokens_per_sec_prefill => tps(PromptTokens, PrefillMs),
        tokens_per_sec_decode => tps(Completion, GenMs),
        cache_hit_kind => maps:get(cache_hit_kind, Stats),
        cache_delta_read => maps:get(read, Delta),
        cache_delta_created => maps:get(created, Delta)
    }.

measure_continue_3turn(Model, ShortPrompt, RespTokens) ->
    SessionId = make_ref(),
    %% Turn 1: infer/4 establishes the session.
    {ok, PromptTokens} = erllama:tokenize(Model, ShortPrompt),
    {ok, Ref1} = erllama_model:infer(
        Model,
        PromptTokens,
        #{response_tokens => RespTokens, session_id => SessionId},
        self()
    ),
    Stats1 = drain_done(Ref1, 120000),
    Turn1 = turn_record(1, <<"infer">>, Stats1),
    %% Turn 2: continue/3 with a tokenised suffix.
    {ok, Suffix2} = erllama:tokenize(Model, <<" Tell me more.">>),
    {ok, Ref2} = erllama:continue(
        Model,
        Suffix2,
        #{
            session_id => SessionId,
            caller_pid => self(),
            response_tokens => RespTokens
        }
    ),
    Stats2 = drain_done(Ref2, 120000),
    Turn2 = turn_record(2, <<"continue">>, Stats2),
    %% Turn 3: another continue/3.
    {ok, Suffix3} = erllama:tokenize(Model, <<" And then?">>),
    {ok, Ref3} = erllama:continue(
        Model,
        Suffix3,
        #{
            session_id => SessionId,
            caller_pid => self(),
            response_tokens => RespTokens
        }
    ),
    Stats3 = drain_done(Ref3, 120000),
    Turn3 = turn_record(3, <<"continue">>, Stats3),
    ok = erllama:end_session(Model, SessionId),
    [Turn1, Turn2, Turn3].

turn_record(N, Method, Stats) ->
    Delta = maps:get(cache_delta, Stats),
    PromptTokens = maps:get(prompt_tokens, Stats),
    Completion = maps:get(completion_tokens, Stats),
    PrefillMs = maps:get(prefill_ms, Stats),
    GenMs = maps:get(generation_ms, Stats),
    #{
        turn => N,
        method => Method,
        prompt_tokens => PromptTokens,
        completion_tokens => Completion,
        prefill_ms => PrefillMs,
        generation_ms => GenMs,
        tokens_per_sec_prefill => tps(PromptTokens, PrefillMs),
        tokens_per_sec_decode => tps(Completion, GenMs),
        cache_hit_kind => maps:get(cache_hit_kind, Stats),
        cache_delta_read => maps:get(read, Delta),
        cache_delta_created => maps:get(created, Delta)
    }.

drain_done(Ref, TimeoutMs) ->
    receive
        {erllama_done, Ref, Stats} -> Stats;
        {erllama_token, Ref, _} -> drain_done(Ref, TimeoutMs);
        {erllama_token_id, Ref, _} -> drain_done(Ref, TimeoutMs);
        {erllama_thinking_end, Ref, _} -> drain_done(Ref, TimeoutMs);
        {erllama_tool_call_end, Ref, _} -> drain_done(Ref, TimeoutMs);
        {erllama_error, Ref, R} -> erlang:error({stream_error, R})
    after TimeoutMs ->
        erlang:error({timeout, drain_done})
    end.

%% ----- metadata -----------------------------------------------------------

host_meta() ->
    #{
        hostname => binstr(os:getenv("HOSTNAME", net_adm:localhost())),
        os_kernel => binstr(os:getenv("ERLLAMA_BENCH_OS_KERNEL", "")),
        os_release => binstr(os:getenv("ERLLAMA_BENCH_OS_RELEASE", "")),
        arch => binstr(os:getenv("ERLLAMA_BENCH_ARCH", "")),
        cpu_brand => binstr(os:getenv("ERLLAMA_BENCH_CPU_BRAND", "")),
        physical_cores => env_int("ERLLAMA_BENCH_PHYSICAL_CORES", 0),
        ram_mb => env_int("ERLLAMA_BENCH_RAM_MB", 0)
    }.

gpu_meta() ->
    %% Populated by the shell wrapper (`bench/collect.sh`), which knows
    %% how to probe nvidia-smi / rocm-smi / system_profiler.
    Kind = binstr(os:getenv("ERLLAMA_BENCH_GPU_KIND", "cpu")),
    #{
        kind => Kind,
        name => binstr(os:getenv("ERLLAMA_BENCH_GPU_NAME", "")),
        memory_mb => env_int("ERLLAMA_BENCH_GPU_MEMORY_MB", 0),
        driver => binstr(os:getenv("ERLLAMA_BENCH_GPU_DRIVER", ""))
    }.

erllama_meta() ->
    Vsn =
        case application:get_key(erllama, vsn) of
            {ok, V} -> binstr(V);
            _ -> <<"unknown">>
        end,
    #{
        vsn => Vsn,
        otp_release => binstr(erlang:system_info(otp_release)),
        nif_loaded => is_nif_loaded(),
        git_commit => binstr(os:getenv("ERLLAMA_BENCH_GIT_COMMIT", "")),
        git_dirty => binstr(os:getenv("ERLLAMA_BENCH_GIT_DIRTY", "unknown"))
    }.

model_meta(Path) ->
    Basename = filename:basename(Path),
    Size =
        case file:read_file_info(Path) of
            {ok, #file_info{size = S}} -> S;
            _ -> 0
        end,
    Sha = binstr(os:getenv("ERLLAMA_BENCH_MODEL_SHA256", "")),
    #{
        path => binstr(Path),
        basename => binstr(Basename),
        sha256 => Sha,
        size_bytes => Size
    }.

config_meta(Config) ->
    CtxOpts = maps:get(context_opts, Config, #{}),
    ModelOpts = maps:get(model_opts, Config, #{}),
    #{
        n_gpu_layers => maps:get(n_gpu_layers, ModelOpts, 0),
        n_ctx => maps:get(n_ctx, CtxOpts, 0),
        n_batch => maps:get(n_batch, CtxOpts, 0),
        n_seq_max => maps:get(n_seq_max, CtxOpts, 1),
        response_tokens => env_int("BENCH_RESPONSE_TOKENS", 32)
    }.

%% ----- helpers ------------------------------------------------------------

ensure_app_started() ->
    application:set_env(erllama, scheduler, #{enabled => false}),
    {ok, _} = application:ensure_all_started(erllama),
    ok.

model_config(Path, DiskSrv) ->
    NGpuLayers = env_int("N_GPU_LAYERS", 999),
    NCtx = env_int("N_CTX", 4096),
    NBatch = env_int("N_BATCH", 4096),
    NSeqMax = env_int("N_SEQ_MAX", 1),
    %% Compute fingerprint from file bytes for stable cache keys
    %% across runs of the same GGUF (skipped if the caller already
    %% computed the sha256, to avoid double-reading large files).
    Fp =
        case os:getenv("ERLLAMA_BENCH_MODEL_SHA256") of
            false -> file_sha256(Path);
            "" -> file_sha256(Path);
            HexStr -> hex_to_bin(HexStr)
        end,
    #{
        backend => erllama_model_llama,
        model_path => Path,
        model_opts => #{n_gpu_layers => NGpuLayers},
        context_opts => #{n_ctx => NCtx, n_batch => NBatch, n_seq_max => NSeqMax},
        tier_srv => DiskSrv,
        tier => disk,
        fingerprint => Fp,
        fingerprint_mode => safe,
        quant_type => q4_k_m,
        quant_bits => 4,
        ctx_params_hash => crypto:hash(sha256, term_to_binary({NCtx, NBatch, NSeqMax})),
        context_size => NCtx,
        policy => #{
            min_tokens => 32,
            cold_min_tokens => 32,
            cold_max_tokens => 8192,
            continued_interval => 256,
            boundary_trim_tokens => 16,
            boundary_align_tokens => 32,
            session_resume_wait_ms => 500
        }
    }.

file_sha256(Path) ->
    {ok, Bin} = file:read_file(Path),
    crypto:hash(sha256, Bin).

hex_to_bin(Hex) ->
    Lower = string:to_lower(Hex),
    Pairs = [list_to_integer([A, B], 16) || [A, B] <- chunks_of_2(Lower)],
    list_to_binary(Pairs).

chunks_of_2([]) -> [];
chunks_of_2([A, B | T]) -> [[A, B] | chunks_of_2(T)];
chunks_of_2(_) -> [].

generate_prompt(TargetTokens) ->
    SeedLen = string:length(?SEED_SENTENCE),
    Reps = max(1, (TargetTokens * 5) div SeedLen + 1),
    list_to_binary(lists:duplicate(Reps, ?SEED_SENTENCE)).

make_tmp_dir(Tag) ->
    Base = os:getenv("TMPDIR", "/tmp"),
    Dir = filename:join(Base, "erllama_" ++ Tag),
    ok = filelib:ensure_path(Dir),
    Dir.

rm_rf(Dir) ->
    case file:list_dir(Dir) of
        {ok, Entries} -> [file:delete(filename:join(Dir, E)) || E <- Entries];
        _ -> ok
    end,
    file:del_dir(Dir).

tps(_Tokens, 0) -> 0.0;
tps(Tokens, Ms) -> Tokens * 1000 / Ms.

env_int(Name, Default) ->
    case os:getenv(Name) of
        false -> Default;
        "" -> Default;
        Str ->
            try
                list_to_integer(Str)
            catch
                _:_ -> Default
            end
    end.

binstr(B) when is_binary(B) -> B;
binstr(L) when is_list(L) -> list_to_binary(L);
binstr(A) when is_atom(A) -> atom_to_binary(A, utf8).

iso8601_now() ->
    %% UTC, second precision: 2026-05-18T13:00:00Z
    Now = erlang:system_time(second),
    list_to_binary(calendar:system_time_to_rfc3339(Now, [{offset, "Z"}, {time_designator, $T}])).

is_nif_loaded() ->
    try
        erllama_nif:module_info(exports),
        true
    catch
        _:_ -> false
    end.
