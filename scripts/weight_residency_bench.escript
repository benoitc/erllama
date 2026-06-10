#!/usr/bin/env escript
%%! -hidden

%% Compare weight_residency modes on a local GGUF.
%% Reports load wall-time, peak RSS, tokens/sec, resident_bytes, and
%% post-idle RSS for each of: eager, lazy, pinned, lazy_pin_resident.
%%
%% Run from the repo root after `rebar3 compile':
%%   BARREL_BENCH_GGUF=/path/to/model.gguf \
%%     escript apps/barrel_inference/scripts/weight_residency_bench.escript
%%
%% Note on OS page cache: modes run back-to-back in the same BEAM, so
%% by mode 2 the GGUF bytes are warm in the kernel cache. The bench
%% therefore shows the relative DIRECTION of each mode's effect (lazy
%% lower than eager, pinned highest, etc.), not magazine-perfect cold
%% numbers. For a true cold-cache run, restart the host between modes.

main(_) ->
    Gguf = case os:getenv("BARREL_BENCH_GGUF") of
        false ->
            io:format(standard_error,
                "error: set BARREL_BENCH_GGUF to the path of a local GGUF "
                "with a chat template (e.g. a Mistral / Qwen / Llama "
                "instruct quant).~n", []),
            halt(2);
        Path -> Path
    end,
    setup_paths(),
    {ok, _} = application:ensure_all_started(barrel_inference),
    Modes = [eager, lazy, pinned, lazy_pin_resident],
    Results = [bench_mode(M, Gguf) || M <- Modes],
    print_table(Results),
    halt(0).

bench_mode(Mode, Gguf) ->
    io:format("=== ~p ===~n", [Mode]),
    Model = atom_to_binary(Mode, utf8),
    Cfg = #{
        backend => barrel_inference_model_llama,
        model_path => list_to_binary(Gguf),
        model_opts => model_opts_for(Mode),
        context_opts => #{n_ctx => 4096, n_batch => 512, n_seq_max => 1}
    },
    LoadT0 = mono_ms(),
    {ok, _Pid} = barrel_inference:load_model(Model, Cfg),
    LoadMs = mono_ms() - LoadT0,
    timer:sleep(200),
    RssLoaded = rss_kb(),
    ResLoaded = barrel_inference:resident_bytes(Model),
    io:format("loaded: ~B ms, RSS=~.1f GB, resident=~.1f GB~n",
              [LoadMs, gb(RssLoaded * 1024), gb(ResLoaded)]),

    Prompt = render_prompt(Model),
    {ok, Tokens} = barrel_inference:tokenize(
        Model, Prompt, #{add_special => false, parse_special => true}
    ),
    io:format("prompt: ~B tokens~n", [length(Tokens)]),

    InferT0 = mono_ms(),
    {ok, Ref} = barrel_inference:infer(
        Model, Tokens,
        #{response_tokens => 64, temperature => 0.7, top_p => 0.9, top_k => 40},
        self()
    ),
    {Stats, _Buf} = collect(Ref),
    InferMs = mono_ms() - InferT0,
    RssAfterInfer = rss_kb(),
    ResAfterInfer = barrel_inference:resident_bytes(Model),
    io:format("inferred: ~B ms wall, RSS=~.1f GB, resident=~.1f GB~n",
              [InferMs, gb(RssAfterInfer * 1024), gb(ResAfterInfer)]),

    PromptToks = maps:get(prompt_tokens, Stats, 0),
    CompToks = maps:get(completion_tokens, Stats, 0),
    GenMs = maps:get(generation_ms, Stats, 1),
    PrefillMs = maps:get(prefill_ms, Stats, 0),
    Tps =
        case GenMs > 0 of
            true -> CompToks * 1000.0 / GenMs;
            false -> 0.0
        end,
    io:format("stats: ~B prompt + ~B gen tok, prefill=~B ms, gen=~B ms (~.1f tok/s)~n",
              [PromptToks, CompToks, PrefillMs, GenMs, Tps]),

    io:format("idle 30s...~n", []),
    timer:sleep(30_000),
    RssIdle = rss_kb(),
    ResIdle = barrel_inference:resident_bytes(Model),
    io:format("idle: RSS=~.1f GB, resident=~.1f GB~n",
              [gb(RssIdle * 1024), gb(ResIdle)]),

    ok = barrel_inference:unload(Model),
    timer:sleep(500),
    #{
        mode => Mode,
        load_ms => LoadMs,
        rss_loaded_gb => gb(RssLoaded * 1024),
        res_loaded_gb => gb(ResLoaded),
        rss_after_infer_gb => gb(RssAfterInfer * 1024),
        res_after_infer_gb => gb(ResAfterInfer),
        rss_idle_gb => gb(RssIdle * 1024),
        res_idle_gb => gb(ResIdle),
        prompt_tokens => PromptToks,
        gen_tokens => CompToks,
        prefill_ms => PrefillMs,
        gen_ms => GenMs,
        tokens_per_sec => Tps
    }.

%% Build a fake OTP lib layout under /tmp/bench_lib so that
%% application:load(barrel_inference) finds the .app file AND code:priv_dir/1
%% resolves to the directory holding our freshly built NIF .so.
setup_paths() ->
    Root = "/tmp/bench_lib",
    AppDir = filename:join(Root, "barrel_inference-0.1.0"),
    Ebin = filename:join(AppDir, "ebin"),
    Priv = filename:join(AppDir, "priv"),
    ok = filelib:ensure_path(Ebin),
    ok = filelib:ensure_path(Priv),
    %% Copy the compiled beams + the prebuilt NIF .so + the .app file.
    [{ok, _} = file:copy(B, filename:join(Ebin, filename:basename(B)))
        || B <- filelib:wildcard("/tmp/bench_ebin/*.beam")],
    {ok, _} = file:copy(
        "apps/barrel_inference/priv/barrel_inference_nif.so",
        filename:join(Priv, "barrel_inference_nif.so")
    ),
    %% Synthesize a minimal .app file. Modules are derived from the
    %% beams we just copied. No deps that would force boot ordering.
    Mods = [list_to_atom(filename:basename(F, ".beam"))
             || F <- filelib:wildcard(filename:join(Ebin, "*.beam"))],
    AppSpec = {application, barrel_inference,
        [{description, "barrel inference (bench fake)"},
         {vsn, "0.1.0"},
         {modules, Mods},
         {registered,
             [barrel_inference_sup, barrel_inference_cache_sup,
              barrel_inference_cache_meta_srv]},
         {mod, {barrel_inference_app, []}},
         {applications, [kernel, stdlib, sasl, crypto]},
         {env, [
             {tiers, [#{backend => ram, quota_mb => 4096}]},
             {default_fingerprint_mode, safe}
         ]}]},
    ok = file:write_file(
        filename:join(Ebin, "barrel_inference.app"),
        io_lib:format("~p.~n", [AppSpec])
    ),
    true = code:add_pathz(Ebin),
    ok.

model_opts_for(eager) ->
    #{use_mmap => true, use_mlock => false, prefetch => true};
model_opts_for(lazy) ->
    #{use_mmap => true, use_mlock => false, prefetch => false};
model_opts_for(pinned) ->
    #{use_mmap => true, use_mlock => true, prefetch => true};
model_opts_for(lazy_pin_resident) ->
    #{
        use_mmap => true, use_mlock => false, prefetch => false,
        pin_resident_after_first_request => true
    }.

render_prompt(Model) ->
    Messages = [
        #{<<"role">> => <<"user">>,
          <<"content">> => <<"Briefly: why is the sky blue? Two sentences.">>}
    ],
    Inputs = #{
        messages => iolist_to_binary(json:encode(Messages)),
        tools => <<"[]">>,
        tool_choice => auto,
        parallel_tool_calls => false
    },
    {ok, _Params, Prompt} = barrel_inference:chat_apply(Model, Inputs),
    Prompt.

collect(Ref) -> collect(Ref, []).
collect(Ref, Acc) ->
    receive
        {barrel_inference_token, Ref, Bin} when is_binary(Bin) ->
            collect(Ref, [Acc, Bin]);
        {barrel_inference_token, Ref, _} -> collect(Ref, Acc);
        {barrel_inference_thinking_end, Ref, _} -> collect(Ref, Acc);
        {barrel_inference_token_id, Ref, _} -> collect(Ref, Acc);
        {barrel_inference_done, Ref, Stats} ->
            {Stats, iolist_to_binary(Acc)};
        {barrel_inference_error, Ref, Reason} ->
            io:format("engine error: ~p~n", [Reason]),
            {#{}, iolist_to_binary(Acc)}
    after 300000 ->
        io:format("timeout~n", []),
        {#{}, iolist_to_binary(Acc)}
    end.

rss_kb() ->
    Cmd = "ps -o rss= -p " ++ os:getpid(),
    case string:trim(os:cmd(Cmd)) of
        "" -> 0;
        S ->
            try list_to_integer(S)
            catch _:_ -> 0
            end
    end.

mono_ms() -> erlang:monotonic_time(millisecond).

gb(Bytes) -> Bytes / (1024 * 1024 * 1024).

print_table(Results) ->
    io:format("~n~n## Results~n~n", []),
    io:format(
        "| mode | load (ms) | RSS @ load (GB) | resident @ load (GB) "
        "| RSS @ infer (GB) | resident @ infer (GB) | RSS @ idle (GB) "
        "| resident @ idle (GB) | prefill (ms) | gen (ms) | tok/s |~n", []
    ),
    io:format(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|~n", []
    ),
    lists:foreach(
        fun(R) ->
            io:format(
                "| ~p | ~B | ~.2f | ~.2f | ~.2f | ~.2f | ~.2f | ~.2f "
                "| ~B | ~B | ~.1f |~n",
                [
                    maps:get(mode, R),
                    maps:get(load_ms, R),
                    maps:get(rss_loaded_gb, R),
                    maps:get(res_loaded_gb, R),
                    maps:get(rss_after_infer_gb, R),
                    maps:get(res_after_infer_gb, R),
                    maps:get(rss_idle_gb, R),
                    maps:get(res_idle_gb, R),
                    maps:get(prefill_ms, R),
                    maps:get(gen_ms, R),
                    maps:get(tokens_per_sec, R)
                ]
            )
        end,
        Results
    ),
    ok.
