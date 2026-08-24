%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% Cache-safety Common Test against non-dense model families.
%%
%% Two env-gated model groups; a case is skipped when its variable is
%% unset, so each group can run independently:
%%
%% ```
%%   LLAMA_TEST_RECURRENT_MODEL=/path/to/mamba-130m.Q8_0.gguf \
%%   LLAMA_TEST_SWA_MODEL=/path/to/gemma-3-270m-it-Q8_0.gguf \
%%       rebar3 ct --suite=erllama_family_SUITE
%% ```
%%
%% What it covers:
%% - model_info/1 reports the probed family (recurrent flag, arch)
%% - exact warm hit on a recurrent model either succeeds via
%%   recurrent-state rollback or takes the documented cold fallback
%%   (restore_failed bumped, reply identical to the cold run - the
%%   pre-0.11 bug produced a duplicated token here)
%% - partial warm hit prefills the suffix directly, so it stays warm
%%   on recurrent models (no partial seq_rm at all)
%% - the SWA (iSWA cache) model takes warm hits with restore_failed 0
%% @end
-module(erllama_family_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([
    all/0,
    init_per_suite/1,
    end_per_suite/1,
    init_per_testcase/2,
    end_per_testcase/2
]).

-export([
    recurrent_reports_family/1,
    recurrent_exact_hit_safe/1,
    recurrent_partial_resume_stays_warm/1,
    swa_reports_family/1,
    swa_exact_hit_stays_warm/1
]).

-define(RECURRENT_ENV, "LLAMA_TEST_RECURRENT_MODEL").
-define(SWA_ENV, "LLAMA_TEST_SWA_MODEL").
-define(LONG_PROMPT, <<
    "Once upon a time in a quiet village there lived a clever old woman "
    "who told stories to anyone who would listen. Her favorite was about "
    "a small fox that learned how to outwit a hungry wolf by hiding under "
    "a haystack and reading old letters. Every evening the children "
    "gathered around her hearth and she began the same way: "
>>).

%% =============================================================================
%% Common Test boilerplate
%% =============================================================================

all() ->
    [
        recurrent_reports_family,
        recurrent_exact_hit_safe,
        recurrent_partial_resume_stays_warm,
        swa_reports_family,
        swa_exact_hit_stays_warm
    ].

init_per_suite(Config) ->
    case {os:getenv(?RECURRENT_ENV), os:getenv(?SWA_ENV)} of
        {false, false} ->
            {skip, "set " ?RECURRENT_ENV " and/or " ?SWA_ENV " to GGUF paths to enable this suite"};
        _ ->
            {ok, _} = application:ensure_all_started(erllama),
            Config
    end.

end_per_suite(_Config) ->
    application:stop(erllama),
    ok.

init_per_testcase(TC, Config) ->
    case model_path_for(TC) of
        {skip, _} = Skip ->
            Skip;
        Path ->
            PrivDir = ?config(priv_dir, Config),
            Dir = filename:join(PrivDir, atom_to_list(TC) ++ "_dir"),
            ok = filelib:ensure_path(Dir),
            {evicted, _} = erllama_cache_meta_srv:gc(),
            DiskSrv = list_to_atom("family_disk_" ++ atom_to_list(TC)),
            {ok, _} = erllama_cache_disk_srv:start_link(DiskSrv, Dir),
            Model = iolist_to_binary(["family_model_", atom_to_binary(TC)]),
            {ok, _} = erllama_model:start_link(Model, model_config(Path, DiskSrv)),
            erllama_cache_counters:reset(),
            [{disk_srv, DiskSrv}, {model, Model}, {dir, Dir} | Config]
    end.

end_per_testcase(_TC, Config) ->
    erllama_test_helpers:stop_model_quiet(?config(model, Config)),
    erllama_test_helpers:stop_quiet(?config(disk_srv, Config)),
    ok.

model_path_for(TC) ->
    Env =
        case lists:prefix("recurrent", atom_to_list(TC)) of
            true -> ?RECURRENT_ENV;
            false -> ?SWA_ENV
        end,
    case os:getenv(Env) of
        false ->
            {skip, "set " ++ Env ++ " to a GGUF path to enable this case"};
        "" ->
            {skip, "empty " ++ Env};
        Path ->
            case filelib:is_regular(Path) of
                true -> Path;
                false -> {skip, "not a file: " ++ Path}
            end
    end.

model_config(Path, DiskSrv) ->
    Fp = file_sha256(Path),
    #{
        backend => erllama_model_llama,
        model_path => Path,
        model_opts => #{n_gpu_layers => 0},
        context_opts => #{n_ctx => 2048, n_batch => 512},
        tier_srv => DiskSrv,
        tier => disk,
        fingerprint => Fp,
        fingerprint_mode => safe,
        quant_type => f16,
        quant_bits => 16,
        ctx_params_hash => crypto:hash(sha256, term_to_binary({2048, 512})),
        context_size => 2048,
        %% trim 0 / align 1 so the cold save lands exactly on the
        %% full prompt: the second identical complete takes the
        %% EXACT hit path (the one that needs the seq_rm primer).
        policy => #{
            min_tokens => 16,
            cold_min_tokens => 16,
            cold_max_tokens => 4096,
            continued_interval => 4096,
            boundary_trim_tokens => 0,
            boundary_align_tokens => 1,
            session_resume_wait_ms => 1000
        }
    }.

%% =============================================================================
%% Recurrent model (Mamba-style)
%% =============================================================================

recurrent_reports_family(Config) ->
    Info = erllama_model:model_info(?config(model, Config)),
    ct:log("model_info: ~p", [Info]),
    ?assert(is_binary(maps:get(arch, Info))),
    ?assert(byte_size(maps:get(arch, Info)) > 0),
    ?assert(maps:get(recurrent, Info) orelse maps:get(hybrid, Info)),
    ?assert(maps:get(n_params, Info) > 0),
    ok.

recurrent_exact_hit_safe(Config) ->
    %% Cold run, then the same prompt again (exact hit). On archs
    %% without recurrent-state rollback the 1-token primer seq_rm
    %% fails; the engine must fall back to cold with restore_failed
    %% bumped and reproduce the cold reply byte-identically (the
    %% pre-0.11 bug re-prefilled the last token at a position the
    %% memory still held, corrupting the continuation). On rollback-
    %% capable archs the hit stays exact with restore_failed 0.
    Model = ?config(model, Config),
    Opts = #{response_tokens => 8, temperature => 0.0},
    {ok, #{reply := R1}} = erllama_model:complete(Model, ?LONG_PROMPT, Opts),
    timer:sleep(400),
    {ok, R2Map} = erllama_model:complete(Model, ?LONG_PROMPT, Opts),
    #{reply := R2, cache_hit_kind := Kind} = R2Map,
    Failed = maps:get(restore_failed, erllama_cache:get_counters()),
    ct:log("kind=~p restore_failed=~p~ncold=~ts~nsecond=~ts", [Kind, Failed, R1, R2]),
    case Kind of
        cold ->
            %% Documented fallback: same cold path both times, so the
            %% greedy replies must be byte-identical.
            ?assert(Failed >= 1),
            ?assertEqual(R1, R2);
        _Warm ->
            ?assertEqual(0, Failed),
            Common = binary:longest_common_prefix([R1, R2]),
            ?assert(Common >= byte_size(R1) div 2)
    end,
    ok.

recurrent_partial_resume_stays_warm(Config) ->
    %% Extend the cached prompt: the warm path prefills only the
    %% suffix at position N, with no partial seq_rm at all, so it
    %% must stay warm even on recurrent memories.
    Model = ?config(model, Config),
    Opts = #{response_tokens => 8, temperature => 0.0},
    {ok, _} = erllama_model:complete(Model, ?LONG_PROMPT, Opts),
    timer:sleep(400),
    Before = erllama_cache:get_counters(),
    Extended = <<?LONG_PROMPT/binary, " The next morning brought a quiet rain.">>,
    {ok, #{reply := Reply, cache_hit_kind := Kind}} =
        erllama_model:complete(Model, Extended, Opts),
    After = erllama_cache:get_counters(),
    Warm =
        (maps:get(hits_resume, After) - maps:get(hits_resume, Before)) +
            (maps:get(hits_longest_prefix, After) - maps:get(hits_longest_prefix, Before)),
    ct:log("kind=~p warm=~p reply=~ts", [Kind, Warm, Reply]),
    ?assertEqual(partial, Kind),
    ?assert(Warm >= 1),
    ?assertEqual(0, maps:get(restore_failed, After)),
    ?assert(byte_size(Reply) > 0),
    ok.

%% =============================================================================
%% SWA model (iSWA cache; partial seq_rm succeeds)
%% =============================================================================

swa_reports_family(Config) ->
    Info = erllama_model:model_info(?config(model, Config)),
    ct:log("model_info: ~p", [Info]),
    ?assert(maps:get(n_swa, Info) > 0),
    ?assertEqual(false, maps:get(recurrent, Info)),
    ok.

swa_exact_hit_stays_warm(Config) ->
    %% iSWA caches accept the 1-token primer removal in the vendored
    %% llama.cpp, so the exact hit must stay warm with no fallback.
    Model = ?config(model, Config),
    Opts = #{response_tokens => 8, temperature => 0.0},
    {ok, #{reply := R1}} = erllama_model:complete(Model, ?LONG_PROMPT, Opts),
    timer:sleep(400),
    {ok, #{reply := R2, cache_hit_kind := Kind}} =
        erllama_model:complete(Model, ?LONG_PROMPT, Opts),
    After = erllama_cache:get_counters(),
    ct:log("kind=~p~ncold=~ts~nwarm=~ts", [Kind, R1, R2]),
    ?assertEqual(exact, Kind),
    ?assertEqual(0, maps:get(restore_failed, After)),
    Common = binary:longest_common_prefix([R1, R2]),
    ?assert(Common >= byte_size(R1) div 2),
    ok.

%% =============================================================================
%% Helpers
%% =============================================================================

file_sha256(Path) ->
    {ok, Fd} = file:open(Path, [read, binary, raw, read_ahead]),
    try
        file_sha256_loop(Fd, crypto:hash_init(sha256))
    after
        file:close(Fd)
    end.

file_sha256_loop(Fd, Ctx) ->
    case file:read(Fd, 1 bsl 20) of
        {ok, Chunk} -> file_sha256_loop(Fd, crypto:hash_update(Ctx, Chunk));
        eof -> crypto:hash_final(Ctx)
    end.
