%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Opt-in, on CUDA hosts (ERLLAMA_TP_HOSTS, see erllama_tp_nccl_SUITE):
%% KV checkpoints of a split model. A checkpoint holds every rank's
%% share of the KV at the same committed token, published as one cache
%% entry; a share missing publishes nothing.
%%
%% Each case starts its own serving node (so its own, empty, cache) and
%% fresh rank nodes for each group.
%%
%% The model is ERLLAMA_TP_MODEL (TinyLlama by default, whose KV the
%% split backend splits). The GLM cases (ERLLAMA_TP_GLM, a GLM-5.3-Flash
%% GGUF; skipped without it) cover a model whose MLA latent cache and
%% indexer the split backend mirrors on every rank.
-module(erllama_tp_checkpoint_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([warm_restore_after_reload/1, interrupted_save_is_a_miss/1]).
-export([glm_restore_one/1, glm_restore_four/1]).

-define(GLM_MOPTS, #{n_gpu_layers => 999, load_mode => direct_io}).
-define(GLM_COPTS, #{
    n_ctx => 4096,
    n_batch => 1024,
    flash_attn => enabled,
    n_rs_seq => 2,
    n_seq_max => 4,
    kv_unified => true
}).
-define(NO_CACHE, #{
    min_tokens => infinity, cold_min_tokens => infinity, continued_interval => infinity
}).
%% Each prompt's length (the same for every sequence, so the batches
%% and the KV cells are laid out alike on both paths) and the steps
%% decoded after the checkpoint.
-define(GLM_PROMPT_TOKENS, 48).
-define(GLM_STEPS, 32).

%% One cold save, at the largest 64-token boundary of a prompt of 128
%% tokens or more, and a finish save; no continued saves.
-define(POLICY, #{
    max_ladder_rows => 0,
    min_tokens => 64,
    cold_min_tokens => 128,
    cold_max_tokens => 1536,
    boundary_trim_tokens => 32,
    boundary_align_tokens => 64,
    continued_interval => infinity
}).

all() ->
    [warm_restore_after_reload, interrupted_save_is_a_miss, glm_restore_one, glm_restore_four].

init_per_suite(Config) ->
    case erllama_tp_hosts:hosts() of
        false -> {skip, "set ERLLAMA_TP_HOSTS to the CUDA hosts (see scripts/tp_hosts.sh)"};
        Hosts -> [{hosts, Hosts} | Config]
    end.

end_per_suite(_Config) ->
    ok.

%% A request saves a checkpoint; the group is unloaded and a new one
%% formed (new nodes); the same prompt then resumes from the
%% checkpoint, restored on every rank, and gives the same tokens.
warm_restore_after_reload(Config) ->
    Hosts = ?config(hosts, Config),
    Prompt = erllama_tp_hosts:long_prompt(30),
    erllama_tp_hosts:with_serving(Hosts, fun(S) ->
        G1 = erllama_tp_hosts:serve(Hosts, S, 10, ?POLICY),
        {Cold, Stats1} = erllama_tp_hosts:complete_stats(S, G1, Prompt),
        Pack = peer:call(S, erllama_tp_rank, checkpoint_stats, [rank0(G1)]),
        ct:pal("first group: ~p~ncheckpoint on rank 0: ~p", [cache(Stats1), Pack]),
        ?assertEqual(cold, maps:get(cache_hit_kind, Stats1)),
        ?assertMatch(#{pack := ok, pack_ok := 2}, Pack),
        ?assertEqual(0, maps:get(pack_failed, Pack, 0)),
        ok = erllama_tp_hosts:stop_group(G1),
        G2 = erllama_tp_hosts:serve(Hosts, S, 11, ?POLICY),
        {Warm, Stats2} = erllama_tp_hosts:complete_stats(S, G2, Prompt),
        Restore = peer:call(S, erllama_tp_rank, checkpoint_stats, [rank0(G2)]),
        ct:pal("second group: ~p~nrestore on rank 0: ~p", [cache(Stats2), Restore]),
        ?assertNotEqual(cold, maps:get(cache_hit_kind, Stats2)),
        ?assertMatch(#{unpack := ok}, Restore),
        ?assertEqual(Cold, Warm),
        ok = erllama_tp_hosts:stop_group(G2)
    end).

%% Rank 1 sends no share for the first checkpoint (the cold save): rank
%% 0 holds only its own share, so the save fails and publishes nothing.
%% A new group then misses on the same prompt and serves it cold, with
%% the same tokens.
interrupted_save_is_a_miss(Config) ->
    Hosts = ?config(hosts, Config),
    Prompt = erllama_tp_hosts:long_prompt(30),
    erllama_tp_hosts:with_serving(Hosts, fun(S) ->
        G1 = erllama_tp_hosts:serve(Hosts, S, 12, ?POLICY, #{tp_kv_timeout => 3000}),
        #{rank_pids := [_, R1 | _], rank_peers := [_, P1 | _]} = G1,
        ok = peer:call(P1, erllama_tp_rank, fault, [R1, drop_kv_pack]),
        {First, Stats1} = erllama_tp_hosts:complete_stats(S, G1, Prompt),
        Pack = peer:call(S, erllama_tp_rank, checkpoint_stats, [rank0(G1)]),
        ct:pal("save without rank 1's share: ~p~ncheckpoint on rank 0: ~p", [cache(Stats1), Pack]),
        %% The cold save failed (no share from rank 1), the finish save
        %% was made.
        ?assertMatch(#{pack_failed := 1, pack_ok := 1}, Pack),
        ok = erllama_tp_hosts:stop_group(G1),
        G2 = erllama_tp_hosts:serve(Hosts, S, 13, ?POLICY),
        {Again, Stats2} = erllama_tp_hosts:complete_stats(S, G2, Prompt),
        ct:pal("the same prompt on a new group: ~p", [cache(Stats2)]),
        ?assertEqual(cold, maps:get(cache_hit_kind, Stats2)),
        ?assertEqual(First, Again),
        ok = erllama_tp_hosts:stop_group(G2)
    end).

%% A split GLM checkpoints one sequence (or four together) after its
%% prompt on a fresh group, which then decodes on; another fresh group
%% restores the checkpoints and decodes the same inputs. Right after the
%% restore, before any decode, a checkpoint of every rank holds the same
%% bytes as the one restored, and rank 1's holds the mirrored tensors
%% (the same, non-zero bytes as rank 0's). The first step then leaves
%% the same KV on both groups, byte for byte, and the greedy tokens
%% that follow are the same.
glm_restore_one(Config) ->
    glm_restore(Config, [0]).

glm_restore_four(Config) ->
    glm_restore(Config, [0, 1, 2, 3]).

glm_restore(Config, Seqs) ->
    Hosts = ?config(hosts, Config),
    Model = glm_model(),
    os:putenv("ERLLAMA_TP_MODEL", Model),
    try
        erllama_tp_hosts:with_serving(Hosts, fun(S) ->
            ok = erllama_tp_hosts:load_module(S, erllama_tp_glm_checkpoint),
            G1 = glm_group(Hosts, S, 20),
            Prompts = glm_prompts(S, G1, Seqs),
            ok = glm_call(S, reference, [rank0(G1), Prompts, ?GLM_STEPS]),
            ok = erllama_tp_hosts:stop_group(G1),
            G2 = glm_group(Hosts, S, 21),
            Rows = glm_call(S, restored, [rank0(G2), ?GLM_STEPS, Seqs]),
            ok = erllama_tp_hosts:stop_group(G2),
            [ct:pal("~p", [R]) || R <- Rows],
            [
                begin
                    ?assert(MS > 16384),
                    ?assertEqual(MS, MR),
                    ?assert(Same),
                    ?assert(First),
                    ?assert(Toks)
                end
             || #{
                    mirrored_saved := MS,
                    mirrored_restored := MR,
                    restored_same := Same,
                    first_step_same := First,
                    tokens_same := Toks
                } <- Rows
            ]
        end)
    after
        os:unsetenv("ERLLAMA_TP_MODEL")
    end.

glm_model() ->
    case os:getenv("ERLLAMA_TP_GLM") of
        false -> exit({skip, "set ERLLAMA_TP_GLM to a GLM-5.3-Flash GGUF on the hosts"});
        M -> M
    end.

glm_group(Hosts, S, Gen) ->
    G0 = erllama_tp_hosts:form(Hosts, S, Gen),
    erllama_tp_hosts:load(
        G0,
        ?NO_CACHE,
        #{model_opts => ?GLM_MOPTS, context_opts => ?GLM_COPTS, tp_call_timeout => 3600000},
        3600000
    ).

%% The first prompt tokens of a different text for each sequence.
glm_prompts(S, #{id := Id}, Seqs) ->
    Texts = [
        <<"The history of ocean navigation begins with the stars, the currents and the winds.">>,
        <<"A modern operating system schedules its threads with run queues, priorities and time slices.">>,
        <<"Suspension bridges carry their decks on cables hung between towers and anchored at each end.">>,
        <<"The printing press made books cheap, and with them the ideas of the Reformation spread.">>
    ],
    [
        begin
            {ok, Toks} = peer:call(S, erllama, tokenize, [Id, lists:nth(Q + 1, Texts)]),
            {Q, lists:sublist(Toks ++ Toks, ?GLM_PROMPT_TOKENS)}
        end
     || Q <- Seqs
    ].

glm_call(S, Fn, Args) ->
    peer:call(S, erllama_tp_glm_checkpoint, Fn, Args, 3600000).

rank0(#{rank_pids := [R0 | _]}) -> R0.

cache(Stats) ->
    maps:with([cache_hit_kind, cache_delta, prompt_tokens, prefill_ms, finish_key], Stats).
