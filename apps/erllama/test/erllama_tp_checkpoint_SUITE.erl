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
-module(erllama_tp_checkpoint_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([warm_restore_after_reload/1, interrupted_save_is_a_miss/1]).

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
    [warm_restore_after_reload, interrupted_save_is_a_miss].

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

rank0(#{rank_pids := [R0 | _]}) -> R0.

cache(Stats) ->
    maps:with([cache_hit_kind, cache_delta, prompt_tokens, prefill_ms, finish_key], Stats).
