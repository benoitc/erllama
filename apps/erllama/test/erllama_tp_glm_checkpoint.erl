%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Test helper for erllama_tp_checkpoint_SUITE's GLM cases, loaded onto
%% the serving node: a checkpoint of each sequence taken on a fresh
%% group after its prompt (reference/3), then restored into another
%% fresh group and continued with the same inputs (restored/3), calls
%% going to rank 0's hosted backend as the serving node's do.
-module(erllama_tp_glm_checkpoint).

-export([reference/3, restored/3]).

%% `Prompts' ([{Seq, Tokens}]) prefilled in one step, a checkpoint of
%% each sequence, then `Steps' greedy decode steps of every sequence
%% together; the checkpoints after the first step too. Kept on this node
%% (a checkpoint is the whole KV of a sequence on every rank) for
%% restored/3.
reference(R0, Prompts, Steps) ->
    Seqs = [S || {S, _} <- Prompts],
    Sm = samplers(R0, Seqs),
    {ok, _} = be(R0, step, [[{S, {prefill, P}} || {S, P} <- Prompts]]),
    Packs = packs(R0, Seqs),
    {ok, First} = be(R0, step, [[{S, {decode, maps:get(S, Sm)}} || S <- Seqs]]),
    T1 = maps:from_list([{S, T} || {S, {token, T, _}} <- First]),
    After1 = packs(R0, Seqs),
    Tokens = decode(R0, Sm, Seqs, Steps - 1, #{}),
    persistent_term:put(?MODULE, #{
        packs => Packs, first => T1, after_first => After1, tokens => Tokens
    }),
    ok.

%% The reference's checkpoints restored on every rank of another group,
%% packed again at once (before any decode), then each sequence's first
%% token prefilled (a restore carries no logits; the reference decoded
%% the same token in a batch of the same shape) and `Steps - 1' greedy
%% decode steps. One row a sequence comparing both paths.
restored(R0, Steps, Seqs) ->
    #{packs := Packs, first := T1, after_first := RefAfter1, tokens := RefTokens} =
        persistent_term:get(?MODULE),
    persistent_term:erase(?MODULE),
    Sm = samplers(R0, Seqs),
    [ok = kv(R0, {kv_unpack, maps:get(S, Packs), S}) || S <- Seqs],
    Repacks = packs(R0, Seqs),
    {ok, _} = be(R0, step, [[{S, {prefill, [maps:get(S, T1)]}} || S <- Seqs]]),
    After1 = packs(R0, Seqs),
    Tokens = decode(R0, Sm, Seqs, Steps - 1, #{}),
    [
        #{
            seq => S,
            checkpoint_bytes => byte_size(maps:get(S, Packs)),
            mirrored_saved => mirrored_bytes(maps:get(S, Packs)),
            mirrored_restored => mirrored_bytes(maps:get(S, Repacks)),
            restored_same => maps:get(S, Repacks) =:= maps:get(S, Packs),
            first_step_same => maps:get(S, After1) =:= maps:get(S, RefAfter1),
            first_step_common => common(maps:get(S, After1), maps:get(S, RefAfter1)),
            tokens_same => maps:get(S, Tokens) =:= maps:get(S, RefTokens),
            tokens => {maps:get(S, RefTokens), maps:get(S, Tokens)}
        }
     || S <- Seqs
    ].

%% Each rank's share: the bytes before the first that differs.
common(F1, F2) ->
    {ok, A} = erllama_tp_checkpoint:unframe(F1, 2),
    {ok, B} = erllama_tp_checkpoint:unframe(F2, 2),
    [binary:longest_common_prefix([X, Y]) || {X, Y} <- lists:zip(A, B)].

%% Bytes (in 8-byte words) that are not zero and the same in rank 0's
%% and rank 1's shares of a checkpoint: a mirrored tensor is the same on
%% both ranks, while each rank's share of a split tensor holds the other
%% rank's slice as zeros. A checkpoint that read rank 1's mirrored
%% tensors from a placeholder has (almost) none.
mirrored_bytes(Frame) ->
    {ok, [A, B]} = erllama_tp_checkpoint:unframe(Frame, 2),
    same_nonzero(A, B, 0).

same_nonzero(<<X:64, RA/binary>>, <<X:64, RB/binary>>, N) when X =/= 0 ->
    same_nonzero(RA, RB, N + 8);
same_nonzero(<<_:64, RA/binary>>, <<_:64, RB/binary>>, N) ->
    same_nonzero(RA, RB, N);
same_nonzero(_, _, N) ->
    N.

decode(_R0, _Sm, Seqs, 0, Acc) ->
    maps:from_list([{S, lists:reverse(maps:get(S, Acc, []))} || S <- Seqs]);
decode(R0, Sm, Seqs, N, Acc) ->
    {ok, Res} = be(R0, step, [[{S, {decode, maps:get(S, Sm)}} || S <- Seqs]]),
    Acc1 = lists:foldl(
        fun({S, {token, T, _}}, A) -> maps:update_with(S, fun(L) -> [T | L] end, [T], A) end,
        Acc,
        Res
    ),
    decode(R0, Sm, Seqs, N - 1, Acc1).

samplers(R0, Seqs) ->
    maps:from_list([
        {S, element(2, {ok, _} = be(R0, sampler_new, [#{temperature => 0.0}]))}
     || S <- Seqs
    ]).

packs(R0, Seqs) ->
    maps:from_list([{S, kv(R0, {kv_pack, S})} || S <- Seqs]).

be(R0, Fn, Args) -> erllama_tp_rank:backend(R0, Fn, Args).

kv(R0, Req) -> gen_server:call(R0, {hosting, Req}, 600000).
