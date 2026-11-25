%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
-module(erllama_model_tp).
-moduledoc false.
%% Experimental: the backend of a model split over a tensor-parallel
%% group whose GPUs are in other OS processes. The serving VM loads the
%% model with this backend (`erllama:load_model/2', `backend =>
%% erllama_model_tp'), so it is used like any loaded model; every rank,
%% the sampling one included, is a separate node, and a collective that
%% fails or wedges takes down a rank, never the serving VM.
%%
%% Tokenizing, detokenizing and chat templates run here, on a copy of
%% the GGUF's vocabulary (no weights). Everything that decodes, samples
%% or changes the KV runs on rank 0, which hosts erllama_model_llama
%% for this backend (erllama_tp_rank:host/3).
%%
%% Config, on top of erllama_model_llama's: `tp_worker' (rank 0's
%% erllama_tp_rank pid, joined to its group), `tp_followers' (the other
%% ranks' pids with their erllama_tp_rank:describe/3, every one loaded
%% with load/3), `tp_call_timeout' (ms, default 120000) and
%% `tp_kv_timeout' (how long rank 0 waits for the other ranks' shares of
%% a checkpoint, ms, default 30000).
%%
%% A rank lost (its node down, or a call past the timeout) makes every
%% call answer `{error, {tp_rank_lost, Why}}': the model fails its
%% requests once and stops, and the caller forms a new group. Past the
%% timeout rank 0 is killed: a native call blocked in a collective
%% ends only with its OS process.
-behaviour(erllama_model_backend).

-export([
    init/1,
    terminate/1,
    shared_state/1,
    tokenize/2,
    tokenize/3,
    detokenize/2,
    detokenize/3,
    vocab_info/1,
    get_model_ref/1,
    apply_chat_template/2,
    prefill/2,
    decode_one/2,
    kv_pack/2,
    kv_pack/3,
    kv_unpack/2,
    kv_unpack/3,
    seq_clear/1,
    seq_rm/2,
    seq_cp/3,
    seq_rm_last/2,
    seq_rm_last/3,
    step/2,
    sampler_new/2,
    sampler_free/1,
    set_grammar/2,
    configure_sampler/2,
    clear_sampler/1,
    reset_context/1,
    extra_metadata/1,
    thinking_signature/3,
    spec_supported/1,
    spec_new/2,
    spec_begin/4,
    spec_draft/5,
    spec_accept/4,
    spec_free/2,
    spec_step/4,
    spec_rollback/1,
    mtp_step_batch/4
]).

-ifdef(TEST).
-export([kv_format/2]).
-endif.

-record(tp, {
    %% This VM's copy of the vocabulary.
    vocab :: erllama_nif:model_ref() | undefined,
    %% Rank 0's process; `undefined' in the shared state.
    worker :: pid() | undefined,
    %% The split, part of the KV's identity: a checkpoint restores only
    %% into a group split the same way.
    layout = <<>> :: binary(),
    timeout = 120000 :: timeout()
}).

-spec init(map()) -> {ok, #tp{}} | {error, term()}.
init(#{model_path := Path, tp_worker := Worker} = Config) when is_pid(Worker) ->
    case erllama_nif:load_model(Path, #{vocab_only => true}) of
        {ok, Vocab} ->
            S = #tp{
                vocab = Vocab,
                worker = Worker,
                layout = layout(Config),
                timeout = maps:get(tp_call_timeout, Config, 120000)
            },
            Hosted = maps:without([tp_worker, tp_followers, tp_call_timeout], Config),
            case call(S, {hosting, {host, Hosted, maps:get(tp_followers, Config, [])}}, infinity) of
                ok ->
                    {ok, S};
                {error, _} = E ->
                    _ = erllama_nif:free_model(Vocab),
                    E
            end;
        {error, _} = E ->
            E
    end;
init(_Config) ->
    {error, {badarg, tp_worker}}.

-spec terminate(#tp{}) -> ok.
terminate(#tp{vocab = Vocab, worker = W}) ->
    %% Stopping rank 0 frees the model on every rank: the others stop
    %% at the gap or with their group.
    _ = catch gen_server:stop(W, normal, 30000),
    _ = erllama_nif:free_model(Vocab),
    ok.

-spec shared_state(#tp{}) -> #tp{}.
shared_state(#tp{vocab = V}) -> #tp{vocab = V}.

tokenize(#tp{vocab = M}, Text) ->
    erllama_nif:tokenize(M, Text, #{add_special => true, parse_special => false}).

tokenize(#tp{vocab = M}, Text, Opts) when is_map(Opts) ->
    erllama_nif:tokenize(M, Text, Opts).

detokenize(#tp{vocab = M}, Tokens) ->
    erllama_nif:detokenize(M, Tokens).

detokenize(#tp{vocab = M}, Tokens, Opts) when is_map(Opts) ->
    erllama_nif:detokenize(M, Tokens, Opts).

vocab_info(#tp{vocab = M}) ->
    case erllama_nif:vocab_info(M) of
        Map when is_map(Map) -> {ok, Map};
        {error, _} = E -> E
    end.

-spec get_model_ref(#tp{}) -> erllama_nif:model_ref().
get_model_ref(#tp{vocab = M}) -> M.

apply_chat_template(#tp{vocab = M}, Request) ->
    erllama_nif:apply_chat_template(M, Request).

prefill(S, Tokens) -> remote(S, prefill, [Tokens]).
decode_one(S, ContextTokens) -> remote(S, decode_one, [ContextTokens]).
%% A checkpoint holds every rank's share, packed at the same token
%% (erllama_tp_rank:kv_pack/2); one share missing fails the save, so
%% nothing is published.
kv_pack(S, Tokens) -> kv_pack(S, Tokens, 0).
kv_pack(#tp{timeout = T} = S, _Tokens, SeqId) -> call(S, {hosting, {kv_pack, SeqId}}, T).
kv_unpack(S, Bin) -> kv_unpack(S, Bin, 0).
kv_unpack(#tp{timeout = T} = S, Bin, SeqId) -> call(S, {hosting, {kv_unpack, Bin, SeqId}}, T).
seq_clear(S) -> remote(S, seq_clear, []).
seq_rm(S, SeqId) -> remote(S, seq_rm, [SeqId]).
seq_cp(S, Src, Dst) -> remote(S, seq_cp, [Src, Dst]).
seq_rm_last(S, N) -> remote(S, seq_rm_last, [N]).
seq_rm_last(S, SeqId, N) -> remote(S, seq_rm_last, [SeqId, N]).
step(S, Ops) -> remote(S, step, [Ops]).
sampler_new(S, Cfg) -> remote(S, sampler_new, [Cfg]).
extra_metadata(#tp{layout = L} = S) ->
    case remote(S, extra_metadata, []) of
        #{kv_format := F} = Meta -> Meta#{kv_format => kv_format(F, L)};
        Other -> Other
    end.

%% A checkpoint of the split model is laid out by the backend's format
%% and the group's layout. `tp2': checkpoints saved before the split
%% backend read mirrored tensors from the rank's own device hold zeros
%% for them on every rank but the first, so they must not be found.
kv_format(F, Layout) -> <<F/binary, "/tp2=", Layout/binary>>.

sampler_free({erllama_tp_ref, Worker, Id}) ->
    case call(#tp{worker = Worker}, {hosting, {free_ref, Id}}, 30000) of
        {error, {tp_rank_lost, _}} -> ok;
        R -> R
    end.

set_grammar(S, Grammar) -> remote_state(S, set_grammar, [Grammar]).
configure_sampler(S, Cfg) -> remote_state(S, configure_sampler, [Cfg]).
clear_sampler(S) -> remote_state(S, clear_sampler, []).
reset_context(S) -> remote_state(S, reset_context, []).

thinking_signature(_S, SeqId, Bytes) ->
    erllama_model_llama:thinking_signature(undefined, SeqId, Bytes).

%% Speculative decoding, MTP only (experimental): rank 0 drafts with the
%% draft model's MTP head (`draft_model', loaded on rank 0's GPU) and
%% verifies on the split model, so every round is one batch on every
%% rank. A hybrid model needs `n_rs_seq' >= the draft length on every
%% rank to take a rejected draft back.
spec_supported(S) ->
    case remote(S, spec_supported, []) of
        true -> true;
        _ -> false
    end.
spec_new(S, #{type := mtp} = Cfg) -> remote(S, spec_new, [Cfg]);
spec_new(_S, _Cfg) -> {error, not_supported}.
spec_begin(S, Ref, SeqId, Prompt) -> remote(S, spec_begin, [Ref, SeqId, Prompt]).
spec_draft(S, Ref, SeqId, IdLast, Delta) -> remote(S, spec_draft, [Ref, SeqId, IdLast, Delta]).
spec_accept(S, Ref, SeqId, NAcc) -> remote(S, spec_accept, [Ref, SeqId, NAcc]).
%% The drafter goes with its handle on rank 0.
spec_free(_S, {mtp, Handle}) -> sampler_free(Handle);
spec_free(S, Ref) -> remote(S, spec_free, [Ref]).
spec_step(S, SamplerRef, SeqId, Draft) -> remote(S, spec_step, [SamplerRef, SeqId, Draft]).
mtp_step_batch(S, Mtp, Items, MaxDraft) -> remote(S, mtp_step_batch, [Mtp, Items, MaxDraft]).
spec_rollback(S) ->
    case remote(S, spec_rollback, []) of
        N when is_integer(N); N =:= infinity -> N;
        _ -> 0
    end.

%% -----------------------------------------------------------------------------

%% The ranks and the share of each split tensor each holds.
layout(Config) ->
    World = 1 + length(maps:get(tp_followers, Config, [])),
    Split = maps:get(tensor_split, maps:get(model_opts, Config, #{}), even),
    iolist_to_binary(io_lib:format("~b:~w", [World, Split])).

remote(#tp{worker = undefined}, _Fn, _Args) ->
    {error, not_supported};
remote(#tp{timeout = T} = S, Fn, Args) ->
    call(S, {hosting, {backend, Fn, Args}}, T).

%% Rank 0 keeps the new backend state; this one does not change.
remote_state(S, Fn, Args) ->
    case remote(S, Fn, Args) of
        ok -> {ok, S};
        {error, _} = E -> E
    end.

call(#tp{worker = W}, Msg, Timeout) ->
    try
        gen_server:call(W, Msg, Timeout)
    catch
        exit:{timeout, _} ->
            ok = erllama_tp_rank:kill(W),
            {error, {tp_rank_lost, timeout}};
        exit:{Why, _} ->
            {error, {tp_rank_lost, Why}}
    end.
