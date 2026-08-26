%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_model_llama).
-moduledoc false.
%% Real-llama.cpp backend for `erllama_model`.
%%
%% Owns a `model_ref` and a `context_ref` from `erllama_nif`. The
%% gen_statem hands its decode/kv operations through this module;
%% this module forwards to the NIF.
%%
%% Config (passed through `erllama_model:start_link/2`):
%%   model_path :: file:name() | binary()  (required)
%%   model_opts :: map()  (forwarded to erllama_nif:load_model/2)
%%   context_opts :: map()  (forwarded to erllama_nif:new_context/2)
%%
%% `model_opts` and `context_opts` flow through to the NIF unchanged,
%% with one exception: on recurrent / hybrid models `n_rs_seq`
%% defaults to 1 (see family_context_opts/2). See
%% `erllama_nif:load_model/2` and `erllama_nif:new_context/2` for the
%% full set of recognised keys, including the llama.cpp option
%% passthroughs `split_mode`, `main_gpu`, `tensor_split`,
%% `flash_attn`, `type_k`, and `type_v`.
-behaviour(erllama_model_backend).

-export([
    init/1,
    terminate/1,
    tokenize/2,
    tokenize/3,
    detokenize/2,
    detokenize/3,
    vocab_info/1,
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
    spec_supported/1,
    spec_new/2,
    spec_begin/4,
    spec_draft/5,
    spec_accept/4,
    spec_free/2,
    spec_step/4,
    sampler_new/2,
    sampler_free/1,
    apply_chat_template/2,
    embed/2,
    set_grammar/2,
    configure_sampler/2,
    clear_sampler/1,
    load_adapter/2,
    unload_adapter/2,
    apply_adapters/2,
    extra_metadata/1,
    verify/4,
    thinking_signature/3,
    reset_context/1,
    abort_handle/1,
    get_model_ref/1
]).

-record(s, {
    model :: erllama_nif:model_ref(),
    ctx :: erllama_nif:context_ref(),
    %% Context options passed to erllama_nif:new_context/2. Retained
    %% so reset_context/1 can rebuild the context in place after a
    %% wedged/aborted decode without reloading the model.
    context_opts = #{} :: map(),
    %% Captured once at init for vram_estimate_b derivation. The
    %% values are immutable once the model is loaded; the gen_statem
    %% reads them via extra_metadata/1 at its own init time and
    %% caches the derived estimate in #data.
    model_size_bytes = 0 :: non_neg_integer(),
    total_layers = 0 :: non_neg_integer(),
    %% Signed: llama.cpp uses negative (typically -1) to mean
    %% "offload all layers".
    n_gpu_layers = 0 :: integer(),
    %% Token ids that open / close a thinking block. Populated at
    %% init from the optional `thinking_markers` config key by
    %% tokenising the start / end strings through this model's own
    %% vocabulary. Empty lists disable marker recognition entirely;
    %% the backend then behaves identically to a non-thinking
    %% backend regardless of the per-request thinking flag.
    thinking_start_ids = [] :: [erllama_nif:token_id()],
    thinking_end_ids = [] :: [erllama_nif:token_id()],
    %% Family / GGUF metadata map probed once at init via
    %% erllama_nif:model_family/1. Drives the load-time policy
    %% (unsupported archs, n_rs_seq injection) and is surfaced
    %% through extra_metadata/1 into model_info/1.
    family = #{} :: map(),
    %% Auto-fit result from the load NIF (`undefined` when the fit
    %% option was off): #{fit => ok | failed, n_gpu_layers, n_ctx,
    %% tensor_split => [float()]}. Surfaced via extra_metadata/1.
    fit_info = undefined :: map() | undefined
}).

init(Config) ->
    Path = maps:get(model_path, Config),
    MOpts0 = progress_opts(maps:get(model_opts, Config, #{}), Config),
    MOpts = fit_opts(MOpts0, Config),
    case erllama_nif:load_model(Path, MOpts) of
        {ok, Model} ->
            post_load(Model, Config, MOpts, undefined);
        {ok, Model, FitInfo} ->
            post_load(Model, Config, MOpts, FitInfo);
        {error, _} = E ->
            E
    end.

post_load(Model, Config, MOpts, FitInfo) ->
    Family = probe_family(Model),
    case check_family(Family) of
        ok ->
            open_context(Model, Config, MOpts, Family, FitInfo);
        {error, _} = E ->
            erllama_nif:free_model(Model),
            E
    end.

%% Normalize the `fit` model option for the NIF: `true` becomes an
%% empty map, `margin_mib` (single or list, MiB) becomes the
%% `margins_mib` list the NIF broadcasts, `n_ctx => auto` becomes
%% the n_ctx_auto flag, and the planned context options ship as
%% `fit_context` so the fit measures the context the model layer
%% will actually open.
fit_opts(MOpts, Config) ->
    case maps:find(fit, MOpts) of
        error ->
            MOpts;
        {ok, FitOpt} ->
            Fit0 =
                case FitOpt of
                    true -> #{};
                    M when is_map(M) -> M
                end,
            Fit1 =
                case maps:find(margin_mib, Fit0) of
                    error -> #{};
                    {ok, N} when is_integer(N) -> #{margins_mib => [N]};
                    {ok, L} when is_list(L) -> #{margins_mib => L}
                end,
            Fit2 =
                case maps:find(min_ctx, Fit0) of
                    error -> Fit1;
                    {ok, MC} -> Fit1#{min_ctx => MC}
                end,
            Fit3 =
                case maps:get(n_ctx, Fit0, undefined) of
                    auto -> Fit2#{n_ctx_auto => true};
                    _ -> Fit2
                end,
            MOpts#{
                fit := Fit3,
                fit_context => maps:get(context_opts, Config, #{})
            }
    end.

%% Wire the optional `progress_to` load option into the NIF opts:
%% the NIF sends `{erllama_load_progress, Tag, Float}` messages to
%% the pid, tagged with the model id the gen_statem injected.
progress_opts(MOpts, Config) ->
    case maps:find(progress_to, Config) of
        {ok, Pid} when is_pid(Pid) ->
            MOpts#{
                progress_to => Pid,
                progress_tag => maps:get(model_id, Config, <<>>)
            };
        _ ->
            MOpts
    end.

probe_family(Model) ->
    case erllama_nif:model_family(Model) of
        Family when is_map(Family) -> Family;
        {error, _} -> #{}
    end.

%% Load-time family policy. Encoder-bearing archs (T5 and friends)
%% and diffusion archs need llama_encode / diffusion stepping, which
%% the engine does not drive; reject at load instead of failing
%% confusingly at the first decode.
check_family(#{has_encoder := true}) ->
    {error, {unsupported_model, encoder_decoder}};
check_family(#{diffusion := true}) ->
    {error, {unsupported_model, diffusion}};
check_family(_) ->
    ok.

open_context(Model, Config, MOpts, Family, FitInfo) ->
    COpts0 = family_context_opts(maps:get(context_opts, Config, #{}), Family),
    COpts = fit_context_opts(COpts0, MOpts, FitInfo),
    case erllama_nif:new_context(Model, COpts) of
        {ok, Ctx} ->
            {ok, build_state(Model, Ctx, Config, MOpts, COpts, Family, FitInfo)};
        {error, _} = E ->
            erllama_nif:free_model(Model),
            E
    end.

%% With `fit => #{n_ctx => auto}` and a successful fit, the chosen
%% context size replaces whatever context_opts carried.
fit_context_opts(COpts, MOpts, FitInfo) when is_map(FitInfo) ->
    FitMap = maps:get(fit, MOpts, #{}),
    Auto = is_map(FitMap) andalso maps:get(n_ctx_auto, FitMap, false),
    case Auto andalso maps:get(fit, FitInfo) =:= ok of
        true -> COpts#{n_ctx => maps:get(n_ctx, FitInfo)};
        false -> COpts
    end;
fit_context_opts(COpts, _MOpts, undefined) ->
    COpts.

%% Recurrent / hybrid models keep compressed state, not per-token KV
%% cells, so the warm-restore primer's 1-token seq_rm only succeeds
%% when the arch supports recurrent-state rollback AND the context
%% keeps at least one rollback snapshot. Default n_rs_seq to 1 for
%% those families (an explicit user value wins); dense models are
%% untouched.
family_context_opts(COpts, Family) ->
    Recurrent = maps:get(recurrent, Family, false),
    Hybrid = maps:get(hybrid, Family, false),
    case (Recurrent orelse Hybrid) andalso not maps:is_key(n_rs_seq, COpts) of
        true -> COpts#{n_rs_seq => 1};
        false -> COpts
    end.

build_state(Model, Ctx, Config, MOpts, COpts, Family, FitInfo) ->
    ThinkingMarkers = maps:get(thinking_markers, Config, #{}),
    {ThinkStart, ThinkEnd} = tokenize_markers(Model, ThinkingMarkers),
    #s{
        model = Model,
        ctx = Ctx,
        context_opts = COpts,
        model_size_bytes = safe_uint(erllama_nif:model_size(Model)),
        total_layers = safe_uint(erllama_nif:model_n_layer(Model)),
        n_gpu_layers = effective_n_gpu_layers(MOpts, FitInfo),
        thinking_start_ids = ThinkStart,
        thinking_end_ids = ThinkEnd,
        family = Family,
        fit_info = FitInfo
    }.

%% The value the VRAM estimate runs on: the fitted layer count when
%% fit ran, otherwise the requested option. -1 (the llama default)
%% means all layers.
effective_n_gpu_layers(MOpts, undefined) ->
    maps:get(n_gpu_layers, MOpts, -1);
effective_n_gpu_layers(_MOpts, FitInfo) ->
    maps:get(n_gpu_layers, FitInfo, -1).

safe_uint(N) when is_integer(N), N >= 0 -> N;
safe_uint(_) -> 0.

%% Resolve the configured `thinking_markers => #{start => Binary,
%% end => Binary}` map into two token-id lists by running each
%% string through the model's own vocabulary. Missing or invalid
%% entries yield an empty list, which keeps the step wrapper a
%% no-op for that side.
tokenize_markers(_Model, Markers) when map_size(Markers) =:= 0 ->
    {[], []};
tokenize_markers(Model, Markers) when is_map(Markers) ->
    Start = tokenize_marker(Model, maps:get(start, Markers, undefined)),
    End = tokenize_marker(Model, maps:get('end', Markers, undefined)),
    {Start, End};
tokenize_markers(_Model, _Other) ->
    {[], []}.

tokenize_marker(_Model, undefined) ->
    [];
tokenize_marker(_Model, <<>>) ->
    [];
tokenize_marker(Model, Bin) when is_binary(Bin) ->
    case erllama_nif:tokenize(Model, Bin, #{add_special => false, parse_special => true}) of
        Tokens when is_list(Tokens) -> Tokens;
        _ -> []
    end.

terminate(#s{ctx = Ctx, model = Model}) ->
    erllama_nif:free_context(Ctx),
    erllama_nif:free_model(Model),
    ok.

%% Recreate the context in place after a wedged/aborted decode. The
%% model stays loaded; all live KV is dropped. On a new_context
%% failure the old context is already gone, so we surface the error
%% and let the gen_statem stop (supervisor reload as a last resort).
reset_context(#s{ctx = OldCtx, model = Model, context_opts = COpts} = S) ->
    _ = erllama_nif:free_context(OldCtx),
    case erllama_nif:new_context(Model, COpts) of
        {ok, NewCtx} -> {ok, S#s{ctx = NewCtx}};
        {error, _} = E -> E
    end.

abort_handle(#s{ctx = Ctx}) ->
    {ok, Ctx}.

tokenize(#s{model = M}, Text) ->
    erllama_nif:tokenize(M, Text, #{add_special => true, parse_special => false}).

tokenize(#s{model = M}, Text, Opts) when is_map(Opts) ->
    erllama_nif:tokenize(M, Text, Opts).

detokenize(#s{model = M}, Tokens) ->
    erllama_nif:detokenize(M, Tokens).

%% Options-aware detokenize (`remove_special` / `unparse_special`),
%% backed by llama_detokenize. Distinct from the arity-2 path, whose
%% output the cache byte-keys are computed over.
detokenize(#s{model = M}, Tokens, Opts) when is_map(Opts) ->
    erllama_nif:detokenize(M, Tokens, Opts).

%% Special / FIM vocab tokens for erllama:vocab_info/1.
vocab_info(#s{model = M}) ->
    case erllama_nif:vocab_info(M) of
        Map when is_map(Map) -> {ok, Map};
        {error, _} = E -> E
    end.

prefill(#s{ctx = C}, Tokens) ->
    erllama_nif:prefill(C, Tokens).

decode_one(#s{ctx = C}, _ContextTokens) ->
    erllama_nif:decode_one(C).

kv_pack(#s{ctx = C}, Tokens) ->
    erllama_nif:kv_pack(C, Tokens, length(Tokens)).

%% Seq-aware kv_pack. The 2-arity above keeps the v0.1 contract
%% (seq_id=0); the 3-arity is driven by the multi-sequence scheduler.
kv_pack(#s{ctx = C}, Tokens, SeqId) when is_integer(SeqId), SeqId >= 0 ->
    erllama_nif:kv_pack(C, Tokens, length(Tokens), SeqId).

kv_unpack(#s{ctx = C}, Bin) ->
    erllama_nif:kv_unpack(C, Bin, 0).

kv_unpack(#s{ctx = C}, Bin, SeqId) when is_integer(SeqId), SeqId >= 0 ->
    erllama_nif:kv_unpack(C, Bin, SeqId).

%% Drop the cell at position N-1 from seq 0 so the model layer can
%% re-prefill the corresponding token and regenerate logits.
%% `llama_state_seq_*` only persists KV cells, never the per-context
%% logits buffer; without this primer the next sample would read stale
%% (or zero) logits.
seq_rm_last(#s{ctx = C}, NTokens) when NTokens > 0 ->
    erllama_nif:kv_seq_rm(C, 0, NTokens - 1, -1).

%% Seq-aware variant of seq_rm_last/2.
seq_rm_last(#s{ctx = C}, SeqId, NTokens) when
    is_integer(SeqId), SeqId >= 0, NTokens > 0
->
    erllama_nif:kv_seq_rm(C, SeqId, NTokens - 1, -1).

%% Full-sequence KV copy (session fork). See erllama_nif:kv_seq_cp/3.
seq_cp(#s{ctx = C}, SrcSeq, DstSeq) when
    is_integer(SrcSeq), SrcSeq >= 0, is_integer(DstSeq), DstSeq >= 0
->
    erllama_nif:kv_seq_cp(C, SrcSeq, DstSeq).

%% Free all KV cells of a specific seq_id. Used by the scheduler when
%% a request finishes and its seq_id returns to the idle pool.
seq_rm(#s{ctx = C}, SeqId) when is_integer(SeqId), SeqId >= 0 ->
    erllama_nif:kv_seq_rm(C, SeqId, 0, -1).

%% Wipe seq 0 entirely. p0=0, p1=-1 means "from position 0 to infinity".
seq_clear(#s{ctx = C}) ->
    erllama_nif:kv_seq_rm(C, 0, 0, -1).

%% Drive one batched-decode tick. Forwards to the NIF and, when
%% thinking or tool-call markers are configured, maps any sampled
%% token that matches one of them into the corresponding step-result
%% variant. With every marker list empty the mapping is the
%% identity and the return shape is exactly what the NIF produced.
%% Thinking checks come first so a token id that is in both sets
%% routes to thinking.
step(#s{ctx = C} = S, Ops) ->
    case erllama_nif:step(C, Ops) of
        {ok, Results} ->
            case all_markers_empty(S) of
                true -> {ok, Results};
                false -> {ok, [map_marker(R, S) || R <- Results]}
            end;
        Other ->
            Other
    end.

all_markers_empty(#s{thinking_start_ids = [], thinking_end_ids = []}) ->
    true;
all_markers_empty(_) ->
    false.

map_marker({SeqId, {token, Tok, _Eog}} = R, #s{
    thinking_start_ids = TSI,
    thinking_end_ids = TEI
}) ->
    case {lists:member(Tok, TSI), lists:member(Tok, TEI)} of
        {true, _} -> {SeqId, {thinking_token, Tok}};
        {_, true} -> {SeqId, thinking_end};
        _ -> R
    end;
map_marker(R, _S) ->
    R.

%% Ngram speculation: thin forwards to the NIF shim. Speculation is
%% vetoed while thinking markers are configured because spec_step
%% bypasses the map_marker routing above.
spec_supported(#s{} = S) ->
    all_markers_empty(S).

spec_new(#s{}, Cfg) ->
    erllama_nif:spec_new(Cfg).

spec_begin(#s{}, SpecRef, SeqId, PromptTokens) ->
    erllama_nif:spec_begin(SpecRef, SeqId, PromptTokens).

spec_draft(#s{}, SpecRef, SeqId, IdLast, Delta) ->
    erllama_nif:spec_draft(SpecRef, SeqId, IdLast, Delta).

spec_accept(#s{}, SpecRef, SeqId, NAcc) ->
    erllama_nif:spec_accept(SpecRef, SeqId, NAcc).

spec_free(#s{}, SpecRef) ->
    erllama_nif:spec_free(SpecRef).

spec_step(#s{ctx = C}, SamplerRef, SeqId, Draft) ->
    erllama_nif:spec_step(C, SamplerRef, SeqId, Draft).

%% Surface the underlying NIF model resource for callers that need to
%% hand it to `erllama_chat:init/2' (the autoparser
%% templates init expects a model_ref). The scheduler holds the
%% backend state; the chat-API path goes through a model gen_statem
%% call (`erllama_model:chat_apply/2') so the resource
%% lifetime is tied to the model the same way it is for other ops.
-spec get_model_ref(#s{}) -> erllama_nif:model_ref().
get_model_ref(#s{model = Model}) -> Model.

%% Build a per-request sampler chain. The opaque sampler_ref is held
%% by the scheduler for the request's lifetime and freed when the
%% request finishes.
sampler_new(#s{ctx = C}, Cfg) when is_map(Cfg) ->
    erllama_nif:sampler_new(C, Cfg).

sampler_free(SamplerRef) ->
    erllama_nif:sampler_free(SamplerRef).

apply_chat_template(#s{model = M}, Request) ->
    erllama_nif:apply_chat_template(M, Request).

embed(#s{ctx = C}, Tokens) ->
    erllama_nif:embed(C, Tokens).

set_grammar(#s{ctx = C} = S, Grammar) when is_binary(Grammar) ->
    ok_state(S, erllama_nif:set_grammar(C, Grammar));
set_grammar(#s{} = S, undefined) ->
    {ok, S}.

configure_sampler(#s{} = S, Cfg) when map_size(Cfg) =:= 0 ->
    %% No sampler params at all - leave the existing chain alone so
    %% the lazy greedy fallback in the NIF kicks in on first decode.
    {ok, S};
configure_sampler(#s{ctx = C} = S, Cfg) when is_map(Cfg) ->
    ok_state(S, erllama_nif:configure_sampler(C, Cfg)).

clear_sampler(#s{ctx = C} = S) ->
    ok = erllama_nif:clear_sampler(C),
    {ok, S}.

load_adapter(#s{model = M} = S, Path) ->
    case erllama_nif:adapter_load(M, Path) of
        {ok, AdapterRef} -> {ok, AdapterRef, S};
        {error, _} = E -> E
    end.

unload_adapter(#s{} = S, AdapterRef) ->
    case erllama_nif:adapter_free(AdapterRef) of
        %% Double-free is a no-op: the adapter is already gone.
        {error, released} -> {ok, S};
        Result -> ok_state(S, Result)
    end.

apply_adapters(#s{ctx = C} = S, Adapters) ->
    ok_state(S, erllama_nif:set_adapters(C, Adapters)).

%% Wrap a NIF's bare ok / {error, _} return into the backend's
%% {ok, State} shape.
-spec ok_state(#s{}, ok | {error, term()}) -> {ok, #s{}} | {error, term()}.
ok_state(S, ok) -> {ok, S};
ok_state(_S, {error, _} = E) -> E.

extra_metadata(#s{
    model_size_bytes = SB,
    total_layers = TL,
    n_gpu_layers = NL,
    family = Family,
    fit_info = FitInfo
}) ->
    Meta = #{
        model_size_bytes => SB,
        total_layers => TL,
        n_gpu_layers => NL,
        family => Family
    },
    case FitInfo of
        undefined -> Meta;
        _ -> Meta#{fit => FitInfo}
    end.

%% Speculative-decoding verifier. Snapshot+restore protocol so
%% the caller's KV view is unchanged after the call. Empty prefix
%% is rejected: the acceptance / NextToken indexing both require
%% at least one prefix token.
verify(_S, [], _Candidates, _K) ->
    {error, empty_prefix};
verify(#s{ctx = Ctx} = S, PrefixTokens, Candidates, K) when
    is_list(PrefixTokens), is_list(Candidates), is_integer(K), K > 0
->
    KCap = min(K, length(Candidates)),
    Truncated = lists:sublist(Candidates, KCap),
    Input = PrefixTokens ++ Truncated,
    case erllama_nif:forward_with_argmax(Ctx, Input) of
        {ok, Argmax} ->
            P = length(PrefixTokens),
            Accepted = count_accepted(0, P, Truncated, Argmax),
            %% NextToken comes from logits at 0-indexed position
            %% P - 1 + Accepted (the verifier's prediction given
            %% the prefix and the accepted prefix of candidates).
            %% lists:nth/2 is 1-indexed: P + Accepted.
            NextToken = lists:nth(P + Accepted, Argmax),
            ok = restore_after_verify(Ctx, P, lists:last(PrefixTokens)),
            {ok, Accepted, NextToken, S};
        {error, _} = E ->
            E
    end.

count_accepted(Acc, _P, [], _Argmax) ->
    Acc;
count_accepted(Acc, P, [C | Rest], Argmax) ->
    Predicted = lists:nth(P + Acc, Argmax),
    case Predicted of
        C when is_integer(C) -> count_accepted(Acc + 1, P, Rest, Argmax);
        _ -> Acc
    end.

%% Bring the context's KV cells back to the caller's pre-call
%% length P, then re-prefill the last prefix token so the logits
%% buffer is in a sampleable state for any follow-up decode_one.
%% The caller's pre-call decode_ready flag is not preserved: after
%% verify the context is always ready to sample at the prefix end.
%% Documented in the public erllama:verify/4 doc.
restore_after_verify(Ctx, P, LastPrefixToken) ->
    %% kv_seq_rm with p1 = -1 means "to infinity"; combined with
    %% p0 = P, that drops every cell from the candidate batch.
    _ = erllama_nif:kv_seq_rm(Ctx, 0, P, -1),
    _ = erllama_nif:prefill(Ctx, [LastPrefixToken]),
    ok.

%% Sign the observed thinking-phase bytes with an HMAC-SHA256 over
%% the node-wide signing key. The Anthropic SDK round-trips the
%% resulting binary as `signature_delta` so the server can verify
%% the thinking text on the next turn. The key is read from the
%% application environment:
%%
%%     application:set_env(erllama, thinking_signing_key, <<"...">>).
%%
%% Leaving the key unset returns `<<>>`, the documented
%% "no signature available" path: the downstream omits
%% `signature_delta` from its SSE output.
thinking_signature(_S, _SeqId, Bytes) when is_binary(Bytes) ->
    case application:get_env(erllama, thinking_signing_key) of
        {ok, Key} when is_binary(Key), Key =/= <<>> ->
            crypto:mac(hmac, sha256, Key, Bytes);
        _ ->
            <<>>
    end.
