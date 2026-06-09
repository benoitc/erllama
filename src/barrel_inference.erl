%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(barrel_inference).
-moduledoc """
Public façade for the barrel_inference application.

The cache subsystem (`barrel_inference_cache`) is independent. This module
is the user-facing surface for loading and running models.

Typical usage:

```
  ok = application:ensure_all_started(barrel_inference).
  {ok, Bin} = file:read_file("/srv/models/tinyllama-1.1b-q4_k_m.gguf").
  {ok, Model} = barrel_inference:load_model(#{
      backend => barrel_inference_model_llama,
      model_path => "/srv/models/tinyllama-1.1b-q4_k_m.gguf",
      fingerprint => crypto:hash(sha256, Bin)
  }).
  {ok, #{reply := Reply, finish_key := FK}} =
      barrel_inference:complete(Model, <<"hello">>).
  %% On the next turn, pass FK as parent_key for byte-exact warm
  %% restore:
  {ok, #{reply := Reply2}} =
      barrel_inference:complete(Model, <<"hello world">>, #{parent_key => FK}).
  ok = barrel_inference:unload(Model).
```

Extra cache parameters (`tier`, `tier_srv`, `quant_type`,
`ctx_params_hash`, `policy`, ...) are optional; the defaults route
saves to the RAM tier (`barrel_inference_cache_ram`). See the loading guide
for the full option map and instructions to wire up
`ram_file` / `disk` tier servers.

Models are dynamic children of `barrel_inference_model_sup` (simple_one_for_one).
A registered name is auto-generated when the caller does not provide
an explicit `model_id` in the config map.
""".

-export([
    load_model/1,
    load_model/2,
    unload/1,
    unload_model/1,
    complete/2,
    complete/3,
    prefill_only/2,
    prefill_only/3,
    infer/4,
    continue/3,
    cancel/1,
    end_session/2,
    reset_session/2,
    status/1,
    evict/1,
    shutdown/1,
    models/0,
    list_models/0,
    model_info/1,
    resident_bytes/1,
    tokenize/2,
    tokenize/3,
    detokenize/2,
    apply_chat_template/2,
    chat_apply/2,
    chat_parse/3,
    embed/2,
    load_adapter/2,
    unload_adapter/2,
    set_adapter_scale/3,
    list_adapters/1,
    counters/0,
    vram_info/0,
    queue_depth/0,
    queue_depth/1,
    pending_len/1,
    phase/1,
    last_cache_hit/1,
    list_cached_prefixes/2,
    draft_tokens/3,
    verify/4
]).

-export_type([model/0, model_id/0, model_info/0]).

-type model_id() :: barrel_inference_registry:model_id().
-type model() :: barrel_inference_model:model().
-type model_info() :: barrel_inference_model:model_info().

%% =============================================================================
%% Public API
%% =============================================================================

-doc "Load a model with an auto-generated id.".
-spec load_model(map()) -> {ok, model_id()} | {error, term()}.
load_model(Config) when is_map(Config) ->
    load_model(default_id(), Config).

-doc "Load a model with an explicit id.".
-spec load_model(model_id(), map()) -> {ok, model_id()} | {error, term()}.
load_model(ModelId, Config) when is_binary(ModelId), is_map(Config) ->
    case barrel_inference_model_sup:start_model(ModelId, Config) of
        {ok, _Pid} -> {ok, ModelId};
        {error, {already_started, _}} -> {error, already_loaded};
        {error, _} = E -> E
    end.

-doc "Unload a model. Terminates the gen_statem cleanly.".
-spec unload(model()) -> ok | {error, term()}.
unload(Model) ->
    barrel_inference_model_sup:stop_model(Model).

-doc """
Alias for `unload/1`. Provided for API symmetry with `load_model/1,2`
and the OpenAI/Ollama-style naming used by downstream HTTP servers.
""".
-spec unload_model(model()) -> ok | {error, term()}.
unload_model(Model) ->
    unload(Model).

-doc """
Run a completion against a loaded model.

Returns `{ok, Result}` where `Result` is an `barrel_inference_model:completion_result()`
map carrying the detokenised reply, the generated token list, the
full context tokens, the cache `finish_key` to use as
`parent_key` on the next turn, and per-request stats.
""".
-spec complete(model(), binary()) ->
    {ok, barrel_inference_model:completion_result()} | {error, term()}.
complete(Model, Prompt) ->
    barrel_inference_model:complete(Model, Prompt).

-doc """
Run a completion against a loaded model with options.

Recognised keys in `Opts`:

- `response_tokens` (`non_neg_integer()`) — cap on the number of
  tokens generated. Defaults to the model's `n_ctx` minus prompt
  length.
- `parent_key` (`barrel_inference_cache:cache_key()`) — the previous turn's
  `finish_key`. Skips the longest-prefix walk and resumes directly
  from that row.
- `stop_sequences` (`[binary()]`) — caller-supplied stop strings.
  Generation halts on the first occurrence (by list order) of any
  element in the accumulated detokenised output; the matched string
  is trimmed from `reply` and reported as `stop_sequence`.
- `on_full` (`block | error`, default `block`) — admission behaviour
  when no seq_id is free. `block` queues the request until a slot
  frees; `error` fails fast with `{error, seq_capacity}`. Pair with
  `available_seqs` / `n_seq_max` from `model_info/1` to size capacity
  and avoid an unbounded wait when all seqs are pinned.

Returns `{ok, Result}` where `Result` is a `completion_result()` map
carrying:

- `reply` — detokenised reply text (trimmed at the matched stop
  string when one fired)
- `generated` — tokens produced by this request
- `context_tokens` — full token list (prompt ++ generated)
- `committed_tokens` — `length(context_tokens)`
- `finish_key` — cache key for the full context, or `undefined` if
  the finish save was suppressed
- `cache_hit_kind` — `exact | partial | cold | sticky | continuation`
- `finish_reason` — `stop | length | cancelled`
- `cache_delta` — `#{read := N, created := N}`. `read` is the warm
  prefix length restored from cache at admission; `created` is the
  largest contribution this request added to the cache beyond that
  prefix (0 when no save fired)
- `stop_sequence` — only present when a `stop_sequences` entry
  fired; the binary of the matched stop string
- `stats` — per-request timing and cache stats (also carries
  `cache_delta` for streaming callers)
""".
-spec complete(model(), binary(), map()) ->
    {ok, barrel_inference_model:completion_result()} | {error, term()}.
complete(Model, Prompt, Opts) ->
    barrel_inference_model:complete(Model, Prompt, Opts).

-doc """
Decode a prompt into KV state and fire a finish save without
sampling any output tokens. Returns the `finish_key` so the caller
can hand it as `parent_key` on a subsequent `complete/3` or
`infer/4` for byte-exact warm restore.

`PromptTokens` is the prompt as a list of token ids. Tokenisation
is the caller's responsibility (use `tokenize/2` or apply a chat
template first).

`finish_key` is `undefined` if the finish save was suppressed
because the token count is below the configured `min_tokens`.
""".
-spec prefill_only(model(), [barrel_inference_nif:token_id()]) ->
    {ok, barrel_inference_model:prefill_result()} | {error, term()}.
prefill_only(Model, PromptTokens) ->
    barrel_inference_model:prefill_only(Model, PromptTokens).

-doc """
Same as `prefill_only/2` but accepts `Opts`. The only recognised
key today is `parent_key`: when set to a prior turn's `finish_key`
and `PromptTokens` extends that prior context, barrel_inference warm-restores
from the prior row and prefills only the new suffix before firing
the finish save. Useful for chaining cache-warming calls across
multiple turns deterministically rather than relying on the
implicit longest-prefix walk.
""".
-spec prefill_only(model(), [barrel_inference_nif:token_id()], map()) ->
    {ok, barrel_inference_model:prefill_result()} | {error, term()}.
prefill_only(Model, PromptTokens, Opts) when is_map(Opts) ->
    barrel_inference_model:prefill_only(Model, PromptTokens, Opts).

-doc """
Streaming inference. Returns immediately with a `reference()` that
identifies this request; tokens are delivered to `CallerPid` via
async messages:

- `{barrel_inference_token, Ref, Bin :: binary()}` — text fragment
- `{barrel_inference_token, Ref, {thinking_delta, Bin :: binary()}}` —
  fragment of an extended-thinking block; only emitted when
  `Params` carries `thinking => enabled` and the backend supports
  it
- `{barrel_inference_thinking_end, Ref, Sig :: binary()}` — close marker for
  a thinking block, carrying an opaque integrity signature; emitted
  exactly once per closed block before any subsequent
  `{barrel_inference_token, _, _}` message. `Sig` is `<<>>` when no
  signature is available
- `{barrel_inference_done, Ref, Stats}` — normal completion
- `{barrel_inference_error, Ref, Reason}` — failure

`Tokens` is the prompt as a list of token ids; tokenisation is the
caller's responsibility (use `tokenize/2` or apply a chat template
first).

When `Params` carries `stop_sequences => [binary()]` and one of the
strings appears in the accumulated detokenised output, generation
halts. The match is trimmed from the streamed `{barrel_inference_token, _,
_}` chunks and the matched value is reported as `stop_sequence` in
the final `{barrel_inference_done, _, Stats}`.

`Stats` carries `cache_delta => #{read := N, created := N}` for
Anthropic-style cache accounting: `read` tokens came from the warm
prefix at admission, `created` tokens were added to the cache by
this request.

`Stats` also carries `generated => [token_id()]`, the exact
generated token ids for this turn in order, so a caller can feed a
byte-exact suffix to `continue/3` without re-tokenising detokenised
text.

When `Params` carries a binary `grammar`, it constrains every token
of the turn, including tool-call syntax: the greedy-on-syntax swap
(used for byte-deterministic tool-call markers on the default
no-grammar path) is disabled for the request, so a
`tool_choice = required` / `response_format` grammar is honoured
end to end rather than abandoned once a tool-call span opens.

When `Params` carries `thinking_budget_tokens => N` and the
thinking phase emits `N` or more `{thinking_delta, _}` payloads,
the scheduler synthesises the `{barrel_inference_thinking_end, _, _}` close
early. Any subsequent backend `{thinking_token, _}` is routed
through the normal token pipeline so generation still progresses.

When `Params` carries `session_id => Term`, the underlying seq_id is
pinned to that session across successive requests. The next
request with the same `session_id` truncates-and-prefills in place
on the live KV cells instead of warm-restoring from disk, as long
as its prompt continues the stored tokens. Concurrent requests
on the same `session_id` are out of scope: the second one returns
`{error, sticky_busy}`. Release the session explicitly with
`end_session/2` when the conversation ends.

When `Params` carries `on_full => error`, an admission that finds no
free seq_id returns `{error, seq_capacity}` immediately instead of
queueing. The default `block` preserves the queue-and-wait
behaviour. Use `available_seqs` / `n_seq_max` from `model_info/1` to
decide capacity up front.
""".
-spec infer(
    model(),
    [barrel_inference_nif:token_id()],
    barrel_inference_model:infer_params(),
    pid()
) ->
    {ok, reference()} | {error, term()}.
infer(Model, Tokens, Params, CallerPid) ->
    barrel_inference_model:infer(Model, Tokens, Params, CallerPid).

-doc """
Streaming inference that extends a pinned sticky session by
prefilling `SuffixTokens` directly onto the session's already-live
KV cells, skipping both the chat-template prefix-equality check and
any cache lookup.

Use this when the chat template renders a different leading byte
sequence depending on history length (typical for production chat
templates that wrap the first user message differently in a
1-message vs N-message history). In that situation `infer/4` with
`session_id` would re-tokenise the full prompt, fail the
`is_strict_prefix_or_equal` check, evict the sticky seq, and admit
cold. `continue/3` lets the caller assert "here is the new
tokenised tail" and the engine appends it on top of the resident
KV without re-tokenising the prior turns.

`Opts` must carry:

- `session_id => Term` — identifies a previously pinned session.
- `caller_pid => pid()` — where streaming events are sent.

All other `infer_params()` keys are honoured
(`response_tokens`, `temperature`, `top_p`, `grammar`,
`thinking`, `thinking_budget_tokens`, `stop_sequences`, ...), exactly
as on `infer/4`. A binary `grammar` is applied identically here: it
governs tool-call syntax tokens too (the greedy-on-syntax swap is
disabled for the request), so `tool_choice` / `response_format`
constraints hold on continued rounds.
`parent_key` is ignored on this path because no cache lookup runs.

Contract: the caller asserts that `SuffixTokens` is the exact
tokenised tail to append on top of the session's stored prefix. The
engine performs no prefix equality check. A wrong tail produces
garbage generation but does not corrupt engine state.

To catch a divergent caller mental-model rather than generate
garbage, pass `Opts` key `expect_committed => [token_id()]` — the
tokens the caller believes the session has already committed
(reconstruct them from the prior turn's `Stats.generated`). It must
equal the session's stored context exactly; on divergence the call
returns `{error, {transcript_mismatch, #{stored_len, expected_len,
diverge_at}}}` and prefills nothing, leaving the seq pinned so the
caller can re-sync and retry. Omit it for the historical
"trust the tail" behaviour.

Errors:

- `{error, no_session}` — `session_id` is unknown (no prior turn
  pinned this session, or it was released).
- `{error, sticky_busy}` — the session's seq is currently in flight
  with an earlier request.
- `{error, {transcript_mismatch, Detail}}` — `expect_committed` did
  not match the session's stored committed tokens.

Streaming wire shape is identical to `infer/4`. On success
`Stats.cache_hit_kind` is `continuation`,
`Stats.cache_delta.read` equals the length of the session's stored
tokens before this call, and `Stats.cache_delta.created` equals
`length(SuffixTokens) + completion_tokens`.

See the "Chat-template continuation" example in the guides for the
recommended pattern (render the full prompt, slice off the first
`prior_token_count` tokens, pass the tail).
""".
-spec continue(
    model(),
    [barrel_inference_nif:token_id()],
    map()
) ->
    {ok, reference()}
    | {error, no_session | sticky_busy | {transcript_mismatch, map()} | term()}.
continue(Model, SuffixTokens, Opts) ->
    barrel_inference_model:continue(Model, SuffixTokens, Opts).

-doc """
Cancel an in-flight streaming inference. Idempotent and
fire-and-forget; cancellation is observed at the next inter-token
boundary. The caller still receives a final `{barrel_inference_done, Ref,
Stats}` with `cancelled => true`.

A healthy decode is fast, so the next inter-token boundary is
typically milliseconds away. As a safety net, cancel also asks the
backend to interrupt the in-flight decode directly (via the
context's abort callback), so a stalled or wedged decode is bounded
rather than blocking forever. An interrupt that actually fires
recreates the context in place (the model stays loaded), so under
`n_seq_max > 1` co-batched siblings on other seqs are reset too.
Only the live in-context KV and sticky-session pins are dropped; the
persistent tiered cache survives, so warm restore from cache still
applies on the next admission.
""".
-spec cancel(reference()) -> ok.
cancel(Ref) ->
    barrel_inference_model:cancel(Ref).

-doc """
Release a sticky session: free the pinned seq_id's KV cells and
return the seq to the idle pool so a new conversation can take it.
Idempotent — unknown session ids are a no-op. Call when the user's
conversation ends, on TTL expiry, or any time the live KV is no
longer worth keeping resident.
""".
-spec end_session(model(), term()) -> ok.
end_session(Model, SessionId) ->
    barrel_inference_model:end_session(Model, SessionId).

-doc """
Forcibly drop a session's live KV cells and any in-flight request
row tagged to its seq, then return the seq to the idle pool.

Unlike `end_session/2`, this primitive uses a short timeout and is
reachable when the engine's hot path is wedged (a `gen_statem:call`
with `infinity` would never return). It also force-fails any
in-flight `#req{}` on the session's seq, signalling the caller with
`{barrel_inference_error, Ref, engine_reset}` for streaming callers or
`{error, engine_reset}` to sync callers.

Returns:
- `{ok, recovered}` — session existed, seq was reclaimed.
- `{ok, not_found}` — no such sticky session (TTL evict, never
  registered). The caller can fall through to a plain `infer/4`.
- `{error, timeout}` — the gen_statem mailbox itself is unreachable
  within 5 s. Restart the model.

Only the session's live KV cells and seq pin are dropped; the
persistent tiered cache is untouched, so a later request can still
warm-restore from cache rather than starting fully cold.

For graceful conversation teardown prefer `end_session/2`.
""".
-spec reset_session(model(), term()) ->
    {ok, recovered | not_found} | {error, timeout}.
reset_session(Model, SessionId) ->
    barrel_inference_model:reset_session(Model, SessionId).

-doc """
Current model state. `idle` means no request is in flight;
`prefilling` and `generating` are the two active phases.
""".
-spec status(model()) -> idle | prefilling | generating.
status(Model) ->
    barrel_inference_model:status(Model).

-doc """
Fire an `evict` save synchronously and release the model's live KV
state. Used by an external memory-pressure scheduler when it wants
this model's working set off the heap without unloading the model.
""".
-spec evict(model()) -> ok.
evict(Model) ->
    barrel_inference_model:evict(Model).

-doc """
Fire a `shutdown` save synchronously and return. Called from a
release stop hook; bounded by `evict_save_timeout_ms`.
""".
-spec shutdown(model()) -> ok.
shutdown(Model) ->
    barrel_inference_model:shutdown(Model).

-doc """
List currently-loaded model pids (low-level supervisor view). Most
callers want `list_models/0`, which returns metadata maps.
""".
-spec models() -> [pid()].
models() ->
    [Pid || {_, Pid, _, _} <- barrel_inference_model_sup:models(), is_pid(Pid)].

-doc """
List currently-loaded models as `model_info()` maps. Each entry
includes the model id, status, backend, context size, and
quantisation.
""".
-spec list_models() -> [model_info()].
list_models() ->
    lists:filtermap(
        fun({_ModelId, Pid}) ->
            try
                {true, barrel_inference_model:model_info(Pid)}
            catch
                _:_ -> false
            end
        end,
        barrel_inference_registry:all()
    ).

-doc """
Inspect a single loaded model. Returns the same map shape
`list_models/0` produces. Crashes with `noproc` if the model is not
loaded.
""".
-spec model_info(model()) -> model_info().
model_info(Model) ->
    barrel_inference_model:model_info(Model).

-doc """
Diagnostic resident-set size in bytes for a loaded model. Walks the
backend's mmap regions and counts pages currently faulted in via
`mincore(2)`. Backends without an mmap representation (the stub)
return `0`. Used to drive the Prometheus
`barrel_inference_resident_bytes{model=...}` gauge.
""".
-spec resident_bytes(model()) -> non_neg_integer().
resident_bytes(Model) ->
    barrel_inference_model:resident_bytes(Model).

-doc """
Tokenise text against a loaded model's tokenizer. Safe to call
concurrently with `complete/2,3`.
""".
-spec tokenize(model(), binary()) ->
    {ok, [barrel_inference_nif:token_id()]} | {error, term()}.
tokenize(Model, Text) ->
    barrel_inference_model:tokenize(Model, Text).

-doc """
Tokenise text with explicit options. `parse_special => true' lets
chat-template markers (e.g. `[INST]', `[AVAILABLE_TOOLS]') decode to
single special-token ids; the 2-arity form keeps them as raw bytes
to match user-typed input. The pipeline uses the 3-arity form for
prompt rendered through `chat_apply/2'.
""".
-spec tokenize(model(), binary(), map()) ->
    {ok, [barrel_inference_nif:token_id()]} | {error, term()}.
tokenize(Model, Text, Opts) when is_map(Opts) ->
    barrel_inference_model:tokenize(Model, Text, Opts).

-doc "Detokenise a list of token ids back to text.".
-spec detokenize(model(), [barrel_inference_nif:token_id()]) ->
    {ok, binary()} | {error, term()}.
detokenize(Model, Tokens) ->
    barrel_inference_model:detokenize(Model, Tokens).

-doc """
Render a chat request through the model's chat template and
tokenise. The Request map carries `messages`, `system`, and `tools`.
""".
-spec apply_chat_template(model(), barrel_inference_model_backend:chat_request()) ->
    {ok, [barrel_inference_nif:token_id()]} | {error, term()}.
apply_chat_template(Model, Request) ->
    barrel_inference_model:apply_chat_template(Model, Request).

-doc """
Build a chat_params_ref + rendered prompt for `Model' via llama.cpp's
autoparser. `Inputs' is the map fed to `barrel_inference_chat:apply/2'
(`messages', `tools', `tool_choice', `parallel_tool_calls').
Each call synthesizes fresh params + prompt; only the model-level
`templates_ref' is cached.
""".
-spec chat_apply(model(), map()) ->
    {ok, barrel_inference_nif:chat_params_ref(), binary()} | {error, term()}.
chat_apply(Model, Inputs) ->
    barrel_inference_model:chat_apply(Model, Inputs).

-doc """
Parse a model output string into a structured assistant message via
llama.cpp's autoparser. `IsPartial' = true accepts streaming
prefixes. See `barrel_inference_chat:parse/3' for the returned map
shape.
""".
-spec chat_parse(
    barrel_inference_nif:chat_params_ref(), binary(), boolean()
) ->
    {ok, map()} | {error, term()}.
chat_parse(Params, Input, IsPartial) ->
    barrel_inference_chat:parse(Params, Input, IsPartial).

-doc "Compute an embedding vector for the given prompt tokens.".
-spec embed(model(), [barrel_inference_nif:token_id()]) ->
    {ok, [float()]} | {error, term()}.
embed(Model, Tokens) ->
    barrel_inference_model:embed(Model, Tokens).

-doc """
Load a LoRA adapter from a GGUF file and attach it to the model with
scale 1.0. Returns an opaque handle to pass to `set_adapter_scale/3`
and `unload_adapter/2`.

The adapter's file sha256 is folded into the model's effective
fingerprint so cache rows produced with the adapter attached never
collide with rows from a different attachment set. In-flight
requests keep their original fingerprint snapshot; the new value
takes effect from the next request.
""".
-spec load_adapter(model(), file:filename_all()) ->
    {ok, term()} | {error, term()}.
load_adapter(Model, Path) ->
    barrel_inference_model:load_adapter(Model, Path).

-doc """
Detach and free a previously loaded adapter. Idempotent.
""".
-spec unload_adapter(model(), term()) -> ok | {error, term()}.
unload_adapter(Model, Handle) ->
    barrel_inference_model:unload_adapter(Model, Handle).

-doc """
Change an attached adapter's scale. The scale is folded into the
effective fingerprint, so changes split the cache namespace.
""".
-spec set_adapter_scale(model(), term(), float()) -> ok | {error, term()}.
set_adapter_scale(Model, Handle, Scale) ->
    barrel_inference_model:set_adapter_scale(Model, Handle, Scale).

-doc """
List currently attached adapters with their scales.
""".
-spec list_adapters(model()) -> [#{handle := term(), scale := float()}].
list_adapters(Model) ->
    barrel_inference_model:list_adapters(Model).

-doc "Snapshot of the cache subsystem operational counters.".
-spec counters() -> #{atom() => non_neg_integer()}.
counters() ->
    barrel_inference_cache:get_counters().

-doc """
VRAM probe across all loaded ggml backends. Sums free / total bytes
across non-CPU devices (GPU, integrated GPU, accelerator). Returns
`{error, no_gpu}` on a CPU-only build rather than reporting a fake
number; the caller should fall back to a system memory probe of its
own choosing in that case.

Used by the `barrel_inference_cluster` scheduler for bin-packing model
placement.
""".
-spec vram_info() ->
    {ok, #{
        total_b := non_neg_integer(),
        free_b := non_neg_integer(),
        used_b := non_neg_integer()
    }}
    | {error, atom()}.
vram_info() ->
    barrel_inference_nif:vram_info().

-doc """
O(1) snapshot of currently-admitted streaming inference requests
across all loaded models. Counts only rows registered in
`barrel_inference_inflight` from the `infer/4` admission path; pending
requests queued inside an individual model gen_statem are not
included.

Used by the `barrel_inference_cluster` load balancer (least_loaded,
power_of_two strategies) as a more accurate alternative to
client-side outgoing-request counters.
""".
-spec queue_depth() -> non_neg_integer().
queue_depth() ->
    barrel_inference_inflight:queue_depth().

-doc """
Per-model inflight count. Counts only admitted streaming requests
(`infer/4`); for pending FIFO depth (calls queued inside the model
gen_statem behind an in-flight request) use `pending_len/1`.

Returns 0 if the model is not loaded.
""".
-spec queue_depth(model_id()) -> non_neg_integer().
queue_depth(ModelId) when is_binary(ModelId) ->
    case barrel_inference_registry:whereis_name(ModelId) of
        undefined -> 0;
        Pid -> barrel_inference_inflight:queue_depth(Pid)
    end.

-doc """
Lock-free per-model snapshot of the gen_statem's pending FIFO
length — i.e. how many `complete/2,3`, `prefill_only/2`, and
`infer/4` calls are queued behind whatever the model is currently
running. Returns 0 if the model is idle or not loaded.

Reads a named public ETS row written by the model on every queue
mutation; the call does not cross the model gen_statem, so it
returns instantly even while a decode step is in flight. That
matters: the whole point of asking "is this model busy?" is to
answer without serialising behind the work you are probing.

Used by `barrel_inference_cluster` routers to bin-pack requests onto the
least-loaded node.
""".
-spec pending_len(model_id()) -> non_neg_integer().
pending_len(ModelId) when is_binary(ModelId) ->
    case barrel_inference_inflight:obs_get(ModelId) of
        {_Id, _Phase, PendingLen, _Kind, _PrefixLen} -> PendingLen;
        undefined -> 0
    end.

-doc """
Lock-free per-model phase snapshot. Returns `idle`, `prefilling`,
or `generating`; falls back to `idle` if the model is not loaded.

Like `pending_len/1`, this reads a public ETS row without crossing
the model gen_statem.
""".
-spec phase(model_id()) -> idle | prefilling | generating.
phase(ModelId) when is_binary(ModelId) ->
    case barrel_inference_inflight:obs_get(ModelId) of
        {_Id, Phase, _, _, _} -> Phase;
        undefined -> idle
    end.

-doc """
Lock-free snapshot of the model's most recent cache-hit summary:
the kind (`exact | partial | cold`) and the warm prefix token count.
Returns `undefined` if the model has not admitted any request yet
or is not loaded.

A `cold` kind with `prefix_len = 0` means the previous admission
took the full cold path; an `exact` kind means a byte-exact warm
restore (whole prompt cached); `partial` means a longest-byte-prefix
lookup hit at `prefix_len` checkpoint tokens.

Used by cache-affinity routers to bias new requests toward the node
whose last admission for this model produced the longest warm
prefix.
""".
-spec last_cache_hit(model_id()) ->
    #{kind := exact | partial | cold, prefix_len := non_neg_integer()} | undefined.
last_cache_hit(ModelId) when is_binary(ModelId) ->
    case barrel_inference_inflight:obs_get(ModelId) of
        {_Id, _Phase, _Pending, undefined, _PrefixLen} ->
            undefined;
        {_Id, _Phase, _Pending, Kind, PrefixLen} ->
            #{kind => Kind, prefix_len => PrefixLen};
        undefined ->
            undefined
    end.

-doc """
Probe how much of `PromptTokens` is already cached for `ModelId`
on this node. Returns `{ok, MatchBytes}` where `MatchBytes` is the
byte length of the longest cached rendered-prompt prefix (across all
tiers: RAM, ram_file, disk). The cache is content-addressed by the
detokenised prompt bytes (ds4-style), so the probe detokenises
`PromptTokens` and matches on bytes. Returns `{ok, 0}` if no prefix
is cached or the prompt is empty. Returns `{error, model_not_loaded}`
if `ModelId` is not registered locally.

Lookup uses the model's effective fingerprint, so attached LoRA
adapters are honoured: cached rows produced under one adapter set
will not match a probe taken under a different adapter set.

Used by the `barrel_inference_cluster` cache-affinity router to route
prompts to the node with the longest matching cached prefix.
""".
-spec list_cached_prefixes(model_id(), [barrel_inference_nif:token_id()]) ->
    {ok, non_neg_integer()} | {error, term()}.
list_cached_prefixes(_ModelId, []) ->
    {ok, 0};
list_cached_prefixes(ModelId, PromptTokens) when is_binary(ModelId), is_list(PromptTokens) ->
    case barrel_inference_registry:whereis_name(ModelId) of
        undefined ->
            {error, model_not_loaded};
        _Pid ->
            KeyMeta = barrel_inference_model:cache_key_meta(ModelId),
            case barrel_inference_model:detokenize(ModelId, PromptTokens) of
                {ok, PromptBytes} ->
                    case
                        barrel_inference_cache:lookup_longest_text_prefix(
                            KeyMeta, PromptBytes
                        )
                    of
                        {ok, MatchBytes, _Row} -> {ok, MatchBytes};
                        miss -> {ok, 0}
                    end;
                {error, _} = E ->
                    E
            end
    end.

-doc """
Synchronous speculative draft. Generates up to `max` next-token
ids from the model given the supplied prefix and returns them as
a list. The list may be shorter than `max` if the model hits EOS
or its response_tokens limit first; an empty list is valid.

Implementation reuses `infer/4` and collects the
`{barrel_inference_token_id, Ref, Id}` messages it emits, so the path is
identical to ordinary streaming inference apart from the
synchronous reply. The 30 s default timeout cancels the
underlying request and drains any pending messages so they do
not leak into the caller's mailbox.

Used by the upcoming barrel_inference_cluster speculative-decoding
strategy to produce K candidate tokens for verification.
""".
-spec draft_tokens(
    model_id(),
    [barrel_inference_nif:token_id()],
    #{max => pos_integer(), atom() => term()}
) ->
    {ok, [barrel_inference_nif:token_id()]} | {error, term()}.
draft_tokens(_ModelId, [], _Opts) ->
    {error, empty_prefix};
draft_tokens(ModelId, PrefixTokens, Opts) when
    is_binary(ModelId), is_list(PrefixTokens), is_map(Opts)
->
    Params = draft_params(Opts),
    case barrel_inference:infer(ModelId, PrefixTokens, Params, self()) of
        {ok, Ref} -> collect_draft_tokens(Ref, [], 30_000);
        {error, _} = E -> E
    end.

draft_params(Opts) ->
    case maps:find(max, Opts) of
        {ok, Max} when is_integer(Max), Max > 0 ->
            #{response_tokens => Max};
        _ ->
            #{}
    end.

collect_draft_tokens(Ref, Acc, Timeout) ->
    receive
        {barrel_inference_token_id, Ref, Id} ->
            collect_draft_tokens(Ref, [Id | Acc], Timeout);
        {barrel_inference_token, Ref, _Bin} ->
            collect_draft_tokens(Ref, Acc, Timeout);
        {barrel_inference_done, Ref, _Stats} ->
            {ok, lists:reverse(Acc)};
        {barrel_inference_error, Ref, Reason} ->
            {error, Reason}
    after Timeout ->
        ok = barrel_inference:cancel(Ref),
        ok = drain_draft(Ref),
        {error, timeout}
    end.

%% Drain any messages still in transit after a timeout-driven
%% cancel so the caller's mailbox stays clean. A short tail
%% timeout is enough; the model's cancel handling fires the
%% terminal {barrel_inference_done, _, _} or {barrel_inference_error, _, _} within
%% one inter-token boundary.
drain_draft(Ref) ->
    receive
        {barrel_inference_token, Ref, _} -> drain_draft(Ref);
        {barrel_inference_token_id, Ref, _} -> drain_draft(Ref);
        {barrel_inference_done, Ref, _} -> ok;
        {barrel_inference_error, Ref, _} -> ok
    after 100 -> ok
    end.

-doc """
Speculative-decoding verifier. Runs `PrefixTokens ++ Candidates`
(truncated to `K` candidates) through the model in a single
forward pass with per-position argmax, returns the longest
accepted prefix length and the model's own next token after it.

Behaviour:
- The verifier model gen_statem is locked for the duration of
  the call; concurrent `infer/4` requests on the same model
  return `{error, busy}`. Verify only proceeds when the model
  is idle.
- The context's KV cells are mutated during the forward pass
  but restored before return: post-call the seq_id=0 KV ends at
  the same length the caller had before, with logits buffered
  for the last prefix token (so a follow-up `decode_one` is
  immediately valid). The caller's pre-call `decode_ready`
  flag is not preserved; after verify the context is always
  ready to sample.
- An empty `PrefixTokens` returns `{error, empty_prefix}`
  because the acceptance and NextToken indexing both require
  at least one prefix token.
- `NextToken` may be the atom `eos` if the verifier's argmax
  at the relevant position is an end-of-generation token; map
  it to terminate the decode loop.

Used by the upcoming barrel_inference_cluster speculative-decoding
strategy after `draft_tokens/3`.
""".
-spec verify(
    model_id(),
    [barrel_inference_nif:token_id()],
    [barrel_inference_nif:token_id()],
    pos_integer()
) ->
    {ok, non_neg_integer(), barrel_inference_nif:token_id() | eos} | {error, term()}.
verify(ModelId, PrefixTokens, Candidates, K) when
    is_binary(ModelId),
    is_list(PrefixTokens),
    is_list(Candidates),
    is_integer(K),
    K > 0
->
    barrel_inference_model:verify(ModelId, PrefixTokens, Candidates, K).

%% =============================================================================
%% Internal
%% =============================================================================

default_id() ->
    Int = erlang:unique_integer([positive]),
    iolist_to_binary(["barrel_inference_model_", integer_to_binary(Int)]).
