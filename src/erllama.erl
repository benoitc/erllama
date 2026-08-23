%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama).
-moduledoc """
Public API of erllama: load llama.cpp models as supervised OTP
processes, run completions, stream tokens, and reuse the KV cache.

```erlang
{ok, _} = application:ensure_all_started(erllama),
{ok, Model} = erllama:load_model(#{model_path => "/srv/models/tinyllama.gguf"}),
{ok, #{reply := Reply, finish_key := Key}} = erllama:complete(Model, <<"hello">>),
{ok, #{reply := Reply2}} =
    erllama:complete(Model, <<"hello world">>, #{parent_key => Key}),
ok = erllama:unload(Model).
```

Every function that addresses a model returns `{ok, Result}` or
`{error, Reason}` with `Reason :: error_reason()`; an unknown or
stopped model is `{error, not_loaded}`, never a crash. Option maps
are validated: an unknown key is `{error, {unknown_option, Key}}`.

Models are dynamic children of `erllama_model_sup`. The id returned by
`load_model/1` (or supplied as `model_id` / to `load_model/2`) is the
handle for every other call; a pid works too. The cache subsystem is
`erllama_cache`; observability hooks are `erllama_middleware`.
""".

-export([
    load_model/1,
    load_model/2,
    unload/1,
    whereis/1,
    list_models/0,
    model_info/1,
    status/1,
    phase/1,
    pending_len/1,
    queue_depth/0,
    queue_depth/1,
    last_cache_hit/1,
    complete/2,
    complete/3,
    prefill_only/2,
    prefill_only/3,
    stream/3,
    collect/2,
    continue/3,
    cancel/1,
    end_session/2,
    reset_session/2,
    evict/1,
    shutdown/1,
    tokenize/2,
    tokenize/3,
    detokenize/2,
    render_chat_template/2,
    chat/3,
    chat_apply/3,
    chat_parse/3,
    embed/2,
    embed_batch/2,
    load_adapter/2,
    unload_adapter/2,
    set_adapter_scale/3,
    list_adapters/1,
    counters/0,
    vram_info/0,
    pressure/0,
    pressure_sources/0,
    requests/0,
    request_info/1,
    cached_prefix_len/2,
    draft_tokens/3,
    verify/4
]).

-export_type([
    model/0,
    model_id/0,
    model_info/0,
    load_config/0,
    token_id/0,
    cache_key/0,
    completion_result/0,
    prefill_result/0,
    stats/0,
    finish_reason/0,
    cache_hit_kind/0,
    cache_hit/0,
    phase/0,
    request_opts/0,
    prefill_opts/0,
    adapter/0,
    chat_params/0,
    chat_request/0,
    parsed_message/0,
    chat_message/0,
    chat_tool/0,
    chat_opts/0,
    chat_result/0,
    stream_event/0,
    stream_result/0,
    error_reason/0,
    middleware/0
]).

-doc "A model id (binary) or the model process pid.".
-type model() :: model_id() | pid().
-doc "The registered model id returned by `load_model/1,2`.".
-type model_id() :: binary().
-doc "Snapshot returned by `model_info/1` and `list_models/0`.".
-type model_info() :: #{
    id := binary(),
    %% Alias for `id`. Added for cluster registry rows so callers
    %% match on a name that does not collide with their own
    %% process-id-typed `id` fields.
    model_id := binary(),
    pid := pid(),
    status := idle | prefilling | generating,
    backend := module(),
    context_size := non_neg_integer(),
    quant_type := atom(),
    quant_bits := non_neg_integer(),
    %% String tag like <<"q4_k_m">> / <<"f16">>. Derived from
    %% quant_type + quant_bits.
    quant_tag := binary(),
    tier := ram | disk | ram_file,
    fingerprint := binary(),
    %% erlang:monotonic_time(nanosecond) at gen_statem init.
    loaded_at_monotonic := integer(),
    %% Best-effort estimate of VRAM footprint for the model when
    %% the gpu offload is configured. 0 if no GPU layers are
    %% offloaded or if the backend cannot report the underlying
    %% sizes (stub backend, etc.).
    vram_estimate_b := non_neg_integer(),
    %% Total seq capacity of the live context (`context_opts.n_seq_max`).
    n_seq_max := pos_integer(),
    %% Number of seq slots currently free for a new admission. A
    %% sticky session with no in-flight req still holds its slot
    %% off the free list, so this counts only seqs the engine could
    %% admit a brand-new request onto right now.
    available_seqs := non_neg_integer(),
    %% Sticky-session seqs that are pinned but idle (no in-flight
    %% request) - reclaimable under seq-pool pressure. `available_seqs`
    %% stays ~0 since an admitted request immediately re-pins, so this is
    %% the meaningful "headroom" signal.
    pinned_idle_seqs := non_neg_integer()
}.
-doc "A token id in the model vocabulary.".
-type token_id() :: non_neg_integer().
-doc "SHA-256 cache key of a committed context (`finish_key`, `parent_key`).".
-type cache_key() :: <<_:256>>.
-doc "Result of `complete/2,3`.".
-type completion_result() :: #{
    %% Detokenised reply text. Trimmed at the first occurrence of a
    %% matched `stop_sequences` entry when one fired.
    reply := binary(),
    %% Tokens produced by this request (not including the prompt).
    generated := [non_neg_integer()],
    %% Full context as a token list (prompt ++ generated).
    context_tokens := [non_neg_integer()],
    %% Convenience: length(context_tokens).
    committed_tokens := non_neg_integer(),
    %% Token-exact cache key for the full context. Pass as
    %% `parent_key` on the next request to resume from the warm row.
    %% `undefined` if the finish save was suppressed.
    finish_key := binary() | undefined,
    %% How this request resolved against the cache on admission.
    cache_hit_kind := cache_hit_kind(),
    finish_reason := finish_reason(),
    %% Per-request Anthropic cache breakdown. See `stats()`.
    cache_delta := #{
        read := non_neg_integer(),
        created := non_neg_integer()
    },
    stats := stats(),
    %% Only present when a caller-supplied `stop_sequences` entry
    %% fired. The value is the binary of the matched stop string.
    stop_sequence => binary()
}.
-doc "Result of `prefill_only/2,3`.".
-type prefill_result() :: #{
    context_tokens := [non_neg_integer()],
    committed_tokens := non_neg_integer(),
    finish_key := binary() | undefined,
    cache_hit_kind := cache_hit_kind(),
    %% Per-request Anthropic cache breakdown. See `stats()`.
    cache_delta := #{
        read := non_neg_integer(),
        created := non_neg_integer()
    }
}.
-doc "Per-request statistics (`stats` in results, `{done, Stats}` in streams).".
-type stats() :: #{
    prompt_tokens := non_neg_integer(),
    completion_tokens := non_neg_integer(),
    %% Exact generated token ids for this turn, in order. Lets a
    %% caller reconstruct a byte-exact suffix for `continue/3`
    %% without re-tokenising detokenised text. Mirrors the
    %% `generated` key in the standard-mode (`complete`) result map.
    generated := [token_id()],
    prefill_ms := non_neg_integer(),
    generation_ms := non_neg_integer(),
    cache_hit_kind := cache_hit_kind(),
    finish_reason := finish_reason(),
    cancelled := boolean(),
    %% Token-exact cache key for the full context (prompt ++ generated).
    %% `undefined` when the finish save was suppressed (e.g. live token
    %% count below `min_tokens`).
    finish_key := binary() | undefined,
    %% Length of `context_tokens` at finish (prompt + generated). Equal
    %% to `prompt_tokens + completion_tokens` unless the cache pruned
    %% the live context (not currently possible).
    committed_tokens := non_neg_integer(),
    %% Anthropic-style per-request cache breakdown. `read` is the
    %% warm prefix length restored from cache at admission;
    %% `created` is the largest contribution this request added to
    %% the cache beyond that prefix (saves below `min_tokens` and
    %% writer back-pressure leave `created` unchanged). Both
    %% default to 0.
    cache_delta := #{
        read := non_neg_integer(),
        created := non_neg_integer()
    },
    %% Only present when a caller-supplied `stop_sequences` entry
    %% fired. The value is the binary of the matched stop string.
    %% Absent on `length`, `cancelled`, or natural end-of-generation
    %% without a stop-string match.
    stop_sequence => binary()
}.
-type finish_reason() :: stop | length | cancelled.
-doc "How the last admission found its prefix in the cache.".
-type cache_hit_kind() :: exact | partial | cold | sticky | continuation.
-doc "Last admission summary from `last_cache_hit/1`.".
-type cache_hit() :: #{kind := cache_hit_kind(), prefix_len := non_neg_integer()}.
-type phase() :: idle | prefilling | generating.
-doc "Handle of an attached LoRA adapter; treat as opaque.".
-type adapter() :: term().
-doc "Parser handle produced by `chat_apply/3` for `chat_parse/3`; treat as opaque.".
-type chat_params() :: reference().
-type chat_request() :: erllama_model_backend:chat_request().
-doc "Assistant message parsed by `chat_parse/3` / returned by `chat/3`.".
-type parsed_message() :: #{
    role := binary(),
    content := binary(),
    reasoning_content := binary() | undefined,
    tool_calls := [#{name := binary(), arguments := map(), id := binary() | undefined}]
}.
-doc """
A chat message. `role` is `system | user | assistant | tool`; `content`
is a binary or a list of content-part maps in the OpenAI shape;
assistant messages may carry `tool_calls`, tool results `tool_call_id`.
""".
-type chat_message() :: #{
    role := system | user | assistant | tool | binary(),
    content := binary() | [map()] | null,
    tool_calls => [map()],
    tool_call_id => binary(),
    name => binary()
}.
-doc "A tool definition: `name`, `description`, JSON-schema `parameters`.".
-type chat_tool() :: #{
    name := binary(),
    description => binary(),
    parameters => map()
}.
-doc """
Options for `chat/3`: `tools`, `tool_choice` (`auto | required | none`),
`parallel_tool_calls`, plus any `request_opts()` key.
""".
-type chat_opts() :: #{
    tools => [chat_tool()],
    tool_choice => auto | required | none,
    parallel_tool_calls => boolean(),
    response_tokens => pos_integer(),
    parent_key => cache_key() | undefined,
    session_id => term(),
    on_full => block | error,
    stop_sequences => [binary()],
    thinking => enabled | disabled,
    thinking_budget_tokens => pos_integer(),
    temperature => number(),
    top_p => number(),
    top_k => integer(),
    min_p => number(),
    repetition_penalty => number(),
    seed => non_neg_integer(),
    grammar => binary(),
    prefix_checkpoint_len => non_neg_integer(),
    middleware => [middleware()]
}.
-doc "Result of `chat/3`.".
-type chat_result() :: #{
    message := parsed_message(),
    prompt := binary(),
    reply := binary(),
    stats := stats()
}.
-doc """
Events delivered to the `to` process of `stream/3` and `continue/3`
as `{erllama, Ref, Event}`.

- `{token, Bin}`: text fragment (omitted when empty)
- `{token_id, Id}`: every generated token id, in order
- `{thinking, Bin}`: extended-thinking fragment (`thinking => enabled`)
- `{thinking_end, Sig}`: close of a thinking block with its signature
- `{done, Stats}`: completion; after `cancel/1` `Stats` carries
  `finish_reason => cancelled`
- `{error, Reason}`: failure; no `done` follows
""".
-type stream_event() ::
    {token, binary()}
    | {token_id, token_id()}
    | {thinking, binary()}
    | {thinking_end, binary()}
    | {done, stats()}
    | {error, error_reason()}.
-doc "Result of `collect/2`: the stream folded into a map.".
-type stream_result() :: #{
    reply := binary(),
    thinking := binary(),
    generated := [token_id()],
    committed_tokens := non_neg_integer(),
    finish_key := cache_key() | undefined,
    cache_hit_kind := cache_hit_kind(),
    finish_reason := finish_reason(),
    cache_delta := #{read := non_neg_integer(), created := non_neg_integer()},
    stats := stats(),
    stop_sequence => binary()
}.
-doc "A middleware: `fun(Request, Next) -> Response`; see `erllama_middleware`.".
-type middleware() :: erllama_middleware:middleware().

-doc """
Config map for `load_model/1,2`.

- `model_path` (required unless `backend` is `erllama_model_stub`):
  path to a GGUF file.
- `backend`: `erllama_model_llama` (default) or `erllama_model_stub`.
- `model_id`: explicit id (`load_model/1` only).
- `model_opts`: `n_gpu_layers`, `main_gpu`, `load_mode` (`auto | none |
  mmap | mlock | mmap_mlock | direct_io`; `use_mmap` / `use_mlock`
  booleans map onto it), `vocab_only`, `split_mode`, `tensor_split`.
- `context_opts`: `n_ctx`, `n_batch`, `n_ubatch`, `n_seq_max`,
  `n_threads`, `n_threads_batch`, `embeddings`, `offload_kqv`,
  `kv_unified`, `flash_attn`, `type_k`, `type_v`, `decode_budget_ms`.
- `fingerprint`, `fingerprint_mode`, `quant_type`, `quant_bits`,
  `ctx_params_hash`, `context_size`: cache-key inputs; see the loading
  guide.
- `tier`, `tier_srv`: cache tier for this model's saves (default RAM).
- `policy`: cache policy overrides (`min_tokens`, `cold_min_tokens`, ...).
- `thinking_markers`: `#{start := binary(), 'end' := binary()}`.
- `chat_template`: Jinja source that replaces the template stored in
  the GGUF for `chat/3` and `chat_apply/3` (for files that ship a broken
  or outdated template).
""".
-type load_config() :: #{
    model_path => file:name_all(),
    backend => module(),
    model_id => model_id(),
    model_opts => map(),
    context_opts => map(),
    fingerprint => <<_:256>>,
    fingerprint_mode => safe | gguf_chunked | fast_unsafe,
    quant_type => atom(),
    quant_bits => pos_integer(),
    ctx_params_hash => <<_:256>>,
    context_size => pos_integer(),
    tier => ram | ram_file | disk,
    tier_srv => atom(),
    policy => map(),
    thinking_markers => #{start := binary(), 'end' := binary()},
    chat_template => binary(),
    %% erllama_model_stub only
    step_delay_ms => non_neg_integer(),
    thinking_capable => boolean()
}.

-doc """
Options for `complete/3`, `stream/3` and `continue/3`.

- `response_tokens` (default 64): cap on generated tokens.
- `parent_key`: a previous `finish_key`; resumes from that cached row.
- `session_id`: pin the KV cells to a session across turns.
- `on_full`: `block` (default) or `error` when no seq is free.
- `stop_sequences`: stop strings; the match is trimmed from the reply.
- `thinking`, `thinking_budget_tokens`: extended-thinking control.
- `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`,
  `seed`, `grammar`: sampling.
- `prefix_checkpoint_len`: pin the first N tokens as a static prefix
  checkpoint.
- `to`: process receiving the stream events (`stream/3`, `continue/3`).
- `middleware`: per-call middleware chain (see `erllama_middleware`).
""".
-type request_opts() :: #{
    response_tokens => pos_integer(),
    parent_key => cache_key() | undefined,
    session_id => term(),
    on_full => block | error,
    stop_sequences => [binary()],
    thinking => enabled | disabled,
    thinking_budget_tokens => pos_integer(),
    temperature => number(),
    top_p => number(),
    top_k => integer(),
    min_p => number(),
    repetition_penalty => number(),
    seed => non_neg_integer(),
    grammar => binary(),
    prefix_checkpoint_len => non_neg_integer(),
    to => pid(),
    expect_committed => [token_id()],
    middleware => [middleware()]
}.

-doc "Options for `prefill_only/3`: `parent_key`, `session_id`, `on_full`, `prefix_checkpoint_len`.".
-type prefill_opts() :: #{
    parent_key => cache_key() | undefined,
    session_id => term(),
    on_full => block | error,
    prefix_checkpoint_len => non_neg_integer(),
    middleware => [middleware()]
}.

-doc """
Every `{error, Reason}` the API returns.

- `not_loaded`: no model with that id or pid.
- `already_loaded`: `load_model/2` with an id in use.
- `busy`, `seq_capacity`, `sticky_busy`: admission refused.
- `no_session`, `{transcript_mismatch, _}`: session errors
  (`continue/3`, `end_session/2`).
- `not_supported`, `chat_not_supported`, `no_template`: the backend or
  model lacks the feature.
- `timeout`, `engine_reset`, `decode_timeout`, `decode_aborted`,
  `{decode_failed, Rc}`, `context_overflow`: runtime failures.
- `oom`, `load_failed`, `malformed_gguf`, `no_gpu`, `too_large`,
  `not_found`: NIF-level failures.
- `{missing_config, Key}`, `{invalid_config, Key, Value}`,
  `{missing_option, Key}`, `{unknown_option, Key}`,
  `{invalid_option, Key, Value}`: validation.

Backends may add their own atoms; they are documented on the backend.
""".
-type error_reason() ::
    not_loaded
    | already_loaded
    | busy
    | seq_capacity
    | sticky_busy
    | no_session
    | {transcript_mismatch, #{
        stored_len := non_neg_integer(),
        expected_len := non_neg_integer(),
        diverge_at := non_neg_integer()
    }}
    | not_supported
    | chat_not_supported
    | no_template
    | timeout
    | engine_reset
    | context_overflow
    | decode_timeout
    | decode_aborted
    | {decode_failed, integer()}
    | oom
    | load_failed
    | malformed_gguf
    | no_gpu
    | too_large
    | not_found
    | empty_prefix
    | {missing_config, atom()}
    | {invalid_config, atom(), term()}
    | {missing_option, atom()}
    | {unknown_option, atom() | {atom(), atom()}}
    | {invalid_option, atom(), term()}
    | term().

%% =============================================================================
%% Lifecycle
%% =============================================================================

-doc """
Load a model. The id is taken from `model_id` in `Config` or
generated (`<<"erllama_model_N">>`). Returns `{error, {missing_config,
model_path}}` when no path is given for the llama backend and
`{error, {invalid_config, model_path, Path}}` when the file does not
exist; the model process is never started on a bad config.
""".
-spec load_model(load_config()) -> {ok, model_id()} | {error, error_reason()}.
load_model(Config) when is_map(Config) ->
    load_model(maps:get(model_id, Config, default_id()), Config).

-doc "Load a model under an explicit id. `{error, already_loaded}` if the id is in use.".
-spec load_model(model_id(), load_config()) -> {ok, model_id()} | {error, error_reason()}.
load_model(ModelId, Config) when is_binary(ModelId), is_map(Config) ->
    case erllama_opts:load_config(Config) of
        {ok, Config1} ->
            Req = #{
                op => load_model,
                model => ModelId,
                args => #{config => maps:remove(model_id, Config1)}
            },
            erllama_middleware:run(Req, fun do_load_model/1);
        {error, _} = E ->
            E
    end.

do_load_model(#{model := ModelId, args := #{config := Config}}) ->
    case erllama_model_sup:start_model(ModelId, Config) of
        {ok, _Pid} -> {ok, ModelId};
        {error, {already_started, _}} -> {error, already_loaded};
        {error, _} = E -> E
    end.

-doc "Unload a model and free its context. `{error, not_loaded}` if it is not running.".
-spec unload(model()) -> ok | {error, not_loaded}.
unload(Model) ->
    erllama_middleware:run(#{op => unload, model => Model, args => #{}}, fun do_unload/1).

do_unload(#{model := Model}) ->
    case erllama_model_sup:stop_model(Model) of
        ok -> ok;
        {error, _} -> {error, not_loaded}
    end.

-doc "Pid of a loaded model, e.g. to `erlang:monitor/2` it.".
-spec whereis(model()) -> {ok, pid()} | {error, not_loaded}.
whereis(Pid) when is_pid(Pid) ->
    case is_process_alive(Pid) of
        true -> {ok, Pid};
        false -> {error, not_loaded}
    end;
whereis(ModelId) when is_binary(ModelId) ->
    case erllama_registry:whereis_name(ModelId) of
        undefined -> {error, not_loaded};
        Pid -> {ok, Pid}
    end.

-doc "Loaded models as `model_info()` maps (id, status, backend, context size, quantisation).".
-spec list_models() -> [model_info()].
list_models() ->
    lists:filtermap(
        fun({_ModelId, Pid}) ->
            case erllama_model:model_info(Pid) of
                {error, _} -> false;
                Info -> {true, Info}
            end
        end,
        erllama_registry:all()
    ).

-doc "Inspect one loaded model (same map shape as `list_models/0`).".
-spec model_info(model()) -> {ok, model_info()} | {error, not_loaded | timeout}.
model_info(Model) ->
    case erllama_model:model_info(Model) of
        {error, _} = E -> E;
        Info -> {ok, Info}
    end.

-doc "Current phase, read through the model process.".
-spec status(model()) -> {ok, phase()} | {error, not_loaded | timeout}.
status(Model) ->
    case erllama_model:status(Model) of
        {error, _} = E -> E;
        Phase -> {ok, Phase}
    end.

-doc """
Lock-free phase snapshot from the model's observability row; answers
without crossing the model process, so it returns instantly while a
decode step is in flight.
""".
-spec phase(model_id()) -> {ok, phase()} | {error, not_loaded}.
phase(ModelId) when is_binary(ModelId) ->
    obs(ModelId, fun({_Id, Phase, _Pending, _Kind, _PrefixLen}) -> Phase end).

-doc """
Lock-free count of calls queued behind the model's current request
(`complete/2,3`, `prefill_only/2,3`, `stream/3`).
""".
-spec pending_len(model_id()) -> {ok, non_neg_integer()} | {error, not_loaded}.
pending_len(ModelId) when is_binary(ModelId) ->
    obs(ModelId, fun({_Id, _Phase, PendingLen, _Kind, _PrefixLen}) -> PendingLen end).

-doc "Number of admitted streaming requests across all loaded models.".
-spec queue_depth() -> non_neg_integer().
queue_depth() ->
    erllama_inflight:queue_depth().

-doc "Number of admitted streaming requests for one model.".
-spec queue_depth(model_id()) -> {ok, non_neg_integer()} | {error, not_loaded}.
queue_depth(ModelId) when is_binary(ModelId) ->
    case erllama_registry:whereis_name(ModelId) of
        undefined -> {error, not_loaded};
        Pid -> {ok, erllama_inflight:queue_depth(Pid)}
    end.

-doc """
Lock-free summary of the model's most recent admission: the cache hit
kind and the warm prefix length in tokens. `{ok, undefined}` before the
first admission.
""".
-spec last_cache_hit(model_id()) -> {ok, cache_hit() | undefined} | {error, not_loaded}.
last_cache_hit(ModelId) when is_binary(ModelId) ->
    obs(ModelId, fun
        ({_Id, _Phase, _Pending, undefined, _PrefixLen}) -> undefined;
        ({_Id, _Phase, _Pending, Kind, PrefixLen}) -> #{kind => Kind, prefix_len => PrefixLen}
    end).

%% =============================================================================
%% Completion
%% =============================================================================

-doc "`complete/3` with default options.".
-spec complete(model(), binary()) -> {ok, completion_result()} | {error, error_reason()}.
complete(Model, Prompt) ->
    complete(Model, Prompt, #{}).

-doc """
Run a completion and wait for the result.

`Result` carries `reply` (detokenised text, trimmed at a matched stop
string), `generated` (token ids produced), `context_tokens` (prompt ++
generated), `committed_tokens`, `finish_key` (cache key of the full
context, or `undefined` when the finish save was suppressed),
`cache_hit_kind`, `finish_reason` (`stop | length | cancelled`),
`cache_delta` (`#{read := N, created := N}`), `stop_sequence` (only when
one fired) and `stats`.

Pass the previous turn's `finish_key` as `parent_key` to resume from the
cached row instead of walking the longest cached prefix. See
`request_opts()` for every option.
""".
-spec complete(model(), binary(), request_opts()) ->
    {ok, completion_result()} | {error, error_reason()}.
complete(Model, Prompt, Opts) when is_binary(Prompt), is_map(Opts) ->
    with_opts(complete, Model, Opts, fun erllama_opts:request_opts/1, #{prompt => Prompt}, fun(
        #{model := M, args := #{prompt := P, opts := O}}
    ) ->
        erllama_model:complete(M, P, O)
    end).

-doc "`prefill_only/3` with default options.".
-spec prefill_only(model(), [token_id()]) -> {ok, prefill_result()} | {error, error_reason()}.
prefill_only(Model, PromptTokens) ->
    prefill_only(Model, PromptTokens, #{}).

-doc """
Decode a prompt into KV state and fire a finish save without sampling
any tokens. Returns `finish_key` for a later `parent_key`, or
`undefined` when the prompt is shorter than the policy's `min_tokens`.
With `parent_key` set and `PromptTokens` extending that context, only
the new suffix is prefilled.
""".
-spec prefill_only(model(), [token_id()], prefill_opts()) ->
    {ok, prefill_result()} | {error, error_reason()}.
prefill_only(Model, PromptTokens, Opts) when is_list(PromptTokens), is_map(Opts) ->
    with_opts(
        prefill_only, Model, Opts, fun erllama_opts:prefill_opts/1, #{tokens => PromptTokens}, fun(
            #{model := M, args := #{tokens := T, opts := O}}
        ) ->
            erllama_model:prefill_only(M, T, O)
        end
    ).

-doc """
Streaming inference. Returns `{ok, Ref}` at once; events arrive at the
`to` process (default the caller) as `{erllama, Ref, stream_event()}`.
`Prompt` is text (tokenised with `tokenize/2`) or a token list.

`session_id` pins the KV cells to a session so the next request with
the same id extends them in place; release it with `end_session/2`. A
concurrent request on a pinned session returns `{error, sticky_busy}`;
`on_full => error` returns `{error, seq_capacity}` instead of queueing
when no seq is free. `Stats.generated` in the `done` event is the
exact generated token list, usable as the suffix for `continue/3`.
Use `collect/2` to wait for the result without writing the receive
loop.
""".
-spec stream(model(), binary() | [token_id()], request_opts()) ->
    {ok, reference()} | {error, error_reason()}.
stream(Model, Prompt, Opts) when is_binary(Prompt), is_map(Opts); is_list(Prompt), is_map(Opts) ->
    with_opts(
        stream, Model, Opts, fun erllama_opts:request_opts/1, #{prompt => Prompt}, fun do_stream/1
    ).

do_stream(#{model := Model, args := #{prompt := Prompt, opts := Opts}}) when is_binary(Prompt) ->
    case erllama_model:tokenize(Model, Prompt) of
        {ok, Tokens} -> do_stream_tokens(Model, Tokens, Opts);
        {error, _} = E -> E
    end;
do_stream(#{model := Model, args := #{prompt := Tokens, opts := Opts}}) when is_list(Tokens) ->
    do_stream_tokens(Model, Tokens, Opts).

do_stream_tokens(Model, Tokens, Opts) ->
    {To, Params} = take_to(Opts),
    erllama_model:infer(Model, Tokens, Params, To).

-doc """
Wait for a streaming request started with `stream/3` or `continue/3`
and fold its events into a `stream_result()`. `{error, timeout}` after
`Timeout` milliseconds without any event (the request is cancelled and
its remaining events drained).
""".
-spec collect(reference(), timeout()) -> {ok, stream_result()} | {error, error_reason()}.
collect(Ref, Timeout) when is_reference(Ref) ->
    erllama_stream:collect(Ref, Timeout).

-doc """
Extend a pinned session by prefilling `SuffixTokens` on top of its live
KV cells, skipping the prefix-equality check and any cache lookup. Use
it when the chat template renders history-dependent prefixes that would
defeat `stream/3`'s sticky path.

`Opts` must carry `session_id`; events go to `to` (default the caller).
`expect_committed => [token_id()]` makes the engine verify the
session's stored tokens first and fail with `{error,
{transcript_mismatch, #{stored_len, expected_len, diverge_at}}}` on
divergence, leaving the session pinned for a retry. `parent_key` is
ignored. Events are those of `stream/3`; `Stats.cache_hit_kind` is
`continuation`.
""".
-spec continue(model(), [token_id()], request_opts()) ->
    {ok, reference()} | {error, error_reason()}.
continue(Model, SuffixTokens, Opts) when is_list(SuffixTokens), is_map(Opts) ->
    case maps:is_key(session_id, Opts) of
        false ->
            {error, {missing_option, session_id}};
        true ->
            with_opts(
                continue,
                Model,
                Opts,
                fun erllama_opts:request_opts/1,
                #{tokens => SuffixTokens},
                fun(
                    #{model := M, args := #{tokens := T, opts := O}}
                ) ->
                    {To, Params} = take_to(O),
                    erllama_model:continue(M, T, Params#{caller_pid => To})
                end
            )
    end.

-doc """
Cancel an in-flight streaming request. Idempotent. The caller still
receives the terminal `{erllama, Ref, {done, Stats}}` with
`finish_reason => cancelled`. The running decode is interrupted through
the backend's abort callback; an interrupt that fires recreates the
context in place, which also resets co-batched requests on other seqs.
The persistent cache is untouched.
""".
-spec cancel(reference()) -> ok.
cancel(Ref) when is_reference(Ref) ->
    erllama_model:cancel(Ref).

-doc """
Release a sticky session: free its KV cells and return the seq to the
idle pool. Unknown session ids are a no-op.
""".
-spec end_session(model(), term()) -> ok | {error, not_loaded | timeout}.
end_session(Model, SessionId) ->
    erllama_model:end_session(Model, SessionId).

-doc """
Forcibly drop a session's live KV cells and fail any in-flight request
on its seq (the caller gets `{erllama, Ref, {error, engine_reset}}`).
Bounded by a 5 s timeout so it stays usable when the engine is wedged:
`{error, timeout}` means the model process itself is unreachable.
Returns `{ok, recovered}` or `{ok, not_found}`. Prefer `end_session/2`
for normal teardown.
""".
-spec reset_session(model(), term()) ->
    {ok, recovered | not_found} | {error, not_loaded | timeout}.
reset_session(Model, SessionId) ->
    erllama_model:reset_session(Model, SessionId).

-doc """
Fire an `evict` save and release the model's live KV state without
unloading it. Bounded by the `evict_save_timeout_ms` application
environment key (default 30 s).
""".
-spec evict(model()) -> ok | {error, not_loaded | timeout}.
evict(Model) ->
    erllama_model:evict(Model).

-doc "Fire a `shutdown` save and return; same bound as `evict/1`.".
-spec shutdown(model()) -> ok | {error, not_loaded | timeout}.
shutdown(Model) ->
    erllama_model:shutdown(Model).

%% =============================================================================
%% Tokenizer
%% =============================================================================

-doc "Tokenise text (`add_special => true`, `parse_special => false`). Safe during inference.".
-spec tokenize(model(), binary()) -> {ok, [token_id()]} | {error, error_reason()}.
tokenize(Model, Text) when is_binary(Text) ->
    tokenize(Model, Text, #{}).

-doc """
Tokenise with explicit options. `parse_special => true` turns
chat-template markers in `Text` into their special token ids; use it
for prompts rendered by `chat_apply/2`. `add_special => false` skips
the BOS token.
""".
-spec tokenize(model(), binary(), #{add_special => boolean(), parse_special => boolean()}) ->
    {ok, [token_id()]} | {error, error_reason()}.
tokenize(Model, Text, Opts) when is_binary(Text), is_map(Opts) ->
    Req = #{op => tokenize, model => Model, args => #{text => Text, opts => Opts}},
    erllama_middleware:run(Req, fun do_tokenize/1).

do_tokenize(#{model := Model, args := #{text := Text, opts := Opts}}) when map_size(Opts) =:= 0 ->
    erllama_model:tokenize(Model, Text);
do_tokenize(#{model := Model, args := #{text := Text, opts := Opts}}) ->
    erllama_model:tokenize(Model, Text, Opts).

-doc "Detokenise token ids back to text.".
-spec detokenize(model(), [token_id()]) -> {ok, binary()} | {error, error_reason()}.
detokenize(Model, Tokens) when is_list(Tokens) ->
    Req = #{op => detokenize, model => Model, args => #{tokens => Tokens}},
    erllama_middleware:run(Req, fun(#{model := M, args := #{tokens := T}}) ->
        erllama_model:detokenize(M, T)
    end).

%% =============================================================================
%% Chat
%% =============================================================================

-doc """
Render a chat request through the model's built-in template with the
legacy renderer and tokenise it. `Request` carries `messages`, `system`
and `tools`. Fallback for models whose template the autoparser
(`chat_apply/2`) cannot handle; `{error, no_template}` when the GGUF
ships none.
""".
-spec render_chat_template(model(), chat_request()) ->
    {ok, [token_id()]} | {error, error_reason()}.
render_chat_template(Model, Request) when is_map(Request) ->
    erllama_model:apply_chat_template(Model, Request).

-doc """
One chat turn: render `Messages` (and the `tools` in `Opts`) through
the model's template, generate, and parse the output into a structured
assistant message with content, reasoning and tool calls. Messages and
tools are Erlang maps; see `chat_message()` and `chat_tool()`.

```erlang
{ok, #{message := #{content := Text, tool_calls := Calls}}} =
    erllama:chat(Model, [#{role => user, content => <<"hi">>}],
                 #{tools => [#{name => <<"weather">>, parameters => Schema}]}).
```

Streaming callers use `chat_apply/2` + `stream/3` + `chat_parse/3`.
""".
-spec chat(model(), [chat_message()], chat_opts()) ->
    {ok, chat_result()} | {error, error_reason()}.
chat(Model, Messages, Opts) when is_list(Messages), is_map(Opts) ->
    case erllama_opts:request_opts(maps:without([tools, tool_choice, parallel_tool_calls], Opts)) of
        {ok, _} ->
            {Chain, Opts1} = erllama_middleware:take(Opts),
            Req = #{op => chat, model => Model, args => #{messages => Messages, opts => Opts1}},
            erllama_middleware:run(Req, Chain, fun(
                #{model := M, args := #{messages := Ms, opts := O}}
            ) ->
                erllama_chat:chat(M, Ms, O)
            end);
        {error, _} = E ->
            E
    end.

-doc """
Render the prompt and build the output parser for one request with
llama.cpp's `common_chat_templates_apply`. `Messages` and `Opts` are
those of `chat/3` (only the chat keys of `Opts` are used). Returns the
prompt bytes and the `chat_params()` to hand to `chat_parse/3` for this
request's output; tokenise the prompt with `tokenize/3` and
`#{add_special => false, parse_special => true}`. The parser is not
reusable across requests.
""".
-spec chat_apply(model(), [chat_message()], chat_opts()) ->
    {ok, #{prompt := binary(), params := chat_params()}} | {error, error_reason()}.
chat_apply(Model, Messages, Opts) when is_list(Messages), is_map(Opts) ->
    {Chain, Opts1} = erllama_middleware:take(Opts),
    Req = #{op => chat_apply, model => Model, args => #{messages => Messages, opts => Opts1}},
    erllama_middleware:run(Req, Chain, fun do_chat_apply/1).

do_chat_apply(#{model := Model, args := #{messages := Messages, opts := Opts}}) ->
    case erllama_model:chat_apply(Model, erllama_chat:inputs(Messages, Opts)) of
        {ok, Params, Prompt} -> {ok, #{prompt => Prompt, params => Params}};
        {error, _} = E -> E
    end.

-doc """
Parse model output into a structured assistant message (content,
reasoning, tool calls) with the parser from `chat_apply/3`.
`IsPartial = true` accepts a streaming prefix.
""".
-spec chat_parse(chat_params(), binary(), boolean()) ->
    {ok, parsed_message()} | {error, error_reason()}.
chat_parse(Params, Input, IsPartial) when is_binary(Input), is_boolean(IsPartial) ->
    Req = #{
        op => chat_parse,
        model => undefined,
        args => #{params => Params, input => Input, partial => IsPartial}
    },
    erllama_middleware:run(Req, fun(#{args := #{params := P, input := I, partial := Partial}}) ->
        erllama_chat:parse(P, I, Partial)
    end).

%% =============================================================================
%% Embeddings
%% =============================================================================

-doc """
Embedding vector for a text or a token list. The model must be loaded
with `context_opts => #{embeddings => true}`.
""".
-spec embed(model(), binary() | [token_id()]) -> {ok, [float()]} | {error, error_reason()}.
embed(Model, Input) when is_binary(Input); is_list(Input) ->
    Req = #{op => embed, model => Model, args => #{input => Input}},
    erllama_middleware:run(Req, fun do_embed/1).

do_embed(#{model := Model, args := #{input := Text}}) when is_binary(Text) ->
    case erllama_model:tokenize(Model, Text) of
        {ok, Tokens} -> erllama_model:embed(Model, Tokens);
        {error, _} = E -> E
    end;
do_embed(#{model := Model, args := #{input := Tokens}}) when is_list(Tokens) ->
    erllama_model:embed(Model, Tokens).

-doc """
Embedding vectors for several inputs (texts or token lists) in one
round-trip to the model process. Stops at the first error.
""".
-spec embed_batch(model(), [binary() | [token_id()]]) ->
    {ok, [[float()]]} | {error, error_reason()}.
embed_batch(Model, Inputs) when is_list(Inputs) ->
    Req = #{op => embed_batch, model => Model, args => #{input => Inputs}},
    erllama_middleware:run(Req, fun(#{model := M, args := #{input := In}}) ->
        case tokenize_all(M, In, []) of
            {ok, TokenLists} -> erllama_model:embed_batch(M, TokenLists);
            {error, _} = E -> E
        end
    end).

tokenize_all(_Model, [], Acc) ->
    {ok, lists:reverse(Acc)};
tokenize_all(Model, [Text | Rest], Acc) when is_binary(Text) ->
    case erllama_model:tokenize(Model, Text) of
        {ok, Tokens} -> tokenize_all(Model, Rest, [Tokens | Acc]);
        {error, _} = E -> E
    end;
tokenize_all(Model, [Tokens | Rest], Acc) when is_list(Tokens) ->
    tokenize_all(Model, Rest, [Tokens | Acc]).

%% =============================================================================
%% Adapters
%% =============================================================================

-doc """
Load a LoRA adapter from a GGUF file and attach it with scale 1.0. The
adapter's sha256 is folded into the model's effective fingerprint, so
cached rows never mix across adapter sets. In-flight requests keep the
previous fingerprint.
""".
-spec load_adapter(model(), file:name_all()) -> {ok, adapter()} | {error, error_reason()}.
load_adapter(Model, Path) ->
    erllama_model:load_adapter(Model, Path).

-doc "Detach and free an adapter. Idempotent.".
-spec unload_adapter(model(), adapter()) -> ok | {error, error_reason()}.
unload_adapter(Model, Adapter) ->
    erllama_model:unload_adapter(Model, Adapter).

-doc "Change an attached adapter's scale (also splits the cache namespace).".
-spec set_adapter_scale(model(), adapter(), number()) -> ok | {error, error_reason()}.
set_adapter_scale(Model, Adapter, Scale) when is_number(Scale) ->
    erllama_model:set_adapter_scale(Model, Adapter, Scale).

-doc "Attached adapters with their scales.".
-spec list_adapters(model()) ->
    {ok, [#{adapter := adapter(), scale := float()}]} | {error, not_loaded | timeout}.
list_adapters(Model) ->
    case erllama_model:list_adapters(Model) of
        {error, _} = E -> E;
        List -> {ok, [#{adapter => H, scale => S} || #{handle := H, scale := S} <- List]}
    end.

%% =============================================================================
%% Observability
%% =============================================================================

-doc "Snapshot of the cache counters; see `erllama_cache:get_counters/0` for the keys.".
-spec counters() -> #{atom() => non_neg_integer()}.
counters() ->
    erllama_cache:get_counters().

-doc """
Free / total / used bytes summed over the non-CPU ggml devices.
`{error, no_gpu}` on a CPU-only build; fall back to a system memory
probe in that case.
""".
-spec vram_info() ->
    {ok, #{total_b := non_neg_integer(), free_b := non_neg_integer(), used_b := non_neg_integer()}}
    | {error, no_gpu | error_reason()}.
vram_info() ->
    erllama_nif:vram_info().

-doc """
Host or accelerator memory pressure from the scheduler's configured
`pressure_source` (`system` when the scheduler is off or on `noop`):
used and total bytes plus the source that produced them.
""".
-spec pressure() ->
    {ok, #{
        source := erllama_pressure:source(),
        used_b := non_neg_integer(),
        total_b := non_neg_integer()
    }}
    | {error, term()}.
pressure() ->
    Source =
        case erllama_scheduler:status() of
            #{pressure_source := noop} -> system;
            #{pressure_source := S} -> S;
            _ -> system
        end,
    try erllama_pressure:sample(Source) of
        {Used, Total} -> {ok, #{source => Source, used_b => Used, total_b => Total}}
    catch
        Class:Reason -> {error, {Class, Reason}}
    end.

-doc "Pressure sources the scheduler accepts (`noop`, `system`, `nvidia_smi`, ...).".
-spec pressure_sources() -> [erllama_pressure:source()].
pressure_sources() ->
    erllama_pressure:available_sources().

-doc "Admitted streaming requests across all models: the ref and the model pid and id.".
-spec requests() -> [#{ref := reference(), pid := pid(), model := model_id() | undefined}].
requests() ->
    Ids = maps:from_list([{Pid, Id} || {Id, Pid} <- erllama_registry:all()]),
    [
        #{ref => Ref, pid => Pid, model => maps:get(Pid, Ids, undefined)}
     || {Ref, Pid} <- erllama_inflight:all()
    ].

-doc "The model serving a streaming request, or `{error, not_found}` once it has finished.".
-spec request_info(reference()) ->
    {ok, #{ref := reference(), pid := pid(), model := model_id() | undefined}} | {error, not_found}.
request_info(Ref) when is_reference(Ref) ->
    case erllama_inflight:lookup(Ref) of
        {ok, Pid} ->
            Model =
                case [Id || {Id, P} <- erllama_registry:all(), P =:= Pid] of
                    [Id] -> Id;
                    [] -> undefined
                end,
            {ok, #{ref => Ref, pid => Pid, model => Model}};
        {error, not_found} ->
            {error, not_found}
    end.

-doc """
How many bytes of the detokenised `PromptTokens` are already cached
for this model, across all tiers. `{ok, 0}` when nothing matches.
Attached adapters are honoured through the effective fingerprint.
""".
-spec cached_prefix_len(model(), [token_id()]) ->
    {ok, non_neg_integer()} | {error, error_reason()}.
cached_prefix_len(_Model, []) ->
    {ok, 0};
cached_prefix_len(Model, PromptTokens) when is_list(PromptTokens) ->
    case erllama_model:cache_key_meta(Model) of
        {error, _} = E ->
            E;
        KeyMeta ->
            case erllama_model:detokenize(Model, PromptTokens) of
                {ok, PromptBytes} ->
                    case erllama_cache:lookup_longest_text_prefix(KeyMeta, PromptBytes) of
                        {ok, MatchBytes, _Row} -> {ok, MatchBytes};
                        miss -> {ok, 0}
                    end;
                {error, _} = E ->
                    E
            end
    end.

%% =============================================================================
%% Speculative decoding
%% =============================================================================

-doc """
Draft up to `max` next tokens after `PrefixTokens` and return their
ids. Shorter lists (EOS, `response_tokens`) are valid. Built on
`stream/3`; a 30 s silence cancels the request and returns
`{error, timeout}`.
""".
-spec draft_tokens(model(), [token_id()], #{max => pos_integer()}) ->
    {ok, [token_id()]} | {error, error_reason()}.
draft_tokens(_Model, [], _Opts) ->
    {error, empty_prefix};
draft_tokens(Model, PrefixTokens, Opts) when is_list(PrefixTokens), is_map(Opts) ->
    Params = draft_params(Opts),
    case stream(Model, PrefixTokens, Params) of
        {ok, Ref} -> erllama_stream:collect_token_ids(Ref, 30_000);
        {error, _} = E -> E
    end.

draft_params(Opts) ->
    case maps:find(max, Opts) of
        {ok, Max} when is_integer(Max), Max > 0 -> #{response_tokens => Max};
        _ -> #{}
    end.

-doc """
Verify speculative `Candidates` (first `K`) after `PrefixTokens` in one
forward pass. Returns the accepted prefix length and the model's own
next token (`eos` at end of generation). The model must be idle;
concurrent requests get `{error, busy}`. KV state is restored before
returning.
""".
-spec verify(model(), [token_id()], [token_id()], pos_integer()) ->
    {ok, #{accepted := non_neg_integer(), next := token_id() | eos}} | {error, error_reason()}.
verify(Model, PrefixTokens, Candidates, K) when
    is_list(PrefixTokens), is_list(Candidates), is_integer(K), K > 0
->
    case erllama_model:verify(Model, PrefixTokens, Candidates, K) of
        {ok, Accepted, Next} -> {ok, #{accepted => Accepted, next => Next}};
        {error, _} = E -> E
    end.

%% =============================================================================
%% Internal
%% =============================================================================

%% Validate an option map, split the per-call middleware chain out of
%% it, and run `Fun' on the request through that chain.
with_opts(Op, Model, Opts, Validate, Args, Fun) ->
    case Validate(Opts) of
        {ok, Opts1} ->
            {Chain, Opts2} = erllama_middleware:take(Opts1),
            Req = #{op => Op, model => Model, args => Args#{opts => Opts2}},
            erllama_middleware:run(Req, Chain, Fun);
        {error, _} = E ->
            E
    end.

%% Split the `to` option (default: the caller) from the request params.
take_to(Opts) ->
    To = maps:get(to, Opts, self()),
    {To, maps:remove(to, Opts)}.

%% Read the model's lock-free observability row and project it.
obs(ModelId, Project) ->
    case erllama_inflight:obs_get(ModelId) of
        undefined -> {error, not_loaded};
        Row -> {ok, Project(Row)}
    end.

default_id() ->
    Int = erlang:unique_integer([positive]),
    iolist_to_binary(["erllama_model_", integer_to_binary(Int)]).
