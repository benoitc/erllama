# Roadmap

What erllama does not do yet, with rough scope and rationale for each
item. Issues / PRs welcome.

## Recently shipped: engine robustness

The `erllama_server` hardening brief (cold-admit decode wedges and
agentic tool-continue loops under real 30B/Metal load) is fully
shipped. For reference so these are not re-filed:

- **Bounded, interruptible, self-recovering decode** (0.8.0). Per-step
  wall-clock budget via a ggml abort callback
  (`context_opts.decode_budget_ms`, default 30000) returning
  `{error, decode_timeout}`; `erllama_nif:request_abort/1` for
  mid-decode interruption (`{error, decode_aborted}`, wired into
  `cancel/1`); in-place recovery via the backend `reset_context/1`
  (model stays loaded). Recovery drops only live KV + sticky pins; the
  persistent tiered cache survives.
- **`reset_session/2`** (0.7.0). Forcibly drop a wedged session's live
  KV and in-flight req, reachable when the hot path is blocked.
- **Non-blocking admission** (0.8.0). `on_full => error` returns
  `{error, seq_capacity}` instead of queueing; `model_info/1` exposes
  `available_seqs` / `n_seq_max` (0.7.0).
- **Grammar honoured through tool-call syntax** (0.8.0). A binary
  `grammar` disables the greedy-on-syntax swap, so
  `tool_choice=required` / `response_format` hold end to end on
  `infer/4` and `continue/3`.
- **Byte-exact continuation aids** (0.8.0). `generated` token ids in
  the `erllama_done` Stats map, and `continue/3` `expect_committed`
  verification (`{error, {transcript_mismatch, _}}`).
- **Structured NIF errors** (0.7.0/0.8.0). `nif_decode_one` and
  `nif_step` surface `{error, {decode_failed, Rc}}`.

## Sister project: erllama_cluster (in development)

A separate OTP application that coordinates a fleet of erllama nodes
into one inference cluster. Each node still runs erllama as a
standalone library; the cluster layer sits on top and decides which
node serves which request.

Three v1 strategies:

- **Request distribution** with cache-affinity routing (follow-up
  requests prefer the node that warmed the KV cache for the prefix).
- **Speculative decoding across nodes** — small draft on one node,
  large verifier on another.
- **Pipeline parallelism** — models too large for one node split by
  layer ranges, hidden states passed as Erlang binaries.

Transport is QUIC via [erlang_quic](https://github.com/benoitc/erlang_quic)
(pure Erlang, no C NIF in the protocol path). Repository:
<https://github.com/erllama/erllama_cluster>.

### Pipeline parallelism: blocked on upstream llama.cpp

The cluster's pipeline-parallelism strategy needs three new
NIFs (`forward_partial/3`, `forward_continue/3`,
`forward_final/3`) that run a configurable subrange of model
layers and pass hidden states between nodes. llama.cpp's public
API does not expose layer-range execution today: `llama_decode`
always runs the full model, and `llama_model_n_layer/1` only
exposes the count. Implementing the NIFs requires patching
llama.cpp itself, which the cluster brief
(`AGENTS_TASKS.md`) puts out of scope.

The cluster already gates pipeline mode on
`erlang:function_exported(erllama, forward_partial, 3)` and
falls back to the other strategies (request distribution and
speculative decoding) when the export is missing, so this is a
graceful degradation rather than a blocker for the cluster's
v1 release.

If the NIFs become critical, the right next step is a separate
upstream-llama.cpp investigation: identify the minimum patch to
`llama_decode` (likely a `layer_start` / `layer_end` field on a
new `llama_decode_params` struct) and either upstream it or
maintain a fork.

### Verifier context isolation (item 6 design note)

The current `verify/4` implementation snapshots the caller's
KV length, runs the verifier forward pass, and restores via
`kv_seq_rm` plus a re-prefill of the last prefix token. The
`decode_ready` flag is left in the "ready" state after verify
regardless of its pre-call value, on the assumption that any
caller of `verify/4` will follow up with a `decode_one`
imminently. Callers that need bit-identical pre-call state
beyond the sampling distribution (e.g. pre-call `decode_ready`
preserved as `false`) should clear the sampler explicitly
after verify. A v2 path that snapshots and restores
`decode_ready` is straightforward but waits for a concrete
caller need.

## Backlog (no fixed milestone)

### Speculative decoding

Pair a small draft model with a target model; speculate-and-verify
to improve throughput. Needs a "verify N tokens at offset" NIF and
a draft-model registry. The KV cache layer is largely orthogonal
(verifications run on the target context).

### Vision / LLaVA

`llama.cpp` supports the LLaVA family via `llava_init_from_*` and
`llava_eval_image_embed`. The Erlang surface needs an
`apply_image/2` callback on the backend, an embed-cache integration
(so re-uploaded images don't re-tokenize), and chat-template
extensions for multi-modal messages.

### Audio (Whisper)

A different model class than the GGUF chat models 0.1 targets;
`whisper.cpp` has its own context shape. Could be a sister
application (`erllama_whisper`) sharing the cache subsystem.

### Non-GGUF model loading

ONNX, safetensors, raw PyTorch checkpoints. llama.cpp doesn't load
these natively, so the path is either a converter step at
`fetch`-time (the `erllama_server` repo handles fetch) or a second
backend that targets a different runtime.

### Stateful streaming with bit-exact KV resume

Today's warm restore re-prefills the last KV cell to regenerate
logits, which can shift a near-tied sample. `erllama:continue/3`
(0.6.0) extends a pinned sticky session with a caller-asserted
token tail and avoids the warm-restore primer cost entirely for
multi-turn workloads, but it's still not bit-exact: the sampler
and RNG state are rebuilt per request. A turn-boundary save that
persists the sampler+RNG state alongside the KV cells would make
multi-turn replies bit-identical to the unbroken stream.

### Persistent sampler chain state across turns

If a chain carries internal state (e.g., `repetition_penalty`'s
sliding window), the current design rebuilds it per request.
Persisting it through the cache so a multi-turn resume picks up the
exact same sampler internal state is a 0.2 nice-to-have.

### Telemetry / OTel hooks

Counters today are bare atomics. A telemetry-style event surface
(`telemetry:execute([erllama, complete, start], ...)`) would let
operators wire Prometheus, OTel, statsd without forking the metrics
module.

### Memory-pressure NVIDIA-multi-GPU

`erllama_pressure_nvidia_smi` reads `nvidia-smi` once and sums; a
real multi-GPU deployment wants per-GPU pressure with per-context
eviction. The single-source pressure model in 0.1 collapses to "all
GPUs together".

### TurboQuant / KV state compression

KV state is bulky (~1 GB for 30k tokens on a 70B model). Generic
lz4/zstd helps a little; TurboQuant is unproven for this. We have no
benchmark data we trust enough to ship a default; both stay
deferred.

### KTM-inside-KV-files persistence

ds4 packs its tool-id → bytes map ("KTM") as appended sections
inside the disk KV cache files. Tempting because the lifetime
matches the KV row, but couples erllama's stable cache binary
format to an evolving HTTP-layer concept. Keep the two stores
separate; revisit only if running them side-by-side produces
measurable I/O amplification.

### Cluster / distributed inference

A model loaded on node A served from node B. The cache subsystem is
node-local; cross-node cache sharing (via the disk tier on a shared
filesystem, or a small announce protocol) is a 0.3+ topic.

### Incremental chat-template render

`erllama:continue/3` (0.6.0) lets callers pass only the new turn's
tokens, but they still have to render the full conversation through
`apply_chat_template/2` and slice off the prior tokens — wasteful
when the history is large. A companion `apply_chat_template_delta/3`
that takes the prior token count plus the new messages and emits
just the tail bytes would let HTTP layers skip the full Jinja pass
on every turn. The primitive is well-defined; the work is mostly
plumbing through the existing chat-template detector.

### Streaming tokenize for very large prompts

`erllama:tokenize/2` and `apply_chat_template/2` allocate the full
output buffer up front (worst case ~256 MB at the new 64 MiB text
cap). A streaming variant that yields tokens in chunks would let
the prefill scheduler start work before tokenization completes and
cap peak NIF allocations to a fixed chunk size. The C++ tokenize
already builds an internal vector incrementally, so the path is
exposing it through a cursor-based NIF rather than the one-shot
copy.
