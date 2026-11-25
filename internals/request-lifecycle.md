# Request lifecycle

This is the contributor-facing walkthrough of a single inference request.
The public usage patterns live in [Examples](../guides/examples.md); this
page explains how admission, cache restore, batching, decode, and saving fit
together inside the model process.

## Model process states

Each loaded model is a supervised `gen_statem` with one underlying
`llama_context*`. It has two broad runtime states:

- `idle` when no request is active.
- `running` when one or more requests are prefilling, decoding, or queued
  for the multi-sequence scheduler.

The default context has `n_seq_max => 1`, which keeps single-request
behavior simple and deterministic. Setting `context_opts.n_seq_max > 1`
opts into concurrent prefill/decode across multiple active sequences.

## Before admission: work outside the model process

While the model generates, its process spends most of its time in the
decode step, so any call it has to answer waits for that step. Reading
the vocabulary or the chat templates does not need the context, so it
does not go through the model process:

- At init the model publishes its backend's shared state
  (`shared_state/1` in `erllama_model_backend`, the model reference for
  the llama.cpp backend) in its `erllama_registry` row. The row goes
  away with the process.
- `tokenize`, `detokenize` and, once the templates are built,
  `chat_apply` run in the caller's process with it. Tokenising takes
  the model's lock, a decode step takes the context's, so they run side
  by side; a read racing an unload gets `{error, not_loaded}`.
- The model process alone builds the chat templates (on the first
  `chat_apply`) and stores them in `erllama_chat_cache` under its id and
  pid, so a model loaded again under the same id builds its own.

Called with a pid, or on a backend without `shared_state/1`, these
calls still go through the model process. Tests:
`erllama_model_tests` (the stub) and the `shared_reads_*` cases in
`erllama_real_model_SUITE`.

## Admission

Every request starts by taking a sequence id from the model's idle pool. If
the caller supplied a `session_id`, the model tries to reuse that session's
pinned sequence instead. If the session is already active in another
request, admission returns `{error, sticky_busy}`.

`continue/3` is a separate admission path used by callers that have a
pinned session and want to extend its live KV with a caller-asserted token
tail. It short-circuits the cache resolution stage entirely: no `parent_key`
lookup, no longest-prefix walk, no prompt prefix-equality check. The only
admission checks are that the `session_id` is known (otherwise
`{error, no_session}`) and that the pinned seq is not in flight
(otherwise `{error, sticky_busy}`).

At admission, the model also prepares request-local state: token cursor,
sampling options, optional grammar, optional thinking/tool-call markers, and
the reply destination for streaming calls.

## Cache resolution

The admitted request resolves its warm prefix before any decode work starts.
The result is reported as `cache_hit_kind`:

| Kind | Meaning |
|---|---|
| `exact` | The supplied key or computed prompt key matched the full prompt. |
| `partial` | A shorter exact prefix matched and the suffix must be prefed. |
| `cold` | No usable prefix was found. |
| `sticky` | A pinned session sequence already contains the prefix live. |
| `continuation` | Caller asserted the tail on a pinned session via `continue/3`; no prefix check, no cache lookup. |

The cache is byte-addressed (ds4-style): the key is SHA-256 over the
rendered prompt bytes (`detokenize(tokens)`), not the token-id list.
The lookup order is:

1. Exact byte-key hit on the whole rendered prompt.
2. Supplied `parent_key`, if present (session fast path), accepted
   when its rendered bytes are a byte-prefix of this prompt; it also
   waits up to `session_resume_wait_ms` for an in-flight finish save.
3. Longest-byte-prefix lookup over the new prompt's rendered bytes.

Any failure of 1 or 2 falls through to 3; only a byte-prefix miss is
cold. The longest-byte-prefix lookup is still exact: it picks the
longest stored rendered-byte prefix whose recomputed SHA-256 key
matches and resumes from that checkpoint's exact tokens. The suffix
beyond the match reuses the caller's ORIGINAL tokens when the byte
boundary lands on a token boundary (token-exact, the common case);
only a mid-token boundary re-tokenises the byte remainder
(`checkpoint_tokens ++ tokenize(byte_suffix)` - identical byte
stream, ds4's contract).

## Restore or prefill

After cache resolution, the request takes one of four paths:

- **Warm restore.** `kv_unpack` restores the saved KV bytes into the
  sequence. The model then prefills the last token again as a logits primer
  so generation starts from fresh logits.
- **Cold prefill.** The sequence is cleared and the full prompt is prefed.
- **Sticky continuation.** The live sequence already holds the prefix, so
  the model truncates or extends in place and only prefills the new suffix.
- **Caller-asserted continuation.** `continue/3` enters with the session's
  stored tokens already resident on the pinned seq. The supplied suffix is
  queued for prefill without any prefix equality check; correctness is the
  caller's contract.

Long prompts are sliced by `prefill_chunk_size`, defaulting to
`max(64, n_batch div 4)`, so one large prefill cannot monopolize every
scheduler tick when other sequences are decoding.

## Decode

The request scheduler batches active rows into one `llama_decode` call
per tick, bounded by `n_batch`. A tick can contain a mix of prefill
tokens and decode tokens across different sequences.

Request-local samplers stay isolated. One request can use a grammar or a
fixed seed while another uses a different temperature. Tool-call and
thinking markers are recognized inline and surfaced on the streaming wire
without changing unrelated requests.

## The tick

Each tick (`step_tick/1` in `erllama_model`) runs the first of these
that applies:

1. **Cancellations.** Requests with a pending cancel are marked
   finishing, so the finisher pass drops them.
2. **A media prefill**, when the only active request has media pending:
   libmtmd's chunk evaluation cannot share a batch, so a media request
   waits while others are active.
3. **A speculative round**, when exactly one request is active, drafts,
   is ready to decode and has generated at least one token: an ngram
   draft verified in one batch, or a lone MTP round.
4. **A batched MTP round**, when the model has an MTP drafter, two to
   16 requests are active, every one of them is ready to decode (no
   prefill left, no media pending) and none asks for `logprobs`, and at
   least one drafts. The backend's `mtp_step_batch/4` decodes every
   request's next token and drafts in one batch.
5. **The co-batched step** otherwise: prefill slices and decode rows
   within the batch budget, through the backend's `step/2`.

A batched MTP round replaces every logits row, so it runs only when it
can hold every request waiting for a token; otherwise the co-batched
step holds them all.

## Speculation

Speculation is a per-request option (`speculative`), set up at
admission (`setup_spec/3`) against one speculator per model, shared by
the requests that use the same config.

- **ngram** drafts only while the request is alone on the model
  (step 3 above); with others active it decodes normally and resumes
  drafting when alone again.
- **MTP** drafts with the `draft_model` head, alone (step 3) or with
  other requests (step 4). The drafted length is capped by `n_max`,
  the batch budget and the request's remaining tokens.
- In a batched MTP round, a backend refusal (`not_greedy`,
  `not_supported`) turns drafting off for the request it names and runs
  the ordinary co-batched step for that tick; the other requests keep
  their state. A sampler that is not greedy can still join later rounds
  without drafts.
- Committed tokens from a round go through the same commit path as an
  ordinary decode (stop strings, stream events, saves, terminal
  checks). Overshoot past a stop is trimmed from the KV.

## Recovery

When a step fails, the model decides by the reason
(`is_recoverable_decode_error/1`):

- **`decode_timeout`, `decode_aborted`, `decode_failed`** (and
  `stale_logits` from a batched MTP round): `recover_in_place/2`. Every
  in-flight request gets `{error, Reason}`, so do queued admissions,
  sticky sessions are dropped, the backend recreates its context
  (`reset_context/1`) and the model goes back to `idle`. The model stays
  loaded. If the context cannot be recreated, the model stops.
- **Any other step error**: every in-flight request gets
  `{error, Reason}` and the model process stops with
  `{step_failed, Reason}`. Models are temporary children of
  `erllama_model_sup`, so it is not restarted: load it again.

## Save points

The model fires cache saves at stable boundaries:

| Reason | Trigger |
|---|---|
| `cold` | During a cold prefill, at each ladder boundary and at the trimmed prefix boundary. |
| `agent_prefix` | During a cold prefill, at `prefix_checkpoint_len`; the row is pinned. |
| `continued` | Every configured generated-token interval. |
| `finish` | When generation completes. |
| `evict` | When a holder must release a slab under pressure. |
| `shutdown` | During unload or model process stop. |

A cold prefill is split into segments (`setup_cold/2`, from
`erllama_cache_policy:cold_save_segments/2` and `build_cold_ladder/3`):
each segment boundary but the last fires a save as the prefill cursor
reaches it (`maybe_fire_cold_save/2`). With the default
`max_ladder_rows => 0` that is one save at the trimmed boundary; see
[prefix-checkpoint-ladder.md](prefix-checkpoint-ladder.md).

Async saves go through `erllama_cache_writer`. Synchronous saves block the
caller because the backing resource is about to be released.

Disk publication is handled by the reservation protocol described in
[Publish protocol](publish-protocol.md).

## Completion

When a request finishes, the model reports final stats, including:

- `finish_reason`
- `finish_key`
- `cache_hit_kind`
- `cache_delta`
- cancellation state, when applicable

One-shot requests release their sequence back to the idle pool. Sticky
requests keep the sequence pinned until a later request diverges or the
caller invokes `erllama:end_session/2`.

## Observability

Routers can inspect model state without crossing the model `gen_statem`:

```erlang
erllama:phase(ModelId).
erllama:pending_len(ModelId).
erllama:queue_depth(ModelId).
erllama:last_cache_hit(ModelId).
```

These read a public ETS row and are safe to call from a hot path.

## Where the code lives

| Stage | Module and functions |
|---|---|
| Admission, sequences, sessions | `erllama_model`: `admit/2`, `start_request/4`, the `setup_*` functions |
| Cache resolution | `erllama_model` (lookup chain), `erllama_cache_meta_srv:lookup_longest_text_prefix/2`, `lookup_exact_or_wait/2` |
| Ladder and save gating | `erllama_cache_policy` |
| The tick | `erllama_model`: `step_tick/1`, `maybe_media_tick/1`, `maybe_spec_tick/1`, `maybe_mtp_batch_tick/1`, `step_tick_ops/1` |
| Recovery | `erllama_model`: `recover_in_place/2`, `fail_all_requests/2` |
| Saves | `erllama_model` (`fire_save_if/5`), `erllama_cache_writer`, [publish protocol](publish-protocol.md) |
| Backend callbacks | `erllama_model_backend`; `erllama_model_llama` (llama.cpp), `erllama_model_tp` (a split model) |
| Request options | `erllama_opts` |

