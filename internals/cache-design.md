# Cache design

This is the contributor's-eye view of why the cache looks the way
it does. The user-facing description of *what* the cache provides
lives in the [caching guide](../guides/caching.md). This document
is the *why*.

## Byte-prefix addressing, not approximate

Cache keys are SHA-256 over `(model_fp || quant || ctx_params ||
rendered_prompt_bytes)`, where `rendered_prompt_bytes =
detokenize(tokens)`. The key is over the **rendered bytes**, not the
token-id list (the model layer detokenises before keying). This is
the ds4 model: content-addressed by the surface text, no session id.

Why bytes and not tokens? Across agent turns the *same* logical
prompt routinely retokenises (chat-template wrapping, tool
rendering, the assistant's generated ids vs the re-tokenised
assistant text). Keying on tokens makes those turns miss and re-cold-
prefill every time. Keying on the rendered bytes makes the leading
system + tools + history (stable text) hit even when the
tokenisation drifts. The exact tokens still travel in the checkpoint
payload (TLV `0x09`) for KV resume; only the *key* is byte-based.

A hit is still exact, not fuzzy: SHA-256 over the bytes means a key
match implies the bytes match (collision-negligible), so there is no
memcmp and no approximate distance metric. Approximate / semantic
matching was an early temptation and stays rejected:

1. **Correctness is not a tunable.** A "close enough" cache hit
   silently changes the model's output for the user. Either the
   state is the right state or it isn't.
2. **Approximate match needs a candidate proposer** (which embedding
   model? which distance metric?) that does not generalise across
   tenants. Out of scope for v1; tracked but not roadmapped.

The **longest byte-prefix** lookup
(`erllama_cache_meta_srv:lookup_longest_text_prefix` (arity 2))
solves "this prompt is yesterday's prompt plus a new turn": over the
`available` rows it picks the longest stored `text_bytes <=
byte_size(prompt_bytes)` whose recomputed key matches, restores that
checkpoint's exact tokens + KV, and prefills only the uncovered
suffix. When the checkpoint's byte boundary lands on a token boundary
of the caller's prompt (the common case, including an identical
re-send), the suffix is the caller's ORIGINAL remaining tokens, so
resume is token-exact and a re-sent prompt reproduces its reply. Only
when the boundary falls mid-token (a genuine retokenisation) is the
byte remainder re-tokenised (`checkpoint_tokens ++
tokenize(byte_suffix)`); the token boundaries at the seam then differ
from a fresh full tokenisation, but the byte stream is identical, so
the resume is sound (ds4's contract). The recomputed key folds in
the current model's `fp/quant/ctx`, so rows from another
model/quant/context never match; there are no separate namespace
fields.

Old token-keyed cache files (KVC format v1) are not adopted: the
format version was bumped to v2 and v1 files are rejected on the
startup disk scan. There is no backward-compatible reading of the
old key scheme; the cache simply refills under the byte scheme.

## Multi-tier, not just RAM

We ship three tiers because no single layer is the right answer
across deployments:

- **`ram`** (ETS slabs): lowest latency, smallest budget, and the
  default tier. ETS reads are sub-µs and hot-path-friendly; stores go
  through `erllama_cache_ram`, which enforces `ram_tier_budget` by
  evicting the lowest-frecency RAM rows before inserting.
- **`ram_file`** (`/dev/shm`): fast and effectively unlimited by
  process address space. Survives a model-supervisor restart but
  not a node restart.
- **`disk`**: survives everything. The cheap tier; deploy with the
  largest quota of the three.

Each tier is independently supervised. The RAM tier has a byte
budget (`ram_tier_budget`); the file tiers are bounded by their
storage and emptied by eviction (lowest frecency first). A save
written to one tier never moves; the disk tier
is intentionally not a "promotion target" of the RAM tier: that
would force every saved row to be re-encoded twice.

Why is the disk tier first-class instead of a "fallback"? Because
big-model deployments cannot fit a working set of warm KV state in
RAM alongside the weights. A 70B-class model in Q4 takes ~40 GB
of RAM for weights alone; a 30 000-token KV state can easily exceed
1 GB. With ten warm sessions you've blown a 24 GB GPU. Disk is the
realistic place for that working set, and modern NVMe is fast
enough to keep restore cost in the millisecond range.

## Sole-writer arbitration

The meta server (`erllama_cache_meta_srv`) is the only process that
mutates the meta ETS, the recency index (an ordered ETS table keyed
on last use) and the reservations. Every write (claim, release,
evict, save announce) goes through a gen_server call; only
`set_live_key/1` is a cast. Reads stay on ETS directly via
`ets:lookup/2`.

The split is deliberate: hot-path reads must not contend on a
gen_server message queue, but writes must serialise so we never
race two reservations for the same key. `ets:select_replace/2` was
considered for in-place atomic updates but rejected: the
reservation state machine is rich enough that single-row CAS would
not be enough, and we'd need a lock anyway.

## Eviction order: frecency

Eviction is not plain LRU. Each row's score is its last use, moved
forward by its accumulated hits: every hit counts one second
(`HIT_BONUS_NS`), decayed by a 6 h half-life on the row's age
(`HIT_HALF_LIFE_NS` in `erllama_cache_meta_srv`). A row hit often
recently outranks an old one, and old hits fade. Byte-targeted
eviction (`evict_bytes/2`, used by the RAM tier and the cache eviction
scheduler under memory pressure) drops the lowest scores first.

Protected rows (pinned `agent_prefix` checkpoints and the live key)
sort in a separate, higher rank: byte-targeted eviction takes them
only as a last resort, and `gc/0` never takes them. `gc/0` drops every
other available, unreferenced row, node-wide, and returns
`{evicted, N}`.

## The live key

`live_key` is one value for the whole node, held by the meta server:
the finish key of the most recent request that saved one
(`erllama_model` calls `erllama_cache_meta_srv:set_live_key/1` after a
finish save). It protects that row, the most likely resume target of
the next turn, from eviction until another request's finish replaces
it. It is not per model or per session: with several models or
sessions, only the last recorded key is protected. Pin a row
(`agent_prefix`) when it must stay.

## Save reasons taxonomy

Six reasons, each with distinct semantics:

| Reason | Trigger | Why it's a separate reason |
|---|---|---|
| `cold` | During a cold prefill, at each ladder boundary and the trimmed-prefix boundary | First save for this prefix; we want it stored as soon as possible. |
| `continued` | Every `continued_interval` tokens during generation | Keeps the cache useful even if generation is interrupted. |
| `finish` | End of generation, captures prompt+reply | Multi-turn flows resume from this row. |
| `agent_prefix` | At `prefix_checkpoint_len` | A static prefix shared by many requests; pinned against eviction. |
| `evict` | `erllama:evict/1` | Saves the live context of in-flight requests on demand. |
| `shutdown` | `erllama:shutdown/1`, `unload/1`, application stop | Best-effort save of in-flight work before the model goes away; capped by `evict_save_timeout_ms`. |

Every save runs in the model process, on the request path: the KV
state is packed and written before the model takes its next step,
so a save delays every request co-batched on that model. That cost
is why the save policy gates exist, and why file saves that find the
writer pool full are dropped (`saves_dropped`) instead of waiting.

## What the cache is not

- **Not a session manager.** It does not track conversations or
  authenticate callers. The session layer above passes a
  `parent_key` if it has one; the cache treats it as a hint, not a
  capability.
- **Not a request scheduler.** Concurrency, queueing, and rate
  limiting live above the cache. The cache only owns "the
  on-disk/in-RAM mapping from rendered-byte-prefix to KV bytes".
- **Not a generic blob store.** Slab format is opinionated:
  fixed-size per-layer regions, 48-byte header, CRC32C trailer.
  Repurposing the format for non-llama.cpp data would be miserable.
- **Not GPU-aware.** The cache stores KV bytes; whether they go on
  GPU or CPU is a property of the `llama_context*` that consumes
  them. The cache doesn't care.

## Two remaining v1 deferrals

Both intentional:

1. **No semantic candidate proposer.** Discussed above. v1 is
   exact-only.
2. **No KV state compression.** TurboQuant is unproven at this
   layer. Generic lz4/zstd would help a little on some quant
   schemes and hurt others. The breakeven is unclear and we don't
   have benchmark data we trust enough to ship a default.

Both are tracked for v2; neither blocks production use of v1.
