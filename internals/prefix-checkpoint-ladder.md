# Design proposal: prefix-checkpoint ladder for warm agent turns

Status: proposal, for review before implementing. Companion to
[cache-design.md](cache-design.md).

## Problem

Agent clients (Claude Code over `/v1/messages`, similar OpenAI loops) get
`cache_hit_kind = cold` on every turn. Measured live on a qwen3-coder:30b
load serving Claude Code:

```
longest_prefix_probes => 19296   hits_longest_prefix => 0
saves_finish => 11   saves_continued => 11   saves_cold => 0
```

Lookups run constantly and never hit. Two causes:

1. **No checkpoint at the shared boundary.** The cold path writes at most
   one save row, at the trimmed-prefix boundary (`cold` reason, see the
   save-reasons table in cache-design.md), and that boundary is near the
   full prompt length (`boundary_align_tokens` is small, `boundary_trim`
   is 0). The `finish` row is the full prompt+reply. So every stored row
   for a session is near-full-length. The longest-prefix walk probes the
   new prompt at stride alignments and can only hit a row whose tokens
   equal the new prompt's prefix at that length. When the conversation
   diverges anywhere before the end (a tool result inserted mid-history,
   client-side compaction, a re-render), the only common region is the
   head, and there is no row at a head-length boundary to hit.

2. **Turns are not token-exact extensions.** Claude Code re-renders and
   compacts: logged prompt sizes are non-monotonic. The session
   continuation path (`maybe_arm_continuation`, strict `lists:prefix`)
   and the engine sticky prefix-equality both require an exact extension,
   so neither arms.

(`cold_max_tokens` was also capping cold-save at 8192 while these prompts
are 25-32k, which zeroed `saves_cold` outright. That cap is raised
separately; this proposal is the structural fix that helps even when
turns are not clean extensions.)

The lookup side already does the right thing (walk at stride alignments,
longest hit wins). The gap is on the save side: we never produce rows at
the boundaries the walk probes below the full length.

## Proposal

During a cold prefill, emit `cold` save rows at a **ladder** of stride
boundaries, not just the final trimmed boundary. A later prompt that
shares the head up to boundary K then hits at K and resumes there,
re-prefilling only the divergent tail.

Concretely:

- `barrel_inference_cache_policy`: add a function that yields the ladder of
  save boundaries for a prompt given `(ladder_interval, cold_min_tokens,
  cold_max_tokens, boundary_align_tokens)`, e.g. every `ladder_interval`
  tokens from `cold_min_tokens` up to the trimmed boundary. Keep
  `cold_save_split/2` for the final boundary; the ladder is the set of
  earlier ones.
- `barrel_inference_model` cold prefill (`setup_cold/2`, the cursor-drain
  in `apply_step_results`, `maybe_fire_cold_save/2`): fire a `cold` save
  each time the prefill cursor crosses a ladder boundary, reusing the
  existing async save plumbing (`fire_save_for_tokens(cold, ...)`).
- Consumer (`lookup_longest_prefix`, `try_longest_prefix`) is unchanged.

No new save reason; the rows are ordinary `cold` rows at more boundaries.
Token-exactness, sole-writer arbitration, and the slab format are all
untouched.

## The cost tradeoff (the part that needs a decision)

Cumulative prefixes are redundant: a row at length L contains every
shorter row's bytes. A full ladder at the lookup stride (2048) for a 29k
prompt is ~14 rows totalling ~7x the full-prompt KV, with a pack pause at
each boundary during prefill. That is too expensive to take blindly.

Realistic shapes to choose between:

- **(B1) Coarse, bounded ladder.** Pick a `ladder_interval` much larger
  than the lookup stride (e.g. 8192) and cap the row count (e.g. 3-4).
  Bounds both storage and pack pauses; reuse granularity is coarse (a
  divergence at 12k reuses the 8k checkpoint). Smallest change, ships on
  the existing slab format and LRU. Disk tier absorbs the extra rows;
  per-tier quota + LRU already evict the cold ones.
- **(B2) Paged / block KV (radix-style).** Store KV in fixed-size blocks
  and reconstruct any prefix from shared blocks, so checkpoints cost
  one block each and there is no cumulative redundancy. This is the
  vLLM/SGLang answer and the right long-term shape, but it is a slab and
  meta-format rework (cache-design.md explicitly calls the slab format
  "opinionated, fixed-size per-layer regions"). Large; out of scope for a
  first cut.

Recommendation: ship **B1** behind policy knobs (`ladder_interval`,
`max_ladder_rows`, default off or conservative), measure the warm-hit
rate against a real Claude Code session, and only pursue B2 if B1's
coarse granularity proves insufficient.

## Open questions for review

1. B1 vs jumping straight to B2.
2. Ladder placement: fixed interval from the head, or biased toward the
   tail (agent turns most often diverge near the end, where one near-tail
   checkpoint already helps once `cold_max_tokens` is raised)?
3. Budget: extra cold rows compete with finish/continued rows in the LRU.
   Do we want a separate quota or a TTL for ladder rows so they cannot
   evict the more valuable finish rows?
4. Pack-pause budget during prefill: cap total ladder pack time per
   request so a long cold prefill does not stall the response.
5. Eviction interaction with `n_seq_max` concurrency (multiple cold
   prefills laddering at once).

## Tests

- Policy: ladder boundaries for representative lengths and knobs;
  count is bounded; empty below `cold_min_tokens`.
- Engine: a cold prefill emits `cold` saves at each ladder boundary
  (assert via counters / save announcements), and a follow-up prompt
  sharing only a head prefix resumes at the largest ladder boundary
  that still matches (`partial` hit), not cold.
- Budget: ladder rows are evictable and do not starve finish rows.
