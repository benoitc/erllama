# Prefix-checkpoint ladder

Status: shipped (option B1 of the original proposal), off by default.
This page records why the ladder exists, what was built, where it lives
and how to check it. Read it before you change cold-prefill saves,
`erllama_cache_policy` or the `agent_prefix` checkpoint. Companion to
[cache-design.md](cache-design.md); the user-facing knobs are in the
[caching guide](../guides/caching.md).

## Problem

Agent clients (Claude Code over `/v1/messages`, similar OpenAI loops)
got `cache_hit_kind = cold` on every turn. Measured live on a
qwen3-coder:30b load serving Claude Code:

```
longest_prefix_probes => 19296   hits_longest_prefix => 0
saves_finish => 11   saves_continued => 11   saves_cold => 0
```

Lookups ran constantly and never hit, for two reasons:

1. **No checkpoint at the shared boundary.** A cold prefill wrote at
   most one save, at the trimmed-prefix boundary, which is close to
   the full prompt length, and the finish save is the full
   prompt plus reply. So every stored row for a session was close to
   full length. When a conversation diverges before its end (a tool
   result inserted mid-history, client-side compaction, a re-render),
   the only common region is the head, and no row ended there.
2. **Turns are not token-exact extensions.** Claude Code re-renders
   and compacts, so the session continuation path and the sticky
   prefix check, which both need an exact extension, never armed.

(`cold_max_tokens` was also capping cold saves at 8192 while these
prompts were 25k to 32k tokens, which zeroed `saves_cold`; that was
fixed separately.)

The lookup side already did the right thing: the longest-byte-prefix
lookup takes the longest stored row whose bytes prefix the new prompt.
The gap was on the save side.

## What shipped

A cold prefill can write extra `cold` rows at a ladder of boundaries
below the trimmed boundary, so a later prompt that shares only the head
hits the largest boundary it still matches and prefills only the
divergent tail. The rows are ordinary `cold` rows: no new save reason,
no format change, the same reservation and publication rules.

- **Boundaries** (`erllama_cache_policy:cold_save_segments/2`): up to
  `max_ladder_rows` points spaced `ladder_interval` apart (rounded down
  to a multiple of `boundary_align_tokens`), at or above
  `cold_min_tokens` and strictly below the trimmed boundary, plus the
  trimmed boundary itself. A prompt outside
  `[cold_min_tokens, cold_max_tokens]` gets no cold save at all.
- **Defaults:** `max_ladder_rows => 0` (the ladder is off: one cold
  save at the trimmed boundary, as before) and
  `ladder_interval => 16384`. Both are keys of a model's `policy` map
  (accepted by `erllama_opts`).
- **Firing** (`erllama_model`): `setup_cold/2` splits the prompt into
  segments; the first goes into the prefill cursor, and each time the
  cursor drains at an internal boundary `maybe_fire_cold_save/2` packs
  the sequence (its KV then holds exactly that prefix), fires an async
  save and rotates the next segment in. The last segment is the
  remainder, covered by the finish save.
- **Static prefix** (`build_cold_ladder/3`): a request with
  `prefix_checkpoint_len => N` adds a boundary at N, tagged
  `agent_prefix` instead of `cold`. That row ignores the cold band
  (only `min_tokens` applies) and is pinned, so `gc/0` keeps it and
  byte-targeted eviction takes it only as a last resort.

## Decisions on the open questions

- **B1 or B2.** B1 (a coarse, bounded ladder on the existing slab
  format) shipped. B2 (paged, block-level KV shared between prefixes)
  was not built: it is a slab and meta-format rework.
- **Placement.** Fixed interval from the head, plus the optional
  `prefix_checkpoint_len` boundary for a known static prefix.
- **Budget.** No separate quota or TTL: ladder rows compete with the
  other rows under frecency eviction (see cache-design.md). Rows that
  must stay are the pinned `agent_prefix` ones.
- **Pack pauses.** Bounded only by `max_ladder_rows`: each boundary
  costs one pack pause during the prefill.

## Cost

Cumulative prefixes are redundant: a row at length L holds every
shorter row's KV. With `max_ladder_rows` R, one cold prefill writes up
to R + 1 rows and pauses R + 1 times to pack. Keep `ladder_interval`
large and R small.

## Tests

- `erllama_cache_policy_tests`:
  `cold_save_segments_no_ladder_matches_legacy_test` (no ladder keys
  gives the single legacy boundary),
  `cold_save_segments_ladder_below_trim_test`,
  `cold_save_segments_bounded_by_max_ladder_rows_test`.
- `erllama_model_tests:ladder_fires_multiple_cold_saves_test`: a cold
  prefill with the ladder on fires several cold saves, and the head
  boundary's row is in the cache.
- `erllama_model_tests:build_cold_ladder_*_test`: the segments are a
  partition of the prompt, and only the boundary at N is tagged
  `agent_prefix`.
- `erllama_model_tests:agent_prefix_checkpoint_fires_pinned_and_survives_evict_test`
  and `no_agent_prefix_without_boundary_test`: the static-prefix row is
  pinned, survives `gc/0`, and a later prompt resumes from it.
- Not covered: a follow-up prompt that shares only a ladder boundary
  (no `agent_prefix`) resuming from that row as a `partial` hit.
