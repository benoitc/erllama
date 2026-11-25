# Sessions

A session pins a conversation to one live sequence in the model's KV
memory, so each turn only prefills the new tokens instead of the
whole transcript. You need this page when you build anything
multi-turn: an agent loop, a chat UI, or branching exploration over
a shared prefix.

## The concept

erllama keeps conversation state at two levels:

- **Live KV (sessions).** A session owns a sequence whose KV cells
  stay in the model context between turns. Continuing it is nearly
  free: only the new suffix is prefilled. Capacity is bounded by
  `context_opts.n_seq_max`.
- **The tiered cache.** Finished prefixes are also saved to the
  byte-keyed cache (RAM/disk), so even a session that was evicted, or
  a stateless caller resending the whole transcript, warm-restores
  instead of prefilling cold. See [caching](caching.md).

Sessions are the fast path; the cache is the safety net underneath.

## `session_id` or `parent_key`?

Both carry a conversation from one turn to the next, at different
levels:

- `session_id` names a **live sequence** pinned in the model's context
  between turns. The next turn decodes on the cells already there: no
  restore, only the new suffix is prefilled. The pin lasts while the
  model runs and the sequence is not reclaimed (`end_session/2`,
  `reset_session/2`, an engine recovery, or the pool taking back the
  least-recently-used idle pin).
- `parent_key` names a **saved cache row**: the `finish_key` a previous
  `complete/3` or `prefill_only/3` returned. The next turn restores
  that row into a free sequence, then prefills the suffix. The row
  outlives the sequence: it stays until the cache evicts it, and on a
  `disk` tier it survives a restart of the model.

```erlang
%% Live pin: one sequence held for the conversation.
{ok, _} = erllama:complete(M, Turn1, #{session_id => chat_42}),
{ok, _} = erllama:complete(M, Turn2, #{session_id => chat_42}),

%% Saved row: no sequence held between turns.
{ok, #{finish_key := K}} = erllama:complete(M, Turn1, #{}),
{ok, _} = erllama:complete(M, Turn2, #{parent_key => K}).
```

Use `session_id` for active conversations when you have a free
sequence for each (`context_opts.n_seq_max`). Use `parent_key` when
conversations outnumber sequences, or when the next turn may come
much later. `finish_key` is `undefined` when the prompt was too short
to save (below `min_tokens`). If you track neither, a caller that
resends the whole transcript still warm-restores through the
longest-prefix lookup (see [caching](caching.md#when-does-a-hit-happen)).

## Start and continue a session

Pass any term as `session_id`; the first request creates the pin:

```erlang
{ok, R1} = erllama:complete(M, Prompt, #{session_id => my_chat, response_tokens => 64}),
Transcript = <<Prompt/binary, (maps:get(reply, R1))/binary>>,

%% Next turn: send the FULL transcript plus the new user text. The
%% engine detects the stored prefix and only prefills the suffix.
{ok, R2} = erllama:complete(M, <<Transcript/binary, "\nUser: and then?">>, #{
    session_id => my_chat, response_tokens => 64
}),
sticky = maps:get(cache_hit_kind, R2).
```

The stored transcript is the previous turn's prompt plus generated
tokens. When your next prompt strictly extends it you get
`cache_hit_kind => sticky` and `cache_delta.read` equal to the stored
length. When it diverges (a re-rendered template, an edited
history), the engine reuses the longest common prefix in place
(`partial`) or falls back to a cold admit below the `min_tokens`
floor.

`erllama:chat/3` uses the same mechanism: pass `session_id` in its
options and each round of an agent loop reuses the rendered prefix
(see `examples/agent_loop`).

## Token-level continuation: `continue/3`

Chat templates sometimes render history-dependent bytes that defeat
prefix matching. `continue/3` skips the check: you assert the
transcript and hand only the suffix tokens.

```erlang
{ok, Ref} = erllama:continue(M, SuffixTokens, #{
    session_id => my_chat,
    expect_committed => StoredTokens   %% optional transcript guard
}),
{ok, #{reply := Reply}} = erllama:collect(Ref, 30000).
```

With `expect_committed` the engine verifies the session's stored
tokens first and fails with
`{error, {transcript_mismatch, #{stored_len, expected_len, diverge_at}}}`
on divergence, leaving the session intact for a retry. Stats report
`cache_hit_kind => continuation`.

## Fork a session

`fork_session/3` duplicates a session's live KV into a new session,
so two continuations explore different branches without re-prefilling
the shared prefix:

```erlang
ok = erllama:fork_session(M, my_chat, branch_b),
{ok, RA} = erllama:complete(M, <<Transcript/binary, " Option one:">>, #{session_id => my_chat}),
{ok, RB} = erllama:complete(M, <<Transcript/binary, " Option two:">>, #{session_id => branch_b}).
```

Notes on forking:

- You need free sequences: set `context_opts => #{n_seq_max => N}`
  at load, and `kv_unified => true` makes the copy metadata-only
  (branches share cells until they diverge).
- The copy carries no logits, so the fork's first request must
  extend the transcript; any normal continuation does.
- It works on every model family; recurrent models copy the
  compressed state tail.
- It never queues. With no free sequence (after reclaiming the
  least-recently-used idle session, never your source) you get
  `{error, seq_capacity}`. Other errors: `no_session`,
  `session_exists`, `sticky_busy` (source has an in-flight request),
  `seq_cp_failed`.

## End and reset

```erlang
ok = erllama:end_session(M, my_chat).
```

`end_session/2` frees the sequence and returns it to the pool.
Idempotent: unknown ids and busy sequences are no-ops (call again
after the in-flight request finishes). The transcript's prefix
usually survives in the tiered cache, so a later turn on the same
text still warm-restores.

```erlang
{ok, recovered} = erllama:reset_session(M, my_chat).
```

`reset_session/2` is the recovery hammer: it force-fails any
in-flight request on the session (the caller sees
`{error, engine_reset}`) and frees the sequence. It runs with a 5 s
timeout so it stays usable when the engine is wedged.

## Capacity

- `n_seq_max` sequences exist; each idle session pins one. When a
  new request needs a sequence and the pool is empty, the engine
  reclaims the least-recently-used idle pin (that session's next
  turn warm-restores from the cache instead), or queues when every
  sequence has an in-flight request.
- `model_info/1` reports the balance: `n_seq_max`,
  `available_seqs`, and `pinned_idle_seqs` (reclaimable headroom).

## See also

- [Generating text](generation.md): the request model and options
- [Caching](caching.md): what happens when a session is evicted
- [Tool calls](tool-calls.md): `chat/3` with sessions in agent loops
