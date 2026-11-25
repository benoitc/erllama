# Crash-safe publish protocol

This is the contract between the writer pool
(`erllama_cache_writer`), the meta server (`erllama_cache_meta_srv`)
and the file tiers (`erllama_cache_disk_srv`, which serves both `disk`
and `ram_file` roots). Read it before you change any of them. It exists
because cache files must appear atomically to readers, must never be
partially written, and must be cleaned up correctly when a writer
crashes mid-save.

## The six stages

A file-tier save runs in the caller's process (`erllama_cache_writer`
`do_save/4` and the functions after it), once it holds a writer slot:

```
1. erllama_cache_meta_srv:reserve_save(Key, Tier, self())
       placeholder `writing` row + reservation; returns {ok, Token}
2. erllama_cache_disk_srv:write_tmp(Root, BuildMeta, Payload)
       <hex>.kvc.<pid>.<unique>.tmp, opened O_EXCL;
       prim_file:write/2 of [Prefix, Payload]; datasync; close
3. erllama_cache_meta_srv:check_reservation(Key, Token)
       the token is still the live one
4. erllama_cache_disk_srv:publish(WriteOut)
       link(2) to <hex>.kvc (EEXIST: validate, adopt or replace);
       fsync the directory (erllama_nif:fsync_dir/1);
       reopen and parse the published file
5. erllama_cache_meta_srv:mark_published(Key, Token, FinalPath)
       the reservation moves to stage post_link
6. erllama_cache_meta_srv:announce_saved(Key, Token, Size, Header, TokensBin, TextBytes)
       the row becomes `available`; waiters are answered
```

A failure in stage 2 or 4 deletes the temp file and cancels the
reservation (`cancel_reservation/2`); an `{error, expired}` in stage 3
deletes the temp file. The RAM tier has no file: it reserves, stores
the slab in `erllama_cache_ram` and announces.

## Why each stage exists

### Stage 1: reserve

Prevents two writers for the same key from both reaching stage 4. The
meta server inserts a placeholder row with status `writing` and keeps
a reservation (writer pid, monitor, token, a 60 s TTL, stage
`pre_link`). `reserve_save/3` answers:

- `{error, already_present}` when the row is `available` (the writer
  counts it as a dropped duplicate);
- `{error, conflict}` when another writer's reservation is live (its
  process alive and its TTL not passed), or the row is being evicted;
- `{ok, Token}` otherwise, after cleaning up a stale reservation for
  the key if there is one.

### Stage 2: write_tmp

The temp file lives in the same directory as the final file, so
`link(2)` never crosses file systems, and its name carries the
writer's pid and a unique integer, so two writers never share a temp
file. The payload goes out as an iolist (no flattening of multi-GB
binaries). The file is `datasync`ed before stage 3, so the bytes stage
4 publishes are durable.

### Stage 3: check_reservation

The window between stage 1 and stage 4 can be long on a slow disk. In
that window the writer's reservation can expire and be taken over by
another writer. `check_reservation/2` confirms the token is still the
live one and extends its TTL. `{error, expired}` means "someone else
owns this key now; abandon".

### Stage 4: publish

`link(2)` is the durable publish: atomic create-if-not-exists. On
`EEXIST` the existing file is never skipped silently:

- **Adopt** it if it parses as a valid file for this key: the temp
  file is deleted and the directory fsynced.
- **Replace** it otherwise: delete it and link the temp file again,
  once.

Then the directory is fsynced, so the new entry is on stable storage,
and the published file is reopened and parsed; a file that fails to
parse is deleted.

### Stage 5: mark_published

The reservation moves to `post_link` and records the final path. From
here, a writer that dies leaves a file the meta server can validate
and adopt, instead of a placeholder to delete.

### Stage 6: announce_saved

The meta server installs the `available` row, drops the reservation,
and answers callers parked in `lookup_exact_or_wait/2` (the
`parent_key` wait, `session_resume_wait_ms`). A writer whose token
expired between stages 5 and 6 gets `{error, expired}`: the file is
published and the meta server adopts it itself.

## Cleanup of a dead or late writer

The meta server cleans a reservation when its writer goes down (the
monitor), when its TTL has passed (a periodic sweep), or when a new
`reserve_save` finds it stale. What it does depends on the stage:

- **`pre_link`:** delete the placeholder row (and, for the RAM tier, a
  slab already stored). No file was published.
- **`post_link`:** validate the file at the recorded path. If it parses
  for this key, install it as an `available` row and answer the
  waiters; otherwise delete the file and the placeholder.

So a writer can crash at any point and the bytes it published are
still recovered.

On startup, each file tier deletes leftover `*.tmp` files, registers
every `<hex>.kvc` that parses, and deletes the ones that do not.

## Read path: plain read I/O

The disk tier reads cache files with `file:read_file/1` into a fresh
BEAM heap binary. mmap was removed: the process already mmaps the GGUF
(multi-GB), and a region binary that outlives the NIF call would
expose the BEAM to SIGBUS from any external truncation. See
[nif-safety.md](nif-safety.md).

## Test surface

| Invariant | Test |
|---|---|
| `EEXIST` adopted when the existing file is valid | `erllama_cache_disk_srv_tests:eexist_with_valid_existing_file_adopts_test` |
| `EEXIST` replaced when the existing file is junk | `erllama_cache_disk_srv_tests:eexist_with_corrupt_existing_file_replaces_test` |
| Writer down after link, valid file adopted | `erllama_cache_meta_srv_tests:down_post_link_with_valid_file_adopts_test` |
| Writer down after link, invalid file deleted | `erllama_cache_meta_srv_tests:down_post_link_with_invalid_file_deletes_and_clears_test` |
| Writer down before link, placeholder cleared | `erllama_cache_meta_srv_tests:down_pre_link_clears_placeholder_test` |
| A live reservation rejects a second writer | `erllama_cache_meta_srv_tests:reserve_rejects_concurrent_writer_test` |
| A stale token is refused before link | `erllama_cache_meta_srv_tests:check_reservation_expired_for_unknown_token_test` |
| A waiter is answered when the save publishes | `erllama_cache_meta_srv_tests:wait_replies_when_save_publishes_test` |
| `parent_key` resumes the previous turn | `erllama_model_tests:parent_key_session_resume_test` |
| Leftover temp files and corrupt files dropped at startup | `erllama_cache_disk_srv_tests:init_sweeps_tmp_files_test`, `init_drops_corrupt_kvc_files_test` |
| TTL expiry of a live but slow writer (the sweep) | no test |
| Concurrent writers racing through all six stages, one publishes | no test (only the reservation step is covered) |
| Directory fsync after link | no test |

If you change any stage, surface the tension with these tests to a
reviewer before landing, and add a test where the table says there is
none.
