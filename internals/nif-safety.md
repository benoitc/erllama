# NIF safety

The NIF (`erllama_nif`) is the only place erllama touches unmanaged
memory. This page records the patterns that keep the BEAM alive in the
face of llama.cpp's quirks, and where each one is tested. Read it
before you change anything under `apps/erllama/c_src/`. The dated audit
that led to several of these rules is
[c-safety-audit.md](c-safety-audit.md).

## One NIF, dirty schedulers by default

There is exactly one shared library (`priv/erllama_nif.so`), built from
several C and C++ files under `apps/erllama/c_src/` (`erllama_nif.c`
holds the function table, `erllama_nif_*.c` the per-concern code, the
`*.cpp` files the C++ shims). Every native call lives in it: cache
pack and unpack, model load, contexts, tokenisation, decode, KV
operations, chat templates, speculation, multimodal, tensor
parallelism.

Exports are marked `ERL_NIF_DIRTY_JOB_CPU_BOUND` or
`ERL_NIF_DIRTY_JOB_IO_BOUND`, with five exceptions that run on a
regular scheduler because they do constant, tiny work (the entries
with flag `0` in the `nif_funcs` table in `erllama_nif.c`):

- `nif_kv_format/0` returns a short version string;
- `nif_set_log_receiver/2` and `nif_clear_log_receiver/0` swap the
  native log target;
- `nif_request_abort/1` sets an atomic flag. It must not be dirty: as a
  dirty job it could queue behind busy dirty CPU schedulers, the decode
  it is meant to interrupt among them;
- `nif_tp_bootstrap_reply/3` hands every rank's bootstrap bytes to the
  thread waiting in `nif_tp_open` (a dirty IO job) and wakes it.

Why one library? The resource lifetime story is easier when every
type that participates lives in the same shared library. Cross-NIF
resource handling works in principle, but its failure modes
(mismatched destructor pointers when one side reloads) are not worth
the modularity.

Why dirty schedulers? `llama_decode` of a 4096-token batch takes
hundreds of milliseconds on CPU; calling it on a regular scheduler
would starve every other process on the node.

## Two-resource lifetime

Two resource types matter:

```
  model_ref       owns: llama_model* (weights, vocab)
  context_ref     owns: llama_context*; holds a keep-resource on its model_ref
```

`enif_keep_resource` keeps the model resource alive while any context
references it. `erllama_nif:free_model/1` returns `{ok, deferred}`
while contexts or LoRA adapters still use the model; the last one to
drop it performs the free (`context_drops_model` and
`model_drops_adapter` in `erllama_nif.c`). `llama_free_context` reads from the
model, so this is the only safe order under concurrent frees.

erllama does **not** use `enif_resource_refcount` for any decision: it
is not part of the public API, and control flow built on it breaks
when OTP changes its internals. Keep-references give the same effect
through documented APIs.

## Per-resource mutex

Each resource carries a `pthread_mutex` that every NIF function takes
before touching the llama struct:

1. **Race-free explicit free.** `erllama_nif:free_context/1` takes the
   mutex, frees the llama context and sets the pointer to `NULL` under
   it. A later call on the same resource finds the pointer `NULL` when
   it takes the lock (`erllama_lock_ctx` in `erllama_nif_int.h`) and
   answers `{error, released}`, even if it was waiting on another
   scheduler.
2. **One decode at a time per context.** llama.cpp does not document
   re-entrancy of `llama_decode`. The mutex serialises calls on a
   context, which matches the model process driving one step at a
   time.

## Exception safety: the `extern "C" noexcept` shims

llama.cpp is C++. It can throw `std::bad_alloc` on OOM and other
exceptions on bad input. An exception unwinding across the C NIF
boundary into the BEAM is undefined behaviour and crashes the VM.

Every llama call that can throw goes through an `extern "C"`
`noexcept` shim in `erllama_safe.cpp` (declared in `erllama_safe.h`);
the other C++ files (`erllama_chat_nif.cpp`, `erllama_mtmd_nif.cpp`,
`erllama_mtp.cpp`, `erllama_spec_nif.cpp`, `erllama_tp_nif.cpp`)
catch at their own entry points. The decode shim:

```cpp
int erllama_safe_decode(struct llama_context *c,
                        struct llama_batch batch) noexcept {
    try {
        return llama_decode(c, batch);
    } catch (...) {
        return ERLLAMA_DECODE_EXC_SENTINEL;   /* INT_MIN */
    }
}
```

The C side maps a sentinel to an Erlang error: `{error, exception}`
for a thrown exception, `{error, oom}` where an allocation failed, and
the decode codes (`decode_failed`, `decode_timeout`, `decode_aborted`)
for `llama_decode`'s own returns. The BEAM never sees a C++ exception
cross the boundary.

## No sampling without logits

`llama_sampler_sample` calls `llama_get_logits_ith`, which asserts
`logits != nullptr` and aborts the process when no decode has produced
logits. The context resource carries a `decode_ready` flag, set by a
successful prefill or decode and cleared by `kv_unpack`, `kv_seq_rm`
and a failed decode. `nif_decode_one` answers `{error, no_logits}`
when it is clear, instead of calling the sampler. For the batched
`step`, each sequence carries `last_logits_idx`: `-1` before its first
decode (`{error, no_logits}`), and a stale marker when another
sequence's decode replaced the logits rows (`{error, stale_logits}`).
The model layer re-prefills the last token before sampling again.

## Disk reads: plain read I/O, not mmap

The disk tier reads cache files with `file:read_file/1` into a fresh
BEAM heap binary. mmap was removed because:

- the process already mmaps the GGUF (multi-GB on a large model); a
  second mapping per cache restore doubles the VM footprint;
- a region binary returned to the BEAM outlives the NIF call, and any
  external truncation of the cache file then crashes the VM with
  SIGBUS in places no NIF can intercept (sub-binary creation, message
  send, GC).

For typical cache files the page cache already makes the second
`read(2)` cheap.

## `fsync_dir` and link durability

`erllama_nif:fsync_dir/1` opens a directory and `fsync(2)`s it. The
[publish protocol](publish-protocol.md) calls it after `link(2)`, so
the new directory entry is on stable storage. The path is copied with
`copy_path`, which rejects empty and oversized paths and embedded NUL
bytes (`badarg`), so `<<"a\0b">>` never reaches `open(2)` as `"a"`.

## Test surface

| Concern | Test |
|---|---|
| Decode without a prefill answers `no_logits` | `erllama_nif_tests:nif_step_decode_without_prefill_rejects_test_` (needs `LLAMA_TEST_MODEL`) |
| A row replaced by another sequence's decode answers `stale_logits` | `erllama_stale_logits_SUITE` (needs `LLAMA_TEST_MODEL`) |
| Bad input returns an error, never a crash | `erllama_nif_tests:prefill_context_overflow_test_`, `prefill_batch_overflow_test_`, `prefill_fuzz_test_` (runs `prop_erllama_nif:prop_prefill_never_crashes`) |
| A freed speculator answers `released` | `erllama_nif_tests:spec_draft_roundtrip_test` |
| A C++ exception caught at the boundary | no test forces one |
| Concurrent free and decode | no test |
| Model free deferred until the last context goes | no test |
| `fsync_dir` rejects NUL paths | no test |

The failure modes these guard against crash the BEAM rather than
return an error, so there is no graceful fallback: if one of these
tests flips on a change, the change is wrong. Add a test where the
table says there is none before relying on the behaviour.
