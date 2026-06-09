# Weight residency: real-model bench

Measures the four `weight_residency` modes against a real GGUF so the
manifest knob's behaviour matches the spec the loader documents.

## Setup

- Box: Apple M4 Pro, 48 GB unified memory, macOS 15.5, Metal backend.
- Model: `Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf` (~13 GB on
  disk, mistral-small architecture, mmap-loadable).
- Build: `barrel_inference` main (post PRs #32-#36) with the four modes
  exposed via `weight_residency`.
- Method: a single BEAM process loads the model in each mode in
  sequence, runs one 64-token generation against a tiny user prompt,
  idles for 30 s, then unloads. The bench script is at
  `/tmp/weight_residency_bench.escript` in the maintainer's notes.

The page cache stays warm between runs (the GGUF bytes are still in
the kernel cache when mode N+1 loads), so load wall-time for modes
2-4 is artificially low. The relative shape of the columns is what
matters: peak RSS, resident_bytes, and tok/s.

## Results

| mode | load (ms) | RSS @ load (GB) | RSS @ infer (GB) | RSS @ idle (GB) | resident (mincore) (GB) | prefill (ms) | gen (ms) | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `eager`             | 19722 | 8.52 | 8.57 | 8.57 | 13.34 | 310 | 8248 | 7.8 |
| `lazy`              |   435 | 0.73 | 0.74 | 0.74 | 13.34 |   2 | 8256 | 7.8 |
| `pinned`            |  1634 | 14.07 | 14.08 | 14.07 | 13.34 |   2 | 8271 | 7.7 |
| `lazy_then_pin_resident` | 444 | 0.73 | 0.73 | **13.99** | 13.34 |   2 | 8342 | 7.7 |

## Reading the table

- **`RSS @ load` vs `RSS @ idle`** is the operator-facing knob. `lazy`
  reports 0.7 GB of process-RSS while `pinned` reports 14 GB on the
  same model, even though both end up with the same working set sized
  by mincore (13.3 GB). The difference is accounting: macOS does not
  attribute shared file-backed mmap pages to the process's RSS unless
  they are mlocked. `pinned` mlocks the whole file; `lazy` lets the
  shared page cache hold the bytes without claiming them against the
  BEAM's resident-set budget. Useful when many BEAMs share one model.

- **`lazy_then_pin_resident`** behaves like `lazy` during the first
  request, then jumps to `pinned`-shaped RSS at idle once the
  scheduler's `maybe_pin_resident_pages/1` hook fires inside
  `finish_req`. The post-infer reading is still 0.73 GB because macOS's
  RSS accounting catches up after the mlock; mincore sees the pages as
  resident either way.

- **`resident_bytes`** (the new gauge) reports the same 13.3 GB across
  all four modes because mincore counts every page in physical memory,
  not just the process's owned slice. So once the model has been used
  at least once on this host, `resident_bytes` saturates at the
  effective working-set size regardless of mode. The signal is in
  changes BEFORE the first request (lazy starts low, eager / pinned
  start at the full size).

- **`load (ms)`** shows the eager mode actually waiting for the kernel
  to read the file (~20 s here, cold cache). lazy and lazy_pin_resident
  return almost immediately because they only set up the mmap; the
  pages will fault in as the first prefill needs them. `pinned` pays
  the load time (mlock walks the file) but the bytes were already in
  the cache from the prior `eager` run, hence the 1.6 s instead of
  ~20 s.

- **`prefill (ms)`** is 310 ms for eager (cold cache, lots of
  faults during prefill) vs ~2 ms for every other mode because by then
  the working set has been pulled in. **tok/s is identical** across
  modes at 7.7-7.8 — page-resident weights cost the same to read
  regardless of which knob put them there.

## Operator guidance

- **Single-tenant, plenty of RAM:** stay on `eager` (default). The
  load-time read-ahead pays off; nothing to tune.
- **Sparse-prompt agent workload (tool-call heavy, narrow templates):**
  `lazy` keeps unused weight rows out of process RSS. Same tok/s.
- **Multi-tenant inference with several models on one box:**
  `lazy_then_pin_resident` after the first warm-up request gives each
  model a guaranteed-resident working set without paying the full file
  cost up front. Idle RSS lands between lazy's floor and pinned's
  ceiling.
- **High-jitter sensitivity (latency-critical fleet nodes):** `pinned`
  on a node with enough RAM headroom. Pages cannot be evicted under
  pressure; no surprise faults during decode.

## Reproducing

```sh
rebar3 compile
escript /tmp/weight_residency_bench.escript
```

The escript reads the GGUF path from a `?GGUF` macro at the top; change
it to point at any local GGUF that the autoparser path can render
(any model with a chat template will do). For a true cold-cache
comparison, restart the host between modes — back-to-back in one BEAM
artificially benefits modes 2-4.
