# erllama bench

Two harnesses live here:

- **Local "cold vs warm" markdown** (`bench/run.sh`) — quick eyeball
  comparison printed to stdout. Two scenarios: `cold_vs_warm` across
  prompt lengths and `multi_agent` for shared-prefix concurrency.
- **Cross-machine JSON collect** (`bench/collect.sh` and
  `bench/bootstrap.sh`) — one JSON artifact per `{machine, model}`
  pair, designed to be shipped off-box and aggregated. Use this when
  comparing GPUs / models across hosts.

## Cross-machine collect mode

```bash
# already cloned + built:
bench/collect.sh /path/to/model.gguf

# one-shot (clone, build, run):
curl -fsSL https://raw.githubusercontent.com/erllama/erllama/main/bench/bootstrap.sh \
  | bash -s -- /path/to/model.gguf
```

Output lands in `bench/results/` with a deterministic name:

```
<gpu-slug>__<model-basename>__erllama-<vsn>__<utc-ts>.json
```

The script auto-detects the host (kernel, arch, CPU brand, RAM) and
GPU (probes `nvidia-smi`, then `rocm-smi`, then `system_profiler` on
macOS, else falls back to CPU). One file per run; collect the files
back from each machine for off-line summarisation.

Tunables (env vars on `collect.sh` or `bootstrap.sh`):

| Var | Default | Meaning |
|---|---|---|
| `N_GPU_LAYERS` | `999` | Layers offloaded to GPU (high = all of them) |
| `N_CTX` | `4096` | `llama_context` size |
| `N_BATCH` | `4096` | `llama_decode` batch size |
| `N_SEQ_MAX` | `1` | Concurrent sequence slots |
| `BENCH_SHORT_TOKENS` | `50` | Short-prompt target length |
| `BENCH_LONG_TOKENS` | `500` | Long-prompt target length |
| `BENCH_RESPONSE_TOKENS` | `32` | Tokens to generate per workload |
| `SKIP_SHA256=1` | unset | Skip GGUF sha256 (faster, less model identity info) |
| `ERLLAMA_REF` | `main` | (bootstrap only) git ref to check out |
| `ERLLAMA_DIR` | `~/.erllama-bench/erllama` | (bootstrap only) clone target |

The collect harness runs four workloads:

1. `cold_short` — fresh cache, ~50-token prompt, 32 tokens generated.
2. `cold_long` — fresh cache, ~500-token prompt, 32 tokens generated.
3. `warm_long` — same prompt as `cold_long`, hits the row that
   `cold_long` just wrote.
4. `continue_3turn` — pinned session, turn 1 via `infer/4`, turns 2/3
   via `continue/3`. Captures per-turn Stats including the new
   `cache_hit_kind => continuation`.

The cache is cleared with `erllama_cache_meta_srv:gc/0` between
unrelated workloads so cross-prompt prefix overlap doesn't accidentally
warm a "cold" number. The `cold_long` → `warm_long` pair deliberately
shares state.

A run takes roughly 5-10 seconds on Metal/CUDA with TinyLlama, longer
on CPU or with larger models. Local results are gitignored.

### Caveats

- Timings are millisecond-granularity. Workloads under ~5ms of real
  work (notably `cold_short` against TinyLlama on Metal/CUDA) hit
  timer noise; on real-size models (Llama-3-8B and up) the same
  workload produces stable numbers.
- A warmup pass is run before measurement to amortise Metal/CUDA
  kernel-compilation cost on the first big-batch prefill. The
  warmup result is discarded.

### Summarising collected results

The JSON shape is flat enough for `jq` one-liners. Headline cache
speedup per file:

```bash
jq -r '
  .gpu.name as $gpu
  | .model.basename as $model
  | (.workloads[] | select(.name=="cold_long").prefill_ms) as $cold
  | (.workloads[] | select(.name=="warm_long").prefill_ms) as $warm
  | "\($gpu)\t\($model)\tcold=\($cold)ms\twarm=\($warm)ms\tspeedup=\(($cold/$warm)|tostring[0:4])x"
' bench/results/*.json
```

Continuation reuse per file:

```bash
jq -r '
  .gpu.name as $gpu
  | (.workloads[] | select(.name=="continue_3turn").turns) as $t
  | "\($gpu)\tturn2 read=\($t[1].cache_delta_read)/created=\($t[1].cache_delta_created)\tturn3 read=\($t[2].cache_delta_read)/created=\($t[2].cache_delta_created)"
' bench/results/*.json
```

For richer aggregation (a Markdown table across many runs), ship the
result files back to one place and feed them through whatever
post-processor fits — the schema is stable per `schema_version`.

## Local cold-vs-warm markdown mode

```bash
bench/run.sh              # all configured models
bench/run.sh tiny         # only the small one
bench/run.sh large        # only the large one
```

Models are selected by env var, with `tiny` defaulting to
`$HOME/Models/tinyllama-1.1b-chat.gguf`:

```bash
LLAMA_BENCH_TINY=/path/to/tinyllama-1.1b-chat.gguf \
LLAMA_BENCH_LARGE=/path/to/llama-3.1-8b-instruct.Q4_K_M.gguf \
  bench/run.sh all
```

A model whose path is unset or doesn't resolve to a file is skipped
with a stderr note. The bench creates a per-run tmp directory under
`$TMPDIR` for the disk tier and removes it on exit.

## What to compare

These are **internal** comparisons by default — they let you see
the cache speedup on a single binary. For external comparisons:

| Target | What it tells you |
|---|---|
| `bench/run.sh tiny` cold column | Equivalent to running raw `llama.cpp llama-cli` on the same prompt — that's the prefill cost the cache amortises. |
| `bench/run.sh tiny` warm column | Cache-restored next-token latency. |
| `multi_agent` p50/p99 | Latency under shared-prefix concurrency, the agent-loop scenario. |

For an apples-to-apples run against unwrapped llama.cpp, time
`./c_src/llama.cpp/build/bin/llama-cli -m $MODEL -n 1 -p '<same prompt>'`
and compare against the cold column. The warm column / cold column
ratio is the headline cache benefit.

## Caveats

- TinyLlama on CPU on M-series Macs gives a useful but not
  representative baseline; for production-shape numbers run the
  large variant on a Mac Studio or comparable.
- The first `complete/3` on a freshly-started Erlang node pays the
  llama.cpp model-load cost (seconds for a multi-GB GGUF). The
  bench's pre-warm covers that for `multi_agent`; `cold_vs_warm`'s
  cold column includes it implicitly for the first prompt-length
  bucket only.
- `n_gpu_layers => 0` keeps things on CPU for repeatability across
  machines. Override at the source if you want Metal/CUDA numbers.
- **`cold_vs_warm` reuses the same llama_context across the cold
  and warm call.** With short prompts (~512 tokens) the warm path
  is the expected ~13× faster. With longer prompts (>=1024 tokens)
  the speedup collapses to ~1× — this is *not* a cache failure
  (the `lp` column confirms the longest-prefix path is hitting),
  it's a llama-side cost: `kv_unpack` into a context that already
  has the prior call's KV cells in seq 0 isn't a clean
  reset+restore. The realistic agent-loop scenario has each agent
  on its own context — that's what `multi_agent` measures, and
  there the speedup over cold is the headline number. A
  `cold_vs_warm_fresh_context` follow-up that allocates a new
  context per warm call would isolate this further.
