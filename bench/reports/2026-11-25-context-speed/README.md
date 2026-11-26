# How erllama speed changes with context length

25 November 2026. On two GB10s, GLM-5.3-Flash generated 20.1–20.7 tokens/s
across contexts from 2,048 to 8,192 tokens. On the M4 Pro, Qwen2.5-1.5B
processed a new 2,048-token block at 1,811 tokens/s with an empty history
and 1,252 tokens/s after 6,144 tokens were already in memory, a 31% drop.
Its generation rate varied from 122 to 137 tokens/s. Each point averages
two runs. The short sweep gives no basis for extrapolating beyond 8,192.

![The left and centre columns show erllama on Qwen and GLM. The right column compares DeepSeek V4 Flash on two GB10s with erllama and one GB10 with ds4. Circles mark erllama runs and squares mark upstream ds4 observations. Each panel has its own vertical scale.](context-speed.png)

| Context before generation | Qwen prompt extension | Qwen generation | GLM prompt extension | GLM generation |
|---:|---:|---:|---:|---:|
| 2,048 | 1,811 | 132.8 | 527 | 20.1 |
| 4,096 | 1,661 | 136.6 | 517 | 20.7 |
| 6,144 | 1,465 | 121.6 | 522 | 20.6 |
| 8,192 | 1,252 | 124.9 | 520 | 20.3 |

All rates are tokens/s. Read down a column to see the effect of context
length on that setup. Both hardware and model change between columns.

## DeepSeek V4 Flash on GX10

[Same-prompt comparison of erllama, ds4 and DSpark on two GX10s](dspark/README.md)

[One versus two GX10s with erllama, and the Q5_K trial](shared-expert/q5/README.md)

[GX10 custom GGUF build and shared-expert measurements](shared-expert/README.md)

erllama ran the ds4 Flash 0731 importance-matrix GGUF across two GX10s,
one GB10 per rank over RoCE. Steady generation stayed between **21.7
and 22.8 tokens/s**. Prompt extension fell from **715 to 558 tokens/s**
as the history grew. All eight trials returned 128 greedy tokens, with
zero end-of-generation tokens. This sweep measures throughput. Quality and checkpoint restoration require separate tests.

| Context before generation | erllama · two GB10s · prompt | ds4 · one GB10 · prompt | erllama · two GB10s · generation | ds4 · one GB10 · generation |
|---:|---:|---:|---:|---:|
| 2,048 | 715 | 826 | 22.3 | 18.2 |
| 4,096 | 639 | 900 | 22.8 | 15.5 |
| 6,144 | 588 | 889 | 22.0 | 15.4 |
| 8,192 | 558 | 883 | 21.7 | 15.3 |

Rates are tokens/s. The right-hand panels plot both DeepSeek series on the
same axes. erllama's circles show individual runs and its line connects
their means. The squares show ds4's published measurements on a single
GB10. Both generation rates use 127 intervals. ds4 used a different
prompt and timed prefill separately from the first generated token.
Its historical CSV gives no weight hash. An engine speedup or a
two-host scaling factor would require a matched run.
[erllama timings](deepseek-erllama.csv) · [ds4 plotted rows](ds4-gb10.csv)
· [ds4 source CSV](https://github.com/antirez/ds4/blob/e0c63d9345b6b249ab3d556b329102c4d96a935e/speed-bench/gb10.csv)

The [GX10 config](gx10-deepseek.env) pins the 86,720,111,488-byte file.
Both hosts verified its SHA-256 before loading. Across all 43 layers,
the routed expert gate and up projections use IQ2_XXS and the down
projections use Q2_K. The shared experts, main attention projections
and output head use Q8_0. Both hosts used the erllama-val build, CUDA
driver 580.178.04, direct I/O, and ordinary greedy decoding.
[Pinned weights](https://huggingface.co/antirez/deepseek-v4-gguf/tree/f71f23d552d664e523b422157b2befbf74040380)

The [GX10 expert-format benchmark](expert-formats/README.md) compares
the existing IQ2_XXS gate and up weights with Q4_K for decode and prefill.

## What the clocks measure

The workload follows the 2,048-token prompt extension and 128-token
greedy generation used in the [ds4 speed benchmark](https://github.com/antirez/ds4#speed).
We drive erllama's backend directly with one live sequence. The model
stays loaded, the KV history stays in memory during each measurement,
and ordinary decoding runs throughout. The earlier [TinyLlama HTTP and
cache retry comparison with Ollama](../2026-11-25-tinyllama-ollama/README.md)
measures a different workload.

- **Prompt extension** starts with the first of four 512-token batches
  and stops when the backend returns the first greedy token. The rate is
  2,048 divided by that elapsed time. This includes first-token sampling
  and submission of its decode. At 2,048 the history is empty. At 8,192
  the first 6,144 tokens are already in the live KV cache.
- **Generation** measures the 127 token-return intervals between the
  first and 128th sampled tokens. All 24 erllama measurements reached 128
  tokens without sampling an end-of-generation token. The timings
  exclude model loading, HTTP streaming and persistent checkpoint I/O.
- We warmed the backend with 512 prompt tokens and eight decode steps.
  We then ran all four lengths in ascending order and repeated in
  descending order. Each point starts from a cleared sequence and
  rebuilds its history outside the timer. On Metal, an untimed read of
  that history waits for pending GPU work. On the split backend each
  prefill already waits for the reduced logits.
- All three tokenizers received the same text, the repository README from
  commit `2907030`, repeated with a newline between copies. Each model
  took the first N tokens in its own vocabulary. The runs used one
  sequence slot, a 9,216-token context, a 512-token batch, Flash
  Attention and temperature zero.

## Builds and weights

| Setup | Weight file | SHA-256 |
|---|---|---|
| Apple M4 Pro, Metal, macOS 27.2 | Qwen2.5-1.5B-Instruct Q4_K_M | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| Two NVIDIA GB10s, CUDA, NCCL over RoCE | Original GLM-5.3-Flash Q4_K, shard 1 | `a1e9bcead2fce5ca96f5a4ef54dc77ad6dd9ba77ecde57f812e875e6d3e33f07` |
| Same split model | Shard 2 | `27636dd16f76d39aa95db1de081cf458f705f1c78f8f1010b2fac3edc416c116` |

The Mac used the local release-blocker build whose library sources match
`2907030`. The Sparks used the existing `erllama-val` integration build,
one rank per host, with direct I/O for loading. The split GLM and DeepSeek timings include
the rank communication and logits reduction.

## Reproduce

The raw files record durations in microseconds, measured on the Mac or on
the serving Spark. The chart uses the arithmetic mean of the two rates.
[Qwen timings](qwen.csv) · [GLM timings](glm.csv) · [Benchmark source](context_sweep.erl)

From the repository root, build erllama and compile the standalone
driver with OTP 28. The text hash should be
`833bca3212d693f381a8f389b31b96e1d95fe46e03cdde7a21ea1bf13b209a77`.

```sh
rebar3 compile
report=bench/reports/2026-11-25-context-speed
scratch=$(mktemp -d)
git show 2907030:README.md > "$scratch/corpus.md"
export SWEEP_TEXT="$scratch/corpus.md"
erlc -o "$scratch" "$report/context_sweep.erl" \
  apps/erllama/test/erllama_tp_hosts.erl \
  apps/erllama/test/erllama_tp_coordinator.erl \
  apps/erllama/test/erllama_tp_stream.erl
export SWEEP_MODEL=/absolute/path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf
SWEEP_OUTPUT="$scratch/qwen.csv" erl -noshell -pa "$scratch" _build/default/lib/*/ebin \
  -s context_sweep mac -s init stop
```

For GLM, prepare the two hosts as described in
[scripts/tp_hosts.sh](../../../scripts/tp_hosts.sh). The complete GGUF
must be available at the same path on both hosts. Set the output to a
writable path on the **first Spark**, where the measurements run.

```sh
export ERLLAMA_TP_HOSTS=gx10-fa8e.local,gx10-e25e.local
export ERLLAMA_TP_MODEL=erllama-tp/models/glm53-flash-q4k/GLM-5.3-Flash-Q4_K-00001-of-00002.gguf
export ERLLAMA_TP_NCCL_ENV=NCCL_SOCKET_IFNAME=enP7s7
SWEEP_OUTPUT=/tmp/glm-context-speed.csv erl -noshell -pa "$scratch" \
  -s context_sweep split -s init stop
```

With Matplotlib installed, run `python3 "$report/plot.py"` to redraw the
figure from the archived CSVs beside the script.

For DeepSeek, use the same compiled driver and corpus. On **each GX10**,
source the config from the report directory and stage one verified copy.
The temporary download can be resumed if interrupted.

```sh
. ./gx10-deepseek.env
model="$HOME/$ERLLAMA_TP_MODEL"
mkdir -p "$(dirname "$model")"
if [ ! -f "$model" ]; then
    curl --fail --location --retry 3 --continue-at - --output "$model.part" "$DS4_URL" &&
        printf '%s  %s\n' "$DS4_SHA256" "$model.part" | sha256sum --check --status &&
        mv "$model.part" "$model"
fi
printf '%s  %s\n' "$DS4_SHA256" "$model" | sha256sum --check
```

On the controller, during a free GPU window, run from the repository
root with the same `scratch` and `SWEEP_TEXT` as above. The CSV is
written to `SWEEP_OUTPUT` on the first GX10.

```sh
. "$report/gx10-deepseek.env"
erl -noshell -pa "$scratch" -s context_sweep split -s init stop
```
