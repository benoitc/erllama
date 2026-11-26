# DeepSeek V4 Flash 0731 with Q4_K shared experts on GX10

The candidate changes the three shared-expert projections in each of the
43 layers. It quantizes these 129 tensors directly from DeepSeek's official
FP8 weights and block scales to Q4_K using erllama's ggml, without an
importance matrix. All other tensor payloads, the tokenizer and the chat
template come from the pinned ds4 mixed GGUF. The baseline has Q8_0
shared experts. [Official weights](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062)
· [Baseline weights](https://huggingface.co/antirez/deepseek-v4-gguf/tree/f71f23d552d664e523b422157b2befbf74040380)

[Follow-up with Q5_K and the one-versus-two-GX10 comparison](q5/README.md)

## Full-model result

On two GB10s over RoCE, the candidate generated **23.50 tokens/s** across
eight runs and the ds4 baseline generated **22.24 tokens/s**. The candidate
won all eight pairs. Prompt processing, including the first decode, was
620.8 and 622.5 tokens/s respectively. Both used the same erllama CUDA build, direct I/O, greedy decoding, the same README corpus and
disabled checkpoint reuse. Each model ran the four depths in forward and
reverse order. The baseline ran first. The rates pool tokens and elapsed
time across repetitions. Each decode rate covers the 127 intervals after
the first token. All runs generated 128 tokens without encountering EOG.
[Baseline timings](baseline.csv) · [Candidate timings](candidate.csv)

| Context | Baseline decode, tok/s | Candidate decode, tok/s | Change |
|---:|---:|---:|---:|
| 2,048 | 22.74 | 23.78 | +4.6% |
| 4,096 | 22.99 | 24.07 | +4.7% |
| 6,144 | 22.25 | 23.55 | +5.9% |
| 8,192 | 21.09 | 22.64 | +7.4% |

**The candidate failed the agreed quality gate. Keep the ds4 GGUF for serving.** On 56 documents and 57,281 scored
tokens, the equal-workload mean change was +0.0173 nats/token.
The table shows candidate minus baseline NLL, averaged by document,
with paired two-sided 95% t-intervals. Every document had identical
token hashes and scored counts across the two split models.
[Baseline scores](baseline.quality.json) · [Candidate scores](candidate.quality.json)

| Workload | Documents | Mean change | 95% interval |
|---|---:|---:|---:|
| chat | 16 | +0.0167 | [+0.0052, +0.0283] |
| code | 16 | +0.0080 | [+0.0027, +0.0132] |
| multilingual | 16 | +0.0292 | [+0.0051, +0.0534] |
| long | 8 | +0.0151 | [+0.0063, +0.0238] |

The candidate is 86,179,046,272 bytes, saving 541,065,216 bytes.
The other 1,199 tensors were checked byte for byte. [File checksum and changed tensors](manifest.json)

## One-layer result

On one NVIDIA GB10, using the layer-zero matrix dimensions for one rank
of an even two-host split, Q4_K reduced the time for gate, up, clamped
SwiGLU and down together. Timings include graph submission through GPU
synchronization. [All paired timings](timings.csv)

| Input | Pairs | ds4 Q8_0 median | Candidate Q4_K median | Median paired saving | Pairs faster |
|---|---:|---:|---:|---:|---:|
| One token | 100 | 67.8 µs | 44.4 µs | 23.5 µs | 98 |
| 512 tokens | 40 | 328.8 µs | 298.2 µs | 30.3 µs | 39 |

The probe used the same deterministic FP32 activations for both formats.
It warmed up eight pairs per shape, alternated format order and displaced
L2 before each timed graph. Q8_0 was read directly from the baseline.
Its weight RMS difference from the official decoded weights was
0.55–0.56% relative to the source RMS. The output difference between
the two formats on the synthetic activations was 11.9% and 11.7%
relative RMS for the two batch sizes. These are kernel measurements.
The full-model results above include the other layers, routing and rank communication.

## Reproduce

Run in an empty scratch directory on a prepared Spark with the report's
source files copied into it. Use the host's Python environment with gguf
and NumPy installed. The microbenchmark stages byte ranges and uses a
sparse GGUF for reading the tensor directory. Its GPU stage needs an
idle Spark.

```sh
python stage.py
bash build.sh
./shared_probe > timings.csv
```

The measured build used gguf 0.19.0, NumPy 2.5.5 and GCC 13.3.0.
The full build runs on CPU and needs roughly 180 GB of free disk for the
baseline, candidate, and intermediate tensors. Download the baseline
from the parent report's config, verify its hash, and build:

```sh
. /path/to/report/gx10-deepseek.env
curl --fail --location --retry 3 --output baseline.gguf "$DS4_URL"
printf '%s  baseline.gguf\n' "$DS4_SHA256" | sha256sum -c -
base="$HOME/erllama-tp/erllama/apps/erllama"
g++ -O3 -std=c++17 -I"$base/c_src/llama.cpp/ggml/include" quantize.cpp \
    "$base/_build/cmake/llama.cpp/ggml/src/libggml-base.a" -ldl -lpthread -lm -o quantize
python candidate.py quantize
python candidate.py assemble
```

Assembly verifies the baseline SHA-256, reopens the candidate GGUF,
checks its metadata and shapes, and reads back every tensor against
the bytes from its source. The output checksum and changed tensor
names are recorded in `manifest.json`.

For the full-model comparison, use the parent report's context sweep
on both files with the same prompts, warm-up, cache settings and rank
split. Use the frozen 56-document quality set retokenized for DeepSeek.
The agreed limit on the candidate's increase in mean negative log
likelihood is 0.01 nats/token overall, with the upper bound of every
workload's 95% interval at most 0.02. CPU preprocessing found 57,281
scored tokens across the four workloads and located every assistant turn.
