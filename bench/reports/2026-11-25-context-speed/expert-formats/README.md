# DeepSeek V4 Flash expert formats on GX10

25 November 2026. Keep the ds4 IQ2_XXS format for the routed experts’
gate and up projections on the two-GX10 setup. Q4_K took longer in
every paired trial of this one-layer benchmark. Building a full
custom GGUF for this change offers no demonstrated speed benefit.

| Work on one rank | Pairs | Experts touched | IQ2_XXS median | Q4_K median | Q4_K / IQ2_XXS time |
|---|---:|---:|---:|---:|---:|
| Decode, one token | 80 | 6 | 71.9 µs | 156.6 µs | 2.18× |
| Prefill, 512 tokens | 24 | 252–255 | 6.345 ms | 9.461 ms | 1.49× |

Each measurement includes both projections and clamped SwiGLU. The
median paired time added by Q4_K was 84.2 µs for decode and 3.132 ms
for prefill. Q4_K used 576 MiB per projection on this rank, compared
with 264 MiB for IQ2_XXS. Expert selection, attention, the down
projection, communication and sampling are outside the timer.
These results describe layer 0. Generation quality and full-model
throughput require separate measurements. [Raw paired timings](timings.csv)

## Method

We extracted gate and up weights for layer 0 of two pinned 0731 GGUFs
using HTTP byte ranges. Both files have the same token-to-expert table
for this layer. For each of 256 experts we used the first 1,024 output
rows of 2,048, with all 4,096 input features, matching one half of the
even tensor split. Both projections use the same six expert IDs per
token. We tokenized the context sweep’s README corpus with the model
vocabulary and selected different offsets for successive pairs. Hidden
activations are deterministic synthetic FP32 values shared across
formats. [Pinned weight files](https://huggingface.co/antirez/deepseek-v4-gguf/tree/f71f23d552d664e523b422157b2befbf74040380)

The probe links the ggml CUDA libraries from the erllama-val build on
one NVIDIA GB10. It builds two indexed matrix products followed by the
model’s clamped SwiGLU with limit 10. For each batch size we warmed up
four pairs and alternated format order across the timed pairs. Before
each graph, a CUDA kernel read 96 MiB to displace the preceding
trial’s weights from the 24 MiB L2. The host’s monotonic timer covers
graph submission through GPU synchronization. Inputs, weights, and
cache eviction are outside the timing.

Separate Nsight captures measured one warmed CUDA graph per case.
IQ2_XXS and Q4_K took 81 and 160 µs on the GPU for decode, and 6.176
and 9.272 ms for prefill. Nsight captured graph spans only. Individual
kernel names and the cause of the latency difference remain unverified.
[GPU graph timings](graphs.csv)

## Reproduce

Use the project’s existing CUDA build on a GX10 prepared with
`scripts/tp_hosts.sh`. Copy this directory into a fresh scratch
directory on the GX10. Provide the same corpus and run from scratch:

```sh
git -C /path/to/erllama show 2907030:README.md > corpus.md
bash inputs.sh
bash build.sh
./expert_probe > timings.csv
```

`inputs.sh` extracts only the required tensors and creates a sparse
file for the native tokenizer. Allow about 2 GiB of physical disk and
use a filesystem supporting sparse files. The timed run needs an idle
GPU. To capture individual CUDA kernel names on a repeat, use Nsight’s
`--cuda-graph-trace=node` along with `--capture-range=cudaProfilerApi`
and run `./expert_probe QUANT BATCH` with QUANT 0 for IQ2_XXS or 1 for
Q4_K, and BATCH 1 or 512. Discard the scratch files after saving the CSVs.
