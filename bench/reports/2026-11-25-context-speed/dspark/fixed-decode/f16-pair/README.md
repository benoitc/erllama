# DeepSeek compressor projection pair on one GB10

The ds4 paired CUDA kernel took longer than two stock ggml matrix-vector operations on all three projection shapes. This trial used one input token and the original Flash 0731 F16 weights. TensorFold stayed resident on the host and reported zero connections at both ends of the uninstrumented run.

| Projections, input × output | Pairs in model | Two stock operations, µs | One stock operation on stacked rows, µs | ds4 paired operation, µs |
|---|---:|---:|---:|---:|
| Attention 4096 × 1024 | 21 | 71.22 | 69.79 | 72.55 |
| Attention 4096 × 512 | 20 | 39.54 | 38.04 | 40.98 |
| Indexer 4096 × 256 | 21 | 23.81 | 22.30 | 25.99 |

Each number is the median synchronized wall time over 48 trials. The probe rotated method order, read 96 MiB to displace the previous weights from L2 before each trial, and prepared the ds4 layout outside the timer. The two stock operations ran as a single ggml graph. A separate Nsight run confirmed the same ordering using GPU kernel durations alone. The weights came from one real layer per shape. The three methods received the same deterministic synthetic activations. Three additional input vectors per shape were checked against FP64 dot products for both outputs. The stacked variant matched the stock outputs bit for bit. The ds4 variant accumulated more accurately against the reference and produced different rounding.

A single stock multiply can compute both projections if the gate rows already follow the KV rows in memory. The observed benefit, weighted by the 62 projection pairs in this model, comes to about 0.09 ms per token before integration costs. This is a projection from isolated timings. The ds4 kernel missed the speed criterion for a two-host trial.

The llama.cpp CLI offers weight repacking by default through `--repack`. In erllama’s pin this permits the loader to use extra CPU buffer types. The DeepSeek graph builds separate KV and gate matrix products. Its CUDA gated-FFN fusion combines two projections and a gated activation into one output. The compressor consumes both outputs separately. [llama.cpp options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) · [ds4 paired CUDA kernel](https://github.com/antirez/ds4/blob/b3f50180ec2c7324892be1f50bc83555ccb8d5e7/ds4_cuda.cu#L4732) · [Wall and GPU timings](summary.csv)
