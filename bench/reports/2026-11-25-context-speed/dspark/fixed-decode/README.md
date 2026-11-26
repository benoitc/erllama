# Where ordinary DeepSeek decoding spends its time on two GX10s

On 25 November 2026 erllama took 45.05 ms per token and ds4 took 34.15 ms on the same 32 positions. Both engines picked the same greedy token at all 32 positions. Switching off ds4’s aligned IQ2/Q2 expert kernels raised its time to 37.69 ms. This establishes a 3.53 ms benefit within ds4. A 7.36 ms gap to erllama remained in that comparison.

| Mean ms per token | erllama | ds4 |
|---|---:|---:|
| Ordinary decode, profiler off | 45.05 | 34.15 |
| Ordinary decode, profiler on | 43.15 | 34.52 |
| Rank 0 GPU busy during traced decode | 40.54 | 28.84 |
| Rank 1 GPU busy during traced decode | 39.68 | 28.25 |

## GPU work

Mean GPU kernel time in ms per token on rank 0. Each row groups kernels by their recorded GPU symbol and weight format. For example, the Q8 row includes several different projections. The two engines can fuse different operations into one kernel.

| Operation | erllama | ds4 |
|---|---:|---:|
| Q8 weight matrix-vector products | 18.24 | 16.38 |
| F16 weight matrix-vector products | 6.35 | 3.30 |
| IQ2/Q2 routed expert projections | 5.58 | 5.33 |
| Attention score and output kernels | 0.70 | 0.71 |
| Other GPU kernels | 6.58 | 3.10 |
| GPU copies and fills | 0.29 | 0.01 |
| NCCL all-reduce kernels, including waits | 5.25 | — |

Rank 1 showed the same pattern. erllama ran 4,429 GPU kernels per token on each rank. ds4 ran about 1,649 per rank. This counts GPU kernel executions, including nodes of CUDA graphs. It does not measure CPU launch cost. ds4 uses its own RDMA protocol, whose time is included in its wall clock. Its lack of an NCCL kernel gives no estimate of network savings. GPU kernel durations can overlap, and profiler overhead and run variation affect the two engines differently. The rows should not be subtracted to make a wall-time budget.

The named routed kernels on rank 0 differ by only 0.25 ms between the engines at their normal settings. The separate 3.53 ms result measures what ds4 loses when its own optimized implementation is disabled. Q8 and F16 matrix-vector kernels together differ by 4.91 ms. In the ds4 trace, a paired F16 kernel ran 62 times per token. Its DeepSeek code can compute the compressor’s KV and gate projections from the same input. A concrete first kernel trial is to benchmark those two projections together on GB10 using the existing F16 weights, then check the output and full decode time. This removes the cost of changing or evaluating another GGUF. [ds4 implementation](https://github.com/antirez/ds4/blob/b3f50180ec2c7324892be1f50bc83555ccb8d5e7/ds4.c#L25081)

On rank 0, erllama’s CPU-side sampling call took **0.12 ms**. The function that sums the final vocabulary scores across ranks took **2.75 ms** on each rank. That call includes staging buffers, copies and communication, and overlaps some of the GPU time above. The trace does not isolate time on the wire.

## Procedure

One coding prompt from the [serving comparison](../README.md), 70 input IDs, 16 single-token warm-up decodes and 32 measured decodes. Both hosts used the original 86.72 GB mixed GGUF (SHA-256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`), a 2,048-token context, one sequence and RoCE. Each leg started a fresh rank group. The target was fully loaded before timing. Drafting was off.

erllama used its ordinary rank step and asserted the sampled token against the archived sequence. The ds4 bench driver read the same IDs, computed greedy argmax, and evaluated the specified token. All measured argmax values matched in all five legs. Each clock covers token selection and evaluation. The erllama step also includes its Erlang rank call. These are single runs per configuration on one fixed sequence.

Nsight Systems 2025.3.2 captured CUDA graph nodes on both ranks. NVTX markers isolated the 32-token interval. Additional markers enclosed erllama’s native decode, sampler and host logits sum in the profiled run. GPU busy time is the union of kernel and copy intervals. The ds4 control used `DS4_CUDA_MOE_NO_IQ2_ALIGNED=1` and `DS4_CUDA_MOE_NO_Q2K_ALIGNED=1` on both ranks.

[Fixed token IDs](fixed.ids) · [Per-step timings and token checks](steps.csv) · [Both GPUs by operation](gpu.csv) · [Individual kernel names](kernel-times.csv) · [Native function timings](native.csv)

The [one-GPU projection trial](f16-pair/README.md) found the ds4 paired F16 kernel slower on all three measured shapes.
