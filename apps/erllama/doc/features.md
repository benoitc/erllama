# erllama: feature record

Library milestones that span several releases (tensor parallelism,
drafting with a model's MTP head) with what was measured on which
hardware. The CHANGELOG says what each release changes for you; this
page keeps the runs behind it. Later entries supersede the "Not done"
lines of earlier ones. The server's record is in
[apps/erllama_server/doc/features.md](../../erllama_server/doc/features.md).

## Tensor parallelism, milestone 1: a reduction between hosts (2026-11-23)

Experimental. See [internals/tensor-parallel-design.md](../../../internals/tensor-parallel-design.md).

| Feature | State | Notes |
|---|---|---|
| rank-local API | done | `erllama_tp`: open this node's GPU as one rank, matvec, reduce, close; `erllama_tp_rank` holds a handle in a process for a remote coordinator |
| bootstrap bridge | done | the communicator's bootstrap callback (`ggml_backend_comm_init_rank`) goes through an Erlang agent and a coordinator, rank-ordered, bounded by the open's deadline |
| group coordinator | done | `erllama_tp_coordinator` exchanges bootstrap records and watches the ranks for one generation |
| portable tests | done | `erllama_tp_bootstrap_SUITE`, in CI |
| hardware tests | done on two hosts | `erllama_tp_nccl_SUITE`, opt-in; hosts prepared with `scripts/tp_hosts.sh` |

Hardware run: two ASUS GX10 (NVIDIA GB10, compute capability 12.1, about 121 GB unified memory, driver 580.178.04, CUDA 13.0), one rank each; llama.cpp pin v1.0.0; NCCL 2.32.6 (pip wheel `nvidia-nccl-cu13`); OTP 28.3.1.

| Case | Result |
|---|---|
| fixture: rows `[1,2,3,4]`, `[4,3,2,1]`, input `[1,2,3,4]`, columns split | partials `[5,10]` and `[25,10]`; both ranks `[30,20]`, exact |
| 4096 values per rank (FP32 path) | tolerance 1e-6 of the summed magnitudes + 1e-6; worst error 0.02 of the bound |
| 2,097,152 values per rank (BF16 path) | tolerance 1.6e-2 of the summed magnitudes + 1e-6; worst error 0.29 of the bound |
| rank 0 alone | `{error, join_timeout}` after 2031 ms (coordinator deadline 2000 ms); the node stays usable |
| open, reduce, close three times | each round correct; free device memory flat (rank 0 about 5.2 GB free, rank 1 about 7.7 GB, another process holding the rest) |
| data path | NCCL `NET/IB` over RoCE, `rocep1s0f0` and `roceP2p1s0f0` at 200 Gb/s each (back-to-back links), out-of-band on `enP7s7`; GPU Direct RDMA disabled on GB10 |

The run is from a third machine:

```bash
scripts/tp_hosts.sh setup gx10-fa8e.local gx10-e25e.local
scripts/tp_hosts.sh sync  gx10-fa8e.local gx10-e25e.local
scripts/tp_hosts.sh build gx10-fa8e.local gx10-e25e.local
ERLLAMA_TP_HOSTS=gx10-fa8e.local,gx10-e25e.local \
    rebar3 ct --dir apps/erllama/test --suite erllama_tp_nccl_SUITE
```

Not done: a transformer block split across ranks (milestone 2), a full model (milestone 3), failure handling during generation and coordinated checkpoints (milestone 4), three or more hosts, Metal. No throughput or latency is claimed: nothing was measured beyond correctness.

## Tensor parallelism, milestone 2: a model split over the hosts (2026-11-23)

Experimental. A GGUF loaded with `tp_rank => Rank` is split with llama.cpp's own tensor split over one device per rank: this rank's GPU, wrapped, and placeholders for the others that hold and compute nothing. Collectives go through the rank's communicator; the logits are summed over the ranks exactly. No vendored file is changed. See [internals/tensor-parallel-design.md](../../../internals/tensor-parallel-design.md).

Run on the two GB10 hosts (pin v1.0.0, NCCL 2.32.6 over RoCE), TinyLlama 1.1B Chat (Llama architecture, Q4), `n_ctx 512`, flash attention on, prompt "The quick brown fox jumps over the lazy dog" (12 tokens), then one greedy token from the cache. Reference: the same file on one GB10.

| Case | Prompt | Next token from the cache | Reductions | Placeholders reserve (the other rank's share) | Free GPU memory used by the load |
|---|---|---|---|---|---|
| even split | max \|logit − one GPU\| 0.272 (largest \|logit\| 13.96), top 5 identical | 0.220 (largest 14.41), top 5 identical | 88 on each rank, same sizes in the same order | 385.3 MiB on each | 436 MiB and 412 MiB |
| uneven split (`tensor_split => [3.0, 1.0]`) | 0.246, top 5 identical | 0.213, top 5 identical | 88 on each, same order | 232.5 MiB on rank 0, 538.1 MiB on rank 1 | 605 MiB and 303 MiB |

- Tolerance, set before comparing: the same argmax, and a largest logit difference within 2% of the largest logit. Both ranks hold identical logits.
- 88 reductions: two per layer (attention output and feed-forward output) over 22 layers, for each of the two decodes.
- Memory is the free memory reported by the GPU before and after the load (unified memory shared with another process), so it is approximate; the placeholder figure is exact.
- The differences come from the split itself (matrix shapes change the CUDA kernels used); they are not shown to be rounding only. Throughput and latency were not measured.

Not done: serving through the normal erllama path with one sampling rank driving the others, failure handling during generation, coordinated checkpoints, a model larger than one GPU, more than two hosts, Metal.

## Tensor parallelism, milestone 3: serving a split model (2026-11-23)

Experimental. Rank 0 loads the split model through `erllama:load_model/2` (`tp_rank` and `tp_sink` options) and serves it with the normal API; it samples once per step. Every batch and KV change is sent, numbered, to the other ranks, which apply them in order. Ranks with a different file, options or engine build are refused. A split model saves no KV yet. See [internals/tensor-parallel-design.md](../../../internals/tensor-parallel-design.md).

Run on the two GB10 hosts over RoCE, TinyLlama 1.1B Chat Q4, `n_ctx 1024`, flash attention on, one sequence, greedy; reference: the same file loaded through erllama on one GB10.

| Check | Result |
|---|---|
| greedy completion of "The quick brown fox", 32 tokens | identical to one GPU, all 32 |
| stream of 128 tokens after a 242-token prompt | ends `length`, 128 tokens |
| cancel after 10 tokens | ends cancelled; the next request returns the same 32 greedy tokens |
| followers in step | 9196 reductions on each rank, same sizes in the same order |

| 242-token prompt, 128 tokens | First token | Prompt tokens/s | Decode tokens/s | Median gap |
|---|---|---|---|---|
| split over the two hosts (two runs) | 39.0 and 53.1 ms | 6202 and 4559 | 233.7 and 172.8 | 4.17 and 5.75 ms |
| one GB10 (two runs) | 22.7 and 22.1 ms | 10673 and 10935 | 288.9 and 290.8 | 3.41 ms |

- A 2048-value reduction (FP32), timed in each rank's process with the copy to and from the GPU: about 230 µs. Each token makes 44 reductions inside the model graph (two per layer), without the copies.
- For a model this small, splitting is slower than one GPU: the time saved computing half of each layer is smaller than the time spent reducing. The point of splitting is a model that does not fit one GPU; that run is still to be made (the test hosts' GPUs are shared with another workload).
- The split runs vary between runs (39 to 53 ms first token); the one-GPU runs do not. Only these configurations were measured.

Not done: failure handling during generation (a rank lost mid-stream, a stale or duplicate step, cancel during a collective), coordinated checkpoints and cache hits, a model larger than one GPU, more than two hosts, Metal.

## Tensor parallelism over plain TCP (2026-11-23)

Experimental. A group runs over ordinary Ethernet with NCCL's socket transport: set `NCCL_NET=Socket` and `NCCL_SOCKET_IFNAME` in each rank node's environment. Nothing in erllama changed; the launch recipe, the diagnostics and the checks are in [internals/tensor-parallel-design.md](../../../internals/tensor-parallel-design.md).

Same two GB10 hosts, same build, TinyLlama 1.1B Chat Q4, `n_ctx 1024`, a 242-token prompt and 128 streamed tokens, rank 0 serving (`erllama_tp_nccl_SUITE`, all 9 cases passed in each configuration). Three configurations: TCP over the two 200 Gb/s links (`NCCL_SOCKET_IFNAME==enp1s0f0np0,enP2p1s0f0np0`), TCP over the 2.5 Gb/s management port (`==enP7s7`), and RoCE (no NCCL variable set).

| What NCCL used | TCP, 200 Gb/s links | TCP, 2.5 Gb/s port | RoCE |
|---|---|---|---|
| network | `Using network Socket` | `Using network Socket` | `Using network IB` |
| devices | `[0]enp1s0f0np0 [1]enP2p1s0f0np0` | `[0]enP7s7` | `[0]rocep1s0f0 [1]roceP2p1s0f0` (RoCE) |
| channels | 8, `via NET/Socket/0` and `/1` alternating | 4, `via NET/Socket/0` | 8, `via NET/IB/0` and `/1` alternating |
| bytes sent per host by 17 reductions of 2M values (BF16) | 37.4 and 37.3 MB, nothing on the 2.5 Gb/s port | 74.7 MB | none in the interface counters (RDMA) |
| one such reduction | 7.2 ms | 18.7 ms | 4.5 ms |

| Two runs each | First token | Decode tokens/s | Median gap | 2048-value reduction |
|---|---|---|---|---|
| TCP, 200 Gb/s links | 132.8 and 126.8 ms | 65.6 and 55.8 | 14.90 and 18.18 ms | 342 and 581 µs |
| TCP, 2.5 Gb/s port | 197.5 and 213.8 ms | 43.3 and 47.5 | 24.24 and 21.54 ms | 421 and 492 µs |
| RoCE | 45.2 and 48.7 ms | 177.1 and 228.9 | 5.70 and 4.35 ms | 163 and 118 µs |
| one GB10, no split | 21.3 to 22.9 ms | 283.9 to 289.5 | 3.42 to 3.50 ms | |

- In every configuration the greedy completion was identical to one GPU for all 32 tokens, the 128-token stream finished, a cancel after 10 tokens ended cleanly and the next request gave the same tokens, and both ranks made 9196 reductions in the same order.
- Decoding is bound by the latency of the 44 small reductions per token, not by bandwidth: TCP over the 200 Gb/s links is only 1.2 to 1.5 times faster than over the 2.5 Gb/s port, and 3 to 4 times slower than RoCE.
- The NIC speeds are the hosts' reported link speeds (`/sys/class/net/*/speed`). The two hosts' GPUs are shared with another workload; the runs vary.

Not run over TCP: the rank-loss, stale-command and checkpoint cases, which come with milestone 4.

## Tensor parallelism, milestone 4a: failures while serving (2026-11-23)

Experimental. The split model is now served from a node that is no rank (`erllama_model_tp`): tokenizing and chat templates run there on the GGUF's vocabulary, and decoding, sampling and KV changes run on rank 0. Commands to the other ranks carry the group's generation; a stale or duplicate one is dropped before any GPU work, a gap stops the follower. A split model's decode is not interrupted: a cancel waits for the step in flight. When a rank goes, the coordinator kills the others and the model fails its requests once and stops. See [internals/tensor-parallel-design.md](../../../internals/tensor-parallel-design.md).

Run on the two GB10 hosts over RoCE, TinyLlama 1.1B Chat Q4, `n_ctx 2048`, `n_batch 512`, flash attention on, greedy; the serving node on the first host. Each case starts its own nodes (`erllama_tp_failure_SUITE`).

| Case | Result |
|---|---|
| rank 1's OS process killed after 10 streamed tokens | one terminal error, `{tp_rank_lost, {nodedown, rank 0}}`, 6.3 ms after the kill (the coordinator killed rank 0, held in the collective); the serving node lists its models in 8 ms; the model has stopped |
| a new group (new nodes, generation 2) | the same 32 greedy tokens as the first group |
| a command of the previous generation, then one already applied | both dropped (stale 1, duplicate 1); no reduction made; the next request gives the same tokens |
| a command past a gap | the follower stops (`{out_of_order, 70, 75}`); the next request fails once in 540 ms and the model stops |
| cancel right after a 962-token prompt is admitted | one terminal result, `cancelled`, after the first 128-token step; 2948 reductions on each rank, same order; the next request gives the same tokens |

Serving from the separate node, `erllama_tp_nccl_SUITE` (same model, `n_ctx 1024`, 242-token prompt, 128 tokens): greedy identical to one GPU for all 32 tokens; first token 64.3 ms, decode 150.6 tokens/s, median gap 6.57 ms (one GB10: 34.2 ms, 287.9 tokens/s, 3.42 ms); 9196 reductions on each rank. Each step now crosses the serving node and rank 0 once more than in milestone 3 (rank 0 serving: 173 to 234 tokens/s); one run each, so the share of that hop is not measured apart from the variation between runs.

Over TCP (`NCCL_NET=Socket` on the two 200 Gb/s links) the three cases pass too: the terminal error came 6.3 ms after the kill, the request after a gap failed once in 119 ms, and the cancel during prefill ended once, 68 ms after the cancel.

Not done: coordinated checkpoints and cache hits, a model larger than one GPU, more than two hosts, Metal.

## Tensor parallelism, milestone 4b: checkpoints of a split model (2026-11-23)

Experimental. A split model's cache entry holds every rank's share of the KV, packed at the same committed token: rank 0 puts the pack in the stream the other ranks follow, after every batch it decoded, and frames the shares (size and CRC32C each) into one payload, so the cache publishes all of them or none. A share that does not come within `tp_kv_timeout` fails the save and nothing is published. The world size and `tensor_split` are part of the KV identity. See [internals/tensor-parallel-design.md](../../../internals/tensor-parallel-design.md).

Run on the two GB10 hosts over RoCE, TinyLlama 1.1B Chat Q4, `n_ctx 2048`, a 330-token prompt, 32 greedy tokens; one cold save at the 320-token boundary and a finish save (`erllama_tp_checkpoint_SUITE`, two runs).

| Case | Result |
|---|---|
| a checkpoint, then the group unloaded and a new one formed on new nodes | the same prompt resumes from it (`partial`, 320 tokens read, restored on both ranks), same 32 tokens as the first group |
| rank 1 sends no share for the cold save | that save fails and the finish save is made (rank 0: 1 failed, 1 made); a new group misses on the same prompt (`cold`, nothing read) and gives the same tokens |

| | Time on rank 0 | Size |
|---|---|---|
| checkpoint (pack on both ranks, gather, frame) | 78 to 105 ms | 17.8 MB |
| restore (check, unpack on both ranks) | 49 and 68 ms | |

- Each rank's share is the size of the whole sequence's KV: the split backend reads the other ranks' slices as zeros. The checkpoint is twice the single-GPU size with two ranks.
- Timed apart from decoding, in rank 0's process; the cache's own framing and publication come on top.
- Over TCP (`NCCL_NET=Socket` on the two 200 Gb/s links) both cases pass too, with the same hits and misses (checkpoint 81 to 93 ms).

## Tensor parallelism: a model larger than one GPU (2026-11-23)

Experimental. Llama 3.3 70B Instruct in F16 (four GGUF parts, 131.4 GiB, 80 layers) on the two GB10 hosts, each with 121 GB of memory shared by its CPU and GPU, the hosts' other workload stopped for the run (`erllama_tp_capacity_SUITE`). `n_ctx 2048`, flash attention on, over RoCE, served from its own node through `erllama_model_tp`.

| | One GB10 | Split over the two |
|---|---|---|
| placement | auto-fit (8 GiB margin) chose 68 of 81 layers on the GPU, 13 on the CPU | 81 of 81 layers on the GPU of each rank, each holding its share; only the 2.0 GB token embedding table in CPU memory, as llama.cpp always places it |
| result | the load ran the host out of memory: the kernel killed the node (`Out of memory: Killed process ... beam.smp`, 144.6 GB of address space) | loaded and served |
| memory available before | 117 GB | 125.4 and 125.5 GB |
| peak use during load and generation | (killed) | 74.3 GB on rank 0, 73.9 GB on rank 1 |
| per rank | | 66,287 MiB of weights (its half), 320 MiB of KV, 317 MiB of compute buffers; the other rank's share (70.2 GB) reserved as address space only |

| Streamed completion, 64 tokens after a 220-token prompt (two runs) | First token | Decode | Median gap |
|---|---|---|---|
| split over the two GB10s | 1635 and 1582 ms | 3.34 and 3.33 tokens/s | 299.5 and 300.0 ms |

- Both streams ended `length` with 64 tokens. There is no one-GPU reference to compare the tokens with: no single host holds the model.
- Decoding reads each rank's 66 GB of F16 weights once per token, so the rate is bound by memory bandwidth, not by the reductions; a quantized file would be faster and smaller.
- The one-host case runs only with `ERLLAMA_TP_CAPACITY_ONE_HOST=1`: on hosts whose GPU shares the host's memory it ends with the node killed.

Not done: more than two hosts, Metal.

## Tensor parallelism: GLM-5.3-Flash over two GB10s (2026-11-23)

Experimental. GLM-5.3-Flash (`glm5-next`, 320B parameters with 18B active, 45 layers; MLA and KDA linear attention, a sparse-attention indexer) split over the two GB10 hosts with llama.cpp's own tensor split and served through `erllama_model_tp`, over RoCE, drafting off. Files: `ggml-org/GLM-5.3-Flash-GGUF` Q4_K (181 GB, revision `93ae818`) for speed and capacity, `unsloth/GLM-5.3-Flash-GGUF` UD-IQ2_XXS (102 GB, revision `a38483c`, small enough for one GB10) for correctness (`erllama_tp_glm_SUITE`). See [internals/tensor-parallel-design.md](../../../internals/tensor-parallel-design.md).

What it took:

- The other ranks now open the same context as rank 0: rank 0 adds recurrent-state snapshots for hybrid models, the others did not.
- Split loads keep no recurrent-state snapshots by default: without the local llama.cpp fix that came later (see the MTP entry below), the split backend aborted every rank at the first decode (a view of the recurrent cache it could not split). Without snapshots a hit on a cached prefix followed by new tokens still restores; a hit on a whole prompt is served cold. A split load can now keep them when asked (`n_rs_seq`).
- `load_mode => direct_io`: on a GB10 the page cache of a buffered load is not given back to CUDA, and a CUDA init failed with out of memory while the host showed 117 GB available.

| Correctness (UD-IQ2_XXS, split against one GB10) | Result |
|---|---|
| next token at the prompt and three decodes | identical at every step; top 5 identical or 4 of 5 |
| largest logit difference | 4.4% at the prompt (its reductions go through BF16), growing to 7.1% by the third decode |
| 32-token greedy replies to four prompts, alone | 3 of 4 byte-identical; the fourth is a near-tie that batching alone changes on one GB10 |
| reductions | 360 on each rank, same sizes, same order |
| four prompts at once | replies differ from the solo ones, and between two batched runs, on one GB10 as on the split model (when each request is admitted changes what a step computes); a cancel midway ends only that request |
| checkpoint, then a new group | the same prompt resumes from it (`partial`, 256 tokens restored on both ranks) with the same 32 tokens; 318 MB, 1.55 s to pack, 0.73 s to restore |

| Q4_K, 250 tokens, TensorFold's four prompts (`docs/dec.py`), 3 trials | erllama split, no drafts | TensorFold, no drafts | TensorFold, MTP | TensorFold, DFlash2 |
|---|---|---|---|---|
| 1 stream | 19.9 to 20.5 tokens/s (first token about 295 ms) | 28.8 | 41.3 | 40.9 |
| 4 streams, total | 39.7 to 40.6 tokens/s | 28.5 | 61.9 | 78.5 |

- Memory with Q4_K loaded, `n_ctx 8192`, 4 sequences: 30 to 32 GB left available on each GB10 (125 to 126 before).
- TensorFold runs a 4-bit EXL3 quant with FP8 KV and its own kernels; tokens are compared within one engine, never across.
- Not done then: speculative decoding (TensorFold's single-stream lead is mostly drafts; done since, see below), a longer context, vision, and FP8 KV (refused by the pin for this model).

## Drafting with the MTP head, on one GPU and on a split model (2026-11-24 to 2026-11-25)

Experimental. See [internals/tensor-parallel-design.md](../../../internals/tensor-parallel-design.md) for the split-model rounds.

| Feature | State | Notes |
|---|---|---|
| MTP drafting | done | `speculative => #{type => mtp}` with the head's GGUF as `draft_model`; the head guesses up to `n_max` tokens (default 2, at most 16) and the model checks them in one step; greedy requests only |
| on a split model | done | the head runs on rank 0; every rank decodes the known token and the guesses in one batch, and the rejected rows are taken back on every rank |
| recurrent-state snapshots on a split hybrid model | done | a local fix to the vendored llama.cpp lets the split backend keep them (`n_rs_seq`); MTP on GLM-5.3-Flash needs `n_rs_seq => 2` on every rank |
| several requests at once | done | with two or more requests (at most 16), a step checks every request's next token and the guesses of those that draft in one decode, with each drafting request's guesses shortened so the batch fits; a request that cannot draft joins without guesses and does not stop the others from drafting |

Measured on GLM-5.3-Flash Q4_K split over the two GB10 hosts, one stream went from 19.5 tokens/s without drafts to 26.2 with them; a round takes about 85 ms (head 10 ms, verify batch 73 ms, rollback 1 ms) and produces 2.1 to 2.4 tokens. The text can differ from decoding without drafts where the two most likely tokens are almost tied.

Not done: testing on more than two hosts, Metal.
