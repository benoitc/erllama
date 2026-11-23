# Tensor parallelism across hosts

This page is for contributors working on cross-host tensor parallelism:
several machines, one GPU each, computing parts of the same layers for
the same request. Read it before changing `erllama_tp`, the rank NIFs
in `c_src/erllama_tp_nif.cpp` or the tests that drive them. The work
is experimental and comes in milestones; this page describes what
exists now (a reduction between ranks, and a model split over them)
and how failures behave.

## The model

- **A rank is an OS process.** Each rank is an Erlang node with one
  GPU. The GPU backend aborts the process when a collective fails
  (`NCCL_CHECK` in the CUDA backend), so a separate process per rank
  keeps that failure to one rank and lets the coordinator see it as a
  node going down.
- **erllama holds only rank-local resources.** `erllama_tp` opens this
  node's GPU backend, joins the backend's communicator and runs
  collectives. Forming a group (which ranks, when, which generation)
  is the coordinator's job: `erllama_cluster` later, a test coordinator
  (`test/erllama_tp_coordinator.erl`) here.
- **Tensor data stays native.** Only the communicator's bootstrap
  records cross Erlang. Reductions go GPU to GPU through the backend's
  communicator (NCCL for CUDA).

## Joining a group

The pinned llama.cpp gives each backend an optional
`ggml_backend_comm_init_rank(backend, rank, world_size, callback,
data)` (`ggml/include/ggml-backend.h`). The CUDA backend has it when
built with NCCL. It calls `callback(data, send, recv, size)` one or
more times; each call must fill `recv` with every rank's `size` bytes
in rank order. CUDA uses one round, carrying each rank's local checks
and rank 0's NCCL unique id, so all ranks join or fail together.

erllama connects that callback to Erlang:

```
rank process            agent process             coordinator
erllama_tp:open/5       (serves the bridge)       (one per group)
  nif_tp_open  ──callback──> {erllama_tp_bootstrap, Bridge, Round, Bytes}
  (dirty IO, waits                Join(Round, Bytes) ──> {exchange, Rank, Round, Bytes}
   until its deadline)                               <── every rank's bytes, rank order
               <──── nif_tp_bootstrap_reply(Bridge, Round, All)
```

- The callback runs on the dirty thread of `nif_tp_open`, sends the
  bytes to the agent and waits on a condition variable until the
  open's deadline. A reply of the wrong size, an `error` reply or the
  deadline ends the wait, and `init_rank` returns no communicator.
- `Join` is a fun, or `{call, Server}` for a coordinator that answers
  `{exchange, Rank, Round, Bytes}` calls, possibly on another node.
- The test coordinator answers each round once every rank's record is
  in, and fails the group if a round is not complete in time.

## Using a rank

```erlang
{ok, R} = erllama_tp:open(0, Rank, World, {call, Coordinator}, 60000),
{ok, Partial, Sum} = erllama_tp:matvec(R, Rows, X),   %% this rank's columns
{ok, Summed} = erllama_tp:reduce(R, Values),          %% same length on every rank
ok = erllama_tp:close(R).
```

`erllama_tp_rank` wraps a handle in a process, so a coordinator on
another node can drive the rank by pid. Every rank must make the same
collective calls in the same order, with buffers of the same length.

The CUDA backend reduces fewer than 32768 elements (two ranks) in FP32,
and larger buffers in BF16: each partial is rounded to BF16, summed,
and expanded back. A buffer handed to the all-reduce must carry
`GGML_TENSOR_FLAG_COMPUTE`, or the backend treats it as no contribution
and zeroes it; `nif_tp_reduce` sets it.

## Splitting a model over the ranks

A model is loaded with llama.cpp's own tensor split
(`LLAMA_SPLIT_MODE_TENSOR`) over one device per rank, in rank order:
load it with `tp_rank => Rank` (an open erllama_tp handle) and every
rank loads the same file with the same options. erllama supplies the
devices (`c_src/erllama_tp_nif.cpp`, built on ggml's backend
interfaces):

- the device at this rank's slot wraps the local GPU and forwards
  everything to it, except that its backends report the wrapper, so the
  split backend asks the wrapper's registry for its collectives;
- the devices at the other slots stand for the other ranks: their
  buffers reserve address space (`PROT_NONE`) and no memory, their
  graphs compute nothing and their tensors read as zeros.

So each rank holds and computes only its slices, from the same split
rules (`llama_meta_device_get_split_state`), in the split backend's
order, and each all-reduce the split backend asks for is this rank's
tensor through the rank's communicator, at the same point on every
rank. No vendored file is changed.

Two things the split backend does not do across processes:

- **Stream order.** The communicator runs on the CUDA stream of the
  backend it was created with, not the one the model computes on. CUDA
  events order the two around every reduction.
- **The output head.** `output.weight` is split along the vocabulary,
  so each rank computes part of the logits (the other parts read as
  zeros). `erllama_nif:tp_decode/2` sums them over the ranks in chunks
  below the FP32 threshold, so the sum is exact and every rank holds
  the full logits; rank 0 samples.

The devices are never registered with ggml: llama.cpp's default device
selection would otherwise pick them up for ordinary loads. A rank stays
open while a model is loaded on it (`erllama_tp:close/1` answers
`{error, in_use}`).

`erllama_tp_rank` drives this for tests: `load/3`, `decode/2` (every
rank at once), `stats/1` (the reductions made, with their sizes in
order, and what the placeholders reserve) and a single-GPU reference
copy (`ref_load/3`, `ref_decode/2`).

## What a failure does

| Failure | Result |
|---|---|
| a rank never joins | the coordinator fails the round: every joined rank gets `{error, join_timeout}` |
| the coordinator never answers | each rank's wait ends at its open deadline: `{error, timeout}` |
| a rank fails its local checks, or NCCL's init fails | every rank's `init_rank` returns no communicator: `{error, init_failed}` |
| a build without per-process communicators | `{error, not_supported}`; `erllama_tp:caps/0` reports it |
| a failure inside a collective (a lost peer) | the backend aborts that rank's OS process; its node goes down |
| a rank blocked in a native call | ended by stopping its node; an Erlang timeout never ends a native call |

After any failure the communicator is discarded and the group is
formed again, under a new coordinator (a new generation).

## Tests

- `test/erllama_tp_bootstrap_SUITE.erl` runs everywhere: the bridge,
  the agent and the coordinator through
  `erllama_tp:bootstrap_selftest/5`, which makes the same callback
  rounds as a communicator's init without a GPU. Ranks in rank order,
  a missing rank, a silent coordinator, a rank lost between rounds, a
  duplicate rank, and `not_supported` without a communicator.
- `test/erllama_tp_nccl_SUITE.erl` is opt-in, on CUDA hosts prepared
  with `scripts/tp_hosts.sh`. The plan's matrix fixture, a
  decode-sized FP32 reduction and a prefill-sized BF16 one with stated
  tolerances, a missing rank, repeated open and close, a GGUF split
  evenly and unevenly against the same model on one GPU, and the
  transport NCCL chose.

```bash
scripts/tp_hosts.sh setup gx10-fa8e.local gx10-e25e.local   # once: NCCL, OpenSSL, OTP, rebar3
scripts/tp_hosts.sh sync  gx10-fa8e.local gx10-e25e.local
scripts/tp_hosts.sh build gx10-fa8e.local gx10-e25e.local   # CUDA + NCCL
ERLLAMA_TP_HOSTS=gx10-fa8e.local,gx10-e25e.local \
    rebar3 ct --dir apps/erllama/test --suite erllama_tp_nccl_SUITE
```

The suite runs on any machine with ssh access to the hosts. The hosts
must reach each other by the names given (rank 0's node runs the
coordinator), and the build must be `rebar3 as test compile` so the
rank nodes load the test build.

## When you bump llama.cpp

The devices implement ggml's internal backend interfaces
(`ggml/src/ggml-backend-impl.h`), which change between pins (new entries
in the buffer type interface between b10593 and v1.0.0), and rely on
how the split backend behaves (`ggml/src/ggml-backend-meta.cpp`, the
fastest-moving file). A changed interface fails the build; a changed
behaviour fails `erllama_tp_nccl_SUITE`, so run it on the CUDA hosts at
every bump.

## Not done yet

Serving through the normal erllama path (load, stream, cancel) with one
sampling rank driving the others, failure handling during generation,
coordinated cache checkpoints, a model too large for one GPU, groups of
more than two hosts, and Metal.
