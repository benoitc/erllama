# Tensor parallelism across hosts

This page is for contributors working on cross-host tensor parallelism:
several machines, one GPU each, computing parts of the same layers for
the same request. Read it before changing `erllama_tp`, the rank NIFs
in `c_src/erllama_tp_nif.cpp` or the tests that drive them. The work
is experimental and comes in milestones; this page describes what
exists now (milestone 1: a reduction between ranks) and how failures
behave.

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
  tolerances, a missing rank, repeated open and close, and the
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

## Not done yet

Running a model across ranks (milestones 2 to 4): rank-local execution
of a transformer block from the pinned split rules, the full model with
one sampling rank, failure handling during generation and coordinated
cache checkpoints. Groups of more than two hosts and Metal are not
qualified.
