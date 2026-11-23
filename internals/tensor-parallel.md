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

## Serving a split model

The model is served from a node that is no rank: the serving VM loads
it with `backend => erllama_model_tp` (`src/erllama_model_tp.erl`), so
callers use the normal API (`complete`, `stream`, `cancel`), and every
rank, the sampling one included, is a separate OS process. A
collective that fails or hangs then takes down a rank, never the
serving VM.

```erlang
%% On the serving node, once the group is formed and every rank but 0
%% has run erllama_tp_rank:load/3 with the same file and options:
{ok, Id} = erllama:load_model(<<"split">>, #{
    backend => erllama_model_tp,
    model_path => Path,
    model_opts => MOpts,
    context_opts => COpts,
    tp_worker => Rank0Pid,                     %% rank 0's erllama_tp_rank
    tp_followers => [{Pid, erllama_tp_rank:describe(Path, {MOpts, COpts}, World)}]
}).
```

- **What runs where.** Tokenizing, detokenizing and chat templates run
  on the serving node, on a copy of the GGUF's vocabulary (no weights;
  the file must be readable there). Decoding, sampling and KV changes
  run on rank 0, which hosts `erllama_model_llama` for the serving node
  (`erllama_tp_rank:host/3`); samplers stay on rank 0 and travel as
  handles.
- **Rank 0 keeps the others in step.** Before every `llama_decode`
  (`erllama_ctx_decode` in `c_src/erllama_nif_decode.c`) its context
  sends the exact batch to a forwarder, and every KV change
  (`kv_seq_rm`, `kv_seq_cp`) too, numbered from 1. The forwarder tags
  each with the group's generation (`{erllama_tp, Gen, Seq, Cmd}`) and
  sends it to the other ranks, which apply it with
  `erllama_nif:tp_follow/2`. The ranks meet in the decode's first
  collective, and after it every rank sums the output logits, so rank 0
  samples from the full logits.
- **Before serving**, every rank describes its model (file size, hash
  of its first MiB, options, world size, engine build); a rank that
  differs is refused (`{error, {incompatible_ranks, _}}`).
- **A command that does not fit**, decided before any GPU work:

  | Command | Result |
  |---|---|
  | of another generation | stale: dropped and counted |
  | already applied (`Seq` at or below the last) | duplicate: dropped and counted |
  | the next one | applied |
  | past a gap | the follower stops: it no longer holds rank 0's KV state |

  `erllama_tp_rank:follow_stats/1` gives the counts.
- **Cancel.** A split model's decode is never interrupted (no abort
  callback on its context): a step stopped on one rank and not on the
  others would leave them waiting in a collective. A cancel takes
  effect between steps, once every rank finished the step in flight.
- **Embeddings, speculative decoding, the argmax forward and media
  prefill** answer `not_supported` on a split model.

## Checkpoints of a split model

Each rank holds its share of the KV, so a checkpoint is every rank's
share at the same committed token, and the cache must never expose
part of one.

- **Same token on every rank.** To save, rank 0 puts a `kv_pack`
  command in the stream the other ranks follow, after every batch it
  decoded (`erllama_nif:tp_send/2`), then packs its own share
  (`erllama_nif:tp_kv_pack/2`). Each other rank packs its share when it
  reaches the command, so after the same batches, and sends it to rank
  0. A restore goes the same way with `kv_unpack`.
- **One entry, all or nothing.** Rank 0 frames the shares into one
  binary, each with its size and CRC32C
  (`src/erllama_tp_checkpoint.erl`), and `erllama_model_tp:kv_pack/3`
  hands that to the model like any payload, so the cache's own
  reservation and publication rules apply unchanged. A share that does
  not come within `tp_kv_timeout` (default 30 s) fails the pack, and the
  model publishes nothing for a pack that failed: a later lookup misses.
- **Checked before any rank restores.** A frame of another world size,
  a share of the wrong size or CRC fails the restore before any rank
  applies it; a rank that fails its own share fails it after. The model
  then clears the sequence on every rank and serves the request cold.
- **Part of the KV identity.** The world size and `tensor_split` are
  appended to the KV format (`/tp=2:even`), so a checkpoint restores
  only into a group split the same way.
- **Size.** Each rank's share is the size of the whole sequence's KV:
  the split backend reads the other ranks' slices from the placeholders,
  as zeros. A checkpoint is the world size times the single-GPU size.
- `erllama_tp_rank:checkpoint_stats/1` (rank 0) gives the last pack's
  and restore's time and size.

## Plain TCP, without RDMA

NCCL picks the network itself: RDMA (`NET/IB`, RoCE included) when the
hosts have it, otherwise its socket transport. To run a group over
ordinary Ethernet, or to force TCP on hosts that have RDMA, set the
transport and the interface in each rank node's environment before the
node's first `erllama_tp:open/5` (NCCL reads them once per OS process).
Nothing in erllama changes.

```bash
# host A, rank 0: the interface on the link to the other hosts
NCCL_NET=Socket NCCL_SOCKET_IFNAME==enp1s0f0np0 \
    erl -name rank0@hostA -setcookie "$COOKIE" ...
# host B, rank 1: its own interface name
NCCL_NET=Socket NCCL_SOCKET_IFNAME==enp1s0f0np0 \
    erl -name rank1@hostB -setcookie "$COOKIE" ...
```

- `NCCL_SOCKET_IFNAME` takes a comma-separated list; a leading `=`
  matches names exactly. With two interfaces NCCL spreads its channels
  over both.
- The interface carries both NCCL's own bootstrap and the reductions.
  Erlang distribution, which forms the group and carries the serving
  rank's commands, keeps its own route.
- Check what NCCL chose in the rank's log (`NCCL_DEBUG=INFO
  NCCL_DEBUG_SUBSYS=INIT,NET`): `Using network Socket`, then
  `NET/Socket : Using [0]enp1s0f0np0:...` (device 0 is that interface),
  then one line per channel, `via NET/Socket/0`. The channel lines are
  the data path; the bootstrap connection alone says nothing about it.
- Unset, the default is unchanged: RoCE hosts keep using RDMA.

## What a failure does

| Failure | Result |
|---|---|
| a rank never joins | the coordinator fails the round: every joined rank gets `{error, join_timeout}` |
| the coordinator never answers | each rank's wait ends at its open deadline: `{error, timeout}` |
| a rank fails its local checks, or NCCL's init fails | every rank's `init_rank` returns no communicator: `{error, init_failed}` |
| a build without per-process communicators | `{error, not_supported}`; `erllama_tp:caps/0` reports it |
| a failure inside a collective (a lost peer) | the backend aborts that rank's OS process, or the collective waits forever |
| a rank blocked in a native call | ended by stopping its node; an Erlang timeout never ends a native call |
| a rank lost while serving | the coordinator kills the other ranks; `erllama_model_tp` answers `{error, {tp_rank_lost, Why}}`, the model fails its requests once and stops; the serving VM is unaffected |
| rank 0 silent past `tp_call_timeout` (default 120 s) | the same, and rank 0 is killed |

Watching the group is the coordinator's job (`erllama_tp_coordinator:watch/2`
in the tests): when a rank process goes, it kills the others with
`erllama_tp_rank:kill/1`, because a rank left in a collective with a
lost peer never returns.

After any failure the communicator is discarded and the group is
formed again, under a new coordinator (a new generation).

## Tests

- `test/erllama_tp_bootstrap_SUITE.erl` runs everywhere: the bridge,
  the agent and the coordinator through
  `erllama_tp:bootstrap_selftest/5`, which makes the same callback
  rounds as a communicator's init without a GPU. Ranks in rank order,
  a missing rank, a silent coordinator, a rank lost between rounds, a
  duplicate rank, and `not_supported` without a communicator.
- `test/erllama_tp_failure_SUITE.erl` is opt-in on the same hosts, with
  fresh nodes for each case: a rank killed mid-stream, stale, duplicate
  and out-of-order commands, and a cancel during prefill, each followed
  by a request served.
- `test/erllama_tp_checkpoint_SUITE.erl` is opt-in on the same hosts:
  a checkpoint restored by a new group, and a save missing a rank's
  share that publishes nothing. `test/erllama_tp_checkpoint_tests.erl`
  checks the frame everywhere.
- `test/erllama_tp_capacity_SUITE.erl` is opt-in on the same hosts,
  with their GPUs free and `ERLLAMA_TP_BIG_MODEL` naming a GGUF larger
  than one GPU: the model split over the group and served, each rank's
  peak memory and llama.cpp's placement. Its one-host case
  (`ERLLAMA_TP_CAPACITY_ONE_HOST=1`) can run a host out of memory.
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
    rebar3 ct --dir apps/erllama/test \
    --suite erllama_tp_nccl_SUITE,erllama_tp_failure_SUITE,erllama_tp_checkpoint_SUITE
```

Over plain TCP, with the transport and the interfaces checked (from
NCCL's channel lines and from each host's interface byte counters
around a burst of reductions):

```bash
ERLLAMA_TP_NCCL_ENV="NCCL_NET=Socket NCCL_SOCKET_IFNAME==enp1s0f0np0" \
ERLLAMA_TP_EXPECT_NET=Socket ERLLAMA_TP_EXPECT_IF=enp1s0f0np0 \
ERLLAMA_TP_HOSTS=gx10-fa8e.local,gx10-e25e.local \
    rebar3 ct --dir apps/erllama/test --suite erllama_tp_nccl_SUITE
```

`ERLLAMA_TP_NCCL_ENV_<Rank>` sets one rank's environment, for hosts
whose interfaces are named differently. `ERLLAMA_TP_EXPECT_NET=IB`
checks RDMA; RDMA traffic does not show in the interface counters.

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

Groups of more than two hosts, and Metal.
