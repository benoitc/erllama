# Tensor parallelism across hosts

This guide shows you how to split one model over several machines,
one GPU each, and serve it through the normal erllama API. You need it
when a model does not fit one GPU, or when you want two GPUs to share
the work of every layer. The feature is experimental: two hosts are
verified; larger groups have no limit in the code (`erllama_tp:open/5`
only checks `Rank < World`) but have not been tested. For the design
(collective ordering, the devices, the command stream, the checkpoint
format, failure behaviour) read
[internals/tensor-parallel-design.md](../internals/tensor-parallel-design.md).

You end up with these Erlang nodes:

- one **rank node** per host, holding that host's GPU as one rank of
  the group;
- one **serving node** that holds no rank: it loads the model with
  `backend => erllama_model_tp` and answers `complete`, `stream` and
  `cancel`. A collective that fails or hangs takes down a rank node,
  never the serving node.

## 1. Build with CUDA and NCCL

Build erllama on every host with CUDA and NCCL (see
[building](building.md#cuda-with-nccl-tensor-parallelism)):

```bash
ERLLAMA_OPTS="-DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DNCCL_ROOT=$HOME/nccl" \
ERLLAMA_BUILDOPTS=-j6 \
rebar3 compile
```

To prepare Linux hosts without root, use `scripts/tp_hosts.sh`. It
installs NCCL, OpenSSL, OTP 28.3.1 and rebar3 under `~/erllama-tp`,
copies your checkout there and builds the test profile:

```bash
scripts/tp_hosts.sh setup hostA hostB   # once, runs in the background
scripts/tp_hosts.sh sync  hostA hostB
scripts/tp_hosts.sh build hostA hostB
scripts/tp_hosts.sh env                 # the environment a rank node needs
```

On each host, check that the build can join a group:

```erlang
{ok, #{gpus := 1, init_rank := true, allreduce := true}} = erllama_tp:caps().
```

`init_rank => false` means the GPU backend was built without NCCL;
`erllama_tp:open/5` then answers `{error, not_supported}`.

## 2. Start a coordinator

Start `erllama_tp_coordinator` on the serving node. Give it the number
of ranks and the time in milliseconds to wait from the first arrival
in each bootstrap round.

```erlang
World = 2,
{ok, Coord} = erllama_tp_coordinator:start(World, 60000).
```

Each rank receives every rank's bytes in rank order. If a rank misses
the deadline, waiting and late joins receive `{error, join_timeout}`.
Stop the failed coordinator with `erllama_tp_coordinator:stop(Coord)`
and start a fresh one for the next generation.

## 3. Form the ranks

Start an Erlang node on each host with the same cookie, erllama
started, and the hosts able to reach each other by name. Then start
every rank at the same time: `erllama_tp_rank:start/1` returns only
once every rank has joined, or with an error within `timeout`.

```erlang
Start = fun(Node, Rank) ->
    erpc:call(Node, erllama_tp_rank, start, [#{
        rank => Rank,
        world => World,
        join => {call, Coord},
        timeout => 60000,
        generation => 0
    }], 90000)
end,
Parent = self(),
[spawn(fun() -> Parent ! {Rank, Start(Node, Rank)} end)
 || {Rank, Node} <- lists:enumerate(0, ['rank0@hostA', 'rank1@hostB'])],
[{ok, Rank0}, {ok, Rank1}] =
    [receive {R, Res} -> Res end || R <- lists:seq(0, World - 1)],
ok = erllama_tp_coordinator:watch(Coord, [Rank0, Rank1]).
```

`gpu` (default 0) picks the host's device. The rank process is not
linked to its caller: it lives until `erllama_tp_rank:stop/1` or until
its node goes. After a normal exit the coordinator stops the other
ranks. After an abnormal exit it terminates the remaining rank nodes
to release blocked collectives. Read `erllama_tp_coordinator:lost(Coord)`
for the first lost rank's `{Pid, Reason}`.

### Start a rank from application configuration

Use this mode when each rank has its own OTP application and you want
to check its state locally. Start a named coordinator on the serving
node first. Give each group its own registered name:

```erlang
{ok, Coord} = erllama_tp_coordinator:start(
    glm_group, {<<"glm">>, 1}, 2, 60000).
```

On each dedicated rank node, place this in `sys.config`. Set `rank` to
0 on the first node and 1 on the second. Both nodes need the same erllama
version, native library build, absolute model path, model file and options.

```erlang
[{erllama, [
    {tp_rank, #{group_id => <<"glm">>, generation => 1,
                rank => 0, world => 2, gpu => 0,
                coordinator => {glm_group, 'serve@hostA'},
                model_path => "/models/glm.gguf",
                model_opts => #{n_gpu_layers => 99},
                context_opts => #{n_ctx => 4096, n_seq_max => 4},
                timeout => 60000}}
]}].
```

Start the node with that configuration and start the `erllama` application.
The rank joins and loads its share in the background, including on rank 0.
Use `erllama_tp_rank_role:status()` on the rank node while it starts:

```erlang
#{group_id := <<"glm">>, generation := 1, world := 2,
  rank := 0, state := ready} = erllama_tp_rank_role:status().
```

You see `forming` during bootstrap and loading, then `ready` once the local
load succeeds. A failed rank or coordinator leaves a surviving rank at
`lost` with a `reason`. An identity mismatch gives
`{incompatible_ranks, RankNumbers}` before native bootstrap. Stop the rank
applications, start a fresh coordinator, and increase the generation on
all nodes to form the group again. With no configured rank, the local
status is `undefined`.

The coordinator address can also be a node name when you register it as
`erllama_tp_coordinator`. The default local GPU is 0 and the default join
timeout is 60000 ms. This mode starts rank processes. The serving example
below uses manually started ranks.

## 4. Load the model on every rank but 0

Every rank but 0 loads the same file with the same options. Rank 0
loads it when the serving node loads the model.

```erlang
Path = "/models/model.gguf",
MOpts = #{n_gpu_layers => 99},
COpts = #{n_ctx => 4096, n_batch => 512, flash_attn => enabled},
ok = erllama_tp_rank:load(Rank1, Path, {MOpts, COpts}).
```

The path is read on the rank's host. `tensor_split` in `MOpts` gives
uneven shares per rank (default `even`).

## 5. Load the model on the serving node

Describe each follower on its own node (the description hashes the
file there and records the engine build), then load:

```erlang
Desc1 = erpc:call(node(Rank1), erllama_tp_rank, describe, [Path, {MOpts, COpts}, World]),
{ok, Id} = erllama:load_model(<<"split">>, #{
    backend => erllama_model_tp,
    model_path => Path,
    model_opts => MOpts,
    context_opts => COpts,
    tp_worker => Rank0,
    tp_followers => [{Rank1, Desc1}]
}).
```

- The serving node reads the GGUF's vocabulary (no weights) from
  `model_path`, and rank 0 loads the model from the same path, so the
  file must be readable at that path on both. Running the serving node
  on rank 0's host is the simple way.
- A follower whose file, options, world size or build differ from rank
  0's is refused with `{error, {incompatible_ranks, _}}`.
- `tp_call_timeout` (ms, default 120000) bounds every call to rank 0;
  `tp_kv_timeout` (ms, default 30000) bounds the wait for the other
  ranks' checkpoint shares.
- Other load options (`policy`, `draft_model`, ...) work as for a local
  model.

## 6. Serve a request

```erlang
{ok, #{reply := Text, cache_hit_kind := Kind}} =
    erllama:complete(Id, <<"Hello">>, #{response_tokens => 32}).
```

`stream` and `cancel` work too. A cancel takes effect between decode
steps, once every rank has finished the step in flight: a split
model's decode is never interrupted mid-step.

If a rank is lost, or rank 0 stays silent past `tp_call_timeout`, every
call answers `{error, {tp_rank_lost, Why}}`, the model fails its
requests once and stops. Unload it, stop the remaining ranks, and form
a new group with a new coordinator and a higher `generation`.

## 7. Check a warm restore

KV cache entries of a split model hold every rank's share, packed at
the same token and published as one entry: a share that does not come
in time publishes nothing. The group's world size and split are part of
the cache key (`/tp2=2:even` is appended to the KV format), so an entry
restores only into a group split the same way.

Send the same long prompt twice, or once, then again after forming a
new group and loading the model again on the same serving node:

```erlang
{ok, #{cache_hit_kind := cold}} = erllama:complete(Id, Prompt, #{response_tokens => 32}),
{ok, #{cache_hit_kind := Warm}} = erllama:complete(Id, Prompt, #{response_tokens => 32}),
true = Warm =/= cold,
{ok, #{kind := Warm, prefix_len := N}} = erllama:last_cache_hit(Id).
```

On rank 0's node, `erllama_tp_rank:checkpoint_stats(Rank0)` gives how
long the last checkpoint and restore took and their sizes. A
checkpoint is the world size times the size of the single-GPU KV.

## Draft with an MTP head

MTP drafting works on a split model. Load the model's MTP head whole on
rank 0's GPU with `draft_model` (a path on rank 0's host) and ask for it
per request:

```erlang
{ok, Id} = erllama:load_model(<<"split">>, #{
    backend => erllama_model_tp,
    model_path => Path,
    model_opts => MOpts,
    context_opts => COpts,
    draft_model => "/models/model-mtp.gguf",
    draft_model_opts => #{load_mode => direct_io},
    tp_worker => Rank0,
    tp_followers => [{Rank1, Desc1}]
}),
{ok, _} = erllama:complete(Id, Prompt, #{
    response_tokens => 128,
    temperature => 0.0,
    speculative => #{type => mtp, n_max => 2}
}).
```

- `n_max` is the number of drafted tokens per round: default 2, at
  most 16.
- With `n_seq_max > 1` in `context_opts`, several requests at once
  draft together in one batch per round (up to 16 requests), when
  every active request is ready to decode and none asks for
  `logprobs`.
- Leave room on rank 0's GPU for the head: a `tensor_split` such as
  `[0.46, 0.54]` gives rank 0 the smaller share.

## Hybrid models (GLM-5.3-Flash)

Split loads default to `n_rs_seq => 0` on every rank (no
recurrent-state snapshots). Without snapshots a hybrid model cannot
take back its last tokens: a cache hit on a whole prompt is served
cold, and MTP cannot roll back a rejected draft. Set `n_rs_seq` in the
context options, the same on every rank, at least `n_max`:

```erlang
COpts = #{
    n_ctx => 4096,
    n_batch => 1024,
    flash_attn => enabled,
    n_rs_seq => 2,
    n_seq_max => 4,
    kv_unified => true
},
MOpts = #{n_gpu_layers => 999, load_mode => direct_io}.
```

On a GB10, whose GPU shares the host's memory, load large models with
`load_mode => direct_io`: the page cache a buffered load leaves behind
is not given back to CUDA.

## Run over plain TCP

NCCL picks RDMA when the hosts have it, otherwise its socket
transport. To force TCP, set the transport and interface in each rank
node's environment before the node's first `erllama_tp:open/5`:

```bash
NCCL_NET=Socket NCCL_SOCKET_IFNAME==enp1s0f0np0 \
    erl -name rank0@hostA -setcookie "$COOKIE" ...
```

Check what NCCL chose with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET`
in the rank's log. See
[internals/tensor-parallel-design.md](../internals/tensor-parallel-design.md#plain-tcp-without-rdma).

## What works and what does not

| Feature | On a split model |
|---|---|
| `complete`, `stream`, `cancel` | yes (cancel between steps) |
| KV cache, warm restore | yes |
| MTP drafting, one request or several at once | yes, `draft_model` on rank 0 |
| ngram speculation | no: the backend refuses the drafter and the request decodes without drafts |
| embeddings | no |
| `erllama:verify/4` | no |
| media input | no |
| LoRA adapters | no |

## Notes

- Two hosts are verified. Groups of more than two hosts are
  experimental; four-host testing is open.
- The opt-in suites that check all of this on real hosts are listed in
  [internals/hardware-suites.md](../internals/hardware-suites.md).
- Watching the group, and killing the other ranks when one goes, is the
  coordinator's job.
