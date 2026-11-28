# Start a serving node and two ranks

Use these configurations when you want the erllama release to serve one GGUF
split across two GPUs. You run the same release three times. The serving node
accepts chat requests and owns the group coordinator. Each rank loads its
share of the model and exposes three HTTP probes.

## Prepare the three roles

This example runs on one machine with two CUDA GPUs and an erllama build with
NCCL. Use the same release build and absolute model path on all three nodes.
Run the nodes as a user with a private `~/.erlang.cookie` file (mode 0400 or
0600). The example binds Erlang distribution and HTTP to loopback.

Create three config directories under your working directory:

```sh
mkdir -p serve rank0 rank1
for role in serve rank0 rank1; do
    cat > "$role/vm.args" <<EOF_VM
-name $role@127.0.0.1
-kernel inet_dist_use_interface {127,0,0,1}
+K true
+A 16
EOF_VM
done
```

Save this as `serve/sys.config`. Replace `/models/model.gguf` in all three
files. Choose data directories your service user can write.

```erlang
[{erllama_server, [
    {role, serving},
    {http, #{ip => {127,0,0,1}, port => 8080}},
    {data_dir, "/var/tmp/erllama/serve"},
    {models, [#{name => <<"split">>, path => "/models/model.gguf",
                group => #{group_id => <<"demo">>, generation => 1, world => 2,
                           coordinator => demo_group,
                           followers => ['rank1@127.0.0.1']}}]}
]}].
```

Save this as `rank0/sys.config`:

```erlang
[{erllama, [
    {tp_rank, #{group_id => <<"demo">>, generation => 1, world => 2,
                rank => 0, gpu => 0, coordinator => {demo_group, 'serve@127.0.0.1'},
                model_path => "/models/model.gguf", model_opts => #{n_gpu_layers => 99},
                context_opts => #{n_ctx => 4096, n_seq_max => 4}}}
]}, {erllama_server, [
    {role, rank}, {http, #{ip => {127,0,0,1}, port => 8081}},
    {data_dir, "/var/tmp/erllama/rank0"}
]}].
```

Save this as `rank1/sys.config`:

```erlang
[{erllama, [
    {tp_rank, #{group_id => <<"demo">>, generation => 1, world => 2,
                rank => 1, gpu => 1, coordinator => {demo_group, 'serve@127.0.0.1'},
                model_path => "/models/model.gguf", model_opts => #{n_gpu_layers => 99},
                context_opts => #{n_ctx => 4096, n_seq_max => 4}}}
]}, {erllama_server, [
    {role, rank}, {http, #{ip => {127,0,0,1}, port => 8082}},
    {data_dir, "/var/tmp/erllama/rank1"}
]}].
```

## Put the follower on a second host

Use TLS distribution for the coordinator, rank commands and Erlang node
connections. Both peers present a certificate. In this example the serving
node and rank 0 run on `a.cluster.example` (10.20.0.11), and rank 1 runs
on `b.cluster.example` (10.20.0.12). Both names must resolve on both hosts.

1. Provision these files on each host for the release's service account.

   | File | Contents |
   |---|---|
   | `~/.erlang.cookie` | The same secret on both hosts, owned by the service account, mode 0400 or 0600. |
   | `/etc/erllama/dist/tp-dist.conf` | Copy `etc/tp-dist.conf` from the unpacked release. OTP reads the three PEM files beside it. |
   | `/etc/erllama/dist/ca.pem` | The public CA certificate dedicated to this cluster. |
   | `/etc/erllama/dist/node.pem` | This host's certificate and any intermediate chain, with its DNS name in the subject alternative names and both client and server TLS usage. |
   | `/etc/erllama/dist/node.key` | The matching private key, readable only by the service account. |

2. Give each role its own `vm.args`. Here is the complete serving example.
   For the other two roles replace the node name, interface and **both** port
   values with the row in the table.

   ```text
   -name serve@a.cluster.example
   -proto_dist inet_tls
   -ssl_dist_optfile /etc/erllama/dist/tp-dist.conf
   -kernel inet_dist_use_interface {10,20,0,11}
   -kernel inet_dist_listen_min 9100 inet_dist_listen_max 9100
   +K true
   +A 16
   ```

   | Role and node name | Interface | Incoming distribution TCP port |
   |---|---|---:|
   | `serve@a.cluster.example` | `{10,20,0,11}` | 9100 |
   | `rank0@a.cluster.example` | `{10,20,0,11}` | 9101 |
   | `rank1@b.cluster.example` | `{10,20,0,12}` | 9102 |

3. In both rank configs set `coordinator => {demo_group, 'serve@a.cluster.example'}`.
   In the serving config set `followers => ['rank1@b.cluster.example']`.
   On a follower host with a single GPU, set `gpu => 0`. Keep HTTP on
   loopback. Start each role on its assigned host using the foreground
   command below. Supervise this process and stop it with SIGTERM.
   The relx `daemon`, `ping` and `stop` helpers can pass the cookie on a
   process command line. Foreground startup with a full node name reads
   the cookie from the account's file. Omit `-setcookie` and leave
   `ERLLAMA_COOKIE` and `RELX_COOKIE` unset.
   If you use the CLI, `ERLLAMA_DIST_OPTFILE` takes the **path** to its
   own TLS profile. The release itself reads `vm.args`.

4. Limit inbound connections to the peer host on the private network.
   Allow TCP **4369** on both hosts for EPMD name discovery, **9100 and
   9101** on A, and **9102** on B. The coordinator shares the serving
   node's TLS port. HTTP ports 8080, 8081 and 8082 stay local in this
   example.

   GPU collectives use NCCL's own network connections. For a TCP deployment
   on an approved private interface named `eth1`, set `NCCL_NET=Socket`
   and `NCCL_SOCKET_IFNAME==eth1` in each rank's environment. Use the
   local interface name on each host. On Linux, read
   `sysctl -n net.ipv4.ip_local_port_range` and allow that destination
   TCP range between the two rank IPs on the private link. For example,
   a configured range of `50000 51000` needs TCP 50000–51000 on both.
   Erlang TLS covers the control traffic. The private interface carries
   the tensor data. [NCCL network configuration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/networking_troubleshooting.html)

## Start and check the group

Set `RELEASE` to your unpacked release directory. Start the serving node first
and wait for its health probe before you start the ranks. Leave
`ERLLAMA_HTTP_PORT` unset so each listener uses its configured port.

```sh
RELEASE=/opt/erllama
start_role() {
    role_dir="$PWD/$1"
    RELX_CONFIG_PATH="$role_dir/sys.config" VMARGS_PATH="$role_dir/vm.args" \
        ERL_CRASH_DUMP="$role_dir/erl_crash.dump" \
        "$RELEASE/bin/erllama_server" foreground >"$role_dir/console.log" 2>&1 &
    role_pid=$!
}
start_role serve
serve_pid=$role_pid
until curl -fsS http://127.0.0.1:8080/health >/dev/null; do
    kill -0 "$serve_pid" || exit 1
    sleep 1
done
start_role rank0
rank0_pid=$role_pid
start_role rank1
rank1_pid=$role_pid
```

Read `/rank` on ports 8081 and 8082 to see the rank number, generation and
state. The rank's `/ready` returns 200 once its share is loaded. The serving
node loads the model when both ranks join. Once its `/ready` returns 200,
send your request to port 8080:

```sh
curl -fsS http://127.0.0.1:8081/rank
curl -fsS http://127.0.0.1:8082/rank
curl -fsS http://127.0.0.1:8080/ready
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"split","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'
```

Stop these instances from the shell that started them:

```sh
kill -TERM "$serve_pid" "$rank0_pid" "$rank1_pid"
wait
```

After a lost rank, increase `generation` in all three configs and start the
group again. For daemon mode in the single-host example, give each instance
its own `RUNNER_LOG_DIR` and `PIPE_DIR`. If you configure disk cache tiers, give each rank
its own writable cache directory.

## Two-host qualification, 28 November 2026

Use this result when deploying the release under a host supervisor such as
barrellm. One NVIDIA GB10 on each of two Linux arm64 hosts ran the same
production release. The serving process shared rank 0's host. Erlang
distribution used mutual TLS. NCCL reported `NET/Socket` on the private
Ethernet link. The qualification covered two ranks and one active sequence.

| Check | Result through the release |
|---|---|
| Form the group | Both ranks and the serving model reached `ready` in generation 1. |
| Buffered and streamed chat | Buffered chat returned visible text. All three measured streams delivered 64 tokens. |
| Cancel mid-stream | The client disconnected after visible text. The stored Response reached `cancelled`, and the next request completed. |
| Restore a checkpoint | Repeating a 440-token prompt restored a 384-token prefix in 60.0 ms. Both runs returned the same greedy reply. |
| Kill the follower during generation | After the first text delta, rank 1 received SIGKILL. The active Response ended with `response.failed`. Serving readiness returned 503, and group status identified rank 1 with `detail: noconnection`. |
| Send a new request to the lost generation | HTTP 503 with `error.code: tp_rank_lost`. |
| Form generation 2 | Restarting the three roles with the incremented generation served a new buffered reply under the same model name. |

The single-host baseline ran the full model on rank 0's GPU through the
same release and chat endpoint. Both configurations used the same 38-token
prompt, greedy sampling, a 2,048-token context and one sequence slot. These
are medians of three 64-token streams after a warm-up. The timing prompts
were below the 160-token checkpoint threshold.

| TinyLlama 1.1B | First visible text | Decode tokens/s | End-to-end tokens/s |
|---|---:|---:|---:|
| One GB10 | 16.2 ms | 261.6 | 249.3 |
| Two GB10s, NCCL sockets | 70.4 ms | 37.7 | 36.8 |

Decode speed counts the remaining 63 tokens after the first visible text.
The split decode rate ranged from 29.6 to 46.5 tokens/s. These measurements
apply to this small model over sockets. The raw HTTP observations and
individual timings are in `bench/results/2026-11-28-gb10-release-tp.json`.

At the median, generating the next token took 3.82 ms on one GPU and
26.55 ms on the pair. TinyLlama has 22 layers. Every token needs two
cross-host sums per layer before the next token can start. A single GPU
holds the whole model and can advance through all 22 layers without a
network round trip.

An earlier controlled test on the same pair changed the NCCL transport
and held the model, build, prompt and output length fixed. The private
200 Gb/s links delivered 56–66 tokens/s with sockets and 177–229 with
RoCE. Sockets on a 2.5 Gb/s port delivered 43–48. This points to the
latency of repeated small collectives. Link bandwidth alone predicts a
much larger difference between the two socket cases. The release test
used one private interface, a shorter prompt and the HTTP serving path.
The archived log shows all eight NCCL channels used that interface.
The release benchmark did not time the in-graph sums or include a matched RoCE leg, so the
22.73 ms difference per token also includes other split-serving costs.
[Transport comparison and reduction timings](https://github.com/benoitc/erllama/blob/v0.14.0/apps/erllama/doc/features.md#tensor-parallelism-over-plain-tcp-2026-11-23)

Both GB10s also completed a separate 2,048-value FP32 sum with
`NCCL_NET=IB` on `rocep1s0f0`. Both logged `Using network IB` and
`NET/IB : GPU Direct RDMA Disabled`. CUDA reported zero for
`GPU_DIRECT_RDMA_SUPPORTED` and `DMA_BUF_SUPPORTED` on each host. On
Spark the NIC can use RoCE with registered host buffers. Direct NIC
access to CUDA device allocations is unsupported on this platform.
[NVIDIA DGX Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/latest/porting/cuda.html#gpudirect-rdma)

**Build and model.** The working tree was based on
`b50a5d32698daa9ad6ebf19b49e1f6b961aad65c` and included the local
release changes. The source archive SHA-256 was
`20fe5311c87ee57dd58d66041dd2fabcedebbce413ff9d780fc0ac1503580833`.
The build used OTP 28.3.1, CUDA 13.0.88 for SM 121, NCCL 2.32.6 and
driver 580.178.04. Both ranks reported `gpus => 1`, `device => "CUDA0"`,
`init_rank => true` and `allreduce => true` from `erllama_tp:caps/0`.
The TinyLlama GGUF SHA-256 was
`9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`
on both hosts.

### Fleet acceptance and matched transport comparison

Two Fleet agents ran erllama `0f69622` on the same GB10 pair. In epoch 10
the gateway served chat through the lead. After rank 1 received SIGKILL,
Fleet removed the replica from its ready catalog and the next gateway
request returned HTTP 502. A new activation (epoch 31) served the same
chat again. Fleet released its slots at the end. The follower readiness
report includes the model revision in barrellm `d5b508c`.

The timing client ran on the lead host and used the erllama streaming
endpoint directly. Both transports used one private 200 Gb/s interface,
the same release and GGUF, one active sequence, and the 38-token prompt
from the qualification above. Each leg had one warm-up and three
64-token streams. The cache counters recorded zero hits throughout.
Both ranks logged the selected NCCL transport.

| Two GB10s, one private link | First visible text, median | Decode tokens/s, median (range) |
|---|---:|---:|
| RoCE (`NET/IB`) | 25.9 ms | 135.4 (129.6–144.9) |
| TCP (`NET/Socket`) | 76.6 ms | 35.3 (28.1–41.4) |

RoCE decoded 3.83 times as fast as sockets in this run. RoCE was measured
first. The earlier single-GPU result of 261.6 tokens/s came from a separate
window. [Individual runs and Fleet outcomes](https://github.com/benoitc/erllama/blob/v0.14.0/bench/results/2026-11-28-gb10-fleet-roce-socket.json)

### Fleet adapter handoff

1. Write one serving `sys.config` with `erllama_server.models` containing
   `name`, `path` and `group`. Inside `group`, set `group_id`, `generation`,
   `world`, `coordinator` and `followers` in rank order starting at rank 1.
   For each rank, set `erllama_server.role` to `rank`, configure its `http`
   listener, and set `erllama.tp_rank` with `group_id`, `generation`,
   `world`, `rank`, `gpu`, `coordinator`, `model_path`, `model_opts`
   and `context_opts`. The coordinator address on a rank is
   `{RegisteredName, ServingNode}`. Use the role-specific `VMARGS_PATH`
   and `RELX_CONFIG_PATH` shown above.
2. Require serving `GET /ready` to return 200. In `GET /status`, check
   `api` is 2 and match `groups[]` by `model`, `group_id` and `generation`.
   Check `world`, `n_seq_max`, group `state` and each `ranks[].rank` and
   `ranks[].state`. You can read an individual rank's local state at `/rank`.
3. On `lost`, stop sending requests to that generation. Read `reason.code`
   and, when available, `reason.rank` and `reason.detail`. A fresh request
   gets `tp_rank_lost`; an already-open Responses stream can end with
   `response.failed` and the generic event code `server_error`. Use group
   status to identify the failed rank. For `incompatible_ranks`, inspect
   `reason.ranks` and correct the mismatched settings or build before
   starting a new generation. For `group_unavailable`, inspect `/status`.
