# Contributing

This page tells you how to change erllama safely: the toolchain, the
checks every change must pass, where the tests live, how to open a pull
request, and where to start for the most common changes. Read
[internals/overview.md](internals/overview.md) first if you are new to
the code; it explains the terms used below.

## Set up the toolchain

You need Erlang/OTP 28.3, rebar3 3.25 or later, CMake and a C/C++
toolchain. The build uses `warnings_as_errors`, and a newer OTP (for
example a package manager's OTP 29) can emit new warnings that break
it: use OTP 28.3 (kerl or asdf).

```bash
git clone https://github.com/benoitc/erllama.git
cd erllama
rebar3 compile    # builds the vendored llama.cpp and the NIF on first run
```

Run every rebar3 command from the repository root: it is a rebar3
umbrella with the library in `apps/erllama` and the server in
`apps/erllama_server`.

## Run the checks

Every change must be formatted and pass all of these before you open a
pull request:

```bash
rebar3 fmt          # auto-format (run first)
rebar3 compile      # must compile cleanly
rebar3 eunit        # unit tests
rebar3 proper       # property tests
rebar3 ct           # Common Test suites
rebar3 lint         # Elvis
rebar3 dialyzer     # types
rebar3 xref         # cross references
```

A change to the library must also keep it buildable on its own, without
the server's dependencies:

```bash
scripts/check_library_package.sh
```

To run one module or suite:

```bash
rebar3 eunit --module=erllama_cache_kvc_tests
rebar3 ct --dir apps/erllama/test --suite erllama_SUITE
```

## Know where the tests are

| File | Kind |
|---|---|
| `apps/<app>/test/<mod>_tests.erl` | EUnit |
| `apps/<app>/test/prop_<mod>.erl` | PropEr |
| `apps/<app>/test/<feature>_SUITE.erl` | Common Test |

Most tests run on the stub backend (`erllama_model_stub`) and need no
model file. The suites that need a real model, a GPU or several hosts
skip unless you set their environment variables:

```bash
LLAMA_TEST_MODEL=/path/to/tinyllama-1.1b-chat.gguf \
    rebar3 ct --dir apps/erllama/test --suite erllama_real_model_SUITE
```

Every opt-in suite, its variables, models and hosts, and which one to
run after which change, are in
[internals/hardware-suites.md](internals/hardware-suites.md).

## Open a pull request

- One change per pull request.
- The description states the intent (what was wrong and why it
  matters), the solution in a few lines, and any breaking change.
- A change visible to users gets a CHANGELOG entry in the app it
  touches (`apps/erllama/CHANGELOG.md` or
  `apps/erllama_server/CHANGELOG.md`): short, plain English, what was
  wrong or what is new.
- Comments explain only a non-obvious why. Behaviour is defined by the
  tests: if your change conflicts with an existing test, say so in the
  pull request instead of changing the test quietly.

## If you want to...

### Add a request option

1. Add the key and its check to `apps/erllama/src/erllama_opts.erl`
   (`request_opts/1`); an unknown or invalid value must return
   `{error, {invalid_option, Key, Value}}`.
2. Use it in `erllama_model` where the request is admitted or sampled.
3. Test it in `erllama_opts_tests` and, for its effect, in
   `erllama_model_tests` on the stub.
4. Document it in `guides/generation.md` or `guides/configuration.md`.

### Add a load or context option

1. Add the key to `MODEL_OPT_KEYS` or `CONTEXT_OPT_KEYS` in
   `erllama_opts.erl`.
2. `erllama_model_llama` forwards `model_opts` and `context_opts` to
   `erllama_nif:load_model/2` and `new_context/2`; read the key in
   `c_src/erllama_nif_model.c` (`nif_load_model`) or
   `c_src/erllama_nif_context.c` (`nif_new_context`).
3. If it changes the KV layout, fold it into the cache key
   (`kv_format/1` in `erllama_model_llama`), or a restore can read rows
   saved with another layout.
4. Test it in `erllama_opts_tests` and `erllama_model_llama_tests`; a
   real-model case in `erllama_real_model_SUITE` if it needs llama.cpp.

### Add a NIF

1. Write the function in the `c_src/erllama_nif_*.c` file of its area.
2. Register it in the `nif_funcs` table of `c_src/erllama_nif.c`. Run
   it on a dirty scheduler unless it is cheap and cannot block.
3. Add the Erlang stub to `apps/erllama/src/erllama_nif.erl`.
4. Follow [internals/nif-safety.md](internals/nif-safety.md): argument
   checks, the resource lifetime pattern, no exception crossing into
   the VM.
5. Test it in `erllama_nif_tests` (and `prop_erllama_nif` for input
   fuzzing).

### Add a backend

1. Implement the `erllama_model_backend` behaviour; its module doc lists
   the required and optional callbacks.
2. Use `erllama_model_stub` as the reference: it implements the
   contract without native code.
3. Test it through `erllama_model` as `erllama_model_stub_tests` and
   `erllama_model_tests` do.

### Change the cache tiers or the save policy

1. Read [internals/cache-design.md](internals/cache-design.md) and
   [internals/publish-protocol.md](internals/publish-protocol.md); the
   save invariants in [AGENTS.md](https://github.com/benoitc/erllama/blob/main/AGENTS.md) must hold.
2. When to save: `erllama_cache_policy`. How: `erllama_cache_writer`.
   Where: `erllama_cache_ram`, `erllama_cache_disk_srv` (disk and
   `ram_file` roots). Who may publish: `erllama_cache_meta_srv`.
3. Test with `erllama_cache_policy_tests`, `prop_erllama_cache_policy`,
   `erllama_cache_writer_tests`, `erllama_cache_disk_srv_tests`,
   `erllama_cache_meta_srv_tests` and `erllama_cache_tier_tests`.

### Debug a request

1. Library: `erllama:last_cache_hit/1`, `erllama:request_info/1`,
   `erllama:counters/0` and the native logs; see
   [guides/observability.md](guides/observability.md) and
   [guides/troubleshooting.md](guides/troubleshooting.md).
2. Server: `erllama status`, `erllama inspect ID`, then
   `erllama attach RUN --from 0` to replay a run's events. The recent
   failures are in `erllama_server_api:failures/1`.
3. Follow the request path in
   [internals/overview.md](internals/overview.md#from-an-http-request-to-a-token)
   and [internals/request-lifecycle.md](internals/request-lifecycle.md).

### Bump llama.cpp

Follow [apps/erllama/UPDATE_LLAMA.md](apps/erllama/UPDATE_LLAMA.md).

### Extend the public API

1. Export and document the function in `apps/erllama/src/erllama.erl`
   with `-doc` and `-spec`.
2. Decide whether it goes through the middleware chain
   (`erllama_middleware`) and test that in `erllama_middleware_tests`.
3. Test it in `erllama_tests` or `erllama_facade_tests`.
4. Add it to the API table in `apps/erllama/README.md`.

### Change speculative decoding or MTP

1. Native side: `c_src/erllama_nif_mtp.c` (the single and the batched
   MTP round: a fix to one usually belongs in both),
   `c_src/erllama_mtp.cpp`, `c_src/erllama_spec_nif.cpp`, and
   `nif_spec_step` in `c_src/erllama_nif_decode.c` for ngram drafting.
2. Model side: the ngram speculation and MTP sections of
   `erllama_model` (`spec_gates_ok/3`, `maybe_mtp_batch_tick/1`), and
   `erllama_model_tp:mtp_step_batch/4` for split models.
3. Run `erllama_stale_logits_SUITE` and `erllama_speculative_tests`
   locally, and `erllama_tp_mtp_SUITE` on hardware.

### Change tensor parallelism

1. Read [internals/tensor-parallel-design.md](internals/tensor-parallel-design.md).
2. The rank process is `erllama_tp_rank`, the serving backend
   `erllama_model_tp`, the rank-local API `erllama_tp`, split
   checkpoints `erllama_tp_checkpoint`, the native side
   `c_src/erllama_tp_nif.cpp`, and the local llama.cpp fixes
   `apps/erllama/scripts/patches/`.
3. Run `erllama_tp_bootstrap_SUITE` anywhere; the hardware suites run
   on hosts prepared with `scripts/tp_hosts.sh` (see
   [internals/hardware-suites.md](internals/hardware-suites.md)).

### Add an HTTP endpoint or dialect

1. Add the route to `routes/0` in
   `apps/erllama_server/src/erllama_server_http.erl`.
2. Build the handler on `erllama_server_http_kit`: `read_json/2` for
   the body, `start_run/2` and `drive/3` for a generation, `api_error/2`
   for the protocol's error shape. Reach models, embeddings and token
   counts through `erllama_server_api`.
3. Test it in `erllama_server_http_SUITE` and
   `erllama_server_protocols_tests`; list it in `guides/server/http.md`.

### Add an agent tool

1. Add its spec to `specs/0` in `erllama_server_tools.erl` with
   `tool(Class, Name, Description, Properties, Required)`: `Class` is
   `read` or `write`. A tool that changes the workspace or runs a
   command is `write`: it asks for approval by default and never runs
   alongside other calls.
2. Dispatch it in `execute/3`; resolve every path with
   `erllama_server_workspace:resolve/2`.
3. Test it in `erllama_server_tools_tests`, and the loop in
   `erllama_server_agent_SUITE`, which scripts the model with
   `erllama_server_agent_script` through the application env key
   `agent_generator`.

### Add a CLI command

1. Add a clause to `command/1` in `erllama_server_cli.erl` and its line
   to `?USAGE`.
2. The handler calls the release only through
   `erllama_server_client:call/3`; never call an erllama module from the
   CLI.
3. Test the parsing in `erllama_server_cli_tests`; in
   `erllama_server_rpc_SUITE`, `leaked/1` fails if the client node
   loaded an erllama module.

### Add an MCP integration

1. Client side: `erllama_server_mcp` (connections, catalog, calls) and
   `erllama_server_agent_policy` (what a session offers and asks);
   configuration in `valid_mcp/1` of `erllama_server_config`.
2. Server side (the `/mcp` endpoint): `register_tools/0` in
   `erllama_server_mcp_endpoint`.
3. Test with `erllama_server_mcp_SUITE` (real servers over Streamable
   HTTP and stdio) and `erllama_server_mcp_tests`.
