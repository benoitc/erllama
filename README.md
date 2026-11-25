# erllama

erllama runs llama.cpp models inside the Erlang VM. You use it in one
of two ways:

- as a **library** (the Hex package `erllama`): your OTP application
  loads a model, streams tokens as messages to its own processes, and
  supervises inference like any other work;
- as **erllama_server**, a release that keeps running: it owns the
  models, conversations, runs and agent sessions, and the `erllama`
  CLI, HTTP clients (OpenAI, Anthropic, Ollama) and other Erlang nodes
  share them. Clients can leave and come back; the work goes on.

Both use the same engine: a model serves several requests at once in
one batch, and an exact, tiered KV cache restores prefixes it has
already computed. You start here to find the part you need.

## How it fits together

```
your OTP application          CLI   HTTP clients   Erlang nodes
        |                       \        |         /
        |                  erllama_server (a release)
        |                  conversations, runs, agent sessions,
        |                  tools, MCP: one process each
        |                            |
        +------------ erllama (the library) -------------+
                one process per loaded model (a gen_statem),
                the KV cache tiers, one NIF over llama.cpp
```

- The library knows models, requests and the cache. It has no notion
  of users or conversations.
- The server adds what a shared service needs: who owns which work,
  what survives a client or a restart, limits, and what an operator
  sees. Its failures stay in the process that owns them; a crash in
  native code ends the VM, for both.

| You want to | Go to |
|---|---|
| Use erllama as a dependency (Hex package `erllama`) | [apps/erllama](apps/erllama/README.md), [examples/embedded_app](examples/embedded_app) |
| Run the server release | [apps/erllama_server](apps/erllama_server/README.md) |
| See the server at work: several clients, a slow tool, a restart | [the walkthrough](guides/server/demo.md) |
| Call the server from Erlang | [examples/service_client](examples/service_client) |
| Serve `wasi:nn` components of erlang_wasm from your models | [apps/erllama_wasi_nn](apps/erllama_wasi_nn/README.md) |
| Read the usage guides | [guides/](guides/) |
| Understand the cache, NIF and request design | [internals/](internals/) |
| Contribute | [AGENTS.md](AGENTS.md) |
| See what is planned | [ROADMAP.md](ROADMAP.md) |

## Layout

```
apps/erllama/          the library: Erlang sources, NIF (c_src), tests
apps/erllama_server/   the server release and the `erllama` CLI
apps/erllama_wasi_nn/  a wasi:nn backend for erlang_wasm components
config/, rel/          release configuration and the CLI launcher
scripts/               checks, e.g. erllama built standalone from its Hex package
guides/, internals/    documentation, published with the library
bench/, examples/      benchmarks and standalone example projects
rebar.config           shared tooling for both apps
```

## Build and test

Run every command from the repository root. You need Erlang/OTP 28,
rebar3 3.25+, CMake and a C/C++ toolchain.

```bash
rebar3 compile    # builds the vendored llama.cpp and the NIF on first run
rebar3 eunit
rebar3 ct
```

The full set of checks is in [AGENTS.md](AGENTS.md).

## License

MIT. See [LICENSE](LICENSE).
