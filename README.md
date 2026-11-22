# erllama

This repository holds erllama, a native Erlang/OTP runtime for
llama.cpp with a tiered KV cache, and erllama_server, a release built
on it. You start here to find the part you need: use the library from
your own OTP application, or run the server.

| You want to | Go to |
|---|---|
| Use erllama as a dependency (Hex package `erllama`) | [apps/erllama](apps/erllama/README.md) |
| Run the server release | [apps/erllama_server](apps/erllama_server/README.md) |
| Read the usage guides | [guides/](guides/) |
| Understand the cache, NIF and request design | [internals/](internals/) |
| Contribute | [AGENTS.md](AGENTS.md) |
| See what is planned | [ROADMAP.md](ROADMAP.md) |

## Layout

```
apps/erllama/          the library: Erlang sources, NIF (c_src), tests
apps/erllama_server/   the server release and the `erllama` CLI
config/, rel/          release configuration and the CLI launcher
scripts/               checks, e.g. erllama built standalone from its Hex package
guides/, internals/    documentation, published with the library
bench/, examples/      benchmarks and a standalone example project
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
