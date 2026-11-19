# erllama_server

The erllama_server release owns loaded models, KV state, conversations,
agent runs and MCP connections, and serves them to a terminal client
and to HTTP clients. You need this page if you want to run the server
rather than embed the [erllama library](../erllama/README.md) in your
own application.

This app is being built in phases; the current state is tracked in
[doc/features.md](../../doc/features.md).

## Build and run the release

From the repository root:

```bash
rebar3 release
_build/default/rel/erllama_server/bin/erllama_server daemon
_build/default/rel/erllama_server/bin/erllama_server ping
_build/default/rel/erllama_server/bin/erllama_server stop
```

The node is named `erllama_server@127.0.0.1` and Erlang distribution
listens on the loopback interface only. The cookie is not stored in
the release: the node and the control commands (`ping`, `stop`) use
`~/.erlang.cookie` of the user that runs them. `RELX_COOKIE` only
changes the cookie the control commands use, not the node's own.

For a self-contained release that carries its own Erlang runtime, use
`rebar3 as prod release` (output in `_build/prod/rel/erllama_server`).

## Talk to the release with the CLI

`bin/erllama` is the command-line client shipped beside the release.
It runs on the release's own runtime when the release carries one, and
connects to the running node from a temporary hidden node. It never
starts the inference application or loads the NIF itself.

```bash
_build/default/rel/erllama_server/bin/erllama --version
_build/default/rel/erllama_server/bin/erllama ping
```

Point it at another node or cookie with `ERLLAMA_NODE` and
`ERLLAMA_COOKIE`. Only `ping` and `--version` exist so far; the other
commands arrive with the service API.
