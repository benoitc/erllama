# erllama_server: progress record

The plan (7 phases) is in
[apps/erllama_server/PLAN.md](../apps/erllama_server/PLAN.md). This
page records what is implemented, what is outstanding and how it was
validated. A phase is only marked done when its required behaviour is
implemented and tested.

Branch: `refactor/umbrella-layout`, rebased on erllama 0.12.0
(`ad47520`, GitHub release `v0.12.0`).

## Phase 1: repository and build migration (implemented, awaiting review)

| Step | State | Notes |
|---|---|---|
| rebar3 umbrella behaviour checked | done | app hooks run from the app dir; `priv` is linked; PropEr finds `apps/*/test`; `hex build --app` works; empty fmt globs pass silently, so globs were fixed |
| move into `apps/erllama` | done | history kept; rebased onto 0.12.0 (the two files 0.12.0 added were placed under `apps/erllama`) |
| umbrella build | done | app `rebar.config` (NIF hooks, Hex, ex_doc), root tooling, elvis and erlfmt globs, Hex `files`, CI paths, bench, doc paths, root README |
| `apps/erllama_server` + release | done | app, sup, CT suite; relx release with `config/sys.config` and `config/vm.args.src`; node on loopback only |
| CLI escript | done (scaffold) | `erllama_server_cli`, built as the `erllama` escript before every release, shipped as `bin/erllama_cli` with the `bin/erllama` launcher; `--version` and `ping` only |
| standalone library check | done | `scripts/check_library_package.sh` and the `library-package` CI job |
| release check | done | `release` CI job: prod release, daemon, ping, CLI `--version` and `ping`, stop |

Validation (OTP 28.3.1, local):
- full gate: eunit 580/580, proper 13/13, ct 9 passed / 69 skipped, lint, xref and dialyzer clean, fmt clean;
- real-model suite via `--dir apps/erllama/test`: stories260K 40 passed / 4 skipped, TinyLlama 44/44;
- `scripts/check_library_package.sh`: the 0.12.0 package compiles its NIF and runs in a consumer project, with no livery or barrel_mcp;
- release: starts, `bin/erllama ping` answers `pong` with the right cookie and fails cleanly with a wrong one; the prod release runs the CLI on its bundled ERTS with no Erlang on `PATH`;
- the CLI loads no erllama module (checked in a fresh peer node).

Known limits and findings:
- rebar3's escriptize bundles the main app's dependencies, so the escript carries erllama's beam files (not its NIF library). The CLI never loads them; the peer-node test guards this;
- the relx start script uses `RELX_COOKIE` only for its control commands; the node itself uses `~/.erlang.cookie`. Phase 2 documents the cookie and node setup;
- stale generated HTML may remain in the root `doc/` of old checkouts (ignored).

## Phases 2 to 7

Not started.
