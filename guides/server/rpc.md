# Driving the release from the CLI

The `erllama` command talks to a running `erllama_server` release over
Erlang distribution. You need this page to run generations from a
terminal, to follow or cancel them, to point the CLI at a release on
another node or machine, and to understand what the CLI reports when
something goes wrong.

The release owns everything: models, KV state, conversations and runs.
The CLI only sends commands and prints the events it receives, so
closing the terminal never stops a generation, unloads a model or
stops the release.

## Start the release and load a model

```bash
rebar3 as prod release
R=_build/prod/rel/erllama_server
$R/bin/erllama_server daemon
$R/bin/erllama load ~/Models/tinyllama-1.1b-chat.gguf --name tiny --n-ctx 2048
$R/bin/erllama status
```

`load` registers a GGUF file on the release machine and loads it;
`pull` downloads one. See [Managing models](models.md).

## Run a chat turn

```bash
$R/bin/erllama run tiny "Say hello in one short sentence."
$R/bin/erllama run tiny --max-tokens 400 "Write a short story about a cat."
$R/bin/erllama chat tiny
```

Each `run` is one turn of a conversation, and the tokens stream to your
terminal as the model produces them. Pass `--conversation ID` to add a
turn to an existing conversation: the release prompts the model with
the whole history and passes the previous turn's cache key, so the
cache can restore the shared prefix instead of recomputing it.
A conversation runs one turn at a time.

## Chat

```bash
$R/bin/erllama chat tiny
> Plan a picnic for four.
...
> /fork
[going on in conv_19be..., a copy of conv_7f3a...]
> /quit
```

- A line ending in `\` goes on to the next line; `"""` on its own
  line starts a block that runs to the next `"""`.
- In a terminal, the reply streams as it is generated. Reasoning shows
  dimmed. While the release loads the model, compacts the conversation
  or prefills the prompt, a status line counts the seconds.

| Command | Does |
|---|---|
| `/new` | the next message starts a new conversation |
| `/sessions` | lists the conversations |
| `/switch ID` | continues conversation `ID` |
| `/fork` | goes on in a copy of the current conversation |
| `/compact` | shortens the current conversation now |
| `/help` | lists the commands |
| `/quit` | leaves (so does the end of input) |

## Ctrl-C

In a terminal, Ctrl-C acts on what is running, never on the release:

| While | Ctrl-C |
|---|---|
| `run` or a `chat` turn streams | cancels the run; the turn is not added to the conversation |
| an `agent` turn runs | interrupts the turn; the session stays |
| `pull` downloads | cancels it; the part already downloaded is kept for the next pull |
| `attach` follows a run | detaches; the run goes on |
| you type at a prompt | drops the line |
| anything else | leaves the CLI |

- During a turn, a second Ctrl-C leaves the CLI. The release keeps
  what it was doing.
- Wait for the prompt before you type the next message: keys pressed
  while a reply streams are not kept.
- Closing the terminal ends only the CLI. Runs, sessions and loaded
  models stay on the release.

## Detach, reattach, cancel

```bash
RUN=$($R/bin/erllama run tiny --max-tokens 600 --detach "Count to two hundred.")
$R/bin/erllama runs
$R/bin/erllama attach "$RUN"
$R/bin/erllama attach "$RUN" --from 120
$R/bin/erllama cancel "$RUN"
$R/bin/erllama result "$RUN"
```

- `--detach` prints the run id and returns. The run goes on.
- `attach` prints the run from an event sequence number (`--from`,
  default 0) to its end. If you press Ctrl-C or lose the connection,
  the run keeps going, and you can attach again.
- `cancel` stops the generation. The run ends with `[cancelled]` and
  the turn is not added to its conversation.
- `result` prints a finished run's text, finish reason and token
  counts. The release keeps finished runs for `run_retention_ms`
  (default 5 minutes), then drops them.

Every event of a run has a sequence number. The release keeps the last
`run_buffer_events` events of each run (default 1024). If you attach
from a sequence the release no longer holds, the CLI says which events
were lost and continues from the oldest one still held. A client that
reads too slowly is cut off after `subscriber_max_unacked` unread
events (default 256) and reattaches on its own from where it stopped.

## Point the CLI at a release

| Variable | Default | Meaning |
|---|---|---|
| `ERLLAMA_NODE` | `erllama_server@127.0.0.1` | the release node |
| `ERLLAMA_COOKIE` | the cookie in `~/.erlang.cookie` | distribution cookie |
| `ERLLAMA_TIMEOUT` | `10000` | timeout of a control command, in ms |
| `ERLLAMA_DIST_OPTFILE` | unset | use TLS distribution with this `-ssl_dist_optfile` |

The CLI starts a temporary hidden node named `erllama_cli_<N>@<host of
the release>`. Hidden means it does not appear in the release's
`nodes()` and is not connected to the release's other nodes.

The release node uses the cookie in `~/.erlang.cookie` of the user that
starts it. Create that file before the first `daemon` on a new machine:
the start script reads the cookie before the node exists, and without
one it waits forever for a node it cannot reach.

```bash
head -c 24 /dev/urandom | base64 | tr -dc A-Za-z0-9 > ~/.erlang.cookie
chmod 400 ~/.erlang.cookie
```
 Run the CLI as the same user, or set `ERLLAMA_COOKIE` to
that cookie. `RELX_COOKIE` only changes the cookie of the release's own
control commands (`ping`, `stop`, `eval`), not the node's.

## Administer a release on another machine

By default the release listens on the loopback interface only
(`-kernel inet_dist_use_interface {127,0,0,1}` in `config/vm.args.src`).
To reach it from another machine, use TLS distribution on both sides.

1. Write an `ssl_dist.conf` for each side, with certificates both sides
   trust:

```erlang
[{server, [{certfile, "/etc/erllama/server_cert.pem"},
           {keyfile, "/etc/erllama/server_key.pem"},
           {cacertfile, "/etc/erllama/ca.pem"},
           {verify, verify_peer},
           {fail_if_no_peer_cert, true}]},
 {client, [{certfile, "/etc/erllama/client_cert.pem"},
           {keyfile, "/etc/erllama/client_key.pem"},
           {cacertfile, "/etc/erllama/ca.pem"},
           {verify, verify_peer}]}].
```

2. In the release's `vm.args`, replace the loopback line with the TLS
   flags and a host name the certificates cover:

```text
-name erllama_server@llm.example.com
-proto_dist inet_tls
-ssl_dist_optfile /etc/erllama/ssl_dist.conf
```

3. Run the CLI with the client side's file:

```bash
ERLLAMA_NODE=erllama_server@llm.example.com \
ERLLAMA_DIST_OPTFILE=$HOME/.erllama/ssl_dist.conf \
    bin/erllama status
```

The release carries the `ssl` application for this. Agent tools that
read or write files always run on the release machine: a workspace is
a path on that machine, and the CLI never uploads your local files.

## Events you receive

A process attached to a job (`erllama_server_api:attach/3`) receives
`{erllama_server_run, JobId, Event}`, where `Event` is
`#{seq := N, type := Type, data := Data}`. Ack with
`JobPid ! {ack, self(), Seq}`; a subscriber that falls too far behind
gets `{erllama_server_run, JobId, {dropped, NextSeq}}` and reattaches
from `NextSeq`.

A run:

| Type | When |
|---|---|
| `started` | the run begins: `run_id`, `conversation_id`, `model` |
| `loading` | its model is not loaded yet and is being loaded |
| `compacting`, `compacted` | its conversation is shortened first (see [Conversations](conversations.md)) |
| `token` | a piece of the reply's content |
| `thinking` | a piece of the reasoning |
| `tool_call` | a parsed tool call (the run never executes it) |
| `logprobs` | token log-probabilities, when the request asked for them |
| `done` | terminal: the result, with `finish_reason`, the text, tool calls and stats |
| `cancelled` | terminal: the run was cancelled |
| `error` | terminal: `#{error => Reason}` |

An agent session (see [Agent sessions](agent.md)):

| Type | When |
|---|---|
| `started` | the session begins: `agent_id`, `conversation_id`, `model`, `workspace` |
| `turn_started` | a turn begins with its `input` |
| `compacting`, `compacted` | the conversation is shortened at the start of a turn, or on request |
| `round` | a generation round of the turn begins |
| `token`, `thinking`, `tool_call` | the round's reply, as for a run |
| `approval_request` | a call waits for `agent_approve/3`: `call_id`, `name`, `arguments` |
| `tool_started` | a call runs: `call_id`, `name`, `arguments` |
| `tool_result` | a call finished: `call_id`, `name`, `ok`, a preview of the `output` |
| `turn_done` | the turn ended: `rounds`, `tokens` and its `reason` |
| `done` | terminal: the session stopped |

A download (`pull`):

| Type | When |
|---|---|
| `started` | the download begins: `job_id`, `source` |
| `progress` | the `file` being fetched, bytes `done` and `total` |
| `done`, `cancelled`, `error` | terminal; `done` carries the registered `model` |

## When something goes wrong

| Message | What it means |
|---|---|
| `release not reachable at NODE` | the node is not running, the name is wrong, or the cookie differs |
| `the release speaks API version S and this client C` | the CLI comes from another build; use the `bin/erllama` shipped with the release |
| `has no erllama service API` | the node runs but is not an erllama_server release, or an older one |
| `model M is not loaded` | load it first, or check `erllama models` |
| `the conversation already has a run in progress` | wait for it, cancel it, or use another conversation |
| `the release is running its maximum number of runs` | `max_active_runs` runs are generating; try again later |

## Notes

- The CLI never starts the inference application or loads the NIF; it
  runs on the release's bundled runtime when the release has one.
- The CLI and the release check each other's API version on every
  connection. The CLI, other Erlang nodes and the HTTP adapters all go
  through the same service API, the module `erllama_server_api`.
- When the CLI's input is not a terminal (a pipe, a script), Ctrl-C
  ends the CLI at once and the release keeps any run going; use
  `erllama cancel` to stop it.
