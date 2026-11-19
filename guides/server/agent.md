# Agent sessions

An agent session is a model that works in a directory with tools: it
reads and searches files, edits them and runs commands, in a loop, until
it can answer you. You need this page to start a session, to control
what its tools may do, and to understand what happens when you
interrupt it or close your terminal.

Everything runs on the release machine: the workspace is a directory
there, and the CLI never sends your local files. A session belongs to
the release, not to the terminal that started it: it keeps running
when the CLI exits, and you can come back to it.

## Start a session

```bash
R=_build/prod/rel/erllama_server
$R/bin/erllama agent qwen --workspace /srv/projects/app
you> Why does the build fail? Fix it and run the tests.
[tool] read_file {"path":"rebar.config"}
[ok] 1	{erl_opts, [debug_info]}.
allow shell {"command":"rebar3 eunit"}? [y]es / [n]o / [a]lways: y
[tool] shell {"command":"rebar3 eunit"}
[ok] exit 0
The build failed because ...
you> /quit
```

- `/prompt SERVER NAME [KEY=VALUE ...]` sends an MCP server's prompt
  as your input (see [MCP servers](mcp.md)).
- `/quit` (or the end of input) leaves the session running;
  `erllama agents` lists sessions, and `erllama agent-stop AGENT` ends
  one.
- `--conversation ID` continues an earlier conversation; the session's
  history is kept in it.
- `--max-rounds N` limits how many times the model may call tools in
  one turn.

## The tools

| Tool | Does | Asks by default |
|---|---|---|
| `read_file` | numbered lines, 400 at a time, with the offset to continue from | no |
| `list_dir` | a directory, up to 3 levels | no |
| `search` | a regular expression over the text files (skips hidden directories, binary and large files) | no |
| `write_file` | creates a file; refuses to replace one | yes |
| `edit_file` | replaces an exact text, refused when absent or ambiguous | yes |
| `shell` | runs `/bin/sh` in the workspace, with a timeout (default 2 min) and output capped to its first and last 16 KiB; `background: true` returns a job id | yes |
| `shell_status` | output so far of a background job | no |
| `shell_kill` | stops a background job | yes |

- Paths resolve inside the workspace. `..` and symbolic links that
  lead outside are refused.
- `list_dir` and `search` never follow a symbolic link to a directory
  (`list_dir` shows it as `name@`); `search` reads a linked file only
  when it resolves inside the workspace.
- A tool that fails returns an error to the model, which can correct
  itself, instead of stopping the turn.

## Approvals and policy

When a call needs approval, the session sends an `approval_request`
and waits, even when no terminal is attached.
- `y` allows this call;
- `n` declines it, and the model is told it was declined;
- `a` allows this tool for the rest of the session.

| `erllama agent` flag | Policy |
|---|---|
| (none) | reading runs, changing the workspace or running commands asks |
| `--yes` | every tool runs without asking |
| `--read-only` | tools that change the workspace are refused |

Through the service API, `policy` can also be a map of tool name to
`auto`, `ask` or `deny`.

## How a turn runs

1. Your input is added to the conversation and the model generates.
2. If it calls tools, the calls that need it wait for approval. Then
   they run:
   - calls that only read run together;
   - a call that changes the workspace waits for the calls before it
     and holds back the calls after it, so changes happen in the order
     the model asked for them.
3. The results go back to the model in call order, and it generates
   again.
4. The turn ends when the model answers without calling a tool, or when
   a budget runs out.

| Budget | Default |
|---|---|
| `max_rounds` | 20 generations per turn |
| `max_tokens` | 32768 generated tokens per turn |
| `max_elapsed_ms` | 10 minutes per turn |

- Each generation may produce at most the tokens left in the turn.
- When the time runs out, the turn ends where it is: a generation is
  cancelled, running tools are stopped, and a pending approval is
  dropped. The time counts from the start of the turn, waiting for
  approvals included.
- A generation that dies without answering ends the turn with an
  error; the session stays ready for the next input.

Input you send during a turn waits and starts the next one.

## Interrupting

Interrupting a turn (`agent_interrupt`) stops it at once:
- the generation is cancelled;
- every running tool is stopped, and a running command's whole process
  group is killed, including what it started in the background;
- unfinished calls are recorded as interrupted.

The session stays, ready for the next input. Ending the session, or
stopping the release, also stops its background jobs, whole process
groups included, before the session goes.

## Tools from HTTP clients

Tools declared in an HTTP request (`tools` in OpenAI, Anthropic or
Ollama requests) are never run by the server: the model's calls go back
to the client, which runs them. Only agent sessions run tools, and only
the tools listed here and those of the MCP servers the release is
configured with (see [MCP servers](mcp.md)).
