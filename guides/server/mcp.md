# MCP servers

The release can use the tools of MCP (Model Context Protocol) servers
in agent sessions, and can itself be an MCP server so other MCP clients
can use its models. You need this page to connect a server such as a
Git, database or search server to your agent sessions, to control what
its tools may do, or to expose the release at `/mcp`.

## Connect a server

List the servers in the release's `sys.config`. A server runs as a
local command (stdio) or is reached over HTTP (Streamable HTTP).

```erlang
{erllama_server, [
    {mcp_servers, [
        #{name => "git", transport => stdio,
          command => "/usr/local/bin/mcp-server-git",
          args => ["--repository", "/srv/projects/app"],
          env => [{"GIT_PAGER", "cat"}]},
        #{name => "search", transport => http,
          url => "https://search.internal/mcp",
          token => "...",
          policy => auto}
    ]}
]}
```

- `name`: letters, digits, `_` and `-`, unique. The model sees the
  server's tools as `name__tool`, for example `git__git_status`.
- stdio: `command` (an absolute path), optional `args` and `env`. The
  release starts the command and stops it when the release stops.
- HTTP: `url`, optional `token` (sent as a bearer token) and `headers`
  (`[{Name, Value}]`).
- `policy`: `ask` (the default), `auto` or `deny`, for every tool of
  the server.
- `mcp_timeout_ms` (default 60000) bounds every call to a server.

The release starts with the servers in the background. A server that is
slow, down or misconfigured never stops the release or a session: its
tools are simply not offered until it answers. The release reconnects
to a server that goes away, with growing delays up to a minute, and
refreshes each server's tool list every 30 seconds.

```bash
R=_build/prod/rel/erllama_server
$R/bin/erllama mcp
git  stdio  ready  12 tools
search  http  connecting  0 tools
$R/bin/erllama mcp tools
git__git_status  ask  read  Shows the working tree status
...
```

`erllama mcp resources SERVER` lists a server's resources and
`erllama mcp read SERVER URI` prints one.

## What a session sees

- Each generation round offers the native tools and the tools of the
  servers that are ready at that moment. The model's calls in that
  round resolve against that same list, so a server that comes or goes
  mid-round changes nothing until the next round.
- A tool the server marks read-only (`readOnlyHint`) runs alongside
  the other reads. Any other MCP tool is ordered like a native write:
  it waits for the calls before it and holds back the calls after it.
- The model gets the tool's text. Structured content is added as JSON;
  images and audio are described (type and size), not shown; embedded
  resources are shown when they are text.
- A tool error, a timeout or a server that went away becomes an error
  result the model can act on. The turn goes on.

## Policy

The server's `policy` is the default for its tools. In a session:

| Session | MCP tool |
|---|---|
| default | the server's policy |
| `--yes` | runs without asking |
| `--read-only` | refused unless the server marks it read-only |
| `policy => #{<<"git__git_commit">> => deny}` (service API) | that tool's own policy |

A server set to `deny` stays denied whatever the session says: its
tools are not offered, and a call to one is refused. Answering `a`
(always) to an approval allows that one tool for the rest of the
session.

## Serve the release over MCP

Turn on the endpoint with the HTTP listener:

```erlang
{erllama_server, [
    {http, #{port => 8080}},
    {mcp_endpoint, true},
    {api_keys, ["change-me"]}
]}
```

The endpoint is `/mcp` on the HTTP listener, behind the same API keys
(`Authorization: Bearer KEY`). It offers two tools:

| Tool | Does |
|---|---|
| `erllama_models` | the models the release can run, and which are loaded |
| `erllama_chat` | one stateless chat turn: `prompt`, optional `model`, `system`, `max_tokens` (default 1024) |

A chat counts against `max_active_runs` like any HTTP request, shows in
`erllama runs`, and is cancelled when the MCP client gives up on it.

From Erlang, with barrel_mcp:

```erlang
{ok, C} = barrel_mcp_client:start(#{
    transport => {http, <<"http://127.0.0.1:8080/mcp">>},
    auth => {bearer, <<"change-me">>}
}),
{ok, Answer} = barrel_mcp_client:call_tool(C, <<"erllama_chat">>, #{
    <<"prompt">> => <<"Name three Erlang behaviours.">>
}).
```
