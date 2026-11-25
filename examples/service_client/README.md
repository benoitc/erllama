# service_client

An Erlang node that uses a running `erllama_server` release through its
service API (`erllama_server_api`), over Erlang distribution: start a
run, follow its events, inspect it, cancel it. The release owns the
model, the conversation and the run; your node only starts work and
reads events, so it can stop and another node can pick the run up
where it is. Use it when an Erlang system shares one inference service
instead of loading models itself.

`service_client.erl` is one module with no dependency beyond OTP.

## Run

Start the release and load a model, then compile the module:

```sh
R=_build/prod/rel/erllama_server
$R/bin/erllama_server daemon
$R/bin/erllama load ~/Models/qwen2.5-1.5b-instruct-q4_k_m.gguf --name qwen
erlc examples/service_client/service_client.erl
```

One node starts a run and exits:

```sh
erl -noshell -name a@127.0.0.1 -hidden -eval '
    N = '"'"'erllama_server@127.0.0.1'"'"',
    ok = service_client:connect(N),
    {ok, Run} = service_client:start(N, <<"qwen">>,
        <<"List the numbers from 1 to 40, one per line.">>,
        #{opts => #{response_tokens => 200}}),
    io:format("~s~n", [Run]), halt().'
```

Another node, later, follows the run from its first event:

```erlang
1> N = 'erllama_server@127.0.0.1', ok = service_client:connect(N).
2> service_client:inspect(N, <<"run_5e44005676c38077">>).
{ok,#{status => done,next_seq => 128, ...}}
3> service_client:follow(N, <<"run_5e44005676c38077">>, 0).
Here are the numbers from 1 to 40, one per line:
1
...
40
[done: stop]
```

The release keeps the last `run_buffer_events` events of each run
(default 1024). Following a 1500-token run from 0 after it ended:

```text
[events 0 to 477 are no longer kept]
...
[done: length]
```

The node must use the release's cookie (`~/.erlang.cookie`, or
`-setcookie`). A hidden node (`-hidden`) is not connected to the
release's other nodes.

## The protocol

| Call | Does |
|---|---|
| `erllama_server_api:start_run(Spec)` | starts a run: `model`, `messages` (a turn of `conversation_id`, created when absent) or `prompt`, `opts` |
| `erllama_server_api:attach(Run, Pid, Cursor)` | the events from `Cursor` on, the run's pid, its status; `Pid` then receives the live events |
| `RunPid ! {ack, self(), Seq}` | flow control: a subscriber more than `subscriber_max_unacked` events behind is sent `{dropped, Next}` and attaches again |
| `erllama_server_api:inspect(Run)` | the run's status, phase and next event sequence |
| `erllama_server_api:cancel(Run)` | stops the run; the turn is not added to its conversation |
| `erllama_server_api:status()` | what the release runs, waits for and recently failed |

Events are `{erllama_server_run, RunId, #{seq, type, data}}`; a run
ends with `done`, `cancelled` or `error`. `attach` answers
`{error, {gap, Oldest}}` for a cursor the release no longer holds.

## Library or service

- [embedded_app](../embedded_app): erllama inside your application.
  The model is a process of your node; you get tokens straight from it,
  and you decide its lifecycle. Use it when the model serves that
  application.
- This example: the release owns models, conversations and runs, and
  many clients (the CLI, HTTP clients, your nodes) share them. Runs
  outlive their clients and conversations survive restarts. Use it
  when the model is a shared service.
