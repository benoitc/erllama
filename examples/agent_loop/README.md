# agent_loop

A tool-calling agent loop on top of `erllama:chat/3`: the model gets
three tools (current time, arithmetic, directory listing), calls them as
it sees fit, and the loop feeds the results back until the model answers
in plain text. Use it as the starting point for your own agent; it is a
standalone rebar3 application that depends on the `erllama` hex package.

You need a GGUF whose chat template supports tools (Qwen2.5 Instruct,
Llama 3.1 Instruct, Mistral, Hermes, ...). A 1.5B Qwen2.5 is enough to
see the loop work.

## Run

```sh
cd examples/agent_loop
rebar3 shell
```

```erlang
1> agent_loop:run("/srv/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                  <<"What is 17 * 23, and what time is it now?">>,
                  #{chat_template => "priv/qwen2.5-instruct.jinja"}).
round 1: 322 prompt tokens (0 from cache), 41 generated, 2 tool call(s)
  tool calculate({"expression":"17 * 23"}) -> 391
  tool get_current_time({}) -> 2026-08-23T14:46:55Z
round 2: 413 prompt tokens (363 from cache), 43 generated, 0 tool call(s)
  assistant: The result of 17 * 23 is 391. The current time is 2026-08-23T14:46:55Z.
{ok,<<"The result of 17 * 23 is 391. The current time is 2026-08-23T14:46:55Z.">>}
```

`run/3` takes options: `max_rounds` (default 8), `verbose` (default
true), `system`, `chat_template` (path to a Jinja file that replaces the
GGUF's template), and any `erllama:request_opts()` key such as
`temperature`.

About `chat_template`: the `Qwen/Qwen2.5-*-Instruct-GGUF` files ship a
template whose tool-call example uses doubled braces (`{{"name": ...}}`);
the model copies them and the parser rejects the call. llama.cpp keeps a
corrected template under `models/templates/`, copied here as
`priv/qwen2.5-instruct.jinja`. Newer conversions do not need it.

## What the code does

1. `erllama:load_model/1` with the model path; `run/2` unloads at the end.
2. Builds the messages (`system`, `user`) as Erlang maps and calls
   `erllama:chat(Model, Messages, #{tools => agent_loop:tools(), ...})`.
3. If the parsed message carries `tool_calls`, runs each through
   `agent_loop:call_tool/2`, appends the assistant turn (with the calls in
   the OpenAI `tool_calls` shape) and one `tool` message per result, and
   calls `chat/3` again.
4. Every call in the loop passes the same `session_id`, so erllama keeps
   the conversation's KV cells pinned between rounds and only prefills the
   new suffix; `cache_delta.read` in the printed stats shows it. The
   session is released with `erllama:end_session/2` when the loop ends.

Streaming variant: replace `chat/3` with `chat_apply/3` + `stream/3` +
`chat_parse/3` (see the tool-calls guide).

From a script, end the VM with `init:stop()` rather than `halt()`:
`halt/0` runs ggml's Metal teardown before the model resources are
released and trips an assertion on exit (harmless, but noisy).
