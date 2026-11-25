# embedded_app

erllama used as a library inside an OTP application: a supervised
`summarizer` process loads a model and summarizes text for the other
processes of the application. There is no HTTP server and no separate
service; inference is one more process in your supervision tree. Use it
as a starting point when your Erlang system needs a local model for
some of its own work.

What it shows:

- the model loads when the application starts and unloads when it
  stops (`summarizer_app:stop/1`);
- a summary streams to the caller as messages
  (`{summarizer, Job, {text, Bin}}`, then `{done, Summary}`);
- as many jobs as the model has sequences (`parallel`) run at once;
  one more is refused with `{error, seq_capacity}` instead of waiting;
- a caller that exits gets its job cancelled (the summarizer monitors
  callers);
- if the summarizer crashes, its supervisor restarts it; the model
  stays loaded (it belongs to erllama's supervisor) and erllama
  cancels the generations that were streaming to the dead process;
- `summarizer:info/0` reads the model's sequences, its waiting
  requests, the last cache hit and the cache counters from erllama.

## Run

You need a GGUF chat model; a 1.5B Qwen2.5 Instruct is enough.

```sh
cd examples/embedded_app
SUMMARIZER_MODEL=/srv/models/qwen2.5-1.5b-instruct-q4_k_m.gguf rebar3 shell
```

```erlang
1> summarizer:summarize_sync(<<"Erlang processes are isolated...">>, 30000).
{ok,<<"Erlang processes are isolated and can recover from faults without stopping.">>}
2> summarizer_demo:run().
```

`summarizer_demo:run/0` goes through each behaviour above. With
`parallel` set to 2 (`application:set_env(summarizer, parallel, 2)`
before the application starts), on an Apple M-series machine with Metal:

```text
Erlang processes are isolated and can recover from faults without stopping.
[14 messages, 161 ms]
one more while 2 run: {error,seq_capacity}
info: #{waiting => 0,
        cache => #{hits_longest_prefix => 0,saves_finish => 0,misses => 3,hits_exact => 0},
        last_cache_hit => #{kind => cold,prefix_len => 0},
        jobs => 2,
        sequences => #{n_seq_max => 2,available_seqs => 0}}
its caller exited; erllama's request: {error,not_found}
restarted: true, model: {ok,<0.117.0>}
the stream to the killed process: {error,not_found}
after the restart: Erlang processes are isolated and can recover from faults without stopping.
```

`{error, not_found}` from `erllama:request_info/1` means the request is
gone: it was cancelled.

The application environment: `model_path` (or `SUMMARIZER_MODEL`),
`n_ctx` (default 4096, shared by the jobs running at once) and
`parallel` (default 4).

## Notes

- The prompts here are short, so the cache never saves them (erllama
  keeps prompts of at least `min_tokens`, 512 by default): every job
  is a cold prefill. A long shared system prompt would be restored
  from the cache instead.
- Stop the application (or `q()` in the shell) before the VM exits:
  `halt()` with a model still loaded makes llama.cpp's Metal teardown
  abort at exit on macOS.
- To run against this repository's erllama instead of the Hex package:
  `mkdir _checkouts && ln -s ../../../apps/erllama _checkouts/erllama`.
- The server release does the same for other programs, over HTTP or
  Erlang distribution: see [service_client](../service_client).
