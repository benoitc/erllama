# wasi-nn guests

`erllama_wasi_nn` lets a WebAssembly component that imports `wasi:nn`
run its models on erllama. The component names a model and hands over
text; erlang_wasm calls this backend, and the backend runs a completion
or an embedding on a model you loaded. You need this page when you run
components with [erlang_wasm](https://github.com/benoitc/erlang_wasm)
and want their inference served by the models of your node.

It is a separate application, `apps/erllama_wasi_nn`. The erllama
library does not depend on erlang_wasm.

## Add the application

`wasi_nn` is not in a Hex release of erlang_wasm yet, so
`erllama_wasi_nn` depends on it by git and is not on Hex itself. Use
it from this repository, and start it with its dependencies:

```erlang
{ok, _} = application:ensure_all_started(erllama_wasi_nn).
```

## Load the models

The backend never loads a model. Load each one yourself, under an id
you choose:

```erlang
{ok, <<"qwen">>} = erllama:load_model(<<"qwen">>, #{
    model_path => "/srv/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    context_opts => #{n_ctx => 4096}
}),
{ok, <<"bge">>} = erllama:load_model(<<"bge">>, #{
    model_path => "/srv/models/bge-small-en-v1.5.Q8_0.gguf",
    context_opts => #{embeddings => true, n_ctx => 512}
}).
```

See [Loading a model](loading.md) and [Embeddings](embeddings.md).

## Build the grant

The grant names the backend, the names the guest may load, and for
each name the model that answers it:

```erlang
Grant = #{
    backend => erllama_wasi_nn,
    models => [<<"default">>, <<"embed">>],
    backend_opts => #{
        <<"default">> => #{
            model => <<"qwen">>,
            task => completion,
            request_opts => #{response_tokens => 256}
        },
        <<"embed">> => #{model => <<"bge">>, task => embedding}
    }
}.
```

| key | meaning | left out |
| --- | --- | --- |
| `model` | the erllama model id | required |
| `task` | `completion` or `embedding` | required |
| `request_opts` | the options of `erllama:complete/3` | `#{}` |

A name must be in `models` and have an entry in `backend_opts`. A
guest that loads any other name reads `not-found`, as it does when the
model of an entry is not loaded.

## Run a component

Pass the grant as the `nn` option:

```erlang
{ok, Bin} = file:read_file("summarise.component.wasm"),
{ok, #{stdout := Out, exit_code := 0}} =
    wasi_preview2:run_command(Bin, <<"Once upon a time">>, #{nn => Grant}).
```

When you instantiate the component yourself, build the imports from
the same grant:

```erlang
{ok, Inst} = wasm_component:instantiate(
    Bin,
    wasi_preview2:command(#{nn => Grant}),
    #{resource_closer => fun wasi_preview2:close_resource/1}
).
```

## What the guest sends and receives

The guest calls `load-by-name`, `init-execution-context` and `compute`.
The backend reads the first input tensor only. Its name and its
dimensions are ignored: guests commonly send text with dimensions
`[1]`.

| task | first input | output |
| --- | --- | --- |
| `completion` | type `u8`, a UTF-8 prompt | one tensor named `output`: type `u8`, dimensions `[bytes]`, the reply as UTF-8 |
| `embedding` | type `u8`, UTF-8 text | one tensor named `embedding`: type `fp32`, dimensions `[N]`, `N` floats of 32 bits, little-endian |

The guest sets nothing but the text. Sampling, length and stop strings
come from `request_opts`. The reply comes once, when the model has
finished: nothing streams.

## Errors the guest reads

| erllama answers | the guest reads |
| --- | --- |
| `not_loaded`, `not_found` | `not-found` |
| `timeout`, `decode_timeout` | `timeout` |
| `too_large`, `context_overflow`, `oom` | `too-large` |
| `not_supported`, `chat_not_supported` | `unsupported-operation` |
| anything else | `runtime-error` |

No input, a first input that is not `u8`, and bytes that are not UTF-8
are `invalid-argument`. The message is the name of erllama's reason,
or a fixed text: no path and no term from the host reaches the guest.

## Notes

- **Time.** erlang_wasm does not bound `compute`. A completion is
  bounded by `response_tokens` (64 when left out) and by the decode
  budget of the model; see [Configuration](configuration.md). To bound
  the wall-clock time, run the component in a worker process and kill
  the worker.
- **A caller that dies.** The model is not told. It finishes the
  completion nobody waits for, up to `response_tokens`, then serves the
  next request. Keep `response_tokens` as low as the task allows.
- **Concurrency.** Several components can share one model. Requests
  run together up to `context_opts.n_seq_max` and queue beyond it; set
  `on_full => error` in `request_opts` to fail instead of waiting.
- **Tensor bytes.** erlang_wasm holds 16 MiB of tensors for one
  process. A reply over that reads `too-large`.
- **Closing.** Dropping a graph or a context frees nothing in erllama.
  Unload a model with `erllama:unload/1` when no grant names it.
- Not supported: streaming, options set by the guest, chat templates,
  tools, media, LoRA adapters, and the older `wasi_ephemeral_nn`
  interface.
