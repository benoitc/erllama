# erllama_wasi_nn

A `wasi:nn` backend for [erlang_wasm](https://github.com/benoitc/erlang_wasm):
a WebAssembly component that imports `wasi:nn` runs text completions
and embeddings on models loaded in erllama. You need this app when you
run components with erlang_wasm and want their inference served by
your node.

```erlang
{ok, _} = application:ensure_all_started(erllama_wasi_nn),
{ok, <<"qwen">>} = erllama:load_model(<<"qwen">>, #{model_path => Path}),
Grant = #{
    backend => erllama_wasi_nn,
    models => [<<"default">>],
    backend_opts => #{<<"default">> => #{model => <<"qwen">>, task => completion}}
},
{ok, #{stdout := Out}} = wasi_preview2:run_command(Component, Prompt, #{nn => Grant}).
```

The guide is [guides/wasi-nn.md](../../guides/wasi-nn.md).

## Notes

- The app depends on erlang_wasm by git, pinned to the commit that
  added `wasi_nn`. It cannot be published to Hex until erlang_wasm
  releases that module.
- The tests run without a model file. Set `LLAMA_TEST_MODEL` to a GGUF
  file to also run one command against a real model:

```bash
LLAMA_TEST_MODEL=/path/to/model.gguf rebar3 ct --dir apps/erllama_wasi_nn/test
```
