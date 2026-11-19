# Managing models

The release keeps a registry of the models it knows: GGUF files you
register from disk and models it downloads for you. You need this page
to get a model onto the release, to choose what stays loaded and for
how long, and to free memory.

A request names a model; the release loads it on demand, keeps it
while requests use it, and unloads it once it has been idle for its
keep-alive.

## Download a model

```bash
R=_build/prod/rel/erllama_server
$R/bin/erllama pull hf://ggml-org/models/tinyllamas/stories260K.gguf
$R/bin/erllama pull hf://bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M
$R/bin/erllama pull llama3.2:1b
$R/bin/erllama pull https://example.com/model.gguf --sha256 9f86d0...
```

| Source | What is fetched | Registered as |
|---|---|---|
| `hf://ORG/REPO/FILE.gguf` | that file | `hf.co/org/repo:file` |
| `hf://ORG/REPO[:QUANT]` | the GGUF for `QUANT` (default `Q4_K_M`, else the only one) and the repository's mmproj projector, if any | `hf.co/org/repo:quant` |
| `ollama://[NS/]NAME[:TAG]`, or `NAME[:TAG]` | the model layer, the projector layer and the default parameters (stop strings, sampling) | `name:tag` |
| `https://...gguf` | that file, checked against `--sha256` when given | the file name |

- The progress shows on the terminal, and `--detach` returns the job id
  at once: `erllama attach JOB` follows it again and `erllama cancel
  JOB` stops it.
- Each file is checked against the size and sha256 the source
  publishes. A file that does not match is deleted and the pull fails
  with `integrity_mismatch`.
- A transfer that breaks off resumes from the bytes already on disk.
  So does a pull you cancelled: run the same `pull` again.
- Set `HF_TOKEN` (or the `hf_token` setting) for gated Hugging Face
  repositories. The token is sent to huggingface.co only, never to the
  CDN it redirects to.
- An Ollama model's Go template is not used; the chat template inside
  the GGUF file is.

## Register a file you already have

```bash
$R/bin/erllama load ~/Models/tinyllama-1.1b-chat.gguf --name tiny --n-ctx 2048
$R/bin/erllama load ~/Models/smolvlm/SmolVLM-256M-Instruct-Q8_0.gguf --name smolvlm \
    --mmproj ~/Models/smolvlm/mmproj-SmolVLM-256M-Instruct-f16.gguf
```

An argument ending in `.gguf` is a file on the release machine: it is
registered under `--name` (default: the file name) and loaded. The file
stays where it is; removing the model from the registry never deletes
it.

## See what you have

```bash
$R/bin/erllama models          # the registry, and what is loaded
$R/bin/erllama show tiny       # file, size, sha256, architecture, quantisation, context
$R/bin/erllama ps              # loaded models, who holds them, when they unload
$R/bin/erllama status          # versions, memory pressure, active runs and downloads
```

`show` reads the GGUF header, so it works on a model that is not
loaded. A model whose file has disappeared is listed as `FILE MISSING`.

## Names, copies and the default model

```bash
$R/bin/erllama cp tiny tiny:chat
$R/bin/erllama default tiny
$R/bin/erllama run tiny "Hello"
$R/bin/erllama rm tiny:chat
```

A name without a tag means `:latest`. `default` sets the model used
when a request names none. `rm` unloads the model and forgets it; a
downloaded file that no other name uses is deleted.

## Choose how long a model stays loaded

```erlang
%% config/sys.config
{erllama_server, [
    {keep_alive_ms, 300000},   %% idle time before unloading (ms or infinity)
    {default_n_ctx, 4096}      %% context size when a model sets none
]}
```

- A model stays loaded while any request uses it. Once the last one
  ends, it stays for its keep-alive, then unloads.
- The keep-alive comes from the request (`keep_alive`), else the
  model's registry entry, else `keep_alive_ms`. `0` unloads as soon as
  the request ends; `infinity` keeps the model loaded.
- `erllama load NAME` loads a model now and starts its keep-alive.
  `erllama unload NAME` unloads it at once. Runs still using it then
  end with an error.
- The context size is the model entry's `n_ctx`, else `default_n_ctx`
  capped at the context length the model was trained for.

## Memory pressure

The release turns on erllama's scheduler: when memory runs short it
evicts KV cache rows first, then asks the server to unload the least
recently used model that no request holds. Configure it under
`erllama` in `config/sys.config` (see the
[configuration guide](../configuration.md)); `model_evictor` must stay
`erllama_server_models`.

## Where things are stored

| Setting | Default | Holds |
|---|---|---|
| `data_dir` | `ERLLAMA_DATA_DIR`, else `~/.local/share/erllama` (Linux) or `~/Library/Application Support/erllama` (macOS) | `registry.json` and `blobs/` |

- `registry.json` is rewritten atomically on each change.
- A registry file that cannot be read is renamed
  `registry.json.corrupt-<time>`, and the release starts with an empty
  registry. Your files in `blobs/` are untouched: pull or register them
  again.

## Notes

- Downloads use these settings:
  - `hf_base_url` and `ollama_base_url` (point them at a mirror);
  - `max_active_pulls` (default 4);
  - `pull_idle_timeout_ms` (default 60000: a transfer that sends nothing
    for that long is resumed).
- Split GGUF files (`-00001-of-00003`) are not supported yet.
