# The HTTP APIs

The release serves OpenAI-, Anthropic- and Ollama-compatible HTTP
endpoints, so existing SDKs and tools can use the models it holds. You need this page to point a
client at the release, to secure the listener, and to know exactly
which request fields are honoured.

What a request costs depends on its work:
- a generation request (chat, completions, Responses, Anthropic
  messages, Ollama chat and generate) is a run: it shows in
  `erllama runs`, counts against `max_active_runs`, and can be
  cancelled from the CLI;
- an embedding request takes one of the same `max_active_runs` slots
  while it runs, and answers `429` when they are all taken; it is not
  listed as a run;
- token counts and model lookups (`/v1/models`, `/api/tags`,
  `/api/show`, `/api/ps`) take no slot, so they answer while the
  release is full.

A run is cancelled as soon as its client disconnects, so no model keeps
generating for a closed connection.

## Turn it on and call it

```erlang
%% config/sys.config
{erllama_server, [
    {http, #{port => 8080}},            %% 127.0.0.1 unless `ip' is set
    {api_keys, [<<"sk-local-1">>]}      %% optional on loopback
]}
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'authorization: Bearer sk-local-1' \
  -H 'content-type: application/json' \
  -d '{"model":"tiny","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local-1")
reply = client.chat.completions.create(
    model="tiny", messages=[{"role": "user", "content": "Hello"}])
print(reply.choices[0].message.content)
```

The model is a registry name ([Managing models](models.md)), loaded on
demand, or omitted for the default model.

## Serving MLX on a Mac

Install the [optional MLX runtime](../building.md#optional-mlx-lm-backend-on-apple-silicon)
and export `ERLLAMA_MLX_PYTHON` in the server's environment. Point a registry
entry at the local checkpoint directory in `sys.config`. The registry keeps
the backend and slot count under the same model name after restart.

```erlang
{erllama_server, [
    {http, #{ip => {127,0,0,1}, port => 8080}},
    {models, [#{name => <<"qwen">>, path => "/models/Qwen3.8-27B-MLX-4bit",
                backend => erllama_model_mlx, n_ctx => 4096, parallel => 4}]}
]}
```

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Explain OTP supervision."}],"temperature":0,"reasoning_effort":"none","max_tokens":64,"stream":true}'
```

Set `stream` to `false` for a single JSON response. Closing a stream cancels
its request and frees its slot. Cache reuse follows the configured minimum
prompt length and checkpoint policy.

## Security and limits

| Setting | Default | Meaning |
|---|---|---|
| `http` | off in the library, `#{port => 8080}` in the release | listener; `ip` defaults to `{127,0,0,1}`; `ERLLAMA_HTTP_PORT` overrides the port |
| `api_keys` | `[]` | accepted keys, as `Authorization: Bearer KEY` or `x-api-key: KEY`; `[]` means no check |
| `allow_unauthenticated` | `false` | allow a non-loopback `ip` without keys (refused at start otherwise) |
| `max_body_bytes` | 32 MiB | larger bodies get `413` |
| `max_http_requests` | 64 | requests in flight; more get `429` with `Retry-After` |
| `max_active_runs` | 64 | generations at once across HTTP and CLI; more get `429` |

Requests on the same model run together, up to the model's `parallel`
(default 4); past that they wait for a free one. See
[Serve several clients at once](models.md#serve-several-clients-at-once).

`/health` and `/ready` never need a key, so probes work behind an
authenticated listener.

## OpenAI endpoints

| Endpoint | Streaming |
|---|---|
| `GET /v1/models`, `GET /v1/models/{id}` | no |
| `POST /v1/chat/completions` | SSE `chat.completion.chunk`, then `[DONE]` |
| `POST /v1/completions` | SSE `text_completion`, then `[DONE]` |
| `POST /v1/embeddings` | no |

### Chat completions fields

| Field | Support |
|---|---|
| `model`, `messages` | supported; roles `system`, `user`, `assistant`, `tool` |
| message `content` | a string, or parts: `text`, `image_url` with a base64 `data:` URL, `input_audio` (base64) |
| `tool_calls` (assistant), `tool_call_id`, `name` | supported, kept across turns |
| `tools` | function tools only |
| `tool_choice` | `auto`, `none`, `required`, or a named function (only that tool is offered, and a call is required) |
| `parallel_tool_calls` | supported |
| `response_format` | `text`, `json_object`, `json_schema` (enforced by grammar) |
| `stream`, `stream_options.include_usage` | supported |
| `max_tokens`, `max_completion_tokens` | supported |
| `temperature`, `top_p`, `seed`, `stop`, `frequency_penalty`, `presence_penalty` | supported |
| `logprobs`, `top_logprobs` (0 to 20) | supported |
| `reasoning_effort` | `none` turns thinking off; any other value turns it on |
| `n` | only `1` |
| `keep_alive` (extension) | seconds, or `"30s"`, `"5m"`, `"1h"`; negative keeps the model loaded |
| `user`, `metadata`, `store` | accepted and not used |
| anything else (`logit_bias`, `audio`, `modalities`, ...) | refused with `400`, naming the field |

Notes:
- With tools, the template's tool grammar constrains the output, so a
  `json_schema` `response_format` with tools is refused, as in erllama.
- Tool calls are parsed by llama.cpp's parser for the model's template.
  In a stream, each call is announced with its id and name as soon as
  it is recognised, and its arguments follow at the end; call syntax
  never appears in `content`.
- Reasoning models stream their reasoning as `reasoning_content`.
- `finish_reason` is `stop`, `length` or `tool_calls`.

### Images and audio

A model takes images or audio only when its registry entry has a
projector that covers the modality (`erllama pull hf://ORG/REPO` brings
the repository's mmproj; `erllama load FILE --mmproj PROJ`). Otherwise
the request gets a `400` that says why. Images must be base64 `data:`
URLs: the server does not fetch remote URLs. Requests with media are
evaluated whole and do not reuse the KV cache.

### Completions and embeddings

- `/v1/completions` takes `prompt` (a string), the sampling and length
  fields above, `stream` and `logprobs`. `echo` and `n > 1` are
  refused.
- `/v1/embeddings` takes `input` (a string, an array of strings, or
  token arrays) and `encoding_format` (`float` or `base64`).
  `dimensions` is refused.

## OpenAI Responses

| Endpoint | Streaming |
|---|---|
| `POST /v1/responses` | SSE typed events: `response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`/`.done`, `response.function_call_arguments.delta`/`.done`, `response.output_item.done`, then `response.completed`, `response.incomplete` (stopped at `max_output_tokens`, or cancelled) or `response.failed`, each with `sequence_number` |
| `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}` | no |

| Field | Support |
|---|---|
| `input` | a string, or items: messages (roles `user`, `assistant`, `system`, `developer`; parts `input_text`, `output_text`, `input_image` as a base64 `data:` URL), `function_call`, `function_call_output`; `reasoning` items are skipped |
| `instructions` | added as a system message to this turn |
| `previous_response_id` | continues that response's conversation, with its history and KV cache warmth |
| `store` | default `true`: the response can be read back and continued; `false` without `previous_response_id` is stateless |
| `tools`, `tool_choice`, `parallel_tool_calls` | function tools; `auto`, `none`, `required`, or a named function |
| `text.format` | `text`, `json_object`, `json_schema` |
| `reasoning.effort` | `none` turns thinking off; any other value turns it on |
| `max_output_tokens`, `temperature`, `top_p`, `stream` | supported |
| `truncation` | only `disabled` |
| `metadata`, `user`, `service_tier` | accepted and not used |
| anything else (`background`, `include`, built-in tools, ...) | refused with `400` |

A stream ends with `response.completed` when the answer is complete,
and with `response.incomplete` when it is not: stopped at
`max_output_tokens` (status `incomplete`, `incomplete_details.reason`
`max_output_tokens`), or cancelled. OpenAI has a `cancelled` status
but no event for it; erllama's convention is to end a cancelled stream
with `response.incomplete` carrying status `cancelled` and the text
made so far. The response is cancelled when the client goes away or
with `erllama cancel`.

With `store` on, `GET` reads back the object the stream ended with,
including a cancelled one, even when the client went away before the
run had stopped. Continuing a cancelled response with
`previous_response_id` starts from the last turn that completed: the
cancelled turn's input and partial reply are not added to the
conversation. With `store: false` nothing is kept.

Stored responses are written under `data_dir/responses` and are loaded
again when the release starts.

## Anthropic Messages

| Endpoint | Streaming |
|---|---|
| `POST /v1/messages` | SSE: `message_start`, `content_block_start`/`_delta`/`_stop` (`text_delta`, `thinking_delta`, `input_json_delta`), `message_delta` with the stop reason and usage, `message_stop` |
| `POST /v1/messages/count_tokens` | no |

| Field | Support |
|---|---|
| `model`, `max_tokens` (required), `messages` | supported; content as a string or blocks: `text`, `image` (base64 source), `tool_use`, `tool_result` (`is_error` prefixes the result with `Error:`) |
| `system` | a string or text blocks |
| `tools`, `tool_choice` | `auto`, `any`, `tool` (named), `none`; `disable_parallel_tool_use` |
| `thinking` | `enabled` or `disabled`; the budget is not applied |
| `stop_sequences`, `temperature`, `top_p`, `top_k`, `stream` | supported |
| `metadata`, `service_tier` | accepted and not used |
| anything else | refused with `400` |

`x-api-key` and `Authorization: Bearer` both carry the key; the
`anthropic-version` header is accepted and not checked. `stop_reason`
is `end_turn`, `max_tokens`, `stop_sequence` or `tool_use`. A
`tool_use` block is sent whole once its arguments are known, after the
text.

## Ollama

| Endpoint | Notes |
|---|---|
| `POST /api/chat` | NDJSON stream by default, one object with `"stream": false`; `messages` with `images` (base64), `tool_calls`, and `tool` results (matched to the calls in order); `tools`, `format` (`"json"` or a schema), `options`, `think`, `keep_alive` |
| `POST /api/generate` | as chat for one prompt (`system`, `images`); `raw: true` sends the prompt without the template; an empty prompt loads the model (`keep_alive: 0` unloads it) |
| `POST /api/embed`, `POST /api/embeddings` | `input` (a string or strings), `prompt` (legacy) |
| `GET /api/tags`, `POST /api/show`, `GET /api/ps` | the registry, one model's metadata and template, the loaded models |
| `POST /api/pull` | a registry name (`llama3.2:1b`) or `hf.co/ORG/REPO[:QUANT]`; NDJSON status lines (`pulling manifest`, `pulling model` with `completed`/`total`, `success`) |
| `DELETE /api/delete`, `POST /api/copy`, `GET /api/version`, `GET /` | as Ollama |
| `POST /api/create`, `POST /api/push` | `501`: models come from pulls or GGUF files |

`options` accepts `num_predict`, `temperature`, `top_p`, `top_k`,
`min_p`, `seed`, `stop`, `repeat_penalty`, `repeat_last_n`,
`frequency_penalty` and `presence_penalty`. `num_ctx` is accepted but
not applied: the context size is the model entry's. Any other option is
refused with `400`. `suffix`, `template` and `context` on generate are
refused.

## Errors

Errors use the protocol's own shape:
- OpenAI: `{"error": {"message", "type", "param", "code"}}`;
- Anthropic: `{"type": "error", "error": {"type", "message"}}`;
- Ollama: `{"error": "message"}`. The status tells
you the kind:
- `400`: an invalid or unsupported field, with `param` naming it;
- `401`: no valid key;
- `404`: an unknown model;
- `413`: the body is too large;
- `429`: the server is at capacity;
- `503`: a new inference request cannot use its rank group. Read
  `error.code` (OpenAI and Anthropic) or `code` (Ollama).
  `tp_rank_lost` means the generation lost a rank. Start a new generation.
  `incompatible_ranks` means the ranks disagree on the model, options,
  group size or build. Check the rank numbers in `/status` before restarting.
  `group_unavailable` covers other group failures, such as a timed-out join;
- `500`: other server failures.

An error during a stream arrives as a final event in the protocol's
own shape:
- OpenAI chat and completions: a chunk with an `error` object, then
  `data: [DONE]` (a stream that ends normally also ends with
  `[DONE]`);
- OpenAI Responses: `response.failed`;
- Anthropic: an `error` event (a stream that ends normally ends with
  `message_stop`);
- Ollama: a last NDJSON line `{"error": "message"}` (a stream that
  ends normally ends with an object whose `done` is `true`).

## Operations

| Endpoint | Answers |
|---|---|
| `GET /health` | `200` while the release runs |
| `GET /ready` | `200` when the engine, registry, model manager and all grouped models are ready, `503` and the failed checks otherwise |
| `GET /metrics` | Prometheus text: HTTP request metrics, erllama's cache counters, loaded models, active and waiting runs, downloads, failures by kind and rank group gauges |
| `GET /rank` | on a rank node, its local identity, state and reason |
| `GET /status` | HTTP API version, ordinary server status and rank groups as JSON |

The serving node reports `api: 2` in `/status`. You can read the same
model, run, cache and failure fields as in API 1. The CLI uses its existing
RPC version. The new `groups` array has one entry per grouped model and is
empty on a standalone server.

| Group field | Meaning |
|---|---|
| `model` | Registered model name, including its tag. |
| `group_id`, `generation` | Group name and generation number. Use both to identify a run. |
| `world` | Number of configured ranks. |
| `state` | `forming` while waiting for ranks or model load, `ready` once all shares and the serving model are loaded, `lost` when this generation has failed. |
| `reason` | `null` until loss, then an object with a `code`. A rank exit can include its `rank` and exit `detail`. An identity mismatch includes the list of disagreeing `ranks`, relative to rank 0. |
| `ranks` | One entry per slot, in order from zero, with `rank`, `state` and `reason`. A rank becomes ready after loading its share. All slots become lost when the generation fails. |
| `n_seq_max` | Number of sequence slots in the model context, or `null` until rank 0 reports its capacity. |

On a grouped server, `/ready` includes the same `groups` array. When any
group is forming or lost, the response is HTTP 503 and `failed` includes
`group`. On a rank node, `/rank` returns `group_id`, `generation`, `world`,
`rank`, `state` and `reason`. Its `/ready` includes this object under
`rank`. It returns 503 with `rank` in `failed` until ready and after loss.
You can still read `/rank` after the local role reports a loss.

Scrape `erllama_server_tp_group_state` with labels `model`, `group_id`,
`generation` and `state`. Only the current state's sample is 1.
`erllama_server_tp_lost_ranks` uses the first three labels and counts slots
in the lost state. If one host fails and the group shuts down, that count
includes every slot that can no longer serve.
