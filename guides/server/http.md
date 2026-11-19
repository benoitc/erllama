# The HTTP APIs

The release serves OpenAI-compatible HTTP endpoints, so existing SDKs
and tools can use the models it holds. You need this page to point a
client at the release, to secure the listener, and to know exactly
which request fields are honoured.

Every request is a run: it shows in `erllama runs`, counts against
`max_active_runs`, and can be cancelled from the CLI. A request is
cancelled as soon as its client disconnects, so no model keeps
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

## Security and limits

| Setting | Default | Meaning |
|---|---|---|
| `http` | off in the library, `#{port => 8080}` in the release | listener; `ip` defaults to `{127,0,0,1}`; `ERLLAMA_HTTP_PORT` overrides the port |
| `api_keys` | `[]` | accepted keys, as `Authorization: Bearer KEY` or `x-api-key: KEY`; `[]` means no check |
| `allow_unauthenticated` | `false` | allow a non-loopback `ip` without keys (refused at start otherwise) |
| `max_body_bytes` | 32 MiB | larger bodies get `413` |
| `max_http_requests` | 64 | requests in flight; more get `429` with `Retry-After` |
| `max_active_runs` | 64 | generations at once across HTTP and CLI; more get `429` |

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

## Errors

Errors use OpenAI's shape:
`{"error": {"message", "type", "param", "code"}}`. The status tells
you the kind:
- `400`: an invalid or unsupported field, with `param` naming it;
- `401`: no valid key;
- `404`: an unknown model;
- `413`: the body is too large;
- `429`: the server is at capacity;
- `500`: anything else.

An error during a stream arrives as a final event with an `error`
object, followed by `[DONE]`.

## Operations

| Endpoint | Answers |
|---|---|
| `GET /health` | `200` while the release runs |
| `GET /ready` | `200` when the engine, the registry and the model manager are up, `503` and the failed checks otherwise |
| `GET /metrics` | Prometheus text: HTTP request metrics, erllama's cache counters, loaded models, active runs and downloads |
