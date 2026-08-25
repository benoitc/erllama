# Tool calls

Letting the model call functions you define: you describe your tools,
the model asks for calls, you execute them and feed the results back
until the model answers in plain text. You need this page when you
build agents, assistants with actions, or anything where the model
drives your code.

## The round trip

One tool interaction is a loop of chat turns:

```text
you                          erllama                     model
 |-- messages + tools ------> chat/3 --- rendered ------> |
 |                                                        | decides
 |<-- message with tool_calls --- parsed <--- output ---- |
 |-- run each call
 |-- messages ++ assistant turn ++ tool results
 |------------------------------> chat/3 (same session) ->|
 |<-- message with content (or more calls: loop again) ---|
```

`erllama:chat/3` does the erllama half in one call: it renders your
messages and tool definitions through the model's own chat template,
generates with the template's tool grammar enforced, and parses the
output into a structured message. Every template family llama.cpp
knows (Qwen, Llama 3, Mistral, Hermes, GPT-OSS, ...) works without
per-model configuration. Your half is running the calls and
appending the results.

## 1. Define your tools

A tool is a name, a description the model reads, and a JSON schema
for its arguments:

```erlang
Tools = [
    #{name => <<"get_weather">>,
      description => <<"Current weather for a city">>,
      parameters => #{type => object,
                      properties => #{city => #{type => string}},
                      required => [city]}}
].
```

Write descriptions for the model, not for humans: say what the tool
answers and when to use it. Keep schemas small; every property is
prompt tokens on every turn.

## 2. Ask for a turn

```erlang
Messages = [
    #{role => system, content => <<"You are a helpful assistant.">>},
    #{role => user, content => <<"What is the weather in Paris?">>}
],
{ok, #{message := Msg}} =
    erllama:chat(Model, Messages, #{tools => Tools, temperature => 0.0}).
```

`Msg` is an `erllama:parsed_message()`:

```erlang
#{role => <<"assistant">>,
  content => <<>>,                      %% text the model said, if any
  reasoning_content => undefined,       %% thinking text, if extracted
  tool_calls => [#{name => <<"get_weather">>,
                   arguments => #{<<"city">> => <<"Paris">>},
                   id => undefined}]}
```

`arguments` is already decoded from JSON. `tool_calls =:= []` means
the model answered directly; `content` is your reply and the loop
ends. `id` is `undefined` unless the template mints ids; mint your
own when your client protocol needs them.

## 3. Execute the calls

Run each requested call yourself and produce a result string
(usually JSON). Treat arguments as untrusted input: validate against
your schema, bound what the tool can touch, and return errors as
data so the model can react:

```erlang
run_tool(#{name := <<"get_weather">>, arguments := #{<<"city">> := City}}) ->
    case weather_api:lookup(City) of
        {ok, W} -> iolist_to_binary(json:encode(W));
        {error, unknown_city} -> <<"{\"error\": \"unknown city\"}">>
    end;
run_tool(#{name := Name}) ->
    iolist_to_binary([<<"{\"error\": \"unknown tool ">>, Name, <<"\"}">>]).
```

## 4. Feed the results back and loop

Append two things to the conversation: the assistant turn that made
the calls (in the OpenAI shape, `arguments` re-encoded as a JSON
string) and one `tool` message per result, then call `chat/3` again:

```erlang
tool_loop(Model, Messages, Tools) ->
    Opts = #{tools => Tools, temperature => 0.0, session_id => agent},
    {ok, #{message := Msg}} = erllama:chat(Model, Messages, Opts),
    case maps:get(tool_calls, Msg) of
        [] ->
            {ok, maps:get(content, Msg)};
        Calls ->
            Assistant = #{role => assistant,
                          content => maps:get(content, Msg),
                          tool_calls => [call_json(C) || C <- Calls]},
            Results = [#{role => tool,
                         tool_call_id => call_id(C),
                         content => run_tool(C)} || C <- Calls],
            tool_loop(Model, Messages ++ [Assistant | Results], Tools)
    end.

call_id(#{id := undefined, name := Name}) -> Name;
call_id(#{id := Id}) -> Id.

call_json(#{name := Name, arguments := Args} = C) ->
    #{id => call_id(C),
      type => function,
      function => #{name => Name,
                    arguments => iolist_to_binary(json:encode(Args))}}.
```

Two things make this loop cheap and robust:

- The `session_id` pins the conversation's KV cells, so every round
  only prefills the new suffix (the tool results), not the whole
  transcript. See [sessions](sessions.md).
- Always cap the loop (a max-rounds counter): a confused model can
  request calls forever.

`examples/agent_loop` in the repository is this loop as a complete
runnable application.

## Message shapes

The conversation list uses the OpenAI shapes throughout:

| Role | Keys | Meaning |
|---|---|---|
| `system` | `content` | instructions, once at the top |
| `user` | `content` | the human turn |
| `assistant` | `content`, optional `tool_calls` | a model turn; `tool_calls` entries are `#{id, type => function, function => #{name, arguments}}` with `arguments` as a JSON **string** |
| `tool` | `content`, `tool_call_id`, optional `name` | one result per call, matched by id |

## Enforcement: `tool_choice`

With tools present, the grammar llama.cpp synthesizes from the chat
template is enforced during sampling, not just suggested by the
prompt:

- `tool_choice => auto` (default): the model may answer in text or
  call tools. A lazy grammar arms itself only when the model opens a
  call, so free text stays unconstrained, but an opened call is
  always syntactically valid.
- `tool_choice => required`: the whole reply is constrained to a
  call; `tool_calls` is never empty.
- `tool_choice => none`: tools stay in the prompt for context, but
  the model cannot call them.

`parallel_tool_calls => true` lets one turn request several calls;
run them all and append one `tool` message each.

Passing your own `grammar` together with active tools is rejected
(`{error, {invalid_option, grammar, conflicts_with_tools}}`); a
caller grammar otherwise replaces the template's.

## Forcing a schema instead of tools

When you want structured output rather than actions, constrain the
reply itself:

```erlang
Schema = #{type => object,
           properties => #{answer => #{type => string}},
           required => [answer]},
{ok, #{message := #{content := Json}}} =
    erllama:chat(Model, Messages, #{json_schema => Schema}),
#{<<"answer">> := Answer} = json:decode(Json).
```

`json_schema` (OpenAI `response_format` semantics) grammar-constrains
the content, so `json:decode/1` always succeeds. It cannot be
combined with `tools`
(`{error, {invalid_option, json_schema, conflicts_with_tools}}`).

## Streaming a tool turn

For token-by-token delivery use the three-step form: render, stream,
parse. Merge the `sampler_opts` and `stop_sequences` that
`chat_apply/3` returns into the stream options; that is the
template's constraint set (`chat/3` does the same merge internally):

```erlang
{ok, #{prompt := Prompt, params := Params,
       sampler_opts := SamplerOpts, stop_sequences := Stops}} =
    erllama:chat_apply(Model, Messages, #{tools => Tools}),
{ok, Tokens} = erllama:tokenize(Model, Prompt,
                                #{add_special => false, parse_special => true}),
StreamOpts = maps:merge(SamplerOpts,
                        #{temperature => 0.0, stop_sequences => Stops}),
{ok, Ref} = erllama:stream(Model, Tokens, StreamOpts),
{ok, #{reply := Reply}} = erllama:collect(Ref, 60000),
{ok, Msg} = erllama:chat_parse(Params, Reply, false).
```

While the stream runs, `chat_parse(Params, PartialReply, true)`
parses a prefix and returns what is complete so far: content deltas
to show the user, and a tool call whose name is known while its
arguments are still generating. `Params` is valid for this request
only; call `chat_apply/3` again for the next turn.

## Thinking and prefill

- `enable_thinking => false` suppresses the thinking preamble on
  templates that support it; `chat_apply/3` reports
  `supports_thinking` plus the template's thinking tags.
- `reasoning_format` (default `deepseek`) extracts thinking text into
  `reasoning_content`; `reasoning_format => none` leaves it inline in
  `content`.
- `continue_final_message => content` (or `auto` / `reasoning`) turns
  a trailing assistant message into a prefill: the model continues it
  instead of starting a new turn. The prefill text ends up at the
  tail of the rendered prompt and is included in the parsed message.

## Troubleshooting

- **`{error, {chat_parse_failed, Reason}}`**: the model produced
  output the parser rejects. With enforcement on this is rare; when
  it happens with a `Qwen/Qwen2.5-*-Instruct-GGUF` file, the shipped
  template has doubled braces in its tool example. Load the corrected
  template with the `chat_template` option (see
  `examples/agent_loop/priv/qwen2.5-instruct.jinja`).
- **The model never calls tools** under `auto`: strengthen the tool
  descriptions, or use `required` for the turn where a call is
  mandatory.
- **`{error, chat_not_supported}`**: the stub backend has no chat
  template; test tool loops against a small real model
  (see [testing](testing.md)).
- **`{error, no_template}`**: the GGUF ships no chat template;
  `render_chat_template/2` is the legacy renderer for that case (it
  returns tokens and does no parsing).

## See also

- [Sessions](sessions.md) - pinning the loop's KV across rounds
- [Generating text](generation.md) - streaming events, options
- [Examples](examples.md) - more recipes, including streaming parses
