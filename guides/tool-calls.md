# Tool calls

`erllama:chat/3` runs one chat turn: it renders the messages and the
tool definitions through the model's own chat template, generates, and
parses the output back into a structured assistant message with
`content`, `reasoning_content` and `tool_calls`. The parser is
llama.cpp's autoparser, so every template family llama.cpp knows
(Qwen, Llama 3, Mistral, Hermes, GPT-OSS, ...) works without per-model
configuration. You need this page when your application lets the
model call functions.

## One turn

```erlang
Tools = [
    #{name => <<"get_weather">>,
      description => <<"Current weather for a city">>,
      parameters => #{type => object,
                      properties => #{city => #{type => string}},
                      required => [city]}}
],
Messages = [
    #{role => system, content => <<"You are a helpful assistant.">>},
    #{role => user, content => <<"What is the weather in Paris?">>}
],
{ok, #{message := Msg, stats := Stats}} =
    erllama:chat(Model, Messages, #{tools => Tools, temperature => 0.0}).
```

`Msg` is an `erllama:parsed_message()`:

```erlang
#{role => <<"assistant">>,
  content => <<>>,
  reasoning_content => undefined,
  tool_calls => [#{name => <<"get_weather">>,
                   arguments => #{<<"city">> => <<"Paris">>},
                   id => undefined}]}
```

`arguments` is already decoded from JSON. `id` is `undefined` unless
the template carries ids; mint your own when the client protocol needs
them.

Options: `tools`, `tool_choice` (`auto` default, `required`, `none`),
`parallel_tool_calls` (boolean), plus every `erllama:request_opts()`
key (`response_tokens`, `temperature`, `session_id`, `stop_sequences`,
...).

## The tool loop

Run the tool, append the result as a `tool` message, and call
`chat/3` again with the whole conversation. The KV cache makes the
second call cheap: the rendered prompt shares its prefix with the
first one.

```erlang
tool_loop(Model, Messages, Tools) ->
    {ok, #{message := Msg}} = erllama:chat(Model, Messages, #{tools => Tools}),
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
      function => #{name => Name, arguments => iolist_to_binary(json:encode(Args))}}.

run_tool(#{name := <<"get_weather">>, arguments := #{<<"city">> := City}}) ->
    iolist_to_binary(json:encode(#{city => City, temperature_c => 21})).
```

Messages follow the OpenAI shapes: an assistant message that made
calls carries `tool_calls` with `function => #{name, arguments}` where
`arguments` is a JSON string; a `tool` message answers one call by
`tool_call_id`.

## Streaming

For token-by-token delivery use the three-step form: render, stream,
parse.

```erlang
{ok, #{prompt := Prompt, params := Params}} =
    erllama:chat_apply(Model, Messages, #{tools => Tools}),
{ok, Tokens} = erllama:tokenize(Model, Prompt,
                                #{add_special => false, parse_special => true}),
{ok, Ref} = erllama:stream(Model, Tokens, #{temperature => 0.0}),
{ok, #{reply := Reply}} = erllama:collect(Ref, 60000),
{ok, Msg} = erllama:chat_parse(Params, Reply, false).
```

While the stream is running, `chat_parse(Params, PartialReply, true)`
parses a prefix and returns what is complete so far (content deltas,
a tool call whose arguments are still being generated). `Params` is
valid for this request only: call `chat_apply/3` again for the next
turn.

## Forcing a call or a schema

- `tool_choice => required` makes the template and parser expect a
  call.
- A `grammar` (GBNF) in the request options constrains every sampled
  token, tool-call syntax included, for strict output formats.

## Models without a template

`chat_apply/3` returns `{error, no_template}` for a GGUF that ships
no chat template and `{error, chat_not_supported}` for the stub
backend. `render_chat_template/2` is the legacy renderer for the
first case: it returns tokens and does no parsing.
