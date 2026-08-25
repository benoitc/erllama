%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_chat).
-moduledoc false.
%% Thin Erlang facade over llama.cpp's `common_chat_*` autoparser NIF entries.
%%
%% `init/2` builds the templates for a model, `apply/2` renders the prompt
%% and synthesises the parser for one request (upstream's
%% `common_chat_templates_apply`), `parse/3` turns model output back into a
%% structured message with that request's parser. No caching here; the
%% per-model templates ref is cached by `erllama_chat_cache`, and
%% `erllama_model:chat_apply/2` is the entry point used by the façade.

-export([init/2, apply/2, parse/3]).
-export([chat/3, inputs/2, chat_keys/0, merge_params/2]).

-export_type([templates_ref/0, params_ref/0, parsed_msg/0, message/0, tool/0]).

-type templates_ref() :: erllama_nif:chat_templates_ref().
-type params_ref() :: erllama_nif:chat_params_ref().

-type parsed_msg() :: erllama:parsed_message().

-type message() :: erllama:chat_message().

-type tool() :: erllama:chat_tool().

-define(CHAT_KEYS, [
    tools,
    tool_choice,
    parallel_tool_calls,
    json_schema,
    enable_thinking,
    reasoning_format,
    continue_final_message
]).

%% The option keys `chat/3` consumes itself (everything else is a
%% request opt). The facade strips these before request validation.
-spec chat_keys() -> [atom()].
chat_keys() -> ?CHAT_KEYS.

-doc """
Build the NIF inputs map for `apply/2` from Erlang terms: messages and
tools are JSON-encoded, `tool_choice` (`auto | required | none`),
`parallel_tool_calls`, `enable_thinking`, `reasoning_format`
(`deepseek | none`, default `deepseek`) and `continue_final_message`
(`none | auto | content | reasoning`) are passed through, and
`json_schema` (map or JSON binary) is JSON-encoded.
""".
-spec inputs([message()], map()) -> map().
inputs(Messages, Opts) when is_list(Messages), is_map(Opts) ->
    Tools = [oaicompat_tool(T) || T <- maps:get(tools, Opts, [])],
    #{
        messages => iolist_to_binary(json:encode(Messages)),
        tools => iolist_to_binary(json:encode(Tools)),
        tool_choice => maps:get(tool_choice, Opts, auto),
        parallel_tool_calls => maps:get(parallel_tool_calls, Opts, false),
        json_schema => json_schema_input(maps:get(json_schema, Opts, undefined)),
        enable_thinking => maps:get(enable_thinking, Opts, true),
        reasoning_format => maps:get(reasoning_format, Opts, deepseek),
        continue_final_message => maps:get(continue_final_message, Opts, none)
    }.

json_schema_input(undefined) -> <<>>;
json_schema_input(Schema) when is_binary(Schema) -> Schema;
json_schema_input(Schema) when is_map(Schema) -> iolist_to_binary(json:encode(Schema)).

%% llama.cpp parses tools in the OpenAI shape
%% `{"type": "function", "function": {name, description, parameters}}';
%% `erllama:chat_tool()' is the flat inner map. Already wrapped tools
%% (e.g. forwarded from an OpenAI-style client) pass through.
oaicompat_tool(#{type := _, function := _} = Tool) ->
    Tool;
oaicompat_tool(#{<<"type">> := _, <<"function">> := _} = Tool) ->
    Tool;
oaicompat_tool(Tool) when is_map(Tool) ->
    Params = maps:get(parameters, Tool, #{type => object, properties => #{}}),
    Function0 = #{name => maps:get(name, Tool), parameters => Params},
    Function =
        case maps:find(description, Tool) of
            {ok, D} -> Function0#{description => D};
            error -> Function0
        end,
    #{type => function, function => Function}.

-doc """
One synchronous chat turn: render the prompt and parser for
`Messages`, tokenise, stream the completion to this process, collect
it, and parse the output into a structured message. `Opts` carries
the chat keys (`chat_keys/0`) plus any `erllama:request_opts()` key.
The constraint set the template pass synthesizes (tool-call grammar
with lazy triggers, additional stop strings) is merged into the
request opts via `merge_params/2`, so `tool_choice => required` and
`json_schema` are enforced during sampling, not just suggested by the
prompt.
""".
-spec chat(erllama:model(), [message()], map()) ->
    {ok, #{
        message := parsed_msg(), prompt := binary(), reply := binary(), stats := erllama:stats()
    }}
    | {error, term()}.
chat(Model, Messages, Opts) when is_list(Messages), is_map(Opts) ->
    RequestOpts = maps:without(?CHAT_KEYS, Opts),
    case check_conflicts(Opts) of
        ok ->
            case erllama_model:chat_apply(Model, inputs(Messages, Opts)) of
                {ok, Params, Render} ->
                    chat_generate(Model, Params, Render, merge_params(Render, RequestOpts));
                {error, _} = E ->
                    E
            end;
        {error, _} = E ->
            E
    end.

%% Mirror upstream's hard error (chat.cpp: "Cannot specify grammar
%% with tools") and reject json_schema + tools instead of upstream's
%% silent tool dropping.
check_conflicts(Opts) ->
    HasTools = maps:get(tools, Opts, []) =/= [],
    ToolsActive = HasTools andalso maps:get(tool_choice, Opts, auto) =/= none,
    HasSchema = maps:get(json_schema, Opts, undefined) =/= undefined,
    case {ToolsActive andalso is_map_key(grammar, Opts), ToolsActive andalso HasSchema} of
        {true, _} -> {error, {invalid_option, grammar, conflicts_with_tools}};
        {_, true} -> {error, {invalid_option, json_schema, conflicts_with_tools}};
        _ -> ok
    end.

-doc """
Merge the constraint set from an `apply/2` render map into request
opts: the template-synthesized grammar (with its lazy trigger
patterns/tokens, or the generation-prompt prefill for a non-lazy
grammar) and the template's additional stop strings. Caller-supplied
keys always win; a caller `grammar` suppresses the template grammar
entirely.
""".
-spec merge_params(map(), map()) -> map().
merge_params(Render, RequestOpts) when is_map(Render), is_map(RequestOpts) ->
    merge_stops(Render, merge_grammar(Render, RequestOpts)).

merge_grammar(#{grammar := G} = Render, Opts) when is_binary(G), G =/= <<>> ->
    case is_map_key(grammar, Opts) of
        true ->
            Opts;
        false ->
            case maps:get(grammar_lazy, Render, false) of
                true ->
                    Opts#{
                        grammar => G,
                        grammar_lazy => true,
                        trigger_patterns => maps:get(trigger_patterns, Render, []),
                        trigger_tokens => maps:get(trigger_tokens, Render, [])
                    };
                false ->
                    with_prefill(Opts#{grammar => G}, Render)
            end
    end;
merge_grammar(_Render, Opts) ->
    Opts.

%% A non-lazy template grammar covers the assistant header already in
%% the prompt; the sampler must accept those tokens first.
with_prefill(Opts, Render) ->
    case maps:get(generation_prompt, Render, <<>>) of
        <<>> -> Opts;
        GP -> Opts#{grammar_prefill => GP}
    end.

merge_stops(#{additional_stops := [_ | _] = Stops}, Opts) ->
    Existing = maps:get(stop_sequences, Opts, []),
    New = [S || S <- Stops, is_binary(S), not lists:member(S, Existing)],
    Opts#{stop_sequences => Existing ++ New};
merge_stops(_Render, Opts) ->
    Opts.

chat_generate(Model, Params, Render, RequestOpts) ->
    Prompt = maps:get(prompt, Render),
    TokOpts = #{add_special => false, parse_special => true},
    case erllama_model:tokenize(Model, Prompt, TokOpts) of
        {ok, Tokens} ->
            %% Straight to the model process: the façade's `chat' op is the
            %% one the middleware chain sees, not a nested `stream'.
            case erllama_model:infer(Model, Tokens, maps:remove(to, RequestOpts), self()) of
                {ok, Ref} ->
                    chat_collect(Ref, Params, Prompt);
                {error, _} = E ->
                    E
            end;
        {error, _} = E ->
            E
    end.

chat_collect(Ref, Params, Prompt) ->
    case erllama_stream:collect(Ref, infinity) of
        {ok, #{reply := Reply, stats := Stats}} ->
            case parse(Params, Reply, false) of
                {ok, Msg} ->
                    {ok, #{message => Msg, prompt => Prompt, reply => Reply, stats => Stats}};
                {error, _} = E ->
                    E
            end;
        {error, _} = E ->
            E
    end.

-spec init(erllama_nif:model_ref(), binary() | undefined) ->
    {ok, templates_ref()} | {error, term()}.
init(Model, TemplateOverride) ->
    erllama_nif:chat_templates_init(Model, TemplateOverride).

%% Returns the params ref plus the render map: prompt, and the
%% constraint set the template pass synthesized (grammar,
%% grammar_lazy, trigger_patterns, trigger_tokens, additional_stops,
%% generation_prompt, supports_thinking, thinking tags, format).
-spec apply(templates_ref(), map()) ->
    {ok, params_ref(), map()} | {error, term()}.
apply(Templates, Inputs) when is_map(Inputs) ->
    erllama_nif:chat_templates_apply(Templates, Inputs).

-spec parse(params_ref(), binary(), boolean()) ->
    {ok, parsed_msg()} | {error, term()}.
parse(Params, Input, IsPartial) when
    is_binary(Input), is_boolean(IsPartial)
->
    case erllama_nif:chat_parse(Params, Input, IsPartial) of
        {ok, Msg} -> {ok, decode_tool_calls(Msg)};
        Err -> Err
    end.

%% The NIF returns each tool call's arguments as a raw JSON binary
%% (`arguments_json' key) so the C++ side carries no JSON-decode
%% logic. Decode at the Erlang boundary into the documented map shape.
decode_tool_calls(Msg = #{tool_calls := Calls}) ->
    Decoded = [decode_call(C) || C <- Calls],
    Msg#{tool_calls => Decoded};
decode_tool_calls(Msg) ->
    Msg.

decode_call(#{name := Name, arguments_json := Json, id := Id}) ->
    Args =
        case Json of
            <<>> ->
                #{};
            _ ->
                try json:decode(Json) of
                    M when is_map(M) -> M;
                    _ -> #{}
                catch
                    _:_ -> #{}
                end
        end,
    #{name => Name, arguments => Args, id => Id}.
