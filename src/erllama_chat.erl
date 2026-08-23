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
-export([chat/3, inputs/2]).

-export_type([templates_ref/0, params_ref/0, parsed_msg/0, message/0, tool/0]).

-type templates_ref() :: erllama_nif:chat_templates_ref().
-type params_ref() :: erllama_nif:chat_params_ref().

-type parsed_msg() :: erllama:parsed_message().

-type message() :: erllama:chat_message().

-type tool() :: erllama:chat_tool().

-define(CHAT_KEYS, [tools, tool_choice, parallel_tool_calls]).

-doc """
Build the NIF inputs map for `apply/2` from Erlang terms: messages and
tools are JSON-encoded, `tool_choice` (`auto | required | none`) and
`parallel_tool_calls` are passed through.
""".
-spec inputs([message()], map()) -> map().
inputs(Messages, Opts) when is_list(Messages), is_map(Opts) ->
    Tools = maps:get(tools, Opts, []),
    #{
        messages => iolist_to_binary(json:encode(Messages)),
        tools => iolist_to_binary(json:encode(Tools)),
        tool_choice => maps:get(tool_choice, Opts, auto),
        parallel_tool_calls => maps:get(parallel_tool_calls, Opts, false)
    }.

-doc """
One synchronous chat turn: render the prompt and parser for
`Messages`, tokenise, stream the completion to this process, collect
it, and parse the output into a structured message. `Opts` carries
the chat keys (`tools`, `tool_choice`, `parallel_tool_calls`) plus any
`erllama:request_opts()` key.
""".
-spec chat(erllama:model(), [message()], map()) ->
    {ok, #{
        message := parsed_msg(), prompt := binary(), reply := binary(), stats := erllama:stats()
    }}
    | {error, term()}.
chat(Model, Messages, Opts) when is_list(Messages), is_map(Opts) ->
    RequestOpts = maps:without(?CHAT_KEYS, Opts),
    case erllama_model:chat_apply(Model, inputs(Messages, Opts)) of
        {ok, Params, Prompt} ->
            chat_generate(Model, Params, Prompt, RequestOpts);
        {error, _} = E ->
            E
    end.

chat_generate(Model, Params, Prompt, RequestOpts) ->
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

-spec apply(templates_ref(), map()) ->
    {ok, params_ref(), binary()} | {error, term()}.
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
