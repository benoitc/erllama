%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(barrel_inference_chat).
-moduledoc """
Thin Erlang facade over llama.cpp's `common_chat_*` autoparser NIF entries.

Three call-once-per-thing functions wrapping the NIFs one-to-one with
no caching. Callers that want per-(model x tools) caching go through
`barrel_inference_chat_cache' instead.

This is the DORMANT layer shipped in Phase 3.B; the server handlers
do not consume it yet. Phase 3.C wires it into the chat / messages /
responses paths.
""".

-export([init/2, apply/2, parse/3]).

-export_type([templates_ref/0, params_ref/0, parsed_msg/0]).

-type templates_ref() :: barrel_inference_nif:chat_templates_ref().
-type params_ref() :: barrel_inference_nif:chat_params_ref().

-type parsed_msg() :: #{
    role := binary(),
    content := binary(),
    reasoning_content := binary() | undefined,
    tool_calls := [
        #{
            name := binary(),
            arguments := map(),
            id := binary() | undefined
        }
    ]
}.

-spec init(barrel_inference_nif:model_ref(), binary() | undefined) ->
    {ok, templates_ref()} | {error, term()}.
init(Model, TemplateOverride) ->
    barrel_inference_nif:chat_templates_init(Model, TemplateOverride).

-spec apply(templates_ref(), map()) ->
    {ok, params_ref(), binary()} | {error, term()}.
apply(Templates, Inputs) when is_map(Inputs) ->
    barrel_inference_nif:chat_templates_apply(Templates, Inputs).

-spec parse(params_ref(), binary(), boolean()) ->
    {ok, parsed_msg()} | {error, term()}.
parse(Params, Input, IsPartial) when
    is_binary(Input), is_boolean(IsPartial)
->
    case barrel_inference_nif:chat_parse(Params, Input, IsPartial) of
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
