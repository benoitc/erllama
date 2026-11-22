%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_conv_export).
-moduledoc false.
%% A conversation as one document: JSON (its fields and messages, media
%% as base64, the same encoding as its log) or Markdown (for reading).

-export([export/2]).

-spec export(json | markdown, map()) -> binary().
export(json, #{messages := Msgs} = C) ->
    Meta = maps:with([id, model, kind, parent, workspace, unresolved, created_at, updated_at], C),
    Doc = maps:map(fun(_, V) -> json_value(V) end, Meta),
    iolist_to_binary(
        json:encode(Doc#{messages => erllama_server_conv_codec:encode_msgs(Msgs)})
    );
export(markdown, #{id := Id, messages := Msgs} = C) ->
    Head = io_lib:format("# Conversation ~ts~n~nModel: ~ts~n", [Id, text(maps:get(model, C))]),
    Unresolved =
        case maps:get(unresolved, C, []) of
            [] -> [];
            U -> io_lib:format("Unresolved tool calls: ~ts~n", [lists:join(", ", U)])
        end,
    iolist_to_binary([Head, Unresolved | [md_message(M) || M <- Msgs]]).

md_message(#{role := R} = M) ->
    Calls = [
        io_lib:format("~n- call `~ts` ~ts~n", [N, A])
     || #{function := #{name := N, arguments := A}} <- maps:get(tool_calls, M, [])
    ],
    Who =
        case M of
            #{name := N} when R =:= <<"tool">> -> [R, " (", N, ")"];
            _ -> R
        end,
    [io_lib:format("~n## ~ts~n~n", [Who]), md_content(maps:get(content, M, <<>>)), "\n" | Calls].

md_content(C) when is_binary(C) -> C;
md_content(Parts) when is_list(Parts) ->
    [
        case P of
            #{text := T} -> T;
            #{type := T} -> io_lib:format("[~0p]", [T])
        end
     || P <- Parts
    ];
md_content(_) ->
    <<>>.

text(V) when is_binary(V) -> V;
text(V) -> io_lib:format("~0p", [V]).

json_value(undefined) -> null;
json_value(V) when is_binary(V); is_integer(V); is_list(V); is_atom(V) -> V;
json_value(V) -> text(V).
