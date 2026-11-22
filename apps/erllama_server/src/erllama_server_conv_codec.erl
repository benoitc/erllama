%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_conv_codec).
-moduledoc false.
%% Conversation messages to JSON and back, for the conversation logs.
%%
%% Messages are maps with atom keys (`role', `content', `tool_calls',
%% `tool_call_id', `name', and inside them `id', `type', `function',
%% `arguments', `text', `data'). Decoding turns exactly those keys back
%% into atoms and leaves any other key a binary: no atom is made from
%% what a file holds. Image and audio parts carry raw bytes, stored as
%% base64. A finish key (a binary cache key) is stored as hex.

-export([encode_msgs/1, decode_msgs/1, encode_key/1, decode_key/1]).

-define(KEYS, [
    role, content, tool_calls, tool_call_id, name, id, type, function, arguments, text, data
]).

-spec encode_msgs([map()]) -> [map()].
encode_msgs(Msgs) -> [encode(M) || M <- Msgs].

-spec decode_msgs([map()]) -> [map()].
decode_msgs(Msgs) -> [decode(M) || M <- Msgs, is_map(M)].

-spec encode_key(binary() | undefined) -> binary() | null.
encode_key(undefined) -> null;
encode_key(K) when is_binary(K) -> binary:encode_hex(K, lowercase).

-spec decode_key(term()) -> binary() | undefined.
decode_key(Hex) when is_binary(Hex) ->
    try
        binary:decode_hex(Hex)
    catch
        error:_ -> undefined
    end;
decode_key(_) ->
    undefined.

%% -----------------------------------------------------------------------------

encode(#{type := T, data := D} = Part) when
    T =:= image, is_binary(D); T =:= audio, is_binary(D)
->
    Rest = encode_map(maps:without([type, data], Part)),
    Rest#{<<"type">> => atom_to_binary(T), <<"data_base64">> => base64:encode(D)};
encode(M) when is_map(M) ->
    encode_map(M);
encode(L) when is_list(L) ->
    [encode(X) || X <- L];
encode(V) ->
    V.

encode_map(M) ->
    maps:fold(fun(K, V, Acc) -> Acc#{key_out(K) => encode(V)} end, #{}, M).

key_out(K) when is_atom(K) -> atom_to_binary(K);
key_out(K) -> K.

decode(#{<<"data_base64">> := B64, <<"type">> := T} = Part) when
    T =:= <<"image">>; T =:= <<"audio">>
->
    Rest = decode_map(maps:without([<<"data_base64">>, <<"type">>], Part)),
    Rest#{type => binary_to_existing_atom(T), data => base64:decode(B64)};
decode(M) when is_map(M) ->
    decode_map(M);
decode(L) when is_list(L) ->
    [decode(X) || X <- L];
decode(V) ->
    V.

decode_map(M) ->
    maps:fold(fun(K, V, Acc) -> Acc#{key_in(K) => decode(V)} end, #{}, M).

key_in(K) when is_binary(K) ->
    case [A || A <- ?KEYS, atom_to_binary(A) =:= K] of
        [A] -> A;
        [] -> K
    end;
key_in(K) ->
    K.
