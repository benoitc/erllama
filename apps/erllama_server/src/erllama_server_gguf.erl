%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_gguf).
-moduledoc false.
%% Reads the metadata of a GGUF file without loading the model, so the
%% registry can describe and check a model (architecture, context
%% length, quantisation, chat template, projector or not) before any
%% memory is committed to it. The header is streamed: large arrays (the
%% tokenizer tables run to megabytes) are skipped, never held.
%%
%% Format (gguf.h, version 2 and 3, little endian): magic "GGUF",
%% u32 version, u64 tensor count, u64 key/value count, then each pair
%% as a u64-length string key, a u32 value type and the value.

-export([read/1, summary/1]).

-define(CHUNK, 65536).
%% Arrays longer than this are reported by type and length only.
-define(KEEP_ARRAY, 64).

-type value() ::
    integer() | float() | boolean() | binary() | {array, atom(), non_neg_integer(), list()}.
-export_type([value/0]).

%% All key/value pairs of the header, as a map of binary keys.
-spec read(file:filename_all()) ->
    {ok, #{version := pos_integer(), tensors := non_neg_integer(), kv := #{binary() => value()}}}
    | {error, term()}.
read(Path) ->
    case file:open(Path, [read, binary, raw]) of
        {ok, Fd} ->
            try
                parse({Fd, <<>>})
            catch
                throw:Reason -> {error, Reason}
            after
                file:close(Fd)
            end;
        {error, Reason} ->
            {error, Reason}
    end.

parse(R0) ->
    {Magic, R1} = take(4, R0),
    Magic =:= <<"GGUF">> orelse throw(not_gguf),
    {<<Version:32/little>>, R2} = take(4, R1),
    lists:member(Version, [2, 3]) orelse throw({unsupported_gguf_version, Version}),
    {<<Tensors:64/little, Count:64/little>>, R3} = take(16, R2),
    {KV, _} = pairs(Count, R3, #{}),
    {ok, #{version => Version, tensors => Tensors, kv => KV}}.

pairs(0, R, Acc) ->
    {Acc, R};
pairs(N, R0, Acc) ->
    {Key, R1} = string(R0),
    {<<Type:32/little>>, R2} = take(4, R1),
    {Value, R3} = value(Type, R2),
    pairs(N - 1, R3, Acc#{Key => Value}).

value(0, R) ->
    int(1, unsigned, R);
value(1, R) ->
    int(1, signed, R);
value(2, R) ->
    int(2, unsigned, R);
value(3, R) ->
    int(2, signed, R);
value(4, R) ->
    int(4, unsigned, R);
value(5, R) ->
    int(4, signed, R);
value(6, R) ->
    float(4, R);
value(7, R) ->
    {<<B:8>>, R1} = take(1, R),
    {B =/= 0, R1};
value(8, R) ->
    string(R);
value(9, R0) ->
    {<<ElemType:32/little, Len:64/little>>, R1} = take(12, R0),
    array(ElemType, Len, R1);
value(10, R) ->
    int(8, unsigned, R);
value(11, R) ->
    int(8, signed, R);
value(12, R) ->
    float(8, R);
value(Type, _R) ->
    throw({bad_value_type, Type}).

int(Bytes, Sign, R0) ->
    Bits = Bytes * 8,
    {Bin, R1} = take(Bytes, R0),
    V =
        case Sign of
            unsigned ->
                <<U:Bits/little-unsigned>> = Bin,
                U;
            signed ->
                <<S:Bits/little-signed>> = Bin,
                S
        end,
    {V, R1}.

float(Bytes, R0) ->
    Bits = Bytes * 8,
    {Bin, R1} = take(Bytes, R0),
    case Bin of
        <<F:Bits/little-float>> -> {F, R1};
        _ -> {nan, R1}
    end.

string(R0) ->
    {<<Len:64/little>>, R1} = take(8, R0),
    take(Len, R1).

%% Small arrays are kept; long ones (tokens, merges, scores) are walked
%% over: fixed-size elements by seeking, strings one length at a time.
array(Type, Len, R) when Len =< ?KEEP_ARRAY ->
    {Vals, R1} = lists:foldl(
        fun(_, {Acc, Rx}) ->
            {V, Ry} = value(Type, Rx),
            {[V | Acc], Ry}
        end,
        {[], R},
        lists:seq(1, Len)
    ),
    {{array, type_name(Type), Len, lists:reverse(Vals)}, R1};
array(8, Len, R) ->
    {{array, string, Len, []}, skip_strings(Len, R)};
array(Type, Len, R) ->
    {{array, type_name(Type), Len, []}, skip(Len * fixed_size(Type), R)}.

skip_strings(0, R) ->
    R;
skip_strings(N, R0) ->
    {<<Len:64/little>>, R1} = take(8, R0),
    skip_strings(N - 1, skip(Len, R1)).

fixed_size(T) when T =:= 0; T =:= 1; T =:= 7 -> 1;
fixed_size(T) when T =:= 2; T =:= 3 -> 2;
fixed_size(T) when T =:= 4; T =:= 5; T =:= 6 -> 4;
fixed_size(T) when T =:= 10; T =:= 11; T =:= 12 -> 8;
fixed_size(T) -> throw({unsupported_array_type, T}).

type_name(0) -> u8;
type_name(1) -> i8;
type_name(2) -> u16;
type_name(3) -> i16;
type_name(4) -> u32;
type_name(5) -> i32;
type_name(6) -> f32;
type_name(7) -> bool;
type_name(8) -> string;
type_name(9) -> array;
type_name(10) -> u64;
type_name(11) -> i64;
type_name(12) -> f64;
type_name(T) -> throw({bad_value_type, T}).

%% Buffered reader: `{Fd, Buffered}'.
take(N, {Fd, Buf}) when byte_size(Buf) >= N ->
    <<Bin:N/binary, Rest/binary>> = Buf,
    {Bin, {Fd, Rest}};
take(N, {Fd, Buf}) ->
    case file:read(Fd, max(?CHUNK, N - byte_size(Buf))) of
        {ok, More} -> take(N, {Fd, <<Buf/binary, More/binary>>});
        eof -> throw(truncated);
        {error, Reason} -> throw(Reason)
    end.

skip(N, {Fd, Buf}) when byte_size(Buf) >= N ->
    <<_:N/binary, Rest/binary>> = Buf,
    {Fd, Rest};
skip(N, {Fd, Buf}) ->
    {ok, _} = file:position(Fd, {cur, N - byte_size(Buf)}),
    {Fd, <<>>}.

%% What the registry and `show' report about a model file.
-spec summary(#{kv := map(), _ => _}) -> map().
summary(#{kv := KV} = Header) ->
    Arch = maps:get(<<"general.architecture">>, KV, undefined),
    ArchKey = fun(Suffix) -> maps:get(<<Arch/binary, ".", Suffix/binary>>, KV, undefined) end,
    Base = #{
        architecture => Arch,
        name => maps:get(<<"general.name">>, KV, undefined),
        size_label => maps:get(<<"general.size_label">>, KV, undefined),
        file_type => file_type(maps:get(<<"general.file_type">>, KV, undefined)),
        tensors => maps:get(tensors, Header, 0),
        chat_template => maps:get(<<"tokenizer.chat_template">>, KV, undefined),
        projector => is_projector(KV)
    },
    case is_binary(Arch) of
        true ->
            Base#{
                context_length => ArchKey(<<"context_length">>),
                embedding_length => ArchKey(<<"embedding_length">>),
                block_count => ArchKey(<<"block_count">>)
            };
        false ->
            Base
    end.

%% A multimodal projector (mmproj) and the modalities it encodes.
is_projector(KV) ->
    case maps:get(<<"general.architecture">>, KV, undefined) of
        <<"clip">> ->
            Vision = flag(maps:get(<<"clip.has_vision_encoder">>, KV, false)),
            Audio = flag(maps:get(<<"clip.has_audio_encoder">>, KV, false)),
            #{vision => Vision, audio => Audio};
        _ ->
            false
    end.

flag(true) -> true;
flag(_) -> false.

%% llama.h `llama_ftype'.
file_type(undefined) -> undefined;
file_type(0) -> <<"F32">>;
file_type(1) -> <<"F16">>;
file_type(2) -> <<"Q4_0">>;
file_type(3) -> <<"Q4_1">>;
file_type(7) -> <<"Q8_0">>;
file_type(8) -> <<"Q5_0">>;
file_type(9) -> <<"Q5_1">>;
file_type(10) -> <<"Q2_K">>;
file_type(11) -> <<"Q3_K_S">>;
file_type(12) -> <<"Q3_K_M">>;
file_type(13) -> <<"Q3_K_L">>;
file_type(14) -> <<"Q4_K_S">>;
file_type(15) -> <<"Q4_K_M">>;
file_type(16) -> <<"Q5_K_S">>;
file_type(17) -> <<"Q5_K_M">>;
file_type(18) -> <<"Q6_K">>;
file_type(19) -> <<"IQ2_XXS">>;
file_type(20) -> <<"IQ2_XS">>;
file_type(21) -> <<"Q2_K_S">>;
file_type(22) -> <<"IQ3_XS">>;
file_type(23) -> <<"IQ3_XXS">>;
file_type(24) -> <<"IQ1_S">>;
file_type(25) -> <<"IQ4_NL">>;
file_type(26) -> <<"IQ3_S">>;
file_type(27) -> <<"IQ3_M">>;
file_type(28) -> <<"IQ2_S">>;
file_type(29) -> <<"IQ2_M">>;
file_type(30) -> <<"IQ4_XS">>;
file_type(31) -> <<"IQ1_M">>;
file_type(32) -> <<"BF16">>;
file_type(36) -> <<"TQ1_0">>;
file_type(37) -> <<"TQ2_0">>;
file_type(38) -> <<"MXFP4_MOE">>;
file_type(39) -> <<"NVFP4">>;
file_type(40) -> <<"Q1_0">>;
file_type(41) -> <<"Q2_0">>;
file_type(42) -> <<"F8_E4M3">>;
file_type(N) -> integer_to_binary(N).
