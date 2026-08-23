%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Real-model CT for the chat-autoparser NIF wrapper. Gated on
%% `LLAMA_TEST_MODEL'; mirrors `erllama_real_model_SUITE'.
-module(erllama_chat_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([
    templates_init_returns_ref/1,
    apply_then_parse_round_trip/1,
    parse_partial_then_full/1,
    facade_chat_apply_then_parse/1
]).

all() ->
    [
        templates_init_returns_ref,
        apply_then_parse_round_trip,
        parse_partial_then_full,
        facade_chat_apply_then_parse
    ].

init_per_suite(Config) ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {skip, "set LLAMA_TEST_MODEL to a GGUF path to enable this suite"};
        Path ->
            ok = application:ensure_started(erllama),
            {ok, Model} = erllama_nif:load_model(Path, #{}),
            [{model, Model} | Config]
    end.

end_per_suite(Config) ->
    case ?config(model, Config) of
        undefined -> ok;
        Model -> _ = erllama_nif:free_model(Model)
    end,
    ok.

templates_init_returns_ref(Config) ->
    Model = ?config(model, Config),
    {ok, Ref} = erllama_chat:init(Model, undefined),
    ?assert(is_reference(Ref)).

apply_then_parse_round_trip(Config) ->
    Model = ?config(model, Config),
    {ok, Templates} = erllama_chat:init(Model, undefined),
    Inputs = #{
        messages => iolist_to_binary(
            json:encode([
                #{<<"role">> => <<"user">>, <<"content">> => <<"hi">>}
            ])
        ),
        tools => <<"[]">>
    },
    {ok, Params, Prompt} = erllama_chat:apply(Templates, Inputs),
    ?assert(is_reference(Params)),
    ?assert(is_binary(Prompt) andalso byte_size(Prompt) > 0),
    {ok, Msg} = erllama_chat:parse(Params, <<"hello there">>, false),
    #{role := Role, content := Content} = Msg,
    ?assertEqual(<<"assistant">>, Role),
    ?assert(is_binary(Content)).

parse_partial_then_full(Config) ->
    Model = ?config(model, Config),
    {ok, Templates} = erllama_chat:init(Model, undefined),
    Inputs = #{
        messages => iolist_to_binary(
            json:encode([
                #{<<"role">> => <<"user">>, <<"content">> => <<"hi">>}
            ])
        ),
        tools => <<"[]">>
    },
    {ok, Params, _Prompt} = erllama_chat:apply(Templates, Inputs),
    {ok, Partial} = erllama_chat:parse(Params, <<"hel">>, true),
    {ok, Full} = erllama_chat:parse(Params, <<"hello">>, false),
    ?assert(is_map(Partial)),
    ?assert(is_map(Full)),
    #{content := PartialContent} = Partial,
    #{content := FullContent} = Full,
    ?assert(byte_size(FullContent) >= byte_size(PartialContent)).

%% Public façade path: erllama:chat_apply/2 runs one upstream
%% common_chat_templates_apply per request (prompt + parser), and the
%% returned params ref drives erllama:chat_parse/3 for that request.
%% Two calls with the same inputs yield distinct params refs; only the
%% per-model templates ref is cached.
facade_chat_apply_then_parse(_Config) ->
    Path = os:getenv("LLAMA_TEST_MODEL"),
    {ok, ModelId} = erllama:load_model(#{
        backend => erllama_model_llama,
        model_path => Path
    }),
    Inputs = #{
        messages => iolist_to_binary(
            json:encode([
                #{<<"role">> => <<"user">>, <<"content">> => <<"hi">>}
            ])
        ),
        tools => <<"[]">>
    },
    try
        {ok, Params1, Prompt} = erllama:chat_apply(ModelId, Inputs),
        ?assert(is_reference(Params1)),
        ?assert(is_binary(Prompt) andalso byte_size(Prompt) > 0),
        {ok, Params2, Prompt} = erllama:chat_apply(ModelId, Inputs),
        ?assertNotEqual(Params1, Params2),
        {ok, Msg} = erllama:chat_parse(Params2, <<"hello there">>, false),
        ?assertEqual(<<"assistant">>, maps:get(role, Msg)),
        ?assertEqual(<<"hello there">>, maps:get(content, Msg))
    after
        ok = erllama:unload(ModelId)
    end.
