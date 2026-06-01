%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Real-model CT for the chat-autoparser NIF wrapper. Gated on
%% `LLAMA_TEST_MODEL'; mirrors `barrel_inference_real_model_SUITE'.
-module(barrel_inference_chat_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([
    templates_init_returns_ref/1,
    apply_then_parse_round_trip/1,
    parse_partial_then_full/1
]).

all() ->
    [
        templates_init_returns_ref,
        apply_then_parse_round_trip,
        parse_partial_then_full
    ].

init_per_suite(Config) ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {skip, "set LLAMA_TEST_MODEL to a GGUF path to enable this suite"};
        Path ->
            ok = application:ensure_started(barrel_inference),
            {ok, Model} = barrel_inference_nif:load_model(Path, #{}),
            [{model, Model} | Config]
    end.

end_per_suite(Config) ->
    case ?config(model, Config) of
        undefined -> ok;
        Model -> _ = barrel_inference_nif:free_model(Model)
    end,
    ok.

templates_init_returns_ref(Config) ->
    Model = ?config(model, Config),
    {ok, Ref} = barrel_inference_chat:init(Model, undefined),
    ?assert(is_reference(Ref)).

apply_then_parse_round_trip(Config) ->
    Model = ?config(model, Config),
    {ok, Templates} = barrel_inference_chat:init(Model, undefined),
    Inputs = #{
        messages => iolist_to_binary(
            json:encode([
                #{<<"role">> => <<"user">>, <<"content">> => <<"hi">>}
            ])
        ),
        tools => <<"[]">>
    },
    {ok, Params, Prompt} = barrel_inference_chat:apply(Templates, Inputs),
    ?assert(is_reference(Params)),
    ?assert(is_binary(Prompt) andalso byte_size(Prompt) > 0),
    {ok, Msg} = barrel_inference_chat:parse(Params, <<"hello there">>, false),
    #{role := Role, content := Content} = Msg,
    ?assertEqual(<<"assistant">>, Role),
    ?assert(is_binary(Content)).

parse_partial_then_full(Config) ->
    Model = ?config(model, Config),
    {ok, Templates} = barrel_inference_chat:init(Model, undefined),
    Inputs = #{
        messages => iolist_to_binary(
            json:encode([
                #{<<"role">> => <<"user">>, <<"content">> => <<"hi">>}
            ])
        ),
        tools => <<"[]">>
    },
    {ok, Params, _Prompt} = barrel_inference_chat:apply(Templates, Inputs),
    {ok, Partial} = barrel_inference_chat:parse(Params, <<"hel">>, true),
    {ok, Full} = barrel_inference_chat:parse(Params, <<"hello">>, false),
    ?assert(is_map(Partial)),
    ?assert(is_map(Full)),
    #{content := PartialContent} = Partial,
    #{content := FullContent} = Full,
    ?assert(byte_size(FullContent) >= byte_size(PartialContent)).
