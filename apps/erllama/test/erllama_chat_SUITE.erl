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
    apply_tools_synthesizes_lazy_grammar/1,
    apply_required_synthesizes_plain_grammar/1,
    apply_json_schema_synthesizes_grammar/1,
    apply_enable_thinking_prompt/1,
    apply_continue_final_message_prefills/1,
    facade_chat_apply_then_parse/1,
    facade_chat_round_trip/1,
    facade_chat_with_tools/1,
    facade_chat_required_forces_call/1,
    facade_chat_json_schema_constrains_content/1
]).

all() ->
    [
        templates_init_returns_ref,
        apply_then_parse_round_trip,
        parse_partial_then_full,
        apply_tools_synthesizes_lazy_grammar,
        apply_required_synthesizes_plain_grammar,
        apply_json_schema_synthesizes_grammar,
        apply_enable_thinking_prompt,
        apply_continue_final_message_prefills,
        facade_chat_apply_then_parse,
        facade_chat_round_trip,
        facade_chat_with_tools,
        facade_chat_required_forces_call,
        facade_chat_json_schema_constrains_content
    ].

init_per_suite(Config) ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {skip, "set LLAMA_TEST_MODEL to a GGUF path to enable this suite"};
        Path ->
            {ok, _} = application:ensure_all_started(erllama),
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
    {ok, Params, Render} = erllama_chat:apply(Templates, Inputs),
    ?assert(is_reference(Params)),
    #{prompt := Prompt} = Render,
    ?assert(is_binary(Prompt) andalso byte_size(Prompt) > 0),
    %% No tools, no schema: nothing to constrain.
    ?assertEqual(<<>>, maps:get(grammar, Render)),
    ?assertEqual([], maps:get(additional_stops, Render)),
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
    {ok, Params, _Render} = erllama_chat:apply(Templates, Inputs),
    {ok, Partial} = erllama_chat:parse(Params, <<"hel">>, true),
    {ok, Full} = erllama_chat:parse(Params, <<"hello">>, false),
    ?assert(is_map(Partial)),
    ?assert(is_map(Full)),
    #{content := PartialContent} = Partial,
    #{content := FullContent} = Full,
    ?assert(byte_size(FullContent) >= byte_size(PartialContent)).

%% Test tool set shared by the constraint cases below.
suite_tools() ->
    [
        #{
            name => <<"calculate">>,
            description => <<"Evaluate an arithmetic expression.">>,
            parameters => #{
                type => object,
                properties => #{expression => #{type => string}},
                required => [expression]
            }
        }
    ].

suite_inputs(Opts) ->
    erllama_chat:inputs(
        [#{role => user, content => <<"Compute 17 * 23 with the tool.">>}], Opts
    ).

%% The template pass synthesizes a tool grammar only for templates
%% with inferable call markers; the CI model (stories260K, ChatML
%% fallback) has none, so the enforcement cases skip there and run on
%% a tool-capable model (Qwen2.5, Llama 3.1, ...) locally.
-define(NO_TOOL_GRAMMAR, "template synthesizes no tool grammar").

%% tools + tool_choice auto: the template pass synthesizes a LAZY
%% grammar with at least one trigger pattern, so free-text replies stay
%% unconstrained until the model opens a call.
apply_tools_synthesizes_lazy_grammar(Config) ->
    Model = ?config(model, Config),
    {ok, Templates} = erllama_chat:init(Model, undefined),
    {ok, _Params, Render} =
        erllama_chat:apply(Templates, suite_inputs(#{tools => suite_tools()})),
    ct:log("render: ~p", [maps:without([prompt], Render)]),
    case maps:get(grammar, Render) of
        <<>> ->
            {skip, ?NO_TOOL_GRAMMAR};
        _ ->
            ?assertEqual(true, maps:get(grammar_lazy, Render)),
            ?assertMatch([_ | _], maps:get(trigger_patterns, Render)),
            %% merge_params turns it into ready-to-use request opts.
            Merged = erllama_chat:merge_params(Render, #{}),
            ?assertEqual(true, maps:get(grammar_lazy, Merged)),
            ?assertMatch([_ | _], maps:get(trigger_patterns, Merged)),
            ?assertNot(maps:is_key(grammar_prefill, Merged))
    end.

%% tool_choice required: the grammar is not lazy (the whole reply must
%% be a call) and merge_params carries the generation-prompt prefill.
apply_required_synthesizes_plain_grammar(Config) ->
    Model = ?config(model, Config),
    {ok, Templates} = erllama_chat:init(Model, undefined),
    {ok, _Params, Render} =
        erllama_chat:apply(
            Templates,
            suite_inputs(#{tools => suite_tools(), tool_choice => required})
        ),
    case maps:get(grammar, Render) of
        <<>> ->
            {skip, ?NO_TOOL_GRAMMAR};
        _ ->
            ?assertEqual(false, maps:get(grammar_lazy, Render)),
            Merged = erllama_chat:merge_params(Render, #{}),
            ?assertNot(maps:get(grammar_lazy, Merged, false)),
            case maps:get(generation_prompt, Render) of
                <<>> -> ?assertNot(maps:is_key(grammar_prefill, Merged));
                GP -> ?assertEqual(GP, maps:get(grammar_prefill, Merged))
            end
    end.

%% json_schema (response format): a non-lazy grammar constrains the
%% whole reply to the schema.
apply_json_schema_synthesizes_grammar(Config) ->
    Model = ?config(model, Config),
    {ok, Templates} = erllama_chat:init(Model, undefined),
    Schema = #{
        type => object,
        properties => #{answer => #{type => string}},
        required => [answer]
    },
    {ok, _Params, Render} =
        erllama_chat:apply(Templates, suite_inputs(#{json_schema => Schema})),
    ?assert(byte_size(maps:get(grammar, Render)) > 0),
    ?assertEqual(false, maps:get(grammar_lazy, Render)).

%% enable_thinking only changes the render when the template supports
%% thinking; either way the flag round-trips without error and the
%% support probe is reported.
apply_enable_thinking_prompt(Config) ->
    Model = ?config(model, Config),
    {ok, Templates} = erllama_chat:init(Model, undefined),
    {ok, _P1, R1} =
        erllama_chat:apply(Templates, suite_inputs(#{enable_thinking => true})),
    {ok, _P2, R2} =
        erllama_chat:apply(Templates, suite_inputs(#{enable_thinking => false})),
    ct:log("supports_thinking=~p", [maps:get(supports_thinking, R1)]),
    case maps:get(supports_thinking, R1) of
        true -> ?assertNotEqual(maps:get(prompt, R1), maps:get(prompt, R2));
        false -> ?assertEqual(maps:get(prompt, R1), maps:get(prompt, R2))
    end.

%% Assistant prefill: the trailing assistant message is popped from the
%% rendered history and appended to the prompt as the beginning of the
%% reply.
apply_continue_final_message_prefills(Config) ->
    Model = ?config(model, Config),
    {ok, Templates} = erllama_chat:init(Model, undefined),
    Prefill = <<"The answer is">>,
    Messages = [
        #{role => user, content => <<"What is 17 * 23?">>},
        #{role => assistant, content => Prefill}
    ],
    {ok, _Params, Render} =
        erllama_chat:apply(
            Templates,
            erllama_chat:inputs(Messages, #{continue_final_message => content})
        ),
    #{prompt := Prompt} = Render,
    PSize = byte_size(Prompt),
    FSize = byte_size(Prefill),
    ?assert(PSize > FSize),
    ?assertEqual(Prefill, binary:part(Prompt, PSize - FSize, FSize)).

%% Public façade path: erllama:chat_apply/3 runs one upstream
%% common_chat_templates_apply per request (prompt + parser), and the
%% returned params ref drives erllama:chat_parse/3 for that request.
%% Two calls with the same inputs yield distinct params refs; only the
%% per-model templates ref is cached.
facade_chat_apply_then_parse(_Config) ->
    Path = os:getenv("LLAMA_TEST_MODEL"),
    {ok, ModelId} = erllama:load_model(#{model_path => Path}),
    Messages = [#{role => user, content => <<"hi">>}],
    try
        {ok, #{params := Params1, prompt := Prompt}} = erllama:chat_apply(ModelId, Messages, #{}),
        ?assert(is_reference(Params1)),
        ?assert(is_binary(Prompt) andalso byte_size(Prompt) > 0),
        {ok, #{params := Params2, prompt := Prompt}} = erllama:chat_apply(ModelId, Messages, #{}),
        ?assertNotEqual(Params1, Params2),
        {ok, Msg} = erllama:chat_parse(Params2, <<"hello there">>, false),
        ?assertEqual(<<"assistant">>, maps:get(role, Msg)),
        ?assertEqual(<<"hello there">>, maps:get(content, Msg))
    after
        ok = erllama:unload(ModelId)
    end.

%% erllama:chat/3 end to end: render, generate, parse.
facade_chat_round_trip(_Config) ->
    Path = os:getenv("LLAMA_TEST_MODEL"),
    {ok, ModelId} = erllama:load_model(#{model_path => Path}),
    Messages = [
        #{role => system, content => <<"You are terse.">>},
        #{role => user, content => <<"Say hello.">>}
    ],
    try
        {ok, #{message := Msg, prompt := Prompt, reply := Reply, stats := Stats}} =
            erllama:chat(ModelId, Messages, #{response_tokens => 8, temperature => 0.0}),
        ?assert(byte_size(Prompt) > 0),
        ?assert(is_binary(Reply)),
        ?assertEqual(<<"assistant">>, maps:get(role, Msg)),
        ?assert(is_binary(maps:get(content, Msg))),
        ?assertEqual([], maps:get(tool_calls, Msg)),
        ?assert(maps:get(completion_tokens, Stats) > 0),
        ?assertEqual(
            {error, {unknown_option, max_tokens}},
            erllama:chat(ModelId, Messages, #{max_tokens => 1})
        )
    after
        ok = erllama:unload(ModelId)
    end.

%% Tools reach the template in the OpenAI shape and a tool-capable
%% model answers with a parsed call. Models whose template has no tool
%% support (tinyllama) still render and parse; the assertion on the call
%% itself is only made when the model actually produced one.
facade_chat_with_tools(_Config) ->
    Path = os:getenv("LLAMA_TEST_MODEL"),
    {ok, ModelId} = erllama:load_model(#{model_path => Path}),
    Tools = [
        #{
            name => <<"calculate">>,
            description => <<"Evaluate an arithmetic expression.">>,
            parameters => #{
                type => object,
                properties => #{expression => #{type => string}},
                required => [expression]
            }
        }
    ],
    Messages = [#{role => user, content => <<"Use the calculate tool to compute 17 * 23.">>}],
    try
        {ok, #{prompt := Prompt}} = erllama:chat_apply(ModelId, Messages, #{tools => Tools}),
        ?assert(binary:match(Prompt, <<"calculate">>) =/= nomatch),
        {ok, #{message := Msg}} =
            erllama:chat(ModelId, Messages, #{
                tools => Tools, response_tokens => 64, temperature => 0.0
            }),
        case maps:get(tool_calls, Msg) of
            [] -> ct:comment("model produced no tool call (template without tool support?)");
            [#{name := <<"calculate">>, arguments := Args} | _] -> ?assert(is_map(Args))
        end
    after
        ok = erllama:unload(ModelId)
    end.

%% True when the loaded model's template synthesizes a tool grammar
%% under tool_choice => required (see ?NO_TOOL_GRAMMAR above).
model_has_tool_grammar(ModelId) ->
    {ok, #{sampler_opts := S}} =
        erllama:chat_apply(
            ModelId,
            [#{role => user, content => <<"probe">>}],
            #{tools => suite_tools(), tool_choice => required}
        ),
    maps:get(grammar, S, <<>>) =/= <<>>.

%% tool_choice => required with the grammar enforced during sampling:
%% the reply MUST parse into at least one call.
facade_chat_required_forces_call(_Config) ->
    Path = os:getenv("LLAMA_TEST_MODEL"),
    {ok, ModelId} = erllama:load_model(#{model_path => Path}),
    Messages = [#{role => user, content => <<"Compute 17 * 23.">>}],
    try
        case model_has_tool_grammar(ModelId) of
            false ->
                {skip, ?NO_TOOL_GRAMMAR};
            true ->
                {ok, #{message := Msg, reply := Reply}} =
                    erllama:chat(ModelId, Messages, #{
                        tools => suite_tools(),
                        tool_choice => required,
                        response_tokens => 96,
                        temperature => 0.0
                    }),
                ct:log("reply: ~ts", [Reply]),
                ?assertMatch([#{name := <<"calculate">>} | _], maps:get(tool_calls, Msg))
        end
    after
        ok = erllama:unload(ModelId)
    end.

%% json_schema with the grammar enforced: content decodes as JSON with
%% the required key. Gated on the same tool-grammar probe: a model
%% whose reply is grammar-shaped but semantically random (the CI story
%% model) proves nothing.
facade_chat_json_schema_constrains_content(_Config) ->
    Path = os:getenv("LLAMA_TEST_MODEL"),
    {ok, ModelId} = erllama:load_model(#{model_path => Path}),
    Schema = #{
        type => object,
        properties => #{answer => #{type => string}},
        required => [answer]
    },
    Messages = [#{role => user, content => <<"What is the capital of France?">>}],
    try
        case model_has_tool_grammar(ModelId) of
            false ->
                {skip, ?NO_TOOL_GRAMMAR};
            true ->
                {ok, #{message := #{content := Content}, reply := Reply}} =
                    erllama:chat(ModelId, Messages, #{
                        json_schema => Schema,
                        response_tokens => 64,
                        temperature => 0.0
                    }),
                ct:log("content: ~ts~nreply: ~ts", [Content, Reply]),
                Decoded = json:decode(Content),
                ?assert(is_map(Decoded)),
                ?assert(maps:is_key(<<"answer">>, Decoded))
        end
    after
        ok = erllama:unload(ModelId)
    end.
