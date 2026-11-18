%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% erllama_chat:inputs/2: the Erlang-term -> NIF inputs boundary.
-module(erllama_chat_inputs_tests).
-include_lib("eunit/include/eunit.hrl").

tools_are_wrapped_in_the_openai_shape_test() ->
    Tool = #{
        name => <<"calc">>,
        description => <<"Arithmetic">>,
        parameters => #{type => object, properties => #{e => #{type => string}}}
    },
    #{tools := Json} = erllama_chat:inputs([], #{tools => [Tool]}),
    [Decoded] = json:decode(Json),
    ?assertEqual(<<"function">>, maps:get(<<"type">>, Decoded)),
    Fun = maps:get(<<"function">>, Decoded),
    ?assertEqual(<<"calc">>, maps:get(<<"name">>, Fun)),
    ?assertEqual(<<"Arithmetic">>, maps:get(<<"description">>, Fun)),
    ?assertEqual(<<"object">>, maps:get(<<"type">>, maps:get(<<"parameters">>, Fun))).

already_wrapped_tools_pass_through_test() ->
    Wrapped = #{type => function, function => #{name => <<"x">>, parameters => #{}}},
    #{tools := Json} = erllama_chat:inputs([], #{tools => [Wrapped]}),
    [Decoded] = json:decode(Json),
    ?assertEqual(<<"x">>, maps:get(<<"name">>, maps:get(<<"function">>, Decoded))).

tool_without_parameters_gets_an_empty_object_schema_test() ->
    #{tools := Json} = erllama_chat:inputs([], #{tools => [#{name => <<"now">>}]}),
    [#{<<"function">> := #{<<"parameters">> := Params}}] = json:decode(Json),
    ?assertEqual(<<"object">>, maps:get(<<"type">>, Params)).

messages_and_defaults_test() ->
    Msgs = [#{role => user, content => <<"hi">>}],
    Inputs = erllama_chat:inputs(Msgs, #{}),
    ?assertEqual(
        [#{<<"role">> => <<"user">>, <<"content">> => <<"hi">>}],
        json:decode(maps:get(messages, Inputs))
    ),
    ?assertEqual(<<"[]">>, maps:get(tools, Inputs)),
    ?assertEqual(auto, maps:get(tool_choice, Inputs)),
    ?assertEqual(false, maps:get(parallel_tool_calls, Inputs)),
    ?assertEqual(<<>>, maps:get(json_schema, Inputs)),
    ?assertEqual(true, maps:get(enable_thinking, Inputs)),
    ?assertEqual(deepseek, maps:get(reasoning_format, Inputs)),
    ?assertEqual(none, maps:get(continue_final_message, Inputs)).

json_schema_map_is_encoded_binary_passes_through_test() ->
    Schema = #{type => object, properties => #{a => #{type => string}}},
    #{json_schema := Encoded} = erllama_chat:inputs([], #{json_schema => Schema}),
    ?assertEqual(<<"object">>, maps:get(<<"type">>, json:decode(Encoded))),
    Raw = <<"{\"type\":\"object\"}">>,
    #{json_schema := Raw} = erllama_chat:inputs([], #{json_schema => Raw}).

chat_control_keys_pass_through_test() ->
    Inputs = erllama_chat:inputs([], #{
        enable_thinking => false,
        reasoning_format => none,
        continue_final_message => content
    }),
    ?assertEqual(false, maps:get(enable_thinking, Inputs)),
    ?assertEqual(none, maps:get(reasoning_format, Inputs)),
    ?assertEqual(content, maps:get(continue_final_message, Inputs)).

%% =============================================================================
%% merge_params/2: template constraint set -> request opts
%% =============================================================================

render(Overrides) ->
    maps:merge(
        #{
            prompt => <<"P">>,
            format => <<"PEG Simple">>,
            grammar => <<>>,
            grammar_lazy => false,
            trigger_patterns => [],
            trigger_tokens => [],
            additional_stops => [],
            generation_prompt => <<>>,
            supports_thinking => false,
            thinking_start_tag => <<>>,
            thinking_end_tags => []
        },
        Overrides
    ).

merge_lazy_grammar_injects_triggers_test() ->
    Render = render(#{
        grammar => <<"root ::= call">>,
        grammar_lazy => true,
        trigger_patterns => [<<"<tool_call>">>],
        trigger_tokens => [42]
    }),
    Merged = erllama_chat:merge_params(Render, #{temperature => 0.0}),
    ?assertEqual(<<"root ::= call">>, maps:get(grammar, Merged)),
    ?assertEqual(true, maps:get(grammar_lazy, Merged)),
    ?assertEqual([<<"<tool_call>">>], maps:get(trigger_patterns, Merged)),
    ?assertEqual([42], maps:get(trigger_tokens, Merged)),
    ?assertNot(maps:is_key(grammar_prefill, Merged)),
    ?assertEqual(0.0, maps:get(temperature, Merged)).

merge_plain_grammar_carries_prefill_test() ->
    Render = render(#{
        grammar => <<"root ::= call">>,
        generation_prompt => <<"<|assistant|>">>
    }),
    Merged = erllama_chat:merge_params(Render, #{}),
    ?assertEqual(<<"root ::= call">>, maps:get(grammar, Merged)),
    ?assertNot(maps:is_key(grammar_lazy, Merged)),
    ?assertEqual(<<"<|assistant|>">>, maps:get(grammar_prefill, Merged)).

merge_plain_grammar_without_generation_prompt_has_no_prefill_test() ->
    Merged = erllama_chat:merge_params(render(#{grammar => <<"root ::= x">>}), #{}),
    ?assertNot(maps:is_key(grammar_prefill, Merged)).

merge_caller_grammar_wins_test() ->
    Render = render(#{grammar => <<"root ::= template">>, grammar_lazy => true}),
    Merged = erllama_chat:merge_params(Render, #{grammar => <<"root ::= mine">>}),
    ?assertEqual(<<"root ::= mine">>, maps:get(grammar, Merged)),
    ?assertNot(maps:is_key(grammar_lazy, Merged)),
    ?assertNot(maps:is_key(trigger_patterns, Merged)).

merge_empty_grammar_is_a_no_op_test() ->
    Opts = #{temperature => 0.5},
    ?assertEqual(Opts, erllama_chat:merge_params(render(#{}), Opts)).

merge_additional_stops_append_and_dedup_test() ->
    Render = render(#{additional_stops => [<<"</assistant>">>, <<"STOP">>]}),
    Merged = erllama_chat:merge_params(Render, #{stop_sequences => [<<"STOP">>]}),
    ?assertEqual([<<"STOP">>, <<"</assistant>">>], maps:get(stop_sequences, Merged)).

%% =============================================================================
%% chat/3 conflict checks (run before any model call, so no model
%% needs to be loaded)
%% =============================================================================

grammar_conflicts_with_tools_test() ->
    Opts = #{
        tools => [#{name => <<"t">>}],
        grammar => <<"root ::= x">>
    },
    ?assertEqual(
        {error, {invalid_option, grammar, conflicts_with_tools}},
        erllama_chat:chat(<<"no_such_model">>, [], Opts)
    ).

grammar_with_tool_choice_none_is_allowed_past_conflicts_test() ->
    %% tool_choice => none deactivates the tools; the caller grammar is
    %% then legal and the call proceeds to the model lookup (which,
    %% with the app not running here, raises badarg on the missing
    %% registry table - proof the conflicts check passed).
    Opts = #{
        tools => [#{name => <<"t">>}],
        tool_choice => none,
        grammar => <<"root ::= x">>
    },
    Result =
        try erllama_chat:chat(<<"no_such_model">>, [], Opts) of
            R -> R
        catch
            error:badarg -> model_lookup_reached
        end,
    ?assertEqual(model_lookup_reached, Result).

json_schema_conflicts_with_tools_test() ->
    Opts = #{
        tools => [#{name => <<"t">>}],
        json_schema => #{type => object}
    },
    ?assertEqual(
        {error, {invalid_option, json_schema, conflicts_with_tools}},
        erllama_chat:chat(<<"no_such_model">>, [], Opts)
    ).

%% =============================================================================
%% Media content-part extraction (extract_media/1)
%% =============================================================================

extract_media_replaces_parts_in_order_test() ->
    Img1 = <<"img-one">>,
    Aud = <<"aud-one">>,
    {Msgs, Media} = erllama_chat:extract_media([
        #{
            role => user,
            content => [
                #{type => text, text => <<"first ">>},
                #{type => image, data => Img1},
                #{type => text, text => <<" then ">>},
                #{type => audio, data => Aud}
            ]
        }
    ]),
    ?assertEqual(
        [#{type => image, data => Img1}, #{type => audio, data => Aud}],
        Media
    ),
    [#{content := Parts}] = Msgs,
    ?assertEqual(
        [
            #{type => text, text => <<"first ">>},
            #{type => media_marker, text => <<"<__media__>">>},
            #{type => text, text => <<" then ">>},
            #{type => media_marker, text => <<"<__media__>">>}
        ],
        Parts
    ).

extract_media_across_messages_keeps_order_test() ->
    {_, Media} = erllama_chat:extract_media([
        #{role => user, content => [#{type => image, data => <<"a">>}]},
        #{role => assistant, content => <<"seen">>},
        #{role => user, content => [#{type => image, data => <<"b">>}]}
    ]),
    ?assertEqual([<<"a">>, <<"b">>], [D || #{data := D} <- Media]).

extract_media_no_media_is_identity_test() ->
    Msgs = [
        #{role => system, content => <<"sys">>},
        #{role => user, content => [#{type => text, text => <<"hi">>}]}
    ],
    ?assertEqual({Msgs, []}, erllama_chat:extract_media(Msgs)).

extract_media_binary_content_untouched_test() ->
    Msgs = [#{role => user, content => <<"plain">>}],
    ?assertEqual({Msgs, []}, erllama_chat:extract_media(Msgs)).
