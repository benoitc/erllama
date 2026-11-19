%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_generate_tests).
-include_lib("eunit/include/eunit.hrl").

-define(G, erllama_server_generate).

tools() -> [#{name => <<"weather">>, parameters => #{type => object}}].

%% -----------------------------------------------------------------------------
%% Option conflicts, as erllama:chat/3 enforces them
%% -----------------------------------------------------------------------------

grammar_conflicts_with_active_tools_test() ->
    ?assertEqual(
        {error, {invalid_option, grammar, conflicts_with_tools}},
        ?G:check_opts(#{tools => tools(), grammar => <<"root ::= \"x\"">>})
    ),
    ?assertEqual(
        {error, {invalid_option, json_schema, conflicts_with_tools}},
        ?G:check_opts(#{tools => tools(), json_schema => #{type => object}})
    ).

no_conflict_without_active_tools_test() ->
    ?assertEqual(ok, ?G:check_opts(#{tools => tools(), tool_choice => none, grammar => <<"g">>})),
    ?assertEqual(ok, ?G:check_opts(#{tools => [], grammar => <<"g">>})),
    ?assertEqual(ok, ?G:check_opts(#{grammar => <<"g">>})),
    ?assertEqual(ok, ?G:check_opts(#{tools => tools()})).

start_run_refuses_conflicts_before_running_test() ->
    ?assertEqual(
        {error, {invalid_option, grammar, conflicts_with_tools}},
        erllama_server_api:start_run(#{
            model => <<"m">>,
            messages => [#{role => user, content => <<"x">>}],
            opts => #{tools => tools(), grammar => <<"g">>}
        })
    ).

%% -----------------------------------------------------------------------------
%% Template constraints combined with caller options
%% -----------------------------------------------------------------------------

template_lazy() ->
    #{
        grammar => <<"tmpl">>,
        grammar_lazy => true,
        trigger_patterns => [<<"<tool">>],
        trigger_tokens => [7]
    }.

template_grammar_is_kept_without_caller_grammar_test() ->
    Req = ?G:constrain(template_lazy(), [], #{temperature => 0.1}),
    ?assertEqual(<<"tmpl">>, maps:get(grammar, Req)),
    ?assertEqual(true, maps:get(grammar_lazy, Req)),
    ?assertEqual([<<"<tool">>], maps:get(trigger_patterns, Req)),
    ?assertEqual(0.1, maps:get(temperature, Req)).

caller_grammar_replaces_the_whole_template_constraint_test() ->
    Req = ?G:constrain(template_lazy(), [], #{grammar => <<"mine">>}),
    ?assertEqual(#{grammar => <<"mine">>}, Req),
    Prefilled = ?G:constrain(
        #{grammar => <<"tmpl">>, grammar_prefill => <<"<|im_start|>">>}, [], #{
            grammar => <<"mine">>
        }
    ),
    ?assertEqual(#{grammar => <<"mine">>}, Prefilled).

template_stops_follow_the_callers_test() ->
    Req = ?G:constrain(#{}, [<<"</s>">>, <<"STOP">>], #{stop_sequences => [<<"STOP">>]}),
    ?assertEqual([<<"STOP">>, <<"</s>">>], maps:get(stop_sequences, Req)),
    ?assertEqual(#{}, ?G:constrain(#{}, [], #{})).

%% -----------------------------------------------------------------------------
%% Chat messages and tool-call history
%% -----------------------------------------------------------------------------

flat_tool_call_becomes_openai_shape_test() ->
    {ok, Msg} = ?G:message(#{
        role => assistant,
        tool_calls => [
            #{name => <<"weather">>, arguments => #{city => <<"Paris">>}, id => <<"call_1">>}
        ]
    }),
    ?assertMatch(
        #{
            role := <<"assistant">>,
            content := null,
            tool_calls := [
                #{id := <<"call_1">>, type := <<"function">>, function := #{name := <<"weather">>}}
            ]
        },
        Msg
    ),
    #{tool_calls := [#{function := #{arguments := Args}}]} = Msg,
    ?assertEqual(#{<<"city">> => <<"Paris">>}, json:decode(Args)).

openai_tool_call_is_accepted_test() ->
    Call = #{
        id => <<"call_2">>,
        type => <<"function">>,
        function => #{name => <<"weather">>, arguments => <<"{\"city\":\"Oslo\"}">>}
    },
    ?assertEqual(
        {ok, #{role => <<"assistant">>, content => <<"checking">>, tool_calls => [Call]}},
        ?G:message(#{role => <<"assistant">>, content => <<"checking">>, tool_calls => [Call]})
    ).

tool_call_without_id_gets_one_test() ->
    {ok, #{tool_calls := [#{id := Id}]}} =
        ?G:message(#{role => assistant, tool_calls => [#{name => <<"w">>, arguments => #{}}]}),
    ?assertMatch(<<"call_", _/binary>>, Id).

tool_result_keeps_its_call_id_test() ->
    ?assertEqual(
        {ok, #{
            role => <<"tool">>,
            content => <<"sunny">>,
            tool_call_id => <<"call_1">>,
            name => <<"weather">>
        }},
        ?G:message(#{
            role => tool,
            content => <<"sunny">>,
            tool_call_id => <<"call_1">>,
            name => <<"weather">>
        })
    ).

invalid_messages_are_refused_test() ->
    Bad = [
        #{role => tool, content => <<"no id">>},
        #{role => tool, content => <<"x">>, tool_call_id => <<>>},
        #{role => assistant},
        #{role => assistant, content => null},
        #{role => user, content => null},
        #{role => assistant, tool_calls => [#{name => <<"w">>, arguments => <<"[1,2]">>}]},
        #{role => assistant, tool_calls => [#{name => <<"w">>, arguments => <<"not json">>}]},
        #{
            role => assistant,
            tool_calls => [#{type => <<"other">>, function => #{name => <<"w">>}}]
        },
        #{role => assistant, tool_calls => not_a_list},
        #{role => robot, content => <<"x">>},
        #{content => <<"no role">>}
    ],
    [?assertMatch({error, {invalid_message, _}}, ?G:message(M)) || M <- Bad].

reply_with_calls_is_a_valid_history_message_test() ->
    Result = #{
        content => <<>>,
        tool_calls => [
            #{name => <<"weather">>, arguments => #{city => <<"Rome">>}, id => <<"call_9">>}
        ]
    },
    Reply = ?G:reply_message(Result),
    ?assertMatch(
        #{
            role := <<"assistant">>,
            content := null,
            tool_calls := [
                #{id := <<"call_9">>, type := <<"function">>, function := #{name := <<"weather">>}}
            ]
        },
        Reply
    ),
    %% What the conversation stores is accepted again as input.
    ?assertEqual({ok, Reply}, ?G:message(Reply)).

reply_without_calls_is_plain_text_test() ->
    ?assertEqual(
        #{role => <<"assistant">>, content => <<"hi">>},
        ?G:reply_message(#{content => <<"hi">>, tool_calls => []})
    ).
