%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_acp_tests).
-include_lib("eunit/include/eunit.hrl").

%% ---------------------------------------------------------------------------
%% Framing (erllama_server_acp_stdio)
%% ---------------------------------------------------------------------------

parse_empty_is_more_test() ->
    ?assertEqual(more, erllama_server_acp_stdio:parse(<<>>)).

parse_partial_header_is_more_test() ->
    ?assertEqual(more, erllama_server_acp_stdio:parse(<<"Content-Len">>)).

parse_partial_body_is_more_test() ->
    ?assertEqual(more, erllama_server_acp_stdio:parse(<<"Content-Length: 10\r\n\r\n{\"a\":1">>)).

parse_one_frame_test() ->
    F = erllama_server_acp_stdio:frame(#{<<"a">> => 1}),
    Body = iolist_to_binary(F),
    {ok, B, <<>>} = erllama_server_acp_stdio:parse(Body),
    ?assertEqual({ok, #{<<"a">> => 1}}, erllama_server_acp_stdio:decode(B)),
    %% ACP frames are newline-delimited JSON lines.
    ?assertEqual(<<"{">>, binary:part(Body, 0, 1)),
    ?assertEqual($\n, binary:last(Body)).

parse_two_frames_in_one_buffer_test() ->
    F1 = iolist_to_binary(erllama_server_acp_stdio:frame(#{<<"id">> => 1})),
    F2 = iolist_to_binary(erllama_server_acp_stdio:frame(#{<<"id">> => 2})),
    {ok, B1, Rest} = erllama_server_acp_stdio:parse(<<F1/binary, F2/binary>>),
    ?assertEqual({ok, #{<<"id">> => 1}}, erllama_server_acp_stdio:decode(B1)),
    {ok, B2, <<>>} = erllama_server_acp_stdio:parse(Rest),
    ?assertEqual({ok, #{<<"id">> => 2}}, erllama_server_acp_stdio:decode(B2)).

parse_header_name_is_case_insensitive_test() ->
    ?assertMatch(
        {ok, _, <<>>},
        erllama_server_acp_stdio:parse(<<"content-LENGTH: 7\r\n\r\n{\"a\":1}">>)
    ).

parse_ndjson_line_test() ->
    {ok, B, <<>>} = erllama_server_acp_stdio:parse(<<"{\"a\":1}\n">>),
    ?assertEqual({ok, #{<<"a">> => 1}}, erllama_server_acp_stdio:decode(B)).

parse_ndjson_crlf_line_test() ->
    {ok, B, <<>>} = erllama_server_acp_stdio:parse(<<"{\"a\":1}\r\n">>),
    ?assertEqual({ok, #{<<"a">> => 1}}, erllama_server_acp_stdio:decode(B)).

parse_ndjson_two_lines_test() ->
    {ok, B1, Rest} = erllama_server_acp_stdio:parse(<<"{\"id\":1}\n{\"id\":2}\n">>),
    {ok, B2, <<>>} = erllama_server_acp_stdio:parse(Rest),
    ?assertEqual({ok, #{<<"id">> => 1}}, erllama_server_acp_stdio:decode(B1)),
    ?assertEqual({ok, #{<<"id">> => 2}}, erllama_server_acp_stdio:decode(B2)).

parse_skips_noise_lines_test() ->
    {ok, B, <<>>} = erllama_server_acp_stdio:parse(<<"=ERROR REPORT==\n{\"a\":1}\n">>),
    ?assertEqual({ok, #{<<"a">> => 1}}, erllama_server_acp_stdio:decode(B)).

parse_lsp_with_extra_headers_test() ->
    In = <<"Content-Length: 7\r\nContent-Type: application/vscode-jsonrpc\r\n\r\n{\"a\":1}">>,
    {ok, B, <<>>} = erllama_server_acp_stdio:parse(In),
    ?assertEqual({ok, #{<<"a">> => 1}}, erllama_server_acp_stdio:decode(B)).

parse_header_only_frame_then_line_test() ->
    %% LSP framing without a `Content-Length' header is not entered: the
    %% blank line just separates and the next line is one JSON message.
    {ok, B, <<>>} = erllama_server_acp_stdio:parse(
        <<"Content-Type: x\r\n\r\n{\"a\":1}\n">>
    ),
    ?assertEqual({ok, #{<<"a">> => 1}}, erllama_server_acp_stdio:decode(B)).

parse_lsp_body_short_is_more_test() ->
    ?assertEqual(
        more,
        erllama_server_acp_stdio:parse(<<"Content-Length: 20\r\n\r\n{\"a\":1}">>)
    ).

parse_bad_content_length_test() ->
    ?assertEqual(
        {error, invalid_content_length},
        erllama_server_acp_stdio:parse(<<"Content-Length: abc\r\n\r\n{}">>)
    ).

frame_decode_roundtrip_utf8_test() ->
    Term = #{<<"text">> => <<"café ☕"/utf8>>},
    {ok, Body, <<>>} = erllama_server_acp_stdio:parse(
        iolist_to_binary(erllama_server_acp_stdio:frame(Term))
    ),
    ?assertEqual({ok, Term}, erllama_server_acp_stdio:decode(Body)).

decode_rejects_non_object_test() ->
    ?assertEqual({error, not_an_object}, erllama_server_acp_stdio:decode(<<"[1,2]">>)),
    ?assertEqual({error, invalid_json}, erllama_server_acp_stdio:decode(<<"{oops">>)).

%% ---------------------------------------------------------------------------
%% Mappings (erllama_server_acp_map)
%% ---------------------------------------------------------------------------

prompt_text_joins_text_blocks_test() ->
    Params = #{
        <<"prompt">> => [
            #{<<"type">> => <<"text">>, <<"text">> => <<"hello ">>},
            #{<<"type">> => <<"text">>, <<"text">> => <<"world">>}
        ]
    },
    ?assertEqual({ok, <<"hello world">>}, erllama_server_acp_map:prompt_text(Params)).

prompt_text_rejects_non_text_test() ->
    Params = #{<<"prompt">> => [#{<<"type">> => <<"image">>, <<"data">> => <<"...">>}]},
    ?assertEqual(
        {error, {unsupported_content, <<"image">>}},
        erllama_server_acp_map:prompt_text(Params)
    ),
    ?assertEqual({error, empty_prompt}, erllama_server_acp_map:prompt_text(#{})).

new_session_spec_uses_cwd_and_defaults_test() ->
    Params = #{
        <<"cwd">> => <<"/tmp/ws">>, <<"mcpServers">> => [], <<"additionalDirectories">> => []
    },
    ?assertEqual(
        {ok, #{workspace => <<"/tmp/ws">>, model => <<"fast">>, policy => yes}},
        erllama_server_acp_map:new_session_spec(Params, #{model => <<"fast">>, policy => yes})
    ).

new_session_spec_workspace_flag_wins_test() ->
    Params = #{<<"cwd">> => <<"/tmp/ws">>},
    ?assertEqual(
        {ok, #{workspace => <<"/other">>, policy => default}},
        erllama_server_acp_map:new_session_spec(Params, #{workspace => <<"/other">>})
    ).

new_session_spec_accepts_and_ignores_additional_directories_test() ->
    Params = #{
        <<"cwd">> => <<"/tmp/ws">>,
        <<"mcpServers">> => [],
        <<"additionalDirectories">> => [<<"/tmp/other">>]
    },
    ?assertEqual(
        {ok, #{workspace => <<"/tmp/ws">>, policy => yes}},
        erllama_server_acp_map:new_session_spec(Params, #{policy => yes})
    ),
    ?assertEqual(
        {ok, #{workspace => <<"/tmp/ws">>, policy => default}},
        erllama_server_acp_map:new_session_spec(#{<<"cwd">> => <<"/tmp/ws">>}, #{})
    ).

new_session_spec_rejects_test() ->
    ?assertEqual(
        {error, missing_cwd},
        erllama_server_acp_map:new_session_spec(#{}, #{})
    ),
    ?assertEqual(
        {error, mcp_servers_unsupported},
        erllama_server_acp_map:new_session_spec(
            #{<<"cwd">> => <<"/w">>, <<"mcpServers">> => [#{}]}, #{}
        )
    ).

load_session_spec_test() ->
    ?assertEqual(
        {ok, #{conversation_id => <<"conv_1">>}},
        erllama_server_acp_map:load_session_spec(#{
            <<"sessionId">> => <<"conv_1">>, <<"cwd">> => <<"/w">>
        })
    ),
    ?assertEqual({error, missing_session_id}, erllama_server_acp_map:load_session_spec(#{})).

modes_test() ->
    M = erllama_server_acp_map:modes(yes),
    ?assertEqual(<<"yes">>, maps:get(<<"currentModeId">>, M)),
    Ids = [maps:get(<<"id">>, O) || O <- maps:get(<<"availableModes">>, M)],
    ?assertEqual([<<"default">>, <<"yes">>, <<"read_only">>], Ids),
    ?assertEqual({ok, read_only}, erllama_server_acp_map:mode_from_id(<<"read_only">>)),
    ?assertEqual({error, unknown_mode}, erllama_server_acp_map:mode_from_id(<<"bogus">>)).

config_options_lists_models_test() ->
    Models = [#{name => <<"fast">>, loaded => true}, #{name => <<"big">>, loaded => false}],
    [Opt] = erllama_server_acp_map:config_options(Models, <<"fast">>),
    ?assertEqual(<<"model">>, maps:get(<<"id">>, Opt)),
    ?assertEqual(<<"fast">>, maps:get(<<"currentValue">>, Opt)),
    ?assertEqual(
        [
            #{<<"value">> => <<"fast">>, <<"name">> => <<"fast">>},
            #{<<"value">> => <<"big">>, <<"name">> => <<"big">>}
        ],
        maps:get(<<"options">>, Opt)
    ).

session_info_test() ->
    S = erllama_server_acp_map:session_info(#{
        session_id => <<"conv_1">>, cwd => <<"/tmp/ws">>, updated_at => 0
    }),
    ?assertEqual(<<"conv_1">>, maps:get(<<"sessionId">>, S)),
    ?assertEqual(<<"/tmp/ws">>, maps:get(<<"cwd">>, S)),
    ?assert(is_binary(maps:get(<<"updatedAt">>, S))).

event_updates_test() ->
    U = fun(E) -> erllama_server_acp_map:event_updates(E) end,
    [User] = U(#{type => turn_started, data => #{input => <<"hi">>}}),
    ?assertEqual(<<"user_message_chunk">>, maps:get(<<"sessionUpdate">>, User)),
    ?assertEqual(
        #{<<"type">> => <<"text">>, <<"text">> => <<"hi">>}, maps:get(<<"content">>, User)
    ),
    [Msg] = U(#{type => token, data => <<"a">>}),
    ?assertEqual(<<"agent_message_chunk">>, maps:get(<<"sessionUpdate">>, Msg)),
    [Thought] = U(#{type => thinking, data => <<"hmm">>}),
    ?assertEqual(<<"agent_thought_chunk">>, maps:get(<<"sessionUpdate">>, Thought)),
    [Tool] = U(#{
        type => tool_started,
        data => #{
            call_id => <<"c1">>, name => <<"write_file">>, arguments => #{<<"path">> => <<"a.txt">>}
        }
    }),
    ?assertEqual(<<"tool_call">>, maps:get(<<"sessionUpdate">>, Tool)),
    ?assertEqual(<<"c1">>, maps:get(<<"toolCallId">>, Tool)),
    ?assertEqual(<<"edit">>, maps:get(<<"kind">>, Tool)),
    ?assertEqual([#{<<"path">> => <<"a.txt">>}], maps:get(<<"locations">>, Tool)),
    [Progress] = U(#{
        type => tool_output, data => #{call_id => <<"c1">>, output => <<"ready">>, omitted => 0}
    }),
    ?assertMatch(
        #{
            <<"toolCallId">> := <<"c1">>,
            <<"status">> := <<"in_progress">>,
            <<"content">> := [#{<<"content">> := #{<<"text">> := <<"ready">>}}]
        },
        Progress
    ),
    [Upd] = U(#{
        type => tool_result, data => #{call_id => <<"c1">>, ok => false, output => <<"boom">>}
    }),
    ?assertEqual(<<"tool_call_update">>, maps:get(<<"sessionUpdate">>, Upd)),
    ?assertEqual(<<"failed">>, maps:get(<<"status">>, Upd)),
    [{permission, Req}] = U(#{
        type => approval_request,
        data => #{call_id => <<"c1">>, name => <<"shell">>, arguments => #{}}
    }),
    ?assertEqual(<<"c1">>, maps:get(call_id, Req)),
    [{turn_done, D}] = U(#{type => turn_done, data => #{reason => stop}}),
    ?assertEqual(#{reason => stop}, D),
    ?assertEqual([session_ended], U(#{type => done, data => #{}})),
    ?assertEqual([], U(#{type => round, data => #{}})).

stop_reason_test() ->
    SR = fun erllama_server_acp_map:stop_reason/1,
    ?assertEqual(<<"end_turn">>, SR(#{reason => stop})),
    ?assertEqual(<<"end_turn">>, SR(#{reason => idle})),
    ?assertEqual(<<"cancelled">>, SR(#{reason => cancelled})),
    ?assertEqual(<<"refusal">>, SR(#{reason => #{error => <<"x">>}})),
    ?assertEqual(<<"max_tokens">>, SR(#{reason => #{budget => tokens}})),
    ?assertEqual(<<"max_turn_requests">>, SR(#{reason => #{budget => rounds}})).

tool_kind_test() ->
    TK = fun erllama_server_acp_map:tool_kind/1,
    ?assertEqual(<<"read">>, TK(<<"read_file">>)),
    ?assertEqual(<<"read">>, TK(<<"list_dir">>)),
    ?assertEqual(<<"search">>, TK(<<"search">>)),
    ?assertEqual(<<"edit">>, TK(<<"write_file">>)),
    ?assertEqual(<<"edit">>, TK(<<"edit_file">>)),
    ?assertEqual(<<"execute">>, TK(<<"shell">>)),
    ?assertEqual(<<"execute">>, TK(<<"shell_bg">>)),
    ?assertEqual(<<"other">>, TK(<<"mcp_tool">>)).

permission_flow_test() ->
    Req = erllama_server_acp_map:permission_request(
        7, <<"conv_1">>, #{
            call_id => <<"c1">>, name => <<"shell">>, arguments => #{<<"cmd">> => <<"ls">>}
        }
    ),
    ?assertEqual(7, maps:get(<<"id">>, Req)),
    ?assertEqual(<<"session/request_permission">>, maps:get(<<"method">>, Req)),
    Params = maps:get(<<"params">>, Req),
    ?assertEqual(<<"conv_1">>, maps:get(<<"sessionId">>, Params)),
    OptIds = [maps:get(<<"optionId">>, O) || O <- maps:get(<<"options">>, Params)],
    ?assertEqual([<<"allow_once">>, <<"allow_always">>, <<"reject_once">>], OptIds),
    PD = fun erllama_server_acp_map:permission_decision/1,
    ?assertEqual(
        allow,
        PD(#{
            <<"outcome">> => #{<<"outcome">> => <<"selected">>, <<"optionId">> => <<"allow_once">>}
        })
    ),
    ?assertEqual(
        always,
        PD(#{
            <<"outcome">> => #{
                <<"outcome">> => <<"selected">>, <<"optionId">> => <<"allow_always">>
            }
        })
    ),
    ?assertEqual(
        deny,
        PD(#{
            <<"outcome">> => #{<<"outcome">> => <<"selected">>, <<"optionId">> => <<"reject_once">>}
        })
    ),
    ?assertEqual(deny, PD(#{<<"outcome">> => #{<<"outcome">> => <<"cancelled">>}})).

envelopes_test() ->
    R = erllama_server_acp_map:response(1, #{<<"ok">> => true}),
    ?assertEqual(
        #{<<"jsonrpc">> => <<"2.0">>, <<"id">> => 1, <<"result">> => #{<<"ok">> => true}}, R
    ),
    E = erllama_server_acp_map:error(2, -32601, <<"nope">>),
    ?assertEqual(-32601, maps:get(<<"code">>, maps:get(<<"error">>, E))),
    N = erllama_server_acp_map:notification(<<"session/update">>, #{<<"a">> => 1}),
    ?assert(is_map_key(<<"method">>, N)),
    ?assert(not is_map_key(<<"id">>, N)).

incoming_test() ->
    ?assertEqual(
        {request, 1, <<"initialize">>, #{<<"x">> => 1}},
        erllama_server_acp_map:incoming(#{
            <<"jsonrpc">> => <<"2.0">>,
            <<"id">> => 1,
            <<"method">> => <<"initialize">>,
            <<"params">> => #{<<"x">> => 1}
        })
    ),
    ?assertEqual(
        {notification, <<"session/cancel">>, #{}},
        erllama_server_acp_map:incoming(#{<<"method">> => <<"session/cancel">>})
    ),
    ?assertEqual(
        {response, 9, #{<<"outcome">> => <<"x">>}},
        erllama_server_acp_map:incoming(#{
            <<"id">> => 9, <<"result">> => #{<<"outcome">> => <<"x">>}
        })
    ),
    ?assertEqual(
        {error_response, 9, -32000, #{<<"code">> => -32000}},
        erllama_server_acp_map:incoming(#{<<"id">> => 9, <<"error">> => #{<<"code">> => -32000}})
    ),
    ?assertEqual(invalid, erllama_server_acp_map:incoming(#{<<"jsonrpc">> => <<"2.0">>})).

initialize_result_test() ->
    R = erllama_server_acp_map:initialize_result(),
    ?assertEqual(1, maps:get(<<"protocolVersion">>, R)),
    Caps = maps:get(<<"agentCapabilities">>, R),
    ?assertEqual(true, maps:get(<<"loadSession">>, Caps)),
    ?assertEqual(<<"erllama">>, maps:get(<<"name">>, maps:get(<<"agentInfo">>, R))).
