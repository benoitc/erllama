%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_conv_tests).
-include_lib("eunit/include/eunit.hrl").

-define(LOG, erllama_server_conv_log).
-define(CODEC, erllama_server_conv_codec).

%% Every message shape the release keeps comes back as it was.
codec_round_trip_test() ->
    Msgs = [
        #{role => <<"system">>, content => <<"be brief">>},
        #{role => <<"user">>, content => <<"hello é"/utf8>>},
        #{
            role => <<"assistant">>,
            content => null,
            tool_calls => [
                #{
                    id => <<"c1">>,
                    type => <<"function">>,
                    function => #{name => <<"read_file">>, arguments => <<"{\"path\":\"a\"}">>}
                }
            ]
        },
        #{
            role => <<"tool">>,
            tool_call_id => <<"c1">>,
            name => <<"read_file">>,
            content => <<"x">>
        },
        #{
            role => <<"user">>,
            content => [
                #{type => <<"text">>, text => <<"what is this?">>},
                #{type => image, data => <<0, 1, 2, 255>>}
            ]
        }
    ],
    Back = ?CODEC:decode_msgs(json_round(?CODEC:encode_msgs(Msgs))),
    ?assertEqual(Msgs, Back).

%% A key no message uses stays a binary: no atom from file contents.
codec_unknown_key_test() ->
    [M] = ?CODEC:decode_msgs([#{<<"role">> => <<"user">>, <<"surprise">> => 1}]),
    ?assertEqual(#{role => <<"user">>, <<"surprise">> => 1}, M).

codec_key_test() ->
    K = crypto:strong_rand_bytes(32),
    ?assertEqual(K, ?CODEC:decode_key(?CODEC:encode_key(K))),
    ?assertEqual(null, ?CODEC:encode_key(undefined)),
    ?assertEqual(undefined, ?CODEC:decode_key(null)),
    ?assertEqual(undefined, ?CODEC:decode_key(<<"not hex">>)).

json_round(T) -> json:decode(iolist_to_binary(json:encode(T))).

%% -----------------------------------------------------------------------------
%% Logs
%% -----------------------------------------------------------------------------

with_dir(Fun) ->
    Dir = filename:join([
        os:getenv("TMPDIR", "/tmp"),
        "erllama_conv_tests",
        os:getpid() ++ "-" ++ integer_to_list(erlang:unique_integer([positive]))
    ]),
    ok = application:set_env(erllama_server, data_dir, Dir),
    try
        Fun()
    after
        application:unset_env(erllama_server, data_dir),
        file:del_dir_r(Dir)
    end.

user(T) -> #{role => <<"user">>, content => T}.

%% A log is read back in order; a torn last line is skipped.
torn_tail_test() ->
    with_dir(fun() ->
        Id = <<"conv_a">>,
        K = crypto:strong_rand_bytes(32),
        ok = ?LOG:append(Id, [
            ?LOG:create(Id, <<"m">>, conversation, undefined, 1),
            ?LOG:msgs([user(<<"one">>)], K)
        ]),
        P = filename:join(?LOG:dir(), <<Id/binary, ".jsonl">>),
        ok = file:write_file(P, <<"{\"t\":\"msgs\",\"msgs\":[{\"ro">>, [append]),
        ?assertEqual([Id], ?LOG:ids()),
        {ok, Records} = ?LOG:read(Id),
        ?assertEqual(2, length(Records)),
        {ok, C} = ?LOG:replay(Records),
        ?assertMatch(#{id := Id, model := <<"m">>, last_finish_key := K}, C),
        ?assertEqual([user(<<"one">>)], maps:get(messages, C))
    end).

%% A finished run needs no repair.
clean_run_test() ->
    {ok, C} = replay([
        ?LOG:run(<<"r1">>, #{phase => 'begin', kind => run, model => <<"m">>}),
        ?LOG:msgs([user(<<"q">>), #{role => <<"assistant">>, content => <<"a">>}], undefined),
        ?LOG:run(<<"r1">>, #{phase => 'end'})
    ]),
    {Fixed, Repairs} = ?LOG:recover(C),
    ?assertEqual([], Repairs),
    ?assertEqual([], maps:get(unresolved, Fixed)).

%% A run cut in the middle of its tool calls: one finished (its outcome
%% is used as it was), one had started (unresolved), one never started.
cut_run_test() ->
    Call = fun(Id) ->
        #{
            id => Id,
            type => <<"function">>,
            function => #{name => <<"shell">>, arguments => <<"{}">>}
        }
    end,
    {ok, C} = replay([
        ?LOG:run(<<"a1">>, #{
            phase => 'begin', kind => agent, model => <<"m">>, workspace => <<"/w">>
        }),
        ?LOG:msgs(
            [
                user(<<"go">>),
                #{
                    role => <<"assistant">>,
                    content => null,
                    tool_calls => [Call(<<"done">>), Call(<<"running">>), Call(<<"queued">>)]
                }
            ],
            undefined
        ),
        ?LOG:tool_intent(<<"a1">>, <<"done">>, <<"shell">>, #{}),
        ?LOG:tool_outcome(<<"a1">>, <<"done">>, ok, <<"exit 0">>),
        ?LOG:tool_intent(<<"a1">>, <<"running">>, <<"shell">>, #{})
    ]),
    ?assertEqual(<<"/w">>, maps:get(workspace, C)),
    {Fixed, Repairs} = ?LOG:recover(C),
    Results = [
        {Id, Content}
     || #{role := <<"tool">>, tool_call_id := Id, content := Content} <- maps:get(messages, Fixed)
    ],
    ?assertMatch(
        [
            {<<"done">>, <<"exit 0">>},
            {<<"running">>, <<"Error: the release stopped while this call was running", _/binary>>},
            {<<"queued">>, <<"Error: the release stopped before this call ran", _/binary>>}
        ],
        Results
    ),
    ?assertEqual([<<"running">>], maps:get(unresolved, Fixed)),
    ?assertNot(maps:is_key(open_run, Fixed)),
    %% The repair is durable: replaying with it needs no further repair,
    %% and the unresolved call stays listed.
    {ok, Again} = replay_records(
        [
            ?LOG:run(<<"a1">>, #{phase => 'begin', kind => agent, model => <<"m">>})
        ] ++ Repairs
    ),
    ?assertEqual([<<"running">>], maps:get(unresolved, element(1, ?LOG:recover(Again)))).

replay(Records) ->
    replay_records([?LOG:create(<<"c">>, <<"m">>, conversation, undefined, 1) | Records]).

replay_records(Records) ->
    Recs = [json_round(R) || R <- Records],
    case Recs of
        [#{<<"t">> := <<"create">>} | _] ->
            ?LOG:replay(Recs);
        _ ->
            ?LOG:replay([
                json_round(?LOG:create(<<"c">>, <<"m">>, conversation, undefined, 1)) | Recs
            ])
    end.
