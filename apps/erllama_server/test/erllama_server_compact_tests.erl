%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_compact_tests).
-include_lib("eunit/include/eunit.hrl").

-define(C, erllama_server_compact).

sys(T) -> #{role => <<"system">>, content => T}.
user(T) -> #{role => <<"user">>, content => T}.
bot(T) -> #{role => <<"assistant">>, content => T}.
calls(Id) ->
    #{
        role => <<"assistant">>,
        content => null,
        tool_calls => [
            #{
                id => Id,
                type => <<"function">>,
                function => #{name => <<"f">>, arguments => <<"{}">>}
            }
        ]
    }.
result(Id) -> #{role => <<"tool">>, tool_call_id => Id, name => <<"f">>, content => <<"ok">>}.

%% One turn, or only the first request plus recent ones: nothing to drop.
nothing_to_drop_test() ->
    ?assertEqual(nothing, ?C:plan([sys(<<"s">>), user(<<"u1">>), bot(<<"a1">>)], 2)),
    ?assertEqual(nothing, ?C:plan([user(<<"u1">>), user(<<"u2">>)], 1)).

%% The cut is at a user message that keeps at least `Keep' messages; the
%% first request is kept apart, the rest between goes to the summary.
cut_at_user_test() ->
    Msgs = [
        sys(<<"rules">>),
        user(<<"task">>),
        bot(<<"a1">>),
        user(<<"u2">>),
        calls(<<"c1">>),
        result(<<"c1">>),
        bot(<<"a2">>),
        user(<<"u3">>),
        bot(<<"a3">>)
    ],
    {ok, #{system := S, first := F, middle := M, tail := T}} = ?C:plan(Msgs, 2),
    ?assertEqual([sys(<<"rules">>)], S),
    ?assertEqual(user(<<"task">>), F),
    ?assertEqual(
        [bot(<<"a1">>), user(<<"u2">>), calls(<<"c1">>), result(<<"c1">>), bot(<<"a2">>)], M
    ),
    ?assertEqual([user(<<"u3">>), bot(<<"a3">>)], T),
    %% More to keep: the cut moves back, with the tool call and its result.
    {ok, #{tail := T4}} = ?C:plan(Msgs, 4),
    ?assertEqual(
        [
            user(<<"u2">>),
            calls(<<"c1">>),
            result(<<"c1">>),
            bot(<<"a2">>),
            user(<<"u3">>),
            bot(<<"a3">>)
        ],
        T4
    ).

%% A tail never starts with a tool result or an assistant message, even
%% when every turn is long: then the latest turn is kept whole.
long_turns_test() ->
    Msgs = [
        user(<<"task">>),
        bot(<<"a">>),
        user(<<"now">>),
        calls(<<"c1">>),
        result(<<"c1">>),
        calls(<<"c2">>),
        result(<<"c2">>)
    ],
    {ok, #{tail := T}} = ?C:plan(Msgs, 50),
    ?assertMatch([#{role := <<"user">>, content := <<"now">>} | _], T),
    ?assertEqual(5, length(T)).

%% One leading system message: the first, with the others and the summary
%% section appended; the first request is quoted in the section.
apply_test() ->
    Plan = #{
        system => [sys(<<"rules">>), sys(<<"late rule">>)],
        first => user(<<"do X">>),
        middle => [],
        tail => [user(<<"u3">>), bot(<<"a3">>)]
    },
    [#{role := <<"system">>, content := Head} | Tail] = ?C:apply_summary(Plan, <<"did half of X">>),
    ?assertEqual([user(<<"u3">>), bot(<<"a3">>)], Tail),
    ?assertMatch(<<"rules\n\nlate rule\n\nEarlier part of this conversation", _/binary>>, Head),
    ?assertNotEqual(nomatch, binary:match(Head, <<"The first request was:\ndo X">>)),
    ?assertNotEqual(nomatch, binary:match(Head, <<"did half of X">>)),
    %% No system message: one is made for the section.
    [#{role := <<"system">>} | _] = ?C:apply_summary(Plan#{system => []}, <<"s">>).

%% The summariser sees calls and results, each clipped.
render_test() ->
    Big = binary:copy(<<"x">>, 5000),
    Text = ?C:render([bot(Big), calls(<<"c1">>), result(<<"c1">>)]),
    ?assert(byte_size(Text) < 4300),
    ?assertNotEqual(nomatch, binary:match(Text, <<"[calls f {}]">>)),
    ?assertNotEqual(nomatch, binary:match(Text, <<"[result of f] ok">>)).
