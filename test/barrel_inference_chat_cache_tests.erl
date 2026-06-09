%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Cache-level eunit. ETS-observable semantics only (put/lookup,
%% LRU eviction, purge). The `get_or_init/3' path needs a real
%% model resource and is covered in `barrel_inference_chat_SUITE'.
-module(barrel_inference_chat_cache_tests).
-include_lib("eunit/include/eunit.hrl").

-define(M, barrel_inference_chat_cache).
-define(TAB, barrel_inference_chat_cache).

setup() ->
    application:set_env(barrel_inference, chat_params_cache_size, 4),
    {ok, Pid} = ?M:start_link(),
    Pid.

cleanup(Pid) ->
    application:unset_env(barrel_inference, chat_params_cache_size),
    case is_process_alive(Pid) of
        true ->
            MRef = monitor(process, Pid),
            unlink(Pid),
            exit(Pid, shutdown),
            receive
                {'DOWN', MRef, process, Pid, _} -> ok
            after 1000 -> ok
            end;
        false ->
            ok
    end,
    ok.

cache_test_() ->
    {foreach, fun setup/0, fun cleanup/1, [
        fun put_get_round_trip/1,
        fun lookup_miss_returns_not_found/1,
        fun lru_evicts_oldest/1,
        fun purge_drops_model_entries/1,
        fun purge_leaves_other_models/1
    ]}.

put_get_round_trip(_Pid) ->
    Key = {templates, <<"model-a">>},
    Ref = make_ref(),
    ?M:put(?TAB, Key, Ref),
    ?_assertEqual({ok, Ref}, ?M:lookup(?TAB, Key)).

lookup_miss_returns_not_found(_Pid) ->
    ?_assertEqual(not_found, ?M:lookup(?TAB, {templates, <<"missing">>})).

lru_evicts_oldest(_Pid) ->
    %% Cache size is 4 (set in setup). After 5 inserts the oldest
    %% must be evicted.
    Keys = [
        {templates, <<"m1">>},
        {templates, <<"m2">>},
        {templates, <<"m3">>},
        {templates, <<"m4">>},
        {templates, <<"m5">>}
    ],
    [?M:put(?TAB, K, make_ref()) || K <- Keys],
    [_Oldest | Rest] = Keys,
    [
        ?_assertEqual(not_found, ?M:lookup(?TAB, {templates, <<"m1">>})),
        [?_assertMatch({ok, _}, ?M:lookup(?TAB, K)) || K <- Rest]
    ].

purge_drops_model_entries(_Pid) ->
    ?M:put(?TAB, {templates, <<"model-a">>}, make_ref()),
    ok = ?M:purge(<<"model-a">>),
    ?_assertEqual(not_found, ?M:lookup(?TAB, {templates, <<"model-a">>})).

purge_leaves_other_models(_Pid) ->
    ?M:put(?TAB, {templates, <<"keep">>}, keep_ref),
    ?M:put(?TAB, {templates, <<"drop">>}, drop_ref),
    ok = ?M:purge(<<"drop">>),
    [
        ?_assertEqual({ok, keep_ref}, ?M:lookup(?TAB, {templates, <<"keep">>})),
        ?_assertEqual(not_found, ?M:lookup(?TAB, {templates, <<"drop">>}))
    ].
