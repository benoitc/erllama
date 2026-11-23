%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Cache-level eunit. ETS-observable semantics only (put/lookup,
%% LRU eviction, purge). The `get_or_init/4' path needs a real
%% model resource and is covered in `erllama_chat_SUITE'.
-module(erllama_chat_cache_tests).
-include_lib("eunit/include/eunit.hrl").

-define(M, erllama_chat_cache).
-define(TAB, erllama_chat_cache).

setup() ->
    application:set_env(erllama, chat_cache_size, 4),
    {ok, Pid} = ?M:start_link(),
    Pid.

cleanup(Pid) ->
    application:unset_env(erllama, chat_cache_size),
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
        fun purge_leaves_other_models/1,
        fun purge_drops_every_load_of_the_model/1,
        fun cached_reads_only_the_same_load/1
    ]}.

%% Any live pid stands in for the model process owning an entry; the
%% cache server's is the same in the fixture and in the test processes.
owner() -> whereis(?M).

purge_drops_every_load_of_the_model(_Pid) ->
    Old = spawn(fun() -> ok end),
    ?M:put(?TAB, {templates, <<"model-a">>, Old}, make_ref()),
    ?M:put(?TAB, {templates, <<"model-a">>, owner()}, make_ref()),
    ok = ?M:purge(<<"model-a">>),
    [
        ?_assertEqual(not_found, ?M:cached(<<"model-a">>, Old)),
        ?_assertEqual(not_found, ?M:cached(<<"model-a">>, owner()))
    ].

%% A model loaded again under the same id is a new process: it must
%% not get the templates the previous load built.
cached_reads_only_the_same_load(_Pid) ->
    Old = spawn(fun() -> ok end),
    ?M:put(?TAB, {templates, <<"model-a">>, Old}, old_ref),
    [
        ?_assertEqual({ok, old_ref}, ?M:cached(<<"model-a">>, Old)),
        ?_assertEqual(not_found, ?M:cached(<<"model-a">>, owner()))
    ].

put_get_round_trip(_Pid) ->
    Key = {templates, <<"model-a">>, owner()},
    Ref = make_ref(),
    ?M:put(?TAB, Key, Ref),
    ?_assertEqual({ok, Ref}, ?M:lookup(?TAB, Key)).

lookup_miss_returns_not_found(_Pid) ->
    ?_assertEqual(not_found, ?M:lookup(?TAB, {templates, <<"missing">>, owner()})).

lru_evicts_oldest(_Pid) ->
    %% Cache size is 4 (set in setup). After 5 inserts the oldest
    %% must be evicted.
    Keys = [
        {templates, <<"m1">>, owner()},
        {templates, <<"m2">>, owner()},
        {templates, <<"m3">>, owner()},
        {templates, <<"m4">>, owner()},
        {templates, <<"m5">>, owner()}
    ],
    [?M:put(?TAB, K, make_ref()) || K <- Keys],
    [_Oldest | Rest] = Keys,
    [
        ?_assertEqual(not_found, ?M:lookup(?TAB, {templates, <<"m1">>, owner()})),
        [?_assertMatch({ok, _}, ?M:lookup(?TAB, K)) || K <- Rest]
    ].

purge_drops_model_entries(_Pid) ->
    ?M:put(?TAB, {templates, <<"model-a">>, owner()}, make_ref()),
    ok = ?M:purge(<<"model-a">>),
    ?_assertEqual(not_found, ?M:lookup(?TAB, {templates, <<"model-a">>, owner()})).

purge_leaves_other_models(_Pid) ->
    ?M:put(?TAB, {templates, <<"keep">>, owner()}, keep_ref),
    ?M:put(?TAB, {templates, <<"drop">>, owner()}, drop_ref),
    ok = ?M:purge(<<"drop">>),
    [
        ?_assertEqual({ok, keep_ref}, ?M:lookup(?TAB, {templates, <<"keep">>, owner()})),
        ?_assertEqual(not_found, ?M:lookup(?TAB, {templates, <<"drop">>, owner()}))
    ].
