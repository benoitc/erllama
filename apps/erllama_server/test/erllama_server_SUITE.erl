%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% The server application starts on top of erllama and stops cleanly
%% without taking erllama down with it.
%% @end
-module(erllama_server_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, end_per_testcase/2]).
-export([starts_with_erllama/1, stop_leaves_erllama_running/1]).

all() ->
    [starts_with_erllama, stop_leaves_erllama_running].

end_per_testcase(_TC, _Config) ->
    _ = application:stop(erllama_server),
    _ = application:stop(erllama),
    ok.

starts_with_erllama(_Config) ->
    {ok, Started} = application:ensure_all_started(erllama_server),
    ?assert(lists:member(erllama_server, Started)),
    ?assert(is_pid(whereis(erllama_server_sup))),
    ?assert(lists:keymember(erllama, 1, application:which_applications())),
    ok.

stop_leaves_erllama_running(_Config) ->
    {ok, _} = application:ensure_all_started(erllama_server),
    ok = application:stop(erllama_server),
    ?assertEqual(undefined, whereis(erllama_server_sup)),
    ?assert(lists:keymember(erllama, 1, application:which_applications())),
    ok.
