%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_config).
-moduledoc false.
%% The `erllama_server' application environment: defaults, validation
%% (run once from the top supervisor's init, so a bad value fails the
%% start) and reads.

-export([validate/0, validate/1, get/1, defaults/0]).

-spec defaults() -> #{atom() => term()}.
defaults() ->
    #{
        run_buffer_events => 1024,
        subscriber_max_unacked => 256,
        run_retention_ms => 300000,
        max_active_runs => 64
    }.

-spec validate() -> ok | {error, term()}.
validate() ->
    validate(maps:from_list(application:get_all_env(erllama_server))).

-spec validate(map()) -> ok | {error, term()}.
validate(Env) when is_map(Env) ->
    Bad = [
        {K, V}
     || K := V <- maps:with(maps:keys(defaults()), Env), not valid(K, V)
    ],
    case Bad of
        [] -> ok;
        [{K, V} | _] -> {error, {invalid_config, K, V}}
    end.

valid(run_retention_ms, V) -> is_integer(V) andalso V >= 0;
valid(_, V) -> is_integer(V) andalso V > 0.

-spec get(atom()) -> term().
get(Key) ->
    application:get_env(erllama_server, Key, maps:get(Key, defaults())).
