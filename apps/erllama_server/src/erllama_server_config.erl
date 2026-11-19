%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_config).
-moduledoc false.
%% The `erllama_server' application environment: defaults, validation
%% (run once when the application starts, so a bad value fails the
%% start) and reads.

-export([validate/0, validate/1, get/1, defaults/0, data_dir/0]).
-compile({no_auto_import, [get/1]}).

-spec defaults() -> #{atom() => term()}.
defaults() ->
    #{
        run_buffer_events => 1024,
        subscriber_max_unacked => 256,
        run_retention_ms => 300000,
        max_active_runs => 64,
        max_active_pulls => 4,
        pull_idle_timeout_ms => 60000,
        data_dir => undefined,
        default_model => undefined,
        keep_alive_ms => 300000,
        default_n_ctx => 4096,
        models => [],
        hf_token => undefined,
        hf_base_url => <<"https://huggingface.co">>,
        ollama_base_url => <<"https://registry.ollama.ai">>
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
valid(keep_alive_ms, V) -> V =:= infinity orelse (is_integer(V) andalso V >= 0);
valid(data_dir, V) -> V =:= undefined orelse is_path(V);
valid(default_model, V) -> V =:= undefined orelse is_name(V);
valid(hf_token, V) -> V =:= undefined orelse is_name(V);
valid(hf_base_url, V) -> is_url(V);
valid(ollama_base_url, V) -> is_url(V);
valid(models, V) -> is_list(V) andalso lists:all(fun valid_model/1, V);
valid(_, V) -> is_integer(V) andalso V > 0.

%% A model registered at boot: `#{name, path}' plus load options.
valid_model(#{name := N, path := P}) -> is_name(N) andalso is_path(P);
valid_model(_) -> false.

is_name(V) ->
    (is_binary(V) andalso V =/= <<>>) orelse
        (is_list(V) andalso V =/= [] andalso io_lib:printable_unicode_list(V)).
is_path(V) -> is_name(V).
is_url(V) when is_binary(V) -> is_url(binary_to_list(V));
is_url("http://" ++ _) -> true;
is_url("https://" ++ _) -> true;
is_url(_) -> false.

-spec get(atom()) -> term().
get(Key) ->
    application:get_env(erllama_server, Key, maps:get(Key, defaults())).

%% Where the registry and downloaded models live: `data_dir', else the
%% `ERLLAMA_DATA_DIR' environment variable, else the user data
%% directory of the platform (`~/.local/share/erllama',
%% `~/Library/Application Support/erllama'). Never inside the KV cache
%% directories.
-spec data_dir() -> file:filename_all().
data_dir() ->
    case get(data_dir) of
        undefined ->
            case os:getenv("ERLLAMA_DATA_DIR") of
                false -> filename:basedir(user_data, "erllama");
                "" -> filename:basedir(user_data, "erllama");
                Env -> Env
            end;
        Dir ->
            Dir
    end.
