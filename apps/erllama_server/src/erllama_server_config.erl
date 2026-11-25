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
        max_agents => 16,
        pull_idle_timeout_ms => 60000,
        data_dir => undefined,
        default_model => undefined,
        keep_alive_ms => 300000,
        default_n_ctx => 4096,
        models => [],
        hf_token => undefined,
        hf_base_url => <<"https://huggingface.co">>,
        ollama_base_url => <<"https://registry.ollama.ai">>,
        %% HTTP APIs: off unless `http' is set.
        http => undefined,
        api_keys => [],
        allow_unauthenticated => false,
        max_body_bytes => 33554432,
        max_http_requests => 64,
        %% MCP: servers whose tools agent sessions may use, and the opt-in
        %% endpoint exposing the release to MCP clients.
        mcp_servers => [],
        mcp_timeout_ms => 60000,
        mcp_endpoint => false,
        %% Compaction: above this fraction of the model's context a
        %% transcript is compacted before the next turn (`false' never);
        %% the recent messages kept verbatim; the summary's length.
        compact_threshold => 0.75,
        compact_keep_messages => 6,
        compact_summary_tokens => 512,
        %% Sequences a model serves at once (its `parallel' load option
        %% overrides it), sharing one KV pool of `n_ctx' when `kv_unified'.
        default_parallel => 4,
        kv_unified => true
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
        [] -> check_http(maps:merge(defaults(), Env));
        [{K, V} | _] -> {error, {invalid_config, K, V}}
    end.

%% An HTTP listener reachable from other machines needs API keys, unless
%% the operator says otherwise.
check_http(#{http := #{ip := Ip}, api_keys := [], allow_unauthenticated := false}) when
    Ip =/= {127, 0, 0, 1}, Ip =/= {0, 0, 0, 0, 0, 0, 0, 1}
->
    {error, {invalid_config, http, {no_api_keys_on_public_address, Ip}}};
check_http(_) ->
    ok.

valid(run_retention_ms, V) ->
    is_integer(V) andalso V >= 0;
valid(keep_alive_ms, V) ->
    V =:= infinity orelse (is_integer(V) andalso V >= 0);
valid(data_dir, V) ->
    V =:= undefined orelse is_path(V);
valid(default_model, V) ->
    V =:= undefined orelse is_name(V);
valid(hf_token, V) ->
    V =:= undefined orelse is_name(V);
valid(hf_base_url, V) ->
    is_url(V);
valid(ollama_base_url, V) ->
    is_url(V);
valid(models, V) ->
    is_list(V) andalso lists:all(fun valid_model/1, V);
valid(http, V) ->
    valid_http(V);
valid(api_keys, V) ->
    is_list(V) andalso lists:all(fun is_name/1, V);
valid(K, V) when K =:= allow_unauthenticated; K =:= mcp_endpoint; K =:= kv_unified ->
    is_boolean(V);
valid(mcp_servers, V) ->
    is_list(V) andalso lists:all(fun valid_mcp/1, V) andalso unique_names(V);
valid(compact_threshold, V) ->
    valid_fraction(V);
valid(default_parallel, V) ->
    is_integer(V) andalso V >= 1 andalso V =< 64;
valid(_, V) ->
    is_integer(V) andalso V > 0.

%% A fraction of the context, or `false' for never.
valid_fraction(false) -> true;
valid_fraction(V) -> is_number(V) andalso V > 0 andalso V =< 1.

valid_http(undefined) ->
    true;
valid_http(#{port := P} = H) ->
    is_integer(P) andalso P >= 0 andalso P < 65536 andalso
        valid_ip(maps:get(ip, H, {127, 0, 0, 1}));
valid_http(_) ->
    false.

%% An MCP server: a name (letters, digits, `_', `-') and a transport,
%% with every optional field checked too (and no unknown key): a bad
%% value is refused at startup, never found by a crash later.
valid_mcp(#{name := N, transport := stdio, command := C} = S) ->
    only_keys(S, [name, transport, command, args, env, policy]) andalso
        mcp_name(N) andalso is_path(C) andalso mcp_policy(S) andalso
        optional(S, args, fun(L) -> is_list(L) andalso lists:all(fun is_text/1, L) end) andalso
        optional(S, env, fun(L) -> is_list(L) andalso lists:all(fun env_pair/1, L) end);
valid_mcp(#{name := N, transport := http, url := U} = S) ->
    only_keys(S, [name, transport, url, token, headers, policy]) andalso
        mcp_name(N) andalso is_url(U) andalso mcp_policy(S) andalso
        optional(S, token, fun is_name/1) andalso
        optional(S, headers, fun(L) -> is_list(L) andalso lists:all(fun header/1, L) end);
valid_mcp(_) ->
    false.

only_keys(M, Keys) -> maps:keys(maps:without(Keys, M)) =:= [].

optional(M, K, Check) ->
    case M of
        #{K := V} -> Check(V);
        _ -> true
    end.

is_text(V) -> is_binary(V) orelse (is_list(V) andalso io_lib:printable_unicode_list(V)).

env_pair({K, false}) -> is_name(K);
env_pair({K, V}) -> is_name(K) andalso is_text(V);
env_pair(_) -> false.

header({K, V}) -> is_name(K) andalso is_text(V);
header(_) -> false.

mcp_name(N) ->
    is_name(N) andalso re:run(unicode:characters_to_binary(N), <<"^[A-Za-z0-9_-]+$">>) =/= nomatch.

mcp_policy(S) -> lists:member(maps:get(policy, S, ask), [auto, ask, deny]).

unique_names(Servers) ->
    Names = [unicode:characters_to_binary(N) || #{name := N} <- Servers],
    length(Names) =:= length(lists:usort(Names)).

%% A model registered at boot: `#{name, path}' plus load options.
valid_model(#{name := N, path := P}) -> is_name(N) andalso is_path(P);
valid_model(_) -> false.

valid_ip(Ip) when is_tuple(Ip) -> tuple_size(Ip) =:= 4 orelse tuple_size(Ip) =:= 8;
valid_ip(_) -> false.

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
