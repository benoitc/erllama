%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_http).
-moduledoc false.
%% The HTTP listener, served by Livery. Off unless the `http' setting is
%% present (`#{port => 8080}', on 127.0.0.1 unless `ip' says otherwise).
%% Every request goes through one middleware stack (request ids, access
%% log, API keys, body size, concurrency limit, metrics, crash
%% handling) and then to the adapter of its protocol, or to the MCP
%% endpoint when `mcp_endpoint' is on.

-export([start_link/0, port/0, routes/0, middleware/0]).

%% The supervisor's child: the Livery service process, or `ignore'
%% when HTTP is off.
-spec start_link() -> {ok, pid()} | ignore | {error, term()}.
start_link() ->
    case erllama_server_config:get(http) of
        undefined ->
            ignore;
        Http ->
            Listen = #{ip => maps:get(ip, Http, {127, 0, 0, 1}), port => port(Http)},
            Opts = #{
                http => Listen,
                router => livery_router:compile(routes() ++ mcp_routes()),
                middleware => middleware()
            },
            case livery:start_service(Opts) of
                {ok, Pid} ->
                    true = register(?MODULE, Pid),
                    {ok, Pid};
                {error, _} = E ->
                    E
            end
    end.

%% The MCP endpoint, when `mcp_endpoint' turns it on.
mcp_routes() ->
    case erllama_server_config:get(mcp_endpoint) of
        true ->
            ok = erllama_server_mcp_endpoint:register_tools(),
            erllama_server_mcp_endpoint:routes();
        false ->
            []
    end.

%% `ERLLAMA_HTTP_PORT' overrides the configured port.
port(Http) ->
    case string:to_integer(os:getenv("ERLLAMA_HTTP_PORT", "")) of
        {P, []} when P >= 0, P < 65536 -> P;
        _ -> maps:get(port, Http)
    end.

%% The port the listener got (useful with `port => 0').
-spec port() -> inet:port_number() | undefined.
port() ->
    case whereis(?MODULE) of
        undefined ->
            undefined;
        Pid ->
            case livery:which_listeners(Pid) of
                #{h1 := [P | _]} -> P;
                _ -> undefined
            end
    end.

-spec routes() -> list().
routes() ->
    [
        {<<"GET">>, <<"/v1/models">>, {erllama_server_openai, models}},
        {<<"GET">>, <<"/v1/models/*id">>, {erllama_server_openai, model}},
        {<<"POST">>, <<"/v1/chat/completions">>, {erllama_server_openai, chat_completions}},
        {<<"POST">>, <<"/v1/completions">>, {erllama_server_openai, completions}},
        {<<"POST">>, <<"/v1/embeddings">>, {erllama_server_openai, embeddings}},
        {<<"POST">>, <<"/v1/responses">>, {erllama_server_openai_responses, create}},
        {<<"GET">>, <<"/v1/responses/:id">>, {erllama_server_openai_responses, retrieve}},
        {<<"DELETE">>, <<"/v1/responses/:id">>, {erllama_server_openai_responses, delete}},
        {<<"POST">>, <<"/v1/messages">>, {erllama_server_anthropic, messages}},
        {<<"POST">>, <<"/v1/messages/count_tokens">>, {erllama_server_anthropic, count_tokens}},
        {<<"GET">>, <<"/">>, {erllama_server_ollama_models, root}},
        {<<"POST">>, <<"/api/chat">>, {erllama_server_ollama, chat}},
        {<<"POST">>, <<"/api/generate">>, {erllama_server_ollama, generate}},
        {<<"POST">>, <<"/api/embed">>, {erllama_server_ollama, embed}},
        {<<"POST">>, <<"/api/embeddings">>, {erllama_server_ollama, embeddings}},
        {<<"GET">>, <<"/api/tags">>, {erllama_server_ollama_models, tags}},
        {<<"POST">>, <<"/api/show">>, {erllama_server_ollama_models, show}},
        {<<"GET">>, <<"/api/ps">>, {erllama_server_ollama_models, ps}},
        {<<"POST">>, <<"/api/pull">>, {erllama_server_ollama_models, pull}},
        {<<"DELETE">>, <<"/api/delete">>, {erllama_server_ollama_models, delete}},
        {<<"POST">>, <<"/api/copy">>, {erllama_server_ollama_models, copy}},
        {<<"GET">>, <<"/api/version">>, {erllama_server_ollama_models, version}},
        {<<"POST">>, <<"/api/create">>, {erllama_server_ollama_models, not_supported}},
        {<<"POST">>, <<"/api/push">>, {erllama_server_ollama_models, not_supported}},
        {<<"GET">>, <<"/health">>, {erllama_server_http_ops, health}},
        {<<"GET">>, <<"/ready">>, {erllama_server_http_ops, ready}},
        {<<"GET">>, <<"/metrics">>, {erllama_server_http_ops, metrics}},
        {<<"GET">>, <<"/status">>, {erllama_server_http_ops, status}}
    ].

-spec middleware() -> list().
middleware() ->
    Limit = erllama_server_config:get(max_http_requests),
    %% The limiter answers as text; the body is the OpenAI error shape.
    Busy = iolist_to_binary(
        json:encode(#{
            <<"error">> => #{
                <<"message">> => <<"too many concurrent requests; retry later">>,
                <<"type">> => <<"rate_limit_error">>,
                <<"param">> => null,
                <<"code">> => null
            }
        })
    ),
    [
        {livery_request_id, undefined},
        {livery_access_log, #{}},
        {erllama_server_http_auth, [
            unicode:characters_to_binary(K)
         || K <- erllama_server_config:get(api_keys)
        ]},
        {livery_body_limit, #{max => erllama_server_config:get(max_body_bytes)}},
        {livery_concurrency,
            livery_concurrency:limiter(Limit, #{status => 429, body => Busy, retry_after => 1})},
        {livery_instrument_metrics, #{}},
        livery_middleware:wrap(fun erllama_server_http_kit:crash/3)
    ].
