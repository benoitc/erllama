%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_http_ops).
-moduledoc false.
%% Operational endpoints: `/health' (the process is up), `/ready' (the
%% inference engine and the model registry answer) and `/metrics'
%% (Prometheus text: the HTTP metrics Livery records, erllama's
%% counters, and the server's models and jobs).

-export([health/1, ready/1, metrics/1]).

health(Req) ->
    (livery_health:live())(Req).

ready(Req) ->
    Checks = [
        {<<"erllama">>, fun() -> alive(erllama_sup) end},
        {<<"registry">>, fun() -> alive(erllama_server_registry) end},
        {<<"models">>, fun() -> alive(erllama_server_models) end}
    ],
    (livery_health:ready(Checks))(Req).

alive(Name) ->
    case whereis(Name) of
        undefined -> {error, down};
        _ -> ok
    end.

metrics(_Req) ->
    Http = instrument_prometheus:format(),
    livery_resp:text(
        200,
        [{<<"content-type">>, instrument_prometheus:content_type()}],
        [Http, erllama_counters(), server_gauges()]
    ).

erllama_counters() ->
    [
        metric(<<"counter">>, <<"erllama_", (atom_to_binary(K))/binary>>, V)
     || K := V <- erllama:counters(), is_integer(V)
    ].

server_gauges() ->
    Loaded = length(erllama:list_models()),
    Runs = erllama_server_runs:active_count(run),
    Pulls = erllama_server_runs:active_count(pull),
    [
        gauge(<<"erllama_server_models_loaded">>, Loaded),
        gauge(<<"erllama_server_active_runs">>, Runs),
        gauge(<<"erllama_server_active_pulls">>, Pulls)
    ].

gauge(Name, V) ->
    metric(<<"gauge">>, Name, V).

metric(Type, Name, V) ->
    [<<"# TYPE ">>, Name, <<" ">>, Type, <<"\n">>, Name, <<" ">>, integer_to_binary(V), <<"\n">>].
