%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_http_ops).
-moduledoc false.
%% Operational endpoints: `/health' (the process is up), `/ready' (the
%% inference engine and the model registry answer), `/metrics'
%% (Prometheus text: the HTTP metrics Livery records, erllama's
%% counters, and the server's models, jobs and failures) and `/status'
%% (erllama_server_api:status/0 as JSON: what runs, waits and failed).

-export([health/1, ready/1, metrics/1, status/1]).

status(_Req) ->
    {ok, S} = erllama_server_api:status(),
    erllama_server_http_kit:json(200, jsonable(S)).

%% Status values include terms JSON has no form for (error reasons,
%% pids): they are shown as Erlang text.
jsonable(M) when is_map(M) ->
    #{key(K) => jsonable(V) || K := V <- M};
jsonable(L) when is_list(L) ->
    case io_lib:printable_unicode_list(L) of
        true when L =/= [] -> unicode:characters_to_binary(L);
        _ -> [jsonable(X) || X <- L]
    end;
jsonable(V) when is_binary(V); is_number(V); is_atom(V) ->
    V;
jsonable(V) ->
    iolist_to_binary(io_lib:format("~0tp", [V])).

key(K) when is_atom(K) -> K;
key(K) when is_binary(K) -> K;
key(K) -> iolist_to_binary(io_lib:format("~0tp", [K])).

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
    Waiting = length([
        R
     || #{kind := run, status := running, phase := waiting} = R <- erllama_server_runs:list()
    ]),
    [
        gauge(<<"erllama_server_models_loaded">>, Loaded),
        gauge(<<"erllama_server_active_runs">>, Runs),
        gauge(<<"erllama_server_waiting_runs">>, Waiting),
        gauge(<<"erllama_server_active_pulls">>, Pulls),
        failure_counters()
    ].

%% One counter per failure kind, labelled.
failure_counters() ->
    case erllama_server_failures:counts() of
        C when map_size(C) =:= 0 ->
            [];
        C ->
            [
                <<"# TYPE erllama_server_failures_total counter\n">>,
                [
                    [
                        <<"erllama_server_failures_total{kind=\"">>,
                        atom_to_binary(K),
                        <<"\"} ">>,
                        integer_to_binary(N),
                        <<"\n">>
                    ]
                 || K := N <- maps:iterator(C, ordered)
                ]
            ]
    end.

gauge(Name, V) ->
    metric(<<"gauge">>, Name, V).

metric(Type, Name, V) ->
    [<<"# TYPE ">>, Name, <<" ">>, Type, <<"\n">>, Name, <<" ">>, integer_to_binary(V), <<"\n">>].
