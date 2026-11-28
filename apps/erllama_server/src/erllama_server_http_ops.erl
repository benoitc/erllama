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

-export([health/1, ready/1, rank_ready/1, rank/1, metrics/1, status/1]).

status(_Req) ->
    {ok, S} = erllama_server_api:status(),
    erllama_server_http_kit:json(200, (jsonable(S))#{
        api := 2, groups => erllama_server_group_status:groups()
    }).

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

ready(_Req) ->
    Infra = [
        N
     || {N, P} <- [
            {erllama, erllama_sup},
            {registry, erllama_server_registry},
            {models, erllama_server_models}
        ],
        whereis(P) =:= undefined
    ],
    try erllama_server_group_status:groups() of
        [] ->
            probe(Infra, #{});
        Groups ->
            Failed = [group || lists:any(fun(#{state := S}) -> S =/= ready end, Groups)],
            probe(Infra ++ Failed, #{groups => Groups})
    catch
        exit:_ -> probe(lists:usort([group | Infra]), #{})
    end.

probe([], Extra) ->
    erllama_server_http_kit:json(200, Extra#{status => ok});
probe(Failed, Extra) ->
    erllama_server_http_kit:json(503, Extra#{status => unavailable, failed => Failed}).

rank_ready(_Req) ->
    Rank = local_rank(),
    probe([rank || maps:get(state, Rank) =/= ready], #{rank => Rank}).

rank(_Req) ->
    case local_rank() of
        #{state := unavailable} = Status -> erllama_server_http_kit:json(503, Status);
        Status -> erllama_server_http_kit:json(200, Status)
    end.

local_rank() ->
    case erllama_tp_rank_role:status() of
        undefined -> #{state => unavailable};
        Status -> erllama_server_group_status:rank(Status)
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
        erllama_server_group_status:metrics(erllama_server_group_status:groups()),
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
