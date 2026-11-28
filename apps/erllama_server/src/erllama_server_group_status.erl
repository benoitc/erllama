%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
-module(erllama_server_group_status).
-moduledoc false.
-export([groups/0, rank/1, metrics/1]).

-spec groups() -> [map()].
groups() ->
    Config =
        #{
            erllama_server_registry:normalize(N) => #{group => G, status => missing}
         || #{name := N, group := G} <- erllama_server_config:get(models)
        },
    Entries = #{N => E || #{name := N} = E <- erllama_server_registry:grouped()},
    [group(N, E) || N := E <- maps:iterator(maps:merge(Config, Entries), ordered)].

group(Name, #{group := #{world := W} = G} = Entry) ->
    {Ready, Failure} =
        case erllama_server_group:snapshot(G) of
            {ok, R, undefined} -> {R, maps:get(reason, Entry, undefined)};
            {ok, R, Why} -> {R, Why};
            {error, group_forming} -> {#{}, maps:get(reason, Entry, undefined)};
            {error, Why} -> {#{}, Why}
        end,
    Ranks = [rank_state(I, Ready, Failure) || I <- lists:seq(0, W - 1)],
    State =
        case Failure of
            undefined ->
                case
                    map_size(Ready) =:= W andalso maps:get(status, Entry) =:= ok andalso
                        erllama_server_models:group_loaded(Name, G)
                of
                    true -> ready;
                    false -> forming
                end;
            _ ->
                lost
        end,
    (maps:with([group_id, generation, world], G))#{
        model => Name,
        state => State,
        reason => reason(Failure),
        ranks => Ranks,
        n_seq_max => maps:get(0, Ready, null)
    }.

rank_state(I, _, Failure) when Failure =/= undefined ->
    #{rank => I, state => lost, reason => reason(Failure)};
rank_state(I, Ready, _) when is_map_key(I, Ready) ->
    #{rank => I, state => ready, reason => null};
rank_state(I, _, _) ->
    #{rank => I, state => forming, reason => null}.

-spec rank(map()) -> map().
rank(Status) -> Status#{reason => reason(maps:get(reason, Status, undefined))}.

reason(undefined) ->
    null;
reason({incompatible_ranks, Ranks}) ->
    #{code => incompatible_ranks, ranks => Ranks};
reason({tp_rank_lost, R, Why}) ->
    #{code => tp_rank_lost, rank => R, detail => detail(Why)};
reason({tp_rank_lost, {R, Why}}) when is_integer(R) -> reason({tp_rank_lost, R, Why});
reason({Code, Why}) when Code =:= tp_rank_lost; Code =:= coordinator_lost ->
    #{code => Code, detail => detail(Why)};
reason(Why) ->
    #{code => detail(Why)}.

detail(A) when is_atom(A) -> A;
detail(_) -> rank_failed.

-spec metrics([map()]) -> iolist().
metrics(Groups) ->
    [
        <<"# TYPE erllama_server_tp_group_state gauge\n# TYPE erllama_server_tp_lost_ranks gauge\n">>,
        [samples(G) || G <- Groups]
    ].

samples(#{model := M, group_id := Id, generation := Gen, state := State, ranks := Ranks}) ->
    Labels = [
        <<"model=\"">>,
        escape(M),
        <<"\",group_id=\"">>,
        escape(Id),
        <<"\",generation=\"">>,
        integer_to_binary(Gen),
        <<"\"">>
    ],
    [
        [
            [
                <<"erllama_server_tp_group_state{">>,
                Labels,
                <<",state=\"">>,
                atom_to_binary(S),
                <<"\"} ">>,
                case S =:= State of
                    true -> <<"1\n">>;
                    false -> <<"0\n">>
                end
            ]
         || S <- [forming, ready, lost]
        ],
        <<"erllama_server_tp_lost_ranks{">>,
        Labels,
        <<"} ">>,
        integer_to_binary(length([ok || #{state := lost} <- Ranks])),
        <<"\n">>
    ].

escape(B) ->
    Esc = binary:replace(B, <<"\\">>, <<"\\\\">>, [global]),
    Quotes = binary:replace(Esc, <<"\"">>, <<"\\\"">>, [global]),
    binary:replace(Quotes, <<"\n">>, <<"\\n">>, [global]).
