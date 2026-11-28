%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
-module(erllama_server_group_boot).
-moduledoc false.
-export([validate/1, coordinators/1, loaders/1, start_link/1, init/1]).

-spec validate([map()]) -> ok | {error, term()}.
validate(Models) ->
    Groups = [G || #{group := G} <- Models],
    %% A configured rank hosts one model. Registry aliases can name it after startup.
    Names = [text(maps:get(coordinator, G)) || G <- Groups],
    Ids = [Id || #{group_id := Id} <- Groups],
    ModelNames = [erllama_server_registry:normalize(N) || #{name := N} <- Models],
    Valid = lists:all(fun(N) -> byte_size(N) =< 255 andalso N =/= <<"undefined">> end, Names),
    case
        Valid andalso unique(Names) andalso unique(Ids) andalso
            (Groups =:= [] orelse unique(ModelNames))
    of
        true -> ok;
        false -> {error, {invalid_config, models, conflicting_group_declarations}}
    end.

unique(L) -> length(L) =:= length(lists:usort(L)).
text(A) when is_atom(A) -> atom_to_binary(A);
text(B) -> B.

-spec coordinators([map()]) -> [supervisor:child_spec()].
coordinators(Models) ->
    [coordinator(G) || #{group := G} <- Models].

coordinator(#{group_id := Id, generation := Gen, world := W, coordinator := C}) ->
    %% Only sys.config can create a registered coordinator name.
    Name = binary_to_atom(text(C)),
    #{
        id => {coordinator, Id},
        restart => temporary,
        start => {erllama_tp_coordinator, start_link, [Name, {Id, Gen}, W, 60000]}
    }.

-spec loaders([map()]) -> [supervisor:child_spec()].
loaders(Models) ->
    [
        #{
            id => {group_load, N},
            restart => temporary,
            start => {?MODULE, start_link, [erllama_server_registry:normalize(N)]}
        }
     || #{group := _, name := N} <- Models
    ].

-spec start_link(binary()) -> {ok, pid()}.
start_link(Name) -> proc_lib:start_link(?MODULE, init, [Name]).

-spec init(binary()) -> ok.
init(Name) ->
    proc_lib:init_ack({ok, self()}),
    load(Name).

load(Name) ->
    case erllama_server_models:load(Name, undefined) of
        {error, group_forming} ->
            timer:sleep(250),
            load(Name);
        {ok, _} ->
            ok;
        {error, Why} ->
            logger:error("Group model ~ts failed to load: ~p", [Name, Why])
    end.
