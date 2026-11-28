%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
-module(erllama_server_group).
-moduledoc false.

-export([entry/2, attach/2, init/0, bind/2, down/2, view/1, snapshot/1]).

-spec entry(map(), map()) -> {ok, map()} | {error, invalid_group}.
entry(#{group := G}, Entry) ->
    case G of
        #{group_id := Id, generation := Gen, world := W, coordinator := C, followers := Fs} when
            map_size(G) =:= 5,
            is_binary(Id),
            byte_size(Id) > 0,
            is_integer(Gen),
            Gen >= 0,
            is_integer(W),
            W >= 2,
            is_list(Fs),
            length(Fs) =:= W - 1
        ->
            Names = [name(F) || F <- Fs],
            case
                name(Id) =/= <<>> andalso name(C) =/= <<>> andalso not lists:member(<<>>, Names) andalso
                    length(lists:usort(Names)) =:= length(Names)
            of
                true -> {ok, Entry#{group => G#{coordinator := name(C), followers := Names}}};
                false -> {error, invalid_group}
            end;
        _ ->
            {error, invalid_group}
    end;
entry(_, Entry) ->
    {ok, Entry}.

name(A) when is_atom(A) -> atom_to_binary(A);
name(B) when is_binary(B) ->
    case unicode:characters_to_binary(B) of
        B -> B;
        _ -> <<>>
    end;
name(_) ->
    <<>>.

key(#{group_id := Id, generation := Gen}) -> {Id, Gen}.

-spec init() -> atom().
init() -> ets:new(?MODULE, [named_table, protected, set]).

-spec view(map()) -> map().
view(#{group := Group} = Entry) ->
    case ets:lookup(?MODULE, key(Group)) of
        [{_, {lost, Why}}] -> Entry#{status => lost, reason => Why};
        _ -> Entry
    end;
view(Entry) ->
    Entry.

-spec attach(map(), map()) -> {ok, map()} | {error, term()}.
attach(#{group := G} = Entry, Base) ->
    try discover(G, Entry, Base) of
        {error, {tp_rank_lost, Rank, Reason}} -> lost(G, {Rank, Reason});
        Result -> Result
    catch
        exit:{timeout, _} -> {error, group_timeout};
        exit:{Reason, _} -> lost(G, Reason)
    end.

discover(G, Entry, Base) ->
    maybe
        {ok, Coord} ?= coordinator(maps:get(coordinator, G)),
        {ok, [_ | Followers] = Roles} ?= erllama_tp_coordinator:members(Coord, key(G)),
        true ?= [atom_to_binary(node(R)) || R <- Followers] =:= maps:get(followers, G),
        {ok, Ranks} ?= attachments(Roles, Coord, []),
        configure(Ranks, Entry, Base, Coord)
    else
        false -> {error, incompatible_group_nodes};
        {error, _} = Error -> Error
    end.

attachments([], _, Acc) ->
    {ok, lists:reverse(Acc)};
attachments([Role | Rest], Coord, Acc) ->
    maybe
        {ok, Rank} ?= erllama_tp_rank_role:attachment(Role, Coord),
        attachments(Rest, Coord, [Rank | Acc])
    end.

lost(G, Reason) ->
    ok = gen_server:call(erllama_server_registry, {group_lost, key(G), Reason}),
    {error, {tp_rank_lost, Reason}}.

-spec snapshot(map()) -> {ok, map(), term()} | {error, term()}.
snapshot(G) ->
    maybe
        {ok, Coord} ?= coordinator(name(maps:get(coordinator, G))),
        try
            erllama_tp_coordinator:snapshot(Coord, key(G))
        catch
            exit:_ -> {error, group_forming}
        end
    end.

coordinator(Name) ->
    try whereis(binary_to_existing_atom(Name)) of
        undefined -> {error, group_forming};
        Pid -> {ok, Pid}
    catch
        error:badarg -> {error, group_forming}
    end.

configure([{Worker, P, Desc} | Fs] = Ranks, #{group := G, path := Path} = Entry, Base, Coord) ->
    #{context_opts := C, model_opts := M, world := W} = Desc,
    maybe
        true ?= unicode:characters_to_binary(P) =:= Path,
        true ?= Desc =:= erllama_tp_rank:describe(P, {M, C}, W),
        true ?= lists:all(fun({_, FP, FD}) -> FP =:= P andalso FD =:= Desc end, Fs),
        true ?= matching_options(maps:get(load, Entry, #{}), M, C),
        ok ?=
            gen_server:call(
                erllama_server_registry,
                {bind_group, key(G), [Coord | [Pid || {Pid, _, _} <- Ranks]]}
            ),
        {ok, Base#{
            backend => erllama_model_tp,
            model_opts => M,
            context_opts => C,
            context_size => maps:get(n_ctx, C, 512),
            tp_worker => Worker,
            tp_followers => [{Pid, D} || {Pid, _, D} <- Fs]
        }}
    else
        false -> {error, {incompatible_ranks, [0]}};
        {error, _} = Error -> Error
    end.

matching_options(Load, Model, Context) ->
    Overrides = maps:with([n_ctx, kv_unified], Load),
    C =
        case maps:find(parallel, Load) of
            {ok, N} -> Overrides#{n_seq_max => N};
            error -> Overrides
        end,
    maps:merge(Context, C) =:= Context andalso
        maps:merge(Model, maps:with([n_gpu_layers], Load)) =:= Model.

%% The registry owns these monitors. A lost generation stays lost even if
%% its coordinator name is later registered by another process.
-spec bind(term(), [pid()]) -> ok | {error, term()}.
bind(Key, Pids) ->
    case ets:lookup(?MODULE, Key) of
        [] ->
            Refs = [monitor(process, P, [{tag, {tp_member, Key}}]) || P <- Pids],
            true = ets:insert(?MODULE, {Key, {watch, Pids, Refs}}),
            ok;
        [{_, {watch, Pids, _}}] ->
            ok;
        [{_, {lost, Why}}] ->
            {error, Why};
        _ ->
            {error, wrong_generation}
    end.

-spec down(term(), term()) -> ok.
down(Key, Why) ->
    case ets:lookup(?MODULE, Key) of
        [{_, {watch, _, Refs}}] ->
            [demonitor(Ref, [flush]) || Ref <- Refs],
            mark_lost(Key, Why);
        [{_, {lost, _}}] ->
            ok;
        [] ->
            mark_lost(Key, Why)
    end.

mark_lost(Key, Why) ->
    true = ets:insert(?MODULE, {Key, {lost, {tp_rank_lost, Why}}}),
    ok.
