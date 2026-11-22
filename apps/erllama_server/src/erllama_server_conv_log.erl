%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_conv_log).
-moduledoc false.
%% The durable form of a conversation: an append-only log of JSON
%% records, one per line, in `<data_dir>/conversations/<id>.jsonl',
%% apart from the KV cache. Only erllama_server_convs writes them.
%%
%% Records (`t'):
%%   create     id, model, kind (conversation | snapshot), parent, at
%%   msgs       messages appended, and the finish key when it changed
%%   snapshot   the whole transcript and key (a fork's start, a
%%              compaction)
%%   run        begin (run, kind, model, workspace), end, release
%%   tool       intent before a call runs, outcome once it finished
%%   recovered  what a restart repaired
%%
%% Every write is synced before it returns. A torn last line (a crash
%% in the middle of a write) is skipped when the log is read.
%%
%% replay/1 rebuilds a conversation from its records; a run that began
%% and never ended was cut by a stop or a crash, and recover/1 gives
%% the records that repair it.

-export([dir/0, append/2, read/1, ids/0, delete/1]).
-export([replay/1, recover/1]).
-export([create/5, msgs/2, snapshot/2, run/2, tool_intent/4, tool_outcome/4]).

-define(UNRESOLVED, <<
    "the release stopped while this call was running; its outcome is unknown "
    "and it was not run again"
>>).
-define(NOT_RUN, <<"the release stopped before this call ran; it was not run">>).

-spec dir() -> file:filename_all().
dir() ->
    filename:join(erllama_server_config:data_dir(), "conversations").

path(Id) ->
    filename:join(dir(), <<Id/binary, ".jsonl">>).

-spec append(binary(), [map()]) -> ok | {error, term()}.
append(_Id, []) ->
    ok;
append(Id, Records) ->
    P = path(Id),
    ok = filelib:ensure_dir(P),
    Lines = [[json:encode(R), $\n] || R <- Records],
    file:write_file(P, Lines, [append, raw, binary, sync]).

%% The records of a log, oldest first. Lines that do not decode (a torn
%% tail) are skipped.
-spec read(binary()) -> {ok, [map()]} | {error, term()}.
read(Id) ->
    case file:read_file(path(Id)) of
        {ok, Bin} ->
            {ok, [R || L <- binary:split(Bin, <<"\n">>, [global]), L =/= <<>>, R <- decode(L)]};
        {error, _} = E ->
            E
    end.

decode(Line) ->
    try json:decode(Line) of
        #{<<"t">> := _} = R -> [R];
        _ -> []
    catch
        _:_ -> []
    end.

-spec ids() -> [binary()].
ids() ->
    [
        unicode:characters_to_binary(filename:basename(F, ".jsonl"))
     || F <- filelib:wildcard("*.jsonl", dir())
    ].

-spec delete(binary()) -> ok.
delete(Id) ->
    _ = file:delete(path(Id)),
    ok.

%% -----------------------------------------------------------------------------
%% Records
%% -----------------------------------------------------------------------------

-spec create(binary(), term(), conversation | snapshot, binary() | undefined, integer()) -> map().
create(Id, Model, Kind, Parent, At) ->
    #{
        t => create,
        id => Id,
        model => model_out(Model),
        kind => Kind,
        parent => null_if_undefined(Parent),
        at => At
    }.

-spec msgs([map()], binary() | undefined | keep) -> map().
msgs(Msgs, Key) ->
    R = #{t => msgs, msgs => erllama_server_conv_codec:encode_msgs(Msgs), at => now_ms()},
    case Key of
        keep -> R;
        _ -> R#{key => erllama_server_conv_codec:encode_key(Key)}
    end.

-spec snapshot([map()], binary() | undefined) -> map().
snapshot(Msgs, Key) ->
    #{
        t => snapshot,
        msgs => erllama_server_conv_codec:encode_msgs(Msgs),
        key => erllama_server_conv_codec:encode_key(Key),
        at => now_ms()
    }.

%% `Info': `run', `phase' (begin | end | release) and, at begin, `kind',
%% `model' and `workspace'.
-spec run(binary(), map()) -> map().
run(RunId, Info) ->
    maps:merge(
        #{t => run, run => RunId, at => now_ms()},
        maps:map(fun(_, V) -> out(V) end, Info)
    ).

-spec tool_intent(binary(), binary(), binary(), map()) -> map().
tool_intent(RunId, CallId, Name, Args) ->
    #{t => tool, phase => intent, run => RunId, call => CallId, name => Name, args => Args}.

-spec tool_outcome(binary(), binary(), ok | error, binary()) -> map().
tool_outcome(RunId, CallId, Status, Text) ->
    #{t => tool, phase => outcome, run => RunId, call => CallId, status => Status, text => Text}.

out(undefined) -> null;
out(V) when is_atom(V) -> V;
out(V) -> model_out(V).

model_out(V) when is_binary(V) -> V;
model_out(V) when is_atom(V) -> atom_to_binary(V);
model_out(V) -> iolist_to_binary(io_lib:format("~0p", [V])).

null_if_undefined(undefined) -> null;
null_if_undefined(V) -> V.

now_ms() -> erlang:system_time(millisecond).

%% -----------------------------------------------------------------------------
%% Replay and recovery
%% -----------------------------------------------------------------------------

%% A conversation from its records, as erllama_server_convs holds it.
-spec replay([map()]) -> {ok, map()} | {error, no_create}.
replay([#{<<"t">> := <<"create">>} = C | Rest]) ->
    At = maps:get(<<"at">>, C, now_ms()),
    Conv0 = #{
        id => maps:get(<<"id">>, C),
        model => maps:get(<<"model">>, C, null),
        kind => kind(maps:get(<<"kind">>, C, <<"conversation">>)),
        parent => undefined_if_null(maps:get(<<"parent">>, C, null)),
        messages => [],
        last_finish_key => undefined,
        active_run => undefined,
        workspace => undefined,
        unresolved => [],
        created_at => At,
        updated_at => At,
        %% Replay state, dropped once done.
        open_run => undefined,
        tools => #{}
    },
    {ok, lists:foldl(fun apply_record/2, Conv0, Rest)};
replay(_) ->
    {error, no_create}.

apply_record(#{<<"t">> := <<"msgs">>, <<"msgs">> := Ms} = R, C) ->
    touch(R, with_key(R, C#{messages := maps:get(messages, C) ++ decode_msgs(Ms)}));
apply_record(#{<<"t">> := <<"snapshot">>, <<"msgs">> := Ms} = R, C) ->
    touch(R, with_key(R, C#{messages := decode_msgs(Ms)}));
apply_record(#{<<"t">> := <<"run">>, <<"phase">> := <<"begin">>, <<"run">> := Run} = R, C) ->
    C1 =
        case maps:get(<<"model">>, R, null) of
            null -> C;
            M -> C#{model := M}
        end,
    C2 =
        case maps:get(<<"workspace">>, R, null) of
            null -> C1;
            W -> C1#{workspace := W}
        end,
    touch(R, C2#{open_run := Run, tools := #{}});
apply_record(#{<<"t">> := <<"run">>, <<"run">> := Run} = R, #{open_run := Run} = C) ->
    touch(R, C#{open_run := undefined});
apply_record(
    #{<<"t">> := <<"tool">>, <<"call">> := Id, <<"phase">> := P} = R, #{tools := T} = C
) when
    P =:= <<"intent">>; P =:= <<"outcome">>
->
    Prev = maps:get(Id, T, #{}),
    C#{tools := T#{Id => Prev#{phase(P) => R}}};
apply_record(#{<<"t">> := <<"recovered">>, <<"unresolved">> := U}, C) when is_list(U) ->
    C#{unresolved := lists:usort(maps:get(unresolved, C) ++ U)};
apply_record(_Other, C) ->
    C.

phase(<<"intent">>) -> intent;
phase(<<"outcome">>) -> outcome.

with_key(#{<<"key">> := K}, C) -> C#{last_finish_key := erllama_server_conv_codec:decode_key(K)};
with_key(_, C) -> C.

touch(#{<<"at">> := At}, C) when is_integer(At) -> C#{updated_at := At};
touch(_, C) -> C.

decode_msgs(Ms) when is_list(Ms) -> erllama_server_conv_codec:decode_msgs(Ms);
decode_msgs(_) -> [].

kind(<<"snapshot">>) -> snapshot;
kind(_) -> conversation.

undefined_if_null(null) -> undefined;
undefined_if_null(V) -> V.

%% A conversation made consistent, and the records that make the repair
%% durable. Nothing to repair unless a run is open (cut by a stop or a
%% crash, or being released by a session that died): then the last
%% assistant message's calls without a result get one (their recorded
%% outcome, or unresolved, or not run), and the run is released.
%% Unresolved calls are never run again.
-spec recover(map()) -> {map(), [map()]}.
recover(#{open_run := undefined} = C) ->
    {clean(C), []};
recover(#{open_run := Run, messages := Msgs, tools := Tools} = C) ->
    Missing = missing_results(Msgs),
    Results = [result(Call, maps:get(Id, Tools, #{})) || #{id := Id} = Call <- Missing],
    Unresolved = [Id || #{id := Id} = Call <- Missing, state(Call, Tools) =:= unresolved],
    Repairs =
        [msgs(Results, keep) || Results =/= []] ++
            [
                #{t => recovered, run => Run, unresolved => Unresolved, at => now_ms()}
             || Results =/= []
            ] ++
            [run(Run, #{phase => release})],
    C1 = C#{
        messages := Msgs ++ Results,
        unresolved := lists:usort(maps:get(unresolved, C) ++ Unresolved)
    },
    {clean(C1), Repairs}.

clean(C) -> maps:without([open_run, tools], C).

%% The calls of the last assistant message that have no tool result.
missing_results(Msgs) ->
    case lists:splitwith(fun(M) -> not has_calls(M) end, lists:reverse(Msgs)) of
        {After, [#{tool_calls := Calls} | _]} ->
            Answered = [Id || #{role := <<"tool">>, tool_call_id := Id} <- After],
            [Call || #{id := Id} = Call <- Calls, not lists:member(Id, Answered)];
        {_, []} ->
            []
    end.

has_calls(#{role := <<"assistant">>, tool_calls := [_ | _]}) -> true;
has_calls(_) -> false.

state(#{id := Id}, Tools) ->
    case maps:get(Id, Tools, #{}) of
        #{outcome := _} -> outcome;
        #{intent := _} -> unresolved;
        _ -> not_run
    end.

result(#{id := Id} = Call, Recorded) ->
    Name =
        case Call of
            #{function := #{name := N}} -> N;
            _ -> <<>>
        end,
    Content =
        case Recorded of
            #{outcome := #{<<"status">> := <<"ok">>, <<"text">> := T}} -> T;
            #{outcome := #{<<"text">> := T}} -> <<"Error: ", T/binary>>;
            #{intent := _} -> <<"Error: ", ?UNRESOLVED/binary>>;
            _ -> <<"Error: ", ?NOT_RUN/binary>>
        end,
    #{role => <<"tool">>, tool_call_id => Id, name => Name, content => Content}.
