%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_registry).
-moduledoc false.
%% The models the release knows, persisted in `<data_dir>/registry.json'.
%% An entry names a GGUF file (registered from disk or downloaded into
%% `<data_dir>/blobs'), its optional multimodal projector, how to load
%% it and the generation defaults its source gave. Names are
%% `family:tag'; a name given without a tag means `:latest'. Aliases
%% point at a name; `default_model' answers when no model is named.
%%
%% Every change is written to a temporary file, synced and renamed over
%% the registry, so a crash leaves the old or the new file, never half
%% of one. A file that cannot be read is moved aside and the release
%% starts with an empty registry. Entries whose file has gone are kept
%% and reported as `missing'.

-behaviour(gen_server).

-export([
    start_link/0,
    register_local/2,
    add/1,
    resolve/1,
    list/0,
    alias/2,
    copy/2,
    remove/1,
    set_default/1,
    normalize/1,
    blobs_dir/0
]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-define(VERSION, 1).
%% How to load the model. `backend' and `step_delay_ms' exist for tests:
%% they load the model on erllama's stub backend instead of llama.cpp.
-define(LOAD_KEYS, [
    n_ctx, n_gpu_layers, parallel, kv_unified, chat_template, backend, step_delay_ms
]).

-record(state, {
    file :: file:filename_all(),
    entries = #{} :: #{binary() => map()},
    aliases = #{} :: #{binary() => binary()},
    default :: binary() | undefined
}).
-type state() :: #state{}.

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

%% Register a GGUF file already on this machine. `Opts': `name'
%% (default: the file name), `mmproj', `n_ctx', `n_gpu_layers',
%% `chat_template', `keep_alive'.
-spec register_local(file:filename_all(), map()) -> {ok, map()} | {error, term()}.
register_local(Path, Opts) ->
    gen_server:call(?MODULE, {register_local, Path, Opts}, 30000).

%% Add or replace a complete entry (a finished download).
-spec add(map()) -> {ok, map()} | {error, term()}.
add(Entry) ->
    gen_server:call(?MODULE, {add, Entry}, 30000).

%% A name, alias or (with `undefined') the default model.
-spec resolve(binary() | undefined) -> {ok, map()} | {error, term()}.
resolve(Name) ->
    gen_server:call(?MODULE, {resolve, Name}).

-spec list() -> [map()].
list() ->
    gen_server:call(?MODULE, list).

-spec alias(binary(), binary()) -> ok | {error, term()}.
alias(Alias, Name) ->
    gen_server:call(?MODULE, {alias, Alias, Name}).

%% A second name for the same file (Ollama `copy').
-spec copy(binary(), binary()) -> {ok, map()} | {error, term()}.
copy(Src, Dst) ->
    gen_server:call(?MODULE, {copy, Src, Dst}).

%% Forget a model. A downloaded blob that no other entry uses is
%% deleted; a file registered from disk is never deleted.
-spec remove(binary()) -> ok | {error, term()}.
remove(Name) ->
    gen_server:call(?MODULE, {remove, Name}, 30000).

-spec set_default(binary()) -> ok | {error, term()}.
set_default(Name) ->
    gen_server:call(?MODULE, {set_default, Name}).

-spec normalize(binary() | string()) -> binary().
normalize(Name0) ->
    Name = string:lowercase(unicode:characters_to_binary(Name0)),
    case binary:match(Name, <<":">>) of
        nomatch -> <<Name/binary, ":latest">>;
        _ -> Name
    end.

-spec blobs_dir() -> file:filename_all().
blobs_dir() ->
    filename:join(erllama_server_config:data_dir(), "blobs").

%% -----------------------------------------------------------------------------

-spec init([]) -> {ok, state()}.
init([]) ->
    Dir = erllama_server_config:data_dir(),
    ok = filelib:ensure_path(filename:join(Dir, "blobs")),
    File = filename:join(Dir, "registry.json"),
    St0 = load(File),
    St = boot_models(erllama_server_config:get(models), St0),
    {ok, St}.

handle_call({register_local, Path, Opts}, _From, St) ->
    reply_change(local_entry(Path, Opts), St);
handle_call({add, Entry}, _From, St) ->
    reply_change({ok, Entry#{name := normalize(maps:get(name, Entry))}}, St);
handle_call({resolve, Name}, _From, St) ->
    {reply, find(Name, St), St};
handle_call(list, _From, #state{entries = E, default = D} = St) ->
    {reply, [view(X, D) || _ := X <- maps:iterator(E, ordered)], St};
handle_call({alias, Alias0, Name}, _From, St) ->
    Alias = string:lowercase(unicode:characters_to_binary(Alias0)),
    case find(Name, St) of
        {ok, #{name := Real}} ->
            St1 = St#state{aliases = (St#state.aliases)#{Alias => Real}},
            {reply, save(St1), St1};
        {error, _} = E ->
            {reply, E, St}
    end;
handle_call({copy, Src, Dst}, _From, St) ->
    case find(Src, St) of
        {ok, Entry} ->
            reply_change({ok, Entry#{name := normalize(Dst), created_at := now_ms()}}, St);
        {error, _} = E ->
            {reply, E, St}
    end;
handle_call({remove, Name}, _From, St) ->
    case find(Name, St) of
        {ok, #{name := Real} = Entry} ->
            Entries = maps:remove(Real, St#state.entries),
            Aliases = maps:filter(fun(_, N) -> N =/= Real end, St#state.aliases),
            Default =
                case St#state.default of
                    Real -> undefined;
                    D -> D
                end,
            St1 = St#state{entries = Entries, aliases = Aliases, default = Default},
            ok = delete_unused_blobs(Entry, Entries),
            {reply, save(St1), St1};
        {error, _} = E ->
            {reply, E, St}
    end;
handle_call({set_default, Name}, _From, St) ->
    case find(Name, St) of
        {ok, #{name := Real}} ->
            St1 = St#state{default = Real},
            {reply, save(St1), St1};
        {error, _} = E ->
            {reply, E, St}
    end;
handle_call(_Req, _From, St) ->
    {reply, {error, unknown_call}, St}.

handle_cast(_Msg, St) ->
    {noreply, St}.

handle_info(_Msg, St) ->
    {noreply, St}.

%% -----------------------------------------------------------------------------
%% Entries
%% -----------------------------------------------------------------------------

reply_change({ok, #{name := Name} = Entry}, St) ->
    St1 = St#state{entries = (St#state.entries)#{Name => Entry}},
    case save(St1) of
        ok -> {reply, {ok, view(Entry, St1#state.default)}, St1};
        {error, _} = E -> {reply, E, St}
    end;
reply_change({error, _} = E, St) ->
    {reply, E, St}.

local_entry(Path0, Opts) ->
    Path = unicode:characters_to_binary(filename:absname(Path0)),
    case erllama_server_gguf:read(Path) of
        {ok, #{kv := _}} ->
            Name = normalize(maps:get(name, Opts, filename:basename(Path, <<".gguf">>))),
            MmProj = maps:get(mmproj, Opts, undefined),
            case check_mmproj(MmProj) of
                ok ->
                    {ok, #{
                        name => Name,
                        source => #{kind => local, ref => Path},
                        path => Path,
                        sha256 => undefined,
                        size => filelib:file_size(Path),
                        mmproj => abs_or_undefined(MmProj),
                        load => load_opts(Opts),
                        defaults => #{},
                        keep_alive => maps:get(keep_alive, Opts, undefined),
                        created_at => now_ms()
                    }};
                {error, _} = E ->
                    E
            end;
        {error, enoent} ->
            {error, {enoent, Path}};
        {error, Reason} ->
            {error, {not_a_gguf, Path, Reason}}
    end.

check_mmproj(undefined) ->
    ok;
check_mmproj(Path) ->
    case erllama_server_gguf:read(Path) of
        {ok, H} ->
            case erllama_server_gguf:summary(H) of
                #{projector := #{}} -> ok;
                _ -> {error, {not_a_projector, Path}}
            end;
        {error, Reason} ->
            {error, {mmproj, Path, Reason}}
    end.

abs_or_undefined(undefined) -> undefined;
abs_or_undefined(P) -> unicode:characters_to_binary(filename:absname(P)).

find(undefined, #state{default = undefined} = St) ->
    case erllama_server_config:get(default_model) of
        undefined -> {error, no_default_model};
        Name -> find(unicode:characters_to_binary(Name), St#state{default = <<>>})
    end;
find(undefined, #state{default = D} = St) ->
    find(D, St);
find(Name0, #state{entries = E, aliases = A, default = D}) ->
    Name = string:lowercase(unicode:characters_to_binary(Name0)),
    Lookup = [Name, maps:get(Name, A, none), normalize(Name), maps:get(normalize(Name), A, none)],
    case [maps:get(N, E) || N <- Lookup, is_map_key(N, E)] of
        [Entry | _] -> {ok, view(Entry, D)};
        [] -> {error, {unknown_model, Name0}}
    end.

%% An entry as callers see it: plus its status, aliases are not part of
%% the entry itself.
view(#{path := Path} = Entry, Default) ->
    Status =
        case filelib:is_regular(Path) of
            true -> ok;
            false -> missing
        end,
    Entry#{status => Status, default => maps:get(name, Entry) =:= Default}.

delete_unused_blobs(Entry, Remaining) ->
    Blobs = unicode:characters_to_binary(filename:absname(blobs_dir())),
    Used = lists:usort(
        lists:append([[maps:get(path, R), maps:get(mmproj, R, undefined)] || _ := R <- Remaining])
    ),
    _ = [
        file:delete(P)
     || P <- [maps:get(path, Entry), maps:get(mmproj, Entry, undefined)],
        is_binary(P),
        string:prefix(P, Blobs) =/= nomatch,
        not lists:member(P, Used)
    ],
    ok.

boot_models([], St) ->
    St;
boot_models([#{name := N, path := P} = M | Rest], #state{entries = E} = St) ->
    Name = normalize(N),
    case is_map_key(Name, E) of
        true ->
            boot_models(Rest, St);
        false ->
            case local_entry(P, maps:put(name, Name, M)) of
                {ok, Entry} ->
                    boot_models(Rest, St#state{entries = E#{Name => Entry}});
                {error, Reason} ->
                    logger:warning("erllama_server: model ~ts not registered: ~p", [Name, Reason]),
                    boot_models(Rest, St)
            end
    end.

now_ms() -> erlang:system_time(millisecond).

load_opts(Opts) ->
    maps:filtermap(
        fun
            (backend, B) when B =:= erllama_model_stub; B =:= <<"erllama_model_stub">> ->
                {true, erllama_model_stub};
            (backend, _) ->
                false;
            (parallel, P) ->
                is_integer(P) andalso P >= 1 andalso P =< 64;
            (kv_unified, U) ->
                is_boolean(U);
            (_, _) ->
                true
        end,
        maps:with(?LOAD_KEYS, Opts)
    ).

%% -----------------------------------------------------------------------------
%% Persistence
%% -----------------------------------------------------------------------------

load(File) ->
    case file:read_file(File) of
        {ok, Bin} ->
            try decode(json:decode(Bin)) of
                {Entries, Aliases, Default} ->
                    #state{file = File, entries = Entries, aliases = Aliases, default = Default}
            catch
                _:Reason ->
                    Aside = <<
                        (unicode:characters_to_binary(File))/binary,
                        ".corrupt-",
                        (integer_to_binary(now_ms()))/binary
                    >>,
                    _ = file:rename(File, Aside),
                    logger:error(
                        "erllama_server: registry ~ts unreadable (~p); moved to ~ts, starting empty",
                        [
                            File, Reason, Aside
                        ]
                    ),
                    #state{file = File}
            end;
        {error, enoent} ->
            #state{file = File}
    end.

save(#state{file = File, entries = E, aliases = A, default = D}) ->
    Doc = #{
        <<"version">> => ?VERSION,
        <<"default">> => null_or(D),
        <<"aliases">> => A,
        <<"models">> => [encode_entry(X) || _ := X <- maps:iterator(E, ordered)]
    },
    Tmp = <<(unicode:characters_to_binary(File))/binary, ".tmp">>,
    case file:open(Tmp, [write, binary, raw]) of
        {ok, Fd} ->
            Result =
                case file:write(Fd, json:encode(Doc)) of
                    ok -> file:sync(Fd);
                    {error, _} = E1 -> E1
                end,
            ok = file:close(Fd),
            case Result of
                ok -> file:rename(Tmp, File);
                {error, _} = E2 -> E2
            end;
        {error, _} = E3 ->
            E3
    end.

null_or(undefined) -> null;
null_or(V) -> V.

encode_entry(E) ->
    #{
        <<"name">> => maps:get(name, E),
        <<"source">> => #{
            <<"kind">> => maps:get(kind, maps:get(source, E)),
            <<"ref">> => maps:get(ref, maps:get(source, E))
        },
        <<"path">> => maps:get(path, E),
        <<"sha256">> => null_or(maps:get(sha256, E, undefined)),
        <<"size">> => maps:get(size, E, 0),
        <<"mmproj">> => null_or(maps:get(mmproj, E, undefined)),
        <<"load">> => maps:get(load, E, #{}),
        <<"defaults">> => maps:get(defaults, E, #{}),
        <<"keep_alive">> => null_or(maps:get(keep_alive, E, undefined)),
        <<"created_at">> => maps:get(created_at, E, 0)
    }.

decode(#{<<"version">> := ?VERSION, <<"models">> := Models} = Doc) ->
    Entries = maps:from_list([{maps:get(name, X), X} || X <- [decode_entry(M) || M <- Models]]),
    Aliases = maps:get(<<"aliases">>, Doc, #{}),
    Default = undefined_or(maps:get(<<"default">>, Doc, null)),
    {Entries, Aliases, Default}.

decode_entry(
    #{<<"name">> := N, <<"path">> := P, <<"source">> := #{<<"kind">> := K, <<"ref">> := R}} = M
) ->
    #{
        name => N,
        source => #{kind => source_kind(K), ref => R},
        path => P,
        sha256 => undefined_or(maps:get(<<"sha256">>, M, null)),
        size => maps:get(<<"size">>, M, 0),
        mmproj => undefined_or(maps:get(<<"mmproj">>, M, null)),
        load => load_opts(atom_keys(maps:get(<<"load">>, M, #{}), ?LOAD_KEYS)),
        defaults => atom_keys(
            maps:get(<<"defaults">>, M, #{}), erllama_server_generate:option_keys()
        ),
        keep_alive => keep_alive(maps:get(<<"keep_alive">>, M, null)),
        created_at => maps:get(<<"created_at">>, M, 0)
    }.

source_kind(<<"local">>) -> local;
source_kind(<<"hf">>) -> hf;
source_kind(<<"https">>) -> https;
source_kind(<<"ollama">>) -> ollama.

keep_alive(null) -> undefined;
keep_alive(<<"infinity">>) -> infinity;
keep_alive(N) when is_integer(N) -> N.

undefined_or(null) -> undefined;
undefined_or(V) -> V.

%% Only known option names become atoms.
atom_keys(Map, Known) ->
    maps:from_list([{K, V} || K <- Known, {ok, V} <- [maps:find(atom_to_binary(K), Map)]]).
