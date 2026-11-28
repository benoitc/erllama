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
    grouped/0,
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
    n_ctx,
    n_gpu_layers,
    parallel,
    kv_unified,
    early_stream_token,
    chat_template,
    backend,
    step_delay_ms
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

-spec grouped() -> [map()].
grouped() -> gen_server:call(?MODULE, grouped).

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
    _ = erllama_server_group:init(),
    Dir = erllama_server_config:data_dir(),
    ok = filelib:ensure_path(filename:join(Dir, "blobs")),
    File = filename:join(Dir, "registry.json"),
    St0 = load(File),
    St = boot_models(erllama_server_config:get(models), St0),
    {ok, St}.

handle_call({group_lost, Key, Why}, _, St) ->
    {reply, erllama_server_group:down(Key, Why), St};
handle_call({bind_group, Key, Pids}, _, St) ->
    {reply, erllama_server_group:bind(Key, Pids), St};
handle_call({register_local, Path, Opts}, _From, St) ->
    reply_change(local_entry(Path, Opts), St);
handle_call({add, #{shard_digests := [_, _ | _] = Digests, name := Name} = Entry}, _From, St) ->
    Result =
        case erllama_server_shards:link_set(Digests) of
            {ok, Path, Sha} -> {ok, Entry#{path => Path, sha256 => Sha, name := normalize(Name)}};
            {error, _} = E -> E
        end,
    reply_change(Result, St);
handle_call({add, Entry}, _From, St) ->
    reply_change({ok, Entry#{name := normalize(maps:get(name, Entry))}}, St);
handle_call({resolve, Name}, _From, St) ->
    {reply, find(Name, St), St};
handle_call(grouped, _, #state{entries = E, default = D} = St) ->
    {reply, [view(X, D) || _ := #{group := _} = X <- maps:iterator(E, ordered)], St};
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
            case save(St1) of
                ok ->
                    ok = delete_unused_blobs(Entry, Entries),
                    {reply, ok, St1};
                {error, _} = E ->
                    {reply, E, St}
            end;
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

handle_info({{tp_member, Key}, _, process, _, Reason}, St) ->
    ok = erllama_server_group:down(Key, Reason),
    {noreply, St};
handle_info(_Msg, St) ->
    {noreply, St}.

%% -----------------------------------------------------------------------------
%% Entries
%% -----------------------------------------------------------------------------

reply_change({ok, Entry}, St) ->
    store_entry(
        erllama_server_group:entry(Entry, maps:without([status, reason, default], Entry)), St
    );
reply_change({error, _} = E, St) ->
    {reply, E, St}.

store_entry({ok, #{name := Name} = Entry}, St) ->
    St1 = St#state{entries = (St#state.entries)#{Name => Entry}},
    case save(St1) of
        ok ->
            case maps:find(Name, St#state.entries) of
                {ok, Old} -> delete_unused_blobs(Old, St1#state.entries);
                error -> ok
            end,
            {reply, {ok, view(Entry, St1#state.default)}, St1};
        {error, _} = E ->
            {reply, E, St}
    end;
store_entry({error, _} = E, St) ->
    {reply, E, St}.

local_entry(Path0, Opts) ->
    Path = unicode:characters_to_binary(filename:absname(Path0)),
    case local_format(Path, load_opts(Opts)) of
        {ok, First, Load, Size} ->
            Base =
                case erllama_gguf_shards:parts(First) of
                    {Prefix, _, _} -> filename:basename(Prefix);
                    single -> filename:basename(First, <<".gguf">>)
                end,
            Name = normalize(maps:get(name, Opts, Base)),
            MmProj = maps:get(mmproj, Opts, undefined),
            case check_projector(Load, MmProj) of
                ok ->
                    erllama_server_group:entry(Opts, #{
                        name => Name,
                        source => #{kind => local, ref => Path},
                        path => First,
                        sha256 => undefined,
                        size => Size,
                        mmproj => abs_or_undefined(MmProj),
                        load => Load,
                        defaults => #{},
                        keep_alive => maps:get(keep_alive, Opts, undefined),
                        created_at => now_ms()
                    });
                {error, _} = E ->
                    E
            end;
        Error ->
            Error
    end.

local_format(Path, Load) ->
    case filelib:is_dir(Path) orelse maps:get(backend, Load, undefined) =:= erllama_model_mlx of
        true ->
            Files = filelib:wildcard(filename:join(binary_to_list(Path), "*.safetensors")),
            Required = ["config.json", "tokenizer.json", "tokenizer_config.json"],
            case
                Files =/= [] andalso
                    lists:all(
                        fun(F) ->
                            filelib:is_regular(filename:join(Path, F))
                        end,
                        Required
                    )
            of
                true ->
                    {ok, Path, Load#{backend => erllama_model_mlx},
                        lists:sum([filelib:file_size(F) || F <- Files])};
                false ->
                    {error, {invalid_mlx_checkpoint, Path}}
            end;
        false ->
            local_gguf(erllama_gguf_shards:paths(Path), Load)
    end.

local_gguf({ok, [First | _] = Paths}, Load) ->
    case erllama_server_shards:inspect(Paths) of
        {ok, Size} -> {ok, First, Load, Size};
        {error, _} = E -> E
    end;
local_gguf({error, _} = E, _) ->
    E.

check_projector(#{backend := erllama_model_mlx}, P) when P =/= undefined ->
    {error, {mlx_unsupported_option, mmproj}};
check_projector(_, P) ->
    check_mmproj(P).

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
        case entry_present(Entry, Path) of
            true -> ok;
            false -> missing
        end,
    erllama_server_group:view(Entry#{status => Status, default => maps:get(name, Entry) =:= Default}).

entry_present(#{load := #{backend := erllama_model_mlx}}, Path) ->
    filelib:is_regular(filename:join(Path, "config.json"));
entry_present(_, Path) ->
    lists:all(fun filelib:is_regular/1, model_paths(Path)).

model_paths(Path) ->
    case erllama_gguf_shards:paths(Path) of
        {ok, Paths} -> Paths;
        {error, _} -> [Path]
    end.

entry_files(Entry) ->
    model_paths(maps:get(path, Entry)) ++ [maps:get(mmproj, Entry, undefined)] ++
        [erllama_server_shards:blob(H) || H <- maps:get(shard_digests, Entry, [])].

delete_unused_blobs(#{source := #{kind := local}}, _) ->
    ok;
delete_unused_blobs(Entry, Remaining) ->
    Blobs = unicode:characters_to_binary(filename:absname(blobs_dir())),
    Used = lists:usort(
        lists:append([entry_files(R) || _ := R <- Remaining])
    ),
    _ = [
        file:delete(P)
     || P <- entry_files(Entry),
        is_binary(P),
        filename:dirname(P) =:= Blobs,
        not lists:member(P, Used)
    ],
    ok.

boot_models([], St) ->
    St;
boot_models([#{name := N, path := P, group := _} = M | Rest], #state{entries = Entries} = St) ->
    case local_entry(P, M) of
        {ok, Entry} -> boot_models(Rest, St#state{entries = Entries#{normalize(N) => Entry}});
        {error, Why} -> error({invalid_group_model, N, Why})
    end;
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
            (backend, B) when B =:= erllama_model_mlx; B =:= <<"erllama_model_mlx">> ->
                {true, erllama_model_mlx};
            (backend, _) ->
                false;
            (parallel, P) ->
                is_integer(P) andalso P >= 1 andalso P =< 64;
            (K, U) when K =:= kv_unified; K =:= early_stream_token ->
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
    (maps:with([shard_digests, group], E))#{
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
    Extra =
        case maps:find(<<"group">>, M) of
            {ok, Raw} ->
                Group = atom_keys(Raw, [group_id, generation, world, coordinator, followers]),
                {ok, Part} = erllama_server_group:entry(#{group => Group}, #{}),
                Part;
            error ->
                #{}
        end,
    (maps:merge(Extra, atom_keys(M, [shard_digests])))#{
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
