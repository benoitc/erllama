%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_source).
-moduledoc false.
%% Model sources a pull accepts, and what to fetch for each:
%%
%%   hf://ORG/REPO/PATH.gguf    one file of a Hugging Face repository
%%   hf://ORG/REPO[:QUANT]      the repository's GGUF for QUANT (default
%%                              Q4_K_M, else the only one), plus its
%%                              mmproj projector when it ships one
%%   https://HOST/PATH.gguf     a plain URL (checked with `sha256' if given)
%%   ollama://[NS/]NAME[:TAG]   an Ollama registry model (also written
%%                              without the scheme)
%%
%% `parse/1' only reads the string; `plan/3' asks the registries what
%% the files are, their sizes and their sha256.

-export([parse/1, plan/3, origin/1, resolve/2, hf_headers/0]).

-define(DEFAULT_QUANT, <<"q4_k_m">>).
-define(OLLAMA_MANIFEST, <<"application/vnd.docker.distribution.manifest.v2+json">>).

-type source() :: #{kind := hf | https | ollama, ref := binary(), _ => _}.
-type file() :: #{
    role := model | projector,
    url := binary(),
    sha256 := binary() | undefined,
    size := non_neg_integer() | undefined,
    auth := boolean()
}.
-export_type([source/0, file/0]).

-spec parse(binary() | string()) -> {ok, source()} | {error, {invalid_source, binary()}}.
parse(Ref0) ->
    Ref = string:trim(unicode:characters_to_binary(Ref0)),
    case Ref of
        <<"hf://", Rest/binary>> -> parse_hf(Rest, Ref);
        <<"https://", _/binary>> -> {ok, #{kind => https, ref => Ref, url => Ref}};
        <<"http://", _/binary>> -> {ok, #{kind => https, ref => Ref, url => Ref}};
        <<"ollama://", Rest/binary>> -> parse_ollama(Rest, Ref);
        _ -> parse_ollama(Ref, <<"ollama://", Ref/binary>>)
    end.

parse_hf(Rest, Ref) ->
    case binary:split(Rest, <<"/">>, [global]) of
        [Org, Repo | Path] when Org =/= <<>>, Repo =/= <<>> ->
            case hf_target(Repo, Path) of
                {ok, Name, Target} ->
                    {ok, Target#{kind => hf, ref => Ref, repo => <<Org/binary, "/", Name/binary>>}};
                error ->
                    {error, {invalid_source, Ref}}
            end;
        _ ->
            {error, {invalid_source, Ref}}
    end.

%% `REPO:QUANT', `REPO' or `REPO/PATH.gguf'.
hf_target(Repo, []) ->
    case binary:split(Repo, <<":">>) of
        [R, Quant] when Quant =/= <<>> -> {ok, R, #{quant => string:lowercase(Quant)}};
        [R] -> {ok, R, #{quant => undefined}};
        _ -> error
    end;
hf_target(Repo, Path) ->
    File = iolist_to_binary(lists:join(<<"/">>, Path)),
    case filename:extension(File) of
        <<".gguf">> -> {ok, Repo, #{file => File}};
        _ -> error
    end.

parse_ollama(Rest, Ref) ->
    {Path, Tag} =
        case binary:split(Rest, <<":">>) of
            [P, T] when T =/= <<>> -> {P, T};
            [P] -> {P, <<"latest">>};
            _ -> {<<>>, <<>>}
        end,
    case binary:split(Path, <<"/">>, [global]) of
        [Name] when Name =/= <<>> -> {ok, ollama(Ref, <<"library">>, Name, Tag)};
        [Ns, Name] when Ns =/= <<>>, Name =/= <<>> -> {ok, ollama(Ref, Ns, Name, Tag)};
        _ -> {error, {invalid_source, Ref}}
    end.

ollama(Ref, Ns, Name, Tag) ->
    #{
        kind => ollama,
        ref => Ref,
        namespace => string:lowercase(Ns),
        name => string:lowercase(Name),
        tag => Tag
    }.

%% The files to fetch, the registry name to give the model and the
%% generation defaults the source declares.
-spec plan(source(), livery_client:client(), map()) ->
    {ok, #{name := binary(), files := [file()], defaults := map()}} | {error, term()}.
plan(#{kind := https, url := Url} = S, _Client, Opts) ->
    Name = maps:get(name, Opts, filename:basename(uri_path(Url), <<".gguf">>)),
    Sha = maps:get(sha256, Opts, undefined),
    _ = S,
    {ok, #{
        name => Name,
        files => [
            #{role => model, url => Url, sha256 => lower(Sha), size => undefined, auth => false}
        ],
        defaults => #{}
    }};
plan(#{kind := hf, repo := Repo} = S, Client, Opts) ->
    Base = erllama_server_config:get(hf_base_url),
    Rev = maps:get(revision, Opts, <<"main">>),
    TreeUrl = <<Base/binary, "/api/models/", Repo/binary, "/tree/", Rev/binary, "?recursive=true">>,
    case get_json(Client, TreeUrl, hf_headers()) of
        {ok, Tree} when is_list(Tree) -> hf_plan(S, gguf_files(Tree), Base, Rev);
        {ok, _} -> {error, {unexpected_response, TreeUrl}};
        {error, _} = E -> E
    end;
plan(#{kind := ollama, namespace := Ns, name := Name, tag := Tag}, Client, _Opts) ->
    Base = erllama_server_config:get(ollama_base_url),
    Repo = <<Ns/binary, "/", Name/binary>>,
    Url = <<Base/binary, "/v2/", Repo/binary, "/manifests/", Tag/binary>>,
    case get_json(Client, Url, [{<<"accept">>, ?OLLAMA_MANIFEST}]) of
        {ok, #{<<"layers">> := Layers}} -> ollama_plan(Client, Base, Repo, Ns, Name, Tag, Layers);
        {ok, _} -> {error, {unexpected_response, Url}};
        {error, _} = E -> E
    end.

gguf_files(Tree) ->
    [
        F
     || #{<<"type">> := <<"file">>, <<"path">> := P} = F <- Tree,
        filename:extension(P) =:= <<".gguf">>
    ].

hf_plan(#{repo := Repo} = S, Files, Base, Rev) ->
    {Models, Projectors} = lists:partition(fun(#{<<"path">> := P}) -> not is_mmproj(P) end, Files),
    case pick_model(S, Models) of
        {ok, #{<<"path">> := Path} = Model} ->
            Tag =
                case S of
                    #{file := _} -> string:lowercase(filename:basename(Path, <<".gguf">>));
                    _ -> quant_of(Path)
                end,
            Proj = [
                hf_file(projector, P, Repo, Base, Rev)
             || P <- pick_projector(Path, Projectors), not is_map_key(file, S)
            ],
            {ok, #{
                name => <<"hf.co/", (string:lowercase(Repo))/binary, ":", Tag/binary>>,
                files => [hf_file(model, Model, Repo, Base, Rev) | Proj],
                defaults => #{}
            }};
        {error, _} = E ->
            E
    end.

pick_model(#{file := File}, Models) ->
    case [M || #{<<"path">> := P} = M <- Models, P =:= File] of
        [M | _] -> {ok, M};
        [] -> {error, {not_in_repository, File}}
    end;
pick_model(#{quant := Quant}, Models) ->
    Want =
        case Quant of
            undefined -> ?DEFAULT_QUANT;
            Q -> Q
        end,
    %% Split GGUFs (`-00001-of-00003') are not supported yet.
    Single = [M || #{<<"path">> := P} = M <- Models, binary:match(P, <<"-of-0">>) =:= nomatch],
    case {[M || #{<<"path">> := P} = M <- Single, quant_of(P) =:= Want], Quant, Single} of
        {[M | _], _, _} ->
            {ok, M};
        {[], undefined, [Only]} ->
            %% No quantisation asked for and only one file: take it.
            {ok, Only};
        {[], _, _} ->
            {error, {quant_not_found, Want, [quant_of(P) || #{<<"path">> := P} <- Single]}}
    end.

%% The projector matching the model's quantisation, else F16, else any.
pick_projector(_Path, []) ->
    [];
pick_projector(Path, Projectors) ->
    Q = quant_of(Path),
    Ranked = lists:sort(
        fun(A, B) -> rank(A, Q) =< rank(B, Q) end,
        Projectors
    ),
    [hd(Ranked)].

rank(#{<<"path">> := P}, Q) ->
    case quant_of(P) of
        Q -> 0;
        <<"f16">> -> 1;
        _ -> 2
    end.

hf_file(Role, #{<<"path">> := Path} = F, Repo, Base, Rev) ->
    {Sha, Size} =
        case F of
            #{<<"lfs">> := #{<<"oid">> := Oid, <<"size">> := Sz}} -> {Oid, Sz};
            #{<<"size">> := Sz} -> {undefined, Sz};
            _ -> {undefined, undefined}
        end,
    #{
        role => Role,
        url => <<Base/binary, "/", Repo/binary, "/resolve/", Rev/binary, "/", Path/binary>>,
        sha256 => lower(Sha),
        size => Size,
        auth => true
    }.

is_mmproj(Path) ->
    binary:match(string:lowercase(filename:basename(Path)), <<"mmproj">>) =/= nomatch.

%% `TinyLlama-1.1B-Chat.Q4_K_M.gguf' -> `q4_k_m': the last `.' or `-'
%% separated word of the file name that looks like a quantisation.
quant_of(Path) ->
    Base = string:lowercase(filename:basename(Path, <<".gguf">>)),
    Words = re:split(Base, <<"[.-]">>, [{return, binary}]),
    case [W || W <- Words, is_quant(W)] of
        [] -> Base;
        Qs -> lists:last(Qs)
    end.

is_quant(W) ->
    re:run(W, <<"^(i?q[0-9]_[a-z0-9_]+|q[0-9]_[0-9]|f16|f32|bf16|tq[0-9]_[0-9]|mxfp4[a-z_]*)$">>) =/=
        nomatch.

ollama_plan(Client, Base, Repo, Ns, Name, Tag, Layers) ->
    Blob = fun(Digest) -> <<Base/binary, "/v2/", Repo/binary, "/blobs/", Digest/binary>> end,
    Files = [
        #{role => role(MT), url => Blob(D), sha256 => digest_hex(D), size => Sz, auth => false}
     || #{<<"mediaType">> := MT, <<"digest">> := D, <<"size">> := Sz} <- Layers,
        role(MT) =/= ignore
    ],
    case [F || #{role := model} = F <- Files] of
        [_] ->
            Params = ollama_params(Client, Blob, Layers),
            {ok, #{name => ollama_name(Ns, Name, Tag), files => Files, defaults => Params}};
        _ ->
            {error, {no_model_layer, Repo}}
    end.

%% The `params' layer, when there is one, as erllama request options.
ollama_params(Client, Blob, Layers) ->
    Params = [
        D
     || #{<<"mediaType">> := <<"application/vnd.ollama.image.params">>, <<"digest">> := D} <- Layers
    ],
    case Params of
        [P | _] ->
            case get_json(Client, Blob(P), []) of
                {ok, Map} when is_map(Map) -> ollama_defaults(Map);
                _ -> #{}
            end;
        [] ->
            #{}
    end.

ollama_name(<<"library">>, Name, Tag) -> <<Name/binary, ":", Tag/binary>>;
ollama_name(Ns, Name, Tag) -> <<Ns/binary, "/", Name/binary, ":", Tag/binary>>.

role(<<"application/vnd.ollama.image.model">>) -> model;
role(<<"application/vnd.ollama.image.projector">>) -> projector;
role(_) -> ignore.

digest_hex(<<"sha256:", Hex/binary>>) -> string:lowercase(Hex);
digest_hex(_) -> undefined.

%% Ollama `params' into erllama request options.
ollama_defaults(P) ->
    Map = [
        {<<"stop">>, stop_sequences},
        {<<"temperature">>, temperature},
        {<<"top_p">>, top_p},
        {<<"top_k">>, top_k},
        {<<"min_p">>, min_p},
        {<<"repeat_penalty">>, repetition_penalty}
    ],
    maps:from_list([{K, normalize_param(K, V)} || {B, K} <- Map, {ok, V} <- [maps:find(B, P)]]).

normalize_param(stop_sequences, V) when is_list(V) -> [S || S <- V, is_binary(S)];
normalize_param(stop_sequences, V) when is_binary(V) -> [V];
normalize_param(_, V) -> V.

hf_headers() ->
    case token() of
        undefined -> [];
        T -> [{<<"authorization">>, <<"Bearer ", T/binary>>}]
    end.

token() ->
    case erllama_server_config:get(hf_token) of
        undefined ->
            case os:getenv("HF_TOKEN") of
                false -> undefined;
                "" -> undefined;
                T -> list_to_binary(T)
            end;
        T ->
            unicode:characters_to_binary(T)
    end.

get_json(Client, Url, Headers) ->
    case livery_client:request(Client, get, Url, #{headers => Headers, timeout => 30000}) of
        {ok, Resp} ->
            case {livery_client:status(Resp), livery_client:body(Resp)} of
                {200, {full, Body}} ->
                    try
                        {ok, json:decode(Body)}
                    catch
                        error:_ -> {error, {invalid_json, Url}}
                    end;
                {Status, _} when Status >= 300, Status < 400 ->
                    case livery_client:header(<<"location">>, Resp) of
                        undefined -> {error, {http_status, Status, Url}};
                        Loc -> get_json(Client, resolve(Url, Loc), Headers)
                    end;
                {Status, _} ->
                    {error, {http_status, Status, Url}}
            end;
        {error, Reason} ->
            {error, {http, Reason, Url}}
    end.

%% Scheme, host and port of a URL: a redirect to another origin drops
%% the Authorization header.
-spec origin(binary()) -> {binary(), binary(), integer()} | undefined.
origin(Url) ->
    case uri_string:parse(Url) of
        #{scheme := S, host := H} = U ->
            Port = maps:get(port, U, default_port(S)),
            {string:lowercase(S), string:lowercase(H), Port};
        _ ->
            undefined
    end.

default_port(<<"https">>) -> 443;
default_port(_) -> 80.

resolve(Base, Location) ->
    unicode:characters_to_binary(uri_string:resolve(Location, Base)).

uri_path(Url) ->
    case uri_string:parse(Url) of
        #{path := P} -> P;
        _ -> Url
    end.

lower(undefined) -> undefined;
lower(B) -> string:lowercase(unicode:characters_to_binary(B)).
