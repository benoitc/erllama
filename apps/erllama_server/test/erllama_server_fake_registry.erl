%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% A stand-in for the Hugging Face hub, its CDN and the Ollama registry,
%% served by Livery on two local origins (`127.0.0.1' for the hub and
%% registry, `localhost' for the CDN), so pulls can be tested end to end
%% without the network. Its behaviour is set through `set/2':
%% `interrupt_after' (bytes sent before the first transfer of a blob is
%% cut), `ignore_range', `corrupt' (serve other bytes than announced)
%% and `require_token'. `seen/0' returns what each origin received.
%% @end
-module(erllama_server_fake_registry).

-export([start/0, stop/1, set/2, seen/0, blob/1, gguf/1, sha/1]).
-export([tree/1, resolve/1, cdn/1, manifest/1, ollama_blob/1, plain/1]).

-define(TAB, ?MODULE).

%% A small valid GGUF (no tensors, one key) padded to `Size' bytes.
gguf(Size) ->
    Key = <<"general.architecture">>,
    Val = <<"llama">>,
    Header =
        <<"GGUF", 3:32/little, 0:64/little, 1:64/little, (byte_size(Key)):64/little, Key/binary,
            8:32/little, (byte_size(Val)):64/little, Val/binary>>,
    Pad = Size - byte_size(Header),
    <<Header/binary, (rand_bytes(Pad))/binary>>.

rand_bytes(N) -> crypto:strong_rand_bytes(N).

sha(Bin) -> binary:encode_hex(crypto:hash(sha256, Bin), lowercase).

start() ->
    %% The table and the listeners outlive the process that starts the
    %% fake (a CT init_per_suite), so a small process of its own owns
    %% them.
    Self = self(),
    Owner = spawn(fun() -> owner(Self) end),
    receive
        {fake_ready, Owner, Ports} -> {ok, Ports#{owner => Owner}}
    after 10000 -> {error, fake_did_not_start}
    end.

owner(Parent) ->
    _ = ets:new(?TAB, [named_table, public, set]),
    Model = gguf(300 * 1024),
    Proj = gguf(64 * 1024),
    Params = json:encode(#{<<"stop">> => [<<"<|end|>">>], <<"temperature">> => 0.5}),
    true = ets:insert(?TAB, [
        {{blob, model}, Model},
        {{blob, projector}, Proj},
        {{blob, params}, iolist_to_binary(Params)},
        {seen, []}
    ]),
    {ok, Hub} = livery:start_service(#{
        http => #{ip => {127, 0, 0, 1}, port => 0}, router => hub_router()
    }),
    {ok, Cdn} = livery:start_service(#{
        http => #{ip => {127, 0, 0, 1}, port => 0}, router => cdn_router()
    }),
    #{h1 := [HubPort]} = livery:which_listeners(Hub),
    #{h1 := [CdnPort]} = livery:which_listeners(Cdn),
    true = ets:insert(?TAB, [{hub_port, HubPort}, {cdn_port, CdnPort}]),
    Parent !
        {fake_ready, self(), #{
            base => <<"http://127.0.0.1:", (integer_to_binary(HubPort))/binary>>
        }},
    receive
        stop ->
            ok = livery:stop_service(Hub),
            ok = livery:stop_service(Cdn)
    end.

stop(#{owner := Owner}) ->
    Owner ! stop,
    ok.

set(Key, Value) ->
    true = ets:insert(?TAB, {{opt, Key}, Value}),
    ok.

opt(Key, Default) ->
    case ets:lookup(?TAB, {opt, Key}) of
        [{_, V}] -> V;
        [] -> Default
    end.

seen() ->
    [{seen, L}] = ets:lookup(?TAB, seen),
    lists:reverse(L).

record(Origin, Req) ->
    [{seen, L}] = ets:lookup(?TAB, seen),
    Entry = #{
        origin => Origin,
        path => livery_req:path(Req),
        authorization => livery_req:header(<<"authorization">>, Req),
        range => livery_req:header(<<"range">>, Req)
    },
    true = ets:insert(?TAB, {seen, [Entry | L]}).

blob(Name) ->
    blob_by(Name).

blob_by(Name) ->
    [{_, Bin}] = ets:lookup(?TAB, {blob, Name}),
    Bin.

hub_router() ->
    livery_router:compile([
        {<<"GET">>, <<"/api/models/:org/:repo/tree/:rev">>, {?MODULE, tree}},
        {<<"GET">>, <<"/:org/:repo/resolve/:rev/*file">>, {?MODULE, resolve}},
        {<<"GET">>, <<"/v2/:ns/:name/manifests/:tag">>, {?MODULE, manifest}},
        {<<"GET">>, <<"/v2/:ns/:name/blobs/:digest">>, {?MODULE, ollama_blob}},
        {<<"GET">>, <<"/plain/*file">>, {?MODULE, plain}}
    ]).

cdn_router() ->
    livery_router:compile([{<<"GET">>, <<"/cdn/*file">>, {?MODULE, cdn}}]).

files() ->
    [
        {<<"tiny-model.Q4_K_M.gguf">>, model},
        {<<"tiny-model.Q8_0.gguf">>, model},
        {<<"mmproj-tiny-model-f16.gguf">>, projector}
    ].

tree(Req) ->
    record(hub, Req),
    case authorized(Req) of
        false ->
            livery_resp:json(401, <<"{\"error\":\"unauthorized\"}">>);
        true ->
            Entries =
                [
                    #{
                        <<"type">> => <<"file">>,
                        <<"path">> => Path,
                        <<"size">> => byte_size(blob_by(Kind)),
                        <<"lfs">> => #{
                            <<"oid">> => sha(blob_by(Kind)), <<"size">> => byte_size(blob_by(Kind))
                        }
                    }
                 || {Path, Kind} <- files()
                ] ++ [#{<<"type">> => <<"file">>, <<"path">> => <<"README.md">>, <<"size">> => 10}],
            livery_resp:json(200, json:encode(Entries))
    end.

authorized(Req) ->
    case opt(require_token, undefined) of
        undefined -> true;
        Token -> livery_req:header(<<"authorization">>, Req) =:= <<"Bearer ", Token/binary>>
    end.

%% The hub answers file downloads with a redirect to the CDN origin.
resolve(Req) ->
    record(hub, Req),
    File = livery_req:binding(<<"file">>, Req),
    [{_, CdnPort}] = ets:lookup(?TAB, cdn_port),
    Loc = <<"http://localhost:", (integer_to_binary(CdnPort))/binary, "/cdn/", File/binary>>,
    livery_resp:redirect(302, Loc).

cdn(Req) ->
    record(cdn, Req),
    File = livery_req:binding(<<"file">>, Req),
    Kind = proplists:get_value(File, files(), model),
    serve(Req, {cdn, File}, blob_by(Kind)).

manifest(Req) ->
    record(hub, Req),
    Model = blob_by(model),
    Params = blob_by(params),
    Layer = fun(MT, Bin) ->
        #{
            <<"mediaType">> => MT,
            <<"digest">> => <<"sha256:", (sha(Bin))/binary>>,
            <<"size">> => byte_size(Bin)
        }
    end,
    Doc = #{
        <<"schemaVersion">> => 2,
        <<"layers">> => [
            Layer(<<"application/vnd.ollama.image.model">>, Model),
            Layer(<<"application/vnd.ollama.image.template">>, <<"{{ .Prompt }}">>),
            Layer(<<"application/vnd.ollama.image.params">>, Params)
        ]
    },
    livery_resp:json(200, json:encode(Doc)).

ollama_blob(Req) ->
    record(hub, Req),
    <<"sha256:", Hex/binary>> = livery_req:binding(<<"digest">>, Req),
    Bin = hd([B || {{blob, _}, B} <- ets:tab2list(?TAB), sha(B) =:= Hex]),
    serve(Req, {ollama, Hex}, Bin).

plain(Req) ->
    record(hub, Req),
    serve(Req, plain, blob_by(model)).

%% Serve `Bin', honouring `Range: bytes=N-' unless told not to. The
%% first transfer of each blob can be cut after `interrupt_after' bytes.
serve(Req, Key, Bin0) ->
    Bin =
        case opt(corrupt, false) of
            true -> crypto:strong_rand_bytes(byte_size(Bin0));
            false -> Bin0
        end,
    Size = byte_size(Bin),
    {Status, Offset} =
        case {opt(ignore_range, false), livery_req:header(<<"range">>, Req)} of
            {false, <<"bytes=", R/binary>>} ->
                [From | _] = binary:split(R, <<"-">>),
                {206, binary_to_integer(From)};
            _ ->
                {200, 0}
        end,
    Body = binary:part(Bin, Offset, Size - Offset),
    Range =
        case Status of
            206 ->
                [
                    {<<"content-range">>,
                        iolist_to_binary(io_lib:format("bytes ~b-~b/~b", [Offset, Size - 1, Size]))}
                ];
            200 ->
                []
        end,
    Headers = [
        {<<"content-type">>, <<"application/octet-stream">>},
        {<<"content-length">>, integer_to_binary(byte_size(Body))}
        | Range
    ],
    Cut =
        case {opt(interrupt_after, undefined), ets:insert_new(?TAB, {{interrupted, Key}, true})} of
            {N, true} when is_integer(N), N < byte_size(Body) -> N;
            _ -> undefined
        end,
    livery_resp:stream(Status, Headers, fun(Emit) -> send(Emit, Body, Cut) end).

send(Emit, Body, undefined) ->
    chunks(Emit, Body);
send(Emit, Body, Cut) ->
    ok = chunks(Emit, binary:part(Body, 0, Cut)),
    %% Stall half way through, as a dropped transfer looks to a client.
    %% (Crashing the handler instead makes Livery write a 500 response
    %% into the unfinished body.)
    timer:sleep(5000),
    ok.

chunks(_Emit, <<>>) ->
    ok;
chunks(Emit, Bin) when byte_size(Bin) =< 16384 ->
    _ = Emit(Bin),
    ok;
chunks(Emit, <<Chunk:16384/binary, Rest/binary>>) ->
    _ = Emit(Chunk),
    chunks(Emit, Rest).
