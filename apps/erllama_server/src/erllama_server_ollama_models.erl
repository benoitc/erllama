%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_ollama_models).
-moduledoc false.
%% The Ollama model-management endpoints: `/api/tags', `/api/show',
%% `/api/ps', `/api/pull' (a download job, reported as Ollama's status
%% lines), `/api/delete', `/api/copy', `/api/version', and `/' (the
%% liveness text Ollama clients probe). `/api/create' and `/api/push'
%% answer 501: models come from pulls or GGUF files.

-export([tags/1, show/1, ps/1, pull/1, delete/1, copy/1, version/1, root/1, not_supported/1]).

tags(_Req) ->
    {ok, Entries} = erllama_server_api:model_entries(),
    Models = [tag(E) || E <- Entries],
    erllama_server_http_kit:json(200, #{<<"models">> => Models}).

tag(#{name := N} = E) ->
    #{
        <<"name">> => N,
        <<"model">> => N,
        <<"modified_at">> => rfc3339(maps:get(created_at, E, 0)),
        <<"size">> => maps:get(size, E, 0),
        <<"digest">> => digest(E),
        <<"details">> => details(E)
    }.

digest(#{sha256 := S}) when is_binary(S) -> S;
digest(_) -> <<>>.

details(#{path := P}) ->
    Meta =
        case erllama_server_gguf:read(P) of
            {ok, H} -> erllama_server_gguf:summary(H);
            {error, _} -> #{}
        end,
    Family = null_or(maps:get(architecture, Meta, undefined)),
    #{
        <<"format">> => <<"gguf">>,
        <<"family">> => Family,
        <<"families">> => [Family],
        <<"parameter_size">> => null_or(maps:get(size_label, Meta, undefined)),
        <<"quantization_level">> => null_or(maps:get(file_type, Meta, undefined))
    }.

null_or(undefined) -> null;
null_or(V) -> V.

rfc3339(Ms) ->
    list_to_binary(calendar:system_time_to_rfc3339(Ms, [{unit, millisecond}, {offset, "Z"}])).

show(Req) ->
    with_name(Req, fun(Name) ->
        case erllama_server_api:show(Name) of
            {ok, #{metadata := Meta} = E} ->
                erllama_server_http_kit:json(200, #{
                    <<"modelfile">> => <<>>,
                    <<"parameters">> => parameters(maps:get(defaults, E, #{})),
                    <<"template">> => null_or(maps:get(chat_template, Meta, undefined)),
                    <<"details">> => details(E),
                    <<"model_info">> => model_info(Meta),
                    <<"capabilities">> => capabilities(E)
                });
            {ok, _} ->
                erllama_server_http_kit:err(ollama, 404, [
                    <<"model '">>, Name, <<"' is not in the registry">>
                ]);
            {error, R} ->
                erllama_server_http_kit:api_error(ollama, R)
        end
    end).

parameters(Defaults) ->
    iolist_to_binary([
        [atom_to_binary(K), <<" ">>, io_lib:format("~0p", [V]), <<"\n">>]
     || K := V <- Defaults
    ]).

model_info(Meta) ->
    Arch = maps:get(architecture, Meta, undefined),
    Base = #{<<"general.architecture">> => null_or(Arch)},
    case is_binary(Arch) of
        true ->
            Base#{
                <<Arch/binary, ".context_length">> => null_or(
                    maps:get(context_length, Meta, undefined)
                )
            };
        false ->
            Base
    end.

capabilities(#{mmproj := M}) when is_binary(M) -> [<<"completion">>, <<"vision">>];
capabilities(_) -> [<<"completion">>].

ps(_Req) ->
    Models = [
        #{
            <<"name">> => Id,
            <<"model">> => Id,
            <<"size">> => 0,
            <<"digest">> => <<>>,
            <<"expires_at">> => expires(P),
            <<"size_vram">> => maps:get(vram_estimate_b, P, 0)
        }
     || #{id := Id} = P <- element(2, erllama_server_api:ps())
    ],
    erllama_server_http_kit:json(200, #{<<"models">> => Models}).

expires(#{expires_at := E}) when is_integer(E) -> rfc3339(E);
expires(_) -> null.

delete(Req) ->
    with_name(Req, fun(Name) ->
        case erllama_server_api:remove(Name) of
            ok -> livery_resp:empty(200);
            {error, R} -> erllama_server_http_kit:api_error(ollama, R)
        end
    end).

copy(Req) ->
    case erllama_server_http_kit:read_json(Req, ollama) of
        {ok, #{<<"source">> := S, <<"destination">> := D}} ->
            case erllama_server_api:copy(S, D) of
                {ok, _} -> livery_resp:empty(200);
                {error, R} -> erllama_server_http_kit:api_error(ollama, R)
            end;
        {ok, _} ->
            erllama_server_http_kit:err(ollama, 400, <<"source and destination are required">>);
        {error, Resp} ->
            Resp
    end.

%% `/api/pull': an Ollama name, or `hf.co/ORG/REPO[:QUANT]'.
pull(Req) ->
    with_name(Req, fun(Name, Body) ->
        Source =
            case Name of
                <<"hf.co/", Repo/binary>> -> <<"hf://", Repo/binary>>;
                _ -> Name
            end,
        Stream =
            case Body of
                #{<<"stream">> := false} -> false;
                _ -> true
            end,
        case erllama_server_api:pull(Source, #{}) of
            {ok, #{job_id := Job}} when Stream ->
                livery_resp:ndjson(200, fun(Emit) -> pull_stream(Job, Emit) end);
            {ok, #{job_id := Job}} ->
                pull_whole(Job);
            {error, R} ->
                erllama_server_http_kit:api_error(ollama, R)
        end
    end).

pull_whole(Job) ->
    case erllama_server_http_kit:drive(Job, fun(_, A) -> {ok, A} end, ok) of
        {done, #{type := done}, _} ->
            erllama_server_http_kit:json(200, #{<<"status">> => <<"success">>});
        {done, #{type := error, data := #{error := R}}, _} ->
            erllama_server_http_kit:api_error(ollama, {pull_failed, R});
        _ ->
            erllama_server_http_kit:err(ollama, 500, <<"pull cancelled">>)
    end.

pull_stream(Job, Emit) ->
    Send = erllama_server_http_kit:sender(Emit),
    Fun = fun
        (#{type := started}, A) ->
            Send(#{<<"status">> => <<"pulling manifest">>}, A);
        (#{type := progress, data := #{file := F, done := D, total := T}}, A) ->
            Send(
                #{
                    <<"status">> => <<"pulling ", (atom_to_binary(F))/binary>>,
                    <<"completed">> => D,
                    <<"total">> => null_or(T)
                },
                A
            );
        (#{type := done}, A) ->
            Send(#{<<"status">> => <<"success">>}, A);
        (#{type := error, data := #{error := R}}, A) ->
            Send(#{<<"error">> => iolist_to_binary(io_lib:format("~0p", [R]))}, A);
        (_, A) ->
            {ok, A}
    end,
    _ = erllama_server_http_kit:drive(Job, Fun, ok),
    ok.

%% The model a management request names (`model', or the older `name'),
%% with the whole body.
with_name(Req, Fun) ->
    case erllama_server_http_kit:read_json(Req, ollama) of
        {ok, #{<<"model">> := N} = B} when is_binary(N) -> call_named(Fun, N, B);
        {ok, #{<<"name">> := N} = B} when is_binary(N) -> call_named(Fun, N, B);
        {ok, _} -> erllama_server_http_kit:err(ollama, 400, <<"model is required">>);
        {error, Resp} -> Resp
    end.

call_named(Fun, Name, _Body) when is_function(Fun, 1) -> Fun(Name);
call_named(Fun, Name, Body) -> Fun(Name, Body).

version(_Req) ->
    erllama_server_http_kit:json(200, #{<<"version">> => app_vsn()}).

app_vsn() ->
    case application:get_key(erllama_server, vsn) of
        {ok, V} -> list_to_binary(V);
        undefined -> <<"0.0.0">>
    end.

root(_Req) ->
    livery_resp:text(200, <<"erllama is running">>).

not_supported(Req) ->
    erllama_server_http_kit:err(ollama, 501, [
        livery_req:path(Req), <<" is not supported: models come from pulls or GGUF files">>
    ]).
