%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_http_auth).
-moduledoc false.
%% API keys for the HTTP APIs. A request carries its key the way its
%% SDK sends it: `Authorization: Bearer KEY' (OpenAI, Ollama clients)
%% or `x-api-key: KEY' (Anthropic). Keys are compared in constant time.
%% With no keys configured every request is let through (the listener
%% is then on loopback, or `allow_unauthenticated' was set). The health
%% probes never need a key.

-behaviour(livery_middleware).

-export([call/3, key/1]).

-define(OPEN_PATHS, [<<"/health">>, <<"/ready">>]).

call(Req, Next, []) ->
    Next(Req);
call(Req, Next, Keys) ->
    case lists:member(livery_req:path(Req), ?OPEN_PATHS) orelse valid(key(Req), Keys) of
        true ->
            Next(Req);
        false ->
            Resp = erllama_server_http_kit:err(proto(Req), 401, <<"invalid or missing API key">>),
            livery_resp:with_header(<<"www-authenticate">>, <<"Bearer">>, Resp)
    end.

%% The key a request presents, if any.
-spec key(livery_req:req()) -> binary() | undefined.
key(Req) ->
    case livery_ext:bearer_token(Req) of
        undefined -> livery_req:header(<<"x-api-key">>, Req);
        Token -> Token
    end.

valid(undefined, _Keys) ->
    false;
valid(Key, Keys) ->
    %% Every key is compared (equal/2 runs before `orelse' looks at the
    %% accumulator), so the time taken does not tell which key matched.
    lists:foldl(
        fun(K, Acc) -> equal(Key, unicode:characters_to_binary(K)) orelse Acc end, false, Keys
    ).

equal(A, B) when byte_size(A) =:= byte_size(B) -> crypto:hash_equals(A, B);
equal(_, _) -> false.

%% Answer in the shape of the API that was called.
proto(Req) ->
    case livery_req:path(Req) of
        <<"/v1/messages", _/binary>> -> anthropic;
        <<"/api/", _/binary>> -> ollama;
        _ -> openai
    end.
