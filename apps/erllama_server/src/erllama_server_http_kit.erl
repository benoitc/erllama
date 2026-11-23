%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_http_kit).
-moduledoc false.
%% What every HTTP adapter shares: reading a JSON body, the error shape
%% of each protocol, and driving a run from a request handler.
%%
%% An HTTP request is a run without a conversation. The handler process
%% subscribes to the run's events, acks them, and cancels the run as
%% soon as the client goes away (Livery's `{livery_disconnect, _, _}'
%% message, or a write that fails), so ordinary HTTP inference never
%% outlives its client.

-export([read_json/2, json/2, err/3, err/4, crash/3, api_error/2]).
-export([drive/3, start_run/2, now_s/0, id/2, sender/1]).

-define(ACK_EVERY, 16).

-type protocol() :: openai | anthropic | ollama.
-export_type([protocol/0]).

%% The request body as a JSON object.
-spec read_json(livery_req:req(), protocol()) -> {ok, map()} | {error, livery_resp:resp()}.
read_json(Req, Proto) ->
    case body(Req) of
        {ok, <<>>} ->
            {error, err(Proto, 400, <<"the request body must be a JSON object">>)};
        {ok, Bin} ->
            try json:decode(Bin) of
                M when is_map(M) -> {ok, M};
                _ -> {error, err(Proto, 400, <<"the request body must be a JSON object">>)}
            catch
                error:_ -> {error, err(Proto, 400, <<"the request body is not valid JSON">>)}
            end;
        {error, body_too_large} ->
            {error, err(Proto, 413, <<"the request body is too large">>)};
        {error, _} ->
            {error, err(Proto, 400, <<"the request body could not be read">>)}
    end.

body(Req) ->
    case livery_req:body(Req) of
        empty ->
            {ok, <<>>};
        {buffered, IoData} ->
            {ok, iolist_to_binary(IoData)};
        {stream, Reader} ->
            case livery_body:read_all(Reader) of
                {ok, Data, _} -> {ok, iolist_to_binary(Data)};
                {error, body_too_large, _} -> {error, body_too_large};
                {error, R, _} -> {error, R}
            end
    end.

-spec json(100..599, term()) -> livery_resp:resp().
json(Status, Term) ->
    livery_resp:json(Status, json:encode(Term)).

%% An error in the shape the protocol's SDKs parse.
-spec err(protocol(), 100..599, iodata()) -> livery_resp:resp().
err(Proto, Status, Msg) ->
    err(Proto, Status, Msg, null).

-spec err(protocol(), 100..599, iodata(), binary() | null) -> livery_resp:resp().
err(openai, Status, Msg, Param) ->
    json(Status, #{
        <<"error">> => #{
            <<"message">> => iolist_to_binary(Msg),
            <<"type">> => openai_type(Status),
            <<"param">> => Param,
            <<"code">> => null
        }
    });
err(anthropic, Status, Msg, _Param) ->
    json(Status, #{
        <<"type">> => <<"error">>,
        <<"error">> => #{
            <<"type">> => anthropic_type(Status), <<"message">> => iolist_to_binary(Msg)
        }
    });
err(ollama, Status, Msg, _Param) ->
    json(Status, #{<<"error">> => iolist_to_binary(Msg)}).

openai_type(401) -> <<"authentication_error">>;
openai_type(404) -> <<"not_found_error">>;
openai_type(429) -> <<"rate_limit_error">>;
openai_type(S) when S >= 500 -> <<"server_error">>;
openai_type(_) -> <<"invalid_request_error">>.

anthropic_type(401) -> <<"authentication_error">>;
anthropic_type(404) -> <<"not_found_error">>;
anthropic_type(413) -> <<"request_too_large">>;
anthropic_type(429) -> <<"rate_limit_error">>;
anthropic_type(529) -> <<"overloaded_error">>;
anthropic_type(S) when S >= 500 -> <<"api_error">>;
anthropic_type(_) -> <<"invalid_request_error">>.

%% A handler crash, as a 500 the clients can read (livery_middleware:wrap/1).
-spec crash(throw | error | exit, term(), list()) -> livery_resp:resp().
crash(Class, Reason, Stack) ->
    logger:error("erllama_server http handler crashed: ~p:~p~n~p", [Class, Reason, Stack]),
    err(openai, 500, <<"internal server error">>).

%% A service API refusal as an HTTP error.
-spec api_error(protocol(), term()) -> livery_resp:resp().
api_error(P, {unknown_model, M}) ->
    err(P, 404, [<<"model '">>, to_bin(M), <<"' not found">>], <<"model">>);
api_error(P, {model_file_missing, M}) ->
    err(P, 404, [<<"the file of model '">>, to_bin(M), <<"' is missing">>], <<"model">>);
api_error(P, no_default_model) ->
    err(P, 400, <<"model is required (no default model is set)">>, <<"model">>);
api_error(P, too_many_runs) ->
    err(P, 429, <<"the server is running its maximum number of requests; retry later">>);
api_error(P, {invalid_option, K}) ->
    err(P, 400, [<<"unsupported option: ">>, to_bin(K)], to_bin(K));
api_error(P, {invalid_option, K, Why}) ->
    err(P, 400, io_lib:format("invalid option ~ts: ~p", [to_bin(K), Why]), to_bin(K));
api_error(P, {invalid_message, M}) ->
    err(P, 400, io_lib:format("invalid message: ~0p", [M]), <<"messages">>);
api_error(P, {media_not_supported, M}) ->
    err(P, 400, [<<"model '">>, to_bin(M), <<"' does not accept images or audio">>]);
api_error(P, {bad_request, Msg, Param}) ->
    err(P, 400, Msg, Param);
api_error(P, Other) ->
    err(P, 500, io_lib:format("~0p", [Other])).

to_bin(B) when is_binary(B) -> B;
to_bin(A) when is_atom(A) -> atom_to_binary(A);
to_bin(L) when is_list(L) -> unicode:characters_to_binary(L);
to_bin(T) -> iolist_to_binary(io_lib:format("~0p", [T])).

%% -----------------------------------------------------------------------------
%% Runs
%% -----------------------------------------------------------------------------

%% Start a stateless run (no conversation).
-spec start_run(map(), protocol()) -> {ok, binary()} | {error, livery_resp:resp()}.
start_run(Spec, Proto) ->
    case check_media(Spec) of
        ok ->
            case erllama_server_api:start_run(Spec#{conversation_id => none}) of
                {ok, #{run_id := Run}} -> {ok, Run};
                {error, Reason} -> {error, api_error(Proto, Reason)}
            end;
        {error, Reason} ->
            {error, api_error(Proto, Reason)}
    end.

%% Images and audio need a registry model with a projector; say so up
%% front rather than fail inside the engine.
check_media(#{messages := Msgs} = Spec) ->
    Kinds = lists:usort([
        T
     || #{content := Parts} <- Msgs,
        is_list(Parts),
        #{type := T} <- Parts,
        T =:= image orelse T =:= audio
    ]),
    case Kinds of
        [] ->
            ok;
        _ ->
            case erllama_server_models:resolve(maps:get(model, Spec, undefined)) of
                {ok, {registry, #{name := N} = E}} ->
                    projector_covers(N, maps:get(mmproj, E, undefined), Kinds);
                _ ->
                    ok
            end
    end;
check_media(_) ->
    ok.

projector_covers(Name, undefined, _Kinds) ->
    {error,
        {bad_request,
            [
                <<"model '">>,
                Name,
                <<"' has no multimodal projector; images and audio are not supported">>
            ],
            <<"messages">>}};
projector_covers(Name, MmProj, Kinds) ->
    Caps =
        case erllama_server_gguf:read(MmProj) of
            {ok, H} -> maps:get(projector, erllama_server_gguf:summary(H), false);
            {error, _} -> false
        end,
    Missing = [K || K <- Kinds, not covers(K, Caps)],
    case Missing of
        [] ->
            ok;
        [K | _] ->
            {error,
                {bad_request,
                    [<<"model '">>, Name, <<"' cannot take ">>, atom_to_binary(K), <<" input">>],
                    <<"messages">>}}
    end.

covers(image, #{vision := true}) -> true;
covers(audio, #{audio := true}) -> true;
covers(_, _) -> false.

%% Follow `Run' to its end. `Fun(Event, Acc)' sees every event and
%% returns `{ok, Acc}', or `{stop, Acc}' when it can no longer write to
%% the client. Returns `{done, Terminal, Acc}' or `{gone, Acc}' when
%% the client went away (the run is then cancelled).
-spec drive(binary(), fun((map(), term()) -> {ok | stop, term()}), term()) ->
    {done, map(), term()} | {gone, term()}.
drive(Run, Fun, Acc0) ->
    case erllama_server_api:attach(Run, self(), 0) of
        {ok, #{run_pid := Pid, replay := Replay}} ->
            case feed(Replay, Fun, Acc0) of
                {done, _, _} = Done -> Done;
                {stop, Acc} -> gone(Run, Acc);
                {ok, Acc} -> loop(Run, Pid, Fun, Acc, 0)
            end;
        {error, _} ->
            {gone, Acc0}
    end.

feed([], _Fun, Acc) ->
    {ok, Acc};
feed([E | Rest], Fun, Acc) ->
    case Fun(E, Acc) of
        {ok, Acc1} ->
            case terminal(E) of
                true -> {done, E, Acc1};
                false -> feed(Rest, Fun, Acc1)
            end;
        {stop, Acc1} ->
            {stop, Acc1}
    end.

loop(Run, Pid, Fun, Acc, Unacked) ->
    receive
        {erllama_server_run, Run, #{seq := Seq} = E} ->
            case Fun(E, Acc) of
                {ok, Acc1} ->
                    case terminal(E) of
                        true ->
                            {done, E, Acc1};
                        false when Unacked + 1 >= ?ACK_EVERY ->
                            Pid ! {ack, self(), Seq},
                            loop(Run, Pid, Fun, Acc1, 0);
                        false ->
                            loop(Run, Pid, Fun, Acc1, Unacked + 1)
                    end;
                {stop, Acc1} ->
                    gone(Run, Acc1)
            end;
        {erllama_server_run, Run, {dropped, _}} ->
            %% A local subscriber falls behind only if its client stalls.
            gone(Run, Acc);
        {livery_disconnect, _, _} ->
            gone(Run, Acc)
    after 30000 ->
        %% A long prefill sends nothing for a while; make sure the run
        %% is still there.
        case erllama_server_runs:lookup(Run) of
            {ok, #{status := running}} -> loop(Run, Pid, Fun, Acc, Unacked);
            _ -> gone(Run, Acc)
        end
    end.

gone(Run, Acc) ->
    _ = erllama_server_api:cancel(Run),
    {gone, Acc}.

terminal(#{type := T}) -> lists:member(T, [done, cancelled, error]).

%% A `Send(Obj, Acc)' for drive/3 callbacks over a Livery `Emit':
%% `{stop, Acc}' once the client is gone.
-spec sender(fun((term()) -> ok | {error, term()})) -> fun((term(), term()) -> {ok | stop, term()}).
sender(Emit) ->
    fun(Obj, Acc) ->
        case Emit(Obj) of
            ok -> {ok, Acc};
            {error, _} -> {stop, Acc}
        end
    end.

-spec now_s() -> integer().
now_s() -> erlang:system_time(second).

%% A protocol object id from a run id: `chatcmpl-9f2c...'.
-spec id(binary(), binary()) -> binary().
id(Prefix, <<"run_", Hex/binary>>) -> <<Prefix/binary, Hex/binary>>;
id(Prefix, Run) -> <<Prefix/binary, Run/binary>>.
