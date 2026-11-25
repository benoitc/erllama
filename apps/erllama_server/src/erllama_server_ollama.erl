%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_ollama).
-moduledoc false.
%% The Ollama generation API: `/api/chat' and `/api/generate' (NDJSON
%% streams by default, one JSON object with `"stream": false'),
%% `/api/embed' and `/api/embeddings'. Generation runs through the same
%% run specs as every other front end. Model management is in
%% erllama_server_ollama_models.

-export([chat/1, generate/1, embed/1, embeddings/1]).
-export([chat_request/1, generate_request/1, options/1]).

-define(CHAT_FIELDS, [
    <<"model">>,
    <<"messages">>,
    <<"tools">>,
    <<"format">>,
    <<"options">>,
    <<"stream">>,
    <<"keep_alive">>,
    <<"think">>
]).
-define(GENERATE_FIELDS, [
    <<"model">>,
    <<"prompt">>,
    <<"images">>,
    <<"format">>,
    <<"options">>,
    <<"system">>,
    <<"stream">>,
    <<"raw">>,
    <<"keep_alive">>,
    <<"think">>
]).
-define(OPTIONS, [
    {<<"num_predict">>, response_tokens},
    {<<"temperature">>, temperature},
    {<<"top_p">>, top_p},
    {<<"top_k">>, top_k},
    {<<"min_p">>, min_p},
    {<<"seed">>, seed},
    {<<"stop">>, stop_sequences},
    {<<"repeat_penalty">>, repetition_penalty},
    {<<"repeat_last_n">>, penalty_last_n},
    {<<"frequency_penalty">>, frequency_penalty},
    {<<"presence_penalty">>, presence_penalty}
]).
%% Accepted and not applied per request: the context size is the model
%% entry's.
-define(INFO_OPTIONS, [<<"num_ctx">>]).

%% -----------------------------------------------------------------------------
%% Generation
%% -----------------------------------------------------------------------------

chat(Req) ->
    with_body(Req, fun chat_request/1, fun(Spec, Meta) -> respond(chat, Spec, Meta) end).

generate(Req) ->
    with_body(Req, fun generate_request/1, fun generate_or_load/2).

with_body(Req, Parse, Run) ->
    case erllama_server_http_kit:read_json(Req, ollama) of
        {ok, Body} ->
            case Parse(Body) of
                {ok, Spec, Meta} -> Run(Spec, Meta);
                {error, R} -> erllama_server_http_kit:api_error(ollama, R)
            end;
        {error, Resp} ->
            Resp
    end.

-spec chat_request(map()) -> {ok, map(), map()} | {error, term()}.
chat_request(Body) ->
    case fields(Body, ?CHAT_FIELDS) of
        ok ->
            case {messages(maps:get(<<"messages">>, Body, undefined)), gen_opts(Body)} of
                {{ok, Msgs}, {ok, Opts}} ->
                    {ok, spec(Body, #{messages => Msgs, opts => Opts}), meta(Body)};
                {{error, _} = E, _} ->
                    E;
                {_, {error, _} = E} ->
                    E
            end;
        {error, _} = E ->
            E
    end.

%% `/api/generate': a raw prompt when `raw' is true, otherwise one user
%% turn (after `system') through the chat template.
-spec generate_request(map()) -> {ok, map(), map()} | {error, term()}.
generate_request(Body) ->
    case fields(Body, ?GENERATE_FIELDS) of
        ok ->
            Prompt = maps:get(<<"prompt">>, Body, <<>>),
            case {gen_opts(Body), images(maps:get(<<"images">>, Body, []))} of
                {{ok, Opts}, {ok, Images}} ->
                    {ok, spec(Body, generate_input(Body, Prompt, Images, Opts)), meta(Body)};
                {{error, _} = E, _} ->
                    E;
                {_, {error, _} = E} ->
                    E
            end;
        {error, _} = E ->
            E
    end.

generate_input(#{<<"raw">> := true}, Prompt, [], Opts) ->
    #{prompt => Prompt, opts => maps:without([tools, json_schema, enable_thinking], Opts)};
generate_input(Body, Prompt, Images, Opts) ->
    System =
        case maps:get(<<"system">>, Body, undefined) of
            S when is_binary(S), S =/= <<>> -> [#{role => <<"system">>, content => S}];
            _ -> []
        end,
    Content =
        case Images of
            [] -> Prompt;
            _ -> [#{type => text, text => Prompt} | Images]
        end,
    #{messages => System ++ [#{role => <<"user">>, content => Content}], opts => Opts}.

bad(Msg, Param) -> {error, {bad_request, Msg, Param}}.

%% Ollama streams unless told `"stream": false'.
streamed(#{<<"stream">> := false}) -> false;
streamed(_) -> true.

fields(Body, Allowed) ->
    case [K || K <- maps:keys(Body), not lists:member(K, Allowed)] of
        [] -> ok;
        [K | _] -> bad([<<"unsupported parameter: ">>, K], K)
    end.

spec(Body, Spec0) ->
    Spec1 =
        case maps:find(<<"model">>, Body) of
            {ok, M} when is_binary(M) -> Spec0#{model => M};
            _ -> Spec0
        end,
    case erllama_server_openai_req:duration_ms(maps:get(<<"keep_alive">>, Body, undefined)) of
        {ok, Ms} -> Spec1#{keep_alive => Ms};
        error -> Spec1
    end.

meta(Body) ->
    #{
        stream => streamed(Body),
        model => maps:get(<<"model">>, Body, null)
    }.

%% Ollama messages: `images' are base64 strings, tool calls carry no id
%% and tool results name no call, so calls get ids here and the tool
%% messages that follow answer them in order.
messages([_ | _] = Msgs) ->
    {Converted, _Pending} = lists:mapfoldl(fun message/2, [], Msgs),
    case [E || {error, _} = E <- Converted] of
        [] -> {ok, [M || {ok, M} <- Converted]};
        [E | _] -> E
    end;
messages(_) ->
    bad(<<"messages must be a non-empty array">>, <<"messages">>).

message(#{<<"role">> := <<"assistant">>, <<"tool_calls">> := [_ | _] = Calls} = M, _Pending) ->
    Ids = [erllama_server_ids:tool_call() || _ <- Calls],
    ToolCalls = [
        #{
            type => function,
            id => Id,
            function => #{
                name => N,
                arguments => iolist_to_binary(json:encode(maps:get(<<"arguments">>, F, #{})))
            }
        }
     || {Id, #{<<"function">> := #{<<"name">> := N} = F}} <- lists:zip(Ids, Calls)
    ],
    Content =
        case maps:get(<<"content">>, M, <<>>) of
            <<>> -> null;
            C -> C
        end,
    {{ok, #{role => <<"assistant">>, content => Content, tool_calls => ToolCalls}}, Ids};
message(#{<<"role">> := <<"tool">>, <<"content">> := C} = M, Pending) ->
    {Id, Rest} =
        case Pending of
            [P | Ps] -> {P, Ps};
            [] -> {erllama_server_ids:tool_call(), []}
        end,
    Base = #{role => <<"tool">>, tool_call_id => Id, content => C},
    Msg =
        case maps:find(<<"tool_name">>, M) of
            {ok, N} when is_binary(N) -> Base#{name => N};
            _ -> Base
        end,
    {{ok, Msg}, Rest};
message(#{<<"role">> := Role, <<"content">> := C} = M, Pending) when is_binary(C) ->
    case images(maps:get(<<"images">>, M, [])) of
        {ok, []} ->
            {{ok, #{role => Role, content => C}}, Pending};
        {ok, Images} ->
            {{ok, #{role => Role, content => [#{type => text, text => C} | Images]}}, Pending};
        {error, _} = E ->
            {E, Pending}
    end;
message(_, Pending) ->
    {bad(<<"every message needs a role and content">>, <<"messages">>), Pending}.

images(L) when is_list(L) ->
    Decoded = [erllama_server_openai_req:decode64(B) || B <- L],
    case lists:member(error, Decoded) of
        true -> bad(<<"images must be base64 strings">>, <<"images">>);
        false -> {ok, [#{type => image, data => D} || {ok, D} <- Decoded]}
    end;
images(_) ->
    bad(<<"images must be an array">>, <<"images">>).

gen_opts(Body) ->
    case options(maps:get(<<"options">>, Body, #{})) of
        {ok, O0} ->
            O1 =
                case maps:get(<<"format">>, Body, undefined) of
                    <<"json">> -> O0#{json_schema => #{<<"type">> => <<"object">>}};
                    F when is_map(F) -> O0#{json_schema => F};
                    _ -> O0
                end,
            O2 =
                case maps:get(<<"think">>, Body, undefined) of
                    B when is_boolean(B) -> O1#{enable_thinking => B};
                    _ -> O1
                end,
            case erllama_server_openai_req:tools(maps:get(<<"tools">>, Body, [])) of
                {ok, []} -> {ok, O2};
                {ok, Tools} -> {ok, O2#{tools => Tools}};
                {error, _} = E -> E
            end;
        {error, _} = E ->
            E
    end.

%% The `options' map as erllama request options.
-spec options(term()) -> {ok, map()} | {error, term()}.
options(Opts) when is_map(Opts) ->
    Known = [K || {K, _} <- ?OPTIONS] ++ ?INFO_OPTIONS,
    case [K || K <- maps:keys(Opts), not lists:member(K, Known)] of
        [] ->
            Mapped = [
                {To, V}
             || {From, To} <- ?OPTIONS, {ok, V} <- [maps:find(From, Opts)], V =/= null
            ],
            {ok,
                maps:from_list([
                    option(K, V)
                 || {K, V} <- Mapped, not (K =:= response_tokens andalso V < 0)
                ])};
        [K | _] ->
            bad([<<"unsupported option: ">>, K], K)
    end;
options(_) ->
    bad(<<"options must be an object">>, <<"options">>).

option(stop_sequences, S) when is_binary(S) -> {stop_sequences, [S]};
option(K, V) -> {K, V}.

%% An empty prompt loads the model (or with `keep_alive: 0' unloads it),
%% as Ollama does.
generate_or_load(#{prompt := <<>>} = Spec, Meta) ->
    load_only(Spec, Meta);
generate_or_load(#{messages := [#{role := <<"user">>, content := <<>>}]} = Spec, Meta) ->
    load_only(Spec, Meta);
generate_or_load(Spec, Meta) ->
    respond(generate, Spec, Meta).

load_only(Spec, Meta) ->
    Model = maps:get(model, Spec, undefined),
    {Result, Reason} =
        case maps:get(keep_alive, Spec, undefined) of
            0 -> {erllama_server_api:unload(Model), <<"unload">>};
            KA -> {erllama_server_api:load(Model, #{keep_alive => KA}), <<"load">>}
        end,
    case Result of
        Ok when Ok =:= ok; element(1, Ok) =:= ok ->
            erllama_server_http_kit:json(200, #{
                <<"model">> => maps:get(model, Meta),
                <<"created_at">> => timestamp(),
                <<"response">> => <<>>,
                <<"done">> => true,
                <<"done_reason">> => Reason
            });
        {error, R} ->
            erllama_server_http_kit:api_error(ollama, R)
    end.

%% -----------------------------------------------------------------------------
%% Responses
%% -----------------------------------------------------------------------------

respond(Kind, Spec, #{stream := Stream} = Meta) ->
    case erllama_server_http_kit:start_run(Spec, ollama) of
        {ok, Run} when Stream ->
            livery_resp:ndjson(200, fun(Emit) -> stream(Kind, Run, Meta, Emit) end);
        {ok, Run} ->
            case erllama_server_http_kit:drive(Run, fun(_, Acc) -> {ok, Acc} end, ok) of
                {done, #{type := error, data := #{error := R}}, _} ->
                    erllama_server_http_kit:api_error(ollama, R);
                {done, #{data := R}, _} ->
                    erllama_server_http_kit:json(200, final(Kind, Run, Meta, R, whole));
                {gone, _} ->
                    erllama_server_http_kit:err(ollama, 499, <<"client closed the request">>)
            end;
        {error, Resp} ->
            Resp
    end.

stream(Kind, Run, Meta, Emit) ->
    Send = erllama_server_http_kit:sender(Emit),
    Fun = fun(E, Acc) -> event(Kind, Run, Meta, Send, E, Acc) end,
    _ = erllama_server_http_kit:drive(Run, Fun, ok),
    ok.

event(Kind, Run, Meta, Send, #{type := token, data := D}, Acc) ->
    Send(piece(Kind, Run, Meta, #{content => D}), Acc);
event(Kind, Run, Meta, Send, #{type := thinking, data := D}, Acc) ->
    Send(piece(Kind, Run, Meta, #{thinking => D}), Acc);
event(Kind, Run, Meta, Send, #{type := T, data := R}, Acc) when T =:= done; T =:= cancelled ->
    Calls =
        case {Kind, maps:get(tool_calls, R, [])} of
            {chat, [_ | _] = C} -> [piece(chat, Run, Meta, #{tool_calls => C})];
            _ -> []
        end,
    lists:foldl(
        fun
            (Obj, {ok, A}) -> Send(Obj, A);
            (_, Stop) -> Stop
        end,
        {ok, Acc},
        Calls ++ [final(Kind, Run, Meta, R, last)]
    );
event(_Kind, _Run, _Meta, Send, #{type := error, data := #{error := Reason}}, Acc) ->
    Send(#{<<"error">> => iolist_to_binary(io_lib:format("~0p", [Reason]))}, Acc);
event(_Kind, _Run, _Meta, _Send, _E, Acc) ->
    {ok, Acc}.

%% One streamed line.
piece(chat, Run, Meta, Part) ->
    Msg0 = #{<<"role">> => <<"assistant">>, <<"content">> => maps:get(content, Part, <<>>)},
    Msg1 =
        case Part of
            #{thinking := T} -> Msg0#{<<"thinking">> => T};
            _ -> Msg0
        end,
    Msg =
        case Part of
            #{tool_calls := Calls} -> Msg1#{<<"tool_calls">> => [call(C) || C <- Calls]};
            _ -> Msg1
        end,
    (head(Run, Meta))#{<<"message">> => Msg, <<"done">> => false};
piece(generate, Run, Meta, Part) ->
    Base = (head(Run, Meta))#{<<"response">> => maps:get(content, Part, <<>>), <<"done">> => false},
    case Part of
        #{thinking := T} -> Base#{<<"thinking">> => T};
        _ -> Base
    end.

call(#{name := N, arguments := A}) ->
    #{<<"function">> => #{<<"name">> => N, <<"arguments">> => A}}.

%% The closing object: the whole reply when not streamed, an empty
%% message when it ends a stream, plus the done fields.
final(Kind, Run, Meta, R, How) ->
    Reply =
        case {Kind, How} of
            {chat, whole} ->
                Msg0 = #{
                    <<"role">> => <<"assistant">>, <<"content">> => maps:get(content, R, <<>>)
                },
                Msg1 =
                    case maps:get(reasoning, R, <<>>) of
                        <<>> -> Msg0;
                        T -> Msg0#{<<"thinking">> => T}
                    end,
                Msg =
                    case maps:get(tool_calls, R, []) of
                        [] -> Msg1;
                        Calls -> Msg1#{<<"tool_calls">> => [call(C) || C <- Calls]}
                    end,
                #{<<"message">> => Msg};
            {chat, last} ->
                #{<<"message">> => #{<<"role">> => <<"assistant">>, <<"content">> => <<>>}};
            {generate, whole} ->
                #{<<"response">> => maps:get(content, R, <<>>)};
            {generate, last} ->
                #{<<"response">> => <<>>}
        end,
    maps:merge(maps:merge(head(Run, Meta), Reply), done_fields(R)).

done_fields(R) ->
    #{prompt_tokens := P, completion_tokens := G} = maps:get(usage, R, #{
        prompt_tokens => 0, completion_tokens => 0
    }),
    #{prefill_ms := PMs, generation_ms := GMs} = maps:get(timings, R, #{
        prefill_ms => 0, generation_ms => 0
    }),
    #{
        <<"done">> => true,
        <<"done_reason">> => done_reason(R),
        <<"total_duration">> => (PMs + GMs) * 1000000,
        <<"load_duration">> => 0,
        <<"prompt_eval_count">> => P,
        <<"prompt_eval_duration">> => PMs * 1000000,
        <<"eval_count">> => G,
        <<"eval_duration">> => GMs * 1000000
    }.

done_reason(#{finish_reason := length}) -> <<"length">>;
done_reason(_) -> <<"stop">>.

head(Run, Meta) ->
    #{<<"model">> => model_name(Run, Meta), <<"created_at">> => timestamp()}.

model_name(Run, #{model := null}) ->
    case erllama_server_api:run(Run) of
        {ok, #{model := M}} -> M;
        _ -> null
    end;
model_name(_Run, #{model := M}) ->
    M.

timestamp() ->
    list_to_binary(
        calendar:system_time_to_rfc3339(erlang:system_time(millisecond), [
            {unit, millisecond}, {offset, "Z"}
        ])
    ).

%% -----------------------------------------------------------------------------
%% Embeddings
%% -----------------------------------------------------------------------------

embed(Req) ->
    embed_with(Req, <<"input">>, fun(Model, Vectors, Count) ->
        #{<<"model">> => Model, <<"embeddings">> => Vectors, <<"prompt_eval_count">> => Count}
    end).

embeddings(Req) ->
    embed_with(Req, <<"prompt">>, fun(_Model, [V | _], _Count) -> #{<<"embedding">> => V} end).

embed_with(Req, Field, Shape) ->
    case erllama_server_http_kit:read_json(Req, ollama) of
        {ok, #{<<"model">> := Model} = Body} ->
            Inputs =
                case maps:get(Field, Body, undefined) of
                    B when is_binary(B) -> [B];
                    L when is_list(L) -> L;
                    _ -> []
                end,
            case {Inputs, lists:all(fun is_binary/1, Inputs)} of
                {[_ | _], true} ->
                    run_embed(Model, Inputs, Body, Shape);
                _ ->
                    erllama_server_http_kit:err(ollama, 400, [
                        Field, <<" must be a string or an array of strings">>
                    ])
            end;
        {ok, _} ->
            erllama_server_http_kit:err(ollama, 400, <<"model is required">>);
        {error, Resp} ->
            Resp
    end.

run_embed(Model, Inputs, Body, Shape) ->
    KA =
        case erllama_server_openai_req:duration_ms(maps:get(<<"keep_alive">>, Body, undefined)) of
            {ok, Ms} -> Ms;
            error -> undefined
        end,
    case erllama_server_api:embed(Model, Inputs, #{keep_alive => KA}) of
        {ok, #{vectors := Vectors, prompt_tokens := Count}} ->
            erllama_server_http_kit:json(200, Shape(Model, Vectors, Count));
        {error, R} ->
            erllama_server_http_kit:api_error(ollama, R)
    end.
