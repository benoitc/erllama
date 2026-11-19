%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_openai_responses).
-moduledoc false.
%% The OpenAI Responses API: `POST /v1/responses' (streamed as typed
%% server-sent events when asked), `GET' and `DELETE /v1/responses/{id}'.
%%
%% A stored response (`store', the default) is a turn of a server-side
%% conversation: `previous_response_id' continues it, with its history
%% and the KV cache warmth of the previous turn. `store: false' without
%% a previous response is a stateless request.

-export([create/1, retrieve/1, delete/1]).
-export([request/1]).
%% For erllama_server_openai_responses_stream.
-export([response_object/2, base_object/2, message_item/3, call_item/4, keep/2, indexed/1]).

-define(FIELDS, [
    <<"model">>,
    <<"input">>,
    <<"instructions">>,
    <<"tools">>,
    <<"tool_choice">>,
    <<"previous_response_id">>,
    <<"store">>,
    <<"stream">>,
    <<"max_output_tokens">>,
    <<"temperature">>,
    <<"top_p">>,
    <<"text">>,
    <<"reasoning">>,
    <<"parallel_tool_calls">>,
    <<"truncation">>
]).
-define(INFO_FIELDS, [<<"metadata">>, <<"user">>, <<"service_tier">>]).

%% -----------------------------------------------------------------------------
%% Requests
%% -----------------------------------------------------------------------------

create(Req) ->
    case erllama_server_http_kit:read_json(Req, openai) of
        {ok, Body} ->
            case request(Body) of
                {ok, Spec, Meta} -> respond(Spec, Meta);
                {error, R} -> erllama_server_http_kit:api_error(openai, R)
            end;
        {error, Resp} ->
            Resp
    end.

retrieve(Req) ->
    case erllama_server_responses:get(livery_req:binding(<<"id">>, Req)) of
        {ok, Response, _Conv} -> erllama_server_http_kit:json(200, Response);
        {error, not_found} -> erllama_server_http_kit:err(openai, 404, <<"response not found">>)
    end.

delete(Req) ->
    Id = livery_req:binding(<<"id">>, Req),
    case erllama_server_responses:delete(Id) of
        ok ->
            erllama_server_http_kit:json(200, #{
                <<"id">> => Id, <<"object">> => <<"response.deleted">>, <<"deleted">> => true
            });
        {error, not_found} ->
            erllama_server_http_kit:err(openai, 404, <<"response not found">>)
    end.

%% A Responses request as a run spec (with its conversation), and what
%% the response object needs.
-spec request(map()) -> {ok, map(), map()} | {error, term()}.
request(Body) ->
    case {fields(Body), unsupported_values(Body)} of
        {ok, ok} ->
            case {input(Body), opts(Body), previous(Body)} of
                {{ok, Msgs}, {ok, Opts}, {ok, Conv}} ->
                    {ok, spec(Body, Msgs, Opts, Conv), meta(Body)};
                {{error, _} = E, _, _} ->
                    E;
                {_, {error, _} = E, _} ->
                    E;
                {_, _, {error, _} = E} ->
                    E
            end;
        {{error, _} = E, _} ->
            E;
        {_, {error, _} = E} ->
            E
    end.

bad(Msg, Param) -> {error, {bad_request, Msg, Param}}.

is_true(true) -> true;
is_true(_) -> false.

is_false(false) -> true;
is_false(_) -> false.

fields(Body) ->
    case [K || K <- maps:keys(Body), not lists:member(K, ?FIELDS ++ ?INFO_FIELDS)] of
        [] -> ok;
        [K | _] -> bad([<<"unsupported parameter: ">>, K], K)
    end.

unsupported_values(#{<<"truncation">> := T}) when T =/= <<"disabled">> ->
    bad(<<"only truncation \"disabled\" is supported">>, <<"truncation">>);
unsupported_values(_) ->
    ok.

spec(Body, Msgs, Opts, Conv) ->
    Spec0 = #{messages => Msgs, opts => Opts, conversation_id => Conv},
    case maps:find(<<"model">>, Body) of
        {ok, M} when is_binary(M) -> Spec0#{model => M};
        _ -> Spec0
    end.

meta(Body) ->
    #{
        stream => is_true(maps:get(<<"stream">>, Body, false)),
        store => not is_false(maps:get(<<"store">>, Body, true)),
        model => maps:get(<<"model">>, Body, null),
        echo => maps:with(
            [
                <<"instructions">>,
                <<"tools">>,
                <<"tool_choice">>,
                <<"temperature">>,
                <<"top_p">>,
                <<"text">>,
                <<"metadata">>,
                <<"previous_response_id">>
            ],
            Body
        )
    }.

%% The conversation to continue: a copy of the previous response's
%% (a branch of its own), a new one when this response is stored, none
%% for a stateless request.
previous(#{<<"previous_response_id">> := Prev}) when is_binary(Prev) ->
    case erllama_server_responses:continue(Prev) of
        {ok, Conv} -> {ok, Conv};
        {error, _} -> bad([<<"previous response not found: ">>, Prev], <<"previous_response_id">>)
    end;
previous(Body) ->
    case maps:get(<<"store">>, Body, true) of
        false -> {ok, none};
        _ -> {ok, undefined}
    end.

input(Body) ->
    Instructions =
        case maps:get(<<"instructions">>, Body, undefined) of
            I when is_binary(I), I =/= <<>> -> [#{role => <<"system">>, content => I}];
            _ -> []
        end,
    case items(maps:get(<<"input">>, Body, undefined)) of
        {ok, Msgs} -> {ok, Instructions ++ Msgs};
        {error, _} = E -> E
    end.

items(Text) when is_binary(Text) ->
    {ok, [#{role => <<"user">>, content => Text}]};
items([_ | _] = Items) ->
    fold_items(Items, []);
items(_) ->
    bad(<<"input must be a string or an array of items">>, <<"input">>).

%% Consecutive function calls become one assistant message with several
%% tool calls; their outputs are tool messages.
fold_items([], Acc) ->
    {ok, lists:reverse(Acc)};
fold_items([#{<<"type">> := <<"function_call">>} = Call | Rest], Acc) ->
    Tc = #{
        type => function,
        id => maps:get(<<"call_id">>, Call, null),
        function => #{
            name => maps:get(<<"name">>, Call),
            arguments => maps:get(<<"arguments">>, Call, <<"{}">>)
        }
    },
    case Acc of
        [#{role := <<"assistant">>, tool_calls := Calls} = Prev | Tail] ->
            fold_items(Rest, [Prev#{tool_calls := Calls ++ [Tc]} | Tail]);
        _ ->
            fold_items(Rest, [#{role => <<"assistant">>, content => null, tool_calls => [Tc]} | Acc])
    end;
fold_items(
    [#{<<"type">> := <<"function_call_output">>, <<"call_id">> := Id, <<"output">> := Out} | Rest],
    Acc
) ->
    fold_items(Rest, [#{role => <<"tool">>, tool_call_id => Id, content => output_text(Out)} | Acc]);
fold_items([#{<<"type">> := <<"reasoning">>} | Rest], Acc) ->
    fold_items(Rest, Acc);
fold_items([#{<<"role">> := Role0, <<"content">> := C} = Item | Rest], Acc) ->
    case maps:get(<<"type">>, Item, <<"message">>) of
        <<"message">> ->
            Role =
                case Role0 of
                    <<"developer">> -> <<"system">>;
                    R -> R
                end,
            case content(C) of
                {ok, Content} -> fold_items(Rest, [#{role => Role, content => Content} | Acc]);
                {error, _} = E -> E
            end;
        T ->
            bad([<<"unsupported input item: ">>, T], <<"input">>)
    end;
fold_items([#{<<"type">> := T} | _], _Acc) ->
    bad([<<"unsupported input item: ">>, T], <<"input">>);
fold_items(_, _Acc) ->
    bad(<<"input items need a type or a role">>, <<"input">>).

output_text(O) when is_binary(O) -> O;
output_text(O) -> iolist_to_binary(json:encode(O)).

content(C) when is_binary(C) ->
    {ok, C};
content([_ | _] = Parts) ->
    Converted = [part(P) || P <- Parts],
    case [E || {error, _} = E <- Converted] of
        [] -> {ok, [V || {ok, V} <- Converted]};
        [E | _] -> E
    end;
content(_) ->
    bad(<<"content must be a string or an array of parts">>, <<"input">>).

part(#{<<"type">> := T, <<"text">> := Text}) when T =:= <<"input_text">>; T =:= <<"output_text">> ->
    {ok, #{type => text, text => Text}};
part(#{<<"type">> := <<"input_image">>, <<"image_url">> := Url}) ->
    case erllama_server_openai_req:data_url(Url) of
        {ok, Bytes} -> {ok, #{type => image, data => Bytes}};
        error -> bad(<<"input_image must be a base64 data: URL">>, <<"input">>)
    end;
part(#{<<"type">> := T}) ->
    bad([<<"unsupported content part: ">>, T], <<"input">>).

opts(Body) ->
    Base = maps:from_list(
        [
            {K, V}
         || {F, K} <- [
                {<<"max_output_tokens">>, response_tokens},
                {<<"temperature">>, temperature},
                {<<"top_p">>, top_p}
            ],
            {ok, V} <- [maps:find(F, Body)],
            V =/= null
        ]
    ),
    WithText =
        case maps:get(<<"text">>, Body, undefined) of
            #{<<"format">> := #{<<"type">> := <<"json_schema">>, <<"schema">> := S}} ->
                {ok, Base#{json_schema => S}};
            #{<<"format">> := #{<<"type">> := <<"json_object">>}} ->
                {ok, Base#{json_schema => #{<<"type">> => <<"object">>}}};
            #{<<"format">> := #{<<"type">> := <<"text">>}} ->
                {ok, Base};
            undefined ->
                {ok, Base};
            _ ->
                bad(<<"text.format must be text, json_object or json_schema">>, <<"text">>)
        end,
    WithReasoning =
        case {WithText, maps:get(<<"reasoning">>, Body, undefined)} of
            {{ok, O}, #{<<"effort">> := <<"none">>}} -> {ok, O#{enable_thinking => false}};
            {{ok, O}, #{<<"effort">> := _}} -> {ok, O#{enable_thinking => true}};
            {Other, _} -> Other
        end,
    case WithReasoning of
        {ok, O1} -> tools(Body, O1);
        E -> E
    end.

%% Responses tools are flat: `{type: function, name, parameters}'.
tools(Body, Opts) ->
    case maps:get(<<"tools">>, Body, []) of
        [] ->
            {ok, Opts};
        Tools when is_list(Tools) ->
            Wrapped = [
                #{
                    <<"type">> => <<"function">>,
                    <<"function">> => maps:with(
                        [<<"name">>, <<"description">>, <<"parameters">>], T
                    )
                }
             || #{<<"type">> := <<"function">>} = T <- Tools
            ],
            case length(Wrapped) =:= length(Tools) of
                false ->
                    bad(<<"only function tools are supported">>, <<"tools">>);
                true ->
                    {ok, Converted} = erllama_server_openai_req:tools(Wrapped),
                    choice(maps:get(<<"tool_choice">>, Body, <<"auto">>), Converted, Body, Opts)
            end;
        _ ->
            bad(<<"tools must be an array">>, <<"tools">>)
    end.

choice(Choice, Tools, Body, Opts) ->
    Par =
        case maps:find(<<"parallel_tool_calls">>, Body) of
            {ok, P} when is_boolean(P) -> #{parallel_tool_calls => P};
            _ -> #{}
        end,
    Set = fun(C, Ts) -> {ok, maps:merge(Opts#{tools => Ts, tool_choice => C}, Par)} end,
    case Choice of
        <<"auto">> ->
            Set(auto, Tools);
        <<"none">> ->
            Set(none, Tools);
        <<"required">> ->
            Set(required, Tools);
        #{<<"type">> := <<"function">>, <<"name">> := N} ->
            case [T || #{name := TN} = T <- Tools, TN =:= N] of
                [T | _] -> Set(required, [T]);
                [] -> bad([<<"tool_choice names an unknown function: ">>, N], <<"tool_choice">>)
            end;
        _ ->
            bad(<<"tool_choice must be auto, none, required or a function">>, <<"tool_choice">>)
    end.

%% -----------------------------------------------------------------------------
%% Responses
%% -----------------------------------------------------------------------------

respond(Spec, #{stream := Stream} = Meta) ->
    case erllama_server_api:start_run(Spec) of
        {ok, #{run_id := Run, conversation_id := Conv}} ->
            Id = erllama_server_http_kit:id(<<"resp_">>, Run),
            Ctx = Meta#{
                id => Id, run => Run, conv => Conv, created => erllama_server_http_kit:now_s()
            },
            case Stream of
                true ->
                    livery_resp:sse(200, fun(Emit) ->
                        erllama_server_openai_responses_stream:stream(Ctx, Emit)
                    end);
                false ->
                    complete(Ctx)
            end;
        {error, R} ->
            erllama_server_http_kit:api_error(openai, R)
    end.

complete(#{run := Run} = Ctx) ->
    case erllama_server_http_kit:drive(Run, fun(_, A) -> {ok, A} end, ok) of
        {done, #{type := error, data := #{error := R}}, _} ->
            erllama_server_http_kit:api_error(openai, R);
        {done, #{data := Result}, _} ->
            Obj = response_object(Ctx, Result),
            ok = keep(Ctx, Obj),
            erllama_server_http_kit:json(200, Obj);
        {gone, _} ->
            erllama_server_http_kit:err(openai, 499, <<"client closed the request">>)
    end.

keep(#{store := true, id := Id, conv := Conv}, Obj) when is_binary(Conv) ->
    _ = erllama_server_responses:put(Id, Obj, Conv),
    ok;
keep(_, _) ->
    ok.

response_object(Ctx, R) ->
    Base = base_object(Ctx, completed),
    Status =
        case maps:get(finish_reason, R, stop) of
            length -> <<"incomplete">>;
            _ -> <<"completed">>
        end,
    Incomplete =
        case Status of
            <<"incomplete">> -> #{<<"reason">> => <<"max_output_tokens">>};
            _ -> null
        end,
    Base#{
        <<"status">> => Status,
        <<"incomplete_details">> => Incomplete,
        <<"output">> => output_items(Ctx, R),
        <<"usage">> => usage(R)
    }.

base_object(#{id := Id, created := Created, echo := Echo} = Ctx, _Status) ->
    maps:merge(
        #{
            <<"instructions">> => null,
            <<"tools">> => [],
            <<"tool_choice">> => <<"auto">>,
            <<"previous_response_id">> => null,
            <<"metadata">> => #{},
            <<"text">> => #{<<"format">> => #{<<"type">> => <<"text">>}}
        },
        Echo#{
            <<"id">> => Id,
            <<"object">> => <<"response">>,
            <<"created_at">> => Created,
            <<"status">> => <<"in_progress">>,
            <<"model">> => model_name(Ctx),
            <<"output">> => [],
            <<"error">> => null,
            <<"incomplete_details">> => null,
            <<"parallel_tool_calls">> => true,
            <<"store">> => maps:get(store, Ctx)
        }
    ).

output_items(#{run := Run}, R) ->
    Reasoning =
        case maps:get(reasoning, R, <<>>) of
            <<>> -> [];
            T -> [reasoning_item(Run, T)]
        end,
    Calls = maps:get(tool_calls, R, []),
    Message =
        case {maps:get(content, R, <<>>), Calls} of
            {<<>>, [_ | _]} -> [];
            {C, _} -> [message_item(Run, C, <<"completed">>)]
        end,
    Reasoning ++ Message ++ [call_item(Run, I, C, <<"completed">>) || {I, C} <- indexed(Calls)].

indexed(L) -> lists:zip(lists:seq(0, length(L) - 1), L).

reasoning_item(Run, Text) ->
    #{
        <<"type">> => <<"reasoning">>,
        <<"id">> => erllama_server_http_kit:id(<<"rs_">>, Run),
        <<"summary">> => [#{<<"type">> => <<"summary_text">>, <<"text">> => Text}]
    }.

message_item(Run, Text, Status) ->
    #{
        <<"type">> => <<"message">>,
        <<"id">> => erllama_server_http_kit:id(<<"msg_">>, Run),
        <<"status">> => Status,
        <<"role">> => <<"assistant">>,
        <<"content">> => [
            #{<<"type">> => <<"output_text">>, <<"text">> => Text, <<"annotations">> => []}
        ]
    }.

call_item(Run, I, #{name := N} = C, Status) ->
    #{
        <<"type">> => <<"function_call">>,
        <<"id">> => <<
            (erllama_server_http_kit:id(<<"fc_">>, Run))/binary, "_", (integer_to_binary(I))/binary
        >>,
        <<"call_id">> => maps:get(id, C, null),
        <<"name">> => N,
        <<"arguments">> => args(C),
        <<"status">> => Status
    }.

args(#{arguments := A}) when is_map(A) -> iolist_to_binary(json:encode(A));
args(#{arguments := A}) when is_binary(A) -> A;
args(_) -> <<>>.

usage(R) ->
    #{prompt_tokens := P, completion_tokens := G} = maps:get(usage, R, #{
        prompt_tokens => 0, completion_tokens => 0
    }),
    #{
        <<"input_tokens">> => P,
        <<"output_tokens">> => G,
        <<"total_tokens">> => P + G,
        <<"input_tokens_details">> => #{<<"cached_tokens">> => 0},
        <<"output_tokens_details">> => #{<<"reasoning_tokens">> => 0}
    }.

model_name(#{model := null, run := Run}) ->
    case erllama_server_api:run(Run) of
        {ok, #{model := M}} -> M;
        _ -> null
    end;
model_name(#{model := M}) ->
    M.
