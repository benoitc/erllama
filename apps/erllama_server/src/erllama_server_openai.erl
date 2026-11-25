%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_openai).
-moduledoc false.
%% The OpenAI-compatible HTTP API: `/v1/models', `/v1/chat/completions',
%% `/v1/completions' and `/v1/embeddings'. Each handler turns the
%% request into a run spec (or an embedding call), runs it through the
%% service API and writes the answer in OpenAI's shape, streamed as
%% server-sent events when asked.
%%
%% Fields are strict: each endpoint accepts the fields it implements
%% plus a few informational ones (`user', `metadata', `store'), and
%% refuses any other with a 400 naming it rather than ignoring it.

-export([models/1, model/1, chat_completions/1, completions/1, embeddings/1]).

%% -----------------------------------------------------------------------------
%% Models
%% -----------------------------------------------------------------------------

models(_Req) ->
    {ok, Models} = erllama_server_api:models(),
    Data = [model_object(M) || M <- Models],
    erllama_server_http_kit:json(200, #{<<"object">> => <<"list">>, <<"data">> => Data}).

model(Req) ->
    Id = livery_req:binding(<<"id">>, Req),
    case erllama_server_api:model(Id) of
        {ok, M} -> erllama_server_http_kit:json(200, model_object(M));
        {error, R} -> erllama_server_http_kit:api_error(openai, R)
    end.

model_object(#{name := Name} = M) ->
    #{
        <<"id">> => Name,
        <<"object">> => <<"model">>,
        <<"created">> => maps:get(created_at, M, 0) div 1000,
        <<"owned_by">> => <<"erllama">>
    }.

%% -----------------------------------------------------------------------------
%% Requests
%% -----------------------------------------------------------------------------

chat_completions(Req) ->
    with_body(Req, fun erllama_server_openai_req:chat_request/1, fun(Spec, Meta) ->
        respond(chat, Spec, Meta)
    end).

completions(Req) ->
    with_body(Req, fun erllama_server_openai_req:completion_request/1, fun(Spec, Meta) ->
        respond(text, Spec, Meta)
    end).

embeddings(Req) ->
    with_body(Req, fun erllama_server_openai_req:embedding_request/1, fun embed/2).

with_body(Req, Parse, Run) ->
    case erllama_server_http_kit:read_json(Req, openai) of
        {ok, Body} ->
            case Parse(Body) of
                {ok, Spec, Meta} -> Run(Spec, Meta);
                {error, Reason} -> erllama_server_http_kit:api_error(openai, Reason)
            end;
        {error, Resp} ->
            Resp
    end.

%% -----------------------------------------------------------------------------
%% Responses
%% -----------------------------------------------------------------------------

respond(Kind, Spec, #{stream := Stream} = Meta) ->
    case erllama_server_http_kit:start_run(Spec, openai) of
        {ok, Run} when Stream ->
            livery_resp:sse(200, fun(Emit) -> stream(Kind, Run, Meta, Emit) end);
        {ok, Run} ->
            complete(Kind, Run, Meta);
        {error, Resp} ->
            Resp
    end.

complete(Kind, Run, Meta) ->
    Collect = fun
        (#{type := logprobs, data := E}, Acc) -> {ok, [E | Acc]};
        (_, Acc) -> {ok, Acc}
    end,
    case erllama_server_http_kit:drive(Run, Collect, []) of
        {done, #{type := error, data := #{error := Reason}}, _} ->
            erllama_server_http_kit:api_error(openai, Reason);
        {done, #{data := Result}, Logprobs} ->
            erllama_server_http_kit:json(
                200, completion_object(Kind, Run, Meta, Result, lists:reverse(Logprobs))
            );
        {gone, _} ->
            erllama_server_http_kit:err(openai, 499, <<"client closed the request">>)
    end.

completion_object(chat, Run, Meta, R, Logprobs) ->
    Msg0 = #{<<"role">> => <<"assistant">>, <<"content">> => content_or_null(R)},
    Msg1 =
        case maps:get(reasoning, R, <<>>) of
            <<>> -> Msg0;
            Reasoning -> Msg0#{<<"reasoning_content">> => Reasoning}
        end,
    Msg =
        case maps:get(tool_calls, R, []) of
            [] -> Msg1;
            Calls -> Msg1#{<<"tool_calls">> => [call_object(I, C) || {I, C} <- indexed(Calls)]}
        end,
    Choice = #{
        <<"index">> => 0,
        <<"message">> => Msg,
        <<"finish_reason">> => finish_reason(R),
        <<"logprobs">> => logprobs_object(Meta, Run, Logprobs)
    },
    with_speculative(
        R, envelope(<<"chat.completion">>, <<"chatcmpl-">>, Run, Meta, [Choice], usage(R))
    );
completion_object(text, Run, Meta, R, Logprobs) ->
    Choice = #{
        <<"index">> => 0,
        <<"text">> => maps:get(content, R, <<>>),
        <<"finish_reason">> => finish_reason(R),
        <<"logprobs">> => logprobs_object(Meta, Run, Logprobs)
    },
    envelope(<<"text_completion">>, <<"cmpl-">>, Run, Meta, [Choice], usage(R)).

envelope(Object, Prefix, Run, Meta, Choices, Usage) ->
    #{
        <<"id">> => erllama_server_http_kit:id(Prefix, Run),
        <<"object">> => Object,
        <<"created">> => erllama_server_http_kit:now_s(),
        <<"model">> => model_name(Run, Meta),
        <<"choices">> => Choices,
        <<"usage">> => Usage
    }.

content_or_null(#{content := <<>>, tool_calls := [_ | _]}) -> null;
content_or_null(#{content := C}) -> C.

indexed(L) -> lists:zip(lists:seq(0, length(L) - 1), L).

call_object(I, #{name := N, arguments := A} = C) ->
    #{
        <<"index">> => I,
        <<"id">> => call_id(C),
        <<"type">> => <<"function">>,
        <<"function">> => #{<<"name">> => N, <<"arguments">> => args_json(A)}
    }.

call_id(#{id := Id}) when is_binary(Id) -> Id;
call_id(_) -> null.

args_json(A) when is_map(A) -> iolist_to_binary(json:encode(A));
args_json(A) when is_binary(A) -> A.

-spec finish_reason(map()) -> binary().
finish_reason(#{tool_calls := [_ | _]}) -> <<"tool_calls">>;
finish_reason(#{finish_reason := length}) -> <<"length">>;
finish_reason(_) -> <<"stop">>.

usage(R) ->
    #{prompt_tokens := P, completion_tokens := G} = maps:get(usage, R, #{
        prompt_tokens => 0, completion_tokens => 0
    }),
    #{<<"prompt_tokens">> => P, <<"completion_tokens">> => G, <<"total_tokens">> => P + G}.

model_name(Run, #{model := null}) ->
    case erllama_server_api:run(Run) of
        {ok, #{model := M}} -> M;
        _ -> null
    end;
model_name(_Run, #{model := M}) ->
    M.

%% Token ids from the engine, as the token strings OpenAI reports.
logprobs_object(#{logprobs := false}, _Run, _Entries) ->
    null;
logprobs_object(_Meta, Run, Entries) ->
    Model =
        case erllama_server_api:run(Run) of
            {ok, #{model := M}} -> M;
            _ -> undefined
        end,
    #{<<"content">> => [logprob_entry(Model, E) || E <- Entries]}.

logprob_entry(Model, #{token_id := T, logprob := L, top := Top}) ->
    Tok = token_text(Model, T),
    #{
        <<"token">> => Tok,
        <<"logprob">> => L,
        <<"bytes">> => binary_to_list(Tok),
        <<"top_logprobs">> => [
            #{
                <<"token">> => token_text(Model, Id),
                <<"logprob">> => LP,
                <<"bytes">> => binary_to_list(token_text(Model, Id))
            }
         || {Id, LP} <- Top
        ]
    }.

token_text(undefined, T) ->
    integer_to_binary(T);
token_text(Model, T) ->
    case erllama_server_api:detokenize(Model, [T]) of
        {ok, B} -> B;
        _ -> <<>>
    end.

%% -----------------------------------------------------------------------------
%% Streaming
%% -----------------------------------------------------------------------------

stream(Kind, Run, Meta, Emit) ->
    Send = fun(Obj, Acc) ->
        case Emit(#{data => json:encode(Obj)}) of
            ok -> {ok, Acc};
            {error, _} -> {stop, Acc}
        end
    end,
    Fun = fun(E, Acc) -> stream_event(Kind, Run, Meta, Send, E, Acc) end,
    case erllama_server_http_kit:drive(Run, Fun, #{announced => []}) of
        {done, _, _} -> Emit(#{data => <<"[DONE]">>});
        {gone, _} -> ok
    end.

stream_event(chat, Run, Meta, Send, #{type := started}, Acc) ->
    Send(
        chunk(chat, Run, Meta, #{<<"role">> => <<"assistant">>, <<"content">> => <<>>}, null), Acc
    );
stream_event(Kind, Run, Meta, Send, #{type := token, data := D}, Acc) ->
    Send(chunk(Kind, Run, Meta, text_delta(Kind, D), null), Acc);
stream_event(chat, Run, Meta, Send, #{type := thinking, data := D}, Acc) ->
    Send(chunk(chat, Run, Meta, #{<<"reasoning_content">> => D}, null), Acc);
stream_event(chat, Run, Meta, Send, #{type := tool_call, data := Call}, Acc) ->
    announce(Run, Meta, Send, Call, Acc);
stream_event(_Kind, _Run, _Meta, _Send, #{type := T}, Acc) when T =:= loading; T =:= logprobs ->
    {ok, Acc};
stream_event(Kind, Run, Meta, Send, #{type := Type, data := R}, Acc) when
    Type =:= done; Type =:= cancelled
->
    final(Kind, Run, Meta, Send, R, Acc);
stream_event(_Kind, _Run, _Meta, Send, #{type := error, data := #{error := Reason}}, Acc) ->
    Send(
        #{
            <<"error">> => #{
                <<"message">> => iolist_to_binary(io_lib:format("~0p", [Reason])),
                <<"type">> => <<"server_error">>
            }
        },
        Acc
    );
stream_event(_Kind, _Run, _Meta, _Send, _E, Acc) ->
    {ok, Acc}.

%% A tool call the parser recognised: its id and name now, its
%% arguments with the final chunk.
announce(Run, Meta, Send, #{index := I, id := Id, name := N}, Acc) ->
    Delta = #{
        <<"tool_calls">> => [
            #{
                <<"index">> => I,
                <<"id">> => Id,
                <<"type">> => <<"function">>,
                <<"function">> => #{<<"name">> => N, <<"arguments">> => <<>>}
            }
        ]
    },
    Send(chunk(chat, Run, Meta, Delta, null), Acc#{announced := [I | maps:get(announced, Acc)]}).

text_delta(chat, D) -> #{<<"content">> => D};
text_delta(text, D) -> D.

%% The arguments of each tool call (and any call the stream did not
%% announce), the finish reason, then the usage when asked.
final(Kind, Run, Meta, Send, R, Acc) ->
    Announced = maps:get(announced, Acc),
    Args = [call_args(I, C, Announced) || {I, C} <- indexed(maps:get(tool_calls, R, []))],
    Steps =
        [chunk(chat, Run, Meta, #{<<"tool_calls">> => [A]}, null) || Kind =:= chat, A <- Args] ++
            [chunk(Kind, Run, Meta, end_delta(Kind), finish_reason(R))] ++
            [usage_chunk(Kind, Run, Meta, R) || maps:get(include_usage, Meta)],
    lists:foldl(
        fun
            (Obj, {ok, A0}) -> Send(Obj, A0);
            (_Obj, Stop) -> Stop
        end,
        {ok, Acc},
        Steps
    ).

%% A call's arguments; a call the stream never announced (its name came
%% only with the final parse) gets its id and name here too.
call_args(I, #{arguments := A} = C, Announced) ->
    case lists:member(I, Announced) of
        true -> #{<<"index">> => I, <<"function">> => #{<<"arguments">> => args_json(A)}};
        false -> call_object(I, C)
    end.

end_delta(chat) -> #{};
end_delta(text) -> <<>>.

chunk(chat, Run, Meta, Delta, Finish) ->
    (base_chunk(<<"chat.completion.chunk">>, <<"chatcmpl-">>, Run, Meta))#{
        <<"choices">> => [#{<<"index">> => 0, <<"delta">> => Delta, <<"finish_reason">> => Finish}]
    };
chunk(text, Run, Meta, Text, Finish) ->
    (base_chunk(<<"text_completion">>, <<"cmpl-">>, Run, Meta))#{
        <<"choices">> => [
            #{
                <<"index">> => 0,
                <<"text">> => Text,
                <<"finish_reason">> => Finish,
                <<"logprobs">> => null
            }
        ]
    }.

usage_chunk(Kind, Run, Meta, R) ->
    Object =
        case Kind of
            chat -> <<"chat.completion.chunk">>;
            text -> <<"text_completion">>
        end,
    Prefix =
        case Kind of
            chat -> <<"chatcmpl-">>;
            text -> <<"cmpl-">>
        end,
    with_speculative(R, (base_chunk(Object, Prefix, Run, Meta))#{
        <<"choices">> => [], <<"usage">> => usage(R)
    }).

%% erllama's own field, beside `usage', when the request asked for
%% speculative decoding: the tokens it drafted and had accepted, and the
%% steps it drafted in.
with_speculative(#{speculative := Spec}, Obj) ->
    Obj#{
        <<"speculative">> => maps:from_list([
            {atom_to_binary(K), V}
         || {K, V} <- maps:to_list(maps:with([drafted, accepted, rounds], Spec))
        ])
    };
with_speculative(_R, Obj) ->
    Obj.

base_chunk(Object, Prefix, Run, Meta) ->
    #{
        <<"id">> => erllama_server_http_kit:id(Prefix, Run),
        <<"object">> => Object,
        <<"created">> => erllama_server_http_kit:now_s(),
        <<"model">> => model_name(Run, Meta)
    }.

%% -----------------------------------------------------------------------------
%% Embeddings
%% -----------------------------------------------------------------------------

embed(#{inputs := Inputs} = Spec, #{format := Format} = Meta) ->
    Opts = maps:with([keep_alive], Spec),
    case erllama_server_api:embed(maps:get(model, Spec, undefined), Inputs, Opts) of
        {ok, #{model := Id, vectors := Vectors, prompt_tokens := Tokens}} ->
            Data = [
                #{
                    <<"object">> => <<"embedding">>,
                    <<"index">> => I,
                    <<"embedding">> => encode_vector(Format, V)
                }
             || {I, V} <- indexed(Vectors)
            ],
            Model =
                case maps:get(model, Meta) of
                    null -> Id;
                    M -> M
                end,
            erllama_server_http_kit:json(200, #{
                <<"object">> => <<"list">>,
                <<"data">> => Data,
                <<"model">> => Model,
                <<"usage">> => #{<<"prompt_tokens">> => Tokens, <<"total_tokens">> => Tokens}
            });
        {error, Reason} ->
            erllama_server_http_kit:api_error(openai, Reason)
    end.

encode_vector(<<"float">>, V) -> V;
encode_vector(<<"base64">>, V) -> base64:encode(<<<<F:32/float-little>> || F <- V>>).
