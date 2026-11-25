%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_openai_responses_stream).
-moduledoc false.
%% A Responses request streamed as OpenAI's typed server-sent events:
%% `response.created', `response.in_progress', the output text item
%% (`output_item.added', `content_part.added', `output_text.delta' ...
%% `.done'), each function call whole (`function_call_arguments.delta'
%% and `.done'), then `response.completed' (or `response.failed').
%% Every event carries its `sequence_number'.

-export([stream/2]).

stream(#{run := Run} = Ctx, Emit) ->
    Send = fun(Type, Data, Acc = #{seq := N}) ->
        Payload = Data#{<<"type">> => Type, <<"sequence_number">> => N},
        case Emit(#{event => Type, data => json:encode(Payload)}) of
            ok -> {ok, Acc#{seq := N + 1}};
            {error, _} -> {stop, Acc}
        end
    end,
    Fun = fun(E, Acc) -> event(Ctx, Send, E, Acc) end,
    _ = erllama_server_http_kit:drive(Run, Fun, #{seq => 0, text => none}),
    ok.

event(Ctx, Send, #{type := started}, Acc) ->
    Obj = erllama_server_openai_responses:base_object(Ctx, in_progress),
    then(Send(<<"response.created">>, #{<<"response">> => Obj}, Acc), fun(A) ->
        Send(<<"response.in_progress">>, #{<<"response">> => Obj}, A)
    end);
event(#{run := Run}, Send, #{type := token, data := D}, #{text := none} = Acc) ->
    Item = erllama_server_openai_responses:message_item(Run, <<>>, <<"in_progress">>),
    Part = #{<<"type">> => <<"output_text">>, <<"text">> => <<>>, <<"annotations">> => []},
    ItemId = maps:get(<<"id">>, Item),
    Steps = [
        fun(A) ->
            Send(
                <<"response.output_item.added">>,
                #{<<"output_index">> => 0, <<"item">> => maps:put(<<"content">>, [], Item)},
                A
            )
        end,
        fun(A) ->
            Send(
                <<"response.content_part.added">>,
                #{
                    <<"item_id">> => ItemId,
                    <<"output_index">> => 0,
                    <<"content_index">> => 0,
                    <<"part">> => Part
                },
                A
            )
        end,
        fun(A) -> text_delta(Send, ItemId, D, A) end
    ],
    run_steps(Steps, Acc#{text := {ItemId, D}});
event(_Ctx, Send, #{type := token, data := D}, #{text := {ItemId, T}} = Acc) ->
    text_delta(Send, ItemId, D, Acc#{text := {ItemId, <<T/binary, D/binary>>}});
event(Ctx, Send, #{type := T, data := R}, Acc) when T =:= done; T =:= cancelled ->
    finish(Ctx, Send, R, Acc);
event(Ctx, Send, #{type := error, data := #{error := Reason}}, Acc) ->
    Obj = (erllama_server_openai_responses:base_object(Ctx, failed))#{
        <<"status">> => <<"failed">>,
        <<"error">> => #{
            <<"code">> => <<"server_error">>,
            <<"message">> => iolist_to_binary(io_lib:format("~0p", [Reason]))
        }
    },
    Send(<<"response.failed">>, #{<<"response">> => Obj}, Acc);
event(_Ctx, _Send, _E, Acc) ->
    {ok, Acc}.

text_delta(Send, ItemId, D, Acc) ->
    Send(
        <<"response.output_text.delta">>,
        #{
            <<"item_id">> => ItemId,
            <<"output_index">> => 0,
            <<"content_index">> => 0,
            <<"delta">> => D
        },
        Acc
    ).

%% Close the text item, send each function call whole, then the
%% completed response (stored when asked).
finish(#{run := Run} = Ctx, Send, R, Acc) ->
    TextSteps =
        case maps:get(text, Acc) of
            none ->
                [];
            {ItemId, Text} ->
                Done = erllama_server_openai_responses:message_item(Run, Text, <<"completed">>),
                Part = #{
                    <<"type">> => <<"output_text">>, <<"text">> => Text, <<"annotations">> => []
                },
                [
                    fun(A) ->
                        Send(
                            <<"response.output_text.done">>,
                            #{
                                <<"item_id">> => ItemId,
                                <<"output_index">> => 0,
                                <<"content_index">> => 0,
                                <<"text">> => Text
                            },
                            A
                        )
                    end,
                    fun(A) ->
                        Send(
                            <<"response.content_part.done">>,
                            #{
                                <<"item_id">> => ItemId,
                                <<"output_index">> => 0,
                                <<"content_index">> => 0,
                                <<"part">> => Part
                            },
                            A
                        )
                    end,
                    fun(A) ->
                        Send(
                            <<"response.output_item.done">>,
                            #{<<"output_index">> => 0, <<"item">> => Done},
                            A
                        )
                    end
                ]
        end,
    Offset =
        case TextSteps of
            [] -> 0;
            _ -> 1
        end,
    CallSteps = lists:append([
        call_steps(Run, Send, I, I + Offset, C)
     || {I, C} <- erllama_server_openai_responses:indexed(maps:get(tool_calls, R, []))
    ]),
    Obj = erllama_server_openai_responses:response_object(Ctx, R),
    ok = erllama_server_openai_responses:keep(Ctx, Obj),
    run_steps(
        TextSteps ++ CallSteps ++
            [fun(A) -> Send(<<"response.completed">>, #{<<"response">> => Obj}, A) end],
        Acc
    ).

call_steps(Run, Send, I, Index, Call) ->
    Item = erllama_server_openai_responses:call_item(Run, I, Call, <<"completed">>),
    #{<<"id">> := ItemId, <<"arguments">> := Args} = Item,
    [
        fun(A) ->
            Send(
                <<"response.output_item.added">>,
                #{
                    <<"output_index">> => Index,
                    <<"item">> => Item#{<<"arguments">> => <<>>, <<"status">> => <<"in_progress">>}
                },
                A
            )
        end,
        fun(A) ->
            Send(
                <<"response.function_call_arguments.delta">>,
                #{<<"item_id">> => ItemId, <<"output_index">> => Index, <<"delta">> => Args},
                A
            )
        end,
        fun(A) ->
            Send(
                <<"response.function_call_arguments.done">>,
                #{<<"item_id">> => ItemId, <<"output_index">> => Index, <<"arguments">> => Args},
                A
            )
        end,
        fun(A) ->
            Send(
                <<"response.output_item.done">>,
                #{<<"output_index">> => Index, <<"item">> => Item},
                A
            )
        end
    ].

run_steps(Steps, Acc) ->
    lists:foldl(fun(Step, Res) -> then(Res, Step) end, {ok, Acc}, Steps).

then({ok, Acc}, Fun) -> Fun(Acc);
then(Stop, _Fun) -> Stop.
