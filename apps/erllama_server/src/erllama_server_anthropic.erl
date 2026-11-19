%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_anthropic).
-moduledoc false.
%% The Anthropic Messages API: `POST /v1/messages' (streamed as
%% Anthropic's typed server-sent events when `stream' is true) and
%% `POST /v1/messages/count_tokens'. Requests become the same run specs
%% as every other front end; content blocks map onto chat messages:
%% `tool_use' blocks are assistant tool calls, `tool_result' blocks are
%% tool messages answering them.

-export([messages/1, count_tokens/1]).
-export([request/1]).

-define(FIELDS, [
    <<"model">>,
    <<"max_tokens">>,
    <<"messages">>,
    <<"system">>,
    <<"stop_sequences">>,
    <<"stream">>,
    <<"temperature">>,
    <<"top_p">>,
    <<"top_k">>,
    <<"tools">>,
    <<"tool_choice">>,
    <<"thinking">>
]).
-define(COUNT_FIELDS, [
    <<"model">>, <<"messages">>, <<"system">>, <<"tools">>, <<"tool_choice">>, <<"thinking">>
]).
-define(INFO_FIELDS, [<<"metadata">>, <<"service_tier">>]).

%% -----------------------------------------------------------------------------
%% Requests
%% -----------------------------------------------------------------------------

messages(Req) ->
    case erllama_server_http_kit:read_json(Req, anthropic) of
        {ok, Body} ->
            case request(Body) of
                {ok, Spec, Meta} -> respond(Spec, Meta);
                {error, R} -> erllama_server_http_kit:api_error(anthropic, R)
            end;
        {error, Resp} ->
            Resp
    end.

count_tokens(Req) ->
    case erllama_server_http_kit:read_json(Req, anthropic) of
        {ok, Body} ->
            case parse(Body, ?COUNT_FIELDS, false) of
                {ok, #{messages := Msgs, opts := Opts} = Spec, _} -> count(Spec, Msgs, Opts);
                {error, R} -> erllama_server_http_kit:api_error(anthropic, R)
            end;
        {error, Resp} ->
            Resp
    end.

count(Spec, Msgs, Opts) ->
    case erllama_server_models:acquire(maps:get(model, Spec, undefined), self(), undefined) of
        {ok, Id} ->
            try erllama_server_generate:count_tokens(Id, Msgs, Opts) of
                {ok, N} -> erllama_server_http_kit:json(200, #{<<"input_tokens">> => N});
                {error, R} -> erllama_server_http_kit:api_error(anthropic, R)
            after
                erllama_server_models:release(Id, self())
            end;
        {error, R} ->
            erllama_server_http_kit:api_error(anthropic, R)
    end.

%% A Messages request as a run spec, and what the response needs.
-spec request(map()) -> {ok, map(), map()} | {error, term()}.
request(Body) ->
    parse(Body, ?FIELDS, true).

parse(Body, Fields, NeedMax) ->
    Checks = [fields(Body, Fields), max_tokens(Body, NeedMax)],
    case [E || {error, _} = E <- Checks] of
        [] ->
            case {messages_of(Body), opts(Body)} of
                {{ok, Msgs}, {ok, Opts}} ->
                    Spec0 = #{messages => Msgs, opts => Opts},
                    Spec =
                        case maps:find(<<"model">>, Body) of
                            {ok, M} when is_binary(M) -> Spec0#{model => M};
                            _ -> Spec0
                        end,
                    {ok, Spec, #{
                        stream => is_true(maps:get(<<"stream">>, Body, false)),
                        model => maps:get(<<"model">>, Body, null)
                    }};
                {{error, _} = E, _} ->
                    E;
                {_, {error, _} = E} ->
                    E
            end;
        [E | _] ->
            E
    end.

bad(Msg, Param) -> {error, {bad_request, Msg, Param}}.

is_true(true) -> true;
is_true(_) -> false.

fields(Body, Allowed) ->
    case [K || K <- maps:keys(Body), not lists:member(K, Allowed ++ ?INFO_FIELDS)] of
        [] -> ok;
        [K | _] -> bad([<<"unsupported parameter: ">>, K], K)
    end.

max_tokens(_Body, false) -> ok;
max_tokens(#{<<"max_tokens">> := N}, true) when is_integer(N), N > 0 -> ok;
max_tokens(_, true) -> bad(<<"max_tokens is required">>, <<"max_tokens">>).

messages_of(Body) ->
    System =
        case maps:get(<<"system">>, Body, undefined) of
            undefined ->
                {ok, []};
            S when is_binary(S) -> {ok, [#{role => <<"system">>, content => S}]};
            [_ | _] = Blocks ->
                {ok, [#{role => <<"system">>, content => iolist_to_binary(text_of(Blocks))}]};
            _ ->
                bad(<<"system must be a string or text blocks">>, <<"system">>)
        end,
    case {System, maps:get(<<"messages">>, Body, undefined)} of
        {{ok, Sys}, [_ | _] = Msgs} ->
            case collect([message(M) || M <- Msgs]) of
                {ok, Converted} -> {ok, Sys ++ lists:append(Converted)};
                {error, _} = E -> E
            end;
        {{error, _} = E, _} ->
            E;
        {_, _} ->
            bad(<<"messages must be a non-empty array">>, <<"messages">>)
    end.

collect(Results) ->
    case [E || {error, _} = E <- Results] of
        [] -> {ok, [V || {ok, V} <- Results]};
        [E | _] -> E
    end.

text_of(Blocks) -> [T || #{<<"type">> := <<"text">>, <<"text">> := T} <- Blocks].

%% One Anthropic message can become several chat messages: the tool
%% results it carries come first, as `tool' messages, then its text and
%% images.
message(#{<<"role">> := Role, <<"content">> := C}) when is_binary(C) ->
    {ok, [#{role => Role, content => C}]};
message(#{<<"role">> := <<"assistant">>, <<"content">> := Blocks}) when is_list(Blocks) ->
    Text = iolist_to_binary(text_of(Blocks)),
    Calls = [
        #{
            type => function,
            id => Id,
            function => #{name => N, arguments => iolist_to_binary(json:encode(In))}
        }
     || #{<<"type">> := <<"tool_use">>, <<"id">> := Id, <<"name">> := N, <<"input">> := In} <-
            Blocks
    ],
    case Calls of
        [] -> {ok, [#{role => <<"assistant">>, content => Text}]};
        _ -> {ok, [#{role => <<"assistant">>, content => null_if_empty(Text), tool_calls => Calls}]}
    end;
message(#{<<"role">> := <<"user">>, <<"content">> := Blocks}) when is_list(Blocks) ->
    Results = [tool_result(B) || #{<<"type">> := <<"tool_result">>} = B <- Blocks],
    Rest = [B || #{<<"type">> := T} = B <- Blocks, T =/= <<"tool_result">>],
    case collect([part(B) || B <- Rest]) of
        {ok, []} -> {ok, Results};
        {ok, Parts} -> {ok, Results ++ [#{role => <<"user">>, content => Parts}]};
        {error, _} = E -> E
    end;
message(_) ->
    bad(<<"every message needs a role and content">>, <<"messages">>).

null_if_empty(<<>>) -> null;
null_if_empty(T) -> T.

tool_result(#{<<"tool_use_id">> := Id} = B) ->
    Content =
        case maps:get(<<"content">>, B, <<>>) of
            C when is_binary(C) -> C;
            Blocks when is_list(Blocks) -> iolist_to_binary(text_of(Blocks))
        end,
    Text =
        case maps:get(<<"is_error">>, B, false) of
            true -> <<"Error: ", Content/binary>>;
            _ -> Content
        end,
    #{role => <<"tool">>, tool_call_id => Id, content => Text}.

part(#{<<"type">> := <<"text">>, <<"text">> := T}) ->
    {ok, #{type => text, text => T}};
part(#{<<"type">> := <<"image">>, <<"source">> := #{<<"type">> := <<"base64">>, <<"data">> := D}}) ->
    case erllama_server_openai_req:decode64(D) of
        {ok, Bytes} -> {ok, #{type => image, data => Bytes}};
        error -> bad(<<"image data must be base64">>, <<"messages">>)
    end;
part(#{<<"type">> := <<"image">>}) ->
    bad(
        <<"only base64 image sources are supported; the server does not fetch URLs">>,
        <<"messages">>
    );
part(#{<<"type">> := <<"thinking">>}) ->
    %% Earlier reasoning is not fed back to the model.
    {ok, #{type => text, text => <<>>}};
part(#{<<"type">> := T}) ->
    bad([<<"unsupported content block: ">>, T], <<"messages">>).

opts(Body) ->
    Map = [
        {<<"max_tokens">>, response_tokens},
        {<<"temperature">>, temperature},
        {<<"top_p">>, top_p},
        {<<"top_k">>, top_k},
        {<<"stop_sequences">>, stop_sequences}
    ],
    Opts0 = maps:from_list([{K, V} || {F, K} <- Map, {ok, V} <- [maps:find(F, Body)], V =/= null]),
    Opts1 =
        case maps:get(<<"thinking">>, Body, undefined) of
            #{<<"type">> := <<"enabled">>} -> Opts0#{enable_thinking => true};
            #{<<"type">> := <<"disabled">>} -> Opts0#{enable_thinking => false};
            _ -> Opts0
        end,
    tools(maps:get(<<"tools">>, Body, []), maps:get(<<"tool_choice">>, Body, undefined), Opts1).

tools([], _Choice, Opts) ->
    {ok, Opts};
tools(Tools0, Choice, Opts) when is_list(Tools0) ->
    case collect([tool(T) || T <- Tools0]) of
        {ok, Tools} ->
            case tool_choice(Choice, Tools) of
                {ok, C, Selected, Par} ->
                    {ok, maps:merge(Opts#{tools => Selected, tool_choice => C}, Par)};
                {error, _} = E ->
                    E
            end;
        {error, _} = E ->
            E
    end;
tools(_, _, _) ->
    bad(<<"tools must be an array">>, <<"tools">>).

tool(#{<<"name">> := N} = T) when is_binary(N) ->
    Base = #{
        name => N, parameters => maps:get(<<"input_schema">>, T, #{<<"type">> => <<"object">>})
    },
    case maps:find(<<"description">>, T) of
        {ok, D} -> {ok, Base#{description => D}};
        error -> {ok, Base}
    end;
tool(_) ->
    bad(<<"tools need a name">>, <<"tools">>).

tool_choice(undefined, Tools) ->
    {ok, auto, Tools, #{}};
tool_choice(#{<<"type">> := Type} = C, Tools) ->
    Par =
        case maps:get(<<"disable_parallel_tool_use">>, C, false) of
            true -> #{parallel_tool_calls => false};
            _ -> #{parallel_tool_calls => true}
        end,
    case Type of
        <<"auto">> -> {ok, auto, Tools, Par};
        <<"any">> -> {ok, required, Tools, Par};
        <<"none">> -> {ok, none, Tools, #{}};
        <<"tool">> -> named(maps:get(<<"name">>, C, undefined), Tools, Par);
        _ -> bad(<<"tool_choice type must be auto, any, tool or none">>, <<"tool_choice">>)
    end;
tool_choice(_, _) ->
    bad(<<"tool_choice must be an object">>, <<"tool_choice">>).

named(Name, Tools, Par) ->
    case [T || #{name := N} = T <- Tools, N =:= Name] of
        [T | _] -> {ok, required, [T], Par};
        [] -> bad(<<"tool_choice names an unknown tool">>, <<"tool_choice">>)
    end.

%% -----------------------------------------------------------------------------
%% Responses
%% -----------------------------------------------------------------------------

respond(Spec, #{stream := Stream} = Meta) ->
    case erllama_server_http_kit:start_run(Spec, anthropic) of
        {ok, Run} when Stream ->
            livery_resp:sse(200, fun(Emit) -> stream(Run, Meta, Emit) end);
        {ok, Run} ->
            case erllama_server_http_kit:drive(Run, fun(_, Acc) -> {ok, Acc} end, ok) of
                {done, #{type := error, data := #{error := R}}, _} ->
                    erllama_server_http_kit:api_error(anthropic, R);
                {done, #{data := Result}, _} ->
                    erllama_server_http_kit:json(200, message_object(Run, Meta, Result));
                {gone, _} ->
                    erllama_server_http_kit:err(anthropic, 499, <<"client closed the request">>)
            end;
        {error, Resp} ->
            Resp
    end.

message_object(Run, Meta, R) ->
    (envelope(Run, Meta))#{
        <<"content">> => content_blocks(R),
        <<"stop_reason">> => stop_reason(R),
        <<"stop_sequence">> => stop_sequence(R),
        <<"usage">> => usage(R)
    }.

envelope(Run, Meta) ->
    #{
        <<"id">> => erllama_server_http_kit:id(<<"msg_">>, Run),
        <<"type">> => <<"message">>,
        <<"role">> => <<"assistant">>,
        <<"model">> => model_name(Run, Meta)
    }.

content_blocks(R) ->
    Thinking =
        case maps:get(reasoning, R, <<>>) of
            <<>> -> [];
            T -> [#{<<"type">> => <<"thinking">>, <<"thinking">> => T, <<"signature">> => <<>>}]
        end,
    Text =
        case maps:get(content, R, <<>>) of
            <<>> -> [];
            C -> [#{<<"type">> => <<"text">>, <<"text">> => C}]
        end,
    Thinking ++ Text ++ [tool_use(Call) || Call <- maps:get(tool_calls, R, [])].

tool_use(#{name := N, arguments := A} = C) ->
    #{
        <<"type">> => <<"tool_use">>,
        <<"id">> => maps:get(id, C, null),
        <<"name">> => N,
        <<"input">> => A
    }.

stop_reason(#{tool_calls := [_ | _]}) -> <<"tool_use">>;
stop_reason(#{finish_reason := length}) -> <<"max_tokens">>;
stop_reason(#{stop_sequence := S}) when is_binary(S) -> <<"stop_sequence">>;
stop_reason(_) -> <<"end_turn">>.

stop_sequence(#{stop_sequence := S}) when is_binary(S) -> S;
stop_sequence(_) -> null.

usage(R) ->
    #{prompt_tokens := P, completion_tokens := G} = maps:get(usage, R, #{
        prompt_tokens => 0, completion_tokens => 0
    }),
    #{<<"input_tokens">> => P, <<"output_tokens">> => G}.

model_name(Run, #{model := null}) ->
    case erllama_server_api:run(Run) of
        {ok, #{model := M}} -> M;
        _ -> null
    end;
model_name(_Run, #{model := M}) ->
    M.

%% -----------------------------------------------------------------------------
%% Streaming
%% -----------------------------------------------------------------------------

%% Text and reasoning stream as content blocks; a tool_use block is
%% sent whole once its arguments are known (with the final parse).
stream(Run, Meta, Emit) ->
    Send = fun(Event, Data, Acc) ->
        case Emit(#{event => Event, data => json:encode(Data)}) of
            ok -> {ok, Acc};
            {error, _} -> {stop, Acc}
        end
    end,
    Fun = fun(E, Acc) -> event(Run, Meta, Send, E, Acc) end,
    _ = erllama_server_http_kit:drive(Run, Fun, #{block => none, index => 0}),
    ok.

event(Run, Meta, Send, #{type := started}, Acc) ->
    Msg = (envelope(Run, Meta))#{
        <<"content">> => [],
        <<"stop_reason">> => null,
        <<"stop_sequence">> => null,
        <<"usage">> => #{<<"input_tokens">> => 0, <<"output_tokens">> => 0}
    },
    Send(<<"message_start">>, #{<<"type">> => <<"message_start">>, <<"message">> => Msg}, Acc);
event(_Run, _Meta, Send, #{type := token, data := D}, Acc) ->
    delta(Send, text, #{<<"type">> => <<"text_delta">>, <<"text">> => D}, Acc);
event(_Run, _Meta, Send, #{type := thinking, data := D}, Acc) ->
    delta(Send, thinking, #{<<"type">> => <<"thinking_delta">>, <<"thinking">> => D}, Acc);
event(_Run, _Meta, Send, #{type := T, data := R}, Acc) when T =:= done; T =:= cancelled ->
    finish(Send, R, Acc);
event(_Run, _Meta, Send, #{type := error, data := #{error := Reason}}, Acc) ->
    Send(
        <<"error">>,
        #{
            <<"type">> => <<"error">>,
            <<"error">> => #{
                <<"type">> => <<"api_error">>,
                <<"message">> => iolist_to_binary(io_lib:format("~0p", [Reason]))
            }
        },
        Acc
    );
event(_Run, _Meta, _Send, _E, Acc) ->
    {ok, Acc}.

%% A delta for a `Kind' block, opening one (and closing the previous
%% one) when the kind changes.
delta(Send, Kind, Delta, #{block := Kind, index := I} = Acc) ->
    Send(
        <<"content_block_delta">>,
        #{<<"type">> => <<"content_block_delta">>, <<"index">> => I, <<"delta">> => Delta},
        Acc
    );
delta(Send, Kind, Delta, Acc) ->
    then(close_block(Send, Acc), fun(Acc1) ->
        #{index := I} = Acc1,
        Block =
            case Kind of
                text -> #{<<"type">> => <<"text">>, <<"text">> => <<>>};
                thinking -> #{<<"type">> => <<"thinking">>, <<"thinking">> => <<>>}
            end,
        then(
            Send(
                <<"content_block_start">>,
                #{
                    <<"type">> => <<"content_block_start">>,
                    <<"index">> => I,
                    <<"content_block">> => Block
                },
                Acc1#{block := Kind}
            ),
            fun(Acc2) -> delta(Send, Kind, Delta, Acc2) end
        )
    end).

close_block(_Send, #{block := none} = Acc) ->
    {ok, Acc};
close_block(Send, #{index := I} = Acc) ->
    Send(
        <<"content_block_stop">>, #{<<"type">> => <<"content_block_stop">>, <<"index">> => I}, Acc#{
            block := none, index := I + 1
        }
    ).

then({ok, Acc}, Fun) -> Fun(Acc);
then(Stop, _Fun) -> Stop.

finish(Send, R, Acc0) ->
    Steps =
        [fun(Acc) -> close_block(Send, Acc) end] ++
            [fun(Acc) -> tool_block(Send, Call, Acc) end || Call <- maps:get(tool_calls, R, [])] ++
            [
                fun(Acc) ->
                    Send(
                        <<"message_delta">>,
                        #{
                            <<"type">> => <<"message_delta">>,
                            <<"delta">> => #{
                                <<"stop_reason">> => stop_reason(R),
                                <<"stop_sequence">> => stop_sequence(R)
                            },
                            <<"usage">> => #{
                                <<"output_tokens">> => maps:get(<<"output_tokens">>, usage(R))
                            }
                        },
                        Acc
                    )
                end,
                fun(Acc) -> Send(<<"message_stop">>, #{<<"type">> => <<"message_stop">>}, Acc) end
            ],
    lists:foldl(fun(Step, Res) -> then(Res, Step) end, {ok, Acc0}, Steps).

tool_block(Send, #{arguments := A} = Call, #{index := I} = Acc) ->
    Start = (tool_use(Call))#{<<"input">> => #{}},
    Json = iolist_to_binary(json:encode(A)),
    then(
        Send(
            <<"content_block_start">>,
            #{
                <<"type">> => <<"content_block_start">>,
                <<"index">> => I,
                <<"content_block">> => Start
            },
            Acc
        ),
        fun(Acc1) ->
            then(
                Send(
                    <<"content_block_delta">>,
                    #{
                        <<"type">> => <<"content_block_delta">>,
                        <<"index">> => I,
                        <<"delta">> => #{
                            <<"type">> => <<"input_json_delta">>, <<"partial_json">> => Json
                        }
                    },
                    Acc1
                ),
                fun(Acc2) ->
                    Send(
                        <<"content_block_stop">>,
                        #{<<"type">> => <<"content_block_stop">>, <<"index">> => I},
                        Acc2#{index := I + 1}
                    )
                end
            )
        end
    ).
