%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_openai_req).
-moduledoc false.
%% OpenAI request bodies as service API run specs: which fields each
%% endpoint honours (any other is refused with a 400 naming it), chat
%% messages and their content parts, tools and tool choice, response
%% formats, sampling and length. The Anthropic and Ollama adapters reuse
%% the message and duration helpers.

-export([
    chat_request/1, completion_request/1, embedding_request/1, messages/1, tools/1, duration_ms/1
]).
-export([data_url/1, decode64/1, sampling/1]).

-define(CHAT_FIELDS, [
    <<"model">>,
    <<"messages">>,
    <<"tools">>,
    <<"tool_choice">>,
    <<"parallel_tool_calls">>,
    <<"response_format">>,
    <<"stream">>,
    <<"stream_options">>,
    <<"max_tokens">>,
    <<"max_completion_tokens">>,
    <<"temperature">>,
    <<"top_p">>,
    <<"seed">>,
    <<"stop">>,
    <<"frequency_penalty">>,
    <<"presence_penalty">>,
    <<"logprobs">>,
    <<"top_logprobs">>,
    <<"n">>,
    <<"keep_alive">>,
    <<"reasoning_effort">>
]).
-define(COMPLETION_FIELDS, [
    <<"model">>,
    <<"prompt">>,
    <<"stream">>,
    <<"stream_options">>,
    <<"max_tokens">>,
    <<"temperature">>,
    <<"top_p">>,
    <<"seed">>,
    <<"stop">>,
    <<"frequency_penalty">>,
    <<"presence_penalty">>,
    <<"logprobs">>,
    <<"n">>,
    <<"echo">>,
    <<"keep_alive">>
]).
-define(EMBED_FIELDS, [<<"model">>, <<"input">>, <<"encoding_format">>, <<"keep_alive">>]).
-define(INFO_FIELDS, [<<"user">>, <<"metadata">>, <<"store">>]).

%% A chat completion request as a run spec, and what the response needs
%% (stream or not, usage in the stream, logprobs).
-spec chat_request(map()) -> {ok, map(), map()} | {error, term()}.
chat_request(Body) ->
    case all([fields(Body, ?CHAT_FIELDS), single_choice(Body)]) of
        ok ->
            case {messages(maps:get(<<"messages">>, Body, undefined)), chat_opts(Body)} of
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

-spec completion_request(map()) -> {ok, map(), map()} | {error, term()}.
completion_request(Body) ->
    Checks = [
        fields(Body, ?COMPLETION_FIELDS),
        single_choice(Body),
        no_echo(Body)
    ],
    case all(Checks) of
        ok ->
            case {prompt(maps:get(<<"prompt">>, Body, undefined)), sampling(Body)} of
                {{ok, P}, {ok, Opts}} -> {ok, spec(Body, #{prompt => P, opts => Opts}), meta(Body)};
                {{error, _} = E, _} -> E;
                {_, {error, _} = E} -> E
            end;
        {error, _} = E ->
            E
    end.

-spec embedding_request(map()) -> {ok, map(), map()} | {error, term()}.
embedding_request(Body) ->
    case fields(Body, ?EMBED_FIELDS) of
        ok ->
            Format = maps:get(<<"encoding_format">>, Body, <<"float">>),
            case
                {
                    inputs(maps:get(<<"input">>, Body, undefined)),
                    lists:member(Format, [<<"float">>, <<"base64">>])
                }
            of
                {{ok, Inputs}, true} ->
                    {ok, spec(Body, #{inputs => Inputs}), #{
                        format => Format, model => maps:get(<<"model">>, Body, null)
                    }};
                {{error, _} = E, _} ->
                    E;
                {_, false} ->
                    bad(<<"encoding_format must be float or base64">>, <<"encoding_format">>)
            end;
        {error, _} = E ->
            E
    end.

all([]) -> ok;
all([ok | Rest]) -> all(Rest);
all([{error, _} = E | _]) -> E.

bad(Msg, Param) -> {error, {bad_request, Msg, Param}}.

fields(Body, Allowed) ->
    case [K || K <- maps:keys(Body), not lists:member(K, Allowed ++ ?INFO_FIELDS)] of
        [] -> ok;
        [K | _] -> bad([<<"unsupported parameter: ">>, K], K)
    end.

single_choice(#{<<"n">> := N}) when N =/= 1 -> bad(<<"only n = 1 is supported">>, <<"n">>);
single_choice(_) -> ok.

no_echo(#{<<"echo">> := true}) -> bad(<<"echo is not supported">>, <<"echo">>);
no_echo(_) -> ok.

spec(Body, Spec0) ->
    Spec1 =
        case maps:find(<<"model">>, Body) of
            {ok, M} when is_binary(M) -> Spec0#{model => M};
            _ -> Spec0
        end,
    case maps:find(<<"keep_alive">>, Body) of
        {ok, KA} ->
            case duration_ms(KA) of
                {ok, Ms} -> Spec1#{keep_alive => Ms};
                error -> Spec1
            end;
        error ->
            Spec1
    end.

meta(Body) ->
    #{
        stream => flag(maps:get(<<"stream">>, Body, false)),
        include_usage => include_usage(Body),
        logprobs => flag(maps:get(<<"logprobs">>, Body, false)),
        model => maps:get(<<"model">>, Body, null)
    }.

flag(true) -> true;
flag(_) -> false.

include_usage(#{<<"stream_options">> := #{<<"include_usage">> := true}}) -> true;
include_usage(_) -> false.

%% Ollama-style keep-alive: seconds as a number, a duration string
%% ("30s", "5m", "1h"), and a negative value for "keep loaded".
-spec duration_ms(term()) -> {ok, non_neg_integer() | infinity} | error.
duration_ms(N) when is_number(N), N < 0 -> {ok, infinity};
duration_ms(N) when is_number(N) -> {ok, round(N * 1000)};
duration_ms(B) when is_binary(B) ->
    case re:run(B, <<"^(-?[0-9.]+)(ms|s|m|h)?$">>, [{capture, all_but_first, binary}]) of
        {match, [Num | Unit]} ->
            V = binary_to_number(Num),
            case {V < 0, Unit} of
                {true, _} -> {ok, infinity};
                {false, [<<"ms">>]} -> {ok, round(V)};
                {false, [<<"m">>]} -> {ok, round(V * 60000)};
                {false, [<<"h">>]} -> {ok, round(V * 3600000)};
                {false, _} -> {ok, round(V * 1000)}
            end;
        nomatch ->
            error
    end;
duration_ms(_) ->
    error.

binary_to_number(B) ->
    try
        binary_to_integer(B)
    catch
        error:badarg -> binary_to_float(B)
    end.

%% -----------------------------------------------------------------------------
%% Messages
%% -----------------------------------------------------------------------------

%% OpenAI chat messages as erllama_server_generate messages.
-spec messages(term()) -> {ok, [map()]} | {error, term()}.
messages([_ | _] = Msgs) ->
    collect([message(M) || M <- Msgs]);
messages(_) ->
    bad(<<"messages must be a non-empty array">>, <<"messages">>).

collect(Results) ->
    case [E || {error, _} = E <- Results] of
        [] -> {ok, [V || {ok, V} <- Results]};
        [E | _] -> E
    end.

message(#{<<"role">> := Role} = M) when is_binary(Role) ->
    case content(maps:get(<<"content">>, M, absent)) of
        {ok, C} ->
            Base0 = #{role => Role},
            Base =
                case C of
                    absent -> Base0;
                    _ -> Base0#{content => C}
                end,
            with_extras(M, Base);
        {error, _} = E ->
            E
    end;
message(_) ->
    bad(<<"every message needs a role">>, <<"messages">>).

with_extras(M, Base) ->
    Calls = [openai_call(C) || C <- maps:get(<<"tool_calls">>, M, [])],
    B1 =
        case Calls of
            [] -> Base;
            _ -> Base#{tool_calls => Calls}
        end,
    B2 =
        case maps:find(<<"tool_call_id">>, M) of
            {ok, Id} -> B1#{tool_call_id => Id};
            error -> B1
        end,
    case maps:find(<<"name">>, M) of
        {ok, N} -> {ok, B2#{name => N}};
        error -> {ok, B2}
    end.

openai_call(#{<<"function">> := #{<<"name">> := Name} = F} = C) ->
    Base = #{
        type => function,
        function => #{name => Name, arguments => maps:get(<<"arguments">>, F, <<"{}">>)}
    },
    case maps:find(<<"id">>, C) of
        {ok, Id} -> Base#{id => Id};
        error -> Base
    end;
openai_call(Other) ->
    Other.

content(absent) -> {ok, absent};
content(null) -> {ok, null};
content(B) when is_binary(B) -> {ok, B};
content([_ | _] = Parts) -> collect([part(P) || P <- Parts]);
content(_) -> bad(<<"content must be a string or an array of parts">>, <<"messages">>).

part(#{<<"type">> := <<"text">>, <<"text">> := T}) ->
    {ok, #{type => text, text => T}};
part(#{<<"type">> := <<"image_url">>, <<"image_url">> := #{<<"url">> := Url}}) ->
    case data_url(Url) of
        {ok, Bytes} ->
            {ok, #{type => image, data => Bytes}};
        error ->
            bad(
                <<"image_url must be a base64 data: URL; the server does not fetch remote images">>,
                <<"messages">>
            )
    end;
part(#{<<"type">> := <<"input_audio">>, <<"input_audio">> := #{<<"data">> := B64}}) ->
    case decode64(B64) of
        {ok, Bytes} -> {ok, #{type => audio, data => Bytes}};
        error -> bad(<<"input_audio.data must be base64">>, <<"messages">>)
    end;
part(#{<<"type">> := T}) ->
    bad([<<"unsupported content part type: ">>, T], <<"messages">>);
part(_) ->
    bad(<<"content parts need a type">>, <<"messages">>).

data_url(<<"data:", Rest/binary>>) ->
    case binary:split(Rest, <<";base64,">>) of
        [_Mime, B64] -> decode64(B64);
        _ -> error
    end;
data_url(_) ->
    error.

decode64(B) when is_binary(B) ->
    try
        {ok, base64:decode(B)}
    catch
        error:_ -> error
    end;
decode64(_) ->
    error.

%% -----------------------------------------------------------------------------
%% Options
%% -----------------------------------------------------------------------------

chat_opts(Body) ->
    case sampling(Body) of
        {ok, Opts0} ->
            Steps = [fun tool_opts/2, fun format_opts/2, fun thinking_opts/2, fun logprob_opts/2],
            lists:foldl(
                fun
                    (Step, {ok, Acc}) -> Step(Body, Acc);
                    (_Step, E) -> E
                end,
                {ok, Opts0},
                Steps
            );
        {error, _} = E ->
            E
    end.

%% Sampling and length, shared by chat and text completions.
sampling(Body) ->
    Map = [
        {<<"max_tokens">>, response_tokens},
        {<<"max_completion_tokens">>, response_tokens},
        {<<"temperature">>, temperature},
        {<<"top_p">>, top_p},
        {<<"seed">>, seed},
        {<<"frequency_penalty">>, frequency_penalty},
        {<<"presence_penalty">>, presence_penalty}
    ],
    Opts = maps:from_list([{K, V} || {F, K} <- Map, {ok, V} <- [maps:find(F, Body)], V =/= null]),
    case maps:find(<<"stop">>, Body) of
        {ok, S} when is_binary(S) -> {ok, Opts#{stop_sequences => [S]}};
        {ok, [_ | _] = S} -> {ok, Opts#{stop_sequences => S}};
        {ok, null} -> {ok, Opts};
        {ok, []} -> {ok, Opts};
        {ok, _} -> bad(<<"stop must be a string or an array of strings">>, <<"stop">>);
        error -> {ok, Opts}
    end.

tool_opts(Body, Opts) ->
    case tools(maps:get(<<"tools">>, Body, [])) of
        {ok, []} ->
            {ok, Opts};
        {ok, Tools} ->
            case tool_choice(maps:get(<<"tool_choice">>, Body, <<"auto">>), Tools) of
                {ok, Choice, Selected} ->
                    Par =
                        case maps:find(<<"parallel_tool_calls">>, Body) of
                            {ok, P} when is_boolean(P) -> #{parallel_tool_calls => P};
                            _ -> #{}
                        end,
                    {ok, maps:merge(Opts#{tools => Selected, tool_choice => Choice}, Par)};
                {error, _} = E ->
                    E
            end;
        {error, _} = E ->
            E
    end.

%% OpenAI function tools as erllama chat tools.
-spec tools(term()) -> {ok, [map()]} | {error, term()}.
tools(L) when is_list(L) ->
    collect([tool(T) || T <- L]);
tools(_) ->
    bad(<<"tools must be an array">>, <<"tools">>).

tool(#{<<"type">> := <<"function">>, <<"function">> := #{<<"name">> := N} = F}) when is_binary(N) ->
    T0 = #{
        name => N,
        parameters => maps:get(<<"parameters">>, F, #{
            <<"type">> => <<"object">>, <<"properties">> => #{}
        })
    },
    case maps:find(<<"description">>, F) of
        {ok, D} -> {ok, T0#{description => D}};
        error -> {ok, T0}
    end;
tool(_) ->
    bad(<<"only function tools are supported">>, <<"tools">>).

tool_choice(<<"auto">>, Tools) ->
    {ok, auto, Tools};
tool_choice(<<"none">>, Tools) ->
    {ok, none, Tools};
tool_choice(<<"required">>, Tools) ->
    {ok, required, Tools};
tool_choice(#{<<"type">> := <<"function">>, <<"function">> := #{<<"name">> := N}}, Tools) ->
    %% One named function: require a call and offer only that tool.
    case [T || #{name := TN} = T <- Tools, TN =:= N] of
        [T | _] -> {ok, required, [T]};
        [] -> bad([<<"tool_choice names an unknown function: ">>, N], <<"tool_choice">>)
    end;
tool_choice(_, _) ->
    bad(<<"tool_choice must be auto, none, required or a function">>, <<"tool_choice">>).

format_opts(Body, Opts) ->
    case maps:get(<<"response_format">>, Body, null) of
        null ->
            {ok, Opts};
        #{<<"type">> := <<"text">>} ->
            {ok, Opts};
        #{<<"type">> := <<"json_object">>} ->
            {ok, Opts#{json_schema => #{<<"type">> => <<"object">>}}};
        #{<<"type">> := <<"json_schema">>, <<"json_schema">> := #{<<"schema">> := S}} ->
            {ok, Opts#{json_schema => S}};
        _ ->
            bad(
                <<"response_format must be text, json_object or json_schema">>,
                <<"response_format">>
            )
    end.

thinking_opts(#{<<"reasoning_effort">> := <<"none">>}, Opts) ->
    {ok, Opts#{enable_thinking => false}};
thinking_opts(#{<<"reasoning_effort">> := E}, Opts) when is_binary(E) ->
    {ok, Opts#{enable_thinking => true}};
thinking_opts(_, Opts) ->
    {ok, Opts}.

logprob_opts(#{<<"logprobs">> := true} = Body, Opts) ->
    case maps:get(<<"top_logprobs">>, Body, 0) of
        N when is_integer(N), N >= 0, N =< 20 -> {ok, Opts#{logprobs => N}};
        _ -> bad(<<"top_logprobs must be an integer from 0 to 20">>, <<"top_logprobs">>)
    end;
logprob_opts(_, Opts) ->
    {ok, Opts}.

prompt(P) when is_binary(P), P =/= <<>> -> {ok, P};
prompt([P]) when is_binary(P), P =/= <<>> -> {ok, P};
prompt(_) -> bad(<<"prompt must be a non-empty string">>, <<"prompt">>).

inputs(B) when is_binary(B) -> {ok, [B]};
inputs([_ | _] = L) ->
    case
        lists:all(fun is_binary/1, L) orelse lists:all(fun is_integer/1, L) orelse
            lists:all(fun is_list/1, L)
    of
        true when is_integer(hd(L)) -> {ok, [L]};
        true ->
            {ok, L};
        false ->
            bad(<<"input must be a string, an array of strings, or token arrays">>, <<"input">>)
    end;
inputs(_) ->
    bad(<<"input is required">>, <<"input">>).
