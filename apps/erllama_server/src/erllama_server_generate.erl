%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_generate).
-moduledoc false.
%% Turns a request (chat messages or a raw prompt) into an erllama
%% stream and the final text into a result. Shared by every front end
%% (RPC runs now, the HTTP adapters later) so they all render, sample
%% and parse the same way.
%%
%% Chat goes through llama.cpp's template engine (`erllama:chat_apply'),
%% which also yields the output parser and the sampler constraints for
%% tools and JSON schemas. A backend without it (the stub) falls back
%% to the legacy template renderer and an unparsed reply.

-export([start/4, finish/4, option_keys/0, check_opts/1, message/1, reply_message/1, constrain/3]).
-export([stream_parser/1, feed/2, count_tokens/3]).

-define(CHAT_KEYS, [
    tools, tool_choice, parallel_tool_calls, json_schema, enable_thinking, reasoning_format
]).
-define(REQUEST_KEYS, [
    response_tokens,
    stop_sequences,
    thinking,
    thinking_budget_tokens,
    temperature,
    seed,
    top_k,
    top_p,
    min_p,
    typical_p,
    repetition_penalty,
    frequency_penalty,
    presence_penalty,
    penalty_last_n,
    grammar,
    logprobs
]).

-type input() :: {messages, [map()]} | {prompt, binary()}.
-type parser() :: erllama:chat_params() | undefined.
-export_type([input/0, parser/0]).

%% Streaming parse state (see feed/2).
-type delta() :: {token, binary()} | {thinking, binary()} | {tool_call, map()}.
-record(sp, {
    params :: erllama:chat_params(),
    text = <<>> :: binary(),
    content = <<>> :: binary(),
    reasoning = <<>> :: binary(),
    calls = 0 :: non_neg_integer()
}).
-opaque stream_parser() :: #sp{}.
-export_type([stream_parser/0, delta/0]).

%% The options a client may pass. Keys the server controls
%% (`parent_key', `session_id', `to', middleware) are not among them.
-spec option_keys() -> [atom()].
option_keys() -> ?CHAT_KEYS ++ ?REQUEST_KEYS.

%% Unknown keys are refused, and so are the combinations erllama:chat/3
%% refuses: with tools in play the template's tool grammar is the
%% constraint, so a caller `grammar' or `json_schema' would fight it.
-spec check_opts(map()) ->
    ok | {error, {invalid_option, term()} | {invalid_option, atom(), term()}}.
check_opts(Opts) when is_map(Opts) ->
    case [K || K <- maps:keys(Opts), not lists:member(K, option_keys())] of
        [] -> check_conflicts(Opts);
        [K | _] -> {error, {invalid_option, K}}
    end;
check_opts(Other) ->
    {error, {invalid_option, Other}}.

check_conflicts(Opts) ->
    ToolsActive =
        maps:get(tools, Opts, []) =/= [] andalso maps:get(tool_choice, Opts, auto) =/= none,
    case ToolsActive of
        true when is_map_key(grammar, Opts) ->
            {error, {invalid_option, grammar, conflicts_with_tools}};
        true when is_map_key(json_schema, Opts) ->
            {error, {invalid_option, json_schema, conflicts_with_tools}};
        _ ->
            ok
    end.

%% Start streaming `Input' on `Model'; events go to `To' as
%% `{erllama, Ref, Event}'. `Extra' carries server-side request keys
%% such as `parent_key'.
-spec start(erllama:model(), input(), map(), pid()) ->
    {ok, reference(), parser()} | {error, term()}.
start(Model, {prompt, Prompt}, Opts, To) when is_binary(Prompt) ->
    case maps:keys(maps:with(?CHAT_KEYS, Opts)) of
        [] -> stream(Model, Prompt, Opts, To, undefined);
        [K | _] -> {error, {invalid_option, K, requires_messages}}
    end;
start(Model, {messages, Msgs}, Opts, To) when is_list(Msgs) ->
    ChatOpts = maps:with(?CHAT_KEYS, Opts),
    Req = request_opts(Opts),
    case check_conflicts(Opts) of
        ok -> start_chat(Model, Msgs, ChatOpts, Req, To);
        {error, _} = E -> E
    end.

start_chat(Model, Msgs0, ChatOpts, Req, To) ->
    case extract_media(Msgs0) of
        {Msgs, []} -> start_text_chat(Model, Msgs, ChatOpts, Req, To);
        {Msgs, Media} -> start_media_chat(Model, Msgs, Media, ChatOpts, Req, To)
    end.

%% Images and audio go to the model's projector: the prompt carries one
%% media marker per item and is evaluated whole (no KV cache reuse, so
%% the cache options are dropped). The template already put the BOS
%% token at the start of the prompt.
start_media_chat(Model, Msgs, Media, ChatOpts, Req, To) ->
    case erllama:chat_apply(Model, Msgs, ChatOpts) of
        {ok, #{prompt := P, params := Params, sampler_opts := S, stop_sequences := Stops}} ->
            Req1 = maps:without([parent_key, session_id], constrain(S, Stops, Req)),
            stream(Model, P, Req1#{media => Media, media_add_special => false}, To, Params);
        {error, chat_not_supported} ->
            {error, {media_not_supported, Model}};
        {error, _} = E ->
            E
    end.

start_text_chat(Model, Msgs, ChatOpts, Req, To) ->
    case erllama:chat_apply(Model, Msgs, ChatOpts) of
        {ok, #{prompt := P, params := Params, sampler_opts := S, stop_sequences := Stops}} ->
            case erllama:tokenize(Model, P, #{add_special => false, parse_special => true}) of
                {ok, Tokens} ->
                    stream(Model, Tokens, constrain(S, Stops, Req), To, Params);
                {error, _} = E ->
                    E
            end;
        {error, chat_not_supported} ->
            case erllama:render_chat_template(Model, template_request(Msgs, ChatOpts)) of
                {ok, Tokens} -> stream(Model, Tokens, Req, To, undefined);
                {error, _} = E -> E
            end;
        {error, _} = E ->
            E
    end.

request_opts(Opts) ->
    maps:without(?CHAT_KEYS, Opts).

%% Media parts become the media marker the template renders verbatim;
%% the media, in order, travels beside the prompt.
extract_media(Msgs) ->
    {Rev, Media} = lists:foldl(
        fun
            (#{content := Parts} = M, {MAcc, DAcc}) when is_list(Parts) ->
                {Parts1, DAcc1} = lists:foldl(fun media_part/2, {[], DAcc}, Parts),
                {[M#{content := lists:reverse(Parts1)} | MAcc], DAcc1};
            (M, {MAcc, DAcc}) ->
                {[M | MAcc], DAcc}
        end,
        {[], []},
        Msgs
    ),
    {lists:reverse(Rev), lists:reverse(Media)}.

media_part(#{type := T, data := D}, {PAcc, DAcc}) when T =:= image; T =:= audio ->
    {[#{type => <<"media_marker">>, text => erllama:media_marker()} | PAcc], [
        #{type => T, data => D} | DAcc
    ]};
media_part(Part, {PAcc, DAcc}) ->
    {[Part | PAcc], DAcc}.

template_request(Msgs, #{tools := Tools}) -> #{messages => Msgs, tools => Tools};
template_request(Msgs, _) -> #{messages => Msgs}.

%% The template's constraint set (`S': grammar with its lazy triggers
%% or prefill) and stop strings, combined with the caller's options as
%% erllama:chat/3 combines them: a caller grammar replaces the template
%% grammar together with everything that belongs to it, other caller
%% keys win, and the template's stops are added after the caller's.
constrain(S, Stops, Req) ->
    Req1 =
        case is_map_key(grammar, Req) of
            true -> Req;
            false -> maps:merge(S, Req)
        end,
    Existing = maps:get(stop_sequences, Req1, []),
    case [X || X <- Stops, is_binary(X), not lists:member(X, Existing)] of
        [] -> Req1;
        New -> Req1#{stop_sequences => Existing ++ New}
    end.

stream(Model, Prompt, Req, To, Parser) ->
    case erllama:stream(Model, Prompt, Req#{to => To}) of
        {ok, Ref} -> {ok, Ref, Parser};
        {error, _} = E -> E
    end.

%% The result of a finished stream: the reply text (parsed into
%% content and tool calls when a parser exists), the streamed
%% reasoning, the finish reason, token usage and timings.
-spec finish(parser(), binary(), binary(), map()) -> map().
finish(Parser, Text, Thinking, Stats) ->
    Base = #{
        content => Text,
        reasoning => Thinking,
        tool_calls => [],
        finish_reason => maps:get(finish_reason, Stats, stop),
        finish_key => maps:get(finish_key, Stats, undefined),
        usage => #{
            prompt_tokens => maps:get(prompt_tokens, Stats, 0),
            completion_tokens => maps:get(completion_tokens, Stats, 0)
        },
        timings => #{
            prefill_ms => maps:get(prefill_ms, Stats, 0),
            generation_ms => maps:get(generation_ms, Stats, 0)
        },
        cache_hit_kind => maps:get(cache_hit_kind, Stats, cold),
        stop_sequence => maps:get(stop_sequence, Stats, undefined)
    },
    parse(Parser, Text, Base).

parse(undefined, _Text, Base) ->
    Base;
parse(Parser, Text, #{reasoning := Thinking} = Base) ->
    case erllama:chat_parse(Parser, Text, false) of
        {ok, #{content := C, tool_calls := Calls0} = M} ->
            Calls = [Call#{id := call_id(maps:get(id, Call, undefined))} || Call <- Calls0],
            Reasoning =
                case maps:get(reasoning_content, M, undefined) of
                    R when is_binary(R), R =/= <<>> -> R;
                    _ -> Thinking
                end,
            Base#{content := C, tool_calls := Calls, reasoning := Reasoning};
        {error, _} ->
            Base
    end.

%% -----------------------------------------------------------------------------
%% Chat messages
%% -----------------------------------------------------------------------------

%% Validate a chat message and put it in the shape llama.cpp's template
%% engine reads: `role' as a binary; tool calls in the OpenAI shape
%% (`#{id, type => <<"function">>, function => #{name, arguments}}',
%% arguments as a JSON string), accepted either in that shape or flat
%% as a run result gives them (`#{name, arguments, id}'); and a `tool'
%% message keeping the `tool_call_id' of the call it answers.
-spec message(term()) -> {ok, map()} | {error, {invalid_message, term()}}.
message(#{role := R} = M) ->
    case role(R) of
        {ok, Role} -> message(Role, M);
        error -> {error, {invalid_message, #{role => R}}}
    end;
message(M) ->
    {error, {invalid_message, M}}.

message(<<"assistant">>, M) ->
    Calls = maps:get(tool_calls, M, []),
    case {content(M, Calls =/= []), tool_calls(Calls)} of
        {{ok, C}, {ok, []}} -> {ok, #{role => <<"assistant">>, content => C}};
        {{ok, C}, {ok, Cs}} -> {ok, #{role => <<"assistant">>, content => C, tool_calls => Cs}};
        {{error, _}, _} -> {error, {invalid_message, M}};
        {_, error} -> {error, {invalid_message, M}}
    end;
message(<<"tool">>, #{tool_call_id := Id, content := C} = M) when
    is_binary(Id), Id =/= <<>>, is_binary(C)
->
    Base = #{role => <<"tool">>, content => C, tool_call_id => Id},
    case M of
        #{name := N} when is_binary(N) -> {ok, Base#{name => N}};
        #{name := _} -> {error, {invalid_message, M}};
        _ -> {ok, Base}
    end;
message(<<"tool">>, M) ->
    {error, {invalid_message, M}};
message(Role, #{content := C}) when is_binary(C) ->
    {ok, #{role => Role, content => C}};
message(Role, #{content := [_ | _] = Parts} = M) ->
    case parts(Parts) of
        {ok, Ps} -> {ok, #{role => Role, content => Ps}};
        error -> {error, {invalid_message, M}}
    end;
message(_Role, M) ->
    {error, {invalid_message, M}}.

role(R) when is_atom(R) -> role(atom_to_binary(R));
role(R) when R =:= <<"system">>; R =:= <<"user">>; R =:= <<"assistant">>; R =:= <<"tool">> ->
    {ok, R};
role(_) ->
    error.

%% Content parts: text, and image or audio bytes for a model with a
%% projector.
parts(Parts) ->
    all_ok([part(P) || P <- Parts]).

%% `{ok, List}' unless an element failed to convert.
all_ok(Normal) ->
    case lists:member(error, Normal) of
        true -> error;
        false -> {ok, Normal}
    end.

part(#{type := text, text := Text}) when is_binary(Text) ->
    #{type => <<"text">>, text => Text};
part(#{type := <<"text">>, text := Text}) when is_binary(Text) ->
    #{type => <<"text">>, text => Text};
part(#{type := T, data := D}) when
    T =:= image, is_binary(D), D =/= <<>>; T =:= audio, is_binary(D), D =/= <<>>
->
    #{type => T, data => D};
part(_) ->
    error.

%% An assistant message may omit its content (or set it to null) only
%% when it carries tool calls.
content(#{content := C}, _) when is_binary(C) -> {ok, C};
content(#{content := null}, true) -> {ok, null};
content(M, true) when not is_map_key(content, M) -> {ok, null};
content(_, _) -> {error, content}.

tool_calls(Calls) when is_list(Calls) ->
    all_ok([tool_call(C) || C <- Calls]);
tool_calls(_) ->
    error.

tool_call(#{type := T, function := #{name := N} = F} = C) when
    T =:= function; T =:= <<"function">>
->
    openai_call(N, maps:get(arguments, F, #{}), maps:get(id, C, undefined));
tool_call(#{name := N, arguments := A} = C) ->
    openai_call(N, A, maps:get(id, C, undefined));
tool_call(_) ->
    error.

openai_call(Name, Args, Id) when is_binary(Name), Name =/= <<>> ->
    case arguments_json(Args) of
        {ok, Json} ->
            #{
                id => call_id(Id),
                type => <<"function">>,
                function => #{name => Name, arguments => Json}
            };
        error ->
            error
    end;
openai_call(_, _, _) ->
    error.

arguments_json(Args) when is_map(Args) -> {ok, iolist_to_binary(json:encode(Args))};
arguments_json(Args) when is_binary(Args) ->
    try json:decode(Args) of
        M when is_map(M) -> {ok, Args};
        _ -> error
    catch
        error:_ -> error
    end;
arguments_json(_) ->
    error.

%% A call the model emitted without an id gets one, so the tool result
%% can name it with `tool_call_id' on the next turn.
call_id(Id) when is_binary(Id), Id =/= <<>> -> Id;
call_id(_) -> erllama_server_ids:tool_call().

%% The assistant message a finished turn adds to its conversation.
-spec reply_message(map()) -> map().
reply_message(#{content := Content, tool_calls := []}) ->
    #{role => <<"assistant">>, content => Content};
reply_message(#{content := Content, tool_calls := Calls}) ->
    {ok, Msg} = message(#{role => assistant, content => Content, tool_calls => Calls}),
    case Content of
        <<>> -> Msg#{content := null};
        _ -> Msg
    end.

%% -----------------------------------------------------------------------------
%% Streaming parse
%% -----------------------------------------------------------------------------

%% With a parser, the reply is parsed again after every token
%% (llama.cpp's partial parsing holds back what may still turn into a
%% tool call), and only what is new is emitted: content and reasoning
%% deltas, and each tool call once its name is known. The complete
%% arguments come with the final result.

-spec stream_parser(parser()) -> stream_parser() | undefined.
stream_parser(undefined) -> undefined;
stream_parser(Params) -> #sp{params = Params}.

-spec feed(binary(), stream_parser()) -> {[delta()], stream_parser()}.
feed(Bin, #sp{params = P, text = T} = Sp) ->
    Text = <<T/binary, Bin/binary>>,
    case erllama:chat_parse(P, Text, true) of
        {ok, Msg} -> diff(Msg, Sp#sp{text = Text});
        {error, _} -> {[], Sp#sp{text = Text}}
    end.

diff(Msg, #sp{content = C0, reasoning = R0, calls = N0} = Sp) ->
    C = maps:get(content, Msg, <<>>),
    R =
        case maps:get(reasoning_content, Msg, undefined) of
            B when is_binary(B) -> B;
            _ -> <<>>
        end,
    Calls = maps:get(tool_calls, Msg, []),
    {DC, C1} = suffix(C0, C),
    {DR, R1} = suffix(R0, R),
    New = [
        {tool_call, #{index => I, id => call_id(maps:get(id, Call, undefined)), name => Name}}
     || {I, #{name := Name} = Call} <- lists:zip(lists:seq(0, length(Calls) - 1), Calls),
        I >= N0,
        Name =/= <<>>
    ],
    Deltas = [{thinking, DR} || DR =/= <<>>] ++ [{token, DC} || DC =/= <<>>] ++ New,
    {Deltas, Sp#sp{content = C1, reasoning = R1, calls = N0 + length(New)}}.

%% What `New' adds to `Old'. A parse that rewrote what was already sent
%% (it should not) sends nothing until it extends it again.
suffix(Old, New) ->
    case string:prefix(New, Old) of
        nomatch -> {<<>>, Old};
        Rest -> {iolist_to_binary(Rest), New}
    end.

%% The number of tokens `Msgs' (and the tools in `Opts') take once
%% rendered through the model's template.
-spec count_tokens(erllama:model(), [map()], map()) -> {ok, non_neg_integer()} | {error, term()}.
count_tokens(Model, Msgs0, Opts) ->
    {Msgs, _Media} = extract_media(Msgs0),
    ChatOpts = maps:with(?CHAT_KEYS, Opts),
    case erllama:chat_apply(Model, Msgs, ChatOpts) of
        {ok, #{prompt := P}} ->
            case erllama:tokenize(Model, P, #{add_special => false, parse_special => true}) of
                {ok, Tokens} -> {ok, length(Tokens)};
                {error, _} = E -> E
            end;
        {error, chat_not_supported} ->
            case erllama:render_chat_template(Model, template_request(Msgs, ChatOpts)) of
                {ok, Tokens} -> {ok, length(Tokens)};
                {error, _} = E -> E
            end;
        {error, _} = E ->
            E
    end.
