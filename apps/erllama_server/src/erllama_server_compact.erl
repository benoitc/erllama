%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_compact).
-moduledoc false.
%% Context compaction: a transcript that nears the model's context is
%% replaced by a shorter one that keeps what the conversation needs to
%% go on.
%%
%% Kept: every system message verbatim (the first stays at the head;
%% the others, and the summary section, are appended to it, so a
%% template that takes one leading system message still renders); the
%% first user request verbatim, inside the summary section; a summary
%% of the middle written by the conversation's own model (goal,
%% decisions, state of the work, what remains); and the recent
%% messages verbatim, from a user message on, so tool calls and their
%% results stay in pairs.
%%
%% The compacted transcript is a new exact prompt: callers drop the
%% finish key. Nothing about it goes into the KV cache.

-export([run/3, start/4, conversation/2, needed/2, plan/2, apply_summary/2, render/1]).
-export([with_system/2]).

-define(CLIP, 2048).
-define(SUMMARY_TIMEOUT, 120000).
-define(INSTRUCTION, <<
    "You compact a conversation so that it can go on in less space. Write a "
    "summary of the conversation below that keeps everything needed to continue "
    "it: the user's goal and constraints, the decisions made, the current state of "
    "the work (files read or changed, commands run and their outcome, results "
    "found), and what remains to be done. Be specific and brief. Plain text, no "
    "preamble."
>>).

-type plan() :: #{
    system := [map()],
    first := map(),
    middle := [map()],
    tail := [map()]
}.
-export_type([plan/0]).

%% Compact `Msgs' for `Model' when it needs it (or always, with
%% `force'). The model must be loaded and held by the caller.
%% `on_start' (a fun) is called once a compaction is decided, before
%% the summary is written.
-spec run(erllama:model(), [map()], map()) ->
    {compacted, [map()], map()} | none | {error, term()}.
run(Model, Msgs, Opts) ->
    case maps:get(force, Opts, false) orelse needed(Model, Msgs) of
        true -> compact(Model, Msgs, Opts);
        false -> none
    end.

compact(Model, Msgs, Opts) ->
    Keep = erllama_server_config:get(compact_keep_messages),
    case plan(Msgs, Keep) of
        {ok, Plan} ->
            _ = (maps:get(on_start, Opts, fun() -> ok end))(),
            case summarize(Model, Plan) of
                {ok, Summary} ->
                    New = apply_summary(Plan, Summary),
                    {compacted, New, #{before => length(Msgs), 'after' => length(New)}};
                {error, _} = E ->
                    E
            end;
        nothing ->
            none
    end.

%% A helper, linked to `Owner', that holds `Model' for `Owner', compacts
%% `Msgs' when needed (or always, with `force') and reports
%% `{compacting, Pid}' once it decides to, then
%% `{compact_done, Pid, Outcome}'.
-spec start(pid(), term(), [map()], map()) -> pid().
start(Owner, Model0, Msgs, Opts) ->
    spawn_link(fun() ->
        Outcome =
            case erllama_server_models:acquire(Model0, Owner, undefined) of
                {ok, Model} ->
                    run(Model, Msgs, Opts#{on_start => fun() -> Owner ! {compacting, self()} end});
                {error, _} = E ->
                    E
            end,
        Owner ! {compact_done, self(), Outcome}
    end).

%% Compact an idle conversation now: hold it and its model, compact,
%% record, let go.
-spec conversation(binary(), term()) -> {ok, map()} | {error, term()}.
conversation(Id, Model0) ->
    Holder = erllama_server_ids:run(),
    case erllama_server_convs:begin_run(Id, Model0, Holder, #{kind => compact}) of
        {ok, #{messages := Msgs}} ->
            try
                held(Id, Holder, Model0, Msgs)
            after
                ok = erllama_server_convs:release(Id, Holder)
            end;
        {error, _} = E ->
            E
    end.

held(Id, Holder, Model0, Msgs) ->
    case erllama_server_models:acquire(Model0, self(), undefined) of
        {ok, Model} ->
            try run(Model, Msgs, #{force => true}) of
                {compacted, New, Info} ->
                    ok = erllama_server_convs:compact(Id, Holder, New, Info),
                    {ok, Info};
                none ->
                    {ok, #{compacted => false}};
                {error, _} = E ->
                    E
            after
                erllama_server_models:release(Model, self())
            end;
        {error, _} = E ->
            E
    end.

%% Is the transcript above the threshold of the model's context?
-spec needed(erllama:model(), [map()]) -> boolean().
needed(Model, Msgs) ->
    case erllama_server_config:get(compact_threshold) of
        false ->
            false;
        Threshold ->
            case {context_size(Model), erllama_server_generate:count_tokens(Model, Msgs, #{})} of
                {Ctx, {ok, N}} when is_integer(Ctx), Ctx > 0 -> N > Threshold * Ctx;
                _ -> false
            end
    end.

context_size(Model) ->
    try erllama:model_info(Model) of
        {ok, #{context_size := N}} -> N;
        _ -> undefined
    catch
        _:_ -> undefined
    end.

%% -----------------------------------------------------------------------------
%% The plan
%% -----------------------------------------------------------------------------

%% Where to cut: the tail starts at the latest user message that keeps
%% at least `Keep' messages after it, or, if every turn is long, at the
%% latest user message. Never at the first request (kept anyway).
-spec plan([map()], pos_integer()) -> {ok, plan()} | nothing.
plan(Msgs, Keep) ->
    System = [M || #{role := <<"system">>} = M <- Msgs],
    Rest = [M || M <- Msgs, not is_system(M)],
    Indexed = lists:zip(lists:seq(1, length(Rest)), Rest),
    case [I || {I, #{role := <<"user">>}} <- Indexed] of
        [First | Later] when Later =/= [] ->
            Len = length(Rest),
            Cut =
                case [I || I <- Later, Len - I + 1 >= Keep] of
                    [] -> lists:last(Later);
                    Fit -> lists:last(Fit)
                end,
            Middle =
                lists:sublist(Rest, First - 1) ++ lists:sublist(Rest, First + 1, Cut - First - 1),
            case Middle of
                [] ->
                    nothing;
                _ ->
                    {ok, #{
                        system => System,
                        first => lists:nth(First, Rest),
                        middle => Middle,
                        tail => lists:nthtail(Cut - 1, Rest)
                    }}
            end;
        _ ->
            nothing
    end.

%% `System' first, then `History', with one leading system message:
%% system messages the history holds (a compaction's summary, earlier
%% instructions) are appended to `System', and a copy of `System' itself
%% at their start is not repeated. Agent sessions build their
%% transcript with it when they continue a conversation.
-spec with_system(binary(), [map()]) -> [map()].
with_system(System, History) ->
    Extra = [
        string:trim(strip_prefix(System, C))
     || #{role := <<"system">>, content := C} <- History, is_binary(C)
    ],
    Content = iolist_to_binary(lists:join(<<"\n\n">>, [System | [E || E <- Extra, E =/= <<>>]])),
    [#{role => <<"system">>, content => Content} | [M || M <- History, not is_system(M)]].

strip_prefix(Prefix, C) ->
    case string:prefix(C, Prefix) of
        nomatch -> C;
        Rest -> unicode:characters_to_binary(Rest)
    end.

is_system(#{role := <<"system">>}) -> true;
is_system(_) -> false.

%% The compacted transcript: one leading system message (the first,
%% with the others and the summary section appended), then the tail.
-spec apply_summary(plan(), binary()) -> [map()].
apply_summary(#{system := System, first := First, tail := Tail}, Summary) ->
    Section = iolist_to_binary([
        <<"Earlier part of this conversation, compacted.\n\nThe first request was:\n">>,
        text(First),
        <<"\n\nSummary of what happened since:\n">>,
        Summary
    ]),
    Head =
        case System of
            [] ->
                #{role => <<"system">>, content => Section};
            [S | Others] ->
                Parts = [text(S) | [text(O) || O <- Others]] ++ [Section],
                S#{content => iolist_to_binary(lists:join(<<"\n\n">>, Parts))}
        end,
    [Head | Tail].

%% -----------------------------------------------------------------------------
%% The summary
%% -----------------------------------------------------------------------------

summarize(Model, #{first := First, middle := Middle}) ->
    Max = erllama_server_config:get(compact_summary_tokens),
    Prompt = [
        #{role => <<"system">>, content => ?INSTRUCTION},
        #{
            role => <<"user">>,
            content => iolist_to_binary([
                <<"First request:\n">>, text(First), <<"\n\nConversation:\n">>, render(Middle)
            ])
        }
    ],
    case
        erllama_server_generate:start(Model, {messages, Prompt}, #{response_tokens => Max}, self())
    of
        {ok, Ref, Parser} -> collect(Ref, Parser, []);
        {error, _} = E -> E
    end.

collect(Ref, Parser, Acc) ->
    receive
        {erllama, Ref, {token, B}} ->
            collect(Ref, Parser, [B | Acc]);
        {erllama, Ref, {done, Stats}} ->
            Text = iolist_to_binary(lists:reverse(Acc)),
            #{content := C} = erllama_server_generate:finish(Parser, Text, <<>>, Stats),
            case string:trim(C) of
                <<>> -> {error, empty_summary};
                S -> {ok, S}
            end;
        {erllama, Ref, {error, R}} ->
            {error, R};
        {erllama, Ref, _Other} ->
            collect(Ref, Parser, Acc)
    after ?SUMMARY_TIMEOUT ->
        _ = erllama:cancel(Ref),
        {error, summary_timeout}
    end.

%% Messages as plain text for the summariser, each clipped.
-spec render([map()]) -> binary().
render(Msgs) ->
    iolist_to_binary(lists:join(<<"\n">>, [line(M) || M <- Msgs])).

line(#{role := <<"tool">>} = M) ->
    [<<"[result of ">>, maps:get(name, M, <<"tool">>), <<"] ">>, clip(text(M))];
line(#{role := R, tool_calls := [_ | _] = Calls} = M) ->
    [
        R,
        <<": ">>,
        clip(text(M))
        | [
            [<<"\n[calls ">>, N, <<" ">>, clip(A), <<"]">>]
         || #{function := #{name := N, arguments := A}} <- Calls
        ]
    ];
line(#{role := R} = M) ->
    [R, <<": ">>, clip(text(M))].

clip(B) when byte_size(B) > ?CLIP -> [binary:part(B, 0, ?CLIP), <<"...">>];
clip(B) -> B.

text(#{content := C}) when is_binary(C) -> C;
text(#{content := Parts}) when is_list(Parts) ->
    iolist_to_binary([T || #{text := T} <- Parts]);
text(_) ->
    <<>>.
