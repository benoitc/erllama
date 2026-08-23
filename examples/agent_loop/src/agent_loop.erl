%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(agent_loop).
-moduledoc """
A tool-calling agent loop on top of `erllama:chat/3`.

```erlang
1> agent_loop:run("/srv/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                  <<"What is 17 * 23, and what time is it?">>).
```

Each turn renders the whole conversation through the model's chat
template, generates, and parses the reply. When the model asks for
tools, the loop runs them, appends the results as `tool` messages and
calls the model again, until it answers in plain text or the round
limit is hit. The conversation is pinned to one erllama session so the
KV cells of the shared prefix stay live between turns; the printed
`cache_delta` shows how many prompt tokens were served from cache.
""".

-export([run/2, run/3, tools/0, call_tool/2]).

-define(MAX_ROUNDS, 8).

-doc "Load the model, answer `Question` with tools, unload.".
-spec run(file:name_all(), binary()) -> {ok, binary()} | {error, term()}.
run(ModelPath, Question) ->
    run(ModelPath, Question, #{}).

-doc """
`Opts`: `max_rounds` (default 8), `verbose` (default true), `system`
(system prompt), `chat_template` (path to a Jinja file that replaces
the template stored in the GGUF; `priv/qwen2.5-instruct.jinja` is the
corrected Qwen2.5 template, some Qwen2.5 GGUFs ship one with doubled
braces that breaks tool calls), plus any `erllama:request_opts()` key
such as `temperature` or `response_tokens`.
""".
-spec run(file:name_all(), binary(), map()) -> {ok, binary()} | {error, term()}.
run(ModelPath, Question, Opts) ->
    {ok, _} = application:ensure_all_started(erllama),
    Load0 = #{model_path => ModelPath, context_opts => #{n_ctx => 8192}},
    Load =
        case maps:find(chat_template, Opts) of
            {ok, TemplatePath} ->
                {ok, Tmpl} = file:read_file(TemplatePath),
                Load0#{chat_template => Tmpl};
            error ->
                Load0
        end,
    case erllama:load_model(Load) of
        {ok, Model} ->
            try
                answer(Model, Question, Opts)
            after
                erllama:unload(Model)
            end;
        {error, _} = E ->
            E
    end.

%% Run the loop against an already loaded model.
answer(Model, Question, Opts) ->
    System = maps:get(system, Opts, default_system()),
    Messages = [
        #{role => system, content => System},
        #{role => user, content => Question}
    ],
    Session = make_ref(),
    ChatOpts = maps:merge(
        #{
            tools => tools(),
            tool_choice => auto,
            temperature => 0.0,
            response_tokens => 512,
            session_id => Session
        },
        maps:without([max_rounds, verbose, system, chat_template], Opts)
    ),
    MaxRounds = maps:get(max_rounds, Opts, ?MAX_ROUNDS),
    Verbose = maps:get(verbose, Opts, true),
    try
        loop(Model, Messages, ChatOpts, MaxRounds, Verbose, 1)
    after
        erllama:end_session(Model, Session)
    end.

loop(_Model, _Messages, _Opts, MaxRounds, _Verbose, Round) when Round > MaxRounds ->
    {error, {max_rounds, MaxRounds}};
loop(Model, Messages, Opts, MaxRounds, Verbose, Round) ->
    case erllama:chat(Model, Messages, Opts) of
        {ok, #{message := Msg, stats := Stats}} ->
            report(Verbose, Round, Msg, Stats),
            case maps:get(tool_calls, Msg) of
                [] ->
                    {ok, maps:get(content, Msg)};
                Calls ->
                    Assistant = assistant_message(Msg, Calls),
                    Results = [tool_message(Call, Verbose) || Call <- Calls],
                    loop(
                        Model,
                        Messages ++ [Assistant | Results],
                        Opts,
                        MaxRounds,
                        Verbose,
                        Round + 1
                    )
            end;
        {error, _} = E ->
            E
    end.

%% The assistant turn that requested the calls, in the OpenAI shape the
%% templates expect (arguments as a JSON string).
assistant_message(Msg, Calls) ->
    #{
        role => assistant,
        content => maps:get(content, Msg),
        tool_calls => [
            #{
                id => call_id(Call),
                type => function,
                function => #{
                    name => maps:get(name, Call),
                    arguments => iolist_to_binary(json:encode(maps:get(arguments, Call)))
                }
            }
         || Call <- Calls
        ]
    }.

tool_message(#{name := Name, arguments := Args} = Call, Verbose) ->
    Result = call_tool(Name, Args),
    Verbose andalso
        io:format("  tool ~s(~s) -> ~s~n", [Name, json:encode(Args), Result]),
    #{role => tool, tool_call_id => call_id(Call), name => Name, content => Result}.

%% Templates that do not mint ids leave `id` undefined; reuse the name.
call_id(#{id := undefined, name := Name}) -> Name;
call_id(#{id := Id}) -> Id.

report(false, _Round, _Msg, _Stats) ->
    ok;
report(true, Round, Msg, Stats) ->
    #{prompt_tokens := P, completion_tokens := C, cache_delta := #{read := Read}} = Stats,
    io:format(
        "round ~p: ~p prompt tokens (~p from cache), ~p generated, ~p tool call(s)~n",
        [Round, P, Read, C, length(maps:get(tool_calls, Msg))]
    ),
    case maps:get(content, Msg) of
        <<>> -> ok;
        Text -> io:format("  assistant: ~s~n", [Text])
    end.

default_system() ->
    <<"You are a helpful assistant. Use the available tools when they help answer the question, then reply to the user in one or two sentences.">>.

%% =============================================================================
%% Tools
%% =============================================================================

-doc "The tool definitions handed to the model.".
-spec tools() -> [erllama:chat_tool()].
tools() ->
    [
        #{
            name => <<"get_current_time">>,
            description => <<"Current date and time in UTC, ISO 8601.">>,
            parameters => #{type => object, properties => #{}, required => []}
        },
        #{
            name => <<"calculate">>,
            description =>
                <<"Evaluate an arithmetic expression with + - * / and parentheses, e.g. \"17 * 23\".">>,
            parameters => #{
                type => object,
                properties => #{expression => #{type => string}},
                required => [expression]
            }
        },
        #{
            name => <<"list_files">>,
            description => <<"List the files in a directory on this machine.">>,
            parameters => #{
                type => object,
                properties => #{path => #{type => string}},
                required => [path]
            }
        }
    ].

-doc "Run one tool; returns the result as a binary for the `tool` message.".
-spec call_tool(binary(), map()) -> binary().
call_tool(<<"get_current_time">>, _Args) ->
    iolist_to_binary(calendar:system_time_to_rfc3339(erlang:system_time(second), [{offset, "Z"}]));
call_tool(<<"calculate">>, #{<<"expression">> := Expr}) ->
    calculate(Expr);
call_tool(<<"list_files">>, #{<<"path">> := Path}) ->
    case file:list_dir(Path) of
        {ok, Names} -> iolist_to_binary(json:encode([list_to_binary(N) || N <- Names]));
        {error, Reason} -> iolist_to_binary(io_lib:format("error: ~p", [Reason]))
    end;
call_tool(Name, Args) ->
    iolist_to_binary(io_lib:format("error: unknown tool ~s with ~p", [Name, Args])).

%% Arithmetic only: tokens are restricted to digits, operators and
%% parentheses before the expression reaches erl_eval.
calculate(Expr) when is_binary(Expr) ->
    Str = binary_to_list(Expr),
    case lists:all(fun(C) -> lists:member(C, "0123456789.+-*/() ") end, Str) of
        false ->
            <<"error: only arithmetic is allowed">>;
        true ->
            try
                {ok, Tokens, _} = erl_scan:string(Str ++ "."),
                {ok, [Form]} = erl_parse:parse_exprs(Tokens),
                {value, V, _} = erl_eval:expr(Form, []),
                iolist_to_binary(io_lib:format("~p", [V]))
            catch
                _:_ -> <<"error: cannot evaluate">>
            end
    end.
