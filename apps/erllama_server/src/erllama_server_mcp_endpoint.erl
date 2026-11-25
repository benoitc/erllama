%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_mcp_endpoint).
-moduledoc false.
%% The opt-in MCP endpoint (`mcp_endpoint => true'): `/mcp' on the HTTP
%% listener, behind the same API keys, so MCP clients can use the
%% release's models as tools. Livery serves the transport (livery_mcp)
%% and barrel_mcp the protocol; this module registers the tools:
%%
%%   erllama_models   the models the release can run
%%   erllama_chat     one stateless chat turn with a model
%%
%% A chat goes through erllama_server_api:start_run/1 like any HTTP
%% request: the same admission limit, and it shows in `erllama runs'.

-export([routes/0, register_tools/0]).
-export([models/1, chat/1]).

-define(MAX_TOKENS, 1024).
%% The longest a chat may take before barrel_mcp stops the handler.
-define(CHAT_TIMEOUT_MS, 600000).

-spec routes() -> list().
routes() ->
    Mcp = livery_mcp:handler(#{}),
    [{M, <<"/mcp">>, Mcp} || M <- [<<"POST">>, <<"GET">>, <<"DELETE">>, <<"OPTIONS">>]].

%% The tools live in barrel_mcp's node-wide registry.
-spec register_tools() -> ok.
register_tools() ->
    ok = barrel_mcp:reg_tool(<<"erllama_models">>, ?MODULE, models, #{
        description => <<"List the models this erllama release can run, and which are loaded.">>,
        input_schema => #{<<"type">> => <<"object">>, <<"properties">> => #{}},
        annotations => #{<<"readOnlyHint">> => true}
    }),
    ok = barrel_mcp:reg_tool(<<"erllama_chat">>, ?MODULE, chat, #{
        description => <<
            "Ask a model of this erllama release a question and get its answer. "
            "Each call is independent: no history is kept."
        >>,
        input_schema => #{
            <<"type">> => <<"object">>,
            <<"properties">> => #{
                <<"prompt">> => str(<<"The question or instruction.">>),
                <<"model">> => str(
                    <<"A model name (erllama_models); the default model if absent.">>
                ),
                <<"system">> => str(<<"An optional system prompt.">>),
                <<"max_tokens">> => #{
                    <<"type">> => <<"integer">>,
                    <<"description">> => <<"The most tokens to generate (default 1024).">>
                }
            },
            <<"required">> => [<<"prompt">>]
        },
        annotations => #{<<"readOnlyHint">> => true, <<"openWorldHint">> => false},
        timeout => ?CHAT_TIMEOUT_MS
    }),
    ok.

str(D) -> #{<<"type">> => <<"string">>, <<"description">> => D}.

-spec models(map()) -> binary().
models(_Args) ->
    {ok, Models} = erllama_server_api:models(),
    Rows = [
        #{
            <<"name">> => N,
            <<"loaded">> => maps:get(loaded, M, false),
            <<"default">> => maps:get(default, M, false)
        }
     || #{name := N} = M <- Models
    ],
    iolist_to_binary(json:encode(Rows)).

-spec chat(map()) -> binary() | {tool_error, binary()}.
chat(#{<<"prompt">> := Prompt} = Args) when is_binary(Prompt), Prompt =/= <<>> ->
    System = [
        #{role => <<"system">>, content => S}
     || S <- [maps:get(<<"system">>, Args, <<>>)], is_binary(S), S =/= <<>>
    ],
    MaxTokens =
        case maps:get(<<"max_tokens">>, Args, ?MAX_TOKENS) of
            N when is_integer(N), N > 0 -> N;
            _ -> ?MAX_TOKENS
        end,
    Spec0 = #{
        conversation_id => none,
        messages => System ++ [#{role => <<"user">>, content => Prompt}],
        opts => #{response_tokens => MaxTokens}
    },
    Spec =
        case maps:get(<<"model">>, Args, undefined) of
            M when is_binary(M), M =/= <<>> -> Spec0#{model => M};
            _ -> Spec0
        end,
    case erllama_server_api:start_run(Spec) of
        {ok, #{run_id := Run}} -> answer(Run);
        {error, R} -> {tool_error, describe(R)}
    end;
chat(_Args) ->
    {tool_error, <<"missing argument: prompt">>}.

%% Wait for the run. If this process is stopped (the tool's timeout, or
%% the client cancelling), the run is cancelled with it.
answer(Run) ->
    Self = self(),
    Guard = spawn(fun() ->
        Mon = erlang:monitor(process, Self),
        receive
            {'DOWN', Mon, process, Self, _} -> _ = erllama_server_api:cancel(Run);
            done -> ok
        after ?CHAT_TIMEOUT_MS + 5000 ->
            _ = erllama_server_api:cancel(Run)
        end
    end),
    _ = erllama_server_http_kit:drive(Run, fun(_E, Acc) -> {ok, Acc} end, ok),
    Guard ! done,
    case erllama_server_api:result(Run) of
        {ok, #{status := done, content := Text}} -> Text;
        {ok, #{error := R}} -> {tool_error, describe(R)};
        {ok, #{status := S}} -> {tool_error, describe(S)};
        {error, R} -> {tool_error, describe(R)}
    end.

describe({unknown_model, M}) -> <<"unknown model: ", M/binary>>;
describe(no_default_model) -> <<"no model named and no default model set">>;
describe(too_many_runs) -> <<"the release is busy; retry later">>;
describe(R) -> iolist_to_binary(io_lib:format("~0p", [R])).
