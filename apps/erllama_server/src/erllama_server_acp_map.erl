%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_acp_map).
-moduledoc false.
%% Pure mappings between ACP (JSON-RPC maps with binary keys, the
%% `acp` schema v1) and the release's agent API: request params to
%% `agent_start' specs, job events to `session/update' payloads,
%% turn reasons to stop reasons, and the JSON-RPC envelopes. No side
%% effects, no release calls, so the whole protocol surface is eunit
%% testable.

-export([
    initialize_result/0,
    new_session_spec/2,
    load_session_spec/1,
    prompt_text/1,
    modes/1,
    mode_from_id/1,
    mode_id/1,
    config_options/2,
    session_info/1,
    event_updates/1,
    stop_reason/1,
    tool_kind/1,
    permission_request/3,
    permission_decision/1,
    response/2,
    error/3,
    notification/2,
    request/3,
    incoming/1
]).

-export_type([mode/0]).

-type mode() :: default | yes | read_only.

-define(PROTOCOL_VERSION, 1).
-define(JSONRPC, <<"2.0">>).

%% ---------------------------------------------------------------------------
%% initialize
%% ---------------------------------------------------------------------------

initialize_result() ->
    #{
        <<"protocolVersion">> => ?PROTOCOL_VERSION,
        <<"agentCapabilities">> => #{
            <<"loadSession">> => true,
            <<"promptCapabilities">> => #{
                <<"audio">> => false,
                <<"embeddedContext">> => true,
                <<"image">> => true
            },
            <<"mcpCapabilities">> => #{<<"http">> => false, <<"sse">> => false},
            <<"sessionCapabilities">> => #{
                <<"list">> => #{},
                <<"close">> => #{},
                <<"delete">> => #{}
            }
        },
        <<"agentInfo">> => #{
            <<"name">> => <<"erllama">>,
            <<"title">> => <<"erllama">>,
            <<"version">> => version()
        },
        <<"authMethods">> => []
    }.

version() ->
    case application:get_key(erllama_server, vsn) of
        {ok, V} -> list_to_binary(V);
        _ -> <<"0.2.0">>
    end.

%% ---------------------------------------------------------------------------
%% session/new, session/load, session/prompt params
%% ---------------------------------------------------------------------------

%% `Params': the ACP request params; `Defaults' from the CLI flags.
-spec new_session_spec(map(), map()) -> {ok, map()} | {error, term()}.
new_session_spec(Params, Defaults) ->
    %% `additionalDirectories' are extra read roots the client grants;
    %% the release's tool sandbox is the one `cwd' workspace, so they
    %% are accepted and ignored. MCP servers cannot be run by this ACP
    %% agent (see the initialize capabilities), so an empty list is the
    %% only thing that maps to a session.
    case maps:get(<<"mcpServers">>, Params, []) of
        [] -> new_session_spec_1(Params, Defaults);
        _ -> {error, mcp_servers_unsupported}
    end.

new_session_spec_1(Params, Defaults) ->
    Cwd =
        case maps:get(workspace, Defaults, undefined) of
            undefined -> maps:get(<<"cwd">>, Params, undefined);
            W -> W
        end,
    case Cwd of
        C when is_binary(C), C =/= <<>> ->
            Spec0 = #{workspace => C},
            Spec1 =
                case maps:get(model, Defaults, undefined) of
                    undefined -> Spec0;
                    M -> Spec0#{model => M}
                end,
            case maps:get(policy, Defaults, default) of
                P when P =:= yes; P =:= read_only; P =:= default -> {ok, Spec1#{policy => P}};
                _ -> {ok, Spec1#{policy => default}}
            end;
        _ ->
            {error, missing_cwd}
    end.

-spec load_session_spec(map()) -> {ok, map()} | {error, term()}.
load_session_spec(Params) ->
    case maps:get(<<"sessionId">>, Params, undefined) of
        Sid when is_binary(Sid), Sid =/= <<>> -> {ok, #{conversation_id => Sid}};
        _ -> {error, missing_session_id}
    end.

%% Join the text blocks of a prompt; anything else is refused in v1.
-spec prompt_text(map()) -> {ok, binary()} | {error, term()}.
prompt_text(Params) ->
    Blocks = maps:get(<<"prompt">>, Params, []),
    Texts = [
        T
     || #{<<"type">> := <<"text">>, <<"text">> := T} <- Blocks
    ],
    NonText = [
        maps:get(<<"type">>, B, <<"?">>)
     || B <- Blocks, maps:get(<<"type">>, B, <<>>) =/= <<"text">>
    ],
    case {Texts, NonText} of
        {_, [T | _]} -> {error, {unsupported_content, T}};
        {[], []} -> {error, empty_prompt};
        _ -> {ok, iolist_to_binary(Texts)}
    end.

%% ---------------------------------------------------------------------------
%% modes and config options
%% ---------------------------------------------------------------------------

-spec modes(mode()) -> map().
modes(Current) ->
    #{
        <<"currentModeId">> => mode_id(Current),
        <<"availableModes">> => [
            #{<<"id">> => mode_id(M), <<"name">> => mode_name(M)}
         || M <- [default, yes, read_only]
        ]
    }.

mode_id(default) -> <<"default">>;
mode_id(yes) -> <<"yes">>;
mode_id(read_only) -> <<"read_only">>.

mode_name(default) -> <<"Default">>;
mode_name(yes) -> <<"Yes (auto-approve)">>;
mode_name(read_only) -> <<"Read-only">>.

-spec mode_from_id(binary()) -> {ok, mode()} | {error, unknown_mode}.
mode_from_id(<<"default">>) -> {ok, default};
mode_from_id(<<"yes">>) -> {ok, yes};
mode_from_id(<<"read_only">>) -> {ok, read_only};
mode_from_id(_) -> {error, unknown_mode}.

%% One select for the release's models, as vibe-acp exposes `model`.
-spec config_options([map()], binary()) -> [map()].
config_options(Models, Current) ->
    [
        #{
            <<"type">> => <<"select">>,
            <<"id">> => <<"model">>,
            <<"name">> => <<"Model">>,
            <<"currentValue">> => Current,
            <<"options">> => [
                #{<<"value">> => N, <<"name">> => N}
             || #{name := N} <- Models
            ]
        }
    ].

%% One `SessionInfo' from an `agent_sessions/0' entry.
-spec session_info(map()) -> map().
session_info(S) ->
    #{
        <<"sessionId">> => maps:get(session_id, S),
        <<"cwd">> => maps:get(cwd, S),
        <<"updatedAt">> => rfc3339(maps:get(updated_at, S, 0))
    }.

rfc3339(Ms) when is_integer(Ms) ->
    unicode:characters_to_binary(
        calendar:system_time_to_rfc3339(Ms, [{unit, millisecond}, {offset, "Z"}])
    );
rfc3339(_) ->
    <<>>.

%% ---------------------------------------------------------------------------
%% job events -> session updates
%% ---------------------------------------------------------------------------

%% Side-effect markers (`{permission, _}', `{turn_done, _}',
%% `session_ended') are handled by the dispatcher; the rest become
%% `session/update' notification updates.
-spec event_updates(map()) -> [map() | {permission, map()} | {turn_done, map()} | session_ended].
event_updates(#{type := turn_started, data := #{input := Text}}) ->
    [update(<<"user_message_chunk">>, #{<<"content">> => text_content(Text)})];
event_updates(#{type := token, data := Bin}) ->
    [update(<<"agent_message_chunk">>, #{<<"content">> => text_content(Bin)})];
event_updates(#{type := thinking, data := Bin}) ->
    [update(<<"agent_thought_chunk">>, #{<<"content">> => text_content(Bin)})];
event_updates(#{type := tool_started, data := #{call_id := Id, name := Name, arguments := Args}}) ->
    [
        update(<<"tool_call">>, #{
            <<"toolCallId">> => Id,
            <<"title">> => Name,
            <<"kind">> => tool_kind(Name),
            <<"status">> => <<"in_progress">>,
            <<"rawInput">> => Args,
            <<"locations">> => locations(Args)
        })
    ];
event_updates(#{type := tool_output, data := #{call_id := Id} = Data}) ->
    Content = text_content(erllama_server_tool_shell:display_output(Data)),
    [
        update(<<"tool_call_update">>, #{
            <<"toolCallId">> => Id,
            <<"status">> => <<"in_progress">>,
            <<"content">> => [#{<<"type">> => <<"content">>, <<"content">> => Content}]
        })
    ];
event_updates(#{type := tool_result, data := #{call_id := Id, ok := Ok, output := Out}}) ->
    [
        update(<<"tool_call_update">>, #{
            <<"toolCallId">> => Id,
            <<"status">> =>
                case Ok of
                    true -> <<"completed">>;
                    false -> <<"failed">>
                end,
            <<"content">> => [text_content(Out)]
        })
    ];
event_updates(#{type := approval_request, data := Req}) ->
    [{permission, Req}];
event_updates(#{type := turn_done, data := Data}) ->
    [{turn_done, Data}];
event_updates(#{type := T}) when T =:= done; T =:= cancelled; T =:= error ->
    [session_ended];
event_updates(_) ->
    [].

update(SessionUpdate, Fields) ->
    maps:merge(#{<<"sessionUpdate">> => SessionUpdate}, Fields).

text_content(Text) when is_binary(Text) ->
    #{<<"type">> => <<"text">>, <<"text">> => Text}.

locations(Args) when is_map(Args) ->
    case maps:get(<<"path">>, Args, undefined) of
        P when is_binary(P) -> [#{<<"path">> => P}];
        _ -> []
    end;
locations(_) ->
    [].

-spec tool_kind(binary()) -> binary().
tool_kind(Name) when is_binary(Name) ->
    case Name of
        <<"read_file">> ->
            <<"read">>;
        <<"list_dir">> ->
            <<"read">>;
        <<"search">> ->
            <<"search">>;
        <<"write_file">> ->
            <<"edit">>;
        <<"edit_file">> ->
            <<"edit">>;
        _ ->
            case binary:match(Name, <<"shell">>) of
                {0, _} -> <<"execute">>;
                _ -> <<"other">>
            end
    end.

%% turn_done's `reason' to a PromptResponse stopReason.
-spec stop_reason(map()) -> binary().
stop_reason(#{reason := #{error := _}}) ->
    <<"refusal">>;
stop_reason(#{reason := cancelled}) ->
    <<"cancelled">>;
stop_reason(#{reason := #{budget := tokens}}) ->
    <<"max_tokens">>;
stop_reason(#{reason := #{budget := _}}) ->
    <<"max_turn_requests">>;
stop_reason(_) ->
    <<"end_turn">>.

%% ---------------------------------------------------------------------------
%% session/request_permission (agent -> client)
%% ---------------------------------------------------------------------------

-spec permission_request(term(), binary(), map()) -> map().
permission_request(ReqId, Sid, #{call_id := CallId, name := Name, arguments := Args}) ->
    request(
        ReqId,
        <<"session/request_permission">>,
        #{
            <<"sessionId">> => Sid,
            <<"toolCall">> => #{
                <<"toolCallId">> => CallId,
                <<"title">> => Name,
                <<"kind">> => tool_kind(Name),
                <<"status">> => <<"pending">>,
                <<"rawInput">> => Args
            },
            <<"options">> => [
                #{<<"optionId">> => <<"allow_once">>, <<"name">> => <<"Allow once">>},
                #{<<"optionId">> => <<"allow_always">>, <<"name">> => <<"Always allow this tool">>},
                #{<<"optionId">> => <<"reject_once">>, <<"name">> => <<"Reject">>}
            ]
        }
    ).

%% The client's answer to a permission request, to `agent_approve/3`.
-spec permission_decision(map()) -> allow | always | deny.
permission_decision(#{<<"outcome">> := #{<<"outcome">> := <<"selected">>, <<"optionId">> := Opt}}) ->
    case Opt of
        <<"allow_once">> -> allow;
        <<"allow_always">> -> always;
        _ -> deny
    end;
permission_decision(_) ->
    deny.

%% ---------------------------------------------------------------------------
%% JSON-RPC envelopes
%% ---------------------------------------------------------------------------

-spec response(term(), map()) -> map().
response(Id, Result) ->
    #{<<"jsonrpc">> => ?JSONRPC, <<"id">> => Id, <<"result">> => Result}.

-spec error(term(), integer(), term()) -> map().
error(Id, Code, Msg) ->
    #{
        <<"jsonrpc">> => ?JSONRPC,
        <<"id">> => Id,
        <<"error">> => #{<<"code">> => Code, <<"message">> => format_msg(Msg)}
    }.

-spec notification(binary(), map()) -> map().
notification(Method, Params) ->
    #{<<"jsonrpc">> => ?JSONRPC, <<"method">> => Method, <<"params">> => Params}.

-spec request(term(), binary(), map()) -> map().
request(Id, Method, Params) ->
    #{<<"jsonrpc">> => ?JSONRPC, <<"id">> => Id, <<"method">> => Method, <<"params">> => Params}.

%% Classify one decoded frame from the client.
-spec incoming(map()) ->
    {request, term(), binary(), map()}
    | {response, term(), map()}
    | {error_response, term(), integer(), map()}
    | {notification, binary(), map()}
    | invalid.
incoming(#{<<"method">> := M, <<"id">> := Id} = F) when is_binary(M) ->
    {request, Id, M, maps:get(<<"params">>, F, #{})};
incoming(#{<<"method">> := M} = F) when is_binary(M) ->
    {notification, M, maps:get(<<"params">>, F, #{})};
incoming(#{<<"id">> := Id, <<"result">> := R}) ->
    {response, Id, R};
incoming(#{<<"id">> := Id, <<"error">> := #{<<"code">> := C} = E}) ->
    {error_response, Id, C, E};
incoming(_) ->
    invalid.

format_msg(Msg) when is_binary(Msg) -> Msg;
format_msg(Msg) when is_atom(Msg) -> atom_to_binary(Msg);
format_msg(Msg) -> unicode:characters_to_binary(io_lib:format("~p", [Msg])).
