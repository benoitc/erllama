%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_agent_policy).
-moduledoc false.
%% What an agent session may run, and how: the native tools with their
%% defaults, the MCP tools of this round's catalog snapshot, and the
%% policy (`auto', `ask', `deny') that applies to each call.
%%
%% A session's policy map holds its choices by tool name, and the
%% `always' approvals of MCP tools by their identity `{mcp, Server,
%% RemoteName}', so such a permission stays with that tool.

-export([initial/1, snapshot/0, policy/4, class/2, known/2, grant_key/2, decide/4, budget_left/3]).

-type policy() :: erllama_server_tools:policy().
-type policies() :: #{binary() | {mcp, binary(), term()} => policy()}.
-type mode() :: default | yes | read_only.
-type catalog() :: #{binary() => map()}.
-export_type([policy/0, policies/0, mode/0, catalog/0]).

%% The native tools' defaults, overridden by the session's choices.
-spec initial(map()) -> policies().
initial(Spec) ->
    maps:merge(
        maps:from_list([
            {N, erllama_server_tools:default_policy(N)}
         || N <- erllama_server_tools:names()
        ]),
        maps:get(policy, Spec, #{})
    ).

%% This round's MCP tools, by alias.
-spec snapshot() -> catalog().
snapshot() ->
    case erllama_server_config:get(mcp_servers) of
        [] -> #{};
        _ -> maps:from_list([{N, E} || #{name := N} = E <- erllama_server_mcp:catalog()])
    end.

%% The policy of a tool. A native tool's is in `Policies' (filled from
%% the defaults at start). For an MCP tool, a server the operator set
%% to `deny' stays denied; otherwise what the session allowed for that
%% very tool, then its choice for the name, then its mode, then the
%% server's policy decide.
-spec policy(binary(), policies(), catalog(), mode()) -> policy().
policy(Name, Policies, Catalog, Mode) ->
    case Catalog of
        #{Name := #{policy := deny}} ->
            deny;
        #{Name := #{server := S, remote := R, class := Class, policy := P}} ->
            case Policies of
                #{{mcp, S, R} := Allowed} -> Allowed;
                #{Name := Chosen} -> Chosen;
                _ -> mode_policy(Mode, Class, P)
            end;
        _ ->
            maps:get(Name, Policies, deny)
    end.

mode_policy(yes, _Class, _P) -> auto;
mode_policy(read_only, write, _P) -> deny;
mode_policy(_Mode, _Class, P) -> P.

-spec class(binary(), catalog()) -> erllama_server_tools:class().
class(Name, Catalog) ->
    case Catalog of
        #{Name := #{class := C}} -> C;
        _ -> erllama_server_tools:class(Name)
    end.

-spec known(binary(), catalog()) -> boolean().
known(Name, Catalog) ->
    is_map_key(Name, Catalog) orelse lists:member(Name, erllama_server_tools:names()).

%% The key an `always' approval is kept under.
-spec grant_key(binary(), map() | undefined) -> binary() | {mcp, binary(), term()}.
grant_key(_Name, #{server := S, remote := R}) -> {mcp, S, R};
grant_key(Name, undefined) -> Name.

%% The user's answer to an approval request for tool `Name' (`Mcp' its
%% catalog entry, or undefined): `{done, Result}' for a declined call,
%% `queued' for one to run; `always' also allows the tool for the rest
%% of the session.
-spec decide(allow | deny | always, binary(), map() | undefined, policies()) ->
    {queued | {done, {error, binary()}}, policies()}.
decide(deny, _Name, _Mcp, Policies) ->
    {{done, {error, <<"the user declined this call">>}}, Policies};
decide(allow, _Name, _Mcp, Policies) ->
    {queued, Policies};
decide(always, Name, Mcp, Policies) ->
    {queued, Policies#{grant_key(Name, Mcp) => auto}}.

%% What a turn has left of its budgets: rounds, tokens and time
%% (`Now' on the clock its deadline was set with).
-spec budget_left(map(), map(), integer()) -> ok | {exceeded, rounds | tokens | time}.
budget_left(#{rounds := R, tokens := T, deadline := Dl}, B, Now) ->
    case {R >= maps:get(max_rounds, B), T >= maps:get(max_tokens, B), Now >= Dl} of
        {true, _, _} -> {exceeded, rounds};
        {_, true, _} -> {exceeded, tokens};
        {_, _, true} -> {exceeded, time};
        _ -> ok
    end.
