%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_cli_mcp).
-moduledoc false.
%% `erllama mcp': the MCP servers the release is configured with, the
%% tools they give agent sessions, and their resources.

-export([mcp/4]).

-spec mcp(node(), map(), map(), [string()]) -> {non_neg_integer(), iodata()}.
mcp(Node, _Info, _Opts, []) ->
    erllama_server_cli:reply(erllama_server_client:call(Node, mcp, []), fun servers/1);
mcp(Node, _Info, _Opts, ["tools"]) ->
    erllama_server_cli:reply(erllama_server_client:call(Node, mcp_tools, []), fun tools/1);
mcp(Node, _Info, _Opts, ["resources", Server]) ->
    text(erllama_server_client:call(Node, mcp_resources, [bin(Server)]));
mcp(Node, _Info, _Opts, ["read", Server, Uri]) ->
    text(erllama_server_client:call(Node, mcp_read, [bin(Server), bin(Uri)]));
mcp(_, _, _, _) ->
    erllama_server_cli:usage("mcp").

servers([]) ->
    "no MCP servers configured\n";
servers(L) ->
    [
        io_lib:format("~ts  ~p  ~p  ~b tools~ts~n", [N, Tr, St, T, error_note(E)])
     || #{name := N, transport := Tr, status := St, tools := T, error := E} <- L
    ].

error_note(undefined) -> "";
error_note(E) -> io_lib:format("  (~0p)", [E]).

tools([]) ->
    "no MCP tools available\n";
tools(L) ->
    [
        io_lib:format("~ts  ~p  ~p  ~ts~n", [N, P, C, first_line(D)])
     || #{name := N, policy := P, class := C, description := D} <- L
    ].

first_line(D) -> hd(binary:split(D, <<"\n">>) ++ [<<>>]).

text({ok, T}) -> {0, [T, "\n"]};
text({error, T}) when is_binary(T) -> {1, ["erllama: ", T, "\n"]};
text({error, R}) -> erllama_server_cli:fail(R).

bin(S) -> unicode:characters_to_binary(S).
