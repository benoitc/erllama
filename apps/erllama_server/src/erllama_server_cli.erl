%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_cli).
-moduledoc false.
%% The `erllama` command-line client, built as an escript and shipped
%% beside the release (`bin/erllama` runs it on the release's own
%% runtime). It talks to a running release over Erlang distribution
%% from a temporary hidden node; it never starts the erllama
%% application or loads its NIF.
%%
%% Environment:
%%   ERLLAMA_NODE    release node (default erllama_server@127.0.0.1)
%%   ERLLAMA_COOKIE  distribution cookie (default: ~/.erlang.cookie)

-export([main/1, run/1]).

-define(DEFAULT_NODE, "erllama_server@127.0.0.1").

-spec main([string()]) -> no_return().
main(Args) ->
    {Code, Output} = run(Args),
    io:put_chars(Output),
    halt(Code).

%% Run one command and return its exit code and output, so the
%% commands can be tested without halting the VM.
-spec run([string()]) -> {non_neg_integer(), iodata()}.
run(["--version"]) ->
    {0, ["erllama ", version(), "\n"]};
run(["ping"]) ->
    ping(release_node());
run(["help"]) ->
    {0, usage()};
run([]) ->
    {0, usage()};
run(Other) ->
    {2, ["erllama: unknown command: ", lists:join(" ", Other), "\n", usage()]}.

usage() ->
    [
        "usage: erllama <command>\n",
        "\n",
        "commands:\n",
        "  ping        check that the release is running\n",
        "  --version   print the client version\n",
        "\n",
        "The release node is ERLLAMA_NODE (default " ?DEFAULT_NODE "); the\n",
        "cookie is ERLLAMA_COOKIE or ~/.erlang.cookie.\n"
    ].

version() ->
    _ = application:load(erllama_server),
    case application:get_key(erllama_server, vsn) of
        {ok, Vsn} -> Vsn;
        undefined -> "unknown"
    end.

release_node() ->
    list_to_atom(os:getenv("ERLLAMA_NODE", ?DEFAULT_NODE)).

ping(Node) ->
    case start_client_node(Node) of
        ok ->
            case net_adm:ping(Node) of
                pong ->
                    {0, ["pong ", atom_to_list(Node), "\n"]};
                pang ->
                    {1, [
                        "erllama: release not reachable at ",
                        atom_to_list(Node),
                        " (is it running, and is the cookie the same?)\n"
                    ]}
            end;
        {error, Reason} ->
            {1, ["erllama: cannot start a client node: ", io_lib:format("~p", [Reason]), "\n"]}
    end.

%% A hidden node with a unique name on the release node's host, so a
%% client neither shows up in the release's nodes() nor clashes with
%% another client.
start_client_node(Node) ->
    [_, Host] = string:split(atom_to_list(Node), "@"),
    Name = list_to_atom(
        "erllama_cli_" ++ integer_to_list(erlang:unique_integer([positive])) ++ "@" ++ Host
    ),
    case net_kernel:start(Name, #{name_domain => longnames, hidden => true}) of
        {ok, _} -> set_cookie(os:getenv("ERLLAMA_COOKIE"));
        {error, {already_started, _}} -> set_cookie(os:getenv("ERLLAMA_COOKIE"));
        {error, _} = E -> E
    end.

set_cookie(false) ->
    ok;
set_cookie(Cookie) ->
    true = erlang:set_cookie(list_to_atom(Cookie)),
    ok.
