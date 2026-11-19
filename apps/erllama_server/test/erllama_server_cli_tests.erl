%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_cli_tests).
-include_lib("eunit/include/eunit.hrl").

out({Code, IoData}) ->
    {Code, unicode:characters_to_binary(IoData)}.

version_test() ->
    {0, Out} = out(erllama_server_cli:run(["--version"])),
    ?assertMatch(<<"erllama ", _/binary>>, Out).

no_command_prints_usage_test() ->
    {0, Out} = out(erllama_server_cli:run([])),
    ?assertNotEqual(nomatch, binary:match(Out, <<"usage: erllama">>)).

unknown_command_is_a_usage_error_test() ->
    {2, Out} = out(erllama_server_cli:run(["frobnicate"])),
    ?assertNotEqual(nomatch, binary:match(Out, <<"unknown command: frobnicate">>)).

ping_unreachable_release_test_() ->
    {timeout, 30, fun() ->
        os:putenv("ERLLAMA_NODE", "erllama_server_absent_test@127.0.0.1"),
        try
            {1, Out} = out(erllama_server_cli:run(["ping"])),
            %% Either the release is not reachable or (no epmd on this
            %% host) no client node could start: both are clear errors.
            ?assertMatch(<<"erllama: ", _/binary>>, Out)
        after
            os:unsetenv("ERLLAMA_NODE")
        end
    end}.

%% The CLI must never load the inference application or its NIF. Run
%% it in a fresh node with only erllama_server on the code path, as in
%% the escript, and look at what got loaded there.
commands_do_not_load_erllama_test_() ->
    {timeout, 60, fun() ->
        Ebin = filename:dirname(code:which(erllama_server_cli)),
        {ok, Peer, Node} = peer:start(#{
            name => peer:random_name(?MODULE),
            connection => standard_io,
            args => ["-pa", Ebin]
        }),
        try
            {0, _} = peer:call(Peer, erllama_server_cli, run, [["--version"]]),
            {0, _} = peer:call(Peer, erllama_server_cli, run, [["help"]]),
            Loaded = [M || {M, _} <- peer:call(Peer, code, all_loaded, [])],
            ?assertNot(lists:member(erllama, Loaded)),
            ?assertNot(lists:member(erllama_nif, Loaded)),
            ?assert(is_atom(Node))
        after
            peer:stop(Peer)
        end
    end}.
