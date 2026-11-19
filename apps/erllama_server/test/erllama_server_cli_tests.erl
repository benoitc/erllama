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

parse_opts_test() ->
    Spec = [{"--id", id, string}, {"--n-ctx", n_ctx, int}, {"--detach", detach, flag}],
    ?assertEqual(
        {ok, #{id => "tiny", n_ctx => 2048, detach => true}, ["a", "b"]},
        erllama_server_cli:parse_opts(
            ["a", "--id", "tiny", "--detach", "b", "--n-ctx", "2048"], Spec
        )
    ),
    ?assertEqual(
        {ok, #{}, ["--id", "x"]}, erllama_server_cli:parse_opts(["--", "--id", "x"], Spec)
    ),
    ?assertMatch({error, _}, erllama_server_cli:parse_opts(["--n-ctx", "lots"], Spec)),
    ?assertMatch({error, _}, erllama_server_cli:parse_opts(["--id"], Spec)),
    ?assertMatch({error, _}, erllama_server_cli:parse_opts(["--nope"], Spec)).

bad_options_are_usage_errors_test() ->
    {2, Out} = out(erllama_server_cli:run(["load", "--n-ctx", "x", "f.gguf"])),
    ?assertNotEqual(nomatch, binary:match(Out, <<"--n-ctx">>)).

error_messages_test() ->
    Msg = fun(R) -> unicode:characters_to_binary(erllama_server_cli:format_error(R)) end,
    ?assertNotEqual(nomatch, binary:match(Msg({unreachable, 'a@h'}), <<"not reachable at a@h">>)),
    ?assertNotEqual(
        nomatch,
        binary:match(Msg({incompatible_api, #{server => 2, client => 1}}), <<"API version 2">>)
    ),
    ?assertNotEqual(
        nomatch, binary:match(Msg({no_service_api, 'a@h'}), <<"no erllama service API">>)
    ),
    ?assertNotEqual(
        nomatch, binary:match(Msg({model_not_loaded, <<"m">>}), <<"model m is not loaded">>)
    ).

%% Client node names must not repeat across CLI processes started at
%% the same moment: the OS pid tells live processes apart, the random
%% part covers pid reuse. (unique_integer/1, the old suffix, repeats
%% between fresh VMs.)
client_names_are_unique_across_processes_test() ->
    A = atom_to_list(erllama_server_client:client_name("127.0.0.1")),
    B = atom_to_list(erllama_server_client:client_name("127.0.0.1")),
    ?assertNotEqual(A, B),
    ?assertMatch("erllama_cli_" ++ _, A),
    ?assertNotEqual(nomatch, string:find(A, "_" ++ os:getpid() ++ "_")),
    ?assertMatch([_, "127.0.0.1"], string:split(A, "@")).
