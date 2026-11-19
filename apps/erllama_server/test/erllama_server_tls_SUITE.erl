%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% Remote administration over TLS distribution: the release node and
%% the client node both run `-proto_dist inet_tls' with certificates
%% from a test CA, and the client connects, checks the API and
%% follows a run as it does over plain distribution.
%% @end
-module(erllama_server_tls_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([run_over_tls_distribution/1]).

all() -> [run_over_tls_distribution].

init_per_suite(Config) ->
    case application:ensure_all_started(ssl) of
        {ok, _} ->
            OptFile = write_tls_files(?config(priv_dir, Config)),
            [{optfile, OptFile} | Config];
        {error, R} ->
            {skip, {no_ssl, R}}
    end.

end_per_suite(_Config) ->
    ok.

run_over_tls_distribution(Config) ->
    OptFile = ?config(optfile, Config),
    {ok, Rel, RelNode} = start_peer(erllama_server_tls_rel, OptFile, []),
    try
        {ok, _} = peer:call(Rel, application, ensure_all_started, [erllama_server]),
        {ok, _} = peer:call(Rel, erllama, load_model, [<<"stub">>, #{backend => erllama_model_stub}]),
        {ok, Cli, _} = start_peer(erllama_server_tls_cli, OptFile, ["-hidden"]),
        try
            {ok, _, #{api := _}} = peer:call(Cli, erllama_server_client, connect, [RelNode]),
            ?assertEqual(tls, protocol(Cli, RelNode)),
            Spec = #{model => <<"stub">>, messages => [#{role => user, content => <<"over tls">>}]},
            {ok, #{run_id := Run}} = peer:call(Cli, erllama_server_client, call, [
                RelNode, start_run, [Spec]
            ]),
            {ok, #{type := done}} =
                peer:call(
                    Cli, erllama_server_client, follow, [RelNode, Run, 0, fun(_) -> ok end], 30000
                )
        after
            peer:stop(Cli)
        end
    after
        peer:stop(Rel)
    end.

%% The protocol of the client's live connection to the release
%% (`#net_address.protocol').
protocol(Cli, Node) ->
    {ok, Info} = peer:call(Cli, net_kernel, node_info, [Node]),
    element(4, proplists:get_value(address, Info)).

start_peer(Prefix, OptFile, Extra) ->
    Paths = [P || P <- code:get_path(), string:find(P, "_build") =/= nomatch],
    peer:start(#{
        name => peer:random_name(Prefix),
        host => "127.0.0.1",
        longnames => true,
        connection => standard_io,
        args =>
            [
                "-setcookie",
                "erllama_tls_suite",
                "-proto_dist",
                "inet_tls",
                "-ssl_dist_optfile",
                OptFile
                | lists:append([["-pa", P] || P <- Paths])
            ] ++ Extra
    }).

%% A test CA, a server and a client certificate, and the
%% `-ssl_dist_optfile' both nodes use. P-256 keys with SHA-256: the
%% generator's default curve has no signature algorithm TLS accepts.
write_tls_files(Dir) ->
    K = [{key, {namedCurve, secp256r1}}, {digest, sha256}],
    #{server_config := S, client_config := C} = public_key:pkix_test_data(#{
        server_chain => #{root => K, intermediates => [], peer => K},
        client_chain => #{root => K, intermediates => [], peer => K}
    }),
    Server = write_side(Dir, "server", S),
    Client = write_side(Dir, "client", C),
    Opts = [
        {server, Server ++ [{verify, verify_peer}, {fail_if_no_peer_cert, true}]},
        {client, Client ++ [{verify, verify_peer}, {server_name_indication, disable}]}
    ],
    File = filename:join(Dir, "ssl_dist.conf"),
    ok = file:write_file(File, io_lib:format("~p.~n", [Opts])),
    File.

write_side(Dir, Name, Conf) ->
    Cert = proplists:get_value(cert, Conf),
    {KeyType, KeyDer} = proplists:get_value(key, Conf),
    CaCerts = proplists:get_value(cacerts, Conf),
    CertFile = filename:join(Dir, Name ++ "_cert.pem"),
    KeyFile = filename:join(Dir, Name ++ "_key.pem"),
    CaFile = filename:join(Dir, Name ++ "_ca.pem"),
    ok = file:write_file(CertFile, public_key:pem_encode([{'Certificate', Cert, not_encrypted}])),
    ok = file:write_file(KeyFile, public_key:pem_encode([{KeyType, KeyDer, not_encrypted}])),
    ok = file:write_file(
        CaFile, public_key:pem_encode([{'Certificate', D, not_encrypted} || D <- CaCerts])
    ),
    [{certfile, CertFile}, {keyfile, KeyFile}, {cacertfile, CaFile}].
