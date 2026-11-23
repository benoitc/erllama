%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Helpers for the opt-in tensor-parallel suites on CUDA hosts
%% (ERLLAMA_TP_HOSTS, prepared with scripts/tp_hosts.sh): rank nodes
%% started over ssh, a group formed through a coordinator, and NCCL's
%% log lines.
-module(erllama_tp_hosts).

-include_lib("common_test/include/ct.hrl").

-export([
    hosts/0,
    start_peer/3,
    start_peer/4,
    load_module/2,
    form_group/4,
    stop_ranks/1,
    home/2,
    model/0,
    nccl_lines/1,
    parallel/1
]).

-define(T, "$HOME/erllama-tp").

%% The hosts in rank order, or `false'.
hosts() ->
    case os:getenv("ERLLAMA_TP_HOSTS") of
        false -> false;
        Hosts -> string:lexemes(Hosts, ",")
    end.

%% A node named `Name' on `Host', erllama started. ERLLAMA_TP_NCCL_ENV
%% adds environment (space-separated K=V, e.g. "NCCL_NET=Socket"), and
%% for rank `Rank' ERLLAMA_TP_NCCL_ENV_<Rank> too.
start_peer(Name, Host, Cookie) ->
    start_peer(Name, Host, Cookie, undefined).

start_peer(Name, Host, Cookie, Rank) ->
    Extra =
        string:lexemes(os:getenv("ERLLAMA_TP_NCCL_ENV", ""), " ") ++
            case Rank of
                undefined ->
                    [];
                _ ->
                    string:lexemes(
                        os:getenv("ERLLAMA_TP_NCCL_ENV_" ++ integer_to_list(Rank), ""), " "
                    )
            end,
    Env = [
        "LD_LIBRARY_PATH=" ?T "/nccl/lib:" ?T "/openssl/lib",
        "NCCL_DEBUG=INFO",
        "NCCL_DEBUG_SUBSYS=INIT,NET",
        "NCCL_DEBUG_FILE=/tmp/erllama_tp.%h.%p.nccl"
        | Extra
    ],
    {ok, Peer, Node} = peer:start(#{
        name => Name,
        host => Host,
        longnames => true,
        connection => standard_io,
        exec => {os:find_executable("ssh"), [Host, "env" | Env] ++ [?T "/otp-28.3.1/bin/erl"]},
        args => [
            "-setcookie", Cookie, "-pa", ?T "/erllama/_build/test/lib/erllama/ebin", "-noshell"
        ],
        wait_boot => 120000
    }),
    {ok, _} = peer:call(Peer, application, ensure_all_started, [erllama], 120000),
    ct:pal("~p on ~s", [Node, Host]),
    {Peer, Node}.

%% Load a test module (not in the hosts' build) on a node.
load_module(Peer, Mod) ->
    {Mod, Bin, File} = code:get_object_code(Mod),
    {module, Mod} = peer:call(Peer, code, load_binary, [Mod, File, Bin]),
    ok.

%% Every rank joined at once through coordinator `Coord' (a pid on any
%% node): `[{Peer, RankPid}]' in rank order.
form_group(RankPeers, Coord, Gen, Timeout) ->
    World = length(RankPeers),
    Starts = parallel([
        fun() ->
            peer:call(
                P,
                erllama_tp_rank,
                start,
                [
                    #{
                        rank => R,
                        world => World,
                        join => {call, Coord},
                        timeout => Timeout,
                        generation => Gen
                    }
                ],
                Timeout + 30000
            )
        end
     || {R, P} <- lists:enumerate(0, RankPeers)
    ]),
    case [S || S <- Starts, element(1, S) =/= ok] of
        [] -> ok;
        Bad -> ct:fail({group_failed, Bad})
    end,
    [{P, Pid} || {P, {ok, Pid}} <- lists:zip(RankPeers, Starts)].

stop_ranks(Ranks) ->
    [catch peer:call(P, erllama_tp_rank, stop, [R], 30000) || {P, R} <- Ranks],
    ok.

%% `Path' under the node's home unless absolute.
home(P, Path) ->
    case Path of
        "/" ++ _ -> Path;
        _ -> filename:join(peer:call(P, os, getenv, ["HOME"]), Path)
    end.

model() ->
    os:getenv("ERLLAMA_TP_MODEL", "erllama-tp/tinyllama-1.1b-chat.gguf").

%% NCCL's version, network and channel lines from the node's log.
nccl_lines(Peer) ->
    OsPid = peer:call(Peer, os, getpid, []),
    Files = peer:call(Peer, filelib, wildcard, ["/tmp/erllama_tp.*." ++ OsPid ++ ".nccl"]),
    lists:append([
        [
            unicode:characters_to_list(L)
         || L <- binary:split(Bin, <<"\n">>, [global]),
            interesting(L)
        ]
     || F <- Files, {ok, Bin} <- [peer:call(Peer, file, read_file, [F])]
    ]).

interesting(L) ->
    lists:any(
        fun(P) -> binary:match(L, P) =/= nomatch end,
        [<<"NET/">>, <<"Using network">>, <<"via ">>, <<"NCCL version">>, <<"comm ">>]
    ).

parallel(Funs) ->
    Self = self(),
    Refs = [
        begin
            Ref = make_ref(),
            spawn(fun() -> Self ! {Ref, catch F()} end),
            Ref
        end
     || F <- Funs
    ],
    [
        receive
            {Ref, R} -> R
        after 300000 -> ct:fail(parallel_timeout)
        end
     || Ref <- Refs
    ].
