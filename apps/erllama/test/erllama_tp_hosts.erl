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
    parallel/1,
    with_serving/2,
    serve/4,
    serve/5,
    complete/3,
    complete_stats/3,
    stop_group/1,
    long_prompt/1
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

%% -----------------------------------------------------------------------------
%% A split model served from its own node

%% A serving node on the first host (erllama, the coordinator and the
%% stream helper; no GPU work) for `Fun(S)'.
with_serving([Host0 | _], Fun) ->
    Cookie = cookie(),
    {S, _} = start_peer(name("erllama_tps"), Host0, Cookie),
    put(cookie, Cookie),
    [ok = ?MODULE:load_module(S, M) || M <- [erllama_tp_coordinator, erllama_tp_stream]],
    try
        Fun(S)
    after
        peer:stop(S)
    end.

%% A group of generation `Gen' on fresh rank nodes, serving the model
%% from node `S' through erllama_model_tp.
serve(Hosts, S, Gen, Policy) ->
    serve(Hosts, S, Gen, Policy, #{}).

%% `Extra' adds to the load config (e.g. tp_kv_timeout).
serve(Hosts, S, Gen, Policy, Extra) ->
    Peers = [
        element(
            1, start_peer(name("erllama_tp" ++ integer_to_list(R)), H, get(cookie), R)
        )
     || {R, H} <- lists:enumerate(0, Hosts)
    ],
    {ok, Coord} = peer:call(S, erllama_tp_coordinator, start, [length(Hosts), 60000]),
    Ranks = form_group(Peers, Coord, Gen, 60000),
    RankPids = [R || {_, R} <- Ranks],
    ok = peer:call(S, erllama_tp_coordinator, watch, [Coord, RankPids]),
    [{P0, R0} | Followers] = Ranks,
    Model = model(),
    MOpts = #{n_gpu_layers => 99},
    COpts = #{n_ctx => 2048, n_batch => 512, flash_attn => enabled},
    Loaded = parallel([
        fun() ->
            peer:call(
                P,
                erllama_tp_rank,
                load,
                [R, home(P, Model), {MOpts, COpts}],
                600000
            )
        end
     || {P, R} <- Followers
    ]),
    [ok = L || L <- Loaded],
    Descs = [
        {R,
            peer:call(P, erllama_tp_rank, describe, [
                home(P, Model), {MOpts, COpts}, length(Hosts)
            ])}
     || {P, R} <- Followers
    ],
    Id = iolist_to_binary(["tp_gen", integer_to_list(Gen)]),
    {ok, Id} = peer:call(
        S,
        erllama,
        load_model,
        [
            Id,
            Extra#{
                backend => erllama_model_tp,
                model_path => home(P0, Model),
                model_opts => MOpts,
                context_opts => COpts,
                policy => Policy,
                tp_worker => R0,
                tp_followers => Descs
            }
        ],
        600000
    ),
    #{id => Id, coord => Coord, rank_peers => Peers, rank_pids => RankPids, serving => S}.

%% 32 greedy tokens after `Prompt', and the request's stats.
complete(S, G, Prompt) ->
    element(1, complete_stats(S, G, Prompt)).

complete_stats(S, #{id := Id}, Prompt) ->
    {ok, #{generated := G} = Stats} = peer:call(
        S, erllama, complete, [Id, Prompt, #{response_tokens => 32, temperature => 0.0}], 120000
    ),
    {G, Stats}.

stop_group(#{id := Id, coord := C, rank_peers := Peers, serving := S}) ->
    _ = catch peer:call(S, erllama, unload, [Id], 60000),
    _ = catch peer:call(S, erllama_tp_coordinator, stop, [C], 10000),
    [catch peer:stop(P) || P <- Peers],
    ok.

long_prompt(N) ->
    iolist_to_binary(lists:duplicate(N, <<"The quick brown fox jumps over the lazy dog. ">>)).

cookie() ->
    "erllama_tp_" ++ integer_to_list(erlang:unique_integer([positive])).

name(Base) ->
    list_to_atom(Base ++ "_" ++ integer_to_list(erlang:unique_integer([positive]))).
