%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Opt-in: a tensor-parallel reduction between ranks on separate CUDA
%% hosts, one GPU each, through the pinned CUDA backend's NCCL
%% communicator. Skipped unless ERLLAMA_TP_HOSTS names the hosts (ssh
%% targets, in rank order), each prepared with scripts/tp_hosts.sh
%% (setup, sync, build).
%%
%% Each rank is an Erlang node started over ssh (`peer', controlled
%% through its standard I/O) holding an erllama_tp_rank. The test
%% coordinator runs on rank 0's node; the other ranks reach it over
%% Erlang distribution, so the hosts must reach each other by the names
%% given. The tensors themselves go through NCCL. NCCL's INIT and NET
%% logs are kept per rank (NCCL_DEBUG_FILE) and the transport and NIC
%% it chose are logged.
%%
%%   ERLLAMA_TP_HOSTS=gx10-fa8e.local,gx10-e25e.local \
%%     rebar3 ct --dir apps/erllama/test --suite erllama_tp_nccl_SUITE
%%
%% ERLLAMA_TP_NCCL_ENV adds environment for the ranks (space-separated
%% K=V, e.g. "NCCL_IB_HCA=rocep1s0f0").
-module(erllama_tp_nccl_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1, init_per_testcase/2, end_per_testcase/2]).
-export([
    matvec_fixture/1,
    decode_sized_fp32/1,
    prefill_sized_bf16/1,
    missing_rank_fails_within_the_deadline/1,
    open_and_close_repeat/1,
    model_matches_one_gpu/1,
    uneven_split_matches_one_gpu/1,
    transport_in_use/1
]).

-define(T, "$HOME/erllama-tp").

all() ->
    [
        matvec_fixture,
        decode_sized_fp32,
        prefill_sized_bf16,
        missing_rank_fails_within_the_deadline,
        open_and_close_repeat,
        model_matches_one_gpu,
        uneven_split_matches_one_gpu,
        transport_in_use
    ].

init_per_suite(Config) ->
    case os:getenv("ERLLAMA_TP_HOSTS") of
        false ->
            {skip, "set ERLLAMA_TP_HOSTS to the CUDA hosts (see scripts/tp_hosts.sh)"};
        Hosts0 ->
            Hosts = string:lexemes(Hosts0, ","),
            Cookie = "erllama_tp_" ++ integer_to_list(erlang:unique_integer([positive])),
            Peers = [start_peer(R, H, Cookie) || {R, H} <- lists:enumerate(0, Hosts)],
            [{_, P0, _} | _] = Peers,
            {M, Bin, File} = code:get_object_code(erllama_tp_coordinator),
            {module, M} = peer:call(P0, code, load_binary, [M, File, Bin]),
            {ok, Caps} = peer:call(P0, erllama_tp, caps, []),
            ct:pal("rank 0 caps: ~p", [Caps]),
            case Caps of
                #{init_rank := true} -> [{peers, Peers}, {world, length(Hosts)} | Config];
                _ -> {skip, "the hosts' build has no per-process communicator (CUDA with NCCL)"}
            end
    end.

end_per_suite(Config) ->
    [peer:stop(P) || {_, P, _} <- proplists:get_value(peers, Config, [])],
    ok.

init_per_testcase(transport_in_use, Config) ->
    Config;
init_per_testcase(missing_rank_fails_within_the_deadline, Config) ->
    Config;
init_per_testcase(_TC, Config) ->
    Ranks = form_group(Config, 60000),
    [{ranks, Ranks} | Config].

end_per_testcase(_TC, Config) ->
    stop_ranks(Config, proplists:get_value(ranks, Config, [])),
    ok.

%% -----------------------------------------------------------------------------

%% The plan's fixture: rows [1,2,3,4] and [4,3,2,1], input [1,2,3,4],
%% columns split in two. Each rank checks its own partial, and both get
%% the sum.
matvec_fixture(Config) ->
    2 = ?config(world, Config),
    [{P0, R0}, {P1, R1}] = rank_pids(Config),
    [{ok, Part0, Sum0}, {ok, Part1, Sum1}] = parallel([
        fun() -> peer:call(P0, erllama_tp_rank, matvec, [R0, [[1, 2], [4, 3]], [1, 2]], 60000) end,
        fun() -> peer:call(P1, erllama_tp_rank, matvec, [R1, [[3, 4], [2, 1]], [3, 4]], 60000) end
    ]),
    ct:pal("partials ~p and ~p, sums ~p and ~p", [Part0, Part1, Sum0, Sum1]),
    ?assertEqual([5.0, 10.0], Part0),
    ?assertEqual([25.0, 10.0], Part1),
    ?assertEqual([30.0, 20.0], Sum0),
    ?assertEqual([30.0, 20.0], Sum1).

%% 4096 values per rank (a decode-sized activation): under the
%% backend's 32768-element threshold, reduced in FP32.
decode_sized_fp32(Config) ->
    reduce_case(Config, 4096, 1.0e-6, fp32).

%% 512 x 4096 values per rank (a prefill-sized activation): over the
%% threshold, compressed to BF16 for the reduction.
prefill_sized_bf16(Config) ->
    reduce_case(Config, 512 * 4096, 1.6e-2, bf16).

%% Only rank 0 joins: it fails at the coordinator's deadline, and its
%% node stays usable.
missing_rank_fails_within_the_deadline(Config) ->
    [{_, P0, _} | _] = ?config(peers, Config),
    {ok, Coord} = peer:call(P0, erllama_tp_coordinator, start, [?config(world, Config), 2000]),
    T0 = erlang:monotonic_time(millisecond),
    Result = peer:call(
        P0,
        erllama_tp_rank,
        start,
        [#{rank => 0, world => ?config(world, Config), join => {call, Coord}, timeout => 20000}],
        60000
    ),
    Ms = erlang:monotonic_time(millisecond) - T0,
    ct:pal("rank 0 alone: ~p after ~b ms", [Result, Ms]),
    ?assertEqual({error, join_timeout}, Result),
    ?assert(Ms < 10000),
    ?assertMatch({ok, #{init_rank := true}}, peer:call(P0, erllama_tp, caps, [])).

%% Open, reduce and close three times in a row on the same nodes.
open_and_close_repeat(Config) ->
    Ranks0 = ?config(ranks, Config),
    stop_ranks(Config, Ranks0),
    lists:foreach(
        fun(I) ->
            Ranks = form_group(Config, 60000),
            Results = parallel([
                fun() -> peer:call(P, erllama_tp_rank, reduce, [R, [float(I + Rank)]], 60000) end
             || {Rank, {P, R}} <- lists:enumerate(0, Ranks)
            ]),
            Expected = float(lists:sum([I + Rank || Rank <- lists:seq(0, length(Ranks) - 1)])),
            [?assertEqual({ok, [Expected]}, Res) || Res <- Results],
            Info = [peer:call(P, erllama_tp_rank, info, [R]) || {P, R} <- Ranks],
            ct:pal("round ~b: ~p", [I, [maps:with([rank, devices], X) || X <- Info]]),
            stop_ranks(Config, Ranks)
        end,
        [1, 2, 3]
    ).

%% A GGUF split over the group with llama.cpp's tensor split, each rank
%% holding and computing its slices (ERLLAMA_TP_MODEL, a path on every
%% host, default the TinyLlama the setup copies). Its logits for a
%% prompt, and for the next token from the cache, against the same
%% model on one GPU.
model_matches_one_gpu(Config) ->
    split_model(Config, #{}).

%% The same with an uneven split: rank 0 takes three quarters of each
%% split tensor and rank 1 one quarter (llama.cpp rotates the
%% proportions from layer to layer).
uneven_split_matches_one_gpu(Config) ->
    2 = ?config(world, Config),
    split_model(Config, #{tensor_split => [3.0, 1.0]}).

split_model(Config, Split) ->
    Model = os:getenv("ERLLAMA_TP_MODEL", "erllama-tp/tinyllama-1.1b-chat.gguf"),
    Ranks = rank_pids(Config),
    [{P0, R0} | _] = Ranks,
    MOpts = #{n_gpu_layers => 99},
    CtxOpts = #{n_ctx => 512, n_batch => 512, flash_attn => enabled},
    Opts = {maps:merge(MOpts, Split), CtxOpts},
    Before = [peer:call(P, erllama_tp_rank, info, [R]) || {P, R} <- Ranks],
    Loaded = parallel([
        fun() -> peer:call(P, erllama_tp_rank, load, [R, home(P, Model), Opts], 600000) end
     || {P, R} <- Ranks
    ]),
    ct:pal("split load: ~p", [Loaded]),
    [ok = L || L <- Loaded],
    After = [peer:call(P, erllama_tp_rank, info, [R]) || {P, R} <- Ranks],
    ok = peer:call(P0, erllama_tp_rank, ref_load, [R0, home(P0, Model), {MOpts, CtxOpts}], 600000),
    %% "The quick brown fox jumps over the lazy dog" in the Llama vocabulary.
    Prompt = [1, 450, 4996, 17354, 1701, 29916, 432, 17204, 975, 278, 17366, 11203],
    Step1 = compare_step(Ranks, P0, R0, Prompt, "prompt"),
    Next = argmax(Step1),
    _ = compare_step(Ranks, P0, R0, [Next], "next token from the cache"),
    Stats = [element(2, peer:call(P, erllama_tp_rank, stats, [R])) || {P, R} <- Ranks],
    [
        ct:pal(
            "rank ~b: ~b reductions; placeholders reserve ~.1f MiB (the other ranks' share); free GPU memory ~p -> ~p MiB",
            [
                I,
                maps:get(reductions, St),
                maps:get(remote_bytes, St) / 1048576,
                free_mib(B),
                free_mib(A)
            ]
        )
     || {I, St, {B, A}} <- lists:zip3(
            lists:seq(0, length(Ranks) - 1), Stats, lists:zip(Before, After)
        )
    ],
    %% Every rank made the same reductions, of the same sizes, in order.
    [Sizes0 | Others] = [maps:get(sizes, St) || St <- Stats],
    [?assertEqual(Sizes0, Sz) || Sz <- Others],
    ?assert(length(Sizes0) > 0).

compare_step(Ranks, P0, R0, Tokens, What) ->
    Got = parallel([
        fun() -> peer:call(P, erllama_tp_rank, decode, [R, Tokens], 120000) end
     || {P, R} <- Ranks
    ]),
    {ok, Ref} = peer:call(P0, erllama_tp_rank, ref_decode, [R0, Tokens], 120000),
    RefL = f32s(Ref),
    lists:foreach(
        fun({I, {ok, Bin}}) ->
            L = f32s(Bin),
            MaxDiff = lists:max([abs(A - B) || {A, B} <- lists:zip(L, RefL)]),
            Scale = lists:max([abs(B) || B <- RefL]),
            ct:pal(
                "~s, rank ~b: max |logit - one GPU| = ~.4g (largest |logit| ~.4g); top 5 ~w vs ~w",
                [
                    What, I, MaxDiff, Scale, top(L, 5), top(RefL, 5)
                ]
            ),
            ?assertEqual(argmax(RefL), argmax(L)),
            ?assert(MaxDiff =< 2.0e-2 * Scale)
        end,
        lists:enumerate(0, Got)
    ),
    RefL.

f32s(Bin) -> [V || <<V:32/float-native>> <= Bin].

argmax(L) ->
    {_, I} = lists:max(lists:zip(L, lists:seq(0, length(L) - 1))),
    I.

top(L, N) ->
    [
        I
     || {_, I} <- lists:sublist(
            lists:reverse(lists:sort(lists:zip(L, lists:seq(0, length(L) - 1)))), N
        )
    ].

free_mib(#{devices := [#{free_b := F} | _]}) -> F div 1048576;
free_mib(_) -> undefined.

%% `Path' relative to the home directory of the peer's host.
home(P, Path) ->
    case Path of
        "/" ++ _ -> Path;
        _ -> filename:join(peer:call(P, os, getenv, ["HOME"]), Path)
    end.

%% What NCCL chose for the data path, from each rank's INIT/NET log.
transport_in_use(Config) ->
    Lines = lists:append([nccl_lines(P) || {_, P, _} <- ?config(peers, Config)]),
    ct:pal("NCCL transport and network lines:~n~ts", [lists:join("\n", Lines)]),
    ?assertNotEqual([], Lines).

%% -----------------------------------------------------------------------------

reduce_case(Config, N, Tol, Path) ->
    Ranks = rank_pids(Config),
    Inputs = [values(Rank, N) || {Rank, _} <- lists:enumerate(0, Ranks)],
    ct:pal("~b values per rank, ~p path; tolerance |got - exact| =< ~g x sum of |inputs| + 1e-6", [
        N, Path, Tol
    ]),
    Results = parallel([
        fun() -> peer:call(P, erllama_tp_rank, reduce, [R, In], 120000) end
     || {{P, R}, In} <- lists:zip(Ranks, Inputs)
    ]),
    {Exact, Scale} = exact_sum(Inputs),
    lists:foreach(
        fun({ok, Got}) ->
            Worst = worst(Got, Exact, Scale, Tol),
            ct:pal("worst error over the bound: ~p", [Worst]),
            ?assert(Worst =< 1.0)
        end,
        Results
    ).

%% Deterministic values in [-0.5, 0.5), different per rank, as F32.
values(Rank, N) ->
    <<<<(((I * 7 + Rank * 13) rem 101) / 101 - 0.5):32/float-native>> || I <- lists:seq(0, N - 1)>>.

exact_sum(Inputs) ->
    Lists = [[V || <<V:32/float-native>> <= B] || B <- Inputs],
    Zipped = transpose(Lists),
    {[lists:sum(Col) || Col <- Zipped], [lists:sum([abs(V) || V <- Col]) || Col <- Zipped]}.

transpose([[] | _]) -> [];
transpose(Ls) -> [[hd(L) || L <- Ls] | transpose([tl(L) || L <- Ls])].

%% The largest |error| / bound over the buffer (=< 1.0 passes).
worst(Got, Exact, Scale, Tol) ->
    Gs = [V || <<V:32/float-native>> <= Got],
    lists:max([abs(G - E) / (Tol * S + 1.0e-6) || {G, E, S} <- lists:zip3(Gs, Exact, Scale)]).

start_peer(Rank, Host, Cookie) ->
    Extra = string:lexemes(os:getenv("ERLLAMA_TP_NCCL_ENV", ""), " "),
    Env = [
        "LD_LIBRARY_PATH=" ?T "/nccl/lib:" ?T "/openssl/lib",
        "NCCL_DEBUG=INFO",
        "NCCL_DEBUG_SUBSYS=INIT,NET",
        "NCCL_DEBUG_FILE=/tmp/erllama_tp.%h.%p.nccl"
        | Extra
    ],
    {ok, Peer, Node} = peer:start(#{
        name => list_to_atom("erllama_tp" ++ integer_to_list(Rank)),
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
    ct:pal("rank ~b on ~s: ~p", [Rank, Host, Node]),
    {Rank, Peer, Node}.

%% A coordinator on rank 0's node, then every rank joined at once.
form_group(Config, Timeout) ->
    Peers = ?config(peers, Config),
    [{_, P0, _} | _] = Peers,
    World = ?config(world, Config),
    {ok, Coord} = peer:call(P0, erllama_tp_coordinator, start, [World, Timeout]),
    Starts = parallel([
        fun() ->
            peer:call(
                P,
                erllama_tp_rank,
                start,
                [#{rank => R, world => World, join => {call, Coord}, timeout => Timeout}],
                Timeout + 30000
            )
        end
     || {R, P, _} <- Peers
    ]),
    case [S || S <- Starts, element(1, S) =/= ok] of
        [] -> ok;
        Bad -> ct:fail({group_failed, Bad})
    end,
    [{P, Pid} || {{_, P, _}, {ok, Pid}} <- lists:zip(Peers, Starts)].

rank_pids(Config) ->
    ?config(ranks, Config).

stop_ranks(_Config, Ranks) ->
    [catch peer:call(P, erllama_tp_rank, stop, [R], 30000) || {P, R} <- Ranks],
    ok.

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
