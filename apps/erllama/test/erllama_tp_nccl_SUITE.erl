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
%% ERLLAMA_TP_NCCL_ENV adds environment for every rank (space-separated
%% K=V, e.g. "NCCL_IB_HCA=rocep1s0f0"), ERLLAMA_TP_NCCL_ENV_<Rank> for
%% one rank (e.g. its NCCL_SOCKET_IFNAME). Plain TCP:
%%
%%   ERLLAMA_TP_NCCL_ENV="NCCL_NET=Socket NCCL_SOCKET_IFNAME==enp1s0f0np0" \
%%   ERLLAMA_TP_EXPECT_NET=Socket ERLLAMA_TP_EXPECT_IF=enp1s0f0np0 ...
%%
%% ERLLAMA_TP_EXPECT_NET (Socket or IB) and ERLLAMA_TP_EXPECT_IF (the
%% interfaces, comma-separated, the same on every host) make
%% transport_in_use and data_path_bytes check that the reductions went
%% that way; without them those cases only log what NCCL chose.
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
    serve_through_erllama/1,
    transport_in_use/1,
    data_path_bytes/1
]).

all() ->
    [
        matvec_fixture,
        decode_sized_fp32,
        prefill_sized_bf16,
        missing_rank_fails_within_the_deadline,
        open_and_close_repeat,
        model_matches_one_gpu,
        uneven_split_matches_one_gpu,
        serve_through_erllama,
        transport_in_use,
        data_path_bytes
    ].

init_per_suite(Config) ->
    case erllama_tp_hosts:hosts() of
        false ->
            {skip, "set ERLLAMA_TP_HOSTS to the CUDA hosts (see scripts/tp_hosts.sh)"};
        Hosts ->
            Cookie = "erllama_tp_" ++ integer_to_list(erlang:unique_integer([positive])),
            Peers = [
                begin
                    Name = list_to_atom("erllama_tp" ++ integer_to_list(R)),
                    {P, N} = erllama_tp_hosts:start_peer(Name, H, Cookie, R),
                    {R, P, N}
                end
             || {R, H} <- lists:enumerate(0, Hosts)
            ],
            [{_, P0, _} | _] = Peers,
            ok = erllama_tp_hosts:load_module(P0, erllama_tp_coordinator),
            %% The node serving the split model: erllama, no GPU work.
            {S, _} = erllama_tp_hosts:start_peer(erllama_tps, hd(Hosts), Cookie),
            ok = erllama_tp_hosts:load_module(S, erllama_tp_stream),
            {ok, Caps} = peer:call(P0, erllama_tp, caps, []),
            ct:pal("rank 0 caps: ~p", [Caps]),
            case Caps of
                #{init_rank := true} ->
                    [{peers, Peers}, {serving, S}, {world, length(Hosts)} | Config];
                _ ->
                    {skip, "the hosts' build has no per-process communicator (CUDA with NCCL)"}
            end
    end.

end_per_suite(Config) ->
    [peer:stop(P) || {_, P, _} <- proplists:get_value(peers, Config, [])],
    _ = [peer:stop(S) || S <- [proplists:get_value(serving, Config)], S =/= undefined],
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
    [{ok, Part0, Sum0}, {ok, Part1, Sum1}] = erllama_tp_hosts:parallel([
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
            Results = erllama_tp_hosts:parallel([
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
    Model = erllama_tp_hosts:model(),
    Ranks = rank_pids(Config),
    [{P0, R0} | _] = Ranks,
    MOpts = #{n_gpu_layers => 99},
    CtxOpts = #{n_ctx => 512, n_batch => 512, flash_attn => enabled},
    Opts = {maps:merge(MOpts, Split), CtxOpts},
    Before = [peer:call(P, erllama_tp_rank, info, [R]) || {P, R} <- Ranks],
    Loaded = erllama_tp_hosts:parallel([
        fun() ->
            peer:call(P, erllama_tp_rank, load, [R, erllama_tp_hosts:home(P, Model), Opts], 600000)
        end
     || {P, R} <- Ranks
    ]),
    ct:pal("split load: ~p", [Loaded]),
    [ok = L || L <- Loaded],
    After = [peer:call(P, erllama_tp_rank, info, [R]) || {P, R} <- Ranks],
    ok = peer:call(
        P0,
        erllama_tp_rank,
        ref_load,
        [R0, erllama_tp_hosts:home(P0, Model), {MOpts, CtxOpts}],
        600000
    ),
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
    Got = erllama_tp_hosts:parallel([
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

%% The split model served through erllama, as any loaded model, from a
%% node that is no rank (erllama_model_tp; rank 0 samples): a greedy
%% completion against the same model on one GPU, a timed stream, a
%% cancel and the next request; the other ranks follow rank 0's
%% batches.
serve_through_erllama(Config) ->
    Model = erllama_tp_hosts:model(),
    S = ?config(serving, Config),
    [{P0, R0} | Followers] = rank_pids(Config),
    MOpts = #{n_gpu_layers => 99},
    COpts = #{n_ctx => 1024, n_batch => 512, flash_attn => enabled},
    Opts = {MOpts, COpts},
    Loaded = erllama_tp_hosts:parallel([
        fun() ->
            peer:call(P, erllama_tp_rank, load, [R, erllama_tp_hosts:home(P, Model), Opts], 600000)
        end
     || {P, R} <- Followers
    ]),
    [ok = L || L <- Loaded],
    Descs = [
        {R,
            peer:call(P, erllama_tp_rank, describe, [
                erllama_tp_hosts:home(P, Model), Opts, ?config(world, Config)
            ])}
     || {P, R} <- Followers
    ],
    ok = erllama_tp_hosts:load_module(P0, erllama_tp_stream),
    NoCache = #{
        min_tokens => infinity, cold_min_tokens => infinity, continued_interval => infinity
    },
    {ok, Id} = peer:call(
        S,
        erllama,
        load_model,
        [
            <<"tp_served">>,
            #{
                backend => erllama_model_tp,
                model_path => erllama_tp_hosts:home(P0, Model),
                model_opts => MOpts,
                context_opts => COpts,
                policy => NoCache,
                tp_worker => R0,
                tp_followers => Descs
            }
        ],
        600000
    ),
    {ok, Ref} = peer:call(
        P0,
        erllama,
        load_model,
        [
            <<"tp_one_gpu">>,
            #{
                backend => erllama_model_llama,
                model_path => erllama_tp_hosts:home(P0, Model),
                model_opts => #{n_gpu_layers => 99},
                context_opts => COpts,
                policy => NoCache
            }
        ],
        600000
    ),
    try
        %% Greedy, fixed prompt: the same tokens as on one GPU.
        %% The split model is on the serving node, the one-GPU copy on rank 0's.
        On = fun
            (Mod) when Mod =:= Id -> S;
            (_) -> P0
        end,
        Complete = fun(Mod) ->
            {ok, #{generated := G}} = peer:call(
                On(Mod),
                erllama,
                complete,
                [Mod, <<"The quick brown fox">>, #{response_tokens => 32, temperature => 0.0}],
                120000
            ),
            G
        end,
        Split = Complete(Id),
        One = Complete(Ref),
        Same = length(lists:takewhile(fun({A, B}) -> A =:= B end, lists:zip(Split, One))),
        ct:pal("greedy, 32 tokens: split ~w~none GPU ~w~nidentical for the first ~b", [
            Split, One, Same
        ]),
        ?assert(Same >= 8),
        %% A stream with a longer prompt, timed, on both.
        Long = iolist_to_binary(
            lists:duplicate(20, <<"The quick brown fox jumps over the lazy dog. ">>)
        ),
        {ok, PromptToks} = peer:call(S, erllama, tokenize, [Id, Long]),
        Timed = fun(Mod) ->
            peer:call(On(Mod), erllama_tp_stream, run, [Mod, Long, 128, infinity], 300000)
        end,
        _ = Timed(Ref),
        [SplitRun, OneRun] = [Timed(Mod) || Mod <- [Id, Ref]],
        [
            ct:pal(
                "~s: ~b prompt tokens, first token ~.1f ms (~.1f prompt tokens/s), ~b tokens, decode ~.1f tokens/s, median gap ~.2f ms, ended ~p",
                [
                    Name,
                    length(PromptToks),
                    maps:get(ttft_ms, Run),
                    length(PromptToks) / maps:get(ttft_ms, Run) * 1000,
                    maps:get(tokens, Run),
                    decode_rate(Run),
                    median(maps:get(gaps_ms, Run)),
                    maps:get(ended, Run)
                ]
            )
         || {Name, Run} <- [{"split over the ranks", SplitRun}, {"one GPU", OneRun}]
        ],
        ?assertEqual(128, maps:get(tokens, SplitRun)),
        %% Cancel after 10 tokens; the next request is served.
        Cancelled = peer:call(S, erllama_tp_stream, run, [Id, Long, 128, 10], 300000),
        ?assertMatch(#{ended := cancelled, tokens := 10}, Cancelled),
        ?assertEqual(Split, Complete(Id)),
        %% The followers made every reduction rank 0 made, in order.
        Stats = [
            element(2, peer:call(P, erllama_tp_rank, stats, [R]))
         || {P, R} <- rank_pids(Config)
        ],
        [Sizes0 | Others] = [maps:get(sizes, St) || St <- Stats],
        ct:pal("reductions per rank: ~w", [[maps:get(reductions, St) || St <- Stats]]),
        [?assertEqual(Sizes0, Sz) || Sz <- Others],
        %% A decode-sized reduction (2048 values, FP32), timed in each
        %% rank's process, including the copy to and from the GPU.
        Latency = erllama_tp_hosts:parallel([
            fun() -> peer:call(P, erllama_tp_rank, time_reduce, [R, 2048, 200], 120000) end
         || {P, R} <- rank_pids(Config)
        ]),
        ct:pal("2048-value reduction, mean over 200 in the rank process: ~w us", [Latency])
    after
        peer:call(P0, erllama, unload, [Ref]),
        peer:call(S, erllama, unload, [Id])
    end.

decode_rate(#{gaps_ms := []}) -> 0.0;
decode_rate(#{gaps_ms := Gaps}) -> length(Gaps) / lists:sum(Gaps) * 1000.

median([]) -> 0.0;
median(L) -> lists:nth((length(L) + 1) div 2, lists:sort(L)).

%% What NCCL chose for the data path, from each rank's INIT/NET log.
transport_in_use(Config) ->
    PerRank = [erllama_tp_hosts:nccl_lines(P) || {_, P, _} <- ?config(peers, Config)],
    Lines = lists:append(PerRank),
    ct:pal("NCCL transport and network lines:~n~ts", [lists:join("\n", Lines)]),
    ?assertNotEqual([], Lines),
    case os:getenv("ERLLAMA_TP_EXPECT_NET") of
        false -> ok;
        Net -> [expect_net(Net, L) || L <- PerRank]
    end,
    ok.

%% Every channel the reductions use goes through `Net', and NCCL's
%% devices for it are the expected interfaces, in order.
expect_net(Net, Lines) ->
    Channels = [L || L <- Lines, string:find(L, " via ") =/= nomatch],
    ?assertNotEqual([], Channels),
    [?assertNotEqual(nomatch, string:find(L, "via NET/" ++ Net ++ "/"), L) || L <- Channels],
    case expected_ifs() of
        [] ->
            ok;
        Ifs ->
            Using = [L || L <- Lines, string:find(L, "NET/" ++ Net ++ " : Using [") =/= nomatch],
            ?assertNotEqual([], Using),
            [
                ?assertEqual(Ifs, used_ifs(L), L)
             || L <- Using
            ]
    end.

%% `NET/Socket : Using [0]enp1s0f0np0:10.100.168.2<0> [1]...' -> names.
used_ifs(Line) ->
    {match, Ms} = re:run(Line, "\\[[0-9]+\\]([^:\\s/]+)", [global, {capture, all_but_first, list}]),
    [N || [N] <- Ms].

%% The reductions' bytes, from each host's interface counters around a
%% burst of 17 prefill-sized reductions (BF16, about 4 MB per rank
%% each): with ERLLAMA_TP_EXPECT_IF, the expected interfaces must carry
%% at least 80% of what every interface sent, control traffic included.
%% RDMA traffic does not show in these counters.
data_path_bytes(Config) ->
    Ranks = rank_pids(Config),
    Peers = [P || {P, _} <- Ranks],
    Before = [if_bytes(P) || P <- Peers],
    %% The values are made on each rank's node: only the reductions
    %% cross the network.
    Us = erllama_tp_hosts:parallel([
        fun() -> peer:call(P, erllama_tp_rank, time_reduce, [R, 512 * 4096, 16], 300000) end
     || {P, R} <- Ranks
    ]),
    ct:pal("16 reductions of 512 x 4096 values (BF16): ~w us each", [Us]),
    After = [if_bytes(P) || P <- Peers],
    Deltas = [
        [{If, Tx1 - Tx0, Speed} || {If, Tx1, Speed} <- A, {If0, Tx0, _} <- B, If0 =:= If]
     || {B, A} <- lists:zip(Before, After)
    ],
    [
        ct:pal("rank ~b sent: ~ts", [
            R,
            lists:join(", ", [
                io_lib:format("~s ~.1f MB (~s Mb/s)", [If, D / 1.0e6, Sp])
             || {If, D, Sp} <- Ds, D > 0
            ])
        ])
     || {R, Ds} <- lists:enumerate(0, Deltas)
    ],
    case expected_ifs() of
        [] ->
            ok;
        Ifs ->
            [
                begin
                    Total = lists:sum([D || {_, D, _} <- Ds]),
                    Mine = lists:sum([D || {If, D, _} <- Ds, lists:member(If, Ifs)]),
                    ?assert(Mine >= 0.8 * Total, {Mine, Total})
                end
             || Ds <- Deltas
            ]
    end,
    ok.

%% Every physical or bonded interface's sent bytes and speed on the
%% peer's host (loopback, docker and veth left out).
if_bytes(P) ->
    {ok, All} = peer:call(P, file, list_dir, ["/sys/class/net"]),
    Ifs = [
        If
     || If <- All,
        not lists:any(fun(Pre) -> lists:prefix(Pre, If) end, ["lo", "docker", "veth", "br-"])
    ],
    [{If, sys_int(P, If, "statistics/tx_bytes"), sys_str(P, If, "speed")} || If <- Ifs].

sys_int(P, If, What) ->
    list_to_integer(sys_str(P, If, What)).

sys_str(P, If, What) ->
    case peer:call(P, file, read_file, ["/sys/class/net/" ++ If ++ "/" ++ What]) of
        {ok, B} -> string:trim(binary_to_list(B));
        _ -> "0"
    end.

expected_ifs() ->
    string:lexemes(os:getenv("ERLLAMA_TP_EXPECT_IF", ""), ",").

reduce_case(Config, N, Tol, Path) ->
    Ranks = rank_pids(Config),
    Inputs = [values(Rank, N) || {Rank, _} <- lists:enumerate(0, Ranks)],
    ct:pal("~b values per rank, ~p path; tolerance |got - exact| =< ~g x sum of |inputs| + 1e-6", [
        N, Path, Tol
    ]),
    Results = erllama_tp_hosts:parallel([
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

%% A coordinator on rank 0's node, then every rank joined at once.
form_group(Config, Timeout) ->
    [{_, P0, _} | _] = Peers = ?config(peers, Config),
    {ok, Coord} = peer:call(P0, erllama_tp_coordinator, start, [?config(world, Config), Timeout]),
    erllama_tp_hosts:form_group([P || {_, P, _} <- Peers], Coord, 0, Timeout).

rank_pids(Config) ->
    ?config(ranks, Config).

stop_ranks(_Config, Ranks) ->
    [catch peer:call(P, erllama_tp_rank, stop, [R], 30000) || {P, R} <- Ranks],
    ok.
