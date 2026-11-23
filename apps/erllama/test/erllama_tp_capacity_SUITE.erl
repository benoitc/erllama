%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Opt-in, on CUDA hosts (ERLLAMA_TP_HOSTS, see erllama_tp_nccl_SUITE),
%% with ERLLAMA_TP_BIG_MODEL naming a GGUF (its first part when split,
%% relative to the hosts' home) that does not fit one GPU but fits the
%% group: what one host can hold, and the same model split over the
%% hosts and served.
%%
%% The hosts' GPUs must be free for the run: it needs their memory.
-module(erllama_tp_capacity_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([one_host_cannot_hold_it/1, split_over_the_hosts/1]).

-define(NO_CACHE, #{
    min_tokens => infinity, cold_min_tokens => infinity, continued_interval => infinity
}).
-define(SAMPLE_MS, 250).

all() ->
    [one_host_cannot_hold_it, split_over_the_hosts].

init_per_suite(Config) ->
    case {erllama_tp_hosts:hosts(), os:getenv("ERLLAMA_TP_BIG_MODEL")} of
        {false, _} -> {skip, "set ERLLAMA_TP_HOSTS to the CUDA hosts"};
        {_, false} -> {skip, "set ERLLAMA_TP_BIG_MODEL to a GGUF larger than one GPU"};
        {Hosts, Model} -> [{hosts, Hosts}, {model, Model} | Config]
    end.

end_per_suite(_Config) ->
    ok.

%% What one host can hold of the model: llama.cpp's auto-fit (an 8 GiB
%% margin) puts as many layers as fit on the GPU and the rest on the
%% CPU. On a GB10 both share the host's memory, so a model larger than
%% that memory ends with the kernel killing the node: the case counts
%% that, or a refused load, or layers left off the GPU, as "cannot hold
%% it". It can take the host to its memory limit, so it runs only with
%% ERLLAMA_TP_CAPACITY_ONE_HOST=1.
one_host_cannot_hold_it(Config) ->
    case os:getenv("ERLLAMA_TP_CAPACITY_ONE_HOST") of
        "1" -> one_host(Config);
        _ -> {skip, "set ERLLAMA_TP_CAPACITY_ONE_HOST=1: the load can run the host out of memory"}
    end.

one_host(Config) ->
    [Host0 | _] = ?config(hosts, Config),
    {P, _} = erllama_tp_hosts:start_peer(name("erllama_cap"), Host0, cookie()),
    try
        native_log(P),
        Path = erllama_tp_hosts:home(P, ?config(model, Config)),
        Size = model_bytes(P, Path),
        Before = mem(P),
        Sampler = sampler([P]),
        {Ms, Result} = timer:tc(
            fun() ->
                try
                    peer:call(
                        P,
                        erllama_nif,
                        load_model,
                        [
                            Path,
                            #{
                                fit => #{margins_mib => [8192]},
                                fit_context => #{n_ctx => 2048, flash_attn => enabled}
                            }
                        ],
                        1800000
                    )
                catch
                    exit:Why -> {node_lost, Why}
                end
            end,
            millisecond
        ),
        [Low] = stop_sampler(Sampler),
        ct:pal(
            "~s on one GB10: model ~.1f GB; available before ~.1f GB, lowest during the load ~.1f GB; after ~b ms: ~p",
            [Host0, gb(Size), gb(avail(Before)), gb(Low), Ms, short(Result)]
        ),
        case Result of
            {node_lost, _} ->
                ok;
            {error, _} ->
                ct:pal("llama.cpp placement:~n~ts", [placement(P)]);
            {ok, M, #{n_gpu_layers := N} = Fit} ->
                ct:pal("llama.cpp placement:~n~ts", [placement(P)]),
                _ = peer:call(P, erllama_nif, free_model, [M]),
                %% Llama 3.3 70B has 80 layers.
                ?assert(N < 80, Fit)
        end
    after
        catch peer:stop(P)
    end.

%% The same file split over the group and served from its own node:
%% each rank's peak memory, where the shards went, any CPU offload, and
%% a streamed completion.
split_over_the_hosts(Config) ->
    Hosts = ?config(hosts, Config),
    os:putenv("ERLLAMA_TP_MODEL", ?config(model, Config)),
    erllama_tp_hosts:with_serving(Hosts, fun(S) ->
        {G, Before} = serve_sampled(Hosts, S),
        #{rank_peers := Peers, rank_pids := RankPids} = G,
        Prompt = erllama_tp_hosts:long_prompt(20),
        Run = peer:call(
            S, erllama_tp_stream, run, [maps:get(id, G), Prompt, 64, infinity], 1800000
        ),
        Low = stop_sampler(maps:get(sampler, G)),
        ct:pal("streamed: ~p", [Run]),
        Stats = [
            element(2, peer:call(P, erllama_tp_rank, stats, [R]))
         || {P, R} <- lists:zip(Peers, RankPids)
        ],
        [
            ct:pal(
                "rank ~b: available before ~.1f GB, lowest while loading and serving ~.1f GB "
                "(peak use ~.1f GB); the other rank's share of every split tensor, reserved "
                "here as address space only: ~.1f GB~nllama.cpp placement:~n~ts",
                [
                    R,
                    gb(avail(B)),
                    gb(L),
                    gb(avail(B) - L),
                    maps:get(remote_bytes, St) / 1.0e9,
                    placement(P)
                ]
            )
         || {R, {P, B}, {L, St}} <- lists:zip3(
                lists:seq(0, length(Peers) - 1),
                lists:zip(Peers, Before),
                lists:zip(Low, Stats)
            )
        ],
        ?assertMatch(#{tokens := 64}, Run),
        ok = erllama_tp_hosts:stop_group(G)
    end).

%% The group formed, then the model loaded and served while each rank's
%% memory is sampled: the lowest available memory per rank, and the
%% baseline.
serve_sampled(Hosts, S) ->
    G0 = erllama_tp_hosts:form(Hosts, S, 1),
    Peers = maps:get(rank_peers, G0),
    _ = [native_log(P) || P <- Peers],
    Before = [mem(P) || P <- Peers],
    Sampler = sampler(Peers),
    G = erllama_tp_hosts:load(G0, ?NO_CACHE, #{tp_call_timeout => 1800000}, 1800000),
    {G#{sampler => Sampler}, Before}.

native_log(P) ->
    ok = peer:call(P, application, set_env, [erllama, native_log_level, info]),
    ok = peer:call(P, supervisor, terminate_child, [erllama_sup, erllama_log]),
    {ok, _} = peer:call(P, supervisor, restart_child, [erllama_sup, erllama_log]),
    _ = peer:call(P, logger, set_primary_config, [level, info]),
    _ = peer:call(P, logger, add_handler, [
        tp_capacity,
        logger_std_h,
        #{
            level => info,
            config => #{file => "/tmp/erllama_tp_capacity.log"},
            filter_default => stop,
            filters => [{native, {fun logger_filters:domain/2, {log, sub, [erllama, native]}}}]
        }
    ]),
    ok.

placement(P) ->
    case peer:call(P, file, read_file, ["/tmp/erllama_tp_capacity.log"]) of
        {ok, Bin} ->
            Lines = binary:split(Bin, <<"\n">>, [global]),
            Keep = [
                L
             || L <- Lines,
                lists:any(
                    fun(K) -> binary:match(L, K) =/= nomatch end,
                    [
                        <<"buffer size">>,
                        <<"offloaded">>,
                        <<"offloading">>,
                        <<"failed to allocate">>,
                        <<"out of memory">>,
                        <<"file size">>,
                        <<"split_mode">>
                    ]
                )
            ],
            lists:join("\n", lists:sublist(Keep, 40));
        _ ->
            "(no native log)"
    end.

%% MemAvailable (bytes): the GB10's GPU shares the host's memory.
mem(P) ->
    {ok, Bin} = peer:call(P, file, read_file, ["/proc/meminfo"]),
    {match, [Kb]} = re:run(Bin, "MemAvailable:\\s+([0-9]+) kB", [{capture, all_but_first, binary}]),
    #{avail => binary_to_integer(Kb) * 1024}.

avail(#{avail := A}) -> A.

sampler(Peers) ->
    Self = self(),
    spawn_link(fun() -> sample(Self, Peers, [avail(mem(P)) || P <- Peers]) end).

sample(Owner, Peers, Low) ->
    receive
        {stop, Ref} -> Owner ! {Ref, Low}
    after ?SAMPLE_MS ->
        Now = [
            try
                avail(mem(P))
            catch
                _:_ -> L
            end
         || {P, L} <- lists:zip(Peers, Low)
        ],
        sample(Owner, Peers, [min(A, B) || {A, B} <- lists:zip(Low, Now)])
    end.

stop_sampler(Pid) ->
    Ref = make_ref(),
    Pid ! {stop, Ref},
    receive
        {Ref, Low} -> Low
    after 10000 -> ct:fail(sampler_timeout)
    end.

model_bytes(P, Path) ->
    Dir = filename:dirname(Path),
    {ok, Files} = peer:call(P, file, list_dir, [Dir]),
    lists:sum([
        peer:call(P, filelib, file_size, [filename:join(Dir, F)])
     || F <- Files, filename:extension(F) =:= ".gguf"
    ]).

gb(B) -> B / 1.0e9.

short({node_lost, Why}) -> {node_lost, Why};
short({error, _} = E) -> E;
short({ok, _}) -> loaded;
short({ok, _, Fit}) -> {loaded, Fit};
short(Other) -> Other.

cookie() ->
    "erllama_tp_" ++ integer_to_list(erlang:unique_integer([positive])).

name(Base) ->
    list_to_atom(Base ++ "_" ++ integer_to_list(erlang:unique_integer([positive]))).
