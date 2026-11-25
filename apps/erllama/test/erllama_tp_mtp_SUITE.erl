%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Opt-in, on CUDA hosts (ERLLAMA_TP_HOSTS) with their GPUs free: MTP
%% speculative decoding of GLM-5.3-Flash split over the group. Rank 0
%% drafts two tokens a round with the model's MTP head on its own GPU;
%% the round is one 3-row batch on every rank, and a rejected draft is
%% taken back with the recurrent-state snapshots (n_rs_seq 2 on every
%% rank).
%%
%%   ERLLAMA_TP_GLM      the GGUF to serve (the first part of a split one)
%%   ERLLAMA_TP_GLM_MTP  the companion MTP GGUF, on the first host
%%   ERLLAMA_TP_MTP_SPLIT  optional shares of the model per rank (e.g.
%%                         "0.46,0.54"), to leave rank 0 room for the
%%                         MTP head
%%
%% Paths are relative to the hosts' home.
-module(erllama_tp_mtp_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1]).
-export([mtp_decode/1, divergence_margin/1, rollback_state_matches/1]).

-define(NO_CACHE, #{
    min_tokens => infinity, cold_min_tokens => infinity, continued_interval => infinity
}).
-define(MOPTS, #{n_gpu_layers => 999, load_mode => direct_io}).
-define(COPTS, #{
    n_ctx => 4096,
    n_batch => 1024,
    flash_attn => enabled,
    n_rs_seq => 2,
    n_seq_max => 2,
    kv_unified => true
}).
-define(MTP, #{speculative => #{type => mtp, n_max => 2}}).
-define(TOKENS, 250).

all() ->
    [mtp_decode, divergence_margin, rollback_state_matches].

init_per_suite(Config) ->
    case {erllama_tp_hosts:hosts(), os:getenv("ERLLAMA_TP_GLM"), os:getenv("ERLLAMA_TP_GLM_MTP")} of
        {false, _, _} -> {skip, "set ERLLAMA_TP_HOSTS to the CUDA hosts"};
        {_, false, _} -> {skip, "set ERLLAMA_TP_GLM to a GLM-5.3-Flash GGUF"};
        {_, _, false} -> {skip, "set ERLLAMA_TP_GLM_MTP to its MTP GGUF"};
        {Hosts, Model, Mtp} -> [{hosts, Hosts}, {model, Model}, {mtp, Mtp} | Config]
    end.

end_per_suite(_Config) ->
    ok.

%% TensorFold's four fixed prompts (docs/dec.py of its recipe) through
%% the chat template, thinking off, greedy, 250 tokens, one stream:
%% each one without drafting, then with MTP, in the same context. The
%% decode rate is taken after the first token, as for the ordinary
%% decoding figures. Reports acceptance, draft, verify and rollback
%% time a round, and whether the drafter's context ended where the
%% target's does after every round (it resumes from the accepted
%% position). Then one erllama:complete with and without MTP.
mtp_decode(Config) ->
    serve(Config, fun(S, G, R0) ->
        Prompts = [chat(S, G, T) || T <- dec_prompts()],
        _ = run(S, G, hd(Prompts), 16, #{}),
        _ = run(S, G, hd(Prompts), 16, ?MTP),
        Rows = [prompt_case(S, G, R0, I, P) || {I, P} <- lists:enumerate(Prompts)],
        Plain = [maps:get(plain, R) || R <- Rows],
        Mtp = [maps:get(mtp, R) || R <- Rows],
        ct:pal(
            "over the four prompts: ordinary ~.1f tokens/s, MTP ~.1f tokens/s (mean of the per-prompt rates)",
            [mean(Plain), mean(Mtp)]
        ),
        [?assertEqual(0, maps:get(misaligned, R)) || R <- Rows],
        complete_case(S, G),
        ok
    end).

prompt_case(S, G, R0, I, Prompt) ->
    A = run(S, G, Prompt, ?TOKENS, #{}),
    K0 = mtp_stats(S, R0),
    B = run(S, G, Prompt, ?TOKENS, ?MTP),
    K = diff(mtp_stats(S, R0), K0),
    #{rounds := Rounds, drafted := Drafted, accepted := Acc} = K,
    Same = binary:longest_common_prefix([maps:get(text, A), maps:get(text, B)]),
    PerRound = fun(Key) -> maps:get(Key, K) / max(1, Rounds) / 1000 end,
    ct:pal(
        "prompt ~b: ordinary ~.1f tokens/s (first token ~.1f ms); MTP ~.1f tokens/s (first token ~.1f ms), ~b tokens, ~p~n"
        "  rounds ~b, drafted ~b, accepted ~b (~.3f a draft, ~.2f tokens a round); after a rejection ~b rounds accepted ~b (~.3f a draft)~n"
        "  a round: draft ~.2f ms, verify ~.2f ms (of which the drafter's catch-up ~.2f ms), rollback ~.2f ms; misaligned ~b, catch-up failures ~b~n"
        "  request stats ~p; same text for ~b of ~b bytes",
        [
            I,
            rate(A),
            maps:get(ttft_ms, A),
            rate(B),
            maps:get(ttft_ms, B),
            maps:get(tokens, B),
            maps:get(ended, B),
            Rounds,
            Drafted,
            Acc,
            Acc / max(1, Drafted),
            (Rounds + Acc) / max(1, Rounds),
            maps:get(after_reject_rounds, K),
            maps:get(after_reject_accepted, K),
            maps:get(after_reject_accepted, K) / max(1, 2 * maps:get(after_reject_rounds, K)),
            PerRound(draft_us),
            PerRound(verify_us),
            maps:get(process_us, K) / max(1, maps:get(process_calls, K)) / 1000,
            PerRound(rollback_us),
            maps:get(misaligned, K),
            maps:get(process_failures, K),
            maps:get(speculative, maps:get(stats, B, #{}), none),
            Same,
            byte_size(maps:get(text, A))
        ]
    ),
    ?assert(Rounds > 0),
    #{plain => rate(A), mtp => rate(B), misaligned => maps:get(misaligned, K)}.

%% erllama:complete, the plain-text path, with and without MTP.
complete_case(S, #{id := Id}) ->
    Prompt = <<"The history of ocean navigation begins with">>,
    Opts = #{response_tokens => 64, temperature => 0.0},
    {ok, #{generated := A}} = peer:call(S, erllama, complete, [Id, Prompt, Opts], 600000),
    {ok, #{generated := B} = St} = peer:call(
        S, erllama, complete, [Id, Prompt, maps:merge(Opts, ?MTP)], 600000
    ),
    Common = length(lists:takewhile(fun({X, Y}) -> X =:= Y end, lists:zip(A, B, trim))),
    ct:pal("erllama:complete, 64 tokens: ~b of ~b tokens the same with MTP; ~p", [
        Common, length(A), maps:get(speculative, St, none)
    ]),
    ?assertMatch(#{drafted := D} when D > 0, maps:get(speculative, St)).

%% Where the MTP output first differs from ordinary decoding, the top
%% two candidates of the ordinary run at that token: a small margin and
%% the MTP token being the runner-up says the 3-row verify batch only
%% rounds differently from a 1-row decode (as batching does on one GPU
%% too); a large margin says a rollback went wrong.
divergence_margin(Config) ->
    serve(Config, fun(S, #{id := Id}, _R0) ->
        Prompts = [<<"The history of ocean navigation begins with">> | dec_prompts()],
        Opts = #{response_tokens => 250, temperature => 0.0},
        Rows = [
            begin
                {ok, #{generated := A, logprobs := Lps}} =
                    peer:call(S, erllama, complete, [Id, P, Opts#{logprobs => 2}], 600000),
                {ok, #{generated := B}} =
                    peer:call(S, erllama, complete, [Id, P, maps:merge(Opts, ?MTP)], 600000),
                N = length(lists:takewhile(fun({X, Y}) -> X =:= Y end, lists:zip(A, B, trim))),
                case N < min(length(A), length(B)) of
                    true ->
                        #{top := Top, logprob := Lp} = lists:nth(N + 1, Lps),
                        {I, N, lists:nth(N + 1, A), lists:nth(N + 1, B), Lp, Top};
                    false ->
                        {I, N, same}
                end
            end
         || {I, P} <- lists:enumerate(Prompts)
        ],
        ct:pal(
            "{prompt, tokens the same, ordinary token, MTP token, its logprob, the ordinary run's top two}:~n~p",
            [Rows]
        )
    end).

%% Matching positions do not show that a rollback restored the right
%% state. For the round deciding each prompt's first divergence from
%% ordinary decoding, and the first rounds after a rejection: rebuild
%% the exact prefix the live MTP run had before the round in a fresh
%% group's context (one prefill), decode the same tokens in one batch
%% with every row out, and compare every row with the live run's (its
%% greedy choice and the logits of the live top 8). The batch shape and
%% row are the live ones, so only the state the rounds before left can
%% differ (and the prefix's own rounding: one prefill against rounds).
%% Tolerances, set before the run: a fresh greedy choice may differ
%% only where the live top two are within 0.5; the top-8 logits within
%% 2.0 (the largest difference seen between batch shapes was 1.62).
rollback_state_matches(Config) ->
    Live = serve(Config, fun(S, G, R0) ->
        Prompts = [chat(S, G, T) || T <- dec_prompts()],
        %% The drafter exists from the first MTP request.
        _ = run(S, G, hd(Prompts), 16, ?MTP),
        [live_rounds(S, G, R0, P) || P <- Prompts]
    end),
    Checks = with_group(Config, fun(G) ->
        lists:append([[replay(G, P, B, T) || T <- Targets] || {P, B, Targets} <- Live])
    end),
    ct:pal(
        "{why, after a rejection, position, row, live / fresh choice, live top-two margin, max |diff| over the live top 8}:~n~p",
        [Checks]
    ),
    [
        begin
            ?assert(Diff =< 2.0),
            ?assert(LiveTok =:= FreshTok orelse Margin =< 0.5)
        end
     || {_, _, _, _, {LiveTok, FreshTok}, Margin, Diff} <- Checks
    ],
    ?assert(lists:any(fun({_, After, _, _, _, _, _}) -> After end, Checks)).

%% The prompt, the MTP run's tokens and the rounds to check: the one
%% whose row chose the first token that differs from ordinary decoding,
%% and the first three that follow a rejected draft.
live_rounds(S, G, R0, P) ->
    #{stats := #{generated := A}} = run(S, G, P, ?TOKENS, #{}),
    [ok] = trace(S, R0, on),
    #{stats := #{generated := B}} = run(S, G, P, ?TOKENS, ?MTP),
    [Rounds] = trace(S, R0, read),
    [ok] = trace(S, R0, off),
    L = length(P),
    Indexed = lists:zip([undefined | lists:droplast(Rounds)], Rounds),
    Rejected = fun
        (undefined) -> false;
        (#{tokens := Ts, n_acc := Kept}) -> Kept < length(Ts) - 1
    end,
    N = length(lists:takewhile(fun({X, Y}) -> X =:= Y end, lists:zip(A, B, trim))),
    Div = [
        {divergence, Rejected(Prev), R, L + N - 1 - Pos}
     || N < min(length(A), length(B)),
        {Prev, #{pos := Pos, n_acc := Acc} = R} <- Indexed,
        Pos =< L + N - 1,
        L + N - 1 =< Pos + Acc
    ],
    After = lists:sublist(
        [{after_rejection, true, R, Acc} || {Prev, #{n_acc := Acc} = R} <- Indexed, Rejected(Prev)],
        3
    ),
    {P, B, Div ++ After}.

replay(
    G, P, B, {Why, AfterRej, #{pos := Pos, tokens := Toks, n_acc := Acc, rows := LiveRows}, Row}
) ->
    L = length(P),
    %% The round's kept tokens are the MTP run's at its position.
    ?assertEqual(lists:sublist(Toks, Acc + 1), lists:sublist(B, Pos - L + 1, Acc + 1)),
    {_, ok} = op(G, {drop, 0}),
    {_, {ok, _}} = op(G, {decode, 0, P ++ lists:sublist(B, Pos - L), false}),
    {_, {ok, Bin}} = op(G, {decode, 0, Toks, true}),
    Fresh = lists:nth(Row + 1, rows(Bin, length(Toks))),
    Live = lists:nth(Row + 1, LiveRows),
    [{LiveTok, V1}, {_, V2} | _] = Live,
    Diff = lists:max([abs(V - logit(Fresh, Id)) || {Id, V} <- Live]),
    {Why, AfterRej, Pos + Row, Row, {LiveTok, argmax(Fresh)}, V1 - V2, Diff}.

trace(S, R0, Op) ->
    peer:call(S, gen_server, call, [R0, {hosting, {mtp, Op}}, 60000], 70000).

with_group(Config, Fun) ->
    Hosts = ?config(hosts, Config),
    Model = ?config(model, Config),
    erllama_tp_hosts:with_serving(Hosts, fun(S) ->
        #{rank_peers := Peers, rank_pids := Pids} = erllama_tp_hosts:form(Hosts, S, 2),
        Ranks = lists:zip(Peers, Pids),
        try
            [
                ok = R
             || R <- erllama_tp_hosts:parallel([
                    fun() ->
                        peer:call(
                            P,
                            erllama_tp_rank,
                            load,
                            [R, erllama_tp_hosts:home(P, Model), {model_opts(), ?COPTS}],
                            3600000
                        )
                    end
                 || {P, R} <- Ranks
                ])
            ],
            Fun(#{ranks => Ranks})
        after
            erllama_tp_hosts:stop_ranks(Ranks),
            [catch peer:stop(P) || P <- Peers]
        end
    end).

%% Every rank makes the call and must answer the same.
op(#{ranks := Ranks}, Op) ->
    Res = erllama_tp_hosts:parallel([
        fun() -> peer:call(P, erllama_tp_rank, seq_op, [R, Op], 600000) end
     || {P, R} <- Ranks
    ]),
    Norm = [
        case X of
            {Ms, {ok, Bin, _, _}} -> {Ms, {ok, Bin}};
            Other -> Other
        end
     || X <- Res
    ],
    [{_, First} | _] = Norm,
    [?assertEqual(First, Y, Op) || {_, Y} <- Norm],
    {lists:max([Ms || {Ms, _} <- Norm]), First}.

rows(Bin, K) ->
    Sz = byte_size(Bin) div K,
    [binary:part(Bin, I * Sz, Sz) || I <- lists:seq(0, K - 1)].

logit(Row, Id) ->
    <<V:32/float-native>> = binary:part(Row, Id * 4, 4),
    V.

argmax(Bin) -> erllama_tp_glm_ref:argmax(Bin).

%% -----------------------------------------------------------------------------

serve(Config, Fun) ->
    Hosts = ?config(hosts, Config),
    os:putenv("ERLLAMA_TP_MODEL", ?config(model, Config)),
    erllama_tp_hosts:with_serving(Hosts, fun(S) ->
        G0 = erllama_tp_hosts:form(Hosts, S, 1),
        [P0 | _] = Peers = maps:get(rank_peers, G0),
        Before = [avail(P) || P <- Peers],
        G = erllama_tp_hosts:load(
            G0,
            ?NO_CACHE,
            #{
                model_opts => model_opts(),
                context_opts => ?COPTS,
                tp_call_timeout => 3600000,
                draft_model => erllama_tp_hosts:home(P0, ?config(mtp, Config)),
                draft_model_opts => #{load_mode => direct_io}
            },
            3600000
        ),
        After = [avail(P) || P <- Peers],
        ct:pal(
            "loaded with the MTP head on rank 0: memory available per rank ~w GB before, ~w GB after",
            [
                [round(B / 1.0e9) || B <- Before], [round(A / 1.0e9) || A <- After]
            ]
        ),
        try
            Fun(S, G, hd(maps:get(rank_pids, G)))
        after
            erllama_tp_hosts:stop_group(G)
        end
    end).

model_opts() ->
    case os:getenv("ERLLAMA_TP_MTP_SPLIT") of
        false -> ?MOPTS;
        V -> maps:put(tensor_split, [list_to_float(F) || F <- string:lexemes(V, ",")], ?MOPTS)
    end.

run(S, #{id := Id}, Prompt, Max, Opts) ->
    peer:call(S, erllama_tp_stream, run, [Id, Prompt, Max, infinity, Opts], 600000).

mtp_stats(S, R0) ->
    case peer:call(S, gen_server, call, [R0, {hosting, {mtp, stats}}, 60000], 70000) of
        [K] -> K;
        [] -> #{}
    end.

diff(B, A) ->
    maps:map(fun(K, V) -> V - maps:get(K, A, 0) end, B).

dec_prompts() ->
    [
        <<"Write five detailed paragraphs about the history of ocean navigation.">>,
        <<"Explain how a modern operating system schedules threads. Be thorough and use several paragraphs.">>,
        <<"Describe the engineering of suspension bridges in five paragraphs.">>,
        <<"Write five paragraphs on the development of the printing press and its consequences.">>
    ].

chat(S, #{id := Id}, Text) ->
    peer:call(S, erllama_tp_stream, chat_tokens, [Id, Text], 60000).

rate(#{gaps_ms := []}) -> 0.0;
rate(#{gaps_ms := Gaps}) -> length(Gaps) / lists:sum(Gaps) * 1000.

mean(L) -> lists:sum(L) / max(1, length(L)).

avail(P) ->
    {ok, Bin} = peer:call(P, file, read_file, ["/proc/meminfo"]),
    {match, [Kb]} = re:run(Bin, "MemAvailable:\\s+([0-9]+) kB", [{capture, all_but_first, binary}]),
    binary_to_integer(Kb) * 1024.
