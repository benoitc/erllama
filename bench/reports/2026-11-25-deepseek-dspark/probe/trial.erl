-module(trial).
-export([reference/0, split/0, prepare/1, single/0, command/1]).
-define(D, "/tmp/erllama-dspark-layers").
-define(K, {?MODULE, state}).
opts() ->
    #{
        n_ctx => 2048,
        n_batch => 512,
        n_seq_max => 4,
        n_rs_seq =>
            case os:getenv("DSPARK_LEG") of
                "teacher" -> 0;
                _ -> 2
            end,
        kv_unified => true,
        flash_attn => enabled
    }.
model() -> #{n_gpu_layers => 999, load_mode => direct_io}.
ids() ->
    {ok, B} = file:read_file(?D ++ "/fixed.ids"),
    [binary_to_integer(X) || X <- binary:split(string:trim(B), <<" ">>, [global])].
reference() ->
    {ok, _} = application:ensure_all_started(erllama),
    ok = single(),
    halt().
single() ->
    {ok, M} = erllama_nif:load_model(?D ++ "/model.gguf", model()),
    {ok, C} = erllama_nif:new_context(M, opts()),
    ok = dsprobe:load(?D ++ "/probe"),
    putstate(C),
    ok = command(capture),
    {Prefix, [First | _]} = lists:split(70, ids()),
    [_] = command({decode, 0, Prefix, false}),
    ok = command({save_rows, "reference-pre", length(Prefix)}),
    [_] = command({decode, 0, [First], false}),
    ok = command({save_rows, "reference-one", 1}),
    ok.
putstate(C) -> persistent_term:put(?K, #{ctx => C}).
prepare(R) ->
    {module, erllama_nif} = code:ensure_loaded(erllama_nif),
    C = element(6, sys:get_state(R)),
    true = is_reference(C),
    ok = dsprobe:load(?D ++ "/probe"),
    putstate(C).
command(Op) ->
    S = persistent_term:get(?K),
    C = maps:get(ctx, S),
    case Op of
        capture ->
            dsprobe:capture(C);
        plain ->
            dsprobe:plain(C);
        {many, Ids} ->
            dsprobe:many(C, Ids);
        head_save ->
            B = dsprobe:checkpoint(C, maps:get(h, S)),
            true = is_binary(B),
            persistent_term:put(?K, S#{head_save => B});
        head_restore ->
            dsprobe:restore(C, maps:get(h, S), maps:get(head_save, S));
        head ->
            H = dsprobe:head(C, ?D ++ "/standalone.gguf"),
            is_reference(H) orelse error({head_init, H}),
            persistent_term:put(?K, S#{h => H});
        {decode, I, T, All} ->
            {ok, B, _, _} = erllama_nif:tp_decode_seq(C, I, T, All),
            dsprobe:top(B, 129280);
        {feed, Pos, Ts} ->
            dsprobe:feed(C, maps:get(h, S), Pos, Ts);
        {draft, Pos, K, N} ->
            dsprobe:draft(C, maps:get(h, S), Pos, K, N);
        {trim, I, Keep} ->
            erllama_nif:kv_seq_rm(C, I, Keep, -1);
        {save_rows, Name, Num} ->
            file:write_file(?D ++ "/" ++ Name ++ ".f32", dsprobe:rows(C, Num));
        {pos, I} ->
            erllama_nif:kv_seq_pos_max(C, I);
        save ->
            Blob = erllama_nif:tp_kv_pack(C, 0),
            true = is_binary(Blob),
            persistent_term:put(?K, S#{saved => Blob});
        restore ->
            erllama_nif:tp_kv_unpack(C, maps:get(saved, S), 0)
    end.
call(P, Op) -> peer:call(P, ?MODULE, command, [Op], 90000).
both(Ps, Op) ->
    [X | Xs] = erllama_tp_hosts:parallel([fun() -> call(P, Op) end || P <- Ps]),
    case lists:all(fun(Y) -> Y =:= X end, Xs) of
        true -> X;
        false -> error({rank_disagreement, element(1, Op)})
    end.
split() ->
    H = erllama_tp_hosts:hosts(),
    erllama_tp_hosts:with_serving(H, fun(S) ->
        #{rank_peers := Ps, rank_pids := Rs} = erllama_tp_hosts:form(H, S, 1),
        try
            Pairs = lists:zip(Ps, Rs),
            [
                ok = V
             || V <- erllama_tp_hosts:parallel([
                    fun() ->
                        ok = peer:call(
                            P,
                            erllama_tp_rank,
                            load,
                            [R, ?D ++ "/model.gguf", {model(), opts()}],
                            150000
                        ),
                        [ok = erllama_tp_hosts:load_module(P, Mod) || Mod <- [?MODULE, dsprobe]],
                        peer:call(P, ?MODULE, prepare, [R], 30000)
                    end
                 || {P, R} <- Pairs
                ])
            ],
            benchmark(Ps)
        after
            erllama_tp_hosts:stop_ranks(Pairs0 = lists:zip(Ps, Rs)),
            [catch peer:stop(P) || {P, _} <- Pairs0]
        end
    end).
benchmark([Leader | _] = Ps) ->
    {Prefix, [First | _]} = lists:split(70, ids()),
    K =
        case os:getenv("DSPARK_DIAGNOSTIC") of
            "1" ->
                [Initial] = both(Ps, {decode, 0, Prefix, false}),
                ok = both(Ps, {trim, 0, 0}),
                Initial;
            _ ->
                feature_gate(Ps, Prefix, First)
        end,
    [K] = both(Ps, {decode, 0, Prefix, false}),
    {BaseUs, Reference} = timer:tc(fun() -> ordinary(Ps, K, 65) end),
    write(base, #{us => BaseUs, tokens => Reference}),
    ok = both(Ps, {trim, 0, 0}),
    ok = both(Ps, capture),
    ok = call(Leader, head),
    [Anchor] = both(Ps, {decode, 0, Prefix, false}),
    69 = call(Leader, {feed, 0, Prefix}),
    case os:getenv("DSPARK_LEG") of
        "teacher" -> teacher(Ps, Leader, 70, Reference, []);
        _ -> verified(Ps, Leader, Anchor, Reference)
    end.
feature_gate(Ps, Prefix, First) ->
    ok = both(Ps, capture),
    [K] = both(Ps, {decode, 0, Prefix, false}),
    ok = both(Ps, {save_rows, "split-pre", length(Prefix)}),
    [_] = both(Ps, {decode, 0, [First], false}),
    ok = both(Ps, {save_rows, "split-one", 1}),
    {Msg, Code} = shell_check(),
    io:format("FEATURE COMPARISON ~s~n", [Msg]),
    case Code of
        0 -> ok;
        _ -> error(feature_comparison_failed)
    end,
    ok = both(Ps, {trim, 0, 0}),
    ok = both(Ps, plain),
    K.
teacher(_, _, _, [_Last], Acc) ->
    write(teacher, lists:reverse(Acc));
teacher(Ps, L, Pos, [Anchor | [_ | _] = Rest], Acc) ->
    {Us, G} = timer:tc(fun() -> call(L, {draft, Pos, Anchor, 2}) end),
    [N] = both(Ps, {decode, 0, [Anchor], false}),
    N = hd(Rest),
    Pos = call(L, {feed, Pos, [Anchor]}),
    teacher(Ps, L, Pos + 1, Rest, [
        #{pos => Pos, guesses => G, accepted => same_prefix(G, Rest), draft_us => Us} | Acc
    ]).
verified(Ps, Leader, Anchor, Reference) ->
    Rounds = rounds(Ps, Leader, 70, Anchor, 64, []),
    Tokens = lists:append([maps:get(committed, R) || R <- Rounds]),
    write(draft, #{
        rounds => Rounds, greedy_common_prefix => same_prefix(Tokens, Reference), tokens => Tokens
    }),
    Keep = 70 + length(Tokens),
    Last = lists:last(Rounds),
    % The target predicts the next known token in the last accepted row.
    Known = maps:get(next, Last),
    transitions(Ps, Leader, Keep, Known),
    ok.
transitions(Ps, L, Pos, K) ->
    % Save both target shares and the local drafter at the committed boundary.
    ok = both(Ps, save),
    ok = call(L, head_save),
    G = call(L, {draft, Pos, K, 2}),
    Pred = both(Ps, {decode, 0, [K | G], true}),
    % Cancelling the outstanding guesses leaves the known token committed.
    ok = both(Ps, {trim, 0, Pos + 1}),
    Pos = call(L, {feed, Pos, [K]}),
    ok = both(Ps, {trim, 0, 0}),
    ok = both(Ps, restore),
    ok = call(L, head_restore),
    RestoredG = call(L, {draft, Pos, K, 2}),
    RestoredPred = both(Ps, {decode, 0, [K | G], true}),
    write(restore_check, #{
        before_guesses => G,
        after_guesses => RestoredG,
        before_verify => Pred,
        after_verify => RestoredPred
    }),
    ok = both(Ps, {trim, 0, Pos + 1}),
    Pos = call(L, {feed, Pos, [K]}),
    [First | _] = RestoredPred,
    Q = lists:foldl(
        fun(I, A) ->
            [Seed] = both(Ps, {decode, I, [4, I + 4], false}),
            A ++ [Seed]
        end,
        [First],
        lists:seq(1, 3)
    ),
    {Q1, Pos1} = lists:foldl(
        fun(_, {Ids, P}) ->
            Next = both(Ps, {many, Ids}),
            [Committed | _] = Ids,
            P = call(L, {feed, P, [Committed]}),
            {Next, P + 1}
        end,
        {Q, Pos + 1},
        lists:seq(1, 8)
    ),
    [ok = both(Ps, {trim, I, 0}) || I <- lists:seq(1, 3)],
    [Anchor | _] = Q1,
    Back = rounds(Ps, L, Pos1, Anchor, 8, []),
    write(transitions, #{
        restore_guesses_match => (G =:= RestoredG),
        restore_predictions_match => (Pred =:= RestoredPred),
        four_ordinary_steps => 8,
        solo_rounds_after => Back
    }).
shell_check() ->
    Port = open_port({spawn_executable, "/bin/bash"}, [
        binary, exit_status, use_stdio, stderr_to_stdout, {args, [?D ++ "/compare.sh"]}
    ]),
    collect(Port, []).
collect(P, L) ->
    receive
        {P, {data, B}} -> collect(P, [B | L]);
        {P, {exit_status, N}} -> {iolist_to_binary(lists:reverse(L)), N}
    end.
ordinary(_, _, 0) ->
    [];
ordinary(Ps, K, N) ->
    [Next] = both(Ps, {decode, 0, [K], false}),
    [K | ordinary(Ps, Next, N - 1)].
rounds(_, _, _, _, N, A) when N =< 0 -> lists:reverse(A);
rounds(Ps, L, Pos, K, N, A) ->
    T = erlang:monotonic_time(microsecond),
    G = call(L, {draft, Pos, K, 2}),
    (is_list(G) andalso G =/= []) orelse error({empty_proposal, Pos, G}),
    D = erlang:monotonic_time(microsecond),
    Predictions = both(Ps, {decode, 0, [K | G], true}),
    true = length(Predictions) =:= 1 + length(G),
    V = erlang:monotonic_time(microsecond),
    Count = same_prefix(G, Predictions),
    Committed = [K | lists:sublist(G, Count)],
    Keep = Pos + length(Committed),
    case Count < length(G) of
        true -> ok = both(Ps, {trim, 0, Keep});
        false -> ok
    end,
    End = call(L, {feed, Pos, Committed}),
    (End =:= Keep - 1) orelse error({draft_position, Pos, Keep - 1, End}),
    F = erlang:monotonic_time(microsecond),
    Row = #{
        next => lists:nth(1 + Count, Predictions),
        committed => Committed,
        guesses => G,
        accepted => Count,
        us => F - T,
        draft_us => D - T,
        verify_us => V - D,
        trim_feed_us => F - V
    },
    ok = file:write_file(?D ++ "/rounds.jsonl", [json:encode(Row), <<"\n">>], [append]),
    rounds(Ps, L, Keep, lists:nth(1 + Count, Predictions), N - length(Committed), [Row | A]).
same_prefix([X | Xs], [X | Ys]) -> 1 + same_prefix(Xs, Ys);
same_prefix(_, _) -> 0.
write(Name, Data) -> file:write_file(?D ++ "/" ++ atom_to_list(Name) ++ ".json", json:encode(Data)).
