%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Test helper for erllama_tp_mtp_SUITE:mtp_batch, loaded onto the serving
%% node: four greedy requests on rank 0's hosted backend, decoded either
%% by ordinary co-batched steps (one token a request a step, as the
%% scheduler does) or by batched MTP rounds (one guess a request, every
%% request's token and guess verified in one batch). Calls go to rank 0
%% as the serving node's do (erllama_tp_rank:backend/3).
-module(erllama_tp_mtp_bench).

-export([setup/2, ordinary/4, mtp/5]).

%% The continuation compared after a restore: ?CONT greedy tokens, of
%% which at least ?MIN_CLEAR with the live path's top token leading by
%% ?MARGIN nats.
-define(CONT, 8).
-define(MARGIN, 1.0).
-define(MIN_CLEAR, 4).

%% Greedy samplers for `Seqs' and an MTP drafter of one guess.
setup(R0, Seqs) ->
    Samplers = [element(2, {ok, _} = be(R0, sampler_new, [#{temperature => 0.0}])) || _ <- Seqs],
    {ok, Mtp} = be(R0, spec_new, [#{type => mtp, n_max => 1, n_seq => length(Seqs)}]),
    #{samplers => maps:from_list(lists:zip(Seqs, Samplers)), mtp => Mtp}.

%% Fresh sequences: the prompts prefilled in one step, then each first
%% token (a plain decode step). Returns each sequence's tokens so far.
start(R0, #{samplers := Sm}, Prompts) ->
    Seqs = [S || {S, _} <- Prompts],
    [ok = be(R0, seq_rm, [S]) || S <- Seqs],
    {ok, _} = be(R0, step, [[{S, {prefill, P}} || {S, P} <- Prompts]]),
    {ok, First} = be(R0, step, [[{S, {decode, maps:get(S, Sm)}} || S <- Seqs]]),
    maps:from_list([{S, [T]} || {S, {token, T, _}} <- First]).

%% Ordinary co-batched decoding until every sequence has `Max' tokens.
%% Returns the tokens, the wall time from the first decode, and each
%% step's wall time (ms).
ordinary(R0, Ctx, Prompts, Max) ->
    #{samplers := Sm} = Ctx,
    T0 = now_us(),
    Toks0 = start(R0, Ctx, Prompts),
    loop_ordinary(R0, Sm, Max, Toks0, [], T0).

loop_ordinary(R0, Sm, Max, Toks, Steps, T0) ->
    Active = [S || {S, Ts} <- maps:to_list(Toks), length(Ts) < Max, not eog(Ts)],
    case Active of
        [] ->
            #{tokens => Toks, wall_ms => (now_us() - T0) / 1000, steps_ms => lists:reverse(Steps)};
        _ ->
            T = now_us(),
            {ok, Res} = be(R0, step, [[{S, {decode, maps:get(S, Sm)}} || S <- Active]]),
            Toks1 = lists:foldl(
                fun({S, {token, Tok, E}}, Acc) ->
                    maps:update_with(S, fun(L) -> L ++ [mark(Tok, E)] end, Acc)
                end,
                Toks,
                Res
            ),
            loop_ordinary(R0, Sm, Max, Toks1, [(now_us() - T) / 1000 | Steps], T0)
    end.

%% Batched MTP rounds until every sequence has `Max' tokens. `Plan':
%% `forced => #{Round => #{Seq => [Guess]}}' replaces the drafter's guess
%% for those sequences in that round; `cancel => {Round, Seq}' drops a
%% sequence after that round (its KV removed, as a cancel does);
%% `resume => Round' runs the checkpoint regression (resume/3) after the
%% first round from that one in which every guess was accepted, and ends
%% the run there. A sequence with one token left gets no
%% guess. Returns the tokens, the wall time, each round's wall time,
%% results and native timings, and the round a resume happened after.
mtp(R0, Ctx, Prompts, Max, Plan) ->
    T0 = now_us(),
    Toks0 = start(R0, Ctx, Prompts),
    Ctx1 = Ctx#{prompt_len => maps:from_list([{S, length(P)} || {S, P} <- Prompts])},
    loop_mtp(R0, Ctx1, Max, Plan, Toks0, 1, [], T0).

loop_mtp(R0, #{samplers := Sm, mtp := Mtp} = Ctx, Max, Plan, Toks, N, Rounds, T0) ->
    Active = [S || {S, Ts} <- lists:sort(maps:to_list(Toks)), length(Ts) < Max, not eog(Ts)],
    case Active of
        [] ->
            #{
                tokens => Toks,
                wall_ms => (now_us() - T0) / 1000,
                rounds => lists:reverse(Rounds),
                resumed => none,
                resume_info => none
            };
        _ ->
            Forced = maps:get(N, maps:get(forced, Plan, #{}), #{}),
            Items = [
                {
                    maps:get(S, Sm),
                    S,
                    case {maps:find(S, Forced), Max - length(maps:get(S, Toks))} of
                        {{ok, G}, _} -> G;
                        {error, Left} when Left =< 1 -> [];
                        {error, _} -> auto
                    end
                }
             || S <- Active
            ],
            T = now_us(),
            {ok, Res, Times} = be(R0, mtp_step_batch, [Mtp, Items, 1]),
            Wall = (now_us() - T) / 1000,
            Toks1 = lists:foldl(
                fun({S, Committed, E, _NAcc}, Acc) ->
                    maps:update_with(S, fun(L) -> L ++ marked(Committed, E) end, Acc)
                end,
                Toks,
                Res
            ),
            Toks2 =
                case maps:get(cancel, Plan, none) of
                    {N, Seq} ->
                        ok = be(R0, seq_rm, [Seq]),
                        maps:remove(Seq, Toks1);
                    _ ->
                        Toks1
                end,
            Resume =
                case maps:get(resume, Plan, none) of
                    From when is_integer(From), N >= From, length(Res) =:= length(Active) ->
                        lists:all(fun({_, _, _, A}) -> A >= 1 end, Res);
                    _ ->
                        false
                end,
            Round = #{
                n => N,
                wall_ms => Wall,
                times => Times,
                results => Res,
                drafted => [S || {_, S, F} <- Items, F =/= []]
            },
            case Resume of
                true ->
                    #{
                        tokens => Toks2,
                        wall_ms => (now_us() - T0) / 1000,
                        rounds => lists:reverse([Round | Rounds]),
                        resumed => N,
                        resume_info => resume(R0, Ctx, maps:with(Active, Toks2))
                    };
                false ->
                    loop_mtp(R0, Ctx, Max, Plan, Toks2, N + 1, [Round | Rounds], T0)
            end
    end.

%% The checkpoint regression on every sequence in `Toks', after a round
%% that accepted every guess. With a rollback pending (each one's last
%% token taken back): a checkpoint A, the token decoded on the live
%% state (checkpoint B) and ?CONT greedy steps; then A restored, saved
%% again at once, the same token (C) and the same steps. With none
%% pending: a checkpoint D, a token decoded live (E); D restored, saved
%% again at once, the same token (F). Per sequence: whether each restore
%% gave back the checkpoint (saved_restored), whether the first step
%% left the same state (first_step), and, with a rollback pending, the
%% continuation of both paths (continuation/2).
resume(R0, #{prompt_len := PL}, Toks) ->
    L = lists:sort(maps:keys(Toks)),
    Last = [{Seq, {prefill, [lists:last(maps:get(Seq, Toks))]}} || Seq <- L],
    %% A rollback pending at the checkpoint: the restore lays the cells
    %% out otherwise than the live path keeps them.
    [ok = be(R0, seq_rm_last, [Seq, maps:get(Seq, PL) + length(maps:get(Seq, Toks))]) || Seq <- L],
    A = packs(R0, L),
    {ok, _} = be(R0, step, [Last]),
    B = packs(R0, L),
    LiveCont = continue(R0, L, ?CONT),
    {ARe, C} = restore_step(R0, L, A, Last),
    RestCont = continue(R0, L, ?CONT),
    %% No rollback pending: the restore keeps the layout.
    X = [{Seq, {prefill, [1000]}} || Seq <- L],
    D = packs(R0, L),
    {ok, _} = be(R0, step, [X]),
    E = packs(R0, L),
    {DRe, F} = restore_step(R0, L, D, X),
    maps:from_list([
        {Seq, #{
            pending => #{
                saved_restored => same(maps:get(Seq, A), maps:get(Seq, ARe)),
                first_step => same(maps:get(Seq, B), maps:get(Seq, C)),
                continuation => continuation(maps:get(Seq, LiveCont), maps:get(Seq, RestCont))
            },
            none_pending => #{
                saved_restored => same(maps:get(Seq, D), maps:get(Seq, DRe)),
                first_step => same(maps:get(Seq, E), maps:get(Seq, F))
            }
        }}
     || Seq <- L
    ]).
%% Every sequence restored from `Packs', saved again at once (before any
%% decode), then `Step' decoded and saved.
restore_step(R0, L, Packs, Step) ->
    [
        begin
            ok = be(R0, seq_rm, [Seq]),
            ok = kv(R0, {kv_unpack, maps:get(Seq, Packs), Seq})
        end
     || Seq <- L
    ],
    Re = packs(R0, L),
    {ok, _} = be(R0, step, [Step]),
    {Re, packs(R0, L)}.

packs(R0, L) -> maps:from_list([{Seq, kv(R0, {kv_pack, Seq})} || Seq <- L]).

%% Two checkpoints of one sequence, rank by rank: the same bytes, and
%% the same state by token position (erllama_kv_seq_state; another
%% layout of the same cells is the same state).
same(F1, F2) ->
    {ok, S1} = erllama_tp_checkpoint:unframe(F1, 2),
    {ok, S2} = erllama_tp_checkpoint:unframe(F2, 2),
    [
        #{bytes => X =:= Y, state => erllama_kv_seq_state:equal(X, Y)}
     || {X, Y} <- lists:zip(S1, S2)
    ].

%% `N' greedy decode steps of every sequence together, each token with
%% the top log-probabilities.
continue(R0, L, N) ->
    Sm = maps:from_list([
        {Seq, element(2, {ok, _} = be(R0, sampler_new, [#{temperature => 0.0, logprobs => 5}]))}
     || Seq <- L
    ]),
    Steps = [
        begin
            {ok, Res} = be(R0, step, [[{Seq, {decode, maps:get(Seq, Sm)}} || Seq <- L]]),
            maps:from_list([{Seq, {T, Top}} || {Seq, {token, T, _, {_, Top}}} <- Res])
        end
     || _ <- lists:seq(1, N)
    ],
    maps:from_list([{Seq, [maps:get(Seq, S) || S <- Steps]} || Seq <- L]).

%% The ?CONT greedy tokens of both paths, compared in order: `{ok, Clear}'
%% when they agree, Clear of them with the live path's top token leading
%% the next by at least ?MARGIN nats; a token that differs where the
%% lead is clear fails (`{differs_at, Pos, TopLogprobs}'); one that
%% differs where it is not ends the comparison (the contexts differ
%% after it). Fewer than ?MIN_CLEAR clear tokens compared is
%% `{too_few_clear_tokens, Margins}'.
continuation(Live, Rest) ->
    compare_cont(lists:enumerate(lists:zip(Live, Rest)), 0).

compare_cont([], Clear) ->
    clear_enough(Clear, ok);
compare_cont([{_I, {{T, Top}, {T, _}}} | More], Clear) ->
    compare_cont(More, Clear + clear(Top));
compare_cont([{I, {{_, Top1}, {_, Top2}}} | _], Clear) ->
    case clear(Top1) of
        1 -> {differs_at, I, #{live => Top1, restored => Top2}};
        0 -> clear_enough(Clear, {ends_at, I})
    end.

clear_enough(Clear, _How) when Clear >= ?MIN_CLEAR -> {ok, Clear};
clear_enough(Clear, How) -> {too_few_clear_tokens, Clear, How}.

clear(Top) ->
    case margin(Top) >= ?MARGIN of
        true -> 1;
        false -> 0
    end.

margin([{_, L1}, {_, L2} | _]) -> L1 - L2;
margin(_) -> 0.0.

kv(R0, Req) -> gen_server:call(R0, {hosting, Req}, 600000).

be(R0, Fn, Args) -> erllama_tp_rank:backend(R0, Fn, Args).

marked(Committed, 1) -> lists:droplast(Committed) ++ [{eog, lists:last(Committed)}];
marked(Committed, 0) -> Committed.

mark(Tok, 1) -> {eog, Tok};
mark(Tok, _) -> Tok.

eog(Ts) -> lists:any(fun(X) -> is_tuple(X) end, Ts).

now_us() -> erlang:monotonic_time(microsecond).
