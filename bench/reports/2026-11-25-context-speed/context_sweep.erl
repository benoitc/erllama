-module(context_sweep).
-export([mac/0, cuda/0, split/0, run/4]).

context() -> #{n_ctx => 9216, n_seq_max => 1, n_batch => 512, flash_attn => enabled, n_rs_seq => 0}.

corpus() ->
    {ok, Text} = file:read_file(os:getenv("SWEEP_TEXT", "README.md")),
    binary:copy(<<Text/binary, "\n">>, 32).

mac() ->
    single(#{n_gpu_layers => 999}).

cuda() ->
    single(#{n_gpu_layers => 999, load_mode => direct_io}).

single(Opts) ->
    {ok, _} = application:ensure_all_started(erllama),
    {ok, State} = erllama_model_llama:init(#{
        model_path => os:getenv("SWEEP_MODEL"),
        model_opts => Opts,
        context_opts => context()
    }),
    Call = fun(F, Args) -> apply(erllama_model_llama, F, [State | Args]) end,
    try
        run(Call, local, corpus(), os:getenv("SWEEP_OUTPUT"))
    after
        erllama_model_llama:terminate(State)
    end.

split() ->
    Hosts = erllama_tp_hosts:hosts(),
    Text = corpus(),
    erllama_tp_hosts:with_serving(Hosts, fun(S) ->
        Off = #{
            min_tokens => infinity, cold_min_tokens => infinity, continued_interval => infinity
        },
        G = erllama_tp_hosts:serve(Hosts, S, 1, Off, #{
            model_opts => #{n_gpu_layers => 999, load_mode => direct_io}, context_opts => context()
        }),
        try
            [Rank | _] = maps:get(rank_pids, G),
            ok = erllama_tp_hosts:load_module(S, ?MODULE),
            ok = peer:call(
                S, ?MODULE, run, [Rank, distributed, Text, os:getenv("SWEEP_OUTPUT")], infinity
            )
        after
            erllama_tp_hosts:stop_group(G)
        end
    end).

run(Rank, Mode, Text, Path) when is_pid(Rank) ->
    run(fun(F, Args) -> erllama_tp_rank:backend(Rank, F, Args) end, Mode, Text, Path);
run(Call, Mode, Text, Path) ->
    Tokens = Call(tokenize, [Text]),
    true = length(Tokens) >= 8192,
    {ok, Sampler} = Call(sampler_new, [#{temperature => 0.0}]),
    ok = file:write_file(
        Path,
        "repeat,context_tokens,extension_tokens,sampled_tokens,eog_steps,first_token_us,remaining_us\n"
    ),
    warmup(Call, Sampler, Tokens),
    lists:foreach(
        fun({Rep, Points}) ->
            lists:foreach(
                fun(Depth) ->
                    ok = Call(seq_rm, [0]),
                    {History, Block} = lists:split(Depth - 2048, lists:sublist(Tokens, Depth)),
                    fill(Call, History),
                    sync_history(Mode, Call, History),
                    Start = erlang:monotonic_time(microsecond),
                    fill(Call, Block),
                    Eog = sample(Call, Sampler),
                    First = erlang:monotonic_time(microsecond),
                    Ends = Eog + lists:sum([sample(Call, Sampler) || _ <- lists:seq(1, 127)]),
                    Stop = erlang:monotonic_time(microsecond),
                    Row = io_lib:format("~B,~B,2048,128,~B,~B,~B~n", [
                        Rep, Depth, Ends, First - Start, Stop - First
                    ]),
                    ok = file:write_file(Path, Row, [append])
                end,
                Points
            )
        end,
        [{1, [2048, 4096, 6144, 8192]}, {2, [8192, 6144, 4096, 2048]}]
    ).

%% The Metal backend can return before its last prefill completes.
%% Reading the history state outside the timer fences that work. A split
%% step already reads back the reduced logits before it returns.
sync_history(local, Call, [_ | _] = Tokens) ->
    true = is_binary(Call(kv_pack, [Tokens])),
    ok;
sync_history(_, _, _) ->
    ok.

warmup(Call, Sampler, Tokens) ->
    fill(Call, lists:sublist(Tokens, 512)),
    [sample(Call, Sampler) || _ <- lists:seq(1, 8)],
    ok.

fill(_, []) ->
    ok;
fill(Call, Tokens) ->
    {Slice, Rest} = lists:split(min(512, length(Tokens)), Tokens),
    {ok, [{0, prefilled}]} = Call(step, [[{0, {prefill, Slice}}]]),
    fill(Call, Rest).

sample(Call, Sampler) ->
    {ok, [{0, {token, _, Eog}}]} = Call(step, [[{0, {decode, Sampler}}]]),
    Eog.
