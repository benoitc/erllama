-module(checkpoint_pause).
-export([run/0]).

run() ->
    {ok, _} = application:ensure_all_started(erllama),
    Out = os:getenv("PAUSE_OUTPUT", "pause.csv"),
    ok = file:write_file(Out, "pair,prompt_tokens,save,pack_bytes,pack_ms,neighbor_gap_ms\n"),
    lists:foreach(
        fun(Rep) ->
            lists:foreach(
                fun(N) ->
                    Order =
                        case Rep rem 2 of
                            0 -> [true, false];
                            1 -> [false, true]
                        end,
                    lists:foreach(fun(Save) -> trial(Out, Rep, N, Save) end, Order)
                end,
                [128, 512, 1536]
            )
        end,
        lists:seq(1, list_to_integer(os:getenv("PAUSE_REPEATS", "30")))
    ).

trial(Out, Rep, N, Save) ->
    Model = <<"pause">>,
    Min =
        case Save of
            true -> 100;
            false -> 4096
        end,
    Cfg = #{
        fingerprint => crypto:hash(sha256, term_to_binary({Rep, N, Save})),
        model_path => os:getenv("LLAMA_TEST_MODEL"),
        model_opts => #{n_gpu_layers => 999},
        context_size => 4096,
        context_opts => #{n_ctx => 4096, n_seq_max => 2, kv_unified => true, n_batch => 512},
        tier => ram,
        tier_srv => erllama_cache_ram,
        policy => #{
            min_tokens => Min,
            cold_min_tokens => 4096,
            cold_max_tokens => 4096,
            continued_interval => 4096
        }
    },
    {ok, Model} = erllama:load_model(Model, Cfg),
    try
        {ok, _} = erllama:complete(Model, <<"warm up">>, opts(4)),
        {ok, Pid} = erllama:whereis(Model),
        Tr = trace:session_create(pause, self(), []),
        try
            1 = trace:function(Tr, {erllama_model, fire_finish_save_for_req, 3}, true, [local]),
            1 = trace:function(
                Tr,
                {erllama_model, publish_packed, 3},
                [{['$1', '_', '_'], [{'is_binary', '$1'}], [{message, {byte_size, '$1'}}]}],
                [local]
            ),
            1 = trace:process(Tr, Pid, true, [call, arity, monotonic_timestamp]),
            Before = erllama_cache:get_counters(),
            %% Distinct first tokens between pairs prevent warm-prefix restores.
            Prompt = [1000 + Rep * 3 + N rem 7 | lists:duplicate(N - 1, 1234)],
            {ok, Long} = erllama:stream(Model, Prompt, opts(20)),
            receive
                {erllama, Long, {token, _}} -> ok
            after 60000 -> error(long_timeout)
            end,
            {ok, Neighbor} = erllama:stream(Model, [500, 612, 721], opts(48)),
            {Times, Stats} = collect(Neighbor, Long, [], #{}),
            Mark = trace:delivered(Tr, Pid),
            receive
                {trace_delivered, Pid, Mark} -> ok
            after 3000 -> error(trace_timeout)
            end,
            [Finish | _] = finish_times(Pid, []),
            Bytes = pack_sizes(Pid, []),
            true =
                (length(Bytes) =:=
                    case Save of
                        true -> 1;
                        false -> 0
                    end),
            #{completion_tokens := 20, cache_delta := #{read := 0}} = Stats,
            %% Delivery of a token already sent can overlap the start of a
            %% checkpoint. Include the neighboring gaps on both sides.
            [A0, A1 | _] = lists:reverse([T || T <- Times, T < Finish]),
            [B0, B1 | _] = [T || T <- Times, T >= Finish],
            Gap = lists:max([A0 - A1, B0 - A0, B1 - B0]),
            After = erllama_cache:get_counters(),
            Pack = (maps:get(pack_total_ns, After) - maps:get(pack_total_ns, Before)) / 1.0e6,
            Row = io_lib:format("~B,~B,~p,~B,~.3f,~.3f~n", [
                Rep, N, Save, lists:sum(Bytes), Pack, Gap / 1000
            ]),
            ok = file:write_file(Out, Row, [append])
        after
            trace:session_destroy(Tr)
        end
    after
        ok = erllama:unload(Model)
    end.

opts(N) -> #{temperature => 0.0, response_tokens => N, ignore_eos => true}.

collect(Neighbor, Long, Times, Stats) ->
    receive
        {erllama, Neighbor, {token, _}} ->
            collect(Neighbor, Long, [erlang:monotonic_time(microsecond) | Times], Stats);
        {erllama, Long, {token, _}} ->
            collect(Neighbor, Long, Times, Stats);
        {erllama, Long, {done, S}} ->
            collect(Neighbor, Long, Times, S);
        {erllama, Neighbor, {done, #{completion_tokens := 48}}} ->
            {lists:reverse(Times), Stats};
        {erllama, _, {error, R}} ->
            error(R)
    after 60000 -> error(stream_timeout)
    end.

finish_times(Pid, Acc) ->
    receive
        {trace_ts, Pid, call, {erllama_model, fire_finish_save_for_req, 3}, T} ->
            finish_times(Pid, [erlang:convert_time_unit(T, native, microsecond) | Acc])
    after 0 -> lists:reverse(Acc)
    end.

pack_sizes(Pid, Acc) ->
    receive
        {trace_ts, Pid, call, {erllama_model, publish_packed, 3}, Size, _} ->
            pack_sizes(Pid, [Size | Acc])
    after 0 -> Acc
    end.
