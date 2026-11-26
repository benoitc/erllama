%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
-module(erllama_mlx_SUITE).
-include_lib("stdlib/include/assert.hrl").
-export([all/0, init_per_suite/1, end_per_suite/1, qwen_greedy/1, disk_resume/1, worker_loss/1]).

all() -> [qwen_greedy, disk_resume, worker_loss].

init_per_suite(Config) ->
    case os:getenv("ERLLAMA_MLX_TEST_MODEL") of
        false ->
            {skip, "Set ERLLAMA_MLX_TEST_MODEL to the reference Qwen3.8-27B MLX directory"};
        Path ->
            {ok, _} = application:ensure_all_started(erllama),
            Disk = mlx_test_disk,
            Dir = filename:join(proplists:get_value(priv_dir, Config), "mlx-checkpoints"),
            ok = filelib:ensure_path(Dir),
            {ok, _} = erllama_cache_disk_srv:start_link(Disk, Dir),
            unlink(whereis(Disk)),
            Load = #{
                backend => erllama_model_mlx,
                model_path => Path,
                tier => disk,
                tier_srv => Disk,
                context_opts => #{n_seq_max => 4},
                policy => #{min_tokens => 4}
            },
            {ok, Model} = erllama:load_model(Load),
            [{model, Model}, {load, Load}, {disk, Disk}, {checkpoint_dir, Dir} | Config]
    end.

end_per_suite(Config) ->
    _ = erllama:unload(proplists:get_value(model, Config)),
    ok = gen_server:stop(proplists:get_value(disk, Config)),
    file:del_dir_r(proplists:get_value(checkpoint_dir, Config)).

qwen_greedy(Config) ->
    M = proplists:get_value(model, Config),
    Prompt =
        <<"Explain in detail how the Erlang runtime schedules processes on a multicore machine.">>,
    Expected = [
        28763,
        279,
        12417,
        314,
        279,
        27867,
        11,
        279,
        27867,
        7087,
        11,
        321,
        279,
        27867,
        2767,
        49096,
        1021,
        5134,
        11,
        321,
        7276,
        1204,
        279,
        1324,
        314,
        5400,
        25943,
        369,
        10521,
        13,
        271,
        16
    ],
    Opts = #{temperature => 0.0, response_tokens => length(Expected)},
    {ok, #{generated := Expected, reply := Reply}} = erllama:complete(M, Prompt, Opts),
    ?assertEqual({ok, Reply}, erllama:detokenize(M, Expected)),
    ?assertEqual({error, not_supported}, erllama:complete(M, Prompt, Opts#{temperature => 1.0})),
    {ok, #{generated := Expected}} = erllama:complete(M, Prompt, Opts),
    Text = unicode:characters_to_binary([16#1F642, 16#4E2D, 16#E9]),
    {ok, Tokens} = erllama:tokenize(M, Text),
    {ok, Text} = erllama:detokenize(M, Tokens),
    Pieces = [
        begin
            {ok, B} = erllama:detokenize(M, [T]),
            B
        end
     || T <- Tokens
    ],
    ?assertEqual(Text, iolist_to_binary(Pieces)).

disk_resume(Config) ->
    Model = proplists:get_value(model, Config),
    Opts = #{response_tokens => 6, temperature => 0.0},
    Text = <<"An Erlang process receives messages in its">>,
    {ok, Prompt} = erllama:tokenize(Model, Text),
    Extended = <<Text/binary, " mailbox and">>,
    {ok, #{generated := Expected, cache_hit_kind := cold}} = erllama:complete(Model, Text, Opts),
    {ok, #{generated := ExpectedSuffix, cache_hit_kind := cold}} = erllama:complete(
        Model, Extended, Opts
    ),
    {ok, #{finish_key := Key}} = erllama:prefill_only(Model, Prompt),
    ?assert(is_binary(Key)),
    {ok, _} = erllama_cache_meta_srv:lookup_exact_or_wait(Key, 5000),
    ok = erllama:unload(Model),
    {ok, Model} = erllama:load_model(Model, proplists:get_value(load, Config)),
    {ok, #{generated := Expected, cache_hit_kind := exact}} = erllama:complete(Model, Text, Opts),
    {ok, #{generated := ExpectedSuffix, cache_delta := #{read := Read}}} =
        erllama:complete(Model, Extended, Opts#{parent_key => Key}),
    ?assertEqual(length(Prompt), Read).

worker_loss(Config) ->
    Model = proplists:get_value(model, Config),
    {ok, Pid} = erllama:whereis(Model),
    Monitor = erlang:monitor(process, Pid),
    [Port] = [P || P <- erlang:ports(), erlang:port_info(P, connected) =:= {connected, Pid}],
    {ok, Ref} = erllama:stream(
        Model,
        <<"List the numbers from one to one hundred.">>,
        #{temperature => 0, response_tokens => 128}
    ),
    receive
        {erllama, Ref, {token, _}} -> ok
    after 15000 -> ct:fail(no_first_token)
    end,
    true = erlang:port_close(Port),
    receive
        {erllama, Ref, {error, _}} -> ok
    after 5000 -> ct:fail(no_worker_error)
    end,
    receive
        {'DOWN', Monitor, process, Pid, _} -> ok
    after 5000 -> ct:fail(model_still_alive)
    end,
    {ok, Model} = erllama:load_model(Model, proplists:get_value(load, Config)),
    {ok, #{generated := [_]}} = erllama:complete(Model, <<"Testing recovery">>, #{
        response_tokens => 1
    }).
