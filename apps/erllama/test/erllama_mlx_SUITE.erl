%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
-module(erllama_mlx_SUITE).
-include_lib("stdlib/include/assert.hrl").
-export([all/0, init_per_suite/1, end_per_suite/1, qwen_greedy/1]).

all() -> [qwen_greedy].

init_per_suite(Config) ->
    case os:getenv("ERLLAMA_MLX_TEST_MODEL") of
        false ->
            {skip, "Set ERLLAMA_MLX_TEST_MODEL to the reference Qwen3.8-27B MLX directory"};
        Path ->
            {ok, _} = application:ensure_all_started(erllama),
            {ok, Model} = erllama:load_model(#{backend => erllama_model_mlx, model_path => Path}),
            [{model, Model} | Config]
    end.

end_per_suite(Config) ->
    erllama:unload(proplists:get_value(model, Config)).

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
