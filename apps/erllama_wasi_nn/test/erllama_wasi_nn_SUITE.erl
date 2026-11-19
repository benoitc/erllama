%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% A component that imports `wasi:nn` runs against `erllama_wasi_nn`,
%% with models loaded in erllama on `erllama_model_stub`.
%%
%% The two guests in `test/fixtures/` are copied unchanged from
%% erlang_wasm (`test/fixtures/component/`) at commit
%% `cc56b56676537577806395bfd4969fdb0884a807`, where their Rust sources
%% live. `wasinn.component.wasm` is a library guest whose `run` export
%% loads a model by name, runs it on the input and answers every output.
%% `nncommand.component.wasm` is a `wasi:cli/command` that runs the
%% model named `echo` on stdin and writes the first output to stdout.
%%
%% The first cases are those of erlang_wasm's `wasm_wasi2_nn_SUITE` that
%% a backend answers for; the cases about the tensor bound, forged
%% handles and `load` test the host and stay there. The rest is what
%% only this backend has: the two tasks, `request_opts`, a model that
%% cannot embed, a caller that dies, and two callers at once.
%%
%% The stub draws its tokens from a sampler reference that is new for
%% every request, so two runs of one prompt do not answer the same
%% bytes. A reply is checked by its shape instead: the stub answers one
%% number per token, so the number of tokens asked for can be counted.
%%
%% `a_real_model_answers_a_command` is skipped unless `LLAMA_TEST_MODEL`
%% names a GGUF file.
%% @end
-module(erllama_wasi_nn_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([
    all/0,
    init_per_suite/1,
    end_per_suite/1,
    init_per_testcase/2,
    end_per_testcase/2
]).

-export([
    a_model_runs_and_its_outputs_come_back/1,
    nothing_is_left_live_after_a_run/1,
    a_name_outside_the_grant_is_not_found/1,
    the_backend_may_refuse_a_granted_name/1,
    a_backend_error_reaches_the_guest/1,
    a_command_runs_a_model/1,
    without_the_option_the_import_is_missing/1,
    an_embedding_comes_back_as_fp32/1,
    request_opts_are_honoured/1,
    a_model_that_cannot_embed_is_an_error_the_guest_reads/1,
    the_model_serves_again_after_its_caller_dies/1,
    two_instances_share_a_model/1,
    a_real_model_answers_a_command/1
]).

-define(OUTPUT,
    {record, [
        {<<"name">>, string},
        {<<"dimensions">>, {list, u32}},
        {<<"ty">>, string},
        {<<"data">>, {list, u8}}
    ]}
).

-define(MODEL, <<"wasi_nn_stub">>).
-define(SLOW, <<"wasi_nn_slow">>).
-define(NO_EMBED, <<"wasi_nn_noembed">>).
-define(REAL, <<"wasi_nn_real">>).
-define(MODEL_ENV, "LLAMA_TEST_MODEL").
-define(PROMPT, <<"the quick brown fox">>).
-define(OPTS, #{response_tokens => 4}).

all() ->
    [
        a_model_runs_and_its_outputs_come_back,
        nothing_is_left_live_after_a_run,
        a_name_outside_the_grant_is_not_found,
        the_backend_may_refuse_a_granted_name,
        a_backend_error_reaches_the_guest,
        a_command_runs_a_model,
        without_the_option_the_import_is_missing,
        an_embedding_comes_back_as_fp32,
        request_opts_are_honoured,
        a_model_that_cannot_embed_is_an_error_the_guest_reads,
        the_model_serves_again_after_its_caller_dies,
        two_instances_share_a_model,
        a_real_model_answers_a_command
    ].

init_per_suite(Config) ->
    {ok, _} = application:ensure_all_started(erllama_wasi_nn),
    {ok, Bin} = file:read_file(fixture_path("wasinn")),
    {ok, Command} = file:read_file(fixture_path("nncommand")),
    [{component, Bin}, {command, Command} | Config].

end_per_suite(_Config) ->
    _ = application:stop(erllama_wasi_nn),
    _ = application:stop(erllama),
    ok.

%% Every case runs in a process of its own, so the host table of
%% erlang_wasm, which lives in the process dictionary, starts empty.
init_per_testcase(a_real_model_answers_a_command, Config) ->
    case os:getenv(?MODEL_ENV) of
        false ->
            {skip, "LLAMA_TEST_MODEL not set"};
        Path ->
            case filelib:is_regular(Path) of
                true -> [{model_path, Path} | Config];
                false -> {skip, "LLAMA_TEST_MODEL does not name a file"}
            end
    end;
init_per_testcase(_Case, Config) ->
    {ok, ?MODEL} = erllama:load_model(?MODEL, #{backend => erllama_model_stub}),
    Config.

end_per_testcase(_Case, _Config) ->
    [_ = erllama:unload(M) || M <- [?MODEL, ?SLOW, ?NO_EMBED, ?REAL]],
    ok.

%% =============================================================================
%% Ported from wasm_wasi2_nn_SUITE
%% =============================================================================

a_model_runs_and_its_outputs_come_back(Config) ->
    {ok, #{generated := Generated}} = erllama:complete(?MODEL, ?PROMPT, ?OPTS),
    I = instance(Config),
    {ok, [Output]} = run(I, <<"default">>, ?PROMPT),
    Data = maps:get(<<"data">>, Output),
    ?assertEqual(
        #{
            <<"name">> => <<"output">>,
            <<"dimensions">> => [byte_size(Data)],
            <<"ty">> => <<"U8">>,
            <<"data">> => Data
        },
        Output
    ),
    ?assertEqual(length(Generated), tokens(Data)),
    ?assertEqual(4, tokens(Data)).

nothing_is_left_live_after_a_run(Config) ->
    I = instance(Config),
    {ok, _} = run(I, <<"default">>, ?PROMPT),
    ?assertEqual([], wasm_component:host_live()),
    ?assertEqual(
        {error, <<"not-found: not_loaded">>},
        run(I, <<"unloaded">>, ?PROMPT)
    ),
    ?assertEqual([], wasm_component:host_live()).

%% `secret` has an entry and a loaded model, and is not in `models`.
a_name_outside_the_grant_is_not_found(Config) ->
    I = instance(Config),
    ?assertEqual(
        {error, <<"not-found: no such model">>},
        run(I, <<"secret">>, ?PROMPT)
    ).

the_backend_may_refuse_a_granted_name(Config) ->
    I = instance(Config),
    ?assertEqual(
        {error, <<"not-found: no such model">>},
        run(I, <<"absent">>, ?PROMPT)
    ),
    ?assertEqual(
        {error, <<"not-found: not_loaded">>},
        run(I, <<"unloaded">>, ?PROMPT)
    ),
    %% The refusal comes when the graph is loaded, before any tensor.
    ?assertEqual(
        {error, {not_found, <<"not_loaded">>}},
        erllama_wasi_nn:load_by_name(<<"unloaded">>, maps:get(backend_opts, grant()))
    ).

%% The model runs one request at a time, and `full` asks not to wait
%% for a free sequence: while another request holds it, erllama answers
%% `seq_capacity`.
a_backend_error_reaches_the_guest(Config) ->
    {ok, ?SLOW} = erllama:load_model(?SLOW, #{
        backend => erllama_model_stub, step_delay_ms => 50
    }),
    Holder = spawn_link(fun() ->
        erllama:complete(?SLOW, ?PROMPT, #{response_tokens => 64})
    end),
    ok = wait_for_phase(?SLOW, generating, 5000),
    I = instance(Config),
    ?assertEqual(
        {error, <<"runtime-error: seq_capacity">>},
        run(I, <<"full">>, ?PROMPT)
    ),
    ?assert(is_process_alive(Holder)),
    ?assertEqual([], wasm_component:host_live()).

a_command_runs_a_model(Config) ->
    {ok, #{stdout := Out, exit_code := 0}} =
        wasi_preview2:run_command(?config(command, Config), ?PROMPT, #{nn => grant()}),
    ?assertEqual(4, tokens(Out)),
    ?assertEqual([], wasm_component:host_live()).

without_the_option_the_import_is_missing(Config) ->
    ?assertMatch(
        {error, _},
        wasm_component:instantiate(
            ?config(component, Config),
            wasi_preview2:command(#{}),
            #{}
        )
    ).

%% =============================================================================
%% Only this backend
%% =============================================================================

an_embedding_comes_back_as_fp32(Config) ->
    {ok, Vector} = erllama:embed(?MODEL, ?PROMPT),
    N = length(Vector),
    ?assert(N > 0),
    I = instance(Config),
    {ok, [Output]} = run(I, <<"embed">>, ?PROMPT),
    ?assertMatch(
        #{<<"name">> := <<"embedding">>, <<"ty">> := <<"FP32">>, <<"dimensions">> := [N]},
        Output
    ),
    Data = maps:get(<<"data">>, Output),
    ?assertEqual(4 * N, byte_size(Data)),
    ?assertEqual([as_fp32(F) || F <- Vector], [F || <<F:32/float-little>> <= Data]),
    %% The stub's floats are not all exact in 32 bits, so an encoding
    %% that kept 64 bits or another byte order would not pass above.
    ?assert(lists:any(fun(F) -> as_fp32(F) =/= F end, Vector)).

request_opts_are_honoured(Config) ->
    I = instance(Config),
    {ok, [#{<<"data">> := Short}]} = run(I, <<"one">>, ?PROMPT),
    {ok, [#{<<"data">> := Long}]} = run(I, <<"eight">>, ?PROMPT),
    ?assertEqual(1, tokens(Short)),
    ?assertEqual(8, tokens(Long)),
    ?assert(byte_size(Short) < byte_size(Long)).

a_model_that_cannot_embed_is_an_error_the_guest_reads(Config) ->
    %% Only the stub loads without a model file: any file that exists
    %% passes the check, and this backend never opens it.
    {ok, ?NO_EMBED} = erllama:load_model(?NO_EMBED, #{
        backend => erllama_wasi_nn_noembed_backend,
        model_path => fixture_path("wasinn")
    }),
    I = instance(Config),
    ?assertEqual(
        {error, <<"unsupported-operation: not_supported">>},
        run(I, <<"no-embed">>, ?PROMPT)
    ),
    ?assertEqual([], wasm_component:host_live()).

%% An embedder bounds `compute` by running the instance in a worker and
%% killing the worker. The model finishes the request nobody waits for
%% and serves the next one.
the_model_serves_again_after_its_caller_dies(Config) ->
    {ok, ?SLOW} = erllama:load_model(?SLOW, #{
        backend => erllama_model_stub, step_delay_ms => 50
    }),
    Component = ?config(component, Config),
    Self = self(),
    {Pid, Ref} = spawn_monitor(fun() ->
        I = instance(Component),
        Self ! {answered, run(I, <<"slow">>, ?PROMPT)}
    end),
    ok = wait_for_phase(?SLOW, generating, 5000),
    exit(Pid, kill),
    receive
        {'DOWN', Ref, process, Pid, killed} -> ok
    after 5000 -> ct:fail(the_caller_did_not_die)
    end,
    receive
        {answered, _} -> ct:fail(the_caller_answered_before_it_was_killed)
    after 0 -> ok
    end,
    ok = wait_for_phase(?SLOW, idle, 10000),
    I2 = instance(Config),
    ?assertMatch({ok, [#{<<"name">> := <<"output">>}]}, run(I2, <<"slow">>, ?PROMPT)).

%% Each process has its own instance and its own host table; the model
%% runs one request at a time and the second waits its turn.
two_instances_share_a_model(Config) ->
    {ok, ?SLOW} = erllama:load_model(?SLOW, #{
        backend => erllama_model_stub, step_delay_ms => 10
    }),
    Component = ?config(component, Config),
    Self = self(),
    Pids = [
        spawn_link(fun() ->
            I = instance(Component),
            Self ! {self(), run(I, <<"slow">>, ?PROMPT), wasm_component:host_live()}
        end)
     || _ <- [1, 2]
    ],
    [
        receive
            {Pid, Result, Live} ->
                {ok, [#{<<"name">> := <<"output">>, <<"data">> := Data}]} = Result,
                ?assertEqual(4, tokens(Data)),
                ?assertEqual([], Live)
        after 10000 -> ct:fail(no_answer)
        end
     || Pid <- Pids
    ].

%% =============================================================================
%% A live run
%% =============================================================================

a_real_model_answers_a_command(Config) ->
    {ok, ?REAL} = erllama:load_model(?REAL, #{
        backend => erllama_model_llama,
        model_path => ?config(model_path, Config),
        context_opts => #{n_ctx => 512, n_batch => 64}
    }),
    Grant = #{
        backend => erllama_wasi_nn,
        models => [<<"echo">>],
        backend_opts => #{
            <<"echo">> => #{
                model => ?REAL,
                task => completion,
                request_opts => #{response_tokens => 16}
            }
        }
    },
    {ok, #{stdout := Out, exit_code := 0}} =
        wasi_preview2:run_command(?config(command, Config), <<"Once upon a time">>, #{
            nn => Grant
        }),
    ct:pal("reply: ~ts", [Out]),
    ?assert(byte_size(Out) > 0),
    ?assertEqual(Out, unicode:characters_to_binary(Out, utf8, utf8)).

%% =============================================================================
%% Helpers
%% =============================================================================

%% `secret` and `unloaded` have entries; `secret` is not granted, and
%% `absent` is granted without an entry.
grant() ->
    Completion = #{model => ?MODEL, task => completion, request_opts => ?OPTS},
    #{
        backend => erllama_wasi_nn,
        models => [
            <<"default">>,
            <<"echo">>,
            <<"embed">>,
            <<"one">>,
            <<"eight">>,
            <<"slow">>,
            <<"full">>,
            <<"no-embed">>,
            <<"absent">>,
            <<"unloaded">>
        ],
        backend_opts => #{
            <<"default">> => Completion,
            <<"echo">> => Completion,
            <<"secret">> => Completion,
            <<"embed">> => #{model => ?MODEL, task => embedding},
            <<"one">> => Completion#{request_opts := #{response_tokens => 1}},
            <<"eight">> => Completion#{request_opts := #{response_tokens => 8}},
            <<"slow">> => Completion#{model := ?SLOW},
            <<"full">> => Completion#{model := ?SLOW, request_opts := #{on_full => error}},
            <<"no-embed">> => #{model => ?NO_EMBED, task => embedding},
            <<"unloaded">> => Completion#{model := <<"wasi_nn_never_loaded">>}
        }
    }.

instance(Component) when is_binary(Component) ->
    {ok, I} = wasm_component:instantiate(
        Component,
        wasi_preview2:command(#{nn => grant()}),
        #{resource_closer => fun wasi_preview2:close_resource/1}
    ),
    I;
instance(Config) ->
    instance(?config(component, Config)).

run(I, Model, Input) ->
    {ok, Result} = call(
        I,
        <<"run">>,
        {[string, {list, u8}], {result, {list, ?OUTPUT}, string}},
        [Model, Input]
    ),
    Result.

call(I, Name, Sig, Args) ->
    wasm_component:call(I, Name, Sig, Args).

%% The stub detokenises to one number per token, space separated.
tokens(Reply) ->
    length(binary:split(Reply, <<" ">>, [global, trim_all])).

as_fp32(F) ->
    <<F32:32/float-little>> = <<F:32/float-little>>,
    F32.

wait_for_phase(Model, Phase, Timeout) when Timeout =< 0 ->
    {timeout, Phase, erllama:phase(Model)};
wait_for_phase(Model, Phase, Timeout) ->
    case erllama:phase(Model) of
        {ok, Phase} ->
            ok;
        _ ->
            timer:sleep(10),
            wait_for_phase(Model, Phase, Timeout - 10)
    end.

fixture_path(Name) ->
    filename:join([
        code:lib_dir(erllama_wasi_nn), "test", "fixtures", Name ++ ".component.wasm"
    ]).
