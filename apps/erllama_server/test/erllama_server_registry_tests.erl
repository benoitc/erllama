%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_registry_tests).
-include_lib("eunit/include/eunit.hrl").

-define(R, erllama_server_registry).

%% -----------------------------------------------------------------------------
%% GGUF reader
%% -----------------------------------------------------------------------------

tmp_dir() ->
    Dir = filename:join([
        os:getenv("TMPDIR", "/tmp"),
        "erllama_server_registry_tests",
        os:getpid() ++ "-" ++ integer_to_list(erlang:unique_integer([positive]))
    ]),
    ok = filelib:ensure_path(Dir),
    Dir.

gguf_file(Dir, Name) ->
    Path = filename:join(Dir, Name),
    ok = file:write_file(Path, erllama_server_fake_registry:gguf(4096)),
    Path.

gguf_reads_header_test() ->
    Path = gguf_file(tmp_dir(), "m.gguf"),
    {ok, #{version := 3, tensors := 0, kv := KV} = H} = erllama_server_gguf:read(Path),
    ?assertEqual(#{<<"general.architecture">> => <<"llama">>}, KV),
    ?assertMatch(
        #{architecture := <<"llama">>, projector := false}, erllama_server_gguf:summary(H)
    ).

gguf_refuses_other_files_test() ->
    Dir = tmp_dir(),
    Bad = filename:join(Dir, "bad.gguf"),
    ok = file:write_file(Bad, <<"not a gguf file at all">>),
    ?assertEqual({error, not_gguf}, erllama_server_gguf:read(Bad)),
    Short = filename:join(Dir, "short.gguf"),
    ok = file:write_file(Short, <<"GGUF", 3:32/little, 0:64/little, 5:64/little>>),
    ?assertEqual({error, truncated}, erllama_server_gguf:read(Short)),
    ?assertEqual({error, enoent}, erllama_server_gguf:read(filename:join(Dir, "none.gguf"))).

%% A real model's header, when one is available.
gguf_real_model_test() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            ok;
        Path ->
            {ok, H} = erllama_server_gguf:read(Path),
            #{architecture := Arch, context_length := Ctx} = erllama_server_gguf:summary(H),
            ?assert(is_binary(Arch)),
            ?assert(is_integer(Ctx) andalso Ctx > 0)
    end.

%% -----------------------------------------------------------------------------
%% Sources
%% -----------------------------------------------------------------------------

sources_test() ->
    P = fun erllama_server_source:parse/1,
    ?assertMatch(
        {ok, #{kind := hf, repo := <<"org/repo">>, quant := undefined}}, P(<<"hf://org/repo">>)
    ),
    ?assertMatch(
        {ok, #{kind := hf, repo := <<"org/repo">>, quant := <<"q4_k_m">>}},
        P(<<"hf://org/repo:Q4_K_M">>)
    ),
    ?assertMatch(
        {ok, #{kind := hf, repo := <<"org/repo">>, file := <<"sub/m.gguf">>}},
        P(<<"hf://org/repo/sub/m.gguf">>)
    ),
    ?assertMatch(
        {ok, #{kind := https, url := <<"https://x.example/m.gguf">>}}, P("https://x.example/m.gguf")
    ),
    ?assertMatch(
        {ok, #{kind := ollama, namespace := <<"library">>, name := <<"llama3.2">>, tag := <<"1b">>}},
        P(<<"llama3.2:1b">>)
    ),
    ?assertMatch(
        {ok, #{kind := ollama, namespace := <<"user">>, name := <<"model">>, tag := <<"latest">>}},
        P(<<"ollama://user/model">>)
    ),
    [
        ?assertMatch({error, {invalid_source, _}}, P(B))
     || B <- [<<"hf://">>, <<"hf://org">>, <<"hf://org/repo/readme.md">>, <<"a/b/c">>]
    ].

origins_test() ->
    O = fun erllama_server_source:origin/1,
    ?assertEqual({<<"https">>, <<"huggingface.co">>, 443}, O(<<"https://huggingface.co/x">>)),
    ?assertNotEqual(O(<<"http://127.0.0.1:80/a">>), O(<<"http://localhost:80/a">>)),
    ?assertNotEqual(O(<<"http://127.0.0.1:8080/a">>), O(<<"http://127.0.0.1:8081/a">>)).

%% -----------------------------------------------------------------------------
%% Registry
%% -----------------------------------------------------------------------------

with_registry(Fun) ->
    Dir = tmp_dir(),
    ok = application:set_env(erllama_server, data_dir, Dir),
    {ok, Pid} = ?R:start_link(),
    unlink(Pid),
    try
        Fun(Dir)
    after
        %% By name: a test may have restarted it.
        gen_server:stop(?R),
        application:unset_env(erllama_server, data_dir),
        file:del_dir_r(Dir)
    end.

restart() ->
    gen_server:stop(?R),
    {ok, Pid} = ?R:start_link(),
    unlink(Pid).

register_resolve_and_persist_test() ->
    with_registry(fun(Dir) ->
        Path = gguf_file(Dir, "Tiny.gguf"),
        {ok, #{name := <<"tiny:latest">>, status := ok}} = ?R:register_local(Path, #{n_ctx => 1024}),
        ?assertMatch({ok, #{name := <<"tiny:latest">>}}, ?R:resolve(<<"tiny">>)),
        ?assertMatch({ok, #{name := <<"tiny:latest">>}}, ?R:resolve(<<"TINY:latest">>)),
        ok = ?R:alias(<<"t">>, <<"tiny">>),
        {ok, _} = ?R:copy(<<"tiny">>, <<"tiny:two">>),
        ok = ?R:set_default(<<"tiny:two">>),
        restart(),
        ?assertMatch(
            {ok, #{name := <<"tiny:latest">>, load := #{n_ctx := 1024}}}, ?R:resolve(<<"t">>)
        ),
        ?assertMatch({ok, #{name := <<"tiny:two">>, default := true}}, ?R:resolve(undefined)),
        ?assertEqual([<<"tiny:latest">>, <<"tiny:two">>], [N || #{name := N} <- ?R:list()])
    end).

mlx_directory_persists_test() ->
    with_registry(fun(Dir) ->
        Path = filename:join(Dir, "qwen"),
        ok = file:make_dir(Path),
        ?assertMatch({error, {invalid_mlx_checkpoint, _}}, ?R:register_local(Path, #{})),
        [
            ok = file:write_file(filename:join(Path, F), <<"{}">>)
         || F <- ["tokenizer_config.json", "config.json", "tokenizer.json", "model.safetensors"]
        ],
        {ok, _} = ?R:register_local(Path, #{parallel => 4}),
        restart(),
        ?assertMatch(
            {ok, #{status := ok, size := 2, load := #{backend := erllama_model_mlx, parallel := 4}}},
            ?R:resolve(<<"qwen">>)
        )
    end).

unknown_and_default_test() ->
    with_registry(fun(_Dir) ->
        ?assertEqual({error, {unknown_model, <<"nope">>}}, ?R:resolve(<<"nope">>)),
        ?assertEqual({error, no_default_model}, ?R:resolve(undefined))
    end).

missing_file_is_reported_test() ->
    with_registry(fun(Dir) ->
        Path = gguf_file(Dir, "gone.gguf"),
        {ok, _} = ?R:register_local(Path, #{}),
        ok = file:delete(Path),
        restart(),
        ?assertMatch({ok, #{status := missing}}, ?R:resolve(<<"gone">>))
    end).

local_files_are_never_deleted_test() ->
    with_registry(fun(Dir) ->
        Path = gguf_file(Dir, "keep.gguf"),
        {ok, _} = ?R:register_local(Path, #{}),
        ok = ?R:remove(<<"keep">>),
        ?assert(filelib:is_regular(Path)),
        ?assertEqual([], ?R:list())
    end).

register_refuses_non_gguf_test() ->
    with_registry(fun(Dir) ->
        Bad = filename:join(Dir, "x.gguf"),
        ok = file:write_file(Bad, <<"nope">>),
        ?assertMatch({error, {not_a_gguf, _, not_gguf}}, ?R:register_local(Bad, #{})),
        ?assertMatch({error, {enoent, _}}, ?R:register_local(filename:join(Dir, "none.gguf"), #{})),
        Model = gguf_file(Dir, "m.gguf"),
        ?assertMatch({error, {not_a_projector, _}}, ?R:register_local(Model, #{mmproj => Model}))
    end).

corrupt_registry_is_moved_aside_test() ->
    with_registry(fun(Dir) ->
        Path = gguf_file(Dir, "a.gguf"),
        {ok, _} = ?R:register_local(Path, #{}),
        gen_server:stop(?R),
        File = filename:join(Dir, "registry.json"),
        ok = file:write_file(File, <<"{\"version\": 1, \"models\": [tru">>),
        {ok, Pid} = ?R:start_link(),
        unlink(Pid),
        ?assertEqual([], ?R:list()),
        ?assertMatch([_], filelib:wildcard(filename:join(Dir, "registry.json.corrupt-*")))
    end).
