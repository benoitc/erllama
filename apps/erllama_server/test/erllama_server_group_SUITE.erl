%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
-module(erllama_server_group_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").
-behaviour(gen_server).

-export([
    all/0,
    serve_and_lose_rank/1,
    exercise/4,
    start_link/1,
    load/3,
    init/1,
    handle_call/3,
    handle_cast/2
]).

all() -> [serve_and_lose_rank].

serve_and_lose_rank(Config) ->
    Dir = ?config(priv_dir, Config),
    Path = filename:join(Dir, "vocab.gguf"),
    ok = file:write_file(Path, vocab()),
    with_peer("serve", fun(S, _) ->
        with_peer("leader", fun(_, N0) ->
            with_peer("follower", fun(_, N1) ->
                ok = peer:call(S, ?MODULE, exercise, [Dir, Path, N0, N1], 45000)
            end)
        end)
    end).

with_peer(Name, Fun) ->
    LibRoot = filename:dirname(code:lib_dir(erllama)),
    Paths = filelib:wildcard(filename:join([LibRoot, "*", "ebin"])),
    {ok, Peer, Node} = peer:start(#{
        name => peer:random_name(Name),
        host => "127.0.0.1",
        longnames => true,
        connection => standard_io,
        args => ["+S", "2", "-pa", filename:dirname(code:which(?MODULE)) | Paths]
    }),
    try
        Fun(Peer, Node)
    after
        peer:stop(Peer)
    end.

exercise(Dir, Path, N0, N1) ->
    ok = application:set_env(erllama_server, data_dir, Dir),
    {ok, _} = application:ensure_all_started(erllama_server),
    Group = #{
        group_id => <<"g">>,
        generation => 1,
        world => 2,
        coordinator => <<"tp_test">>,
        followers => [atom_to_binary(N1)]
    },
    Id = <<"split:latest">>,
    Register = fun(G) -> erllama_server_api:register_model(Path, #{name => Id, group => G}) end,
    {error, invalid_group} = Register(Group#{coordinator := self()}),
    {ok, #{group := Group}} = Register(Group),
    %% Exercise the persisted entry through a registry restart.
    ok = application:stop(erllama_server),
    {ok, _} = application:ensure_all_started(erllama_server),
    {ok, #{group := Group}} = erllama_server_api:model(Id),
    Spec = #{model => Id, prompt => <<"x">>, opts => #{response_tokens => 3}},
    ?assertEqual({error, group_forming}, erllama_server_models:load(Id, infinity)),
    {ok, Id} = erllama:load_model(Id, #{backend => erllama_model_stub}),
    false = erllama_server_models:group_loaded(Id, Group),
    ok = erllama:unload(Id),
    C = form(1, Path, N0, N1),
    buffered(Spec),
    true = erllama_server_models:group_loaded(Id, Group),
    false = erllama_server_models:group_loaded(Id, Group#{generation := 2}),
    {ok, #{run_id := Run}} = erllama_server_api:start_run(Spec#{opts := #{response_tokens => 128}}),
    {ok, #{replay := Replay}} = erllama_server_api:attach(Run, self(), 0),
    case lists:any(fun(#{type := T}) -> T =:= token end, Replay) of
        true ->
            ok;
        false ->
            receive
                {erllama_server_run, Run, #{type := token}} -> ok
            after 5000 -> error(no_token)
            end
    end,
    exit(erpc:call(N1, erlang, whereis, [?MODULE]), kill),
    receive
        {erllama_server_run, Run, #{type := error, data := #{error := {tp_rank_lost, _}}}} -> ok
    after 5000 -> error({no_loss_event, erllama_server_api:result(Run)})
    end,
    {ok, #{replay := Events}} = erllama_server_api:attach(Run, self(), 0),
    ?assertEqual([error], [T || #{type := T} <- Events, lists:member(T, [done, error, cancelled])]),
    wait(fun() -> erllama:whereis(Id) =:= {error, not_loaded} end),
    {ok, #{status := lost, loaded := false}} = erllama_server_api:model(Id),
    {ok, #{status := lost}} = Register(Group),
    ?assertMatch({error, {tp_rank_lost, _}}, erllama_server_api:start_run(Spec)),
    {ok, Json} = file:read_file(filename:join(Dir, "registry.json")),
    [Stored] = maps:get(<<"models">>, json:decode(Json)),
    ?assertNot(maps:is_key(<<"status">>, Stored)),
    ?assertNot(maps:is_key(<<"reason">>, Stored)),
    [ok = erpc:call(N, application, stop, [erllama]) || N <- [N0, N1]],
    ok = erllama_tp_coordinator:stop(C),
    _ = form(2, Path, N0, N1),
    {ok, #{status := ok}} = Register(Group#{generation := 2}),
    buffered(Spec),
    exit(erpc:call(N1, erlang, whereis, [?MODULE]), kill),
    wait(fun() ->
        {ok, Model} = erllama_server_api:model(Id),
        maps:get(status, Model) =:= lost andalso not maps:get(loaded, Model)
    end).

form(Gen, Path, N0, N1) ->
    {ok, C} = erllama_tp_coordinator:start(tp_test, {<<"g">>, Gen}, 2, 5000),
    [
        begin
            Config = #{
                rank => R,
                world => 2,
                group_id => <<"g">>,
                generation => Gen,
                coordinator => {tp_test, node()},
                model_path => Path,
                context_opts => #{n_ctx => 256}
            },
            ok = erpc:call(N, application, set_env, [erllama, tp_rank_driver, ?MODULE]),
            ok = erpc:call(N, application, set_env, [erllama, tp_rank, Config]),
            {ok, _} = erpc:call(N, application, ensure_all_started, [erllama])
        end
     || {R, N} <- [{0, N0}, {1, N1}]
    ],
    wait(fun() ->
        lists:all(
            fun(N) ->
                maps:get(state, erpc:call(N, erllama_tp_rank_role, status, [])) =:= ready
            end,
            [N0, N1]
        )
    end),
    C.

buffered(Spec) ->
    {ok, #{run_id := Run}} = erllama_server_api:start_run(Spec),
    wait(fun() ->
        case erllama_server_api:result(Run) of
            {ok, #{status := done}} -> true;
            {ok, #{status := error} = E} -> error(E);
            _ -> false
        end
    end),
    {ok, #{status := done, content := <<"xxx">>}} = erllama_server_api:result(Run),
    {ok, #{replay := Events}} = erllama_server_api:attach(Run, self(), 0),
    ?assertEqual(<<"xxx">>, iolist_to_binary([B || #{type := token, data := B} <- Events])).

wait(F) -> wait(F, 250).
wait(_, 0) ->
    error(timed_out);
wait(F, N) ->
    case F() of
        true ->
            ok;
        false ->
            timer:sleep(20),
            wait(F, N - 1)
    end.

%% Only native rank allocation and math are simulated. The serving VM
%% loads an actual GGUF vocabulary and uses the normal TP proxy.
start_link(C) -> gen_server:start_link({local, ?MODULE}, ?MODULE, C, []).
load(P, Path, Opts) -> gen_server:call(P, {load, Path, Opts}).
init(C) -> {ok, C}.
handle_call({load, _, _}, _, C) ->
    {reply, ok, C};
handle_call({hosting, {host, _, [{_, _}]}}, _, #{rank := 0} = C) ->
    {reply, ok, C};
handle_call({hosting, {backend, extra_metadata, []}}, _, C) ->
    {reply, #{}, C};
handle_call({hosting, {backend, sampler_new, [_]}}, _, C) ->
    {reply, {ok, {erllama_tp_ref, self(), 1}}, C};
handle_call({hosting, {backend, step, [Ops]}}, _, C) ->
    timer:sleep(20),
    {reply, {ok, [step(Op) || Op <- Ops]}, C};
handle_call({hosting, _}, _, C) ->
    {reply, ok, C}.
handle_cast(_, C) -> {noreply, C}.

step({Seq, {prefill, _}}) -> {Seq, prefilled};
step({Seq, {decode, _}}) -> {Seq, {token, 3, 0}}.

vocab() ->
    Tokens = [
        <<"<unk>">>,
        <<"<s>">>,
        <<"</s>">>,
        <<"x">>
        | [iolist_to_binary(io_lib:format("<0x~2.16.0B>", [N])) || N <- lists:seq(0, 255)]
    ],
    Fields = [
        {<<"general.architecture">>, 8, str(<<"llama">>)},
        {<<"tokenizer.ggml.model">>, 8, str(<<"llama">>)},
        {<<"tokenizer.ggml.tokens">>, 9, [<<8:32/little, 260:64/little>>, [str(T) || T <- Tokens]]},
        {<<"tokenizer.ggml.token_type">>, 9, [
            <<5:32/little, 260:64/little>>,
            [<<T:32/little>> || T <- [2, 3, 3, 1 | lists:duplicate(256, 6)]]
        ]}
    ],
    Bytes = iolist_to_binary([
        <<"GGUF", 3:32/little, 0:64/little, (length(Fields)):64/little>>,
        [[str(K), <<T:32/little>>, V] || {K, T, V} <- Fields]
    ]),
    [Bytes, binary:copy(<<0>>, (32 - byte_size(Bytes) rem 32) rem 32)].
str(B) -> [<<(byte_size(B)):64/little>>, B].
