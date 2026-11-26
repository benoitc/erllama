%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% Model downloads against a local stand-in for Hugging Face, its CDN
%% and the Ollama registry (erllama_server_fake_registry): each source
%% kind, progress events, resuming a broken transfer, a server that
%% ignores ranges, integrity checks, cancel then resume, the token kept
%% away from the CDN, and the registry surviving a restart.
%% @end
-module(erllama_server_pull_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_suite/1, end_per_suite/1, init_per_testcase/2, end_per_testcase/2]).
-export([
    hf_repo_with_projector/1,
    hf_single_file/1,
    split_set/1,
    hf_unknown_quant/1,
    ollama_model_with_params/1,
    https_with_checksum/1,
    https_wrong_checksum/1,
    broken_transfer_resumes/1,
    dropped_connection_resumes/1,
    ignored_range_restarts/1,
    corrupt_blob_is_rejected/1,
    cancel_then_resume/1,
    registry_survives_restart/1,
    remove_deletes_unused_blob/1,
    reused_blob_is_verified/1
]).

all() ->
    [
        hf_repo_with_projector,
        hf_single_file,
        split_set,
        hf_unknown_quant,
        ollama_model_with_params,
        https_with_checksum,
        https_wrong_checksum,
        broken_transfer_resumes,
        dropped_connection_resumes,
        ignored_range_restarts,
        corrupt_blob_is_rejected,
        cancel_then_resume,
        registry_survives_restart,
        remove_deletes_unused_blob,
        reused_blob_is_verified
    ].

init_per_suite(Config) ->
    {ok, _} = application:ensure_all_started(livery),
    {ok, Fake} = erllama_server_fake_registry:start(),
    [{fake, Fake} | Config].

end_per_suite(Config) ->
    erllama_server_fake_registry:stop(?config(fake, Config)),
    ok.

init_per_testcase(TC, Config) ->
    Dir = filename:join(?config(priv_dir, Config), atom_to_list(TC)),
    #{base := Base} = ?config(fake, Config),
    Env = [
        {data_dir, Dir},
        {hf_base_url, Base},
        {ollama_base_url, Base},
        {run_retention_ms, 60000},
        {pull_idle_timeout_ms, 1500}
    ],
    [ok = application:set_env(erllama_server, K, V) || {K, V} <- Env],
    reset_fake(),
    {ok, _} = application:ensure_all_started(erllama_server),
    [{data_dir, Dir} | Config].

end_per_testcase(_TC, _Config) ->
    _ = application:stop(erllama_server),
    [
        application:unset_env(erllama_server, K)
     || K <- [
            data_dir, hf_base_url, ollama_base_url, hf_token, run_retention_ms, pull_idle_timeout_ms
        ]
    ],
    ok.

reset_fake() ->
    Tab = erllama_server_fake_registry,
    ets:match_delete(Tab, {{opt, '_'}, '_'}),
    ets:match_delete(Tab, {{interrupted, '_'}, '_'}),
    true = ets:insert(Tab, {seen, []}).

%% Pull and follow to the end; returns the terminal event and all events.
pull(Source) ->
    pull(Source, #{}).

pull(Source, Opts) ->
    {ok, #{job_id := Id}} = erllama_server_api:pull(Source, Opts),
    follow(Id).

follow(Id) ->
    {ok, #{replay := Replay, status := Status}} = erllama_server_api:attach(Id, self(), 0),
    collect(Id, Replay, Status).

collect(_Id, Events, Status) when Status =/= running ->
    {lists:last(Events), Events};
collect(Id, Events, running) ->
    case [E || #{type := T} = E <- Events, lists:member(T, [done, error, cancelled])] of
        [T | _] -> {T, Events};
        [] -> collect_more(Id, Events)
    end.

collect_more(Id, Acc) ->
    receive
        {erllama_server_run, Id, #{type := T} = E} when T =:= done; T =:= error; T =:= cancelled ->
            {E, Acc ++ [E]};
        {erllama_server_run, Id, #{seq := Seq} = E} ->
            ack(Id, Seq),
            collect_more(Id, Acc ++ [E])
    after 30000 -> ct:fail({pull_timeout, Acc})
    end.

ack(Id, Seq) ->
    {ok, #{pid := Pid}} = erllama_server_runs:lookup(Id),
    Pid ! {ack, self(), Seq}.

types(Events) -> [T || #{type := T} <- Events].

file_sha(Path) ->
    {ok, Bin} = file:read_file(Path),
    erllama_server_fake_registry:sha(Bin).

blob_sha(Kind) -> erllama_server_fake_registry:sha(erllama_server_fake_registry:blob(Kind)).

%% -----------------------------------------------------------------------------

hf_repo_with_projector(_Config) ->
    ok = application:set_env(erllama_server, hf_token, <<"secret">>),
    ok = erllama_server_fake_registry:set(require_token, <<"secret">>),
    {#{type := done, data := #{model := M}}, Events} = pull(<<"hf://acme/tiny">>),
    ?assertEqual(<<"hf.co/acme/tiny:q4_k_m">>, maps:get(name, M)),
    ?assert(lists:member(progress, types(Events))),
    ?assertEqual(blob_sha(model), file_sha(maps:get(path, M))),
    ?assertEqual(blob_sha(model), maps:get(sha256, M)),
    ?assertEqual(blob_sha(projector), file_sha(maps:get(mmproj, M))),
    %% The token went to the hub, never to the CDN it redirected to.
    Seen = erllama_server_fake_registry:seen(),
    ?assert(
        lists:all(fun(#{authorization := A}) -> A =:= <<"Bearer secret">> end, [
            S
         || #{origin := hub} = S <- Seen
        ])
    ),
    CdnSeen = [S || #{origin := cdn} = S <- Seen],
    ?assertEqual(2, length(CdnSeen)),
    ?assert(lists:all(fun(#{authorization := A}) -> A =:= undefined end, CdnSeen)),
    {ok, [
        #{name := <<"hf.co/acme/tiny:q4_k_m">>, registered := true, status := ok, mmproj := true}
    ]} =
        erllama_server_api:models().

hf_single_file(_Config) ->
    {#{type := done, data := #{model := M}}, _} = pull(<<"hf://acme/tiny/tiny-model.Q8_0.gguf">>),
    ?assertEqual(<<"hf.co/acme/tiny:tiny-model.q8_0">>, maps:get(name, M)),
    ?assertEqual(undefined, maps:get(mmproj, M)).

split_set(Config) ->
    A = <<"tiny.Q4_K_M-00001-of-00002.gguf">>,
    B = <<"tiny.Q4_K_M-00002-of-00002.gguf">>,
    Reg = erllama_server_fake_registry,
    One = shard_bytes(0),
    Two = shard_bytes(1),
    ets:insert(Reg, [{{blob, first}, One}, {{blob, second}, Two}]),
    ok = Reg:set(paged, true),
    ok = Reg:set(files, [{B, second}]),
    {#{type := error, data := #{error := {missing_shard, A}}}, _} = pull(<<"hf://acme/tiny">>),
    ok = Reg:set(files, [{B, second}, {A, first}]),
    ets:insert(Reg, {{blob, second}, One}),
    {#{type := error}, _} = pull(<<"hf://acme/tiny">>),
    [] = erllama_server_registry:list(),
    ets:insert(Reg, {{blob, second}, Two}),
    ets:match_delete(Reg, {{interrupted, '_'}, '_'}),
    ok = Reg:set(interrupt_after, 16),
    ok = Reg:set(interrupt_how, drop),
    {ok, #{job_id := J1}} = erllama_server_api:pull(<<"hf://acme/tiny">>, #{}),
    {ok, #{job_id := J2}} = erllama_server_api:pull(<<"hf://acme/tiny/", B/binary>>, #{
        name => <<"copy">>
    }),
    {#{type := done, data := #{model := Entry}}, _} = follow(J1),
    #{path := Path, size := Size, name := Name, status := ok} = Entry,
    ?assertEqual(byte_size(One) + byte_size(Two), Size),
    {ok, [Path, Other]} = erllama_gguf_shards:paths(Path),
    ?assertEqual({ok, Two}, file:read_file(Other)),
    {#{type := done, data := #{model := #{path := Path}}}, _} = follow(J2),
    ?assert(lists:any(fun(#{range := R}) -> R =:= <<"bytes=16-">> end, Reg:seen())),
    ok = application:stop(erllama_server),
    {ok, _} = application:ensure_all_started(erllama_server),
    ok = erllama_server_registry:remove(Name),
    ?assertMatch({ok, #{status := ok}}, erllama_server_registry:resolve(<<"copy">>)),
    ok = erllama_server_registry:remove(<<"copy">>),
    ?assertNot(filelib:is_file(Path)),
    ?assertNot(filelib:is_file(Other)),
    Dir = ?config(data_dir, Config),
    P1 = filename:join(Dir, A),
    P2 = filename:join(Dir, B),
    ok = file:write_file(P2, Two),
    ?assertMatch({error, {missing_shard, _}}, erllama_server_registry:register_local(P2, #{})),
    ok = file:write_file(P1, One),
    {ok, #{path := P1, name := Local, size := Size}} = erllama_server_registry:register_local(
        P2, #{}
    ),
    {ok, Fp} = erllama_fingerprint:derive(P1, safe),
    ok = file:write_file(P2, <<Two/binary, 0>>),
    ?assertNotEqual({ok, Fp}, erllama_fingerprint:derive(P1, safe)),
    ok = file:delete(P2),
    ?assertMatch({ok, #{status := missing}}, erllama_server_registry:resolve(Local)),
    ok = erllama_server_registry:remove(Local),
    ?assert(filelib:is_file(P1)).

shard_bytes(Index) ->
    <<"GGUF", 3:32/little, 0:64/little, 2:64/little, 8:64/little, "split.no", 2:32/little,
        Index:16/little, 11:64/little, "split.count", 2:32/little, 2:16/little>>.

hf_unknown_quant(_Config) ->
    {#{type := error, data := #{error := {quant_not_found, <<"q2_k">>, Available}}}, _} =
        pull(<<"hf://acme/tiny:Q2_K">>),
    ?assertEqual([<<"q4_k_m">>, <<"q8_0">>], lists:sort(Available)).

ollama_model_with_params(_Config) ->
    {#{type := done, data := #{model := M}}, _} = pull(<<"tiny:1b">>),
    ?assertEqual(<<"tiny:1b">>, maps:get(name, M)),
    ?assertEqual(#{stop_sequences => [<<"<|end|>">>], temperature => 0.5}, maps:get(defaults, M)),
    ?assertEqual(blob_sha(model), file_sha(maps:get(path, M))),
    %% The defaults survive the registry file.
    {ok, #{defaults := D}} = erllama_server_registry:resolve(<<"tiny:1b">>),
    ?assertEqual(maps:get(defaults, M), D).

https_with_checksum(Config) ->
    Url = plain_url(Config),
    {#{type := done, data := #{model := M}}, _} = pull(Url, #{sha256 => blob_sha(model)}),
    ?assertEqual(<<"model:latest">>, maps:get(name, M)).

https_wrong_checksum(Config) ->
    Url = plain_url(Config),
    Wrong = binary:copy(<<"0">>, 64),
    {#{type := error, data := #{error := {integrity_mismatch, _}}}, _} = pull(Url, #{
        sha256 => Wrong
    }),
    ?assertEqual([], erllama_server_registry:list()),
    ?assertEqual([], [
        F
     || F <- filelib:wildcard("*", blobs(Config)), filename:extension(F) =:= ".part"
    ]).

broken_transfer_resumes(Config) ->
    ok = erllama_server_fake_registry:set(interrupt_after, 100000),
    {#{type := done, data := #{model := M}}, _} = pull(<<"tiny:1b">>),
    ?assertEqual(blob_sha(model), file_sha(maps:get(path, M))),
    Ranges = [R || #{range := R} <- erllama_server_fake_registry:seen(), R =/= undefined],
    ?assertMatch([<<"bytes=", _/binary>> | _], Ranges),
    ?assertEqual([], [
        F
     || F <- filelib:wildcard("*", blobs(Config)), filename:extension(F) =:= ".part"
    ]).

%% The connection breaks in the middle of a blob: the pull asks for the
%% rest at once, without waiting for its idle timeout.
dropped_connection_resumes(Config) ->
    ok = erllama_server_fake_registry:set(interrupt_after, 100000),
    ok = erllama_server_fake_registry:set(interrupt_how, drop),
    T0 = erlang:monotonic_time(millisecond),
    {#{type := done, data := #{model := M}}, _} = pull(<<"tiny:1b">>),
    ?assert(erlang:monotonic_time(millisecond) - T0 < 1500),
    ?assertEqual(blob_sha(model), file_sha(maps:get(path, M))),
    Ranges = [R || #{range := R} <- erllama_server_fake_registry:seen(), R =/= undefined],
    ?assertMatch([<<"bytes=", _/binary>> | _], Ranges),
    ?assertEqual([], [
        F
     || F <- filelib:wildcard("*", blobs(Config)), filename:extension(F) =:= ".part"
    ]).

ignored_range_restarts(_Config) ->
    ok = erllama_server_fake_registry:set(interrupt_after, 100000),
    ok = erllama_server_fake_registry:set(ignore_range, true),
    {#{type := done, data := #{model := M}}, _} = pull(<<"tiny:1b">>),
    ?assertEqual(blob_sha(model), file_sha(maps:get(path, M))).

corrupt_blob_is_rejected(Config) ->
    ok = erllama_server_fake_registry:set(corrupt, true),
    {#{type := error, data := #{error := {integrity_mismatch, #{expected_sha256 := Want}}}}, _} =
        pull(<<"tiny:1b">>),
    ?assertEqual(blob_sha(model), Want),
    ?assertEqual([], erllama_server_registry:list()),
    ?assertEqual([], filelib:wildcard("*", blobs(Config))).

%% Cancelling keeps what was downloaded; the next pull asks for the rest.
cancel_then_resume(Config) ->
    ok = erllama_server_fake_registry:set(interrupt_after, 100000),
    {ok, #{job_id := Id}} = erllama_server_api:pull(<<"tiny:1b">>, #{}),
    ok = wait_part(Config),
    ok = erllama_server_api:cancel(Id),
    {#{type := cancelled}, _} = follow(Id),
    [Part] = [F || F <- filelib:wildcard("*", blobs(Config)), filename:extension(F) =:= ".part"],
    ?assert(filelib:file_size(filename:join(blobs(Config), Part)) > 0),
    ok = erllama_server_fake_registry:set(interrupt_after, undefined),
    {#{type := done, data := #{model := M}}, _} = pull(<<"tiny:1b">>),
    ?assertEqual(blob_sha(model), file_sha(maps:get(path, M))),
    ?assert(
        lists:any(fun(#{range := R}) -> R =/= undefined end, erllama_server_fake_registry:seen())
    ).

registry_survives_restart(_Config) ->
    {#{type := done}, _} = pull(<<"tiny:1b">>),
    ok = application:stop(erllama_server),
    {ok, _} = application:ensure_all_started(erllama_server),
    {ok, [#{name := <<"tiny:1b">>, status := ok}]} = erllama_server_api:models().

remove_deletes_unused_blob(Config) ->
    {#{type := done, data := #{model := #{path := Path}}}, _} = pull(<<"tiny:1b">>),
    {ok, _} = erllama_server_api:copy(<<"tiny:1b">>, <<"tiny:copy">>),
    ok = erllama_server_api:remove(<<"tiny:1b">>),
    %% Still used by the copy.
    ?assert(filelib:is_regular(Path)),
    ok = erllama_server_api:remove(<<"tiny:copy">>),
    ?assertNot(filelib:is_regular(Path)),
    ?assertEqual([], filelib:wildcard("*.gguf", blobs(Config))).

%% A blob already on disk is reused only when its sha256 matches: an
%% intact one is not fetched again, a damaged one of the right size is.
reused_blob_is_verified(_Config) ->
    {#{type := done, data := #{model := #{path := Path}}}, _} = pull(<<"tiny:1b">>),
    BlobFetches = fun() ->
        length([S || #{path := P} = S <- erllama_server_fake_registry:seen(), is_model_blob(P)])
    end,
    Before = BlobFetches(),
    {#{type := done}, _} = pull(<<"tiny:1b">>, #{name => <<"tiny:twice">>}),
    ?assertEqual(Before, BlobFetches()),
    %% Damage one byte, keep the size.
    {ok, Bin} = file:read_file(Path),
    <<First, Rest/binary>> = Bin,
    ok = file:write_file(Path, <<(First bxor 16#ff), Rest/binary>>),
    {#{type := done, data := #{model := M}}, _} = pull(<<"tiny:1b">>, #{name => <<"tiny:thrice">>}),
    ?assert(BlobFetches() > Before),
    ?assertEqual(blob_sha(model), file_sha(maps:get(path, M))),
    ?assertEqual(blob_sha(model), file_sha(Path)).

is_model_blob(P) -> binary:match(P, blob_sha(model)) =/= nomatch.

%% -----------------------------------------------------------------------------

blobs(Config) -> filename:join(?config(data_dir, Config), "blobs").

plain_url(Config) ->
    #{base := Base} = ?config(fake, Config),
    <<Base/binary, "/plain/model.gguf">>.

wait_part(Config) -> wait_part(Config, 400).
wait_part(_Config, 0) ->
    ct:fail(no_part_file);
wait_part(Config, N) ->
    %% The first transfer was cut at 100000 bytes; the pull is waiting
    %% to retry.
    Parts = [
        filename:join(blobs(Config), F)
     || F <- filelib:wildcard("*", blobs(Config)), filename:extension(F) =:= ".part"
    ],
    case [P || P <- Parts, filelib:file_size(P) >= 100000] of
        [_] ->
            ok;
        _ ->
            timer:sleep(5),
            wait_part(Config, N - 1)
    end.
