%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Helpers shared by the eunit and CT suites: quiet teardown, temp
%% directories, application fixture.
-module(erllama_test_helpers).

-export([
    stop_quiet/1,
    stop_model_quiet/1,
    rm_rf/1,
    tmp_dir/1,
    with_app/1
]).

%% Stop a gen_server (by pid or registered name); a server that is
%% already gone is not an error.
-spec stop_quiet(pid() | atom()) -> ok.
stop_quiet(Server) ->
    try
        gen_server:stop(Server)
    catch
        exit:_ -> ok
    end.

%% Stop a model process; a model that is already gone is not an error.
-spec stop_model_quiet(erllama:model()) -> ok.
stop_model_quiet(Model) ->
    try
        erllama_model:stop(Model)
    catch
        exit:_ -> ok
    end.

%% Delete a directory tree.
-spec rm_rf(file:name()) -> ok.
rm_rf(Dir) ->
    case file:list_dir(Dir) of
        {ok, Entries} ->
            lists:foreach(fun(E) -> rm_entry(filename:join(Dir, E)) end, Entries),
            _ = file:del_dir(Dir),
            ok;
        _ ->
            ok
    end.

rm_entry(Path) ->
    case filelib:is_dir(Path) of
        true ->
            rm_rf(Path);
        false ->
            _ = file:delete(Path),
            ok
    end.

%% Fresh empty directory under TMPDIR.
-spec tmp_dir(string()) -> file:name().
tmp_dir(Tag) ->
    Dir = filename:join(
        os:getenv("TMPDIR", "/tmp"),
        "erllama_" ++ Tag ++ "_" ++ integer_to_list(erlang:unique_integer([positive]))
    ),
    ok = filelib:ensure_path(Dir),
    Dir.

%% Run `Body' with the erllama application started, stopping
%% whatever the call started afterwards.
-spec with_app(fun(() -> T)) -> T.
with_app(Body) ->
    {ok, Started} = application:ensure_all_started(erllama),
    try
        Body()
    after
        [application:stop(A) || A <- lists:reverse(Started)],
        ok
    end.
