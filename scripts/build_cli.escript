#!/usr/bin/env escript
%% Build the `erllama' CLI escript from erllama_server's compiled modules
%% only. rebar3's escriptize would pack every dependency of the app
%% (Livery and the rest, none of which the CLI loads, and optional ones
%% that are not even fetched); the CLI needs nothing but this app.
%%
%%   escript scripts/build_cli.escript BUILD_DIR
%%
%% writes BUILD_DIR/bin/erllama from BUILD_DIR/lib/erllama_server/ebin.
-mode(compile).
-include_lib("kernel/include/file.hrl").

main([BuildDir]) ->
    Ebin = filename:join([BuildDir, "lib", "erllama_server", "ebin"]),
    Files = [
        {filename:join(["erllama_server", "ebin", F]), read(filename:join(Ebin, F))}
     || F <- filelib:wildcard("*.{beam,app}", Ebin)
    ],
    Files =/= [] orelse fail("no compiled erllama_server in ~ts", [Ebin]),
    Out = filename:join([BuildDir, "bin", "erllama"]),
    ok = filelib:ensure_dir(Out),
    %% `-noinput' as well as `-noshell': the VM must not let prim_tty
    %% own fd 0, or the ACP agent's {fd, 0, 1} port can never read it
    %% (the steal is logged, then the reader silently starves).
    ok = escript:create(Out, [shebang, {emu_args, "-noshell -noinput -escript main erllama_server_cli"}, {archive, Files, []}]),
    {ok, #file_info{mode = Mode}} = file:read_file_info(Out),
    ok = file:change_mode(Out, Mode bor 8#111),
    io:format("built ~ts (~b files)~n", [Out, length(Files)]);
main(_) ->
    fail("usage: build_cli.escript BUILD_DIR", []).

read(Path) ->
    {ok, Bin} = file:read_file(Path),
    Bin.

fail(Fmt, Args) ->
    io:format(standard_error, Fmt ++ "~n", Args),
    halt(1).
