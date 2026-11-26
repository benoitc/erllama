%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_tools_tests).
-include_lib("eunit/include/eunit.hrl").

-define(FS, erllama_server_tool_fs).

with_ws(Fun) ->
    Dir = filename:join([
        os:getenv("TMPDIR", "/tmp"),
        "erllama_tools_tests",
        os:getpid() ++ "-" ++ integer_to_list(erlang:unique_integer([positive]))
    ]),
    ok = filelib:ensure_path(Dir),
    {ok, Ws} = erllama_server_workspace:open(Dir),
    try
        Fun(Ws)
    after
        file:del_dir_r(Dir)
    end.

write(Ws, Rel, Bin) ->
    P = filename:join(Ws, Rel),
    ok = filelib:ensure_dir(P),
    ok = file:write_file(P, Bin).

workspace_confines_paths_test() ->
    with_ws(fun(Ws) ->
        R = fun(P) -> erllama_server_workspace:resolve(Ws, P) end,
        ?assertMatch({ok, _}, R(<<"a/b.txt">>)),
        ?assertMatch({ok, _}, R(<<"a/../b.txt">>)),
        ?assertEqual({ok, Ws}, R(<<".">>)),
        ?assertMatch({error, {outside_workspace, _}}, R(<<"../x">>)),
        ?assertMatch({error, {outside_workspace, _}}, R(<<"/etc/passwd">>)),
        %% A symlink pointing out is refused, through it and below it.
        ok = file:make_symlink("/etc", filename:join(Ws, "out")),
        ?assertMatch({error, {outside_workspace, _}}, R(<<"out/hosts">>)),
        %% A symlink inside is fine.
        ok = filelib:ensure_path(filename:join(Ws, "real")),
        ok = file:make_symlink(filename:join(Ws, "real"), filename:join(Ws, "in")),
        ?assertMatch({ok, _}, R(<<"in/file">>))
    end).

read_file_pages_test() ->
    with_ws(fun(Ws) ->
        write(
            Ws,
            "f.txt",
            iolist_to_binary([[integer_to_binary(I), <<"\n">>] || I <- lists:seq(1, 10)])
        ),
        {ok, P1} = ?FS:read_file(Ws, #{<<"path">> => <<"f.txt">>, <<"limit">> => 3}),
        ?assertMatch(<<"1\t1\n2\t2\n3\t3\n", _/binary>>, P1),
        ?assertNotEqual(nomatch, binary:match(P1, <<"continue with offset 4">>)),
        {ok, P2} = ?FS:read_file(Ws, #{<<"path">> => <<"f.txt">>, <<"offset">> => 9}),
        ?assertMatch(<<"9\t9\n10\t10\n", _/binary>>, P2),
        ?assertEqual(
            {error, <<"no such file or directory">>}, ?FS:read_file(Ws, #{<<"path">> => <<"none">>})
        ),
        ?assertMatch({error, <<"missing arguments", _/binary>>}, ?FS:read_file(Ws, #{}))
    end).

list_and_search_test() ->
    with_ws(fun(Ws) ->
        write(Ws, "src/a.erl", <<"-module(a).\nfoo() -> ok.\n">>),
        write(Ws, "src/b.txt", <<"foo bar\n">>),
        write(Ws, ".hidden/c.erl", <<"foo\n">>),
        write(Ws, "bin.dat", <<0, 1, 2, "foo">>),
        {ok, L1} = ?FS:list_dir(Ws, #{}),
        ?assertEqual(
            [<<".hidden/">>, <<"bin.dat">>, <<"src/">>], binary:split(L1, <<"\n">>, [global])
        ),
        {ok, L2} = ?FS:list_dir(Ws, #{<<"depth">> => 2}),
        ?assertNotEqual(nomatch, binary:match(L2, <<"src/a.erl">>)),
        {ok, S1} = ?FS:search(Ws, #{<<"pattern">> => <<"foo">>}),
        ?assertEqual(
            [<<"src/a.erl:2: foo() -> ok.">>, <<"src/b.txt:1: foo bar">>],
            binary:split(S1, <<"\n">>, [global])
        ),
        {ok, S2} = ?FS:search(Ws, #{<<"pattern">> => <<"foo">>, <<"glob">> => <<"*.erl">>}),
        ?assertEqual(<<"src/a.erl:2: foo() -> ok.">>, S2),
        ?assertMatch(
            {error, <<"invalid pattern", _/binary>>}, ?FS:search(Ws, #{<<"pattern">> => <<"(">>})
        )
    end).

%% Recursive search and listing never follow a link out of the
%% workspace, whatever the depth.
walks_stay_inside_test() ->
    with_ws(fun(Ws) ->
        Outside = filename:join(
            filename:dirname(Ws), <<"outside-", (filename:basename(Ws))/binary>>
        ),
        ok = filelib:ensure_path(Outside),
        try
            ok = file:write_file(filename:join(Outside, "secret.txt"), <<"MARKER outside\n">>),
            write(Ws, "src/a.txt", <<"MARKER inside\n">>),
            ok = file:make_symlink(Outside, filename:join([Ws, "src", "linkdir"])),
            ok = file:make_symlink(
                filename:join(Outside, "secret.txt"), filename:join([Ws, "src", "linkfile"])
            ),
            ok = file:make_symlink(
                filename:join([Ws, "src", "a.txt"]), filename:join([Ws, "src", "inlink"])
            ),
            {ok, Found} = ?FS:search(Ws, #{<<"pattern">> => <<"MARKER">>}),
            ?assertEqual(nomatch, binary:match(Found, <<"outside">>)),
            ?assertNotEqual(nomatch, binary:match(Found, <<"src/a.txt:1: MARKER inside">>)),
            {ok, Listed} = ?FS:list_dir(Ws, #{<<"depth">> => 3}),
            ?assertEqual(nomatch, binary:match(Listed, <<"secret">>)),
            ?assertNotEqual(nomatch, binary:match(Listed, <<"src/linkdir@">>))
        after
            file:del_dir_r(Outside)
        end
    end).

write_and_edit_test() ->
    with_ws(fun(Ws) ->
        {ok, _} = ?FS:write_file(Ws, #{<<"path">> => <<"d/x.txt">>, <<"content">> => <<"a a b\n">>}),
        ?assertMatch(
            {error, <<"the file exists", _/binary>>},
            ?FS:write_file(Ws, #{<<"path">> => <<"d/x.txt">>, <<"content">> => <<"z">>})
        ),
        Edit = fun(Old, New, All) ->
            ?FS:edit_file(Ws, #{
                <<"path">> => <<"d/x.txt">>,
                <<"old">> => Old,
                <<"new">> => New,
                <<"replace_all">> => All
            })
        end,
        ?assertMatch(
            {error, <<"the text to replace appears 2 times", _/binary>>},
            Edit(<<"a">>, <<"c">>, false)
        ),
        ?assertMatch(
            {error, <<"the text to replace was not found", _/binary>>},
            Edit(<<"q">>, <<"c">>, false)
        ),
        ?assertMatch({ok, <<"replaced 1", _/binary>>}, Edit(<<"b">>, <<"B">>, false)),
        ?assertMatch({ok, <<"replaced 2", _/binary>>}, Edit(<<"a">>, <<"c">>, true)),
        ?assertEqual({ok, <<"c c B\n">>}, file:read_file(filename:join([Ws, "d", "x.txt"]))),
        ?assertMatch(
            {error, <<"the path is outside the workspace">>},
            ?FS:write_file(Ws, #{<<"path">> => <<"../y">>, <<"content">> => <<"y">>})
        )
    end).

shell_output_counts_omitted_bytes_test() ->
    with_ws(fun(Ws) ->
        Bytes = binary:copy(<<"abcd">>, 32768),
        write(Ws, "flood", Bytes),
        Ref = make_ref(),
        Progress = fun(Text, Omitted) -> self() ! {Ref, Text, Omitted} end,
        #{exit := 0, truncated := Truncated} =
            erllama_server_tool_shell:run(Ws, <<"cat flood">>, 5000, Progress),
        ?assertEqual(byte_size(Bytes) - 32768, Truncated),
        ?assertEqual(<<>>, check_chunks(Ref, Bytes))
    end).

check_chunks(Ref, Remaining) ->
    receive
        {Ref, Chunk, Omitted} ->
            Size = byte_size(Chunk),
            ?assert(Size =< 16384),
            <<_:Omitted/binary, Chunk:Size/binary, Rest/binary>> = Remaining,
            check_chunks(Ref, Rest)
    after 0 ->
        Remaining
    end.

shell_output_invalid_utf8_test() ->
    with_ws(fun(Ws) ->
        Bytes = <<$a, 255, 195>>,
        write(Ws, "binary", Bytes),
        Ref = make_ref(),
        Progress = fun(Text, 0) -> self() ! {Ref, Text} end,
        #{exit := 0, output := Bytes} =
            erllama_server_tool_shell:run(Ws, <<"cat binary">>, 5000, Progress),
        Chunks = collect_text(Ref),
        ?assertEqual(<<$a, 16#fffd/utf8, 16#fffd/utf8>>, iolist_to_binary(Chunks))
    end).

collect_text(Ref) ->
    receive
        {Ref, Text} -> [Text | collect_text(Ref)]
    after 0 ->
        []
    end.

tool_catalog_test() ->
    Names = erllama_server_tools:names(),
    ?assertEqual(lists:usort(Names), lists:sort(Names)),
    [
        ?assertEqual(auto, erllama_server_tools:default_policy(N))
     || N <- [<<"read_file">>, <<"list_dir">>, <<"search">>, <<"shell_status">>]
    ],
    [
        ?assertEqual(ask, erllama_server_tools:default_policy(N))
     || N <- [<<"write_file">>, <<"edit_file">>, <<"shell">>, <<"shell_kill">>]
    ],
    ?assertEqual(
        {error, <<"unknown tool: nope">>},
        erllama_server_tools:execute(<<"nope">>, #{}, #{workspace => <<"/tmp">>})
    ).

%% A tool's class lives with its definition: every native tool has one,
%% the schema offered to the model carries no class, and a name the
%% module does not define is never treated as a read.
classes_test() ->
    ?assertEqual(
        [<<"edit_file">>, <<"shell">>, <<"shell_kill">>, <<"write_file">>],
        lists:sort([N || N <- erllama_server_tools:names(), erllama_server_tools:class(N) =:= write])
    ),
    ?assertEqual(read, erllama_server_tools:class(<<"read_file">>)),
    ?assertEqual(write, erllama_server_tools:class(<<"no_such_tool">>)),
    ?assertEqual(ask, erllama_server_tools:default_policy(<<"no_such_tool">>)),
    [
        ?assertEqual([description, name, parameters], lists:sort(maps:keys(D)))
     || D <- erllama_server_tools:definitions()
    ].
