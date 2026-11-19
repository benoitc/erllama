%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_tool_fs).
-moduledoc false.
%% The workspace file tools. Each returns `{ok, Text}' for the model,
%% or `{error, Text}' it can act on (a wrong path, an ambiguous edit):
%% errors are results, not crashes.

-export([read_file/2, list_dir/2, search/2, write_file/2, edit_file/2]).

-include_lib("kernel/include/file.hrl").

-define(READ_LINES, 400).
-define(READ_BYTES, 65536).
-define(LINE_CHARS, 2000).
-define(MAX_FILE, 10485760).
-define(LIST_MAX, 500).
-define(SEARCH_MAX, 200).
-define(SEARCH_FILE_MAX, 1048576).

%% Numbered lines from `offset' (1-based), at most `limit'.
read_file(Ws, #{<<"path">> := P} = Args) ->
    with_path(Ws, P, fun(Path) ->
        case file:read_file_info(Path) of
            {ok, #file_info{size = Size}} when Size > ?MAX_FILE ->
                {error, <<"the file is larger than 10 MiB; use search to find the part you need">>};
            {ok, _} ->
                case file:read_file(Path) of
                    {ok, Bin} ->
                        {ok,
                            page(
                                Bin,
                                arg_int(Args, <<"offset">>, 1),
                                arg_int(Args, <<"limit">>, ?READ_LINES)
                            )};
                    {error, R} ->
                        {error, reason(R)}
                end;
            {error, R} ->
                {error, reason(R)}
        end
    end);
read_file(_Ws, _) ->
    missing(<<"path">>).

page(Bin, Offset0, Limit0) ->
    Lines = binary:split(Bin, <<"\n">>, [global]),
    Total = length(Lines),
    Offset = max(1, Offset0),
    Limit = max(1, min(Limit0, ?READ_LINES)),
    Selected = lists:sublist(Lines, Offset, Limit),
    {Out, Shown} = take_bytes(Selected, Offset, [], 0),
    Last = Offset + Shown - 1,
    Footer =
        case Last < Total of
            true ->
                io_lib:format("~n(lines ~b-~b of ~b; continue with offset ~b)", [
                    Offset, Last, Total, Last + 1
                ]);
            false ->
                io_lib:format("~n(lines ~b-~b of ~b)", [Offset, max(Offset, Last), Total])
        end,
    iolist_to_binary([lists:reverse(Out), Footer]).

take_bytes([], _N, Acc, Shown) ->
    {Acc, Shown};
take_bytes([L | Rest], N, Acc, Shown) ->
    Line = [integer_to_binary(N), <<"\t">>, clip(L), <<"\n">>],
    case iolist_size(Acc) + iolist_size(Line) > ?READ_BYTES andalso Shown > 0 of
        true -> {Acc, Shown};
        false -> take_bytes(Rest, N + 1, [Line | Acc], Shown + 1)
    end.

clip(L) when byte_size(L) > ?LINE_CHARS -> <<(binary:part(L, 0, ?LINE_CHARS))/binary, "...">>;
clip(L) -> L.

%% Entries of a directory, directories marked with `/'; `depth' up to 3.
list_dir(Ws, Args) ->
    P = maps:get(<<"path">>, Args, <<".">>),
    Depth = max(1, min(3, arg_int(Args, <<"depth">>, 1))),
    with_path(Ws, P, fun(Path) ->
        case filelib:is_dir(Path) of
            true ->
                Entries = lists:sort(entries(Path, Depth, Ws)),
                {Shown, More} = split(Entries, ?LIST_MAX),
                Note =
                    case More of
                        0 ->
                            <<>>;
                        _ ->
                            iolist_to_binary(io_lib:format("~n(~b more entries not shown)", [More]))
                    end,
                {ok, iolist_to_binary([lists:join(<<"\n">>, Shown), Note])};
            false ->
                {error, <<"not a directory">>}
        end
    end).

entries(Dir, Depth, Ws) ->
    case file:list_dir(Dir) of
        {ok, Names} ->
            lists:flatmap(
                fun(Name) ->
                    Full = filename:join(Dir, Name),
                    Rel = erllama_server_workspace:relative(Ws, unicode:characters_to_binary(Full)),
                    case kind(Full) of
                        dir when Depth > 1, hd(Name) =/= $. ->
                            [<<Rel/binary, "/">> | entries(Full, Depth - 1, Ws)];
                        dir ->
                            [<<Rel/binary, "/">>];
                        link ->
                            [<<Rel/binary, "@">>];
                        _ ->
                            [Rel]
                    end
                end,
                Names
            );
        {error, _} ->
            []
    end.

split(L, N) when length(L) =< N -> {L, 0};
split(L, N) -> {lists:sublist(L, N), length(L) - N}.

%% Lines matching a regular expression, as `path:line: text'.
search(Ws, #{<<"pattern">> := Pattern} = Args) ->
    case re:compile(Pattern, [unicode]) of
        {ok, Re} ->
            P = maps:get(<<"path">>, Args, <<".">>),
            Glob = glob(maps:get(<<"glob">>, Args, undefined)),
            with_path(Ws, P, fun(Path) ->
                Files = files(Ws, Path, Glob),
                {Matches, Truncated} = scan(Files, Re, Ws, [], 0),
                Body =
                    case Matches of
                        [] -> <<"no matches">>;
                        _ -> lists:join(<<"\n">>, lists:reverse(Matches))
                    end,
                Note =
                    case Truncated of
                        true -> io_lib:format("~n(stopped at ~b matches)", [?SEARCH_MAX]);
                        false -> <<>>
                    end,
                {ok, iolist_to_binary([Body, Note])}
            end);
        {error, {Why, _}} ->
            {error, iolist_to_binary(io_lib:format("invalid pattern: ~s", [Why]))}
    end;
search(_Ws, _) ->
    missing(<<"pattern">>).

glob(undefined) ->
    undefined;
glob(G) ->
    Escaped = re:replace(G, <<"[.+^$(){}|\\[\\]\\\\]">>, <<"\\\\&">>, [global, {return, binary}]),
    Star = binary:replace(Escaped, <<"*">>, <<"[^/]*">>, [global]),
    Q = binary:replace(Star, <<"?">>, <<".">>, [global]),
    {ok, Re} = re:compile(<<"^", Q/binary, "$">>),
    Re.

files(Ws, Path, Glob) ->
    case filelib:is_dir(Path) of
        true -> walk(Ws, Path, Glob);
        false -> [Path]
    end.

%% What a directory entry is, without following it: a directory, a
%% file, or a symbolic link. Walks never descend a link (it may lead
%% outside the workspace, or back into itself).
kind(Full) ->
    case file:read_link_info(Full) of
        {ok, #file_info{type = directory}} -> dir;
        {ok, #file_info{type = symlink}} -> link;
        {ok, #file_info{type = regular}} -> file;
        _ -> other
    end.

%% A linked file is searched only when it resolves inside the workspace.
linked_file(Ws, Full) ->
    case erllama_server_workspace:resolve(Ws, Full) of
        {ok, Real} -> [Real || filelib:is_regular(Real)];
        {error, _} -> []
    end.

walk(Ws, Dir, Glob) ->
    case file:list_dir(Dir) of
        {ok, Names} ->
            lists:flatmap(
                fun
                    ([$. | _]) ->
                        [];
                    (Name) ->
                        Full = filename:join(Dir, Name),
                        case kind(Full) of
                            dir -> walk(Ws, Full, Glob);
                            file -> [Full || matches_glob(Name, Glob)];
                            link -> [F || matches_glob(Name, Glob), F <- linked_file(Ws, Full)];
                            other -> []
                        end
                end,
                lists:sort(Names)
            );
        {error, _} ->
            []
    end.

matches_glob(_Name, undefined) -> true;
matches_glob(Name, Re) -> re:run(Name, Re) =/= nomatch.

scan([], _Re, _Ws, Acc, _N) ->
    {Acc, false};
scan(_Files, _Re, _Ws, Acc, N) when N >= ?SEARCH_MAX ->
    {Acc, true};
scan([F | Rest], Re, Ws, Acc, N) ->
    case readable_text(F) of
        {ok, Bin} ->
            Rel = erllama_server_workspace:relative(Ws, unicode:characters_to_binary(F)),
            {Acc1, N1} = scan_lines(binary:split(Bin, <<"\n">>, [global]), 1, Re, Rel, Acc, N),
            scan(Rest, Re, Ws, Acc1, N1);
        skip ->
            scan(Rest, Re, Ws, Acc, N)
    end.

scan_lines([], _I, _Re, _Rel, Acc, N) ->
    {Acc, N};
scan_lines(_Lines, _I, _Re, _Rel, Acc, N) when N >= ?SEARCH_MAX ->
    {Acc, N};
scan_lines([L | Rest], I, Re, Rel, Acc, N) ->
    case re:run(L, Re) of
        {match, _} ->
            scan_lines(
                Rest,
                I + 1,
                Re,
                Rel,
                [[Rel, <<":">>, integer_to_binary(I), <<": ">>, clip(L)] | Acc],
                N + 1
            );
        nomatch ->
            scan_lines(Rest, I + 1, Re, Rel, Acc, N)
    end.

%% Small text files only: no NUL byte in the first 8 KiB.
readable_text(F) ->
    case filelib:file_size(F) =< ?SEARCH_FILE_MAX andalso file:read_file(F) of
        {ok, Bin} -> text_only(Bin);
        _ -> skip
    end.

text_only(Bin) ->
    case binary:match(binary:part(Bin, 0, min(8192, byte_size(Bin))), <<0>>) of
        nomatch -> {ok, Bin};
        _ -> skip
    end.

%% A new file. Replacing an existing one goes through edit_file.
write_file(Ws, #{<<"path">> := P, <<"content">> := C}) when is_binary(C) ->
    with_path(Ws, P, fun(Path) ->
        case filelib:is_file(Path) of
            true -> {error, <<"the file exists; use edit_file to change it">>};
            false -> create(Path, P, C)
        end
    end);
write_file(_Ws, _) ->
    missing(<<"path and content">>).

create(Path, Name, Content) ->
    case filelib:ensure_dir(Path) =:= ok andalso file:write_file(Path, Content) of
        ok ->
            {ok,
                iolist_to_binary(
                    io_lib:format("created ~ts (~b bytes)", [Name, byte_size(Content)])
                )};
        {error, R} ->
            {error, reason(R)}
    end.

%% Replace an exact, unique occurrence of `old' (or every one with
%% `replace_all').
edit_file(Ws, #{<<"path">> := P, <<"old">> := Old, <<"new">> := New} = Args) when
    is_binary(Old), Old =/= <<>>, is_binary(New)
->
    All =
        case Args of
            #{<<"replace_all">> := true} -> true;
            _ -> false
        end,
    with_path(Ws, P, fun(Path) ->
        case file:read_file(Path) of
            {ok, Bin} -> edit(Path, Bin, Old, New, All);
            {error, R} -> {error, reason(R)}
        end
    end);
edit_file(_Ws, _) ->
    missing(<<"path, old and new">>).

edit(Path, Bin, Old, New, All) ->
    case {binary:matches(Bin, Old), All} of
        {[], _} ->
            {error,
                <<"the text to replace was not found (it must match exactly, whitespace included)">>};
        {[_, _ | _] = Ms, false} ->
            {error,
                iolist_to_binary(
                    io_lib:format(
                        "the text to replace appears ~b times; give more context or set replace_all",
                        [length(Ms)]
                    )
                )};
        {Ms, _} ->
            Out = binary:replace(Bin, Old, New, [global]),
            case file:write_file(Path, Out) of
                ok ->
                    {ok,
                        iolist_to_binary(io_lib:format("replaced ~b occurrence(s)", [length(Ms)]))};
                {error, R} ->
                    {error, reason(R)}
            end
    end.

%% -----------------------------------------------------------------------------

with_path(Ws, P, Fun) when is_binary(P) ->
    case erllama_server_workspace:resolve(Ws, P) of
        {ok, Path} -> Fun(Path);
        {error, {outside_workspace, _}} -> {error, <<"the path is outside the workspace">>};
        {error, R} -> {error, reason(R)}
    end;
with_path(_Ws, _P, _Fun) ->
    {error, <<"path must be a string">>}.

arg_int(Args, K, Default) ->
    case maps:get(K, Args, Default) of
        N when is_integer(N) -> N;
        _ -> Default
    end.

missing(What) -> {error, <<"missing arguments: ", What/binary>>}.

reason(enoent) -> <<"no such file or directory">>;
reason(eacces) -> <<"permission denied">>;
reason(eisdir) -> <<"is a directory">>;
reason(enotdir) -> <<"not a directory">>;
reason(R) -> iolist_to_binary(io_lib:format("~p", [R])).
