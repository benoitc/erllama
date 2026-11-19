%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_workspace).
-moduledoc false.
%% The directory an agent session works in, on the release machine.
%% Every path a tool receives is resolved against it, with `.', `..'
%% and symbolic links resolved, and refused when it lands outside. The
%% client's own filesystem is never involved.

-export([open/1, resolve/2, relative/2]).

-include_lib("kernel/include/file.hrl").

-define(MAX_LINKS, 32).

%% The canonical form of a workspace directory, which must exist.
-spec open(file:filename_all()) -> {ok, binary()} | {error, term()}.
open(Dir) ->
    case realpath(unicode:characters_to_binary(filename:absname(Dir))) of
        {ok, Real} ->
            case filelib:is_dir(Real) of
                true -> {ok, Real};
                false -> {error, {not_a_directory, Real}}
            end;
        {error, _} = E ->
            E
    end.

%% `Path' (relative to the workspace, or absolute) as an absolute path
%% inside it. The file need not exist; its existing ancestors are
%% resolved, so a symlink cannot point outside.
-spec resolve(binary(), file:filename_all()) -> {ok, binary()} | {error, term()}.
resolve(Workspace, Path0) ->
    Path = unicode:characters_to_binary(Path0),
    Abs = normalize(filename:absname(Path, Workspace)),
    case realpath(Abs) of
        {ok, Real} ->
            case inside(Workspace, Real) of
                true -> {ok, Real};
                false -> {error, {outside_workspace, Path}}
            end;
        {error, _} = E ->
            E
    end.

%% `Path' relative to the workspace, for display.
-spec relative(binary(), binary()) -> binary().
relative(Workspace, Workspace) ->
    <<".">>;
relative(Workspace, Path) ->
    case string:prefix(Path, <<Workspace/binary, "/">>) of
        nomatch -> Path;
        Rest -> unicode:characters_to_binary(Rest)
    end.

inside(Workspace, Path) ->
    Path =:= Workspace orelse string:prefix(Path, <<Workspace/binary, "/">>) =/= nomatch.

%% Drop `.' and resolve `..' lexically.
normalize(Path) ->
    Parts = filename:split(Path),
    unicode:characters_to_binary(filename:join(lists:reverse(lists:foldl(fun step/2, [], Parts)))).

step(<<".">>, Acc) -> Acc;
step(".", Acc) -> Acc;
step(D, [Root]) when D =:= <<"..">>; D =:= ".." -> [Root];
step(D, [_ | Acc]) when D =:= <<"..">>; D =:= ".." -> Acc;
step(P, Acc) -> [P | Acc].

%% Resolve symbolic links in the existing part of `Path'; the rest
%% (a file still to be created) is appended as is.
realpath(Path) ->
    [Root | Parts] = filename:split(Path),
    walk(unicode:characters_to_binary(Root), Parts, ?MAX_LINKS).

walk(Acc, [], _N) ->
    {ok, Acc};
walk(_Acc, _Parts, 0) ->
    {error, too_many_links};
walk(Acc, [P | Rest], N) ->
    Next = unicode:characters_to_binary(filename:join(Acc, P)),
    case file:read_link_info(Next) of
        {ok, #file_info{type = symlink}} ->
            follow(Acc, Next, Rest, N);
        {ok, _} ->
            walk(Next, Rest, N);
        {error, enoent} ->
            {ok, unicode:characters_to_binary(filename:join([Next | Rest]))};
        {error, _} = E ->
            E
    end.

follow(Acc, Link, Rest, N) ->
    case file:read_link(Link) of
        {ok, Target} ->
            Resolved = normalize(filename:absname(Target, Acc)),
            [Root | Parts] = filename:split(Resolved),
            walk(unicode:characters_to_binary(Root), Parts ++ Rest, N - 1);
        {error, _} = E ->
            E
    end.
