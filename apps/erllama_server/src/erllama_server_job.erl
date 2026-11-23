%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_job).
-moduledoc false.
%% The calls every job process (a run, a model download) answers:
%% `{attach, Sub, Cursor}', `{detach, Sub}', `cancel', `result' and
%% `info'. Each job keeps its events in erllama_server_events, so a
%% client follows a download the same way it follows a generation.

-export([attach/3, detach/2, cancel/1, result/1, info/1, call/3]).

-spec attach(pid(), pid(), non_neg_integer()) ->
    {ok, map()} | {error, {gap, non_neg_integer()} | not_found}.
attach(Pid, Sub, Cursor) -> call(Pid, {attach, Sub, Cursor}, 5000).

-spec detach(pid(), pid()) -> ok | {error, not_found}.
detach(Pid, Sub) -> call(Pid, {detach, Sub}, 5000).

-spec cancel(pid()) -> ok | {error, not_found}.
cancel(Pid) -> call(Pid, cancel, 5000).

-spec result(pid()) -> {ok, map()} | {error, running | not_found}.
result(Pid) -> call(Pid, result, 5000).

-spec info(pid()) -> {ok, map()} | {error, not_found}.
info(Pid) -> call(Pid, info, 5000).

%% A job that exits between the table lookup and the call is reported
%% as gone rather than crashing the caller.
-spec call(pid(), term(), timeout()) -> term().
call(Pid, Req, Timeout) ->
    try
        gen_server:call(Pid, Req, Timeout)
    catch
        exit:{noproc, _} -> {error, not_found};
        exit:{normal, _} -> {error, not_found}
    end.
