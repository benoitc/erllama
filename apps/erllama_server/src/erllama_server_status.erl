%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_status).
-moduledoc false.
%% The release's runtime view, behind erllama_server_api:status/0 and
%% inspect/1 (and so `erllama status', `erllama inspect' and
%% `GET /status'): what runs, what waits for a sequence, what an agent
%% session is doing, and what failed. It reads the job table, the
%% models' observability rows, the MCP manager and the failure log, and
%% asks each agent session for its summary; nothing here polls or keeps
%% state.

-export([status/3, inspect/1, agent_summary/2]).

-define(CACHE_COUNTERS, [
    hits_exact,
    hits_resume,
    hits_longest_prefix,
    misses,
    saves_cold,
    saves_continued,
    saves_finish,
    evictions,
    bytes_ram,
    bytes_disk
]).

-spec status(pos_integer(), binary(), binary()) -> map().
status(Api, ServerVsn, ErllamaVsn) ->
    #{
        api => Api,
        server_vsn => ServerVsn,
        erllama_vsn => ErllamaVsn,
        node => node(),
        models => erllama_server_models:ps(),
        active_runs => erllama_server_runs:active_count(run),
        active_pulls => erllama_server_runs:active_count(pull),
        runs => run_phases(),
        agents => [agent_brief(A) || A <- agent_summaries()],
        mcp => erllama_server_mcp:status(),
        cache => maps:with(?CACHE_COUNTERS, erllama:counters()),
        pressure => pressure(),
        vram => vram(),
        failures => erllama_server_failures:recent(5),
        failure_counts => erllama_server_failures:counts()
    }.

-spec inspect(binary()) -> {ok, map()} | {error, term()}.
inspect(Id) ->
    case erllama_server_runs:lookup(Id) of
        {ok, #{pid := Pid, started_at := At} = Row} ->
            %% A phase only while the job runs.
            Keys =
                [run_id, kind, conversation_id, model, status] ++
                    [phase || maps:get(status, Row) =:= running],
            Base = (maps:with(Keys, Row))#{started_at => At},
            case catch_info(Pid) of
                {ok, Info} -> {ok, maps:merge(Base, Info)};
                {error, _} -> {ok, Base}
            end;
        {error, _} = E ->
            E
    end.

%% An agent session's summary from its raw state (erllama_server_agent
%% gives the fields, this shapes them): calls as `{Status, Id, Name,
%% Args}', jobs as job id to pid, the turn's counters and deadline.
-spec agent_summary(atom(), map()) -> map().
agent_summary(State, #{calls := Calls, jobs := Jobs, turn := Turn} = Raw) ->
    (maps:without([calls, jobs, turn], Raw))#{
        state => State,
        approvals => [#{call_id => Id, name => N, arguments => A} || {pending, Id, N, A} <- Calls],
        tools => [
            #{call_id => Id, name => N, status => St}
         || {St, Id, N, _} <- Calls, St =:= queued orelse St =:= running
        ],
        jobs => [#{job_id => J, running => is_process_alive(P)} || J := P <- Jobs],
        turn => turn_info(Turn)
    }.

turn_info(undefined) ->
    undefined;
turn_info(#{rounds := R, tokens := T, deadline := Dl}) ->
    #{rounds => R, tokens => T, ms_left => max(0, Dl - erlang:monotonic_time(millisecond))}.

run_phases() ->
    lists:foldl(
        fun(#{phase := P}, Acc) -> maps:update_with(P, fun(N) -> N + 1 end, 1, Acc) end,
        #{},
        [R || #{kind := run, status := running} = R <- erllama_server_runs:list()]
    ).

agent_summaries() ->
    [
        S
     || #{kind := agent, status := running, pid := P} <- erllama_server_runs:list(),
        {ok, S} <- [catch_info(P)]
    ].

catch_info(Pid) ->
    try
        erllama_server_job:info(Pid)
    catch
        exit:_ -> {error, gone}
    end.

agent_brief(#{approvals := Ap, tools := T, jobs := J} = S) ->
    (maps:with([run_id, conversation_id, model, state, queued], S))#{
        approvals => length(Ap),
        tools => length(T),
        jobs => length([X || #{running := true} = X <- J])
    }.

pressure() ->
    case erllama:pressure() of
        {ok, P} -> P;
        {error, _} -> undefined
    end.

vram() ->
    try erllama:vram_info() of
        {ok, V} -> V;
        _ -> undefined
    catch
        _:_ -> undefined
    end.
