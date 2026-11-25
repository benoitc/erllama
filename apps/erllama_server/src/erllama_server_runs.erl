%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_runs).
-moduledoc false.
%% The table of live jobs: generation runs and model downloads. Readers
%% go to ETS directly; every write is made by this process, which also
%% starts the job processes (so the `max_active_runs' check and the
%% insert are one step) and monitors them: when a job exits, its row is
%% deleted and any conversation it still held is released.

-behaviour(gen_server).

-export([
    start_link/0,
    start/1,
    lookup/1,
    list/0,
    set_status/2,
    set_phase/2,
    active_count/0,
    active_count/1
]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-define(TAB, ?MODULE).

-type status() :: running | done | cancelled | error.
-type kind() :: run | pull | agent.
%% Where a running job is: a run loads its model, may compact the
%% conversation, waits for a free sequence, then generates. Other jobs
%% stay `running'.
-type phase() :: running | loading | compacting | waiting | generating.

%% Monitor ref of each run process, to its run id.
-record(state, {mons = #{} :: #{reference() => binary()}}).
-type state() :: #state{}.
-export_type([status/0, kind/0, phase/0]).

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

%% Start a job process for `Spec' (it carries run_id, the job's id,
%% `kind' (default `run'), conversation_id and model; a download's
%% model is its source). The job is registered before this returns.
-spec start(map()) -> {ok, pid()} | {error, term()}.
start(Spec) ->
    gen_server:call(?MODULE, {start, Spec}, 30000).

-spec lookup(binary()) -> {ok, map()} | {error, not_found}.
lookup(RunId) ->
    case ets:lookup(?TAB, RunId) of
        [Row] -> {ok, to_map(Row)};
        [] -> {error, not_found}
    end.

-spec list() -> [map()].
list() ->
    lists:sort(
        fun(#{started_at := A}, #{started_at := B}) -> A =< B end,
        [to_map(R) || R <- ets:tab2list(?TAB)]
    ).

-spec set_status(binary(), status()) -> ok.
set_status(RunId, Status) ->
    gen_server:cast(?MODULE, {set_status, RunId, Status}).

-spec set_phase(binary(), phase()) -> ok.
set_phase(RunId, Phase) ->
    gen_server:cast(?MODULE, {set_phase, RunId, Phase}).

-spec active_count() -> non_neg_integer().
active_count() ->
    active_count(run).

-spec active_count(kind()) -> non_neg_integer().
active_count(Kind) ->
    ets:select_count(?TAB, [{{'_', '_', '_', '_', '_', running, Kind, '_'}, [], [true]}]).

to_map({RunId, Pid, ConvId, Model, StartedAt, Status, Kind, Phase}) ->
    #{
        run_id => RunId,
        kind => Kind,
        pid => Pid,
        conversation_id => ConvId,
        model => Model,
        started_at => StartedAt,
        status => Status,
        phase => Phase
    }.

-spec init([]) -> {ok, state()}.
init([]) ->
    _ = ets:new(?TAB, [named_table, set, protected, {read_concurrency, true}]),
    {ok, #state{}}.

handle_call({start, #{run_id := RunId} = Spec}, _From, #state{mons = Mons} = S) ->
    Kind = maps:get(kind, Spec, run),
    {Sup, Limit} = job_sup(Kind),
    case active_count(Kind) < Limit of
        false ->
            {reply, {error, too_many(Kind)}, S};
        true ->
            case supervisor:start_child(Sup, [Spec]) of
                {ok, Pid} ->
                    Row = {
                        RunId,
                        Pid,
                        maps:get(conversation_id, Spec, undefined),
                        maps:get(model, Spec),
                        erlang:system_time(millisecond),
                        running,
                        Kind,
                        running
                    },
                    true = ets:insert(?TAB, Row),
                    Mon = erlang:monitor(process, Pid),
                    {reply, {ok, Pid}, S#state{mons = Mons#{Mon => RunId}}};
                {error, _} = E ->
                    {reply, E, S}
            end
    end;
handle_call(_Req, _From, S) ->
    {reply, {error, unknown_call}, S}.

handle_cast({set_status, RunId, Status}, S) ->
    _ = ets:update_element(?TAB, RunId, {6, Status}),
    {noreply, S};
handle_cast({set_phase, RunId, Phase}, S) ->
    _ = ets:update_element(?TAB, RunId, {8, Phase}),
    {noreply, S};
handle_cast(_Msg, S) ->
    {noreply, S}.

handle_info({'DOWN', Mon, process, _Pid, Reason}, #state{mons = Mons} = S) ->
    case maps:take(Mon, Mons) of
        {RunId, Mons1} ->
            ok = job_down(RunId, Reason),
            case ets:lookup(?TAB, RunId) of
                [{_, _, ConvId, _, _, _, _, _}] when ConvId =/= undefined ->
                    release_conv(ConvId, RunId);
                _ ->
                    ok
            end,
            true = ets:delete(?TAB, RunId),
            {noreply, S#state{mons = Mons1}};
        error ->
            {noreply, S}
    end;
handle_info(_Msg, S) ->
    {noreply, S}.

%% The conversation store may be restarting; the lock it held is gone
%% with it then.
%% A job process that crashed (not one that ended).
job_down(_RunId, Reason) when Reason =:= normal; Reason =:= shutdown ->
    ok;
job_down(_RunId, {shutdown, _}) ->
    ok;
job_down(RunId, Reason) ->
    Kind =
        case ets:lookup(?TAB, RunId) of
            [{_, _, _, _, _, _, agent, _}] -> agent_down;
            _ -> run_failed
        end,
    erllama_server_failures:record(Kind, RunId, Reason).

release_conv(ConvId, RunId) ->
    try
        erllama_server_convs:release(ConvId, RunId)
    catch
        exit:_ -> ok
    end.

job_sup(run) -> {erllama_server_run_sup, erllama_server_config:get(max_active_runs)};
job_sup(pull) -> {erllama_server_pull_sup, erllama_server_config:get(max_active_pulls)};
job_sup(agent) -> {erllama_server_agent_sup, erllama_server_config:get(max_agents)}.

too_many(run) -> too_many_runs;
too_many(pull) -> too_many_pulls;
too_many(agent) -> too_many_agents.
