%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_app).
-moduledoc false.
-behaviour(application).

-export([start/2, prep_stop/1, stop/1]).

-define(STOP_SLACK_MS, 1000).

start(_StartType, _StartArgs) ->
    ok = erllama_cache_counters:init(),
    %% Enable scheduler_wall_time so callers can observe per-scheduler
    %% busy ratio via erlang:statistics(scheduler_wall_time_all). The
    %% server's metrics module uses this to expose
    %% `erllama_dirty_cpu_scheduler_util' so operators can
    %% see when the dirty CPU pool is the bottleneck for autoparser
    %% calls. Small per-scheduler overhead; on inference workloads
    %% (most time inside NIFs) the cost is well under 1%.
    _ = erlang:system_flag(scheduler_wall_time, true),
    erllama_sup:start_link().

%% Before the supervision tree goes down, fire a `shutdown` save on
%% every loaded model so the live context of in-flight requests
%% survives a graceful stop. Models save in parallel, all bounded by
%% `evict_save_timeout_ms`.
prep_stop(State) ->
    save_live_state(erllama_registry:all()),
    State.

stop(_State) ->
    ok.

save_live_state(Models) ->
    Timeout = application:get_env(erllama, evict_save_timeout_ms, 30000),
    Savers = [
        spawn_monitor(fun() -> _ = erllama_model:shutdown(Pid) end)
     || {_Id, Pid} <- Models
    ],
    Deadline = erlang:monotonic_time(millisecond) + Timeout + ?STOP_SLACK_MS,
    await_savers(Savers, Deadline).

await_savers([], _Deadline) ->
    ok;
await_savers([{Pid, Ref} | Rest], Deadline) ->
    Left = max(0, Deadline - erlang:monotonic_time(millisecond)),
    receive
        {'DOWN', Ref, process, Pid, _} -> await_savers(Rest, Deadline)
    after Left ->
        [exit(P, kill) || {P, _} <- [{Pid, Ref} | Rest]],
        ok
    end.
