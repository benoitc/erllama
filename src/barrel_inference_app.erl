%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(barrel_inference_app).
-behaviour(application).

-export([start/2, stop/1]).

start(_StartType, _StartArgs) ->
    ok = barrel_inference_cache_counters:init(),
    %% Enable scheduler_wall_time so callers can observe per-scheduler
    %% busy ratio via erlang:statistics(scheduler_wall_time_all). The
    %% server's metrics module uses this to expose
    %% `barrel_inference_dirty_cpu_scheduler_util' so operators can
    %% see when the dirty CPU pool is the bottleneck for autoparser
    %% calls. Small per-scheduler overhead; on inference workloads
    %% (most time inside NIFs) the cost is well under 1%.
    _ = erlang:system_flag(scheduler_wall_time, true),
    barrel_inference_sup:start_link().

stop(_State) ->
    ok.
