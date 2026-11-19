%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_sup).
-moduledoc false.
-behaviour(supervisor).

-export([start_link/0]).
-export([init/1]).

-define(SERVER, ?MODULE).

start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

%% rest_for_one, in dependency order: the registry, then the loaded
%% models (holders are runs), the job table and the conversation store
%% (they hold the pids and locks of the jobs), then the jobs. When one
%% restarts, everything after it restarts too, so nothing points at a
%% stale process. The HTTP listener comes last: it only takes requests
%% once everything it calls is up.
init([]) ->
    SupFlags = #{strategy => rest_for_one, intensity => 5, period => 30},
    Children = [
        worker(erllama_server_registry),
        worker(erllama_server_models),
        worker(erllama_server_runs),
        worker(erllama_server_convs),
        supervisor(erllama_server_run_sup),
        supervisor(erllama_server_pull_sup),
        worker(erllama_server_http)
    ],
    {ok, {SupFlags, Children}}.

supervisor(Mod) ->
    #{id => Mod, start => {Mod, start_link, []}, type => supervisor, shutdown => infinity}.

worker(Mod) ->
    #{id => Mod, start => {Mod, start_link, []}, type => worker, shutdown => 5000}.
