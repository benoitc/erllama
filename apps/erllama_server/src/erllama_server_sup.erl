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

%% rest_for_one: the run table and the conversation store hold the
%% pids and locks of the runs, so when either restarts the runs are
%% restarted with it and nothing points at a stale process.
init([]) ->
    SupFlags = #{strategy => rest_for_one, intensity => 5, period => 30},
    Children = [
        worker(erllama_server_runs),
        worker(erllama_server_convs),
        #{
            id => erllama_server_run_sup,
            start => {erllama_server_run_sup, start_link, []},
            type => supervisor,
            shutdown => infinity
        }
    ],
    {ok, {SupFlags, Children}}.

worker(Mod) ->
    #{id => Mod, start => {Mod, start_link, []}, type => worker, shutdown => 5000}.
