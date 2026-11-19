%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_agent_sup).
-moduledoc false.
%% Supervises the agent sessions. They are temporary: a crashed session
%% is not restarted (its conversation keeps what was recorded).

-behaviour(supervisor).

-export([start_link/0, init/1]).

start_link() ->
    supervisor:start_link({local, ?MODULE}, ?MODULE, []).

init([]) ->
    SupFlags = #{strategy => simple_one_for_one, intensity => 10, period => 10},
    Child = #{
        id => erllama_server_agent,
        start => {erllama_server_agent, start_link, []},
        restart => temporary,
        shutdown => 10000,
        type => worker
    },
    {ok, {SupFlags, [Child]}}.
