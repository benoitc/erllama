%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_pull_sup).
-moduledoc false.
%% Supervises the model downloads. They are temporary: a crashed pull is
%% reported to its subscribers by its exit; its part file is kept.

-behaviour(supervisor).

-export([start_link/0, init/1]).

start_link() ->
    supervisor:start_link({local, ?MODULE}, ?MODULE, []).

init([]) ->
    SupFlags = #{strategy => simple_one_for_one, intensity => 10, period => 10},
    Child = #{
        id => erllama_server_pull,
        start => {erllama_server_pull, start_link, []},
        restart => temporary,
        shutdown => 5000,
        type => worker
    },
    {ok, {SupFlags, [Child]}}.
