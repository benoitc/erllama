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

%% No children yet: the service API, model registry and listeners are
%% added by the phases that implement them.
init([]) ->
    SupFlags = #{strategy => one_for_one, intensity => 5, period => 30},
    {ok, {SupFlags, []}}.
