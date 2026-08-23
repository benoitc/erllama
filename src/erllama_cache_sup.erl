%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_cache_sup).
-moduledoc false.
-behaviour(supervisor).

-export([start_link/0]).
-export([init/1]).

-define(SERVER, ?MODULE).

start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

init([]) ->
    SupFlags = #{strategy => one_for_one, intensity => 5, period => 30},
    Children = [
        erllama_cache_meta_srv, erllama_cache_ram, erllama_cache_writer
    ],
    TierSup = #{
        id => erllama_cache_tier_sup,
        start => {erllama_cache_tier_sup, start_link, []},
        restart => permanent,
        shutdown => infinity,
        type => supervisor,
        modules => [erllama_cache_tier_sup]
    },
    Configured = #{
        id => erllama_cache_tiers_boot,
        start => {erllama_cache, start_configured_tiers, []},
        restart => transient,
        shutdown => brutal_kill,
        type => worker,
        modules => [erllama_cache]
    },
    {ok, {SupFlags, [worker(M) || M <- Children] ++ [TierSup, Configured]}}.

worker(Mod) ->
    #{
        id => Mod,
        start => {Mod, start_link, []},
        restart => permanent,
        shutdown => 5000,
        type => worker,
        modules => [Mod]
    }.
