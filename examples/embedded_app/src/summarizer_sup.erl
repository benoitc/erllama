%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
-module(summarizer_sup).
-behaviour(supervisor).

-export([start_link/0, init/1]).

start_link() ->
    supervisor:start_link({local, ?MODULE}, ?MODULE, []).

%% The summarizer is a child of your application like any other worker:
%% if it crashes it is restarted, and erllama cancels the generations
%% that were streaming to it.
init([]) ->
    Child = #{id => summarizer, start => {summarizer, start_link, []}},
    {ok, {#{strategy => one_for_one, intensity => 5, period => 10}, [Child]}}.
