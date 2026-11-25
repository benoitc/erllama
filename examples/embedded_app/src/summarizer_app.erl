%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
-module(summarizer_app).
-behaviour(application).

-export([start/2, stop/1]).

start(_Type, _Args) ->
    summarizer_sup:start_link().

%% The model belongs to this application: it goes when the application
%% stops.
stop(_State) ->
    _ = erllama:unload(<<"summarizer">>),
    ok.
