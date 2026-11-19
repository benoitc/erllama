%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_app).
-moduledoc false.
-behaviour(application).

-export([start/2, stop/1]).

%% A bad application environment fails the start instead of
%% surfacing later as a runtime error.
start(_StartType, _StartArgs) ->
    case erllama_server_config:validate() of
        ok -> erllama_server_sup:start_link();
        {error, _} = E -> E
    end.

stop(_State) ->
    ok.
