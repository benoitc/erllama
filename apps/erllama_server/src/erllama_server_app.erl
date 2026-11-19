%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_app).
-moduledoc false.
-behaviour(application).

-export([start/2, stop/1]).

start(_StartType, _StartArgs) ->
    erllama_server_sup:start_link().

stop(_State) ->
    ok.
