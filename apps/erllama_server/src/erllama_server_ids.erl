%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_ids).
-moduledoc false.
%% Run and conversation identifiers: a type prefix and 64 random bits,
%% hex encoded, e.g. `<<"run_9f2c41d07ab3e655">>'.

-export([run/0, conversation/0, tool_call/0]).

-spec run() -> binary().
run() -> new(<<"run_">>).

-spec conversation() -> binary().
conversation() -> new(<<"conv_">>).

-spec tool_call() -> binary().
tool_call() -> new(<<"call_">>).

new(Prefix) ->
    <<Prefix/binary, (binary:encode_hex(crypto:strong_rand_bytes(8), lowercase))/binary>>.
