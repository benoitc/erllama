%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_acp_session).
-moduledoc false.
%% One follower per ACP session: follows the release agent job's event
%% stream (`erllama_server_client:follow/4', cursor 0 so a reattached
%% session replays its history) and forwards every event to the
%% dispatcher as `{acp_event, ConversationId, Event}'. The follower
%% ends when the job ends.

-export([start/5]).

-spec start(pid(), node(), binary(), binary(), non_neg_integer()) -> pid().
start(Dispatcher, Node, Conv, AgentId, Cursor) ->
    spawn(fun() ->
        _ = erllama_server_client:follow(
            Node, AgentId, Cursor, fun(E) -> Dispatcher ! {acp_event, Conv, E} end
        )
    end).
