%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_cli_acp).
-moduledoc false.
%% The `erllama acp' command: run the release's agent sessions as an
%% ACP (Agent Client Protocol) agent on standard input/output, the way
%% vibe-acp does, so editors and agent clients (Zed, OpenCode, ...) can
%% drive local models. The client spawns this command with the
%% project's directory as its working directory; `session/new' turns
%% that `cwd' into the session's workspace. Standard output carries
%% only framed protocol messages; diagnostics stay on standard error.

-export([acp/4]).

-spec acp(node(), map(), map(), [string()]) -> {non_neg_integer(), iodata()}.
acp(Node, _Info, Opts, _Pos) ->
    Pid = erllama_server_acp:start(Node, defaults(Opts), #{caller => self()}),
    Mon = erlang:monitor(process, Pid),
    receive
        {acp_done, Pid} ->
            {0, ""};
        {acp_failed, Pid, Reason} ->
            {1, erllama_server_cli:format_error(Reason)};
        {'DOWN', Mon, process, Pid, Reason} ->
            {1, erllama_server_cli:format_error({acp_crashed, Reason})}
    %% The dispatcher answers with `acp_done'/`acp_failed' or exits
    %% through the monitor; the CLI stays up while the agent runs.
    after
        infinity -> ok
    end.

defaults(Opts) ->
    D0 = #{},
    D1 =
        case maps:get(model, Opts, undefined) of
            undefined -> D0;
            M -> D0#{model => unicode:characters_to_binary(M)}
        end,
    D2 =
        case maps:get(workspace, Opts, undefined) of
            undefined -> D1;
            W -> D1#{workspace => unicode:characters_to_binary(W)}
        end,
    case {maps:is_key(yes, Opts), maps:is_key(read_only, Opts)} of
        {true, _} -> D2#{policy => yes};
        {_, true} -> D2#{policy => read_only};
        _ -> D2
    end.
