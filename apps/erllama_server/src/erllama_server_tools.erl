%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_tools).
-moduledoc false.
%% The native tools of agent sessions: what the model is told about
%% them (name, description, JSON schema), whether each only reads or
%% changes the workspace, the default permission, and how to run one.
%% Tool names are binaries throughout; no atom is made from a name the
%% model chose.

-export([definitions/0, names/0, class/1, default_policy/1, execute/3]).
-export([decode_args/1, stop_worker/2]).

-type class() :: read | write.
-type policy() :: auto | ask | deny.
-export_type([class/0, policy/0]).

-define(SHELL_TIMEOUT, 120000).
-define(SHELL_MAX_TIMEOUT, 1800000).

-spec names() -> [binary()].
names() ->
    [N || #{name := N} <- definitions()].

-spec definitions() -> [map()].
definitions() ->
    [
        tool(
            <<"read_file">>,
            <<"Read a text file of the workspace as numbered lines, a page at a time.">>,
            #{
                <<"path">> => str(<<"Path relative to the workspace">>),
                <<"offset">> => int(<<"First line to read (1-based)">>),
                <<"limit">> => int(<<"Number of lines (at most 400)">>)
            },
            [<<"path">>]
        ),
        tool(
            <<"list_dir">>,
            <<"List a directory of the workspace; directories end with /.">>,
            #{
                <<"path">> => str(<<"Directory relative to the workspace (default .)">>),
                <<"depth">> => int(<<"Levels to descend, 1 to 3">>)
            },
            []
        ),
        tool(
            <<"search">>,
            <<"Search the workspace's text files for a regular expression.">>,
            #{
                <<"pattern">> => str(<<"Regular expression">>),
                <<"path">> => str(<<"File or directory to search (default .)">>),
                <<"glob">> => str(<<"Only files whose name matches, e.g. *.erl">>)
            },
            [<<"pattern">>]
        ),
        tool(
            <<"write_file">>,
            <<"Create a new file. Fails if it exists: use edit_file to change a file.">>,
            #{
                <<"path">> => str(<<"Path relative to the workspace">>),
                <<"content">> => str(<<"The whole content">>)
            },
            [<<"path">>, <<"content">>]
        ),
        tool(
            <<"edit_file">>,
            <<"Replace an exact text in a file. The old text must match exactly once unless replace_all is set.">>,
            #{
                <<"path">> => str(<<"Path relative to the workspace">>),
                <<"old">> => str(<<"The exact text to replace, whitespace included">>),
                <<"new">> => str(<<"The replacement">>),
                <<"replace_all">> => #{<<"type">> => <<"boolean">>}
            },
            [<<"path">>, <<"old">>, <<"new">>]
        ),
        tool(
            <<"shell">>,
            <<"Run a shell command in the workspace and return its exit status and output.">>,
            #{
                <<"command">> => str(<<"The command, run by /bin/sh">>),
                <<"timeout_ms">> => int(<<"Kill it after this long (default 120000)">>),
                <<"background">> => #{
                    <<"type">> => <<"boolean">>,
                    <<"description">> => <<"Return at once with a job id">>
                }
            },
            [<<"command">>]
        ),
        tool(
            <<"shell_status">>,
            <<"Output so far and status of a background shell job.">>,
            #{<<"job">> => str(<<"Job id">>)},
            [<<"job">>]
        ),
        tool(
            <<"shell_kill">>,
            <<"Stop a background shell job and everything it started.">>,
            #{<<"job">> => str(<<"Job id">>)},
            [<<"job">>]
        )
    ].

tool(Name, Desc, Props, Required) ->
    #{
        name => Name,
        description => Desc,
        parameters => #{
            <<"type">> => <<"object">>, <<"properties">> => Props, <<"required">> => Required
        }
    }.

str(D) -> #{<<"type">> => <<"string">>, <<"description">> => D}.
int(D) -> #{<<"type">> => <<"integer">>, <<"description">> => D}.

-spec class(binary()) -> class().
class(N) when
    N =:= <<"write_file">>; N =:= <<"edit_file">>; N =:= <<"shell">>; N =:= <<"shell_kill">>
->
    write;
class(_) ->
    read.

%% Reading is allowed; changing the workspace or running commands asks.
-spec default_policy(binary()) -> policy().
default_policy(N) ->
    case class(N) of
        read -> auto;
        write -> ask
    end.

%% Run one tool. `Ctx': `workspace', `owner' (the session, which owns
%% background jobs) and `jobs' (`#{JobId => pid()}'). Returns the text for the model,
%% as a result or as an error it can act on, and any new job.
%% With `mcp' (the catalog entry of an MCP tool), the call goes to its
%% server.
-spec execute(binary(), map(), map()) ->
    {ok | error, binary()} | {ok, binary(), {job, binary(), pid()}}.
execute(_Name, _Args, #{mcp := #{server := Server, remote := list_resources}}) ->
    erllama_server_mcp:resources(Server);
execute(_Name, #{<<"uri">> := Uri}, #{mcp := #{server := Server, remote := read_resource}}) when
    is_binary(Uri)
->
    erllama_server_mcp:read_resource(Server, Uri);
execute(_Name, _Args, #{mcp := #{remote := read_resource}}) ->
    {error, <<"missing argument: uri">>};
execute(_Name, Args, #{mcp := #{server := Server, remote := Tool}}) when is_map(Args) ->
    erllama_server_mcp:call(Server, Tool, Args, erllama_server_config:get(mcp_timeout_ms));
execute(Name, Args, #{workspace := Ws} = Ctx) when is_map(Args) ->
    case Name of
        <<"read_file">> -> erllama_server_tool_fs:read_file(Ws, Args);
        <<"list_dir">> -> erllama_server_tool_fs:list_dir(Ws, Args);
        <<"search">> -> erllama_server_tool_fs:search(Ws, Args);
        <<"write_file">> -> erllama_server_tool_fs:write_file(Ws, Args);
        <<"edit_file">> -> erllama_server_tool_fs:edit_file(Ws, Args);
        <<"shell">> -> shell(Ws, Args, Ctx);
        <<"shell_status">> -> job(Args, Ctx, fun erllama_server_tool_shell:status/1);
        <<"shell_kill">> -> job(Args, Ctx, fun erllama_server_tool_shell:kill/1);
        _ -> {error, <<"unknown tool: ", Name/binary>>}
    end;
execute(_Name, _Args, _Ctx) ->
    {error, <<"tool arguments must be a JSON object">>}.

shell(Ws, #{<<"command">> := Cmd} = Args, Ctx) when is_binary(Cmd), Cmd =/= <<>> ->
    Timeout =
        case maps:get(<<"timeout_ms">>, Args, ?SHELL_TIMEOUT) of
            T when is_integer(T), T > 0 -> min(T, ?SHELL_MAX_TIMEOUT);
            _ -> ?SHELL_TIMEOUT
        end,
    case maps:get(<<"background">>, Args, false) of
        true ->
            Job = erllama_server_tool_shell:start(maps:get(owner, Ctx), Ws, Cmd, Timeout),
            Id = erllama_server_ids:job(),
            {ok, <<"started background job ", Id/binary, "; use shell_status to read its output">>,
                {job, Id, Job}};
        _ ->
            Result = erllama_server_tool_shell:run(Ws, Cmd, Timeout),
            Text = erllama_server_tool_shell:format(Result),
            case maps:get(exit, Result) of
                0 -> {ok, Text};
                _ -> {error, Text}
            end
    end;
shell(_Ws, _, _Ctx) ->
    {error, <<"missing argument: command">>}.

job(#{<<"job">> := Id}, #{jobs := Jobs}, Fun) ->
    case Jobs of
        #{Id := Pid} ->
            case Fun(Pid) of
                {running, R} -> {ok, <<"running\n", (erllama_server_tool_shell:format(R))/binary>>};
                {done, R} -> {ok, erllama_server_tool_shell:format(R)};
                {error, gone} -> {error, <<"the job has ended and its output is no longer kept">>}
            end;
        _ ->
            {error, <<"no such job: ", Id/binary>>}
    end;
job(_Args, _Ctx, _Fun) ->
    {error, <<"missing argument: job">>}.

%% A call's arguments as a map, whatever the model sent.
-spec decode_args(map() | binary()) -> map().
decode_args(A) when is_map(A) -> A;
decode_args(A) when is_binary(A) ->
    try json:decode(A) of
        M when is_map(M) -> M;
        _ -> #{}
    catch
        error:_ -> #{}
    end.

%% Stop the worker running a call (`{Pid, Monitor}'). A running command
%% is stopped through its job first: killing the worker alone leaves the
%% group's cleanup to happen later, or never if the release is stopping.
%% For any other call, MCP ones included, stopping the worker is enough:
%% an MCP request belongs to the worker that made it, and barrel_mcp's
%% client cancels it at the server (that request only) when that process
%% exits.
-spec stop_worker(binary(), {pid(), reference()}) -> true.
stop_worker(<<"shell">>, {Pid, Mon}) ->
    erlang:demonitor(Mon, [flush]),
    ok = erllama_server_tool_shell:stop_run(Pid),
    exit(Pid, kill);
stop_worker(_Name, {Pid, Mon}) ->
    erlang:demonitor(Mon, [flush]),
    exit(Pid, kill).
