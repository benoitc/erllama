%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_cli).
-moduledoc false.
%% The `erllama' command-line client, built as an escript and shipped
%% beside the release (`bin/erllama' runs it on the release's own
%% runtime). Commands go to the running release through
%% erllama_server_client; nothing here starts the erllama application
%% or loads its NIF.

-export([main/1, run/1, parse_opts/2, format_error/1]).
%% For erllama_server_cli_agent.
-export([reply/2, fail/1, usage/1, to_bin/1, print_event/1, follow/3, follow_cancelling/2]).

-define(USAGE, [
    "usage: erllama <command> [options]\n",
    "\n",
    "commands:\n",
    "  ping                          check that the release is running\n",
    "  status                        versions, loaded models, active jobs\n",
    "  models                        the registry's models and what is loaded\n",
    "  pull SOURCE [--name NAME] [--sha256 HEX] [--detach]\n",
    "                                download a model: hf://ORG/REPO[:QUANT],\n",
    "                                hf://ORG/REPO/FILE.gguf, https://...gguf,\n",
    "                                ollama://NAME[:TAG] or NAME[:TAG]\n",
    "  show MODEL                    a model's entry and GGUF metadata\n",
    "  ps                            loaded models, holders and keep-alive\n",
    "  load MODEL|PATH.gguf [--name NAME] [--n-ctx N] [--gpu-layers N] [--parallel N] [--mmproj PATH]\n",
    "                                load a model (registering a file first)\n",
    "  unload MODEL                  unload a model\n",
    "  cp SRC DST                    a second name for a model\n",
    "  rm MODEL                      remove a model (deletes downloaded files)\n",
    "  default MODEL                 the model used when none is named\n",
    "  run MODEL [--conversation ID] [--max-tokens N] [--detach] PROMPT...\n",
    "                                one chat turn, streamed\n",
    "  chat MODEL [--conversation ID] [--max-tokens N]\n",
    "                                chat line by line; /help lists the commands\n",
    "  agent MODEL --workspace PATH | --conversation ID [--yes | --read-only] [--max-rounds N]\n",
    "                                work with tools in PATH on the release machine;\n",
    "                                /help lists the commands of a session\n",
    "  agents                        list the agent sessions\n",
    "  agent-stop AGENT              end an agent session\n",
    "  mcp [tools | resources SERVER | read SERVER URI | prompts SERVER\n",
    "      | prompt SERVER NAME [KEY=VALUE ...]]\n",
    "                                the MCP servers agent sessions use\n",
    "  conversations                 list the conversations (kept across restarts)\n",
    "  conversation show|export|fork|compact|rm|attach ID [--format json|md]\n",
    "                                read, export, copy, shorten or delete a\n",
    "                                conversation, or follow what runs in it\n",
    "  runs                          list the runs and downloads the release holds\n",
    "  attach JOB [--from SEQ]       follow a run or download from an event sequence\n",
    "  cancel JOB                    cancel a run or download\n",
    "  result JOB                    print a finished run's or download's result\n",
    "  --version                     print the client version\n",
    "\n",
    "The release node is ERLLAMA_NODE (default erllama_server@127.0.0.1);\n",
    "the cookie is ERLLAMA_COOKIE or ~/.erlang.cookie.\n"
]).

-spec main([string()]) -> no_return().
main(Args) ->
    %% A terminal: Ctrl-C is handled here (see erllama_server_cli_term).
    _ = erllama_server_cli_term:start(),
    {Code, Output} = run(Args),
    ok = erllama_server_cli_term:finish(),
    io:put_chars(Output),
    halt(Code).

%% Run one command and return its exit code and remaining output, so
%% commands can be tested without halting the VM. Streaming commands
%% print as they go. Exit codes: 0 success, 1 failure, 2 usage.
-spec run([string()]) -> {non_neg_integer(), iodata()}.
run(["--version"]) ->
    {0, ["erllama ", version(), "\n"]};
run(["help"]) ->
    {0, ?USAGE};
run([]) ->
    {0, ?USAGE};
run([Cmd | Args]) ->
    case command(Cmd) of
        undefined ->
            {2, ["erllama: unknown command: ", lists:join(" ", [Cmd | Args]), "\n", ?USAGE]};
        {Spec, Fun} ->
            case parse_opts(Args, Spec) of
                {ok, Opts, Pos} -> with_release(fun(Node, Info) -> Fun(Node, Info, Opts, Pos) end);
                {error, Msg} -> {2, ["erllama ", Cmd, ": ", Msg, "\n"]}
            end
    end.

command("ping") ->
    {[], fun ping/4};
command("status") ->
    {[], fun status/4};
command("models") ->
    {[], fun models/4};
command("load") ->
    {
        [
            {"--name", name, string},
            {"--id", name, string},
            {"--n-ctx", n_ctx, int},
            {"--gpu-layers", n_gpu_layers, int},
            {"--parallel", parallel, int},
            {"--mmproj", mmproj, string}
        ],
        fun load/4
    };
command("pull") ->
    {
        [{"--name", name, string}, {"--sha256", sha256, string}, {"--detach", detach, flag}],
        fun pull/4
    };
command("show") ->
    {[], fun show/4};
command("ps") ->
    {[], fun ps/4};
command("cp") ->
    {[], fun cp/4};
command("rm") ->
    {[], fun rm/4};
command("default") ->
    {[], fun default/4};
command("unload") ->
    {[], fun unload/4};
command("run") ->
    {[{"--detach", detach, flag} | turn_opts()], fun run_turn/4};
command("chat") ->
    {turn_opts(), fun erllama_server_cli_chat:chat/4};
command("agent") ->
    {
        [
            {"--workspace", workspace, string},
            {"--conversation", conversation_id, string},
            {"--yes", yes, flag},
            {"--read-only", read_only, flag},
            {"--max-rounds", max_rounds, int}
        ],
        fun erllama_server_cli_agent:agent/4
    };
command("agents") ->
    {[], fun erllama_server_cli_agent:agents/4};
command("agent-stop") ->
    {[], fun erllama_server_cli_agent:stop/4};
command("mcp") ->
    {[], fun erllama_server_cli_mcp:mcp/4};
command("conversations") ->
    {[], fun erllama_server_cli_conv:list/4};
command("conversation") ->
    {[{"--format", format, string}], fun erllama_server_cli_conv:conversation/4};
command("runs") ->
    {[], fun runs/4};
command("attach") ->
    {[{"--from", from, int}], fun attach/4};
command("cancel") ->
    {[], fun cancel/4};
command("result") ->
    {[], fun result/4};
command(_) ->
    undefined.

turn_opts() ->
    [{"--conversation", conversation_id, string}, {"--max-tokens", response_tokens, int}].

with_release(Fun) ->
    case erllama_server_client:connect() of
        {ok, Node, Info} -> Fun(Node, Info);
        {error, Reason} -> fail(Reason)
    end.

%% -----------------------------------------------------------------------------
%% Commands. Each gets the release node, its hello info, the parsed
%% options and the positional arguments.
%% -----------------------------------------------------------------------------

ping(Node, _Info, _Opts, []) ->
    {0, ["pong ", atom_to_list(Node), "\n"]};
ping(_, _, _, _) ->
    usage("ping").

status(Node, _Info, _Opts, []) ->
    reply(
        erllama_server_client:call(Node, status, []), fun erllama_server_cli_format:format_status/1
    );
status(_, _, _, _) ->
    usage("status").

models(Node, _Info, _Opts, []) ->
    reply(
        erllama_server_client:call(Node, models, []), fun erllama_server_cli_format:format_models/1
    );
models(_, _, _, _) ->
    usage("models").

%% An argument ending in `.gguf' is a file on the release machine, to
%% register and load; anything else is a registry name.
load(Node, _Info, Opts, [Target]) ->
    LoadOpts = maps:map(
        fun
            (K, V) when K =:= name; K =:= mmproj -> list_to_binary(V);
            (_, V) -> V
        end,
        Opts
    ),
    Call =
        case filename:extension(Target) of
            ".gguf" -> {load_model, [list_to_binary(Target), LoadOpts]};
            _ -> {load, [list_to_binary(Target), LoadOpts]}
        end,
    {F, A} = Call,
    reply(erllama_server_client:call(Node, F, A), fun(Id) -> ["loaded ", Id, "\n"] end);
load(_, _, _, _) ->
    usage("load").

unload(Node, _Info, _Opts, [Id]) ->
    reply(erllama_server_client:call(Node, unload, [list_to_binary(Id)]), fun(_) -> "" end);
unload(_, _, _, _) ->
    usage("unload").

pull(Node, _Info, Opts, [Source]) ->
    PullOpts = maps:map(fun(_, V) -> list_to_binary(V) end, maps:with([name, sha256], Opts)),
    case erllama_server_client:call(Node, pull, [list_to_binary(Source), PullOpts]) of
        {ok, #{job_id := Id}} when is_map_key(detach, Opts) ->
            {0, [Id, "\n"]};
        {ok, #{job_id := Id}} ->
            %% Ctrl-C cancels the download; the part file is kept.
            follow_cancelling(Node, Id);
        {error, Reason} ->
            fail(Reason)
    end;
pull(_, _, _, _) ->
    usage("pull").

show(Node, _Info, _Opts, [Name]) ->
    reply(
        erllama_server_client:call(Node, show, [list_to_binary(Name)]),
        fun erllama_server_cli_format:format_show/1
    );
show(_, _, _, _) ->
    usage("show").

ps(Node, _Info, _Opts, []) ->
    reply(erllama_server_client:call(Node, ps, []), fun erllama_server_cli_format:format_ps/1);
ps(_, _, _, _) ->
    usage("ps").

cp(Node, _Info, _Opts, [Src, Dst]) ->
    reply(
        erllama_server_client:call(Node, copy, [list_to_binary(Src), list_to_binary(Dst)]),
        fun(#{name := N}) -> ["copied to ", N, "\n"] end
    );
cp(_, _, _, _) ->
    usage("cp").

rm(Node, _Info, _Opts, [Name]) ->
    reply(erllama_server_client:call(Node, remove, [list_to_binary(Name)]), fun(_) -> "" end);
rm(_, _, _, _) ->
    usage("rm").

default(Node, _Info, _Opts, [Name]) ->
    reply(erllama_server_client:call(Node, set_default, [list_to_binary(Name)]), fun(_) -> "" end);
default(_, _, _, _) ->
    usage("default").

run_turn(Node, _Info, Opts, [Model | [_ | _] = Words]) ->
    Prompt = unicode:characters_to_binary(lists:join(" ", Words)),
    case start_turn(Node, Model, Prompt, Opts) of
        {ok, #{run_id := RunId}} when is_map_key(detach, Opts) ->
            {0, [RunId, "\n"]};
        {ok, #{run_id := RunId}} ->
            follow_cancelling(Node, RunId);
        {error, Reason} ->
            fail(Reason)
    end;
run_turn(_, _, _, _) ->
    usage("run").

start_turn(Node, Model, Text, Opts) ->
    Spec0 = #{
        model => list_to_binary(Model),
        messages => [#{role => user, content => Text}],
        opts => maps:with([response_tokens], Opts)
    },
    Spec =
        case Opts of
            #{conversation_id := C} -> Spec0#{conversation_id => to_bin(C)};
            _ -> Spec0
        end,
    erllama_server_client:call(Node, start_run, [Spec]).

runs(Node, _Info, _Opts, []) ->
    reply(erllama_server_client:call(Node, runs, []), fun erllama_server_cli_format:format_runs/1);
runs(_, _, _, _) ->
    usage("runs").

attach(Node, _Info, Opts, [RunId]) ->
    %% Ctrl-C detaches: the run goes on.
    erllama_server_cli_term:interrupt(
        fun detach/0,
        fun() -> follow(Node, list_to_binary(RunId), maps:get(from, Opts, 0)) end
    );
attach(_, _, _, _) ->
    usage("attach").

-spec detach() -> no_return().
detach() ->
    erllama_server_cli_term:err("[detached; the run goes on]\n"),
    ok = erllama_server_cli_term:finish(),
    halt(0).

cancel(Node, _Info, _Opts, [RunId]) ->
    reply(erllama_server_client:call(Node, cancel, [list_to_binary(RunId)]), fun(_) -> "" end);
cancel(_, _, _, _) ->
    usage("cancel").

result(Node, _Info, _Opts, [RunId]) ->
    reply(
        erllama_server_client:call(Node, result, [list_to_binary(RunId)]),
        fun erllama_server_cli_format:format_result/1
    );
result(_, _, _, _) ->
    usage("result").

%% -----------------------------------------------------------------------------
%% Streaming
%% -----------------------------------------------------------------------------

%% Follow a job to its end; Ctrl-C cancels it.
-spec follow_cancelling(node(), binary()) -> {non_neg_integer(), iodata()}.
follow_cancelling(Node, Id) ->
    erllama_server_cli_term:interrupt(
        fun() -> erllama_server_client:call(Node, cancel, [Id]) end,
        fun() -> follow(Node, Id, 0) end
    ).

follow(Node, RunId, Cursor) ->
    case erllama_server_client:follow(Node, RunId, Cursor, fun print_event/1) of
        {ok, #{type := done, data := #{model := #{name := Name}}}} ->
            {0, ["\npulled ", Name, "\n"]};
        {ok, #{type := done}} ->
            {0, "\n"};
        {ok, #{type := cancelled}} ->
            {1, "\n[cancelled]\n"};
        {ok, #{type := error, data := #{error := R}}} ->
            {1, ["\n", format_error({run_failed, R})]};
        {error, Reason} ->
            fail(Reason)
    end.

print_event(#{type := token, data := Bin}) ->
    erllama_server_cli_term:out(Bin);
print_event(#{type := thinking, data := Bin}) ->
    erllama_server_cli_term:dim(Bin);
print_event(#{type := started}) ->
    erllama_server_cli_term:status("waiting");
print_event(#{type := loading, data := #{model := M}}) ->
    ok = erllama_server_cli_term:status("loading"),
    erllama_server_cli_term:err(io_lib:format("[loading ~ts]~n", [M]));
print_event(#{type := compacting}) ->
    ok = erllama_server_cli_term:status("compacting"),
    erllama_server_cli_term:err("[compacting the conversation]\n");
print_event(#{type := compacted, data := #{before := B, 'after' := A}}) ->
    ok = erllama_server_cli_term:status("prefill"),
    erllama_server_cli_term:err(io_lib:format("[compacted: ~b messages, now ~b]~n", [B, A]));
print_event(#{type := tool_call, data := #{name := N}}) ->
    erllama_server_cli_term:err(["\n[call ", N, "]\n"]);
print_event(#{type := progress, data := #{file := F, done := D, total := T}}) ->
    erllama_server_cli_term:err(["\r", erllama_server_cli_format:progress_line(F, D, T)]);
print_event({gap, Wanted, Oldest}) ->
    erllama_server_cli_term:err(
        io_lib:format("~n[events ~b to ~b are no longer held by the release]~n", [
            Wanted, Oldest - 1
        ])
    );
print_event({reconnecting, _}) ->
    erllama_server_cli_term:err("\n[connection lost, reconnecting]\n");
print_event(_) ->
    ok.

%% -----------------------------------------------------------------------------
%% Output
%% -----------------------------------------------------------------------------

reply(ok, Fun) -> {0, Fun(ok)};
reply({ok, V}, Fun) -> {0, Fun(V)};
reply({error, Reason}, _Fun) -> fail(Reason).

fail(Reason) -> {1, format_error(Reason)}.

usage(Cmd) -> {2, ["erllama ", Cmd, ": wrong arguments\n", ?USAGE]}.

-spec format_error(term()) -> iodata().
format_error(Reason) -> ["erllama: ", error_text(Reason), "\n"].

error_text({unreachable, Node}) ->
    [
        "release not reachable at ",
        atom_to_list(Node),
        " (is it running, and is the cookie the same?)"
    ];
error_text({no_service_api, Node}) ->
    ["the release at ", atom_to_list(Node), " has no erllama service API (an older release?)"];
error_text({incompatible_api, #{server := S, client := C}}) ->
    io_lib:format(
        "the release speaks API version ~p and this client ~p; use the erllama command shipped with the release",
        [S, C]
    );
error_text({client_node, Reason}) ->
    io_lib:format("cannot start a client node: ~p (is epmd running?)", [Reason]);
error_text(Reason) ->
    request_error(Reason).

request_error({model_not_loaded, M}) ->
    ["model ", to_bin(M), " is not loaded (see erllama models)"];
request_error(not_found) ->
    "no such run (finished runs are kept for a while, then dropped)";
request_error(running) ->
    "the run has not finished; use erllama attach to follow it";
request_error(conversation_busy) ->
    "the conversation already has a run in progress";
request_error(too_many_runs) ->
    "the release is running its maximum number of runs; try again later";
request_error(timeout) ->
    "the release did not answer in time (ERLLAMA_TIMEOUT)";
request_error({enoent, Path}) ->
    ["no such file: ", Path];
request_error({run_failed, R}) ->
    io_lib:format("run failed: ~p", [R]);
request_error(Other) ->
    io_lib:format("~p", [Other]).

%% -----------------------------------------------------------------------------
%% Argument parsing
%% -----------------------------------------------------------------------------

%% Split `Args' into the options of `Spec' (`{Flag, Key, string | int | flag}')
%% and the positional arguments. `--' ends the options.
-spec parse_opts([string()], [{string(), atom(), string | int | flag}]) ->
    {ok, map(), [string()]} | {error, iodata()}.
parse_opts(Args, Spec) ->
    parse_opts(Args, Spec, #{}, []).

parse_opts([], _Spec, Opts, Pos) ->
    {ok, Opts, lists:reverse(Pos)};
parse_opts(["--" | Rest], _Spec, Opts, Pos) ->
    {ok, Opts, lists:reverse(Pos, Rest)};
parse_opts(["--" ++ _ = Flag | Rest], Spec, Opts, Pos) ->
    case lists:keyfind(Flag, 1, Spec) of
        {_, Key, flag} ->
            parse_opts(Rest, Spec, Opts#{Key => true}, Pos);
        {_, Key, Type} ->
            case Rest of
                [V | Rest1] ->
                    case value(Type, V) of
                        {ok, Val} -> parse_opts(Rest1, Spec, Opts#{Key => Val}, Pos);
                        error -> {error, [Flag, " needs ", atom_to_list(Type), " value"]}
                    end;
                [] ->
                    {error, [Flag, " needs a value"]}
            end;
        false ->
            {error, ["unknown option ", Flag]}
    end;
parse_opts([A | Rest], Spec, Opts, Pos) ->
    parse_opts(Rest, Spec, Opts, [A | Pos]).

value(string, V) ->
    {ok, V};
value(int, V) ->
    case string:to_integer(V) of
        {N, []} when N >= 0 -> {ok, N};
        _ -> error
    end.

to_bin(B) when is_binary(B) -> B;
to_bin(L) when is_list(L) -> unicode:characters_to_binary(L);
to_bin(A) when is_atom(A) -> atom_to_binary(A).

version() ->
    _ = application:load(erllama_server),
    case application:get_key(erllama_server, vsn) of
        {ok, Vsn} -> Vsn;
        undefined -> "unknown"
    end.
