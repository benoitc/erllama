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

-define(USAGE, [
    "usage: erllama <command> [options]\n",
    "\n",
    "commands:\n",
    "  ping                          check that the release is running\n",
    "  status                        versions, loaded models, active runs\n",
    "  models                        list the loaded models\n",
    "  load PATH [--id ID] [--n-ctx N] [--gpu-layers N]\n",
    "                                load a local GGUF file\n",
    "  unload ID                     unload a model\n",
    "  run MODEL [--conversation ID] [--max-tokens N] [--detach] PROMPT...\n",
    "                                one chat turn, streamed\n",
    "  chat MODEL [--conversation ID] [--max-tokens N]\n",
    "                                chat line by line (/quit or end of input)\n",
    "  runs                          list the runs the release holds\n",
    "  attach RUN [--from SEQ]       stream a run from an event sequence\n",
    "  cancel RUN                    cancel a run\n",
    "  result RUN                    print a finished run's result\n",
    "  --version                     print the client version\n",
    "\n",
    "The release node is ERLLAMA_NODE (default erllama_server@127.0.0.1);\n",
    "the cookie is ERLLAMA_COOKIE or ~/.erlang.cookie.\n"
]).

-spec main([string()]) -> no_return().
main(Args) ->
    {Code, Output} = run(Args),
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
        [{"--id", id, string}, {"--n-ctx", n_ctx, int}, {"--gpu-layers", n_gpu_layers, int}],
        fun load/4
    };
command("unload") ->
    {[], fun unload/4};
command("run") ->
    {[{"--detach", detach, flag} | turn_opts()], fun run_turn/4};
command("chat") ->
    {turn_opts(), fun chat/4};
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
    reply(erllama_server_client:call(Node, status, []), fun format_status/1);
status(_, _, _, _) ->
    usage("status").

models(Node, _Info, _Opts, []) ->
    reply(erllama_server_client:call(Node, models, []), fun format_models/1);
models(_, _, _, _) ->
    usage("models").

load(Node, _Info, Opts, [Path]) ->
    LoadOpts = maps:map(
        fun
            (id, V) -> list_to_binary(V);
            (_, V) -> V
        end,
        Opts
    ),
    reply(
        erllama_server_client:call(Node, load_model, [list_to_binary(Path), LoadOpts]),
        fun(Id) -> ["loaded ", Id, "\n"] end
    );
load(_, _, _, _) ->
    usage("load").

unload(Node, _Info, _Opts, [Id]) ->
    reply(erllama_server_client:call(Node, unload_model, [list_to_binary(Id)]), fun(_) -> "" end);
unload(_, _, _, _) ->
    usage("unload").

run_turn(Node, _Info, Opts, [Model | [_ | _] = Words]) ->
    Prompt = unicode:characters_to_binary(lists:join(" ", Words)),
    case start_turn(Node, Model, Prompt, Opts) of
        {ok, #{run_id := RunId}} when is_map_key(detach, Opts) ->
            {0, [RunId, "\n"]};
        {ok, #{run_id := RunId}} ->
            follow(Node, RunId, 0);
        {error, Reason} ->
            fail(Reason)
    end;
run_turn(_, _, _, _) ->
    usage("run").

chat(Node, _Info, Opts, [Model]) ->
    chat_loop(Node, Model, Opts);
chat(_, _, _, _) ->
    usage("chat").

%% `Opts' carries the conversation id once the first turn created it.
chat_loop(Node, Model, Opts) ->
    case io:get_line("> ") of
        eof ->
            {0, ""};
        {error, _} ->
            {0, ""};
        Line ->
            case string:trim(Line) of
                "/quit" -> {0, ""};
                "" -> chat_loop(Node, Model, Opts);
                Text -> chat_turn(Node, Model, Opts, unicode:characters_to_binary(Text))
            end
    end.

chat_turn(Node, Model, Opts, Text) ->
    case start_turn(Node, Model, Text, Opts) of
        {ok, #{run_id := RunId, conversation_id := Conv}} ->
            {_, Out} = follow(Node, RunId, 0),
            io:put_chars(Out),
            chat_loop(Node, Model, Opts#{conversation_id => Conv});
        {error, Reason} ->
            {_, Out} = fail(Reason),
            io:put_chars(standard_error, Out),
            chat_loop(Node, Model, Opts)
    end.

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
    reply(erllama_server_client:call(Node, runs, []), fun format_runs/1);
runs(_, _, _, _) ->
    usage("runs").

attach(Node, _Info, Opts, [RunId]) ->
    follow(Node, list_to_binary(RunId), maps:get(from, Opts, 0));
attach(_, _, _, _) ->
    usage("attach").

cancel(Node, _Info, _Opts, [RunId]) ->
    reply(erllama_server_client:call(Node, cancel, [list_to_binary(RunId)]), fun(_) -> "" end);
cancel(_, _, _, _) ->
    usage("cancel").

result(Node, _Info, _Opts, [RunId]) ->
    reply(erllama_server_client:call(Node, result, [list_to_binary(RunId)]), fun format_result/1);
result(_, _, _, _) ->
    usage("result").

%% -----------------------------------------------------------------------------
%% Streaming
%% -----------------------------------------------------------------------------

follow(Node, RunId, Cursor) ->
    case erllama_server_client:follow(Node, RunId, Cursor, fun print_event/1) of
        {ok, #{type := done}} -> {0, "\n"};
        {ok, #{type := cancelled}} -> {1, "\n[cancelled]\n"};
        {ok, #{type := error, data := #{error := R}}} -> {1, ["\n", format_error({run_failed, R})]};
        {error, Reason} -> fail(Reason)
    end.

print_event(#{type := token, data := Bin}) ->
    io:put_chars(Bin);
print_event({gap, Wanted, Oldest}) ->
    io:format(standard_error, "~n[events ~b to ~b are no longer held by the release]~n", [
        Wanted, Oldest - 1
    ]);
print_event({reconnecting, _}) ->
    io:put_chars(standard_error, "\n[connection lost, reconnecting]\n");
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

format_status(S) ->
    #{node := Node, server_vsn := Sv, erllama_vsn := Ev, api := Api, active_runs := N, models := Ms} =
        S,
    [
        io_lib:format("release   ~s~n", [Node]),
        io_lib:format("versions  erllama_server ~s, erllama ~s, api ~b~n", [Sv, Ev, Api]),
        io_lib:format("runs      ~b active~n", [N]),
        format_pressure(maps:get(pressure, S, undefined)),
        "models\n",
        case Ms of
            [] -> "  (none loaded)\n";
            _ -> [["  ", L] || L <- model_lines(Ms)]
        end
    ].

format_pressure(#{used_b := U, total_b := T, source := Src}) when T > 0 ->
    io_lib:format("memory    ~b of ~b MiB used (~p)~n", [U bsr 20, T bsr 20, Src]);
format_pressure(_) ->
    "".

format_models([]) -> "no model loaded\n";
format_models(Ms) -> model_lines(Ms).

model_lines(Ms) ->
    [
        io_lib:format("~s  ~p  ctx ~p  ~s~n", [
            maps:get(id, M),
            maps:get(status, M, unknown),
            maps:get(context_size, M, 0),
            maps:get(quant_tag, M, <<"-">>)
        ])
     || M <- Ms
    ].

format_runs([]) ->
    "no runs\n";
format_runs(Rs) ->
    [
        io_lib:format("~s  ~p  ~s  ~s~n", [
            Id,
            Status,
            Model,
            case Conv of
                undefined -> <<"-">>;
                _ -> Conv
            end
        ])
     || #{run_id := Id, status := Status, model := Model, conversation_id := Conv} <- Rs
    ].

format_result(#{status := error, error := R}) ->
    io_lib:format("error: ~p~n", [R]);
format_result(#{content := C} = R) ->
    #{finish_reason := F, usage := #{prompt_tokens := P, completion_tokens := G}} = R,
    [C, io_lib:format("~n[~p, ~b prompt tokens, ~b generated]~n", [F, P, G])].

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
