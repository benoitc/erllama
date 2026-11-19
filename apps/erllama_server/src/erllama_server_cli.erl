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
    "  status                        versions, loaded models, active jobs\n",
    "  models                        the registry's models and what is loaded\n",
    "  pull SOURCE [--name NAME] [--sha256 HEX] [--detach]\n",
    "                                download a model: hf://ORG/REPO[:QUANT],\n",
    "                                hf://ORG/REPO/FILE.gguf, https://...gguf,\n",
    "                                ollama://NAME[:TAG] or NAME[:TAG]\n",
    "  show MODEL                    a model's entry and GGUF metadata\n",
    "  ps                            loaded models, holders and keep-alive\n",
    "  load MODEL|PATH.gguf [--name NAME] [--n-ctx N] [--gpu-layers N] [--mmproj PATH]\n",
    "                                load a model (registering a file first)\n",
    "  unload MODEL                  unload a model\n",
    "  cp SRC DST                    a second name for a model\n",
    "  rm MODEL                      remove a model (deletes downloaded files)\n",
    "  default MODEL                 the model used when none is named\n",
    "  run MODEL [--conversation ID] [--max-tokens N] [--detach] PROMPT...\n",
    "                                one chat turn, streamed\n",
    "  chat MODEL [--conversation ID] [--max-tokens N]\n",
    "                                chat line by line (/quit or end of input)\n",
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
        [
            {"--name", name, string},
            {"--id", name, string},
            {"--n-ctx", n_ctx, int},
            {"--gpu-layers", n_gpu_layers, int},
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
            follow(Node, Id, 0);
        {error, Reason} ->
            fail(Reason)
    end;
pull(_, _, _, _) ->
    usage("pull").

show(Node, _Info, _Opts, [Name]) ->
    reply(erllama_server_client:call(Node, show, [list_to_binary(Name)]), fun format_show/1);
show(_, _, _, _) ->
    usage("show").

ps(Node, _Info, _Opts, []) ->
    reply(erllama_server_client:call(Node, ps, []), fun format_ps/1);
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
    io:put_chars(Bin);
print_event(#{type := loading, data := #{model := M}}) ->
    io:format(standard_error, "[loading ~ts]~n", [M]);
print_event(#{type := progress, data := #{file := F, done := D, total := T}}) ->
    io:put_chars(standard_error, ["\r", progress_line(F, D, T)]);
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

format_models([]) ->
    "no model registered or loaded\n";
format_models(Ms) ->
    [
        io_lib:format("~ts~ts  ~ts  ~ts~ts~n", [
            maps:get(name, M),
            case maps:get(default, M, false) of
                true -> " (default)";
                false -> ""
            end,
            case maps:get(loaded, M, false) of
                true -> "loaded";
                false -> "-"
            end,
            size_text(maps:get(size, M, undefined)),
            case maps:get(status, M, ok) of
                missing -> "  FILE MISSING";
                _ -> ""
            end
        ])
     || M <- Ms
    ].

format_ps([]) ->
    "no model loaded\n";
format_ps(Ms) ->
    [
        io_lib:format("~ts  ~p  ctx ~p  ~ts  ~ts~n", [
            maps:get(id, M),
            maps:get(status, M, unknown),
            maps:get(context_size, M, 0),
            maps:get(quant_tag, M, <<"-">>),
            keep_alive_text(M)
        ])
     || M <- Ms
    ].

keep_alive_text(#{managed := true, holders := H}) when H > 0 ->
    io_lib:format("in use by ~b", [H]);
keep_alive_text(#{managed := true, expires_at := undefined, keep_alive := infinity}) ->
    "kept loaded";
keep_alive_text(#{managed := true, expires_at := E}) when is_integer(E) ->
    io_lib:format("unloads in ~bs", [max(0, (E - erlang:system_time(millisecond)) div 1000)]);
keep_alive_text(#{managed := false}) ->
    "loaded outside the registry";
keep_alive_text(_) ->
    "".

format_show(#{name := N} = E) ->
    Meta = maps:get(metadata, E, #{}),
    Line = fun(K, V) -> io_lib:format("~-14s ~ts~n", [K, value_text(V)]) end,
    [
        Line("name", N),
        Line("source", maps:get(ref, maps:get(source, E, #{}), <<"-">>)),
        Line("file", maps:get(path, E, <<"-">>)),
        Line("size", size_text(maps:get(size, E, undefined))),
        Line("sha256", maps:get(sha256, E, undefined)),
        Line("projector", maps:get(mmproj, E, undefined)),
        Line("architecture", maps:get(architecture, Meta, undefined)),
        Line("parameters", maps:get(size_label, Meta, undefined)),
        Line("quantisation", maps:get(file_type, Meta, undefined)),
        Line("context", maps:get(context_length, Meta, undefined)),
        Line("chat template", maps:get(chat_template, Meta, undefined) =/= undefined),
        Line("loaded", maps:get(loaded, E, false))
    ].

value_text(undefined) -> "-";
value_text(true) -> "yes";
value_text(false) -> "no";
value_text(V) when is_integer(V) -> integer_to_list(V);
value_text(V) when is_binary(V) -> V;
value_text(V) when is_list(V) -> V;
value_text(V) -> io_lib:format("~p", [V]).

size_text(undefined) -> "-";
size_text(B) when B >= 1 bsl 30 -> io_lib:format("~.1f GiB", [B / (1 bsl 30)]);
size_text(B) -> io_lib:format("~.1f MiB", [B / (1 bsl 20)]).

progress_line(File, Done, undefined) ->
    io_lib:format("pulling ~p: ~ts", [File, size_text(Done)]);
progress_line(File, Done, Total) when Total > 0 ->
    io_lib:format("pulling ~p: ~b% (~ts of ~ts)", [
        File, Done * 100 div Total, size_text(Done), size_text(Total)
    ]);
progress_line(File, Done, _) ->
    progress_line(File, Done, undefined).

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
        io_lib:format("~s  ~p  ~p  ~ts  ~ts~n", [
            Id,
            maps:get(kind, R, run),
            Status,
            Model,
            case Conv of
                undefined -> <<"-">>;
                _ -> Conv
            end
        ])
     || #{run_id := Id, status := Status, model := Model, conversation_id := Conv} = R <- Rs
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
