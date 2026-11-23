%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_cli_format).
-moduledoc false.
%% How the CLI shows what the release answers: status, models, loaded
%% models, a model's details, jobs, results and download progress.

-export([
    format_status/1,
    format_models/1,
    format_ps/1,
    format_show/1,
    format_runs/1,
    format_result/1,
    progress_line/3,
    size_text/1
]).

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
