%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_cli_status).
-moduledoc false.
%% How `erllama status' and `erllama inspect' show the release: what
%% runs, what waits, what failed and what kept working. The data comes
%% from erllama_server_api:status/0 and inspect/1.

-export([format_status/1, format_inspect/1]).

-spec format_status(map()) -> iodata().
format_status(S) ->
    #{node := Node, server_vsn := Sv, erllama_vsn := Ev, api := Api} = S,
    [
        io_lib:format("release   ~s~n", [Node]),
        io_lib:format("versions  erllama_server ~s, erllama ~s, api ~b~n", [Sv, Ev, Api]),
        pressure(maps:get(pressure, S, undefined)),
        section("models", [model_line(M) || M <- maps:get(models, S, [])], "none loaded"),
        runs_line(maps:get(runs, S, #{})),
        section("agents", [agent_line(A) || A <- maps:get(agents, S, [])], "none running"),
        section("mcp", [mcp_line(M) || M <- maps:get(mcp, S, [])], "no server configured"),
        cache_line(maps:get(cache, S, #{})),
        section("failures", [failure_line(F) || F <- maps:get(failures, S, [])], "none")
    ].

section(Title, [], Empty) ->
    io_lib:format("~-9s (~s)~n", [Title, Empty]);
section(Title, Lines, _Empty) ->
    [Title, "\n", indented(Lines)].

indented(Lines) -> [["  ", L, "\n"] || L <- Lines].

pressure(#{used_b := U, total_b := T, source := Src}) when T > 0 ->
    io_lib:format("memory    ~b of ~b MiB used (~p)~n", [U bsr 20, T bsr 20, Src]);
pressure(_) ->
    "".

model_line(M) ->
    Seqs =
        case M of
            #{n_seq_max := N, available_seqs := A} when is_integer(N), is_integer(A) ->
                io_lib:format("~b of ~b sequences busy", [N - A, N]);
            _ ->
                "-"
        end,
    Waiting =
        case maps:get(waiting, M, 0) of
            0 -> "";
            W -> io_lib:format(", ~b waiting for one", [W])
        end,
    io_lib:format("~ts  ~p  ctx ~p  ~s~s", [
        maps:get(id, M), maps:get(status, M, unknown), maps:get(context_size, M, 0), Seqs, Waiting
    ]).

runs_line(Phases) when map_size(Phases) =:= 0 ->
    "runs      (none running)\n";
runs_line(Phases) ->
    Total = lists:sum(maps:values(Phases)),
    Parts = [
        io_lib:format("~b ~s", [N, phase_text(P)])
     || P := N <- maps:iterator(Phases, ordered)
    ],
    io_lib:format("runs      ~b running: ~s~n", [Total, lists:join(", ", Parts)]).

phase_text(waiting) -> "waiting for a sequence";
phase_text(loading) -> "loading the model";
phase_text(P) -> atom_to_list(P).

agent_line(#{run_id := Id, state := State, conversation_id := Conv} = A) ->
    Extra = [
        T
     || {K, Fmt} <- [
            {approvals, "~b approval(s) pending"},
            {tools, "~b tool call(s) running"},
            {jobs, "~b background job(s)"},
            {queued, "~b message(s) queued"}
        ],
        N <- [maps:get(K, A, 0)],
        N > 0,
        T <- [io_lib:format(Fmt, [N])]
    ],
    io_lib:format("~ts  ~p  ~ts~s", [Id, State, Conv, [[", ", E] || E <- Extra]]).

mcp_line(#{name := N, status := ready} = M) ->
    io_lib:format("~ts  ready, ~b tool(s)", [N, maps:get(tools, M, 0)]);
mcp_line(#{name := N, status := Status} = M) ->
    case maps:get(error, M, undefined) of
        undefined -> io_lib:format("~ts  ~p", [N, Status]);
        E -> io_lib:format("~ts  ~p: ~0tP", [N, Status, E, 8])
    end.

cache_line(C) when map_size(C) =:= 0 ->
    "";
cache_line(C) ->
    G = fun(K) -> maps:get(K, C, 0) end,
    Hits = G(hits_exact) + G(hits_resume) + G(hits_longest_prefix),
    Saves = G(saves_cold) + G(saves_continued) + G(saves_finish),
    io_lib:format(
        "cache     ~b hit(s), ~b miss(es), ~b save(s), ~b eviction(s); ~s in RAM, ~s on disk~n",
        [
            Hits,
            G(misses),
            Saves,
            G(evictions),
            erllama_server_cli_format:size_text(G(bytes_ram)),
            erllama_server_cli_format:size_text(G(bytes_disk))
        ]
    ).

failure_line(#{at := At, kind := K, subject := Sub, reason := R, outcome := O}) ->
    io_lib:format("~s  ~p  ~ts: ~ts~n    ~ts", [clock(At), K, Sub, R, O]).

clock(Ms) ->
    {_, {H, Mi, Se}} = calendar:system_time_to_local_time(Ms, millisecond),
    io_lib:format("~2..0b:~2..0b:~2..0b", [H, Mi, Se]).

-spec format_inspect(map()) -> iodata().
format_inspect(#{kind := agent} = A) ->
    [
        common(A),
        line("state", maps:get(state, A, undefined)),
        line("workspace", maps:get(workspace, A, undefined)),
        turn(maps:get(turn, A, undefined), maps:get(budgets, A, #{})),
        line("latest turn", seq_text(maps:get(turn_seq, A, undefined))),
        list("approvals", [approval(P) || P <- maps:get(approvals, A, [])]),
        list("tools", [
            io_lib:format("~ts ~ts (~p)", [N, Id, St])
         || #{call_id := Id, name := N, status := St} <- maps:get(tools, A, [])
        ]),
        list("jobs", [
            io_lib:format("~ts ~s", [J, running_text(R)])
         || #{job_id := J, running := R} <- maps:get(jobs, A, [])
        ]),
        line("queued", maps:get(queued, A, 0))
    ];
format_inspect(R) ->
    [common(R), line("phase", maps:get(phase, R, undefined))].

common(R) ->
    [
        line("id", maps:get(run_id, R, undefined)),
        line("kind", maps:get(kind, R, undefined)),
        line("status", maps:get(status, R, undefined)),
        line("model", maps:get(model, R, undefined)),
        line("conversation", maps:get(conversation_id, R, undefined)),
        line("started", started(maps:get(started_at, R, undefined))),
        line("next event", maps:get(next_seq, R, undefined))
    ].

turn(undefined, _Budgets) ->
    line("turn", "none running");
turn(#{rounds := R, tokens := T, ms_left := Left}, B) ->
    line(
        "turn",
        io_lib:format("~b of ~p rounds, ~b of ~p tokens, ~bs left", [
            R, maps:get(max_rounds, B, '-'), T, maps:get(max_tokens, B, '-'), Left div 1000
        ])
    ).

approval(#{call_id := Id, name := N, arguments := Args}) ->
    Text = iolist_to_binary(json:encode(Args)),
    Short =
        case byte_size(Text) > 120 of
            true -> <<(binary:part(Text, 0, 120))/binary, "...">>;
            false -> Text
        end,
    io_lib:format("~ts ~ts ~ts", [N, Id, Short]).

seq_text(undefined) -> "-";
seq_text(Seq) -> io_lib:format("from event ~b (erllama attach ID --from ~b)", [Seq, Seq]).

running_text(true) -> "running";
running_text(false) -> "ended".

started(undefined) -> undefined;
started(Ms) -> clock(Ms).

line(_Label, undefined) ->
    "";
line(Label, V) when is_binary(V); is_list(V) ->
    io_lib:format("~-13s ~ts~n", [Label, V]);
line(Label, V) ->
    io_lib:format("~-13s ~p~n", [Label, V]).

list(Label, []) ->
    line(Label, "none");
list(Label, Items) ->
    [io_lib:format("~-13s~n", [Label]), indented(Items)].
