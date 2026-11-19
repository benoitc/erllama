%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_client).
-moduledoc false.
%% The client side of the service API, used by the `erllama' CLI. It
%% runs in its own short-lived VM as a hidden node, talks to the
%% release only through `erpc' calls to erllama_server_api and the
%% event messages of a run, and must never load or call an erllama
%% library module (no NIF on the client).
%%
%% Environment:
%%   ERLLAMA_NODE     release node (default erllama_server@127.0.0.1)
%%   ERLLAMA_COOKIE   distribution cookie (default: ~/.erlang.cookie)
%%   ERLLAMA_TIMEOUT  control call timeout in ms (default 10000)

-include("erllama_server_api.hrl").

-export([connect/0, connect/1, call/3, follow/4, release_node/0, client_name/1]).

-define(DEFAULT_NODE, "erllama_server@127.0.0.1").
-define(DEFAULT_TIMEOUT, 10000).
-define(ACK_EVERY, 16).
-define(RETRIES, 8).
%% With no event for this long, ask the release whether the run is
%% still going, so a silently broken connection is noticed.
-define(IDLE_CHECK, 30000).

-type sink() :: fun(
    (map() | {gap, non_neg_integer(), non_neg_integer()} | {reconnecting, term()}) -> term()
).
-export_type([sink/0]).

-spec release_node() -> node().
release_node() ->
    list_to_atom(os:getenv("ERLLAMA_NODE", ?DEFAULT_NODE)).

-spec connect() -> {ok, node(), map()} | {error, term()}.
connect() ->
    connect(release_node()).

%% Start the client node, then check the release speaks our API.
-spec connect(node()) -> {ok, node(), map()} | {error, term()}.
connect(Node) ->
    case start_client_node(Node) of
        ok ->
            case call(Node, hello, [?ERLLAMA_SERVER_API_VSN]) of
                {ok, Info} -> {ok, Node, Info};
                {error, _} = E -> E
            end;
        {error, Reason} ->
            {error, {client_node, Reason}}
    end.

%% A bounded control call. Transport failures and a release without
%% the service API come back as errors, never as exceptions.
-spec call(node(), atom(), [term()]) -> term().
call(Node, Fun, Args) ->
    try
        erpc:call(Node, erllama_server_api, Fun, Args, timeout())
    catch
        error:{erpc, noconnection} ->
            {error, {unreachable, Node}};
        error:{erpc, timeout} ->
            {error, timeout};
        error:{exception, undef, [{erllama_server_api, _, _, _} | _]} ->
            {error, {no_service_api, Node}};
        error:{exception, Reason, _} ->
            {error, {remote_error, Reason}};
        exit:{exception, Reason} ->
            {error, {remote_exit, Reason}};
        error:{erpc, Reason} ->
            {error, {rpc, Reason}}
    end.

timeout() ->
    case string:to_integer(os:getenv("ERLLAMA_TIMEOUT", "")) of
        {N, []} when N > 0 -> N;
        _ -> ?DEFAULT_TIMEOUT
    end.

%% Follow a run from sequence `Cursor' until its terminal event, which
%% is returned. Every event, a replay gap (`{gap, Wanted, Oldest}')
%% and a reconnection attempt (`{reconnecting, Reason}') go to
%% `Sink'. Being dropped for lagging, a lost connection or a hole in
%% the sequence all resume by reattaching from the next wanted event.
-spec follow(node(), binary(), non_neg_integer(), sink()) -> {ok, map()} | {error, term()}.
follow(Node, RunId, Cursor, Sink) ->
    follow(Node, RunId, Cursor, Sink, 0).

follow(Node, RunId, Cursor, Sink, Tries) ->
    case call(Node, attach, [RunId, self(), Cursor]) of
        {ok, #{run_pid := RunPid, replay := Replay, status := Status}} ->
            case consume(Replay, Cursor, Sink) of
                {terminal, E} ->
                    {ok, E};
                {next, Next} when Status =:= running ->
                    Mon = erlang:monitor(process, RunPid),
                    loop(
                        #{node => Node, run => RunId, pid => RunPid, mon => Mon, sink => Sink},
                        Next,
                        0
                    );
                {next, _} ->
                    {error, {run_ended, Status}}
            end;
        {error, {gap, Oldest}} ->
            _ = Sink({gap, Cursor, Oldest}),
            follow(Node, RunId, Oldest, Sink, Tries);
        {error, {unreachable, _} = Reason} when Tries < ?RETRIES ->
            _ = Sink({reconnecting, Reason}),
            timer:sleep(backoff(Tries)),
            follow(Node, RunId, Cursor, Sink, Tries + 1);
        {error, _} = E ->
            E
    end.

backoff(Tries) -> min(5000, 100 bsl Tries).

consume([], Next, _Sink) ->
    {next, Next};
consume([#{seq := Seq} = E | Rest], Next, Sink) when Seq >= Next ->
    _ = Sink(E),
    case terminal(E) of
        true -> {terminal, E};
        false -> consume(Rest, Seq + 1, Sink)
    end;
consume([_ | Rest], Next, Sink) ->
    consume(Rest, Next, Sink).

loop(#{run := RunId, pid := RunPid, mon := Mon, sink := Sink} = C, Next, Unacked) ->
    receive
        {erllama_server_run, RunId, #{seq := Seq} = E} when Seq =:= Next ->
            _ = Sink(E),
            case terminal(E) of
                true ->
                    erlang:demonitor(Mon, [flush]),
                    {ok, E};
                false when Unacked + 1 >= ?ACK_EVERY ->
                    RunPid ! {ack, self(), Seq},
                    loop(C, Seq + 1, 0);
                false ->
                    loop(C, Seq + 1, Unacked + 1)
            end;
        {erllama_server_run, RunId, #{seq := Seq}} when Seq < Next ->
            loop(C, Next, Unacked);
        {erllama_server_run, RunId, #{seq := _}} ->
            resume(C, Next);
        {erllama_server_run, RunId, {dropped, Seq}} ->
            resume(C, min(Seq, Next));
        {'DOWN', Mon, process, RunPid, noconnection} ->
            _ = Sink({reconnecting, noconnection}),
            resume(C, Next);
        {'DOWN', Mon, process, RunPid, Reason} ->
            {error, {run_exited, Reason}}
    after ?IDLE_CHECK ->
        case call(maps:get(node, C), run, [RunId]) of
            {ok, #{status := running}} -> loop(C, Next, Unacked);
            _ -> resume(C, Next)
        end
    end.

resume(#{node := Node, run := RunId, mon := Mon, sink := Sink}, Next) ->
    erlang:demonitor(Mon, [flush]),
    _ = call(Node, detach, [RunId, self()]),
    flush(RunId),
    follow(Node, RunId, Next, Sink, 0).

%% Discard events already in the mailbox; the reattach replays them.
flush(RunId) ->
    receive
        {erllama_server_run, RunId, _} -> flush(RunId)
    after 0 -> ok
    end.

terminal(#{type := T}) -> lists:member(T, [done, cancelled, error]).

%% A hidden node on the release node's host, so a client neither shows
%% up in the release's nodes() nor is connected to its other nodes.
%% Several clients run at once (one follows a run while another
%% cancels it), so the name must be unique across OS processes: the
%% OS pid separates live clients on this host, the random part covers
%% pid reuse, and a name still taken is retried with a new suffix.
start_client_node(Node) ->
    Started =
        case node() of
            nonode@nohost ->
                [_, Host] = string:split(atom_to_list(Node), "@"),
                start_named(Host, 3);
            _ ->
                ok
        end,
    case Started of
        ok -> set_cookie(Node, os:getenv("ERLLAMA_COOKIE"));
        {error, _} = E -> E
    end.

start_named(Host, Tries) ->
    case net_kernel:start(client_name(Host), #{name_domain => longnames, hidden => true}) of
        {ok, _} -> ok;
        {error, {already_started, _}} -> ok;
        {error, _} when Tries > 1 -> start_named(Host, Tries - 1);
        {error, _} = E -> E
    end.

-spec client_name(string()) -> atom().
client_name(Host) ->
    Suffix = binary_to_list(binary:encode_hex(crypto:strong_rand_bytes(4), lowercase)),
    list_to_atom("erllama_cli_" ++ os:getpid() ++ "_" ++ Suffix ++ "@" ++ Host).

set_cookie(_Node, false) ->
    ok;
set_cookie(Node, Cookie) ->
    true = erlang:set_cookie(Node, list_to_atom(Cookie)),
    ok.
