%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_acp_stdio).
-moduledoc false.
%% The ACP transport: JSON-RPC 2.0 messages as newline-delimited JSON on
%% standard input/output, the framing the ACP specification (and Zed,
%% OpenCode through opencode-acp-provider) use: each message is one
%% line of JSON terminated by `\n'. The reader also accepts LSP-style
%% `Content-Length' frames from clients that speak that instead. The
%% reader opens file descriptor 0 as an fd port
%% (`open_port({fd, 0, 0}, ...)'): whatever stdin is (pipe, redirected
%% file, tty), the process gets `{data, Bytes}' and `{eof}' port
%% messages, no io protocol in between - reads through the io server
%% block unpredictably on redirected devices. The writer writes one
%% frame line through the group leader device with `io:put_chars/2';
%% the caller resolves that device as group_leader(), not the
%% `standard_output' atom, which port-driven stdio refuses.

-export([start_reader/1, start_writer/1, send/2, stop_writer/1]).
-export([parse/1, frame/1, decode/1]).

%% A reader process: decodes frames from standard input and sends
%% `{frame, Map}' to `Sink'; when stdin closes it tells `Sink' `eof'.
-spec start_reader(pid()) -> pid().
start_reader(Sink) ->
    spawn_link(fun() -> read_loop(open_port({fd, 0, 1}, [binary, eof, stream, in]), Sink, <<>>) end).

read_loop(Port, Sink, Acc) ->
    %% A burst may leave several complete frames in the buffer: parse
    %% them all before waiting on the port again.
    case parse(Acc) of
        {ok, Body, Rest} ->
            _ =
                case decode(Body) of
                    {ok, Map} -> Sink ! {frame, Map};
                    {error, _} -> Sink ! {frame, {invalid_json, Body}}
                end,
            read_loop(Port, Sink, Rest);
        more ->
            receive
                {Port, {data, Bytes}} ->
                    read_loop(Port, Sink, <<Acc/binary, Bytes/binary>>);
                {Port, eof} ->
                    Sink ! eof
                %% Port messages only: data keeps the parse loop fed, eof
                %% ends the reader. There is nothing else to time on.
            after infinity -> ok
            end;
        {error, R} ->
            Sink ! {reader_error, R}
    end.

%% Split one frame off the buffer: `more' while a frame line, a header
%% block or a body is still incomplete. A `{' line is one JSON message
%% (the ACP framing). When a `Content-Length' header line arrives the
%% parser switches to LSP-style framing until the body is read: header
%% lines up to the blank one, then exactly that many body bytes. Blank
%% lines and unrecognized lines are skipped, the way ACP clients skip
%% lines they cannot parse.
-spec parse(binary()) -> {ok, binary(), binary()} | more | {error, term()}.
parse(Acc) ->
    case binary:split(Acc, <<"\n">>) of
        [Line0, Rest] ->
            case frame_line(trim_cr(Line0)) of
                {frame, Line} -> {ok, Line, Rest};
                skip -> parse(Rest);
                {body, N} -> lsp_head(Rest, N);
                {error, _} = E -> E
            end;
        [_] ->
            more
    end.

trim_cr(Line) ->
    case Line of
        <<>> ->
            Line;
        _ ->
            Size = byte_size(Line) - 1,
            case Line of
                <<Trimmed:Size/binary, "\r">> -> Trimmed;
                _ -> Line
            end
    end.

frame_line(<<>>) ->
    skip;
frame_line(<<"{", _/binary>> = Line) ->
    {frame, Line};
frame_line(Line) ->
    case binary:split(Line, <<":">>) of
        [K, V] ->
            case string:lowercase(string:trim(K)) of
                <<"content-length">> ->
                    try binary_to_integer(string:trim(V)) of
                        N when N >= 0 -> {body, N};
                        _ -> {error, invalid_content_length}
                    catch
                        _:_ -> {error, invalid_content_length}
                    end;
                _ ->
                    skip
            end;
        _ ->
            skip
    end.

%% LSP-style: header lines up to and including the blank one, then `N'
%% body bytes follow.
lsp_head(Rest, N) ->
    case binary:split(Rest, <<"\n">>) of
        [Line0, More] ->
            case trim_cr(Line0) of
                <<>> ->
                    case More of
                        <<Body:N/binary, Remain/binary>> -> {ok, Body, Remain};
                        _ -> more
                    end;
                _Header ->
                    lsp_head(More, N)
            end;
        [_] ->
            more
    end.

-spec decode(binary()) -> {ok, map()} | {error, term()}.
decode(Body) ->
    try json:decode(Body) of
        Map when is_map(Map) -> {ok, Map};
        _ -> {error, not_an_object}
    catch
        _:_ -> {error, invalid_json}
    end.

%% The writer process: `{send, Term}' writes one frame as a flat
%% binary.
-spec start_writer(file:io_device()) -> pid().
start_writer(Device) ->
    spawn_link(fun() -> write_loop(Device) end).

write_loop(Device) ->
    receive
        {send, Term} ->
            _ = io:put_chars(Device, iolist_to_binary(frame(Term))),
            write_loop(Device);
        stop ->
            ok
        %% Frames keep coming until `stop'.
    after infinity -> ok
    end.

-spec send(pid(), term()) -> ok.
send(Writer, Term) ->
    Writer ! {send, Term},
    ok.

-spec stop_writer(pid()) -> ok.
stop_writer(Writer) ->
    Writer ! stop,
    ok.

-spec frame(term()) -> iodata().
frame(Term) ->
    %% One ACP frame is one line of JSON; json:encode keeps it a line by
    %% escaping any newline inside string values.
    [iolist_to_binary(json:encode(Term)), <<"\n">>].
