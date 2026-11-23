%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_cli_term).
-moduledoc false.
%% The CLI's terminal. When stdin is a terminal, the CLI runs with `+Bc'
%% (rel/erllama) and this process keeps the terminal in raw mode with a
%% key reader, so Ctrl-C is a key: it runs the current handler
%% (interrupt/2), by default leaving with status 130. A prompt switches
%% to cooked mode (line editing) for the line, with the reader paused.
%% An interactive command (chat, agent) works in cooked mode between
%% turns (interactive/0) and in raw mode while a turn streams, so what is
%% typed at a prompt is never read by the key reader.
%%
%% Every write goes through this process: in raw mode `\n' becomes
%% `\r\n', and a status line (`[thinking 4s]') is shown once the output
%% has been quiet for a second, and cleared by the next write.
%%
%% When stdin is not a terminal, all of this is a pass-through.

-export([
    start/0, interactive/0, finish/0, out/1, err/1, prompt/1, interrupt/2, status/1, dim/1, tty/0
]).
-export([crlf/1, safe_put/2]).

-define(NAME, erllama_cli_term).
-define(QUIET_MS, 1000).
-define(TICK_MS, 250).
-define(CLEAR, "\r\e[2K").

-record(t, {
    tty :: boolean(),
    raw = false :: boolean(),
    reader :: pid() | undefined,
    handler = default :: default | fun(() -> term()),
    hits = 0 :: non_neg_integer(),
    status :: {iodata(), integer()} | undefined,
    shown = false :: boolean(),
    last_out = 0 :: integer()
}).

%% Take the terminal (once, at the start of a command).
-spec start() -> tty | plain.
start() ->
    case whereis(?NAME) of
        undefined ->
            Tty = is_terminal(),
            Gl = group_leader(),
            Pid = spawn(fun() ->
                group_leader(Gl, self()),
                init(Tty)
            end),
            true = register(?NAME, Pid),
            call(sync);
        _ ->
            ok
    end,
    case tty() of
        true -> tty;
        false -> plain
    end.

is_terminal() ->
    case proplists:get_value(terminal, io:getopts(standard_io), false) of
        true -> true;
        _ -> false
    end.

-spec tty() -> boolean().
tty() ->
    case whereis(?NAME) of
        undefined -> false;
        _ -> call(tty)
    end.

%% Cooked mode as the resting state (chat, agent): raw mode only while
%% interrupt/2 runs.
-spec interactive() -> ok.
interactive() ->
    case whereis(?NAME) of
        undefined ->
            ok;
        _ ->
            _ = restore(false),
            ok
    end.

%% Give the terminal back (cooked mode, no reader); before the final
%% output and before leaving.
-spec finish() -> ok.
finish() ->
    case whereis(?NAME) of
        undefined -> ok;
        _ -> call(finish)
    end.

-spec out(iodata()) -> ok.
out(Data) -> write(standard_io, Data).

-spec err(iodata()) -> ok.
err(Data) -> write(standard_error, Data).

write(Dev, Data) ->
    case whereis(?NAME) of
        undefined -> safe_put(Dev, Data);
        _ -> call({write, Dev, Data})
    end.

%% Reasoning, dimmed in a terminal.
-spec dim(iodata()) -> ok.
dim(Data) ->
    case tty() of
        true -> out(["\e[2m", Data, "\e[0m"]);
        false -> ok
    end.

%% A line from the user, in cooked mode (line editing).
-spec prompt(iodata()) -> string() | eof | {error, term()}.
prompt(Prompt) ->
    case whereis(?NAME) of
        undefined ->
            io:get_line(Prompt);
        _ ->
            Was = call(cooked),
            try
                io:get_line(Prompt)
            after
                restore(Was)
            end
    end.

restore(true) -> call(raw);
restore(false) -> call(cooked).

%% Run `Fun' with `Handler' as what the first Ctrl-C does; a second one
%% leaves the CLI (status 130). Returns what `Fun' returns.
-spec interrupt(fun(() -> term()), fun(() -> Result)) -> Result.
interrupt(Handler, Fun) ->
    case whereis(?NAME) of
        undefined ->
            Fun();
        _ ->
            Was = call(raw),
            Prev = call({handler, Handler}),
            try
                Fun()
            after
                _ = call({handler, Prev}),
                ok = status(undefined),
                _ = restore(Was)
            end
    end.

%% The status line's label (`undefined' to clear it).
-spec status(iodata() | undefined) -> ok.
status(Label) ->
    case whereis(?NAME) of
        undefined -> ok;
        _ -> call({status, Label})
    end.

%% `\n' to `\r\n' (raw mode turns output processing off).
-spec crlf(iodata()) -> binary().
crlf(Data) ->
    B = unicode:characters_to_binary(Data),
    binary:replace(B, <<"\n">>, <<"\r\n">>, [global]).

call(Req) ->
    Ref = make_ref(),
    ?NAME ! {Req, self(), Ref},
    receive
        {Ref, Reply} -> Reply
    after 10000 -> ok
    end.

%% -----------------------------------------------------------------------------
%% The terminal process
%% -----------------------------------------------------------------------------

init(Tty) ->
    T = #t{tty = Tty},
    _ = erlang:send_after(?TICK_MS, self(), tick),
    loop(
        case Tty of
            true -> raw(T);
            false -> T
        end
    ).

loop(T) ->
    receive
        {Req, From, Ref} when is_pid(From), is_reference(Ref) ->
            {Reply, T1} = request(Req, T),
            From ! {Ref, Reply},
            loop(T1);
        {key, 3} ->
            loop(ctrl_c(T));
        tick ->
            _ = erlang:send_after(?TICK_MS, self(), tick),
            loop(tick(T));
        _ ->
            loop(T)
    after 60000 ->
        loop(T)
    end.

request(sync, T) ->
    {ok, T};
request(tty, T) ->
    {T#t.tty, T};
request({write, Dev, Data}, T) ->
    {ok, do_write(Dev, Data, T)};
request(cooked, T) ->
    {T#t.raw, cooked(clear(T))};
request(raw, T) ->
    {T#t.raw, raw(T)};
request(finish, T) ->
    {ok, (cooked(clear(T)))#t{tty = false}};
request({handler, H}, T) ->
    {T#t.handler, T#t{handler = H, hits = 0}};
request({status, undefined}, T) ->
    {ok, clear(T#t{status = undefined})};
request({status, Label}, T) ->
    {ok, T#t{status = {Label, now_ms()}}};
request(_, T) ->
    {{error, unknown}, T}.

%% Ctrl-C: the handler once, then leave.
ctrl_c(#t{handler = default} = T) ->
    leave(T);
ctrl_c(#t{hits = 0, handler = H} = T) ->
    _ = spawn(H),
    do_write(standard_error, "\n[interrupting; Ctrl-C again to leave]\n", T#t{hits = 1});
ctrl_c(T) ->
    leave(T).

-spec leave(#t{}) -> no_return().
leave(T) ->
    _ = cooked(clear(T)),
    io:put_chars(standard_error, "\n"),
    erlang:halt(130).

do_write(standard_io, Data, #t{raw = true} = T) ->
    %% Raw mode: standard output is not translated (standard error is).
    T1 = clear(T),
    ok = safe_put(standard_io, crlf(Data)),
    T1#t{last_out = now_ms()};
do_write(Dev, Data, #t{raw = true} = T) ->
    T1 = clear(T),
    ok = safe_put(Dev, Data),
    T1#t{last_out = now_ms()};
do_write(Dev, Data, T) ->
    ok = safe_put(Dev, Data),
    T#t{last_out = now_ms()}.

%% Output that nobody reads any more (`erllama attach RUN | head'):
%% the reader is done, like a detach. Leave quietly; the release keeps
%% what it was doing.
-spec safe_put(io:device(), iodata()) -> ok.
safe_put(Dev, Data) ->
    try
        io:put_chars(Dev, Data)
    catch
        error:_ -> output_closed()
    end.

-spec output_closed() -> no_return().
output_closed() ->
    erlang:halt(0, [{flush, false}]).

tick(#t{raw = true, status = {Label, Since}} = T) ->
    Now = now_ms(),
    case Now - max(T#t.last_out, Since) >= ?QUIET_MS of
        true ->
            Secs = (Now - Since) div 1000,
            io:put_chars(standard_error, [
                ?CLEAR, "\e[2m[", Label, io_lib:format(" ~bs]", [Secs]), "\e[0m"
            ]),
            T#t{shown = true};
        false ->
            T
    end;
tick(T) ->
    T.

clear(#t{shown = true} = T) ->
    io:put_chars(standard_error, ?CLEAR),
    T#t{shown = false};
clear(T) ->
    T.

raw(#t{tty = true, raw = false} = T) ->
    ok = shell:start_interactive({noshell, raw}),
    Self = self(),
    Reader = spawn(fun() -> read_keys(Self) end),
    T#t{raw = true, reader = Reader};
raw(T) ->
    T.

cooked(#t{raw = true, reader = R} = T) ->
    exit(R, kill),
    ok = shell:start_interactive({noshell, cooked}),
    T#t{raw = false, reader = undefined};
cooked(T) ->
    T.

read_keys(Term) ->
    case io:get_chars("", 1) of
        [C] ->
            Term ! {key, C},
            read_keys(Term);
        _ ->
            ok
    end.

now_ms() -> erlang:monotonic_time(millisecond).
