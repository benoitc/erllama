%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% A minimal stdio MCP server, run in its own `erl', that honours
%% cancellation as the specification asks: `notifications/cancelled'
%% for a request stops that call. barrel_mcp's own servers do not stop a
%% running tool on cancel, so this one stands in to show that the
%% release cancels exactly the interrupted request.
%%
%% Tools: `echo' (`text') and `effect' (`path': writes the file after a
%% second, or `<path>.cancelled' when cancelled first).
%% @end
-module(erllama_server_fake_mcp).

-export([main/0]).

main() ->
    ok = io:setopts(standard_io, [binary]),
    Self = self(),
    _ = spawn_link(fun() -> read(Self) end),
    loop(#{}).

read(Main) ->
    case io:get_line(standard_io, <<>>) of
        eof ->
            Main ! eof;
        {error, _} ->
            Main ! eof;
        Line ->
            Main ! {line, Line},
            read(Main)
    end.

loop(Calls) ->
    receive
        {line, Line} ->
            loop(handle(json:decode(string:trim(Line)), Calls));
        {finished, Id} ->
            loop(maps:remove(Id, Calls));
        eof ->
            halt(0)
    end.

handle(#{<<"method">> := <<"server/discover">>, <<"id">> := Id}, Calls) ->
    send(#{<<"id">> => Id, <<"error">> => #{<<"code">> => -32601, <<"message">> => <<"no">>}}),
    Calls;
handle(#{<<"method">> := <<"initialize">>, <<"id">> := Id}, Calls) ->
    reply(Id, #{
        <<"protocolVersion">> => <<"2025-06-18">>,
        <<"capabilities">> => #{<<"tools">> => #{}},
        <<"serverInfo">> => #{<<"name">> => <<"fake">>, <<"version">> => <<"1">>}
    }),
    Calls;
handle(#{<<"method">> := <<"tools/list">>, <<"id">> := Id}, Calls) ->
    Obj = #{<<"type">> => <<"object">>},
    reply(Id, #{
        <<"tools">> => [
            #{<<"name">> => <<"echo">>, <<"inputSchema">> => Obj},
            #{<<"name">> => <<"effect">>, <<"inputSchema">> => Obj}
        ]
    }),
    Calls;
handle(#{<<"method">> := <<"ping">>, <<"id">> := Id}, Calls) ->
    reply(Id, #{}),
    Calls;
handle(#{<<"method">> := <<"tools/call">>, <<"id">> := Id, <<"params">> := P}, Calls) ->
    call(Id, maps:get(<<"name">>, P), maps:get(<<"arguments">>, P, #{}), Calls);
handle(#{<<"method">> := <<"notifications/cancelled">>, <<"params">> := P}, Calls) ->
    Id = maps:get(<<"requestId">>, P),
    case maps:take(Id, Calls) of
        {{Pid, Path}, Rest} ->
            exit(Pid, kill),
            ok = file:write_file(<<Path/binary, ".cancelled">>, <<"cancelled">>),
            Rest;
        error ->
            Calls
    end;
handle(#{<<"id">> := Id, <<"method">> := _}, Calls) ->
    send(#{<<"id">> => Id, <<"error">> => #{<<"code">> => -32601, <<"message">> => <<"no">>}}),
    Calls;
handle(_Notification, Calls) ->
    Calls.

call(Id, <<"echo">>, Args, Calls) ->
    text(Id, maps:get(<<"text">>, Args, <<>>)),
    Calls;
call(Id, <<"effect">>, #{<<"path">> := Path}, Calls) ->
    Main = self(),
    Pid = spawn(fun() ->
        timer:sleep(1000),
        ok = file:write_file(Path, <<"done">>),
        text(Id, <<"written">>),
        Main ! {finished, Id}
    end),
    Calls#{Id => {Pid, Path}}.

text(Id, T) ->
    reply(Id, #{<<"content">> => [#{<<"type">> => <<"text">>, <<"text">> => T}]}).

reply(Id, Result) ->
    send(#{<<"id">> => Id, <<"result">> => Result}).

send(Msg) ->
    io:put_chars(standard_io, [json:encode(Msg#{<<"jsonrpc">> => <<"2.0">>}), <<"\n">>]).
