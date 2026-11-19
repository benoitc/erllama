%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% MCP servers as tools of agent sessions, against real servers: a
%% Streamable HTTP one served by livery_mcp in this node and a stdio
%% one in an `erl' subprocess. Sessions use the scripted model
%% (erllama_server_agent_script). Covers namespaced tools in the
%% catalog, calls and their results, server and session policies,
%% tool errors, timeouts, a server that is down or goes away, and the
%% release's own opt-in `/mcp' endpoint.
%% @end
-module(erllama_server_mcp_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([all/0, init_per_testcase/2, end_per_testcase/2]).
-export([
    http_tools_in_a_session/1,
    stdio_server/1,
    policies/1,
    errors_and_timeouts/1,
    server_down_and_back/1,
    endpoint_serves_the_release/1,
    interrupt_cancels_the_remote_call/1,
    aliases_stay_with_their_tool/1,
    bad_server_stays_contained/1,
    resources_and_prompts/1
]).
%% Tools of the test servers.
-export([echo/1, fail/1, slow/1, picture/1, touch/1, named/1, review/1, notes/1]).

all() ->
    [
        http_tools_in_a_session,
        stdio_server,
        policies,
        errors_and_timeouts,
        server_down_and_back,
        endpoint_serves_the_release,
        interrupt_cancels_the_remote_call,
        aliases_stay_with_their_tool,
        bad_server_stays_contained,
        resources_and_prompts
    ].

init_per_testcase(TC, Config) ->
    Base = filename:join(?config(priv_dir, Config), atom_to_list(TC)),
    Ws = filename:join(Base, "ws"),
    ok = filelib:ensure_path(Ws),
    ok = application:set_env(erllama_server, data_dir, filename:join(Base, "data")),
    {ok, _} = application:ensure_all_started(barrel_mcp),
    Config1 = [{ws, list_to_binary(Ws)} | Config],
    setup(TC, Config1).

setup(endpoint_serves_the_release, Config) ->
    ok = application:set_env(erllama_server, http, #{port => 0}),
    ok = application:set_env(erllama_server, mcp_endpoint, true),
    ok = application:set_env(erllama_server, api_keys, ["k1"]),
    boot(Config);
setup(bad_server_stays_contained, Config) ->
    %% Not started here: the case starts the MCP manager alone.
    Config;
setup(interrupt_cancels_the_remote_call, Config) ->
    ok = application:set_env(erllama_server, mcp_servers, [
        #{
            name => "local",
            transport => stdio,
            command => os:find_executable("erl"),
            args => fake_args(),
            policy => auto
        }
    ]),
    %% Long, so the client's own timeout is not what cancels the call.
    ok = application:set_env(erllama_server, mcp_timeout_ms, 30000),
    boot(Config);
setup(stdio_server, Config) ->
    ok = application:set_env(erllama_server, mcp_servers, [
        #{
            name => "local",
            transport => stdio,
            command => os:find_executable("erl"),
            args => stdio_args(),
            policy => auto
        }
    ]),
    boot(Config);
setup(_TC, Config) ->
    {Pid, Port} = test_server(),
    Url = "http://127.0.0.1:" ++ integer_to_list(Port) ++ "/mcp",
    ok = application:set_env(erllama_server, mcp_servers, [
        #{name => "web", transport => http, url => Url, policy => auto},
        #{name => "guarded", transport => http, url => Url},
        #{name => "closed", transport => http, url => Url, policy => deny},
        #{name => "nowhere", transport => http, url => "http://127.0.0.1:1/mcp"}
    ]),
    ok = application:set_env(erllama_server, mcp_timeout_ms, 1000),
    boot([{mcp_server, Pid}, {mcp_port, Port} | Config]).

boot(Config) ->
    {ok, _} = application:ensure_all_started(erllama_server),
    {ok, _} = erllama:load_model(<<"stub">>, #{backend => erllama_model_stub}),
    Config.

end_per_testcase(_TC, Config) ->
    _ = application:stop(erllama_server),
    [erllama:unload(Id) || #{id := Id} <- erllama:list_models()],
    case ?config(mcp_server, Config) of
        undefined -> ok;
        Pid -> stop_server(Pid)
    end,
    [
        barrel_mcp:unreg_tool(N)
     || N <- [
            <<"echo">>,
            <<"fail">>,
            <<"slow">>,
            <<"picture">>,
            <<"touch">>,
            <<"effect">>,
            <<"a.b">>,
            <<"a_b">>
        ]
    ],
    _ = barrel_mcp:unreg_prompt(<<"review">>),
    _ = barrel_mcp:unreg_resource(<<"notes">>),
    [barrel_mcp:unreg_tool(N) || N <- [<<"erllama_models">>, <<"erllama_chat">>]],
    [
        application:unset_env(erllama_server, K)
     || K <- [data_dir, http, mcp_endpoint, api_keys, mcp_servers, mcp_timeout_ms]
    ],
    ok.

%% -----------------------------------------------------------------------------
%% Test servers
%% -----------------------------------------------------------------------------

test_server() ->
    ok = barrel_mcp:reg_tool(<<"echo">>, ?MODULE, echo, #{
        description => <<"Echo the text.">>,
        input_schema => #{
            <<"type">> => <<"object">>,
            <<"properties">> => #{<<"text">> => #{<<"type">> => <<"string">>}}
        },
        annotations => #{<<"readOnlyHint">> => true}
    }),
    ok = barrel_mcp:reg_tool(<<"fail">>, ?MODULE, fail, #{description => <<"Always fails.">>}),
    ok = barrel_mcp:reg_tool(<<"slow">>, ?MODULE, slow, #{description => <<"Takes 5 s.">>}),
    ok = barrel_mcp:reg_tool(<<"picture">>, ?MODULE, picture, #{
        description => <<"A picture and a caption.">>
    }),
    ok = barrel_mcp:reg_tool(<<"touch">>, ?MODULE, touch, #{description => <<"Changes things.">>}),
    ok = barrel_mcp:reg_prompt(<<"review">>, ?MODULE, review, #{
        description => <<"Review a file.">>,
        arguments => [#{name => <<"file">>, description => <<"The file.">>, required => true}]
    }),
    ok = barrel_mcp:reg_resource(<<"notes">>, ?MODULE, notes, #{
        name => <<"Notes">>, uri => <<"notes://today">>, description => <<"Today's notes">>
    }),
    {ok, Pid} = livery:start_service(#{
        http => #{ip => {127, 0, 0, 1}, port => 0}, router => livery_mcp:router()
    }),
    #{h1 := [Port | _]} = livery:which_listeners(Pid),
    {Pid, Port}.

stop_server(Pid) ->
    _ = catch livery:stop_service(Pid),
    ok.

echo(#{<<"text">> := T}) -> T;
echo(_) -> <<"(nothing)">>.

fail(_) -> {tool_error, <<"it broke">>}.

slow(_) ->
    timer:sleep(5000),
    <<"late">>.

picture(_) ->
    [
        <<"a red dot">>,
        #{<<"type">> => <<"image">>, <<"mimeType">> => <<"image/png">>, <<"data">> => <<"AAAA">>}
    ].

touch(_) -> <<"changed">>.

named(_) -> <<"named">>.

review(#{<<"file">> := F}) ->
    #{
        description => <<"Review">>,
        messages => [
            #{
                role => <<"user">>,
                content => #{<<"type">> => <<"text">>, <<"text">> => <<"Please review ", F/binary>>}
            }
        ]
    }.

notes(_) -> <<"buy milk">>.

%% An `erl' running the fake server (erllama_server_fake_mcp).
fake_args() ->
    Dir = filename:dirname(code:which(erllama_server_fake_mcp)),
    ["-noshell", "-pa", Dir, "-eval", "erllama_server_fake_mcp:main()."].

%% An `erl' that serves `echo' over stdio, with nothing else on stdout.
stdio_args() ->
    Root = code:root_dir(),
    Paths = lists:append([
        ["-pa", P]
     || P <- code:get_path(), filelib:is_dir(P), not lists:prefix(Root, P)
    ]),
    Eval =
        "logger:remove_handler(default),"
        "{ok, _} = application:ensure_all_started(barrel_mcp),"
        "ok = barrel_mcp:reg_tool(<<\"echo\">>, erllama_server_mcp_SUITE, echo,"
        " #{description => <<\"Echo the text.\">>}),"
        "barrel_mcp:start_stdio(), halt().",
    ["-noshell" | Paths] ++ ["-eval", Eval].

%% -----------------------------------------------------------------------------
%% Helpers
%% -----------------------------------------------------------------------------

start(Config, Script) ->
    start(Config, Script, #{}).

start(Config, Script, Extra) ->
    Spec = maps:merge(
        #{
            model => <<"stub">>,
            workspace => ?config(ws, Config),
            generator => erllama_server_agent_script
        },
        Extra
    ),
    {ok, #{agent_id := Id}} = erllama_server_api:agent_start(Spec),
    ok = erllama_server_agent_script:set(Id, Script),
    Id.

turn(Id, Text) ->
    ok = erllama_server_api:agent_send(Id, Text),
    collect_turn(Id).

collect_turn(Id) ->
    {ok, #{replay := Replay}} = erllama_server_api:attach(Id, self(), cursor(Id)),
    Events = until_turn_done(Id, Replay),
    ok = erllama_server_api:detach(Id, self()),
    flush(Id),
    put({cursor, Id}, maps:get(seq, lists:last(Events)) + 1),
    Events.

cursor(Id) ->
    case get({cursor, Id}) of
        undefined -> 0;
        C -> C
    end.

until_turn_done(Id, Acc) ->
    case lists:splitwith(fun(#{type := T}) -> T =/= turn_done end, Acc) of
        {Before, [Done | _]} ->
            Before ++ [Done];
        {_, []} ->
            receive
                {erllama_server_run, Id, #{} = E} -> until_turn_done(Id, Acc ++ [E])
            after 30000 -> ct:fail({no_turn_done, Acc})
            end
    end.

flush(Id) ->
    receive
        {erllama_server_run, Id, _} -> flush(Id)
    after 0 -> ok
    end.

%% The tool results the model was shown in its last round.
last_tool_results(Id) ->
    Last = lists:last(erllama_server_agent_script:rounds(Id)),
    [{N, C} || #{role := <<"tool">>, name := N, content := C} <- Last].

wait_ready(Names) ->
    wait(fun() ->
        Ready = [N || #{name := N, status := ready} <- erllama_server_mcp:status()],
        lists:all(fun(N) -> lists:member(N, Ready) end, Names)
    end).

wait(Fun) -> wait(Fun, 500).
wait(_Fun, 0) ->
    ct:fail(condition_not_reached);
wait(Fun, N) ->
    case Fun() of
        true ->
            ok;
        false ->
            timer:sleep(20),
            wait(Fun, N - 1)
    end.

pending_requests(Id) ->
    {ok, #{replay := Replay}} = erllama_server_api:attach(Id, self(), 0),
    ok = erllama_server_api:detach(Id, self()),
    flush(Id),
    Asked = [C || #{type := approval_request, data := #{call_id := C}} <- Replay],
    Started = [C || #{type := tool_started, data := #{call_id := C}} <- Replay],
    Asked -- Started.

%% -----------------------------------------------------------------------------
%% Cases
%% -----------------------------------------------------------------------------

http_tools_in_a_session(Config) ->
    ok = wait_ready([<<"web">>, <<"guarded">>, <<"closed">>]),
    {ok, Tools} = erllama_server_api:mcp_tools(),
    Names = [N || #{name := N} <- Tools],
    ?assert(lists:member(<<"web__echo">>, Names)),
    ?assertMatch([#{class := read, server := <<"web">>, remote := <<"echo">>}], [
        T
     || #{name := <<"web__echo">>} = T <- Tools
    ]),
    ?assertMatch([#{class := write}], [T || #{name := <<"web__touch">>} = T <- Tools]),
    Id = start(Config, [
        #{
            calls => [
                {<<"c1">>, <<"web__echo">>, #{<<"text">> => <<"hello mcp">>}},
                {<<"c2">>, <<"web__picture">>, #{}}
            ]
        },
        #{text => <<"done">>}
    ]),
    Events = turn(Id, <<"use the server">>),
    ?assertMatch(#{data := #{reason := stop}}, lists:last(Events)),
    [Offered | _] = erllama_server_agent_script:tools(Id),
    ?assert(lists:member(<<"web__echo">>, Offered)),
    ?assert(lists:member(<<"read_file">>, Offered)),
    %% A server set to deny offers nothing; the others do.
    ?assertNot(lists:member(<<"closed__echo">>, Offered)),
    ?assert(lists:member(<<"guarded__echo">>, Offered)),
    [{<<"web__echo">>, Echo}, {<<"web__picture">>, Pic}] = last_tool_results(Id),
    ?assertEqual(<<"hello mcp">>, Echo),
    ?assertNotEqual(nomatch, binary:match(Pic, <<"a red dot">>)),
    ?assertNotEqual(nomatch, binary:match(Pic, <<"[image: image/png">>)).

stdio_server(Config) ->
    ok = wait_ready([<<"local">>]),
    Id = start(Config, [
        #{calls => [{<<"s1">>, <<"local__echo">>, #{<<"text">> => <<"over stdio">>}}]},
        #{text => <<"done">>}
    ]),
    _ = turn(Id, <<"go">>),
    ?assertEqual([{<<"local__echo">>, <<"over stdio">>}], last_tool_results(Id)).

policies(Config) ->
    ok = wait_ready([<<"web">>, <<"guarded">>, <<"closed">>]),
    %% `guarded' has the default policy: its tools ask.
    Id = start(Config, [
        #{calls => [{<<"p1">>, <<"guarded__echo">>, #{<<"text">> => <<"asked">>}}]},
        #{calls => [{<<"p2">>, <<"closed__echo">>, #{<<"text">> => <<"never">>}}]},
        #{text => <<"done">>}
    ]),
    ok = erllama_server_api:agent_send(Id, <<"go">>),
    ok = wait(fun() -> lists:member(<<"p1">>, pending_requests(Id)) end),
    ok = erllama_server_api:agent_approve(Id, <<"p1">>, allow),
    _ = collect_turn(Id),
    Shown = [R || Round <- erllama_server_agent_script:rounds(Id), R <- Round],
    ?assert(
        lists:member(#{role => <<"tool">>, name => <<"guarded__echo">>}, [
            maps:with([role, name], M)
         || M <- Shown
        ])
    ),
    ?assertMatch(
        {_, <<"Error: this tool is not allowed", _/binary>>},
        lists:keyfind(<<"closed__echo">>, 1, last_tool_results(Id))
    ),
    %% `--yes' runs them without asking, but a denied server stays denied;
    %% `--read-only' refuses the tools not marked read-only.
    Yes = start(Config, [#{text => <<"x">>}], #{policy => yes}),
    _ = turn(Yes, <<"go">>),
    [YesOffered] = erllama_server_agent_script:tools(Yes),
    ?assertNot(lists:member(<<"closed__echo">>, YesOffered)),
    RO = start(Config, [#{text => <<"x">>}], #{policy => read_only}),
    _ = turn(RO, <<"go">>),
    [ROOffered] = erllama_server_agent_script:tools(RO),
    ?assert(lists:member(<<"web__echo">>, ROOffered)),
    ?assertNot(lists:member(<<"web__touch">>, ROOffered)).

errors_and_timeouts(Config) ->
    ok = wait_ready([<<"web">>]),
    Id = start(Config, [
        #{
            calls => [
                {<<"e1">>, <<"web__fail">>, #{}},
                {<<"e2">>, <<"web__slow">>, #{}}
            ]
        },
        #{text => <<"done">>}
    ]),
    _ = turn(Id, <<"go">>),
    [{<<"web__fail">>, Failed}, {<<"web__slow">>, Slow}] = last_tool_results(Id),
    ?assertMatch(<<"Error: it broke">>, Failed),
    ?assertMatch(<<"Error: ", _/binary>>, Slow),
    %% The server is still up for the next call.
    ok = wait_ready([<<"web">>]).

server_down_and_back(Config) ->
    ok = wait_ready([<<"web">>]),
    %% A server that cannot be reached is never ready and offers nothing.
    [#{status := NowhereStatus}] = [
        S
     || #{name := <<"nowhere">>} = S <- erllama_server_mcp:status()
    ],
    ?assertNotEqual(ready, NowhereStatus),
    {ok, Tools} = erllama_server_api:mcp_tools(),
    ?assertEqual([], [N || #{server := <<"nowhere">>, name := N} <- Tools]),
    stop_server(?config(mcp_server, Config)),
    %% The release notices at its next call or refresh; a session meanwhile
    %% gets an error result, never a hang.
    Id = start(Config, [
        #{calls => [{<<"d1">>, <<"web__echo">>, #{<<"text">> => <<"x">>}}]},
        #{text => <<"done">>}
    ]),
    Events = turn(Id, <<"go">>),
    ?assertMatch(#{data := #{reason := stop}}, lists:last(Events)),
    [{<<"web__echo">>, Shown}] = last_tool_results(Id),
    ?assertMatch(<<"Error: ", _/binary>>, Shown),
    %% A server on the same port again: the client reconnects.
    Port = ?config(mcp_port, Config),
    {ok, Pid2} = livery:start_service(#{
        http => #{ip => {127, 0, 0, 1}, port => Port}, router => livery_mcp:router()
    }),
    try
        wait(
            fun() ->
                {ok, T} = erllama_server_api:mcp_tools(),
                lists:member(<<"web__echo">>, [N || #{name := N} <- T]) andalso
                    erllama_server_mcp:call(<<"web">>, <<"echo">>, #{<<"text">> => <<"y">>}, 1000) =:=
                        {ok, <<"y">>}
            end,
            2000
        )
    after
        stop_server(Pid2)
    end.

endpoint_serves_the_release(_Config) ->
    Url = iolist_to_binary(
        io_lib:format("http://127.0.0.1:~b/mcp", [erllama_server_http:port()])
    ),
    %% The API keys cover the endpoint.
    {ok, Anon} = barrel_mcp_client:start(#{transport => {http, Url}}),
    ok = wait(fun() -> not is_process_alive(Anon) orelse not_ready =/= list(Anon) end),
    ?assertMatch({error, _}, list_or_dead(Anon)),
    {ok, C} = barrel_mcp_client:start(#{transport => {http, Url}, auth => {bearer, <<"k1">>}}),
    ok = wait(fun() -> list(C) =/= not_ready end),
    {ok, Tools} = barrel_mcp_client:list_tools_all(C),
    ?assertEqual(
        [<<"erllama_chat">>, <<"erllama_models">>], lists:sort([N || #{<<"name">> := N} <- Tools])
    ),
    {ok, Models} = barrel_mcp_client:call_tool(C, <<"erllama_models">>, #{}),
    {ok, ModelsText} = erllama_server_mcp:render(Models),
    ?assertMatch([#{<<"name">> := <<"stub">>, <<"loaded">> := true}], json:decode(ModelsText)),
    {ok, Reply} = barrel_mcp_client:call_tool(C, <<"erllama_chat">>, #{
        <<"model">> => <<"stub">>, <<"prompt">> => <<"hello">>, <<"max_tokens">> => 4
    }),
    {ok, Text} = erllama_server_mcp:render(Reply),
    ?assert(byte_size(Text) > 0),
    {ok, Bad} = barrel_mcp_client:call_tool(C, <<"erllama_chat">>, #{
        <<"model">> => <<"nope">>, <<"prompt">> => <<"hello">>
    }),
    ?assertMatch({error, <<"unknown model: nope">>}, erllama_server_mcp:render(Bad)),
    barrel_mcp_client:close(C).

list(C) ->
    case barrel_mcp_client:list_tools(C) of
        {error, not_ready} -> not_ready;
        Other -> Other
    end.

list_or_dead(C) ->
    case is_process_alive(C) of
        false -> {error, closed};
        true -> barrel_mcp_client:list_tools(C)
    end.

%% Interrupting a session cancels its call at the server
%% (`notifications/cancelled' for that request): a server that honours
%% it never performs the side effect. The connection, shared with other
%% sessions, stays up. The server is erllama_server_fake_mcp: barrel_mcp's
%% servers do not stop a running tool on cancel yet.
interrupt_cancels_the_remote_call(Config) ->
    ok = wait_ready([<<"local">>]),
    Marker = filename:join(?config(priv_dir, Config), "effect.txt"),
    Id = start(Config, [
        #{calls => [{<<"x1">>, <<"local__effect">>, #{<<"path">> => list_to_binary(Marker)}}]},
        #{text => <<"never">>}
    ]),
    ok = erllama_server_api:agent_send(Id, <<"go">>),
    ok = wait(fun() -> started(Id, <<"x1">>) end),
    timer:sleep(200),
    ok = erllama_server_api:agent_interrupt(Id),
    timer:sleep(1500),
    ?assertNot(filelib:is_file(Marker)),
    ?assert(filelib:is_file(Marker ++ ".cancelled")),
    ?assertEqual(
        {ok, <<"still here">>},
        erllama_server_mcp:call(<<"local">>, <<"echo">>, #{<<"text">> => <<"still here">>}, 5000)
    ).

started(Id, CallId) ->
    {ok, #{replay := Replay}} = erllama_server_api:attach(Id, self(), 0),
    ok = erllama_server_api:detach(Id, self()),
    flush(Id),
    lists:member(CallId, [C || #{type := tool_started, data := #{call_id := C}} <- Replay]).

%% An alias belongs to one tool for good: a tool that goes does not
%% hand its name, or an approval given to it, to another.
aliases_stay_with_their_tool(Config) ->
    ok = wait_ready([<<"guarded">>]),
    ok = barrel_mcp:reg_tool(<<"a.b">>, ?MODULE, named, #{description => <<"dot">>}),
    ok = barrel_mcp:reg_tool(<<"a_b">>, ?MODULE, named, #{description => <<"underscore">>}),
    ok = erllama_server_mcp:refresh(),
    Remote = fun(Alias) ->
        {ok, T} = erllama_server_api:mcp_tools(),
        [R || #{name := N, server := <<"guarded">>, remote := R} <- T, N =:= Alias]
    end,
    ok = wait(fun() -> Remote(<<"guarded__a_b_2">>) =/= [] end),
    ?assertEqual([<<"a.b">>], Remote(<<"guarded__a_b">>)),
    ?assertEqual([<<"a_b">>], Remote(<<"guarded__a_b_2">>)),
    %% Always allow the first, in a session.
    Id = start(Config, [
        #{calls => [{<<"g1">>, <<"guarded__a_b">>, #{}}]},
        #{text => <<"one">>},
        #{calls => [{<<"g2">>, <<"guarded__a_b_2">>, #{}}]},
        #{text => <<"two">>}
    ]),
    ok = erllama_server_api:agent_send(Id, <<"first">>),
    ok = wait(fun() -> lists:member(<<"g1">>, pending_requests(Id)) end),
    ok = erllama_server_api:agent_approve(Id, <<"g1">>, always),
    _ = collect_turn(Id),
    %% The first goes away; its alias is not given to the second.
    ok = barrel_mcp:unreg_tool(<<"a.b">>),
    ok = erllama_server_mcp:refresh(),
    ok = wait(fun() -> Remote(<<"guarded__a_b">>) =:= [] end),
    ?assertEqual([<<"a_b">>], Remote(<<"guarded__a_b_2">>)),
    %% And the approval stays with the first: the second still asks.
    ok = erllama_server_api:agent_send(Id, <<"second">>),
    ok = wait(fun() -> lists:member(<<"g2">>, pending_requests(Id)) end),
    ok = erllama_server_api:agent_approve(Id, <<"g2">>, deny),
    _ = collect_turn(Id).

%% A server whose connection cannot even be set up is marked down; the
%% manager and the other servers go on. (Startup validation refuses such
%% a value; this checks the manager holds if one gets through.)
bad_server_stays_contained(_Config) ->
    ?assertMatch(
        {error, {invalid_config, mcp_servers, _}},
        erllama_server_config:validate(#{
            mcp_servers => [
                #{name => "bad", transport => http, url => "http://x/mcp", token => 123}
            ]
        })
    ),
    ok = application:set_env(erllama_server, mcp_servers, [
        #{name => "bad", transport => http, url => "http://127.0.0.1:1/mcp", token => 123}
    ]),
    {ok, Pid} = erllama_server_mcp:start_link(),
    try
        ok = wait(fun() ->
            case erllama_server_mcp:status() of
                [#{status := down, error := {connect_failed, _, _}}] -> true;
                _ -> false
            end
        end),
        ?assert(is_process_alive(Pid))
    after
        unlink(Pid),
        exit(Pid, shutdown)
    end.

%% A session reads a server's resources with tools, and takes a
%% server's prompt as its input.
resources_and_prompts(Config) ->
    ok = wait_ready([<<"web">>]),
    {ok, [#{name := <<"review">>, arguments := [#{name := <<"file">>, required := true}]}]} =
        erllama_server_api:mcp_prompts(<<"web">>),
    {ok, [#{role := <<"user">>, content := <<"Please review a.erl">>}]} =
        erllama_server_api:mcp_prompt(<<"web">>, <<"review">>, #{<<"file">> => <<"a.erl">>}),
    Id = start(Config, [
        #{
            calls => [
                {<<"r1">>, <<"web__list_resources">>, #{}},
                {<<"r2">>, <<"web__read_resource">>, #{<<"uri">> => <<"notes://today">>}}
            ]
        },
        #{text => <<"done">>}
    ]),
    ok = erllama_server_api:agent_prompt(Id, <<"web">>, <<"review">>, #{<<"file">> => <<"a.erl">>}),
    _ = collect_turn(Id),
    [First | _] = erllama_server_agent_script:rounds(Id),
    ?assertEqual(
        [<<"Please review a.erl">>], [C || #{role := <<"user">>, content := C} <- First]
    ),
    [{<<"web__list_resources">>, Listed}, {<<"web__read_resource">>, Read}] =
        last_tool_results(Id),
    ?assertNotEqual(nomatch, binary:match(Listed, <<"notes://today">>)),
    ?assertNotEqual(nomatch, binary:match(Read, <<"buy milk">>)).
