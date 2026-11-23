%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_mcp).
-moduledoc false.
%% The MCP servers the release is configured with (`mcp_servers'), as
%% tools for agent sessions. barrel_mcp does the protocol (transports,
%% discovery, version negotiation, authentication); this module keeps
%% one client per server, reconnects it when it goes, and keeps each
%% server's tool catalog current.
%%
%% A server that is down, slow or misconfigured never holds up the
%% release: clients start in the background, calls have timeouts and
%% are wrapped, and a server's tools simply leave the catalog while it
%% is unavailable.
%%
%% Agent sessions see a server's tools as `<server>__<tool>' (letters,
%% digits, `_' and `-', at most 64 bytes, a numeric suffix on a
%% collision); the session takes a snapshot of the catalog for each
%% generation round. A tool's alias is given once, the first time the
%% tool is seen, and never handed to another tool while the release
%% runs, even after the first one goes: a name the model or a user
%% approved always means the same server and tool. Names stay
%% binaries: no atom is ever made from a name a server sent.

-behaviour(gen_server).

-export([
    start_link/0,
    catalog/0,
    refresh/0,
    status/0,
    call/4,
    resources/1,
    read_resource/2,
    prompts/1,
    get_prompt/3
]).
-export([render/1, namespace/2]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-define(REFRESH_MS, 30000).
-define(MAX_BACKOFF_MS, 60000).
-define(NAME_MAX, 64).

-record(srv, {
    spec :: map(),
    pid :: pid() | undefined,
    mon :: reference() | undefined,
    status = connecting :: connecting | ready | down,
    error :: term(),
    tools = [] :: [map()],
    attempts = 0 :: non_neg_integer(),
    %% The current outage is in the failure log (cleared when ready).
    reported = false :: boolean()
}).

-record(state, {
    servers = #{} :: #{binary() => #srv{}},
    %% Every alias given so far, by the tool it names. Never reassigned.
    aliases = #{} :: #{{binary(), binary() | atom()} => binary()},
    taken = #{} :: #{binary() => true}
}).
-type state() :: #state{}.

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

%% The tools of the servers that are up, namespaced, in a stable order:
%% `#{name, server, remote, description, parameters, class, policy}'.
%% Empty while this process is restarting: a session never waits on it.
-spec catalog() -> [map()].
catalog() ->
    safe_call(catalog, []).

%% Ask every connected server for its tools now instead of at the next
%% periodic refresh. Returns at once.
-spec refresh() -> ok.
refresh() ->
    gen_server:cast(?MODULE, refresh).

-spec status() -> [map()].
status() ->
    safe_call(status, []).

safe_call(Req, Default) ->
    try
        gen_server:call(?MODULE, Req, 5000)
    catch
        exit:_ -> Default
    end.

%% Call a server's tool from the calling process (a tool worker): the
%% result as text for the model, `{error, Text}' when the tool failed
%% or the server is unavailable.
-spec call(binary(), binary(), map(), timeout()) -> {ok | error, binary()}.
call(Server, Tool, Args, Timeout) ->
    with_client(Server, fun(Pid) ->
        case barrel_mcp_client:call_tool(Pid, Tool, Args, #{timeout => Timeout}) of
            {ok, Result} ->
                render(Result);
            {error, R} ->
                ok = maybe_unavailable(Server, R),
                {error,
                    iolist_to_binary(
                        io_lib:format("the MCP server answered with an error: ~0p", [R])
                    )}
        end
    end).

%% A JSON-RPC error is the server answering; anything else (a timeout,
%% a transport error) may be the server being unavailable: recorded, and
%% the server is checked now (a catalog refresh) instead of at its next
%% periodic one.
maybe_unavailable(_Server, #{<<"code">> := _}) ->
    ok;
maybe_unavailable(Server, Reason) ->
    ok = erllama_server_failures:record(mcp_call_failed, Server, Reason),
    gen_server:cast(?MODULE, {check, Server}).

-spec resources(binary()) -> {ok | error, binary()}.
resources(Server) ->
    with_client(Server, fun(Pid) ->
        case barrel_mcp_client:list_resources_all(Pid) of
            {ok, Rs} -> {ok, iolist_to_binary(lists:join(<<"\n">>, [resource_line(R) || R <- Rs]))};
            {error, R} -> {error, iolist_to_binary(io_lib:format("~0p", [R]))}
        end
    end).

-spec read_resource(binary(), binary()) -> {ok | error, binary()}.
read_resource(Server, Uri) ->
    with_client(Server, fun(Pid) ->
        case barrel_mcp_client:read_resource(Pid, Uri) of
            {ok, #{<<"contents">> := Contents}} ->
                {ok, iolist_to_binary(lists:join(<<"\n">>, [block(C) || C <- Contents]))};
            {ok, Other} ->
                {ok, iolist_to_binary(json:encode(Other))};
            {error, R} ->
                {error, iolist_to_binary(io_lib:format("~0p", [R]))}
        end
    end).

%% The prompts a server offers: `#{name, description, arguments}'.
-spec prompts(binary()) -> {ok, [map()]} | {error, binary()}.
prompts(Server) ->
    with_client(Server, fun(Pid) ->
        case barrel_mcp_client:list_prompts_all(Pid) of
            {ok, Ps} -> {ok, [prompt_entry(P) || #{<<"name">> := N} = P <- Ps, is_binary(N)]};
            {error, R} -> {error, iolist_to_binary(io_lib:format("~0p", [R]))}
        end
    end).

prompt_entry(P) ->
    #{
        name => maps:get(<<"name">>, P),
        description => text_or_empty(maps:get(<<"description">>, P, <<>>)),
        arguments => [
            #{
                name => A,
                description => text_or_empty(maps:get(<<"description">>, Arg, <<>>)),
                required => is_true(maps:get(<<"required">>, Arg, false))
            }
         || #{<<"name">> := A} = Arg <- maps:get(<<"arguments">>, P, []), is_binary(A)
        ]
    }.

%% A prompt filled with `Args' (binary names and values), as the
%% messages it expands to: `#{role, content}' with the content as text.
-spec get_prompt(binary(), binary(), #{binary() => binary()}) ->
    {ok, [map()]} | {error, binary()}.
get_prompt(Server, Name, Args) ->
    with_client(Server, fun(Pid) ->
        case barrel_mcp_client:get_prompt(Pid, Name, Args) of
            {ok, #{<<"messages">> := Msgs}} when is_list(Msgs) ->
                {ok, [
                    #{role => role(maps:get(<<"role">>, M, <<"user">>)), content => block(C)}
                 || #{<<"content">> := C} = M <- Msgs
                ]};
            {ok, Other} ->
                {error, iolist_to_binary(["unexpected prompt result: ", json:encode(Other)])};
            {error, R} ->
                {error, iolist_to_binary(io_lib:format("~0p", [R]))}
        end
    end).

is_true(true) -> true;
is_true(_) -> false.

role(<<"assistant">>) -> <<"assistant">>;
role(_) -> <<"user">>.

text_or_empty(T) when is_binary(T) -> T;
text_or_empty(_) -> <<>>.

with_client(Server, Fun) ->
    case with_pid(Server, fun(Pid) -> {done, Fun(Pid)} end) of
        {done, Result} ->
            Result;
        {error, R} ->
            {error,
                iolist_to_binary(
                    io_lib:format("the MCP server ~ts is unavailable (~0p)", [Server, R])
                )}
    end.

%% The client of a server that is up. The client exits when its server
%% closes, so every call is guarded.
with_pid(Server, Fun) ->
    case safe_call({pid, Server}, {error, unavailable}) of
        {ok, Pid} ->
            try
                Fun(Pid)
            catch
                exit:R -> {error, {client_exit, R}}
            end;
        {error, _} = E ->
            E
    end.

%% -----------------------------------------------------------------------------
%% Results
%% -----------------------------------------------------------------------------

%% A tool result as text: text blocks as they are, other blocks named
%% (an image is not shown to a text model, but the model is told it was
%% there), structured content as JSON. `isError' makes it an error.
-spec render(map()) -> {ok | error, binary()}.
render(Result) ->
    Blocks = [block(B) || B <- maps:get(<<"content">>, Result, [])],
    Structured =
        case maps:get(<<"structuredContent">>, Result, undefined) of
            undefined -> [];
            Sc -> [iolist_to_binary(json:encode(Sc))]
        end,
    Text = iolist_to_binary(lists:join(<<"\n">>, Blocks ++ Structured)),
    case maps:get(<<"isError">>, Result, false) of
        true -> {error, Text};
        _ -> {ok, Text}
    end.

block(#{<<"type">> := <<"text">>, <<"text">> := T}) ->
    T;
block(#{<<"type">> := T, <<"mimeType">> := M} = B) when T =:= <<"image">>; T =:= <<"audio">> ->
    Size = byte_size(maps:get(<<"data">>, B, <<>>)) * 3 div 4,
    iolist_to_binary(io_lib:format("[~ts: ~ts, about ~b bytes, not shown]", [T, M, Size]));
block(#{<<"type">> := <<"resource">>, <<"resource">> := R}) ->
    case R of
        #{<<"text">> := Text} ->
            iolist_to_binary([<<"[resource ">>, maps:get(<<"uri">>, R, <<>>), <<"]\n">>, Text]);
        _ ->
            iolist_to_binary([
                <<"[resource ">>, maps:get(<<"uri">>, R, <<>>), <<", binary, not shown]">>
            ])
    end;
block(#{<<"type">> := <<"resource_link">>} = B) ->
    iolist_to_binary([
        <<"[link ">>, maps:get(<<"uri">>, B, <<>>), <<" ">>, maps:get(<<"name">>, B, <<>>), <<"]">>
    ]);
block(#{<<"text">> := T}) ->
    T;
block(#{<<"uri">> := U}) ->
    iolist_to_binary([<<"[">>, U, <<", binary, not shown]">>]);
block(Other) ->
    iolist_to_binary([<<"[unsupported content: ">>, json:encode(Other), <<"]">>]).

resource_line(#{<<"uri">> := U} = R) ->
    iolist_to_binary([
        U, <<"  ">>, maps:get(<<"name">>, R, <<>>), <<"  ">>, maps:get(<<"description">>, R, <<>>)
    ]).

%% -----------------------------------------------------------------------------
%% Names
%% -----------------------------------------------------------------------------

%% `<server>__<tool>' made safe for every chat template and API.
-spec namespace(binary(), binary() | atom()) -> binary().
namespace(Server, Tool) when is_atom(Tool) ->
    namespace(Server, atom_to_binary(Tool));
namespace(Server, Tool) ->
    Safe = re:replace(<<Server/binary, "__", Tool/binary>>, <<"[^A-Za-z0-9_-]">>, <<"_">>, [
        global, {return, binary}
    ]),
    binary:part(Safe, 0, min(?NAME_MAX, byte_size(Safe))).

%% Give an alias to each tool not seen before. Tools already named
%% keep theirs; a name once given is never free again.
assign(Tools, #state{aliases = A0, taken = T0} = St) ->
    {A, T} = lists:foldl(
        fun(#{server := S, remote := R}, {A1, T1} = Acc) ->
            case is_map_key({S, R}, A1) of
                true ->
                    Acc;
                false ->
                    Name = free_name(namespace(S, R), T1, 1),
                    {A1#{{S, R} => Name}, T1#{Name => true}}
            end
        end,
        {A0, T0},
        lists:sort(fun(X, Y) -> key(X) =< key(Y) end, Tools)
    ),
    St#state{aliases = A, taken = T}.

key(#{server := S, remote := R}) -> {S, R}.

free_name(Base, Seen, N) ->
    Candidate =
        case N of
            1 ->
                Base;
            _ ->
                <<
                    (binary:part(Base, 0, min(?NAME_MAX - 4, byte_size(Base))))/binary,
                    "_",
                    (integer_to_binary(N))/binary
                >>
        end,
    case is_map_key(Candidate, Seen) of
        true -> free_name(Base, Seen, N + 1);
        false -> Candidate
    end.

%% -----------------------------------------------------------------------------
%% gen_server
%% -----------------------------------------------------------------------------

-spec init([]) -> {ok, state()}.
init([]) ->
    process_flag(trap_exit, true),
    Specs = erllama_server_config:get(mcp_servers),
    Servers = maps:from_list([{name(S), #srv{spec = S}} || S <- Specs]),
    _ = [self() ! {connect, N} || N := _ <- Servers],
    {ok, #state{servers = Servers}}.

handle_call(catalog, _From, #state{servers = Ss, aliases = A} = St) ->
    Tools = lists:append([
        lists:sort(fun(X, Y) -> key(X) =< key(Y) end, Tools)
     || _ := #srv{status = ready, tools = Tools} <- maps:iterator(Ss, ordered)
    ]),
    {reply, [T#{name => maps:get(key(T), A)} || T <- Tools], St};
handle_call(status, _From, #state{servers = Ss} = St) ->
    {reply, [describe(N, S) || N := S <- maps:iterator(Ss, ordered)], St};
handle_call({pid, Name}, _From, #state{servers = Ss} = St) ->
    case Ss of
        #{Name := #srv{status = ready, pid = Pid}} -> {reply, {ok, Pid}, St};
        #{Name := #srv{status = Status}} -> {reply, {error, Status}, St};
        _ -> {reply, {error, unknown_server}, St}
    end;
handle_call(_Req, _From, St) ->
    {reply, {error, unknown_call}, St}.

handle_cast({check, Name}, #state{servers = Ss} = St) ->
    _ = [self() ! {refresh, Name, P} || #{Name := #srv{pid = P}} <- [Ss], is_pid(P)],
    {noreply, St};
handle_cast(refresh, #state{servers = Ss} = St) ->
    _ = [self() ! {refresh, N, P} || N := #srv{pid = P} <- Ss, is_pid(P)],
    {noreply, St};
handle_cast(_Msg, St) ->
    {noreply, St}.

handle_info({connect, Name}, St) ->
    {noreply, update(Name, fun(S) -> connect(Name, S) end, St)};
handle_info({refresh, Name, Pid}, St) ->
    {noreply, update(Name, fun(S) -> refresh(Name, Pid, S) end, St)};
handle_info({catalog, Name, Pid, Result}, St) ->
    St1 = update(Name, fun(S) -> catalog_result(Name, Pid, Result, S) end, St),
    #{Name := #srv{tools = Tools}} = St1#state.servers,
    {noreply, assign(Tools, St1)};
handle_info({'EXIT', _Pid, _Reason}, St) ->
    %% A client's end is handled by its monitor.
    {noreply, St};
handle_info({'DOWN', Mon, process, _, Reason}, #state{servers = Ss} = St) ->
    case [N || N := #srv{mon = M} <- Ss, M =:= Mon] of
        [Name] -> {noreply, update(Name, fun(S) -> lost(Name, Reason, S) end, St)};
        [] -> {noreply, St}
    end;
handle_info(_Msg, St) ->
    {noreply, St}.

update(Name, Fun, #state{servers = Ss} = St) ->
    case Ss of
        #{Name := S} -> St#state{servers = Ss#{Name := Fun(S)}};
        _ -> St
    end.

%% Start the client (barrel_mcp connects in the background) and ask for
%% its tools from a helper, so this process never waits on a server.
%% Clients are linked: they end with this process, so a restart never
%% leaves a second copy of a stdio server running.
%% Whatever goes wrong stays with this server: it is marked down and
%% tried again later, and the other servers and the release go on.
connect(Name, #srv{spec = Spec} = S) ->
    try barrel_mcp_client:start_link(connect_spec(Spec)) of
        {ok, Pid} ->
            Mon = erlang:monitor(process, Pid),
            self() ! {refresh, Name, Pid},
            S#srv{pid = Pid, mon = Mon, status = connecting};
        {error, R} ->
            retry(Name, R, S)
    catch
        Class:Reason ->
            retry(Name, {connect_failed, Class, Reason}, S)
    end.

refresh(Name, Pid, #srv{pid = Pid} = S) ->
    Self = self(),
    _ = spawn(fun() ->
        Result =
            try barrel_mcp_client:list_tools_all(Pid) of
                {ok, Tools} -> {ok, Tools, capabilities(Pid)};
                Other -> Other
            catch
                exit:R -> {error, R}
            end,
        Self ! {catalog, Name, Pid, Result}
    end),
    S;
refresh(_Name, _Pid, S) ->
    S.

catalog_result(Name, Pid, {ok, Tools, Caps}, #srv{pid = Pid, spec = Spec} = S) ->
    _ = erlang:send_after(?REFRESH_MS, self(), {refresh, Name, Pid}),
    Policy = maps:get(policy, Spec, ask),
    Entries = [entry(Name, N, T, Policy) || #{<<"name">> := N} = T <- Tools, is_binary(N)],
    S#srv{
        status = ready,
        tools = Entries ++ resource_tools(Name, Policy, Caps),
        error = undefined,
        attempts = 0,
        reported = false
    };
catalog_result(Name, Pid, {error, not_ready}, #srv{pid = Pid, status = connecting} = S) ->
    %% Still in its handshake (the client stops itself if that never
    %% ends).
    _ = erlang:send_after(200, self(), {refresh, Name, Pid}),
    S;
catalog_result(Name, Pid, {error, R}, #srv{pid = Pid} = S) ->
    %% Up but not answering: try again later, keep the old catalog out.
    _ = erlang:send_after(backoff(S#srv.attempts), self(), {refresh, Name, Pid}),
    (went_down(Name, R, S))#srv{
        status = down, error = R, tools = [], attempts = S#srv.attempts + 1
    };
catalog_result(_Name, _Pid, _Result, S) ->
    S.

entry(Server, Name, Tool, Policy) ->
    #{
        server => Server,
        remote => Name,
        description => text_or_empty(maps:get(<<"description">>, Tool, <<>>)),
        parameters => schema(maps:get(<<"inputSchema">>, Tool, undefined)),
        class => class(Tool),
        policy => Policy
    }.

capabilities(Pid) ->
    try barrel_mcp_client:server_capabilities(Pid) of
        {ok, #{} = C} -> C;
        _ -> #{}
    catch
        _:_ -> #{}
    end.

%% A server that offers resources gets two tools of ours: list them,
%% read one. Their remote side is ours (an atom we chose, never a name
%% the server sent), so they cannot be mistaken for the server's tools.
resource_tools(Name, Policy, #{<<"resources">> := _}) ->
    [
        #{
            server => Name,
            remote => list_resources,
            description => <<"List the resources the ", Name/binary, " MCP server offers.">>,
            parameters => #{<<"type">> => <<"object">>, <<"properties">> => #{}},
            class => read,
            policy => Policy
        },
        #{
            server => Name,
            remote => read_resource,
            description => <<"Read a resource of the ", Name/binary, " MCP server by its URI.">>,
            parameters => #{
                <<"type">> => <<"object">>,
                <<"properties">> => #{
                    <<"uri">> => #{<<"type">> => <<"string">>, <<"description">> => <<"The URI.">>}
                },
                <<"required">> => [<<"uri">>]
            },
            class => read,
            policy => Policy
        }
    ];
resource_tools(_Name, _Policy, _Caps) ->
    [].

schema(#{} = S) -> S;
schema(_) -> #{<<"type">> => <<"object">>}.

%% A tool runs alongside reads only when its server says it does not
%% change anything; otherwise it is ordered like a native write.
class(#{<<"annotations">> := #{<<"readOnlyHint">> := true}}) -> read;
class(_) -> write.

lost(Name, Reason, S) ->
    retry(Name, Reason, S#srv{pid = undefined, mon = undefined, tools = []}).

retry(Name, Reason, #srv{attempts = A} = S) ->
    _ = erlang:send_after(backoff(A), self(), {connect, Name}),
    (went_down(Name, Reason, S))#srv{status = down, error = Reason, attempts = A + 1}.

%% Recorded once per outage, not at every reconnection attempt.
went_down(_Name, _Reason, #srv{reported = true} = S) ->
    S;
went_down(Name, Reason, S) ->
    ok = erllama_server_failures:record(mcp_down, Name, Reason),
    S#srv{reported = true}.

backoff(A) -> min(?MAX_BACKOFF_MS, 1000 bsl min(A, 6)).

describe(Name, #srv{status = Status, tools = Tools, error = E, spec = Spec}) ->
    #{
        name => Name,
        transport => maps:get(transport, Spec),
        status => Status,
        tools => length(Tools),
        error => E
    }.

name(#{name := N}) -> unicode:characters_to_binary(N).

connect_spec(#{transport := stdio, command := Cmd} = S) ->
    Base = #{
        transport =>
            {stdio, #{
                command => Cmd,
                args => maps:get(args, S, []),
                env => [{str(K), env_value(V)} || {K, V} <- maps:get(env, S, [])]
            }}
    },
    common(Base, S);
connect_spec(#{transport := http, url := Url} = S) ->
    Base0 = #{transport => {http, unicode:characters_to_binary(Url)}},
    Base1 =
        case S of
            #{token := T} -> Base0#{auth => {bearer, unicode:characters_to_binary(T)}};
            _ -> Base0
        end,
    Base =
        case S of
            #{headers := H} -> Base1#{http_headers => H};
            _ -> Base1
        end,
    common(Base, S).

common(Base, _S) ->
    Base#{
        client_info => #{name => <<"erllama_server">>, version => <<"0.1.0">>},
        request_timeout => erllama_server_config:get(mcp_timeout_ms)
    }.

str(V) -> unicode:characters_to_list(V).

env_value(false) -> false;
env_value(V) -> str(V).
