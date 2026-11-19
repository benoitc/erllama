%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_api).
-moduledoc """
The service API of the release: the one module the `erllama` CLI
calls over Erlang distribution and the HTTP adapters call in process.

Every function returns `ok`, `{ok, Value}` or `{error, Reason}` and
never raises, so a remote caller can tell a refusal from a crash.
Clients call `hello/1` first with the API version they speak; the
release refuses any other version.

A job (a run, or a model download started with `pull/2`) streams its
events to the processes attached with `attach/3` as
`{erllama_server_run, JobId, Event}`. An event is
`#{seq := N, type := Type, data := Data}`. A run emits `started`,
`loading` (when its model is being loaded), `token`, `thinking`; a
download emits `started` and `progress`; both end with one of the
terminal `done`, `cancelled` and `error`. Subscribers ack with
`RunPid ! {ack, self(), Seq}`; one that falls too far behind gets
`{erllama_server_run, RunId, {dropped, NextSeq}}` and reattaches
from `NextSeq`.
""".

-export([
    api_version/0,
    hello/1,
    status/0,
    models/0,
    show/1,
    ps/0,
    register_model/2,
    load/2,
    load_model/2,
    unload/1,
    unload_model/1,
    alias/2,
    copy/2,
    remove/1,
    set_default/1,
    pull/2,
    start_run/1,
    attach/3,
    detach/2,
    cancel/1,
    result/1,
    run/1,
    runs/0,
    conversation/1,
    conversations/0,
    agent_start/1,
    agent_send/2,
    agent_approve/3,
    agent_interrupt/1,
    agent_stop/1,
    agents/0,
    mcp/0,
    mcp_tools/0,
    mcp_resources/1,
    mcp_read/2,
    mcp_prompts/1,
    mcp_prompt/3,
    agent_prompt/4
]).

-include("erllama_server_api.hrl").

-define(API_VSN, ?ERLLAMA_SERVER_API_VSN).

-doc "The API version this release speaks.".
-spec api_version() -> pos_integer().
api_version() -> ?API_VSN.

-doc """
Check that the client speaks this release's API version. Returns the
release identity, or `{error, {incompatible_api, #{server, client}}}`.
""".
-spec hello(term()) -> {ok, map()} | {error, {incompatible_api, map()}}.
hello(?API_VSN) ->
    {ok, #{
        api => ?API_VSN,
        server_vsn => app_vsn(erllama_server),
        erllama_vsn => app_vsn(erllama),
        node => node()
    }};
hello(Client) ->
    {error, {incompatible_api, #{server => ?API_VSN, client => Client}}}.

-doc "Versions, loaded models, active jobs, memory pressure and VRAM.".
-spec status() -> {ok, map()}.
status() ->
    Pressure =
        case erllama:pressure() of
            {ok, P} -> P;
            {error, _} -> undefined
        end,
    {ok, #{
        api => ?API_VSN,
        server_vsn => app_vsn(erllama_server),
        erllama_vsn => app_vsn(erllama),
        node => node(),
        models => ps_or_empty(),
        active_runs => erllama_server_runs:active_count(run),
        active_pulls => erllama_server_runs:active_count(pull),
        pressure => Pressure,
        vram => vram()
    }}.

ps_or_empty() ->
    {ok, Ps} = ps(),
    Ps.

vram() ->
    try erllama:vram_info() of
        {ok, V} -> V;
        _ -> undefined
    catch
        _:_ -> undefined
    end.

-doc """
The registry's models, with whether each is loaded, plus models loaded
directly through erllama (`registered => false`).
""".
-spec models() -> {ok, [map()]}.
models() ->
    Loaded = maps:from_list([{maps:get(id, I), I} || I <- erllama:list_models()]),
    Registered = [
        maps:merge(
            maps:with([name, source, size, sha256, status, default, keep_alive], E),
            #{
                registered => true,
                loaded => is_map_key(Name, Loaded),
                mmproj => maps:get(mmproj, E, undefined) =/= undefined
            }
        )
     || #{name := Name} = E <- erllama_server_registry:list()
    ],
    Names = [N || #{name := N} <- Registered],
    Direct = [
        #{name => Id, registered => false, loaded => true, status => ok}
     || Id := _ <- Loaded, not lists:member(Id, Names)
    ],
    {ok, Registered ++ Direct}.

-doc "One model: its registry entry, the GGUF metadata and its load state.".
-spec show(binary()) -> {ok, map()} | {error, term()}.
show(Name) ->
    case erllama_server_registry:resolve(Name) of
        {ok, #{name := Id, path := Path} = Entry} ->
            Meta =
                case erllama_server_gguf:read(Path) of
                    {ok, H} -> erllama_server_gguf:summary(H);
                    {error, R} -> #{error => R}
                end,
            {ok, Entry#{metadata => Meta, loaded => erllama:whereis(Id) =/= {error, not_loaded}}};
        {error, _} = E ->
            case erllama:model_info(Name) of
                {ok, Info} ->
                    {ok, #{name => Name, registered => false, info => maps:remove(pid, Info)}};
                {error, _} ->
                    E
            end
    end.

-doc "Loaded models, with holders and keep-alive for registry models.".
-spec ps() -> {ok, [map()]}.
ps() ->
    {ok, erllama_server_models:ps()}.

-doc """
Register a GGUF file on the release machine. `Opts`: `name`, `mmproj`,
`n_ctx`, `n_gpu_layers`, `chat_template`, `keep_alive`.
""".
-spec register_model(binary() | string(), map()) -> {ok, map()} | {error, term()}.
register_model(Path, Opts) when is_map(Opts) ->
    erllama_server_registry:register_local(Path, Opts).

-doc "Load a registry model now; its keep-alive starts at once.".
-spec load(binary(), map()) -> {ok, binary()} | {error, term()}.
load(Name, Opts) when is_map(Opts) ->
    erllama_server_models:load(Name, maps:get(keep_alive, Opts, undefined)).

-doc """
Register a local GGUF file (when `Path` is one and not yet registered)
and load it. `Opts` as for `register_model/2`.
""".
-spec load_model(binary() | string(), map()) -> {ok, binary()} | {error, term()}.
load_model(Path, Opts0) when is_map(Opts0) ->
    Opts =
        case maps:take(id, Opts0) of
            {Id, Rest} -> Rest#{name => Id};
            error -> Opts0
        end,
    case register_model(Path, Opts) of
        {ok, #{name := Name}} -> load(Name, Opts);
        {error, _} = E -> E
    end.

-doc "Unload a model. Runs still using it end with an error event.".
-spec unload(binary()) -> ok | {error, term()}.
unload(Name) ->
    erllama_server_models:unload(Name).

-doc "Same as `unload/1`.".
-spec unload_model(binary()) -> ok | {error, term()}.
unload_model(Name) ->
    unload(Name).

-doc "Give a model a second name.".
-spec alias(binary(), binary()) -> ok | {error, term()}.
alias(Alias, Name) ->
    erllama_server_registry:alias(Alias, Name).

-doc "Copy a registry entry under a new name (same file).".
-spec copy(binary(), binary()) -> {ok, map()} | {error, term()}.
copy(Src, Dst) ->
    erllama_server_registry:copy(Src, Dst).

-doc """
Remove a model from the registry (unloading it first). A downloaded file
no other entry uses is deleted; a registered local file is kept.
""".
-spec remove(binary()) -> ok | {error, term()}.
remove(Name) ->
    case erllama_server_registry:resolve(Name) of
        {ok, #{name := Id}} ->
            _ = erllama_server_models:unload(Id),
            erllama_server_registry:remove(Id);
        {error, _} = E ->
            E
    end.

-doc "The model used when a request names none.".
-spec set_default(binary()) -> ok | {error, term()}.
set_default(Name) ->
    erllama_server_registry:set_default(Name).

-doc """
Download a model and register it; returns the job id to follow with
`attach/3`. `Source`: `hf://ORG/REPO[:QUANT]`, `hf://ORG/REPO/FILE.gguf`,
`https://...gguf`, `ollama://NAME[:TAG]` or `NAME[:TAG]`. `Opts`:
`name`, `sha256` (https), `revision` (hf), `n_ctx`, `n_gpu_layers`,
`keep_alive`.
""".
-spec pull(binary() | string(), map()) -> {ok, #{job_id := binary()}} | {error, term()}.
pull(Source, Opts) when is_map(Opts) ->
    case erllama_server_source:parse(Source) of
        {ok, #{ref := Ref} = S} ->
            Id = erllama_server_ids:pull(),
            Spec = #{run_id => Id, kind => pull, model => Ref, source => S, opts => Opts},
            case erllama_server_runs:start(Spec) of
                {ok, _} -> {ok, #{job_id => Id}};
                {error, _} = E -> E
            end;
        {error, _} = E ->
            E
    end.

-doc """
Start a run and return its id. `Spec`:

- `model`: a registry name or alias (loaded on demand), a model loaded
  directly through erllama, or absent for the default model.
- `keep_alive`: how long the model stays loaded after this run, in ms
  (or `infinity`); default the model's own.
- `messages`: the new chat messages of this turn: `#{role, content}`
  with binary content; an assistant message may carry `tool_calls`
  (OpenAI shape, or flat `#{name, arguments, id}` as `result/1` returns
  them), and a `tool` message must carry the `tool_call_id` it
  answers. The run is a turn of `conversation_id`, created when
  absent; the conversation's history is prepended and the reply
  appended when the turn completes.
- `prompt`: raw text instead of `messages`, outside any conversation.
- `opts`: generation options (see `erllama_server_generate:option_keys/0`).
""".
-spec start_run(map()) ->
    {ok, #{run_id := binary(), conversation_id := binary() | undefined}} | {error, term()}.
start_run(Spec) when is_map(Spec) ->
    Opts0 = maps:get(opts, Spec, #{}),
    case erllama_server_generate:check_opts(Opts0) of
        ok ->
            case erllama_server_models:resolve(maps:get(model, Spec, undefined)) of
                {ok, {registry, #{name := Id} = Entry}} ->
                    Opts = maps:merge(maps:get(defaults, Entry, #{}), Opts0),
                    start_input(input(Spec), Id, Opts, Spec);
                {ok, {loaded, Id}} ->
                    start_input(input(Spec), Id, Opts0, Spec);
                {error, _} = E ->
                    E
            end;
        {error, _} = E ->
            E
    end;
start_run(_Spec) ->
    {error, invalid_spec}.

start_input({error, _} = E, _Model, _Opts, _Spec) ->
    E;
start_input({prompt, _} = Input, Model, Opts, Spec) ->
    launch(base_spec(Input, Model, Opts, Spec));
start_input({messages, New} = Input, Model, Opts, #{conversation_id := none} = Spec) ->
    %% A stateless request (HTTP): the messages are the whole prompt.
    launch((base_spec(Input, Model, Opts, Spec))#{new_messages => New});
start_input({messages, New}, Model, Opts, Spec) ->
    RunSpec0 = base_spec({messages, New}, Model, Opts, Spec),
    #{run_id := RunId} = RunSpec0,
    case erllama_server_convs:begin_run(maps:get(conversation_id, Spec, undefined), Model, RunId) of
        {ok, #{id := ConvId, messages := History, parent_key := PK}} ->
            RunSpec = RunSpec0#{
                conversation_id => ConvId,
                input => {messages, History ++ New},
                new_messages => New,
                parent_key => PK
            },
            launch_turn(ConvId, RunSpec);
        {error, _} = E ->
            E
    end.

base_spec(Input, Model, Opts, Spec) ->
    #{
        run_id => erllama_server_ids:run(),
        model => Model,
        input => Input,
        opts => Opts,
        keep_alive => maps:get(keep_alive, Spec, undefined)
    }.

%% A turn that fails to start frees its conversation again.
launch_turn(ConvId, #{run_id := RunId} = RunSpec) ->
    case launch(RunSpec) of
        {ok, _} = Ok ->
            Ok;
        {error, _} = E ->
            ok = erllama_server_convs:release(ConvId, RunId),
            E
    end.

launch(#{run_id := RunId} = RunSpec) ->
    case erllama_server_runs:start(RunSpec) of
        {ok, Pid} ->
            case go(Pid) of
                ok ->
                    {ok, #{
                        run_id => RunId,
                        conversation_id => maps:get(conversation_id, RunSpec, undefined)
                    }};
                {error, _} = E ->
                    E
            end;
        {error, _} = E ->
            E
    end.

go(Pid) ->
    try
        erllama_server_run:go(Pid)
    catch
        exit:{timeout, _} ->
            _ = supervisor:terminate_child(erllama_server_run_sup, Pid),
            {error, start_timeout}
    end.

input(#{messages := Msgs, prompt := _}) when is_list(Msgs) ->
    {error, {invalid_spec, messages_and_prompt}};
input(#{prompt := _, conversation_id := C}) when C =/= none ->
    {error, {invalid_spec, prompt_with_conversation}};
input(#{prompt := P}) when is_binary(P), P =/= <<>> ->
    {prompt, P};
input(#{messages := [_ | _] = Msgs}) ->
    messages(Msgs, []);
input(_) ->
    {error, {missing, messages}}.

messages([], Acc) ->
    {messages, lists:reverse(Acc)};
messages([M | Rest], Acc) ->
    case erllama_server_generate:message(M) of
        {ok, Msg} -> messages(Rest, [Msg | Acc]);
        {error, _} = E -> E
    end.

-doc """
Subscribe `Pid` to a run from sequence `Cursor` (0: from the start).
Returns the buffered events from `Cursor` on, the run pid to ack to
and the run status; a finished run returns its events without
subscribing. `{error, {gap, Oldest}}` when `Cursor` is older than the
buffer: attach again from `Oldest` to accept the loss.
""".
-spec attach(binary(), pid(), non_neg_integer()) -> {ok, map()} | {error, term()}.
attach(RunId, Pid, Cursor) when is_pid(Pid), is_integer(Cursor), Cursor >= 0 ->
    with_run(RunId, fun(RunPid) -> erllama_server_job:attach(RunPid, Pid, Cursor) end).

-doc "Stop sending a run's events to `Pid`. The run goes on.".
-spec detach(binary(), pid()) -> ok | {error, term()}.
detach(RunId, Pid) ->
    with_run(RunId, fun(RunPid) -> erllama_server_job:detach(RunPid, Pid) end).

-doc "Cancel a run. Its last event is `cancelled`. Idempotent.".
-spec cancel(binary()) -> ok | {error, term()}.
cancel(RunId) ->
    with_run(RunId, fun erllama_server_job:cancel/1).

-doc """
The result of a finished run: `status`, `content`, `reasoning`,
`tool_calls`, `finish_reason`, `usage`, `timings`; or `status` and
`error` for a failed run. `{error, running}` before it finishes.
""".
-spec result(binary()) -> {ok, map()} | {error, term()}.
result(RunId) ->
    with_run(RunId, fun erllama_server_job:result/1).

-doc "One run's id, conversation, model, status and next sequence.".
-spec run(binary()) -> {ok, map()} | {error, term()}.
run(RunId) ->
    with_run(RunId, fun erllama_server_job:info/1).

-doc "The runs the release still holds, oldest first.".
-spec runs() -> {ok, [map()]}.
runs() ->
    {ok, [maps:remove(pid, R) || R <- erllama_server_runs:list()]}.

-doc "A conversation with its messages.".
-spec conversation(binary()) -> {ok, map()} | {error, not_found}.
conversation(Id) ->
    case erllama_server_convs:get(Id) of
        {ok, C} -> {ok, maps:remove(last_finish_key, C)};
        {error, _} = E -> E
    end.

-doc "The conversations, without their messages.".
-spec conversations() -> {ok, [map()]}.
conversations() ->
    {ok, [maps:remove(last_finish_key, C) || C <- erllama_server_convs:list()]}.

with_run(RunId, Fun) ->
    case erllama_server_runs:lookup(RunId) of
        {ok, #{pid := Pid}} -> Fun(Pid);
        {error, _} = E -> E
    end.

app_vsn(App) ->
    case application:get_key(App, vsn) of
        {ok, V} -> list_to_binary(V);
        undefined -> <<"unknown">>
    end.

%% -----------------------------------------------------------------------------
%% Agent sessions
%% -----------------------------------------------------------------------------

-doc """
Start an agent session: a model working with tools in `workspace`, a
directory on the release machine. `Spec`:

- `model` (registry name or alias; default model when absent) and
  `workspace` (required);
- `conversation_id`: continue that conversation (it must be free);
- `policy`: `yes` (every tool runs without asking), `read_only` (tools
  that change the workspace are denied), or `#{ToolName => auto | ask |
  deny}`; by default reading runs and changing asks;
- `budgets`: `max_rounds`, `max_tokens`, `max_elapsed_ms` per turn;
- `system`: the system prompt (`WORKSPACE` is replaced by the path).

Returns the session id, which is also its job id for `attach/3`.
""".
-spec agent_start(map()) ->
    {ok, #{agent_id := binary(), conversation_id := binary()}} | {error, term()}.
agent_start(#{workspace := Dir} = Spec) ->
    case
        {
            erllama_server_models:resolve(maps:get(model, Spec, undefined)),
            erllama_server_workspace:open(Dir)
        }
    of
        {{ok, Target}, {ok, Ws}} ->
            Model =
                case Target of
                    {registry, #{name := N}} -> N;
                    {loaded, N} -> N
                end,
            start_agent(Spec, Model, Ws);
        {{error, _} = E, _} ->
            E;
        {_, {error, R}} ->
            {error, {workspace, R}}
    end;
agent_start(_Spec) ->
    {error, {missing, workspace}}.

start_agent(Spec, Model, Ws) ->
    Id = erllama_server_ids:agent(),
    case erllama_server_convs:begin_run(maps:get(conversation_id, Spec, undefined), Model, Id) of
        {ok, #{id := Conv, messages := History, parent_key := PK}} ->
            AgentSpec0 = #{
                run_id => Id,
                kind => agent,
                model => Model,
                workspace => Ws,
                conversation_id => Conv,
                history => [M || #{role := R} = M <- History, R =/= <<"system">>],
                parent_key => PK,
                policy => agent_policy(maps:get(policy, Spec, #{})),
                mode => agent_mode(maps:get(policy, Spec, #{})),
                budgets => maps:with(
                    [max_rounds, max_tokens, max_elapsed_ms], maps:get(budgets, Spec, #{})
                )
            },
            AgentSpec = maps:merge(AgentSpec0, maps:with([system, generator], Spec)),
            case erllama_server_runs:start(AgentSpec) of
                {ok, _} ->
                    {ok, #{agent_id => Id, conversation_id => Conv}};
                {error, _} = E ->
                    ok = erllama_server_convs:release(Conv, Id),
                    E
            end;
        {error, _} = E ->
            E
    end.

agent_mode(P) when P =:= yes; P =:= read_only -> P;
agent_mode(_) -> default.

agent_policy(yes) ->
    maps:from_list([{N, auto} || N <- erllama_server_tools:names()]);
agent_policy(read_only) ->
    maps:from_list([
        {N, deny}
     || N <- erllama_server_tools:names(), erllama_server_tools:class(N) =:= write
    ]);
agent_policy(M) when is_map(M) ->
    maps:filter(fun(K, V) -> is_binary(K) andalso lists:member(V, [auto, ask, deny]) end, M);
agent_policy(_) ->
    #{}.

-doc "Input for a session: starts a turn, or waits for the current one (`{queued, N}`).".
-spec agent_send(binary(), binary()) -> ok | {queued, pos_integer()} | {error, term()}.
agent_send(Id, Text) ->
    with_agent(Id, fun(Pid) ->
        erllama_server_agent:send(Pid, unicode:characters_to_binary(Text))
    end).

-doc "Answer an `approval_request`: `allow`, `deny`, or `always` (allow this tool for the session).".
-spec agent_approve(binary(), binary(), allow | deny | always) -> ok | {error, term()}.
agent_approve(Id, CallId, Decision) when
    Decision =:= allow; Decision =:= deny; Decision =:= always
->
    with_agent(Id, fun(Pid) -> erllama_server_agent:approve(Pid, CallId, Decision) end).

-doc "Stop the session's current turn (generation and tools). Queued input then runs.".
-spec agent_interrupt(binary()) -> ok | {error, term()}.
agent_interrupt(Id) ->
    with_agent(Id, fun erllama_server_agent:interrupt/1).

-doc "End a session. Its conversation stays and can be continued.".
-spec agent_stop(binary()) -> ok | {error, term()}.
agent_stop(Id) ->
    with_agent(Id, fun erllama_server_agent:stop/1).

-doc "The agent sessions the release holds.".
-spec agents() -> {ok, [map()]}.
agents() ->
    {ok, [
        S
     || #{kind := agent, pid := P} <- erllama_server_runs:list(),
        {ok, S} <- [erllama_server_job:info(P)]
    ]}.

with_agent(Id, Fun) ->
    case erllama_server_runs:lookup(Id) of
        {ok, #{kind := agent, pid := Pid}} -> Fun(Pid);
        {ok, _} -> {error, not_an_agent};
        {error, _} = E -> E
    end.

%% -----------------------------------------------------------------------------
%% MCP servers
%% -----------------------------------------------------------------------------

-doc """
The configured MCP servers: `name`, `transport`, `status` (`connecting`,
`ready` or `down`), the number of `tools`, and the last `error`.
""".
-spec mcp() -> {ok, [map()]}.
mcp() ->
    {ok, erllama_server_mcp:status()}.

-doc """
The MCP tools agent sessions are offered now, by their namespaced name,
with the server, the server's own name for the tool, the policy, and
`class` (`read` when the server marks the tool read-only).
""".
-spec mcp_tools() -> {ok, [map()]}.
mcp_tools() ->
    {ok, [maps:without([parameters], T) || T <- erllama_server_mcp:catalog()]}.

-doc "The resources an MCP server lists, one per line.".
-spec mcp_resources(binary()) -> {ok, binary()} | {error, binary()}.
mcp_resources(Server) ->
    erllama_server_mcp:resources(Server).

-doc "The text of a resource of an MCP server.".
-spec mcp_read(binary(), binary()) -> {ok, binary()} | {error, binary()}.
mcp_read(Server, Uri) ->
    erllama_server_mcp:read_resource(Server, Uri).

-doc "The prompts an MCP server offers: `name`, `description`, `arguments`.".
-spec mcp_prompts(binary()) -> {ok, [map()]} | {error, binary()}.
mcp_prompts(Server) ->
    erllama_server_mcp:prompts(Server).

-doc "An MCP server's prompt filled with `Args`: the messages it expands to.".
-spec mcp_prompt(binary(), binary(), #{binary() => binary()}) -> {ok, [map()]} | {error, binary()}.
mcp_prompt(Server, Name, Args) ->
    erllama_server_mcp:get_prompt(Server, Name, Args).

-doc """
Send an MCP server's prompt to an agent session as its input, as
`agent_send/2` would send text. A prompt's assistant messages are kept,
marked as such.
""".
-spec agent_prompt(binary(), binary(), binary(), #{binary() => binary()}) ->
    ok | {queued, pos_integer()} | {error, term()}.
agent_prompt(Id, Server, Name, Args) ->
    case erllama_server_mcp:get_prompt(Server, Name, Args) of
        {ok, Msgs} -> agent_send(Id, prompt_text(Msgs));
        {error, _} = E -> E
    end.

prompt_text(Msgs) ->
    iolist_to_binary(
        lists:join(<<"\n\n">>, [
            case R of
                <<"assistant">> -> [<<"Assistant: ">>, C];
                _ -> C
            end
         || #{role := R, content := C} <- Msgs
        ])
    ).
