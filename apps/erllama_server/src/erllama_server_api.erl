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

A run streams its events to the processes attached with `attach/3`
as `{erllama_server_run, RunId, Event}`. An event is
`#{seq := N, type := Type, data := Data}` with `Type` one of
`started`, `token`, `thinking`, `done`, `cancelled` and `error`;
the last three are terminal. Subscribers ack with
`RunPid ! {ack, self(), Seq}`; one that falls too far behind gets
`{erllama_server_run, RunId, {dropped, NextSeq}}` and reattaches
from `NextSeq`.
""".

-export([
    api_version/0,
    hello/1,
    status/0,
    models/0,
    load_model/2,
    unload_model/1,
    start_run/1,
    attach/3,
    detach/2,
    cancel/1,
    result/1,
    run/1,
    runs/0,
    conversation/1,
    conversations/0
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

-doc "Versions, loaded models, active runs and memory pressure.".
-spec status() -> {ok, map()}.
status() ->
    {ok, Models} = models(),
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
        models => Models,
        active_runs => erllama_server_runs:active_count(),
        pressure => Pressure
    }}.

-doc "The loaded models.".
-spec models() -> {ok, [map()]}.
models() ->
    Keys = [id, status, backend, context_size, quant_tag, tier],
    {ok, [maps:with(Keys, I) || I <- erllama:list_models()]}.

-doc """
Load a local GGUF file. `Opts`: `id` (default: the file name without
`.gguf`), `n_ctx`, `n_gpu_layers`. Superseded by the model registry.
""".
-spec load_model(binary() | string(), map()) -> {ok, binary()} | {error, term()}.
load_model(Path0, Opts) when is_map(Opts) ->
    Path = unicode:characters_to_binary(Path0),
    case filelib:is_regular(Path) of
        false ->
            {error, {enoent, Path}};
        true ->
            Id = maps:get(id, Opts, filename:basename(Path, <<".gguf">>)),
            Config0 = #{
                model_path => Path,
                model_opts => maps:with([n_gpu_layers], Opts),
                context_opts => maps:with([n_ctx], Opts)
            },
            %% `context_size' is the cache's copy of `n_ctx'.
            Config =
                case Opts of
                    #{n_ctx := N} -> Config0#{context_size => N};
                    _ -> Config0
                end,
            erllama:load_model(Id, Config)
    end.

-doc "Unload a model. Its runs end with an error event.".
-spec unload_model(binary()) -> ok | {error, term()}.
unload_model(Id) ->
    erllama:unload(Id).

-doc """
Start a run and return its id. `Spec`:

- `model` (required): a loaded model id.
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
start_run(#{model := Model} = Spec) ->
    Opts = maps:get(opts, Spec, #{}),
    case erllama_server_generate:check_opts(Opts) of
        ok ->
            case erllama:whereis(Model) of
                {ok, _} -> start_input(input(Spec), Model, Opts, Spec);
                {error, not_loaded} -> {error, {model_not_loaded, Model}}
            end;
        {error, _} = E ->
            E
    end;
start_run(_Spec) ->
    {error, {missing, model}}.

start_input({error, _} = E, _Model, _Opts, _Spec) ->
    E;
start_input({prompt, _} = Input, Model, Opts, _Spec) ->
    RunId = erllama_server_ids:run(),
    launch(#{run_id => RunId, model => Model, input => Input, opts => Opts});
start_input({messages, New}, Model, Opts, Spec) ->
    RunId = erllama_server_ids:run(),
    ConvId0 = maps:get(conversation_id, Spec, undefined),
    case erllama_server_convs:begin_run(ConvId0, Model, RunId) of
        {ok, #{id := ConvId, messages := History, parent_key := PK}} ->
            RunSpec = #{
                run_id => RunId,
                conversation_id => ConvId,
                model => Model,
                input => {messages, History ++ New},
                new_messages => New,
                parent_key => PK,
                opts => Opts
            },
            case launch(RunSpec) of
                {ok, _} = Ok ->
                    Ok;
                {error, _} = E ->
                    ok = erllama_server_convs:release(ConvId, RunId),
                    E
            end;
        {error, _} = E ->
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
input(#{prompt := _, conversation_id := _}) ->
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
    with_run(RunId, fun(RunPid) -> erllama_server_run:attach(RunPid, Pid, Cursor) end).

-doc "Stop sending a run's events to `Pid`. The run goes on.".
-spec detach(binary(), pid()) -> ok | {error, term()}.
detach(RunId, Pid) ->
    with_run(RunId, fun(RunPid) -> erllama_server_run:detach(RunPid, Pid) end).

-doc "Cancel a run. Its last event is `cancelled`. Idempotent.".
-spec cancel(binary()) -> ok | {error, term()}.
cancel(RunId) ->
    with_run(RunId, fun erllama_server_run:cancel/1).

-doc """
The result of a finished run: `status`, `content`, `reasoning`,
`tool_calls`, `finish_reason`, `usage`, `timings`; or `status` and
`error` for a failed run. `{error, running}` before it finishes.
""".
-spec result(binary()) -> {ok, map()} | {error, term()}.
result(RunId) ->
    with_run(RunId, fun erllama_server_run:result/1).

-doc "One run's id, conversation, model, status and next sequence.".
-spec run(binary()) -> {ok, map()} | {error, term()}.
run(RunId) ->
    with_run(RunId, fun erllama_server_run:info/1).

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
