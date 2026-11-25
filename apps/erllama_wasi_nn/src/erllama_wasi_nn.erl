%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_wasi_nn).
-moduledoc """
A `wasi_nn` backend: a WebAssembly component that imports `wasi:nn`
runs its models on erllama.

Name this module as the `backend` of the `nn` grant you pass to
erlang_wasm, and map each name a guest may load to a model you already
loaded with `erllama:load_model/2`:

```erlang
Grant = #{
    backend => erllama_wasi_nn,
    models => [<<"default">>, <<"embed">>],
    backend_opts => #{
        <<"default">> => #{
            model => <<"qwen">>,
            task => completion,
            request_opts => #{response_tokens => 256}
        },
        <<"embed">> => #{model => <<"bge">>, task => embedding}
    }
}.
```

A `completion` graph reads the first input tensor as a UTF-8 prompt and
answers one `u8` tensor named `output` holding the reply. An
`embedding` graph reads the first input tensor as UTF-8 text and
answers one `fp32` tensor named `embedding`, little-endian. The guest
sets nothing but the text: sampling and length come from
`request_opts`. The answer comes once; nothing streams.

The backend never loads or unloads a model, and `close/1` does
nothing: the embedder owns the models. The guide is `guides/wasi-nn.md`.
""".
-behaviour(wasi_nn).

-export([load_by_name/2, init_execution_context/1, compute/2, close/1]).
-export([error_reply/1]).

-export_type([backend_opts/0, entry/0, task/0, graph/0]).

-doc "What a graph does with its input.".
-type task() :: completion | embedding.

-doc """
One guest-visible model: the erllama model id, the task, and for a
completion the options handed to `erllama:complete/3`.
""".
-type entry() :: #{
    model := erllama:model_id(),
    task := task(),
    request_opts => erllama:request_opts()
}.

-doc "The `backend_opts` of the grant: guest-visible name to entry.".
-type backend_opts() :: #{binary() => entry()}.

-doc "A loaded graph, and its execution context: a checked entry.".
-type graph() :: #{
    model := erllama:model_id(),
    task := task(),
    request_opts := erllama:request_opts()
}.

%% =============================================================================
%% wasi_nn callbacks
%% =============================================================================

-doc """
Resolve a granted name to its entry. A name with no entry, or whose
model is not loaded, is `not_found`; an entry of the wrong shape is
`invalid_argument`.
""".
-spec load_by_name(binary(), term()) -> {ok, graph()} | {error, wasi_nn:error()}.
load_by_name(Name, BackendOpts) when is_map(BackendOpts) ->
    case BackendOpts of
        #{Name := Entry} -> graph(Entry);
        _ -> {error, {not_found, <<"no such model">>}}
    end;
load_by_name(_Name, _BackendOpts) ->
    {error, {not_found, <<"no such model">>}}.

-doc "A context is the graph itself: a request holds no state between calls.".
-spec init_execution_context(graph()) -> {ok, graph()}.
init_execution_context(Graph) ->
    {ok, Graph}.

-doc """
Run the task on the first input tensor, which must be `u8` and hold
UTF-8 text. Its name and its dimensions are ignored.
""".
-spec compute(graph(), [{binary(), wasi_nn:tensor()}]) ->
    {ok, [{binary(), wasi_nn:tensor()}]} | {error, wasi_nn:error()}.
compute(#{task := Task} = Graph, [{_Name, #{type := u8, data := Data}} | _]) when
    is_binary(Data)
->
    case unicode:characters_to_binary(Data, utf8, utf8) of
        Data -> run(Task, Graph, Data);
        _ -> {error, {invalid_argument, <<"the input is not UTF-8">>}}
    end;
compute(_Context, [_ | _]) ->
    {error, {invalid_argument, <<"the first input must be a u8 tensor">>}};
compute(_Context, _Inputs) ->
    {error, {invalid_argument, <<"no input">>}}.

-doc "Nothing to free: the embedder owns the model.".
-spec close({graph | context, term()}) -> ok.
close(_GraphOrContext) ->
    ok.

%% =============================================================================
%% Errors
%% =============================================================================

-doc """
The `wasi:nn` error a guest reads for an erllama error reason. The
message is the reason's atom name, or a fixed text when the reason is
not an atom: a reason can carry a path, so no term is formatted into it.
""".
-spec error_reply(erllama:error_reason()) -> wasi_nn:error().
error_reply(Reason) when is_atom(Reason) ->
    {error_code(Reason), atom_to_binary(Reason)};
error_reply(_Reason) ->
    {runtime_error, <<"the request failed">>}.

error_code(not_loaded) -> not_found;
error_code(not_found) -> not_found;
error_code(timeout) -> timeout;
error_code(decode_timeout) -> timeout;
error_code(too_large) -> too_large;
error_code(context_overflow) -> too_large;
error_code(oom) -> too_large;
error_code(not_supported) -> unsupported_operation;
error_code(chat_not_supported) -> unsupported_operation;
error_code(_Reason) -> runtime_error.

%% =============================================================================
%% Internal
%% =============================================================================

graph(#{model := Model, task := Task} = Entry) when
    is_binary(Model), Task =:= completion; is_binary(Model), Task =:= embedding
->
    case maps:get(request_opts, Entry, #{}) of
        Opts when is_map(Opts) ->
            case erllama:phase(Model) of
                {ok, _Phase} -> {ok, #{model => Model, task => Task, request_opts => Opts}};
                {error, Reason} -> {error, error_reply(Reason)}
            end;
        _ ->
            {error, {invalid_argument, <<"request_opts is not a map">>}}
    end;
graph(_Entry) ->
    {error, {invalid_argument, <<"the model entry needs a model id and a task">>}}.

run(completion, #{model := Model, request_opts := Opts}, Prompt) ->
    case erllama:complete(Model, Prompt, Opts) of
        {ok, #{reply := Reply}} ->
            {ok, [
                {<<"output">>, #{dimensions => [byte_size(Reply)], type => u8, data => Reply}}
            ]};
        {error, Reason} ->
            {error, error_reply(Reason)}
    end;
run(embedding, #{model := Model}, Text) ->
    case erllama:embed(Model, Text) of
        {ok, Vector} ->
            Data = <<<<F:32/float-little>> || F <- Vector>>,
            {ok, [
                {<<"embedding">>, #{dimensions => [length(Vector)], type => fp32, data => Data}}
            ]};
        {error, Reason} ->
            {error, error_reply(Reason)}
    end.
