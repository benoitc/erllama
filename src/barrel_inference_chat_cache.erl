%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(barrel_inference_chat_cache).
-moduledoc """
LRU cache for `barrel_inference_chat' templates_ref AND params_ref.

Two slots share one ETS table and one LRU budget:

- `{templates, ModelIdBin}` → `templates_ref'.
  The autoparser's `templates_init' is heavy (jinja parse + setup);
  amortise by caching the resulting `templates_ref' per `ModelIdBin'.
- `{params, ModelIdBin, ToolsHash, ToolChoice, Parallel}` → `params_ref'.
  The autoparser's PEG arena synthesis (formerly inside
  `templates_apply') is even heavier; with the render-only NIF split
  we cache the parser per `(tools-hash, tool_choice,
  parallel_tool_calls)' on the model. Per-request work is the cheap
  jinja render via `barrel_inference_chat:render_only/2'.

The cache is keyed on the stable `ModelIdBin' binary so cached
entries never extend a model's lifetime past unload. The model
layer calls `purge/1' on its `terminate/1' to drop every entry
for the model (both slots).

Eviction (LRU + `purge/1') removes the resource term from ETS;
the underlying NIF resource destructor runs on the next BEAM GC.

`get_or_init/3' and `get_or_make_params/4' invoke NIFs that need
real model + templates resources; the unit tests exercise `put/3' +
`lookup/2' directly on synthetic terms. End-to-end "double-call
returns same ref" guarantees live in `barrel_inference_chat_SUITE'.
""".

-behaviour(gen_server).

%% Public API
-export([
    start_link/0,
    get_or_init/3,
    get_or_make_params/3,
    purge/1
]).

%% Test-only seam: ETS-level put/lookup with opaque payloads. Allow
%% unit tests to exercise LRU + purge semantics without a real model.
-export([put/3, lookup/2]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-define(SERVER, ?MODULE).
-define(TAB, ?MODULE).
-define(DEFAULT_SIZE, 64).

-record(state, {
    max_size :: pos_integer(),
    %% Monotonic insertion counter; ties broken by larger counter wins
    %% (eviction drops the smallest counter).
    seq = 0 :: non_neg_integer()
}).

-type state() :: #state{}.

%%====================================================================
%% Public API
%%====================================================================

start_link() ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, [], []).

%% Initialise (or retrieve) the templates_ref for a model. Caches by
%% ModelIdBin; calling twice with the same id returns the SAME ref.
-spec get_or_init(
    ModelIdBin :: binary(),
    ModelRef :: barrel_inference_nif:model_ref(),
    TemplateOverride :: binary() | undefined
) ->
    {ok, barrel_inference_chat:templates_ref()} | {error, term()}.
get_or_init(ModelIdBin, ModelRef, TemplateOverride) when
    is_binary(ModelIdBin)
->
    cached_or_build(
        {templates, ModelIdBin},
        fun() -> barrel_inference_chat:init(ModelRef, TemplateOverride) end
    ).

%% Get (or build + cache) a params_ref for the given templates +
%% (tools, tool_choice, parallel_tool_calls). The cache key is
%% derived from a SHA-256 of the JSON-encoded tools list, so two
%% requests with the same tools schema (the common case across turns
%% of one conversation) share one parser. Inputs is the SAME map
%% `barrel_inference_chat:make_params/2' expects (messages, tools,
%% tool_choice, parallel_tool_calls).
-spec get_or_make_params(
    ModelIdBin :: binary(),
    Templates :: barrel_inference_chat:templates_ref(),
    Inputs :: map()
) ->
    {ok, barrel_inference_chat:params_ref()} | {error, term()}.
get_or_make_params(ModelIdBin, Templates, Inputs) when
    is_binary(ModelIdBin), is_map(Inputs)
->
    cached_or_build(
        params_key(ModelIdBin, Inputs),
        fun() -> barrel_inference_chat:make_params(Templates, Inputs) end
    ).

%% Shared cache fast path: return the cached ref on hit, or run the
%% builder fun and insert its result on miss.
cached_or_build(Key, BuildFn) ->
    case lookup(?TAB, Key) of
        {ok, Ref} ->
            {ok, Ref};
        not_found ->
            case BuildFn() of
                {ok, Ref} ->
                    ok = put(?TAB, Key, Ref),
                    {ok, Ref};
                Err ->
                    Err
            end
    end.

%% Build the params-cache key. SHA-256 over the tools JSON keeps it
%% bounded regardless of how big the tools array gets; the atoms
%% (`tool_choice', `parallel_tool_calls') hash by value verbatim.
params_key(ModelIdBin, Inputs) ->
    Tools = maps:get(tools, Inputs, <<"[]">>),
    ToolChoice = maps:get(tool_choice, Inputs, auto),
    Parallel = maps:get(parallel_tool_calls, Inputs, false),
    ToolsHash = crypto:hash(sha256, iolist_to_binary(Tools)),
    {params, ModelIdBin, ToolsHash, ToolChoice, Parallel}.

%% Drop every entry whose key references the given ModelIdBin (both
%% the templates slot and any params slots). Called by the model
%% layer's terminate/1 on unload so cached refs do not extend
%% templates_ref / params_ref lifetimes past unload.
-spec purge(binary()) -> ok.
purge(ModelIdBin) when is_binary(ModelIdBin) ->
    gen_server:call(?SERVER, {purge, ModelIdBin}).

%%====================================================================
%% Test-only ETS seam
%%====================================================================

-doc """
Test-only: insert an opaque payload under Key. Used by the cache
unit tests to exercise put/get round-trip + LRU eviction without
invoking the chat NIF.
""".
-spec put(atom(), term(), term()) -> ok.
put(Tab, Key, Value) when Tab =:= ?TAB ->
    gen_server:call(?SERVER, {put, Key, Value}).

-doc """
Test-only: read an opaque payload by Key. Returns {ok, Value} on hit
or `not_found' otherwise.
""".
-spec lookup(atom(), term()) -> {ok, term()} | not_found.
lookup(Tab, Key) when Tab =:= ?TAB ->
    case ets:lookup(?TAB, Key) of
        [{Key, Value, _Seq}] -> {ok, Value};
        [] -> not_found
    end.

%%====================================================================
%% gen_server callbacks
%%====================================================================

-spec init([]) -> {ok, state()}.
init([]) ->
    _ = ets:new(?TAB, [set, named_table, protected, {read_concurrency, true}]),
    MaxSize = application:get_env(
        barrel_inference, chat_params_cache_size, ?DEFAULT_SIZE
    ),
    {ok, #state{max_size = MaxSize}}.

handle_call({put, Key, Value}, _From, S = #state{seq = Seq, max_size = Max}) ->
    NewSeq = Seq + 1,
    true = ets:insert(?TAB, {Key, Value, NewSeq}),
    Size = ets:info(?TAB, size),
    case Size > Max of
        true -> evict_oldest();
        false -> ok
    end,
    {reply, ok, S#state{seq = NewSeq}};
handle_call({purge, ModelIdBin}, _From, S) ->
    %% Walk the table; drop every entry whose key references this id.
    %% ETS is small (LRU-bounded) so a full scan is fine.
    ets:foldl(
        fun({Key, _V, _Seq}, _Acc) ->
            case Key of
                {templates, MId} when MId =:= ModelIdBin ->
                    ets:delete(?TAB, Key);
                {params, MId, _ToolsH, _TC, _Par} when MId =:= ModelIdBin ->
                    ets:delete(?TAB, Key);
                _ ->
                    ok
            end
        end,
        ok,
        ?TAB
    ),
    {reply, ok, S};
handle_call(_Msg, _From, S) ->
    {reply, {error, unknown_call}, S}.

handle_cast(_Msg, S) ->
    {noreply, S}.

handle_info(_Msg, S) ->
    {noreply, S}.

%%====================================================================
%% Internals
%%====================================================================

evict_oldest() ->
    %% Find the key with the smallest seq and delete it. LRU on
    %% insertion order (not access order); promotion-on-read costs
    %% an extra write per hit and is not needed for this workload.
    {OldestKey, _MinSeq} =
        ets:foldl(
            fun({K, _V, S}, {AccK, AccS}) ->
                case AccK of
                    undefined -> {K, S};
                    _ when S < AccS -> {K, S};
                    _ -> {AccK, AccS}
                end
            end,
            {undefined, infinity},
            ?TAB
        ),
    case OldestKey of
        undefined -> ok;
        _ -> ets:delete(?TAB, OldestKey)
    end,
    ok.
