%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_chat_cache).
-moduledoc false.
%% LRU cache for `erllama_chat' templates refs.
%%
%% One slot per model: `{templates, ModelIdBin}` → `templates_ref'.
%% The autoparser's `templates_init' is heavy (jinja parse + setup);
%% amortise by caching the resulting `templates_ref' per `ModelIdBin'.
%% Per-request work (`erllama_chat:apply/2`) is not cached.
%%
%% The cache is keyed on the stable `ModelIdBin' binary so cached
%% entries never extend a model's lifetime past unload. The model
%% layer calls `purge/1' on its `terminate/1' to drop every entry
%% for the model.
%%
%% Eviction (LRU + `purge/1') removes the resource term from ETS;
%% the underlying NIF resource destructor runs on the next BEAM GC.
%%
%% `get_or_init/3' invokes NIFs that need real model + templates
%% resources; the unit tests exercise `put/3' +
%% `lookup/2' directly on synthetic terms. End-to-end "double-call
%% returns same ref" guarantees live in `erllama_chat_SUITE'.

-behaviour(gen_server).

%% Public API
-export([
    start_link/0,
    get_or_init/3,
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
    ModelRef :: erllama_nif:model_ref(),
    TemplateOverride :: binary() | undefined
) ->
    {ok, erllama_chat:templates_ref()} | {error, term()}.
get_or_init(ModelIdBin, ModelRef, TemplateOverride) when
    is_binary(ModelIdBin)
->
    cached_or_build(
        {templates, ModelIdBin},
        fun() -> erllama_chat:init(ModelRef, TemplateOverride) end
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

%% Drop the templates entry for the given ModelIdBin. Called by the model
%% layer's terminate/1 on unload so the cached ref does not extend
%% the templates_ref lifetime past unload.
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
        erllama, chat_cache_size, ?DEFAULT_SIZE
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
