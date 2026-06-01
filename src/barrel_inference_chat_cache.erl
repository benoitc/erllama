%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(barrel_inference_chat_cache).
-moduledoc """
LRU cache for `barrel_inference_chat' templates_ref + params_ref.

The autoparser's templates_apply path is heavy (re-renders the jinja
template several times to diff with/without tools). Parse-only is
cheap. The cache amortises template apply by `{ModelIdBin, ToolsHash}'.

The cache is keyed on the stable `ModelIdBin' binary (NOT on a model
resource ref) so cached entries never extend a model's lifetime past
unload. The model layer calls `purge/1' on its `terminate/1' to drop
every entry for the model.

Eviction (LRU + `purge/1') removes the resource term from ETS; the
underlying NIF resource destructor runs on the next BEAM GC. Tests
do NOT assert destructor timing; the cache contract is "no reachable
Erlang reference after eviction."

`get_or_init/3' and `get_or_apply/4' invoke NIFs that need a real
model resource; the unit tests in
`barrel_inference_chat_cache_tests' exercise `put/3' + `lookup/2'
directly on synthetic terms. The "double-init returns same ref"
high-level guarantee is covered in the real-model
`barrel_inference_chat_SUITE'.
""".

-behaviour(gen_server).

%% Public API
-export([
    start_link/0,
    get_or_init/3,
    get_or_apply/4,
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
    Key = {templates, ModelIdBin},
    case lookup(?TAB, Key) of
        {ok, Ref} ->
            {ok, Ref};
        not_found ->
            case barrel_inference_chat:init(ModelRef, TemplateOverride) of
                {ok, Ref} ->
                    ok = put(?TAB, Key, Ref),
                    {ok, Ref};
                Err ->
                    Err
            end
    end.

%% Apply a tools-set to a templates_ref and cache the result. Caches
%% by {ModelIdBin, ToolsHash}; same key returns the same params_ref
%% and the same rendered prompt bytes.
-spec get_or_apply(
    ModelIdBin :: binary(),
    ToolsHash :: binary(),
    TemplatesRef :: barrel_inference_chat:templates_ref(),
    Inputs :: map()
) ->
    {ok, barrel_inference_chat:params_ref(), binary()} | {error, term()}.
get_or_apply(ModelIdBin, ToolsHash, TemplatesRef, Inputs) when
    is_binary(ModelIdBin), is_binary(ToolsHash), is_map(Inputs)
->
    Key = {params, ModelIdBin, ToolsHash},
    case lookup(?TAB, Key) of
        {ok, {Params, PromptBin}} ->
            {ok, Params, PromptBin};
        not_found ->
            case barrel_inference_chat:apply(TemplatesRef, Inputs) of
                {ok, Params, PromptBin} ->
                    ok = put(?TAB, Key, {Params, PromptBin}),
                    {ok, Params, PromptBin};
                Err ->
                    Err
            end
    end.

%% Drop every entry whose key references the given ModelIdBin.
%% Called by the model layer's terminate/1 on unload so cached refs
%% do not extend templates_ref / params_ref lifetimes past unload.
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
                {params, MId, _Hash} when MId =:= ModelIdBin ->
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
