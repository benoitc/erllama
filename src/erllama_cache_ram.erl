%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_cache_ram).
-moduledoc false.
%% RAM tier slab store.
%%
%% Owns the `erllama_cache_ram_slabs` ETS table. Reads and deletes are
%% direct ETS operations from any process. Stores go through this
%% server (`store/2`) so the byte budget is checked and enforced in
%% one place across every model: when a new slab would exceed the
%% budget, the oldest RAM rows are evicted first through the meta
%% server, which deletes their slabs with the ETS-direct `delete/1`
%% (never a call back into this server). The meta row for a stored
%% slab is registered by the writer (`erllama_cache_writer`), under
%% the same reservation protocol as the file tiers.
%%
%% The budget comes from the `ram_tier_budget` application env:
%% a byte count, `{fraction, F}` of total system memory (default
%% `{fraction, 0.25}`), or `0` to turn RAM saves off. A bad value
%% stops `start_link/0`, so a misconfigured node fails at boot.
%%
%% Slab binaries above 64 bytes live off-heap as refcounted ProcBins
%% so `ets:lookup` returns a binary reference, not a copy. Eviction
%% frees the slab binary by deleting its row.
-behaviour(gen_server).

-export([start_link/0, store/2, load/1, delete/1, size_bytes/0, budget/0]).
-export([resolve_budget/1]).
-export([init/1, handle_call/3, handle_cast/2]).

-define(SLABS, erllama_cache_ram_slabs).
-define(SERVER, ?MODULE).
-define(DEFAULT_BUDGET, {fraction, 0.25}).
%% Used when `{fraction, F}` is configured but the total system
%% memory cannot be read (os_mon not running).
-define(FALLBACK_TOTAL_BYTES, 4 * 1024 * 1024 * 1024).
-define(STORE_TIMEOUT_MS, 30000).

%% =============================================================================
%% Public API
%% =============================================================================

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, [], []).

-doc """
Store `Slab` under `Key` if it fits in the RAM budget, evicting the
oldest RAM rows to make room. Returns `{error, disabled}` when the
budget is 0, `{error, too_large}` when the slab alone exceeds it, and
`{error, ram_full}` when not enough rows could be evicted (checked
out or pinned rows are kept).
""".
-spec store(erllama_cache:cache_key(), binary()) ->
    ok | {error, disabled | too_large | ram_full}.
store(Key, Slab) when is_binary(Slab) ->
    gen_server:call(?SERVER, {store, Key, Slab}, ?STORE_TIMEOUT_MS).

-spec load(erllama_cache:cache_key()) -> {ok, binary()} | miss.
load(Key) ->
    case ets:lookup(?SLABS, Key) of
        [{_, Slab}] -> {ok, Slab};
        [] -> miss
    end.

-spec delete(erllama_cache:cache_key()) -> ok.
delete(Key) ->
    %% Idempotent against a non-existent table; the ram tier may not
    %% have been started yet (early init) or may have crashed and not
    %% yet been restarted.
    try
        ets:delete(?SLABS, Key),
        ok
    catch
        error:badarg -> ok
    end.

-spec size_bytes() -> non_neg_integer().
size_bytes() ->
    ets:foldl(fun({_, S}, Acc) -> Acc + byte_size(S) end, 0, ?SLABS).

-doc "The RAM tier byte budget this node resolved at start.".
-spec budget() -> non_neg_integer().
budget() ->
    gen_server:call(?SERVER, budget).

-doc """
Resolve a `ram_tier_budget` value to a byte count. `{fraction, F}`
reads total system memory from os_mon's memsup; when memsup is not
running it falls back to a fraction of 4 GiB.
""".
-spec resolve_budget(term()) -> {ok, non_neg_integer()} | {error, term()}.
resolve_budget(Bytes) when is_integer(Bytes), Bytes >= 0 ->
    {ok, Bytes};
resolve_budget({fraction, F}) when is_number(F), F > 0, F =< 1 ->
    {ok, round(F * total_memory())};
resolve_budget(Other) ->
    {error, {invalid_config, ram_tier_budget, Other}}.

%% =============================================================================
%% gen_server callbacks
%% =============================================================================

-spec init([]) -> {ok, #{budget := non_neg_integer()}} | {stop, term()}.
init([]) ->
    case resolve_budget(application:get_env(erllama, ram_tier_budget, ?DEFAULT_BUDGET)) of
        {ok, Budget} ->
            _ = ets:new(?SLABS, [
                named_table,
                public,
                set,
                {read_concurrency, true},
                {write_concurrency, true}
            ]),
            {ok, #{budget => Budget}};
        {error, Reason} ->
            {stop, Reason}
    end.

handle_call({store, Key, Slab}, _From, #{budget := Budget} = S) ->
    {reply, do_store(Key, Slab, Budget), S};
handle_call(budget, _From, #{budget := Budget} = S) ->
    {reply, Budget, S};
handle_call(_Msg, _From, S) ->
    {reply, {error, unknown_call}, S}.

-spec handle_cast(term(), term()) -> {noreply, term()}.
handle_cast(_Msg, S) ->
    {noreply, S}.

%% =============================================================================
%% Internal
%% =============================================================================

do_store(_Key, _Slab, 0) ->
    {error, disabled};
do_store(_Key, Slab, Budget) when byte_size(Slab) > Budget ->
    {error, too_large};
do_store(Key, Slab, Budget) ->
    Size = byte_size(Slab),
    case size_bytes() + Size - Budget of
        Need when Need > 0 ->
            _ = erllama_cache_meta_srv:evict_bytes(Need, [ram]),
            ok;
        _ ->
            ok
    end,
    case size_bytes() + Size =< Budget of
        true ->
            ets:insert(?SLABS, {Key, Slab}),
            ok;
        false ->
            {error, ram_full}
    end.

total_memory() ->
    try memsup:get_system_memory_data() of
        Data ->
            case
                proplists:get_value(
                    system_total_memory, Data, proplists:get_value(total_memory, Data)
                )
            of
                Total when is_integer(Total), Total > 0 -> Total;
                _ -> ?FALLBACK_TOTAL_BYTES
            end
    catch
        _:_ -> ?FALLBACK_TOTAL_BYTES
    end.
