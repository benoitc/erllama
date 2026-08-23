%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_cache).
-moduledoc """
Public API of the KV cache: tier setup, inspection, eviction and
counters.

The RAM tier is always on. Add a `disk` or `ram_file` tier with
`add_tier/1` (or the `tiers` application environment key) and point a
model at it with `tier` + `tier_srv` in its load config:

```erlang
ok = erllama_cache:add_tier(#{name => kv_disk, backend => disk,
                              root => "/var/lib/erllama/kv"}),
{ok, M} = erllama:load_model(#{model_path => Path,
                               tier => disk, tier_srv => kv_disk}).
```

Tier servers, the meta server, the writer pool and the policy module
are internal; the runtime plumbing lives in `erllama_cache_meta_srv`
and `erllama_cache_writer`.
""".

-export([
    add_tier/1,
    remove_tier/1,
    list_tiers/0,
    info/0,
    get_counters/0,
    reset_counters/0,
    gc/0,
    evict_bytes/1,
    evict_bytes/2,
    lookup_longest_text_prefix/2
]).
-export([start_configured_tiers/0]).

-include("erllama_cache.hrl").

-export_type([
    cache_key/0,
    tier/0,
    tier_spec/0,
    tier_info/0,
    status/0,
    save_reason/0
]).

-doc "Spec for `add_tier/1` and the `tiers` environment key.".
-type tier_spec() :: #{
    name := atom(),
    backend := disk | ram_file,
    root := file:name()
}.
-doc "One entry of `list_tiers/0`.".
-type tier_info() :: #{
    name := atom(),
    backend := tier(),
    root := file:name() | undefined,
    pid := pid()
}.

-doc """
Start a supervised `disk` or `ram_file` tier server registered as
`name` and rooted at `root` (created if missing). Models reference it
with `tier_srv => name` and `tier => backend`.
""".
-spec add_tier(tier_spec()) ->
    ok | {error, already_started | {invalid_tier, term()} | term()}.
add_tier(#{name := Name, backend := Backend, root := Root}) when
    is_atom(Name), Backend =:= disk; is_atom(Name), Backend =:= ram_file
->
    case filelib:ensure_path(Root) of
        ok ->
            case erllama_cache_tier_sup:start_tier(Name, Backend, Root) of
                {ok, _Pid} -> ok;
                {error, _} = E -> E
            end;
        {error, Reason} ->
            {error, {invalid_tier, {root, Reason}}}
    end;
add_tier(Other) ->
    {error, {invalid_tier, Other}}.

-doc "Stop a tier started with `add_tier/1`. Rows saved in it become unavailable.".
-spec remove_tier(atom()) -> ok | {error, not_found}.
remove_tier(Name) when is_atom(Name) ->
    erllama_cache_tier_sup:stop_tier(Name).

-doc "Every tier server, the built-in RAM tier included.".
-spec list_tiers() -> [tier_info()].
list_tiers() ->
    Ram =
        case whereis(erllama_cache_ram) of
            undefined -> [];
            Pid -> [#{name => erllama_cache_ram, backend => ram, root => undefined, pid => Pid}]
        end,
    Ram ++
        [
            #{
                name => Name,
                backend => erllama_cache_disk_srv:tier_of(Name),
                root => erllama_cache_disk_srv:root_of(Name),
                pid => Pid
            }
         || {Name, Pid} <- erllama_cache_tier_sup:tiers()
        ].

-doc """
Snapshot of the cache: the tiers, the number of cached rows and the
bytes held per tier backend.
""".
-spec info() ->
    #{
        tiers := [tier_info()],
        entries := non_neg_integer(),
        bytes := #{
            ram := non_neg_integer(), ram_file := non_neg_integer(), disk := non_neg_integer()
        }
    }.
info() ->
    Rows = erllama_cache_meta_srv:dump(),
    Bytes = lists:foldl(
        fun(Row, Acc) ->
            Tier = element(?POS_TIER, Row),
            Size = element(?POS_SIZE, Row),
            maps:update_with(Tier, fun(B) -> B + Size end, Size, Acc)
        end,
        #{ram => 0, ram_file => 0, disk => 0},
        Rows
    ),
    #{tiers => list_tiers(), entries => length(Rows), bytes => Bytes}.

-doc false.
%% Started by `erllama_cache_sup' after the tier supervisor: adds every
%% tier listed in the `tiers' application environment key, then exits
%% normally (transient child, so a failed tier is logged, not fatal).
-spec start_configured_tiers() -> ignore.
start_configured_tiers() ->
    Specs = application:get_env(erllama, tiers, []),
    lists:foreach(
        fun(Spec) ->
            case add_tier(Spec) of
                ok ->
                    ok;
                {error, already_started} ->
                    ok;
                {error, Reason} ->
                    logger:error("erllama: cannot start cache tier ~p: ~p", [Spec, Reason])
            end
        end,
        Specs
    ),
    ignore.

-doc """
Snapshot of operational counters as a map of slot name to
non-negative integer. Suitable for forwarding to a metrics exporter
(Prometheus, statsd, OpenTelemetry).
""".
-spec get_counters() -> #{atom() => non_neg_integer()}.
get_counters() ->
    erllama_cache_counters:snapshot().

-doc """
Reset all counters to 0. Mostly for tests; production callers should
treat counters as monotonic-since-boot.
""".
-spec reset_counters() -> ok.
reset_counters() ->
    erllama_cache_counters:reset().

-doc """
Synchronous full eviction pass. Walks the LRU and drops every
available row with refcount=0. Returns the number evicted.
""".
-spec gc() -> {evicted, non_neg_integer()}.
gc() ->
    erllama_cache_meta_srv:gc().

-doc """
Evict oldest available rows until at least TargetBytes have been
freed. Returns `{evicted, NumRows, BytesFreed}`.
""".
-spec evict_bytes(non_neg_integer()) ->
    {evicted, non_neg_integer(), non_neg_integer()}.
evict_bytes(TargetBytes) ->
    erllama_cache_meta_srv:evict_bytes(TargetBytes).

-doc """
Like `evict_bytes/1`, but only considers rows whose tier is in
Tiers. Pass `all` to match every tier, or a subset of
`[ram, ram_file, disk]`. The system-pressure scheduler uses this to
evict only RAM-resident slabs while leaving the disk tier alone.
""".
-spec evict_bytes(non_neg_integer(), all | [tier()]) ->
    {evicted, non_neg_integer(), non_neg_integer()}.
evict_bytes(TargetBytes, Tiers) ->
    erllama_cache_meta_srv:evict_bytes(TargetBytes, Tiers).

-doc """
Find the longest cached rendered-byte prefix of `PromptBytes` for
the given key namespace (`#{fingerprint, quant_type,
ctx_params_hash}`). Content-addressed (ds4-style): returns
`{ok, MatchBytes, Row}` where `MatchBytes` is the byte length of the
longest stored prompt prefix, or `miss`. Operator-friendly entry
point for stateless callers (HTTP front-end, agent loops) that
resend the full conversation each turn.
""".
-spec lookup_longest_text_prefix(map(), binary()) ->
    {ok, non_neg_integer(), tuple()} | miss.
lookup_longest_text_prefix(KeyMeta, PromptBytes) ->
    erllama_cache_meta_srv:lookup_longest_text_prefix(KeyMeta, PromptBytes).

-type cache_key() :: <<_:256>>.
-type tier() :: ram | ram_file | disk.
-type status() :: available | writing | evicting.
-type save_reason() :: cold | continued | finish | evict | shutdown.
