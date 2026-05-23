%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(barrel_inference_cache_ramfile_srv).
-moduledoc """
RAM-file tier server.

Identical mechanics to `barrel_inference_cache_disk_srv` (same KVC framing,
same temp+datasync+link+fsync_dir publish protocol) — the only
difference is the root directory points at a tmpfs mount such as
`/dev/shm` so the bytes never touch a spinning disk. The tier
label `ram_file` is what the meta server records, which lets the
scheduler reason about durability separately from latency.

Implementation is a delegating wrapper; all real work lives in
`barrel_inference_cache_disk_srv`.
""".

-export([start_link/2, save/3, load/2, delete/2, dir/1, scan/1]).

-spec start_link(atom(), file:name()) -> {ok, pid()} | {error, term()}.
start_link(Name, RootDir) ->
    barrel_inference_cache_disk_srv:start_link(Name, ram_file, RootDir).

-spec save(atom(), barrel_inference_cache_kvc:build_meta(), binary()) ->
    {ok, barrel_inference_cache:cache_key(), binary(), non_neg_integer()}
    | {error, term()}.
save(SrvName, BuildMeta, Payload) ->
    barrel_inference_cache_disk_srv:save(SrvName, BuildMeta, Payload).

-spec load(atom(), barrel_inference_cache:cache_key()) ->
    {ok, barrel_inference_cache_kvc:info(), binary()} | miss | {error, term()}.
load(SrvName, Key) ->
    barrel_inference_cache_disk_srv:load(SrvName, Key).

-spec delete(atom(), barrel_inference_cache:cache_key()) -> ok.
delete(SrvName, Key) ->
    barrel_inference_cache_disk_srv:delete(SrvName, Key).

-spec dir(atom()) -> file:name().
dir(SrvName) ->
    barrel_inference_cache_disk_srv:dir(SrvName).

-spec scan(atom()) ->
    [{barrel_inference_cache:cache_key(), binary(), non_neg_integer()}].
scan(SrvName) ->
    barrel_inference_cache_disk_srv:scan(SrvName).
