%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(barrel_inference_model_evictor).
-moduledoc """
Behaviour for proactive model eviction under memory pressure.

`barrel_inference_scheduler` is the engine-level memory-pressure responder, but
the model fleet (which models are loaded, their request recency, how to unload
one cleanly) is owned by the server layer, which the engine must not depend on.
This behaviour is the seam: the scheduler calls `evict_one/0` on an
operator-configured module (`scheduler.model_evictor`) when cache eviction could
not relieve sustained pressure, and the server ships an implementation.

`evict_one/0` returns `{unloaded, ModelId}` when it unloaded an idle model, or
`none` when nothing could be freed (no idle model, or all candidates turned busy).
It must not raise; the scheduler additionally guards the call.
""".

-export([evict_one/1]).

-callback evict_one() -> {unloaded, binary()} | none.

%% Safe dispatch to a configured evictor module. Lives here (the
%% behaviour-defining module) so the dynamic call is idiomatic, mirroring
%% `barrel_inference_pressure:sample/1`. Normalises the return - not just
%% exceptions - because `model_evictor` is an operator-supplied atom, so a
%% wrong/garbage return must not crash the scheduler.
-spec evict_one(module()) -> {unloaded, binary()} | none.
evict_one(Mod) ->
    try Mod:evict_one() of
        {unloaded, Id} when is_binary(Id) -> {unloaded, Id};
        none -> none;
        _ -> none
    catch
        _:_ -> none
    end.
