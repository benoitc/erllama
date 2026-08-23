%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_middleware).
-moduledoc """
Middleware around the `erllama` API calls, in the style of hackney's
middleware: a middleware is a plain fun that receives the request and
a `Next` fun, and returns the response. It can observe, rewrite,
short-circuit or wrap a call. No behaviour, no registry, no
dependencies.

```erlang
-type request() :: #{op := atom(), model := erllama:model() | undefined, args := map()}.
-type next() :: fun((request()) -> term()).
-type middleware() :: fun((request(), next()) -> term()).
```

Chain order is outermost first: `[A, B]` means A wraps B, the
request flows `A -> B -> erllama` and the response unwinds
`erllama -> B -> A`.

Install a global chain for every call:

```erlang
application:set_env(erllama, middleware, [Log, Metrics]).
```

or a per-call chain, which replaces the global one for that call:

```erlang
erllama:complete(Model, Prompt, #{middleware => [Log]}).
```

Wrapped operations and their `args`:

| op | args |
|---|---|
| `load_model` | `#{config := map()}` (`model` is the id) |
| `unload` | `#{}` |
| `complete` | `#{prompt := binary(), opts := map()}` |
| `prefill_only` | `#{tokens := [token_id()], opts := map()}` |
| `stream` | `#{prompt := binary() \\| [token_id()], opts := map()}` |
| `continue` | `#{tokens := [token_id()], opts := map()}` |
| `chat` | `#{messages := [map()], opts := map()}` |
| `chat_apply` | `#{messages := [map()], opts := map()}` |
| `chat_parse` | `#{params := term(), input := binary(), partial := boolean()}` |
| `embed`, `embed_batch` | `#{input := term()}` |
| `tokenize` | `#{text := binary(), opts := map()}` |
| `detokenize` | `#{tokens := [token_id()]}` |

Streaming ops (`stream`, `continue`) return `{ok, Ref}` to the
middleware; the events are delivered later to the `to` process. To
observe completion, set `to` to a proxy process of your own.

A middleware that raises propagates to the caller; erllama does not
catch. Recipes (logging, Prometheus, telemetry, caching) are in the
middleware guide.
""".

-export([run/3, run/2, take/1, global/0]).
-export_type([request/0, next/0, middleware/0]).

-type request() :: #{op := atom(), model := erllama:model() | undefined, args := map()}.
-type next() :: fun((request()) -> term()).
-type middleware() :: fun((request(), next()) -> term()).

-doc """
Run `Request` through `Chain` and then `Next`. An empty chain calls
`Next` directly.
""".
-spec run(request(), [middleware()], next()) -> term().
run(Request, [], Next) ->
    Next(Request);
run(Request, [Mw | Rest], Next) ->
    Mw(Request, fun(Req1) -> run(Req1, Rest, Next) end).

-doc "`run/3` with the global chain from the `middleware` application environment key.".
-spec run(request(), next()) -> term().
run(Request, Next) ->
    run(Request, global(), Next).

-doc "The global chain (`application:get_env(erllama, middleware, [])`).".
-spec global() -> [middleware()].
global() ->
    case application:get_env(erllama, middleware, []) of
        Chain when is_list(Chain) -> Chain;
        _ -> []
    end.

-doc """
Split the `middleware` key out of an option map: returns the chain to
use (the per-call list when present, else the global chain) and the
options without the key.
""".
-spec take(map()) -> {[middleware()], map()}.
take(Opts) ->
    case maps:take(middleware, Opts) of
        {Chain, Rest} when is_list(Chain) -> {Chain, Rest};
        error -> {global(), Opts}
    end.
