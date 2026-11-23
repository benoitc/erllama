%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
-module(erllama_tp).
-moduledoc """
Experimental: this node's GPU as one rank of a tensor-parallel group.

Each rank is its own OS process (an Erlang node) with one GPU. Ranks
join the GPU backend's communicator together; afterwards a collective
such as `reduce/2` sums a buffer over every rank on the GPUs' own data
path. This module holds only rank-local resources: forming the group
(which ranks, when, under which generation) is the caller's, through
the `Join` function given to `open/5`.

`Join(Round, Bytes)` is called once per bootstrap round with this
rank's bytes. It must return every rank's bytes for that round,
concatenated in rank order (`{ok, All}`), or `{error, Reason}`. All
ranks see the same rounds in the same order. `{call, Server}` stands
for a coordinator process (possibly on another node) that answers
`gen_server:call(Server, {exchange, Rank, Round, Bytes})` that way.
`open/5` returns once
every rank joined, or with an error within its timeout: a rank never
waits alone.

A failure inside a collective (a rank lost mid-reduction) ends the
rank's OS process: the backend aborts on it. A native call blocked on
a lost peer is ended by stopping the node, never by an Erlang timeout.

Needs a GPU backend built with per-process communicators (CUDA with
NCCL); `caps/0` says whether this build has one.
""".

-export([caps/0, open/5, matvec/3, reduce/2, close/1, bootstrap_selftest/5]).

-export_type([rank/0, join/0]).

-define(AGENT_IDLE_MS, 3600000).

-type rank() :: erllama_nif:tp_rank_ref().
-type join() ::
    fun((Round :: pos_integer(), Bytes :: binary()) -> {ok, binary()} | {error, term()})
    | {call, gen_server:server_ref()}.

-doc """
The number of GPU devices, the first one's name, and whether its
backend can join a per-process communicator (`init_rank`) and
all-reduce.
""".
-spec caps() ->
    {ok, #{
        gpus := non_neg_integer(),
        init_rank := boolean(),
        allreduce := boolean(),
        device => string()
    }}
    | {error, term()}.
caps() ->
    erllama_nif:tp_caps().

-doc """
Join a group of `World` ranks as rank `Rank`, on this node's `Gpu`-th
GPU device. Blocks until the group is formed or `TimeoutMs` passed.
Errors: `not_supported` (no such GPU, or no per-process
communicator in this build), `timeout`, `init_failed`, or the reason
`Join` returned.
""".
-spec open(non_neg_integer(), non_neg_integer(), pos_integer(), join(), non_neg_integer()) ->
    {ok, rank()} | {error, term()}.
open(Gpu, Rank, World, {call, Server}, TimeoutMs) ->
    Join = fun(Round, Bytes) ->
        gen_server:call(Server, {exchange, Rank, Round, Bytes}, infinity)
    end,
    open(Gpu, Rank, World, Join, TimeoutMs);
open(Gpu, Rank, World, Join, TimeoutMs) when
    is_integer(Gpu),
    Gpu >= 0,
    is_integer(Rank),
    Rank >= 0,
    is_integer(World),
    Rank < World,
    is_function(Join, 2)
->
    with_agent(Join, fun(Agent) -> erllama_nif:tp_open(Gpu, Rank, World, Agent, TimeoutMs) end).

-doc """
`y = W·x` on this rank's GPU for its share of the columns, then the sum
of every rank's `y`. `W` is a list of rows of this rank's columns, `X`
this rank's slice of the input. Returns this rank's partial and the
reduced result.
""".
-spec matvec(rank(), [[float()]], [float()]) -> {ok, [float()], [float()]} | {error, term()}.
matvec(Rank, [Row | _] = W, X) when length(Row) =:= length(X) ->
    Bin = f32([V || R <- W, V <- R]),
    case erllama_nif:tp_matvec(Rank, length(W), length(X), Bin, f32(X)) of
        {ok, Partial, Reduced} -> {ok, floats(Partial), floats(Reduced)};
        {error, _} = E -> E
    end.

-doc """
Sum `Values` over every rank (all ranks call it with buffers of the
same length). Values as a list of floats or a binary of native F32;
the result comes back in the same form.
""".
-spec reduce(rank(), [float()] | binary()) -> {ok, [float()] | binary()} | {error, term()}.
reduce(Rank, Values) when is_binary(Values) ->
    erllama_nif:tp_reduce(Rank, Values);
reduce(Rank, Values) when is_list(Values) ->
    case erllama_nif:tp_reduce(Rank, f32(Values)) of
        {ok, Bin} -> {ok, floats(Bin)};
        {error, _} = E -> E
    end.

-doc "Leave the group: free the communicator and the GPU backend.".
-spec close(rank()) -> ok.
close(Rank) ->
    erllama_nif:tp_close(Rank).

-doc false.
%% The bootstrap exchange of open/5 without a GPU: `Rounds` rounds of
%% `Payload` through `Join`, exactly as a communicator's init makes
%% them. Returns what each round gathered.
-spec bootstrap_selftest(pos_integer(), binary(), pos_integer(), join(), non_neg_integer()) ->
    {ok, [binary()]} | {error, term()}.
bootstrap_selftest(World, Payload, Rounds, Join, TimeoutMs) ->
    with_agent(Join, fun(Agent) ->
        erllama_nif:tp_bootstrap_selftest(World, Payload, Rounds, Agent, TimeoutMs)
    end).

%% -----------------------------------------------------------------------------

%% Run `Fun(Agent)` while an agent process serves the bootstrap
%% requests the native call sends it. A `rejected` error carries the
%% reason `Join` gave.
with_agent(Join, Fun) ->
    {Agent, Mon} = spawn_monitor(fun() -> agent(Join, undefined) end),
    Result =
        try
            Fun(Agent)
        after
            Agent ! stop
        end,
    Why =
        receive
            {'DOWN', Mon, process, Agent, {done, W}} -> W;
            {'DOWN', Mon, process, Agent, Other} -> {agent_crashed, Other}
        after 1000 ->
            %% Still inside a `Join' that never returned: the native
            %% call has already given up on it.
            exit(Agent, kill),
            receive
                {'DOWN', Mon, process, Agent, _} -> undefined
            after 5000 -> undefined
            end
        end,
    case Result of
        {error, rejected} when Why =/= undefined -> {error, Why};
        _ -> Result
    end.

agent(Join, Why) ->
    receive
        {erllama_tp_bootstrap, Bridge, Round, Bytes} ->
            case
                try
                    Join(Round, Bytes)
                catch
                    Class:Crash -> {error, {join_crashed, Class, Crash}}
                end
            of
                {ok, All} when is_binary(All) ->
                    _ = erllama_nif:tp_bootstrap_reply(Bridge, Round, All),
                    agent(Join, Why);
                {error, Reason} ->
                    _ = erllama_nif:tp_bootstrap_reply(Bridge, Round, error),
                    agent(Join, Reason)
            end;
        stop ->
            exit({done, Why})
    after ?AGENT_IDLE_MS ->
        %% Its native call cannot outlive its own deadline; an agent
        %% left waiting this long has lost it.
        exit({done, Why})
    end.

f32(Values) -> <<<<(float(V)):32/float-native>> || V <- Values>>.

floats(Bin) -> [V || <<V:32/float-native>> <= Bin].
