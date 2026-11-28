%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
-module(erllama_server_cpu_rank).
-behaviour(gen_server).
-export([start_link/1, load/3, init/1, handle_call/3, handle_cast/2, terminate/2]).

%% Release smoke test on machines without NCCL. Each rank runs the whole
%% model on its CPU. The leader checks every follower's result before
%% returning a token. No native resource leaves its owning VM.
start_link(C) -> gen_server:start_link(?MODULE, C, []).
load(P, Path, Opts) -> gen_server:call(P, {load, Path, Opts}, infinity).

init(#{rank := Rank, world := World, join := {call, Coord}, timeout := Ms}) ->
    Join = erllama_tp_coordinator:join_fun(Coord, Rank),
    {ok, _} = erllama_tp:bootstrap_selftest(World, <<Rank>>, 1, Join, Ms),
    {ok, #{followers => [], samplers => #{}, next => 0}}.

handle_call({load, Path, {M, C}}, _, S) ->
    #{devices := none} = M,
    {ok, B} = erllama_model_llama:init(#{model_path => Path, model_opts => M, context_opts => C}),
    {reply, ok, S#{backend => B}};
handle_call({hosting, {host, _, Followers}}, _, S) ->
    {reply, ok, S#{followers := [P || {P, _} <- Followers]}};
handle_call({hosting, {backend, sampler_new, [Opts]}}, _, #{next := N} = S) ->
    {ok, S1} = replicated({new, N, Opts}, S),
    {reply, {ok, {erllama_tp_ref, self(), N}}, S1#{next := N + 1}};
handle_call({hosting, {free_ref, N}}, _, S) ->
    {ok, S1} = replicated({free, N}, S),
    {reply, ok, S1};
handle_call({hosting, {backend, Fn, Args}}, _, S) ->
    {Result, S1} = replicated({Fn, Args}, S),
    {reply, Result, S1};
handle_call({replica, Op}, _, S) ->
    {Result, S1} = execute(Op, S),
    {reply, Result, S1};
handle_call(_, _, S) ->
    {reply, {error, not_supported}, S}.

replicated(Op, #{followers := Followers} = S) ->
    {Result, _} = Local = execute(Op, S),
    [Result = gen_server:call(P, {replica, Op}, 10000) || P <- Followers],
    Local.

execute({new, N, Opts}, #{backend := B, samplers := Refs} = S) ->
    {ok, Ref} = erllama_model_llama:sampler_new(B, Opts),
    {ok, S#{samplers := Refs#{N => Ref}}};
execute({free, N}, #{samplers := Refs} = S) ->
    {Ref, Rest} = maps:take(N, Refs),
    ok = erllama_model_llama:sampler_free(Ref),
    {ok, S#{samplers := Rest}};
execute({step, [Ops]}, #{backend := B, samplers := Refs} = S) ->
    Local = lists:map(
        fun
            ({Seq, {decode, {erllama_tp_ref, _, N}}}) -> {Seq, {decode, maps:get(N, Refs)}};
            (Op) -> Op
        end,
        Ops
    ),
    {erllama_model_llama:step(B, Local), S};
execute({extra_metadata, []}, #{backend := B} = S) ->
    {erllama_model_llama:extra_metadata(B), S};
execute({seq_rm, [Seq]}, #{backend := B} = S) ->
    {erllama_model_llama:seq_rm(B, Seq), S};
execute(_, S) ->
    {{error, not_supported}, S}.

handle_cast(_, S) -> {noreply, S}.
terminate(_, #{backend := B}) -> erllama_model_llama:terminate(B);
terminate(_, _) -> ok.
