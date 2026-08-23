%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_middleware_tests).
-include_lib("eunit/include/eunit.hrl").

-define(M, erllama_middleware).

with_app(Body) ->
    {ok, Started} = application:ensure_all_started(erllama),
    try
        Body()
    after
        application:unset_env(erllama, middleware),
        [application:stop(A) || A <- lists:reverse(Started)],
        ok
    end.

%% A middleware that records `{Tag, in}' before and `{Tag, out}' after.
tracer(Tag, Log) ->
    fun(Req, Next) ->
        Log ! {Tag, in},
        Resp = Next(Req),
        Log ! {Tag, out},
        Resp
    end.

collect(N) ->
    collect(N, []).

collect(0, Acc) ->
    lists:reverse(Acc);
collect(N, Acc) ->
    receive
        Msg -> collect(N - 1, [Msg | Acc])
    after 1000 -> lists:reverse(Acc)
    end.

chain_order_test() ->
    Self = self(),
    Chain = [tracer(a, Self), tracer(b, Self)],
    Req = #{op => noop, model => undefined, args => #{}},
    ?assertEqual(done, ?M:run(Req, Chain, fun(_) -> done end)),
    ?assertEqual([{a, in}, {b, in}, {b, out}, {a, out}], collect(4)).

empty_chain_calls_next_test() ->
    Req = #{op => noop, model => undefined, args => #{x => 1}},
    ?assertEqual({seen, Req}, ?M:run(Req, [], fun(R) -> {seen, R} end)).

short_circuit_test() ->
    Cache = fun(_Req, _Next) -> {ok, cached} end,
    Req = #{op => noop, model => undefined, args => #{}},
    ?assertEqual({ok, cached}, ?M:run(Req, [Cache], fun(_) -> erlang:error(should_not_run) end)).

rewrite_request_test() ->
    AddKey = fun(Req = #{args := Args}, Next) -> Next(Req#{args := Args#{extra => true}}) end,
    Req = #{op => noop, model => undefined, args => #{}},
    ?assertEqual(#{extra => true}, ?M:run(Req, [AddKey], fun(#{args := A}) -> A end)).

take_prefers_per_call_chain_test() ->
    application:set_env(erllama, middleware, [fun(R, N) -> N(R) end]),
    try
        PerCall = [fun(R, N) -> N(R) end, fun(R, N) -> N(R) end],
        {Chain, Rest} = ?M:take(#{middleware => PerCall, response_tokens => 3}),
        ?assertEqual(PerCall, Chain),
        ?assertEqual(#{response_tokens => 3}, Rest),
        {Global, Rest2} = ?M:take(#{response_tokens => 3}),
        ?assertEqual(1, length(Global)),
        ?assertEqual(#{response_tokens => 3}, Rest2)
    after
        application:unset_env(erllama, middleware)
    end.

facade_ops_run_through_global_chain_test() ->
    with_app(fun() ->
        Self = self(),
        Spy = fun(Req = #{op := Op}, Next) ->
            Self ! {op, Op},
            Next(Req)
        end,
        application:set_env(erllama, middleware, [Spy]),
        {ok, Id} = erllama:load_model(#{backend => erllama_model_stub}),
        {ok, Toks} = erllama:tokenize(Id, <<"a b">>),
        {ok, _} = erllama:detokenize(Id, Toks),
        {ok, _} = erllama:complete(Id, <<"a b c">>, #{response_tokens => 1}),
        {ok, Ref} = erllama:stream(Id, Toks, #{response_tokens => 1}),
        {ok, _} = erllama:collect(Ref, 5000),
        ok = erllama:unload(Id),
        Ops = [Op || {op, Op} <- collect(6)],
        ?assertEqual([load_model, tokenize, detokenize, complete, stream, unload], Ops)
    end).

per_call_chain_replaces_global_test() ->
    with_app(fun() ->
        Self = self(),
        application:set_env(erllama, middleware, [
            fun(R, N) ->
                Self ! global,
                N(R)
            end
        ]),
        {ok, Id} = erllama:load_model(#{backend => erllama_model_stub}),
        receive
            global -> ok
        after 1000 -> erlang:error(no_global)
        end,
        Local = fun(R, N) ->
            Self ! local,
            N(R)
        end,
        {ok, _} = erllama:complete(Id, <<"a b c">>, #{response_tokens => 1, middleware => [Local]}),
        ?assertEqual([local], collect(1)),
        ok = erllama:unload(Id)
    end).

middleware_sees_args_and_result_test() ->
    with_app(fun() ->
        Self = self(),
        Spy = fun(#{op := complete, args := #{prompt := P, opts := O}} = Req, Next) ->
            Resp = Next(Req),
            Self ! {complete, P, O, Resp},
            Resp
        end,
        {ok, Id} = erllama:load_model(#{backend => erllama_model_stub}),
        {ok, Result} = erllama:complete(Id, <<"hello">>, #{
            response_tokens => 1, middleware => [Spy]
        }),
        receive
            {complete, <<"hello">>, #{response_tokens := 1} = O, {ok, Result}} ->
                ?assertNot(maps:is_key(middleware, O))
        after 1000 -> erlang:error(no_spy)
        end,
        ok = erllama:unload(Id)
    end).

validation_runs_before_middleware_test() ->
    with_app(fun() ->
        {ok, Id} = erllama:load_model(#{backend => erllama_model_stub}),
        Boom = fun(_, _) -> erlang:error(should_not_run) end,
        ?assertEqual(
            {error, {unknown_option, nope}},
            erllama:complete(Id, <<"x">>, #{nope => 1, middleware => [Boom]})
        ),
        ok = erllama:unload(Id)
    end).
