%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% What `erllama_wasi_nn` decides before it reaches a model: the error
%% a guest reads for each erllama reason, the shape of a `backend_opts`
%% entry, and the input tensor `compute/2` accepts.
%% @end
-module(erllama_wasi_nn_tests).
-include_lib("eunit/include/eunit.hrl").

%% One row per line of the error table.
error_table_test_() ->
    [
        ?_assertEqual({Code, atom_to_binary(Reason)}, erllama_wasi_nn:error_reply(Reason))
     || {Reason, Code} <- [
            {not_loaded, not_found},
            {not_found, not_found},
            {timeout, timeout},
            {decode_timeout, timeout},
            {too_large, too_large},
            {context_overflow, too_large},
            {oom, too_large},
            {not_supported, unsupported_operation},
            {chat_not_supported, unsupported_operation},
            {busy, runtime_error},
            {engine_reset, runtime_error}
        ]
    ].

%% A reason that is not an atom can carry a path or a guest's bytes:
%% none of it reaches the message.
a_tuple_reason_gets_a_fixed_message_test_() ->
    Fixed = {runtime_error, <<"the request failed">>},
    [
        ?_assertEqual(Fixed, erllama_wasi_nn:error_reply(Reason))
     || Reason <- [
            {invalid_config, model_path, "/srv/secret/model.gguf"},
            {decode_failed, -3},
            {not_loaded, "/srv/secret"},
            <<"/srv/secret">>,
            "/srv/secret"
        ]
    ].

a_message_fits_the_host_bound_test() ->
    Longest = binary_to_atom(binary:copy(<<"é"/utf8>>, 255)),
    {runtime_error, Message} = erllama_wasi_nn:error_reply(Longest),
    ?assert(byte_size(Message) =< 1024).

a_name_without_an_entry_is_not_found_test_() ->
    NotFound = {error, {not_found, <<"no such model">>}},
    [
        ?_assertEqual(NotFound, erllama_wasi_nn:load_by_name(<<"default">>, #{})),
        ?_assertEqual(NotFound, erllama_wasi_nn:load_by_name(<<"default">>, undefined)),
        ?_assertEqual(
            NotFound,
            erllama_wasi_nn:load_by_name(<<"default">>, #{
                <<"other">> => #{model => <<"m">>, task => completion}
            })
        )
    ].

an_entry_of_the_wrong_shape_is_an_invalid_argument_test_() ->
    [
        ?_assertMatch(
            {error, {invalid_argument, _}},
            erllama_wasi_nn:load_by_name(<<"default">>, #{<<"default">> => Entry})
        )
     || Entry <- [
            #{task => completion},
            #{model => <<"m">>},
            #{model => <<"m">>, task => chat},
            #{model => <<"m">>, task => <<"completion">>},
            #{model => m, task => completion},
            #{model => <<"m">>, task => completion, request_opts => [{response_tokens, 4}]},
            not_a_map
        ]
    ].

%% The guest of the suite always sends one u8 tensor, so these go to
%% the callback directly. The graph names no loaded model: the input is
%% refused before any model is asked.
compute_refuses_what_is_not_text_test_() ->
    Graph = #{model => <<"absent">>, task => completion, request_opts => #{}},
    Tensor = fun(Type, Data) -> {<<"input">>, #{dimensions => [1], type => Type, data => Data}} end,
    [
        ?_assertEqual(
            {error, {invalid_argument, <<"no input">>}},
            erllama_wasi_nn:compute(Graph, [])
        ),
        ?_assertEqual(
            {error, {invalid_argument, <<"the first input must be a u8 tensor">>}},
            erllama_wasi_nn:compute(Graph, [Tensor(fp32, <<1.0:32/float-little>>)])
        ),
        ?_assertEqual(
            {error, {invalid_argument, <<"the first input must be a u8 tensor">>}},
            erllama_wasi_nn:compute(Graph, [Tensor(i32, <<"text">>), Tensor(u8, <<"text">>)])
        ),
        ?_assertEqual(
            {error, {invalid_argument, <<"the input is not UTF-8">>}},
            erllama_wasi_nn:compute(Graph, [Tensor(u8, <<16#ff, 16#fe>>)])
        ),
        ?_assertEqual(
            {error, {invalid_argument, <<"the input is not UTF-8">>}},
            erllama_wasi_nn:compute(Graph, [Tensor(u8, <<"caf", 16#c3>>)])
        )
    ].

close_does_nothing_test() ->
    ?assertEqual(ok, erllama_wasi_nn:close({graph, anything})),
    ?assertEqual(ok, erllama_wasi_nn:close({context, anything})).
