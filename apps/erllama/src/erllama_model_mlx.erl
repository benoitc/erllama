%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
-module(erllama_model_mlx).
-moduledoc """
Experimental single-sequence, greedy MLX-LM backend for Qwen on Apple Silicon.
Set `ERLLAMA_MLX_PYTHON` to the interpreter in an environment with the
dependencies from `priv/mlx/requirements.txt`. Load a local MLX checkpoint
directory with `backend => erllama_model_mlx` and use raw text completions.
Checkpoints, speculative decoding and chat templates are currently unsupported.
""".
-behaviour(erllama_model_backend).

-export([
    init/1, terminate/1, tokenize/2, detokenize/2, detokenize/3, step/2, prefill/2, decode_one/2
]).
-export([
    sampler_new/2, sampler_free/1, seq_rm/2, seq_clear/1, kv_pack/2, kv_unpack/2, extra_metadata/1
]).

init(Config) ->
    Context = maps:get(context_opts, Config, #{}),
    ModelOpts = maps:get(model_opts, Config, #{}),
    case
        {
            os:type(),
            maps:get(n_seq_max, Context, 1),
            maps:without([n_ctx, n_seq_max, n_batch], Context),
            ModelOpts
        }
    of
        {{unix, darwin}, 1, C, M} when map_size(C) =:= 0, map_size(M) =:= 0 ->
            start(
                maps:get(model_path, Config),
                maps:get(n_ctx, Context, maps:get(context_size, Config, 4096))
            );
        _ ->
            {error, not_supported}
    end.

start(Path, Limit) ->
    case os:find_executable(os:getenv("ERLLAMA_MLX_PYTHON", "python3")) of
        false ->
            {error, mlx_python_not_found};
        Python ->
            Script = filename:join([code:priv_dir(erllama), "mlx", "runner.py"]),
            Args = [Script, filename:absname(Path), integer_to_list(Limit)],
            Port = open_port({spawn_executable, Python}, [
                binary, {packet, 4}, exit_status, {args, Args}
            ]),
            case response(Port, 120000) of
                {ok, <<"ready">>} ->
                    _ = erlang:monitor(port, Port),
                    {ok, Port};
                {error, _} = Error ->
                    terminate(Port),
                    Error
            end
    end.

terminate(Port) ->
    try erlang:port_close(Port) of
        true -> ok
    catch
        error:badarg -> ok
    end.

tokenize(Port, Text) ->
    value(call(Port, encode, Text)).

detokenize(Port, Tokens) ->
    detokenize(Port, Tokens, #{}).

detokenize(Port, Tokens, Opts) ->
    Special =
        maps:get(unparse_special, Opts, false) andalso not maps:get(remove_special, Opts, false),
    case call(Port, pieces, [Tokens, Special]) of
        {ok, Encoded} -> base64:decode(Encoded);
        Error -> Error
    end.

step(Port, [{0, {prefill, Tokens}}]) ->
    case call(Port, prefill, Tokens) of
        {ok, null} -> {ok, [{0, prefilled}]};
        Error -> Error
    end;
step(Port, [{0, {decode, greedy}}]) ->
    case call(Port, decode, null) of
        {ok, [Token, Eog]} -> {ok, [{0, {token, Token, Eog}}]};
        Error -> Error
    end;
step(_Port, _Ops) ->
    {error, not_supported}.

prefill(Port, Tokens) ->
    case step(Port, [{0, {prefill, Tokens}}]) of
        {ok, [{0, prefilled}]} -> ok;
        Error -> Error
    end.

decode_one(Port, _Tokens) ->
    case step(Port, [{0, {decode, greedy}}]) of
        {ok, [{0, {token, Token, 0}}]} -> {ok, Token};
        {ok, [{0, {token, Token, 1}}]} -> {eog, Token};
        Error -> Error
    end.

sampler_new(_Port, Opts) ->
    case {maps:get(temperature, Opts, 0), maps:without([temperature, seed], Opts)} of
        {T, Rest} when is_number(T), T >= 0, T =< 0, map_size(Rest) =:= 0 -> {ok, greedy};
        _ -> {error, not_supported}
    end.

sampler_free(greedy) -> ok.

seq_clear(Port) -> seq_rm(Port, 0).

seq_rm(Port, 0) ->
    case call(Port, clear, null) of
        {ok, null} -> ok;
        Error -> Error
    end.

kv_pack(_, _) -> {error, not_supported}.
kv_unpack(_, _) -> {error, not_supported}.
extra_metadata(_) -> #{kv_format => <<"mlx-no-checkpoints-v1">>}.

call(Port, Op, Arg) ->
    try erlang:port_command(Port, json:encode([1, Op, Arg])) of
        true -> response(Port, 60000)
    catch
        error:badarg -> {error, mlx_runner_down}
    end.

response(Port, Timeout) ->
    receive
        {Port, {data, Frame}} ->
            case json:decode(Frame) of
                [1, <<"ok">>, Result] -> {ok, Result};
                [1, <<"load_error">>, Msg] -> {error, {mlx_load_failed, Msg}};
                [1, <<"error">>, <<"context_full">>] -> {error, context_full};
                [1, <<"error">>, Msg] -> {error, {mlx_error, Msg}}
            end;
        {Port, {exit_status, Status}} ->
            {error, {mlx_runner_exit, Status}}
    after Timeout ->
        terminate(Port),
        {error, decode_timeout}
    end.

value({ok, Value}) -> Value;
value(Error) -> Error.
