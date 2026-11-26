%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
-module(erllama_model_mlx).
-moduledoc """
Experimental greedy MLX-LM backend for Qwen on Apple Silicon.
Set `ERLLAMA_MLX_PYTHON` to the interpreter in an environment with the
dependencies from `priv/mlx/requirements.txt`. Load a local MLX checkpoint
directory with `backend => erllama_model_mlx`. The worker batches active
decode slots and saves attention and recurrent state.
""".
-behaviour(erllama_model_backend).

-export([
    init/1, terminate/1, tokenize/2, detokenize/2, detokenize/3, step/2, prefill/2, decode_one/2
]).
-export([
    sampler_new/2, sampler_free/1, seq_rm/2, seq_clear/1, kv_pack/2, kv_unpack/2, extra_metadata/1
]).

-export([kv_pack/3, kv_unpack/3, seq_rm_last/3, chat_templates/1]).

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
        {{unix, darwin}, N, C, M} when is_integer(N), N > 0, map_size(C) =:= 0, map_size(M) =:= 0 ->
            start(
                maps:get(model_path, Config),
                maps:get(n_ctx, Context, maps:get(context_size, Config, 4096)),
                N,
                maps:get(chat_template, Config, undefined)
            );
        _ ->
            {error, not_supported}
    end.

start(Path, Limit, Slots, Override) ->
    case os:find_executable(os:getenv("ERLLAMA_MLX_PYTHON", "python3")) of
        false ->
            {error, mlx_python_not_found};
        Python ->
            Script = filename:join([code:priv_dir(erllama), "mlx", "runner.py"]),
            Args = [Script, filename:absname(Path), integer_to_list(Limit), integer_to_list(Slots)],
            Port = open_port({spawn_executable, Python}, [
                binary, {packet, 4}, exit_status, {args, Args}
            ]),
            case initialize(Port, Override) of
                {ok, _} = Ready ->
                    Ready;
                Error ->
                    terminate(Port),
                    Error
            end
    end.

initialize(Port, Override) ->
    case response(Port, 120000) of
        {ok, [File, Digest, Format, Template, Bos, Eos]} ->
            Source =
                case Override of
                    undefined -> Template;
                    _ -> Override
                end,
            case templates(Source, Bos, Eos) of
                {ok, Chat} ->
                    _ = erlang:monitor(port, Port),
                    {ok, #{
                        port => Port,
                        file => File,
                        templates => Chat,
                        fingerprint => binary:decode_hex(Digest),
                        kv_format => Format
                    }};
                Error ->
                    Error
            end;
        Error ->
            Error
    end.

templates(null, _, _) -> {ok, undefined};
templates(<<>>, _, _) -> {ok, undefined};
templates(Source, Bos, Eos) -> erllama_nif:chat_templates_init({Bos, Eos}, Source).

chat_templates(#{templates := undefined}) -> {error, chat_not_supported};
chat_templates(#{templates := T}) -> {ok, T}.

terminate(#{port := Port, file := File}) ->
    terminate(Port),
    _ = file:delete(File),
    _ = file:del_dir(filename:dirname(File)),
    ok;
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

step(State, Ops) ->
    case call(State, step, lists:map(fun command/1, Ops)) of
        {ok, Rows} -> {ok, lists:map(fun result/1, Rows)};
        Error -> Error
    end.

command({Seq, {prefill, Tokens}}) -> [Seq, prefill, Tokens];
command({Seq, {decode, greedy}}) -> [Seq, decode, null].

result([Seq, null]) -> {Seq, prefilled};
result([Seq, [Token, Eog]]) -> {Seq, {token, Token, Eog}}.

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

seq_rm(State, Seq) -> done(call(State, clear, Seq)).
seq_rm_last(State, Seq, Length) -> done(call(State, trim_last, [Seq, Length])).

kv_pack(State, Tokens) -> kv_pack(State, Tokens, 0).
kv_pack(#{file := File} = State, Tokens, Seq) ->
    try
        case call(State, pack, [Seq, Tokens]) of
            {ok, null} -> value(file:read_file(File));
            Error -> Error
        end
    after
        file:delete(File)
    end.

kv_unpack(State, Bin) -> kv_unpack(State, Bin, 0).
kv_unpack(#{file := File} = State, Bin, Seq) ->
    try
        case file:write_file(File, Bin) of
            ok -> done(call(State, unpack, Seq));
            Error -> Error
        end
    after
        file:delete(File)
    end.

extra_metadata(State) -> maps:with([fingerprint, kv_format], State).

done({ok, null}) -> ok;
done(Error) -> Error.

call(#{port := Port}, Op, Arg) ->
    try erlang:port_command(Port, json:encode([2, Op, Arg])) of
        true -> response(Port, 60000)
    catch
        error:badarg -> {error, mlx_runner_down}
    end.

response(Port, Timeout) ->
    receive
        {Port, {data, Frame}} ->
            case json:decode(Frame) of
                [2, <<"ok">>, Result] -> {ok, Result};
                [2, <<"load_error">>, Msg] -> {error, {mlx_load_failed, Msg}};
                [2, <<"error">>, <<"context_full">>] -> {error, context_full};
                [2, <<"error">>, Msg] -> {error, {mlx_error, Msg}}
            end;
        {'DOWN', _, port, Port, _} = Down ->
            self() ! Down,
            {error, mlx_runner_down};
        {Port, {exit_status, Status}} ->
            {error, {mlx_runner_exit, Status}}
    after Timeout ->
        terminate(Port),
        {error, decode_timeout}
    end.

value({ok, Value}) -> Value;
value(Error) -> Error.
