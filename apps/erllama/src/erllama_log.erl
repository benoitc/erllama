%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_log).
-moduledoc false.
%% Receiver for llama.cpp / ggml native log lines, forwarding them
%% into Erlang's `logger` under the domain `[erllama, native]`.
%%
%% The NIF's log callback (installed process-wide on the first model
%% load) sends `{llama_log, LevelInt, TextBin}` messages to this
%% process for every native line at or above the configured level;
%% CONT fragments (the stderr progress dots) are dropped C-side.
%%
%% Application environment:
%%   {native_log_level, none | error | warning | info | debug}
%% Default `warning`. `none` disables forwarding entirely (this
%% server still starts but registers no receiver). `info` surfaces
%% the full model-load banner.
-behaviour(gen_server).

-export([start_link/0]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

%% ggml_log_level numeric values (ggml.h).
-define(GGML_DEBUG, 1).
-define(GGML_INFO, 2).
-define(GGML_WARN, 3).
-define(GGML_ERROR, 4).

start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

init([]) ->
    case level_int(application:get_env(erllama, native_log_level, warning)) of
        none ->
            ok;
        MinLevel ->
            %% The NIF is part of this application and always
            %% loadable; the try is a belt for exotic embeddings
            %% where the shared object failed to load - native logs
            %% are then simply not forwarded.
            try
                ok = erllama_nif:set_log_receiver(self(), MinLevel)
            catch
                _:_ -> ok
            end
    end,
    {ok, #{}}.

handle_call(_Req, _From, State) ->
    {reply, ok, State}.

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info({llama_log, Level, Text}, State) when is_binary(Text) ->
    Msg = string:trim(Text, trailing, "\n"),
    case Msg of
        <<>> ->
            ok;
        _ ->
            logger:log(logger_level(Level), "~ts", [Msg], #{
                domain => [erllama, native]
            })
    end,
    {noreply, State};
handle_info(_Other, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    try
        erllama_nif:clear_log_receiver()
    catch
        _:_ -> ok
    end,
    ok.

level_int(none) -> none;
level_int(error) -> ?GGML_ERROR;
level_int(warning) -> ?GGML_WARN;
level_int(info) -> ?GGML_INFO;
level_int(debug) -> ?GGML_DEBUG;
%% Unknown values keep the default rather than crashing the tree.
level_int(_Other) -> ?GGML_WARN.

logger_level(?GGML_DEBUG) -> debug;
logger_level(?GGML_INFO) -> info;
logger_level(?GGML_WARN) -> warning;
logger_level(?GGML_ERROR) -> error;
logger_level(_Other) -> info.
