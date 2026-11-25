%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%%
%% A summarizer that lives in your OTP application: one supervised
%% process loads a model with erllama and serves summary jobs to the
%% other processes of the application. No HTTP, no separate service.
%%
%% - The caller gets the summary as messages while it is generated:
%%   `{summarizer, Job, {text, Bin}}', then `{summarizer, Job, {done,
%%   Summary}}' or `{summarizer, Job, {error, Reason}}'.
%% - Jobs run at once, up to the model's `parallel' sequences. One
%%   more is refused with `{error, seq_capacity}' (`on_full => error')
%%   rather than queued, so this process never waits on admission and
%%   the caller decides whether to retry.
%% - A caller that exits (a closed connection, a crashed handler) gets
%%   its job cancelled: this process monitors every caller.
%% - If this process crashes, the supervisor restarts it, and erllama
%%   cancels the generations that were streaming to it.
-module(summarizer).
-behaviour(gen_server).

-export([start_link/0, summarize/1, summarize_sync/2, cancel/1, info/0]).
-export([init/1, handle_continue/2, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-define(MODEL, <<"summarizer">>).

-record(job, {caller :: pid(), mon :: reference(), text = [] :: [binary()]}).
-record(state, {jobs = #{} :: #{reference() => #job{}}}).

start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

%% Start a summary of `Text'; the reply streams to the caller. Returns
%% the job reference the messages carry.
-spec summarize(binary()) -> {ok, reference()} | {error, term()}.
summarize(Text) when is_binary(Text) ->
    gen_server:call(?MODULE, {summarize, Text, self()}, 30000).

%% The same, waiting for the whole summary.
-spec summarize_sync(binary(), timeout()) -> {ok, binary()} | {error, term()}.
summarize_sync(Text, Timeout) ->
    case summarize(Text) of
        {ok, Job} -> collect(Job, Timeout);
        {error, _} = E -> E
    end.

-spec cancel(reference()) -> ok.
cancel(Job) ->
    erllama:cancel(Job).

%% What the model and its cache are doing, from erllama's own APIs.
-spec info() -> map().
info() ->
    gen_server:call(?MODULE, info).

collect(Job, Timeout) ->
    receive
        {summarizer, Job, {text, _}} -> collect(Job, Timeout);
        {summarizer, Job, {done, Summary}} -> {ok, Summary};
        {summarizer, Job, {error, Reason}} -> {error, Reason}
    after Timeout ->
        ok = cancel(Job),
        {error, timeout}
    end.

%% -----------------------------------------------------------------------------

init([]) ->
    {ok, #state{}, {continue, load}}.

%% Load once, after init: a restart finds the model still loaded (it
%% belongs to erllama's supervisor, not to this process).
handle_continue(load, St) ->
    case erllama:whereis(?MODEL) of
        {ok, _} ->
            {noreply, St};
        {error, not_loaded} ->
            {ok, _} = erllama:load_model(?MODEL, model_config()),
            {noreply, St}
    end.

handle_call({summarize, Text, Caller}, _From, #state{jobs = Jobs} = St) ->
    Messages = [
        #{role => <<"system">>, content => <<"Summarize the user's text in one or two sentences.">>},
        #{role => <<"user">>, content => Text}
    ],
    case erllama:render_chat_template(?MODEL, #{messages => Messages}) of
        {ok, Prompt} ->
            %% Tokens come to this process as messages; it forwards them.
            Opts = #{response_tokens => 120, temperature => 0.0, on_full => error},
            case erllama:stream(?MODEL, Prompt, Opts) of
                {ok, Ref} ->
                    Job = #job{caller = Caller, mon = erlang:monitor(process, Caller)},
                    {reply, {ok, Ref}, St#state{jobs = Jobs#{Ref => Job}}};
                {error, _} = E ->
                    {reply, E, St}
            end;
        {error, _} = E ->
            {reply, E, St}
    end;
handle_call(info, _From, #state{jobs = Jobs} = St) ->
    {ok, Model} = erllama:model_info(?MODEL),
    Info = #{
        jobs => map_size(Jobs),
        sequences => maps:with([n_seq_max, available_seqs], Model),
        waiting => element(2, erllama:pending_len(?MODEL)),
        last_cache_hit => element(2, erllama:last_cache_hit(?MODEL)),
        cache => maps:with([hits_exact, hits_longest_prefix, misses, saves_finish], erllama:counters())
    },
    {reply, Info, St}.

handle_cast(_Msg, St) ->
    {noreply, St}.

handle_info({erllama, Ref, {token, Bin}}, #state{jobs = Jobs} = St) when is_map_key(Ref, Jobs) ->
    #{Ref := #job{caller = C, text = T} = J} = Jobs,
    C ! {summarizer, Ref, {text, Bin}},
    {noreply, St#state{jobs = Jobs#{Ref := J#job{text = [Bin | T]}}}};
handle_info({erllama, Ref, {done, _Stats}}, St) ->
    {noreply, finish(Ref, fun(#job{text = T}) -> {done, iolist_to_binary(lists:reverse(T))} end, St)};
handle_info({erllama, Ref, {error, Reason}}, St) ->
    {noreply, finish(Ref, fun(_) -> {error, Reason} end, St)};
handle_info({'DOWN', Mon, process, _Pid, _Reason}, #state{jobs = Jobs} = St) ->
    %% The caller is gone: nobody wants the rest of its summary.
    case [Ref || Ref := #job{mon = M} <- Jobs, M =:= Mon] of
        [Ref] ->
            ok = erllama:cancel(Ref),
            {noreply, St#state{jobs = maps:remove(Ref, Jobs)}};
        [] ->
            {noreply, St}
    end;
handle_info(_Msg, St) ->
    {noreply, St}.

terminate(_Reason, _St) ->
    ok.

finish(Ref, Result, #state{jobs = Jobs} = St) ->
    case maps:take(Ref, Jobs) of
        {#job{caller = C, mon = M} = J, Rest} ->
            erlang:demonitor(M, [flush]),
            C ! {summarizer, Ref, Result(J)},
            St#state{jobs = Rest};
        error ->
            St
    end.

model_config() ->
    {ok, Env} = {ok, application:get_all_env(summarizer)},
    Path =
        case os:getenv("SUMMARIZER_MODEL") of
            false -> proplists:get_value(model_path, Env);
            P -> P
        end,
    NCtx = proplists:get_value(n_ctx, Env, 4096),
    #{
        backend => erllama_model_llama,
        model_path => Path,
        model_opts => #{n_gpu_layers => 99},
        %% `parallel' summaries at once, sharing one pool of n_ctx.
        context_opts => #{
            n_ctx => NCtx,
            n_seq_max => proplists:get_value(parallel, Env, 4),
            kv_unified => true
        },
        context_size => NCtx
    }.
