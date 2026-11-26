%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_pull).
-moduledoc false.
%% One model download: a job like a run, followed and cancelled the same
%% way. It asks the source what to fetch (erllama_server_source), then
%% fetches each file into `<data_dir>/blobs' and registers the model.
%%
%% A file is written to `<blob>.part' with a running sha256. A transfer
%% that breaks off resumes from the bytes already on disk with a
%% `Range' request (the part file is re-hashed first); a server that
%% answers `200' to a range restarts it from zero. A file whose size or
%% sha256 does not match is deleted and the pull fails with
%% `integrity_mismatch'. A blob already on disk (another model shares
%% it) is reused only once its sha256 checks out; otherwise it is
%% deleted and fetched again. Cancelling keeps the part file, so the next
%% pull of the same model resumes it. Redirects are followed by hand,
%% and the Authorization header is only sent to the origin it was
%% meant for.
%%
%% Events: `started', `progress' (#{file, done, total}), then `done'
%% (the registered entry), `cancelled' or `error'.

-behaviour(gen_server).

-export([start_link/1]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, handle_continue/2]).

-define(MAX_REDIRECTS, 5).
-define(MAX_RETRIES, 5).
-define(PROGRESS_EVERY_MS, 250).

-record(dl, {
    file :: erllama_server_source:file(),
    url :: binary(),
    part :: file:filename_all(),
    fd :: file:fd() | undefined,
    hash :: term(),
    offset = 0 :: non_neg_integer(),
    total :: non_neg_integer() | undefined,
    ref :: term() | undefined,
    redirects = 0 :: non_neg_integer(),
    retries = 0 :: non_neg_integer(),
    last_progress = 0 :: integer()
}).

-record(state, {
    id :: binary(),
    source :: erllama_server_source:source(),
    opts :: map(),
    client :: term(),
    events :: erllama_server_events:events(),
    status = running :: erllama_server_runs:status(),
    name :: binary() | undefined,
    defaults = #{} :: map(),
    todo = [] :: [erllama_server_source:file()],
    fetched = [] :: [{model | projector, binary(), binary(), non_neg_integer()}],
    dl :: #dl{} | undefined,
    part_lock :: term(),
    %% A blob already on disk being checked before it is reused.
    verify :: {reference(), erllama_server_source:file(), binary()} | undefined,
    result :: map() | undefined
}).
-type state() :: #state{}.

-spec start_link(map()) -> {ok, pid()} | {error, term()}.
start_link(Spec) ->
    gen_server:start_link(?MODULE, Spec, []).

-spec init(map()) -> {ok, state(), {continue, plan}}.
init(#{run_id := Id, source := Source} = Spec) ->
    St = #state{
        id = Id,
        source = Source,
        opts = maps:get(opts, Spec, #{}),
        client = livery_client:new(#{}),
        events = erllama_server_events:new(Id)
    },
    {ok, St, {continue, plan}}.

handle_continue(plan, #state{source = Source, client = C, opts = Opts} = St) ->
    St1 = emit(started, #{job_id => St#state.id, source => maps:get(ref, Source)}, St),
    case erllama_server_source:plan(Source, C, Opts) of
        {ok, #{name := Name, files := Files, defaults := D}} ->
            {noreply,
                next_file(St1#state{name = maps:get(name, Opts, Name), defaults = D, todo = Files})};
        {error, Reason} ->
            {noreply, fail(Reason, St1)}
    end.

handle_call({attach, Sub, Cursor}, _From, #state{events = Ev, status = Status} = St) ->
    case erllama_server_events:attach(Sub, Cursor, Status =:= running, Ev) of
        {{ok, Replay}, Ev1} ->
            {reply, {ok, #{run_pid => self(), replay => Replay, status => Status}}, St#state{
                events = Ev1
            }};
        {{error, _} = E, Ev1} ->
            {reply, E, St#state{events = Ev1}}
    end;
handle_call({detach, Sub}, _From, #state{events = Ev} = St) ->
    {reply, ok, St#state{events = erllama_server_events:detach(Sub, Ev)}};
handle_call(cancel, _From, #state{status = running} = St) ->
    {reply, ok, terminal(cancelled, #{}, stop_download(St))};
handle_call(cancel, _From, St) ->
    {reply, ok, St};
handle_call(result, _From, #state{status = running} = St) ->
    {reply, {error, running}, St};
handle_call(result, _From, #state{result = R} = St) ->
    {reply, {ok, R}, St};
handle_call(info, _From, #state{id = Id, status = Status, source = S, name = N, events = Ev} = St) ->
    {reply,
        {ok, #{
            run_id => Id,
            kind => pull,
            source => maps:get(ref, S),
            model => N,
            status => Status,
            next_seq => erllama_server_events:next_seq(Ev)
        }},
        St};
handle_call(_Req, _From, St) ->
    {reply, {error, unknown_call}, St}.

handle_cast(_Msg, St) ->
    {noreply, St}.

handle_info({livery_response, Ref, Msg}, #state{dl = #dl{ref = Ref}, status = running} = St) ->
    {noreply, transfer(Msg, St)};
handle_info({verified, Ref, Result}, #state{verify = {Ref, _, _}, status = running} = St) ->
    {noreply, verified(Result, St)};
handle_info(next_file, #state{status = running} = St) ->
    {noreply, next_file(St)};
handle_info({retry, Ref}, #state{dl = #dl{ref = {retry, Ref}} = Dl, status = running} = St) ->
    {noreply, request(Dl, St)};
handle_info({ack, Sub, Seq}, #state{events = Ev} = St) when is_integer(Seq) ->
    {noreply, St#state{events = erllama_server_events:ack(Sub, Seq, Ev)}};
handle_info({'DOWN', Mon, process, Pid, _}, #state{events = Ev} = St) ->
    case erllama_server_events:down(Mon, Pid, Ev) of
        {ok, Ev1} -> {noreply, St#state{events = Ev1}};
        ignore -> {noreply, St}
    end;
handle_info(expire, St) ->
    {stop, normal, St};
handle_info(_Msg, St) ->
    {noreply, St}.

%% -----------------------------------------------------------------------------
%% Files
%% -----------------------------------------------------------------------------

next_file(#state{todo = []} = St) ->
    register_model(St);
next_file(#state{todo = [File | _]} = St) ->
    %% Retries resume a shared .part file. Give it one writer at a time,
    %% including while a finished blob is being verified.
    Lock = {{?MODULE, part_path(File, St)}, self()},
    case global:set_lock(Lock, [node()], 0) of
        true ->
            fetch_file(St#state{part_lock = Lock});
        false ->
            erlang:send_after(100, self(), next_file),
            St
    end.

unlock(#state{part_lock = undefined} = St) ->
    St;
unlock(#state{part_lock = Lock} = St) ->
    global:del_lock(Lock, [node()]),
    St#state{part_lock = undefined}.

fetch_file(#state{todo = [#{sha256 := Sha, size := Size} = File | Rest]} = St) ->
    Final = blob_path(Sha, St),
    case Sha =/= undefined andalso Size =/= undefined andalso filelib:file_size(Final) =:= Size of
        true ->
            %% Already on disk: check it in a helper (it can be large),
            %% so this job keeps answering attach and cancel.
            Self = self(),
            Ref = make_ref(),
            _ = spawn(fun() -> Self ! {verified, Ref, blob_sha256(Final)} end),
            St#state{todo = Rest, verify = {Ref, File, Final}};
        false ->
            start_file(File, St#state{todo = Rest})
    end.

verified(Result, #state{verify = {_, #{sha256 := Sha, size := Size} = File, Final}} = St0) ->
    St = St0#state{verify = undefined},
    case Result of
        {ok, Sha} ->
            next_file(
                unlock(St#state{fetched = [{role(File), Final, Sha, Size} | St#state.fetched]})
            );
        _ ->
            _ = file:delete(Final),
            start_file(File, St)
    end.

blob_sha256(Path) ->
    case resume_point(Path) of
        {ok, _, H} -> {ok, binary:encode_hex(crypto:hash_final(H), lowercase)};
        {error, _} = E -> E
    end.

start_file(#{url := Url} = File, St) ->
    Part = part_path(File, St),
    ok = filelib:ensure_dir(Part),
    case resume_point(Part) of
        {ok, Offset, Hash} ->
            {ok, Fd} = file:open(Part, [append, binary, raw]),
            Dl = #dl{file = File, url = Url, part = Part, fd = Fd, hash = Hash, offset = Offset},
            request(Dl, St);
        {error, Reason} ->
            fail({part_file, Part, Reason}, St)
    end.

%% Bytes already on disk, hashed again so the final digest covers them.
resume_point(Part) ->
    case file:open(Part, [read, binary, raw]) of
        {ok, Fd} ->
            try
                hash_file(Fd, crypto:hash_init(sha256), 0)
            after
                file:close(Fd)
            end;
        {error, enoent} ->
            {ok, 0, crypto:hash_init(sha256)};
        {error, _} = E ->
            E
    end.

hash_file(Fd, H, N) ->
    case file:read(Fd, 1 bsl 20) of
        {ok, Bin} -> hash_file(Fd, crypto:hash_update(H, Bin), N + byte_size(Bin));
        eof -> {ok, N, H};
        {error, _} = E -> E
    end.

request(#dl{url = Url, offset = Offset, file = File} = Dl, #state{client = C} = St) ->
    Range =
        case Offset of
            0 -> [];
            _ -> [{<<"range">>, <<"bytes=", (integer_to_binary(Offset))/binary, "-">>}]
        end,
    Headers = Range ++ auth_headers(File, Url),
    %% The timeout bounds each wait for data: a transfer that stalls is
    %% retried from where it stopped.
    Timeout = erllama_server_config:get(pull_idle_timeout_ms),
    case
        livery_client:request(C, get, Url, #{
            headers => Headers, stream_to => self(), flow => manual, timeout => Timeout
        })
    of
        {ok, #{body := {push, Ref}}} ->
            %% Read from the map: livery_client:body/1's spec leaves out
            %% the push body.
            St#state{dl = Dl#dl{ref = Ref}};
        {error, Reason} ->
            retry(Reason, St#state{dl = Dl})
    end.

%% The Hugging Face token goes to the Hugging Face origin only, never to
%% the CDN it redirects to.
auth_headers(#{auth := true}, Url) ->
    Hf = erllama_server_config:get(hf_base_url),
    case erllama_server_source:origin(Url) =:= erllama_server_source:origin(Hf) of
        true -> erllama_server_source:hf_headers();
        false -> []
    end;
auth_headers(_, _) ->
    [].

transfer({status, Status, Headers}, #state{dl = Dl} = St) when Status >= 300, Status < 400 ->
    case header(<<"location">>, Headers) of
        undefined -> fail({http_status, Status, Dl#dl.url}, stop_download(St));
        Loc -> redirect(Loc, St)
    end;
transfer({status, 206, Headers}, #state{dl = #dl{offset = Off} = Dl} = St) ->
    Total = add(Off, content_length(Headers)),
    next(St#state{dl = Dl#dl{total = Total}});
transfer({status, 200, Headers}, #state{dl = #dl{offset = 0} = Dl} = St) ->
    next(St#state{dl = Dl#dl{total = content_length(Headers)}});
transfer({status, 200, Headers}, #state{dl = Dl} = St) ->
    %% The server ignored the range: start the file over.
    next(St#state{dl = restart(Dl#dl{total = content_length(Headers)})});
transfer({status, 416, _}, #state{dl = #dl{offset = Off, file = #{size := Off}} = Dl} = St) ->
    %% Everything is already there.
    _ = livery_client:stop_stream(Dl#dl.ref),
    finish_file(St);
transfer({status, 416, _}, #state{dl = Dl} = St) ->
    _ = livery_client:stop_stream(Dl#dl.ref),
    request(restart(Dl), St);
transfer({status, Status, _}, #state{dl = Dl} = St) ->
    fail({http_status, Status, Dl#dl.url}, stop_download(St));
transfer({chunk, Bin}, #state{dl = #dl{fd = Fd, hash = H, offset = Off} = Dl} = St) ->
    case file:write(Fd, Bin) of
        ok ->
            Dl1 = Dl#dl{
                hash = crypto:hash_update(H, Bin), offset = Off + byte_size(Bin), retries = 0
            },
            next(progress(St#state{dl = Dl1}));
        {error, Reason} ->
            fail({write, Reason}, stop_download(St))
    end;
transfer(done, St) ->
    finish_file(St);
transfer({error, {redirect, Loc}}, St) ->
    redirect(Loc, St);
transfer({error, Reason}, St) ->
    retry(Reason, St).

next(#state{dl = #dl{ref = Ref}} = St) ->
    _ = livery_client:stream_next(Ref),
    St.

redirect(_Loc, #state{dl = #dl{redirects = N}} = St) when N >= ?MAX_REDIRECTS ->
    fail(too_many_redirects, stop_download(St));
redirect(Loc, #state{dl = #dl{url = Url, redirects = N} = Dl} = St) ->
    _ = livery_client:stop_stream(Dl#dl.ref),
    request(Dl#dl{url = erllama_server_source:resolve(Url, Loc), redirects = N + 1}, St).

%% A broken transfer resumes from what is on disk, with a growing
%% pause between attempts.
retry(Reason, #state{dl = #dl{retries = N}} = St) when N >= ?MAX_RETRIES ->
    fail({transfer_failed, Reason}, stop_download(St));
retry(_Reason, #state{dl = #dl{retries = N} = Dl} = St) ->
    Ref = make_ref(),
    _ = erlang:send_after(min(10000, 500 bsl N), self(), {retry, Ref}),
    St#state{
        dl = Dl#dl{
            ref = {retry, Ref}, retries = N + 1, redirects = 0, url = maps:get(url, Dl#dl.file)
        }
    }.

restart(#dl{fd = Fd} = Dl) ->
    {ok, 0} = file:position(Fd, 0),
    ok = file:truncate(Fd),
    Dl#dl{offset = 0, hash = crypto:hash_init(sha256)}.

finish_file(#state{dl = #dl{fd = Fd, hash = H, offset = Size, part = Part, file = File}} = St) ->
    ok = file:close(Fd),
    Sha = binary:encode_hex(crypto:hash_final(H), lowercase),
    Expected = maps:get(sha256, File),
    ExpectedSize = maps:get(size, File),
    case
        (Expected =:= undefined orelse Expected =:= Sha) andalso
            (ExpectedSize =:= undefined orelse ExpectedSize =:= Size)
    of
        true ->
            Final = blob_path(Sha, St),
            ok = file:rename(Part, Final),
            next_file(
                unlock(St#state{
                    dl = undefined, fetched = [{role(File), Final, Sha, Size} | St#state.fetched]
                })
            );
        false ->
            _ = file:delete(Part),
            fail(
                {integrity_mismatch, #{
                    expected_sha256 => Expected,
                    sha256 => Sha,
                    expected_size => ExpectedSize,
                    size => Size
                }},
                St#state{dl = undefined}
            )
    end.

progress(#state{dl = #dl{offset = Off, total = Total, last_progress = Last, file = F} = Dl} = St) ->
    Now = erlang:monotonic_time(millisecond),
    case Now - Last >= ?PROGRESS_EVERY_MS orelse Off =:= Total of
        true ->
            Data = #{file => maps:get(role, F), done => Off, total => Total},
            emit(progress, Data, St#state{dl = Dl#dl{last_progress = Now}});
        false ->
            St
    end.

stop_download(#state{dl = undefined} = St) ->
    St;
stop_download(#state{dl = #dl{ref = Ref, fd = Fd}} = St) ->
    _ =
        case Ref of
            {retry, _} -> ok;
            undefined -> ok;
            _ -> livery_client:stop_stream(Ref)
        end,
    _ = Fd =/= undefined andalso file:close(Fd),
    St#state{dl = undefined}.

register_model(
    #state{name = Name, fetched = Fetched, source = Source, defaults = D, opts = Opts} = St
) ->
    Parts = [{P, H, Bytes} || {model, P, H, Bytes} <- lists:reverse(Fetched)],
    MmProj =
        case [P || {projector, P, _, _} <- Fetched] of
            [P | _] -> P;
            [] -> undefined
        end,
    Entry = #{
        name => Name,
        source => #{kind => maps:get(kind, Source), ref => maps:get(ref, Source)},
        mmproj => bin_or_undefined(MmProj),
        load => maps:with([n_ctx, n_gpu_layers], Opts),
        defaults => D,
        keep_alive => maps:get(keep_alive, Opts, undefined),
        created_at => erlang:system_time(millisecond)
    },
    case register_parts(Parts, Entry) of
        {ok, Registered} -> terminal(done, #{model => Registered}, St);
        {error, Reason} -> fail({registry, Reason}, St)
    end.

register_parts([{Path, Sha, Size}], Entry) ->
    erllama_server_registry:add(Entry#{
        path => unicode:characters_to_binary(Path), sha256 => Sha, size => Size
    });
register_parts(Parts, Entry) ->
    Paths = [P || {P, _, _} <- Parts],
    case erllama_server_shards:inspect(Paths) of
        {ok, Size} ->
            erllama_server_registry:add(Entry#{
                size => Size, shard_digests => [H || {_, H, _} <- Parts]
            });
        {error, _} = E ->
            E
    end.

bin_or_undefined(undefined) -> undefined;
bin_or_undefined(P) -> unicode:characters_to_binary(P).

%% -----------------------------------------------------------------------------

fail(Reason, St) ->
    terminal(error, #{error => Reason}, St).

terminal(Type, Data, #state{id = Id} = St) ->
    St1 = emit(Type, Data, unlock(St)),
    ok = erllama_server_runs:set_status(Id, Type),
    _ = erlang:send_after(erllama_server_config:get(run_retention_ms), self(), expire),
    St1#state{
        status = Type,
        result = Data#{status => Type},
        events = erllama_server_events:close(St1#state.events)
    }.

emit(Type, Data, #state{events = Ev} = St) ->
    St#state{events = erllama_server_events:emit(Type, Data, Ev)}.

role(#{role := R}) -> R.

blob_path(undefined, _St) ->
    undefined;
blob_path(Sha, _St) ->
    erllama_server_shards:blob(Sha).

%% A file of unknown digest is staged under the job's name until its
%% digest is known.
part_path(#{sha256 := undefined, url := Url}, _St) ->
    Key = binary:encode_hex(crypto:hash(sha256, Url), lowercase),
    filename:join(erllama_server_registry:blobs_dir(), <<"url-", Key/binary, ".part">>);
part_path(#{sha256 := Sha}, St) ->
    <<(unicode:characters_to_binary(blob_path(Sha, St)))/binary, ".part">>.

header(Name, Headers) ->
    case [V || {K, V} <- Headers, string:lowercase(K) =:= Name] of
        [V | _] -> V;
        [] -> undefined
    end.

content_length(Headers) ->
    case header(<<"content-length">>, Headers) of
        undefined -> undefined;
        V -> binary_to_integer(string:trim(V))
    end.

add(_, undefined) -> undefined;
add(A, B) -> A + B.
