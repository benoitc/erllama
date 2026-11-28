%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
-module(erllama_tp_rank_role).
-moduledoc """
Read the local status of a rank started with the erllama application's
`tp_rank` setting. See the tensor-parallel guide for a configuration.
""".
-behaviour(gen_server).
-export([start_link/1, status/0, attachment/2, loss_reason/2]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-record(state, {
    status :: map(),
    worker :: pid(),
    attachment :: tuple() | undefined,
    coordinator :: pid() | undefined
}).
-type state() :: #state{}.

-doc false.
-spec start_link(map()) -> gen_server:start_ret().
start_link(Config) -> gen_server:start_link({local, ?MODULE}, ?MODULE, Config, []).

-doc "Read forming, ready or lost on this node, including the group identity and any loss reason.".
-spec status() -> map() | undefined.
status() ->
    try
        gen_server:call(?MODULE, status)
    catch
        exit:{noproc, _} -> undefined
    end.

-doc false.
-spec attachment(pid(), pid()) -> {ok, {pid(), string(), map()}} | {error, term()}.
attachment(Role, Coordinator) ->
    gen_server:call(Role, {attachment, Coordinator}).

-spec init(map()) -> {ok, state()} | {stop, term()}.
init(Config) ->
    process_flag(trap_exit, true),
    case config(Config) of
        {ok, Opts} ->
            Info = maps:with([rank, world, generation, group_id], Opts),
            Owner = self(),
            Worker = spawn_link(fun() -> form(Owner, Opts) end),
            {ok, #state{status = Info#{state => forming}, worker = Worker}};
        {error, Why} ->
            {stop, Why}
    end.

config(C) when is_map(C) ->
    Defaults = #{gpu => 0, timeout => 60000, model_opts => #{}, context_opts => #{}},
    validate(maps:merge(Defaults, C));
config(_) ->
    {error, invalid_tp_rank_config}.

validate(#{group_id := Id, generation := Gen, rank := R, world := W} = C) when
    map_size(C) =:= 10,
    is_binary(Id),
    byte_size(Id) > 0,
    is_integer(Gen),
    Gen >= 0,
    is_integer(R),
    R >= 0,
    is_integer(W),
    R < W
->
    validate_local(C);
validate(_) ->
    {error, invalid_tp_rank_config}.

validate_local(#{coordinator := Addr, model_path := Path, gpu := Gpu, timeout := Ms} = C) when
    is_integer(Gpu), Gpu >= 0, is_integer(Ms), Ms > 0
->
    case
        (is_binary(Path) orelse is_list(Path)) andalso valid_address(Addr) andalso
            filename:pathtype(Path) =:= absolute
    of
        true -> erllama_opts:rank_options(C#{model_path => unicode:characters_to_list(Path)});
        false -> {error, invalid_tp_rank_config}
    end;
validate_local(_) ->
    {error, invalid_tp_rank_config}.

valid_address({Name, Node}) -> is_atom(Name) andalso is_atom(Node);
valid_address(Node) -> is_atom(Node).

address(Node) when is_atom(Node) -> {erllama_tp_coordinator, Node};
address(Pair) -> Pair.

handle_call(status, _, #state{status = Status} = S) ->
    {reply, Status, S};
handle_call({attachment, _}, _, #state{status = #{state := lost, reason := Why}} = S) ->
    {reply, {error, Why}, S};
handle_call({attachment, C}, _, #state{coordinator = C, attachment = A} = S) when A =/= undefined ->
    {reply, {ok, A}, S};
handle_call({attachment, _}, _, S) ->
    {reply, {error, group_forming}, S};
handle_call({bind, Coord}, {Worker, _}, #state{worker = Worker} = S) ->
    _ = monitor(process, Coord),
    {reply, ok, S#state{coordinator = Coord}}.

handle_cast(_, S) -> {noreply, S}.

handle_info(_, #state{status = #{state := lost}} = S) ->
    {noreply, S};
handle_info(
    {ready, Worker, {_, _, #{context_opts := C}} = A},
    #state{worker = Worker, status = Status, coordinator = Coord} = S
) ->
    gen_server:cast(Coord, {rank_ready, maps:get(rank, Status), self(), maps:get(n_seq_max, C, 1)}),
    {noreply, S#state{attachment = A, status = Status#{state => ready}}};
handle_info({tp_group_lost, Coord, Why}, #state{coordinator = Coord} = S) when is_pid(Coord) ->
    lost(Why, S);
handle_info({'DOWN', _, process, Coord, Reason}, #state{coordinator = Coord} = S) ->
    lost({coordinator_lost, plain_reason(Reason)}, S);
handle_info({'EXIT', Worker, Reason}, #state{worker = Worker, status = #{rank := R}} = S) ->
    lost(loss_reason(R, Reason), S);
handle_info(_, S) ->
    {noreply, S}.

lost(Why, #state{worker = Worker, status = Info} = S) ->
    exit(Worker, shutdown),
    {noreply, S#state{status = Info#{state => lost, reason => Why}}}.

terminate(_, #state{worker = Worker}) ->
    exit(Worker, shutdown),
    ok.

-doc false.
-spec loss_reason(non_neg_integer(), term()) -> term().
loss_reason(_, {rank_error, Why}) -> Why;
loss_reason(Rank, Reason) -> {tp_rank_lost, Rank, plain_reason(Reason)}.

plain_reason(Reason) when is_atom(Reason) -> Reason;
plain_reason(_) -> rank_failed.

form(Owner, #{rank := R, world := W, coordinator := Addr, timeout := Ms, model_path := Path} = C) ->
    Key = {maps:get(group_id, C), maps:get(generation, C)},
    Coord = value(resolve(address(Addr), Key, erlang:monotonic_time(millisecond) + Ms)),
    ok = gen_server:call(Owner, {bind, Coord}),
    Opts = {maps:get(model_opts, C), maps:get(context_opts, C)},
    Desc = erllama_tp_rank:describe(Path, Opts, W),
    {ok, Version} = application:get_key(erllama, vsn),
    Lib = filename:join(code:priv_dir(erllama), "erllama_nif.so"),
    Build = value(erllama_fingerprint:derive(Lib, safe)),
    Signature = crypto:hash(sha256, term_to_binary({Path, Desc, Version, Build}, [deterministic])),
    ok = value(call(Coord, {enroll, R, W, Owner, Signature}, Ms)),
    case maps:get(head, Desc) of
        {error, Reason} -> exit({rank_error, Reason});
        _ -> ok
    end,
    {Start, Load} = rank_calls(),
    Worker = value(Start(C#{join => {call, Coord}})),
    Ref = monitor(process, Worker),
    ok = value(Load(Worker, Path, Opts)),
    Owner ! {ready, self(), {Worker, Path, Desc}},
    wait_rank(Worker, Ref).

%% A supervisor can start the three releases together. Retry only until
%% first contact. After binding, a lost coordinator fences this generation.
resolve(Addr, Key, Deadline) ->
    case Deadline - erlang:monotonic_time(millisecond) of
        Left when Left > 0 ->
            try gen_server:call(Addr, {resolve, Key}, Left) of
                Reply -> Reply
            catch
                exit:{Reason, _} when Reason =:= noproc; element(1, Reason) =:= nodedown ->
                    timer:sleep(min(100, Left)),
                    resolve(Addr, Key, Deadline);
                exit:{timeout, _} ->
                    {error, join_timeout};
                exit:{Reason, _} ->
                    {error, {coordinator_lost, plain_reason(Reason)}}
            end;
        _ ->
            {error, join_timeout}
    end.

wait_rank(Worker, Ref) ->
    receive
        {'DOWN', Ref, process, Worker, Why} -> exit(Why)
    after 3600000 -> wait_rank(Worker, Ref)
    end.

call(Server, Msg, Ms) ->
    try
        gen_server:call(Server, Msg, Ms)
    catch
        exit:{timeout, _} -> {error, join_timeout};
        exit:{Reason, _} -> {error, {coordinator_lost, plain_reason(Reason)}}
    end.

value({ok, V}) -> V;
value(ok) -> ok;
value({error, Why}) -> exit({rank_error, Why}).

-ifdef(TEST).
rank_calls() ->
    Mod = application:get_env(erllama, tp_rank_driver, erllama_tp_rank),
    %% Embedded releases load their boot script eagerly. The CPU test
    %% driver lives only on the test node's extra code path.
    case code:ensure_loaded(Mod) of
        {error, embedded} -> {module, Mod} = code:load_file(Mod);
        {module, Mod} -> ok
    end,
    {fun Mod:start_link/1, fun Mod:load/3}.
-else.
rank_calls() -> {fun erllama_tp_rank:start_link/1, fun erllama_tp_rank:load/3}.
-endif.
