%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_failures).
-moduledoc false.
%% The release's recent failures, for operators: what failed, which job
%% or resource it belonged to, and what happened next (what was ended,
%% what kept working). The process that owns a failure records it when
%% it handles it, so each entry says what the release did about it.
%%
%% A bounded ring in a protected ETS table: readers go to ETS, writes
%% are casts to this process. A count per kind is kept for metrics.
%% Nothing here survives a restart; the conversation logs carry what
%% must.

-behaviour(gen_server).

-export([start_link/0, record/3, record/4, recent/1, counts/0]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-define(TAB, ?MODULE).

-record(state, {}).
-type state() :: #state{}.
-define(COUNTS, erllama_server_failure_counts).
-define(MAX, 200).
-define(REASON_BYTES, 300).

-type kind() ::
    run_failed
    | model_down
    | turn_failed
    | tool_crashed
    | tool_timed_out
    | agent_down
    | mcp_down
    | recovered.
-export_type([kind/0]).

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

%% `Subject' is the job or resource the failure belongs to (a run or
%% agent id, a model, an MCP server). record/3 says what the release
%% does about a failure of that kind; record/4 says it explicitly.
-spec record(kind(), binary(), term()) -> ok.
record(Kind, Subject, Reason) ->
    record(Kind, Subject, Reason, outcome(Kind)).

outcome(run_failed) ->
    <<"the run ended with an error event; other runs went on">>;
outcome(model_down) ->
    <<"its runs ended with an error; the next request loads it again">>;
outcome(turn_failed) ->
    <<"the turn ended; the session kept running">>;
outcome(tool_crashed) ->
    <<"the call got an error result and the turn went on">>;
outcome(tool_timed_out) ->
    <<"its process group was killed; the call got an error result">>;
outcome(agent_down) ->
    <<"the session ended; its conversation was released and other jobs went on">>;
outcome(mcp_down) ->
    <<"its tools were withdrawn; the release reconnects with backoff">>;
outcome(recovered) ->
    <<"the conversation was repaired">>.

-spec record(kind(), binary(), term(), binary()) -> ok.
record(Kind, Subject, Reason, Outcome) ->
    Entry = #{
        at => erlang:system_time(millisecond),
        kind => Kind,
        subject => Subject,
        reason => format(Reason),
        outcome => Outcome
    },
    gen_server:cast(?MODULE, {record, Entry}).

%% The last `N' failures, newest first.
-spec recent(non_neg_integer()) -> [map()].
recent(N) ->
    try
        take(ets:last(?TAB), N, [])
    catch
        error:badarg -> []
    end.

-spec counts() -> #{kind() => non_neg_integer()}.
counts() ->
    try
        maps:from_list(ets:tab2list(?COUNTS))
    catch
        error:badarg -> #{}
    end.

take('$end_of_table', _N, Acc) ->
    lists:reverse(Acc);
take(_Key, 0, Acc) ->
    lists:reverse(Acc);
take(Key, N, Acc) ->
    case ets:lookup(?TAB, Key) of
        [{_, Entry}] -> take(ets:prev(?TAB, Key), N - 1, [Entry | Acc]);
        [] -> take(ets:prev(?TAB, Key), N, Acc)
    end.

format(Reason) when is_binary(Reason), byte_size(Reason) =< ?REASON_BYTES ->
    Reason;
format(Reason) ->
    Bin = iolist_to_binary(io_lib:format("~0tP", [Reason, 12])),
    case byte_size(Bin) > ?REASON_BYTES of
        true -> <<(binary:part(Bin, 0, ?REASON_BYTES))/binary, "...">>;
        false -> Bin
    end.

-spec init([]) -> {ok, state()}.
init([]) ->
    _ = ets:new(?TAB, [named_table, ordered_set, protected, {read_concurrency, true}]),
    _ = ets:new(?COUNTS, [named_table, set, protected, {read_concurrency, true}]),
    {ok, #state{}}.

handle_call(_Req, _From, S) ->
    {reply, {error, unknown_call}, S}.

handle_cast({record, #{kind := Kind} = Entry}, S) ->
    true = ets:insert(?TAB, {erlang:unique_integer([monotonic, positive]), Entry}),
    _ = ets:update_counter(?COUNTS, Kind, 1, {Kind, 0}),
    case ets:info(?TAB, size) > ?MAX of
        true -> ets:delete(?TAB, ets:first(?TAB));
        false -> ok
    end,
    {noreply, S};
handle_cast(_Msg, S) ->
    {noreply, S}.

handle_info(_Msg, S) ->
    {noreply, S}.
