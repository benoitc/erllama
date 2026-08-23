%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_cache_tier_sup).
-moduledoc false.
%% Supervisor for the `disk' and `ram_file' tier servers added with
%% `erllama_cache:add_tier/1' or the `tiers' application environment
%% key. One `erllama_cache_disk_srv' child per tier, transient so a
%% tier that fails to start (bad root directory) does not take the
%% cache supervisor down.
-behaviour(supervisor).

-export([start_link/0, start_tier/3, stop_tier/1, tiers/0]).
-export([init/1]).

-define(SERVER, ?MODULE).

start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

-spec start_tier(atom(), disk | ram_file, file:name()) -> {ok, pid()} | {error, term()}.
start_tier(Name, Backend, Root) ->
    case supervisor:start_child(?SERVER, [Name, Backend, Root]) of
        {ok, Pid} -> {ok, Pid};
        {error, {already_started, _}} -> {error, already_started};
        {error, {Reason, _Stack}} when is_atom(Reason) -> {error, Reason};
        {error, _} = E -> E
    end.

-spec stop_tier(atom()) -> ok | {error, not_found}.
stop_tier(Name) ->
    case whereis(Name) of
        undefined ->
            {error, not_found};
        Pid ->
            case supervisor:terminate_child(?SERVER, Pid) of
                ok -> ok;
                {error, not_found} -> {error, not_found}
            end
    end.

%% Live tier servers as `{Name, Pid}'.
-spec tiers() -> [{atom(), pid()}].
tiers() ->
    [
        {Name, Pid}
     || {_, Pid, worker, _} <- supervisor:which_children(?SERVER),
        is_pid(Pid),
        {registered_name, Name} <- [erlang:process_info(Pid, registered_name)]
    ].

init([]) ->
    SupFlags = #{strategy => simple_one_for_one, intensity => 5, period => 30},
    Child = #{
        id => erllama_cache_disk_srv,
        start => {erllama_cache_disk_srv, start_link, []},
        restart => transient,
        shutdown => 5000,
        type => worker,
        modules => [erllama_cache_disk_srv]
    },
    {ok, {SupFlags, [Child]}}.
