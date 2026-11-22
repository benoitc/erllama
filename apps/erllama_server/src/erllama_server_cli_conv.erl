%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_server_cli_conv).
-moduledoc false.
%% `erllama conversations' and `erllama conversation ...': the
%% conversations the release keeps (they survive a restart), and what
%% can be done with one.

-export([list/4, conversation/4]).

-spec list(node(), map(), map(), [string()]) -> {non_neg_integer(), iodata()}.
list(Node, _Info, _Opts, []) ->
    erllama_server_cli:reply(erllama_server_client:call(Node, conversations, []), fun rows/1);
list(_, _, _, _) ->
    erllama_server_cli:usage("conversations").

-spec conversation(node(), map(), map(), [string()]) -> {non_neg_integer(), iodata()}.
conversation(Node, _Info, _Opts, ["show", Id]) ->
    erllama_server_cli:reply(
        erllama_server_client:call(Node, conversation_export, [bin(Id), markdown]),
        fun(Text) -> [Text, "\n"] end
    );
conversation(Node, _Info, Opts, ["export", Id]) ->
    erllama_server_cli:reply(
        erllama_server_client:call(Node, conversation_export, [bin(Id), format(Opts)]),
        fun(Doc) -> [Doc, "\n"] end
    );
conversation(Node, _Info, _Opts, ["fork", Id]) ->
    erllama_server_cli:reply(
        erllama_server_client:call(Node, conversation_fork, [bin(Id)]),
        fun(#{conversation_id := New}) -> [New, "\n"] end
    );
conversation(Node, _Info, _Opts, ["rm", Id]) ->
    erllama_server_cli:reply(
        erllama_server_client:call(Node, conversation_delete, [bin(Id)]), fun(_) -> "" end
    );
conversation(Node, _Info, _Opts, ["attach", Id]) ->
    attach(Node, Id);
conversation(_, _, _, _) ->
    erllama_server_cli:usage("conversation").

format(Opts) ->
    case maps:get(format, Opts, "json") of
        "md" -> markdown;
        "markdown" -> markdown;
        _ -> json
    end.

%% Follow the run or session working in the conversation.
attach(Node, Id) ->
    case erllama_server_client:call(Node, conversation, [bin(Id)]) of
        {ok, #{active_run := Run}} when is_binary(Run) ->
            erllama_server_cli:run(["attach", binary_to_list(Run)]);
        {ok, _} ->
            {1, ["erllama: nothing is running in conversation ", Id, "\n"]};
        {error, Reason} ->
            erllama_server_cli:fail(Reason)
    end.

rows([]) ->
    "no conversations\n";
rows(L) ->
    [
        io_lib:format("~ts  ~ts  ~b messages  ~ts~ts~ts~n", [
            Id,
            text(maps:get(model, C)),
            T,
            when_text(maps:get(updated_at, C)),
            running(maps:get(active_run, C, undefined)),
            unresolved(maps:get(unresolved, C, []))
        ])
     || #{id := Id, turns := T} = C <- lists:sort(fun newest/2, L)
    ].

newest(#{updated_at := A}, #{updated_at := B}) -> A >= B.

running(undefined) -> "";
running(Run) -> ["  running ", Run].

unresolved([]) -> "";
unresolved(U) -> io_lib:format("  ~b unresolved", [length(U)]).

when_text(Ms) ->
    calendar:system_time_to_rfc3339(Ms div 1000, [{offset, "Z"}]).

text(V) when is_binary(V) -> V;
text(V) -> io_lib:format("~0p", [V]).

bin(S) -> unicode:characters_to_binary(S).
