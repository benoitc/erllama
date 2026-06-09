%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% Deterministic stub backend for `barrel_inference_model`.
%%
%% No NIF, no GGUF. tokenize uses `erlang:phash2/1` over whitespace-
%% delimited words; decode_one produces a deterministic next-token
%% from the context's hash; pack/unpack serialise the token list as
%% bytes. Useful for tests of the cache integration that don't need
%% real inference.
%% @end
-module(barrel_inference_model_stub).
-behaviour(barrel_inference_model_backend).

-export([
    init/1,
    terminate/1,
    tokenize/2,
    detokenize/2,
    prefill/2,
    decode_one/2,
    kv_pack/2,
    kv_pack/3,
    kv_unpack/2,
    kv_unpack/3,
    seq_rm/2,
    step/2,
    sampler_new/2,
    sampler_free/1,
    apply_chat_template/2,
    embed/2,
    set_grammar/2,
    configure_sampler/2,
    clear_sampler/1,
    load_adapter/2,
    unload_adapter/2,
    apply_adapters/2,
    thinking_signature/3,
    reset_context/1,
    abort_handle/1,
    %% Test helpers: read back what the most recent configure_sampler
    %% / clear_sampler / apply_adapters call saw.
    last_sampler_cfg/1,
    cleared/1,
    applied_adapters/1,
    %% Test helper: force the next step/2 to return {error, Reason},
    %% simulating a wedged/aborted decode so recovery can be tested
    %% without a real backend.
    wedge_next_step/1,
    %% Test helpers: read / clear the list of cfgs passed to
    %% sampler_new/2 since the last reset.
    sampler_new_cfgs/0,
    reset_sampler_new_cfgs/0,
    %% Optional backend callback exercised by warm_restore_primer/3
    %% (range KV trim). Instrumented so tests can assert it was called.
    seq_rm_last/3,
    seq_rm_last_calls/0,
    reset_seq_rm_last_calls/0
]).

%% Stub state.
%%
%% `sampler` is the currently-installed chain config (matches the real
%% backend's behaviour: it gets reset to #{} on clear_sampler/1).
%% `last_sampler` is the cfg the most recent configure_sampler/2 saw,
%% preserved across clear_sampler/1 so end-of-request cleanup doesn't
%% wipe out the value tests want to read.
-record(stub, {
    sampler = #{} :: map(),
    last_sampler = #{} :: map(),
    cleared = false :: boolean(),
    %% Most recently applied adapter set, as the {Ref, Scale} list the
    %% model layer passed to apply_adapters/2. Tests read this back to
    %% assert the snapshot rules.
    applied = [] :: [{reference(), float()}],
    %% Opt-in: when true, the first two decode ops per sampler emit
    %% `{thinking_token, _}`, the next emits `thinking_end`. Per-sampler
    %% phase is tracked via the process dictionary because step/2 takes
    %% no mutable state argument.
    thinking_capable = false :: boolean(),
    %% Opt-in: hold each `step/2' call for N ms via timer:sleep/1
    %% before returning results. Lets server-side concurrency tests
    %% deterministically keep a holder request in-flight while other
    %% requests race for the queue (e.g. chat_busy_returns_429).
    %% Default 0 disables the hold, so the cache / integration tests
    %% that use the stub pay no `timer:sleep(0)' syscall in their hot
    %% path.
    step_delay_ms = 0 :: non_neg_integer()
}).

init(Config) ->
    {ok, #stub{
        thinking_capable = bool_opt(thinking_capable, Config),
        step_delay_ms = step_delay_opt(step_delay_ms, Config)
    }}.

bool_opt(Key, Config) ->
    case maps:get(Key, Config, false) of
        true -> true;
        _ -> false
    end.

step_delay_opt(Key, Config) ->
    case maps:get(Key, Config, 0) of
        N when is_integer(N), N >= 0 -> N;
        _ -> 0
    end.

terminate(_S) ->
    ok.

tokenize(_S, Text) when is_binary(Text) ->
    [
        erlang:phash2(W) rem (1 bsl 32)
     || W <- binary:split(Text, <<" ">>, [global, trim_all]),
        W =/= <<>>
    ].

detokenize(_S, Tokens) ->
    list_to_binary(
        lists:join(<<" ">>, [integer_to_binary(T) || T <- Tokens])
    ).

prefill(_S, _Tokens) ->
    ok.

decode_one(_S, ContextTokens) ->
    Token = erlang:phash2({decode, ContextTokens}) rem (1 bsl 32),
    {ok, Token}.

kv_pack(_S, Tokens) ->
    barrel_inference_cache_key:encode_tokens(Tokens).

kv_pack(_S, Tokens, _SeqId) ->
    barrel_inference_cache_key:encode_tokens(Tokens).

kv_unpack(_S, _Bin) ->
    ok.

kv_unpack(_S, _Bin, _SeqId) ->
    ok.

%% Drops the per-seq phase counter when the scheduler releases a
%% seq's KV. Without this, a subsequent admission to the same
%% seq_id would inherit the prior request's state.
seq_rm(_S, SeqId) ->
    erlang:erase({stub_phase, SeqId}),
    ok.

%% Optional callback: warm_restore_primer/3 calls this to trim a seq's KV
%% to a prefix. The stub holds no real KV, so it just records the call so
%% tests can assert the range-trim ran with the expected N (= prefix len).
seq_rm_last(_S, SeqId, N) ->
    Prev = persistent_term:get({?MODULE, seq_rm_last_calls}, []),
    persistent_term:put({?MODULE, seq_rm_last_calls}, [{SeqId, N} | Prev]),
    ok.

seq_rm_last_calls() ->
    lists:reverse(persistent_term:get({?MODULE, seq_rm_last_calls}, [])).

reset_seq_rm_last_calls() ->
    _ = persistent_term:erase({?MODULE, seq_rm_last_calls}),
    ok.

%% Per-tick batched step. Prefill rows just acknowledge. Decode rows
%% derive a deterministic next token from
%% `{decode_step_stub, SeqId, Sampler}` so two concurrent seqs with
%% different prompts produce different streams and the same seq fed
%% the same sampler keeps producing the same token (matching the
%% prior single-seq stub semantics for cache-integration tests).
%% Outer clause peels off the step_delay_ms hold (when configured),
%% sleeps once, then recurses into the no-delay clause to keep the
%% inner wedge dispatch unchanged. step_delay_ms = 0 (the default)
%% short-circuits straight to the no-delay clause - no `timer:sleep(0)'
%% in the cache/integration hot path.
step(#stub{step_delay_ms = D} = S, Ops) when D > 0 ->
    timer:sleep(D),
    step(S#stub{step_delay_ms = 0}, Ops);
step(S, Ops) ->
    case persistent_term:get({?MODULE, wedge}, undefined) of
        undefined ->
            Results = [stub_step_op(Op, S) || Op <- Ops],
            {ok, Results};
        Reason ->
            %% Test knob: simulate a wedged/aborted decode once.
            persistent_term:erase({?MODULE, wedge}),
            {error, Reason}
    end.

stub_step_op({SeqId, {prefill, _Tokens}}, _S) ->
    {SeqId, prefilled};
stub_step_op({SeqId, {decode, Sampler}}, #stub{thinking_capable = false}) ->
    decode_token(SeqId, Sampler);
stub_step_op({SeqId, {decode, Sampler}}, #stub{thinking_capable = true}) ->
    Phase = next_phase(SeqId),
    advance_phase(SeqId, Sampler, Phase).

next_phase(SeqId) ->
    case erlang:get({stub_phase, SeqId}) of
        undefined -> thinking_emit_0;
        Other -> Other
    end.

advance_phase(SeqId, Sampler, thinking_emit_0) ->
    erlang:put({stub_phase, SeqId}, thinking_emit_1),
    T = erlang:phash2({thinking_stub, SeqId, Sampler, 0}) rem (1 bsl 32),
    {SeqId, {thinking_token, T}};
advance_phase(SeqId, Sampler, thinking_emit_1) ->
    erlang:put({stub_phase, SeqId}, thinking_end_due),
    T = erlang:phash2({thinking_stub, SeqId, Sampler, 1}) rem (1 bsl 32),
    {SeqId, {thinking_token, T}};
advance_phase(SeqId, _Sampler, thinking_end_due) ->
    erlang:put({stub_phase, SeqId}, thinking_done),
    {SeqId, thinking_end};
advance_phase(SeqId, Sampler, _) ->
    decode_token(SeqId, Sampler).

decode_token(SeqId, Sampler) ->
    T = erlang:phash2({decode_step_stub, SeqId, Sampler}) rem (1 bsl 32),
    {SeqId, {token, T, 0}}.

%% Deterministic per-seq stub signature. The stub ignores `Bytes`
%% and hashes the seq_id; real backends derive their signature from
%% the observed thinking text via HMAC.
thinking_signature(_S, SeqId, _Bytes) ->
    crypto:hash(sha256, <<"stub-thinking-sig-", (integer_to_binary(SeqId))/binary>>).

%% No real context to recreate; recovery just returns the same state.
reset_context(S) ->
    {ok, S}.

%% No interruptible context; the engine relies on the per-step budget.
abort_handle(_S) ->
    undefined.

%% Test knob: arm the next step/2 to return {error, Reason} once.
wedge_next_step(Reason) ->
    persistent_term:put({?MODULE, wedge}, Reason),
    ok.

%% Test helper: cfgs passed to sampler_new/2 since the last reset.
sampler_new_cfgs() ->
    persistent_term:get({?MODULE, sampler_new_cfgs}, []).

reset_sampler_new_cfgs() ->
    persistent_term:put({?MODULE, sampler_new_cfgs}, []),
    ok.

%% Sampler refs are opaque references. Free drops the per-sampler
%% per-sampler stub phase (a no-op when neither thinking_capable nor
%% tool_call_capable is set).
sampler_new(_S, Cfg) ->
    %% Record the cfg so tests can assert which sampler chains the
    %% scheduler built (e.g. the grammar-less greedy `#{temperature
    %% => 0.0}` chain vs a request chain carrying a grammar).
    Prev = persistent_term:get({?MODULE, sampler_new_cfgs}, []),
    persistent_term:put({?MODULE, sampler_new_cfgs}, Prev ++ [Cfg]),
    {ok, make_ref()}.

sampler_free(_Sampler) ->
    %% Stub state is now keyed on seq_id; cleanup happens in seq_rm.
    ok.

%% Render a chat request as `system\nrole: content\nrole: content\n`
%% and tokenise via the same phash2 scheme as tokenize/2. Tools are
%% inlined into the system prefix. Deterministic and roundtrippable
%% enough for tests.
apply_chat_template(S, Request) when is_map(Request) ->
    Messages = maps:get(messages, Request, []),
    System = maps:get(system, Request, undefined),
    Tools = maps:get(tools, Request, undefined),
    Rendered = render(System, Tools, Messages),
    {ok, tokenize(S, Rendered)}.

%% A 16-dim hash-derived embedding vector. Deterministic per token
%% list. Useful for /v1/embeddings shape testing without a real model.
embed(_S, Tokens) when is_list(Tokens) ->
    Seed = erlang:phash2({embed, Tokens}),
    Vec = [
        float((Seed bsr (I * 4)) band 16#FFFF) / 65535.0
     || I <- lists:seq(0, 15)
    ],
    {ok, Vec}.

%% Stub backend doesn't sample (decode_one returns a phash2-derived
%% token deterministically), so grammar / sampler params are ignored.
%% The most recent config is recorded on the state so tests can read
%% it back via `last_sampler_cfg/1`.
set_grammar(#stub{sampler = Cfg} = S, Grammar) when is_binary(Grammar) ->
    NewCfg = Cfg#{grammar => Grammar},
    {ok, S#stub{sampler = NewCfg, last_sampler = NewCfg, cleared = false}};
set_grammar(#stub{} = S, undefined) ->
    {ok, S}.

configure_sampler(#stub{} = S, Cfg) when is_map(Cfg) ->
    {ok, S#stub{sampler = Cfg, last_sampler = Cfg, cleared = false}}.

clear_sampler(#stub{} = S) ->
    {ok, S#stub{sampler = #{}, cleared = true}}.

last_sampler_cfg(#stub{last_sampler = Cfg}) -> Cfg.
cleared(#stub{cleared = C}) -> C.
applied_adapters(#stub{applied = A}) -> A.

%% LoRA stubs: the adapter handle is a fresh reference per load so
%% tests can distinguish multiple adapters; unload is a no-op;
%% apply_adapters just records the call.
load_adapter(#stub{} = S, _Path) ->
    {ok, make_ref(), S}.

unload_adapter(#stub{} = S, _Ref) ->
    {ok, S}.

apply_adapters(#stub{} = S, Adapters) when is_list(Adapters) ->
    {ok, S#stub{applied = Adapters}}.

%% =============================================================================
%% Internal: chat-template rendering
%% =============================================================================

render(System, Tools, Messages) ->
    Header =
        case System of
            undefined -> [];
            <<>> -> [];
            _ -> [<<"system: ">>, System, <<"\n">>]
        end,
    ToolsBlob = render_tools(Tools),
    Body = [render_message(M) || M <- Messages],
    iolist_to_binary([Header, ToolsBlob, Body]).

render_tools(undefined) ->
    [];
render_tools([]) ->
    [];
render_tools(Tools) when is_list(Tools) ->
    Lines = [render_tool(T) || T <- Tools],
    [<<"tools:\n">>, Lines].

render_tool(#{name := Name} = T) ->
    Desc = maps:get(description, T, <<>>),
    [<<"  - ">>, Name, <<": ">>, Desc, <<"\n">>].

render_message(#{role := Role, content := Content}) when is_binary(Content) ->
    [Role, <<": ">>, Content, <<"\n">>];
render_message(#{role := Role, content := Blocks}) when is_list(Blocks) ->
    Texts = [B || #{type := <<"text">>, text := B} <- Blocks],
    [Role, <<": ">>, lists:join(<<" ">>, Texts), <<"\n">>];
render_message(_) ->
    <<>>.
