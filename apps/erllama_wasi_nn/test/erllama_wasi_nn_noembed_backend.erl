%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% A model backend without the optional `embed/2` callback, as a model
%% that cannot answer embeddings. Everything it does export goes to
%% `erllama_model_stub`.
%% @end
-module(erllama_wasi_nn_noembed_backend).
-behaviour(erllama_model_backend).

-export([
    init/1,
    terminate/1,
    tokenize/2,
    detokenize/2,
    prefill/2,
    decode_one/2,
    kv_pack/2,
    kv_unpack/2,
    step/2,
    sampler_new/2,
    sampler_free/1
]).

init(Config) -> erllama_model_stub:init(Config).
terminate(S) -> erllama_model_stub:terminate(S).
tokenize(S, Text) -> erllama_model_stub:tokenize(S, Text).
detokenize(S, Tokens) -> erllama_model_stub:detokenize(S, Tokens).
prefill(S, Tokens) -> erllama_model_stub:prefill(S, Tokens).
decode_one(S, Tokens) -> erllama_model_stub:decode_one(S, Tokens).
kv_pack(S, Tokens) -> erllama_model_stub:kv_pack(S, Tokens).
kv_unpack(S, Bin) -> erllama_model_stub:kv_unpack(S, Bin).
step(S, Ops) -> erllama_model_stub:step(S, Ops).
sampler_new(S, Cfg) -> erllama_model_stub:sampler_new(S, Cfg).
sampler_free(Ref) -> erllama_model_stub:sampler_free(Ref).
