/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */
/* Decode paths: the multi-sequence step tick, single-seq
 * prefill/decode_one, embeddings, and the speculative-decoding
 * forward pass. All share decode_ready / per_seq bookkeeping. */

#include "erllama_nif_int.h"
#include "erllama_safe.h"

#include <math.h>
#include <string.h>

/* =========================================================================
 * nif_step: multi-sequence batched decode
 *
 *   nif_step(CtxRef, [{SeqId, {prefill, [Token]} | {decode, SamplerRef}}])
 *     -> {ok, [{SeqId, prefilled | {token, Token, Eog :: 0 | 1}}]}
 *      | {error, atom()}
 *
 * Each tick is exactly one llama_decode call that mixes prefill and
 * decode rows freely (SARATHI-style co-batching). Sample-then-decode
 * order matches nif_decode_one: each decode row samples from the
 * PRIOR tick's logits at `per_seq[seq_id].last_logits_idx` BEFORE
 * the new batch is built, so the token Erlang receives is the same
 * token that lands in KV by the time the call returns. That
 * preserves the cache invariant (every committed token is in KV).
 *
 * Prefill rows do not sample: their last token gets logits=true in
 * the batch so the NEXT tick can sample for that seq. The first
 * output token for a freshly admitted seq is produced one tick
 * after its prefill — hidden in production by co-batching with
 * in-flight decoders.
 * ========================================================================= */

typedef struct {
    int seq_id;
    int is_prefill;                /* 1 prefill, 0 decode */
    llama_token *prefill_tokens;   /* prefill only; enif_alloc'd */
    int32_t prefill_n;             /* prefill only */
    erllama_sampler_t *sampler;    /* decode only */
    llama_token sampled;           /* decode only, set during pre-sample */
    int eog;                       /* decode only */
    /* Optional per-token logprobs (sampler->n_probs > 0): the
     * sampled token's logprob plus the top-lp_count ids/logprobs,
     * both enif_alloc'd during pre-sample. lp_count 0 = none. */
    int32_t lp_count;
    float sampled_lp;
    llama_token *lp_ids;
    float *lp_vals;
} erllama_step_op_t;

static void free_step_ops(erllama_step_op_t *ops, unsigned int n) {
    if (!ops) return;
    for (unsigned int i = 0; i < n; i++) {
        if (ops[i].prefill_tokens) enif_free(ops[i].prefill_tokens);
        if (ops[i].lp_ids) enif_free(ops[i].lp_ids);
        if (ops[i].lp_vals) enif_free(ops[i].lp_vals);
    }
    enif_free(ops);
}

ERL_NIF_TERM nif_step(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    unsigned int n_ops;
    if (!enif_get_list_length(env, argv[1], &n_ops)) {
        return enif_make_badarg(env);
    }
    if (n_ops == 0) {
        return enif_make_tuple2(env, atom_ok, enif_make_list(env, 0));
    }

    erllama_step_op_t *ops = enif_alloc(sizeof(erllama_step_op_t) * n_ops);
    if (!ops) {
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    memset(ops, 0, sizeof(erllama_step_op_t) * n_ops);

    /* Parse `[{SeqId, {Tag, Arg}}, ...]` into the working array. Any
     * malformed entry rejects the whole call with badarg. */
    ERL_NIF_TERM head;
    ERL_NIF_TERM tail = argv[1];
    unsigned int parsed = 0;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        const ERL_NIF_TERM *outer;
        int outer_arity;
        if (!enif_get_tuple(env, head, &outer_arity, &outer) || outer_arity != 2) {
            free_step_ops(ops, parsed);
            return enif_make_badarg(env);
        }
        int seq_id;
        if (!enif_get_int(env, outer[0], &seq_id) ||
            seq_id < 0 || seq_id >= ERLLAMA_N_SEQ_MAX_CAP) {
            free_step_ops(ops, parsed);
            return enif_make_badarg(env);
        }
        ops[parsed].seq_id = seq_id;

        const ERL_NIF_TERM *inner;
        int inner_arity;
        if (!enif_get_tuple(env, outer[1], &inner_arity, &inner) || inner_arity != 2) {
            free_step_ops(ops, parsed);
            return enif_make_badarg(env);
        }
        char tag[16];
        if (!enif_get_atom(env, inner[0], tag, sizeof(tag), ERL_NIF_LATIN1)) {
            free_step_ops(ops, parsed);
            return enif_make_badarg(env);
        }
        if (strcmp(tag, "prefill") == 0) {
            ops[parsed].is_prefill = 1;
            int rc = read_token_list(
                env, inner[1], &ops[parsed].prefill_tokens, &ops[parsed].prefill_n
            );
            if (rc != 1 || ops[parsed].prefill_n <= 0) {
                free_step_ops(ops, parsed + 1);
                return enif_make_badarg(env);
            }
        } else if (strcmp(tag, "decode") == 0) {
            ops[parsed].is_prefill = 0;
            erllama_sampler_t *samp;
            if (!enif_get_resource(env, inner[1], SAMPLER_RT, (void **) &samp)) {
                free_step_ops(ops, parsed);
                return enif_make_badarg(env);
            }
            /* Sampler must be bound to this context — its internal
             * chain was built against `c->ctx` at sampler_new time. */
            if (samp->ctx_res != c) {
                free_step_ops(ops, parsed);
                return enif_make_badarg(env);
            }
            ops[parsed].sampler = samp;
        } else {
            free_step_ops(ops, parsed);
            return enif_make_badarg(env);
        }
        parsed++;
    }

    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        free_step_ops(ops, n_ops);
        return enif_make_tuple2(env, atom_error, atom_released);
    }

    const struct llama_model *model = erllama_safe_get_model(c->ctx);
    const struct llama_vocab *vocab = model ? erllama_safe_model_get_vocab(model) : NULL;

    /* Pre-sample: read each decode row's next token from the PRIOR
     * tick's logits (still in ctx because we haven't called
     * llama_decode yet in this tick). */
    for (unsigned int i = 0; i < n_ops; i++) {
        if (ops[i].is_prefill) continue;
        int seq_id = ops[i].seq_id;
        int32_t li = c->per_seq[seq_id].last_logits_idx;
        if (li < 0) {
            pthread_mutex_unlock(&c->mu);
            free_step_ops(ops, n_ops);
            return enif_make_tuple2(env, atom_error, atom_no_logits);
        }
        pthread_mutex_lock(&ops[i].sampler->mu);
        if (!ops[i].sampler->chain) {
            pthread_mutex_unlock(&ops[i].sampler->mu);
            pthread_mutex_unlock(&c->mu);
            free_step_ops(ops, n_ops);
            return enif_make_tuple2(env, atom_error, atom_released);
        }
        int n_probs = ops[i].sampler->n_probs;
        llama_token tok = erllama_safe_sampler_sample(
            ops[i].sampler->chain, c->ctx, li
        );
        pthread_mutex_unlock(&ops[i].sampler->mu);
        if (tok < 0) {
            pthread_mutex_unlock(&c->mu);
            free_step_ops(ops, n_ops);
            return enif_make_tuple2(env, atom_error, atom_exception);
        }
        ops[i].sampled = tok;
        ops[i].eog = vocab ? erllama_safe_vocab_is_eog(vocab, tok) : 0;
        /* Optional logprobs: the raw logits row `li` is untouched by
         * sampling (the chain works on a candidates copy), so the
         * full-vocab log-softmax here reports the model's own
         * distribution, OpenAI-style. Failure downgrades to "no
         * logprobs for this token" rather than failing the tick. */
        if (n_probs > 0 && vocab) {
            int32_t nv = erllama_safe_vocab_n_tokens(vocab);
            llama_token *ids = enif_alloc(sizeof(llama_token) * (size_t) n_probs);
            float *vals = enif_alloc(sizeof(float) * (size_t) n_probs);
            if (ids && vals) {
                int32_t got = erllama_safe_top_logprobs(
                    c->ctx, li, nv, n_probs, tok,
                    ids, vals, &ops[i].sampled_lp);
                if (got > 0) {
                    ops[i].lp_count = got;
                    ops[i].lp_ids = ids;
                    ops[i].lp_vals = vals;
                    ids = NULL;
                    vals = NULL;
                }
            }
            if (ids) enif_free(ids);
            if (vals) enif_free(vals);
        }
    }

    /* Compute the total batch token count: each decode row contributes
     * 1, each prefill row contributes its slice length. */
    int64_t total64 = 0;
    for (unsigned int i = 0; i < n_ops; i++) {
        total64 += ops[i].is_prefill ? (int64_t) ops[i].prefill_n : 1;
    }
    if (total64 <= 0 || total64 > INT32_MAX) {
        pthread_mutex_unlock(&c->mu);
        free_step_ops(ops, n_ops);
        return enif_make_tuple2(env, atom_error, atom_batch_overflow);
    }
    /* Also bound by the live ctx's n_batch — overflowing this is what
     * causes the safe_decode error in production today; surface a
     * clean error tuple so callers (a budget-aware scheduler) can
     * shrink and retry. */
    uint32_t n_batch_cap = erllama_safe_n_batch(c->ctx);
    if ((uint32_t) total64 > n_batch_cap) {
        pthread_mutex_unlock(&c->mu);
        free_step_ops(ops, n_ops);
        return enif_make_tuple2(env, atom_error, atom_batch_overflow);
    }
    int32_t total = (int32_t) total64;

    /* Build the batch. The vendored llama_batch API exposes
     * llama_batch_init + llama_batch_free but no llama_batch_add
     * helper — fields are public POD and we fill them manually.
     * llama_batch_init allocates the arrays with size = total and
     * each seq_id[i] pointing to an array of length 1 (n_seq_max=1
     * argument). */
    struct llama_batch batch = erllama_safe_batch_init(total, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id ||
        !batch.seq_id || !batch.logits) {
        erllama_safe_batch_free(batch);
        pthread_mutex_unlock(&c->mu);
        free_step_ops(ops, n_ops);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    batch.n_tokens = 0;

    int32_t *last_logits_idx_new = enif_alloc(sizeof(int32_t) * n_ops);
    if (!last_logits_idx_new) {
        erllama_safe_batch_free(batch);
        pthread_mutex_unlock(&c->mu);
        free_step_ops(ops, n_ops);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }

    for (unsigned int i = 0; i < n_ops; i++) {
        int seq_id = ops[i].seq_id;
        int32_t start_pos = c->per_seq[seq_id].next_pos;
        int slice_n = ops[i].is_prefill ? ops[i].prefill_n : 1;
        for (int j = 0; j < slice_n; j++) {
            int32_t row = batch.n_tokens;
            llama_token tok =
                ops[i].is_prefill ? ops[i].prefill_tokens[j] : ops[i].sampled;
            batch.token[row]    = tok;
            batch.pos[row]      = start_pos + (llama_pos) j;
            batch.n_seq_id[row] = 1;
            batch.seq_id[row][0] = (llama_seq_id) seq_id;
            batch.logits[row]   = (j == slice_n - 1) ? (int8_t) 1 : (int8_t) 0;
            if (j == slice_n - 1) {
                last_logits_idx_new[i] = row;
            }
            batch.n_tokens++;
        }
    }

    arm_decode(c);
    int dr = erllama_safe_decode(c->ctx, batch);
    if (dr != 0) {
        erllama_safe_batch_free(batch);
        enif_free(last_logits_idx_new);
        c->decode_ready = 0;
        ERL_NIF_TERM why =
            (dr == ERLLAMA_DECODE_EXC_SENTINEL)
                ? atom_exception
                : classify_decode_error(env, c, dr);
        pthread_mutex_unlock(&c->mu);
        free_step_ops(ops, n_ops);
        return enif_make_tuple2(env, atom_error, why);
    }

    /* Update per-seq state from the just-decoded batch. */
    for (unsigned int i = 0; i < n_ops; i++) {
        int seq_id = ops[i].seq_id;
        int slice_n = ops[i].is_prefill ? ops[i].prefill_n : 1;
        c->per_seq[seq_id].next_pos += slice_n;
        c->per_seq[seq_id].last_logits_idx = last_logits_idx_new[i];
    }
    c->decode_ready = 1;

    erllama_safe_batch_free(batch);
    enif_free(last_logits_idx_new);
    pthread_mutex_unlock(&c->mu);

    /* Build result list. Prefill rows -> `{seq_id, prefilled}`;
     * decode rows -> `{seq_id, {token, Token, EogFlag}}`, or with
     * logprobs requested `{seq_id, {token, Token, EogFlag,
     * {SampledLp, [{Id, Lp}, ...]}}}`. */
    ERL_NIF_TERM atom_prefilled = enif_make_atom(env, "prefilled");
    ERL_NIF_TERM atom_token     = enif_make_atom(env, "token");
    ERL_NIF_TERM *results = enif_alloc(sizeof(ERL_NIF_TERM) * n_ops);
    if (!results) {
        free_step_ops(ops, n_ops);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    for (unsigned int i = 0; i < n_ops; i++) {
        ERL_NIF_TERM payload;
        if (ops[i].is_prefill) {
            payload = atom_prefilled;
        } else if (ops[i].lp_count > 0) {
            ERL_NIF_TERM top = enif_make_list(env, 0);
            for (int32_t j = ops[i].lp_count - 1; j >= 0; j--) {
                ERL_NIF_TERM pair = enif_make_tuple2(env,
                    enif_make_int(env, ops[i].lp_ids[j]),
                    enif_make_double(env, (double) ops[i].lp_vals[j]));
                top = enif_make_list_cell(env, pair, top);
            }
            ERL_NIF_TERM lp = enif_make_tuple2(env,
                enif_make_double(env, (double) ops[i].sampled_lp),
                top);
            payload = enif_make_tuple4(env,
                atom_token,
                enif_make_int(env, ops[i].sampled),
                enif_make_int(env, ops[i].eog ? 1 : 0),
                lp);
        } else {
            payload = enif_make_tuple3(env,
                atom_token,
                enif_make_int(env, ops[i].sampled),
                enif_make_int(env, ops[i].eog ? 1 : 0));
        }
        results[i] = enif_make_tuple2(env,
            enif_make_int(env, ops[i].seq_id),
            payload);
    }
    ERL_NIF_TERM result_list = enif_make_list_from_array(env, results, n_ops);
    enif_free(results);
    free_step_ops(ops, n_ops);
    return enif_make_tuple2(env, atom_ok, result_list);
}

ERL_NIF_TERM nif_prefill(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    llama_token *tokens = NULL;
    int32_t n = 0;
    int rc = read_token_list(env, argv[1], &tokens, &n);
    if (rc != 1) return token_list_error(env, rc);
    if (n == 0) {
        if (tokens) enif_free(tokens);
        return atom_ok;
    }
    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    /* Validate token IDs against the model vocab before handing them
     * to llama_decode. An out-of-range positive ID would otherwise
     * reach `id_to_token.at(id)` deep inside llama and throw a C++
     * exception across the C ABI. */
    const struct llama_model *model = erllama_safe_get_model(c->ctx);
    const struct llama_vocab *vocab =
        model ? erllama_safe_model_get_vocab(model) : NULL;
    int32_t n_vocab = vocab ? erllama_safe_vocab_n_tokens(vocab) : 0;
    /* Fail closed if the vocab lookup failed: without n_vocab we
     * cannot validate token IDs, and an out-of-range positive ID
     * would reach `id_to_token.at(id)` deep inside llama and throw
     * a C++ exception across the C ABI. */
    if (n_vocab <= 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    for (int32_t i = 0; i < n; i++) {
        if (tokens[i] >= n_vocab) {
            pthread_mutex_unlock(&c->mu);
            enif_free(tokens);
            return enif_make_tuple2(env, atom_error, atom_invalid_token);
        }
    }
    /* Bounds-check against the live context. llama_decode dereferences
     * past the KV slab when n_tokens >= n_ctx, and is undefined when
     * n_tokens > n_batch -- both produce SIGSEGV under real load. */
    uint32_t n_ctx = erllama_safe_n_ctx(c->ctx);
    uint32_t n_batch = erllama_safe_n_batch(c->ctx);
    if (n_ctx == 0 || n_batch == 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if ((uint32_t) n >= n_ctx) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_context_overflow);
    }
    if ((uint32_t) n > n_batch) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_batch_overflow);
    }
    struct llama_batch batch = erllama_safe_batch_get_one(tokens, n);
    int dr = erllama_safe_decode(c->ctx, batch);
    if (dr == 0) c->decode_ready = 1;
    pthread_mutex_unlock(&c->mu);
    enif_free(tokens);
    if (dr == ERLLAMA_DECODE_EXC_SENTINEL) {
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (dr != 0) {
        return enif_make_tuple2(env, atom_error, enif_make_int(env, dr));
    }
    return atom_ok;
}

ERL_NIF_TERM nif_decode_one(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    /* `llama_sampler_sample` -> `llama_get_logits_ith` aborts via
     * GGML_ASSERT(logits != nullptr) when no decode has produced
     * sample-able logits yet. We can't catch that abort, so we refuse
     * to call sampler_sample unless the last successful op was a
     * decode (set by nif_prefill / by ourselves below). kv_unpack and
     * kv_seq_rm clear the flag; the model layer must re-prefill the
     * last token before sampling. */
    if (!c->decode_ready) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_no_logits);
    }

    /* Lazy-init the sampler chain on first use as a greedy fallback,
     * matching the behaviour callers got before configure_sampler/2
     * existed. The model layer should normally call configure_sampler
     * once per request before the first decode; this fallback keeps
     * the cache-only and stub-backed call sites working without
     * touching them. */
    if (!c->smpl) {
        struct llama_sampler *fallback = build_default_greedy_chain();
        if (!fallback) {
            pthread_mutex_unlock(&c->mu);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        c->smpl = fallback;
    }

    /* llama_sampler_sample calls llama_sampler_accept on the chain
     * internally; the chain stays cached, so accept lands on the
     * cached object. */
    llama_token tok = erllama_safe_sampler_sample(c->smpl, c->ctx, -1);
    if (tok < 0) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }

    const struct llama_model *model = erllama_safe_get_model(c->ctx);
    const struct llama_vocab *vocab =
        model ? erllama_safe_model_get_vocab(model) : NULL;
    int eog = vocab ? erllama_safe_vocab_is_eog(vocab, tok) : 0;

    llama_token tok_buf = tok;
    struct llama_batch batch = erllama_safe_batch_get_one(&tok_buf, 1);
    arm_decode(c);
    int rc = erllama_safe_decode(c->ctx, batch);
    if (rc == 0) c->decode_ready = 1;
    else c->decode_ready = 0;
    ERL_NIF_TERM why = atom_ok;
    if (rc != 0) {
        why = (rc == ERLLAMA_DECODE_EXC_SENTINEL) ? atom_exception
                                                  : classify_decode_error(env, c, rc);
    }
    pthread_mutex_unlock(&c->mu);
    if (rc != 0) {
        return enif_make_tuple2(env, atom_error, why);
    }
    ERL_NIF_TERM tag = eog ? enif_make_atom(env, "eog") : atom_ok;
    return enif_make_tuple2(env, tag, enif_make_int(env, tok));
}

/* =========================================================================
 * Embeddings
 * =========================================================================
 *
 * Decodes a token list with the embeddings flag flipped on, then
 * reads the per-sequence pooled vector via llama_get_embeddings_seq.
 * Falls back to llama_get_embeddings (last-token) for models whose
 * pooling_type is NONE. The context must have been opened with
 * embeddings = true at new_context/2 time, otherwise the underlying
 * llama_decode allocates causal-LM logits buffers and the
 * embeddings reads return NULL.
 */
ERL_NIF_TERM nif_embed(ErlNifEnv *env, int argc,
                              const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    llama_token *tokens = NULL;
    int32_t n = 0;
    int rc = read_token_list(env, argv[1], &tokens, &n);
    if (rc != 1) return token_list_error(env, rc);
    if (n == 0) {
        if (tokens) enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_invalid_token);
    }

    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    const struct llama_model *model = erllama_safe_get_model(c->ctx);
    const struct llama_vocab *vocab =
        model ? erllama_safe_model_get_vocab(model) : NULL;
    int32_t n_vocab = vocab ? erllama_safe_vocab_n_tokens(vocab) : 0;
    if (n_vocab <= 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    for (int32_t i = 0; i < n; i++) {
        if (tokens[i] >= n_vocab) {
            pthread_mutex_unlock(&c->mu);
            enif_free(tokens);
            return enif_make_tuple2(env, atom_error, atom_invalid_token);
        }
    }
    int32_t n_embd = erllama_safe_n_embd(model);
    if (n_embd <= 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_embed_failed);
    }
    /* Bounds-check before touching context state. Same SIGSEGV path
     * as nif_prefill: llama_decode walks past the KV slab when
     * n_tokens >= n_ctx, undefined when n_tokens > n_batch. */
    uint32_t n_ctx = erllama_safe_n_ctx(c->ctx);
    uint32_t n_batch = erllama_safe_n_batch(c->ctx);
    if (n_ctx == 0 || n_batch == 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if ((uint32_t) n >= n_ctx) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_context_overflow);
    }
    if ((uint32_t) n > n_batch) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_batch_overflow);
    }

    /* Flip on embeddings for this call; the caller may have left it
     * off for normal causal-lm decode. We do not flip it back here -
     * the next decode_one call would read garbage logits. The model
     * layer is responsible for using a dedicated context for
     * embeddings, or for arranging not to mix modes on the same ctx. */
    if (erllama_safe_set_embeddings(c->ctx, true) != 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }

    struct llama_batch batch = erllama_safe_batch_get_one(tokens, n);
    arm_decode(c);
    int dr = erllama_safe_decode(c->ctx, batch);
    if (dr == ERLLAMA_DECODE_EXC_SENTINEL) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (dr != 0) {
        ERL_NIF_TERM why = classify_decode_error(env, c, dr);
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, why);
    }
    /* The `decode_ready` flag implies "logits are ready for sampling";
     * after an embeddings decode the logits buffer is repurposed and a
     * follow-on decode_one would crash. Force it off so the model
     * layer must explicitly re-prefill before sampling. */
    c->decode_ready = 0;

    /* Try the pooled vector first; fall back to last-token. */
    float *embd = erllama_safe_get_embeddings_seq(c->ctx, 0);
    if (!embd) {
        embd = erllama_safe_get_embeddings(c->ctx);
    }
    if (!embd) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_embed_failed);
    }

    /* Copy the floats out of the context-owned buffer before unlocking. */
    double *vec = enif_alloc(sizeof(double) * (size_t) n_embd);
    if (!vec) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    for (int32_t i = 0; i < n_embd; i++) vec[i] = (double) embd[i];
    pthread_mutex_unlock(&c->mu);
    enif_free(tokens);

    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int32_t i = n_embd - 1; i >= 0; i--) {
        list = enif_make_list_cell(env, enif_make_double(env, vec[i]), list);
    }
    enif_free(vec);
    return enif_make_tuple2(env, atom_ok, list);
}

/* =========================================================================
 * Speculative-decoding forward + per-position argmax
 *
 * nif_forward_with_argmax is the per-position argmax primitive used
 * by erllama:verify/4. Builds a custom batch with logits[i]=1 on
 * every position so llama_get_logits_ith(c, i) is valid for every
 * i, decodes, and returns the argmax token at each position.
 *
 * Sampler state is intentionally bypassed: we go straight through
 * llama_get_logits_ith and a manual argmax loop, so the cached
 * c->smpl chain (and any seeded RNG state on it) is untouched.
 *
 * The caller (the model gen_statem's verify handler) is responsible
 * for snapshot+restore around this call: the KV cache is mutated
 * with the new tokens at positions [start_pos, start_pos+n) on
 * seq_id=0, and the per-context logits buffer reflects the end of
 * the supplied batch on return. Pre-call snapshot of (KvLen,
 * decode_ready, last token) and post-call kv_seq_rm + re-prefill
 * brings the context back to its pre-call observable state.
 * ========================================================================= */
ERL_NIF_TERM nif_forward_with_argmax(ErlNifEnv *env, int argc,
                                            const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    llama_token *tokens = NULL;
    int32_t n = 0;
    int rc = read_token_list(env, argv[1], &tokens, &n);
    if (rc != 1) return token_list_error(env, rc);
    if (n == 0) {
        if (tokens) enif_free(tokens);
        return enif_make_tuple2(env, atom_ok, enif_make_list(env, 0));
    }

    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    const struct llama_model *model = erllama_safe_get_model(c->ctx);
    const struct llama_vocab *vocab =
        model ? erllama_safe_model_get_vocab(model) : NULL;
    int32_t n_vocab = vocab ? erllama_safe_vocab_n_tokens(vocab) : 0;
    if (n_vocab <= 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    for (int32_t i = 0; i < n; i++) {
        if (tokens[i] >= n_vocab) {
            pthread_mutex_unlock(&c->mu);
            enif_free(tokens);
            return enif_make_tuple2(env, atom_error, atom_invalid_token);
        }
    }
    uint32_t n_ctx = erllama_safe_n_ctx(c->ctx);
    uint32_t n_batch = erllama_safe_n_batch(c->ctx);
    if (n_ctx == 0 || n_batch == 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if ((uint32_t) n >= n_ctx) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_context_overflow);
    }
    if ((uint32_t) n > n_batch) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_batch_overflow);
    }
    long pos_max = erllama_safe_memory_seq_pos_max(c->ctx, 0);
    if (pos_max == -2) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    long start_pos = (pos_max < 0) ? 0 : (pos_max + 1);

    int32_t *out = enif_alloc(sizeof(int32_t) * (size_t) n);
    if (!out) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    int frc = erllama_safe_forward_with_argmax(
        c->ctx, tokens, n, n_vocab, start_pos, out
    );
    if (frc == 0) c->decode_ready = 1;
    pthread_mutex_unlock(&c->mu);
    enif_free(tokens);
    if (frc != 0) {
        enif_free(out);
        ERL_NIF_TERM why = (frc == -2) ? atom_oom :
                           (frc == -3) ? atom_exception :
                           atom_decode_failed;
        return enif_make_tuple2(env, atom_error, why);
    }

    /* EOG mapping happens here so the Erlang side gets a uniform
     * `[non_neg_integer() | eos]` shape, no extra NIF round-trips
     * for vocab_is_eog. */
    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        ERL_NIF_TERM tok_term;
        if (erllama_safe_vocab_is_eog(vocab, out[i])) {
            tok_term = atom_eos;
        } else {
            tok_term = enif_make_int(env, out[i]);
        }
        list = enif_make_list_cell(env, tok_term, list);
    }
    enif_free(out);
    return enif_make_tuple2(env, atom_ok, list);
}
