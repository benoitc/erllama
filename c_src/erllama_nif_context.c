/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */
/* Context lifecycle: create/free, the per-step abort deadline
 * (arm_decode / classify_decode_error) and external aborts. */

#include "erllama_nif_int.h"
#include "erllama_safe.h"

#include <math.h>
#include <string.h>

static const erllama_atom_enum_pair_t FLASH_ATTN_TABLE[] = {
    /* booleans-as-atoms map to the matching enum tristate. `auto`
     * lets llama.cpp pick based on the build / model. */
    {"true",  LLAMA_FLASH_ATTN_TYPE_ENABLED},
    {"false", LLAMA_FLASH_ATTN_TYPE_DISABLED},
    {"auto",  LLAMA_FLASH_ATTN_TYPE_AUTO},
};

static const erllama_atom_enum_pair_t GGML_TYPE_TABLE[] = {
    {"f16",  GGML_TYPE_F16},
    {"f32",  GGML_TYPE_F32},
    {"bf16", GGML_TYPE_BF16},
    {"q4_0", GGML_TYPE_Q4_0},
    {"q5_0", GGML_TYPE_Q5_0},
    {"q5_1", GGML_TYPE_Q5_1},
    {"q8_0", GGML_TYPE_Q8_0},
};

/* =========================================================================
 * Context
 * ========================================================================= */

/* Abort callback installed on every llama_context. ggml calls it
 * periodically during compute, possibly from worker threads, so it
 * reads only atomics and never locks. Returns true to abort the
 * in-flight llama_decode. */
static bool erllama_decode_abort_cb(void *data) {
    erllama_context_t *c = (erllama_context_t *) data;
    if (atomic_load_explicit(&c->abort_flag, memory_order_relaxed)) {
        return true;
    }
    uint_least64_t dl =
        atomic_load_explicit(&c->decode_deadline_ns, memory_order_relaxed);
    if (dl == 0) {
        return false;
    }
    return (uint_least64_t) enif_monotonic_time(ERL_NIF_NSEC) >= dl;
}

/* Arm the interrupt before a llama_decode: clear any stale abort and
 * set the per-step deadline from the configured budget. Called by the
 * NIF decode thread while holding c->mu. */
void arm_decode(erllama_context_t *c) {
    atomic_store_explicit(&c->abort_flag, 0, memory_order_relaxed);
    if (c->decode_budget_ns == 0) {
        atomic_store_explicit(&c->decode_deadline_ns, 0, memory_order_relaxed);
    } else {
        uint_least64_t now = (uint_least64_t) enif_monotonic_time(ERL_NIF_NSEC);
        atomic_store_explicit(&c->decode_deadline_ns,
                              now + c->decode_budget_ns, memory_order_relaxed);
    }
}

/* Classify a non-zero, non-exception llama_decode return after a
 * decode armed by arm_decode/1: an interrupt we requested maps to a
 * structured timeout/aborted reason; anything else is decode_failed. */
ERL_NIF_TERM classify_decode_error(ErlNifEnv *env, erllama_context_t *c,
                                          int rc) {
    if (atomic_load_explicit(&c->abort_flag, memory_order_relaxed)) {
        return atom_decode_aborted;
    }
    uint_least64_t dl =
        atomic_load_explicit(&c->decode_deadline_ns, memory_order_relaxed);
    if (dl != 0 &&
        (uint_least64_t) enif_monotonic_time(ERL_NIF_NSEC) >= dl) {
        return atom_decode_timeout;
    }
    return enif_make_tuple2(env, atom_decode_failed, enif_make_int(env, rc));
}

/* Set the abort flag on a context from outside the decode thread.
 * Deliberately does NOT take c->mu: a wedged decode holds the mutex,
 * and the whole point is to interrupt it. Registered as a regular
 * (non-dirty) NIF: it is a single atomic store. */
ERL_NIF_TERM nif_request_abort(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    if (!c->ctx) {
        return erllama_error(env, atom_released);
    }
    atomic_store_explicit(&c->abort_flag, 1, memory_order_relaxed);
    return atom_ok;
}

ERL_NIF_TERM nif_new_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    if (!enif_is_map(env, argv[1])) {
        return enif_make_badarg(env);
    }

    struct llama_context_params params = llama_context_default_params();

    /* Per-step decode budget. The abort callback trips a wedged
     * llama_decode after this many ms so the call always returns.
     * Default 30 s (a legit step is sub-second on a loaded model);
     * 0 disables the deadline. */
    uint64_t decode_budget_ns = (uint64_t) 30000 * 1000000ULL;

    unsigned int u;
    if (get_map_uint(env, argv[1], "decode_budget_ms", &u)) {
        decode_budget_ns = (uint64_t) u * 1000000ULL;
    }
    if (get_map_uint(env, argv[1], "n_ctx", &u)) params.n_ctx = (uint32_t) u;
    if (get_map_uint(env, argv[1], "n_batch", &u)) params.n_batch = (uint32_t) u;
    if (get_map_uint(env, argv[1], "n_ubatch", &u)) params.n_ubatch = (uint32_t) u;
    if (get_map_uint(env, argv[1], "n_seq_max", &u)) {
        if (u > ERLLAMA_N_SEQ_MAX_CAP) {
            return enif_make_badarg(env);
        }
        params.n_seq_max = (uint32_t) u;
    }
    /* Recurrent-state rollback snapshots per seq (recurrent / hybrid
     * archs only; 0 = no rollback). With >= 1, RS-capable archs accept
     * the 1-token partial seq_rm the warm-restore primer issues. */
    if (get_map_uint(env, argv[1], "n_rs_seq", &u)) {
        params.n_rs_seq = (uint32_t) u;
    }
    int32_t i32;
    if (get_map_int31(env, argv[1], "n_threads", &i32)) params.n_threads = i32;
    if (get_map_int31(env, argv[1], "n_threads_batch", &i32)) {
        params.n_threads_batch = i32;
    }
    int b;
    if (get_map_bool(env, argv[1], "embeddings", &b)) params.embeddings = b ? true : false;
    if (get_map_bool(env, argv[1], "offload_kqv", &b)) params.offload_kqv = b ? true : false;
    /* Unified KV cache: a single sequence may use the full n_ctx and up to
     * n_seq_max sequences share that buffer, instead of splitting n_ctx
     * into n_ctx/n_seq_max per sequence (the default). Lets concurrency
     * (n_seq_max > 1) coexist with large per-request context. */
    if (get_map_bool(env, argv[1], "kv_unified", &b)) params.kv_unified = b ? true : false;

    int enum_v;
    int fa_rc = get_map_atom_enum(
        env, argv[1], "flash_attn",
        FLASH_ATTN_TABLE, sizeof(FLASH_ATTN_TABLE) / sizeof(FLASH_ATTN_TABLE[0]),
        &enum_v
    );
    if (fa_rc < 0) return enif_make_badarg(env);
    if (fa_rc > 0) params.flash_attn_type = (enum llama_flash_attn_type) enum_v;

    int tk_rc = get_map_atom_enum(
        env, argv[1], "type_k",
        GGML_TYPE_TABLE, sizeof(GGML_TYPE_TABLE) / sizeof(GGML_TYPE_TABLE[0]),
        &enum_v
    );
    if (tk_rc < 0) return enif_make_badarg(env);
    if (tk_rc > 0) params.type_k = (enum ggml_type) enum_v;

    int tv_rc = get_map_atom_enum(
        env, argv[1], "type_v",
        GGML_TYPE_TABLE, sizeof(GGML_TYPE_TABLE) / sizeof(GGML_TYPE_TABLE[0]),
        &enum_v
    );
    if (tv_rc < 0) return enif_make_badarg(env);
    if (tv_rc > 0) params.type_v = (enum ggml_type) enum_v;

    /* The release_pending refusal: if free_model/1 has been called
     * and is waiting for the last context to drop, do not let a new
     * caller resurrect the model by attaching another context. The
     * {ok, deferred} return is a release contract: no new contexts
     * allowed past that point. */
    if (!erllama_lock_model_live(m)) {
        return erllama_error(env, atom_released);
    }
    struct llama_context *ctx = erllama_safe_init_from_model(m->model, params);
    if (!ctx) {
        pthread_mutex_unlock(&m->mu);
        return erllama_error(env, atom_context_failed);
    }
    erllama_context_t *res = enif_alloc_resource(CTX_RT, sizeof(*res));
    if (!res) {
        (void) erllama_safe_free(ctx);
        pthread_mutex_unlock(&m->mu);
        return erllama_error(env, atom_oom);
    }
    memset(res, 0, sizeof(*res));
    /* per_seq[].last_logits_idx must start at -1 so the first
     * nif_step decode tick refuses to sample a seq that has no
     * prefill behind it (memset of int32_t -> 0 would be a valid
     * row index, which would silently read garbage logits). */
    for (size_t i = 0; i < (size_t) ERLLAMA_N_SEQ_MAX_CAP; i++) {
        res->per_seq[i].last_logits_idx = -1;
        res->per_seq[i].next_pos = 0;
    }
    if (pthread_mutex_init(&res->mu, NULL) != 0) {
        enif_release_resource(res);
        (void) erllama_safe_free(ctx);
        pthread_mutex_unlock(&m->mu);
        return erllama_error(env, atom_oom);
    }
    res->mu_inited = 1;
    res->ctx = ctx;
    res->model_res = m;
    res->decode_budget_ns = decode_budget_ns;
    /* Install the per-step interrupt. data = res stays valid for the
     * context's lifetime (the context is freed before the resource in
     * ctx_dtor / free_context) and the callback only runs during a
     * llama_decode on this very context. */
    erllama_safe_set_abort_callback(ctx, erllama_decode_abort_cb, res);
    m->active_contexts++;
    enif_keep_resource(m);
    pthread_mutex_unlock(&m->mu);

    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return enif_make_tuple2(env, atom_ok, term);
}

ERL_NIF_TERM nif_free_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    if (!erllama_lock_ctx(c)) {
        return erllama_error(env, atom_released);
    }
    /* Free under the lock so a concurrent reader cannot observe the
     * pointer mid-teardown. On exception we still NULL the pointer
     * to avoid a double-free path through the destructor; the native
     * object is leaked rather than risking UB. */
    if (c->smpl) {
        (void) erllama_safe_sampler_free(c->smpl);
        c->smpl = NULL;
    }
    int free_rc = erllama_safe_free(c->ctx);
    c->ctx = NULL;
    c->decode_ready = 0;
    erllama_model_t *m = c->model_res;
    c->model_res = NULL;
    pthread_mutex_unlock(&c->mu);
    if (m) {
        context_drops_model(m);
        enif_release_resource(m);
    }
    if (free_rc != 0) {
        return erllama_error(env, atom_exception);
    }
    return atom_ok;
}
