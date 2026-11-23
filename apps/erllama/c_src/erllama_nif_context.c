/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */
/* Context lifecycle: create/free, the per-step abort deadline
 * (arm_decode / classify_decode_error) and external aborts. */

#include "erllama_nif_int.h"
#include "erllama_safe.h"

#include <math.h>
#include <string.h>

/* =========================================================================
 * Graph input diagnostics (`diag_inputs' context option)
 *
 * A ggml eval callback that, while armed, records the name, type and
 * shape of every input tensor of the nodes it is asked about (the
 * attention masks give the KV extent a decode attends over). It never
 * asks for a node's data, so the scheduler still computes each split
 * in one piece. Armed and read by nif_ctx_diag; used by benchmarks.
 * ========================================================================= */

#define ERLLAMA_DIAG_MAX 128

typedef struct erllama_diag_s {
    int on;
    int n;
    struct {
        char name[64];
        int type;
        int64_t ne[4];
    } rec[ERLLAMA_DIAG_MAX];
} erllama_diag_t;

static void diag_record(erllama_diag_t *d, const struct ggml_tensor *s) {
    for (int i = 0; i < d->n; i++) {
        if (d->rec[i].type == (int) s->type && strncmp(d->rec[i].name, s->name, sizeof(d->rec[i].name)) == 0 &&
            memcmp(d->rec[i].ne, s->ne, sizeof(d->rec[i].ne)) == 0) {
            return;
        }
    }
    if (d->n >= ERLLAMA_DIAG_MAX) {
        return;
    }
    (void) snprintf(d->rec[d->n].name, sizeof(d->rec[d->n].name), "%s", s->name);
    d->rec[d->n].type = (int) s->type;
    memcpy(d->rec[d->n].ne, s->ne, sizeof(d->rec[d->n].ne));
    d->n++;
}

static bool erllama_diag_cb(struct ggml_tensor *t, bool ask, void *user_data) {
    erllama_diag_t *d = (erllama_diag_t *) user_data;
    if (!ask) {
        return true;
    }
    if (d->on) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor *s = t->src[i];
            if (s && (s->flags & GGML_TENSOR_FLAG_INPUT)) {
                diag_record(d, s);
            }
        }
    }
    return false;
}

void erllama_diag_free(struct erllama_diag_s *d) {
    enif_free(d);
}

/* nif_ctx_diag(Ctx, arm | read | perf):
 *   arm  -> ok: clear the records and record during the next decodes;
 *   read -> [{Name, Type, [Ne0, Ne1, Ne2, Ne3]}], and disarm;
 *   perf -> #{n_reused, n_reallocs_same, n_reallocs_diff, n_eval, n_p_eval}
 *           (llama_perf_context; works without `diag_inputs'). */
ERL_NIF_TERM nif_ctx_diag(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    char op[8];
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c) ||
        !enif_get_atom(env, argv[1], op, sizeof(op), ERL_NIF_LATIN1)) {
        return enif_make_badarg(env);
    }
    if (!erllama_lock_ctx(c)) {
        return erllama_error(env, atom_released);
    }
    ERL_NIF_TERM out;
    if (strcmp(op, "perf") == 0) {
        struct llama_perf_context_data p = llama_perf_context(c->ctx);
        out = enif_make_new_map(env);
        enif_make_map_put(env, out, enif_make_atom(env, "n_reused"), enif_make_int(env, p.n_reused), &out);
        enif_make_map_put(env, out, enif_make_atom(env, "n_reallocs_same"), enif_make_int(env, p.n_reallocs_same), &out);
        enif_make_map_put(env, out, enif_make_atom(env, "n_reallocs_diff"), enif_make_int(env, p.n_reallocs_diff), &out);
        enif_make_map_put(env, out, enif_make_atom(env, "n_eval"), enif_make_int(env, p.n_eval), &out);
        enif_make_map_put(env, out, enif_make_atom(env, "n_p_eval"), enif_make_int(env, p.n_p_eval), &out);
    } else if (!c->diag) {
        out = erllama_error(env, atom_not_supported);
    } else if (strcmp(op, "arm") == 0) {
        c->diag->n = 0;
        c->diag->on = 1;
        out = atom_ok;
    } else if (strcmp(op, "read") == 0) {
        c->diag->on = 0;
        out = enif_make_list(env, 0);
        for (int i = c->diag->n - 1; i >= 0; i--) {
            ERL_NIF_TERM ne = enif_make_list4(env, enif_make_int64(env, c->diag->rec[i].ne[0]),
                                              enif_make_int64(env, c->diag->rec[i].ne[1]),
                                              enif_make_int64(env, c->diag->rec[i].ne[2]),
                                              enif_make_int64(env, c->diag->rec[i].ne[3]));
            ERL_NIF_TERM name;
            size_t len = strnlen(c->diag->rec[i].name, sizeof(c->diag->rec[i].name));
            memcpy(enif_make_new_binary(env, len, &name), c->diag->rec[i].name, len);
            out = enif_make_list_cell(
                env,
                enif_make_tuple3(env, name, enif_make_string(env, ggml_type_name((enum ggml_type) c->diag->rec[i].type), ERL_NIF_LATIN1), ne),
                out);
        }
    } else {
        out = enif_make_badarg(env);
    }
    pthread_mutex_unlock(&c->mu);
    return out;
}

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
    if (!erllama_parse_cparams(env, argv[1], &params)) {
        return enif_make_badarg(env);
    }
    erllama_diag_t *diag = NULL;
    {
        int b;
        if (get_map_bool(env, argv[1], "diag_inputs", &b) && b) {
            diag = enif_alloc(sizeof(*diag));
            if (!diag) {
                return erllama_error(env, atom_oom);
            }
            memset(diag, 0, sizeof(*diag));
            params.cb_eval = erllama_diag_cb;
            params.cb_eval_user_data = diag;
        }
    }
    /* `tp_sink => Pid': only for a model split over a tensor-parallel
     * rank (see erllama_ctx_decode). */
    ErlNifPid tp_sink;
    int tp_sink_set = 0;
    {
        ERL_NIF_TERM v;
        if (enif_get_map_value(env, argv[1], enif_make_atom(env, "tp_sink"), &v)) {
            if (!enif_get_local_pid(env, v, &tp_sink) || !m->tp_rank) {
                if (diag) enif_free(diag);
                return enif_make_badarg(env);
            }
            tp_sink_set = 1;
        }
    }

    /* The release_pending refusal: if free_model/1 has been called
     * and is waiting for the last context to drop, do not let a new
     * caller resurrect the model by attaching another context. The
     * {ok, deferred} return is a release contract: no new contexts
     * allowed past that point. */
    if (!erllama_lock_model_live(m)) {
        if (diag) enif_free(diag);
        return erllama_error(env, atom_released);
    }
    struct llama_context *ctx = erllama_safe_init_from_model(m->model, params);
    if (!ctx) {
        pthread_mutex_unlock(&m->mu);
        if (diag) enif_free(diag);
        return erllama_error(env, atom_context_failed);
    }
    erllama_context_t *res = enif_alloc_resource(CTX_RT, sizeof(*res));
    if (!res) {
        (void) erllama_safe_free(ctx);
        pthread_mutex_unlock(&m->mu);
        if (diag) enif_free(diag);
        return erllama_error(env, atom_oom);
    }
    memset(res, 0, sizeof(*res));
    /* Owned by the resource from here: ctx_dtor frees it, after the
     * context (the callback's only user). */
    res->diag = diag;
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
    if (tp_sink_set) {
        res->tp_sink = tp_sink;
        res->tp_sink_set = 1;
    }
    /* Install the per-step interrupt. data = res stays valid for the
     * context's lifetime (the context is freed before the resource in
     * ctx_dtor / free_context) and the callback only runs during a
     * llama_decode on this very context.
     * Not on a split model: a decode interrupted on one rank and not on
     * the others leaves them waiting in a collective it never reaches.
     * A cancel there takes effect between steps, when every rank has
     * finished the step. */
    if (!m->tp_rank) {
        erllama_safe_set_abort_callback(ctx, erllama_decode_abort_cb, res);
    }
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
