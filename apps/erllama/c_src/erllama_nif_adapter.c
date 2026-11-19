/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */
/* LoRA adapters: load/free and applying {adapter, scale} sets to
 * a context. */

#include "erllama_nif_int.h"
#include "erllama_safe.h"

#include <math.h>
#include <string.h>

/* =========================================================================
 * LoRA adapters
 * ========================================================================= */

ERL_NIF_TERM nif_adapter_load(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    char path[4097];
    if (!copy_path(env, argv[1], path, sizeof(path))) {
        return enif_make_badarg(env);
    }

    /* Mirror nif_new_context: also refuse to attach a new adapter
     * once free_model/1 has flagged the model for deferred release.
     * Otherwise a new wrapper could resurrect an outgoing model. */
    if (!erllama_lock_model_live(m)) {
        return erllama_error(env, atom_released);
    }
    struct llama_adapter_lora *adapter =
        erllama_safe_adapter_lora_init(m->model, path);
    if (!adapter) {
        pthread_mutex_unlock(&m->mu);
        return erllama_error(env, atom_load_failed);
    }
    erllama_adapter_t *res =
        enif_alloc_resource(ADAPTER_RT, sizeof(*res));
    if (!res) {
        erllama_safe_adapter_lora_free(adapter);
        pthread_mutex_unlock(&m->mu);
        return erllama_error(env, atom_oom);
    }
    memset(res, 0, sizeof(*res));
    if (pthread_mutex_init(&res->mu, NULL) != 0) {
        enif_release_resource(res);
        erllama_safe_adapter_lora_free(adapter);
        pthread_mutex_unlock(&m->mu);
        return erllama_error(env, atom_oom);
    }
    res->mu_inited = 1;
    res->adapter = adapter;
    res->model_res = m;
    m->active_adapters++;
    enif_keep_resource(m);
    pthread_mutex_unlock(&m->mu);

    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return enif_make_tuple2(env, atom_ok, term);
}

ERL_NIF_TERM nif_adapter_free(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_adapter_t *a;
    if (!enif_get_resource(env, argv[0], ADAPTER_RT, (void **) &a)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&a->mu);
    if (!a->adapter) {
        pthread_mutex_unlock(&a->mu);
        return erllama_error(env, atom_released);
    }
    erllama_safe_adapter_lora_free(a->adapter);
    a->adapter = NULL;
    pthread_mutex_unlock(&a->mu);
    return atom_ok;
}

/* Install a set of adapters with scales on a context. Takes a list of
 * {AdapterRes, Scale} pairs; an empty list detaches everything.
 * The model layer is responsible for tracking the current attachment
 * set; this NIF just plumbs through to llama_set_adapters_lora.
 *
 * Concurrency: every unique adapter wrapper's mu is held for the
 * full duration of the llama call. nif_adapter_free, which also
 * takes a->mu, is therefore blocked from racing the read of
 * a->adapter against its use in the native call. To avoid AB-BA
 * between two concurrent set_adapters callers passing overlapping
 * adapter sets in different orders, locks are taken in pointer
 * order (qsort by wrapper address). The user-supplied list order
 * is preserved in the arrays passed to llama via orig_idx. */
typedef struct {
    erllama_adapter_t *w;
    float scale;
    unsigned orig_idx;
} adapter_entry_t;

static int adapter_entry_cmp(const void *a, const void *b) {
    erllama_adapter_t *aw = ((const adapter_entry_t *) a)->w;
    erllama_adapter_t *bw = ((const adapter_entry_t *) b)->w;
    if (aw < bw) return -1;
    if (aw > bw) return 1;
    return 0;
}

ERL_NIF_TERM nif_set_adapters(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    ERL_NIF_TERM list = argv[1];
    unsigned n;
    if (!enif_get_list_length(env, list, &n)) {
        return enif_make_badarg(env);
    }

    struct llama_adapter_lora **adapters = NULL;
    float *scales = NULL;
    adapter_entry_t *entries = NULL;
    if (n > 0) {
        adapters = enif_alloc(sizeof(*adapters) * n);
        scales = enif_alloc(sizeof(*scales) * n);
        entries = enif_alloc(sizeof(*entries) * n);
        if (!adapters || !scales || !entries) {
            if (adapters) enif_free(adapters);
            if (scales) enif_free(scales);
            if (entries) enif_free(entries);
            return erllama_error(env, atom_oom);
        }
    }

    /* Resolve the list in user order. No locks taken yet: we just
     * gather the wrapper pointers and scales. */
    ERL_NIF_TERM head, tail = list;
    unsigned i = 0;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        int arity;
        const ERL_NIF_TERM *pair;
        if (!enif_get_tuple(env, head, &arity, &pair) || arity != 2) {
            goto badarg;
        }
        erllama_adapter_t *a;
        if (!enif_get_resource(env, pair[0], ADAPTER_RT, (void **) &a)) {
            goto badarg;
        }
        double scale;
        if (!enif_get_double(env, pair[1], &scale)) {
            long ll;
            if (enif_get_long(env, pair[1], &ll)) {
                scale = (double) ll;
            } else {
                goto badarg;
            }
        }
        /* entries was allocated in the `if (n > 0)` block above;
         * the while loop only enters when the list has elements,
         * which implies n > 0. The analyzer can't tie those two
         * facts together. */
        /* NOLINTNEXTLINE(clang-analyzer-core.NullDereference) */
        entries[i].w = a;
        entries[i].scale = (float) scale;
        entries[i].orig_idx = i;
        i++;
    }

    /* Sort by wrapper pointer so locks always go in a consistent
     * order across concurrent set_adapters callers. Same wrapper
     * appearing twice (caller error or duplicate-scale convention)
     * collapses to one lock acquisition; the user-supplied
     * adapters[]/scales[] still receive both entries via orig_idx. */
    if (n > 1) qsort(entries, n, sizeof(*entries), adapter_entry_cmp);

    /* Lock each unique wrapper in sorted order. On a released
     * adapter, unlock everything held so far and bail out. */
    unsigned k = 0;
    for (k = 0; k < n; k++) {
        if (k > 0 && entries[k].w == entries[k - 1].w) continue;
        pthread_mutex_lock(&entries[k].w->mu);
        if (!entries[k].w->adapter) {
            for (unsigned j = 0; j <= k; j++) {
                if (j > 0 && entries[j].w == entries[j - 1].w) continue;
                pthread_mutex_unlock(&entries[j].w->mu);
            }
            if (adapters) enif_free(adapters);
            if (scales) enif_free(scales);
            if (entries) enif_free(entries);
            return erllama_error(env, atom_released);
        }
    }

    /* Build native arrays in the user's original order. All
     * a->adapter reads happen under the corresponding a->mu held
     * above, so a concurrent nif_adapter_free cannot null any of
     * these pointers between read and use. */
    for (k = 0; k < n; k++) {
        adapters[entries[k].orig_idx] = entries[k].w->adapter;
        scales[entries[k].orig_idx] = entries[k].scale;
    }

    pthread_mutex_lock(&c->mu);
    int rc;
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        for (k = 0; k < n; k++) {
            if (k > 0 && entries[k].w == entries[k - 1].w) continue;
            pthread_mutex_unlock(&entries[k].w->mu);
        }
        if (adapters) enif_free(adapters);
        if (scales) enif_free(scales);
        if (entries) enif_free(entries);
        return erllama_error(env, atom_released);
    }
    rc = erllama_safe_set_adapters_lora(c->ctx, adapters, n, scales);
    pthread_mutex_unlock(&c->mu);

    for (k = 0; k < n; k++) {
        if (k > 0 && entries[k].w == entries[k - 1].w) continue;
        pthread_mutex_unlock(&entries[k].w->mu);
    }

    if (adapters) enif_free(adapters);
    if (scales) enif_free(scales);
    if (entries) enif_free(entries);

    if (rc != 0) {
        return erllama_error(env, atom_exception);
    }
    return atom_ok;

badarg:
    if (adapters) enif_free(adapters);
    if (scales) enif_free(scales);
    if (entries) enif_free(entries);
    return enif_make_badarg(env);
}
