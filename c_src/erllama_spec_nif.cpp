/* C++ NIF wrapper around llama.cpp's draft-model-free ngram
 * speculation (common/speculative.h, COMMON_SPECULATIVE_TYPE_NGRAM_MOD).
 *
 * Entry points exposed to Erlang via the erllama_nif .so:
 *
 *   nif_spec_new(#{n_match, n_max, n_min, n_seq})
 *     -> {ok, SpecRef} | {error, Reason}
 *   nif_spec_begin(SpecRef, SeqId, PromptTokens) -> ok | {error, _}
 *   nif_spec_draft(SpecRef, SeqId, IdLast, DeltaTokens)
 *     -> {ok, [DraftTok]} | {error, _}
 *   nif_spec_accept(SpecRef, SeqId, NAccepted) -> ok | {error, _}
 *   nif_spec_free(SpecRef) -> ok | {error, released}
 *
 * The shim owns the per-seq token vectors the upstream API points
 * into: `prompts[seq]` always holds committed[0..K-2] and the caller
 * passes committed[K-1] as IdLast, shipping only the delta since the
 * previous call. The ngram-mod method needs no llama_context, model,
 * or sampler; its 16 MiB hash table is shared across sequences and
 * survives across requests (a feature: re-sent transcripts draft
 * immediately).
 *
 * GGML_ASSERT paths in upstream that would abort the process are
 * pre-checked here: seq bounds before get_draft_params, the result
 * vector cleared before every draft, and accept(N > 0) rejected
 * unless the shim's own record of the last draft length covers N.
 *
 * All calls for one speculator come from a single model gen_statem
 * process; the mutex is defensive, mirroring the other resources. */

#include "erllama_spec_nif.h"
#include "erllama_nif_util.h"

#include "speculative.h"

#include <pthread.h>

#include <exception>
#include <new>
#include <vector>

/* ============================================================== */
/* Resource                                                       */
/* ============================================================== */

static ErlNifResourceType *SPEC_RT = nullptr;

namespace {

struct spec_holder {
    pthread_mutex_t mu;
    int mu_inited;
    common_speculative *spec; /* nullptr after explicit free */
    uint32_t n_seq;
    /* Per-seq committed[0..K-2]; dparams.prompt points here. */
    std::vector<llama_tokens> prompts;
    /* Per-seq reusable draft output buffers. */
    std::vector<llama_tokens> results;
    /* Per-seq length of the last non-consumed draft; guards the
     * upstream accept() assertion. */
    std::vector<uint32_t> last_draft_len;
};

void spec_dtor(ErlNifEnv *, void *obj) {
    auto *h = static_cast<spec_holder *>(obj);
    if (h->spec) {
        common_speculative_free(h->spec);
        h->spec = nullptr;
    }
    if (h->mu_inited) {
        pthread_mutex_destroy(&h->mu);
        h->mu_inited = 0;
    }
    h->~spec_holder();
}

ERL_NIF_TERM mk_atom(ErlNifEnv *env, const char *name) {
    return enif_make_atom(env, name);
}

/* Read a list of non-negative token ids into `out`. Returns true on
 * success; false on a malformed list. */
bool read_tokens(ErlNifEnv *env, ERL_NIF_TERM list,
                 std::vector<llama_token> &out) {
    unsigned int n;
    if (!enif_get_list_length(env, list, &n)) {
        return false;
    }
    out.reserve(out.size() + n);
    ERL_NIF_TERM head;
    ERL_NIF_TERM tail = list;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        int v;
        if (!enif_get_int(env, head, &v) || v < 0) {
            return false;
        }
        out.push_back((llama_token) v);
    }
    return true;
}

/* Lock h->mu; on a freed speculator unlock and return false. */
bool spec_lock_live(spec_holder *h) {
    pthread_mutex_lock(&h->mu);
    if (!h->spec) {
        pthread_mutex_unlock(&h->mu);
        return false;
    }
    return true;
}

} /* anonymous namespace */

/* ============================================================== */
/* Load                                                           */
/* ============================================================== */

extern "C" int spec_nif_load(ErlNifEnv *env) {
    SPEC_RT = enif_open_resource_type(
        env, nullptr, "erllama_spec", spec_dtor, ERL_NIF_RT_CREATE, nullptr);
    return SPEC_RT ? 0 : -1;
}

/* ============================================================== */
/* nif_spec_new                                                   */
/* ============================================================== */

extern "C" ERL_NIF_TERM nif_spec_new(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
    (void) argc;
    if (!enif_is_map(env, argv[0])) {
        return enif_make_badarg(env);
    }
    /* Defaults are upstream's (common.h common_params_speculative_
     * ngram_mod): n_match 24, n_max 64, n_min 48. */
    common_params_speculative params;
    params.types = {COMMON_SPECULATIVE_TYPE_NGRAM_MOD};

    int32_t i32;
    if (get_map_int31(env, argv[0], "n_match", &i32)) {
        if (i32 < 1) return enif_make_badarg(env);
        params.ngram_mod.n_match = i32;
    }
    if (get_map_int31(env, argv[0], "n_max", &i32)) {
        if (i32 < 1) return enif_make_badarg(env);
        params.ngram_mod.n_max = i32;
    }
    if (get_map_int31(env, argv[0], "n_min", &i32)) {
        if (i32 < 0) return enif_make_badarg(env);
        params.ngram_mod.n_min = i32;
    }
    unsigned int n_seq = 1;
    if (get_map_uint(env, argv[0], "n_seq", &n_seq)) {
        /* Mirror the context's ERLLAMA_N_SEQ_MAX_CAP. */
        if (n_seq < 1 || n_seq > 256) {
            return enif_make_badarg(env);
        }
    }

    try {
        common_speculative *spec = common_speculative_init(params, n_seq);
        if (!spec) {
            return erllama_error(env, mk_atom(env, "spec_init_failed"));
        }
        void *res = enif_alloc_resource(SPEC_RT, sizeof(spec_holder));
        if (!res) {
            common_speculative_free(spec);
            return erllama_error(env, mk_atom(env, "oom"));
        }
        auto *h = new (res) spec_holder{};
        if (pthread_mutex_init(&h->mu, nullptr) != 0) {
            h->~spec_holder();
            common_speculative_free(spec);
            enif_release_resource(res);
            return erllama_error(env, mk_atom(env, "oom"));
        }
        h->mu_inited = 1;
        h->spec = spec;
        h->n_seq = n_seq;
        h->prompts.resize(n_seq);
        h->results.resize(n_seq);
        h->last_draft_len.assign(n_seq, 0);

        ERL_NIF_TERM term = enif_make_resource(env, res);
        enif_release_resource(res);
        return enif_make_tuple2(env, mk_atom(env, "ok"), term);
    } catch (const std::exception &) {
        return erllama_error(env, mk_atom(env, "exception"));
    } catch (...) {
        return erllama_error(env, mk_atom(env, "exception"));
    }
}

/* ============================================================== */
/* nif_spec_begin                                                 */
/* ============================================================== */

extern "C" ERL_NIF_TERM nif_spec_begin(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
    (void) argc;
    spec_holder *h = nullptr;
    if (!enif_get_resource(env, argv[0], SPEC_RT, (void **) &h)) {
        return enif_make_badarg(env);
    }
    int seq;
    if (!enif_get_int(env, argv[1], &seq) || seq < 0) {
        return enif_make_badarg(env);
    }
    if (!spec_lock_live(h)) {
        return erllama_error(env, mk_atom(env, "released"));
    }
    if ((uint32_t) seq >= h->n_seq) {
        pthread_mutex_unlock(&h->mu);
        return enif_make_badarg(env);
    }
    ERL_NIF_TERM ret = mk_atom(env, "ok");
    try {
        h->prompts[seq].clear();
        if (!read_tokens(env, argv[2], h->prompts[seq])) {
            pthread_mutex_unlock(&h->mu);
            return enif_make_badarg(env);
        }
        h->last_draft_len[seq] = 0;
        common_speculative_begin(h->spec, (llama_seq_id) seq,
                                 h->prompts[seq]);
    } catch (...) {
        ret = erllama_error(env, mk_atom(env, "exception"));
    }
    pthread_mutex_unlock(&h->mu);
    return ret;
}

/* ============================================================== */
/* nif_spec_draft                                                 */
/* ============================================================== */

extern "C" ERL_NIF_TERM nif_spec_draft(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
    (void) argc;
    spec_holder *h = nullptr;
    if (!enif_get_resource(env, argv[0], SPEC_RT, (void **) &h)) {
        return enif_make_badarg(env);
    }
    int seq;
    int id_last;
    if (!enif_get_int(env, argv[1], &seq) || seq < 0 ||
        !enif_get_int(env, argv[2], &id_last) || id_last < 0) {
        return enif_make_badarg(env);
    }
    if (!spec_lock_live(h)) {
        return erllama_error(env, mk_atom(env, "released"));
    }
    if ((uint32_t) seq >= h->n_seq) {
        pthread_mutex_unlock(&h->mu);
        return enif_make_badarg(env);
    }
    ERL_NIF_TERM ret;
    try {
        if (!read_tokens(env, argv[3], h->prompts[seq])) {
            pthread_mutex_unlock(&h->mu);
            return enif_make_badarg(env);
        }
        /* The upstream draft() asserts result->empty(). */
        h->results[seq].clear();
        common_speculative_draft_params &dp =
            common_speculative_get_draft_params(h->spec,
                                                (llama_seq_id) seq);
        dp.drafting = true;
        dp.n_max = -1; /* the Erlang side truncates to its own caps */
        dp.n_past = (llama_pos) h->prompts[seq].size();
        dp.id_last = (llama_token) id_last;
        dp.prompt = &h->prompts[seq];
        dp.result = &h->results[seq];
        common_speculative_draft(h->spec);
        h->last_draft_len[seq] = (uint32_t) h->results[seq].size();

        ERL_NIF_TERM list = enif_make_list(env, 0);
        for (size_t i = h->results[seq].size(); i > 0; i--) {
            list = enif_make_list_cell(
                env, enif_make_int(env, h->results[seq][i - 1]), list);
        }
        ret = enif_make_tuple2(env, mk_atom(env, "ok"), list);
    } catch (...) {
        h->last_draft_len[seq] = 0;
        ret = erllama_error(env, mk_atom(env, "exception"));
    }
    pthread_mutex_unlock(&h->mu);
    return ret;
}

/* ============================================================== */
/* nif_spec_accept                                                */
/* ============================================================== */

extern "C" ERL_NIF_TERM nif_spec_accept(ErlNifEnv *env, int argc,
                                        const ERL_NIF_TERM argv[]) {
    (void) argc;
    spec_holder *h = nullptr;
    if (!enif_get_resource(env, argv[0], SPEC_RT, (void **) &h)) {
        return enif_make_badarg(env);
    }
    int seq;
    unsigned int n_acc;
    if (!enif_get_int(env, argv[1], &seq) || seq < 0 ||
        !enif_get_uint(env, argv[2], &n_acc)) {
        return enif_make_badarg(env);
    }
    if (!spec_lock_live(h)) {
        return erllama_error(env, mk_atom(env, "released"));
    }
    /* accept(N > 0) with no live draft for this seq trips a
     * GGML_ASSERT (process abort) upstream and would corrupt its
     * acceptance statistics; reject before calling through. */
    if ((uint32_t) seq >= h->n_seq || n_acc > h->last_draft_len[seq]) {
        pthread_mutex_unlock(&h->mu);
        return enif_make_badarg(env);
    }
    ERL_NIF_TERM ret = mk_atom(env, "ok");
    try {
        if (h->last_draft_len[seq] > 0) {
            common_speculative_accept(h->spec, (llama_seq_id) seq,
                                      (uint16_t) n_acc);
        }
        h->last_draft_len[seq] = 0;
    } catch (...) {
        ret = erllama_error(env, mk_atom(env, "exception"));
    }
    pthread_mutex_unlock(&h->mu);
    return ret;
}

/* ============================================================== */
/* nif_spec_free                                                  */
/* ============================================================== */

extern "C" ERL_NIF_TERM nif_spec_free(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
    (void) argc;
    spec_holder *h = nullptr;
    if (!enif_get_resource(env, argv[0], SPEC_RT, (void **) &h)) {
        return enif_make_badarg(env);
    }
    if (!spec_lock_live(h)) {
        return erllama_error(env, mk_atom(env, "released"));
    }
    common_speculative_free(h->spec);
    h->spec = nullptr;
    pthread_mutex_unlock(&h->mu);
    return mk_atom(env, "ok");
}
