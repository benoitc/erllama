/* C++ NIF wrapper around llama.cpp's multimodal library (libmtmd,
 * tools/mtmd): mmproj projector loading, capability probes, and the
 * whole-prompt media prefill used by nif_media_prefill.
 *
 * Entry points exposed to Erlang via the erllama_nif .so:
 *
 *   nif_mtmd_init(ModelRef, Path, Opts) -> {ok, MtmdRef} | {error, _}
 *   nif_mtmd_free(MtmdRef)              -> ok | {error, released}
 *   nif_mtmd_caps(MtmdRef)              -> {ok, Map} | {error, _}
 *   nif_mtmd_caps_file(Path)            -> {ok, Map} | {error, _}
 *
 * The projector borrows the llama_model, so the model resource
 * gains an `active_mtmd` count mirroring adapters: free_model
 * defers while a projector is alive, and the last drop performs
 * the deferred free (model_drops_mtmd in erllama_nif.c).
 *
 * The eval side (erllama_mtmd_prefill_run) is driven from the C
 * decode file, which owns the context lock and per-seq bookkeeping;
 * this TU never touches erllama_context_t internals. mtmd's eval
 * helpers are not thread-safe - the caller serializes on c->mu.
 * TTS (mtmd_gen_*) is deliberately not exposed. */

#include "erllama_mtmd_nif.h"
#include "erllama_nif_util.h"
#include "erllama_resources.h"
#include "erllama_safe.h"

#include "mtmd.h"
#include "mtmd-helper.h"

#include <pthread.h>

#include <exception>
#include <new>
#include <vector>

ErlNifResourceType *ERLLAMA_MTMD_RT = nullptr;

/* Defined in erllama_nif.c next to model_drops_adapter. */
extern "C" void model_drops_mtmd(erllama_model_t *m);

namespace {

struct mtmd_holder {
    pthread_mutex_t mu;
    int mu_inited;
    mtmd_context *mtmd; /* nullptr after explicit free */
    erllama_model_t *model_res;
};

void mtmd_dtor(ErlNifEnv *, void *obj) {
    auto *h = static_cast<mtmd_holder *>(obj);
    if (h->mtmd) {
        try {
            mtmd_free(h->mtmd);
        } catch (...) {
        }
        h->mtmd = nullptr;
    }
    if (h->model_res) {
        model_drops_mtmd(h->model_res);
        enif_release_resource(h->model_res);
        h->model_res = nullptr;
    }
    if (h->mu_inited) {
        pthread_mutex_destroy(&h->mu);
        h->mu_inited = 0;
    }
    h->~mtmd_holder();
}

ERL_NIF_TERM mk_atom(ErlNifEnv *env, const char *name) {
    return enif_make_atom(env, name);
}

ERL_NIF_TERM caps_map(ErlNifEnv *env, bool vision, bool audio,
                      int sample_rate, const char *marker) {
    ERL_NIF_TERM m = enif_make_new_map(env);
    erllama_map_put(env, &m, "vision",
                    mk_atom(env, vision ? "true" : "false"));
    erllama_map_put(env, &m, "audio",
                    mk_atom(env, audio ? "true" : "false"));
    if (sample_rate > 0) {
        erllama_map_put(env, &m, "audio_sample_rate",
                        enif_make_int(env, sample_rate));
    }
    if (marker) {
        ERL_NIF_TERM bin;
        if (erllama_bin_from(env, marker, strlen(marker), &bin)) {
            erllama_map_put(env, &m, "marker", bin);
        }
    }
    return m;
}

} /* anonymous namespace */

extern "C" int mtmd_nif_load(ErlNifEnv *env) {
    ERLLAMA_MTMD_RT = enif_open_resource_type(
        env, nullptr, "erllama_mtmd", mtmd_dtor, ERL_NIF_RT_CREATE,
        nullptr);
    if (!ERLLAMA_MTMD_RT) {
        return -1;
    }
    /* Route mtmd/clip log lines into the same process-wide capture
     * llama uses, so they reach erllama_log and the malformed-GGUF
     * classifier instead of stderr. */
    mtmd_helper_log_set(erllama_safe_log_callback(), nullptr);
    return 0;
}

/* ============================================================== */
/* nif_mtmd_init                                                  */
/* ============================================================== */

extern "C" ERL_NIF_TERM nif_mtmd_init(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m = nullptr;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    char path[4097];
    if (!copy_path(env, argv[1], path, sizeof(path))) {
        return enif_make_badarg(env);
    }
    if (!enif_is_map(env, argv[2])) {
        return enif_make_badarg(env);
    }
    mtmd_context_params params = mtmd_context_params_default();
    params.print_timings = false;
    int b;
    if (get_map_bool(env, argv[2], "use_gpu", &b)) {
        params.use_gpu = b ? true : false;
    }
    int32_t i32;
    if (get_map_int31(env, argv[2], "n_threads", &i32)) {
        params.n_threads = i32;
    }
    if (get_map_int31(env, argv[2], "image_min_tokens", &i32)) {
        params.image_min_tokens = i32;
    }
    if (get_map_int31(env, argv[2], "image_max_tokens", &i32)) {
        params.image_max_tokens = i32;
    }

    /* Mirror nif_adapter_load: refuse to attach to a released or
     * deferred-release model; the projector borrows m->model. */
    pthread_mutex_lock(&m->mu);
    if (!m->model || m->release_pending) {
        pthread_mutex_unlock(&m->mu);
        return erllama_error(env, mk_atom(env, "released"));
    }
    mtmd_context *ctx = nullptr;
    try {
        ctx = mtmd_init_from_file(path, m->model, params);
    } catch (...) {
        ctx = nullptr;
    }
    if (!ctx) {
        pthread_mutex_unlock(&m->mu);
        return erllama_error(env, mk_atom(env, "mmproj_load_failed"));
    }
    void *res = enif_alloc_resource(ERLLAMA_MTMD_RT, sizeof(mtmd_holder));
    if (!res) {
        try {
            mtmd_free(ctx);
        } catch (...) {
        }
        pthread_mutex_unlock(&m->mu);
        return erllama_error(env, mk_atom(env, "oom"));
    }
    auto *h = new (res) mtmd_holder{};
    if (pthread_mutex_init(&h->mu, nullptr) != 0) {
        h->~mtmd_holder();
        try {
            mtmd_free(ctx);
        } catch (...) {
        }
        enif_release_resource(res);
        pthread_mutex_unlock(&m->mu);
        return erllama_error(env, mk_atom(env, "oom"));
    }
    h->mu_inited = 1;
    h->mtmd = ctx;
    h->model_res = m;
    m->active_mtmd++;
    enif_keep_resource(m);
    pthread_mutex_unlock(&m->mu);

    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return enif_make_tuple2(env, mk_atom(env, "ok"), term);
}

/* ============================================================== */
/* nif_mtmd_free / caps                                           */
/* ============================================================== */

extern "C" ERL_NIF_TERM nif_mtmd_free(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
    (void) argc;
    mtmd_holder *h = nullptr;
    if (!enif_get_resource(env, argv[0], ERLLAMA_MTMD_RT, (void **) &h)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&h->mu);
    if (!h->mtmd) {
        pthread_mutex_unlock(&h->mu);
        return erllama_error(env, mk_atom(env, "released"));
    }
    try {
        mtmd_free(h->mtmd);
    } catch (...) {
    }
    h->mtmd = nullptr;
    pthread_mutex_unlock(&h->mu);
    return mk_atom(env, "ok");
}

extern "C" ERL_NIF_TERM nif_mtmd_caps(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
    (void) argc;
    mtmd_holder *h = nullptr;
    if (!enif_get_resource(env, argv[0], ERLLAMA_MTMD_RT, (void **) &h)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&h->mu);
    if (!h->mtmd) {
        pthread_mutex_unlock(&h->mu);
        return erllama_error(env, mk_atom(env, "released"));
    }
    ERL_NIF_TERM ret;
    try {
        bool vision = mtmd_support_vision(h->mtmd);
        bool audio = mtmd_support_audio(h->mtmd);
        int rate = mtmd_get_audio_sample_rate(h->mtmd);
        const char *marker = mtmd_get_marker(h->mtmd);
        ret = enif_make_tuple2(env, mk_atom(env, "ok"),
                               caps_map(env, vision, audio, rate, marker));
    } catch (...) {
        ret = erllama_error(env, mk_atom(env, "exception"));
    }
    pthread_mutex_unlock(&h->mu);
    return ret;
}

/* Capability probe from the mmproj file alone - no GPU commit, no
 * projector context. */
extern "C" ERL_NIF_TERM nif_mtmd_caps_file(ErlNifEnv *env, int argc,
                                           const ERL_NIF_TERM argv[]) {
    (void) argc;
    char path[4097];
    if (!copy_path(env, argv[0], path, sizeof(path))) {
        return enif_make_badarg(env);
    }
    try {
        mtmd_caps caps = mtmd_get_cap_from_file(path);
        return enif_make_tuple2(
            env, mk_atom(env, "ok"),
            caps_map(env, caps.inp_vision, caps.inp_audio, -1,
                     mtmd_default_marker()));
    } catch (...) {
        return erllama_error(env, mk_atom(env, "mmproj_load_failed"));
    }
}

/* ============================================================== */
/* Media prefill runner (called from nif_media_prefill in C)      */
/* ============================================================== */

extern "C" int erllama_mtmd_lock_live(void *res) {
    auto *h = static_cast<mtmd_holder *>(res);
    pthread_mutex_lock(&h->mu);
    if (!h->mtmd) {
        pthread_mutex_unlock(&h->mu);
        return 0;
    }
    return 1;
}

extern "C" void erllama_mtmd_unlock(void *res) {
    auto *h = static_cast<mtmd_holder *>(res);
    pthread_mutex_unlock(&h->mu);
}

extern "C" int erllama_mtmd_prefill_run(
    void *mtmd_res, struct llama_context *lctx, const char *prompt,
    size_t prompt_len, const erllama_media_item_t *items, size_t n_items,
    int add_special, int seq_id, int32_t n_batch, int32_t n_past,
    int32_t *out_n_tokens, int32_t *out_n_past) {
    auto *h = static_cast<mtmd_holder *>(mtmd_res);
    try {
        /* Decode every media item; bitmap wrappers own the decoded
         * pixels/samples (helper assigns sha-256 ids we do not use
         * yet - media requests bypass the KV cache in v1). */
        std::vector<mtmd::bitmap> bitmaps;
        bitmaps.reserve(n_items);
        for (size_t i = 0; i < n_items; i++) {
            mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_buf(
                h->mtmd, items[i].data, items[i].len, false);
            if (!w.bitmap) {
                return -1;
            }
            bitmaps.emplace_back(w.bitmap);
            /* Audio bytes decoded as an image (or vice versa) still
             * yield a bitmap of the right kind; enforce the declared
             * type so a mislabeled part fails loudly. */
            if ((items[i].is_audio ? true : false) !=
                mtmd_bitmap_is_audio(bitmaps.back().ptr.get())) {
                return -1;
            }
        }
        std::vector<const mtmd_bitmap *> ptrs;
        ptrs.reserve(n_items);
        for (auto &bm : bitmaps) {
            ptrs.push_back(bm.ptr.get());
        }

        std::string text(prompt, prompt_len);
        mtmd_input_text input;
        input.text = text.c_str();
        input.text_len = text.size();
        input.add_special = add_special ? true : false;
        input.parse_special = true;

        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        if (!chunks.ptr) {
            return -5;
        }
        int32_t trc = mtmd_tokenize(h->mtmd, chunks.ptr.get(), &input,
                                    ptrs.data(), ptrs.size());
        if (trc == 1) {
            return -2;
        }
        if (trc != 0) {
            return -3;
        }

        llama_pos total_pos = mtmd_helper_get_n_pos(chunks.ptr.get());
        size_t total_tokens = mtmd_helper_get_n_tokens(chunks.ptr.get());
        uint32_t n_ctx = llama_n_ctx(lctx);
        if (n_ctx == 0 ||
            (uint32_t) (n_past + total_pos) >= n_ctx) {
            return -6;
        }

        llama_pos new_n_past = n_past;
        int32_t erc = mtmd_helper_eval_chunks(
            h->mtmd, lctx, chunks.ptr.get(), (llama_pos) n_past,
            (llama_seq_id) seq_id, n_batch, /* logits_last */ true,
            &new_n_past);
        if (erc != 0) {
            return -4;
        }
        *out_n_tokens = (int32_t) total_tokens;
        *out_n_past = (int32_t) new_n_past;
        return 0;
    } catch (const std::bad_alloc &) {
        return -5;
    } catch (...) {
        return -5;
    }
}
