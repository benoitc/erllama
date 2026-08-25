/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */
/* Model lifecycle and probes: load/free, size/layers, family and
 * vocab metadata, VRAM info, load-progress reporting. */

#include "erllama_nif_int.h"
#include "erllama_safe.h"

#include <math.h>
#include <string.h>

/* Atom -> enum tables for llama.cpp option passthrough. Sized at
 * the call site via the const-array length idiom (sizeof / element
 * size) so the helper sees the count directly. */
static const erllama_atom_enum_pair_t SPLIT_MODE_TABLE[] = {
    {"none",   LLAMA_SPLIT_MODE_NONE},
    {"layer",  LLAMA_SPLIT_MODE_LAYER},
    {"row",    LLAMA_SPLIT_MODE_ROW},
    {"tensor", LLAMA_SPLIT_MODE_TENSOR},
};
static const erllama_atom_enum_pair_t LOAD_MODE_TABLE[] = {
    {"auto",       LLAMA_LOAD_MODE_AUTO},
    {"none",       LLAMA_LOAD_MODE_NONE},
    {"mmap",       LLAMA_LOAD_MODE_MMAP},
    {"mlock",      LLAMA_LOAD_MODE_MLOCK},
    {"mmap_mlock", LLAMA_LOAD_MODE_MMAP_MLOCK},
    {"direct_io",  LLAMA_LOAD_MODE_DIRECT_IO},
};

/* =========================================================================
 * Model accessors (size, layer count)
 *
 * Used by erllama_model to derive `vram_estimate_b` for `list_models`
 * metadata. Read-only, take the model resource lock for safety
 * against a concurrent free.
 * ========================================================================= */
ERL_NIF_TERM nif_model_size(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    if (!erllama_lock_model(m)) {
        return erllama_error(env, atom_released);
    }
    uint64_t sz = erllama_safe_model_size(m->model);
    pthread_mutex_unlock(&m->mu);
    return enif_make_uint64(env, sz);
}

ERL_NIF_TERM nif_model_n_layer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    if (!erllama_lock_model(m)) {
        return erllama_error(env, atom_released);
    }
    int32_t n = erllama_safe_model_n_layer(m->model);
    pthread_mutex_unlock(&m->mu);
    return enif_make_int(env, n);
}

/* Copy a NUL-terminated probe result into a binary term. A negative
 * probe rc (key missing / exception) yields an empty binary; a
 * binary-allocation failure degrades to `undefined`. */
static ERL_NIF_TERM family_str(ErlNifEnv *env, int32_t rc, const char *buf) {
    size_t len = rc < 0 ? 0 : strlen(buf);
    ERL_NIF_TERM bin;
    if (!erllama_bin_from(env, buf, len, &bin)) {
        return enif_make_atom(env, "undefined");
    }
    return bin;
}

/* Model family / GGUF metadata probe:
 *   nif_model_family(ModelRef) -> #{arch, name, desc, n_ctx_train,
 *     n_params, n_embd, n_layer, n_head, n_swa, recurrent, hybrid,
 *     diffusion, has_encoder, has_decoder, ftype}
 * One read-only pass at load time; the model layer keeps the map and
 * derives its cache-restore policy from it. */
ERL_NIF_TERM nif_model_family(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    if (!erllama_lock_model(m)) {
        return erllama_error(env, atom_released);
    }
    char arch[128] = {0};
    char name[256] = {0};
    char desc[256] = {0};
    int32_t arch_rc = erllama_safe_model_meta_val_str(
        m->model, "general.architecture", arch, sizeof(arch));
    int32_t name_rc = erllama_safe_model_meta_val_str(
        m->model, "general.name", name, sizeof(name));
    int32_t desc_rc = erllama_safe_model_desc(m->model, desc, sizeof(desc));
    int32_t n_ctx_train = erllama_safe_model_n_ctx_train(m->model);
    uint64_t n_params = erllama_safe_model_n_params(m->model);
    int32_t n_embd = erllama_safe_n_embd(m->model);
    int32_t n_layer = erllama_safe_model_n_layer(m->model);
    int32_t n_head = erllama_safe_model_n_head(m->model);
    int32_t n_swa = erllama_safe_model_n_swa(m->model);
    int recurrent = erllama_safe_model_is_recurrent(m->model);
    int hybrid = erllama_safe_model_is_hybrid(m->model);
    int diffusion = erllama_safe_model_is_diffusion(m->model);
    int has_encoder = erllama_safe_model_has_encoder(m->model);
    int has_decoder = erllama_safe_model_has_decoder(m->model);
    int ftype = erllama_safe_model_ftype(m->model);
    pthread_mutex_unlock(&m->mu);

    ERL_NIF_TERM map = enif_make_new_map(env);
    erllama_map_put(env, &map, "arch", family_str(env, arch_rc, arch));
    erllama_map_put(env, &map, "name", family_str(env, name_rc, name));
    erllama_map_put(env, &map, "desc", family_str(env, desc_rc, desc));
    erllama_map_put(env, &map, "n_ctx_train", enif_make_int(env, n_ctx_train));
    erllama_map_put(env, &map, "n_params", enif_make_uint64(env, n_params));
    erllama_map_put(env, &map, "n_embd", enif_make_int(env, n_embd));
    erllama_map_put(env, &map, "n_layer", enif_make_int(env, n_layer));
    erllama_map_put(env, &map, "n_head", enif_make_int(env, n_head));
    erllama_map_put(env, &map, "n_swa", enif_make_int(env, n_swa));
    erllama_map_put(env, &map, "recurrent", recurrent ? atom_true : atom_false);
    erllama_map_put(env, &map, "hybrid", hybrid ? atom_true : atom_false);
    erllama_map_put(env, &map, "diffusion", diffusion ? atom_true : atom_false);
    erllama_map_put(env, &map, "has_encoder", has_encoder ? atom_true : atom_false);
    erllama_map_put(env, &map, "has_decoder", has_decoder ? atom_true : atom_false);
    erllama_map_put(env, &map, "ftype", enif_make_int(env, ftype));
    return map;
}

/* A vocab token id term: LLAMA_TOKEN_NULL -> `undefined`. */
static ERL_NIF_TERM vocab_tok_term(ErlNifEnv *env, llama_token t) {
    if (t == LLAMA_TOKEN_NULL) {
        return enif_make_atom(env, "undefined");
    }
    return enif_make_int(env, t);
}

/* Vocab / special-token probe:
 *   nif_vocab_info(ModelRef) -> #{n_vocab, add_bos, add_eos, bos,
 *     eos, eot, sep, nl, pad, mask, fim_pre, fim_suf, fim_mid,
 *     fim_pad, fim_rep, fim_sep}
 * Token keys are integers or `undefined` when the model has no such
 * token. One read-only pass; callers cache as they see fit. */
ERL_NIF_TERM nif_vocab_info(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    if (!erllama_lock_model(m)) {
        return erllama_error(env, atom_released);
    }
    int32_t n_vocab = 0;
    const struct llama_vocab *vocab = erllama_model_vocab(m, &n_vocab);
    if (!vocab) {
        pthread_mutex_unlock(&m->mu);
        return erllama_error(env, atom_exception);
    }
    int add_bos = erllama_safe_vocab_get_add_bos(vocab);
    int add_eos = erllama_safe_vocab_get_add_eos(vocab);
    llama_token bos = erllama_safe_vocab_bos(vocab);
    llama_token eos = erllama_safe_vocab_eos(vocab);
    llama_token eot = erllama_safe_vocab_eot(vocab);
    llama_token sep = erllama_safe_vocab_sep(vocab);
    llama_token nl = erllama_safe_vocab_nl(vocab);
    llama_token pad = erllama_safe_vocab_pad(vocab);
    llama_token mask = erllama_safe_vocab_mask(vocab);
    llama_token fim_pre = erllama_safe_vocab_fim_pre(vocab);
    llama_token fim_suf = erllama_safe_vocab_fim_suf(vocab);
    llama_token fim_mid = erllama_safe_vocab_fim_mid(vocab);
    llama_token fim_pad = erllama_safe_vocab_fim_pad(vocab);
    llama_token fim_rep = erllama_safe_vocab_fim_rep(vocab);
    llama_token fim_sep = erllama_safe_vocab_fim_sep(vocab);
    pthread_mutex_unlock(&m->mu);

    ERL_NIF_TERM map = enif_make_new_map(env);
    erllama_map_put(env, &map, "n_vocab", enif_make_int(env, n_vocab));
    erllama_map_put(env, &map, "add_bos", add_bos ? atom_true : atom_false);
    erllama_map_put(env, &map, "add_eos", add_eos ? atom_true : atom_false);
    erllama_map_put(env, &map, "bos", vocab_tok_term(env, bos));
    erllama_map_put(env, &map, "eos", vocab_tok_term(env, eos));
    erllama_map_put(env, &map, "eot", vocab_tok_term(env, eot));
    erllama_map_put(env, &map, "sep", vocab_tok_term(env, sep));
    erllama_map_put(env, &map, "nl", vocab_tok_term(env, nl));
    erllama_map_put(env, &map, "pad", vocab_tok_term(env, pad));
    erllama_map_put(env, &map, "mask", vocab_tok_term(env, mask));
    erllama_map_put(env, &map, "fim_pre", vocab_tok_term(env, fim_pre));
    erllama_map_put(env, &map, "fim_suf", vocab_tok_term(env, fim_suf));
    erllama_map_put(env, &map, "fim_mid", vocab_tok_term(env, fim_mid));
    erllama_map_put(env, &map, "fim_pad", vocab_tok_term(env, fim_pad));
    erllama_map_put(env, &map, "fim_rep", vocab_tok_term(env, fim_rep));
    erllama_map_put(env, &map, "fim_sep", vocab_tok_term(env, fim_sep));
    return map;
}

/* =========================================================================
 * VRAM probe
 *
 * Walks every loaded ggml backend and sums free / total memory across
 * non-CPU devices (GPU, integrated GPU, accelerator). META is a
 * pseudo-device used by ggml internals and is excluded. CPU memory is
 * also excluded: callers asking for vram_info want VRAM, and there is
 * no good answer for "VRAM on a CPU-only build" -- we return
 * {error, no_gpu} so the caller can fall back to system memory probes
 * of its own choosing rather than reporting a fake number.
 *
 * The probe is rare (cluster scheduler) and runs on the dirty CPU
 * scheduler.
 * ========================================================================= */
ERL_NIF_TERM nif_vram_info(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    (void) argv;
    if (erllama_safe_backend_init_once() != 0) {
        return erllama_error(env, atom_exception);
    }
    size_t n = erllama_safe_backend_dev_count();
    size_t total_sum = 0, free_sum = 0;
    int found_non_cpu = 0;
    for (size_t i = 0; i < n; i++) {
        size_t f = 0, t = 0;
        int dt = 0;
        if (erllama_safe_backend_dev_info(i, &f, &t, &dt) != 0) {
            continue;
        }
        if (dt == GGML_BACKEND_DEVICE_TYPE_GPU
            || dt == GGML_BACKEND_DEVICE_TYPE_IGPU
            || dt == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            total_sum += t;
            free_sum += f;
            found_non_cpu = 1;
        }
    }
    if (!found_non_cpu) {
        return erllama_error(env, atom_no_gpu);
    }
    size_t used_sum = (total_sum > free_sum) ? (total_sum - free_sum) : 0;
    ERL_NIF_TERM map = enif_make_new_map(env);
    enif_make_map_put(env, map, atom_total_b, enif_make_uint64(env, total_sum), &map);
    enif_make_map_put(env, map, atom_free_b, enif_make_uint64(env, free_sum), &map);
    enif_make_map_put(env, map, atom_used_b, enif_make_uint64(env, used_sum), &map);
    return enif_make_tuple2(env, atom_ok, map);
}

/* =========================================================================
 * Model
 * ========================================================================= */

/* Load-progress reporting context. Lives on nif_load_model's stack:
 * llama.cpp invokes the progress callback synchronously on the
 * loading thread (once per tensor, then a final 1.0). Messages are
 * throttled to integer-percent steps; enif_send failure (receiver
 * gone) is ignored so a dead listener never affects the load. */
typedef struct {
    ErlNifPid pid;
    ErlNifEnv *env; /* NULL = progress reporting disabled */
    char tag[128];
    size_t tag_len;
    unsigned last_pct;
    int sent_final;
} erllama_progress_ctx_t;

static bool erllama_load_progress_cb(float progress, void *user_data) {
    erllama_progress_ctx_t *pc = user_data;
    unsigned pct = (unsigned) (progress * 100.0f);
    int final = progress >= 1.0f;
    if (final) {
        if (pc->sent_final) return true;
        pc->sent_final = 1;
    } else if (pct <= pc->last_pct) {
        return true;
    }
    pc->last_pct = pct;
    enif_clear_env(pc->env);
    ERL_NIF_TERM tag;
    unsigned char *p = enif_make_new_binary(pc->env, pc->tag_len, &tag);
    if (!p) {
        return true;
    }
    if (pc->tag_len > 0) {
        memcpy(p, pc->tag, pc->tag_len);
    }
    ERL_NIF_TERM msg = enif_make_tuple3(
        pc->env, enif_make_atom(pc->env, "erllama_load_progress"), tag,
        enif_make_double(pc->env, (double) progress));
    (void) enif_send(NULL, &pc->pid, pc->env, msg);
    return true;
}

ERL_NIF_TERM nif_load_model(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    char path[4097];
    if (!copy_path(env, argv[0], path, sizeof(path))) {
        return enif_make_badarg(env);
    }
    if (!enif_is_map(env, argv[1])) {
        return enif_make_badarg(env);
    }

    if (erllama_safe_backend_init_once() != 0) {
        return erllama_error(env, atom_load_failed);
    }

    /* Allocate the resource BEFORE the model load so `params.tensor_split`
     * can point at the resource-owned buffer (the model copies params by
     * value but the tensor_split pointer inside is not copied — it must
     * outlive the model). Zero-init so the destructor on the
     * alloc-but-not-fully-set-up path sees model=NULL and mu_inited=0 and
     * skips the dangerous frees. */
    erllama_model_t *res = enif_alloc_resource(MODEL_RT, sizeof(*res));
    if (!res) {
        return erllama_error(env, atom_oom);
    }
    memset(res, 0, sizeof(*res));

    struct llama_model_params params = llama_model_default_params();

    int32_t i32;
    if (get_map_int31(env, argv[1], "n_gpu_layers", &i32)) {
        params.n_gpu_layers = i32;
    }
    if (get_map_int31(env, argv[1], "main_gpu", &i32)) {
        params.main_gpu = i32;
    }
    int b;
    /* `load_mode' is the upstream knob (llama_load_mode). `use_mmap' /
     * `use_mlock' are kept as sugar and mapped onto it when `load_mode'
     * is absent: mmap+mlock -> MMAP_MLOCK, mmap -> MMAP, mlock -> MLOCK,
     * neither -> NONE; unspecified -> AUTO. */
    int lm_v;
    int lm_rc = get_map_atom_enum(
        env, argv[1], "load_mode",
        LOAD_MODE_TABLE, sizeof(LOAD_MODE_TABLE) / sizeof(LOAD_MODE_TABLE[0]),
        &lm_v
    );
    if (lm_rc < 0) return enif_make_badarg(env);
    if (lm_rc > 0) {
        params.load_mode = (enum llama_load_mode) lm_v;
    } else {
        int has_mmap = get_map_bool(env, argv[1], "use_mmap", &b);
        int use_mmap = has_mmap ? (b ? 1 : 0) : 1;
        int has_mlock = get_map_bool(env, argv[1], "use_mlock", &b);
        int use_mlock = has_mlock ? (b ? 1 : 0) : 0;
        if (has_mmap || has_mlock) {
            params.load_mode = use_mmap
                ? (use_mlock ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_MMAP)
                : (use_mlock ? LLAMA_LOAD_MODE_MLOCK : LLAMA_LOAD_MODE_NONE);
        }
    }
    if (get_map_bool(env, argv[1], "vocab_only", &b)) params.vocab_only = b ? true : false;

    int enum_v;
    int sm_rc = get_map_atom_enum(
        env, argv[1], "split_mode",
        SPLIT_MODE_TABLE, sizeof(SPLIT_MODE_TABLE) / sizeof(SPLIT_MODE_TABLE[0]),
        &enum_v
    );
    if (sm_rc < 0) {
        enif_release_resource(res);
        return enif_make_badarg(env);
    }
    if (sm_rc > 0) params.split_mode = (enum llama_split_mode) enum_v;

    /* Defensive: if a future vendored llama.cpp raises its max-device
     * cap past our local constant, surface a clear error rather than
     * silently truncating the user's per-device proportions. The bump
     * is one line at the top of this file. */
    if (llama_max_devices() > (size_t) ERLLAMA_MAX_DEVICES) {
        enif_release_resource(res);
        return erllama_error(env, atom_load_failed);
    }
    int ts_n = get_map_float_list(
        env, argv[1], "tensor_split",
        res->tensor_split, ERLLAMA_MAX_DEVICES
    );
    if (ts_n < 0) {
        enif_release_resource(res);
        return enif_make_badarg(env);
    }
    if (ts_n > 0) {
        res->has_tensor_split = 1;
        params.tensor_split = res->tensor_split;
    }

    /* Optional load-progress reporting: `progress_to` (local pid) +
     * `progress_tag` (binary, usually the model id). The callback
     * runs synchronously on THIS dirty thread, so the context struct
     * lives on this stack frame. */
    erllama_progress_ctx_t pctx;
    pctx.env = NULL;
    {
        ERL_NIF_TERM v;
        ErlNifPid ppid;
        if (enif_get_map_value(env, argv[1],
                               enif_make_atom(env, "progress_to"), &v) &&
            enif_get_local_pid(env, v, &ppid)) {
            ErlNifBinary tagbin;
            tagbin.size = 0;
            tagbin.data = NULL;
            ERL_NIF_TERM tv;
            if (enif_get_map_value(env, argv[1],
                                   enif_make_atom(env, "progress_tag"), &tv)) {
                (void) enif_inspect_iolist_as_binary(env, tv, &tagbin);
            }
            ErlNifEnv *penv = enif_alloc_env();
            if (penv) {
                pctx.pid = ppid;
                pctx.env = penv;
                pctx.tag_len = tagbin.size < sizeof(pctx.tag)
                                   ? tagbin.size
                                   : sizeof(pctx.tag);
                if (pctx.tag_len > 0) {
                    memcpy(pctx.tag, tagbin.data, pctx.tag_len);
                }
                pctx.last_pct = 0;
                pctx.sent_final = 0;
                params.progress_callback = erllama_load_progress_cb;
                params.progress_callback_user_data = &pctx;
            }
        }
    }

    erllama_load_status_t status = ERLLAMA_LOAD_FAILED;
    struct llama_model *model =
        erllama_safe_model_load_from_file_v2(path, params, &status);
    if (pctx.env) {
        enif_free_env(pctx.env);
    }
    if (!model) {
        enif_release_resource(res);
        ERL_NIF_TERM why = (status == ERLLAMA_LOAD_MALFORMED)
                               ? atom_malformed_gguf
                               : atom_load_failed;
        return erllama_error(env, why);
    }

    if (pthread_mutex_init(&res->mu, NULL) != 0) {
        (void) erllama_safe_model_free(model);
        enif_release_resource(res);
        return erllama_error(env, atom_oom);
    }
    res->mu_inited = 1;
    res->model = model;
    res->active_contexts = 0;

    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return enif_make_tuple2(env, atom_ok, term);
}

/* free_model/1 returns:
 *   ok                 -> released; subsequent ops on the term return error
 *   {ok, deferred}     -> contexts or adapters still hold this model;
 *                         release flagged. The last context or adapter
 *                         destruction performs the actual llama_model_free
 *                         under context_drops_model / model_drops_adapter.
 *   {error, released}  -> already released
 *
 * The lock blocks for the duration of any concurrent dirty NIF using
 * this resource, which is the point: free can never interleave with a
 * live llama_model_* call. */
ERL_NIF_TERM nif_free_model(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    if (!erllama_lock_model(m)) {
        return erllama_error(env, atom_released);
    }
    if (m->active_contexts > 0 || m->active_adapters > 0) {
        m->release_pending = 1;
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_ok, atom_deferred);
    }
    struct llama_model *to_free = m->model;
    /* Free under the lock so a concurrent state read can't observe a
     * mid-free m->model. The pointer is nulled regardless of the
     * wrapper's return: calling llama_model_free again on a freed
     * pointer is a double-free, and llama destructors are required
     * to be noexcept anyway -- if one throws we leak the native
     * object rather than risk UB. */
    int rc = erllama_safe_model_free(to_free);
    m->model = NULL;
    m->release_pending = 0;
    pthread_mutex_unlock(&m->mu);
    if (rc != 0) {
        return erllama_error(env, atom_exception);
    }
    return atom_ok;
}
