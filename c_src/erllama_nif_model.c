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

/* Per-device probe: nif_list_devices() -> {ok, [#{index, name,
 * description, type, free_b, total_b, device_id, caps}]}. Includes
 * every registered backend device (CPU too); use it to discover the
 * names the `devices` and `tensor_buft_overrides` model options
 * accept. */
static ERL_NIF_TERM dev_type_atom(ErlNifEnv *env, int t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return enif_make_atom(env, "cpu");
        case GGML_BACKEND_DEVICE_TYPE_GPU: return enif_make_atom(env, "gpu");
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return enif_make_atom(env, "igpu");
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return enif_make_atom(env, "accel");
        default: return enif_make_atom(env, "meta");
    }
}

static ERL_NIF_TERM prop_str_bin(ErlNifEnv *env, const char *str) {
    ERL_NIF_TERM bin;
    if (!erllama_bin_from(env, str, strlen(str), &bin)) {
        return enif_make_atom(env, "undefined");
    }
    return bin;
}

ERL_NIF_TERM nif_list_devices(ErlNifEnv *env, int argc,
                              const ERL_NIF_TERM argv[]) {
    (void) argc;
    (void) argv;
    if (erllama_safe_backend_init_once() != 0) {
        return erllama_error(env, atom_exception);
    }
    size_t n = erllama_safe_backend_dev_count();
    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (size_t i = n; i > 0; i--) {
        size_t idx = i - 1;
        erllama_dev_props_t props;
        if (erllama_safe_backend_dev_props(idx, &props) != 0) {
            continue;
        }
        ERL_NIF_TERM caps = enif_make_new_map(env);
        erllama_map_put(env, &caps, "async",
                        props.caps_async ? atom_true : atom_false);
        erllama_map_put(env, &caps, "host_buffer",
                        props.caps_host_buffer ? atom_true : atom_false);
        erllama_map_put(
            env, &caps, "buffer_from_host_ptr",
            props.caps_buffer_from_host_ptr ? atom_true : atom_false);
        erllama_map_put(env, &caps, "events",
                        props.caps_events ? atom_true : atom_false);
        erllama_map_put(env, &caps, "mmap_support",
                        props.caps_mmap ? atom_true : atom_false);
        ERL_NIF_TERM m = enif_make_new_map(env);
        erllama_map_put(env, &m, "index",
                        enif_make_uint64(env, (uint64_t) idx));
        erllama_map_put(env, &m, "name", prop_str_bin(env, props.name));
        erllama_map_put(env, &m, "description",
                        prop_str_bin(env, props.description));
        erllama_map_put(env, &m, "type",
                        dev_type_atom(env, props.dev_type));
        erllama_map_put(env, &m, "free_b",
                        enif_make_uint64(env, props.free_b));
        erllama_map_put(env, &m, "total_b",
                        enif_make_uint64(env, props.total_b));
        erllama_map_put(env, &m, "device_id",
                        props.has_device_id
                            ? prop_str_bin(env, props.device_id)
                            : enif_make_atom(env, "undefined"));
        erllama_map_put(env, &m, "caps", caps);
        list = enif_make_list_cell(env, m, list);
    }
    return enif_make_tuple2(env, atom_ok, list);
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

/* Parse the `devices` model option (list of device-name binaries, or
 * the atom `none` for zero offload) into a NULL-terminated handle
 * array on the caller's stack (llama copies it during load).
 * Returns 1 (params->devices set), 0 (key absent), -1 (bad shape,
 * caller raises badarg), -2 (unknown or CPU device; *err_name holds
 * the offending element's term). */
static int parse_devices(ErlNifEnv *env, ERL_NIF_TERM map,
                         ggml_backend_dev_t *devs,
                         struct llama_model_params *params,
                         ERL_NIF_TERM *err_name) {
    ERL_NIF_TERM v;
    if (!enif_get_map_value(env, map, enif_make_atom(env, "devices"), &v)) {
        return 0;
    }
    if (enif_compare(v, enif_make_atom(env, "none")) == 0) {
        /* Upstream --device none: an empty but non-NULL list. */
        devs[0] = NULL;
        params->devices = devs;
        return 1;
    }
    unsigned int n;
    if (!enif_get_list_length(env, v, &n) || n == 0 ||
        n > (unsigned int) ERLLAMA_MAX_DEVICES) {
        return -1;
    }
    char names[ERLLAMA_MAX_DEVICES][64];
    const char *name_ptrs[ERLLAMA_MAX_DEVICES];
    ERL_NIF_TERM elems[ERLLAMA_MAX_DEVICES];
    ERL_NIF_TERM head;
    ERL_NIF_TERM tail = v;
    unsigned int i = 0;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        ErlNifBinary bin;
        if (!enif_inspect_iolist_as_binary(env, head, &bin) ||
            bin.size == 0 || bin.size >= sizeof(names[0])) {
            return -1;
        }
        memcpy(names[i], bin.data, bin.size);
        names[i][bin.size] = '\0';
        name_ptrs[i] = names[i];
        elems[i] = head;
        i++;
    }
    size_t bad = 0;
    if (erllama_safe_resolve_devices(name_ptrs, n, devs, &bad) != 0) {
        *err_name = elems[bad];
        return -2;
    }
    params->devices = devs;
    return 1;
}

/* Number of bytes the block-regex for `idx` needs (incl. NUL).
 * Regenerating is cheap; the two-pass arena build calls this twice. */
#define ERLLAMA_BLOCK_REGEX_CAP 160

/* Build the resource-owned tensor_buft_overrides arena from the
 * `tensor_buft_overrides` list ([{PatternBin, cpu | DeviceNameBin}])
 * and the `cpu_moe` sugar (true = all expert tensors to CPU, N =
 * first N blocks). Explicit overrides come first (first regex match
 * wins), sugar entries after. The arena holds the (n+1) entries
 * followed by the pattern bytes; llama does NOT copy the override
 * array, so it lives on the resource for the model lifetime.
 * Returns 1 (params->tensor_buft_overrides set), 0 (nothing
 * requested), -1 (bad shape), -2 (unknown device name; *err_name
 * set), -3 (oom). */
static int build_tbo(ErlNifEnv *env, ERL_NIF_TERM map,
                     erllama_model_t *res,
                     struct llama_model_params *params,
                     ERL_NIF_TERM *err_name) {
    ERL_NIF_TERM listv;
    int has_list = enif_get_map_value(
        env, map, enif_make_atom(env, "tensor_buft_overrides"), &listv);
    unsigned int n_list = 0;
    if (has_list && (!enif_get_list_length(env, listv, &n_list))) {
        return -1;
    }

    long moe_n = 0; /* 0 = off, -1 = all, N = first N blocks */
    {
        ERL_NIF_TERM v;
        if (enif_get_map_value(env, map, enif_make_atom(env, "cpu_moe"),
                               &v)) {
            if (enif_compare(v, atom_true) == 0) {
                moe_n = -1;
            } else {
                long l;
                if (!enif_get_long(env, v, &l) || l < 1) return -1;
                moe_n = l;
            }
        }
    }
    size_t n_moe = moe_n < 0 ? 1 : (size_t) moe_n;
    if (moe_n == 0) n_moe = 0;
    size_t n_total = (size_t) n_list + n_moe;
    if (n_total == 0) return 0;
    if (n_total > 4096) return -1;

    /* Pass 1: size the pattern bytes. */
    size_t pat_bytes = 0;
    ERL_NIF_TERM head;
    ERL_NIF_TERM tail = listv;
    if (has_list) {
        while (enif_get_list_cell(env, tail, &head, &tail)) {
            const ERL_NIF_TERM *pair;
            int arity;
            ErlNifBinary pat;
            if (!enif_get_tuple(env, head, &arity, &pair) || arity != 2 ||
                !enif_inspect_iolist_as_binary(env, pair[0], &pat) ||
                pat.size == 0) {
                return -1;
            }
            pat_bytes += pat.size + 1;
        }
    }
    if (moe_n < 0) {
        pat_bytes += strlen(erllama_safe_ffn_exps_regex()) + 1;
    } else {
        char buf[ERLLAMA_BLOCK_REGEX_CAP];
        for (long i = 0; i < moe_n; i++) {
            if (erllama_safe_ffn_exps_block_regex((int) i, buf,
                                                  sizeof(buf)) != 0) {
                return -1;
            }
            pat_bytes += strlen(buf) + 1;
        }
    }

    size_t entry_bytes =
        (n_total + 1) * sizeof(struct llama_model_tensor_buft_override);
    unsigned char *arena = enif_alloc(entry_bytes + pat_bytes);
    if (!arena) return -3;
    struct llama_model_tensor_buft_override *entries =
        (struct llama_model_tensor_buft_override *) arena;
    char *pat_cursor = (char *) (arena + entry_bytes);
    size_t k = 0;

    /* Pass 2: fill. Explicit overrides first. */
    if (has_list) {
        tail = listv;
        while (enif_get_list_cell(env, tail, &head, &tail)) {
            const ERL_NIF_TERM *pair;
            int arity;
            ErlNifBinary pat;
            (void) enif_get_tuple(env, head, &arity, &pair);
            (void) enif_inspect_iolist_as_binary(env, pair[0], &pat);
            ggml_backend_buffer_type_t buft = NULL;
            if (enif_compare(pair[1], enif_make_atom(env, "cpu")) == 0) {
                buft = erllama_safe_cpu_buffer_type();
            } else {
                ErlNifBinary nameb;
                char name[64];
                if (!enif_inspect_iolist_as_binary(env, pair[1], &nameb) ||
                    nameb.size == 0 || nameb.size >= sizeof(name)) {
                    enif_free(arena);
                    return -1;
                }
                memcpy(name, nameb.data, nameb.size);
                name[nameb.size] = '\0';
                buft = erllama_safe_dev_default_buft_by_name(name);
                if (!buft) {
                    enif_free(arena);
                    *err_name = pair[1];
                    return -2;
                }
            }
            if (!buft) {
                enif_free(arena);
                return -1;
            }
            memcpy(pat_cursor, pat.data, pat.size);
            pat_cursor[pat.size] = '\0';
            entries[k].pattern = pat_cursor;
            entries[k].buft = buft;
            pat_cursor += pat.size + 1;
            k++;
        }
    }
    if (moe_n != 0) {
        ggml_backend_buffer_type_t cpu_buft = erllama_safe_cpu_buffer_type();
        if (!cpu_buft) {
            enif_free(arena);
            return -1;
        }
        if (moe_n < 0) {
            const char *pat = erllama_safe_ffn_exps_regex();
            size_t len = strlen(pat);
            memcpy(pat_cursor, pat, len + 1);
            entries[k].pattern = pat_cursor;
            entries[k].buft = cpu_buft;
            pat_cursor += len + 1;
            k++;
        } else {
            char buf[ERLLAMA_BLOCK_REGEX_CAP];
            for (long i = 0; i < moe_n; i++) {
                (void) erllama_safe_ffn_exps_block_regex((int) i, buf,
                                                         sizeof(buf));
                size_t len = strlen(buf);
                memcpy(pat_cursor, buf, len + 1);
                entries[k].pattern = pat_cursor;
                entries[k].buft = cpu_buft;
                pat_cursor += len + 1;
                k++;
            }
        }
    }
    entries[k].pattern = NULL;
    entries[k].buft = NULL;

    res->tbo = arena;
    params->tensor_buft_overrides = entries;
    return 1;
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
    /* Signed read: any negative n_gpu_layers means "all layers"
     * (llama.cpp core semantics; -1 is also the fit default). */
    if (get_map_int32(env, argv[1], "n_gpu_layers", &i32)) {
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
    if (lm_rc < 0) {
        enif_release_resource(res);
        return enif_make_badarg(env);
    }
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

    /* Explicit device selection: `devices => [Name] | none`. The
     * handle array lives on this stack frame; llama copies it during
     * the load. Unknown / CPU names surface as a typed error, not
     * badarg: availability is runtime-dependent. */
    ggml_backend_dev_t devs[ERLLAMA_MAX_DEVICES + 1];
    {
        ERL_NIF_TERM bad_name = atom_error;
        int rc = parse_devices(env, argv[1], devs, &params, &bad_name);
        if (rc == -1) {
            enif_release_resource(res);
            return enif_make_badarg(env);
        }
        if (rc == -2) {
            enif_release_resource(res);
            return erllama_error(
                env,
                enif_make_tuple2(
                    env, enif_make_atom(env, "unknown_device"), bad_name));
        }
    }

    /* MoE offload + generic per-tensor buffer overrides. */
    {
        ERL_NIF_TERM bad_name = atom_error;
        int rc = build_tbo(env, argv[1], res, &params, &bad_name);
        if (rc == -1) {
            enif_release_resource(res);
            return enif_make_badarg(env);
        }
        if (rc == -2) {
            enif_release_resource(res);
            return erllama_error(
                env,
                enif_make_tuple2(
                    env, enif_make_atom(env, "unknown_device"), bad_name));
        }
        if (rc == -3) {
            enif_release_resource(res);
            return erllama_error(env, atom_oom);
        }
    }

    /* Optional auto-fit pre-pass: measure the model against device
     * memory and let common_fit_params pick n_gpu_layers /
     * tensor_split / overrides (and n_ctx when auto). The Erlang
     * validator guarantees fit excludes the manual placement keys,
     * so res->tbo is still free for fit's writable override buffer.
     * On non-success the params snapshot is restored (fit can leave
     * them partially mutated) and the load proceeds with defaults;
     * a broken file is classified by the real load below. */
    int fit_requested = 0;
    int fit_ok = 0;
    uint32_t fit_n_ctx = 0;
    {
        ERL_NIF_TERM fitv;
        if (enif_get_map_value(env, argv[1], enif_make_atom(env, "fit"),
                               &fitv)) {
            if (!enif_is_map(env, fitv) || res->tbo != NULL) {
                enif_release_resource(res);
                return enif_make_badarg(env);
            }
            fit_requested = 1;
            /* Per-device byte margins, default 1 GiB (upstream);
             * `margins_mib` is a non-empty list, the last entry
             * broadcast to the remaining devices. */
            size_t margins[ERLLAMA_MAX_DEVICES];
            for (size_t i = 0; i < ERLLAMA_MAX_DEVICES; i++) {
                margins[i] = (size_t) 1024 * 1024 * 1024;
            }
            ERL_NIF_TERM mv;
            if (enif_get_map_value(env, fitv,
                                   enif_make_atom(env, "margins_mib"),
                                   &mv)) {
                ERL_NIF_TERM head;
                ERL_NIF_TERM tail = mv;
                size_t i = 0;
                size_t last = margins[0];
                while (enif_get_list_cell(env, tail, &head, &tail)) {
                    unsigned long mib;
                    if (i >= ERLLAMA_MAX_DEVICES ||
                        !enif_get_ulong(env, head, &mib)) {
                        enif_release_resource(res);
                        return enif_make_badarg(env);
                    }
                    last = (size_t) mib * 1024 * 1024;
                    margins[i++] = last;
                }
                if (i == 0) {
                    enif_release_resource(res);
                    return enif_make_badarg(env);
                }
                for (; i < ERLLAMA_MAX_DEVICES; i++) {
                    margins[i] = last;
                }
            }
            unsigned int min_ctx = 4096;
            (void) get_map_uint(env, fitv, "min_ctx", &min_ctx);
            /* The fit context params approximate the real context
             * the model layer will open: injected by the Erlang
             * side as `fit_context` (n_ctx = planned size, or absent
             * for n_ctx auto where fit chooses). */
            struct llama_context_params fcp =
                llama_context_default_params();
            ERL_NIF_TERM fctx;
            if (enif_get_map_value(env, argv[1],
                                   enif_make_atom(env, "fit_context"),
                                   &fctx) &&
                enif_is_map(env, fctx)) {
                if (!erllama_parse_cparams(env, fctx, &fcp)) {
                    enif_release_resource(res);
                    return enif_make_badarg(env);
                }
            }
            int nca = 0;
            if (get_map_bool(env, fitv, "n_ctx_auto", &nca) && nca) {
                fcp.n_ctx = 0;
            }
            size_t entry_bytes =
                (size_t) 4097 *
                sizeof(struct llama_model_tensor_buft_override);
            unsigned char *arena = enif_alloc(entry_bytes);
            if (!arena) {
                enif_release_resource(res);
                return erllama_error(env, atom_oom);
            }
            memset(arena, 0, entry_bytes);
            res->tbo = arena;
            struct llama_model_params snapshot = params;
            int frc = erllama_safe_fit_params(
                path, &params, &fcp, res->tensor_split,
                (struct llama_model_tensor_buft_override *) arena,
                margins, (uint32_t) min_ctx);
            if (frc == 0) {
                fit_ok = 1;
            } else {
                params = snapshot;
            }
            fit_n_ctx = fcp.n_ctx;
        }
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
    if (!fit_requested) {
        return enif_make_tuple2(env, atom_ok, term);
    }
    /* Fit info: the effective values the load ran with (fitted on
     * success, the caller's/defaults on failure). n_gpu_layers -1 =
     * all layers. tensor_split is echoed only when fit assigned it. */
    ERL_NIF_TERM info = enif_make_new_map(env);
    erllama_map_put(env, &info, "fit",
                    enif_make_atom(env, fit_ok ? "ok" : "failed"));
    erllama_map_put(env, &info, "n_gpu_layers",
                    enif_make_int(env, params.n_gpu_layers));
    erllama_map_put(env, &info, "n_ctx", enif_make_uint(env, fit_n_ctx));
    if (fit_ok && params.tensor_split == res->tensor_split) {
        ERL_NIF_TERM list = enif_make_list(env, 0);
        for (int i = (int) ERLLAMA_MAX_DEVICES - 1; i >= 0; i--) {
            list = enif_make_list_cell(
                env,
                enif_make_double(env, (double) res->tensor_split[i]),
                list);
        }
        erllama_map_put(env, &info, "tensor_split", list);
    }
    return enif_make_tuple3(env, atom_ok, term, info);
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
