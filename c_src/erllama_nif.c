// Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
// See the LICENSE file at the project root.

/*
 * erllama_nif: single NIF for erllama (cache + llama.cpp surface).
 *
 * v0.2 surface:
 *   crc32c(IoData) -> non_neg_integer()              [dirty CPU]
 *   fsync_dir(Path) -> ok | {error, atom()}          [dirty IO]
 *   load_model(Path, Opts) -> {ok, ModelRes} | ...   [dirty IO]
 *   free_model(ModelRes) -> ok                       [regular]
 *   new_context(ModelRes, Opts) -> {ok, CtxRes} | .. [dirty CPU]
 *   free_context(CtxRes) -> ok                       [regular]
 *   tokenize(ModelRes, Text, Opts) -> [token_id()]   [dirty CPU]
 *   kv_pack(CtxRes, _Tokens, _NTokens) -> Binary     [dirty CPU]
 *   kv_unpack(CtxRes, Binary, SeqId) -> ok | err     [dirty CPU]
 *
 * Resource ownership: model and context resources hold pointers to
 * llama.cpp objects. Their destructors call llama_model_free /
 * llama_free. The context resource also holds a refcount on its
 * model resource via enif_keep_resource so the model survives as
 * long as any context derived from it does.
 */
#include <erl_nif.h>
#include "erllama_chat_nif.h"
#include "erllama_spec_nif.h"
#include "erllama_mtmd_nif.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "crc32c.h"
#include "llama.h"

#include "erllama_nif_int.h"
#include "erllama_safe.h"

/* =========================================================================
 * Atoms
 * ========================================================================= */

ERL_NIF_TERM atom_ok;
ERL_NIF_TERM atom_error;
ERL_NIF_TERM atom_load_failed;
ERL_NIF_TERM atom_malformed_gguf;
ERL_NIF_TERM atom_context_failed;
ERL_NIF_TERM atom_tokenize_failed;
ERL_NIF_TERM atom_pack_failed;
ERL_NIF_TERM atom_unpack_failed;
ERL_NIF_TERM atom_true;
ERL_NIF_TERM atom_false;
ERL_NIF_TERM atom_released;
ERL_NIF_TERM atom_too_large;
ERL_NIF_TERM atom_invalid_token;
ERL_NIF_TERM atom_context_overflow;
ERL_NIF_TERM atom_batch_overflow;
ERL_NIF_TERM atom_oom;
ERL_NIF_TERM atom_deferred;
ERL_NIF_TERM atom_exception;
ERL_NIF_TERM atom_no_logits;
ERL_NIF_TERM atom_no_template;
ERL_NIF_TERM atom_template_failed;
ERL_NIF_TERM atom_grammar_failed;
ERL_NIF_TERM atom_embed_failed;
ERL_NIF_TERM atom_not_supported;
ERL_NIF_TERM atom_invalid_content;
ERL_NIF_TERM atom_no_gpu;
ERL_NIF_TERM atom_total_b;
ERL_NIF_TERM atom_free_b;
ERL_NIF_TERM atom_used_b;
ERL_NIF_TERM atom_eos;
ERL_NIF_TERM atom_decode_failed;
ERL_NIF_TERM atom_decode_timeout;
ERL_NIF_TERM atom_decode_aborted;

/* Forward decl: build_default_greedy_chain is defined in the sampler
 * section but used as a lazy fallback in nif_decode_one. */

ErlNifResourceType *MODEL_RT;
ErlNifResourceType *CTX_RT;
ErlNifResourceType *ADAPTER_RT;
ErlNifResourceType *SAMPLER_RT;

/* Drop the context's (or adapter's) reference on its model; if a
 * previous free_model/1 returned {ok, deferred} and the model is now
 * unreferenced by both contexts and adapters, actually free the
 * underlying llama_model* here. The decision is made under the lock
 * so concurrent destructions can't double-free. The free itself
 * runs while the lock is still held to keep the pointer
 * non-observable mid-teardown.
 *
 * Adapters share this gating because llama_model_free implicitly
 * frees any adapter that wasn't explicitly freed; if we freed the
 * model with an adapter wrapper still holding a (now dangling)
 * llama_adapter_lora* the next adapter_dtor would crash. */
/* True when nothing borrows the model any more; caller holds m->mu. */
static int model_unreferenced(const erllama_model_t *m) {
    return m->active_contexts == 0 && m->active_adapters == 0 &&
           m->active_mtmd == 0;
}

void context_drops_model(erllama_model_t *m) {
    pthread_mutex_lock(&m->mu);
    if (m->active_contexts > 0) {
        m->active_contexts--;
    }
    if (m->release_pending && model_unreferenced(m) && m->model) {
        (void) erllama_safe_model_free(m->model);
        m->model = NULL;
        m->release_pending = 0;
    }
    pthread_mutex_unlock(&m->mu);
}

static void model_drops_adapter(erllama_model_t *m) {
    pthread_mutex_lock(&m->mu);
    if (m->active_adapters > 0) {
        m->active_adapters--;
    }
    if (m->release_pending && model_unreferenced(m) && m->model) {
        (void) erllama_safe_model_free(m->model);
        m->model = NULL;
        m->release_pending = 0;
    }
    pthread_mutex_unlock(&m->mu);
}

/* Same gating for the multimodal projector (called from the mtmd
 * resource destructor in erllama_mtmd_nif.cpp). */
void model_drops_mtmd(erllama_model_t *m) {
    pthread_mutex_lock(&m->mu);
    if (m->active_mtmd > 0) {
        m->active_mtmd--;
    }
    if (m->release_pending && model_unreferenced(m) && m->model) {
        (void) erllama_safe_model_free(m->model);
        m->model = NULL;
        m->release_pending = 0;
    }
    pthread_mutex_unlock(&m->mu);
}

/* Resource destructors run when the BEAM has no remaining references.
 * They must tolerate partial init: if alloc succeeded but mutex_init
 * failed, the dtor sees mu_inited=0 and skips pthread_mutex_destroy.
 * Pointer fields are zero-init'd by the allocation path so freeing a
 * NULL is a no-op here.
 *
 * Two accepted tradeoffs callers should know about:
 *
 *  1. A throwing llama destructor leaks the native object. C++
 *     destructors are required to be `noexcept`; if one throws
 *     anyway, the safe wrapper catches the exception and returns
 *     -1 but we still NULL the pointer so the destructor cannot
 *     be called twice. The native model/context is leaked rather
 *     than risking UB. Fix lives upstream in llama.cpp.
 *
 *  2. GC-triggered dtors run on the scheduler thread that
 *     triggered GC, not on a dirty scheduler. For prompt cleanup
 *     of a multi-MB model, callers should prefer
 *     `erllama:unload/1` (which terminates the per-model
 *     gen_statem and goes through `nif_free_context` -- a dirty
 *     CPU NIF) over relying on Erlang GC to destruct the
 *     resource. */
static void model_dtor(ErlNifEnv *env, void *obj) {
    (void) env;
    erllama_model_t *m = (erllama_model_t *) obj;
    /* The pointer is NULL after any successful or failed explicit
     * release, so this single check covers both paths and avoids
     * double-calling the safe wrapper. */
    if (m->model) {
        (void) erllama_safe_model_free(m->model);
        m->model = NULL;
    }
    if (m->tbo) {
        enif_free(m->tbo);
        m->tbo = NULL;
    }
    if (m->mu_inited) {
        pthread_mutex_destroy(&m->mu);
        m->mu_inited = 0;
    }
}

static void ctx_dtor(ErlNifEnv *env, void *obj) {
    (void) env;
    erllama_context_t *c = (erllama_context_t *) obj;
    grammar_cache_clear(c);
    if (c->smpl) {
        (void) erllama_safe_sampler_free(c->smpl);
        c->smpl = NULL;
    }
    if (c->ctx) {
        (void) erllama_safe_free(c->ctx);
        c->ctx = NULL;
    }
    if (c->model_res) {
        context_drops_model(c->model_res);
        enif_release_resource(c->model_res);
        c->model_res = NULL;
    }
    if (c->mu_inited) {
        pthread_mutex_destroy(&c->mu);
        c->mu_inited = 0;
    }
}

/* Sampler chain destructor. Frees the chain (which may be NULL if
 * the user called sampler_free explicitly) and drops the
 * keep-reference on the owning context. */
static void sampler_dtor(ErlNifEnv *env, void *obj) {
    (void) env;
    erllama_sampler_t *s = (erllama_sampler_t *) obj;
    if (s->chain) {
        (void) erllama_safe_sampler_free(s->chain);
        s->chain = NULL;
    }
    if (s->ctx_res) {
        enif_release_resource(s->ctx_res);
        s->ctx_res = NULL;
    }
    if (s->mu_inited) {
        pthread_mutex_destroy(&s->mu);
        s->mu_inited = 0;
    }
}

/* Adapter destructor. Explicit nif_adapter_free zeroes
 * a->adapter under the lock, so this destructor is either a no-op
 * (already freed) or the implicit final cleanup. Either way it
 * decrements the model's adapter count via model_drops_adapter
 * (which may complete a deferred free_model/1) and releases the
 * keep-reference on the model. */
static void adapter_dtor(ErlNifEnv *env, void *obj) {
    (void) env;
    erllama_adapter_t *a = (erllama_adapter_t *) obj;
    if (a->adapter) {
        erllama_safe_adapter_lora_free(a->adapter);
        a->adapter = NULL;
    }
    if (a->model_res) {
        model_drops_adapter(a->model_res);
        enif_release_resource(a->model_res);
        a->model_res = NULL;
    }
    if (a->mu_inited) {
        pthread_mutex_destroy(&a->mu);
        a->mu_inited = 0;
    }
}

/* =========================================================================
 * Load callback
 * ========================================================================= */

static int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    (void) priv_data;
    (void) load_info;

    if (erllama_crc32c_init() != 0) {
        return -1;
    }
    /* llama_backend_init() is deferred to first model load via
     * erllama_safe_backend_init_once(). NIF load only sets up
     * resources and atoms. Cache-only and cache-test workloads
     * never invoke ggml_backend_load_all, which on some platforms
     * (notably FreeBSD when paired with another NIF that uses
     * mmap and signal handlers) perturbs process state in ways
     * that break unrelated code paths. */

    atom_ok = enif_make_atom(env, "ok");
    atom_error = enif_make_atom(env, "error");
    atom_load_failed = enif_make_atom(env, "load_failed");
    atom_malformed_gguf = enif_make_atom(env, "malformed_gguf");
    atom_context_failed = enif_make_atom(env, "context_failed");
    atom_tokenize_failed = enif_make_atom(env, "tokenize_failed");
    atom_pack_failed = enif_make_atom(env, "pack_failed");
    atom_unpack_failed = enif_make_atom(env, "unpack_failed");
    atom_true = enif_make_atom(env, "true");
    atom_false = enif_make_atom(env, "false");
    atom_released = enif_make_atom(env, "released");
    atom_too_large = enif_make_atom(env, "too_large");
    atom_invalid_token = enif_make_atom(env, "invalid_token");
    atom_context_overflow = enif_make_atom(env, "context_overflow");
    atom_batch_overflow = enif_make_atom(env, "batch_overflow");
    atom_oom = enif_make_atom(env, "oom");
    atom_deferred = enif_make_atom(env, "deferred");
    atom_exception = enif_make_atom(env, "exception");
    atom_no_logits = enif_make_atom(env, "no_logits");
    atom_no_template = enif_make_atom(env, "no_template");
    atom_template_failed = enif_make_atom(env, "template_failed");
    atom_grammar_failed = enif_make_atom(env, "grammar_failed");
    atom_embed_failed = enif_make_atom(env, "embed_failed");
    atom_not_supported = enif_make_atom(env, "not_supported");
    atom_invalid_content = enif_make_atom(env, "invalid_content");
    atom_no_gpu = enif_make_atom(env, "no_gpu");
    atom_total_b = enif_make_atom(env, "total_b");
    atom_free_b = enif_make_atom(env, "free_b");
    atom_used_b = enif_make_atom(env, "used_b");
    atom_eos = enif_make_atom(env, "eos");
    atom_decode_failed = enif_make_atom(env, "decode_failed");
    atom_decode_timeout = enif_make_atom(env, "decode_timeout");
    atom_decode_aborted = enif_make_atom(env, "decode_aborted");

    MODEL_RT = enif_open_resource_type(
        env, NULL, "erllama_model", model_dtor, ERL_NIF_RT_CREATE, NULL);
    if (!MODEL_RT) {
        return -1;
    }

    CTX_RT = enif_open_resource_type(
        env, NULL, "erllama_context", ctx_dtor, ERL_NIF_RT_CREATE, NULL);
    if (!CTX_RT) {
        return -1;
    }

    ADAPTER_RT = enif_open_resource_type(
        env, NULL, "erllama_adapter", adapter_dtor, ERL_NIF_RT_CREATE, NULL);
    if (!ADAPTER_RT) {
        return -1;
    }

    SAMPLER_RT = enif_open_resource_type(
        env, NULL, "erllama_sampler", sampler_dtor, ERL_NIF_RT_CREATE, NULL);
    if (!SAMPLER_RT) {
        return -1;
    }

    /* Register the chat-autoparser resources (defined in
     * erllama_chat_nif.cpp). */
    if (chat_nif_load(env) != 0) {
        return -1;
    }

    /* Register the ngram-speculator resource (defined in
     * erllama_spec_nif.cpp). */
    if (spec_nif_load(env) != 0) {
        return -1;
    }

    /* Register the multimodal projector resource (defined in
     * erllama_mtmd_nif.cpp). */
    if (mtmd_nif_load(env) != 0) {
        return -1;
    }

    return 0;
}

static void unload(ErlNifEnv *env, void *priv_data) {
    (void) env;
    (void) priv_data;
    /* Clear the log callback first: llama_log_set is process-global
     * and points at a function inside this .so. If we left it set,
     * a post-unload log emission would dispatch into freed memory.
     *
     * We intentionally do NOT call llama_backend_free. The
     * backend init runs once per process behind a pthread_once
     * guard whose static state lives in the .so; on a hot-upgrade
     * reload that keeps the .so mapped, the once token would still
     * read "completed" while the global llama state is gone,
     * leaving no path to re-init. The only thing backend_free
     * releases is the ggml quantize table (a small fixed
     * allocation reclaimed by the OS on process exit). */
    erllama_safe_log_unset();
}

/* =========================================================================
 * crc32c
 * ========================================================================= */

static ERL_NIF_TERM nif_crc32c(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    ErlNifBinary bin;
    if (!enif_inspect_iolist_as_binary(env, argv[0], &bin)) {
        return enif_make_badarg(env);
    }
    uint32_t crc = erllama_crc32c_update(0, bin.data, bin.size);
    return enif_make_uint(env, crc);
}

/* =========================================================================
 * fsync_dir (existing)
 * ========================================================================= */

static ERL_NIF_TERM make_errno_atom(ErlNifEnv *env, int e) {
    const char *name;
    switch (e) {
        case EACCES:    name = "eacces";    break;
        case EBUSY:     name = "ebusy";     break;
        case EEXIST:    name = "eexist";    break;
        case EINVAL:    name = "einval";    break;
        case EIO:       name = "eio";       break;
        case EISDIR:    name = "eisdir";    break;
        case ELOOP:     name = "eloop";     break;
        case EMFILE:    name = "emfile";    break;
        case ENAMETOOLONG: name = "enametoolong"; break;
        case ENFILE:    name = "enfile";    break;
        case ENOENT:    name = "enoent";    break;
        case ENOMEM:    name = "enomem";    break;
        case ENOSPC:    name = "enospc";    break;
        case ENOTDIR:   name = "enotdir";   break;
        case EPERM:     name = "eperm";     break;
        case EROFS:     name = "erofs";     break;
#ifdef EINTEGRITY
        /* FreeBSD fsync(2) returns EINTEGRITY on filesystem
         * integrity errors (ZFS checksum failure, ufs2 sb
         * mismatch). Surface it instead of mapping to "unknown". */
        case EINTEGRITY: name = "eintegrity"; break;
#endif
        default:        name = "unknown";   break;
    }
    return enif_make_atom(env, name);
}

static ERL_NIF_TERM nif_fsync_dir(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    char path[4097];
    /* copy_path rejects empty inputs, oversize inputs, and embedded
     * NUL bytes (which would otherwise let `<<"a\0b">>` be passed to
     * open() as just "a"). */
    if (!copy_path(env, argv[0], path, sizeof(path))) {
        return enif_make_badarg(env);
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return erllama_error(env, make_errno_atom(env, errno));
    }
    int rc = fsync(fd);
    int saved = errno;
    close(fd);
    if (rc != 0) {
        return erllama_error(env, make_errno_atom(env, saved));
    }
    return atom_ok;
}

/* Register / clear the process receiving forwarded native llama.cpp
 * log lines as `{llama_log, LevelInt, TextBin}` messages. MinLevel
 * uses the ggml numeric levels (DEBUG 1 .. ERROR 4). */
static ERL_NIF_TERM nif_set_log_receiver(ErlNifEnv *env, int argc,
                                         const ERL_NIF_TERM argv[]) {
    (void) argc;
    ErlNifPid pid;
    int lvl;
    if (!enif_get_local_pid(env, argv[0], &pid) ||
        !enif_get_int(env, argv[1], &lvl)) {
        return enif_make_badarg(env);
    }
    erllama_safe_set_log_receiver(pid, lvl);
    return atom_ok;
}

static ERL_NIF_TERM nif_clear_log_receiver(ErlNifEnv *env, int argc,
                                           const ERL_NIF_TERM argv[]) {
    (void) env;
    (void) argc;
    (void) argv;
    erllama_safe_clear_log_receiver();
    return atom_ok;
}

static ErlNifFunc nif_funcs[] = {
    {"nif_crc32c",       1, nif_crc32c,       ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_vram_info",    0, nif_vram_info,    ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_list_devices", 0, nif_list_devices, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_model_size",   1, nif_model_size,   ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_model_n_layer",1, nif_model_n_layer,ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_model_family", 1, nif_model_family, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_vocab_info",   1, nif_vocab_info,   ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_grammar_cache_stats", 1, nif_grammar_cache_stats, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_forward_with_argmax", 2, nif_forward_with_argmax,
        ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_pack",      3, nif_kv_pack,      ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_pack",      4, nif_kv_pack,      ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_unpack",    3, nif_kv_unpack,    ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_seq_rm",    4, nif_kv_seq_rm,    ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_seq_cp",    3, nif_kv_seq_cp,    ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_set_log_receiver",   2, nif_set_log_receiver,   0},
    {"nif_clear_log_receiver", 0, nif_clear_log_receiver, 0},
    {"nif_step",         2, nif_step,         ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_fsync_dir",    1, nif_fsync_dir,    ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_load_model",   2, nif_load_model,   ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_free_model",   1, nif_free_model,   ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_new_context",  2, nif_new_context,  ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_free_context", 1, nif_free_context, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_request_abort", 1, nif_request_abort, 0},
    {"nif_tokenize",     3, nif_tokenize,     ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_prefill",      2, nif_prefill,      ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_decode_one",   1, nif_decode_one,   ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_detokenize",   2, nif_detokenize,   ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_detokenize",   3, nif_detokenize_opts, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_apply_chat_template", 2, nif_apply_chat_template, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_embed",        2, nif_embed,        ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_set_grammar",  2, nif_set_grammar,  ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_configure_sampler", 2, nif_configure_sampler,
        ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_clear_sampler", 1, nif_clear_sampler, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_adapter_load", 2, nif_adapter_load, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_adapter_free", 1, nif_adapter_free, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_set_adapters", 2, nif_set_adapters, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_sampler_new",  2, nif_sampler_new,  ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_sampler_free", 1, nif_sampler_free, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_chat_templates_init",  2, nif_chat_templates_init,
        ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_chat_templates_apply", 2, nif_chat_templates_apply,
        ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_chat_parse",           3, nif_chat_parse,
        ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_spec_new",    1, nif_spec_new,    ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_spec_begin",  3, nif_spec_begin,  ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_spec_draft",  4, nif_spec_draft,  ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_spec_accept", 3, nif_spec_accept, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_spec_free",   1, nif_spec_free,   ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_spec_step",   4, nif_spec_step,   ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_mtmd_init",      3, nif_mtmd_init,      ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_mtmd_free",      1, nif_mtmd_free,      ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_mtmd_caps",      1, nif_mtmd_caps,      ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_mtmd_caps_file", 1, nif_mtmd_caps_file, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_media_prefill",  6, nif_media_prefill,  ERL_NIF_DIRTY_JOB_CPU_BOUND}
};

ERL_NIF_INIT(erllama_nif, nif_funcs, load, NULL, NULL, unload)
