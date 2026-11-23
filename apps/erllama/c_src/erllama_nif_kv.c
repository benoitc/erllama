/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */
/* KV persistence and per-sequence memory ops: pack/unpack for the
 * tiered cache, seq_rm trims, seq_cp session forks. */

#include "erllama_nif_int.h"
#include "erllama_safe.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef ERLLAMA_LLAMA_TAG
#error "ERLLAMA_LLAMA_TAG must be set from c_src/llama.cpp/.version"
#endif

/* Identifies the byte layout kv_pack produces: the vendored llama.cpp
 * tag plus its state-seq format version. The cache folds it into
 * ctx_params_hash so rows saved by another llama.cpp never reach
 * kv_unpack; the tag is included because upstream does not always
 * bump LLAMA_STATE_SEQ_VERSION when the layout changes. */
ERL_NIF_TERM nif_kv_format(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    (void) argv;
    char buf[128];
    int n = snprintf(buf, sizeof(buf), "llama.cpp/%s/seq-v%d",
                     ERLLAMA_LLAMA_TAG, (int) LLAMA_STATE_SEQ_VERSION);
    if (n < 0 || (size_t) n >= sizeof(buf)) {
        return enif_make_badarg(env);
    }
    ERL_NIF_TERM bin;
    unsigned char *p = enif_make_new_binary(env, (size_t) n, &bin);
    memcpy(p, buf, (size_t) n);
    return bin;
}

/* =========================================================================
 * KV pack / unpack
 *
 * The 3-arg signatures preserve the v0.1 stub API. The Tokens and
 * NTokens / SeqId positional args are interpreted as documented in
 * include/llama.h: NTokens is unused (the in-memory API saves the
 * full state for the configured seq_id, defaulting to 0); SeqId is
 * the destination sequence id for unpack.
 * ========================================================================= */

/* Pack `seq_id''s KV state of `c' (the caller checked the context's
 * kind). */
static ERL_NIF_TERM kv_pack_seq(ErlNifEnv *env, erllama_context_t *c, llama_seq_id seq_id) {
    if (!erllama_lock_ctx(c)) {
        return erllama_error(env, atom_released);
    }
    size_t need = erllama_safe_state_seq_get_size(c->ctx, seq_id);
    if (need == SIZE_MAX) {
        pthread_mutex_unlock(&c->mu);
        return erllama_error(env, atom_exception);
    }
    if (need == 0) {
        pthread_mutex_unlock(&c->mu);
        ERL_NIF_TERM empty;
        if (!erllama_empty_bin(env, &empty)) {
            return erllama_error(env, atom_oom);
        }
        return empty;
    }
    ErlNifBinary out;
    if (!enif_alloc_binary(need, &out)) {
        pthread_mutex_unlock(&c->mu);
        return erllama_error(env, atom_oom);
    }
    size_t written = erllama_safe_state_seq_get_data(
        c->ctx, out.data, out.size, seq_id);
    pthread_mutex_unlock(&c->mu);
    if (written == SIZE_MAX) {
        enif_release_binary(&out);
        return erllama_error(env, atom_exception);
    }
    if (written == 0 || written > need) {
        enif_release_binary(&out);
        return erllama_error(env, atom_pack_failed);
    }
    if (written < need) {
        if (!enif_realloc_binary(&out, written)) {
            enif_release_binary(&out);
            return erllama_error(env, atom_oom);
        }
    }
    return enif_make_binary(env, &out);
}

/* Restore `in' into `seq_id' of `c' (the caller checked the context's
 * kind). */
static ERL_NIF_TERM kv_unpack_seq(ErlNifEnv *env, erllama_context_t *c, const ErlNifBinary *in,
                                  int seq_id) {
    if (!erllama_lock_ctx(c)) {
        return erllama_error(env, atom_released);
    }
    size_t consumed = erllama_safe_state_seq_set_data(
        c->ctx, in->data, in->size, seq_id);
    /* kv_unpack only restores KV cells, not the per-context logits
     * buffer; the model layer must drop the last cell and re-prefill
     * it before the next sample. Mark the context as not ready until
     * that primer runs. */
    c->decode_ready = 0;
    /* Refresh per-seq tracking from the just-restored KV: next_pos is
     * the position immediately past the highest cell now in `seq_id`.
     * last_logits_idx stays at -1 because state_seq_set_data does not
     * carry logits with it. nif_step refuses to sample a seq with
     * last_logits_idx == -1 — the model layer's primer-prefill flow
     * is the only legal next operation for this seq. */
    if (seq_id < ERLLAMA_N_SEQ_MAX_CAP) {
        long pos_max = erllama_safe_memory_seq_pos_max(c->ctx, seq_id);
        c->per_seq[seq_id].next_pos =
            pos_max < 0 ? 0 : (int32_t) (pos_max + 1);
        c->per_seq[seq_id].last_logits_idx = -1;
    }
    pthread_mutex_unlock(&c->mu);
    if (consumed == 0 || consumed != in->size) {
        return erllama_error(env, atom_unpack_failed);
    }
    return atom_ok;
}

ERL_NIF_TERM nif_kv_pack(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    /* A split model's KV is packed on every rank at once, through its
     * serving rank (nif_tp_kv_pack). */
    if (ERLLAMA_IS_TP(c)) {
        return erllama_error(env, atom_not_supported);
    }
    /* Tokens (argv[1]) is informational; NTokens (argv[2]) ignored.
     * The model layer must have prefilled exactly the desired prefix
     * before calling kv_pack. argv[3], when present (arity 4),
     * specifies which sequence to extract from. Default 0 keeps
     * existing 3-arity callers working. */
    llama_seq_id seq_id = 0;
    if (argc == 4) {
        int sid;
        if (!enif_get_int(env, argv[3], &sid) || sid < 0) {
            return enif_make_badarg(env);
        }
        seq_id = (llama_seq_id) sid;
    }

    return kv_pack_seq(env, c, seq_id);
}

ERL_NIF_TERM nif_kv_unpack(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    /* A split model's KV is restored on every rank at once
     * (nif_tp_kv_unpack). */
    if (ERLLAMA_IS_TP(c)) {
        return erllama_error(env, atom_not_supported);
    }
    ErlNifBinary in;
    if (!enif_inspect_binary(env, argv[1], &in)) {
        return enif_make_badarg(env);
    }
    int seq_id;
    if (!enif_get_int(env, argv[2], &seq_id) || seq_id < 0) {
        return enif_make_badarg(env);
    }
    return kv_unpack_seq(env, c, &in, seq_id);
}

/* Remove the cells in [p0, p1) from the given sequence. p0 < 0 means
 * 0; p1 < 0 means infinity. Returns ok or {error, partial}. The save
 * format only stores KV cells; the per-context logits buffer is not
 * restored. So after kv_unpack the model layer drops the last cell of
 * the saved sequence and re-prefills the corresponding token to
 * regenerate logits for the next sample. */
ERL_NIF_TERM nif_kv_seq_rm(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    int seq_id, p0, p1;
    if (!enif_get_int(env, argv[1], &seq_id) || seq_id < 0 ||
        !enif_get_int(env, argv[2], &p0) ||
        !enif_get_int(env, argv[3], &p1)) {
        return enif_make_badarg(env);
    }
    if (!erllama_lock_ctx(c)) {
        return erllama_error(env, atom_released);
    }
    if (c->tp_sink_set) {
        ErlNifEnv *menv = enif_alloc_env();
        erllama_tp_send(env, c, menv,
                        enif_make_tuple4(menv, enif_make_atom(menv, "seq_rm"), enif_make_int(menv, seq_id),
                                         enif_make_int(menv, p0), enif_make_int(menv, p1)));
    }
    int rc = erllama_safe_memory_seq_rm(c->ctx, seq_id, p0, p1);
    /* Removing cells invalidates last-decode logits; force a fresh
     * prefill before the next sample. */
    c->decode_ready = 0;
    /* Refresh per-seq tracking: query the remaining max position
     * for this seq and recompute next_pos. last_logits_idx is
     * cleared because the prior batch's logits no longer correspond
     * to the seq's tail. Refresh on FAILURE too: recurrent / hybrid
     * memories refuse partial removals (unless RS rollback covers
     * them) while keeping all their cells, and Erlang must see the
     * real next_pos to fall back correctly instead of re-prefilling
     * a token at a position the memory still holds. */
    if (seq_id < ERLLAMA_N_SEQ_MAX_CAP) {
        long pos_max = erllama_safe_memory_seq_pos_max(c->ctx, seq_id);
        c->per_seq[seq_id].next_pos =
            pos_max < 0 ? 0 : (int32_t) (pos_max + 1);
        c->per_seq[seq_id].last_logits_idx = -1;
    }
    pthread_mutex_unlock(&c->mu);
    if (rc != 0) {
        return erllama_error(env, atom_unpack_failed);
    }
    return atom_ok;
}

/* Full-sequence KV copy (session fork):
 *   nif_kv_seq_cp(CtxRef, SrcSeq, DstSeq) -> ok | {error, atom()}
 * llama_memory_seq_cp is void with several silent no-op paths, so
 * success is verified via pos_max(dst) == pos_max(src); on mismatch
 * the destination is wiped and {error, seq_cp_failed} returned. The
 * copy carries no logits: the destination behaves exactly like a
 * freshly kv_unpack'ed seq (a suffix prefill must run before it can
 * sample). */
ERL_NIF_TERM nif_kv_seq_cp(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    int src, dst;
    if (!enif_get_int(env, argv[1], &src) || src < 0 ||
        src >= ERLLAMA_N_SEQ_MAX_CAP ||
        !enif_get_int(env, argv[2], &dst) || dst < 0 ||
        dst >= ERLLAMA_N_SEQ_MAX_CAP || src == dst) {
        return enif_make_badarg(env);
    }
    if (!erllama_lock_ctx(c)) {
        return erllama_error(env, atom_released);
    }
    long src_max = erllama_safe_memory_seq_pos_max(c->ctx, src);
    if (src_max < 0) {
        /* Empty (or unreadable) source: nothing to fork. */
        pthread_mutex_unlock(&c->mu);
        return erllama_error(env, enif_make_atom(env, "seq_cp_failed"));
    }
    if (c->tp_sink_set) {
        ErlNifEnv *menv = enif_alloc_env();
        erllama_tp_send(env, c, menv,
                        enif_make_tuple3(menv, enif_make_atom(menv, "seq_cp"), enif_make_int(menv, src),
                                         enif_make_int(menv, dst)));
    }
    int rc = erllama_safe_memory_seq_cp(c->ctx, src, dst);
    long dst_max =
        rc == 0 ? erllama_safe_memory_seq_pos_max(c->ctx, dst) : -2;
    if (rc != 0 || dst_max != src_max) {
        if (c->tp_sink_set) {
            ErlNifEnv *menv = enif_alloc_env();
            erllama_tp_send(env, c, menv,
                            enif_make_tuple4(menv, enif_make_atom(menv, "seq_rm"), enif_make_int(menv, dst),
                                             enif_make_int(menv, 0), enif_make_int(menv, -1)));
        }
        (void) erllama_safe_memory_seq_rm(c->ctx, dst, 0, -1);
        c->per_seq[dst].next_pos = 0;
        c->per_seq[dst].last_logits_idx = -1;
        pthread_mutex_unlock(&c->mu);
        return erllama_error(env, enif_make_atom(env, "seq_cp_failed"));
    }
    c->decode_ready = 0;
    c->per_seq[dst].next_pos = (int32_t) (dst_max + 1);
    c->per_seq[dst].last_logits_idx = -1;
    pthread_mutex_unlock(&c->mu);
    return atom_ok;
}

/* =========================================================================
 * Checkpoints of a model split over a tensor-parallel group
 *
 * Each rank holds its share of the KV. A checkpoint is every rank's
 * share at the same committed token: the serving rank puts a command in
 * its sink's stream (nif_tp_send), after every batch it decoded, and
 * each rank packs or restores its own share (nif_tp_kv_pack,
 * nif_tp_kv_unpack) when it reaches that command. The shares are
 * gathered and checked in Erlang (erllama_tp_rank).
 * ========================================================================= */

/* nif_tp_send(Ctx, Cmd) -> ok | {error, not_supported}: send `Cmd' to
 * the serving rank's sink with the next sequence number. */
ERL_NIF_TERM nif_tp_send(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    if (!ERLLAMA_IS_TP(c) || !c->tp_sink_set) {
        return erllama_error(env, atom_not_supported);
    }
    if (!erllama_lock_ctx(c)) {
        return erllama_error(env, atom_released);
    }
    ErlNifEnv *menv = enif_alloc_env();
    erllama_tp_send(env, c, menv, enif_make_copy(menv, argv[1]));
    pthread_mutex_unlock(&c->mu);
    return atom_ok;
}

/* nif_tp_kv_pack(Ctx, SeqId) -> Bin | {error, Reason}: this rank's
 * share of `SeqId''s KV. */
ERL_NIF_TERM nif_tp_kv_pack(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    int sid;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c) ||
        !enif_get_int(env, argv[1], &sid) || sid < 0) {
        return enif_make_badarg(env);
    }
    if (!ERLLAMA_IS_TP(c)) {
        return erllama_error(env, atom_not_supported);
    }
    return kv_pack_seq(env, c, (llama_seq_id) sid);
}

/* nif_tp_kv_unpack(Ctx, Bin, SeqId) -> ok | {error, Reason}: restore
 * this rank's share of `SeqId''s KV. */
ERL_NIF_TERM nif_tp_kv_unpack(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    ErlNifBinary in;
    int sid;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c) ||
        !enif_inspect_binary(env, argv[1], &in) || !enif_get_int(env, argv[2], &sid) || sid < 0) {
        return enif_make_badarg(env);
    }
    if (!ERLLAMA_IS_TP(c)) {
        return erllama_error(env, atom_not_supported);
    }
    return kv_unpack_seq(env, c, &in, sid);
}

/* nif_kv_seq_pos_max(Ctx, SeqId) -> Pos | {error, Reason}: the highest
 * position the sequence holds, -1 when it is empty. */
ERL_NIF_TERM nif_kv_seq_pos_max(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    int seq;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c) || !enif_get_int(env, argv[1], &seq) ||
        seq < 0 || seq >= ERLLAMA_N_SEQ_MAX_CAP) {
        return enif_make_badarg(env);
    }
    if (!erllama_lock_ctx(c)) {
        return erllama_error(env, atom_released);
    }
    long p = erllama_safe_memory_seq_pos_max(c->ctx, seq);
    pthread_mutex_unlock(&c->mu);
    if (p == -2) {
        return erllama_error(env, atom_exception);
    }
    return enif_make_long(env, p);
}
