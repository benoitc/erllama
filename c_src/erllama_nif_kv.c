/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */
/* KV persistence and per-sequence memory ops: pack/unpack for the
 * tiered cache, seq_rm trims, seq_cp session forks. */

#include "erllama_nif_int.h"
#include "erllama_safe.h"

#include <math.h>
#include <string.h>

/* =========================================================================
 * KV pack / unpack
 *
 * The 3-arg signatures preserve the v0.1 stub API. The Tokens and
 * NTokens / SeqId positional args are interpreted as documented in
 * include/llama.h: NTokens is unused (the in-memory API saves the
 * full state for the configured seq_id, defaulting to 0); SeqId is
 * the destination sequence id for unpack.
 * ========================================================================= */

ERL_NIF_TERM nif_kv_pack(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
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

    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    size_t need = erllama_safe_state_seq_get_size(c->ctx, seq_id);
    if (need == SIZE_MAX) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (need == 0) {
        pthread_mutex_unlock(&c->mu);
        ErlNifBinary empty;
        if (!enif_alloc_binary(0, &empty)) {
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        return enif_make_binary(env, &empty);
    }
    ErlNifBinary out;
    if (!enif_alloc_binary(need, &out)) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    size_t written = erllama_safe_state_seq_get_data(
        c->ctx, out.data, out.size, seq_id);
    pthread_mutex_unlock(&c->mu);
    if (written == SIZE_MAX) {
        enif_release_binary(&out);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (written == 0 || written > need) {
        enif_release_binary(&out);
        return enif_make_tuple2(env, atom_error, atom_pack_failed);
    }
    if (written < need) {
        if (!enif_realloc_binary(&out, written)) {
            enif_release_binary(&out);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
    }
    return enif_make_binary(env, &out);
}

ERL_NIF_TERM nif_kv_unpack(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    ErlNifBinary in;
    if (!enif_inspect_binary(env, argv[1], &in)) {
        return enif_make_badarg(env);
    }
    int seq_id;
    if (!enif_get_int(env, argv[2], &seq_id) || seq_id < 0) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    size_t consumed = erllama_safe_state_seq_set_data(
        c->ctx, in.data, in.size, seq_id);
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
    if (consumed == 0 || consumed != in.size) {
        return enif_make_tuple2(env, atom_error, atom_unpack_failed);
    }
    return atom_ok;
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
    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
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
        return enif_make_tuple2(env, atom_error, atom_unpack_failed);
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
    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    long src_max = erllama_safe_memory_seq_pos_max(c->ctx, src);
    if (src_max < 0) {
        /* Empty (or unreadable) source: nothing to fork. */
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(
            env, atom_error, enif_make_atom(env, "seq_cp_failed"));
    }
    int rc = erllama_safe_memory_seq_cp(c->ctx, src, dst);
    long dst_max =
        rc == 0 ? erllama_safe_memory_seq_pos_max(c->ctx, dst) : -2;
    if (rc != 0 || dst_max != src_max) {
        (void) erllama_safe_memory_seq_rm(c->ctx, dst, 0, -1);
        c->per_seq[dst].next_pos = 0;
        c->per_seq[dst].last_logits_idx = -1;
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(
            env, atom_error, enif_make_atom(env, "seq_cp_failed"));
    }
    c->decode_ready = 0;
    c->per_seq[dst].next_pos = (int32_t) (dst_max + 1);
    c->per_seq[dst].last_logits_idx = -1;
    pthread_mutex_unlock(&c->mu);
    return atom_ok;
}
