/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */
/* Text <-> token conversion: tokenize, the cache-keying
 * detokenize loop (byte-stable), and the options-aware
 * detokenize. */

#include "erllama_nif_int.h"
#include "erllama_safe.h"

#include <math.h>
#include <string.h>

/* =========================================================================
 * Tokenize
 * ========================================================================= */

ERL_NIF_TERM nif_tokenize(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    ErlNifBinary text;
    if (!enif_inspect_iolist_as_binary(env, argv[1], &text)) {
        return enif_make_badarg(env);
    }
    /* NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result) */
    if (text.size > (size_t) ERLLAMA_MAX_TOKEN_TEXT) {
        return erllama_error(env, atom_too_large);
    }
    if (!enif_is_map(env, argv[2])) {
        return enif_make_badarg(env);
    }

    int add_special = 1;
    int parse_special = 0;
    int b;
    if (get_map_bool(env, argv[2], "add_special", &b)) add_special = b;
    if (get_map_bool(env, argv[2], "parse_special", &b)) parse_special = b;

    if (!erllama_lock_model_live(m)) {
        return erllama_error(env, atom_released);
    }
    const struct llama_vocab *vocab = erllama_model_vocab(m, NULL);
    if (!vocab) {
        pthread_mutex_unlock(&m->mu);
        return erllama_error(env, atom_exception);
    }

    llama_token *tokens = NULL;
    int32_t n = 0;
    erllama_tok_status_t st = erllama_tokenize_grow(
        vocab, (const char *) text.data, (int32_t) text.size,
        add_special ? true : false, parse_special ? true : false,
        &tokens, &n);
    pthread_mutex_unlock(&m->mu);
    switch (st) {
        case ERLLAMA_TOK_OOM:
            return erllama_error(env, atom_oom);
        case ERLLAMA_TOK_EXCEPTION:
            return erllama_error(env, atom_exception);
        case ERLLAMA_TOK_TOO_LARGE:
            return erllama_error(env, atom_too_large);
        case ERLLAMA_TOK_FAILED:
            return erllama_error(env, atom_tokenize_failed);
        case ERLLAMA_TOK_OK:
            break;
    }

    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        list = enif_make_list_cell(env, enif_make_int(env, tokens[i]), list);
    }
    enif_free(tokens);
    return list;
}

/* Options-aware detokenize: `nif_detokenize(Model, Tokens, Opts)`
 * with `#{remove_special => boolean(), unparse_special => boolean()}`
 * (both default false), backed by llama_detokenize. Kept SEPARATE
 * from the arity-2 per-token loop below: the cache byte-keys are
 * computed over the arity-2 output and must stay byte-identical. */
ERL_NIF_TERM nif_detokenize_opts(ErlNifEnv *env, int argc,
                                        const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    if (!enif_is_map(env, argv[2])) {
        return enif_make_badarg(env);
    }
    int remove_special = 0;
    int unparse_special = 0;
    {
        int b;
        if (get_map_bool(env, argv[2], "remove_special", &b)) remove_special = b;
        if (get_map_bool(env, argv[2], "unparse_special", &b)) unparse_special = b;
    }
    llama_token *tokens = NULL;
    int32_t n = 0;
    int rc = read_token_list(env, argv[1], &tokens, &n);
    if (rc != 1) return token_list_error(env, rc);
    if (n == 0) {
        if (tokens) enif_free(tokens);
        ERL_NIF_TERM empty;
        if (!erllama_empty_bin(env, &empty)) {
            return erllama_error(env, atom_oom);
        }
        return empty;
    }

    if (!erllama_lock_model_live(m)) {
        enif_free(tokens);
        return erllama_error(env, atom_released);
    }
    int32_t n_vocab = 0;
    const struct llama_vocab *vocab = erllama_model_vocab(m, &n_vocab);
    if (!vocab || n_vocab <= 0) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return erllama_error(env, atom_invalid_token);
    }
    if (erllama_first_oob_token(tokens, n, n_vocab) >= 0) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return erllama_error(env, atom_invalid_token);
    }
    /* Two-pass resize: a negative return is the needed size; the
     * retry may legitimately return FEWER bytes than requested
     * (upstream trims whitespace after per-token detokenization), so
     * the final return value sizes the result binary. */
    int32_t cap = n * 32 + 16;
    char *buf = enif_alloc((size_t) cap);
    if (!buf) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return erllama_error(env, atom_oom);
    }
    int32_t wrote = erllama_safe_detokenize(vocab, tokens, n, buf, cap,
                                            remove_special ? true : false,
                                            unparse_special ? true : false);
    if (wrote < 0 && wrote != INT32_MIN) {
        int32_t need = -wrote;
        char *bigger = enif_realloc(buf, (size_t) need);
        if (!bigger) {
            pthread_mutex_unlock(&m->mu);
            enif_free(buf);
            enif_free(tokens);
            return erllama_error(env, atom_oom);
        }
        buf = bigger;
        cap = need;
        wrote = erllama_safe_detokenize(vocab, tokens, n, buf, cap,
                                        remove_special ? true : false,
                                        unparse_special ? true : false);
    }
    pthread_mutex_unlock(&m->mu);
    enif_free(tokens);
    if (wrote < 0) {
        enif_free(buf);
        return erllama_error(env, atom_exception);
    }
    ERL_NIF_TERM bin;
    if (!erllama_bin_from(env, buf, (size_t) wrote, &bin)) {
        enif_free(buf);
        return erllama_error(env, atom_oom);
    }
    enif_free(buf);
    return bin;
}

ERL_NIF_TERM nif_detokenize(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    llama_token *tokens = NULL;
    int32_t n = 0;
    int rc = read_token_list(env, argv[1], &tokens, &n);
    if (rc != 1) return token_list_error(env, rc);
    if (n == 0) {
        if (tokens) enif_free(tokens);
        ERL_NIF_TERM empty;
        if (!erllama_empty_bin(env, &empty)) {
            return erllama_error(env, atom_oom);
        }
        return empty;
    }

    if (!erllama_lock_model_live(m)) {
        enif_free(tokens);
        return erllama_error(env, atom_released);
    }
    int32_t n_vocab = 0;
    const struct llama_vocab *vocab = erllama_model_vocab(m, &n_vocab);
    if (!vocab) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return erllama_error(env, atom_exception);
    }
    /* Fail closed if the vocab lookup gave us no usable size: without
     * n_vocab we cannot validate token IDs. Mirrors the prefill
     * path; validate before any token_to_piece call. */
    if (n_vocab <= 0 || erllama_first_oob_token(tokens, n, n_vocab) >= 0) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return erllama_error(env, atom_invalid_token);
    }

    /* Per-token piece, concatenated. Pieces are typically a handful
     * of bytes; we grow the buffer on demand and re-call
     * llama_token_to_piece with a sized buffer when 256 bytes isn't
     * enough (it returns the negative needed size). The safe wrapper
     * returns INT32_MIN on a thrown C++ exception. */
    char small_piece[256];
    /* Guard the size computation: clamp n to a sane upper bound so
     * gcc's range analysis can prove cap fits. 16M tokens is far
     * beyond any realistic prompt; reject earlier rather than
     * overflow. */
    if (n < 0 || n > (1 << 24)) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return erllama_error(env, atom_too_large);
    }
    size_t cap = (size_t) n * 32u + 16u;
    char *out = enif_alloc(cap);
    if (!out) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return erllama_error(env, atom_oom);
    }
    size_t used = 0;
    int err = 0;
    for (int32_t i = 0; i < n; i++) {
        char *piece_buf = small_piece;
        int32_t buf_size = (int32_t) sizeof(small_piece);
        char *grown = NULL;
        int32_t got = erllama_safe_token_to_piece(
            vocab, tokens[i], piece_buf, buf_size, 0, false);
        if (got == INT32_MIN) {
            err = 1;
            break;
        }
        if (got < 0) {
            int32_t need = -got;
            if (need <= 0 || need > (1 << 20)) {
                err = 1;
                break;
            }
            grown = enif_alloc((size_t) need);
            if (!grown) { err = 2; break; }
            piece_buf = grown;
            got = erllama_safe_token_to_piece(
                vocab, tokens[i], piece_buf, need, 0, false);
            if (got == INT32_MIN || got < 0) {
                enif_free(grown);
                err = 1;
                break;
            }
        }
        if (used + (size_t) got > cap) {
            size_t new_cap = (used + (size_t) got) * 2 + 16;
            char *new_out = enif_realloc(out, new_cap);
            if (!new_out) {
                if (grown) enif_free(grown);
                err = 2;
                break;
            }
            out = new_out;
            cap = new_cap;
        }
        memcpy(out + used, piece_buf, (size_t) got);
        used += (size_t) got;
        if (grown) enif_free(grown);
    }
    pthread_mutex_unlock(&m->mu);
    enif_free(tokens);
    if (err) {
        enif_free(out);
        if (err == 2) return erllama_error(env, atom_oom);
        return erllama_error(env, atom_invalid_token);
    }

    ERL_NIF_TERM outbin;
    if (!erllama_bin_from(env, out, used, &outbin)) {
        enif_free(out);
        return erllama_error(env, atom_oom);
    }
    enif_free(out);
    return outbin;
}
