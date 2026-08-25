/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */
/* Text <-> token conversion: tokenize, the cache-keying
 * detokenize loop (byte-stable), the options-aware detokenize,
 * and the shared token-list parsing helpers. */

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
        return enif_make_tuple2(env, atom_error, atom_too_large);
    }
    if (!enif_is_map(env, argv[2])) {
        return enif_make_badarg(env);
    }

    int add_special = 1;
    int parse_special = 0;
    int b;
    if (get_map_bool(env, argv[2], "add_special", &b)) add_special = b;
    if (get_map_bool(env, argv[2], "parse_special", &b)) parse_special = b;

    pthread_mutex_lock(&m->mu);
    if (!m->model || m->release_pending) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    const struct llama_vocab *vocab = erllama_safe_model_get_vocab(m->model);
    if (!vocab) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }

    int32_t text_len = (int32_t) text.size;
    /* `n_max = text_len + 8` is a sound upper bound: tokens-per-byte
     * for any tokenizer is `<= 1 + small constant`, the +8 covering
     * BOS/EOS specials. Clamping it down to ERLLAMA_MAX_TOKENS would
     * force an unnecessary retry on inputs over ~1 byte/token average
     * and cap production workloads below the input text cap. The
     * input text cap (ERLLAMA_MAX_TOKEN_TEXT, enforced above) is
     * what bounds the allocation. */
    int32_t n_max = text_len + 8;
    if (n_max < 16) n_max = 16;

    llama_token *tokens = (llama_token *) enif_alloc(sizeof(llama_token) * (size_t) n_max);
    if (!tokens) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    int32_t n = erllama_safe_tokenize(
        vocab, (const char *) text.data, text_len, tokens,
        n_max, add_special ? true : false, parse_special ? true : false);
    if (n == INT32_MIN) {
        enif_free(tokens);
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (n < 0) {
        int32_t needed = -n;
        if (needed > ERLLAMA_MAX_TOKENS) {
            enif_free(tokens);
            pthread_mutex_unlock(&m->mu);
            return enif_make_tuple2(env, atom_error, atom_too_large);
        }
        enif_free(tokens);
        tokens = (llama_token *) enif_alloc(sizeof(llama_token) * (size_t) needed);
        if (!tokens) {
            pthread_mutex_unlock(&m->mu);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        n = erllama_safe_tokenize(
            vocab, (const char *) text.data, text_len, tokens,
            needed, add_special ? true : false, parse_special ? true : false);
    }
    pthread_mutex_unlock(&m->mu);
    if (n == INT32_MIN) {
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (n < 0) {
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_tokenize_failed);
    }
    /* Enforce the output cap post-success: removing the n_max clamp
     * before the first call means the tokenizer can fully populate a
     * buffer larger than ERLLAMA_MAX_TOKENS (e.g. byte-fallback
     * tokenizers at ~1 byte/token on a 60 MiB input). Convert that
     * into a clean too_large error rather than returning an
     * over-cap list to Erlang. */
    if (n > ERLLAMA_MAX_TOKENS) {
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_too_large);
    }

    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        list = enif_make_list_cell(env, enif_make_int(env, tokens[i]), list);
    }
    enif_free(tokens);
    return list;
}

int read_token_list(ErlNifEnv *env, ERL_NIF_TERM list,
                           llama_token **out, int32_t *out_len) {
    unsigned int n;
    if (!enif_get_list_length(env, list, &n)) return 0;
    if (n > (unsigned int) ERLLAMA_MAX_TOKENS) return -2;
    if (n == 0) {
        *out = NULL;
        *out_len = 0;
        return 1;
    }
    llama_token *toks = enif_alloc(sizeof(llama_token) * (size_t) n);
    if (!toks) return -1;
    ERL_NIF_TERM head, tail = list;
    unsigned int i = 0;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        int v;
        if (!enif_get_int(env, head, &v)) {
            enif_free(toks);
            return 0;
        }
        if (v < 0) {
            enif_free(toks);
            return -3;
        }
        toks[i++] = (llama_token) v;
    }
    *out = toks;
    *out_len = (int32_t) n;
    return 1;
}

ERL_NIF_TERM token_list_error(ErlNifEnv *env, int rc) {
    switch (rc) {
        case -1: return enif_make_tuple2(env, atom_error, atom_oom);
        case -2: return enif_make_tuple2(env, atom_error, atom_too_large);
        case -3: return enif_make_tuple2(env, atom_error, atom_invalid_token);
        default: return enif_make_badarg(env);
    }
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
        ErlNifBinary empty;
        if (!enif_alloc_binary(0, &empty)) {
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        return enif_make_binary(env, &empty);
    }

    pthread_mutex_lock(&m->mu);
    if (!m->model || m->release_pending) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    const struct llama_vocab *vocab = erllama_safe_model_get_vocab(m->model);
    int32_t n_vocab = vocab ? erllama_safe_vocab_n_tokens(vocab) : 0;
    if (!vocab || n_vocab <= 0) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_invalid_token);
    }
    for (int32_t i = 0; i < n; i++) {
        if (tokens[i] >= n_vocab) {
            pthread_mutex_unlock(&m->mu);
            enif_free(tokens);
            return enif_make_tuple2(env, atom_error, atom_invalid_token);
        }
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
        return enif_make_tuple2(env, atom_error, atom_oom);
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
            return enif_make_tuple2(env, atom_error, atom_oom);
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
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    ERL_NIF_TERM bin;
    unsigned char *out = enif_make_new_binary(env, (size_t) wrote, &bin);
    if (!out) {
        enif_free(buf);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    if (wrote > 0) memcpy(out, buf, (size_t) wrote);
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
        ErlNifBinary empty;
        if (!enif_alloc_binary(0, &empty)) {
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        return enif_make_binary(env, &empty);
    }

    pthread_mutex_lock(&m->mu);
    if (!m->model || m->release_pending) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    const struct llama_vocab *vocab = erllama_safe_model_get_vocab(m->model);
    if (!vocab) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    int32_t n_vocab = erllama_safe_vocab_n_tokens(vocab);
    /* Fail closed if the vocab lookup gave us no usable size: without
     * n_vocab we cannot validate token IDs, and an out-of-range
     * positive ID would reach `id_to_token.at(id)` deep inside llama
     * and throw across the C ABI. Mirrors the prefill path. */
    if (n_vocab <= 0) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_invalid_token);
    }
    /* Validate before any token_to_piece call so out-of-range IDs do
     * not reach `id_to_token.at(id)` and trigger an internal throw. */
    for (int32_t i = 0; i < n; i++) {
        if (tokens[i] >= n_vocab) {
            pthread_mutex_unlock(&m->mu);
            enif_free(tokens);
            return enif_make_tuple2(env, atom_error, atom_invalid_token);
        }
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
        return enif_make_tuple2(env, atom_error, atom_too_large);
    }
    size_t cap = (size_t) n * 32u + 16u;
    char *out = enif_alloc(cap);
    if (!out) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_oom);
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
        if (err == 2) return enif_make_tuple2(env, atom_error, atom_oom);
        return enif_make_tuple2(env, atom_error, atom_invalid_token);
    }

    ErlNifBinary outbin;
    if (!enif_alloc_binary(used, &outbin)) {
        enif_free(out);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    memcpy(outbin.data, out, used);
    enif_free(out);
    return enif_make_binary(env, &outbin);
}
