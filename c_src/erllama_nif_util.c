/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */
/* Shared helpers used across the per-concern NIF files: option-map
 * getters, token-list parsing, vocab access, and the tokenize
 * grow-retry loop. Term builders and lock guards are static inline
 * in erllama_nif_util.h / erllama_nif_int.h. */

#include "erllama_nif_int.h"
#include "erllama_safe.h"

#include <string.h>

int copy_path(ErlNifEnv *env, ERL_NIF_TERM term, char *out, size_t cap) {
    ErlNifBinary bin;
    if (!enif_inspect_iolist_as_binary(env, term, &bin)) return 0;
    if (bin.size == 0 || bin.size >= cap) return 0;
    /* Reject embedded NUL: a Erlang binary like <<"real\0ignored">>
     * would be silently truncated by C string APIs to "real". */
    if (memchr(bin.data, '\0', bin.size) != NULL) return 0;
    memcpy(out, bin.data, bin.size);
    out[bin.size] = '\0';
    return 1;
}

/* Read an unsigned int but reject values that would wrap when cast
 * to int32_t. Used for llama options (n_gpu_layers, n_threads, etc.)
 * which are signed int32 fields in llama.cpp. */
int get_map_int31(
    ErlNifEnv *env, ERL_NIF_TERM map, const char *key, int32_t *out
) {
    ERL_NIF_TERM v;
    ERL_NIF_TERM k = enif_make_atom(env, key);
    if (!enif_get_map_value(env, map, k, &v)) return 0;
    unsigned int u;
    if (!enif_get_uint(env, v, &u)) return 0;
    if (u > (unsigned int) INT32_MAX) return 0;
    *out = (int32_t) u;
    return 1;
}

int get_map_uint(
    ErlNifEnv *env, ERL_NIF_TERM map, const char *key, unsigned int *out
) {
    ERL_NIF_TERM v;
    ERL_NIF_TERM k = enif_make_atom(env, key);
    if (!enif_get_map_value(env, map, k, &v)) return 0;
    return enif_get_uint(env, v, out);
}

/* Read a number from a map either as a float (`enif_get_double`) or as
 * an integer that gets promoted to double. Lets callers write
 * `temperature => 0` and `temperature => 0.7` interchangeably. */
int get_map_double(
    ErlNifEnv *env, ERL_NIF_TERM map, const char *key, double *out
) {
    ERL_NIF_TERM v;
    ERL_NIF_TERM k = enif_make_atom(env, key);
    if (!enif_get_map_value(env, map, k, &v)) return 0;
    if (enif_get_double(env, v, out)) return 1;
    long ll;
    if (enif_get_long(env, v, &ll)) {
        *out = (double) ll;
        return 1;
    }
    return 0;
}

int get_map_bool(
    ErlNifEnv *env, ERL_NIF_TERM map, const char *key, int *out
) {
    ERL_NIF_TERM v;
    ERL_NIF_TERM k = enif_make_atom(env, key);
    if (!enif_get_map_value(env, map, k, &v)) return 0;
    if (enif_compare(v, atom_true) == 0) {
        *out = 1;
        return 1;
    }
    if (enif_compare(v, atom_false) == 0) {
        *out = 0;
        return 1;
    }
    return 0;
}

/* Read an atom-valued map entry and translate it to a C int via a
 * lookup table. Used for `split_mode`, `flash_attn`, `type_k`, and
 * `type_v` in nif_load_model and nif_new_context.
 *
 * Returns:
 *   1: key present, atom matched the table, *out updated
 *   0: key absent (caller leaves whatever default already filled
 *      `params.<field>` in place)
 *  -1: key present but the value is not an atom in the table; the
 *      caller is expected to raise badarg
 */
int get_map_atom_enum(
    ErlNifEnv *env,
    ERL_NIF_TERM map,
    const char *key,
    const erllama_atom_enum_pair_t *table,
    size_t n,
    int *out
) {
    ERL_NIF_TERM v;
    ERL_NIF_TERM k = enif_make_atom(env, key);
    if (!enif_get_map_value(env, map, k, &v)) return 0;
    char buf[64];
    if (!enif_get_atom(env, v, buf, sizeof(buf), ERL_NIF_LATIN1)) {
        return -1;
    }
    for (size_t i = 0; i < n; i++) {
        if (strcmp(buf, table[i].name) == 0) {
            *out = table[i].value;
            return 1;
        }
    }
    return -1;
}

/* Read a list of floats (or ints promotable to float) from a map and
 * copy them into `out` up to `cap` entries. Returns the count
 * written, 0 if the key is absent, or -1 on a type mismatch or
 * over-cap.
 *
 * Used for `tensor_split` on model load. The destination buffer
 * lives on `erllama_model_t` so its lifetime spans the loaded
 * model; see the struct comment for why we cannot use a
 * stack-local. */
int get_map_float_list(
    ErlNifEnv *env,
    ERL_NIF_TERM map,
    const char *key,
    float *out,
    size_t cap
) {
    ERL_NIF_TERM v;
    ERL_NIF_TERM k = enif_make_atom(env, key);
    if (!enif_get_map_value(env, map, k, &v)) return 0;
    if (!enif_is_list(env, v)) return -1;
    size_t n = 0;
    ERL_NIF_TERM head;
    ERL_NIF_TERM tail = v;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        if (n >= cap) return -1;
        double d;
        long ll;
        if (enif_get_double(env, head, &d)) {
            out[n++] = (float) d;
        } else if (enif_get_long(env, head, &ll)) {
            out[n++] = (float) ll;
        } else {
            return -1;
        }
    }
    return (int) n;
}

/* =========================================================================
 * Token lists
 * ========================================================================= */

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
        case -1: return erllama_error(env, atom_oom);
        case -2: return erllama_error(env, atom_too_large);
        case -3: return erllama_error(env, atom_invalid_token);
        default: return enif_make_badarg(env);
    }
}

/* =========================================================================
 * Vocab access + token validation
 * ========================================================================= */

const struct llama_vocab *erllama_model_vocab(erllama_model_t *m,
                                              int32_t *n_vocab) {
    const struct llama_vocab *vocab = erllama_safe_model_get_vocab(m->model);
    if (n_vocab) {
        *n_vocab = vocab ? erllama_safe_vocab_n_tokens(vocab) : 0;
    }
    return vocab;
}

const struct llama_vocab *erllama_ctx_vocab(erllama_context_t *c,
                                            int32_t *n_vocab) {
    const struct llama_model *model = erllama_safe_get_model(c->ctx);
    const struct llama_vocab *vocab =
        model ? erllama_safe_model_get_vocab(model) : NULL;
    if (n_vocab) {
        *n_vocab = vocab ? erllama_safe_vocab_n_tokens(vocab) : 0;
    }
    return vocab;
}

int32_t erllama_first_oob_token(const llama_token *toks, int32_t n,
                                int32_t n_vocab) {
    for (int32_t i = 0; i < n; i++) {
        if (toks[i] >= n_vocab) return i;
    }
    return -1;
}

/* =========================================================================
 * Tokenize grow-retry
 * ========================================================================= */

erllama_tok_status_t erllama_tokenize_grow(
    const struct llama_vocab *vocab, const char *text, int32_t len,
    bool add_special, bool parse_special,
    llama_token **out, int32_t *out_n
) {
    /* `n_max = len + 8` is a sound upper bound: tokens-per-byte for
     * any tokenizer is `<= 1 + small constant`, the +8 covering
     * BOS/EOS specials. Clamping it down to ERLLAMA_MAX_TOKENS would
     * force an unnecessary retry on inputs over ~1 byte/token average
     * and cap production workloads below the input text cap
     * (ERLLAMA_MAX_TOKEN_TEXT, enforced by the callers), which is
     * what bounds the allocation. */
    int32_t n_max = len + 8;
    if (n_max < 16) n_max = 16;
    llama_token *tokens =
        (llama_token *) enif_alloc(sizeof(llama_token) * (size_t) n_max);
    if (!tokens) return ERLLAMA_TOK_OOM;
    int32_t n = erllama_safe_tokenize(vocab, text, len, tokens, n_max,
                                      add_special, parse_special);
    if (n < 0 && n != INT32_MIN) {
        int32_t needed = -n;
        if (needed > ERLLAMA_MAX_TOKENS) {
            enif_free(tokens);
            return ERLLAMA_TOK_TOO_LARGE;
        }
        enif_free(tokens);
        tokens =
            (llama_token *) enif_alloc(sizeof(llama_token) * (size_t) needed);
        if (!tokens) return ERLLAMA_TOK_OOM;
        n = erllama_safe_tokenize(vocab, text, len, tokens, needed,
                                  add_special, parse_special);
    }
    if (n == INT32_MIN) {
        enif_free(tokens);
        return ERLLAMA_TOK_EXCEPTION;
    }
    if (n < 0) {
        enif_free(tokens);
        return ERLLAMA_TOK_FAILED;
    }
    /* Enforce the output cap post-success: removing the n_max clamp
     * before the first call means the tokenizer can fully populate a
     * buffer larger than ERLLAMA_MAX_TOKENS (e.g. byte-fallback
     * tokenizers at ~1 byte/token on a 60 MiB input). Convert that
     * into a clean too_large error rather than returning an over-cap
     * list to Erlang. */
    if (n > ERLLAMA_MAX_TOKENS) {
        enif_free(tokens);
        return ERLLAMA_TOK_TOO_LARGE;
    }
    *out = tokens;
    *out_n = n;
    return ERLLAMA_TOK_OK;
}
