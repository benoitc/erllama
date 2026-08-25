/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */

/* Shared term/map utilities for the erllama NIF translation units.
 * Everything here depends only on erl_nif.h, so the header is C/C++
 * safe behind the extern "C" guard and erllama_chat_nif.cpp can reuse
 * the map getters and binary builder. Helpers that need the resource
 * structs (lock guards, vocab access, tokenize retry) are declared in
 * erllama_nif_int.h instead. */
#ifndef ERLLAMA_NIF_UTIL_H
#define ERLLAMA_NIF_UTIL_H

#include <erl_nif.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Defined in erllama_nif.c's load callback. Redeclared here so the
 * static inline builders below work in TUs that only include this
 * header (the C++ chat NIF). */
extern ERL_NIF_TERM atom_error;

/* The `{error, Reason}` return tuple. */
static inline ERL_NIF_TERM erllama_error(ErlNifEnv *env, ERL_NIF_TERM reason) {
    return enif_make_tuple2(env, atom_error, reason);
}

/* Copy `len` bytes into a fresh binary term. Returns 1 with *out set,
 * or 0 on allocation failure (each caller keeps its own failure
 * mapping). The destination is an Erlang binary, not a C string, so
 * the copy is deliberately unterminated. */
static inline int erllama_bin_from(ErlNifEnv *env, const void *data,
                                   size_t len, ERL_NIF_TERM *out) {
    ERL_NIF_TERM bin;
    unsigned char *p = enif_make_new_binary(env, len, &bin);
    if (!p) return 0;
    if (len > 0) {
        /* NOLINTNEXTLINE(bugprone-not-null-terminated-result) */
        memcpy(p, data, len);
    }
    *out = bin;
    return 1;
}

/* The empty binary term. Returns 1 with *out set, 0 on allocation
 * failure. */
static inline int erllama_empty_bin(ErlNifEnv *env, ERL_NIF_TERM *out) {
    return erllama_bin_from(env, "", 0, out);
}

/* `Map#{key => Value}` with an atom key named by a C literal. */
static inline void erllama_map_put(ErlNifEnv *env, ERL_NIF_TERM *map,
                                   const char *key, ERL_NIF_TERM value) {
    enif_make_map_put(env, *map, enif_make_atom(env, key), value, map);
}

typedef struct {
    const char *name;
    int value;
} erllama_atom_enum_pair_t;

/* Generic option-map getters (definitions in erllama_nif_util.c). */
int copy_path(ErlNifEnv *env, ERL_NIF_TERM term, char *out, size_t cap);

int get_map_int31(
    ErlNifEnv *env, ERL_NIF_TERM map, const char *key, int32_t *out
);

int get_map_uint(
    ErlNifEnv *env, ERL_NIF_TERM map, const char *key, unsigned int *out
);

int get_map_double(
    ErlNifEnv *env, ERL_NIF_TERM map, const char *key, double *out
);

int get_map_bool(
    ErlNifEnv *env, ERL_NIF_TERM map, const char *key, int *out
);

int get_map_atom_enum(
    ErlNifEnv *env,
    ERL_NIF_TERM map,
    const char *key,
    const erllama_atom_enum_pair_t *table,
    size_t n,
    int *out
);

int get_map_float_list(
    ErlNifEnv *env,
    ERL_NIF_TERM map,
    const char *key,
    float *out,
    size_t cap
);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ERLLAMA_NIF_UTIL_H */
