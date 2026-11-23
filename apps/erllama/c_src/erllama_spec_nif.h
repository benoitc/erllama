/* C-callable shim over the C++ ngram-speculation NIF entry points.
 * The function bodies live in erllama_spec_nif.cpp; this header is
 * what erllama_nif.c includes so it can:
 *   1. Call `spec_nif_load(env)` from its own `load` callback to
 *      register the speculator resource type.
 *   2. Reference the entry-point functions by name in `nif_funcs[]'. */
#ifndef ERLLAMA_SPEC_NIF_H
#define ERLLAMA_SPEC_NIF_H

#include "erl_nif.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Registers the speculator resource type and caches its pointer
 * inside the C++ TU. Returns 0 on success, non-zero on failure. */
int spec_nif_load(ErlNifEnv *env);

/* NIF entry-point declarations. All run on
 * ERL_NIF_DIRTY_JOB_CPU_BOUND and convert C++ exceptions to
 * {error, exception}. */
ERL_NIF_TERM nif_spec_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_spec_begin(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_spec_draft(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_spec_accept(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_spec_free(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ERLLAMA_SPEC_NIF_H */
