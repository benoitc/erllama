/* C-callable shim over the C++ chat-autoparser NIF entry points.
 * The function bodies live in barrel_inference_chat_nif.cpp; this
 * header is what barrel_inference_nif.c includes so it can:
 *   1. Call `chat_nif_load(env)` from its own `load/0` callback to
 *      register the two new resource types.
 *   2. Reference the entry-point functions by name in `nif_funcs[]'. */
#ifndef BARREL_INFERENCE_CHAT_NIF_H
#define BARREL_INFERENCE_CHAT_NIF_H

#include "erl_nif.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Registers the two chat-NIF resource types
 * (chat_templates_ref, chat_params_ref) and caches their pointers
 * inside the C++ TU. Returns 0 on success, non-zero on failure. */
int chat_nif_load(ErlNifEnv *env);

/* NIF entry-point declarations. All three run on
 * ERL_NIF_DIRTY_JOB_CPU_BOUND and convert C++ exceptions to
 * {error, {chat_parse_failed, Reason}} tuples. */
ERL_NIF_TERM nif_chat_templates_init(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_chat_templates_apply(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_chat_parse(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BARREL_INFERENCE_CHAT_NIF_H */
