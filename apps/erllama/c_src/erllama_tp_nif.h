/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */

/* Rank-local tensor-parallel entry points (experimental). One OS
 * process per rank: each opens its GPU backend and joins the
 * backend's communicator (`ggml_backend_comm_init_rank'), whose
 * bootstrap exchange goes through an Erlang agent process. The bodies
 * live in erllama_tp_nif.cpp; erllama_nif.c calls tp_nif_load() from
 * its load callback and lists the entry points in nif_funcs[]. */
#ifndef ERLLAMA_TP_NIF_H
#define ERLLAMA_TP_NIF_H

#include "erl_nif.h"
#include <stddef.h>

struct ggml_backend_device;

#ifdef __cplusplus
extern "C" {
#endif

/* Registers the bridge and rank resource types. 0 on success. */
int tp_nif_load(ErlNifEnv *env);

ERL_NIF_TERM nif_tp_caps(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_tp_open(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_tp_bootstrap_reply(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_tp_matvec(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_tp_reduce(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_tp_close(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_tp_bootstrap_selftest(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_tp_stats(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

/* For the model loader: the rank's group devices (world entries, then
 * NULL) and a reference keeping the rank alive while the model lives;
 * the device count, or -1 when `rank' is not an open rank. */
int erllama_tp_model_devices(ErlNifEnv *env, ERL_NIF_TERM rank, struct ggml_backend_device **out,
                             size_t cap, void **keep);
/* Drops the reference erllama_tp_model_devices took (model freed). */
void erllama_tp_model_released(void *keep);
/* Sums `n' floats in place over every rank of `keep''s group, exactly.
 * 0 on success. */
int erllama_tp_reduce_host(void *keep, float *data, size_t n);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ERLLAMA_TP_NIF_H */
