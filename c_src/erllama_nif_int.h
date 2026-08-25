/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */

/* Internal cross-file surface of the erllama NIF, shared by the
 * per-concern C files (erllama_nif_*.c). The resource structs, the
 * atom table (defined and initialised in erllama_nif.c's load
 * callback), the shared helpers, and every NIF entry point wired
 * into nif_funcs[]. Not for use outside c_src. */
#ifndef ERLLAMA_NIF_INT_H
#define ERLLAMA_NIF_INT_H

#include <erl_nif.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#include "llama.h"

/* Hard upper bound on `n_seq_max`. Caps the per-context `per_seq[]`
 * array on `erllama_context_t`. Sized well above realistic loads
 * (typical multi-tenant inference servers run 1-32 concurrent
 * sequences per context); nif_new_context rejects with `badarg`
 * if a caller asks for more. Lift here, no other code change, if
 * the cap becomes binding. */
#define ERLLAMA_N_SEQ_MAX_CAP 256

#ifndef ERLLAMA_MAX_TOKENS
/* Cap on tokenize output, on the retry-needed-size returned from
 * `llama_tokenize`, and on caller-supplied token-list inputs handed
 * to prefill/decode NIFs. Sized to cover the worst-case rendered
 * chat-template body the downstream HTTP server accepts (64 MiB at
 * roughly 4 bytes/token) with comfortable headroom. Override at
 * build time via -DERLLAMA_MAX_TOKENS=N. */
#define ERLLAMA_MAX_TOKENS (16 * 1024 * 1024)
#endif

#ifndef ERLLAMA_MAX_TOKEN_TEXT
/* Cap on text bytes accepted by tokenize/3 and on the rendered
 * chat-template buffer in apply_chat_template/2. 64 MiB matches the
 * downstream erllama_server's max_request_body_bytes so any input
 * the server passes through is bounded by this cap. Override at
 * build time via -DERLLAMA_MAX_TOKEN_TEXT=N for batch workflows. */
#define ERLLAMA_MAX_TOKEN_TEXT (64 * 1024 * 1024)
#endif


/* =========================================================================
 * Resource types
 * ========================================================================= */

/* Per-resource mutex makes use-after-free between concurrent dirty
 * NIFs and an explicit free call impossible: every NIF entry that
 * dereferences a resource locks it, observes the pointer, and runs
 * llama under that lock; explicit frees take the same lock, so they
 * cannot interleave with a live llama call. The lock is held for the
 * duration of a llama op, but ops on different resources stay
 * independent. */
/* The erllama_model_t layout + the resource type pointers
 * (MODEL_RT etc.) live in erllama_resources.h so C++ TUs
 * (e.g. erllama_chat_nif.cpp) can `enif_get_resource' the
 * model and reach `m->model'. The header declares; this file defines
 * the resource pointers further down. */
#include "erllama_resources.h" /* erllama_model_t + resource type externs */

/* Per-seq state tracked across nif_step ticks so the next tick
 * knows where each seq's logits live (for sampling) and where its
 * next token's position is (for batch construction).
 *
 *   last_logits_idx: row index into the previous llama_decode batch
 *                    at which this seq's logits were emitted, or -1
 *                    if no logits are live for this seq (fresh
 *                    seq_id, post kv_seq_rm, or post kv_unpack —
 *                    the unpacked state has KV but no live logits).
 *
 *   next_pos:        position to assign to the next token decoded
 *                    for this seq. Bumped by the slice length per
 *                    tick. Initialised to 0 for a fresh seq_id and
 *                    to `llama_memory_seq_pos_max(ctx, seq_id) + 1`
 *                    after a kv_unpack. */
typedef struct {
    int32_t last_logits_idx;
    int32_t next_pos;
} erllama_per_seq_t;

#define ERLLAMA_GRAMMAR_CACHE_N 4

/* One cached compiled grammar. `bytes` is an owned copy of the GBNF used
 * to verify identity (a hash alone is not identity); `tmpl` is the parsed
 * grammar sampler, cloned per request and never used for decode directly. */
typedef struct {
    unsigned char *bytes;
    size_t len;
    uint64_t hash;
    struct llama_sampler *tmpl;
    uint64_t lru;
} erllama_grammar_cache_entry_t;

typedef struct erllama_context_s {
    pthread_mutex_t mu;
    int mu_inited;
    struct llama_context *ctx;     /* NULL after successful release */
    erllama_model_t *model_res;    /* keep_resource'd by new_context */
    int decode_ready;              /* set after llama_decode; cleared after kv ops */
    /* Decode interrupt/budget. The abort callback (installed on the
     * llama_context at new_context time) reads these from ggml worker
     * threads; the NIF decode thread arms them before each
     * llama_decode. request_abort sets abort_flag WITHOUT taking mu
     * (a wedged decode holds mu), so these must be atomic. */
    uint64_t decode_budget_ns;          /* per-step budget; 0 = disabled */
    _Atomic uint_least64_t decode_deadline_ns; /* 0 = no live deadline */
    _Atomic int abort_flag;             /* 1 = abort the in-flight decode */
    /* Sampler chain cached on the first nif_decode_one call. The
     * chain is greedy-only and lives for the resource's lifetime;
     * a future sampler-config NIF would free + rebuild this under
     * the resource lock. */
    struct llama_sampler *smpl;
    /* Per-seq state. Only used by nif_step / kv_pack / kv_unpack
     * / kv_seq_rm; the single-seq fast paths (decode_one, prefill)
     * leave this untouched and continue using seq_id=0 semantics
     * implicit in their callers. */
    erllama_per_seq_t per_seq[ERLLAMA_N_SEQ_MAX_CAP];
    /* Compiled-grammar cache. Claude Code (and agentic clients) send the
     * same GBNF every turn; re-parsing it per request dominates infer
     * admission for large tool grammars. Cache the parsed template keyed
     * by the GBNF bytes and clone per request. All access is under c->mu
     * (the sampler builder runs with the lock held), so no extra locking. */
    erllama_grammar_cache_entry_t gcache[ERLLAMA_GRAMMAR_CACHE_N];
    uint64_t gcache_tick;
    uint64_t gcache_hits;
    uint64_t gcache_misses;
} erllama_context_t;

/* LoRA adapter resource. The adapter is bound to a model and stays
 * valid until the model is freed or adapter_lora_free is called
 * explicitly. The wrapping resource holds a keep-reference on its
 * model_res so the underlying llama_model* outlives the adapter even
 * if the user free_model's it. */
typedef struct {
    pthread_mutex_t mu;
    int mu_inited;
    struct llama_adapter_lora *adapter; /* NULL after explicit free */
    erllama_model_t *model_res;         /* keep_resource'd at init */
} erllama_adapter_t;

/* Sampler chain resource. Owned independently from the context so
 * multi-seq batching (v0.2+) can hold one chain per in-flight
 * request without contending on the context's cached `c->smpl`.
 * The chain is built from the same config map configure_sampler/2
 * consumes; freed explicitly via sampler_free/1 or implicitly by
 * the dtor. */
typedef struct {
    pthread_mutex_t mu;
    int mu_inited;
    struct llama_sampler *chain;   /* NULL after explicit free */
    erllama_context_t *ctx_res;    /* keep_resource'd at init */
    /* Per-request logprobs: number of top-token logprobs to report
     * with every sampled token (cfg key `logprobs`, 0 = off). Read
     * by nif_step's pre-sample loop under `mu`. */
    int n_probs;
} erllama_sampler_t;

/* Declared in erllama_resources.h so the C++ chat NIF TU can
 * reference them through extern visibility. */

/* Atoms: defined and initialised once in erllama_nif.c (load). */
extern ERL_NIF_TERM atom_ok;
extern ERL_NIF_TERM atom_error;
extern ERL_NIF_TERM atom_load_failed;
extern ERL_NIF_TERM atom_malformed_gguf;
extern ERL_NIF_TERM atom_context_failed;
extern ERL_NIF_TERM atom_tokenize_failed;
extern ERL_NIF_TERM atom_pack_failed;
extern ERL_NIF_TERM atom_unpack_failed;
extern ERL_NIF_TERM atom_true;
extern ERL_NIF_TERM atom_false;
extern ERL_NIF_TERM atom_released;
extern ERL_NIF_TERM atom_too_large;
extern ERL_NIF_TERM atom_invalid_token;
extern ERL_NIF_TERM atom_context_overflow;
extern ERL_NIF_TERM atom_batch_overflow;
extern ERL_NIF_TERM atom_oom;
extern ERL_NIF_TERM atom_deferred;
extern ERL_NIF_TERM atom_exception;
extern ERL_NIF_TERM atom_no_logits;
extern ERL_NIF_TERM atom_no_template;
extern ERL_NIF_TERM atom_template_failed;
extern ERL_NIF_TERM atom_grammar_failed;
extern ERL_NIF_TERM atom_embed_failed;
extern ERL_NIF_TERM atom_not_supported;
extern ERL_NIF_TERM atom_invalid_content;
extern ERL_NIF_TERM atom_no_gpu;
extern ERL_NIF_TERM atom_total_b;
extern ERL_NIF_TERM atom_free_b;
extern ERL_NIF_TERM atom_used_b;
extern ERL_NIF_TERM atom_eos;
extern ERL_NIF_TERM atom_decode_failed;
extern ERL_NIF_TERM atom_decode_timeout;
extern ERL_NIF_TERM atom_decode_aborted;

typedef struct {
    const char *name;
    int value;
} erllama_atom_enum_pair_t;

/* Shared helpers (definitions in erllama_nif.c). */
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

/* Deferred-free protocol (definition in erllama_nif.c). */
void context_drops_model(erllama_model_t *m);

/* Decode guards (definitions in erllama_nif_context.c). */
void arm_decode(erllama_context_t *c);

ERL_NIF_TERM classify_decode_error(ErlNifEnv *env, erllama_context_t *c,
                                          int rc);

/* Token-list parsing (definitions in erllama_nif_tokens.c). */
int read_token_list(ErlNifEnv *env, ERL_NIF_TERM list,
                           llama_token **out, int32_t *out_len);

ERL_NIF_TERM token_list_error(ErlNifEnv *env, int rc);

/* Sampler construction + grammar cache (erllama_nif_sampler.c). */
struct llama_sampler *build_default_greedy_chain(void);

struct llama_sampler *
build_sampler_chain_from_map(ErlNifEnv *env, ERL_NIF_TERM cfg,
                             erllama_context_t *c,
                             ERL_NIF_TERM *out_err_atom);

void grammar_cache_clear(erllama_context_t *c);

/* NIF entry points, by file. */
/* erllama_nif_model.c */
ERL_NIF_TERM nif_model_size(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_model_n_layer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_model_family(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_vocab_info(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_vram_info(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_load_model(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_free_model(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

/* erllama_nif_context.c */
ERL_NIF_TERM nif_request_abort(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_new_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_free_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

/* erllama_nif_kv.c */
ERL_NIF_TERM nif_kv_pack(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_kv_unpack(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_kv_seq_rm(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_kv_seq_cp(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

/* erllama_nif_tokens.c */
ERL_NIF_TERM nif_tokenize(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_detokenize(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_detokenize_opts(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

/* erllama_nif_decode.c */
ERL_NIF_TERM nif_step(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_prefill(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_decode_one(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_embed(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_forward_with_argmax(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

/* erllama_nif_sampler.c */
ERL_NIF_TERM nif_configure_sampler(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_set_grammar(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_clear_sampler(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_sampler_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_sampler_free(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_grammar_cache_stats(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

/* erllama_nif_chat_template.c */
ERL_NIF_TERM nif_apply_chat_template(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

/* erllama_nif_adapter.c */
ERL_NIF_TERM nif_adapter_load(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_adapter_free(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_set_adapters(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

#endif /* ERLLAMA_NIF_INT_H */
