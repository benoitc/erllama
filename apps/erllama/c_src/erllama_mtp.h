/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */

/* A C surface over llama.cpp's MTP drafter (common/speculative.h,
 * COMMON_SPECULATIVE_TYPE_DRAFT_MTP), for erllama_nif_mtp.c. The
 * drafter reads the target context's nextn embeddings after every
 * target decode (erllama_mtp_process) and proposes the tokens after a
 * known one from a draft model's MTP head, in a context of its own on
 * this process's GPU. Every call is noexcept; the caller serialises
 * them under the target context's lock. */
#ifndef ERLLAMA_MTP_H
#define ERLLAMA_MTP_H

#include <stdint.h>

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

struct erllama_mtp;

typedef struct {
    uint64_t process_calls;
    uint64_t process_failures;
    uint64_t process_us;
    uint64_t drafts;
    uint64_t draft_us;
} erllama_mtp_counters_t;

/* A drafter of `n_max' tokens a round on `tgt' with `draft' (a model
 * loaded with its MTP layers). Turns on the target's unmasked nextn
 * embeddings. NULL with *why set on failure. */
struct erllama_mtp *erllama_mtp_new(struct llama_model *draft, struct llama_context *tgt,
                                    int32_t n_max, const char **why);

/* Frees the drafter and its context; the target keeps its nextn
 * embeddings until erllama_mtp_target_nextn(tgt, 0). */
void erllama_mtp_free(struct erllama_mtp *m);

/* Turn the target's unmasked nextn embeddings on or off (the other
 * ranks of a split target follow the serving rank's). */
void erllama_mtp_target_nextn(struct llama_context *tgt, int on);

/* After the target decoded `b': feed the drafter, first dropping what
 * its own context holds at or after each sequence's first position in
 * `b' (its draft steps, or a target that went back). A failure only
 * makes later drafts worse; it is counted, never fatal. */
void erllama_mtp_process(struct erllama_mtp *m, const struct llama_batch *b);

/* Up to `n_max' tokens after `id_last' at position `pos0' of `seq'
 * (not yet decoded by the target) into `out'; how many. */
int32_t erllama_mtp_draft(struct erllama_mtp *m, int32_t seq, llama_token id_last, int32_t pos0,
                          int32_t n_max, llama_token *out);

/* Draft for `n' sequences at once (one drafter batch a step): up to
 * `n_max' tokens after `id_last[i]' at `pos0[i]' of `seqs[i]' into
 * `out[i * n_max ...]', how many into `n_out[i]'. */
void erllama_mtp_draft_many(struct erllama_mtp *m, int32_t n, const int32_t *seqs, const llama_token *id_last,
                            const int32_t *pos0, int32_t n_max, llama_token *out, int32_t *n_out);

/* The target accepted `n_acc' drafted tokens of the last verified
 * batch of `seq'. */
void erllama_mtp_accept(struct erllama_mtp *m, int32_t seq, int32_t n_acc);

/* Remove positions [p0, p1) of `seq' from the drafter's context. */
void erllama_mtp_seq_rm(struct erllama_mtp *m, int32_t seq, int32_t p0, int32_t p1);

/* The highest position the drafter's context holds for `seq', -1
 * when empty. */
int32_t erllama_mtp_seq_pos_max(struct erllama_mtp *m, int32_t seq);

void erllama_mtp_counters(const struct erllama_mtp *m, erllama_mtp_counters_t *out);

/* 1 when `chain' always picks the most likely token (a greedy chain,
 * optionally after filters that keep it), so a draft can be verified
 * by argmax. */
int erllama_mtp_chain_is_greedy(struct llama_sampler *chain);

#ifdef __cplusplus
}
#endif

#endif
