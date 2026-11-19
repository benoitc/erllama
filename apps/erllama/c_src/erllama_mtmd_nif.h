/* C-callable shim over the C++ multimodal (libmtmd) NIF entry
 * points. The function bodies live in erllama_mtmd_nif.cpp; this
 * header is what the C NIF files include so they can:
 *   1. Call `mtmd_nif_load(env)` from the load callback to register
 *      the projector resource type.
 *   2. Reference the entry-point functions in `nif_funcs[]'.
 *   3. Drive a media prefill from C (nif_media_prefill in
 *      erllama_nif_decode.c) through the accessor + runner below
 *      without seeing any C++ type. */
#ifndef ERLLAMA_MTMD_NIF_H
#define ERLLAMA_MTMD_NIF_H

#include "erl_nif.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Registers the projector resource type; caches its pointer inside
 * the C++ TU and exports it below. Returns 0 on success. */
int mtmd_nif_load(ErlNifEnv *env);

/* The projector resource type (set during mtmd_nif_load); the C
 * side needs it for enif_get_resource in nif_media_prefill. */
extern ErlNifResourceType *ERLLAMA_MTMD_RT;

/* NIF entry points implemented in the C++ TU. */
ERL_NIF_TERM nif_mtmd_init(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_mtmd_free(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_mtmd_caps(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nif_mtmd_caps_file(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

/* --- media prefill support for the C side ------------------------ */

/* One media input: encoded image (jpg/png/bmp/gif) or audio
 * (wav/mp3/flac) bytes; decoding happens inside libmtmd. */
typedef struct {
    int is_audio;
    const unsigned char *data;
    size_t len;
} erllama_media_item_t;

/* Lock the projector resource and check liveness. Returns 1 with
 * the lock held, 0 when the projector was freed. */
int erllama_mtmd_lock_live(void *res);
void erllama_mtmd_unlock(void *res);

/* Run the whole multimodal prefill: decode the media items,
 * mtmd_tokenize the prompt (one <__media__> marker per item),
 * bounds-check against the context, and evaluate every chunk into
 * `lctx` at seq `seq_id` starting from position `n_past`, logits on
 * the last position. Both the projector resource lock and the
 * context lock must be held by the caller.
 *
 * Returns 0 on success with *out_n_tokens (KV cells consumed) and
 * *out_n_past (next position; can differ from n_past + n_tokens on
 * M-RoPE models). Negative on failure:
 *   -1 media decode failed (bad bytes / unsupported format)
 *   -2 marker count != media count
 *   -3 media preprocessing failed
 *   -4 evaluation (llama_decode / encode) failed
 *   -5 allocation failure or C++ exception
 *   -6 prompt + media exceed the context size
 */
int erllama_mtmd_prefill_run(void *mtmd_res, struct llama_context *lctx,
                             const char *prompt, size_t prompt_len,
                             const erllama_media_item_t *items,
                             size_t n_items, int add_special, int seq_id,
                             int32_t n_batch, int32_t n_past,
                             int32_t *out_n_tokens, int32_t *out_n_past);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ERLLAMA_MTMD_NIF_H */
