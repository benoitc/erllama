/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */

/* Declaration surface for c_src/erllama_safe.cpp: exception-safe
 * wrappers around llama.cpp entry points that can throw across the C
 * ABI, plus the shared load-status / sentinel contract. Included by
 * every NIF C file AND by erllama_safe.cpp itself, so the compiler
 * checks these declarations against the definitions. */
#ifndef ERLLAMA_SAFE_H
#define ERLLAMA_SAFE_H

#include <erl_nif.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "llama.h"

#if defined(__cplusplus)
#define ERLLAMA_SAFE_NOEXCEPT noexcept
extern "C" {
#else
#define ERLLAMA_SAFE_NOEXCEPT
#endif

/* Sentinel returned by erllama_safe_decode when llama_decode threw a
 * C++ exception. Distinct from any documented llama_decode return
 * (currently 0/1/-1/2). Defined here and in erllama_safe.cpp; both
 * sides must agree. */
#define ERLLAMA_DECODE_EXC_SENTINEL INT_MIN

/* Mirror of erllama_load_status_t in erllama_safe.cpp; both sides
 * must agree on the integer values. Used by the _v2 model load
 * wrapper to distinguish a generic NULL return from a captured
 * GGML_ASSERT-flavoured failure. */
typedef enum {
    ERLLAMA_LOAD_OK        = 0,
    ERLLAMA_LOAD_FAILED    = 1,
    ERLLAMA_LOAD_MALFORMED = 2,
    ERLLAMA_LOAD_EXCEPTION = 3,
} erllama_load_status_t;

/* Exception-safe wrappers for llama.cpp calls that can throw across
 * the C ABI. Implemented in c_src/erllama_safe.cpp. Each returns a
 * sentinel (NULL, 0, SIZE_MAX, INT32_MIN, etc.) on a thrown C++
 * exception so the C NIF can surface a clean {error, oom} or
 * {error, invalid_token} instead of letting an exception unwind into
 * a C frame. */
extern struct llama_sampler *erllama_safe_sampler_chain_init(
    struct llama_sampler_chain_params p) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_greedy(void) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_dist(uint32_t seed) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_top_k(int32_t k) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_top_p(float p,
                                                             size_t min_keep) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_min_p(float p,
                                                             size_t min_keep) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_temp(float t) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_penalties(
    int32_t n_vocab, int32_t last_n, float repeat, float freq, float present) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_typical(float p,
                                                               size_t min_keep) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_temp_ext(
    float t, float delta, float exponent) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_xtc(
    float p, float t, size_t min_keep, uint32_t seed) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_top_n_sigma(float n) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_mirostat(
    int32_t n_vocab, uint32_t seed, float tau, float eta, int32_t m) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_mirostat_v2(
    uint32_t seed, float tau, float eta) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_dry(
    const struct llama_vocab *vocab, float multiplier, float base,
    int32_t allowed_length, int32_t penalty_last_n,
    const char **seq_breakers, size_t num_breakers) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_logit_bias(
    int32_t n_vocab, int32_t n_bias, const llama_logit_bias *bias) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_infill(
    const struct llama_vocab *vocab) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_sampler_chain_add(struct llama_sampler *chain,
                                          struct llama_sampler *s) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_sampler_free(struct llama_sampler *s) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_clone(
    const struct llama_sampler *smpl) ERLLAMA_SAFE_NOEXCEPT;
extern llama_token erllama_safe_sampler_sample(struct llama_sampler *s,
                                                struct llama_context *ctx,
                                                int32_t idx) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_sampler_accept(struct llama_sampler *s,
                                       llama_token tok) ERLLAMA_SAFE_NOEXCEPT;
extern int32_t erllama_safe_token_to_piece(const struct llama_vocab *vocab,
                                           llama_token tok, char *buf,
                                           int32_t buf_size,
                                           int32_t lstrip,
                                           bool special) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_backend_init(void) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_backend_init_once(void) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_backend_free(void) ERLLAMA_SAFE_NOEXCEPT;
extern void erllama_safe_log_unset(void) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_model *erllama_safe_model_load_from_file(
    const char *path, struct llama_model_params params) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_model *erllama_safe_model_load_from_file_v2(
    const char *path, struct llama_model_params params,
    erllama_load_status_t *out_status) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_model_free(struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_context *erllama_safe_init_from_model(
    struct llama_model *m, struct llama_context_params params) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_free(struct llama_context *c) ERLLAMA_SAFE_NOEXCEPT;
extern void erllama_safe_set_abort_callback(
    struct llama_context *c, bool (*cb)(void *), void *data) ERLLAMA_SAFE_NOEXCEPT;
extern const struct llama_model *erllama_safe_get_model(
    const struct llama_context *c) ERLLAMA_SAFE_NOEXCEPT;
extern const struct llama_vocab *erllama_safe_model_get_vocab(
    const struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern int32_t erllama_safe_vocab_n_tokens(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern uint64_t erllama_safe_model_size(const struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern int32_t erllama_safe_model_n_layer(const struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern int32_t erllama_safe_model_meta_val_str(const struct llama_model *m,
                                               const char *key, char *buf,
                                               size_t buf_size) ERLLAMA_SAFE_NOEXCEPT;
extern int32_t erllama_safe_model_desc(const struct llama_model *m,
                                       char *buf, size_t buf_size) ERLLAMA_SAFE_NOEXCEPT;
extern int32_t erllama_safe_model_n_ctx_train(const struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern uint64_t erllama_safe_model_n_params(const struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern int32_t erllama_safe_model_n_head(const struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern int32_t erllama_safe_model_n_swa(const struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_model_is_recurrent(const struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_model_is_hybrid(const struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_model_is_diffusion(const struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_model_has_encoder(const struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_model_has_decoder(const struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_model_ftype(const struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern uint32_t erllama_safe_n_ctx(const struct llama_context *c) ERLLAMA_SAFE_NOEXCEPT;
extern uint32_t erllama_safe_n_batch(const struct llama_context *c) ERLLAMA_SAFE_NOEXCEPT;
extern size_t erllama_safe_backend_dev_count(void) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_backend_dev_info(size_t idx, size_t *free_b,
                                         size_t *total_b, int *dev_type) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_vocab_is_eog(const struct llama_vocab *v,
                                     llama_token tok) ERLLAMA_SAFE_NOEXCEPT;
extern llama_token erllama_safe_vocab_bos(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern llama_token erllama_safe_vocab_eos(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern llama_token erllama_safe_vocab_eot(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern llama_token erllama_safe_vocab_sep(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern llama_token erllama_safe_vocab_nl(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern llama_token erllama_safe_vocab_pad(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern llama_token erllama_safe_vocab_mask(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern llama_token erllama_safe_vocab_fim_pre(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern llama_token erllama_safe_vocab_fim_suf(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern llama_token erllama_safe_vocab_fim_mid(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern llama_token erllama_safe_vocab_fim_pad(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern llama_token erllama_safe_vocab_fim_rep(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern llama_token erllama_safe_vocab_fim_sep(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_vocab_get_add_bos(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_vocab_get_add_eos(const struct llama_vocab *v) ERLLAMA_SAFE_NOEXCEPT;
extern int32_t erllama_safe_detokenize(const struct llama_vocab *vocab,
                                       const llama_token *tokens,
                                       int32_t n_tokens, char *text,
                                       int32_t text_len_max,
                                       bool remove_special,
                                       bool unparse_special) ERLLAMA_SAFE_NOEXCEPT;
extern int32_t erllama_safe_top_logprobs(struct llama_context *ctx,
                                         int32_t idx, int32_t n_vocab,
                                         int32_t k, llama_token sampled,
                                         llama_token *out_ids, float *out_lps,
                                         float *out_sampled_lp) ERLLAMA_SAFE_NOEXCEPT;
extern int32_t erllama_safe_tokenize(const struct llama_vocab *vocab,
                                     const char *text, int32_t text_len,
                                     llama_token *tokens, int32_t n_max,
                                     bool add_special, bool parse_special) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_decode(struct llama_context *c,
                               struct llama_batch batch) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_batch erllama_safe_batch_init(int32_t n_tokens,
                                                  int32_t embd,
                                                  int32_t n_seq_max) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_batch erllama_safe_batch_get_one(llama_token *tokens,
                                                     int32_t n_tokens) ERLLAMA_SAFE_NOEXCEPT;
extern void erllama_safe_batch_free(struct llama_batch batch) ERLLAMA_SAFE_NOEXCEPT;
extern size_t erllama_safe_state_seq_get_size(struct llama_context *c,
                                              int seq_id) ERLLAMA_SAFE_NOEXCEPT;
extern size_t erllama_safe_state_seq_get_data(struct llama_context *c,
                                              uint8_t *dst, size_t size,
                                              int seq_id) ERLLAMA_SAFE_NOEXCEPT;
extern size_t erllama_safe_state_seq_set_data(struct llama_context *c,
                                              const uint8_t *src,
                                              size_t size, int seq_id) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_memory_seq_rm(struct llama_context *c, int seq_id,
                                      int p0, int p1) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_memory_seq_cp(struct llama_context *c, int seq_src,
                                      int seq_dst) ERLLAMA_SAFE_NOEXCEPT;
extern long erllama_safe_memory_seq_pos_max(struct llama_context *c,
                                            int seq_id) ERLLAMA_SAFE_NOEXCEPT;
extern void erllama_safe_set_log_receiver(ErlNifPid pid, int min_level) ERLLAMA_SAFE_NOEXCEPT;
extern void erllama_safe_clear_log_receiver(void) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_forward_with_argmax(struct llama_context *c,
                                            const llama_token *tokens,
                                            int32_t n_tokens,
                                            int32_t n_vocab,
                                            long start_pos,
                                            int32_t *out_argmax) ERLLAMA_SAFE_NOEXCEPT;
extern const char *erllama_safe_model_chat_template(const struct llama_model *m,
                                                    const char *name) ERLLAMA_SAFE_NOEXCEPT;
extern int32_t erllama_safe_chat_apply_template(
    const char *tmpl, const struct llama_chat_message *msgs, size_t n_msgs,
    bool add_assistant, char *buf, int32_t buf_size) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_grammar(
    const struct llama_vocab *vocab, const char *grammar_str,
    const char *grammar_root) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_sampler *erllama_safe_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab *vocab, const char *grammar_str,
    const char *grammar_root, const char **trigger_patterns,
    size_t num_trigger_patterns, const llama_token *trigger_tokens,
    size_t num_trigger_tokens) ERLLAMA_SAFE_NOEXCEPT;
extern struct llama_adapter_lora *erllama_safe_adapter_lora_init(
    struct llama_model *model, const char *path) ERLLAMA_SAFE_NOEXCEPT;
extern void erllama_safe_adapter_lora_free(struct llama_adapter_lora *a) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_set_adapters_lora(struct llama_context *ctx,
                                          struct llama_adapter_lora **adapters,
                                          size_t n_adapters, float *scales) ERLLAMA_SAFE_NOEXCEPT;
extern float *erllama_safe_get_embeddings_seq(struct llama_context *c,
                                              int seq_id) ERLLAMA_SAFE_NOEXCEPT;
extern float *erllama_safe_get_embeddings(struct llama_context *c) ERLLAMA_SAFE_NOEXCEPT;
extern int32_t erllama_safe_n_embd(const struct llama_model *m) ERLLAMA_SAFE_NOEXCEPT;
extern int erllama_safe_set_embeddings(struct llama_context *c, bool value) ERLLAMA_SAFE_NOEXCEPT;

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* ERLLAMA_SAFE_H */
