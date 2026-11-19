/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */
/* Sampler chains: the per-request chain builder (grammar, lazy
 * triggers, full llama.cpp stage set), the per-context compiled
 * grammar cache, and the sampler resource NIFs. */

#include "erllama_nif_int.h"
#include "erllama_safe.h"

#include <math.h>
#include <string.h>

/* =========================================================================
 * Sampler config
 *
 * configure_sampler/2 is the one entry point that builds the per-context
 * sampler chain. It accepts a config map carrying any of: grammar,
 * repetition_penalty, top_k, top_p, min_p, temperature, seed. Missing
 * fields are skipped; a temperature of 0.0 (or no sampling params at
 * all) ends the chain in greedy.
 *
 * set_grammar/2 is a backwards-compatible alias that builds the same
 * chain with only a grammar entry. clear_sampler/1 drops the cached
 * chain so the next decode_one lazy-inits greedy.
 * ========================================================================= */

struct llama_sampler *build_default_greedy_chain(void) {
    struct llama_sampler_chain_params sp =
        llama_sampler_chain_default_params();
    struct llama_sampler *chain = erllama_safe_sampler_chain_init(sp);
    if (!chain) return NULL;
    struct llama_sampler *greedy = erllama_safe_sampler_init_greedy();
    if (!greedy) {
        (void) erllama_safe_sampler_free(chain);
        return NULL;
    }
    if (erllama_safe_sampler_chain_add(chain, greedy) != 0) {
        (void) erllama_safe_sampler_free(greedy);
        (void) erllama_safe_sampler_free(chain);
        return NULL;
    }
    return chain;
}

/* Append one stage to a chain, freeing the chain and returning NULL on
 * failure so callers can write a tight cleanup ladder. */
static int chain_append(struct llama_sampler *chain,
                        struct llama_sampler *stage) {
    if (!stage) return -1;
    if (erllama_safe_sampler_chain_add(chain, stage) != 0) {
        (void) erllama_safe_sampler_free(stage);
        return -1;
    }
    return 0;
}

/* FNV-1a 64-bit over the GBNF bytes; a fast pre-filter for the grammar
 * cache. Identity is still confirmed by length + memcmp on lookup. */
static uint64_t grammar_hash(const unsigned char *p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* Return the cached parsed grammar template for these exact GBNF bytes, or
 * NULL. Caller holds c->mu. Bumps LRU + hit/miss counters. */
static struct llama_sampler *
grammar_cache_get(erllama_context_t *c, const unsigned char *bytes, size_t len) {
    uint64_t h = grammar_hash(bytes, len);
    for (int i = 0; i < ERLLAMA_GRAMMAR_CACHE_N; i++) {
        erllama_grammar_cache_entry_t *e = &c->gcache[i];
        if (e->tmpl && e->hash == h && e->len == len &&
            memcmp(e->bytes, bytes, len) == 0) {
            e->lru = ++c->gcache_tick;
            c->gcache_hits++;
            return e->tmpl;
        }
    }
    c->gcache_misses++;
    return NULL;
}

/* Store a parsed grammar template under its GBNF bytes, evicting the LRU
 * slot if full. Takes ownership of `tmpl`: on its own alloc failure it
 * frees `tmpl` (the caller has already cloned what it needs). Caller holds
 * c->mu. */
static void
grammar_cache_put(erllama_context_t *c, const unsigned char *bytes,
                  size_t len, struct llama_sampler *tmpl) {
    int victim = 0;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < ERLLAMA_GRAMMAR_CACHE_N; i++) {
        if (!c->gcache[i].tmpl) {
            victim = i;
            break;
        }
        if (c->gcache[i].lru < oldest) {
            oldest = c->gcache[i].lru;
            victim = i;
        }
    }
    unsigned char *copy = enif_alloc(len);
    if (!copy) {
        (void) erllama_safe_sampler_free(tmpl);
        return;
    }
    memcpy(copy, bytes, len);
    erllama_grammar_cache_entry_t *e = &c->gcache[victim];
    if (e->tmpl) {
        (void) erllama_safe_sampler_free(e->tmpl);
        enif_free(e->bytes);
    }
    e->bytes = copy;
    e->len = len;
    e->hash = grammar_hash(bytes, len);
    e->tmpl = tmpl;
    e->lru = ++c->gcache_tick;
}

/* Free every cached grammar template + its bytes. Caller holds c->mu (or
 * is the destructor, where no other thread can reach the resource). */
void grammar_cache_clear(erllama_context_t *c) {
    for (int i = 0; i < ERLLAMA_GRAMMAR_CACHE_N; i++) {
        erllama_grammar_cache_entry_t *e = &c->gcache[i];
        if (e->tmpl) {
            (void) erllama_safe_sampler_free(e->tmpl);
            e->tmpl = NULL;
        }
        if (e->bytes) {
            enif_free(e->bytes);
            e->bytes = NULL;
        }
        e->len = 0;
        e->hash = 0;
    }
}

/* Parse a list of iolists/binaries into an enif_alloc'd array of
 * NUL-terminated C strings. Returns 0 on success, -1 on a bad term or
 * alloc failure. Free with free_string_list. */
static int read_string_list(ErlNifEnv *env, ERL_NIF_TERM list,
                            char ***out, unsigned int *out_n) {
    unsigned int n;
    if (!enif_get_list_length(env, list, &n)) return -1;
    *out = NULL;
    *out_n = 0;
    if (n == 0) return 0;
    char **arr = enif_alloc(sizeof(char *) * n);
    if (!arr) return -1;
    memset(arr, 0, sizeof(char *) * n);
    ERL_NIF_TERM head, tail = list;
    unsigned int i = 0;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        ErlNifBinary bin;
        if (!enif_inspect_iolist_as_binary(env, head, &bin)) goto fail;
        char *s = enif_alloc(bin.size + 1);
        if (!s) goto fail;
        memcpy(s, bin.data, bin.size);
        s[bin.size] = '\0';
        arr[i++] = s;
    }
    *out = arr;
    *out_n = n;
    return 0;
fail:
    for (unsigned int j = 0; j < n; j++) {
        if (arr[j]) enif_free(arr[j]);
    }
    enif_free(arr);
    return -1;
}

static void free_string_list(char **arr, unsigned int n) {
    if (!arr) return;
    for (unsigned int i = 0; i < n; i++) {
        if (arr[i]) enif_free(arr[i]);
    }
    enif_free(arr);
}

static int erllama_is_space(char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
}

/* Feed the generation-prompt tokens into a freshly cloned NON-lazy
 * grammar sampler (mirrors common_sampler_init,
 * common/sampling.cpp:278-308): a template-synthesized grammar covers
 * the assistant header that is already part of the prompt, so those
 * tokens must be accepted before sampling starts. parse_special=true;
 * a synthetic leading-space token is skipped when the text itself does
 * not start with whitespace. Returns 0 on success. */
static int grammar_accept_prefill(const struct llama_vocab *vocab,
                                  struct llama_sampler *g,
                                  const unsigned char *text, size_t len) {
    int32_t cap = (int32_t) len + 16;
    llama_token *toks = enif_alloc(sizeof(llama_token) * (size_t) cap);
    if (!toks) return -1;
    int32_t n = erllama_safe_tokenize(vocab, (const char *) text,
                                      (int32_t) len, toks, cap,
                                      false, true);
    if (n < 0) {
        enif_free(toks);
        return -1;
    }
    for (int32_t i = 0; i < n; i++) {
        if (i == 0) {
            char piece[64];
            int32_t pn = erllama_safe_token_to_piece(
                vocab, toks[0], piece, (int32_t) sizeof(piece), 0, true);
            if (pn > 0 && erllama_is_space(piece[0]) &&
                !erllama_is_space((char) text[0])) {
                continue;
            }
        }
        if (erllama_safe_sampler_accept(g, toks[i]) != 0) {
            enif_free(toks);
            return -1;
        }
    }
    enif_free(toks);
    return 0;
}

/* Compile (or fetch from the per-context cache) the grammar described
 * by `cfg` and append it to `chain`. Handles both plain and lazy
 * grammars plus the non-lazy generation-prompt prefill. Returns 0 on
 * success; on failure returns -1 with *out_err_atom set (the caller
 * frees `chain`). Caller holds c->mu. */
static int append_grammar_sampler(ErlNifEnv *env, ERL_NIF_TERM cfg,
                                  erllama_context_t *c,
                                  struct llama_sampler *chain,
                                  ErlNifBinary grammar_bin,
                                  ERL_NIF_TERM *out_err_atom) {
    int rc = -1;
    char **patterns = NULL;
    unsigned int n_patterns = 0;
    llama_token *trig_tokens = NULL;
    int32_t n_trig_tokens = 0;
    unsigned char *keybuf = NULL;
    struct llama_sampler *g = NULL;

    const struct llama_model *model = erllama_safe_get_model(c->ctx);
    const struct llama_vocab *vocab =
        model ? erllama_safe_model_get_vocab(model) : NULL;
    if (!vocab) {
        *out_err_atom = atom_exception;
        return -1;
    }

    int lazy = 0;
    {
        int b;
        if (get_map_bool(env, cfg, "grammar_lazy", &b)) lazy = b;
    }
    ERL_NIF_TERM v;
    if (enif_get_map_value(env, cfg,
                           enif_make_atom(env, "trigger_patterns"), &v)) {
        if (read_string_list(env, v, &patterns, &n_patterns) != 0) {
            *out_err_atom = enif_make_atom(env, "badarg");
            goto out;
        }
    }
    if (enif_get_map_value(env, cfg,
                           enif_make_atom(env, "trigger_tokens"), &v)) {
        if (read_token_list(env, v, &trig_tokens, &n_trig_tokens) != 1) {
            *out_err_atom = enif_make_atom(env, "badarg");
            goto out;
        }
    }
    ErlNifBinary prefill_bin;
    int has_prefill = 0;
    if (enif_get_map_value(env, cfg,
                           enif_make_atom(env, "grammar_prefill"), &v)) {
        if (!enif_inspect_iolist_as_binary(env, v, &prefill_bin)) {
            *out_err_atom = enif_make_atom(env, "badarg");
            goto out;
        }
        has_prefill = prefill_bin.size > 0;
    }

    /* Cache key: plain grammars keep the raw GBNF bytes (existing
     * behavior). Lazy grammars / trigger sets get a canonical buffer
     * [lazy-byte, grammar, 0, pattern 0, ..., tokens LE] so the same
     * GBNF compiled lazy vs plain occupies two distinct entries. */
    const unsigned char *key = grammar_bin.data;
    size_t keylen = grammar_bin.size;
    if (lazy || n_patterns > 0 || n_trig_tokens > 0) {
        keylen = 1 + grammar_bin.size + 1;
        for (unsigned int i = 0; i < n_patterns; i++) {
            keylen += strlen(patterns[i]) + 1;
        }
        keylen += (size_t) n_trig_tokens * 4;
        keybuf = enif_alloc(keylen);
        if (!keybuf) {
            *out_err_atom = atom_oom;
            goto out;
        }
        unsigned char *p = keybuf;
        *p++ = (unsigned char) (lazy ? 1 : 0);
        memcpy(p, grammar_bin.data, grammar_bin.size);
        p += grammar_bin.size;
        *p++ = 0;
        for (unsigned int i = 0; i < n_patterns; i++) {
            size_t plen = strlen(patterns[i]);
            memcpy(p, patterns[i], plen);
            p += plen;
            *p++ = 0;
        }
        for (int32_t i = 0; i < n_trig_tokens; i++) {
            uint32_t t = (uint32_t) trig_tokens[i];
            *p++ = (unsigned char) (t & 0xFF);
            *p++ = (unsigned char) ((t >> 8) & 0xFF);
            *p++ = (unsigned char) ((t >> 16) & 0xFF);
            *p++ = (unsigned char) ((t >> 24) & 0xFF);
        }
        key = keybuf;
    }

    /* Parse the GBNF at most once per distinct grammar (per context):
     * cache the parsed template and clone it per request. Re-parsing a
     * large tool grammar every turn dominates infer admission. */
    struct llama_sampler *tmpl = grammar_cache_get(c, key, keylen);
    int from_cache = (tmpl != NULL);
    if (!tmpl) {
        char *gstr = enif_alloc(grammar_bin.size + 1);
        if (!gstr) {
            *out_err_atom = atom_oom;
            goto out;
        }
        memcpy(gstr, grammar_bin.data, grammar_bin.size);
        gstr[grammar_bin.size] = '\0';
        if (lazy) {
            tmpl = erllama_safe_sampler_init_grammar_lazy_patterns(
                vocab, gstr, "root",
                (const char **) patterns, (size_t) n_patterns,
                trig_tokens, (size_t) n_trig_tokens);
        } else {
            tmpl = erllama_safe_sampler_init_grammar(vocab, gstr, "root");
        }
        enif_free(gstr);
        if (!tmpl) {
            *out_err_atom = atom_grammar_failed;
            goto out;
        }
    }
    /* Clone before caching: grammar_cache_put may free `tmpl` on its own
     * alloc failure, and a cache hit must not be mutated by decode. */
    g = erllama_safe_sampler_clone(tmpl);
    if (!from_cache) {
        grammar_cache_put(c, key, keylen, tmpl);
    }
    if (!g) {
        *out_err_atom = atom_grammar_failed;
        goto out;
    }
    if (!lazy && has_prefill) {
        if (grammar_accept_prefill(vocab, g, prefill_bin.data,
                                   prefill_bin.size) != 0) {
            *out_err_atom = enif_make_atom(env, "grammar_prefill_failed");
            goto out;
        }
    }
    if (chain_append(chain, g) != 0) {
        g = NULL; /* chain_append freed or adopted it */
        *out_err_atom = atom_oom;
        goto out;
    }
    g = NULL;
    rc = 0;

out:
    if (g) (void) erllama_safe_sampler_free(g);
    if (keybuf) enif_free(keybuf);
    if (trig_tokens) enif_free(trig_tokens);
    free_string_list(patterns, n_patterns);
    return rc;
}

/* Append a stage to the chain; on failure (alloc, or a NULL stage
 * from a failed init) free the WHOLE chain and set *out_err_atom.
 * Returns 0 ok, -1 failed. */
static int chain_append_or_fail(struct llama_sampler *chain,
                                struct llama_sampler *stage,
                                ERL_NIF_TERM *out_err_atom) {
    if (chain_append(chain, stage) != 0) {
        (void) erllama_safe_sampler_free(chain);
        *out_err_atom = atom_oom;
        return -1;
    }
    return 0;
}

/* Build and append the logit-bias stage from the `logit_bias` list
 * (`[{TokenId, Bias}, ...]`) merged with `ignore_eos` (-inf on every
 * end-of-generation token, skipped when the model has no eos - same
 * as upstream common_params expansion). No-op when neither is set.
 * On failure frees the chain and sets *out_err_atom; returns 0 ok. */
static int append_logit_bias_sampler(ErlNifEnv *env, ERL_NIF_TERM cfg,
                                     const struct llama_vocab *vocab,
                                     int32_t n_vocab,
                                     struct llama_sampler *chain,
                                     ERL_NIF_TERM *out_err_atom) {
    ERL_NIF_TERM v;
    unsigned int n_user = 0;
    int has_list =
        enif_get_map_value(env, cfg, enif_make_atom(env, "logit_bias"), &v);
    if (has_list && !enif_get_list_length(env, v, &n_user)) {
        (void) erllama_safe_sampler_free(chain);
        *out_err_atom = enif_make_atom(env, "badarg");
        return -1;
    }
    int ignore_eos = 0;
    {
        int b;
        if (get_map_bool(env, cfg, "ignore_eos", &b)) ignore_eos = b;
    }
    if ((!has_list || n_user == 0) && !ignore_eos) {
        return 0;
    }
    if (!vocab || n_vocab <= 0) {
        return 0;
    }
    /* ignore_eos without an eos token is a silent no-op (upstream
     * logs and skips). Count the eog ids first to size the array. */
    int32_t n_eog = 0;
    if (ignore_eos) {
        if (erllama_safe_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
            ignore_eos = 0;
        } else {
            for (int32_t t = 0; t < n_vocab; t++) {
                if (erllama_safe_vocab_is_eog(vocab, t)) n_eog++;
            }
        }
    }
    size_t total = (size_t) n_user + (size_t) n_eog;
    if (total == 0) {
        return 0;
    }
    llama_logit_bias *bias = enif_alloc(sizeof(llama_logit_bias) * total);
    if (!bias) {
        (void) erllama_safe_sampler_free(chain);
        *out_err_atom = atom_oom;
        return -1;
    }
    size_t n = 0;
    if (has_list) {
        ERL_NIF_TERM head, tail = v;
        while (enif_get_list_cell(env, tail, &head, &tail)) {
            const ERL_NIF_TERM *pair;
            int arity;
            int tok;
            double bval;
            if (!enif_get_tuple(env, head, &arity, &pair) || arity != 2 ||
                !enif_get_int(env, pair[0], &tok) || tok < 0 ||
                tok >= n_vocab || !enif_get_double(env, pair[1], &bval)) {
                enif_free(bias);
                (void) erllama_safe_sampler_free(chain);
                *out_err_atom = enif_make_atom(env, "badarg");
                return -1;
            }
            bias[n].token = (llama_token) tok;
            bias[n].bias = (float) bval;
            n++;
        }
    }
    if (ignore_eos) {
        for (int32_t t = 0; t < n_vocab && n < total; t++) {
            if (erllama_safe_vocab_is_eog(vocab, t)) {
                bias[n].token = t;
                bias[n].bias = -INFINITY;
                n++;
            }
        }
    }
    struct llama_sampler *stage =
        erllama_safe_sampler_init_logit_bias(n_vocab, (int32_t) n, bias);
    enif_free(bias);
    return chain_append_or_fail(chain, stage, out_err_atom);
}

/* Build and append the DRY repetition sampler. Breakers default to
 * upstream's {"\n", ":", "\"", "*"} when the caller supplies none.
 * On failure frees the chain and sets *out_err_atom; returns 0 ok. */
static int append_dry_sampler(ErlNifEnv *env, ERL_NIF_TERM cfg,
                              const struct llama_vocab *vocab,
                              double multiplier, double base,
                              int32_t allowed_length, int32_t penalty_last_n,
                              struct llama_sampler *chain,
                              ERL_NIF_TERM *out_err_atom) {
    static const char *default_breakers[] = {"\n", ":", "\"", "*"};
    if (!vocab) {
        return 0;
    }
    char **user_breakers = NULL;
    unsigned int n_user = 0;
    ERL_NIF_TERM v;
    if (enif_get_map_value(env, cfg,
                           enif_make_atom(env, "dry_sequence_breakers"), &v)) {
        if (read_string_list(env, v, &user_breakers, &n_user) != 0) {
            (void) erllama_safe_sampler_free(chain);
            *out_err_atom = enif_make_atom(env, "badarg");
            return -1;
        }
    }
    struct llama_sampler *stage;
    if (n_user > 0) {
        stage = erllama_safe_sampler_init_dry(
            vocab, (float) multiplier, (float) base, allowed_length,
            penalty_last_n, (const char **) user_breakers, (size_t) n_user);
    } else {
        stage = erllama_safe_sampler_init_dry(
            vocab, (float) multiplier, (float) base, allowed_length,
            penalty_last_n, default_breakers,
            sizeof(default_breakers) / sizeof(default_breakers[0]));
    }
    free_string_list(user_breakers, n_user);
    return chain_append_or_fail(chain, stage, out_err_atom);
}

/* Build a sampler chain from a config map. On failure returns NULL and
 * sets *out_err_atom to one of: atom_oom, atom_grammar_failed,
 * atom_badarg. The lock must already be held by the caller (vocab
 * lookup uses c->ctx). */
struct llama_sampler *
build_sampler_chain_from_map(ErlNifEnv *env, ERL_NIF_TERM cfg,
                             erllama_context_t *c,
                             ERL_NIF_TERM *out_err_atom) {
    if (!enif_is_map(env, cfg)) {
        *out_err_atom = enif_make_atom(env, "badarg");
        return NULL;
    }

    /* Grammar requires the vocab; everything else does not. */
    ErlNifBinary grammar_bin;
    int has_grammar = 0;
    {
        ERL_NIF_TERM v;
        if (enif_get_map_value(env, cfg, enif_make_atom(env, "grammar"), &v)) {
            if (!enif_inspect_iolist_as_binary(env, v, &grammar_bin) ||
                grammar_bin.size == 0) {
                *out_err_atom = enif_make_atom(env, "badarg");
                return NULL;
            }
            has_grammar = 1;
        }
    }

    int32_t i32;
    double f64;
    int has_top_k = get_map_int31(env, cfg, "top_k", &i32);
    int32_t top_k_val = has_top_k ? i32 : 0;

    int has_top_p = get_map_double(env, cfg, "top_p", &f64);
    double top_p_val = has_top_p ? f64 : 1.0;

    int has_min_p = get_map_double(env, cfg, "min_p", &f64);
    double min_p_val = has_min_p ? f64 : 0.0;

    int has_temp = get_map_double(env, cfg, "temperature", &f64);
    double temp_val = has_temp ? f64 : 0.0;

    int has_rep = get_map_double(env, cfg, "repetition_penalty", &f64);
    double rep_val = has_rep ? f64 : 1.0;

    double freq_val = get_map_double(env, cfg, "frequency_penalty", &f64) ? f64 : 0.0;
    double present_val = get_map_double(env, cfg, "presence_penalty", &f64) ? f64 : 0.0;
    int32_t pen_last_n = get_map_int31(env, cfg, "penalty_last_n", &i32) ? i32 : 64;

    int has_typical = get_map_double(env, cfg, "typical_p", &f64);
    double typical_val = has_typical ? f64 : 1.0;

    int has_tns = get_map_double(env, cfg, "top_n_sigma", &f64);
    double tns_val = has_tns ? f64 : -1.0;

    double xtc_p = get_map_double(env, cfg, "xtc_probability", &f64) ? f64 : 0.0;
    double xtc_t = get_map_double(env, cfg, "xtc_threshold", &f64) ? f64 : 0.1;

    double dt_range = get_map_double(env, cfg, "dynatemp_range", &f64) ? f64 : 0.0;
    double dt_exp = get_map_double(env, cfg, "dynatemp_exponent", &f64) ? f64 : 1.0;

    int32_t min_keep = get_map_int31(env, cfg, "min_keep", &i32) ? i32 : 1;
    if (min_keep < 1) min_keep = 1;

    double dry_mult = get_map_double(env, cfg, "dry_multiplier", &f64) ? f64 : 0.0;
    double dry_base = get_map_double(env, cfg, "dry_base", &f64) ? f64 : 1.75;
    int32_t dry_allowed = get_map_int31(env, cfg, "dry_allowed_length", &i32) ? i32 : 2;
    int32_t dry_last_n = get_map_int31(env, cfg, "dry_penalty_last_n", &i32) ? i32 : 64;

    int32_t mirostat_val = get_map_int31(env, cfg, "mirostat", &i32) ? i32 : 0;
    double miro_tau = get_map_double(env, cfg, "mirostat_tau", &f64) ? f64 : 5.0;
    double miro_eta = get_map_double(env, cfg, "mirostat_eta", &f64) ? f64 : 0.1;

    int infill_val = 0;
    {
        int b;
        if (get_map_bool(env, cfg, "infill", &b)) infill_val = b;
    }

    uint32_t seed_val = 0;
    int has_seed = 0;
    {
        ERL_NIF_TERM v;
        if (enif_get_map_value(env, cfg, enif_make_atom(env, "seed"), &v)) {
            unsigned long seed_ul;
            if (!enif_get_ulong(env, v, &seed_ul)) {
                *out_err_atom = enif_make_atom(env, "badarg");
                return NULL;
            }
            seed_val = (uint32_t) seed_ul;
            has_seed = 1;
        }
    }

    const struct llama_vocab *vocab =
        erllama_safe_model_get_vocab(c->model_res->model);
    int32_t n_vocab = vocab ? erllama_safe_vocab_n_tokens(vocab) : 0;

    struct llama_sampler_chain_params sp =
        llama_sampler_chain_default_params();
    struct llama_sampler *chain = erllama_safe_sampler_chain_init(sp);
    if (!chain) {
        *out_err_atom = atom_oom;
        return NULL;
    }

    if (has_grammar) {
        if (append_grammar_sampler(env, cfg, c, chain, grammar_bin,
                                   out_err_atom) != 0) {
            (void) erllama_safe_sampler_free(chain);
            return NULL;
        }
    }

    /* logit_bias + ignore_eos, merged into one stage (upstream puts
     * logit_bias first in the chain, right after the grammar). */
    if (append_logit_bias_sampler(env, cfg, vocab, n_vocab, chain,
                                  out_err_atom) != 0) {
        return NULL; /* helper freed the chain */
    }

    /* Mirostat overrides the whole middle of the chain, exactly as
     * upstream common_sampler_init does: temp -> mirostat terminal. */
    if (mirostat_val == 1 || mirostat_val == 2) {
        if (has_temp && temp_val > 0.0) {
            if (chain_append_or_fail(
                    chain, erllama_safe_sampler_init_temp((float) temp_val),
                    out_err_atom) != 0) {
                return NULL;
            }
        }
        struct llama_sampler *miro =
            mirostat_val == 1
                ? erllama_safe_sampler_init_mirostat(
                      n_vocab, seed_val, (float) miro_tau, (float) miro_eta, 100)
                : erllama_safe_sampler_init_mirostat_v2(
                      seed_val, (float) miro_tau, (float) miro_eta);
        if (chain_append_or_fail(chain, miro, out_err_atom) != 0) {
            return NULL;
        }
        return chain;
    }

    if ((has_rep && rep_val != 1.0) || freq_val != 0.0 || present_val != 0.0) {
        if (chain_append_or_fail(
                chain,
                erllama_safe_sampler_init_penalties(
                    n_vocab, pen_last_n, (float) rep_val, (float) freq_val,
                    (float) present_val),
                out_err_atom) != 0) {
            return NULL;
        }
    }
    if (dry_mult > 0.0) {
        if (append_dry_sampler(env, cfg, vocab, dry_mult, dry_base,
                               dry_allowed, dry_last_n, chain,
                               out_err_atom) != 0) {
            return NULL; /* helper freed the chain */
        }
    }
    if (has_tns && tns_val > 0.0) {
        if (chain_append_or_fail(
                chain, erllama_safe_sampler_init_top_n_sigma((float) tns_val),
                out_err_atom) != 0) {
            return NULL;
        }
    }
    if (has_top_k && top_k_val > 0) {
        if (chain_append_or_fail(chain,
                                 erllama_safe_sampler_init_top_k(top_k_val),
                                 out_err_atom) != 0) {
            return NULL;
        }
    }
    if (has_typical && typical_val < 1.0) {
        if (chain_append_or_fail(
                chain,
                erllama_safe_sampler_init_typical((float) typical_val,
                                                  (size_t) min_keep),
                out_err_atom) != 0) {
            return NULL;
        }
    }
    if (has_top_p && top_p_val < 1.0) {
        if (chain_append_or_fail(
                chain,
                erllama_safe_sampler_init_top_p((float) top_p_val,
                                                (size_t) min_keep),
                out_err_atom) != 0) {
            return NULL;
        }
    }
    if (has_min_p && min_p_val > 0.0) {
        if (chain_append_or_fail(
                chain,
                erllama_safe_sampler_init_min_p((float) min_p_val,
                                                (size_t) min_keep),
                out_err_atom) != 0) {
            return NULL;
        }
    }
    if (xtc_p > 0.0) {
        if (chain_append_or_fail(
                chain,
                erllama_safe_sampler_init_xtc((float) xtc_p, (float) xtc_t,
                                              (size_t) min_keep, seed_val),
                out_err_atom) != 0) {
            return NULL;
        }
    }
    if (infill_val && vocab) {
        if (chain_append_or_fail(chain,
                                 erllama_safe_sampler_init_infill(vocab),
                                 out_err_atom) != 0) {
            return NULL;
        }
    }
    if (has_temp && temp_val > 0.0) {
        struct llama_sampler *temp_stage =
            dt_range > 0.0
                ? erllama_safe_sampler_init_temp_ext(
                      (float) temp_val, (float) dt_range, (float) dt_exp)
                : erllama_safe_sampler_init_temp((float) temp_val);
        if (chain_append_or_fail(chain, temp_stage, out_err_atom) != 0) {
            return NULL;
        }
        if (chain_append_or_fail(chain,
                                 erllama_safe_sampler_init_dist(seed_val),
                                 out_err_atom) != 0) {
            return NULL;
        }
    } else {
        /* temperature == 0 or absent: greedy terminal. */
        if (chain_append_or_fail(chain, erllama_safe_sampler_init_greedy(),
                                 out_err_atom) != 0) {
            return NULL;
        }
        (void) has_seed; /* seed without temperature is a no-op. */
    }

    return chain;
}

/* Shared prologue of nif_configure_sampler / nif_sampler_new: fetch
 * and validate the {CtxRef, CfgMap} argument pair, take the context
 * lock, and build the chain. On success returns the chain with
 * *out_c set and c->mu still HELD (the caller mutates and unlocks).
 * On failure returns NULL with no lock held and *out_ret set to the
 * term to return. */
static struct llama_sampler *chain_from_args(
    ErlNifEnv *env, const ERL_NIF_TERM argv[],
    erllama_context_t **out_c, ERL_NIF_TERM *out_ret
) {
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        *out_ret = enif_make_badarg(env);
        return NULL;
    }
    if (!enif_is_map(env, argv[1])) {
        *out_ret = enif_make_badarg(env);
        return NULL;
    }
    if (!erllama_lock_ctx(c)) {
        *out_ret = erllama_error(env, atom_released);
        return NULL;
    }
    ERL_NIF_TERM err = atom_oom;
    struct llama_sampler *chain =
        build_sampler_chain_from_map(env, argv[1], c, &err);
    if (!chain) {
        pthread_mutex_unlock(&c->mu);
        *out_ret = erllama_error(env, err);
        return NULL;
    }
    *out_c = c;
    return chain;
}

ERL_NIF_TERM nif_configure_sampler(ErlNifEnv *env, int argc,
                                          const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c = NULL;
    ERL_NIF_TERM ret;
    struct llama_sampler *chain = chain_from_args(env, argv, &c, &ret);
    if (!chain) {
        return ret;
    }
    if (c->smpl) {
        (void) erllama_safe_sampler_free(c->smpl);
    }
    c->smpl = chain;
    pthread_mutex_unlock(&c->mu);
    return atom_ok;
}

/* Backwards-compatible: builds a chain with only a grammar entry. */
ERL_NIF_TERM nif_set_grammar(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
    (void) argc;
    ERL_NIF_TERM cfg = enif_make_new_map(env);
    enif_make_map_put(env, cfg, enif_make_atom(env, "grammar"), argv[1], &cfg);
    ERL_NIF_TERM new_argv[2] = {argv[0], cfg};
    return nif_configure_sampler(env, 2, new_argv);
}

ERL_NIF_TERM nif_clear_sampler(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&c->mu);
    if (c->smpl) {
        (void) erllama_safe_sampler_free(c->smpl);
        c->smpl = NULL;
    }
    pthread_mutex_unlock(&c->mu);
    return atom_ok;
}

/* =========================================================================
 * Per-request sampler resource (Phase 4 infrastructure)
 *
 * Wraps a llama_sampler_chain built by build_sampler_chain_from_map
 * so multiple in-flight requests (v0.2+) can hold independent chains.
 * The v0.1 model layer still uses configure_sampler/2 against the
 * context's cached `c->smpl`; this resource is the building block
 * for the eventual decode_and_sample_batch NIF.
 * ========================================================================= */

ERL_NIF_TERM nif_sampler_new(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c = NULL;
    ERL_NIF_TERM ret;
    struct llama_sampler *chain = chain_from_args(env, argv, &c, &ret);
    if (!chain) {
        return ret;
    }
    pthread_mutex_unlock(&c->mu);
    erllama_sampler_t *res = enif_alloc_resource(SAMPLER_RT, sizeof(*res));
    if (!res) {
        (void) erllama_safe_sampler_free(chain);
        return erllama_error(env, atom_oom);
    }
    memset(res, 0, sizeof(*res));
    if (pthread_mutex_init(&res->mu, NULL) != 0) {
        enif_release_resource(res);
        (void) erllama_safe_sampler_free(chain);
        return erllama_error(env, atom_oom);
    }
    res->mu_inited = 1;
    res->chain = chain;
    res->ctx_res = c;
    /* Per-request logprobs count (0 = off). Clamped defensively; the
     * Erlang validator already enforces 0..32. */
    {
        int32_t np;
        if (get_map_int31(env, argv[1], "logprobs", &np)) {
            if (np < 0) np = 0;
            if (np > 32) np = 32;
            res->n_probs = (int) np;
        }
    }
    enif_keep_resource(c);

    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return enif_make_tuple2(env, atom_ok, term);
}

ERL_NIF_TERM nif_sampler_free(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_sampler_t *s;
    if (!enif_get_resource(env, argv[0], SAMPLER_RT, (void **) &s)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&s->mu);
    if (!s->chain) {
        pthread_mutex_unlock(&s->mu);
        return erllama_error(env, atom_released);
    }
    (void) erllama_safe_sampler_free(s->chain);
    s->chain = NULL;
    pthread_mutex_unlock(&s->mu);
    return atom_ok;
}

/* Per-context grammar-cache stats: #{hits => N, misses => N}. Advisory
 * metric so callers can confirm the compiled-grammar cache is taking
 * effect (repeat tool turns should be hits). */
ERL_NIF_TERM
nif_grammar_cache_stats(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&c->mu);
    uint64_t hits = c->gcache_hits;
    uint64_t misses = c->gcache_misses;
    pthread_mutex_unlock(&c->mu);
    ERL_NIF_TERM m = enif_make_new_map(env);
    erllama_map_put(env, &m, "hits", enif_make_uint64(env, hits));
    erllama_map_put(env, &m, "misses", enif_make_uint64(env, misses));
    return m;
}
