/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */

/* MTP speculative decoding (experimental).
 *
 *   nif_mtp_new(Ctx, DraftModel, #{n_max}) -> {ok, Mtp} | {error, _}
 *   nif_mtp_step(Ctx, Sampler, Mtp, Seq, MaxDraft)
 *     -> {ok, {spec, [Tok], Eog, NAcc}} | {error, _}
 *   nif_mtp_stats(Mtp) -> map()
 *   nif_mtp_trace(Mtp, on | off | read) -> ok | [map()]
 *   nif_mtp_free(Mtp) -> ok | {error, released}
 *
 * A drafter (erllama_mtp.h) attached to a context: one per context,
 * fed after every decode of that context (erllama_ctx_decode), so its
 * own context follows the target's. When the target is a model split
 * over a tensor-parallel group, the drafter runs on the serving rank
 * only; the other ranks turn on the same nextn embeddings (`{nextn,
 * On}' in the command stream), so every rank builds the same graphs.
 *
 * A step is one round: sample the next token K from the last logits,
 * draft up to MaxDraft tokens after it, decode K and the draft in one
 * batch with logits on every row, keep the drafted tokens the target
 * agrees with (argmax, so only for a greedy sampler), and remove the
 * others from the target and from the drafter. The token after the
 * kept ones is not decoded: the next round samples it from the kept
 * row, as a plain decode would. Every returned token is in the KV.
 * On a hybrid model the removal needs recurrent-state snapshots
 * (`n_rs_seq' >= MaxDraft).
 *
 * Lock order: the context's, then the sampler's. The drafter is used
 * only under the context's lock. */

#include "erllama_nif_int.h"
#include "erllama_safe.h"
#include "erllama_mtp.h"

#include <string.h>

#define ERLLAMA_MTP_MAX_DRAFT 16
#define ERLLAMA_MTP_TRACE_TOP 8
#define ERLLAMA_MTP_TRACE_CAP 4096

/* One traced round (nif_mtp_trace): where it started, the batch, how
 * many drafts were kept, and the top logits of every row. */
typedef struct {
    int32_t pos;
    int32_t n;
    int32_t n_acc;
    int32_t tok[ERLLAMA_MTP_MAX_DRAFT + 1];
    int32_t top_id[ERLLAMA_MTP_MAX_DRAFT + 1][ERLLAMA_MTP_TRACE_TOP];
    float top_v[ERLLAMA_MTP_MAX_DRAFT + 1][ERLLAMA_MTP_TRACE_TOP];
} erllama_mtp_round_t;

typedef struct erllama_mtp_res {
    erllama_context_t *ctx_res;  /* keep_resource'd */
    erllama_model_t *draft_res;  /* keep_resource'd, counted as a context */
    struct erllama_mtp *m;       /* NULL once detached */
    uint64_t rounds;
    uint64_t drafted;
    uint64_t accepted;
    uint64_t draft_us;
    uint64_t verify_us;
    uint64_t rollback_us;
    /* Rounds after which the drafter's context did not end where the
     * target's does (it should resume from the accepted position). */
    uint64_t misaligned;
    /* The rounds right after a rejection, and what they accepted. */
    uint64_t after_reject_rounds;
    uint64_t after_reject_accepted;
    uint8_t rejected[ERLLAMA_N_SEQ_MAX_CAP];
    /* Rounds recorded while tracing; NULL when off. */
    erllama_mtp_round_t *trace;
    int32_t n_trace;
} erllama_mtp_res_t;

static ErlNifResourceType *MTP_RT = NULL;

static uint64_t mono_us(void) {
    return (uint64_t) enif_monotonic_time(ERL_NIF_USEC);
}

static void send_nextn(ErlNifEnv *env, erllama_context_t *c, int on) {
    if (c->tp_sink_set) {
        ErlNifEnv *menv = enif_alloc_env();
        erllama_tp_send(env, c, menv,
                        enif_make_tuple2(menv, enif_make_atom(menv, "nextn"), enif_make_int(menv, on)));
    }
}

/* Caller holds c->mu. Frees the drafter; with `target' still live,
 * turns its nextn embeddings off on every rank. */
static void mtp_detach_locked(ErlNifEnv *env, erllama_mtp_res_t *r) {
    erllama_context_t *c = r->ctx_res;
    if (!r->m) {
        return;
    }
    erllama_mtp_free(r->m);
    r->m = NULL;
    if (c->mtp == r) {
        c->mtp = NULL;
    }
    if (c->ctx) {
        erllama_mtp_target_nextn(c->ctx, 0);
        send_nextn(env, c, 0);
    }
    context_drops_model(r->draft_res);
}

static void mtp_dtor(ErlNifEnv *env, void *obj) {
    erllama_mtp_res_t *r = (erllama_mtp_res_t *) obj;
    if (r->ctx_res) {
        pthread_mutex_lock(&r->ctx_res->mu);
        mtp_detach_locked(env, r);
        pthread_mutex_unlock(&r->ctx_res->mu);
        enif_release_resource(r->ctx_res);
        r->ctx_res = NULL;
    }
    if (r->draft_res) {
        enif_release_resource(r->draft_res);
        r->draft_res = NULL;
    }
    if (r->trace) {
        enif_free(r->trace);
        r->trace = NULL;
    }
}

int erllama_mtp_nif_load(ErlNifEnv *env) {
    MTP_RT = enif_open_resource_type(env, NULL, "erllama_mtp", mtp_dtor, ERL_NIF_RT_CREATE, NULL);
    return MTP_RT ? 0 : -1;
}

void erllama_ctx_mtp_after_decode(erllama_context_t *c, const struct llama_batch *b) {
    if (c->mtp && c->mtp->m) {
        erllama_mtp_process(c->mtp->m, b);
    }
}

void erllama_ctx_mtp_seq_reset(erllama_context_t *c, int seq) {
    if (c->mtp && c->mtp->m) {
        erllama_mtp_seq_rm(c->mtp->m, seq, 0, -1);
    }
}

void erllama_ctx_mtp_release(ErlNifEnv *env, erllama_context_t *c) {
    if (c->mtp) {
        mtp_detach_locked(env, c->mtp);
    }
}

ERL_NIF_TERM nif_mtp_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    erllama_model_t *dm;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c) ||
        !enif_get_resource(env, argv[1], MODEL_RT, (void **) &dm) || !enif_is_map(env, argv[2])) {
        return enif_make_badarg(env);
    }
    int32_t n_max = 2;
    if (get_map_int31(env, argv[2], "n_max", &n_max) && (n_max < 1 || n_max > ERLLAMA_MTP_MAX_DRAFT)) {
        return enif_make_badarg(env);
    }
    if (!erllama_lock_ctx(c)) {
        return erllama_error(env, atom_released);
    }
    if (c->mtp) {
        pthread_mutex_unlock(&c->mu);
        return erllama_error(env, enif_make_atom(env, "mtp_attached"));
    }
    if (!erllama_lock_model(dm)) {
        pthread_mutex_unlock(&c->mu);
        return erllama_error(env, atom_released);
    }
    erllama_mtp_res_t *r = enif_alloc_resource(MTP_RT, sizeof(*r));
    if (!r) {
        pthread_mutex_unlock(&dm->mu);
        pthread_mutex_unlock(&c->mu);
        return erllama_error(env, atom_oom);
    }
    memset(r, 0, sizeof(*r));
    const char *why = "exception";
    r->m = erllama_mtp_new(dm->model, c->ctx, n_max, &why);
    if (!r->m) {
        pthread_mutex_unlock(&dm->mu);
        pthread_mutex_unlock(&c->mu);
        enif_release_resource(r);
        return erllama_error(env, enif_make_atom(env, why));
    }
    /* The drafter turned on the target's nextn embeddings: the other
     * ranks do the same before the next batch. */
    send_nextn(env, c, 1);
    dm->active_contexts++;
    enif_keep_resource(dm);
    r->draft_res = dm;
    pthread_mutex_unlock(&dm->mu);
    enif_keep_resource(c);
    r->ctx_res = c;
    c->mtp = r;
    pthread_mutex_unlock(&c->mu);
    ERL_NIF_TERM term = enif_make_resource(env, r);
    enif_release_resource(r);
    return enif_make_tuple2(env, atom_ok, term);
}

ERL_NIF_TERM nif_mtp_free(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_mtp_res_t *r;
    if (!enif_get_resource(env, argv[0], MTP_RT, (void **) &r)) {
        return enif_make_badarg(env);
    }
    if (!r->ctx_res) {
        return erllama_error(env, atom_released);
    }
    pthread_mutex_lock(&r->ctx_res->mu);
    int live = r->m != NULL;
    mtp_detach_locked(env, r);
    pthread_mutex_unlock(&r->ctx_res->mu);
    return live ? atom_ok : erllama_error(env, atom_released);
}

static ERL_NIF_TERM put_u64(ErlNifEnv *env, ERL_NIF_TERM map, const char *k, uint64_t v) {
    ERL_NIF_TERM out;
    (void) enif_make_map_put(env, map, enif_make_atom(env, k), enif_make_uint64(env, v), &out);
    return out;
}

ERL_NIF_TERM nif_mtp_stats(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_mtp_res_t *r;
    if (!enif_get_resource(env, argv[0], MTP_RT, (void **) &r)) {
        return enif_make_badarg(env);
    }
    if (!r->ctx_res) {
        return erllama_error(env, atom_released);
    }
    pthread_mutex_lock(&r->ctx_res->mu);
    erllama_mtp_counters_t k;
    erllama_mtp_counters(r->m, &k);
    ERL_NIF_TERM map = enif_make_new_map(env);
    map = put_u64(env, map, "rounds", r->rounds);
    map = put_u64(env, map, "drafted", r->drafted);
    map = put_u64(env, map, "accepted", r->accepted);
    map = put_u64(env, map, "draft_us", r->draft_us);
    map = put_u64(env, map, "verify_us", r->verify_us);
    map = put_u64(env, map, "rollback_us", r->rollback_us);
    map = put_u64(env, map, "misaligned", r->misaligned);
    map = put_u64(env, map, "after_reject_rounds", r->after_reject_rounds);
    map = put_u64(env, map, "after_reject_accepted", r->after_reject_accepted);
    map = put_u64(env, map, "process_calls", k.process_calls);
    map = put_u64(env, map, "process_failures", k.process_failures);
    map = put_u64(env, map, "process_us", k.process_us);
    pthread_mutex_unlock(&r->ctx_res->mu);
    return map;
}

/* The `k' largest logits of `row', largest first. */
static void top_k(const float *row, int32_t n_vocab, int32_t k, int32_t *ids, float *vals) {
    int32_t n = 0;
    for (int32_t v = 0; v < n_vocab; v++) {
        if (n == k && row[v] <= vals[k - 1]) {
            continue;
        }
        int32_t j = n < k ? n++ : k - 1;
        while (j > 0 && vals[j - 1] < row[v]) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j--;
        }
        vals[j] = row[v];
        ids[j] = v;
    }
}

static void trace_round(erllama_mtp_res_t *r, int32_t pos, const llama_token *toks, int32_t n,
                        int32_t n_acc, const float *logits, int32_t n_vocab) {
    if (!r->trace || r->n_trace >= ERLLAMA_MTP_TRACE_CAP) {
        return;
    }
    erllama_mtp_round_t *t = &r->trace[r->n_trace++];
    t->pos = pos;
    t->n = n;
    t->n_acc = n_acc;
    for (int32_t i = 0; i < n; i++) {
        t->tok[i] = toks[i];
        top_k(logits + (size_t) i * (size_t) n_vocab, n_vocab, ERLLAMA_MTP_TRACE_TOP, t->top_id[i],
              t->top_v[i]);
    }
}

/* nif_mtp_trace(Mtp, on | off | read): record every round from `on'
 * (up to 4096, dropping what was recorded), stop, or return the rounds
 * recorded since `on' as maps #{pos, tokens, n_acc, rows => [[{Id,
 * Logit}]]} (tests: a round checked against a fresh context). */
ERL_NIF_TERM nif_mtp_trace(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_mtp_res_t *r;
    char op[8];
    if (!enif_get_resource(env, argv[0], MTP_RT, (void **) &r) ||
        !enif_get_atom(env, argv[1], op, sizeof(op), ERL_NIF_LATIN1)) {
        return enif_make_badarg(env);
    }
    if (!r->ctx_res) {
        return erllama_error(env, atom_released);
    }
    ERL_NIF_TERM ret = atom_ok;
    pthread_mutex_lock(&r->ctx_res->mu);
    if (strcmp(op, "on") == 0) {
        if (!r->trace) {
            r->trace = enif_alloc(sizeof(erllama_mtp_round_t) * ERLLAMA_MTP_TRACE_CAP);
        }
        r->n_trace = 0;
        ret = r->trace ? atom_ok : erllama_error(env, atom_oom);
    } else if (strcmp(op, "off") == 0) {
        if (r->trace) {
            enif_free(r->trace);
            r->trace = NULL;
        }
        r->n_trace = 0;
    } else if (strcmp(op, "read") == 0) {
        ret = enif_make_list(env, 0);
        for (int32_t k = (r->trace ? r->n_trace : 0) - 1; k >= 0; k--) {
            const erllama_mtp_round_t *t = &r->trace[k];
            ERL_NIF_TERM toks = enif_make_list(env, 0);
            ERL_NIF_TERM rows = enif_make_list(env, 0);
            for (int32_t i = t->n - 1; i >= 0; i--) {
                toks = enif_make_list_cell(env, enif_make_int(env, t->tok[i]), toks);
                ERL_NIF_TERM row = enif_make_list(env, 0);
                for (int32_t j = ERLLAMA_MTP_TRACE_TOP - 1; j >= 0; j--) {
                    row = enif_make_list_cell(
                        env,
                        enif_make_tuple2(env, enif_make_int(env, t->top_id[i][j]),
                                         enif_make_double(env, (double) t->top_v[i][j])),
                        row);
                }
                rows = enif_make_list_cell(env, row, rows);
            }
            ERL_NIF_TERM m = enif_make_new_map(env);
            (void) enif_make_map_put(env, m, enif_make_atom(env, "pos"), enif_make_int(env, t->pos), &m);
            (void) enif_make_map_put(env, m, enif_make_atom(env, "tokens"), toks, &m);
            (void) enif_make_map_put(env, m, enif_make_atom(env, "n_acc"), enif_make_int(env, t->n_acc), &m);
            (void) enif_make_map_put(env, m, enif_make_atom(env, "rows"), rows, &m);
            ret = enif_make_list_cell(env, m, ret);
        }
    } else {
        ret = enif_make_badarg(env);
    }
    pthread_mutex_unlock(&r->ctx_res->mu);
    return ret;
}

static int32_t argmax_row(const float *row, int32_t n_vocab) {
    int32_t best = 0;
    for (int32_t v = 1; v < n_vocab; v++) {
        if (row[v] > row[best]) {
            best = v;
        }
    }
    return best;
}

/* After a failed decode or removal the KV and logits may be torn: a
 * fresh prefill must run before the next sample. */
static ERL_NIF_TERM step_fail(ErlNifEnv *env, erllama_context_t *c, erllama_sampler_t *s, int seq,
                              ERL_NIF_TERM why) {
    c->decode_ready = 0;
    c->per_seq[seq].last_logits_idx = -1;
    pthread_mutex_unlock(&s->mu);
    pthread_mutex_unlock(&c->mu);
    return erllama_error(env, why);
}

ERL_NIF_TERM nif_mtp_step(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    erllama_sampler_t *s;
    erllama_mtp_res_t *r;
    int seq, max_draft;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c) ||
        !enif_get_resource(env, argv[1], SAMPLER_RT, (void **) &s) ||
        !enif_get_resource(env, argv[2], MTP_RT, (void **) &r) || !enif_get_int(env, argv[3], &seq) ||
        seq < 0 || seq >= ERLLAMA_N_SEQ_MAX_CAP || !enif_get_int(env, argv[4], &max_draft) ||
        max_draft < 0 || max_draft > ERLLAMA_MTP_MAX_DRAFT) {
        return enif_make_badarg(env);
    }
    if (!erllama_lock_ctx(c)) {
        return erllama_error(env, atom_released);
    }
    if (c->mtp != r || !r->m) {
        pthread_mutex_unlock(&c->mu);
        return erllama_error(env, atom_released);
    }
    int32_t li = c->per_seq[seq].last_logits_idx;
    /* The seq's own row, as nif_step and the batched round check:
     * removing another seq (a request that finished) clears the
     * context's decode_ready but leaves this row in place. */
    if (li == -1 || li == ERLLAMA_LOGITS_STALE) {
        pthread_mutex_unlock(&c->mu);
        return erllama_error(env, li == ERLLAMA_LOGITS_STALE ? atom_stale_logits : atom_no_logits);
    }
    int32_t n_vocab = 0;
    const struct llama_vocab *vocab = erllama_ctx_vocab(c, &n_vocab);
    uint32_t n_batch_cap = erllama_safe_n_batch(c->ctx);
    if (!vocab || n_vocab <= 0 || n_batch_cap < (uint32_t) max_draft + 1) {
        pthread_mutex_unlock(&c->mu);
        return erllama_error(env, atom_batch_overflow);
    }
    pthread_mutex_lock(&s->mu);
    if (!s->chain) {
        pthread_mutex_unlock(&s->mu);
        pthread_mutex_unlock(&c->mu);
        return erllama_error(env, atom_released);
    }
    if (s->ctx_res != c || s->n_probs > 0 || !erllama_mtp_chain_is_greedy(s->chain)) {
        pthread_mutex_unlock(&s->mu);
        pthread_mutex_unlock(&c->mu);
        return erllama_error(env, enif_make_atom(env, "not_greedy"));
    }

    const int32_t pos = c->per_seq[seq].next_pos;
    llama_token k = erllama_safe_sampler_sample(s->chain, c->ctx, li == ERLLAMA_LOGITS_LAST ? -1 : li);
    if (k < 0) {
        return step_fail(env, c, s, seq, atom_exception);
    }
    const int eog_k = erllama_safe_vocab_is_eog(vocab, k);

    llama_token draft[ERLLAMA_MTP_MAX_DRAFT];
    int32_t n_draft = 0;
    const uint64_t t0 = mono_us();
    if (!eog_k && max_draft > 0) {
        n_draft = erllama_mtp_draft(r->m, seq, k, pos, max_draft, draft);
        for (int32_t i = 0; i < n_draft; i++) {
            if (draft[i] < 0 || draft[i] >= n_vocab) {
                n_draft = i;
                break;
            }
        }
    }
    const uint64_t t1 = mono_us();

    struct llama_batch batch = erllama_safe_batch_init(n_draft + 1, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        erllama_safe_batch_free(batch);
        return step_fail(env, c, s, seq, atom_oom);
    }
    for (int32_t i = 0; i <= n_draft; i++) {
        batch.token[i] = i == 0 ? k : draft[i - 1];
        batch.pos[i] = pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = (llama_seq_id) seq;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n_draft + 1;
    arm_decode(c);
    int dr = erllama_ctx_decode(env, c, batch);
    erllama_safe_batch_free(batch);
    if (dr != 0) {
        return step_fail(env, c, s, seq,
                         dr == ERLLAMA_DECODE_EXC_SENTINEL ? atom_exception
                                                           : classify_decode_error(env, c, dr));
    }
    const uint64_t t2 = mono_us();

    const float *logits = erllama_safe_get_logits(c->ctx);
    if (!logits) {
        return step_fail(env, c, s, seq, atom_exception);
    }
    llama_token committed[ERLLAMA_MTP_MAX_DRAFT + 1];
    int32_t n_committed = 1;
    int32_t n_acc = 0;
    int eog = eog_k;
    committed[0] = k;
    for (int32_t i = 0; i < n_draft && !eog; i++) {
        if (argmax_row(logits + (size_t) i * (size_t) n_vocab, n_vocab) != draft[i]) {
            break;
        }
        committed[n_committed++] = draft[i];
        n_acc++;
        eog = erllama_safe_vocab_is_eog(vocab, draft[i]);
    }
    const int32_t keep = pos + 1 + n_acc;
    if (r->trace) {
        llama_token toks[ERLLAMA_MTP_MAX_DRAFT + 1];
        toks[0] = k;
        for (int32_t i = 0; i < n_draft; i++) {
            toks[i + 1] = draft[i];
        }
        trace_round(r, pos, toks, n_draft + 1, n_acc, logits, n_vocab);
    }
    if (n_acc < n_draft) {
        if (erllama_ctx_seq_rm(env, c, seq, keep, -1) != 0) {
            return step_fail(env, c, s, seq, atom_decode_failed);
        }
        erllama_mtp_seq_rm(r->m, seq, keep, -1);
    }
    erllama_mtp_accept(r->m, seq, n_acc);
    const uint64_t t3 = mono_us();
    if (erllama_mtp_seq_pos_max(r->m, seq) != keep - 1) {
        r->misaligned++;
    }
    if (r->rejected[seq] && n_draft > 0) {
        r->after_reject_rounds++;
        r->after_reject_accepted += (uint64_t) n_acc;
    }
    if (n_draft > 0) {
        r->rejected[seq] = n_acc < n_draft;
    }

    c->per_seq[seq].next_pos = keep;
    c->per_seq[seq].last_logits_idx = n_acc;
    c->decode_ready = 1;
    r->rounds++;
    r->drafted += (uint64_t) n_draft;
    r->accepted += (uint64_t) n_acc;
    r->draft_us += t1 - t0;
    r->verify_us += t2 - t1;
    r->rollback_us += t3 - t2;
    pthread_mutex_unlock(&s->mu);
    pthread_mutex_unlock(&c->mu);

    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int32_t i = n_committed - 1; i >= 0; i--) {
        list = enif_make_list_cell(env, enif_make_int(env, committed[i]), list);
    }
    return enif_make_tuple2(env, atom_ok,
                            enif_make_tuple4(env, enif_make_atom(env, "spec"), list,
                                             enif_make_int(env, eog), enif_make_int(env, n_acc)));
}

/* =========================================================================
 * nif_mtp_step_batch(Ctx, Mtp, [{Sampler, Seq, Drafts}], MaxDraft)
 *   -> {ok, [{Seq, [Tok], Eog, NAcc, NDrafted}], #{draft_us, verify_us, rollback_us, rows}}
 *    | {error, _}
 *
 * One round of nif_mtp_step for several sequences at once: sample each
 * sequence's next token K from its own logits row, draft after every K
 * in one drafter call, decode every K and draft in one batch with every
 * row out, then keep or take back each sequence's drafts on its own.
 * Same contract as nif_mtp_step for each sequence.
 *
 * Drafts: `auto' (up to MaxDraft), `{auto, Cap}' (up to Cap, at most
 * MaxDraft), or a list of tokens to verify instead (tests); `[]' or a
 * Cap of 0 decodes K alone, with any sampler. NDrafted is how many
 * guesses were verified for the sequence (the drafter can give fewer
 * than asked, and none after an end-of-generation K). Each item has
 * its own sequence and its own sampler (badarg otherwise). A sequence that drafts
 * needs a greedy sampler (argmax verify, `{not_greedy, Seq}' otherwise);
 * a sampler with logprobs is refused (`{not_supported, Seq}'). Both
 * refusals come before any sampling or decode, so every row is left as
 * it was. The call holds every sequence that needs a
 * token this tick: a sequence it leaves out loses its logits row
 * (stale_logits on its next sample).
 * ========================================================================= */

#define ERLLAMA_MTP_MAX_SEQS 16

ERL_NIF_TERM nif_mtp_step_batch(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    erllama_mtp_res_t *r;
    int max_draft;
    unsigned int n;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c) ||
        !enif_get_resource(env, argv[1], MTP_RT, (void **) &r) || !enif_get_list_length(env, argv[2], &n) ||
        n < 1 || n > ERLLAMA_MTP_MAX_SEQS || !enif_get_int(env, argv[3], &max_draft) || max_draft < 0 ||
        max_draft > ERLLAMA_MTP_MAX_DRAFT) {
        return enif_make_badarg(env);
    }
    erllama_sampler_t *smp[ERLLAMA_MTP_MAX_SEQS];
    int32_t seqs[ERLLAMA_MTP_MAX_SEQS];
    int forced_n[ERLLAMA_MTP_MAX_SEQS];
    int cap[ERLLAMA_MTP_MAX_SEQS];
    llama_token forced[ERLLAMA_MTP_MAX_SEQS][ERLLAMA_MTP_MAX_DRAFT];
    {
        /* Exactly `n' items, each checked before the next; no two with the
         * same sequence or the same sampler (each sampler is locked once
         * below, so a repeated one would lock its mutex twice). */
        ERL_NIF_TERM head, tail = argv[2];
        for (unsigned int i = 0; i < n; i++) {
            int arity;
            const ERL_NIF_TERM *t;
            int seq;
            if (!enif_get_list_cell(env, tail, &head, &tail) || !enif_get_tuple(env, head, &arity, &t) ||
                arity != 3 || !enif_get_resource(env, t[0], SAMPLER_RT, (void **) &smp[i]) ||
                !enif_get_int(env, t[1], &seq) || seq < 0 || seq >= ERLLAMA_N_SEQ_MAX_CAP) {
                return enif_make_badarg(env);
            }
            for (unsigned int j = 0; j < i; j++) {
                if (seqs[j] == seq || smp[j] == smp[i]) {
                    return enif_make_badarg(env);
                }
            }
            seqs[i] = seq;
            forced_n[i] = -1;
            cap[i] = max_draft;
            {
                int ta;
                const ERL_NIF_TERM *tt;
                int cv;
                if (enif_get_tuple(env, t[2], &ta, &tt)) {
                    if (ta != 2 || !enif_is_identical(tt[0], enif_make_atom(env, "auto")) ||
                        !enif_get_int(env, tt[1], &cv) || cv < 0) {
                        return enif_make_badarg(env);
                    }
                    cap[i] = cv < max_draft ? cv : max_draft;
                } else if (!enif_is_list(env, t[2]) && !enif_is_identical(t[2], enif_make_atom(env, "auto"))) {
                    return enif_make_badarg(env);
                }
            }
            if (enif_is_list(env, t[2])) {
                unsigned int fl;
                ERL_NIF_TERM fh, ft = t[2];
                if (!enif_get_list_length(env, t[2], &fl) || fl > (unsigned int) ERLLAMA_MTP_MAX_DRAFT) {
                    return enif_make_badarg(env);
                }
                forced_n[i] = 0;
                for (unsigned int k = 0; k < fl; k++) {
                    int v;
                    if (!enif_get_list_cell(env, ft, &fh, &ft) || !enif_get_int(env, fh, &v) || v < 0) {
                        return enif_make_badarg(env);
                    }
                    forced[i][forced_n[i]++] = v;
                }
            }
        }
    }
    if (!erllama_lock_ctx(c)) {
        return erllama_error(env, atom_released);
    }
    if (c->mtp != r || !r->m) {
        pthread_mutex_unlock(&c->mu);
        return erllama_error(env, atom_released);
    }
    int32_t n_vocab = 0;
    const struct llama_vocab *vocab = erllama_ctx_vocab(c, &n_vocab);
    uint32_t n_batch_cap = erllama_safe_n_batch(c->ctx);
    if (!vocab || n_vocab <= 0 || n_batch_cap < n * ((uint32_t) max_draft + 1)) {
        pthread_mutex_unlock(&c->mu);
        return erllama_error(env, atom_batch_overflow);
    }
    unsigned int locked = 0;
    ERL_NIF_TERM err = 0;
    for (; locked < n; locked++) {
        pthread_mutex_lock(&smp[locked]->mu);
        const int drafts = forced_n[locked] >= 0 ? forced_n[locked] > 0 : cap[locked] > 0;
        if (!smp[locked]->chain || smp[locked]->ctx_res != c) {
            err = atom_released;
            locked++;
            break;
        }
        /* A refusal names its sequence: the caller turns drafting off
         * there and keeps it for the others. */
        if (smp[locked]->n_probs > 0) {
            err = enif_make_tuple2(env, atom_not_supported, enif_make_int(env, seqs[locked]));
            locked++;
            break;
        }
        if (drafts && !erllama_mtp_chain_is_greedy(smp[locked]->chain)) {
            err = enif_make_tuple2(env, enif_make_atom(env, "not_greedy"), enif_make_int(env, seqs[locked]));
            locked++;
            break;
        }
        /* Each sequence's own logits row, as nif_step checks: removing
         * another sequence (a cancel) leaves this row in place; a
         * decode that left this sequence out does not. */
        int32_t li = c->per_seq[seqs[locked]].last_logits_idx;
        if (li == -1 || li == ERLLAMA_LOGITS_STALE) {
            err = li == -1 ? atom_no_logits : atom_stale_logits;
            locked++;
            break;
        }
    }
#define UNLOCK_ALL()                                    \
    do {                                                \
        for (unsigned int u = 0; u < locked; u++) {     \
            pthread_mutex_unlock(&smp[u]->mu);          \
        }                                               \
        pthread_mutex_unlock(&c->mu);                   \
    } while (0)
    if (err) {
        UNLOCK_ALL();
        return erllama_error(env, err);
    }

    int32_t pos[ERLLAMA_MTP_MAX_SEQS];
    llama_token k[ERLLAMA_MTP_MAX_SEQS];
    int eog_k[ERLLAMA_MTP_MAX_SEQS];
    for (unsigned int i = 0; i < n; i++) {
        int32_t li = c->per_seq[seqs[i]].last_logits_idx;
        pos[i] = c->per_seq[seqs[i]].next_pos;
        k[i] = erllama_safe_sampler_sample(smp[i]->chain, c->ctx, li == ERLLAMA_LOGITS_LAST ? -1 : li);
        if (k[i] < 0) {
            for (unsigned int j = 0; j < n; j++) {
                c->per_seq[seqs[j]].last_logits_idx = -1;
            }
            c->decode_ready = 0;
            UNLOCK_ALL();
            return erllama_error(env, atom_exception);
        }
        eog_k[i] = erllama_safe_vocab_is_eog(vocab, k[i]);
    }

    /* Drafts: one drafter call for every sequence that drafts. */
    llama_token draft[ERLLAMA_MTP_MAX_SEQS * ERLLAMA_MTP_MAX_DRAFT];
    int32_t n_draft[ERLLAMA_MTP_MAX_SEQS];
    int32_t dseqs[ERLLAMA_MTP_MAX_SEQS], dpos[ERLLAMA_MTP_MAX_SEQS], didx[ERLLAMA_MTP_MAX_SEQS];
    llama_token dlast[ERLLAMA_MTP_MAX_SEQS];
    int32_t nd = 0;
    const uint64_t t0 = mono_us();
    for (unsigned int i = 0; i < n; i++) {
        n_draft[i] = 0;
        if (eog_k[i] || max_draft == 0) {
            continue;
        }
        if (forced_n[i] >= 0) {
            n_draft[i] = forced_n[i] < max_draft ? forced_n[i] : max_draft;
        } else if (cap[i] == 0) {
            continue;
        }
        if (forced_n[i] >= 0) {
            for (int32_t j = 0; j < n_draft[i]; j++) {
                draft[(size_t) i * ERLLAMA_MTP_MAX_DRAFT + (size_t) j] = forced[i][j];
            }
        } else {
            dseqs[nd] = seqs[i];
            dlast[nd] = k[i];
            dpos[nd] = pos[i];
            didx[nd] = (int32_t) i;
            nd++;
        }
    }
    if (nd > 0) {
        llama_token dout[ERLLAMA_MTP_MAX_SEQS * ERLLAMA_MTP_MAX_DRAFT];
        int32_t dn[ERLLAMA_MTP_MAX_SEQS];
        erllama_mtp_draft_many(r->m, nd, dseqs, dlast, dpos, max_draft, dout, dn);
        for (int32_t d = 0; d < nd; d++) {
            const int32_t i = didx[d];
            n_draft[i] = dn[d] < cap[i] ? dn[d] : cap[i];
            for (int32_t j = 0; j < n_draft[i]; j++) {
                draft[(size_t) i * ERLLAMA_MTP_MAX_DRAFT + (size_t) j] = dout[(size_t) d * (size_t) max_draft + (size_t) j];
            }
        }
    }
    for (unsigned int i = 0; i < n; i++) {
        for (int32_t j = 0; j < n_draft[i]; j++) {
            llama_token t = draft[(size_t) i * ERLLAMA_MTP_MAX_DRAFT + (size_t) j];
            if (t < 0 || t >= n_vocab) {
                n_draft[i] = j;
                break;
            }
        }
    }
    const uint64_t t1 = mono_us();

    int32_t rows = 0, row0[ERLLAMA_MTP_MAX_SEQS];
    for (unsigned int i = 0; i < n; i++) {
        row0[i] = rows;
        rows += 1 + n_draft[i];
    }
    struct llama_batch batch = erllama_safe_batch_init(rows, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        erllama_safe_batch_free(batch);
        UNLOCK_ALL();
        return erllama_error(env, atom_oom);
    }
    for (unsigned int i = 0; i < n; i++) {
        for (int32_t j = 0; j <= n_draft[i]; j++) {
            const int32_t b = row0[i] + j;
            batch.token[b] = j == 0 ? k[i] : draft[(size_t) i * ERLLAMA_MTP_MAX_DRAFT + (size_t) (j - 1)];
            batch.pos[b] = pos[i] + j;
            batch.n_seq_id[b] = 1;
            batch.seq_id[b][0] = (llama_seq_id) seqs[i];
            batch.logits[b] = 1;
        }
    }
    batch.n_tokens = rows;
    arm_decode(c);
    int dr = erllama_ctx_decode(env, c, batch);
    erllama_safe_batch_free(batch);
    if (dr != 0) {
        for (unsigned int j = 0; j < n; j++) {
            c->per_seq[seqs[j]].last_logits_idx = -1;
        }
        c->decode_ready = 0;
        UNLOCK_ALL();
        return erllama_error(env, dr == ERLLAMA_DECODE_EXC_SENTINEL ? atom_exception : classify_decode_error(env, c, dr));
    }
    const uint64_t t2 = mono_us();
    const float *logits = erllama_safe_get_logits(c->ctx);
    if (!logits) {
        c->decode_ready = 0;
        UNLOCK_ALL();
        return erllama_error(env, atom_exception);
    }

    ERL_NIF_TERM out = enif_make_list(env, 0);
    int64_t drafted = 0, accepted = 0;
    for (int ii = (int) n - 1; ii >= 0; ii--) {
        const unsigned int i = (unsigned int) ii;
        llama_token committed[ERLLAMA_MTP_MAX_DRAFT + 1];
        int32_t n_committed = 1, n_acc = 0;
        int eog = eog_k[i];
        committed[0] = k[i];
        for (int32_t j = 0; j < n_draft[i] && !eog; j++) {
            const llama_token g = draft[(size_t) i * ERLLAMA_MTP_MAX_DRAFT + (size_t) j];
            if (argmax_row(logits + (size_t) (row0[i] + j) * (size_t) n_vocab, n_vocab) != g) {
                break;
            }
            committed[n_committed++] = g;
            n_acc++;
            eog = erllama_safe_vocab_is_eog(vocab, g);
        }
        const int32_t keep = pos[i] + 1 + n_acc;
        if (r->trace) {
            llama_token toks[ERLLAMA_MTP_MAX_DRAFT + 1];
            toks[0] = k[i];
            for (int32_t j = 0; j < n_draft[i]; j++) {
                toks[j + 1] = draft[(size_t) i * ERLLAMA_MTP_MAX_DRAFT + (size_t) j];
            }
            trace_round(r, pos[i], toks, n_draft[i] + 1, n_acc, logits + (size_t) row0[i] * (size_t) n_vocab, n_vocab);
        }
        if (n_acc < n_draft[i]) {
            if (erllama_ctx_seq_rm(env, c, seqs[i], keep, -1) != 0) {
                for (unsigned int j = 0; j < n; j++) {
                    c->per_seq[seqs[j]].last_logits_idx = -1;
                }
                c->decode_ready = 0;
                UNLOCK_ALL();
                return erllama_error(env, atom_decode_failed);
            }
            erllama_mtp_seq_rm(r->m, seqs[i], keep, -1);
        }
        erllama_mtp_accept(r->m, seqs[i], n_acc);
        if (erllama_mtp_seq_pos_max(r->m, seqs[i]) != keep - 1) {
            r->misaligned++;
        }
        c->per_seq[seqs[i]].next_pos = keep;
        c->per_seq[seqs[i]].last_logits_idx = row0[i] + n_acc;
        drafted += n_draft[i];
        accepted += n_acc;
        ERL_NIF_TERM list = enif_make_list(env, 0);
        for (int32_t j = n_committed - 1; j >= 0; j--) {
            list = enif_make_list_cell(env, enif_make_int(env, committed[j]), list);
        }
        out = enif_make_list_cell(
            env,
            enif_make_tuple5(env, enif_make_int(env, seqs[i]), list, enif_make_int(env, eog), enif_make_int(env, n_acc),
                             enif_make_int(env, n_draft[i])),
            out);
    }
    const uint64_t t3 = mono_us();
    c->decode_ready = 1;
    r->rounds++;
    r->drafted += (uint64_t) drafted;
    r->accepted += (uint64_t) accepted;
    r->draft_us += t1 - t0;
    r->verify_us += t2 - t1;
    r->rollback_us += t3 - t2;
    UNLOCK_ALL();
#undef UNLOCK_ALL
    ERL_NIF_TERM times = enif_make_new_map(env);
    times = put_u64(env, times, "draft_us", t1 - t0);
    times = put_u64(env, times, "verify_us", t2 - t1);
    times = put_u64(env, times, "rollback_us", t3 - t2);
    times = put_u64(env, times, "rows", (uint64_t) rows);
    return enif_make_tuple3(env, atom_ok, out, times);
}
