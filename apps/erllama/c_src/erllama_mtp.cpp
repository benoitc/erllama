/* Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
 * See the LICENSE file at the project root. */

/* llama.cpp's MTP drafter behind a C surface (erllama_mtp.h). The
 * drafter owns a context of the draft model (ctx_type MTP) and pairs
 * every token the target decodes with the target's hidden state one
 * position back, so its context follows the target's. */

#include "erllama_mtp.h"

#include "common.h"
#include "speculative.h"

#include <chrono>
#include <cstring>
#include <vector>

struct erllama_mtp {
    common_speculative *spec = nullptr;
    llama_context *dft = nullptr;
    llama_context *tgt = nullptr;
    uint32_t n_seq = 0;
    int32_t n_max = 0;
    common_batch batch;
    std::vector<llama_tokens> prompts;
    std::vector<llama_tokens> results;
    std::vector<int32_t> first_pos;
    erllama_mtp_counters_t counters = {};
};

namespace {

uint64_t now_us() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} /* namespace */

extern "C" struct erllama_mtp *erllama_mtp_new(struct llama_model *draft, struct llama_context *tgt,
                                               int32_t n_max, const char **why) {
    *why = "exception";
    erllama_mtp *m = nullptr;
    try {
        if (llama_model_n_layer_nextn(draft) <= 0) {
            *why = "no_mtp_layers";
            return nullptr;
        }
        llama_context_params cp = llama_context_default_params();
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.n_ctx = llama_n_ctx(tgt);
        cp.n_batch = llama_n_batch(tgt);
        cp.n_ubatch = llama_n_ubatch(tgt);
        cp.n_seq_max = llama_n_seq_max(tgt);
        cp.n_rs_seq = 0;
        cp.kv_unified = true;
        cp.no_perf = true;
        cp.ctx_other = tgt;
        llama_context *dft = llama_init_from_model(draft, cp);
        if (!dft) {
            *why = "mtp_context_failed";
            return nullptr;
        }
        m = new erllama_mtp();
        m->dft = dft;
        m->tgt = tgt;
        m->n_seq = cp.n_seq_max;
        m->n_max = n_max;
        m->batch = common_batch(dft);
        m->prompts.resize(m->n_seq);
        m->results.resize(m->n_seq);
        m->first_pos.assign(m->n_seq, -1);

        common_params_speculative params;
        params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        params.draft.ctx_tgt = tgt;
        params.draft.ctx_dft = dft;
        params.draft.n_max = n_max;
        params.draft.n_min = 0;
        /* Every round drafts n_max tokens, so the verify batch keeps one
         * shape and its graph is reused. */
        params.draft.p_min = 0.0f;
        m->spec = common_speculative_init(params, m->n_seq);
        if (!m->spec) {
            *why = "spec_init_failed";
            llama_free(dft);
            delete m;
            return nullptr;
        }
        return m;
    } catch (...) {
        if (m) {
            if (m->spec) {
                common_speculative_free(m->spec);
            }
            if (m->dft) {
                llama_free(m->dft);
            }
            delete m;
        }
        return nullptr;
    }
}

extern "C" void erllama_mtp_free(struct erllama_mtp *m) {
    if (!m) {
        return;
    }
    try {
        common_speculative_free(m->spec);
        llama_free(m->dft);
    } catch (...) {
    }
    delete m;
}

extern "C" void erllama_mtp_target_nextn(struct llama_context *tgt, int on) {
    try {
        llama_set_embeddings_nextn(tgt, on != 0, false);
    } catch (...) {
    }
}

extern "C" void erllama_mtp_process(struct erllama_mtp *m, const struct llama_batch *b) {
    if (!m || !b || b->n_tokens <= 0 || !b->token) {
        return;
    }
    const uint64_t t0 = now_us();
    try {
        std::fill(m->first_pos.begin(), m->first_pos.end(), -1);
        m->batch.clear();
        for (int32_t i = 0; i < b->n_tokens; i++) {
            const llama_seq_id s = b->seq_id[i][0];
            if (s < 0 || (uint32_t) s >= m->n_seq) {
                m->counters.process_failures++;
                return;
            }
            if (m->first_pos[s] < 0) {
                m->first_pos[s] = b->pos[i];
            }
            m->batch.add(b->token[i], b->pos[i], s, b->logits[i] != 0);
        }
        llama_memory_t mem = llama_get_memory(m->dft);
        for (uint32_t s = 0; s < m->n_seq; s++) {
            if (m->first_pos[s] >= 0) {
                (void) llama_memory_seq_rm(mem, (llama_seq_id) s, m->first_pos[s], -1);
            }
        }
        if (!common_speculative_process(m->spec, m->batch)) {
            m->counters.process_failures++;
        }
    } catch (...) {
        m->counters.process_failures++;
    }
    m->counters.process_calls++;
    m->counters.process_us += now_us() - t0;
}

extern "C" int32_t erllama_mtp_draft(struct erllama_mtp *m, int32_t seq, llama_token id_last,
                                     int32_t pos0, int32_t n_max, llama_token *out) {
    if (!m || seq < 0 || (uint32_t) seq >= m->n_seq || n_max <= 0) {
        return 0;
    }
    const uint64_t t0 = now_us();
    int32_t n = 0;
    try {
        auto &dp = common_speculative_get_draft_params(m->spec, seq);
        m->results[seq].clear();
        dp.drafting = true;
        dp.n_max = n_max < m->n_max ? n_max : m->n_max;
        dp.pos0 = pos0;
        dp.id_last = id_last;
        dp.prompt = &m->prompts[seq];
        dp.result = &m->results[seq];
        common_speculative_draft(m->spec);
        const auto &r = m->results[seq];
        for (size_t i = 0; i < r.size() && n < dp.n_max; i++) {
            out[n++] = r[i];
        }
    } catch (...) {
        n = 0;
    }
    m->counters.drafts++;
    m->counters.draft_us += now_us() - t0;
    return n;
}

extern "C" void erllama_mtp_accept(struct erllama_mtp *m, int32_t seq, int32_t n_acc) {
    if (!m || seq < 0 || (uint32_t) seq >= m->n_seq || n_acc < 0) {
        return;
    }
    try {
        common_speculative_accept(m->spec, seq, (uint16_t) n_acc);
    } catch (...) {
    }
}

extern "C" void erllama_mtp_seq_rm(struct erllama_mtp *m, int32_t seq, int32_t p0, int32_t p1) {
    if (!m || seq < 0 || (uint32_t) seq >= m->n_seq) {
        return;
    }
    try {
        (void) llama_memory_seq_rm(llama_get_memory(m->dft), seq, p0, p1);
    } catch (...) {
    }
}

extern "C" int32_t erllama_mtp_seq_pos_max(struct erllama_mtp *m, int32_t seq) {
    if (!m || seq < 0 || (uint32_t) seq >= m->n_seq) {
        return -1;
    }
    try {
        return (int32_t) llama_memory_seq_pos_max(llama_get_memory(m->dft), seq);
    } catch (...) {
        return -1;
    }
}

extern "C" void erllama_mtp_counters(const struct erllama_mtp *m, erllama_mtp_counters_t *out) {
    if (m) {
        *out = m->counters;
    } else {
        std::memset(out, 0, sizeof(*out));
    }
}

extern "C" int erllama_mtp_chain_is_greedy(struct llama_sampler *chain) {
    static const char *keeps_max[] = {"top-k", "top-p", "min-p", "typical", "top-n-sigma"};
    try {
        const int32_t n = llama_sampler_chain_n(chain);
        if (n < 1 || std::strcmp(llama_sampler_name(llama_sampler_chain_get(chain, n - 1)), "greedy") != 0) {
            return 0;
        }
        for (int32_t i = 0; i < n - 1; i++) {
            const char *name = llama_sampler_name(llama_sampler_chain_get(chain, i));
            bool ok = false;
            for (const char *k : keeps_max) {
                ok = ok || std::strcmp(name, k) == 0;
            }
            if (!ok) {
                return 0;
            }
        }
        return 1;
    } catch (...) {
        return 0;
    }
}
