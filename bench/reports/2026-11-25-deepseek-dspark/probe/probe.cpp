#include "common.h"
#include "erl_nif.h"
#include "speculative.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

extern "C" void *dspark_lock(ErlNifEnv *, ERL_NIF_TERM, llama_context **);
extern "C" void dspark_unlock(void *);
using term = ERL_NIF_TERM;
struct Target {
  llama_context *ctx = nullptr;
  void *r;
  Target(ErlNifEnv *e, term t) : r(dspark_lock(e, t, &ctx)) {
    if (!r)
      throw std::runtime_error("target_closed");
  }
  ~Target() { dspark_unlock(r); }
};
static constexpr std::array<unsigned, 3> layers{41, 42, 43};
static ErlNifResourceType *head_rt;
struct Head {
  void *target_ref = nullptr;
  llama_model *m = nullptr;
  llama_context *d = nullptr;
  common_speculative *s = nullptr;
  ~Head() {
    if (s)
      common_speculative_free(s);
    if (d)
      llama_free(d);
    if (m)
      llama_model_free(m);
    if (target_ref)
      enif_release_resource(target_ref);
  }
};
static term atom(ErlNifEnv *e, const char *s) { return enif_make_atom(e, s); }
static int integer(ErlNifEnv *e, term v) {
  int n;
  if (!enif_get_int(e, v, &n) || n < 0)
    throw std::invalid_argument("integer");
  return n;
}
static Head *get(ErlNifEnv *e, term t, Target &target) {
  Head *h;
  if (!enif_get_resource(e, t, head_rt, (void **)&h) ||
      h->target_ref != target.r)
    throw std::invalid_argument("head");
  return h;
}
static term capture(ErlNifEnv *e, int, const term a[]) {
  Target t(e, a[0]);
  for (auto i : layers)
    llama_set_embeddings_layer_inp(t.ctx, i, true);
  llama_set_embeddings_layer_inp_dev(t.ctx, false);
  return atom(e, "ok");
}
static term rows(ErlNifEnv *e, int, const term a[]) {
  Target t(e, a[0]);
  int n = integer(e, a[1]);
  if (n < 1 || n > (int)llama_n_batch(t.ctx))
    throw std::invalid_argument("rows");
  const size_t bytes =
      size_t(n) * sizeof(float) * llama_model_n_embd(llama_get_model(t.ctx));
  term bin;
  auto *p = enif_make_new_binary(e, bytes * layers.size(), &bin);
  for (auto i : layers) {
    float *v = llama_get_embeddings_layer_inp(t.ctx, i);
    if (!v)
      throw std::runtime_error("missing_features");
    memcpy(p, v, bytes);
    p += bytes;
  }
  return bin;
}
static term init(ErlNifEnv *e, int, const term a[]) {
  Target t(e, a[0]);
  char path[4096];
  if (enif_get_string(e, a[1], path, sizeof(path), ERL_NIF_LATIN1) <= 0)
    throw std::invalid_argument("path");
  auto *h = new (enif_alloc_resource(head_rt, sizeof(Head))) Head;
  try {
    enif_keep_resource(t.r);
    h->target_ref = t.r;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
    h->m = llama_model_load_from_file(path, mp);
    if (!h->m)
      throw std::runtime_error("head_load_failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = llama_n_ctx(t.ctx);
    cp.n_seq_max = 1;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_outputs_max = 5;
    cp.n_outputs_max_per_seq = 5;
    cp.n_rs_seq = 0;
    cp.kv_unified = true;
    cp.ctx_other = t.ctx;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    h->d = llama_init_from_model(h->m, cp);
    if (!h->d)
      throw std::runtime_error("draft_context_failed");
    common_params_speculative par;
    par.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK};
    par.draft.ctx_tgt = t.ctx;
    par.draft.ctx_dft = h->d;
    par.draft.n_max = 4;
    par.draft.n_min = 0;
    par.draft.p_min = 0;
    h->s = common_speculative_init(par, 1);
    if (!h->s)
      throw std::runtime_error("draft_init_failed");
    term v = enif_make_resource(e, h);
    enif_release_resource(h);
    return v;
  } catch (...) {
    enif_release_resource(h);
    throw;
  }
}
static term feed(ErlNifEnv *e, int, const term a[]) {
  Target t(e, a[0]);
  auto *h = get(e, a[1], t);
  int pos = integer(e, a[2]);
  term list = a[3], v;
  common_batch b(h->d);
  int next = pos;
  while (enif_get_list_cell(e, list, &v, &list))
    b.add(integer(e, v), next++, 0, true);
  if (!b.size() || !enif_is_empty_list(e, list))
    throw std::invalid_argument("batch");
  auto mem = llama_get_memory(h->d);
  if (llama_memory_seq_pos_max(mem, 0) >= pos &&
      !llama_memory_seq_rm(mem, 0, pos, -1))
    throw std::runtime_error("draft_trim_failed");
  if (!common_speculative_process(h->s, b))
    throw std::runtime_error("draft_feed_failed");
  return enif_make_int(e, llama_memory_seq_pos_max(mem, 0));
}
static term draft(ErlNifEnv *e, int, const term a[]) {
  Target t(e, a[0]);
  auto *h = get(e, a[1], t);
  int cap = integer(e, a[4]);
  if (cap < 1 || cap > 4)
    throw std::invalid_argument("cap");
  llama_tokens prompt, result;
  auto &dp = common_speculative_get_draft_params(h->s, 0);
  dp.drafting = true;
  dp.n_max = cap;
  dp.id_last = integer(e, a[3]);
  dp.pos0 = integer(e, a[2]);
  dp.prompt = &prompt;
  dp.result = &result;
  common_speculative_draft(h->s);
  term out = enif_make_list(e, 0);
  for (int i = std::min<int>(cap, result.size()) - 1; i >= 0; --i)
    out = enif_make_list_cell(e, enif_make_int(e, result[i]), out);
  dp.prompt = nullptr;
  dp.result = nullptr;
  return out;
}
static term top(ErlNifEnv *e, int, const term a[]) {
  ErlNifBinary b;
  if (!enif_inspect_binary(e, a[0], &b))
    throw std::invalid_argument("logits");
  size_t n = integer(e, a[1]);
  if (n == 0 || !b.size || b.size % (n * sizeof(float)))
    throw std::invalid_argument("shape");
  auto *p = (const float *)b.data;
  term out = enif_make_list(e, 0);
  for (size_t i = b.size / sizeof(float); i; i -= n)
    out = enif_make_list_cell(
        e, enif_make_int(e, std::max_element(p + i - n, p + i) - (p + i - n)),
        out);
  return out;
}
extern "C" void *dspark_rank(void *);
extern "C" int erllama_tp_reduce_host(void *, float *, size_t);
static term plain(ErlNifEnv *e, int, const term a[]) {
  Target t(e, a[0]);
  for (auto i : layers)
    llama_set_embeddings_layer_inp(t.ctx, i, false);
  return atom(e, "ok");
}
static term checkpoint(ErlNifEnv *e, int, const term a[]) {
  Target t(e, a[0]);
  auto *h = get(e, a[1], t);
  size_t n = llama_state_seq_get_size(h->d, 0);
  term bin;
  if (!n)
    throw std::runtime_error("empty_draft_checkpoint");
  uint8_t *dst = enif_make_new_binary(e, n, &bin);
  if (llama_state_seq_get_data(h->d, dst, n, 0) != n)
    throw std::runtime_error("draft_save_failed");
  return bin;
}
static term restore(ErlNifEnv *e, int, const term a[]) {
  Target t(e, a[0]);
  auto *h = get(e, a[1], t);
  ErlNifBinary b;
  if (!enif_inspect_binary(e, a[2], &b))
    throw std::invalid_argument("checkpoint");
  if (!llama_memory_seq_rm(llama_get_memory(h->d), 0, -1, -1) ||
      llama_state_seq_set_data(h->d, b.data, b.size, 0) != b.size)
    throw std::runtime_error("draft_restore_failed");
  return atom(e, "ok");
}
// One known token per sequence. Sequence zero occupies the first feature row.
static term many(ErlNifEnv *e, int, const term a[]) {
  Target t(e, a[0]);
  term list = a[1], v;
  std::vector<int> ids;
  while (enif_get_list_cell(e, list, &v, &list))
    ids.push_back(integer(e, v));
  if (ids.empty() || ids.size() > llama_n_seq_max(t.ctx))
    throw std::invalid_argument("sequences");
  struct Buf {
    llama_batch b;
    ~Buf() { llama_batch_free(b); }
  } mem{llama_batch_init(ids.size(), 0, 1)};
  auto &b = mem.b;
  b.n_tokens = ids.size();
  for (int j = 0; j < b.n_tokens; j++) {
    b.token[j] = ids[j];
    b.pos[j] = llama_memory_seq_pos_max(llama_get_memory(t.ctx), j) + 1;
    b.n_seq_id[j] = 1;
    b.seq_id[j][0] = j;
    b.logits[j] = 1;
  }
  if (llama_decode(t.ctx, b))
    throw std::runtime_error("ordinary_batch_failed");
  float *p = llama_get_logits(t.ctx);
  size_t nv =
      llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(t.ctx)));
  auto rank = dspark_rank(t.r);
  if (rank && erllama_tp_reduce_host(rank, p, nv * ids.size()))
    throw std::runtime_error("logits_sum_failed");
  term out = enif_make_list(e, 0);
  for (int j = b.n_tokens - 1; j >= 0; j--) {
    float *row = p + j * nv;
    out = enif_make_list_cell(
        e, enif_make_int(e, std::max_element(row, row + nv) - row), out);
  }
  return out;
}

template <term (*F)(ErlNifEnv *, int, const term *)>
static term call(ErlNifEnv *e, int n, const term a[]) {
  try {
    return F(e, n, a);
  } catch (const std::exception &ex) {
    return enif_make_tuple2(e, atom(e, "error"),
                            enif_make_string(e, ex.what(), ERL_NIF_LATIN1));
  }
}
static int load(ErlNifEnv *e, void **, term) {
  head_rt = enif_open_resource_type(
      e, nullptr, "dspark_probe_head",
      [](ErlNifEnv *, void *p) { static_cast<Head *>(p)->~Head(); },
      ERL_NIF_RT_CREATE, nullptr);
  return head_rt ? 0 : -1;
}
static ErlNifFunc fs[] = {
    {"plain", 1, call<plain>, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"checkpoint", 2, call<checkpoint>, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"restore", 3, call<restore>, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"many", 2, call<many>, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"capture", 1, call<capture>, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"rows", 2, call<rows>, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"head", 2, call<init>, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"feed", 4, call<feed>, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"draft", 5, call<draft>, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"top", 2, call<top>, ERL_NIF_DIRTY_JOB_CPU_BOUND}};
ERL_NIF_INIT(dsprobe, fs, load, nullptr, nullptr, nullptr)
