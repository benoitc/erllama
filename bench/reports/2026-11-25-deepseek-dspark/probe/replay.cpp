#include "common.h"
#include "ggml-backend.h"
#include "sampling.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

static void inject(llama_context *d, const std::string &file, int n, int pos) {
  std::vector<float> raw(n * 4096 * 3), ordered(raw.size());
  std::ifstream f(file, std::ios::binary);
  if (!f.read(reinterpret_cast<char *>(raw.data()), raw.size() * sizeof(float)))
    throw std::runtime_error(file);
  common_batch b(d);
  for (int i = 0; i < n; i++) {
    for (int l = 0; l < 3; l++)
      std::copy_n(raw.data() + (l * n + i) * 4096, 4096,
                  ordered.data() + (i * 3 + l) * 4096);
    llama_pos p[] = {pos + i, pos + i, pos + i, 0};
    b.add_embd({ordered.data() + i * 12288, 1, 12288}, p, 0, false);
  }
  if (llama_process(d, LLAMA_PROCESS_TYPE_DECODE, b.get()))
    throw std::runtime_error("feature injection");
}
static int guess(llama_context *d, common_sampler *sampler, int pos,
                 const std::vector<int> &ids) {
  auto *v = llama_model_get_vocab(llama_get_model(d));
  int nv = llama_vocab_n_tokens(v);
  common_batch b(d);
  for (int i = 0; i < 4; i++)
    b.add(i ? llama_vocab_mask(v) : ids.at(pos), pos + i, 0, true);
  const auto rc = llama_process(d, LLAMA_PROCESS_TYPE_DECODE, b.get());
  if (rc) {
    std::cout << " decode_rc " << rc << std::endl;
    return rc;
  }
  for (int i = 0; i < 2; i++) {
    common_sampler_sample(sampler, d, i, true);
    const auto *rows = common_sampler_get_candidates(sampler, true);
    const auto id = rows->data[0].id;
    common_sampler_accept(sampler, id, true);
    std::cout << ' ' << id;
  }
  std::cout << " reference " << ids.at(pos + 1) << ' ' << ids.at(pos + 2)
            << '\n'
            << std::flush;
  return 0;
}
int main() {
  llama_backend_init();
  ggml_backend_load_all();
  auto mp = llama_model_default_params();
  ggml_backend_dev_t cpu_only[] = {nullptr};
  mp.devices = cpu_only;
  mp.n_gpu_layers = 0;
  mp.use_extra_bufts = false;
  auto *m = llama_model_load_from_file("standalone.gguf", mp);
  if (!m)
    return 1;
  std::ifstream f("fixed.ids");
  std::vector<int> ids;
  int id;
  while (f >> id)
    ids.push_back(id);
  auto cp = llama_context_default_params();
  cp.n_ctx = 512;
  cp.n_batch = 128;
  cp.n_ubatch = 128;
  cp.n_seq_max = 1;
  cp.n_threads = 6;
  cp.n_threads_batch = 6;
  cp.offload_kqv = false;
  for (std::string leg : {"one-output-control", "reference", "split"}) {
    const bool control = leg == "one-output-control";
    std::string src = control ? "reference" : leg;
    cp.n_outputs_max = 5;
    cp.n_outputs_max_per_seq = control ? 1 : 5;
    auto *d = llama_init_from_model(m, cp);
    if (!d)
      return 2;
    auto *chain =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));
    bool attached = llama_set_sampler(d, 0, chain);
    std::cout << "backend_sampler " << attached << " " << leg << std::endl;
    common_params_sampling params;
    params.top_k = 10;
    params.samplers = {COMMON_SAMPLER_TYPE_TOP_K};
    common_sampler_ptr smpl(common_sampler_init(m, params));
    inject(d, src + "-pre.f32", 70, 0);
    std::cout << leg << " 70";
    int rc = guess(d, smpl.get(), 70, ids);
    if (!control) {
      if (rc)
        throw std::runtime_error("draft noise batch");
      llama_memory_seq_rm(llama_get_memory(d), 0, 70, -1);
      inject(d, src + "-one.f32", 1, 70);
      common_sampler_reset(smpl.get());
      std::cout << leg << " 71";
      if (guess(d, smpl.get(), 71, ids))
        throw std::runtime_error("second noise batch");
    }
    if (attached)
      llama_set_sampler(d, 0, nullptr);
    llama_sampler_free(chain);
    llama_free(d);
  }
  llama_model_free(m);
  llama_backend_free();
}
