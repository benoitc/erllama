#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const std::string &msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
static void gpu(cudaError_t err) { check(err == cudaSuccess, cudaGetErrorString(err)); }

__global__ void read_cache(const uint32_t *src, size_t n, uint32_t *dst) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t sum = 0;
    for (size_t j = i; j < n; j += gridDim.x * blockDim.x) {
        uint32_t word;
        asm volatile("ld.global.cg.u32 %0,[%1];" : "=r"(word) : "l"(src + j));
        sum += word;
    }
    dst[i] = sum;
}

static std::vector<int32_t> read_ids(const char *path, size_t n) {
    std::vector<int32_t> data(n);
    std::ifstream file(path, std::ios::binary);
    check(bool(file.read(reinterpret_cast<char *>(data.data()), n * 4)), path);
    return data;
}

struct Fixture {
    ggml_context *ctx;
    ggml_backend_buffer_t buf;
    ggml_cgraph *graph;
    ggml_tensor *activation, *ids, *result;

    Fixture(ggml_backend_t backend, ggml_tensor *up, ggml_tensor *gate, int batch) {
        ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        graph = ggml_new_graph_custom(ctx, 64, false);
        activation = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4096, 1, batch);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6, batch);
        auto u = ggml_mul_mat_id(ctx, up, activation, ids);
        auto g = ggml_mul_mat_id(ctx, gate, activation, ids);
        result = ggml_swiglu_clamp(ctx, g, u, 10.0f);
        for (auto t : {u, g, result})
            check(ggml_backend_supports_op(backend, t), "unsupported CUDA op");
        ggml_build_forward_expand(graph, result);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        check(buf, "graph allocation");
        std::mt19937 rng(42);
        std::vector<float> a(4096 * batch);
        for (auto &x : a)
            x = float(int(rng() % 65536) - 32768) / 32768.f;
        ggml_backend_tensor_set(activation, a.data(), 0, a.size() * sizeof(float));
    }
    ~Fixture() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

struct Weights {
    ggml_context *ctx;
    ggml_backend_buffer_t buf;
    ggml_tensor *up, *gate;
    Weights(ggml_backend_t backend, const std::string &name, ggml_type type) {
        ctx = ggml_init({16384, nullptr, true});
        up = ggml_new_tensor_3d(ctx, type, 4096, 1024, 256);
        gate = ggml_new_tensor_3d(ctx, type, 4096, 1024, 256);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        check(buf, "weight allocation");
        for (auto t : {up, gate}) {
            std::string path = name + (t == up ? "-up.bin" : "-gate.bin");
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            check(file && size_t(file.tellg()) == ggml_nbytes(t), path + " size");
            file.seekg(0);
            std::vector<char> chunk(4 * 1024 * 1024);
            for (size_t offset = 0; offset < ggml_nbytes(t); offset += chunk.size()) {
                size_t bytes = std::min(chunk.size(), ggml_nbytes(t) - offset);
                check(bool(file.read(chunk.data(), bytes)), path + " read");
                ggml_backend_tensor_set(t, chunk.data(), offset, bytes);
            }
        }
    }
    ~Weights() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

static void bench(ggml_backend_t backend, int trace_quant, int trace_batch) {
    auto tokens = read_ids("tokens.bin", 8192);
    auto route_table = read_ids("routes.bin", 6 * 129280);
    Weights iq(backend, "iq2", GGML_TYPE_IQ2_XXS), q4(backend, "q4", GGML_TYPE_Q4_K);
    cudaDeviceProp prop{};
    gpu(cudaGetDeviceProperties(&prop, 0));
    const size_t bytes = std::max(size_t(64) << 20, 4 * size_t(prop.l2CacheSize));
    uint32_t *src, *dst;
    gpu(cudaMalloc(&src, bytes));
    gpu(cudaMalloc(&dst, 256 * 256 * 4));
    gpu(cudaMemset(src, 1, bytes));
    fprintf(stderr, "%s, L2 %d bytes, eviction read %zu bytes\n", prop.name, prop.l2CacheSize,
            bytes);
    puts("batch,pair,format,order,token_offset,distinct_experts,elapsed_us");
    for (int batch : {1, 512}) {
        if (trace_quant >= 0 && batch != trace_batch)
            continue;
        Fixture a(backend, iq.up, iq.gate, batch), b(backend, q4.up, q4.gate, batch);
        const int trials = (trace_quant >= 0 ? 1 : batch == 1 ? 80 : 24);
        for (int rep = -4; rep < trials; ++rep) {
            const int off = (257 * (rep + 4) + 1024) % (int(tokens.size()) - batch + 1);
            std::vector<int32_t> routes;
            std::set<int> distinct;
            for (int i = 0; i < batch; ++i) {
                int id = tokens.at(off + i);
                check(id >= 0 && id < 129280, "token id");
                std::set<int> active;
                for (int k = 0; k < 6; ++k) {
                    int expert = route_table.at(6 * id + k);
                    check(expert >= 0 && expert < 256 && active.insert(expert).second, "expert id");
                    distinct.insert(expert);
                    routes.push_back(expert);
                }
            }
            for (int i = 0; i < 2; ++i) {
                int q = (i + (rep + 4) % 2) % 2;
                if (trace_quant >= 0 && q != trace_quant)
                    continue;
                auto &f = q == 0 ? a : b;
                ggml_backend_tensor_set(f.ids, routes.data(), 0, routes.size() * 4);
                gpu(cudaDeviceSynchronize());
                read_cache<<<256, 256>>>(src, bytes / 4, dst);
                gpu(cudaDeviceSynchronize());
                if (trace_quant >= 0 && rep == 0)
                    gpu(cudaProfilerStart());
                auto start = std::chrono::steady_clock::now();
                check(ggml_backend_graph_compute(backend, f.graph) == GGML_STATUS_SUCCESS,
                      "compute");
                ggml_backend_synchronize(backend);
                auto end = std::chrono::steady_clock::now();
                if (trace_quant >= 0 && rep == 0)
                    gpu(cudaProfilerStop());
                if (rep < 0) {
                    std::vector<float> test(1024);
                    ggml_backend_tensor_get(f.result, test.data(), 0, test.size() * 4);
                    check(std::all_of(test.begin(), test.end(),
                                      [](float x) { return std::isfinite(x); }) &&
                              std::any_of(test.begin(), test.end(), [](float x) { return x != 0; }),
                          "invalid output");
                } else {
                    double us = std::chrono::duration<double, std::micro>(end - start).count();
                    printf("%d,%d,%s,%d,%d,%zu,%.3f\n", batch, rep, q ? "Q4_K" : "IQ2_XXS", i + 1,
                           off, distinct.size(), us);
                }
            }
        }
    }
    gpu(cudaFree(src));
    gpu(cudaFree(dst));
}

int main(int argc, char **argv) {
    try {
        check(argc == 1 || argc == 3, "expected [quant-index batch]");
        auto backend = ggml_backend_cuda_init(0);
        check(backend, "CUDA init");
        bench(backend, argc == 3 ? std::stoi(argv[1]) : -1, argc == 3 ? std::stoi(argv[2]) : 0);
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
