#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
static void gpu(cudaError_t err) { require(err == cudaSuccess, cudaGetErrorString(err)); }

template <class T> static std::vector<T> read_file(const std::string &path, size_t count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    require(f && size_t(f.tellg()) == count * sizeof(T), path);
    f.seekg(0);
    std::vector<T> data(count);
    require(bool(f.read((char *)data.data(), count * sizeof(T))), path);
    return data;
}

struct Graph {
    ggml_context *ctx;
    ggml_backend_buffer_t buffer;
    ggml_cgraph *graph;
    ggml_tensor *output;
    Graph(ggml_backend_t backend, ggml_type type, int batch, float limit) {
        ctx = ggml_init({4 << 20, nullptr, true});
        require(ctx, "ggml context");
        graph = ggml_new_graph_custom(ctx, 64, false);
        std::array<ggml_tensor *, 3> weights;
        for (int i = 0; i < 3; ++i)
            weights[i] = ggml_new_tensor_2d(ctx, type, i == 2 ? 1024 : 4096, i == 2 ? 4096 : 1024);
        auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, batch);
        auto gate = ggml_mul_mat(ctx, weights[0], input);
        auto up = ggml_mul_mat(ctx, weights[1], input);
        auto active =
            limit > 0 ? ggml_swiglu_clamp(ctx, gate, up, limit) : ggml_swiglu_split(ctx, gate, up);
        output = ggml_mul_mat(ctx, weights[2], active);
        for (auto op : {gate, up, active, output})
            require(ggml_backend_supports_op(backend, op), "unsupported op");
        ggml_build_forward_expand(graph, output);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer, "GPU buffer");
        const char *names[] = {"gate", "up", "down"};
        for (int i = 0; i < 3; ++i) {
            auto t = weights[i];
            std::vector<unsigned char> packed;
            if (type == GGML_TYPE_Q8_0) {
                packed = read_file<unsigned char>(std::string(names[i]) + ".q8", ggml_nbytes(t));
            } else {
                auto f = read_file<float>(std::string(names[i]) + ".f32", ggml_nelements(t));
                packed.resize(ggml_nbytes(t));
                require(ggml_quantize_chunk(type, f.data(), packed.data(), 0, t->ne[1], t->ne[0],
                                            nullptr) == packed.size(),
                        "quantize");
            }
            ggml_backend_tensor_set(t, packed.data(), 0, packed.size());
        }
        std::mt19937 rng(42);
        std::vector<float> values(4096 * batch);
        for (auto &x : values)
            x = float(int(rng() % 65536) - 32768) / 32768.f;
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    }
    ~Graph() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

__global__ void evict(const unsigned *src, size_t n, unsigned *dst) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned sum = 0;
    for (size_t j = i; j < n; j += gridDim.x * blockDim.x) {
        unsigned x;
        asm volatile("ld.global.cg.u32 %0,[%1];" : "=r"(x) : "l"(src + j));
        sum += x;
    }
    dst[i] = sum;
}

static void run(ggml_type target) {
    float limit;
    require(bool(std::ifstream("clamp.txt") >> limit), "clamp");
    auto backend = ggml_backend_cuda_init(0);
    require(backend, "CUDA init");
    cudaDeviceProp prop{};
    gpu(cudaGetDeviceProperties(&prop, 0));
    size_t bytes = std::max(size_t(64) << 20, 4 * size_t(prop.l2CacheSize));
    unsigned *src, *dst;
    gpu(cudaMalloc(&src, bytes));
    gpu(cudaMalloc(&dst, 256 * 256 * sizeof(unsigned)));
    gpu(cudaMemset(src, 1, bytes));
    puts("batch,pair,order,format,elapsed_us");
    for (int batch : {1, 512}) {
        Graph q8(backend, GGML_TYPE_Q8_0, batch, limit), trial(backend, target, batch, limit);
        std::vector<float> reference(4096 * batch);
        for (int pair = -8; pair < (batch == 1 ? 100 : 40); ++pair) {
            for (int order = 0; order < 2; ++order) {
                bool candidate = (pair + 8 + order) % 2;
                Graph &g = candidate ? trial : q8;
                evict<<<256, 256>>>(src, bytes / 4, dst);
                gpu(cudaDeviceSynchronize());
                auto start = std::chrono::steady_clock::now();
                require(ggml_backend_graph_compute(backend, g.graph) == GGML_STATUS_SUCCESS,
                        "compute");
                ggml_backend_synchronize(backend);
                auto end = std::chrono::steady_clock::now();
                if (pair == -8) {
                    std::vector<float> out(reference.size());
                    ggml_backend_tensor_get(g.output, out.data(), 0, out.size() * sizeof(float));
                    if (!candidate)
                        reference = out;
                    else {
                        double err = 0, norm = 0;
                        for (size_t j = 0; j < out.size(); ++j) {
                            err += std::pow(out[j] - reference[j], 2);
                            norm += std::pow(reference[j], 2);
                        }
                        double rms = std::sqrt(err / norm);
                        fprintf(stderr, "batch %d relative output RMS %.6f\n", batch, rms);
                        require(std::isfinite(rms) && rms < 0.5, "invalid output");
                    }
                }
                if (pair >= 0)
                    printf("%d,%d,%d,%s,%.3f\n", batch, pair, order + 1,
                           candidate ? (target == GGML_TYPE_Q5_K ? "Q5_K" : "Q4_K") : "Q8_0",
                           std::chrono::duration<double, std::micro>(end - start).count());
            }
        }
    }
    gpu(cudaFree(src));
    gpu(cudaFree(dst));
    ggml_backend_free(backend);
}
int main(int argc, char **argv) {
    try {
        require(argc == 1 || (argc == 2 && std::string(argv[1]) == "Q5_K"),
                "usage: shared_probe [Q5_K]");
        run(argc == 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K);
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
