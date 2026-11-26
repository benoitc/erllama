set -eu
cd "$(dirname "$0")"
base="$HOME/erllama-tp/erllama/apps/erllama"
inc="$base/c_src/llama.cpp/ggml/include"
libs="$base/_build/cmake/llama.cpp/ggml/src"
cuda=/usr/local/cuda
"$cuda/bin/nvcc" -std=c++17 -O3 -arch=sm_120 -I"$inc" -c expert_probe.cu -o expert_probe.o
g++ expert_probe.o -Wl,--start-group "$libs/libggml.a" "$libs/libggml-cpu.a" "$libs/ggml-cuda/libggml-cuda.a" "$libs/libggml-base.a" -Wl,--end-group \
    -L"$cuda/targets/sbsa-linux/lib" -Wl,-rpath,"$cuda/targets/sbsa-linux/lib" -lcublas -lcublasLt -lcudart -lcuda -L"$HOME/erllama-tp/nccl/lib" -Wl,-rpath,"$HOME/erllama-tp/nccl/lib" -lnccl -ldl -lpthread -lgomp -lm -o expert_probe
