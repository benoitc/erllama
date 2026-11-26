#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4)
        return 2;
    const int cols = std::atoi(argv[1]), rows = std::atoi(argv[2]);
    if (!((cols == 4096 && rows == 2048) || (cols == 2048 && rows == 4096)))
        return 2;
    const std::string format = argc == 4 ? argv[3] : "Q4_K";
    if (format != "Q4_K" && format != "Q5_K")
        return 2;
    const ggml_type type = format == "Q5_K" ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
    constexpr int chunk = 64;
    std::vector<float> values(chunk * cols);
    std::vector<unsigned char> packed(chunk * ggml_row_size(type, cols));
    for (int row = 0; row < rows; row += chunk) {
        if (std::fread(values.data(), sizeof(float), values.size(), stdin) != values.size())
            return 1;
        if (ggml_quantize_chunk(type, values.data(), packed.data(), 0, chunk, cols, nullptr) !=
            packed.size())
            return 1;
        if (std::fwrite(packed.data(), 1, packed.size(), stdout) != packed.size())
            return 1;
    }
    return std::fgetc(stdin) == EOF && !std::ferror(stdin) && std::fflush(stdout) == 0 ? 0 : 1;
}
