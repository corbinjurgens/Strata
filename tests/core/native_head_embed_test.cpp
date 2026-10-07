// The UD-Q2_K_XL head (Q4_K) and embedding (Q5_K) through their real GGUF loaders and GPU entry points.
// Synthetic split shards; the numerical oracle is GGML CPU dequantization, not another GPU wrapper.
#include "gguf_fixture.hpp"
#include "strata/core/native_head.hpp"
#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {
constexpr int WIDTH = 512, VOCAB = 17;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}
void cuda_check(cudaError_t status, const char* what) {
    require(status == cudaSuccess, std::string(what) + ": " + cudaGetErrorString(status));
}

struct TempDir {
    fs::path path = fs::temp_directory_path() /
        ("strata-head-embed-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempDir() { fs::create_directories(path); }
    ~TempDir() { std::error_code ignored; fs::remove_all(path, ignored); }
};

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};

template<class T> struct Device {
    T* data = nullptr;
    explicit Device(size_t count) { cuda_check(cudaMalloc((void**) &data, count * sizeof(T)), "allocate buffer"); }
    ~Device() { if (data) cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

struct Matrix {
    std::vector<uint8_t> packed;
    std::vector<float> decoded;
};

Matrix matrix(ggml_type type, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.f, 0.15f);
    std::vector<float> weights((size_t) WIDTH * VOCAB);
    for (int r = 0; r < VOCAB; ++r)
        for (int c = 0; c < WIDTH; ++c)
            weights[(size_t) r * WIDTH + c] = normal(rng) + 0.025f * (float) (r % 5 - 2);
    const size_t row_bytes = ggml_row_size(type, WIDTH);
    Matrix result{std::vector<uint8_t>(row_bytes * VOCAB), std::vector<float>(weights.size())};
    require(ggml_quantize_chunk(type, weights.data(), result.packed.data(), 0, VOCAB, WIDTH, nullptr) ==
                result.packed.size(), "GGML quantized byte count");
    const auto* traits = ggml_get_type_traits(type);
    for (int r = 0; r < VOCAB; ++r)
        traits->to_float(result.packed.data() + r * row_bytes, result.decoded.data() + r * WIDTH, WIDTH);
    for (float value : result.decoded) require(std::isfinite(value), "GGML reference must be finite");
    return result;
}

void write_payload(const fs::path& path, const fixture::Written& written, size_t tensor, const Matrix& m) {
    std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
    out.seekp((std::streamoff) (written.data_start + written.offsets.at(tensor)));
    out.write((const char*) m.packed.data(), (std::streamsize) m.packed.size());
    require((bool) out, "write quantized payload");
}

std::vector<std::string> write_shards(const fs::path& dir, const Matrix& head, const Matrix& embedding) {
    const fs::path first = dir / "model-00001-of-00003.gguf";
    const fs::path second = dir / "model-00002-of-00003.gguf";
    const fs::path third = dir / "model-00003-of-00003.gguf";
    auto metadata = fixture::split_keys(0, 3, 4);
    const std::vector<fixture::Kv> arch{
        fixture::str("general.architecture", "qwen4exp"), fixture::u32("qwen4exp.block_count", 48),
        fixture::u32("qwen4exp.embedding_length", 2560), fixture::u32("qwen4exp.expert_count", 512),
        fixture::u32("qwen4exp.expert_used_count", 10), fixture::u32("qwen4exp.attention.head_count", 24),
        fixture::u32("qwen4exp.attention.head_count_kv", 2)};
    metadata.insert(metadata.end(), arch.begin(), arch.end());
    fixture::write(first, metadata, {});  // metadata-only first shard, as in Unsloth's split GGUFs
    // A preceding tensor makes both matrix offsets nonzero; metadata is deliberately absent from these shards.
    const auto emb = fixture::write(second, fixture::split_keys(1, 3, 4),
        {{"padding.emb", {3}, 0, 1}, {"token_embd.weight", {WIDTH, VOCAB}, GGML_TYPE_Q5_K, 2}});
    const auto out = fixture::write(third, fixture::split_keys(2, 3, 4),
        {{"padding.head", {5}, 0, 3}, {"output.weight", {WIDTH, VOCAB}, GGML_TYPE_Q4_K, 4}});
    write_payload(second, emb, 1, embedding);
    write_payload(third, out, 1, head);
    return strata::gguf_split_paths(first.string());
}

void check_head(const std::vector<std::string>& shards, const Matrix& weights, cudaStream_t stream) {
    strata::core::NativeHead head;
    std::string error;
    const bool loaded = head.load(shards, WIDTH, VOCAB, error);
    require(loaded, "Q4_K head load: " + error);
    require(head.loaded() && head.type() == GGML_TYPE_Q4_K && head.weight_bytes() == weights.packed.size(),
            "Q4_K head loader state");
    Device<float> input(WIDTH), output(VOCAB);
    std::mt19937 rng(173);
    std::normal_distribution<float> normal(0.f, 1.f);
    for (int sample = 0; sample < 3; ++sample) {
        std::vector<float> x(WIDTH), got(VOCAB);
        for (float& value : x) value = normal(rng);
        cuda_check(cudaMemcpyAsync(input.data, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
                   "upload head input");
        const bool ran = head.run(input.data, output.data, stream, error);
        require(ran, "Q4_K head run: " + error);
        cuda_check(cudaMemcpyAsync(got.data(), output.data, got.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
                   "read head output");
        cuda_check(cudaStreamSynchronize(stream), "head synchronize");
        double difference = 0, magnitude = 0, max_error = 0, peak = 0;
        for (int r = 0; r < VOCAB; ++r) {
            double reference = 0;
            for (int c = 0; c < WIDTH; ++c)
                reference += (double) weights.decoded[(size_t) r * WIDTH + c] * x[c];
            require(std::isfinite(got[r]) && std::isfinite(reference), "finite head output/reference");
            const double delta = std::fabs((double) got[r] - reference);
            difference += delta;
            magnitude += std::fabs(reference);
            max_error = (std::max)(max_error, delta);
            peak = (std::max)(peak, std::fabs(reference));
        }
        const double relative = difference / (magnitude + 1e-30);
        std::printf("Q4_K head sample %d: relative L1 %.3e, max absolute %.3e, reference peak %.3e\n",
                    sample, relative, max_error, peak);
        // The CPU oracle uses unrounded activations; NativeHead rounds them to Q8_1 before the GPU dot.
        require(magnitude > 1 && relative < 0.02 && max_error < 0.02 * peak + 1e-5,
                "Q4_K head differs from GGML-dequantized CPU matrix product");
    }
}

void check_embedding(const std::vector<std::string>& shards, const Matrix& weights, cudaStream_t stream) {
    strata::core::NativeEmbed embedding;
    std::string error;
    const bool loaded = embedding.load(shards, WIDTH, VOCAB, error);
    require(loaded, "Q5_K embedding load: " + error);
    require(embedding.type() == GGML_TYPE_Q5_K && embedding.bytes() == weights.packed.size(),
            "Q5_K embedding loader state");
    const std::vector<int32_t> tokens{VOCAB - 1, 0, 7, 2, 7};  // boundaries, out of order, and a repeated row
    Device<int32_t> ids(tokens.size());
    Device<float> single((size_t) WIDTH * tokens.size()), batched((size_t) WIDTH * tokens.size());
    cuda_check(cudaMemcpyAsync(ids.data, tokens.data(), tokens.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                              stream), "upload token ids");
    for (size_t i = 0; i < tokens.size(); ++i)
        embedding.gather_one(tokens[i], single.data + i * WIDTH, stream);
    embedding.gather_dev(ids.data, (int64_t) tokens.size(), batched.data, stream);
    cuda_check(cudaGetLastError(), "embedding launch");
    std::vector<float> one((size_t) WIDTH * tokens.size()), many(one.size());
    cuda_check(cudaMemcpyAsync(one.data(), single.data, one.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
               "read single embedding rows");
    cuda_check(cudaMemcpyAsync(many.data(), batched.data, many.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
               "read batched embedding rows");
    cuda_check(cudaStreamSynchronize(stream), "embedding synchronize");
    double maximum[2] = {};
    for (size_t i = 0; i < tokens.size(); ++i) {
        for (int c = 0; c < WIDTH; ++c) {
            const float reference = weights.decoded[(size_t) tokens[i] * WIDTH + c];
            const float actual[] = {one[i * WIDTH + c], many[i * WIDTH + c]};
            for (int path = 0; path < 2; ++path) {
                require(std::isfinite(actual[path]), "finite embedding output");
                const double delta = std::fabs((double) actual[path] - reference);
                maximum[path] = (std::max)(maximum[path], delta);
                require(delta <= 1e-6 * (1.0 + std::fabs((double) reference)),
                        "Q5_K embedding differs from GGML CPU dequantization");
            }
        }
    }
    std::printf("Q5_K embedding: gather_one max absolute %.3e, gather_dev %.3e vs GGML CPU\n",
                maximum[0], maximum[1]);
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("native_head_embed_test: no CUDA device, skipped\n");
        return 77;
    }
    try {
        TempDir tmp;
        Stream stream;
        const Matrix head = matrix(GGML_TYPE_Q4_K, 71), embedding = matrix(GGML_TYPE_Q5_K, 93);
        const auto shards = write_shards(tmp.path, head, embedding);
        require(shards.size() == 3, "discover all split shards from the metadata shard");
        check_head(shards, head, stream.value);
        check_embedding(shards, embedding, stream.value);
        std::printf("native_head_embed_test: all passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "native_head_embed_test: FAIL: %s\n", e.what());
        return 1;
    }
}
