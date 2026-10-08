// GPU allocation/capacity regression without model weights. Also run in a separate process with
// STRATA_PREFILL_MMQ=0 to retain and exercise the fallback layout.
#include "buffer_layout.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace pf = strata::prefill;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void require(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }

struct Dev {
    void* p = nullptr;
    size_t bytes;
    explicit Dev(size_t n) : bytes(n) {
        ck(cudaMalloc(&p, n + 256), "allocate guarded buffer");
        ck(cudaMemset(p, 0xa5, n + 256), "set guards");
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template<class T> T* as() { return static_cast<T*>(p); }
    template<class T> void upload(const std::vector<T>& x) {
        require(x.size() * sizeof(T) <= bytes, "fixture upload exceeds buffer");
        ck(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    void guard() {
        std::vector<unsigned char> tail(256);
        ck(cudaMemcpy(tail.data(), as<unsigned char>() + bytes, tail.size(), cudaMemcpyDeviceToHost), "read guard");
        require(std::all_of(tail.begin(), tail.end(), [](auto b) { return b == 0xa5; }), "kernel exceeded exact scratch capacity");
    }
};

void allocation_layouts() {
    strata::core::ModelGeometry g;
    strata::core::WeightTable wt;
    strata::core::QsaState q;
    q.max_cells = 131072;
    q.n_pages = q.max_cells / 4;
    q.kv_mode = 1;
    q.kv_q4 = true;
    strata::core::SessionState ss;
    ss.qsa_states = &q;
    std::string err;
    // A nonexistent pack has the canonical Q2_0 layout; no file is created or model data read.
    const auto absent = std::filesystem::temp_directory_path() / "strata-prefill-allocation-fixture-no-pack";
    require(!std::filesystem::exists(absent), "fixture's absent pack path unexpectedly exists");
    require(k::cpu::expert_layout_load(absent.string(), g.n_layers, g.n_expert, err), err);
    pf::Prefill prefill;
    for (int64_t chunk : {1, 16, 17, 32, 64, 257}) {
        err.clear();
        require(prefill.init(wt, g, ss, nullptr, nullptr, nullptr, chunk, nullptr, err), err);
        require(prefill.chunk() == chunk, "owned chunk capacity changed");
        prefill.reset();
    }
    const uint64_t borrowed = pf::Prefill::bytes_needed(g, ss, 257);
    Dev arena((size_t) borrowed);
    err.clear();
    require(prefill.init(wt, g, ss, nullptr, nullptr, nullptr, 257, nullptr, err, arena.p, borrowed), err);
    for (int64_t chunk : {257, 64, 16, 1, 17, 32, 257}) {
        err.clear();
        const uint64_t priced = pf::Prefill::bytes_needed(g, ss, chunk);
        require(priced <= borrowed, "smaller layout costs more than initialized capacity");
        require(prefill.relayout(chunk, arena.p, priced, err), err);
        require(prefill.chunk() == chunk, "relayout chunk capacity differs");
    }
    arena.guard();
    prefill.reset();
    require(pf::Prefill::bytes_needed(g, ss, 0) == 0, "zero chunk must not price a huge layout");
    require(!prefill.init(wt, g, ss, nullptr, nullptr, nullptr, 0, nullptr, err), "zero chunk accepted");
}

void selection_bounds(int64_t queries, bool tensor_core) {
    const auto s = k::qsa_real_shapes();
    constexpr int64_t cells = 131072;
    const int64_t blocks = cells / s.idx_block + 2;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<float> pooled((size_t) blocks * s.idx_dim), dead((size_t) s.idx_dim), x((size_t) queries * 512);
    for (size_t i = 0; i < pooled.size(); ++i) pooled[i] = ((int) (i % 113) - 56) * 0.003f;
    for (size_t i = 0; i < dead.size(); ++i) dead[i] = ((int) (i % 11) - 5) * 0.05f;
    for (size_t i = 0; i < x.size(); ++i) x[i] = ((int) (i % 37) - 18) * 0.025f;
    std::vector<int32_t> steps((size_t) queries * k::kStepCount);
    for (int64_t i = 0; i < queries; ++i) {
        auto* st = steps.data() + i * k::kStepCount;
        const int64_t n = cells - queries + i;
        st[k::kStepPos] = (int32_t) (n - 1);
        st[k::kStepNKv] = (int32_t) n;
        st[k::kStepNBid] = (int32_t) (n / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(n, s);
    }
    Dev dp(pooled.size() * 4), dd(dead.size() * 4), dx(x.size() * 4), ds(steps.size() * 4),
        tight((size_t) queries * blocks * 4), large((size_t) 256 * blocks * 4),
        ids((size_t) queries * cap * 4), ref((size_t) queries * cap * 4);
    dp.upload(pooled); dd.upload(dead); dx.upload(x); ds.upload(steps);
    auto run = [&](Dev& scores, Dev& out) {
        const bool tc = tensor_core && k::qsa_block_scores_tc(dp.as<float>(), dd.as<float>(), dx.as<float>(),
            ds.as<int32_t>(), queries, blocks, s, scores.as<float>(), nullptr, blocks);
        if (!tc) k::qsa_block_scores(dp.as<float>(), dd.as<float>(), dx.as<float>(), ds.as<int32_t>(), queries,
                                    blocks, s, scores.as<float>(), nullptr, blocks);
        k::qsa_block_topk(scores.as<float>(), ds.as<int32_t>(), queries, blocks, cap, s, out.as<int32_t>(), nullptr, blocks);
    };
    run(tight, ids);
    run(large, ref);
    ck(cudaDeviceSynchronize(), "selection sync");
    tight.guard(); ids.guard(); dx.guard(); ds.guard();
    std::vector<int32_t> a((size_t) queries * cap), b(a.size());
    ck(cudaMemcpy(a.data(), ids.p, a.size() * 4, cudaMemcpyDeviceToHost), "selection results");
    ck(cudaMemcpy(b.data(), ref.p, b.size() * 4, cudaMemcpyDeviceToHost), "selection reference");
    require(a == b, "selection differs with exact-sized scratch");
}

void attention_bounds(int64_t queries, bool q4) {
    const auto s = k::qsa_real_shapes();
    constexpr int64_t cells = 128, width = 68;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const size_t stride = (size_t) k::qsa_decode_attn_scratch_floats(cap, s);
    std::vector<float> x((size_t) queries * s.n_head * s.head_dim);
    for (size_t i = 0; i < x.size(); ++i) x[i] = ((int) (i % 37) - 18) * 0.025f;
    std::vector<int32_t> steps((size_t) queries * k::kStepCount), ids((size_t) queries * cap), table((size_t) cells / s.page_size);
    for (size_t i = 0; i < table.size(); ++i) table[i] = (int32_t) i;
    for (int64_t i = 0; i < queries; ++i) {
        steps[(size_t) i * k::kStepCount + k::kStepWidth] = (int32_t) (width - i % 4);
        for (int64_t j = 0; j < width; ++j) ids[(size_t) (i * cap + j)] = (int32_t) j;
    }
    const size_t elements = (size_t) cells * s.n_head_kv * s.head_dim;
    std::vector<uint16_t> half(elements);
    for (size_t i = 0; i < half.size(); ++i) half[i] = (uint16_t) (0x3400 + (i % 5) * 0x100);
    std::vector<uint8_t> quant(elements / 32 * 18);
    for (size_t i = 0; i < quant.size(); i += 18) {
        const uint16_t scale = 0x3000;
        std::memcpy(quant.data() + i, &scale, 2);
        for (size_t j = 2; j < 18; ++j) quant[i + j] = (uint8_t) ((i + j) % 256);
    }
    Dev kv(q4 ? quant.size() : half.size() * 2), dx(x.size() * 4), ds(steps.size() * 4), di(ids.size() * 4), dt(table.size() * 4),
        tight((size_t) queries * stride * 4), large(32 * stride * 4), out(x.size() * 4), ref(x.size() * 4);
    if (q4) kv.upload(quant); else kv.upload(half);
    dx.upload(x); ds.upload(steps); di.upload(ids); dt.upload(table);
    k::QsaAttnPools p;
    p.page_table = dt.as<int32_t>();
    if (q4) p.k_q4 = p.v_q4 = kv.as<uint8_t>();
    else p.k_pool = p.v_pool = kv.as<uint16_t>();
    k::qsa_decode_attn_batch(dx.as<float>(), p, di.as<int32_t>(), ds.as<int32_t>(), cap, s,
                            tight.as<float>(), out.as<float>(), queries, nullptr);
    k::qsa_decode_attn_batch(dx.as<float>(), p, di.as<int32_t>(), ds.as<int32_t>(), cap, s,
                            large.as<float>(), ref.as<float>(), queries, nullptr);
    ck(cudaDeviceSynchronize(), "attention sync");
    tight.guard(); out.guard();
    std::vector<float> a(x.size()), b(x.size());
    ck(cudaMemcpy(a.data(), out.p, a.size() * 4, cudaMemcpyDeviceToHost), "attention results");
    ck(cudaMemcpy(b.data(), ref.p, b.size() * 4, cudaMemcpyDeviceToHost), "attention reference");
    require(std::memcmp(a.data(), b.data(), a.size() * 4) == 0, "attention differs with exact-sized scratch");
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        allocation_layouts();
        for (int64_t n : {1, 15, 16, 17, 31, 32, 33, 63, 64, 255, 256}) {
            selection_bounds(n, false);
            selection_bounds(n, true);
            if (n <= 32) { attention_bounds(n, false); attention_bounds(n, true); }
        }
        std::puts("prefill allocations and scratch bounds: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill allocations and scratch bounds: %s\n", e.what());
        return 1;
    }
}
