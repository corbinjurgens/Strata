// GPU allocation/capacity regression without model weights. Also run in a separate process with
// STRATA_PREFILL_MMQ=0 to retain and exercise the fallback layout.
#include "buffer_layout.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/core/peer_experts.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
        require(!prefill.shared_stage(), "default/manual init selected the private layout");
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

void shared_owner_layouts() {
    strata::core::ModelGeometry g;
    strata::core::WeightTable wt;
    strata::core::QsaState q;
    q.kv_mode = 1;
    strata::core::SessionState ss;
    ss.qsa_states = &q;
    std::string err;
    const char* prefetch = std::getenv("STRATA_KV_PREFETCH");
    const bool excluded = std::getenv("STRATA_KV_STAGE_OWN") != nullptr || (prefetch && std::atoi(prefetch) != 0);
#if defined(_WIN32) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL)
    const bool expect_shared = !excluded;
#else
    const bool expect_shared = false;
    (void) excluded;
#endif
    pf::Prefill shared, normal;
    for (int64_t cells : {65536, 131072}) {
        q.max_cells = cells; q.n_pages = cells / 4;
        q.kv_mode = 0;
        const uint64_t no_stage = pf::Prefill::bytes_needed_owned(g, ss, 17);
        q.kv_mode = 1;
        // Independent expected page-rounded staging prices: q4, int8, fp16 and K8V4, in that order.
        const uint64_t prices64[] = {40, 72, 132, 56}, prices128[] = {76, 140, 260, 108};
        for (int format = 0; format < 4; ++format) {
            q.kv_q4 = format == 0; q.kv_int8 = format == 1; q.kv_hybrid = format == 3;
            const uint64_t owned = pf::Prefill::bytes_needed_owned(g, ss, 17);
            const uint64_t loan = pf::Prefill::bytes_needed(g, ss, 17);
            if (std::getenv("STRATA_KV_STAGE_OWN") == nullptr)
                require(owned - no_stage == (cells == 65536 ? prices64[format] : prices128[format]) * (1ull << 20),
                        "format-aware unshared stage price changed");
            // A maximal remaining-priced hint forces pressure without a production query/fault hook.
            require(shared.init(wt, g, ss, nullptr, nullptr, nullptr, 17, nullptr, err,
                                nullptr, 0, true, UINT64_MAX), err);
            require(shared.shared_stage() == expect_shared, "private pressure/backend/prefetch/stage-own admission differs");
            require(pf::Prefill::bytes_needed_owned(g, ss, 17) == owned && pf::Prefill::bytes_needed(g, ss, 17) == loan,
                    "selecting a physical shared slab changed the conservative prices");
            if (expect_shared) {
                require(!shared.set_stage_helper(&normal, err), "shared owner accepted a helper");
                require(!normal.set_stage_helper(&shared, err), "normal owner accepted a shared helper");
                require(!shared.set_stage_helper(nullptr, err), "shared owner accepted helper detachment");
                require(!shared.set_stage_helper(&shared, err), "shared owner accepted helper no-op");
                strata::core::PeerExperts invalid;
                require(!shared.set_peer(nullptr, 1, err), "shared owner accepted null peer detachment");
                require(!shared.set_peer(&invalid, 1, err), "shared owner accepted invalid peer detachment");
                require(shared.shared_stage() && shared.chunk() == 17 && !normal.shared_stage() && normal.chunk() == 0,
                        "rejected attachments changed either participant's selected layout/chunk");
            }
            shared.reset();
            require(!shared.shared_stage() && shared.chunk() == 0, "reset retained the shared layout");
            err.clear();
        }
    }
    q.kv_hybrid = false; q.kv_q4 = true;
    const uint64_t borrowed = pf::Prefill::bytes_needed(g, ss, 17);
    Dev arena((size_t) borrowed);
    require(shared.init(wt, g, ss, nullptr, nullptr, nullptr, 17, nullptr, err,
                        arena.p, borrowed, true, UINT64_MAX), err);
    require(!shared.shared_stage(), "borrowed arena selected sharing");
    shared.reset();
    q.kv_mode = 0;
    require(shared.init(wt, g, ss, nullptr, nullptr, nullptr, 17, nullptr, err,
                        nullptr, 0, true, UINT64_MAX), err);
    require(!shared.shared_stage(), "resident KV selected sharing");
    shared.reset();
    q.kv_mode = 1;
    shared.set_stage(1, -1, nullptr);
    require(shared.init(wt, g, ss, nullptr, nullptr, nullptr, 17, nullptr, err,
                        nullptr, 0, true, UINT64_MAX), err);
    require(!shared.shared_stage(), "partial layer stage selected sharing");
    shared.reset();
    arena.guard();
}

// Run separately with STRATA_PREFILL_DEMAND_ONLY=1 and STRATA_PREFILL_STREAM_MIN=1024. A 96-slot allocation
// must retain full-chunk fallback scratch, even on CUDA hardware that normally takes the fused experts.
void demand_only_layout() {
    const char* demand = std::getenv("STRATA_PREFILL_DEMAND_ONLY");
    require(demand && demand[0] == '1', "demand-only layout fixture requires STRATA_PREFILL_DEMAND_ONLY=1");
    require(pf::Prefill::stream_all_min_tokens() == 1024, "demand-only fixture requires the original 1024-token threshold");
    struct RingEnv {
        std::string old;
        bool present;
        RingEnv() : present(std::getenv("STRATA_PREFILL_RING") != nullptr) {
            if (present) old = std::getenv("STRATA_PREFILL_RING");
        }
        static void set(const char* value) {
#if defined(_WIN32)
            require(_putenv_s("STRATA_PREFILL_RING", value ? value : "") == 0, "set fixture ring");
#else
            require((value ? setenv("STRATA_PREFILL_RING", value, 1) : unsetenv("STRATA_PREFILL_RING")) == 0,
                    "set fixture ring");
#endif
        }
        ~RingEnv() { set(present ? old.c_str() : nullptr); }
    } ring_env;
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
    const auto absent = std::filesystem::temp_directory_path() / "strata-prefill-allocation-fixture-no-pack";
    require(!std::filesystem::exists(absent), "fixture's absent pack path unexpectedly exists");
    require(k::cpu::expert_layout_load(absent.string(), g.n_layers, g.n_expert, err), err);
    struct NoReadSource : strata::core::ExpertSource {
        const uint8_t* blob(int64_t, int64_t) override { throw std::runtime_error("allocation fixture read a model blob"); }
    } source;
    RingEnv::set("96");
    require(pf::Prefill::ring_slots_for(1023) == 8, "demand-only changed the small-chunk ring");
    require(pf::Prefill::ring_slots_for(1024) == 96, "demand-only changed the original streaming threshold/ring");
    constexpr int64_t chunk = 4096;
    const uint64_t priced96 = pf::Prefill::bytes_needed(g, ss, chunk);
    const uint64_t no_ring = pf::Prefill::bytes_needed_no_ring(g, ss, chunk);
    require(priced96 - no_ring == 96ull * 1382400, "96-slot ring payload price differs from canonical Q2_0");
    RingEnv::set("8");
    const uint64_t priced8 = pf::Prefill::bytes_needed(g, ss, chunk);
    // The explicit eight-slot layout already requires full-chunk fallback buffers. Comparing it against 96
    // catches an accidental fused scratch shrink in the demand-only layout, rather than copying its formula.
    require(priced96 == priced8 + 88ull * (1382400 + 256), "demand-only retained undersized fused MoE scratch");
    RingEnv::set("96");
    Dev arena((size_t) priced96);
    pf::Prefill prefill;
    require(!prefill.init(wt, g, ss, &source, nullptr, nullptr, chunk, nullptr, err, arena.p, priced96 - (64ull << 20)),
            "demand-only accepted an undersized borrowed region");
    require(!err.empty(), "demand-only allocation failure omitted its error");
    prefill.reset();
    err.clear();
    require(prefill.init(wt, g, ss, &source, nullptr, nullptr, chunk, nullptr, err, arena.p, priced96), err);
    for (int64_t n : {int64_t(1023), int64_t(1024), chunk}) {
        const uint64_t price = pf::Prefill::bytes_needed(g, ss, n);
        require(price <= priced96, "demand-only relayout exceeds the initial capacity");
        require(prefill.relayout(n, arena.p, price, err), err);
        require(prefill.chunk() == n, "demand-only relayout changed chunk capacity");
    }
    arena.guard();
    std::printf("demand-only layout: chunk %lld ring 96 bytes %llu; eight-slot bytes %llu\n",
                (long long) chunk, (unsigned long long) priced96, (unsigned long long) priced8);
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

// Exercise the real stage DMA and append into a previously used GEMM slab, in every KV storage format. Only the old
// prefix and appended cells become finite; the rest of each full-capacity pool stays NaN-poisoned, including a
// partial last page. Each storage run's bytes come from its own formula and end in a guard: two equal planes for
// fp16 and q4_0; int8's K/V codes each ride their per-64 fp16 scale run; K8V4 (kKvHybrid, kv_stream.hpp) is int8 K
// codes, their scales, and rotated q4_0 V - three runs, like take_stage stages them in prefill.cpp.
void staged_attention_bounds(int64_t queries, int fmt, int64_t cells, int64_t prefix) {
    const auto s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const size_t stride = (size_t) k::qsa_decode_attn_scratch_floats(cap, s);
    const size_t rows = (size_t) (cells * s.n_head_kv), codes = (size_t) s.head_dim,
                 scales = (size_t) (s.head_dim / 64) * 2;
    std::vector<size_t> row_bytes;
    switch (fmt) {
    case k::kKvF16:    row_bytes = {512, 512}; break;                      // fp16 planes
    case k::kKvInt8:   row_bytes = {codes, codes, scales, scales}; break;   // K/V codes, then K/V scale runs
    case k::kKvQ4:     row_bytes = {144, 144}; break;                      // q4_0 planes
    case k::kKvHybrid: row_bytes = {codes, scales, 144}; break;             // K codes, K scales, q4_0 V
    }
    std::vector<size_t> run_off(row_bytes.size(), 0), run_bytes(row_bytes.size(), 0);
    for (size_t r = 0; r < row_bytes.size(); ++r) {
        if (r) run_off[r] = run_off[r - 1] + (size_t) pf::detail::buffer_bytes((uint64_t) run_bytes[r - 1]);
        run_bytes[r] = rows * row_bytes[r];
    }
    const size_t stage_bytes = run_off.back() + (size_t) pf::detail::buffer_bytes((uint64_t) run_bytes.back());
    const auto layout = pf::detail::shared_gemm_stage_layout(64ull << 20, 8ull << 20, stage_bytes);
    Dev shared((size_t) layout.shared_payload_bytes), separate(stage_bytes);
    ck(cudaMemset(shared.p, 0xff, shared.bytes), "poison old GEMM image and workspace");
    ck(cudaMemset(separate.p, 0x7f, separate.bytes), "poison separate staging image differently");
    struct Host {
        unsigned char* p = nullptr;
        explicit Host(size_t bytes) {
            ck(cudaHostAlloc((void**) &p, bytes, cudaHostAllocMapped), "allocate stage host prefix");
            std::memset(p, 0xff, bytes);
        }
        ~Host() { cudaFreeHost(p); }
    } host(stage_bytes);
    const auto fill_f16 = [&](unsigned char* dst, size_t seed) {
        for (size_t d = 0; d < 512; d += 2) {
            const uint16_t value = (uint16_t) (0x3400 + (seed + d) % 5 * 0x100);
            std::memcpy(dst + d, &value, 2);
        }
    };
    const auto fill_q4 = [&](unsigned char* dst, size_t seed) {
        for (size_t b = 0; b < 144; b += 18) {
            const uint16_t scale = 0x3000;
            std::memcpy(dst + b, &scale, 2);
            std::memset(dst + b + 2, (int) (0x65 + seed % 5), 16);
        }
    };
    const auto fill_int8 = [&](unsigned char* dst, size_t seed) {
        for (size_t d = 0; d < codes; ++d) dst[d] = (unsigned char) (0x70 - (seed + d) % 32);
    };
    const auto fill_scale = [&](unsigned char* dst) {
        for (size_t d = 0; d < scales; d += 2) {
            const uint16_t scale = 0x3000;
            std::memcpy(dst + d, &scale, 2);
        }
    };
    for (int64_t cell = 0; cell < prefix; ++cell) for (int64_t h = 0; h < s.n_head_kv; ++h) {
        const size_t row = (size_t) ((cell / s.page_size * s.n_head_kv + h) * s.page_size + cell % s.page_size);
        const size_t seed = (size_t) (cell + h);
        for (size_t r = 0; r < row_bytes.size(); ++r) {
            unsigned char* dst = host.p + run_off[r] + row * row_bytes[r];
            const bool scale_run = (fmt == k::kKvInt8 && r >= 2) || (fmt == k::kKvHybrid && r == 1);
            const bool q4_run = fmt == k::kKvQ4 || (fmt == k::kKvHybrid && r == 2);
            if (scale_run) fill_scale(dst);
            else if (q4_run) fill_q4(dst, seed);
            else if (fmt == k::kKvF16) fill_f16(dst, seed);
            else fill_int8(dst, seed);
        }
    }
    // The format's pool pointers into one image, one run per offset.
    auto host_pools = [&](unsigned char* base) {
        k::KvHostPools p;
        switch (fmt) {
        case k::kKvF16:
            p.k_pool = reinterpret_cast<uint16_t*>(base + run_off[0]);
            p.v_pool = reinterpret_cast<uint16_t*>(base + run_off[1]);
            break;
        case k::kKvInt8:
            p.k_q = reinterpret_cast<int8_t*>(base + run_off[0]);
            p.v_q = reinterpret_cast<int8_t*>(base + run_off[1]);
            p.k_scale = reinterpret_cast<uint16_t*>(base + run_off[2]);
            p.v_scale = reinterpret_cast<uint16_t*>(base + run_off[3]);
            break;
        case k::kKvQ4:
            p.k_q4 = base + run_off[0];
            p.v_q4 = base + run_off[1];
            break;
        case k::kKvHybrid:
            p.k_q = reinterpret_cast<int8_t*>(base + run_off[0]);
            p.k_scale = reinterpret_cast<uint16_t*>(base + run_off[1]);
            p.v_q4 = base + run_off[2];
            break;
        }
        return p;
    };
    const auto hp = host_pools(host.p);
    const auto a = host_pools(shared.as<unsigned char>()), b = host_pools(separate.as<unsigned char>());
    std::vector<int32_t> table((size_t) cells / s.page_size), missing(table.size(), -1);
    for (size_t i = 0; i < table.size(); ++i) table[i] = (int32_t) i;
    std::vector<int32_t> steps((size_t) queries * k::kStepCount), ids((size_t) queries * cap, -1);
    for (int64_t i = 0; i < queries; ++i) {
        k::qsa_step_fill(steps.data() + i * k::kStepCount, prefix + i, s);
        // A partial 32-cell tensor tile and 64-cell decode tile, reading the new tail as well as the old prefix.
        const int64_t width = std::min<int64_t>(prefix + i + 1, 65 + i % 4);
        steps[(size_t) i * k::kStepCount + k::kStepWidth] = (int32_t) width;
        for (int64_t j = 0; j < width; ++j)
            ids[(size_t) (i * cap + j)] = (int32_t) (prefix + i + 1 - width + j);
    }
    std::vector<float> x((size_t) queries * s.n_head * s.head_dim), current((size_t) queries * 512);
    for (size_t i = 0; i < x.size(); ++i) x[i] = ((int) (i % 37) - 18) * 0.025f;
    for (size_t i = 0; i < current.size(); ++i) current[i] = ((int) (i % 29) - 14) * 0.0625f;
    Dev dt(table.size() * 4), dm(missing.size() * 4), ds(steps.size() * 4), di(ids.size() * 4),
        dx(x.size() * 4), dc(current.size() * 4), dv(current.size() * 4), scratch((size_t) queries * stride * 4),
        out(x.size() * 4), ref(x.size() * 4);
    dt.upload(table); dm.upload(missing); ds.upload(steps); di.upload(ids); dx.upload(x); dc.upload(current); dv.upload(current);
    auto pools = [&](const k::KvHostPools& p) {
        k::QsaAttnPools r;
        r.k_pool = p.k_pool; r.v_pool = p.v_pool; r.k_q = p.k_q; r.v_q = p.v_q;
        r.k_scale = p.k_scale; r.v_scale = p.v_scale; r.k_q4 = p.k_q4; r.v_q4 = p.v_q4;
        r.page_table = dt.as<int32_t>();
        return r;
    };
    if (fmt == k::kKvHybrid) k::fwht256_inplace_cuda(dv.as<float>(), queries * 2, nullptr);   // K8V4: only V rotates
    for (Dev* image : {&shared, &separate}) {
        unsigned char* base = image->as<unsigned char>();
        for (size_t r = 0; r < run_bytes.size(); ++r)
            ck(cudaMemset(base + run_off[r] + run_bytes[r], 0xa5, 256), "set a pool run guard");
        // The real staging DMA: the host copy's finite prefix (its poison) over the pool's poison.
        k::kv_stage_from_host(pools(host_pools(base)), hp, fmt, (prefix + s.page_size - 1) / s.page_size, s, nullptr);
        const k::KvHostPools p = host_pools(base);
        // The real appends, shaped like production's (prefill.cpp): the direct pool args point at the staged pools,
        // whose table passes no resident blocks (`missing`), so every appended cell rides the stage path.
        if (fmt == k::kKvQ4)
            k::kv_append_q4(p.k_q4, p.v_q4, dm.as<int32_t>(), prefix, queries, dc.as<float>(), dc.as<float>(), s,
                            nullptr, nullptr, &p);
        else if (fmt == k::kKvF16)
            pf::kv_append(dc.as<float>(), dc.as<float>(), queries, prefix, dm.as<int32_t>(), s.page_size,
                          p.k_pool, p.v_pool, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &p);
        else if (fmt == k::kKvInt8)
            pf::kv_append(dc.as<float>(), dc.as<float>(), queries, prefix, dm.as<int32_t>(), s.page_size,
                          nullptr, nullptr, p.k_q, p.v_q, p.k_scale, p.v_scale, nullptr, nullptr, &p);
        else if (fmt == k::kKvHybrid) {   // the folded halves: int8 codes/scales for K, rotated q4_0 for V
            const k::KvHostPools sk = k::kv_hybrid_k_half(p), sv = k::kv_hybrid_v_half(p);
            pf::kv_append(dc.as<float>(), dc.as<float>(), queries, prefix, dm.as<int32_t>(), s.page_size,
                          nullptr, nullptr, p.k_q, p.k_q, p.k_scale, p.k_scale, nullptr, nullptr, &sk);
            k::kv_append_q4(p.v_q4, p.v_q4, dm.as<int32_t>(), prefix, queries, dv.as<float>(), dv.as<float>(), s,
                            nullptr, nullptr, &sv);
        }
    }
    auto compare = [&](bool tensor) {
        const bool launched_a = tensor && k::qsa_prompt_attn_batch(dx.as<float>(), pools(a), di.as<int32_t>(),
            ds.as<int32_t>(), cap, s, out.as<float>(), queries, nullptr);
        const bool launched_b = tensor && k::qsa_prompt_attn_batch(dx.as<float>(), pools(b), di.as<int32_t>(),
            ds.as<int32_t>(), cap, s, ref.as<float>(), queries, nullptr);
        require(launched_a == launched_b, "shared pool changed tensor attention dispatch");
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL)
        if (tensor) {
            int device = 0, major = 0;
            ck(cudaGetDevice(&device), "tensor fixture device");
            ck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "tensor fixture capability");
            const char* warp = std::getenv("STRATA_QSA_WARP");
            const bool forced_warp = warp && (!std::strcmp(warp, "1") || !std::strcmp(warp, "attn"));
            if (fmt == k::kKvQ4) {
                // sm_80+ only: on Turing mode 4 would run as pairs of m16n8k8 MMAs, which no parity run has checked yet
                const char* q4_mode = std::getenv("STRATA_PROMPT_ATTN_Q4");
                if (major >= 8 && !forced_warp && !(q4_mode && q4_mode[0] == '0'))
                    require(launched_a, "supported default q4 tensor attention was not exercised");
            } else if (major >= 7) {
                require(launched_a, "supported tensor attention was not exercised for the staged format");
            }
        }
#endif
        if (!launched_a) {
            k::qsa_decode_attn_batch(dx.as<float>(), pools(a), di.as<int32_t>(), ds.as<int32_t>(), cap, s,
                                    scratch.as<float>(), out.as<float>(), queries, nullptr);
            k::qsa_decode_attn_batch(dx.as<float>(), pools(b), di.as<int32_t>(), ds.as<int32_t>(), cap, s,
                                    scratch.as<float>(), ref.as<float>(), queries, nullptr);
        }
        ck(cudaDeviceSynchronize(), "poisoned stage attention sync");
        std::vector<float> av(x.size()), bv(x.size());
        ck(cudaMemcpy(av.data(), out.p, av.size() * 4, cudaMemcpyDeviceToHost), "shared poisoned attention result");
        ck(cudaMemcpy(bv.data(), ref.p, bv.size() * 4, cudaMemcpyDeviceToHost), "separate poisoned attention result");
        require(std::all_of(av.begin(), av.end(), [](float v) { return std::isfinite(v); }) &&
                std::all_of(bv.begin(), bv.end(), [](float v) { return std::isfinite(v); }),
                "attention read poisoned unwritten cells");
        require(std::memcmp(av.data(), bv.data(), av.size() * 4) == 0, "shared and separate staged attention differ");
        std::printf("staged attention: cells=%lld prefix=%lld queries=%lld format=%d tensor=%d\n",
                    (long long) cells, (long long) prefix, (long long) queries, fmt, (int) launched_a);
    };
    compare(false);   // Forced decode fallback, regardless of tensor availability.
    compare(true);
    for (Dev* image : {&shared, &separate}) {
        const unsigned char* base = image->as<unsigned char>();
        for (size_t r = 0; r < run_bytes.size(); ++r) {
            std::vector<unsigned char> guard(256);
            ck(cudaMemcpy(guard.data(), base + run_off[r] + run_bytes[r], guard.size(), cudaMemcpyDeviceToHost),
               "read a pool run guard");
            require(std::all_of(guard.begin(), guard.end(), [](auto v) { return v == 0xa5; }),
                    "stage exceeded a pool run guard");
        }
        image->guard();
    }
    scratch.guard(); out.guard(); ref.guard(); dt.guard(); dm.guard(); ds.guard(); di.guard(); dc.guard(); dv.guard(); dx.guard();
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
    if (queries == 17) for (int64_t capacity : {65536, 131072})
        for (int64_t prefix : {int64_t(0), int64_t(67), capacity - 35})
            staged_attention_bounds(queries, q4 ? k::kKvQ4 : k::kKvF16, capacity, prefix);
}
}  // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        if (argc == 2 && std::strcmp(argv[1], "--demand-only-layout") == 0) {
            demand_only_layout();
            std::puts("prefill demand-only allocation bounds: PASS");
            return 0;
        }
        require(argc == 1, "unknown allocation fixture argument");
        allocation_layouts();
        shared_owner_layouts();
        for (int64_t n : {1, 15, 16, 17, 31, 32, 33, 63, 64, 255, 256}) {
            selection_bounds(n, false);
            selection_bounds(n, true);
            if (n <= 32) { attention_bounds(n, false); attention_bounds(n, true); }
        }
        // The int8 and K8V4 staged formats ride the same query count, capacities and prefixes the fp16/q4 bounds use.
        for (int format : {k::kKvInt8, k::kKvHybrid})
            for (int64_t capacity : {65536, 131072})
                for (int64_t prefix : {int64_t(0), int64_t(67), capacity - 35})
                    staged_attention_bounds(17, format, capacity, prefix);
        std::puts("prefill allocations and scratch bounds: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill allocations and scratch bounds: %s\n", e.what());
        return 1;
    }
}
