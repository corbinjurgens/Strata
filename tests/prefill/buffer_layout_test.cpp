// CPU-only invariants for the prompt batch bounds and aliased weight-workspace layouts.
#include "buffer_layout.hpp"

#include <array>
#include <cstdio>
#include <vector>

namespace d = strata::prefill::detail;

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
    };
    for (size_t chunk : {1u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 255u, 256u, 257u, 1024u}) {
        const auto c = d::prompt_batch_caps(chunk);
        for (int64_t batch : {c.select, c.attention}) {
            check(batch > 0 && batch <= (int64_t) chunk, "positive batch no larger than allocated chunk");
            // A short request followed by a full request uses the same layout, including its final remainder.
            for (size_t request : {size_t(1), chunk / 2 + 1, chunk}) {
                int64_t seen = 0;
                while (seen < (int64_t) request) {
                    const int64_t n = std::min(batch, (int64_t) request - seen);
                    check(n <= batch, "launch fits scratch capacity");
                    seen += n;
                }
                check(seen == (int64_t) request, "sub-batches cover every query once");
            }
        }
        check(c.select <= 256 && c.attention <= 32, "retain original large-chunk launch limits");
    }
    constexpr uint64_t MiB = 1ull << 20;
    static_assert(!d::shared_stage_pressure(420 * MiB, 80 * MiB, 20 * MiB), "exact boundary is healthy");
    check(!d::shared_stage_pressure(420 * MiB, 80 * MiB, 20 * MiB), "exact pressure threshold stays separate");
    check(d::shared_stage_pressure(420 * MiB - 1, 80 * MiB, 20 * MiB), "one byte below threshold selects pressure");
    check(!d::shared_stage_pressure(420 * MiB + 1, 80 * MiB, 20 * MiB), "one byte above threshold stays separate");
    check(d::shared_stage_pressure(0, 0, 0), "zero free memory is pressure");
    check(d::shared_stage_pressure(UINT64_MAX, UINT64_MAX, 1), "normal plus late overflow cannot look healthy");
    check(d::shared_stage_pressure(UINT64_MAX, UINT64_MAX - 1, 0), "floor overflow cannot look healthy");
    check(!d::shared_stage_pressure(UINT64_MAX, 0, 0), "large healthy free count is retained");
    const std::array<uint64_t, 2> fallback = {1280ull * 2560 * 2, 2560ull * 640 * 2};
    // Mixed low-bit packs, Q8 experts whose group exceeds 64 MiB, and builds without MMQ. Include odd tails to
    // ensure alignment does not make the two simultaneous images overlap.
    const uint64_t cases[][2] = {{20 * MiB + 4096, 14 * MiB + 4096}, {54 * MiB + 4096, 28 * MiB + 4096},
                                 {127, 259}, {0, 0}};
    for (const auto& c : cases) for (size_t pairs : {size_t(0), size_t(2)}) {
        const auto w = d::weight_scratch_layout(64 * MiB, c[0], c[1], fallback[0], fallback[1], pairs);
        check(w.bytes >= 64 * MiB, "dense capacity preserved");
        check(w.group_down % 256 == 0 && w.fallback_down % 256 == 0 && w.fallback_pair % 256 == 0,
              "every weight image aligned");
        std::vector<unsigned char> arena((size_t) w.bytes + 256, 0xa5);
        if (c[0]) {
            std::fill_n(arena.begin(), (size_t) c[0], 0x11);
            std::fill_n(arena.begin() + (ptrdiff_t) w.group_down, (size_t) c[1], 0x22);
            check(std::all_of(arena.begin(), arena.begin() + (ptrdiff_t) c[0], [](auto b) { return b == 0x11; }),
                  "MMQ down image cannot overwrite gate/up");
            check(w.group_down >= c[0] + 256 && w.group_down + c[1] + 256 <= w.bytes,
                  "MMQ images retain guards including full-group tails");
        }
        for (size_t i = 0; i < pairs; ++i) {
            const uint64_t gu = i * w.fallback_pair, down = gu + w.fallback_down;
            check(down + fallback[1] + 256 <= w.bytes, "fallback ring contained");
            std::fill_n(arena.begin() + (ptrdiff_t) gu, (size_t) fallback[0], (unsigned char) (3 + 2 * i));
            std::fill_n(arena.begin() + (ptrdiff_t) down, (size_t) fallback[1], (unsigned char) (4 + 2 * i));
        }
        for (size_t i = 0; i < pairs; ++i) {
            const uint64_t gu = i * w.fallback_pair, down = gu + w.fallback_down;
            check(std::all_of(arena.begin() + (ptrdiff_t) gu, arena.begin() + (ptrdiff_t) (gu + fallback[0]),
                              [=](auto b) { return b == 3 + 2 * i; }), "fallback gate/up survives other ring writes");
            check(std::all_of(arena.begin() + (ptrdiff_t) down, arena.begin() + (ptrdiff_t) (down + fallback[1]),
                              [=](auto b) { return b == 4 + 2 * i; }), "fallback down survives other ring writes");
        }
        check(std::all_of(arena.end() - 256, arena.end(), [](auto b) { return b == 0xa5; }), "outer guard intact");
        if (!pairs) check(w.fallback_pair == 0, "no fallback allocation for an all-MMQ plan");
    }

    constexpr auto tiny = d::shared_gemm_stage_layout(127, 259, 768);
    static_assert(tiny.workspace_offset_bytes == 512 && tiny.gemm_extent_bytes == 1280 &&
                  tiny.shared_payload_bytes == 1280, "shared layout is usable in constant expressions");
    struct SharedCase {
        uint64_t weights, workspace, stage, workspace_offset, gemm_extent, shared_payload;
    };
    const SharedCase shared_cases[] = {
        {0, 0, 0, 256, 512, 512},
        {1, 1, 0, 512, 1024, 1024},
        {255, 257, 1024, 512, 1280, 1280},
        {256, 256, 0, 512, 1024, 1024},
        {257, 255, 0, 768, 1280, 1280},
        {127, 259, 768, 512, 1280, 1280},
        {127, 259, 1280, 512, 1280, 1280},
        {127, 259, 1792, 512, 1280, 1792},
    };
    for (const auto& c : shared_cases) {
        const auto l = d::shared_gemm_stage_layout(c.weights, c.workspace, c.stage);
        check(l.workspace_offset_bytes == c.workspace_offset, "guard-sensitive workspace offset is exact");
        check(l.gemm_extent_bytes == c.gemm_extent, "complete GEMM extent includes both guards and odd tails");
        check(l.shared_payload_bytes == c.shared_payload, "GEMM, equal and stage extents retain their exact maximum");
        check(l.workspace_offset_bytes >= c.weights + 256, "workspace follows the full weight arena and its guard");
        check(l.gemm_extent_bytes >= l.workspace_offset_bytes + c.workspace + 256,
              "workspace guard fits inside the complete GEMM extent");
        check(l.workspace_offset_bytes % 256 == 0 && l.gemm_extent_bytes % 256 == 0 &&
              l.shared_payload_bytes % 256 == 0, "shared layout retains 256-byte alignment without page rounding");
    }

    const auto dense_arena = d::weight_scratch_layout(64 * MiB, 0, 0, 0, 0, 0);
    const auto mmq_arena = d::weight_scratch_layout(64 * MiB, 54 * MiB + 4096, 28 * MiB + 4096, 0, 0, 0);
    const auto fallback_arena = d::weight_scratch_layout(64 * MiB, 0, 0, fallback[0], fallback[1], 8);
    check(dense_arena.bytes == 67108864, "shared layout input retains the full dense capacity");
    check(mmq_arena.bytes == 85991936, "shared layout input retains the MMQ arena larger than 64 MiB");
    check(fallback_arena.bytes == 78647296, "shared layout input retains the entire fallback ring");
    struct ArenaCase { uint64_t payload, workspace_offset, gemm_extent; };
    const std::array<ArenaCase, 3> arenas = {{
        {dense_arena.bytes, 67109120, 75497984},
        {mmq_arena.bytes, 85992192, 94381056},
        {fallback_arena.bytes, 78647552, 87036416},
    }};
    // Representative retained-128K counted extents: q4, int8, fp16 and K8V4. These arithmetic fixtures do not
    // exercise take_stage or establish an actual-format integration or a model fit.
    struct StageCase { uint64_t extent, shared_payload[3]; };
    const StageCase stages[] = {
        {75497984, {75497984, 94381056, 87036416}},
        {138413056, {138413056, 138413056, 138413056}},
        {268435968, {268435968, 268435968, 268435968}},
        {106955520, {106955520, 106955520, 106955520}},
    };
    for (const auto& stage : stages) for (size_t i = 0; i < arenas.size(); ++i) {
        const auto& arena = arenas[i];
        const auto l = d::shared_gemm_stage_layout(arena.payload, 8388608, stage.extent);
        check(l.workspace_offset_bytes == arena.workspace_offset, "large workspace follows the complete weight arena");
        check(l.gemm_extent_bytes == arena.gemm_extent, "large GEMM extent preserves full workspace and guards");
        check(l.shared_payload_bytes == stage.shared_payload[i],
              "already-counted stage has no extra guard or page rounding");
        check(l.workspace_offset_bytes >= arena.payload + 256 &&
              l.gemm_extent_bytes >= l.workspace_offset_bytes + 8388608 + 256,
              "large simultaneous weight and workspace images retain their guards");
        check(l.workspace_offset_bytes % 256 == 0 && l.gemm_extent_bytes % 256 == 0 &&
              l.shared_payload_bytes % 256 == 0, "large extents remain 256-byte aligned");
    }

    // Small odd images cover simultaneous GEMM writes, then sequential stage reuse, without large allocations.
    const uint64_t reuse_cases[][2] = {{768, 1280}, {1280, 1280}, {1792, 1792}};
    for (const auto& c : reuse_cases) {
        const auto l = d::shared_gemm_stage_layout(127, 259, c[0]);
        std::array<unsigned char, 2048> arena;
        arena.fill(0xa5);
        std::fill_n(arena.begin(), 127, 0x11);
        std::fill_n(arena.begin() + 127, 256, 0x31);
        std::fill_n(arena.begin() + (ptrdiff_t) l.workspace_offset_bytes, 259, 0x22);
        std::fill_n(arena.begin() + (ptrdiff_t) l.workspace_offset_bytes + 259, 256, 0x32);
        check(std::all_of(arena.begin(), arena.begin() + 127, [](auto b) { return b == 0x11; }),
              "simultaneous workspace writes preserve the weight image");
        check(std::all_of(arena.begin() + 127, arena.begin() + 383, [](auto b) { return b == 0x31; }),
              "simultaneous workspace writes preserve the weight guard");
        check(std::all_of(arena.begin() + 512, arena.begin() + 771, [](auto b) { return b == 0x22; }),
              "workspace image starts at the independent guarded offset");
        check(std::all_of(arena.begin() + 771, arena.begin() + 1027, [](auto b) { return b == 0x32; }),
              "workspace retains its full read-ahead guard");
        check(std::all_of(arena.begin() + 383, arena.begin() + 512, [](auto b) { return b == 0xa5; }) &&
              std::all_of(arena.begin() + 1027, arena.begin() + 1280, [](auto b) { return b == 0xa5; }),
              "odd GEMM tails retain alignment padding");
        check(std::all_of(arena.begin() + (ptrdiff_t) l.shared_payload_bytes,
                          arena.begin() + (ptrdiff_t) l.shared_payload_bytes + 256,
                          [](auto b) { return b == 0xa5; }), "GEMM images preserve the separate outer guard");
        std::fill_n(arena.begin(), (size_t) c[0], 0x44);
        check(l.shared_payload_bytes == c[1], "stage reuse retains the exact enclosing payload extent");
        check(std::all_of(arena.begin() + (ptrdiff_t) l.shared_payload_bytes,
                          arena.begin() + (ptrdiff_t) l.shared_payload_bytes + 256,
                          [](auto b) { return b == 0xa5; }), "stage reuse preserves the separate outer guard");
    }
    std::printf("prefill buffer layouts: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
