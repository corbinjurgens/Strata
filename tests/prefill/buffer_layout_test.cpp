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
    std::printf("prefill buffer layouts: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
