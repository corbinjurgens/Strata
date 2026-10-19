#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::prefill::detail {

// Every carved buffer keeps the allocator's read-ahead guard and 256-byte alignment.
constexpr uint64_t buffer_bytes(uint64_t payload) { return (payload + 256 + 255) & ~255ull; }

struct PromptBatchCaps { int64_t select, attention; };
constexpr PromptBatchCaps prompt_batch_caps(size_t chunk) {
    return {static_cast<int64_t>(std::min<size_t>(chunk, 256)),
            static_cast<int64_t>(std::min<size_t>(chunk, 32))};
}

// Heuristic only: retain the normal owned price, remaining explicitly priced late buffers, the existing
// 256 MiB Windows CUDA floor and 64 MiB allocator allowance. Subtraction also handles an overflowing sum.
constexpr bool shared_stage_pressure(uint64_t free_bytes, uint64_t normal_owned_bytes, uint64_t remaining_late_bytes) {
    if (normal_owned_bytes > free_bytes) return true;
    free_bytes -= normal_owned_bytes;
    if (remaining_late_bytes > free_bytes) return true;
    return free_bytes - remaining_late_bytes < (256ull << 20) + (64ull << 20);
}

// These weight images are used successively on the compute stream. The MMQ gate/up and down images coexist,
// as do the fallback ring's pairs; neither overlaps its own slice. The dense GEMM keeps its original capacity.
struct WeightScratchLayout {
    uint64_t bytes, group_down, fallback_down, fallback_pair;
};
constexpr WeightScratchLayout weight_scratch_layout(uint64_t dense_bytes, uint64_t group_gu, uint64_t group_d,
                                                     uint64_t fallback_gu, uint64_t fallback_d, size_t pairs) {
    const uint64_t group_down = group_gu ? buffer_bytes(group_gu) : 0;
    const uint64_t group_bytes = group_down + (group_d ? buffer_bytes(group_d) : 0);
    const uint64_t fallback_down = pairs ? buffer_bytes(fallback_gu) : 0;
    const uint64_t fallback_pair = fallback_down + (pairs ? buffer_bytes(fallback_d) : 0);
    return {std::max({dense_bytes, group_bytes, pairs * fallback_pair}), group_down, fallback_down, fallback_pair};
}

// Weight arena and workspace coexist; the KV stage reuses their complete extent later. The stage extent
// already includes its buffers' guards and alignment. Alloc::take adds the separate outer guard; no page rounding here.
struct SharedGemmStageLayout {
    uint64_t workspace_offset_bytes, gemm_extent_bytes, shared_payload_bytes;
};
constexpr SharedGemmStageLayout shared_gemm_stage_layout(uint64_t weight_arena_payload_bytes,
                                                        uint64_t workspace_payload_bytes, uint64_t stage_extent_bytes) {
    const uint64_t workspace_offset_bytes = buffer_bytes(weight_arena_payload_bytes);
    const uint64_t gemm_extent_bytes = workspace_offset_bytes + buffer_bytes(workspace_payload_bytes);
    return {workspace_offset_bytes, gemm_extent_bytes, std::max(gemm_extent_bytes, stage_extent_bytes)};
}

}  // namespace strata::prefill::detail
