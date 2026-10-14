#pragma once

#include <cstdint>
#include <string>

namespace strata::core {

enum class ExpertCacheRetryStatus { ready, disabled, allocation_failed, validation_failed };

struct ExpertCacheRetryResult {
    ExpertCacheRetryStatus status;
    int allocation_retries;
};

// Startup retry control only: storage, CUDA operations and the existing shrink policy stay with the caller.
// Open, touch and synchronize return false on failure; touch and synchronize name the failing operation in err.
template <class Open, class Touch, class Synchronize, class FreeBytes, class ShrinkAllocation, class ShrinkReserve>
ExpertCacheRetryResult retry_expert_cache(bool automatic, int64_t reserve_bytes, std::string& err,
                                        Open open, Touch touch, Synchronize synchronize, FreeBytes free_bytes,
                                        ShrinkAllocation shrink_allocation, ShrinkReserve shrink_reserve) {
    constexpr int max_allocation_retries = 8;
    constexpr int max_candidates = 7;
    int allocation_retries = 0;
    int candidates = 0;
    for (;;) {
        if (!open(err)) {
            if (automatic && allocation_retries < max_allocation_retries && shrink_allocation()) {
                ++allocation_retries;
                continue;
            }
            return {ExpertCacheRetryStatus::allocation_failed, allocation_retries};
        }
        if (!automatic) return {ExpertCacheRetryStatus::ready, allocation_retries};
        ++candidates;
        if (!touch(err) || !synchronize(err))
            return {ExpertCacheRetryStatus::validation_failed, allocation_retries};
        const int64_t available = free_bytes();
        if (available >= reserve_bytes - (64ll << 20))
            return {ExpertCacheRetryStatus::ready, allocation_retries};
        if (candidates == max_candidates) {
            err = "expert cache auto: reserve still insufficient after " + std::to_string(max_candidates) +
                  " candidates; bounded retries exhausted";
            return {ExpertCacheRetryStatus::validation_failed, allocation_retries};
        }
        if (!shrink_reserve(available, reserve_bytes))
            return {ExpertCacheRetryStatus::disabled, allocation_retries};
    }
}

} // namespace strata::core
