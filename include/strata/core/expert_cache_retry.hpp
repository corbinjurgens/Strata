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
// The free-memory query writes its reading into `available` and names the failing query in err; a failed query
// fails validation before any readiness arithmetic, so it can never pass as free memory (not even as a zero
// reading at a low reserve) and never triggers reserve shrinking.
template <class Open, class Touch, class Synchronize, class FreeBytes, class ShrinkAllocation, class ShrinkReserve>
ExpertCacheRetryResult retry_expert_cache(bool automatic, int64_t reserve_bytes, std::string& err,
                                        Open open, Touch touch, Synchronize synchronize, FreeBytes free_bytes,
                                        ShrinkAllocation shrink_allocation, ShrinkReserve shrink_reserve,
                                        bool strict = false) {
    constexpr int max_allocation_retries = 8;
    constexpr int max_candidates = 7;
    int allocation_retries = 0;
    int candidates = 0;
    for (;;) {
        if (!open(err)) {
            if (!strict && automatic && allocation_retries < max_allocation_retries && shrink_allocation()) {
                ++allocation_retries;
                continue;
            }
            return {ExpertCacheRetryStatus::allocation_failed, allocation_retries};
        }
        if (!automatic && !strict) return {ExpertCacheRetryStatus::ready, allocation_retries};
        ++candidates;
        if (!touch(err) || !synchronize(err))
            return {ExpertCacheRetryStatus::validation_failed, allocation_retries};
        // Consume the query's success before the readiness arithmetic: a failed query stops here with its own
        // error text, instead of being read as a low (or zero) free figure that a small reserve would accept
        // or a big reserve would send into reserve shrinking.
        int64_t available = 0;
        if (!free_bytes(available, err))
            return {ExpertCacheRetryStatus::validation_failed, allocation_retries};
        if (available >= reserve_bytes - (strict ? 0 : (64ll << 20)))
            return {ExpertCacheRetryStatus::ready, allocation_retries};
        if (strict) {
            err = "insufficient_resources: strict expert cache cannot retain the approved reserve and later buffers";
            return {ExpertCacheRetryStatus::validation_failed, allocation_retries};
        }
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
