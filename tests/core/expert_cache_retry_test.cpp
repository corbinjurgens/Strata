// Automatic startup retry control on injected operations: no model, toolkit or GPU.
#include "strata/core/expert_cache_retry.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {
using strata::core::ExpertCacheRetryStatus;
constexpr int64_t mib = 1ll << 20;
int checks = 0;
int failures = 0;

void check(bool ok, const char* what) {
    ++checks;
    std::printf("  %-82s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

struct Operations {
    std::vector<int64_t> readings{1024 * mib};
    std::vector<int> failed_opens;
    int fail_touch = 0;
    int fail_sync = 0;
    int fail_query = 0;
    bool allocation_shrink_ok = true;
    bool reserve_shrink_ok = true;
    int opens = 0, touches = 0, syncs = 0, queries = 0, allocation_shrinks = 0, reserve_shrinks = 0;
    int64_t last_free = -1, last_reserve = -1;
    std::string err, order;

    strata::core::ExpertCacheRetryResult run(bool automatic = true, int64_t reserve = 512 * mib, bool strict = false) {
        return strata::core::retry_expert_cache(automatic, reserve, err,
            [&](std::string& why) {
                order += 'O';
                ++opens;
                if (std::find(failed_opens.begin(), failed_opens.end(), opens) == failed_opens.end()) return true;
                why = "injected allocation failure";
                return false;
            },
            [&](std::string& why) {
                order += 'T';
                if (++touches != fail_touch) return true;
                why = "cudaMemset failed: injected touch error";
                return false;
            },
            [&](std::string& why) {
                order += 'S';
                if (++syncs != fail_sync) return true;
                why = "cudaDeviceSynchronize failed: injected synchronization error";
                return false;
            },
            [&](int64_t& free_b, std::string& why) -> bool {
                order += 'Q';
                const int nth = ++queries;
                if (nth == fail_query) {
                    why = "cudaMemGetInfo failed: injected query error";
                    return false;
                }
                const size_t index = (size_t) (nth - 1);
                if (index < readings.size()) {
                    free_b = readings[index];
                    return true;
                }
                check(false, "each queried free-memory reading was explicitly supplied");
                free_b = 0;
                return true;
            },
            [&]() {
                order += 'A';
                ++allocation_shrinks;
                return allocation_shrink_ok;
            },
            [&](int64_t available, int64_t reserve_bytes) {
                order += 'R';
                ++reserve_shrinks;
                last_free = available;
                last_reserve = reserve_bytes;
                return reserve_shrink_ok;
            }, strict);
    }
};
} // namespace

int main() {
    std::printf("expert_cache_retry_test\n");
    for (bool automatic : {false, true}) {
        Operations ok;
        auto result=ok.run(automatic,512*mib,true);
        check(result.status==ExpertCacheRetryStatus::ready && ok.order=="OTSQ", "strict manual and auto validate the actual touched candidate");
        Operations allocation;
        allocation.failed_opens={1};
        result=allocation.run(automatic,512*mib,true);
        check(result.status==ExpertCacheRetryStatus::allocation_failed && allocation.order=="O" && result.allocation_retries==0,
              "strict allocation failure never calls shrink or retries");
        Operations low;
        low.readings={511*mib};
        result=low.run(automatic,512*mib,true);
        check(result.status==ExpertCacheRetryStatus::validation_failed && low.order=="OTSQ" && low.reserve_shrinks==0,
              "strict post-touch reserve refuses even inside ordinary 64 MiB tolerance");
        for (int fault=0;fault<3;++fault) {
            Operations op;
            op.fail_touch=fault==0; op.fail_sync=fault==1; op.fail_query=fault==2;
            result=op.run(automatic,512*mib,true);
            check(result.status==ExpertCacheRetryStatus::validation_failed && op.opens==1 &&
                  op.allocation_shrinks==0 && op.reserve_shrinks==0, "strict touch/sync/query failure cannot resize");
        }
    }
    {
        Operations op;
        const auto result = op.run();
        check(result.status == ExpertCacheRetryStatus::ready && result.allocation_retries == 0,
              "first candidate passes with no allocation retries");
        check(op.order == "OTSQ" && op.opens == 1 && op.queries == 1,
              "first candidate is opened, touched, synchronized and queried before acceptance");
    }
    {
        Operations op;
        op.readings = {400 * mib, 400 * mib, 400 * mib, 400 * mib, 400 * mib, 400 * mib, 448 * mib};
        const auto result = op.run();
        check(result.status == ExpertCacheRetryStatus::ready && op.opens == 7 && op.touches == 7 &&
              op.syncs == 7 && op.queries == 7 && op.reserve_shrinks == 6,
              "six low candidates then a passing seventh validate all seven candidates");
        check(op.order == "OTSQROTSQROTSQROTSQROTSQROTSQROTSQ",
              "every reserve retry follows touch, synchronization and query");
    }
    {
        Operations op;
        op.readings = {400 * mib, 400 * mib, 400 * mib, 400 * mib, 400 * mib, 400 * mib, 400 * mib};
        const auto result = op.run();
        check(result.status == ExpertCacheRetryStatus::validation_failed && op.opens == 7 &&
              op.touches == 7 && op.syncs == 7 && op.queries == 7 && op.reserve_shrinks == 6,
              "seven low candidates fail after seven validations without an eighth open");
        check(op.err == "expert cache auto: reserve still insufficient after 7 candidates; bounded retries exhausted",
              "bounded-retry exhaustion identifies the failed reserve validation");
    }
    {
        Operations op;
        op.fail_touch = 1;
        const auto result = op.run();
        check(result.status == ExpertCacheRetryStatus::validation_failed && op.order == "OT" &&
              op.syncs == 0 && op.queries == 0 && op.reserve_shrinks == 0,
              "touch failure stops before synchronization, query or shrinking");
        check(op.err == "cudaMemset failed: injected touch error", "touch operation and error are retained");
    }
    {
        Operations op;
        op.fail_sync = 1;
        const auto result = op.run();
        check(result.status == ExpertCacheRetryStatus::validation_failed && op.order == "OTS" &&
              op.queries == 0 && op.reserve_shrinks == 0,
              "synchronization failure stops before query or shrinking");
        check(op.err == "cudaDeviceSynchronize failed: injected synchronization error",
              "synchronization operation and error are retained");
    }
    {
        Operations op;
        op.fail_query = 1;
        const auto result = op.run();
        check(result.status == ExpertCacheRetryStatus::validation_failed && result.allocation_retries == 0 &&
              op.order == "OTSQ" && op.opens == 1 && op.touches == 1 && op.syncs == 1 && op.queries == 1 &&
              op.reserve_shrinks == 0 && op.allocation_shrinks == 0,
              "failed query at an ordinary reserve fails validation without shrink, reopen or retry");
        check(op.err == "cudaMemGetInfo failed: injected query error",
              "free-memory query failure retains its error text");
    }
    {
        Operations op;
        op.fail_query = 1;
        const auto result = op.run(true, 64 * mib);
        check(result.status == ExpertCacheRetryStatus::validation_failed && op.order == "OTSQ" &&
              op.opens == 1 && op.queries == 1 && op.reserve_shrinks == 0,
              "failed query at a low reserve fails validation instead of accepting the candidate");
        check(op.err == "cudaMemGetInfo failed: injected query error",
              "low-reserve free-memory query failure retains its error text");
    }
    {
        Operations op;
        op.readings = {0};
        const auto result = op.run(true, 64 * mib);
        check(result.status == ExpertCacheRetryStatus::ready && op.order == "OTSQ" && op.queries == 1 &&
              op.reserve_shrinks == 0,
              "a successful zero-byte reading at a low reserve stays a real reading, not a failure");
    }
    {
        Operations op;
        op.fail_query = 2;
        op.readings = {400 * mib};
        const auto result = op.run();
        check(result.status == ExpertCacheRetryStatus::validation_failed && op.order == "OTSQROTSQ" &&
              op.opens == 2 && op.queries == 2 && op.reserve_shrinks == 1 && op.allocation_shrinks == 0,
              "query failure after a reserve shrink stops the loop with no further shrink or reopen");
        check(op.err == "cudaMemGetInfo failed: injected query error",
              "mid-retry free-memory query failure retains its error text");
    }
    {
        Operations op;
        op.failed_opens = {1, 2, 3, 4, 5, 6, 7, 8};
        op.readings = {400 * mib, 400 * mib, 400 * mib, 400 * mib, 400 * mib, 400 * mib, 448 * mib};
        const auto result = op.run();
        check(result.status == ExpertCacheRetryStatus::ready && result.allocation_retries == 8 &&
              op.opens == 15 && op.allocation_shrinks == 8 && op.reserve_shrinks == 6 &&
              op.touches == 7 && op.syncs == 7 && op.queries == 7,
              "eight allocation retries preserve the full seven successful-candidate allowance");
    }
    {
        Operations op;
        op.failed_opens = {1, 4, 7};
        op.readings = {400 * mib, 400 * mib, 400 * mib, 400 * mib, 400 * mib, 400 * mib, 448 * mib};
        const auto result = op.run();
        check(result.status == ExpertCacheRetryStatus::ready && result.allocation_retries == 3 &&
              op.opens == 10 && op.allocation_shrinks == 3 && op.reserve_shrinks == 6 && op.queries == 7,
              "interleaved allocation failures do not consume reserve-validation retries");
    }
    {
        Operations op;
        op.failed_opens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        const auto result = op.run();
        check(result.status == ExpertCacheRetryStatus::allocation_failed && result.allocation_retries == 8 &&
              op.opens == 9 && op.allocation_shrinks == 8 && op.touches == 0 && op.queries == 0,
              "ninth allocation failure stops at the independent eight-retry bound");
        check(op.err == "injected allocation failure", "terminal allocation failure retains its error");
    }
    {
        Operations op;
        op.readings = {448 * mib};
        const auto result = op.run();
        check(result.status == ExpertCacheRetryStatus::ready && op.queries == 1 && op.reserve_shrinks == 0,
              "512 MiB reserve accepts exactly 448 MiB free at the 64 MiB tolerance boundary");
    }
    {
        Operations op;
        op.readings = {448 * mib - 1, 448 * mib};
        const auto result = op.run();
        check(result.status == ExpertCacheRetryStatus::ready && op.opens == 2 && op.queries == 2 &&
              op.reserve_shrinks == 1 && op.last_free == 448 * mib - 1 && op.last_reserve == 512 * mib,
              "one byte below the tolerance boundary requires a smaller candidate");
    }
    {
        Operations op;
        op.readings = {576 * mib - 1, 576 * mib};
        const auto result = op.run(true, 640 * mib);
        check(result.status == ExpertCacheRetryStatus::ready && op.queries == 2 && op.reserve_shrinks == 1 &&
              op.last_free == 576 * mib - 1 && op.last_reserve == 640 * mib,
              "reserve validation includes an additional 128 MiB of late buffers");
    }
    {
        Operations op;
        op.readings = {0};
        op.reserve_shrink_ok = false;
        const auto result = op.run();
        check(result.status == ExpertCacheRetryStatus::disabled && op.order == "OTSQR" && op.opens == 1,
              "reserve shrinking that leaves no slots disables the cache without reopening");
    }
    {
        Operations op;
        op.failed_opens = {1};
        op.allocation_shrink_ok = false;
        const auto result = op.run();
        check(result.status == ExpertCacheRetryStatus::allocation_failed && result.allocation_retries == 0 &&
              op.order == "OA" && op.opens == 1,
              "allocation shrinking that leaves no slots stops without consuming a retry");
    }
    {
        Operations op;
        const auto result = op.run(false);
        check(result.status == ExpertCacheRetryStatus::ready && op.order == "O" && op.touches == 0 &&
              op.syncs == 0 && op.queries == 0,
              "explicit cache size retains its early exit without automatic validation");
    }
    {
        Operations op;
        op.failed_opens = {1};
        const auto result = op.run(false);
        check(result.status == ExpertCacheRetryStatus::allocation_failed && result.allocation_retries == 0 &&
              op.order == "O" && op.allocation_shrinks == 0,
              "explicit cache allocation failure does not shrink or retry");
    }
    std::printf("%d/%d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
