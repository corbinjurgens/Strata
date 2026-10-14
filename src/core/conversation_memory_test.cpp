#include "strata/core/conversation_memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}

int main() {
    for (const char* text : {"", "MemFree: 100 kB\n", "MemAvailable: -1 kB\n",
                            "MemAvailable: +1 kB\n", "MemAvailable: 1 MB\n",
                            "MemAvailable: 1\n", "MemAvailable: 1x kB\n",
                            "MemAvailable: 18446744073709551615 kB\n",
                            "MemAvailable: 18446744073709551616 kB\n",
                            "MemAvailable: 1 kB trailing\n",
                            "MemAvailable: 1 kB\nMemAvailable: 2 kB\n"}) {
        std::istringstream input(text);
        check(!conversation_mem_available(input), "missing/malformed telemetry is unknown");
    }
    std::istringstream normal("MemTotal: 999999 kB\nMemAvailable:    12345 kB\nSwapFree: 777 kB\n");
    check(conversation_mem_available(normal) == 12345ULL * 1024, "only MemAvailable is counted");
    std::istringstream zero("MemAvailable: 0 kB");
    check(conversation_mem_available(zero) == 0, "zero available memory is known");
    std::istringstream broken("MemAvailable: 123 kB\n");
    broken.setstate(std::ios::badbit);
    check(!conversation_mem_available(broken), "I/O failure declines admission");
    check(!conversation_memory_admit({}, 0, 0), "unknown fails closed even with zero floor");
    check(conversation_memory_admit(100, 40, 60), "exact allocation plus floor fits");
    check(!conversation_memory_admit(99, 40, 60), "one byte below required memory rejected");
    check(!conversation_memory_admit(59, 0, 60), "floor subtraction cannot underflow");
    check(conversation_memory_admit(60, 0, 60), "post-capture floor check");
    check(!conversation_memory_admit(100, std::numeric_limits<uint64_t>::max(), 1), "allocation arithmetic cannot overflow");
    check(conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                   std::numeric_limits<uint64_t>::max(), 0), "maximal exact bound");
    check(!conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                    std::numeric_limits<uint64_t>::max(), 1), "maximal sum overflow rejected");
    // A snapshot may fit in free RAM but leave too little Windows commit for the floor.
    // Use deterministic available-memory samples; do not create pressure on the test machine.
    constexpr uint64_t gib = 1ULL << 30;
    check(conversation_memory_admit(16 * gib, 2 * gib, 4 * gib), "RAM alone would admit the snapshot");
    check(!conversation_memory_admit(5 * gib, 2 * gib, 4 * gib), "commit-limited sample rejects the snapshot");
    check(conversation_memory_admit(6 * gib, 2 * gib, 4 * gib), "commit-limited exact fit preserves the floor");
    check(!conversation_memory_admit(0, 0, 4 * gib), "exhausted commit rejects a post-capture floor check");
    // Test the real provider without assuming any particular amount of free RAM.
    const auto available = conversation_available_memory();
    check(!available || conversation_memory_admit(available, 0, 0), "provider returns bytes or unknown");
#if defined(_WIN32)
    // Unknown must fail closed in production, but must not let a broken Windows provider pass this test.
    // Compare with total limits, not a second available reading: other processes can allocate
    // between calls. No pressure allocation or fixed free-memory assumption is needed.
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof status;
    check(GlobalMemoryStatusEx(&status) != 0, "Windows memory-status API is available");
    check(available.has_value() && *available <= status.ullTotalPhys && *available <= status.ullTotalPageFile,
          "Windows provider returns a known sample bounded by physical memory and commit");
#endif
    std::printf("conversation_memory_test: %d checks passed\n", checks);
}
