#pragma once

#include <cstdint>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace strata::core {

// Attribution only: the allocation, upload and cache owners retain resource lifetime.
// An empty shard means the source-to-model mapping has not been proved (canonical packs).
struct WeightSlice {
    uint64_t offset = 0, bytes = 0;
    std::string shard;
    uint64_t source_offset = 0;
};

struct WeightOwnership {
    uint64_t capacity = 0, scratch = 0, shared = 0, mapped_host = 0;
    std::vector<WeightSlice> slices;
    bool known = true;
};

struct ShardHolding {
    uint64_t bytes = 0, source_unique = 0;
    bool operator==(const ShardHolding&) const = default;
};
struct WeightSnapshot {
    bool known = true;
    uint64_t capacity = 0, logical = 0, padding = 0, scratch = 0, shared = 0, unattributed = 0, mapped_host = 0;
    std::vector<ShardHolding> shards;
    bool operator==(const WeightSnapshot&) const = default;
};

inline constexpr size_t ownership_shard_limit = 64;
inline constexpr size_t ownership_slice_limit = 1u << 20;
inline constexpr size_t ownership_frame_limit = 16384;

// Validates disjoint destination slices, checked sums and source spans. Unknown
// owners suppress all numeric results; they must never look like a successful zero.
bool aggregate_weight_ownership(const std::vector<const WeightOwnership*>& owners,
                                const std::vector<std::string>& shards, WeightSnapshot& out, std::string& error);
std::string weight_shard_identity(const std::string& path);

// Transport for the existing captured-process stdout pump (strict launch only).
// ASCII header STRATA-OWNERSHIP/1\n, decimal sequence\n, decimal payload bytes\n,
// exactly that payload, then \nSTRATA-OWNERSHIP-END/1 <sequence>\n. Sequence is
// process scoped, starts at 1, never wraps. Entire frame <= 16384 bytes.
// Payload is ASCII lines (tab-separated); string fields are lowercase UTF-8 hex:
// device <backend-hex> <ordinal> <identity-hex>
// state known|unknown
// physical unavailable
// totals <capacity> <logical> <padding> <scratch> <shared> <unattributed> <mapped_host>
// shards <count>
// shard <index> <held> <unique-source> <duplicate-copy>
// Unknown frames omit totals and shard records, with shards 0. Every known frame
// includes every shard, including zero holdings, in native argv shard order (<=64).
// capacity covers reported weight allocations only, not all process allocations;
// logical includes duplicate copies; padding is capacity-logical-scratch-shared.
// shared is scratch reused by multiple tensors/roles (including cache loans).
// mapped_host is separately owned host storage, never dedicated device capacity.
// Shard indices map to --native's gguf_split_paths order; no filenames are emitted.
bool frame_weight_ownership(const WeightSnapshot&, const std::string& backend, unsigned device,
                            const std::string& identity, uint64_t& sequence, std::string& frame, std::string& error);
// Envelope reader for CPU protocol tests and native consumers. Leaves subsequent
// command/output bytes unread. It rejects truncation, oversized frames and bad ends.
bool read_weight_ownership_frame(std::istream&, uint64_t& sequence, std::string& payload, std::string& error);

// Current owned storage, recorded only after successful allocation/initialization.
// rounded is a comparable allocator-granule model, never driver usage or residency.
// Host mapped/pinned are subsets of requested, not extra allocations. Allowances
// are policy budgets and must not be added to measured storage.
// logical is useful payload/capacity: confirmed tensor contents for weight/cache
// owners, usable buffer/carve bytes for scratch, and zero for file views (mapped
// holds their logical address span). It excludes allocator-granule rounding.
struct RuntimeMemory {
    bool known = true;
    uint64_t requested = 0, rounded = 0, logical = 0, mapped = 0, pinned = 0, allowance = 0;
    void device_allocation(uint64_t bytes);
    void merge(const RuntimeMemory&);
    bool operator==(const RuntimeMemory&) const = default;
};
struct RuntimeSnapshot {
    std::map<std::string, uint64_t> settings;
    std::map<std::string, RuntimeMemory> vram, ram;
    WeightSnapshot weights;
    bool operator==(const RuntimeSnapshot&) const = default;
};
// One complete snapshot on the existing stdout owner. Same bounded envelope and
// commit-on-complete sequence rules as ownership, with STRATA-RUNTIME/1 and
// STRATA-RUNTIME-END/1 markers. Payload: device/state/physical/totals/shards/shard
// as above, then setting <name> <uint64> and component <vram|ram> <name>
// <known|unknown> <requested> <rounding> <logical> <mapped> <pinned> <allowance>.
// All quantities are bytes except settings. Unknown component numeric fields
// are '-'. total_allocated and driver_os are always unknown. Names are ASCII
// [a-z_]+; no estimates, paths or payload fingerprints enter this protocol.
bool frame_runtime_snapshot(const RuntimeSnapshot&, const std::string& backend, unsigned device,
                            const std::string& identity, uint64_t& sequence, std::string& frame, std::string& error);
bool read_runtime_frame(std::istream&, uint64_t& sequence, std::string& payload, std::string& error);

// Preserve a proved typed cause at the beginning of a wrapper's actual message.
// Untyped allocation/device failures remain untyped; callers prove OOM from the
// backend result, not from an allocation function name or free-memory guess.
std::string native_error_message(const std::string& context, const std::string& cause, const std::string& fallback_code = {});

} // namespace strata::core
