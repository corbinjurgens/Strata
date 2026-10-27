#pragma once

#include <cstdint>
#include <cstddef>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace strata::plan {

inline constexpr size_t launch_payload_limit = 1u << 20;

struct LaunchResources {
    uint64_t vram_free = 0, ram_available = 0, commit_available = 0;
    unsigned devices = 1;
    bool cuda = true;
};

struct LaunchRequest {
    std::string pack, profile;
    std::vector<std::string> shards;
    uint64_t context = 0, prefill = 0, spec = 2, kv = 1, kv_resident = 0;
    int64_t expert_cache = -1;
    uint64_t resident_budget = 0, file_cache = 0, host_headroom = 0, vram_reserve = 700ull << 20;
    bool no_prefill_borrow = false, reserve_given = false, per_layer = false;
};

struct LaunchSlot {
    uint64_t layer = 0, expert = 0, bytes = 0, offset = 0;
    bool operator==(const LaunchSlot&) const = default;
};

/// Native settings only. Estimates and public plan JSON deliberately have separate owners.
struct LaunchSettings {
    uint64_t context = 0, prefill = 0, spec = 0, kv = 0, kv_resident = 0;
    uint64_t batch = 0, batch_groups = 1, parallel = 1;
    uint64_t auto_cache = 0, per_layer = 0, no_prefill_borrow = 1;
    uint64_t resident_budget = 0, resident_bytes = 0, resident_experts = 0, resident_pin = 1;
    uint64_t file_cache = 0, host_headroom = 0, vram_reserve = 0;
    uint64_t shared_stage = 0, ring = 0, workspace = 0, weight_scratch = 0, common = 0, region = 0;
    uint64_t prefill_mmq = 0, prefill_stage_own = 0, prefill_reserve_bo = 0, prefill_ring_enabled = 0;
    uint64_t experts = 0, top_k = 0, vocab = 0, max_blob = 0;
    std::vector<uint64_t> layer_blobs, kv_stage_runs;
    std::vector<LaunchSlot> slots;
    bool operator==(const LaunchSettings&) const = default;
};

struct LaunchResolution {
    LaunchSettings settings;
    std::map<std::string, uint64_t> vram, ram;
    uint64_t vram_required = 0, ram_required = 0, commit_required = 0;
    // Unknown runtime/library/OS costs are not invented measurements. Reserve is an admission allowance.
    std::string unknown_costs = "CUDA context/modules, graph capture, CPU TLS/request scratch, driver and OS residency";
    std::string payload;
};

uint64_t launch_add(uint64_t a, uint64_t b);
uint64_t launch_mul(uint64_t a, uint64_t b);
uint64_t launch_align(uint64_t bytes, uint64_t alignment);
bool validate_launch_settings(const LaunchSettings&, std::string& error);
std::string encode_launch_settings(const LaunchSettings&);
bool decode_launch_settings(const std::string& payload, LaunchSettings&, std::string& error);
std::string frame_launch_settings(const std::string& payload);
/// Consumes exactly one initial frame, leaving following engine commands unread.
bool read_launch_frame(std::istream&, std::string& payload, std::string& error);
bool resolve_launch(const LaunchRequest&, const LaunchResources&, LaunchResolution&, std::string& error);
/// Early metadata CLI; no strict runtime consumer is advertised by this unit.
int launch_metadata_main(int argc, char** argv);

}  // namespace strata::plan
