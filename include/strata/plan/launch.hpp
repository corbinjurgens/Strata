#pragma once
#include "strata/prefill/prefill.hpp"

#include <cstdint>
#include <cstddef>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace strata::plan {

inline constexpr size_t launch_payload_limit = 1u << 20;
inline constexpr unsigned launch_schema = 2;
inline constexpr uint64_t launch_startup_scratch_allowance = 512ull << 20;

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
    bool no_prefill_borrow = false, reserve_given = false, per_layer = false, prefill_auto = false;
    bool price_only = false, budget_given = false;
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
    uint64_t cache_request = 0, environment = 0, file_cache_effective = 0, file_cache_compact = 0;
    uint64_t vram_required = 0, ram_required = 0, commit_required = 0;
    uint64_t stager_threads = 0, stager_ring = 0, stager_batch = 0;
    std::vector<uint64_t> layer_blobs, kv_stage_runs, resident_order;
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
    // Optional automatic-policy capacity threshold, not an additional allocated component.
    uint64_t minimum_policy_vram_free_bytes = 0;
};

/// Pure metadata pricing. Within-budget covers priced components only, never total runtime fit.
struct LaunchPrice {
    LaunchResolution resolution;
    bool configuration_complete = true;
    std::string status = "not_evaluated";
    uint64_t vram_shortfall = 0, ram_shortfall = 0, commit_shortfall = 0;
};

uint64_t launch_add(uint64_t a, uint64_t b);
uint64_t launch_mul(uint64_t a, uint64_t b);
uint64_t launch_align(uint64_t bytes, uint64_t alignment);
bool validate_launch_settings(const LaunchSettings&, std::string& error);
std::string encode_launch_settings(const LaunchSettings&);
/// Compact verified-settings handoff; aggregate cache layout identity, never a per-expert telemetry dump.
std::string encode_effective_launch(const LaunchSettings&);
std::string frame_effective_launch(const LaunchSettings&, uint64_t sequence);
// Applied owner values only; ordinary serving does not construct/validate a plan.
std::map<std::string, uint64_t> runtime_launch_settings(const LaunchSettings&, uint64_t cache_slots, uint64_t cache_bytes);
bool decode_launch_settings(const std::string& payload, LaunchSettings&, std::string& error);
std::string frame_launch_settings(const std::string& payload);
/// Consumes exactly one initial frame, leaving following engine commands unread.
bool read_launch_frame(std::istream&, std::string& payload, std::string& error);
/// Binary, unbuffered stdin handoff to the later raw-fd command reader.
bool read_launch_stdin(std::string& payload, std::string& error);
bool resolve_launch(const LaunchRequest&, const LaunchResources&, LaunchResolution&, std::string& error);
/// A null budget prices explicit settings only. Zero-valued budgets are explicit insufficiency.
/// Never discovers resources, mutates the environment, or emits an approved replay payload.
bool price_launch(const LaunchRequest&, const LaunchResources* budget, LaunchPrice&, std::string& error);
/// Reprice metadata and admit saved settings without selecting or resizing any settings.
bool admit_launch(const LaunchRequest&, const LaunchResources&, const LaunchSettings&, std::string& error);
/// Strict argv subset, shared with the metadata CLI. Rejects unsupported startup owners.
bool parse_launch_request(int argc, char** argv, LaunchRequest&, LaunchResources&, bool strict, std::string& error);
bool discover_launch_resources(LaunchResources&, std::string& error);
/// Existing engine process default, applied before the first CUDA discovery call.
void default_launch_environment();
/// Real pre-allocation startup boundary. The observation seam permits CPU-only admission/fault tests.
bool consume_strict_launch(int argc, char** argv, LaunchSettings&, std::string& error,
                           bool (*observe)(LaunchResources&, std::string&) = discover_launch_resources);
/// Equality is the final gate before publishing effective settings / READY.
bool check_effective_launch(const LaunchSettings& approved, const LaunchSettings& effective, std::string& error);
prefill::Prefill::Sizing launch_prefill_sizing(const LaunchSettings&);
int launch_metadata_main(int argc, char** argv);

}  // namespace strata::plan
