#pragma once
#include "strata/core/weight_ownership.hpp"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace strata::core {
class WeightTable;

// Experimental GDN/QSA/shared-expert projection overrides. Upload unchanged native GGUF
// blocks once, then attach them to the matching canonical WeightRef. Unsupported
// types retain their canonical paths. Native matrices and their shared Q8_1
// scratch occupy one aligned device allocation, avoiding per-matrix page tails.
// These projections share scratch, so use one ordered session stream and keep this object
// alive until all graphs that reference it have been destroyed and synchronized.
class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    /// With layer_hi >= 0, only the `blk.<l>.` matrices with layer_lo <= l < layer_hi are uploaded (a layer
    /// split's stage holds its own layers' projections, not the whole model's); the others keep data == nullptr.
    /// track_ownership confirms upload completion and records logical source spans for planned startup.
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false, int64_t layer_lo = 0, int64_t layer_hi = -1,
              bool track_ownership = false);
    /// Plan v0.3 P1: the canonical tensor names `load` would serve natively from these shards (eligible name,
    /// supported type, 2-D), read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_names(const std::vector<std::string>& shards, bool include_ple_key,
                             std::set<std::string>& out, std::string& err);
    /// Layer split: load only blocks [lb, le) (every other `blk.N.` projection belongs to another GPU's stage; the
    /// PLE tensors are loaded everywhere).  Process-wide, read by the next `load`; (-1, -1) = all layers.
    static void set_layer_range(int lb, int le);
    /// Layer split: the device bytes `load` with set_layer_range(lb, le) would allocate, retaining PLE on every stage:
    /// the same filters, aligned slices and shared scratch in one granule-rounded arena, reading GGUF headers and `table`
    /// only, so it allocates nothing and needs no device (#1238).  `table` must hold every tensor the shards carry.
    static bool weight_bytes_for(const std::vector<std::string>& shards, WeightTable& table, bool include_ple_key,
                                 int64_t lb, int64_t le, uint64_t& out, std::string& err);
    /// #326: a native pack whose `blk.1.ple_key.weight` row is unquantized (iq_pack --compat-bf16 of a GGUF key
    /// the native kernel also reads, e.g. OrcaRouter's IQ3_XXS) serves the PLE from that row, so it is taken out
    /// of `skip` and `load` does not upload the GGUF key over it.  A quantized row leaves `skip` unchanged.
    static bool keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip, std::string& err);
    uint64_t weight_bytes() const { return bytes_; }
    /// The arena's granule-rounded footprint plus optional HC experiment allocations.
    uint64_t allocated_bytes() const { return allocated_; }
    size_t tensor_count() const { return tensor_count_; }
    const WeightOwnership& weight_ownership() const { return ownership_; }

private:
    void* arena_ = nullptr;
    std::vector<void*> weights_;   // optional HC experiment allocations
    std::vector<const void*> packed_keys_;   // GGUF-layout pointers registered with STRATA_Q8_PACKED=1
    size_t tensor_count_ = 0;
    uint64_t bytes_ = 0;
    uint64_t allocated_ = 0;
    WeightOwnership ownership_;
};
} // namespace strata::core
