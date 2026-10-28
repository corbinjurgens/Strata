#include "strata/core/weight_ownership.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <istream>
#include <limits>
#include <map>
#include <sstream>
#include <set>
#include <stdexcept>

namespace strata::core {
namespace {
uint64_t add(uint64_t a, uint64_t b) {
    if (b > (std::numeric_limits<uint64_t>::max)() - a) throw std::overflow_error("ownership byte overflow");
    return a + b;
}
std::string hex(const std::string& s) {
    if (s.size() > 256) throw std::runtime_error("ownership identity too long");
    constexpr char digits[] = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) { out += digits[c >> 4]; out += digits[c & 15]; }
    return out;
}
bool number(const std::string& s, uint64_t& n) {
    if (s.empty() || s.size() > 20 || (s.size() > 1 && s[0] == '0')) return false;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), n);
    return r.ec == std::errc{} && r.ptr == s.data() + s.size();
}
bool line(std::istream& in, std::string& s, size_t limit) {
    s.clear(); char c;
    while (in.get(c)) {
        if (c == '\n') return true;
        if (s.size() >= limit || c == '\r') return false;
        s += c;
    }
    return false;
}
}

std::string weight_shard_identity(const std::string& path) {
    auto s = std::filesystem::absolute(std::filesystem::path(path)).lexically_normal().generic_string();
#if defined(_WIN32)
    for (auto& c : s) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
#endif
    return s;
}

bool aggregate_weight_ownership(const std::vector<const WeightOwnership*>& owners,
                                const std::vector<std::string>& shards, WeightSnapshot& out, std::string& error) {
    out = {};
    try {
        if (shards.size() > ownership_shard_limit || owners.size() > 1024) throw std::runtime_error("ownership bounds");
        std::map<std::string, size_t> index;
        for (size_t i = 0; i < shards.size(); ++i)
            if (!index.emplace(weight_shard_identity(shards[i]), i).second) throw std::runtime_error("duplicate ownership shard");
        WeightSnapshot s;
        s.shards.resize(shards.size());
        std::vector<std::vector<std::pair<uint64_t, uint64_t>>> sources(shards.size());
        size_t count = 0;
        std::set<const WeightOwnership*> seen;
        for (const auto* owner : owners) {
            if (!owner) throw std::runtime_error("null ownership owner");
            if (!seen.insert(owner).second) throw std::runtime_error("duplicate ownership owner");
            if (!owner->known) { out.known = false; return true; }
            if (owner->slices.size() > ownership_slice_limit - count) throw std::runtime_error("ownership slice bound");
            count += owner->slices.size();
            s.capacity = add(s.capacity, owner->capacity);
            s.scratch = add(s.scratch, owner->scratch);
            s.shared = add(s.shared, owner->shared);
            s.mapped_host = add(s.mapped_host, owner->mapped_host);
            uint64_t logical = 0, end = 0;
            auto slices = owner->slices;
            std::sort(slices.begin(), slices.end(), [](const auto& a, const auto& b) { return a.offset < b.offset; });
            for (const auto& slice : slices) {
                if (!slice.bytes || slice.offset < end || slice.offset > owner->capacity || slice.bytes > owner->capacity - slice.offset)
                    throw std::runtime_error("overlapping or out-of-bounds ownership slice");
                end = add(slice.offset, slice.bytes);
                logical = add(logical, slice.bytes);
                if (!slice.shard.empty()) (void)add(slice.source_offset, slice.bytes);
                auto it = slice.shard.empty() ? index.end() : index.find(weight_shard_identity(slice.shard));
                if (it == index.end()) s.unattributed = add(s.unattributed, slice.bytes);
                else {
                    auto& h = s.shards[it->second];
                    h.bytes = add(h.bytes, slice.bytes);
                    sources[it->second].emplace_back(slice.source_offset, add(slice.source_offset, slice.bytes));
                }
            }
            if (logical > owner->capacity || add(owner->scratch, owner->shared) > owner->capacity - logical)
                throw std::runtime_error("ownership exceeds allocation capacity");
            s.logical = add(s.logical, logical);
        }
        s.padding = s.capacity - s.logical - s.scratch - s.shared;
        for (size_t i = 0; i < sources.size(); ++i) {
            auto& ranges = sources[i];
            std::sort(ranges.begin(), ranges.end());
            uint64_t end = 0;
            for (const auto& r : ranges) {
                if (r.second > end) s.shards[i].source_unique = add(s.shards[i].source_unique, r.second - (std::max)(end, r.first));
                end = (std::max)(end, r.second);
            }
        }
        out = std::move(s);
        return true;
    } catch (const std::exception& e) { error = e.what(); out = {}; out.known = false; return false; }
}

bool frame_weight_ownership(const WeightSnapshot& s, const std::string& backend, unsigned device,
                            const std::string& identity, uint64_t& sequence, std::string& frame, std::string& error) {
    frame.clear();
    try {
        if (sequence == (std::numeric_limits<uint64_t>::max)() || s.shards.size() > ownership_shard_limit)
            throw std::runtime_error("ownership sequence or shard bound exhausted");
        if (backend.empty() || identity.empty()) throw std::runtime_error("ownership device identity unavailable");
        std::ostringstream p;
        p << "device\t" << hex(backend) << '\t' << device << '\t' << hex(identity) << '\n'
          << "state\t" << (s.known ? "known" : "unknown") << "\nphysical\tunavailable\n";
        if (s.known) {
            uint64_t attributed = s.unattributed;
            for (const auto& h : s.shards) attributed = add(attributed, h.bytes);
            if (attributed != s.logical || add(add(s.logical, s.padding), add(s.scratch, s.shared)) != s.capacity)
                throw std::runtime_error("ownership totals do not reconcile");
            p << "totals\t" << s.capacity << '\t' << s.logical << '\t' << s.padding << '\t' << s.scratch << '\t'
              << s.shared << '\t' << s.unattributed << '\t' << s.mapped_host << '\n';
        }
        p << "shards\t" << (s.known ? s.shards.size() : 0) << '\n';
        if (s.known) for (size_t i = 0; i < s.shards.size(); ++i) {
            const auto& h = s.shards[i];
            if (h.source_unique > h.bytes) throw std::runtime_error("ownership source exceeds held bytes");
            p << "shard\t" << i << '\t' << h.bytes << '\t' << h.source_unique << '\t' << h.bytes - h.source_unique << '\n';
        }
        const auto payload = p.str(), seq = std::to_string(sequence + 1);
        auto candidate = "STRATA-OWNERSHIP/1\n" + seq + '\n' + std::to_string(payload.size()) + '\n' + payload +
                         "\nSTRATA-OWNERSHIP-END/1 " + seq + '\n';
        if (candidate.size() > ownership_frame_limit) throw std::runtime_error("ownership frame exceeds bound");
        frame = std::move(candidate);
        ++sequence;
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

bool read_weight_ownership_frame(std::istream& in, uint64_t& sequence, std::string& payload, std::string& error) {
    std::string magic, seq, len, end, body;
    uint64_t n = 0, next = 0;
    if (!line(in, magic, 32) || magic != "STRATA-OWNERSHIP/1" || !line(in, seq, 20) || !number(seq, next) ||
        !next || next <= sequence || !line(in, len, 20) || !number(len, n) || n > ownership_frame_limit - 128) {
        error = "invalid ownership frame header"; return false;
    }
    body.resize((size_t)n);
    if (!in.read(body.data(), (std::streamsize)n) || in.get() != '\n' || !line(in, end, 64) ||
        end != "STRATA-OWNERSHIP-END/1 " + seq) { error = "incomplete ownership frame"; return false; }
    payload = std::move(body);
    sequence = next;
    return true;
}
} // namespace strata::core
