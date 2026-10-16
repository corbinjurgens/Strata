#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#if defined(STRATA_NATIVE_EXPERTS)
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#endif

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define PSAPI_VERSION 2
#include <windows.h>
#include <psapi.h>
#endif

namespace fs = std::filesystem;

namespace strata::core::detail {
// Test-only access to handles/state and synthetic CPU memory snapshots, without environment fault controls.
struct FileExpertSourceTestAccess {
    using CM = FileExpertSource::CopyMetric;
    static uint64_t copy_count(const FileExpertSource& s, CM metric) {
        return s.copy_counts_[(size_t) metric].load(std::memory_order_relaxed);
    }
    static bool copy_ready(const FileExpertSource& s) { return s.copy_ready_; }
    static bool copy_stats(const FileExpertSource& s) { return s.copy_stats_; }
    static auto stage_policy(FileExpertSource& s) {
        std::lock_guard<std::mutex> lock(s.stage_mu_);
        return std::make_tuple(s.stage_seq_, s.epoch_, s.stage_key_, s.stage_epoch_, s.stage_used_, s.stage_pf_,
                               s.stage_allocated_bytes_, s.stage_of_.size());
    }
    static uint8_t* seed(FileExpertSource& s, int64_t layer, int64_t expert, const uint8_t* bytes, bool busy = false) {
        std::lock_guard<std::mutex> lock(s.stage_mu_);
        for (uint64_t b : s.layer_blob_bytes_) s.stage_blob_ = std::max(s.stage_blob_, b);
        size_t v = 0;
        bool fill = false;
        if (!s.claim_stage(layer * s.n_expert_ + expert, v, fill) && !fill)
            throw std::runtime_error("ready-copy fixture could not claim a stage buffer");
        std::memcpy(s.stage_buf_[v].get(), bytes, (size_t) s.layer_blob_bytes_[(size_t) layer]);
        s.stage_busy_[v] = busy ? 1 : 0;
        s.stage_pf_[v] = 1;
        return s.stage_buf_[v].get();
    }
    static void busy_key(FileExpertSource& s, int64_t key, bool busy) {
        std::lock_guard<std::mutex> lock(s.stage_mu_);
        s.stage_busy_[s.stage_of_.at(key)] = busy ? 1 : 0;
    }
    static int64_t exchange_stage_owner(FileExpertSource& s, int64_t key, int64_t owner) {
        std::lock_guard<std::mutex> lock(s.stage_mu_);
        return std::exchange(s.stage_key_[s.stage_of_.at(key)], owner);
    }
    static uint64_t exchange_stage_capacity(FileExpertSource& s, int64_t key, uint64_t bytes) {
        std::lock_guard<std::mutex> lock(s.stage_mu_);
        return std::exchange(s.stage_capacity_[s.stage_of_.at(key)], bytes);
    }
    static void resident(FileExpertSource& s, int64_t key, const uint8_t* p) {
        s.complement_offsets_.assign((size_t) s.blobs_, FileExpertSource::kNoComplement);
        s.complement_offsets_[(size_t) key] = 0;
        s.complement_host_ = p;
        s.complement_ready_ = true;
    }
    static void override_blob(FileExpertSource& s, int64_t key, const uint8_t* p) {
        s.prepare_overrides();
        s.override_[(size_t) key] = p;
    }
#if defined(_WIN32)
    static void* exchange_handle(FileExpertSource& s, size_t file, void* h, bool buffered) {
        void*& slot = buffered ? s.buffered_[file] : s.direct_[file];
        return std::exchange(slot, h);
    }
    static std::string exchange_path(FileExpertSource& s, size_t file, std::string path) {
        return std::exchange(s.paths_[file], std::move(path));
    }
    static void set_busy(FileExpertSource& s, size_t slot, bool busy) { s.stage_busy_[slot] = busy ? 1 : 0; }
#endif
    static void stage_geometry(FileExpertSource& s, uint64_t small, uint64_t large) {
        s.n_layers_ = 2;
        s.n_expert_ = 32;
        s.blobs_ = 64;
        s.role_ptr_.resize(6, nullptr);
        s.layer_blob_bytes_ = {small, large};
        s.stage_blob_ = large;
    }
    static const uint8_t* claim(FileExpertSource& s, int64_t key, const HostMemory& memory,
                                uint8_t value, bool publish = true) {
        std::lock_guard<std::mutex> lock(s.stage_mu_);
        size_t slot = 0;
        bool fill = false;
        if (!s.claim_stage(key, slot, fill, false, &memory) && !fill)
            throw std::runtime_error("synthetic stage claim failed");
        if (fill) {
            std::memset(s.stage_buf_[slot].get(), value, (size_t) s.stage_capacity_[slot]);
            s.stage_busy_[slot] = publish ? 0 : 1;
        }
        return s.stage_buf_[slot].get();
    }
    static void age(FileExpertSource& s) { s.epoch_ += FileExpertSource::kStageAge; }
    static bool has(FileExpertSource& s, int64_t key) { return s.stage_of_.count(key) != 0; }
};
}

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

void set_env(const char* name, const char* value);

struct SavedEnv {
    const char* name;
    bool present;
    std::string value;
    explicit SavedEnv(const char* n) : name(n), present(std::getenv(n) != nullptr),
        value(present ? std::getenv(n) : "") {}
    ~SavedEnv() { set_env(name, present ? value.c_str() : nullptr); }
};

struct TempDirectory {
    fs::path path;

    explicit TempDirectory(const fs::path& base = fs::temp_directory_path()) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = base / ("strata-file-source-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void create_pack(const fs::path& dir, uint64_t bytes, const std::vector<std::pair<uint64_t, char>>& markers = {}) {
    const fs::path file = dir / "experts.bin";
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        require((bool) out, "could not create synthetic experts.bin");
        if (bytes > 0) {
            out.seekp((std::streamoff) (bytes - 1));
            out.put('\0');
        }
        require((bool) out, "could not size synthetic experts.bin");
    }
    for (const auto& [offset, marker] : markers) {
        require(offset < bytes, "synthetic marker lies beyond experts.bin");
        std::fstream file_out(file, std::ios::binary | std::ios::in | std::ios::out);
        require((bool) file_out, "could not open synthetic experts.bin for markers");
        file_out.seekp((std::streamoff) offset);
        file_out.put(marker);
        require((bool) file_out, "could not write synthetic expert marker");
    }
}

void check_file_size_rejection(strata::core::FileExpertSource& source, const fs::path& dir, int64_t layers,
                               int64_t experts, uint64_t expected_bytes) {
    std::string err;
    source.close();
    fs::resize_file(dir / "experts.bin", expected_bytes - 1);
    require(!source.open(dir.string(), layers, experts, err), "a truncated expert file was accepted");
    require(!source.mapped() && !err.empty(), "truncated-file rejection left the source mapped or unreported");

    fs::resize_file(dir / "experts.bin", expected_bytes + 1);
    err.clear();
    require(!source.open(dir.string(), layers, experts, err), "an oversized expert file was accepted");
    require(!source.mapped() && !err.empty(), "oversized-file rejection left the source mapped or unreported");
}

#if defined(_WIN32)
void check_mapped_release(strata::core::FileExpertSource& source, int64_t layer, int64_t expert, uint64_t bytes) {
    const uint8_t* p = source.blob(layer, expert);
    require(p != nullptr, "release fixture has no mapped blob");
    const std::vector<uint8_t> saved(p, p + (size_t) bytes);
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    const uintptr_t page = info.dwPageSize;
    const uintptr_t first = ((uintptr_t) p + page - 1) / page * page;
    const uintptr_t end = ((uintptr_t) p + bytes) / page * page;
    require(end > first, "release fixture has no full interior page");
    auto resident = [&](uintptr_t address) {
        PSAPI_WORKING_SET_EX_INFORMATION state{};
        state.VirtualAddress = (void*) address;
        require(QueryWorkingSetEx(GetCurrentProcess(), &state, sizeof state) != 0,
                "QueryWorkingSetEx failed for the release fixture");
        return state.VirtualAttributes.Valid != 0;
    };
    const char* setting = std::getenv("STRATA_FILE_RELEASE");
    const bool enabled = setting != nullptr && std::strcmp(setting, "1") == 0;
    for (int round = 0; round < 3; ++round) {
        require(std::memcmp(p, saved.data(), saved.size()) == 0, "release changed the mapped expert bytes");
        require(resident(first), "reading the expert did not bring its interior page into the working set");
        const uint64_t released = source.release(layer, expert);
        require(released == (enabled ? end - first : 0), "release did not honor full pages or its opt-in switch");
        // Off: the page stays.  On: Windows may keep a trimmed page on its standby list and map it back at once, so a
        // page that is still valid is reported, not failed (the byte count above is the contract).
        if (!enabled) require(resident(first), "release trimmed pages with the switch off");
        else if (resident(first)) std::fprintf(stderr, "note: the trimmed page is already back in the working set\n");
        if ((uintptr_t) p < first)
            require(resident((uintptr_t) p), "release trimmed the expert's shared first page");
        if (end < (uintptr_t) p + bytes)
            require(resident(end), "release trimmed the expert's shared last page");
    }
    require(std::memcmp(p, saved.data(), saved.size()) == 0, "the expert changed after repeated release/read cycles");
    require(source.release(-1, expert) == 0 && source.release(layer, -1) == 0 &&
                source.release(std::numeric_limits<int64_t>::max(), expert) == 0 &&
                source.release(layer, std::numeric_limits<int64_t>::max()) == 0,
            "release accepted an invalid layer or expert");
}
#endif

void test_canonical_layout() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    const uint64_t layer_bytes = (uint64_t) experts * BLOB;
    const uint64_t total = (uint64_t) layers * layer_bytes;
    TempDirectory dir;

    std::string err;
    const bool layout_ok = expert_layout_load(dir.path.string(), layers, experts, err);
    require(layout_ok, "could not load canonical layout: " + err);
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) BLOB, 'b'}, {layer_bytes, 'c'},
                                  {layer_bytes + (uint64_t) BLOB, 'd'}});

    FileExpertSource source;
    bool opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "could not map canonical pack: " + err);
    const uint64_t mapped_before = source.mapped_view_bytes();
    require(!source.retire_direct_mappings(err) && source.mapped_view_bytes() == mapped_before &&
            !source.direct_mappings_retired(), "unsupported canonical retirement changed mappings");
    require(source.blobs() == layers * experts, "canonical blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    require(first && second && next_layer, "valid canonical blob lookup failed");
    require(second - first == (ptrdiff_t) BLOB && next_layer - first == (ptrdiff_t) layer_bytes,
            "canonical expert or layer stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c',
            "canonical blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "canonical bounds check accepted an invalid layer or expert");
    require(source.reads() == 3, "invalid canonical lookups changed the read count");
#if defined(_WIN32)
    check_mapped_release(source, 0, 0, BLOB);
    check_mapped_release(source, 0, 1, BLOB);
#endif

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_pack(dir.path, total, {{layer_bytes, 'c'}});
    err.clear();
    opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened canonical source kept stale state");
    source.close();
    require(source.release(0, 0) == 0, "a closed file source released pages");
}

#if defined(STRATA_NATIVE_EXPERTS)
void test_direct_mapping_retirement() {
#if defined(_WIN32)
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    using Access = strata::core::detail::FileExpertSourceTestAccess;
    using FM = FileExpertSource::FetchMetric;
    constexpr int64_t layers = 2, experts = 3;
    TempDirectory dir(fs::current_path());
    std::string err;
    NativeFmt fm[2];
    require(native_fmt(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, 2560, 640, fm[0], err), err);
    require(native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, 2560, 640, fm[1], err), err);
    require(fm[0].bytes == 1868800 && fm[1].bytes == 2176000, "retirement fixture geometry changed");
    std::vector<uint8_t> truth[2], payload[2], header[2];
    uint64_t absolute[6] = {}, relative[6] = {}, per[6] = {};
    const char* files[2] = {"roles-a.gguf", "roles-b.gguf"};
    const char* roles[3] = {"gate", "up", "down"};
    auto put = [](std::vector<uint8_t>& out, uint64_t n, size_t bytes) {
        for (size_t i = 0; i < bytes; ++i) out.push_back((uint8_t) (n >> (8 * i)));
    };
    auto str = [&](std::vector<uint8_t>& out, const std::string& s) {
        put(out, s.size(), 8); out.insert(out.end(), s.begin(), s.end());
    };
    for (int f = 0; f < 2; ++f) {
        put(header[f], 0x46554747, 4); put(header[f], 3, 4);
        put(header[f], 3, 8); put(header[f], 0, 8); // three tensors, no metadata
    }
    for (int l = 0; l < 2; ++l) {
        truth[l].resize((size_t) (experts * fm[l].bytes));
        for (size_t i = 0; i < truth[l].size(); ++i)
            truth[l][i] = (uint8_t) ((i * 131u + l * 37u) ^ (i >> 7) ^ (i >> 17));
        uint64_t at = 0;
        for (int r = 0; r < 3; ++r) {
            const int file = (l + r) % 2, i = 3 * l + r;
            per[i] = r < 2 ? fm[l].up_off : fm[l].bytes - fm[l].down_off;
            relative[i] = payload[file].size();
            str(header[file], "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight");
            put(header[file], 3, 4);
            put(header[file], r < 2 ? 2560 : 640, 8);
            put(header[file], r < 2 ? 640 : 2560, 8);
            put(header[file], experts, 8);
            put(header[file], r < 2 ? fm[l].gu_type : fm[l].d_type, 4);
            put(header[file], relative[i], 8);
            for (int e = 0; e < experts; ++e) {
                const auto first = truth[l].begin() + (ptrdiff_t) (e * fm[l].bytes + at);
                payload[file].insert(payload[file].end(), first, first + (ptrdiff_t) per[i]);
            }
            at += per[i];
        }
    }
    for (int f = 0; f < 2; ++f) {
        header[f].resize((header[f].size() + 31) / 32 * 32, 0);
        for (int l = 0; l < 2; ++l) for (int r = 0; r < 3; ++r)
            if ((l + r) % 2 == f) absolute[3 * l + r] = header[f].size() + relative[3 * l + r];
        std::ofstream out(dir.path / files[f], std::ios::binary);
        out.write((const char*) header[f].data(), (std::streamsize) header[f].size());
        out.write((const char*) payload[f].data(), (std::streamsize) payload[f].size());
        require((bool) out, "could not write split-role GGUF fixture");
    }
    {
        std::ofstream metadata(dir.path / "native_experts.txt");
        for (int l = 0; l < 2; ++l) {
            metadata << l << ' ' << fm[l].gu_type << ' ' << fm[l].d_type << ' '
                     << (l == 0 ? 0 : experts * fm[0].bytes) << ' ' << fm[l].bytes;
            for (int r = 0; r < 3; ++r) metadata << ' ' << absolute[3 * l + r];
            metadata << ' ' << files[l % 2] << ',' << files[(l + 1) % 2] << ',' << files[l % 2] << '\n';
        }
        require((bool) metadata, "could not write split-role native layout");
    }
    require(expert_layout_load(dir.path.string(), layers, experts, err), err);
    FileExpertSource source;
    source.set_gguf((dir.path / files[0]).string());
    require(source.open(dir.path.string(), layers, experts, err), err);
    const uint64_t mapped = source.mapped_view_bytes();
    require(mapped == header[0].size() + header[1].size() + 12134400 && source.gguf_mode(),
            "split-role GGUF mapped size is wrong");
    require(!source.retire_direct_mappings(err) && source.mapped_view_bytes() == mapped,
            "mapped-only source allowed retirement");
    auto matches = [&](const uint8_t* p, int l, int e) {
        return p && !std::memcmp(p, truth[l].data() + e * fm[l].bytes, (size_t) fm[l].bytes);
    };
    const int32_t batch_layers[] = {1, 0, 1, 0, 1, 0, 1};
    const int32_t batch_experts[] = {2, 2, 0, 0, 1, 1, 2};   // all experts, mixed sizes and a repeated blob
    std::vector<std::vector<uint8_t>> batch_copy;
    std::vector<uint8_t*> batch_dst;
    for (int l : batch_layers) batch_copy.emplace_back((size_t) fm[l].bytes);
    for (auto& copy : batch_copy) batch_dst.push_back(copy.data());
    auto check_batch = [&] {
        require(source.copy_blobs(batch_layers, batch_experts, batch_dst.data(), batch_dst.size()),
                "split-role batch copy failed");
        for (size_t i = 0; i < batch_copy.size(); ++i)
            require(matches(batch_copy[i].data(), batch_layers[i], batch_experts[i]), "split-role batch bytes differ");
        require(source.copy_blobs(nullptr, nullptr, nullptr, 0), "empty batch failed");
        const int32_t invalid[] = {0, (int32_t) layers};
        require(!source.copy_blobs(invalid, batch_experts, batch_dst.data(), 2), "batch accepted an invalid layer");
    };
    check_batch();
    const auto* live = source.blob(0, 0);
    require(matches(live, 0, 0), "mapped split-role fixture differs");
    set_env("STRATA_UNBUFFERED_LOAD", "1");
    std::string why;
    const bool direct = source.set_unbuffered(0, why);
    set_env("STRATA_UNBUFFERED_LOAD", nullptr);
    require(direct, "retirement fixture could not enable unbuffered reads: " + why);
    check_batch();
    source.set_file_cache_bytes(3 * 2176000ull, true);
    Access::set_busy(source, 0, true);
    require(!source.retire_direct_mappings(err) && source.mapped_view_bytes() == mapped,
            "retirement accepted an in-flight fill");
    Access::set_busy(source, 0, false);
    const auto original_path = Access::exchange_path(source, 1, (dir.path / "missing.gguf").string());
    const bool invalid_path = source.retire_direct_mappings(err);
    Access::exchange_path(source, 1, original_path);
    require(!invalid_path && source.mapped_view_bytes() == mapped && !source.direct_mappings_retired() &&
            matches(source.blob(0, 0), 0, 0), "failed fallback open did not preserve mapped source");
    source.set_fetch_profiling(true);
    const auto before = source.fetch_stats();
    require(source.retire_direct_mappings(err), err);
    require(source.direct_mappings_retired() && !source.mapped() && source.mapped_view_bytes() == 0 &&
            source.gguf_mode() && source.unbuffered(), "retirement did not release all views or preserve mode");
    require(source.fetch_stats().values == before.values, "retirement read model payloads");
    require(matches(live, 0, 0) && source.blob(0, 0) == live, "retirement changed a live staging pointer");
    require(source.retire_direct_mappings(err), "retirement is not idempotent");
    require(source.recheck_unbuffered(why), "retired source switched to absent mappings");
    check_batch();   // base_ is null after retirement, but positional direct reads remain valid
    const int64_t batch[3] = {0, 1, 2};
    source.prefetch(1, batch, 3);
    for (int e = 0; e < 3; ++e) require(matches(source.blob(1, e), 1, e), "retired batch changed expert bytes");
    require(matches(live, 0, 0), "retired batch overwrote a protected live pointer");
    std::vector<uint8_t> copy((size_t) fm[0].bytes);
    require(source.copy_blob(0, 2, copy.data()) && matches(copy.data(), 0, 2), "retired copy_blob differs");
    require(source.release(0, 0) == 0 && !source.warms(), "retired source offered mapped page operations");
    source.warm(0, batch, 3); // explicit callers must also be safe when no views remain
    require(source.warmed() == 0, "retired warm touched missing views");
    require(source.direct_fallbacks() == 0, "normal retired reads unexpectedly fell back");

    const auto direct_handle = Access::exchange_handle(source, 0, INVALID_HANDLE_VALUE, false);
    const auto fallback_before = source.fetch_stats();
    require(source.copy_blob(0, 2, copy.data()) && matches(copy.data(), 0, 2), "positional fallback bytes differ");
    require(source.fetch_stats().get(FM::buffered_read_bytes) - fallback_before.get(FM::buffered_read_bytes) == 1868800,
            "buffered fallback read an unexpected byte count");
    check_batch();   // a failed direct batch must use the same buffered positional fallback as copy_blob
    std::vector<uint8_t> a((size_t) fm[0].bytes), b((size_t) fm[1].bytes);
    bool ok_a = false, ok_b = false;
    std::thread ta([&] { ok_a = source.copy_blob(0, 1, a.data()); });
    std::thread tb([&] { ok_b = source.copy_blob(1, 2, b.data()); });
    ta.join(); tb.join();
    require(ok_a && ok_b && matches(a.data(), 0, 1) && matches(b.data(), 1, 2),
            "concurrent positional fallbacks shared a file position or buffer");
    const auto fallback_handle = Access::exchange_handle(source, 0, INVALID_HANDLE_VALUE, true);
    const bool failed_batch = source.copy_blobs(batch_layers, batch_experts, batch_dst.data(), batch_dst.size());
    const bool failed_copy = source.copy_blob(0, 1, a.data());
    const auto* failed_blob = source.blob(0, 1); // not previously cached: publish must invalidate this failed fill
    Access::exchange_handle(source, 0, fallback_handle, true);
    Access::exchange_handle(source, 0, direct_handle, false);
    require(!failed_batch && !failed_copy && failed_blob == nullptr, "double I/O failure published partial expert bytes");
    check_batch();
    require(matches(source.blob(0, 1), 0, 1), "failed fill poisoned a later valid lookup");
    require(source.mapped_view_bytes() == 0, "fallback recreated whole-file mappings");
    source.close();
    require(!source.direct_mappings_retired() && !source.gguf_mode() && source.mapped_view_bytes() == 0,
            "close retained retired source state");
    require(source.open(dir.path.string(), layers, experts, err) && source.mapped_view_bytes() == mapped &&
            !source.direct_mappings_retired() && matches(source.blob(0, 1), 0, 1), "reopen retained experimental state");
    check_batch();
    {
        SavedEnv saved_ready("STRATA_FILE_COPY_READY"), saved_stats("STRATA_FILE_COPY_STATS");
        set_env("STRATA_FILE_COPY_READY", "1");
        set_env("STRATA_FILE_COPY_STATS", "1");
        source.close();
        require(source.open(dir.path.string(), layers, experts, err), err);
        set_env("STRATA_UNBUFFERED_LOAD", "1");
        const bool enabled = source.set_unbuffered(0, why);
        set_env("STRATA_UNBUFFERED_LOAD", nullptr);
        require(enabled && source.retire_direct_mappings(err), "ready fault fixture could not retire mappings");
        require(matches(source.blob(0, 0), 0, 0) && matches(source.blob(1, 0), 1, 0),
                "ready fault fixture could not seed entries");
        Access::busy_key(source, experts, true);
        const auto policy = Access::stage_policy(source);
        const uint64_t read_before = source.file_read_bytes();
        const auto broken_direct = Access::exchange_handle(source, 0, INVALID_HANDLE_VALUE, false);
        const auto broken_buffered = Access::exchange_handle(source, 0, INVALID_HANDLE_VALUE, true);
        const bool hit_ok = source.copy_blob(0, 0, copy.data());
        const bool miss_ok = source.copy_blob(0, 1, a.data());
        const int32_t fault_layers[] = {0, 1}, fault_experts[] = {1, 0};
        uint8_t* fault_dst[] = {a.data(), b.data()};
        const bool batch_ok = source.copy_blobs(fault_layers, fault_experts, fault_dst, 2);
        Access::exchange_handle(source, 0, broken_buffered, true);
        Access::exchange_handle(source, 0, broken_direct, false);
        Access::busy_key(source, experts, false);
        require(hit_ok && matches(copy.data(), 0, 0) && !miss_ok && !batch_ok,
                "ready hit failed during I/O fault, or a busy/missing entry bypassed a failed file read");
        require(source.file_read_bytes() == read_before && !Access::has(source, 1) &&
                    Access::stage_policy(source) == policy,
                "failed copy admitted cache entries, changed stage lifetime, or counted successful file bytes");
        using CM = Access::CM;
        require(Access::copy_count(source, CM::requests) == 4 && Access::copy_count(source, CM::ready) == 1 &&
                    Access::copy_count(source, CM::busy) == 1 && Access::copy_count(source, CM::misses) == 2 &&
                    Access::copy_count(source, CM::reused_bytes) == 1868800,
                "fault-path counters treated busy data as ready or lost failed-copy demands");
        source.close();
    }
    std::cout << "direct mapping retirement: split/mixed bytes, live buffers, fallback/open/I/O faults, concurrency, reset PASS\n";
#else
    std::cout << "direct mapping retirement: Windows-only experiment, not exercised on this platform\n";
#endif
}

void test_native_variable_layout() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    constexpr int64_t embedding = 2560;
    constexpr int64_t feed_forward = 640;
    TempDirectory dir;

    NativeFmt first_fmt, second_fmt;
    std::string err;
    const bool first_ok = native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, embedding, feed_forward, first_fmt, err);
    require(first_ok, "IQ3_XXS/IQ4_NL synthetic format is unavailable: " + err);
    err.clear();
    const bool second_ok = native_fmt(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, embedding, feed_forward, second_fmt, err);
    require(second_ok, "IQ2_XS/IQ4_NL synthetic format is unavailable: " + err);
    require(first_fmt.bytes != second_fmt.bytes, "chosen native formats do not exercise variable layer sizes");

    const uint64_t layer0_bytes = (uint64_t) first_fmt.bytes * experts;
    const uint64_t total = layer0_bytes + (uint64_t) second_fmt.bytes * experts;
    {
        std::ofstream metadata(dir.path / "native_experts.txt");
        require((bool) metadata, "could not create synthetic native_experts.txt");
        metadata << "0 " << GGML_TYPE_IQ3_XXS << ' ' << GGML_TYPE_IQ4_NL << " 0 " << first_fmt.bytes << '\n';
        metadata << "1 " << GGML_TYPE_IQ2_XS << ' ' << GGML_TYPE_IQ4_NL << ' ' << layer0_bytes << ' '
                 << second_fmt.bytes << '\n';
        require((bool) metadata, "could not write synthetic native_experts.txt");
    }
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) first_fmt.bytes, 'b'},
                                  {layer0_bytes, 'c'}, {layer0_bytes + (uint64_t) second_fmt.bytes, 'd'}});

    const bool layout_ok = expert_layout_load(dir.path.string(), layers, experts, err);
    require(layout_ok, "could not load synthetic native layout: " + err);
    FileExpertSource source;
    bool opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "could not map native pack: " + err);
    require(source.blobs() == layers * experts, "native blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    const uint8_t* next_layer_second = source.blob(1, 1);
    require(first && second && next_layer && next_layer_second, "valid native blob lookup failed");
    require(second - first == (ptrdiff_t) first_fmt.bytes && next_layer - first == (ptrdiff_t) layer0_bytes &&
                next_layer_second - next_layer == (ptrdiff_t) second_fmt.bytes,
            "native per-layer expert stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c' && next_layer_second[0] == 'd',
            "native blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "native bounds check accepted an invalid layer or expert");
    require(source.reads() == 4, "invalid native lookups changed the read count");
#if defined(_WIN32)
    check_mapped_release(source, 0, 1, first_fmt.bytes);
    check_mapped_release(source, 1, 1, second_fmt.bytes);
#endif

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_pack(dir.path, total, {{layer0_bytes, 'c'}});
    err.clear();
    opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened native source kept stale state");
    source.close();
}
#endif

void test_complement_plan() {
    using namespace strata::core::detail;
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string error;
    require(make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    require(bytes == 16 && offsets == std::vector<uint64_t>{0, kNoCacheComplement, 3, 6, 11, kNoCacheComplement},
            "wrong compact offsets for variable native layer sizes");
    const uint8_t resident[16] = {}, mapped[5] = {};
    require(cache_complement_blob_or_fallback(1, offsets, resident, mapped) == mapped,
            "GPU-resident expert lost mmap fallback");
    require(cache_complement_blob_or_fallback(4, offsets, resident, mapped) == resident + 11,
            "CPU miss did not use resident complement");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {0, 1}}, {}, offsets, bytes, error)
            && offsets.empty() && bytes == 0, "duplicate pair accepted");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}}, {{0, 1}}, offsets, bytes, error),
            "overlapping tiers accepted");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{2, 0}}, {}, offsets, bytes, error),
            "out-of-range pair accepted");
    require(!make_cache_complement_plan(2, 3, {0, 5}, {}, {}, offsets, bytes, error), "zero-size layer accepted");
}

void test_resident_lend_region() {
    using namespace strata::core::detail;
    // four slots (sizes 5, 3, 3, 5); the experts no slot holds take 10 bytes
    const std::vector<uint64_t> slots{5, 3, 3, 5};
    require(choose_resident_keep_from(slots, 10, 9, 1) == -1, "a base larger than the budget was accepted");
    require(choose_resident_keep_from(slots, 10, 10, 1) == 4, "no room: the lend region must stay on the file");
    require(choose_resident_keep_from(slots, 10, 15, 1) == 3, "the last slot fits and must be kept first");
    require(choose_resident_keep_from(slots, 10, 17, 1) == 3, "a slot that does not fit ends the lend region's walk");
    require(choose_resident_keep_from(slots, 10, 18, 1) == 2, "two slots fit");
    require(choose_resident_keep_from(slots, 10, 100, 1) == 1, "the walk must stop at the lend region's first slot");
    require(choose_resident_keep_from(slots, 10, 100, -1) == 4, "no lend region kept slots in RAM");
    require(choose_resident_keep_from(slots, 10, 100, 9) == 4, "an out-of-range lend region kept slots in RAM");
    require(choose_resident_keep_from({}, 0, 0, 0) == 0, "an empty cache");
}

void test_resident_exchange() {
    using namespace strata::core;
    using namespace strata::core::detail;
    // 2 layers x 3 experts; the GPU holds (0,1) and (1,2): the compact copy holds the other four
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string error;
    require(make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    const std::vector<uint64_t> before = offsets;
    // (1,0) moves into the GPU, (1,2) leaves it: (1,2) takes (1,0)'s bytes' place
    require(exchange_cache_complement(offsets, 3, 5), "a valid exchange was refused");
    require(offsets[5] == before[3] && offsets[3] == kNoCacheComplement, "the exchange did not move the place");
    for (size_t i : {0u, 1u, 2u, 4u}) require(offsets[i] == before[i], "an exchange touched another expert");
    // the copy's size and its set of places are unchanged
    std::vector<uint64_t> a, b;
    for (uint64_t o : before) if (o != kNoCacheComplement) a.push_back(o);
    for (uint64_t o : offsets) if (o != kNoCacheComplement) b.push_back(o);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    require(a == b, "an exchange changed the compact copy's places");
    const std::vector<uint64_t> after = offsets;
    require(!exchange_cache_complement(offsets, 3, 0), "an `in` the copy does not hold was accepted");
    require(!exchange_cache_complement(offsets, 0, 2), "an `out` the copy holds already was accepted");
    require(!exchange_cache_complement(offsets, 0, 0), "a self-exchange was accepted");
    require(!exchange_cache_complement(offsets, 0, 6), "an out-of-range `out` was accepted");
    require(offsets == after, "a refused exchange changed the offsets");
    // and back: the copy is again what the plan makes for the original placement
    require(exchange_cache_complement(offsets, 5, 3), "the reverse exchange was refused");
    require(offsets == before, "exchanging back did not restore the plan");
}

// #1250: when the page-locked complement is registered in steps, and what a meminfo reading says
void test_pin_pacing() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    // the reporter's reading: 54 GiB available but 16 GiB really free for a 26 GiB complement
    require(pin_depends_on_reclaim(16 * GiB, 26 * GiB, 4 * GiB), "a complement bigger than the free pages was not flagged");
    require(!pin_depends_on_reclaim(40 * GiB, 26 * GiB, 4 * GiB), "a complement the free pages cover was flagged");
    require(pin_depends_on_reclaim(30 * GiB, 26 * GiB, 4 * GiB) == false, "exactly free = bytes + reserve is enough");
    require(!pin_depends_on_reclaim(0, 0, 4 * GiB), "no complement, nothing to pin");
    require(pin_step_fits(6 * GiB, 1 * GiB, 4 * GiB) && !pin_step_fits(4 * GiB, 1 * GiB, 4 * GiB),
            "a step must leave the reserve");
    const std::string info =
        "MemTotal:       65011484 kB\nMemFree:         1284096 kB\nMemAvailable:   33463296 kB\nCached:         54507520 kB\n";
    require(meminfo_bytes(info, "MemFree") == 1284096ull * 1024, "MemFree was misread");
    require(meminfo_bytes(info, "MemAvailable") == 33463296ull * 1024, "MemAvailable was misread");
    require(meminfo_bytes(info, "Mem") == 0 && meminfo_bytes(info, "Shmem") == 0, "an absent key was not 0");
}

void test_resident_memory_budget() {
    using strata::core::detail::clamp_resident_budget;
    using strata::core::detail::resident_gib_bytes;
    constexpr uint64_t GiB = 1ull << 30, margin = 256ull << 20, headroom = 4 * GiB;
    constexpr uint64_t unlimited = std::numeric_limits<uint64_t>::max();
    require(resident_gib_bytes(0) == 0 && resident_gib_bytes(4) == 4 * GiB && resident_gib_bytes(0.25, true) == GiB / 4,
            "ordinary GiB requests changed during byte conversion");
    require(resident_gib_bytes(1e-20, true) == 1 && resident_gib_bytes(std::numeric_limits<double>::denorm_min(), true) == 1,
            "a positive sub-byte budget became the zero/full-complement sentinel");
    require(resident_gib_bytes(1.0 / GiB, true) == 1 && resident_gib_bytes(1e-20) == 0,
            "one-byte budget or sub-byte headroom conversion changed");
    require(resident_gib_bytes(1e308, true) == unlimited && resident_gib_bytes(1e308) == unlimited &&
                resident_gib_bytes(17179869184.0, true) == unlimited,
            "an oversized finite GiB request did not saturate safely");
    require(resident_gib_bytes(std::nextafter(17179869184.0, 0.0), true) == unlimited - 2047,
            "the representable value below the uint64 boundary was cast out of range");
    require(clamp_resident_budget(resident_gib_bytes(1e308, true), 10 * GiB, 10 * GiB, 8 * GiB) == 2 * GiB - margin &&
                clamp_resident_budget(6 * GiB, 10 * GiB, 10 * GiB, resident_gib_bytes(1e308)) == 0,
            "saturated budgets or headroom bypassed allocation checks");
    // #730: RAM can hold the requested cache, but Windows cannot commit it.
    require(clamp_resident_budget(66 * GiB, 69 * GiB, 47 * GiB, headroom) == 43 * GiB - margin,
            "a RAM budget exceeded available commit capacity");
    require(clamp_resident_budget(66 * GiB, 47 * GiB, 69 * GiB, headroom) == 43 * GiB - margin,
            "a RAM budget exceeded available physical memory");
    require(clamp_resident_budget(40 * GiB, 69 * GiB, 47 * GiB, headroom) == 40 * GiB,
            "a smaller explicit budget was changed");
    require(clamp_resident_budget(unlimited, 69 * GiB, 47 * GiB, headroom) == 43 * GiB - margin,
            "the what-fits budget ignored commit capacity");
    require(clamp_resident_budget(66 * GiB, 47 * GiB, unlimited, headroom) == 43 * GiB - margin,
            "a platform without a commit reading lost its RAM limit");
    require(clamp_resident_budget(43 * GiB, 69 * GiB, 47 * GiB, headroom) == 43 * GiB,
            "a budget that exactly fits received the clamping margin");
    require(clamp_resident_budget(66 * GiB, 69 * GiB, 0, headroom) == 0,
            "exhausted commit capacity was treated as unlimited");
    require(clamp_resident_budget(66 * GiB, 0, 69 * GiB, headroom) == 0,
            "exhausted RAM produced a positive budget");
    require(clamp_resident_budget(66 * GiB, 69 * GiB, headroom - 1, headroom) == 0 &&
                clamp_resident_budget(66 * GiB, 69 * GiB, headroom, headroom) == 0,
            "subtracting headroom underflowed");
    require(clamp_resident_budget(66 * GiB, 69 * GiB, headroom + margin - 1, headroom) == 0 &&
                clamp_resident_budget(66 * GiB, 69 * GiB, headroom + margin, headroom) == 0,
            "subtracting the clamping margin underflowed");

    // The recent cache must leave the resident policy's resolved floor after the RAM arena is allocated.
    constexpr uint64_t resident_headroom = 8 * GiB;
    require(clamp_resident_budget(6 * GiB, 10 * GiB, 10 * GiB, resident_headroom) == 2 * GiB - margin,
            "a recent cache request consumed the resident policy's headroom");
    require(clamp_resident_budget(6 * GiB, 10 * GiB, 12 * GiB, resident_headroom) == 2 * GiB - margin,
            "a recent cache request exceeded physical memory above the resident floor");
    require(clamp_resident_budget(6 * GiB, 12 * GiB, 10 * GiB, resident_headroom) == 2 * GiB - margin,
            "a recent cache request exceeded commit capacity above the resident floor");
    require(clamp_resident_budget(1 * GiB, 10 * GiB, 10 * GiB, resident_headroom) == 1 * GiB,
            "a smaller recent cache request received a clamping margin");
    require(clamp_resident_budget(2 * GiB, 10 * GiB, 10 * GiB, resident_headroom) == 2 * GiB,
            "a recent cache request exactly above the resident floor was reduced");
    require(clamp_resident_budget(6 * GiB, 8 * GiB, 10 * GiB, resident_headroom) == 0 &&
                clamp_resident_budget(6 * GiB, 10 * GiB, 8 * GiB, resident_headroom) == 0,
            "a recent cache request was admitted at the resident floor");
    require(clamp_resident_budget(6 * GiB, 10 * GiB, 10 * GiB, 0) == 6 * GiB,
            "a zero headroom override reduced a fitting recent cache request");
    require(clamp_resident_budget(6 * GiB, 2 * GiB, 10 * GiB, 0) == 2 * GiB - margin &&
                clamp_resident_budget(6 * GiB, 10 * GiB, 2 * GiB, 0) == 2 * GiB - margin,
            "a zero headroom override ignored the memory limits");
    require(clamp_resident_budget(6 * GiB, 0, 10 * GiB, 0) == 0 &&
                clamp_resident_budget(6 * GiB, 10 * GiB, 0, 0) == 0,
            "a zero headroom override admitted a cache with no available memory");
}

void test_cgroup_memory_budget() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    uint64_t bytes = 0;

    CgroupMemoryStat clean_cache{40 * GiB, 32 * GiB, 0, 0, true};
    require(cgroup_available_bytes(56 * GiB, clean_cache, bytes), "valid cgroup memory.stat was rejected");
    require(bytes == 48 * GiB, "clean inactive file cache was not credited against the cgroup cap");

    CgroupMemoryStat dirty_cache{40 * GiB, 32 * GiB, 4 * GiB, 2 * GiB, true};
    require(cgroup_available_bytes(56 * GiB, dirty_cache, bytes), "valid dirty-cache counters were rejected");
    require(bytes == 42 * GiB, "dirty/writeback pages were incorrectly counted as reclaimable");

    CgroupMemoryStat oversized_inactive{10 * GiB, 20 * GiB, 0, 0, true};
    require(cgroup_available_bytes(12 * GiB, oversized_inactive, bytes),
            "valid oversized inactive-file counter was rejected");
    require(bytes == 12 * GiB, "inactive-file accounting exceeded charged current usage");

    const uint64_t max = std::numeric_limits<uint64_t>::max();
    CgroupMemoryStat large_counters{max, max, max - 1, max, true};
    require(cgroup_available_bytes(max, large_counters, bytes),
            "large memory.stat counters were rejected");
    require(bytes == 0, "dirty/writeback subtraction overflowed or escaped the usage cap");

    CgroupMemoryStat usage_over_limit{60 * GiB, 0, 0, 0, true};
    require(cgroup_available_bytes(56 * GiB, usage_over_limit, bytes),
            "valid over-limit memory counters were rejected");
    require(bytes == 0, "over-limit cgroup usage produced a positive budget");

    CgroupMemoryStat missing_stat{};
    bytes = 123;
    require(!cgroup_available_bytes(56 * GiB, missing_stat, bytes) && bytes == 0,
            "missing memory.stat counters did not fail closed");
}

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

void test_ready_copy_out() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    using Access = detail::FileExpertSourceTestAccess;
    using CM = Access::CM;
    using FM = FileExpertSource::FetchMetric;
    SavedEnv saved_ready("STRATA_FILE_COPY_READY"), saved_stats("STRATA_FILE_COPY_STATS"),
             saved_unbuffered("STRATA_UNBUFFERED_LOAD");
    constexpr int64_t layers = 2, experts = 4;
    require(BLOB == 1382400, "ready-copy fixture geometry changed");
    TempDirectory dir(fs::current_path());
    std::string err;
    require(expert_layout_load(dir.path.string(), layers, experts, err), err);
    std::vector<uint8_t> truth((size_t) (layers * experts * BLOB));
    for (size_t i = 0; i < truth.size(); ++i)
        truth[i] = (uint8_t) ((i * 131u + (i / BLOB) * 37u) ^ (i >> 8) ^ (i >> 17));
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary);
        out.write((const char*) truth.data(), (std::streamsize) truth.size());
        require((bool) out, "could not write ready-copy fixture");
    }
    auto matches = [&](const uint8_t* p, int l, int e) {
        return !std::memcmp(p, truth.data() + (l * experts + e) * BLOB, (size_t) BLOB);
    };
    auto run = [&](bool direct, const char* reuse_env, const char* stats_env, bool reuse, bool observe) {
        set_env("STRATA_FILE_COPY_READY", reuse_env);
        set_env("STRATA_FILE_COPY_STATS", stats_env);
        FileExpertSource source;
        require(source.open(dir.path.string(), layers, experts, err), err);
        require(Access::copy_ready(source) == reuse && Access::copy_stats(source) == observe,
                "ready-copy switches did not use exact opt-in parsing");
        set_env("STRATA_FILE_COPY_READY", nullptr);
        set_env("STRATA_FILE_COPY_STATS", nullptr);   // changing the environment cannot change a live source
        if (direct) {
            set_env("STRATA_UNBUFFERED_LOAD", "1");
            std::string why;
            const bool available = source.set_unbuffered(0, why);
            set_env("STRATA_UNBUFFERED_LOAD", nullptr);
            if (!available) return false;
            source.set_file_cache_bytes(3 * BLOB);
        }
        Access::seed(source, 0, 0, truth.data());
        Access::seed(source, 0, 1, truth.data() + BLOB, true);
        Access::seed(source, 1, 0, truth.data() + experts * BLOB);
        const auto policy = Access::stage_policy(source);
        source.set_fetch_profiling(true);
        std::vector<uint8_t> copy((size_t) BLOB, 0xFF);
        require(source.copy_blob(0, 0, copy.data()) && matches(copy.data(), 0, 0),
                "ready scalar copy returned the wrong expert");
        const int32_t batch_layers[] = {0, 0, 1, 0, 0};
        const int32_t batch_experts[] = {0, 1, 0, 2, 0};
        std::vector<std::vector<uint8_t>> copies(5, std::vector<uint8_t>((size_t) BLOB, 0xFF));
        uint8_t* destinations[] = {copies[0].data(), copies[1].data(), copies[2].data(), copies[3].data(), copies[4].data()};
        require(source.copy_blobs(batch_layers, batch_experts, destinations, 5), "mixed ready/busy/miss batch failed");
        for (size_t i = 0; i < 5; ++i)
            require(matches(copies[i].data(), batch_layers[i], batch_experts[i]), "mixed batch copied the wrong bytes");
        require(Access::stage_policy(source) == policy && !Access::has(source, 2),
                "copy-out admitted a miss, changed recency, or consumed a prefetch marker");
        const uint64_t expected_reads = reuse ? 2764800 : 8294400;
        const auto fetched = source.fetch_stats();
        require(source.file_read_bytes() == expected_reads &&
                    fetched.get(direct ? FM::direct_copy_bytes : FM::mapped_copy_bytes) == expected_reads,
                "ready-copy bytes matched but the source still re-read cached experts");
        if (direct) require(fetched.get(FM::read_batches) == (reuse ? 1 : 2) && source.direct_fallbacks() == 0,
                            "ready-copy batch failed to preserve direct reads for only the remaining experts");
        Access::busy_key(source, 1, false);
        require(source.copy_blob(0, 1, copy.data()) && matches(copy.data(), 0, 1),
                "a published busy entry did not become copyable");
        require(source.file_read_bytes() == (reuse ? 2764800 : 9676800), "published ready entry was re-read");

        // Resident and exchange bytes deliberately disagree with the old stage entry: precedence must win.
        std::vector<uint8_t> resident((size_t) BLOB, 0xA7), exchanged((size_t) BLOB, 0xD2);
        Access::resident(source, 0, resident.data());
        Access::override_blob(source, experts, exchanged.data());
        const int32_t held_layers[] = {0, 1}, held_experts[] = {0, 0};
        uint8_t* held_dst[] = {copies[0].data(), copies[1].data()};
        require(source.copy_blobs(held_layers, held_experts, held_dst, 2) && copies[0] == resident && copies[1] == exchanged,
                "recent-cache lookup bypassed the resident or exchange override");
        require(Access::stage_policy(source) == policy, "resident/busy copies changed stage policy");
        require(!source.copy_blob(-1, 0, copy.data()) && !source.copy_blob(0, experts, copy.data()) &&
                    !source.copy_blob(0, 0, nullptr), "copy-out accepted invalid input");
        auto count = [&](CM m) { return Access::copy_count(source, m); };
        if (observe) {
            require(count(CM::requests) == 9 && count(CM::request_bytes) == 12441600 &&
                        count(CM::resident) == 2 && count(CM::resident_bytes) == 2764800 &&
                        count(CM::ready) == 5 && count(CM::ready_bytes) == 6912000 &&
                        count(CM::busy) == 1 && count(CM::busy_bytes) == 1382400 &&
                        count(CM::misses) == 1 && count(CM::miss_bytes) == 1382400 &&
                        count(CM::reused) == (reuse ? 5 : 0) && count(CM::reused_bytes) == (reuse ? 6912000 : 0),
                    "copy opportunity counters include resident entries, busy entries, or incorrect byte sizes");
            require(count(CM::lock_hold_ns) > 0, "copy opportunity lock time was not observed");
        } else {
            for (size_t i = 0; i < (size_t) CM::count; ++i)
                require(count((CM) i) == 0, "disabled opportunity counters collected observations");
        }
        source.close();
        require(!Access::copy_ready(source) && !Access::copy_stats(source), "close retained copy-out switches");
        for (size_t i = 0; i < (size_t) CM::count; ++i)
            require(count((CM) i) == 0, "close retained copy opportunity counters");
        require(source.open(dir.path.string(), layers, experts, err), err);
        require(!Access::copy_ready(source) && !Access::copy_stats(source) && !Access::has(source, 0),
                "reopen retained copy-out state");
        return true;
    };
    for (bool direct : {false, true}) {
        if (!run(direct, nullptr, nullptr, false, false)) {
            std::cout << "ready copy-out: direct branch skipped (unbuffered reads unavailable)\n";
            break;
        }
        require(run(direct, "0", "0", false, false), "disabled ready-copy fixture reopen failed");
        require(run(direct, "true", "01", false, false), "nonexact ready-copy fixture reopen failed");
        require(run(direct, "", "", false, false), "empty ready-copy fixture reopen failed");
        require(run(direct, "0", "1", false, true), "observe-only ready-copy fixture reopen failed");
        require(run(direct, "1", "0", true, false), "unobserved ready-copy fixture reopen failed");
        require(run(direct, "1", "1", true, true), "observed ready-copy fixture reopen failed");
    }

    set_env("STRATA_FILE_COPY_READY", "1");
    set_env("STRATA_FILE_COPY_STATS", "1");
    FileExpertSource source;
    require(source.open(dir.path.string(), layers, experts, err), err);
    const uint8_t* first = Access::seed(source, 0, 0, truth.data());
    Access::age(source);   // already eligible: a copy hit must not renew its lifetime
    const auto policy = Access::stage_policy(source);
    std::vector<uint8_t> owned((size_t) BLOB);
    require(source.copy_blob(0, 0, owned.data()) && matches(owned.data(), 0, 0) &&
                Access::stage_policy(source) == policy, "copy hit promoted an aged entry");
    const uint8_t* replacement = Access::seed(source, 0, 1, truth.data() + BLOB);
    require(first == replacement && !Access::has(source, 0) && matches(owned.data(), 0, 0),
            "synchronous copy retained a cache pointer or prevented safe replacement");
    const auto ready_owned = owned;
    const auto original_owner = Access::exchange_stage_owner(source, 1, -1);
    const bool bad_owner = source.copy_blob(0, 1, owned.data());
    Access::exchange_stage_owner(source, 1, original_owner);
    require(bad_owner && matches(owned.data(), 0, 1) && Access::copy_count(source, CM::misses) == 1 &&
                source.file_read_bytes() == 1382400, "copy-out trusted a stage index without matching ownership");
    const auto original_capacity = Access::exchange_stage_capacity(source, 1, 1);
    const bool short_buffer = source.copy_blob(0, 1, owned.data());
    Access::exchange_stage_capacity(source, 1, original_capacity);
    require(short_buffer && matches(owned.data(), 0, 1) && Access::copy_count(source, CM::misses) == 2 &&
                source.file_read_bytes() == 2764800, "copy-out used a buffer smaller than the exact layer blob");
    bool concurrent_ok = true;
    std::thread pressure([&] {
        for (int i = 0; i < 64; ++i) {
            Access::age(source);
            Access::seed(source, 0, 0, truth.data());
            Access::age(source);
            Access::seed(source, 0, 1, truth.data() + BLOB);
        }
    });
    for (int i = 0; i < 128; ++i)
        if (!source.copy_blob(0, 0, owned.data()) || !matches(owned.data(), 0, 0)) concurrent_ok = false;
    pressure.join();
    require(concurrent_ok && source.file_cache_stats().buffers == 1,
            "concurrent stage replacement corrupted a synchronous copy or copy misses populated new buffers");
    source.close();
    require(matches(ready_owned.data(), 0, 0) && matches(owned.data(), 0, 0),
            "caller-owned destination changed after source closure");
    std::cout << "ready copy-out: parsing, scalar/batch bytes, observed/actual reads, busy/miss, resident, no promotion, reset PASS\n";
}

// #286: the unbuffered file tier (Windows: FILE_FLAG_NO_BUFFERING; Linux: O_DIRECT + io_submit) hands out the same
// bytes as the mapping - one blob at a time, a batch of adjacent experts (merged into one request), and copy_blob - and
// never falls back to the mapping.  The pack lives in the working directory, not the temp directory: a tmpfs /tmp has
// no O_DIRECT, and there the test says so and skips.
void test_unbuffered_reads() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    const uint64_t layer_bytes = (uint64_t) experts * BLOB;
    const uint64_t total = (uint64_t) layers * layer_bytes;
    TempDirectory dir(fs::current_path());
    std::string err;
    require(expert_layout_load(dir.path.string(), layers, experts, err), "could not load canonical layout: " + err);
    std::vector<uint8_t> bytes((size_t) total);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (uint8_t& b : bytes) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        b = (uint8_t) (x >> 56);
    }
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary | std::ios::trunc);
        out.write((const char*) bytes.data(), (std::streamsize) bytes.size());
        require((bool) out, "could not write the synthetic experts.bin");
    }
    FileExpertSource source;
    require(source.open(dir.path.string(), layers, experts, err), "could not map the synthetic pack: " + err);
    const int32_t batch_layers[] = {1, 0, 1, 0};
    const int32_t batch_experts[] = {2, 0, 0, 2};
    std::vector<std::vector<uint8_t>> copies(4, std::vector<uint8_t>((size_t) BLOB));
    uint8_t* destinations[] = {copies[0].data(), copies[1].data(), copies[2].data(), copies[3].data()};
    auto check_batch = [&] {
        require(source.copy_blobs(batch_layers, batch_experts, destinations, 4), "canonical batch copy failed");
        for (size_t i = 0; i < copies.size(); ++i)
            require(std::memcmp(copies[i].data(), bytes.data() + batch_layers[i] * layer_bytes + batch_experts[i] * BLOB,
                                (size_t) BLOB) == 0, "canonical batch copied the wrong bytes");
    };
    check_batch();
    set_env("STRATA_UNBUFFERED_LOAD", "1");
    std::string why;
    const bool unbuffered = source.set_unbuffered(0, why);
    set_env("STRATA_UNBUFFERED_LOAD", nullptr);
#if defined(_WIN32) || defined(__linux__)
    if (!unbuffered) {
        std::cout << "file_expert_source_test: unbuffered reads skipped (" << why << ")\n";
        source.close();
        return;
    }
#else
    require(!unbuffered, "unbuffered reads on a platform without them");
    source.close();
    return;
#endif
    require(source.unbuffered(), "set_unbuffered succeeded but the source is not unbuffered");
    const int64_t batch[experts] = {0, 1, 2};
    source.prefetch(1, batch, experts);   // layer 1 in one batch: three adjacent blobs
    for (int64_t l = 0; l < layers; ++l)
        for (int64_t e = 0; e < experts; ++e) {
            const uint8_t* b = source.blob(l, e);
            require(b != nullptr, "an unbuffered blob lookup failed");
            require(std::memcmp(b, bytes.data() + (size_t) (l * (int64_t) layer_bytes + e * (int64_t) BLOB),
                                (size_t) BLOB) == 0,
                    "an unbuffered blob differs from the file");
        }
    std::vector<uint8_t> copy((size_t) BLOB);
    require(source.copy_blob(0, 2, copy.data()) &&
                std::memcmp(copy.data(), bytes.data() + (size_t) (2 * BLOB), (size_t) BLOB) == 0,
            "an unbuffered copy_blob differs from the file");
    require(source.direct_fallbacks() == 0, "an unbuffered read fell back to the mapping");
    check_batch();
    source.close();
    require(source.open(dir.path.string(), layers, experts, err), "could not reopen the batch fixture: " + err);
    check_batch();
}

void test_fetch_profiling() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    using FM = FileExpertSource::FetchMetric;
    constexpr int64_t layers = 8, experts = 3;
    require(BLOB == 1382400, "fetch-profile fixture geometry changed");
    TempDirectory dir(fs::current_path());
    std::string err;
    require(expert_layout_load(dir.path.string(), layers, experts, err), err);
    std::vector<uint8_t> truth((size_t) (layers * experts * BLOB));
    for (size_t i = 0; i < truth.size(); ++i) truth[i] = (uint8_t) ((i * 71u) ^ (i >> 11));
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary);
        out.write((const char*) truth.data(), (std::streamsize) truth.size());
        require((bool) out, "could not write fetch-profile fixture");
    }
    FileExpertSource source;
    require(source.open(dir.path.string(), layers, experts, err), err);
    std::vector<uint8_t> copy((size_t) BLOB);
    require(source.copy_blob(0, 0, copy.data()), "disabled mapped copy failed");
    for (uint64_t n : source.fetch_stats().values) require(n == 0, "disabled profile collected counters");
    source.set_fetch_profiling(true);
    require(source.copy_blob(0, 0, copy.data()) && !std::memcmp(copy.data(), truth.data(), (size_t) BLOB),
            "profiled mapped copy changed bytes");
    const auto mapped = source.fetch_stats();
    require(mapped.get(FM::mapped_copy_bytes) == 1382400 && mapped.get(FM::mapped_copy_ns) > 0 &&
            mapped.get(FM::submitted_bytes) == 0, "mapped-copy profile did not separate useful bytes from OS reads");
    source.close();
    require(!source.fetch_profiling(), "close retained profiling opt-in");
    for (uint64_t n : source.fetch_stats().values) require(n == 0, "close retained fetch-profile counters");

    require(source.open(dir.path.string(), layers, experts, err), err);
    set_env("STRATA_UNBUFFERED_LOAD", "1");
    std::string why;
    const bool direct = source.set_unbuffered(0, why);
    set_env("STRATA_UNBUFFERED_LOAD", nullptr);
    if (!direct) {
        std::cout << "fetch profiling: mapped bytes/reset PASS; direct counters skipped (" << why << ")\n";
        return;
    }
    source.set_file_cache_bytes(3 * BLOB);
    source.set_fetch_profiling(true);
    source.begin_layer(0, nullptr, 0);
    const int64_t batch[3] = {0, 1, 2};
    source.prefetch(0, batch, 3);
    const auto cold = source.fetch_stats();
    require(cold.get(FM::claims) == 3 && cold.get(FM::misses) == 3 && cold.get(FM::hits) == 0 &&
            cold.get(FM::allocations) == 3 && cold.get(FM::allocation_bytes) == 4147200 &&
            cold.get(FM::scan_slots) == 3, "cold fetch-profile claim/allocation counts are wrong");
    require(cold.get(FM::read_batches) == 1 && cold.get(FM::read_requests) == 1 &&
            cold.get(FM::submitted_bytes) == 4149248 && cold.get(FM::completed_bytes) == 4149248 &&
            cold.get(FM::direct_copy_bytes) == 4147200 && cold.get(FM::mapped_copy_bytes) == 0,
            "batch profile did not distinguish aligned OS bytes from useful expert bytes");
    require(cold.get(FM::read_ns) >= cold.get(FM::read_plan_ns) + cold.get(FM::read_submit_ns) +
            cold.get(FM::read_wait_ns) + cold.get(FM::direct_copy_ns) && cold.get(FM::read_wait_ns) > 0 &&
            cold.get(FM::direct_copy_ns) > 0, "direct-read phases overlap or were not timed");
    auto check = [&](int64_t l, int64_t e) {
        const auto* p = source.blob(l, e);
        require(p && !std::memcmp(p, truth.data() + (size_t) ((l * experts + e) * BLOB), (size_t) BLOB),
                "fetch profiling changed expert bytes");
    };
    for (int64_t e : batch) check(0, e);
    source.prefetch(0, batch, 3);
    for (int64_t e : batch) check(0, e);
    const auto warm = source.fetch_stats();
    require(warm.get(FM::claims) == 12 && warm.get(FM::hits) == 9 && warm.get(FM::misses) == 3 &&
            warm.get(FM::busy_hits) == 0 && warm.get(FM::fill_wait_ns) == 0 &&
            warm.get(FM::read_batches) == 1 && warm.get(FM::direct_copy_bytes) == 4147200,
            "warm profile counted extra reads, misses, or fill waits");
    source.set_fetch_profiling(false);
    source.begin_layer(1, nullptr, 0);
    check(1, 0);
    require(source.fetch_stats().values == warm.values, "disabled profile changed existing counters");
    source.set_fetch_profiling(true);
    for (int64_t l : {2, 3, 4, 5}) source.begin_layer(l, nullptr, 0);
    check(6, 0);
    const auto evicted = source.fetch_stats();
    require(evicted.get(FM::evictions) == 1 && evicted.get(FM::allocations) == 3 &&
            evicted.get(FM::misses) == 4 && evicted.get(FM::direct_copy_bytes) == 5529600,
            "profile did not count one safe buffer reuse and exactly one new read");
    require(source.direct_fallbacks() == 0, "profile fixture fell back to mapped reads");
    source.close();
    for (uint64_t n : source.fetch_stats().values) require(n == 0, "direct profile counters survived close");
    std::cout << "fetch profiling: disabled/reset, mapped/direct bytes, cold/warm counts, phases, safe reuse PASS\n";
}

void test_recent_file_cache() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    TempDirectory dir(fs::current_path());
    constexpr int64_t layers = 8, experts = 2;
    const uint64_t blob_bytes = (uint64_t) BLOB;
    std::string err;
    require(expert_layout_load(dir.path.string(), layers, experts, err), err);
    std::vector<uint8_t> truth((size_t) (layers * experts * blob_bytes));
    for (size_t i = 0; i < truth.size(); ++i) truth[i] = (uint8_t) ((i * 131u + (i / blob_bytes) * 37u) ^ (i >> 8));
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary);
        out.write((const char*) truth.data(), (std::streamsize) truth.size());
        require((bool) out, "could not write recent-cache fixture");
    }
    FileExpertSource source;
    auto open = [&] {
        require(source.open(dir.path.string(), layers, experts, err), err);
        set_env("STRATA_UNBUFFERED_LOAD", "1");
        std::string why;
        const bool ok = source.set_unbuffered(0, why);
        set_env("STRATA_UNBUFFERED_LOAD", nullptr);
        return ok;
    };
    if (!open()) {
        std::cout << "recent file cache: skipped (unbuffered reads unavailable)\n";
        return;
    }
    auto traverse = [&](int expert) {
        for (int64_t layer = 0; layer < layers; ++layer) {
            source.begin_layer(layer, nullptr, 0);
            if (layer % 2) continue;
            const int64_t e = expert;
            source.prefetch(layer, &e, 1);
            const uint8_t* b = source.blob(layer, expert);
            require(b && !std::memcmp(b, truth.data() + (layer * experts + expert) * blob_bytes, (size_t) blob_bytes),
                    "recent cache returned stale or wrong expert bytes");
        }
    };
    for (int i = 0; i < 3; ++i) traverse(0);
    const uint64_t legacy_reads = source.file_read_bytes();
    source.close();
    require(open(), "could not reopen recent-cache fixture");
    require(source.set_file_cache_bytes(4 * blob_bytes + 1) == 4 * blob_bytes, "cache target was not rounded down");
    for (int i = 0; i < 3; ++i) traverse(0);
    require(source.file_read_bytes() == 4 * blob_bytes, "cached experts were re-read on the next token");
    require(source.file_read_bytes() < legacy_reads, "recent cache did not reduce reads versus legacy staging");
    traverse(1); // Evict the four aged entries with a different expert working set.
    require(source.file_read_bytes() == 8 * blob_bytes, "pressure did not load the new working set once");
    traverse(1);
    require(source.file_read_bytes() == 8 * blob_bytes, "new working set was not retained");
    traverse(0);
    require(source.file_read_bytes() == 12 * blob_bytes, "evicted working set was not reloaded exactly");
    source.close();
    require(open(), "could not reopen after cache eviction");
    for (int i = 0; i < 3; ++i) traverse(0);
    require(source.file_read_bytes() == legacy_reads, "reopen retained the opt-in cache policy");
    std::cout << "recent file cache: exact bytes, reuse, eviction, reopen PASS\n";
}

#if defined(STRATA_NATIVE_EXPERTS)
void test_compact_file_cache() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 8, experts = 4;
    // The actual two UD-Q2_K_XL expert sizes, not values derived from the cache's accounting.
    constexpr uint64_t small = 1868800, large = 2176000;
    NativeFmt small_fmt, large_fmt;
    std::string err;
    require(native_fmt(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, 2560, 640, small_fmt, err), err);
    require(native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, 2560, 640, large_fmt, err), err);
    require(small_fmt.bytes == small && large_fmt.bytes == large, "mixed-cache fixture geometry changed");
    TempDirectory dir(fs::current_path());
    std::vector<uint64_t> offsets;
    std::vector<uint8_t> truth;
    {
        std::ofstream metadata(dir.path / "native_experts.txt");
        for (int64_t l = 0; l < layers; ++l) {
            const uint64_t bytes = l == 2 ? large : small;
            offsets.push_back(truth.size());
            metadata << l << ' ' << (l == 2 ? GGML_TYPE_IQ3_XXS : GGML_TYPE_IQ2_XS) << ' '
                     << GGML_TYPE_IQ4_NL << ' ' << truth.size() << ' ' << bytes << '\n';
            for (int64_t e = 0; e < experts; ++e)
                for (uint64_t i = 0; i < bytes; ++i)
                    truth.push_back((uint8_t) ((i * 131u + (l * experts + e) * 37u) ^ (i >> 8)));
        }
        require((bool) metadata, "could not write mixed-cache layout");
        std::ofstream out(dir.path / "experts.bin", std::ios::binary);
        out.write((const char*) truth.data(), (std::streamsize) truth.size());
        require((bool) out, "could not write mixed-cache experts");
    }
    require(expert_layout_load(dir.path.string(), layers, experts, err), err);
    FileExpertSource source;
    auto open = [&] {
        require(source.open(dir.path.string(), layers, experts, err), err);
        set_env("STRATA_UNBUFFERED_LOAD", "1");
        std::string why;
        const bool ok = source.set_unbuffered(0, why);
        set_env("STRATA_UNBUFFERED_LOAD", nullptr);
        return ok;
    };
    if (!open()) {
        std::cout << "compact file cache: skipped (unbuffered reads unavailable)\n";
        return;
    }
    auto bytes_match = [&](const uint8_t* p, int64_t l, int64_t e) {
        const uint64_t bytes = l == 2 ? large : small;
        return p && std::memcmp(p, truth.data() + offsets[(size_t) l] + e * bytes, (size_t) bytes) == 0;
    };
    auto get = [&](int64_t l, int64_t e) {
        source.begin_layer(l, nullptr, 0);
        const uint8_t* p = source.blob(l, e);
        require(bytes_match(p, l, e), "compact cache returned the wrong mixed-layout bytes");
        return p;
    };
    auto age = [&] { for (int64_t l : {0, 1, 2, 3}) source.begin_layer(l, nullptr, 0); };
    auto check = [&](uint64_t bytes, size_t buffers) {
        const auto stats = source.file_cache_stats();
        require(stats.allocated_bytes == bytes && stats.buffers == buffers, "compact buffer accounting is wrong");
    };

    // Same byte budget: five small experts fit only with compact storage. Repeated traversal must be all hits.
    const uint64_t target = 5 * small + 1;
    require(source.set_file_cache_bytes(target) == 4 * large, "legacy mixed-size budget rounding changed");
    for (int round = 0; round < 3; ++round) for (int64_t l : {0, 1, 3, 4, 5}) get(l, 0);
    const uint64_t legacy_reads = source.file_read_bytes();
    check(4 * large, 4);
    require(!source.file_cache_stats().compact, "compact storage enabled by default");
    source.close();
    require(open(), "mixed-cache reopen failed");
    require(source.set_file_cache_bytes(target, true) == target, "compact budget was rounded to maximum slots");
    for (int round = 0; round < 3; ++round) for (int64_t l : {0, 1, 3, 4, 5}) get(l, 0);
    require(source.file_read_bytes() == 5 * small && legacy_reads > 5 * small,
            "compact storage did not retain five small experts under the byte budget");
    check(5 * small, 5);
    require(source.file_cache_stats().compact, "compact storage was not enabled");

    // A safe small victim grows for a large expert, and a safe large victim shrinks for a small one.
    source.close();
    require(open(), "mixed-cache resize reopen failed");
    require(source.set_file_cache_bytes(large, true) == large, "mixed-cache resize target changed");
    get(0, 0);
    check(small, 1);
    age();
    get(2, 0);
    check(large, 1);
    age();
    get(1, 1);
    check(small, 1);
    require(source.file_read_bytes() == 2 * small + large, "cross-size replacement read accounting is wrong");

    // A fresh live pointer cannot be resized or evicted even when another layer exceeds the target.
    source.close();
    require(open(), "mixed-cache overflow reopen failed");
    source.set_file_cache_bytes(2 * small, true);
    const uint8_t* p = get(0, 0);
    const uint8_t* q = get(2, 0);
    check(small + large, 2);
    require(bytes_match(p, 0, 0) && bytes_match(q, 2, 0), "live pointers changed during mixed-size overflow");
    // The next layer is still inside both lifetimes: even vector growth must preserve their addresses/bytes.
    get(1, 0);
    check(2 * small + large, 3);
    require(bytes_match(p, 0, 0) && bytes_match(q, 2, 0), "pool growth overwrote a live mixed-size buffer");
    age();
    get(4, 0);
    check(2 * small, 2);
    // Repeated pressure recovers freed slots and returns below target after both cross-size replacements.
    for (int round = 0; round < 8; ++round) {
        age();
        get(2, round % experts);
        check(large, 1);
        age();
        get(5, round % experts);
        check(small, 1);
        get(6, round % experts);
        check(2 * small, 2);
    }
    require(source.direct_fallbacks() == 0, "mixed-cache reads fell back to mapping");
    source.close();
    auto reset = source.file_cache_stats();
    require(reset.target_bytes == 0 && reset.allocated_bytes == 0 && reset.buffers == 0 && !reset.compact,
            "close retained compact-cache state");
    require(open(), "mixed-cache final reopen failed");
    require(source.file_read_bytes() == 0 && !source.file_cache_stats().compact,
            "reopen retained compact mode or counters");
    require(source.set_file_cache_bytes(0, true) == 0 && !source.file_cache_stats().compact,
            "zero target enabled compact storage");
    get(0, 0);
    check(large, 1);
    {
        using Access = detail::FileExpertSourceTestAccess;
        using CM = Access::CM;
        SavedEnv saved_ready("STRATA_FILE_COPY_READY"), saved_stats("STRATA_FILE_COPY_STATS");
        set_env("STRATA_FILE_COPY_READY", "1");
        set_env("STRATA_FILE_COPY_STATS", "1");
        source.close();
        require(open(), "mixed ready-copy reopen failed");
        source.set_file_cache_bytes(small + large, true);
        get(0, 0);
        get(2, 1);
        const auto policy = Access::stage_policy(source);
        const uint64_t read_before = source.file_read_bytes();
        source.set_fetch_profiling(true);
        std::vector<uint8_t> scalar((size_t) small + 16, 0xEC);
        require(source.copy_blob(0, 0, scalar.data() + 8) && bytes_match(scalar.data() + 8, 0, 0),
                "mixed-size scalar ready copy changed bytes");
        const int32_t copy_layers[] = {2, 0, 2}, copy_experts[] = {1, 0, 1};
        std::vector<uint8_t> a((size_t) large + 16, 0xEC), b((size_t) small + 16, 0xEC), c((size_t) large + 16, 0xEC);
        uint8_t* destinations[] = {a.data() + 8, b.data() + 8, c.data() + 8};
        require(source.copy_blobs(copy_layers, copy_experts, destinations, 3), "mixed-size ready batch failed");
        require(bytes_match(a.data() + 8, 2, 1) && bytes_match(b.data() + 8, 0, 0) && bytes_match(c.data() + 8, 2, 1),
                "mixed-size ready batch copied a neighbouring expert");
        for (const auto* copy : {&scalar, &a, &b, &c})
            for (size_t i = 0; i < 8; ++i)
                require((*copy)[i] == 0xEC && (*copy)[copy->size() - 1 - i] == 0xEC,
                        "ready copy wrote beyond the layer's exact blob size");
        require(source.file_read_bytes() == read_before && source.fetch_stats().get(FileExpertSource::FetchMetric::read_batches) == 0 &&
                    Access::stage_policy(source) == policy, "mixed-size ready hit re-read files or promoted stage entries");
        require(Access::copy_count(source, CM::requests) == 4 && Access::copy_count(source, CM::ready) == 4 &&
                    Access::copy_count(source, CM::ready_bytes) == 8089600 &&
                    Access::copy_count(source, CM::reused_bytes) == 8089600,
                "mixed-size opportunity accounting used maximum buffer capacity instead of exact blob bytes");
        source.close();
    }
    std::cout << "compact file cache: mixed bytes, byte budget, cross-size reuse, live overflow, recovery, reset PASS\n";
}
#endif

void test_file_cache_headroom() {
    using namespace strata::core;
    using Access = detail::FileExpertSourceTestAccess;
    constexpr uint64_t small = 32ull << 10, large = 64ull << 10, headroom = 1ull << 20;
    constexpr uint64_t margin = 256ull << 20;
    const uint64_t unlimited = ~uint64_t{0};
    auto memory = [&](uint64_t physical, uint64_t commit) {
        detail::HostMemory m;
        m.available = physical;
        m.commit = commit;
        return m;
    };
    FileExpertSource source;
    auto reset = [&](bool compact = true, uint64_t floor = 1ull << 20) {
        source.close();
        Access::stage_geometry(source, small, large);
        source.set_file_cache_bytes(4 * large, compact, floor);
    };
    auto check = [&](uint64_t bytes, size_t buffers) {
        const auto stats = source.file_cache_stats();
        require(stats.allocated_bytes == bytes && stats.buffers == buffers,
                "headroom cache capacity accounting changed");
    };

    // A late startup allocation lowers commit after initial admission. Growth uses the fresh reading,
    // and existing charged capacities are not subtracted from that reading a second time.
    reset();
    source.set_file_cache_bytes(2 * small, true, headroom);
    const uint8_t* first = Access::claim(source, 0, memory(unlimited, headroom + margin + 2 * small), 11);
    Access::age(source);
    Access::claim(source, 1, memory(unlimited, headroom + margin + small), 22);
    check(2 * small, 2);
    require(first[0] == 11, "room for one additional expert failed to retain the aged expert");
    source.set_file_cache_bytes(4 * small, true, headroom);
    Access::age(source);
    const uint8_t* reused = Access::claim(source, 2, memory(unlimited, headroom + margin + small - 1), 32);
    check(2 * small, 2);
    require(reused == first, "one byte below replacement margin admitted optional growth");
    Access::age(source);
    Access::claim(source, 3, memory(unlimited, headroom + margin + small), 34);
    check(3 * small, 3);
    require(first[0] == 32, "exact allocation-margin boundary failed to grow without eviction");
    Access::age(source);
    reused = Access::claim(source, 4, memory(unlimited, headroom), 33);
    check(small, 1);
    require(!Access::has(source, 0) && reused[0] == 33, "commit pressure did not reuse safe storage and reclaim cold buffers");

    // Physical RAM independently limits optional retention, including the largest-blob policy.
    reset(false);
    first = Access::claim(source, 0, memory(headroom + margin + large, unlimited), 44);
    Access::age(source);
    reused = Access::claim(source, 1, memory(headroom, unlimited), 55);
    check(large, 1);
    require(reused == first, "physical pressure grew the largest-blob cache");

    // Prefer a fitting capacity even when a smaller, older victim cannot hold the requested bytes.
    reset();
    Access::claim(source, 0, memory(unlimited, unlimited), 66);
    const uint8_t* fitting = Access::claim(source, 32, memory(unlimited, unlimited), 77);
    Access::age(source);
    reused = Access::claim(source, 33, memory(headroom, headroom), 88);
    require(reused == fitting && reused[large - 1] == 88,
            "pressure allocated a replacement instead of reusing an oversized capacity");
    check(large, 1);
    require(!Access::has(source, 0), "pressure failed to reclaim excess eligible retention");
    // A compact shrink would allocate before freeing the old capacity. Keep the charged larger buffer.
    reset();
    source.set_file_cache_bytes(large, true, headroom);
    fitting = Access::claim(source, 32, memory(unlimited, unlimited), 89);
    Access::age(source);
    reused = Access::claim(source, 0, memory(headroom, headroom), 90);
    require(reused == fitting && reused[large - 1] == 90, "compact shrink allocated under pressure");
    check(large, 1);

    // Other applications can take memory after retention grows. Reclaim only aged entries; buffers
    // still inside their layer lifetime and pending fills remain valid even if mandatory staging grows.
    reset();
    first = Access::claim(source, 0, memory(unlimited, unlimited), 99, false);
    const uint8_t* live = Access::claim(source, 1, memory(unlimited, unlimited), 100);
    Access::claim(source, 32, memory(0, 0), 101);
    check(2 * small + large, 3);
    require(first[0] == 99 && live[0] == 100, "mandatory staging overwrote protected buffers");
    Access::age(source);
    Access::claim(source, 2, memory(0, 0), 102);
    check(2 * small, 2);
    require(first[0] == 99 && Access::has(source, 0), "pressure reclaimed an in-flight buffer");

    // Saturated readings and Linux's absent commit limit do not wrap the retained-capacity allowance.
    reset();
    Access::claim(source, 0, memory(unlimited - small, unlimited), 103);
    Access::age(source);
    Access::claim(source, 1, memory(unlimited - small, unlimited), 104);
    check(2 * small, 2);
    Access::age(source);
    Access::claim(source, 2, memory(unlimited - small, unlimited), 105);
    check(3 * small, 3);
    reset(true, 0);
    first = Access::claim(source, 0, memory(small, small), 105);
    Access::age(source);
    reused = Access::claim(source, 1, memory(0, 0), 106);
    require(reused == first, "zero headroom ignored exhausted memory");
    check(small, 1);

    // Existing callers retain their original caller-budgeted growth behavior.
    source.close();
    Access::stage_geometry(source, small, large);
    source.set_file_cache_bytes(4 * large, true);
    Access::claim(source, 0, memory(0, 0), 107);
    Access::age(source);
    Access::claim(source, 1, memory(0, 0), 108);
    check(2 * small, 2);
    std::cout << "recent cache headroom: fresh RAM/commit, charged capacities, safe reuse, protected staging, reset PASS\n";
}

// #633: the host RAM probe with fake /proc and cgroup trees: v2 (a limit, "max", a missing or malformed limit), v1
// (a limit, unlimited), and no cgroup line at all.  Elsewhere than Linux it reads the machine's RAM.
void test_host_memory() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    HostMemory m;
#if defined(__linux__)
    TempDirectory t;
    auto put = [&](const fs::path& rel, const std::string& text) {
        fs::create_directories((t.path / rel).parent_path());
        std::ofstream(t.path / rel) << text;
    };
    put("meminfo", "MemTotal: 134217728 kB\nMemAvailable: 104857600 kB\n");   // 100 GiB available
    const std::string mi = (t.path / "meminfo").string(), cg = (t.path / "cgroup").string(),
                      root = (t.path / "fs").string();
    auto probe = [&](const std::string& self) {
        put("cgroup", self);
        m.commit = 0;
        const bool ok = host_available_memory(m, mi, cg, root);
        require(m.commit == ~0ull, "Linux memory probe retained a commit limit");
        return ok;
    };
    // no cgroup line at all: MemAvailable alone (before #633: "cannot determine")
    require(probe("") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "no cgroup: MemAvailable alone");
    // v2, a 48 GiB limit with 16 GiB charged, 4 GiB of it clean cache
    put("fs/cgroup.controllers", "memory\n");
    put("fs/box/memory.max", std::to_string(48 * GiB) + "\n");
    put("fs/box/memory.current", std::to_string(16 * GiB) + "\n");
    put("fs/box/memory.stat", "inactive_file " + std::to_string(4 * GiB) + "\nfile_dirty 0\nfile_writeback 0\n");
    require(probe("0::/box\n") && m.available == 36 * GiB && m.cgroup_limit == 48 * GiB, "v2 limit");
    put("fs/box/memory.max", "max\n");
    require(probe("0::/box\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "v2 max");
    put("fs/box/memory.max", "48G\n");
    require(!probe("0::/box\n"), "v2 malformed limit accepted");
    fs::remove(t.path / "fs/box/memory.max");
    require(!probe("0::/box\n"), "v2 group without memory.max accepted");
    // v1: the memory controller's group (a hybrid line list), a 32 GiB limit with 10 GiB used
    put("fs/memory/docker/abc/memory.limit_in_bytes", std::to_string(32 * GiB) + "\n");
    put("fs/memory/docker/abc/memory.usage_in_bytes", std::to_string(10 * GiB) + "\n");
    put("fs/memory/memory.limit_in_bytes", "9223372036854771712\n");   // the root: unlimited
    put("fs/memory/memory.usage_in_bytes", std::to_string(50 * GiB) + "\n");
    require(probe("12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n1:name=systemd:/docker/abc\n") &&
            m.available == 22 * GiB && m.cgroup_limit == 32 * GiB, "v1 limit");
    put("fs/memory/docker/abc/memory.limit_in_bytes", "9223372036854771712\n");
    require(probe("4:memory:/docker/abc\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "v1 unlimited");
    // a v1 group not visible here (another namespace): MemAvailable alone
    require(probe("4:memory:/elsewhere\n") && m.available == 100 * GiB, "v1 group not mounted");
#else
    require(host_available_memory(m) && m.available > 0 && m.cgroup_limit == ~0ull, "this PC's RAM");
#if defined(_WIN32)
    require(m.commit != ~0ull, "Windows memory probe did not report commit capacity");
#else
    require(m.commit == ~0ull, "a platform without commit accounting imposed a limit");
#endif
    (void) GiB;
#endif
}

// STRATA_IO_PREFETCH: the experts.bin file tier hands out the same bytes with the Linux I/O path on (whole-blob
// preads, the read-ahead workers, the mapping for cached blobs), however the reads and the predictions interleave.
void test_io_prefetch() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 3;
    constexpr int64_t experts = 8;
    const uint64_t layer_bytes = (uint64_t) experts * BLOB;
    const uint64_t total = (uint64_t) layers * layer_bytes;
    TempDirectory dir(fs::current_path());
    std::string err;
    require(expert_layout_load(dir.path.string(), layers, experts, err), "could not load canonical layout: " + err);
    std::vector<uint8_t> bytes((size_t) total);
    uint64_t x = 0xD1B54A32D192ED03ull;
    for (uint8_t& b : bytes) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        b = (uint8_t) (x >> 56);
    }
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary | std::ios::trunc);
        out.write((const char*) bytes.data(), (std::streamsize) bytes.size());
        require((bool) out, "could not write the synthetic experts.bin");
    }
    FileExpertSource source;
    require(source.open(dir.path.string(), layers, experts, err), "could not map the synthetic pack: " + err);
    source.set_io_prefetch(true, 3);
#if defined(__linux__)
    require(source.io_prefetch() && source.warms(), "io prefetch did not turn on");
    // predictions for the next layers, some of them wrong or repeated, while the layers are read
    for (int round = 0; round < 20; ++round)
        for (int64_t l = 0; l < layers; ++l) {
            const int64_t pred[5] = {(round + l) % experts, (round * 3 + 1) % experts, (round + 2 * l + 5) % experts,
                                     (round + l) % experts, 7};
            source.warm((l + 1) % layers, pred, 5);
            source.begin_layer(l, nullptr, 0);
            std::vector<int64_t> want = {(round + l) % experts, (round * 5 + 2) % experts, (l * 3 + round) % experts};
            source.prefetch(l, want.data(), (int64_t) want.size());
            for (int64_t e : want) {
                const uint8_t* b = source.blob(l, e);
                require(b != nullptr, "an io-prefetch blob lookup failed");
                require(std::memcmp(b, bytes.data() + (size_t) (l * (int64_t) layer_bytes + e * (int64_t) BLOB), (size_t) BLOB) == 0,
                        "an io-prefetch blob differs from the file");
                const uint8_t* s2 = source.blob_stable(l, e);
                require(s2 != nullptr && std::memcmp(s2, bytes.data() + (size_t) (l * (int64_t) layer_bytes + e * (int64_t) BLOB), (size_t) BLOB) == 0,
                        "a blob_stable pointer differs from the file");
            }
        }
    std::vector<uint8_t> copy((size_t) BLOB);
    require(source.copy_blob(1, 6, copy.data()) &&
                std::memcmp(copy.data(), bytes.data() + (size_t) (layer_bytes + 6 * BLOB), (size_t) BLOB) == 0,
            "an io-prefetch copy_blob differs from the file");
    const FileExpertSource::IoCounters c = source.io_counters();
    require(c.cached_bytes + c.uncached_bytes > 0, "no residency counted");
    std::cout << "io prefetch: " << c.pf_read_blobs << " read ahead, " << c.pf_used << " used, " << c.pf_resident_skips
              << " skipped, pread " << c.pread_bytes << " B: bytes PASS\n";
#else
    require(!source.io_prefetch(), "io prefetch on a platform without it");
#endif
    source.close();
}

void test_rotating_source(bool rotate, bool pin) {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
#if defined(_WIN32)
    _putenv_s("STRATA_EXCHANGE_ROTATE", rotate ? "1" : "0");
#else
    setenv("STRATA_EXCHANGE_ROTATE", rotate ? "1" : "0", 1);
#endif
    TempDirectory dir;
    std::string err;
    require(expert_layout_load(dir.path.string(), 1, 5, err), err);
    const size_t bytes = (size_t)BLOB;
    std::vector<std::vector<uint8_t>> truth(5, std::vector<uint8_t>(bytes));
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary);
        for (size_t e = 0; e < truth.size(); ++e) {
            for (size_t i = 0; i < bytes; ++i) truth[e][i] = (uint8_t)((i * 131u + e * 37u) ^ (i >> 8));
            out.write((const char*)truth[e].data(), (std::streamsize)bytes);
        }
        require((bool)out, "fixture write failed");
    }
    FileExpertSource src;
    ExpertCache cache;
    require(src.open(dir.path.string(), 1, 5, err), err);
    require(cache.open(2, 1, 5, BLOB, err), err);
    for (int e : {2, 3}) {
        const int slot = cache.admit(0, e);
        require(slot >= 0 && cache.fill_slot_blocking(slot, truth[e].data(), err), err);
    }
    const std::vector<std::pair<int32_t, int32_t>> rank{{0,0},{0,1},{0,2},{0,3},{0,4}};
    // Keep expert 4 on the file tier to check that the fallback survives rotation.
    require(src.pin_cache_complement(cache, err, pin, {}, -1, 0, 2 * bytes, &rank), err);
    require(src.reserve_exchanges(2, err), err);
    require(src.exchange_rotation() == (rotate && pin), "rotation activation/fallback wrong");
    require(src.reserve_exchanges(1, err), "smaller capacity rejected");
    if (rotate && pin) require(!src.reserve_exchanges(3, err), "live arena growth accepted");
    set_env("STRATA_UNBUFFERED_LOAD", "1");
    std::string why;
    const bool direct = src.set_unbuffered(0, why);
    set_env("STRATA_UNBUFFERED_LOAD", nullptr);
    if (!direct) std::cout << "rotation batch uses mapped reads (" << why << ")\n";
    std::vector<std::vector<uint8_t>> batch_copy(5, std::vector<uint8_t>(bytes));
    uint8_t* batch_dst[] = {batch_copy[0].data(), batch_copy[1].data(), batch_copy[2].data(),
                          batch_copy[3].data(), batch_copy[4].data()};
    const int32_t batch_layers[] = {0, 0, 0, 0, 0}, batch_experts[] = {0, 1, 2, 3, 4};
    auto check_batch = [&] {
        require(src.copy_blobs(batch_layers, batch_experts, batch_dst, 5), "rotation batch copy failed");
        require(batch_copy == truth, "batch lost a staged, rotated or file-backed expert");
    };
    int incoming[2] = {0, 1}, outgoing[2] = {2, 3};
    std::vector<uint8_t> actual(bytes);
    double commit_ms = 0;
    for (int round = 0; round < 32; ++round) {
        const uint8_t* prior[2]{};
        uint8_t* eviction[2]{};
        for (int q = 0; q < 2; ++q) {
            prior[q] = src.blob(0, incoming[q]); eviction[q] = src.exchange_buffer(q);
            require(prior[q] && !std::memcmp(prior[q], truth[incoming[q]].data(), bytes), "wrong incoming bytes");
            require(cudaMemcpy(eviction[q], cache.device_slot(q), bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
                    "D2H eviction failed");
            require(src.stage_exchange(0, incoming[q], outgoing[q], q), "stage rejected");
            require(!std::memcmp(src.blob(0, outgoing[q]), truth[outgoing[q]].data(), bytes), "staged override wrong");
            require(cache.fill_slot_blocking(q, prior[q], err), err); // H2D completed before commit
        }
        require(!src.stage_exchange(0, incoming[0], outgoing[0], 0), "duplicate stage accepted");
        require(!src.stage_exchange(0, incoming[0], 5, 0), "invalid expert accepted");
        check_batch();
        const auto t0 = std::chrono::steady_clock::now();
        require(src.commit_exchanges() == 2, "commit count wrong");
        commit_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        require(src.commit_exchanges() == 0, "empty commit not empty");
        for (int q = 0; q < 2; ++q) {
            require(!src.has_resident(0, incoming[q]) && src.has_resident(0, outgoing[q]), "residency wrong");
            const uint8_t* held = src.blob(0, outgoing[q]);
            require(!std::memcmp(held, truth[outgoing[q]].data(), bytes), "evicted bytes corrupted");
            require(held == ((rotate && pin) ? eviction[q] : prior[q]), "wrong storage selected");
            if (rotate && pin) require(src.exchange_buffer(q) == prior[q], "old input not recycled");
            require(src.copy_blob(0, outgoing[q], actual.data()) && actual == truth[outgoing[q]], "copy_blob mismatch");
            require(!src.transient(0, outgoing[q]), "resident became transient");
            if (pin) {
                require(src.pinned(0, outgoing[q]) && src.device_alias(0, outgoing[q]), "mapping lost");
                require(cudaMemcpy(actual.data(), src.device_alias(0, outgoing[q]), bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
                        "device alias read failed");
                require(actual == truth[outgoing[q]], "CUDA alias points at stale expert");
            }
            require(cache.verify_slot(q, truth[incoming[q]].data(), err), err);
            std::swap(incoming[q], outgoing[q]);
        }
        require(!src.has_resident(0, 4) && src.copy_blob(0, 4, actual.data()) && actual == truth[4], "file fallback changed");
        check_batch();
    }
    require(src.exchanges() == 64, "exchange accounting wrong");
    require(src.rotated_exchanges() == ((rotate && pin) ? 64u : 0u), "rotation accounting wrong");
    require(src.avoided_exchange_copy_bytes() == ((rotate && pin) ? 64 * bytes : 0u), "avoided bytes wrong");
    std::cout << "rotation integration: rotate=" << rotate << " pinned=" << pin
              << " exchanges=64 commit_ms=" << commit_ms << " exact_bytes=PASS\n";
    src.close();
    require(!src.exchange_rotation() && !src.exchange_buffer(0), "close retained storage");
    require(src.open(dir.path.string(), 1, 5, err), err);
    require(!src.has_resident(0, 0) && !src.exchange_rotation(), "reopen retained residency");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        test_complement_plan();
        test_resident_lend_region();
        test_resident_exchange();
        test_resident_memory_budget();
        test_pin_pacing();
        test_cgroup_memory_budget();
        test_host_memory();
        test_canonical_layout();
        test_unbuffered_reads();
        test_fetch_profiling();
        test_ready_copy_out();
        test_recent_file_cache();
        test_io_prefetch();
        test_file_cache_headroom();
#if defined(STRATA_NATIVE_EXPERTS)
        test_native_variable_layout();
        test_direct_mapping_retirement();
        test_compact_file_cache();
#endif
        if (argc == 2 && std::string(argv[1]) == "--rotation-gpu") {
            test_rotating_source(false, true);
            test_rotating_source(true, true);
            test_rotating_source(true, false);
        }
        std::cout << "file_expert_source_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "file_expert_source_test: " << error.what() << '\n';
        return 1;
    }
}
