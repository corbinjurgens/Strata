// tests/core/native_dense_ple_key_test.cpp - #326: a native pack whose PLE key row is BF16 (iq_pack --compat-bf16 of
// OrcaRouter's IQ3_XXS key) loads with that row in the arena, and NativeDense does not upload the GGUF key over it.
//
//   1. BF16 key row + IQ3_XXS GGUF key: served_names lists the key, keep_unquantized_ple_key takes it out of the
//      skip set, the arena holds the BF16 bytes, and NativeDense::load attaches no native key (0.1.25-0.1.31 refused
//      the load with "incompatible matrix blk.1.ple_key.weight").  The other matrix still gets its native upload.
//   2. A quantized key row (the Q2_0 / UD-Q4_K_XL packs: a shape-only row) + a Q8_0 GGUF key: unchanged - the key
//      stays skipped and is served natively.
//   3. index_code_bits on a missing row and a missing index.
//   4. Mixed-format native arena slices, failed-load rollback, packed-Q8 copies and layer-stage ownership.
// Synthetic files at small dimensions; needs a CUDA device for the arena (exits 77 without one).
#include "gguf_fixture.hpp"

#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::NativeDense;
using strata::core::WeightTable;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-ple-key-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

const char* KEY = "blk.1.ple_key.weight";
const char* QKV = "blk.0.attn_qkv.weight";
constexpr uint64_t NE0 = 256, NE1 = 4;

std::vector<fixture::Kv> arch_keys() {
    return {fixture::str("general.architecture", "qwen4exp"), fixture::u32("qwen4exp.block_count", 48),
            fixture::u32("qwen4exp.embedding_length", 2560), fixture::u32("qwen4exp.expert_count", 512),
            fixture::u32("qwen4exp.expert_used_count", 10), fixture::u32("qwen4exp.attention.head_count", 24),
            fixture::u32("qwen4exp.attention.head_count_kv", 2)};
}

// A shape-only quantized row (what iq_pack writes for a tensor the GGUF serves) and, with `bf16_key`, the key as a
// raw BF16 row (index kind 4) in dense.bin.
void write_pack(const fs::path& dir, bool bf16_key) {
    fs::create_directories(dir);
    const uint64_t key_bytes = NE0 * NE1 * 2;
    std::string idx = "# align 256 pool " + std::to_string(bf16_key ? 2048 + 256 : 256) + " tensors 2\n";
    idx += std::string(QKV) + " 0 0 0 0 0 0 256 4 8 0 32 0 0 0 0 0 0 0\n";
    if (bf16_key)
        idx += std::string(KEY) + " 0 4 0 " + std::to_string(key_bytes) + " 256 " + std::to_string(key_bytes) +
               " 256 4 0 0 1 0 0 0 0 0 0 0\n";
    else
        idx += std::string(KEY) + " 0 0 0 0 256 0 256 4 8 0 32 0 0 0 0 0 0 0\n";
    std::ofstream(dir / "index.txt", std::ios::binary) << idx;
    std::vector<char> bytes(bf16_key ? key_bytes : 16);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (char) fixture::pattern(9, i);
    std::ofstream(dir / "dense.bin", std::ios::binary).write(bytes.data(), (std::streamsize) bytes.size());
}

struct Loaded {
    bool ok = false;
    std::string err;
    std::set<std::string> skip;
    WeightTable wt;
    NativeDense dense;
    void* arena = nullptr;
    ~Loaded() { if (arena) cudaFree(arena); }
};

// generate.cpp's order: served_names, keep_unquantized_ple_key (native packs), pool_bytes, load, NativeDense::load.
void load(const fs::path& pack, const std::string& shard, Loaded& l) {
    const std::vector<std::string> shards{shard};
    if (!NativeDense::served_names(shards, true, l.skip, l.err)) return;
    if (!NativeDense::keep_unquantized_ple_key(pack.string(), l.skip, l.err)) return;
    uint64_t pool = 0;
    if (!WeightTable::pool_bytes(pack.string(), pool, l.err, &l.skip)) return;
    if (cudaMalloc(&l.arena, pool ? pool : 256) != cudaSuccess) { l.err = "cudaMalloc"; return; }
    if (!l.wt.load(pack.string(), l.arena, pool ? pool : 256, l.err, &l.skip)) return;
    l.ok = l.dense.load(shards, l.wt, l.err, true);
}

// Mixed formats, partial row groups and non-256-byte payload sizes exercise
// interior arena pointers. The larger Q8 matrix also exercises STRATA_Q8_PACKED=1.
void arena_test(const fs::path& root) {
    namespace kernels = strata::kernels;
    const std::vector<fixture::Tensor> tensors{
        {"blk.0.attn_qkv.weight", {256, 5}, 13, 11},
        {"blk.1.attn_q.weight", {256, 7}, 14, 12},
        {"blk.1.attn_k.weight", {640, 3}, 8, 13},
        {"blk.2.ffn_gate_shexp.weight", {1024, 2048}, 8, 14},
        {KEY, {256, 3}, 8, 15},
    };
    const auto pack = root / "arena-pack";
    fs::create_directories(pack);
    std::ofstream idx(pack / "index.txt", std::ios::binary);
    idx << "# align 256 pool 256 tensors " << tensors.size() << "\n";
    std::set<std::string> skip;
    for (const auto& t : tensors) {
        idx << t.name << " 0 0 0 0 0 0 " << t.shape[0] << " " << t.shape[1]
            << " 8 0 32 0 0 0 0 0 0 0\n";
        skip.insert(t.name);
    }
    idx.close();
    std::ofstream(pack / "dense.bin", std::ios::binary).put('\0');
    auto first_meta = arch_keys();
    const auto first_split = fixture::split_keys(0, 2, tensors.size());
    first_meta.insert(first_meta.end(), first_split.begin(), first_split.end());
    const auto second_meta = fixture::split_keys(1, 2, tensors.size());
    const auto first = root / "arena-1.gguf", second = root / "arena-2.gguf", bad = root / "arena-bad.gguf";
    fixture::write(first, first_meta, {tensors[0], tensors[1], tensors[2]});
    fixture::write(second, second_meta, {tensors[3], tensors[4]});
    auto wrong = tensors[4];
    wrong.shape[0] *= 2;
    fixture::write(bad, second_meta, {tensors[3], wrong});

    Loaded l;
    if (cudaMalloc(&l.arena, 256) != cudaSuccess) { check(false, "allocate canonical test arena"); return; }
    const bool table_ok = l.wt.load(pack.string(), l.arena, 256, l.err, &skip);
    check(table_ok, "load mixed-format shape table " + l.err);
    if (!table_ok) return;
    check(!l.dense.load({first.string(), bad.string()}, l.wt, l.err, true), "later incompatible shard is rejected");
    bool untouched = l.dense.tensor_count() == 0 && l.dense.weight_bytes() == 0;
    for (const auto& t : tensors) untouched = untouched && !l.wt.find(t.name)->native_data;
    check(untouched, "failed load leaves every native reference and counter untouched");
    l.err.clear();
    const bool ok = l.dense.load({first.string(), second.string()}, l.wt, l.err, true);
    check(ok, "same object and table load successfully after failure " + l.err);
    if (!ok) return;
    check(l.dense.tensor_count() == tensors.size(), "packed copies do not inflate the native tensor count");
    const auto* base = static_cast<const unsigned char*>(l.wt.find(tensors[0].name)->native_data);
    size_t at = 0, payload = 0;
    bool same = true, aligned = true, slices = true, packed_same = true;
    const bool packed_on = kernels::native_q8_0_packed_enabled();
    for (const auto& t : tensors) {
        const auto* ref = l.wt.find(t.name);
        const size_t bytes = kernels::native_mmvq_weight_bytes(t.type, (int) t.shape[0], (int) t.shape[1]);
        std::vector<unsigned char> back(bytes), expected(bytes);
        for (size_t i = 0; i < bytes; ++i) expected[i] = fixture::pattern(t.seed, i);
        same = same && cudaMemcpy(back.data(), ref->native_data, bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
               back == expected;
        aligned = aligned && reinterpret_cast<uintptr_t>(ref->native_data) % 256 == 0;
        slices = slices && ref->native_data == base + at;
        at += (bytes + 255) & ~(size_t) 255;
        payload += bytes;
        if (packed_on && t.type == 8 && t.name != KEY &&
            kernels::native_q8_0_packed_eligible((int) t.shape[0], (int) t.shape[1])) {
            std::vector<unsigned char> packed(bytes);
            kernels::native_q8_0_pack_host(expected.data(), packed.data(), (int) t.shape[0], (int) t.shape[1]);
            packed_same = packed_same && cudaMemcpy(back.data(), base + at, bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                          back == packed;
            at += (bytes + 255) & ~(size_t) 255;
        }
    }
    check(same, "all native weight bytes survive arena loading unchanged");
    check(aligned && slices, "mixed matrices occupy nonoverlapping 256-byte-aligned slices");
    check(packed_same, "optional packed-Q8 slices retain the exact packed bytes");
    check(l.dense.weight_bytes() == payload, "weight_bytes retains its original payload accounting");
    bool shared = true;
    for (const auto& t : tensors) shared = shared && l.wt.find(t.name)->native_q8_1 == base + at;
    check(shared, "shared Q8_1 scratch follows every weight slice in the same allocation");
    const void* original = l.wt.find(tensors[0].name)->native_data;
    check(!l.dense.load({first.string(), second.string()}, l.wt, l.err, true) &&
              l.wt.find(tensors[0].name)->native_data == original, "second successful load is rejected without mutation");

    // A stage must count and upload only its requested layers. The original
    // allocation stays live, so accidental global ownership would corrupt it.
    Loaded stage;
    if (cudaMalloc(&stage.arena, 256) != cudaSuccess) { check(false, "allocate split test arena"); return; }
    const bool stage_table = stage.wt.load(pack.string(), stage.arena, 256, stage.err, &skip);
    const bool stage_ok = stage_table && stage.dense.load({first.string(), second.string()}, stage.wt, stage.err,
                                                         true, 1, 2);
    check(stage_ok, "load a separate layer stage from the same shards " + stage.err);
    if (stage_ok) {
        check(stage.dense.tensor_count() == 3 && !stage.wt.find(tensors[0].name)->native_data &&
                  !stage.wt.find(tensors[3].name)->native_data, "stage excludes layers outside its interval");
        check(stage.wt.find(tensors[1].name)->native_data != l.wt.find(tensors[1].name)->native_data,
              "separate stage owns its own allocation");
    }
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("native_dense_ple_key_test: no CUDA device, skipped\n");
        return 77;
    }
    TempDir tmp;

    std::printf("1. BF16 key row (--compat-bf16), IQ3_XXS GGUF key\n");
    {
        const fs::path pack = tmp.path / "orca";
        write_pack(pack, true);
        const std::string shard = (tmp.path / "orca.gguf").string();
        fixture::write(shard, arch_keys(), {{QKV, {NE0, NE1}, 8, 1}, {KEY, {NE0, NE1}, 18, 2}});
        int bits = 99;
        std::string err;
        check(WeightTable::index_code_bits(pack.string(), KEY, bits, err) && bits == 0, "index_code_bits: key row is 0");
        check(WeightTable::index_code_bits(pack.string(), QKV, bits, err) && bits == 8, "index_code_bits: qkv row is 8");
        Loaded l;
        load(pack, shard, l);
        check(l.ok, "the pack loads (was: incompatible matrix) " + l.err);
        check(!l.skip.count(KEY) && l.skip.count(QKV), "the key is not skipped, qkv is");
        const auto* k = l.wt.find(KEY);
        check(k && k->resident && k->data && !k->quantized(), "the key is resident and BF16");
        check(k && !k->native_data, "no native key attached");
        if (k && k->data) {
            std::vector<unsigned char> back(NE0 * NE1 * 2);
            cudaMemcpy(back.data(), k->data, back.size(), cudaMemcpyDeviceToHost);
            bool same = true;
            for (size_t i = 0; i < back.size(); ++i) same = same && back[i] == fixture::pattern(9, i);
            check(same, "the arena holds the pack's BF16 bytes");
        }
        const auto* q = l.wt.find(QKV);
        check(q && q->native_data && q->native_type == 8, "qkv still served natively (Q8_0)");
        check(l.dense.tensor_count() == 1, "one native matrix");
    }

    std::printf("2. quantized key row (Q2_0 / UD-Q4_K_XL packs), Q8_0 GGUF key: unchanged\n");
    {
        const fs::path pack = tmp.path / "q8";
        write_pack(pack, false);
        const std::string shard = (tmp.path / "q8.gguf").string();
        fixture::write(shard, arch_keys(), {{QKV, {NE0, NE1}, 8, 1}, {KEY, {NE0, NE1}, 8, 2}});
        Loaded l;
        load(pack, shard, l);
        check(l.ok, "the pack loads " + l.err);
        check(l.skip.count(KEY) == 1, "the key stays skipped");
        const auto* k = l.wt.find(KEY);
        check(k && !k->resident && k->native_data && k->native_type == 8, "the key is served natively");
        check(l.dense.tensor_count() == 2, "two native matrices");
    }

    std::printf("3. index_code_bits edge cases\n");
    {
        int bits = 0;
        std::string err;
        check(WeightTable::index_code_bits((tmp.path / "orca").string(), "blk.9.nothing.weight", bits, err) &&
                  bits == -1, "a missing row is -1");
        check(!WeightTable::index_code_bits((tmp.path / "none").string(), KEY, bits, err) && !err.empty(),
              "a missing index is an error");
        std::set<std::string> skip{QKV};
        check(NativeDense::keep_unquantized_ple_key((tmp.path / "none").string(), skip, err) && skip.size() == 1,
              "no key in the skip set: the index is not read");
    }

    std::printf("4. aligned native arena and failed-load rollback\n");
    arena_test(tmp.path);

    std::printf(g_fail ? "native_dense_ple_key_test: %d FAILED\n" : "native_dense_ple_key_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
