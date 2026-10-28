#include "strata/plan/launch.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/session.hpp"
#include "strata/core/verify.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/prefill/prefill.hpp"

#include <algorithm>
#include <charconv>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#endif
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#if !defined(_WIN32)
extern char** environ;
#endif

namespace strata::plan {
uint64_t launch_add(uint64_t a, uint64_t b) {
    if (a > UINT64_MAX - b) throw std::overflow_error("launch size addition overflow");
    return a + b;
}
uint64_t launch_mul(uint64_t a, uint64_t b) {
    if (b && a > UINT64_MAX / b) throw std::overflow_error("launch size multiplication overflow");
    return a * b;
}
uint64_t launch_align(uint64_t b, uint64_t a) {
    if (!a || (a & (a - 1))) throw std::invalid_argument("invalid launch alignment");
    return launch_add(b, a - 1) / a * a;
}
namespace {
constexpr uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
constexpr const char* default_modules = "EAGER";
bool replayable_kv_residency(uint64_t context, uint64_t resident) {
    const auto page = (uint64_t) kernels::qsa_real_shapes().page_size;
    return resident == 0 || (resident >= (uint64_t) core::qsa_kv_resident_min() &&
                             resident < context && resident % page == 0);
}
// The serve command reader owns the same raw descriptor after this frame. stdio/iostream buffering can
// retain its following commands even with _IONBF, so every underlying read here is bounded to this request.
class StdinFrameBuffer final : public std::streambuf {
    char byte_ = 0;
    static std::streamsize read_exact(char* dst, std::streamsize count) {
        std::streamsize done=0;
        while (done<count) {
#if defined(_WIN32)
            const int n=_read(0,dst+done,(unsigned)(count-done));
#else
            const auto n=::read(0,dst+done,(size_t)(count-done));
#endif
            if (n<0 && errno==EINTR) continue;
            if (n<=0) break;
            done+=n;
        }
        return done;
    }
    int_type underflow() override {
        if (gptr() && gptr()<egptr()) return traits_type::to_int_type(*gptr());
        if (read_exact(&byte_,1)!=1) return traits_type::eof();
        setg(&byte_,&byte_,&byte_+1);
        return traits_type::to_int_type(byte_);
    }
    std::streamsize xsgetn(char* dst, std::streamsize count) override {
        std::streamsize done=0;
        if (count>0 && gptr() && gptr()<egptr()) { dst[done++]=*gptr(); gbump(1); }
        return done+read_exact(dst+done,count-done);
    }
};
#define LAUNCH_FIELDS(X) \
    X(context) X(prefill) X(spec) X(kv) X(kv_resident) X(batch) X(batch_groups) X(parallel) \
    X(auto_cache) X(per_layer) X(no_prefill_borrow) X(resident_budget) X(resident_bytes) X(resident_experts) \
    X(resident_pin) X(file_cache) X(host_headroom) X(vram_reserve) X(shared_stage) X(ring) X(workspace) \
    X(weight_scratch) X(common) X(region) X(prefill_mmq) X(prefill_stage_own) X(prefill_reserve_bo) \
    X(prefill_ring_enabled) X(experts) X(top_k) X(vocab) X(max_blob) X(cache_request) X(environment) \
    X(file_cache_effective) X(file_cache_compact) X(vram_required) X(ram_required) X(commit_required) \
    X(stager_threads) X(stager_ring) X(stager_batch)
bool number(const std::string& text, uint64_t& value) {
    if (text.empty() || (text.size() > 1 && text[0] == '0')) return false;
    const auto r = std::from_chars(text.data(), text.data() + text.size(), value);
    return r.ec == std::errc() && r.ptr == text.data() + text.size();
}
std::string list(const std::vector<uint64_t>& values) {
    std::string out;
    for (auto n : values) { if (!out.empty()) out += ','; out += std::to_string(n); }
    return out;
}
bool parse_list(const std::string& text, std::vector<uint64_t>& values, size_t bound) {
    values.clear();
    if (text.empty()) return true;
    size_t begin = 0;
    for (;;) {
        const size_t end = text.find(',', begin);
        uint64_t n;
        if (values.size() >= bound || !number(text.substr(begin, end == std::string::npos ? end : end - begin), n)) return false;
        values.push_back(n);
        if (end == std::string::npos) return true;
        begin = end + 1;
    }
}
uint64_t sum(const std::map<std::string, uint64_t>& values) {
    uint64_t n = 0;
    for (const auto& entry : values) n = launch_add(n, entry.second);
    return n;
}
uint64_t env_number(const char* key, uint64_t fallback, uint64_t unit = 1) {
    const char* v = std::getenv(key);
    if (!v) return fallback;
    uint64_t n;
    if (!number(v, n)) throw std::invalid_argument(std::string("invalid ") + key);
    return launch_mul(n, unit);
}
void reject_unpriced_modes() {
    // These change allocation geometry or add owners outside this bounded profile.
    const char* keys[] = {"STRATA_KV_STAGE_OWN", "STRATA_KV_RING_OFF", "STRATA_KV_MAIN_OFF", "STRATA_HC_Q8",
        "STRATA_HC_Q8_PACK", "STRATA_Q6_PACKED", "STRATA_VERIFY_TRACE", "STRATA_VERIFY_PROFILE",
        "STRATA_ROUTE_RESIDENT", "STRATA_PREFILL_CPU_SHARE", "STRATA_PREFILL_DEMAND_ONLY", "STRATA_RESIDENT_PIN",
        "STRATA_KV_PREFETCH", "STRATA_DF_BRANCH", "STRATA_VERIFY_DEVICE_PLAN", "STRATA_BATCH_MTP", "STRATA_IO_PREFETCH",
        "STRATA_UNBUFFERED_LOAD"};
    for (const char* k : keys) if (std::getenv(k)) throw std::invalid_argument(std::string("unpriced planned mode: ") + k);
}
uint64_t native_environment() {
    // Compatibility fingerprint only, never an authorization digest. Do not serialize environment values.
    std::map<std::string, std::string> values;
#if defined(_WIN32)
    char** env = _environ;
#else
    char** env = environ;
#endif
    for (; env && *env; ++env) {
        std::string entry = *env;
        const auto eq = entry.find('=');
        if (eq == std::string::npos) continue;
        auto key = entry.substr(0, eq);
#if defined(_WIN32)
        for (char& c : key) if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
#endif
        if (key == "STRATA_API_KEY" || key == "STRATA_API_KEY_FILE") continue;
        if (key.starts_with("STRATA_") || key.starts_with("CUDA_") || key.starts_with("GGML_") ||
            key.starts_with("CUBLAS_") || key.starts_with("OMP_") || key=="PATH" || key=="LD_LIBRARY_PATH" ||
            key=="LD_PRELOAD" || key=="SYSTEMROOT") values.emplace(key, entry.substr(eq + 1));
    }
    if (!values.count("CUDA_MODULE_LOADING")) values.emplace("CUDA_MODULE_LOADING", default_modules);
    uint64_t hash = 14695981039346656037ull;
    for (const auto& [key, value] : values)
        for (unsigned char c : std::to_string(key.size()) + ":" + key + std::to_string(value.size()) + ":" + value) {
            hash ^= c; hash *= 1099511628211ull;
        }
    return hash;
}
std::string quote(const std::string& text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char) c; }
        else if (c == '\n') out += "\\n";
        else if (c < 32 || c > 126) throw std::invalid_argument("non-ASCII metadata output");
        else out += (char) c;
    }
    return out + '"';
}
void print_components(const std::map<std::string, uint64_t>& c) {
    std::cout << '{'; bool first = true;
    for (const auto& [k, v] : c) { if (!first) std::cout << ','; first = false; std::cout << quote(k) << ':' << v; }
    std::cout << '}';
}
}

bool validate_launch_settings(const LaunchSettings& s, std::string& error) {
    try {
        if (!s.context || s.context > 1048576 || !s.prefill || s.prefill > 32768 || s.prefill > s.context ||
            s.spec != 2 || s.kv > 3 || !replayable_kv_residency(s.context, s.kv_resident) ||
            s.batch || s.batch_groups != 1 || s.parallel != 1 ||
            s.auto_cache > 1 || s.per_layer > 1 || s.no_prefill_borrow != 1 || s.shared_stage != 0 ||
            s.resident_pin != 1 || !s.resident_budget || s.resident_budget > (1ull << 40) ||
            s.resident_bytes > s.resident_budget || s.file_cache > (1ull << 40) || s.host_headroom > (1ull << 40) ||
            s.vram_reserve > (1ull << 40) || !prefill::Prefill::valid_native_ring(s.ring) || s.workspace != 32 * MiB ||
            !s.weight_scratch || s.weight_scratch > GiB || !s.common || s.common > 64 * GiB ||
            !s.region || s.region > 64 * GiB || s.prefill_mmq > 1 || s.prefill_stage_own ||
            s.prefill_reserve_bo > 1 || s.prefill_ring_enabled > 1 || !s.experts || s.experts > 512 || s.top_k != 10 ||
            !s.vocab || s.vocab > 1048576 || !s.max_blob || s.max_blob > GiB || s.layer_blobs.size() != 48 ||
            s.slots.empty() || s.slots.size() > 48 * s.experts || s.kv_stage_runs.size() > 4 ||
            s.resident_experts > 48 * s.experts - s.slots.size() || s.resident_order.size() != s.resident_experts ||
            s.file_cache_compact > 1 || s.file_cache_effective > s.file_cache || s.cache_request > 48 * 512 ||
            !s.vram_required || !s.ram_required || s.commit_required < s.ram_required ||
            s.vram_required > (1ull << 40) || s.commit_required > (4ull << 40))
            throw std::invalid_argument("unsupported native settings");
        if (!s.stager_threads || s.stager_threads>32 || s.stager_ring<2 || s.stager_ring>256 ||
            !s.stager_batch || s.stager_batch>32 || s.stager_batch>s.stager_ring)
            throw std::invalid_argument("invalid host stager geometry");
        for (auto b : s.layer_blobs) if (!b || b > s.max_blob) throw std::invalid_argument("invalid layer blob");
        if (*std::max_element(s.layer_blobs.begin(), s.layer_blobs.end()) != s.max_blob)
            throw std::invalid_argument("max blob mismatch");
        for (auto b : s.kv_stage_runs) if (!b || b > 64 * GiB) throw std::invalid_argument("invalid KV stage run");
        if ((s.kv_resident > 0 && s.kv_resident < s.context) != !s.kv_stage_runs.empty())
            throw std::invalid_argument("KV stage disagrees with streaming");
        std::vector<uint64_t> expected_stage;
        if (s.kv_resident && s.kv_resident < s.context) {
            const auto cells = launch_align(s.context,4);
            if (s.kv==0) expected_stage={launch_mul(cells,1024),launch_mul(cells,1024)};
            if (s.kv==1) expected_stage={launch_mul(cells,512),launch_mul(cells,512),launch_mul(cells,16),launch_mul(cells,16)};
            if (s.kv==2) expected_stage={launch_mul(cells,288),launch_mul(cells,288)};
            if (s.kv==3) expected_stage={launch_mul(cells,512),launch_mul(cells,16),launch_mul(cells,288)};
        }
        if (s.kv_stage_runs!=expected_stage) throw std::invalid_argument("KV stage layout mismatch");
        std::set<std::pair<uint64_t, uint64_t>> seen;
        uint64_t offset = 0;
        for (const auto& slot : s.slots) {
            if (slot.layer >= 48 || slot.expert >= s.experts || slot.bytes != s.layer_blobs[slot.layer] ||
                slot.offset != offset || !seen.emplace(slot.layer, slot.expert).second)
                throw std::invalid_argument("invalid ordered cache layout");
            offset = launch_add(offset, launch_align(slot.bytes, 256));
        }
        if (s.per_layer) {
            if (s.slots.size()%48) throw std::invalid_argument("unequal per-layer slot quotas");
            const auto quota=s.slots.size()/48;
            for (size_t i=0;i<s.slots.size();++i)
                if (s.slots[i].layer!=i/quota) throw std::invalid_argument("invalid per-layer slot order");
        }
        uint64_t resident_bytes = 0;
        for (auto index : s.resident_order) {
            const auto l = index / s.experts, e = index % s.experts;
            if (l >= 48 || !seen.emplace(l, e).second) throw std::invalid_argument("invalid resident membership");
            resident_bytes = launch_add(resident_bytes, s.layer_blobs[l]);
        }
        if (resident_bytes != s.resident_bytes) throw std::invalid_argument("resident bytes mismatch");
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

std::string encode_launch_settings(const LaunchSettings& s) {
    std::string err;
    if (!validate_launch_settings(s, err)) throw std::invalid_argument(err);
    std::string out = "schema="+std::to_string(launch_schema)+'\n';
#define WRITE_FIELD(name) out += #name "=" + std::to_string(s.name) + '\n';
    LAUNCH_FIELDS(WRITE_FIELD)
#undef WRITE_FIELD
    out += "layer_blobs=" + list(s.layer_blobs) + "\nkv_stage_runs=" + list(s.kv_stage_runs) +
           "\nresident_order=" + list(s.resident_order) + '\n';
    std::vector<uint64_t> flat;
    for (const auto& slot : s.slots) { flat.push_back(slot.layer); flat.push_back(slot.expert); flat.push_back(slot.bytes); flat.push_back(slot.offset); }
    out += "slots=" + list(flat) + '\n';
    if (out.size() > launch_payload_limit) throw std::invalid_argument("native payload exceeds 1 MiB");
    return out;
}

std::string encode_effective_launch(const LaunchSettings& s) {
    const auto payload=encode_launch_settings(s);
    uint64_t fingerprint=14695981039346656037ull;
    for (unsigned char c:payload) { fingerprint^=c; fingerprint*=1099511628211ull; }
    std::string out="schema=1\nverified=1\n";
#define WRITE_EFFECTIVE(name) out += #name "=" + std::to_string(s.name) + '\n';
    LAUNCH_FIELDS(WRITE_EFFECTIVE)
#undef WRITE_EFFECTIVE
    const auto& last=s.slots.back();
    out+="cache_slots="+std::to_string(s.slots.size())+"\ncache_bytes="+
        std::to_string(launch_add(last.offset,launch_align(last.bytes,256)))+
        "\nkv_stage_runs="+list(s.kv_stage_runs)+"\nnative_payload_fingerprint="+std::to_string(fingerprint)+'\n';
    if (out.size()>16384) throw std::invalid_argument("effective launch exceeds 16 KiB");
    return out;
}
std::string frame_effective_launch(const LaunchSettings& s, uint64_t sequence) {
    if (!sequence) throw std::invalid_argument("effective sequence must be positive");
    const auto payload=encode_effective_launch(s);
    return "STRATA-EFFECTIVE/1\n"+std::to_string(sequence)+'\n'+std::to_string(payload.size())+'\n'+payload;
}

bool decode_launch_settings(const std::string& payload, LaunchSettings& settings, std::string& error) {
    try {
        if (payload.empty() || payload.size() > launch_payload_limit || payload.back() != '\n')
            throw std::invalid_argument("invalid payload length or truncated payload");
        for (unsigned char c : payload) if (c != '\n' && (c < 32 || c > 126)) throw std::invalid_argument("invalid codec byte");
        std::map<std::string, std::string> fields;
        std::istringstream input(payload); std::string line;
        while (std::getline(input, line)) {
            const auto eq = line.find('=');
            if (eq == std::string::npos || !fields.emplace(line.substr(0, eq), line.substr(eq + 1)).second)
                throw std::invalid_argument("malformed or duplicate native field");
        }
        auto take = [&](const char* name) {
            auto it = fields.find(name);
            if (it == fields.end()) throw std::invalid_argument(std::string("missing native field: ") + name);
            auto value = it->second; fields.erase(it); return value;
        };
        if (take("schema") != std::to_string(launch_schema)) throw std::invalid_argument("unsupported native schema");
        LaunchSettings s;
#define READ_FIELD(name) if (!number(take(#name), s.name)) throw std::invalid_argument("invalid native number: " #name);
        LAUNCH_FIELDS(READ_FIELD)
#undef READ_FIELD
        std::vector<uint64_t> flat;
        if (!parse_list(take("layer_blobs"), s.layer_blobs, 48) || !parse_list(take("kv_stage_runs"), s.kv_stage_runs, 4) ||
            !parse_list(take("resident_order"), s.resident_order, 48 * 512) ||
            !parse_list(take("slots"), flat, 48 * 512 * 4) || flat.size() % 4)
            throw std::invalid_argument("invalid native list");
        for (size_t n = 0; n < flat.size(); n += 4) s.slots.push_back({flat[n], flat[n+1], flat[n+2], flat[n+3]});
        if (!fields.empty()) throw std::invalid_argument("unknown native field");
        if (!validate_launch_settings(s, error)) return false;
        settings = std::move(s);
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

std::string frame_launch_settings(const std::string& payload) {
    LaunchSettings s; std::string error;
    if (!decode_launch_settings(payload, s, error)) throw std::invalid_argument(error);
    return "STRATA-PLAN/1\n" + std::to_string(payload.size()) + '\n' + payload;
}
bool read_launch_frame(std::istream& in, std::string& payload, std::string& error) {
    auto line = [&](size_t limit, std::string& out) {
        out.clear(); char c;
        while (in.get(c)) { if (c == '\n') return true; if (out.size() == limit) return false; out += c; }
        return false;
    };
    std::string magic, length;
    uint64_t n;
    if (!line(13, magic) || magic != "STRATA-PLAN/1" || !line(7, length) || !number(length, n) ||
        !n || n > launch_payload_limit) { error = "invalid STRATA-PLAN/1 frame header"; return false; }
    std::string bytes((size_t) n, '\0');
    if (!in.read(bytes.data(), (std::streamsize) n)) { error = "truncated native frame"; return false; }
    LaunchSettings s;
    if (!decode_launch_settings(bytes, s, error)) return false;
    payload = std::move(bytes);
    return true;
}
bool read_launch_stdin(std::string& payload, std::string& error) {
#if defined(_WIN32)
    if (_setmode(_fileno(stdin),_O_BINARY)==-1) { error="cannot set binary plan input"; return false; }
#endif
    StdinFrameBuffer buffer;
    std::istream input(&buffer);
    return read_launch_frame(input,payload,error);
}

static bool prepare_launch(const LaunchRequest& r, const LaunchResources& resources, LaunchResolution& result,
                           const LaunchSettings* approved, std::string& error) {
    try {
        if (!resources.cuda || resources.devices != 1) throw std::invalid_argument("planned metadata supports one CUDA device only");
        if (!resources.vram_free || !resources.ram_available || !resources.commit_available)
            throw std::invalid_argument("required resource snapshot is unavailable");
        if (!r.context || r.context > 1048576 || !r.prefill || r.prefill > 32768 || r.prefill > r.context || r.spec != 2 ||
            r.kv > 3 || !replayable_kv_residency(r.context, r.kv_resident) ||
            r.expert_cache < -1 || !r.no_prefill_borrow || !r.resident_budget || r.resident_budget > (1ull << 40) ||
            r.file_cache > (1ull << 40) || r.host_headroom > (1ull << 40) || r.vram_reserve > (1ull << 40) || r.shards.empty())
            throw std::invalid_argument("unsupported launch request (spec2, explicit prefill, owned buffers and resident budget required)");
        reject_unpriced_modes();
        if (approved && !validate_launch_settings(*approved, error)) return false;
        bool compact;
        if (!core::parse_file_cache_compact(std::getenv("STRATA_FILE_CACHE_COMPACT"), compact, error)) return false;
        const strata::GgufModel model(r.shards);
        error = strata::check_architecture(model.meta());
        if (!error.empty()) return false;
        size_t ple_shard = 0;
        if (!model.find("per_layer_token_embd.weight", &ple_shard))
            throw std::invalid_argument("native model requires per_layer_token_embd.weight");
        if (!kernels::PleTable::check_metadata(model.shard(ple_shard), error)) return false;
        core::ModelGeometry g;
        if (const auto* m = model.meta().get("qwen4exp.expert_count")) g.n_expert = (int64_t) m->u;
        uint64_t k = 10;
        if (const auto* m = model.meta().get("qwen4exp.expert_used_count")) k = m->u;
        if (g.n_expert <= 0 || g.n_expert > 512 || k != 10) throw std::invalid_argument("unsupported model expert geometry");
        core::WeightTable table;
        if (!table.read_metadata(r.pack, error) || !table.check_ple_metadata(error)) return false;
        if (!kernels::cpu::expert_layout_load(r.pack, g.n_layers, g.n_expert, error)) return false;
        const auto& layout = kernels::cpu::expert_layout();
        if (!layout.native || layout.n_expert != g.n_expert || layout.max_blob > GiB)
            throw std::invalid_argument("native pack/model geometry mismatch");
        for (const auto& fmt:layout.fmt)
            if (!kernels::native_expert_supported(fmt.gu_type,fmt.d_type,fmt.n_embd,fmt.n_ff))
                throw std::invalid_argument("expert format unsupported by this CUDA build");
        if (std::filesystem::exists(r.pack + "/experts.bin")) {
            const auto path = r.pack + "/experts.bin";
            if (!core::check_expert_file_size(path, std::filesystem::file_size(path), layout.total, error)) return false;
        } else {
            if (!core::check_experts_gguf(r.shards.front(), layout, error)) return false;
        }
        std::set<std::string> skip;
        if (!core::NativeDense::served_names(r.shards, true, skip, error) ||
            !core::NativeDense::keep_unquantized_ple_key(r.pack, skip, error)) return false;
        skip.insert("output.weight"); skip.insert("token_embd.weight");
        uint64_t canonical = 0, dense = 0;
        if (!core::WeightTable::pool_bytes(r.pack, canonical, error, &skip) ||
            !core::NativeDense::weight_bytes_for(r.shards, table, true, 0, g.n_layers, dense, error)) return false;
        if (!dense) throw std::invalid_argument("native projection metadata unavailable");
        const char* files[]={"dense.bin","embd.bin","experts.bin","extra.bin"};
        for (const auto& [name, ref]:table.all()) if (!skip.count(name) && ref.src_bytes) {
            const auto file=std::filesystem::path(r.pack)/files[ref.file_id];
            if (launch_add(ref.src_off,ref.src_bytes)>std::filesystem::file_size(file))
                throw std::invalid_argument("truncated pack tensor resource");
        }
        const auto* head = model.find("output.weight");
        const auto* emb = model.find("token_embd.weight");
        if (!head || !emb || head->shape.size() != 2 || emb->shape.size() != 2 || head->shape[0] != 2560 ||
            emb->shape[0] != 2560 || head->shape[1] != 248320 || emb->shape[1] != 248320 ||
            !kernels::native_mmvq_supported((int) head->type) || !kernels::embed_type_supported((int) emb->type))
            throw std::invalid_argument("missing or unsupported native head/embedding metadata");
        // The current readers validate mapped spans without reading tensor payloads.
        for (size_t i = 0; i < r.shards.size(); ++i)
            for (const auto& tensor : model.shard(i).tensors()) if (!model.in_bounds(tensor, i))
                throw std::invalid_argument("truncated GGUF tensor");
        std::vector<std::pair<int32_t,int32_t>> ranked; int64_t requested_slots;
        {
            std::ifstream header(r.profile,std::ios::binary);
            char magic[4]; uint32_t values[5]{};
            if (!header.read(magic,4) || !header.read((char*)values,sizeof values) ||
                std::string(magic,4)!="STRP" || values[0]!=1 || values[1]!=48 || values[2]!=(uint32_t)g.n_expert ||
                values[3]>48*g.n_expert || values[4]>values[3])
                throw std::invalid_argument("invalid or unbounded profile metadata");
        }
        if (!core::read_expert_profile(r.profile, 48, g.n_expert, ranked, requested_slots, error)) return false;
        std::set<std::pair<int32_t,int32_t>> unique(ranked.begin(), ranked.end());
        if (ranked.empty() || unique.size() != ranked.size()) throw std::invalid_argument("empty or duplicate expert ranking");
        LaunchResolution out;
        auto& s = out.settings;
        s.context=r.context; s.prefill=r.prefill; s.spec=r.spec; s.kv=r.kv; s.kv_resident=r.kv_resident;
        s.auto_cache=r.expert_cache == -1; s.per_layer=r.per_layer; s.resident_budget=r.resident_budget;
        s.file_cache=r.file_cache; s.host_headroom=r.host_headroom; s.vram_reserve=r.vram_reserve;
        s.cache_request=r.expert_cache < 0 ? 0 : r.expert_cache;
        s.environment=native_environment();
        // Planned replay keeps the mapped source. A contiguous experts.bin needs no assembly-cache buffers.
        const auto target=core::detail::file_cache_target(s.file_cache,compact,!std::filesystem::exists(r.pack+"/experts.bin"),
                                                         48*g.n_expert,layout.max_blob,layout.total);
        s.file_cache_effective=target.bytes; s.file_cache_compact=target.compact;
        s.experts=g.n_expert; s.top_k=k; s.vocab=head->shape[1]; s.max_blob=layout.max_blob;
        for (int l=0; l<48; ++l) s.layer_blobs.push_back(layout.blob_bytes(l));
        core::SessionSizing kv; kv.kv=(core::SessionSizing::Kv) r.kv; kv.resident_cells=(int64_t) r.kv_resident;
        auto pf = prefill::Prefill::sizing_inputs(g, (int64_t) r.context, (int64_t) r.prefill, kv);
        prefill::Prefill::resolve_stager(pf,!std::filesystem::exists(r.pack+"/experts.bin"));
        s.stager_threads=pf.stager_threads; s.stager_ring=pf.stager_ring; s.stager_batch=pf.stager_batch;
        s.ring=pf.ring; s.workspace=pf.workspace; s.weight_scratch=pf.weight_scratch; s.common=pf.common; s.region=pf.region;
        s.prefill_mmq=pf.mmq; s.prefill_stage_own=pf.stage_own; s.prefill_reserve_bo=pf.reserve_bo;
        s.prefill_ring_enabled=pf.ring_enabled; s.kv_stage_runs=pf.kv_runs;
        uint64_t mapped=0;
        out.vram["weights_canonical"] = launch_align(canonical, 2*MiB);
        out.vram["weights_native_dense"] = dense;
        out.vram["weights_native_head"] = launch_align(kernels::native_mmvq_weight_bytes((int) head->type,2560,248320),2*MiB);
        out.vram["head_scratch_logits"] = launch_add(2*MiB, launch_align(s.vocab*4,2*MiB));
        out.vram["session_including_resident_kv"] = launch_align(core::session_bytes_for(g,r.context,k,kv),2*MiB);
        out.vram["verifier"] = launch_align(core::Verifier::workspace_bytes(g,r.context,k,2,s.vocab,s.max_blob,mapped),2*MiB);
        out.vram["prefill_scratch_ring_stage"] = prefill::Prefill::bytes_needed_for(pf,true);
        out.vram["prefill_token_ids"] = launch_align(launch_mul(r.prefill,4),2*MiB);
        out.vram["prefill_kv_identity"] = pf.kv_runs.empty() ? 0 : launch_align(launch_align(r.context,4),2*MiB);
        out.vram["cache_tables"] = 4*MiB; // residency and slot-pointer allocations, rounded separately
        out.vram["decode_hit_scratch"] = launch_align(kernels::moe_hit_grouped_scratch_bytes(10,2560,640),2*MiB) + 10*MiB;
        uint64_t base = sum(out.vram);
        if (!approved && launch_add(base,s.vram_reserve) >= resources.vram_free) throw std::invalid_argument("insufficient VRAM before expert slots");
        if (approved) {
            s.slots=approved->slots;
            for (const auto& slot:s.slots)
                if (!unique.count({(int32_t)slot.layer,(int32_t)slot.expert}))
                    throw std::invalid_argument("saved cache pair is absent from profile");
        } else {
        const uint64_t available = resources.vram_free-base-s.vram_reserve;
        uint64_t budget = r.expert_cache == -1 ? available : launch_mul((uint64_t) r.expert_cache,s.max_blob);
        if (budget > available) throw std::invalid_argument("explicit expert cache exceeds resource admission");
        // Preserve the native uniform-grant provenance, then fill that byte budget in ranking order.
        // Cache allocations round to backing pages; retain that rounding before selecting slots.
        budget = budget / (2*MiB) * (2*MiB);
        if (r.expert_cache == -1) budget = launch_mul(std::min<uint64_t>(budget/s.max_blob,ranked.size()),s.max_blob);
        uint64_t used=0;
        auto add_slot = [&](uint64_t l,uint64_t e) {
            const auto b=s.layer_blobs[l], aligned=launch_align(b,256);
            if (launch_add(used,aligned)>budget) return false;
            s.slots.push_back({l,e,b,used}); used=launch_add(used,aligned); return true;
        };
        if (r.per_layer) {
            uint64_t layer_sum=0;
            for (auto b:s.layer_blobs) layer_sum=launch_add(layer_sum,launch_align(b,256));
            const auto quota=std::min<uint64_t>(budget/layer_sum,s.experts);
            for (uint64_t l=0;l<48;++l) {
                uint64_t n=0;
                for (auto [pl,pe]:ranked) if ((uint64_t) pl==l && n<quota) { add_slot(l,pe); ++n; }
                if (n!=quota) throw std::invalid_argument("profile cannot fill per-layer quota");
            }
        } else for (auto [l,e]:ranked) if (!add_slot(l,e)) break;
        if (s.slots.empty()) throw std::invalid_argument("no expert cache slots fit");
        }
        uint64_t used=0;
        for (auto slot:s.slots) used=launch_add(used,launch_align(slot.bytes,256));
        out.vram["expert_cache"] = launch_align(used,2*MiB);
        out.vram["overhead_allowance"] = s.vram_reserve;
        // Resident-budget mode keeps hottest non-GPU experts, in the profile's exact order.
        std::vector<std::pair<int32_t,int32_t>> gpu;
        for (auto slot:s.slots) gpu.emplace_back((int32_t)slot.layer,(int32_t)slot.expert);
        std::vector<uint64_t> offsets;
        if (!core::detail::make_cache_complement_plan(48,s.experts,s.layer_blobs,gpu,{},offsets,s.resident_bytes,error)) return false;
        int64_t held=0;
        core::detail::rank_cache_complement(offsets,s.layer_blobs,s.experts,&ranked,s.resident_budget,s.resident_bytes,held);
        s.resident_experts=held;
        for (auto [l,e]:ranked) if (offsets[l*s.experts+e]!=core::detail::kNoCacheComplement)
            s.resident_order.push_back(l*s.experts+e);
        out.ram["resident_experts"] = s.resident_bytes;
        out.ram["file_cache_target"] = s.file_cache;
        out.ram["mapped_embedding"] = strata::tensor_payload_bytes(*emb);
        out.ram["verifier_mapped"] = mapped;
        uint64_t kv_host=0;
        for (auto run:s.kv_stage_runs) kv_host=launch_add(kv_host,run);
        out.ram["streamed_kv"] = launch_mul(kv_host,12);
        out.ram["host_headroom_allowance"] = s.host_headroom;
        out.ram["startup_scratch_allowance"] = 512*MiB;
        out.ram["prefill_host_stager"] = launch_mul(s.stager_ring,s.max_blob);
        out.vram_required=sum(out.vram); out.ram_required=sum(out.ram);
        out.commit_required=launch_add(out.ram_required,out.vram_required);
        s.vram_required=out.vram_required; s.ram_required=out.ram_required; s.commit_required=out.commit_required;
        if (approved && !(s==*approved)) throw std::invalid_argument("saved launch disagrees with metadata, argv or environment");
        if (out.vram_required>resources.vram_free || out.ram_required>resources.ram_available ||
            out.commit_required>resources.commit_available) throw std::invalid_argument("insufficient launch resources");
        out.payload=encode_launch_settings(s);
        LaunchSettings roundtrip;
        if (!decode_launch_settings(out.payload,roundtrip,error) || !(roundtrip==s)) throw std::logic_error("native codec round trip mismatch");
        result=std::move(out); return true;
    } catch (const std::exception& e) { error=e.what(); return false; }
}

bool resolve_launch(const LaunchRequest& r, const LaunchResources& resources, LaunchResolution& out, std::string& error) {
    return prepare_launch(r,resources,out,nullptr,error);
}
bool admit_launch(const LaunchRequest& r, const LaunchResources& resources, const LaunchSettings& approved, std::string& error) {
    LaunchResolution out;
    return prepare_launch(r,resources,out,&approved,error);
}
bool check_effective_launch(const LaunchSettings& approved, const LaunchSettings& effective, std::string& error) {
    if (!validate_launch_settings(effective,error)) return false;
    if (!(approved==effective)) { error="effective native settings differ from approved launch"; return false; }
    return true;
}
prefill::Prefill::Sizing launch_prefill_sizing(const LaunchSettings& s) {
    prefill::Prefill::Sizing p;
    p.weight_scratch=s.weight_scratch; p.workspace=s.workspace; p.common=s.common; p.region=s.region;
    p.max_blob=s.max_blob; p.chunk=s.prefill; p.experts=s.experts; p.ring=(int)s.ring;
    p.mmq=s.prefill_mmq; p.ring_enabled=s.prefill_ring_enabled; p.reserve_bo=s.prefill_reserve_bo;
    p.stage_own=s.prefill_stage_own; p.kv_runs=s.kv_stage_runs;
    p.stager_threads=(int)s.stager_threads; p.stager_ring=(int)s.stager_ring; p.stager_batch=(int)s.stager_batch;
    return p;
}

bool parse_launch_request(int argc, char** argv, LaunchRequest& r, LaunchResources& resources, bool strict, std::string& error) {
    try {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_USE_SYCL)
        resources.cuda=false;
#endif
        r.file_cache=env_number("STRATA_FILE_CACHE_MIB",0,MiB);
        r.host_headroom=env_number("STRATA_RESIDENT_HEADROOM_GIB",4*GiB,GiB);
        std::set<std::string> seen;
        for (int i=1;i<argc;++i) {
            const std::string key=argv[i];
            if (key=="generate" || (!strict && key=="--resolve-launch")) continue;
            if (!seen.insert(key).second) throw std::invalid_argument("duplicate metadata option");
            if (strict && (key=="--strict-launch-stdin" || key=="--serve")) continue;
            if (key=="--no-prefill-borrow") { r.no_prefill_borrow=true; continue; }
            if (key=="--expert-cache-per-layer") { r.per_layer=true; continue; }
            if (i+1>=argc) throw std::invalid_argument("metadata option requires a value");
            const std::string value=argv[++i];
            if (strict && key.starts_with("--plan-")) throw std::invalid_argument("strict launch uses current resources only");
            if (key=="--pack") { r.pack=value; continue; }
            if (key=="--native") { r.shards=strata::gguf_split_paths(value); continue; }
            if (key=="--expert-profile") { r.profile=value; continue; }
            if (key=="--kv") {
                if (value=="fp16") r.kv=0; else if (value=="int8") r.kv=1; else if (value=="q4_0") r.kv=2;
                else if (value=="k8v4") r.kv=3; else throw std::invalid_argument("invalid KV format"); continue;
            }
            if (key=="--expert-cache" && value=="auto") { r.expert_cache=-1; continue; }
            uint64_t n;
            if (!number(value,n)) throw std::invalid_argument("invalid metadata number");
            if (key=="--max-context") r.context=n;
            else if (key=="--prefill") r.prefill=n;
            else if (key=="--spec") r.spec=n;
            else if (key=="--kv-resident") r.kv_resident=n;
            else if (key=="--expert-cache") { if (n>INT64_MAX) throw std::invalid_argument("cache count overflow"); r.expert_cache=(int64_t)n; }
            else if (key=="--resident-budget-gib") r.resident_budget=launch_mul(n,GiB);
            else if (key=="--vram-reserve-mib") { r.vram_reserve=launch_mul(n,MiB); r.reserve_given=true; }
            else if (key=="--plan-vram-free-bytes") resources.vram_free=n;
            else if (key=="--plan-ram-available-bytes") resources.ram_available=n;
            else if (key=="--plan-commit-available-bytes") resources.commit_available=n;
            else if (key=="--suffix-draft" || key=="--lookup-chain") { if (n) throw std::invalid_argument("planned drafting unsupported"); }
            else throw std::invalid_argument("unsupported metadata option: "+key);
        }
        if (strict && (!seen.count("--serve") || !seen.count("--suffix-draft") || !seen.count("--lookup-chain")))
            throw std::invalid_argument("strict startup requires --serve --suffix-draft 0 --lookup-chain 0");
        return true;
    } catch (const std::exception& e) { error=e.what(); return false; }
}

void default_launch_environment() {
    if (std::getenv("CUDA_MODULE_LOADING")) return;
#if defined(_WIN32)
    _putenv_s("CUDA_MODULE_LOADING",default_modules);
#else
    setenv("CUDA_MODULE_LOADING",default_modules,0);
#endif
}

bool discover_launch_resources(LaunchResources& resources, std::string& error) {
    try {
        default_launch_environment();
        if (!resources.vram_free) {
            size_t free=0,total=0; int devices=0;
            if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices!=1 || cudaMemGetInfo(&free,&total)!=cudaSuccess)
                throw std::invalid_argument("CUDA resource discovery unavailable or unsupported topology");
            resources.vram_free=free; resources.devices=(unsigned) devices;
        }
        if (!resources.ram_available || !resources.commit_available) {
            core::detail::HostMemory host;
            if (!core::detail::host_available_memory(host)) throw std::invalid_argument("host resources unavailable");
            resources.ram_available=host.available; resources.commit_available=host.commit;
        }
        return true;
    } catch (const std::exception& e) { error=e.what(); return false; }
}

int launch_metadata_main(int argc, char** argv) {
    try {
        LaunchRequest r; LaunchResources resources; std::string error;
        if (!parse_launch_request(argc,argv,r,resources,false,error) || !discover_launch_resources(resources,error))
            throw std::invalid_argument(error);
        LaunchResolution out;
        if (!resolve_launch(r,resources,out,error)) throw std::invalid_argument(error);
        std::cout << "{\"schema\":1,\"native_schema\":" << launch_schema
                  << ",\"kind\":\"native_metadata\",\"strict_consumer\":true,\"ownership_telemetry\":false,"
                  << "\"effective_protocol\":\"STRATA-EFFECTIVE/1\",\"effective_payload\":" << quote(encode_effective_launch(out.settings))
                  << ",\"native_payload\":" << quote(out.payload)
                  << ",\"settings\":{";
        bool first=true;
#define JSON_FIELD(name) if (!first) std::cout << ','; first=false; std::cout << "\"" #name "\":" << out.settings.name;
        LAUNCH_FIELDS(JSON_FIELD)
#undef JSON_FIELD
        std::cout << ",\"resident_order\":[" << list(out.settings.resident_order) << "],\"layer_blobs\":[" << list(out.settings.layer_blobs) << "],\"kv_stage_runs\":[" << list(out.settings.kv_stage_runs) << "],\"slots\":[";
        first=true;
        for (auto slot:out.settings.slots) {
            if (!first) std::cout << ','; first=false;
            std::cout << '[' << slot.layer << ',' << slot.expert << ',' << slot.bytes << ',' << slot.offset << ']';
        }
        std::cout << "]},\"estimates\":{\"vram\":"; print_components(out.vram);
        std::cout << ",\"ram\":"; print_components(out.ram);
        std::cout << ",\"vram_required\":" << out.vram_required << ",\"ram_required\":" << out.ram_required
                  << ",\"commit_required\":" << out.commit_required << ",\"unknown_costs\":" << quote(out.unknown_costs)
                  << "},\"resources\":{\"vram_free\":" << resources.vram_free << ",\"ram_available\":" << resources.ram_available
                  << ",\"commit_available\":" << resources.commit_available << ",\"devices\":" << resources.devices << "}}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "invalid_launch_plan: " << e.what() << '\n'; return 2;
    }
}

bool consume_strict_launch(int argc, char** argv, LaunchSettings& settings, std::string& error,
                           bool (*observe)(LaunchResources&, std::string&)) {
    std::string payload; LaunchSettings saved; LaunchRequest request; LaunchResources resources;
    if (!read_launch_stdin(payload,error) || !decode_launch_settings(payload,saved,error) ||
        !parse_launch_request(argc,argv,request,resources,true,error) || !observe(resources,error) ||
        !admit_launch(request,resources,saved,error)) return false;
#if defined(_WIN32)
    if (_setmode(_fileno(stdout),_O_BINARY)==-1) { error="cannot set binary effective-settings output"; return false; }
#endif
    settings=std::move(saved);
    return true;
}
}  // namespace strata::plan
