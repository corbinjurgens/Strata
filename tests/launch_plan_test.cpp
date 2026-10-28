#include "strata/plan/launch.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/session.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/prefill/prefill.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#if defined(_WIN32)
#include <winioctl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace strata;
using namespace strata::plan;
namespace fs = std::filesystem;
namespace strata::core::detail {
// Omit only device storage; exercise the cache's real byte-layout and admission consumers without a model/GPU.
struct ExpertCacheTestAccess {
    static bool initialize(ExpertCache& cache, const LaunchSettings& s, std::string& error) {
        std::vector<int64_t> bytes;
        for (auto slot:s.slots) bytes.push_back((int64_t)slot.bytes);
        if (!ExpertCache::sized_layout(bytes,cache.off_,cache.blob_,error)) return false;
        cache.slots_=cache.live_slots_=(int64_t)s.slots.size(); cache.n_layers_=48; cache.n_expert_=s.experts;
        cache.residency_.assign(48*s.experts,kNotResident); cache.per_layer_=s.per_layer;
        for (int64_t l=0;l<48;++l) {
            int64_t lo=0,hi=0; cache.layer_slot_range(l,lo,hi); cache.layer_next_.push_back((int32_t)lo);
        }
        return true;
    }
};
}
namespace {
int checks = 0;
void require(bool pass, const char* what) {
    ++checks;
    if (!pass) throw std::runtime_error(what);
}
template<class T> void put(std::ostream& out, T value) { out.write((const char*) &value, sizeof value); }
void str(std::ostream& out, const std::string& value) { put<uint64_t>(out, value.size()); out << value; }
void sparse(const fs::path& path, uint64_t bytes) {
    { std::ofstream out(path, std::ios::binary | std::ios::app); }
#if defined(_WIN32)
    HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    DWORD returned = 0;
    require(h != INVALID_HANDLE_VALUE, "open synthetic sparse fixture");
    const bool ok = DeviceIoControl(h, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr) != 0;
    CloseHandle(h);
    require(ok, "mark synthetic fixture sparse");
#endif
    fs::resize_file(path, bytes);
}
void fixture(const fs::path& root, bool gguf_experts = false, bool wrong_offset = false, bool ple = true) {
    fs::create_directories(root);
    std::ofstream gguf(root / "model.gguf", std::ios::binary);
    gguf.write("GGUF", 4); put<uint32_t>(gguf, 3);
    put<uint64_t>(gguf, 50 + (ple ? 1 : 0) + (gguf_experts ? 144 : 0)); put<uint64_t>(gguf, 7);
    str(gguf, "general.architecture"); put<uint32_t>(gguf, 8); str(gguf, "qwen4exp");
    for (auto [key, value] : std::vector<std::pair<std::string,uint32_t>>{
        {"block_count",48},{"embedding_length",2560},{"expert_count",512},{"expert_used_count",10},
        {"attention.head_count",24},{"attention.head_count_kv",2}}) {
        str(gguf,"qwen4exp."+key); put<uint32_t>(gguf,4); put<uint32_t>(gguf,value);
    }
    uint64_t offset = 0;
    std::ofstream index(root / "index.txt");
    index << "# align 256 pool 201326592 tensors 57\n";
    index << "synthetic_norm 0 2 0 134217728 0 134217728 33554432 1 0 0 0 0 0 0 0 0 0 0\n";
    auto tensor = [&](const std::string& name, uint64_t rows) {
        str(gguf,name); put<uint32_t>(gguf,2); put<uint64_t>(gguf,2560); put<uint64_t>(gguf,rows);
        put<uint32_t>(gguf,2); put<uint64_t>(gguf,offset);
        offset += 2560 * rows / 32 * 18;
        index << name << " 0 0 0 0 134217728 0 2560 " << rows << " 4 0 32 0 0 0 0 0 0 0\n";
    };
    tensor("output.weight",248320); tensor("token_embd.weight",248320);
    for (int l=0;l<48;++l) tensor("blk."+std::to_string(l)+".attn_qkv.weight",10240);
    uint64_t pack_offset = 134217728;
    for (const char* suffix : {"key", "value", "norm_key", "norm_query", "norm_conv", "conv1d"}) {
        const std::string name = std::string("blk.1.ple_") + suffix + ".weight";
        const bool matrix = std::string(suffix) == "key" || std::string(suffix) == "value";
        const bool conv = std::string(suffix) == "conv1d";
        const uint64_t cols = 2560, rows = matrix ? 2560 : conv ? 4 : 1;
        const uint64_t bytes = cols * rows * (matrix || conv ? 2 : 4);
        index << name << " 0 " << (matrix ? 4 : conv ? 5 : 0) << ' ' << pack_offset << ' ' << bytes
              << ' ' << pack_offset << ' ' << bytes << ' ' << cols << ' ' << rows << " 0 0 0 0 0 0 0 0 0 0\n";
        pack_offset += launch_align(bytes, 256);
    }
    if (ple) {
        str(gguf, "per_layer_token_embd.weight"); put<uint32_t>(gguf, 2);
        put<uint64_t>(gguf, 160); put<uint64_t>(gguf, 320001536);
        put<uint32_t>(gguf, 20); put<uint64_t>(gguf, offset);
        offset += 320001536ull * 90;
    }
    std::vector<uint64_t> expert_offsets;
    if (gguf_experts) for (int l = 0; l < 48; ++l) {
        kernels::cpu::NativeFmt fmt; std::string error;
        const int type = l % 2 ? 23 : 18;
        require(kernels::cpu::native_fmt(type, 20, 2560, 640, fmt, error), error.c_str());
        for (int role = 0; role < 3; ++role) {
            offset = launch_align(offset, 32);
            str(gguf, "blk." + std::to_string(l) + ".ffn_" + (role == 0 ? "gate" : role == 1 ? "up" : "down") + "_exps.weight");
            put<uint32_t>(gguf, 3); put<uint64_t>(gguf, role == 2 ? 640 : 2560);
            put<uint64_t>(gguf, role == 2 ? 2560 : 640); put<uint64_t>(gguf, 512);
            put<uint32_t>(gguf, role == 2 ? 20 : type); put<uint64_t>(gguf, offset);
            expert_offsets.push_back(offset);
            offset += (role == 2 ? fmt.bytes - fmt.down_off : fmt.up_off) * 512;
        }
    }
    const auto data = launch_align((uint64_t) gguf.tellp(),32);
    gguf.close(); index.close();
    sparse(root / "model.gguf", data+offset);
    sparse(root / "dense.bin",201326592);
    std::ofstream layout(root / "native_experts.txt");
    layout << "# strata native experts v3: synthetic (n_expert 512)\n";
    uint64_t at=0;
    for (int l=0;l<48;++l) {
        kernels::cpu::NativeFmt fmt; std::string error;
        const int type = l%2 ? 23 : 18;
        const bool valid=kernels::cpu::native_fmt(type,20,2560,640,fmt,error);
        require(valid,error.c_str());
        layout << l << ' ' << type << " 20 " << at << ' ' << fmt.bytes;
        if (gguf_experts) for (int role = 0; role < 3; ++role)
            layout << ' ' << data + expert_offsets[3*l+role] + (wrong_offset && l == 0 && role == 0 ? 32 : 0);
        layout << '\n';
        at += fmt.bytes*512;
    }
    layout.close();
    if (!gguf_experts) sparse(root / "experts.bin",at);
    std::ofstream profile(root / "profile.bin",std::ios::binary);
    profile.write("STRP",4);
    for (uint32_t v : {1u,48u,512u,24576u,24576u}) put(profile,v);
    // Interleave layer formats so the ordered layout cannot be represented by a count.
    for (uint16_t e=0;e<512;++e) for (uint16_t l=0;l<48;++l) { put(profile,l); put(profile,e); }
}
std::string refusal(const LaunchRequest& r, const LaunchResources& resources, const char* label) {
    LaunchResolution untouched; untouched.payload="sentinel"; std::string error;
    require(!resolve_launch(r,resources,untouched,error),label);
    require(untouched.payload=="sentinel" && !error.empty(),"failed resolution does not publish a plan");
    return error;
}
void environment(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
void consume_slots(const LaunchSettings& saved) {
    core::ExpertCache cache; std::string error;
    require(core::detail::ExpertCacheTestAccess::initialize(cache,saved,error),"real cache layout consumes saved byte sizes");
    for (size_t i=0;i<saved.slots.size();++i) {
        const auto& slot=saved.slots[i];
        require(cache.slot_offset(i)==slot.offset,"actual aligned offset equals saved offset");
        require(cache.admit(slot.layer,slot.expert)==i,"actual cache consumer preserves ordered pair admission");
        require(cache.slot_of(slot.layer,slot.expert)==i,"actual mapping agrees with saved ordered slot");
    }
    require(cache.resident()==saved.slots.size(),"all saved slots consumed");
}
}

int main(int argc, char** argv) {
    try {
        if (argc>1 && std::string(argv[1])=="--consume") {
            std::string error;
            LaunchSettings saved;
            const auto observe=[](LaunchResources& resources,std::string&) {
                resources={6ull<<30,48ull<<30,48ull<<30,1,true}; return true;
            };
            require(consume_strict_launch(argc-1,argv+1,saved,error,observe),error.c_str());
            consume_slots(saved);
            core::ModelGeometry geometry; geometry.n_expert=saved.experts;
            core::SessionSizing kv; kv.kv=(core::SessionSizing::Kv)saved.kv; kv.resident_cells=saved.kv_resident;
            auto current=prefill::Prefill::sizing_inputs(geometry,saved.context,saved.prefill,kv);
            const auto approved=launch_prefill_sizing(saved);
            current.stager_threads=approved.stager_threads; current.stager_ring=approved.stager_ring;
            current.stager_batch=approved.stager_batch;
            require(prefill::Prefill::check_strict_sizing(approved,current,false,false,error),"real prefill consumer accepts exact generated profile");
            std::string command;
            while (command.size()<5) {
                char chunk[5];
#if defined(_WIN32)
                const int n=_read(0,chunk,(unsigned)(5-command.size()));
#else
                const auto n=::read(0,chunk,5-command.size());
#endif
                if (n<=0) break;
                command.append(chunk,(size_t)n);
            }
            require(command=="QUIT\n","real stdin frame leaves raw-fd engine command unread");
            std::cout<<frame_effective_launch(saved,1);
            std::cout<<"PASS strict_profile context="<<saved.context<<" checks="<<checks<<" slots="<<saved.slots.size()<<'\n';
            return 0;
        }
        const auto root=argc>1 ? fs::path(argv[1]) : fs::temp_directory_path()/"strata-launch-plan-test";
        require(root.is_absolute(),"fixtures must be outside checkout at an absolute path");
        fixture(root);
        LaunchRequest r;
        r.pack=root.string(); r.shards={(root/"model.gguf").string()}; r.profile=(root/"profile.bin").string();
        r.context=8192; r.prefill=64; r.spec=2; r.kv=1;
        r.no_prefill_borrow=true;
        r.resident_budget=22ull<<30; r.file_cache=10176ull<<20; r.host_headroom=8ull<<30;
        r.vram_reserve=448ull<<20; r.reserve_given=true;
        LaunchResources resources{6ull<<30,48ull<<30,48ull<<30,1,true};
        LaunchResolution a,b; std::string error;
        bool resolved=resolve_launch(r,resources,a,error);
        require(resolved,error.c_str());
        require(a.settings.spec==2 && a.settings.no_prefill_borrow==1 && a.settings.prefill==64,"8K preserves generated settings");
        require(a.vram.at("verifier")>16ull<<20,"mandatory verifier is priced");
        require(a.ram.at("streamed_kv")==0,"8K fully resident KV");
        require(a.settings.slots.size()>1 && a.settings.slots[0].bytes!=a.settings.slots[1].bytes,"variable aligned ordered slots");
        require(resolve_launch(r,resources,b,error) && a.payload==b.payload,"fixed inputs resolve deterministically");
        auto too_many_shards = r;
        too_many_shards.shards.assign(core::ownership_shard_limit + 1, r.shards.front());
        require(refusal(too_many_shards,resources,"ownership shard bound refuses before reading files").find("shard bound")!=std::string::npos,
                "resolver uses shared ownership transport bound");
        const auto index_path = root / "index.txt";
        std::ifstream index_input(index_path);
        const std::string original_index((std::istreambuf_iterator<char>(index_input)), {});
        index_input.close();
        for (const char* suffix : {"key", "value", "norm_key", "norm_query", "norm_conv", "conv1d"}) {
            auto changed = original_index;
            const auto start = changed.find(std::string("blk.1.ple_") + suffix + ".weight ");
            require(start != std::string::npos, "positive fixture has compulsory PLE weight");
            changed.erase(start, changed.find('\n', start) - start + 1);
            { std::ofstream out(index_path); out << changed; }
            refusal(r, resources, "missing compulsory PLE metadata refuses");
        }
        { std::ofstream out(index_path); out << original_index; }
        const auto conv_at = original_index.find("blk.1.ple_conv1d.weight 0 5 ");
        require(conv_at != std::string::npos, "raw F16 fixture row is explicit");
        auto promoted = original_index;
        promoted[conv_at + std::string("blk.1.ple_conv1d.weight 0 ").size()] = '3';
        { std::ofstream out(index_path); out << promoted; }
        const auto storage_error = refusal(r, resources, "promoted F16 source width mismatch refuses");
        require(storage_error.find("promoted tensor") != std::string::npos, "shared storage check supplies refusal");
        const auto width_at = promoted.find(" 20480 ", conv_at);
        promoted.replace(width_at, 7, " 40960 ");
        { std::ofstream out(index_path); out << promoted; }
        require(resolve_launch(r, resources, b, error), "valid promoted F16 source remains supported");
        { std::ofstream out(index_path); out << original_index; }
        const auto expert_path = root / "experts.bin";
        const auto expert_size = fs::file_size(expert_path);
        for (uint64_t size : {expert_size - 1, expert_size + 1}) {
            fs::resize_file(expert_path, size);
            const auto size_error = refusal(r, resources, "expert file requires exact layout length");
            require(size_error.find("loaded expert layout requires") != std::string::npos, "shared expert size refusal");
        }
        fs::resize_file(expert_path, expert_size);
        require(resolve_launch(r, resources, b, error), "exact expert file length remains supported");
        const auto storage_dir = root / "storage";
        fs::create_directories(storage_dir);
        auto storage = [&](const char* row, bool expected) {
            { std::ofstream out(storage_dir / "index.txt"); out << row << '\n'; }
            core::WeightTable table;
            require(table.read_metadata(storage_dir.string(), error) == expected, "shared tensor storage segment validation");
        };
        storage("p 0 3 0 16 0 8 4 1 0 0 0 0 0 0 0 0 0 0", true);
        storage("p 0 3 0 8 0 8 4 1 0 0 0 0 0 0 0 0 0 0", false);
        storage("raw 0 5 0 8 0 8 4 1 0 0 0 0 0 0 0 0 0 0", true);
        storage("q 0 0 0 12 0 16 4 1 2 0 2 0 1 4 8 4 1 0", true);
        storage("q 0 0 0 9 0 10 4 1 2 0 2 0 1 4 2 4 1 0", false);
        storage("q 0 0 0 11 0 14 4 1 2 0 2 0 1 4 6 4 1 0", false);
        storage("q 0 0 0 12 0 15 4 1 2 0 2 0 1 4 7 4 1 0", false);
        storage("q 0 0 0 13 0 16 4 1 2 0 2 0 1 4 8 4 1 0", false);
        storage("copy 0 2 0 512 0 512 128 1 0 0 0 0 0 0 0 0 0 0", true);
        storage("copy 0 2 0 512 0 256 128 1 0 0 0 0 0 0 0 0 0 0", false);
        storage("copy 0 2 0 256 0 512 128 1 0 0 0 0 0 0 0 0 0 0", false);
        fixture(root / "no-ple", false, false, false);
        auto variant = r;
        variant.pack = (root / "no-ple").string(); variant.shards = {(root / "no-ple/model.gguf").string()};
        variant.profile = (root / "no-ple/profile.bin").string();
        refusal(variant, resources, "missing PLE table refuses");
        fixture(root / "gguf", true);
        variant.pack = (root / "gguf").string(); variant.shards = {(root / "gguf/model.gguf").string()};
        variant.profile = (root / "gguf/profile.bin").string();
        require(resolve_launch(variant, resources, b, error), error.c_str());
        fixture(root / "wrong-offset", true, true);
        variant.pack = (root / "wrong-offset").string(); variant.shards = {(root / "wrong-offset/model.gguf").string()};
        variant.profile = (root / "wrong-offset/profile.bin").string();
        const auto mismatch = refusal(variant, resources, "expert GGUF tensor offset disagreement refuses");
        require(mismatch.find("starts at byte") != std::string::npos, "expert span checker supplies refusal");
        const char* compact_env = std::getenv("STRATA_FILE_CACHE_COMPACT");
        const bool compact_was_set = compact_env != nullptr;
        const std::string saved_compact = compact_env ? compact_env : "";
        environment("STRATA_FILE_CACHE_COMPACT", "2");
        refusal(r, resources, "unsupported compact environment refuses");
        for (const char* value : {"0", "1"}) {
            environment("STRATA_FILE_CACHE_COMPACT", value);
            require(resolve_launch(r, resources, b, error), "supported compact mode resolves");
        }
        environment("STRATA_FILE_CACHE_COMPACT", compact_was_set ? saved_compact.c_str() : nullptr);
        LaunchSettings decoded;
        require(decode_launch_settings(a.payload,decoded,error) && decoded==a.settings,"exact codec round trip");
        auto old_schema=a.payload; old_schema.replace(0,8,"schema=1");
        require(!decode_launch_settings(old_schema,decoded,error) && error=="unsupported native schema","old native schema fails closed explicitly");
        for (const char* ring : {"0", "9", "15", "600"}) {
            auto payload = a.payload;
            const auto at = payload.find("\nring=") + 1;
            payload.replace(at, payload.find('\n', at) - at, std::string("ring=") + ring);
            require(!decode_launch_settings(payload, decoded, error), "impossible ring payload refuses");
        }
        for (uint64_t ring : {8u, 16u, 512u}) {
            auto valid = a.settings; valid.ring = ring;
            require(decode_launch_settings(encode_launch_settings(valid), decoded, error) && decoded.ring == ring,
                    "supported staging and streaming ring bounds round trip");
        }
        auto reordered=a.settings;
        std::swap(reordered.slots[0],reordered.slots[1]);
        uint64_t off=0;
        for (auto& slot:reordered.slots) { slot.offset=off; off+=launch_align(slot.bytes,256); }
        const auto other=encode_launch_settings(reordered);
        require(other!=a.payload && reordered.slots.size()==a.settings.slots.size(),"same-count layouts retain order");
        require(decode_launch_settings(other,decoded,error) && decoded==reordered,"second ordered layout round trip");
        require(admit_launch(r,resources,a.settings,error),"saved 8K settings admitted without resolving autos");
        require(core::detail::admit_exact_resident_budget(22ull<<30,30ull<<30,30ull<<30,8ull<<30,error),"exact resident budget fits boundary");
        require(!core::detail::admit_exact_resident_budget(22ull<<30,(30ull<<30)-1,30ull<<30,8ull<<30,error),"resident clamp refuses physical shortfall");
        require(!core::detail::admit_exact_resident_budget(22ull<<30,30ull<<30,(30ull<<30)-1,8ull<<30,error),"resident clamp refuses commit shortfall");
        require(!core::detail::admit_exact_resident_budget(0,7ull<<30,30ull<<30,8ull<<30,error),"zero file cache cannot waive physical headroom");
        require(!core::detail::admit_exact_resident_budget(0,30ull<<30,7ull<<30,8ull<<30,error),"zero file cache cannot waive commit headroom");
        require(core::detail::admit_exact_pinning(1024,1024,error),"full resident pin accepted");
        require(!core::detail::admit_exact_pinning(1024,0,error),"resident pageable fallback refused");
        require(!core::detail::admit_exact_pinning(1024,1023,error),"partial resident pin refused");
        require(core::detail::file_cache_target(1000,true,true,10,256,2560).bytes==1000,"compact source target stays exact");
        require(core::detail::file_cache_target(1000,false,true,10,256,2560).bytes==768,"uniform source target rounds at the real owner");
        require(core::detail::file_cache_target(1000,true,false,10,256,2560).bytes==0,"mapped contiguous source has no assembly cache");
        require(core::detail::file_cache_target(4096,true,true,10,256,2560).bytes==2560,"source inventory bounds compact retention");
        prefill::Prefill::Sizing cold_stager;
        prefill::Prefill::resolve_stager(cold_stager,true);
        require(cold_stager.stager_threads==32 && cold_stager.stager_ring==128 && cold_stager.stager_batch==1,
                "metadata captures existing cold file-source host stager policy");
        require(a.ram.at("prefill_host_stager")==42598400,"contiguous source host ring is priced independently");
        environment("STRATA_STAGER_RING","256");
        require(!admit_launch(r,resources,a.settings,error),"host stager environment drift refuses replay");
        require(resolve_launch(r,resources,b,error),"explicit host stager ring can be planned");
        require(b.settings.stager_ring==256 && b.ram.at("prefill_host_stager")==681574400,
                "explicit host ring uses its actual source-owned buffer cost");
        environment("STRATA_STAGER_RING",nullptr);
        require(admit_launch(r,resources,reordered,error),"same-count second slot order admitted unchanged");
        consume_slots(a.settings); consume_slots(reordered);
        auto per_layer_request=r; per_layer_request.per_layer=true;
        LaunchResolution per_layer_plan;
        require(resolve_launch(per_layer_request,resources,per_layer_plan,error),"per-layer layout resolves at existing owner");
        require(per_layer_plan.settings.slots.size()%48==0 && per_layer_plan.settings.per_layer==1,"resolved per-layer quotas are explicit");
        require(admit_launch(per_layer_request,resources,per_layer_plan.settings,error),"per-layer layout replays exactly");
        consume_slots(per_layer_plan.settings);
        auto explicit_request=r; explicit_request.expert_cache=600;
        LaunchResolution explicit_plan;
        require(resolve_launch(explicit_request,resources,explicit_plan,error),"explicit uniform grant resolves variable aligned slots");
        require(explicit_plan.settings.auto_cache==0 && explicit_plan.settings.cache_request==600 && explicit_plan.settings.slots.size()>600,
                "original explicit cache grant is distinct from applied slot count");
        require(admit_launch(explicit_request,resources,explicit_plan.settings,error),"explicit grant replays saved variable slots without reconstruction");
        consume_slots(explicit_plan.settings);
        auto more=resources; more.vram_free=12ull<<30;
        require(admit_launch(r,more,a.settings,error),"more current VRAM does not grow saved slots");
        require(check_effective_launch(a.settings,a.settings,error),"exact applied settings pass READY gate");
        require(!check_effective_launch(a.settings,reordered,error),"same-count changed effective order blocks READY");
        const auto effective_payload=encode_effective_launch(a.settings);
        require(effective_payload.size()<16384 && effective_payload.starts_with("schema=1\nverified=1\n"),"bounded versioned effective handoff");
        require(effective_payload.find("\nslots=")==std::string::npos && effective_payload.find("resident_order=")==std::string::npos,
                "runtime handoff contains no per-expert slot/member dump");
        require(effective_payload!=encode_effective_launch(reordered),"aggregate handoff distinguishes same-count ordered layouts");
        require(frame_effective_launch(a.settings,1)=="STRATA-EFFECTIVE/1\n1\n"+
                std::to_string(effective_payload.size())+'\n'+effective_payload,"effective frame has coherent sequence and exact byte length");
        for (int change=0;change<14;++change) {
            auto changed=a.settings;
            if (change==0) changed.auto_cache=0;
            if (change==1) changed.resident_budget+=1ull<<30;
            if (change==2) changed.file_cache-=1ull<<20;
            if (change==3) changed.host_headroom+=1ull<<30;
            if (change==4) changed.shared_stage=1;
            if (change==5) changed.workspace=8ull<<20;
            if (change==6) changed.prefill=32;
            if (change==7) changed.batch=2;
            if (change==8) changed.batch_groups=2;
            if (change==9) changed.parallel=2;
            if (change==10) changed.resident_pin=0;
            if (change==11) changed.vram_required+=1;
            if (change==12) changed.environment^=1;
            if (change==13) std::swap(changed.resident_order[0],changed.resident_order[1]);
            require(!admit_launch(r,resources,changed,error),"changed approval refuses native admission");
            require(!check_effective_launch(a.settings,changed,error),"changed actual settings block READY");
        }
        for (int loss=0;loss<4;++loss) {
            auto current=resources;
            if (loss==0) current.vram_free=a.vram_required-1;
            if (loss==1) current.ram_available=a.ram_required-1;
            if (loss==2) current.commit_available=a.commit_required-1;
            if (loss==3) current.devices=2;
            require(!admit_launch(r,current,a.settings,error),"current resource loss refuses without adjustment");
        }
        auto argv_changed=r; argv_changed.expert_cache=100;
        require(!admit_launch(argv_changed,resources,a.settings,error),"original auto-cache provenance is bound");
        environment("STRATA_TEST_STRICT_ENV","changed");
        require(!admit_launch(r,resources,a.settings,error),"native environment drift refuses");
        environment("STRATA_TEST_STRICT_ENV",nullptr);
        for (const std::string bad_value:{"-1","+2","02","18446744073709551616"}) {
            auto bytes=a.payload;
            const auto at=bytes.find("spec=2\n");
            bytes.replace(at,7,"spec="+bad_value+'\n');
            require(!decode_launch_settings(bytes,decoded,error),"invalid numeric grammar refused");
        }
        const auto frame=frame_launch_settings(a.payload);
        std::istringstream in(frame+"QUIT\n"); std::string payload,command;
        require(read_launch_frame(in,payload,error) && payload==a.payload,"framed bytes round trip");
        std::getline(in,command); require(command=="QUIT","initial frame leaves engine commands unread");
        for (const auto& invalid:std::vector<std::string>{"",a.payload.substr(0,a.payload.size()-1),a.payload+"context=8192\n",
                a.payload+"unknown=1\n","schema=2\n",std::string(launch_payload_limit+1,'x')})
            require(!decode_launch_settings(invalid,decoded,error),"malformed/duplicate/unknown/oversized payload refused");
        auto invalid=a.settings; invalid.slots[0].offset=256;
        require(!validate_launch_settings(invalid,error),"invalid layout refused");
        invalid=a.settings; invalid.slots[1]=invalid.slots[0];
        require(!validate_launch_settings(invalid,error),"duplicate slot refused");
        for (const auto& bytes:std::vector<std::string>{frame.substr(0,frame.size()-1),"STRATA-PLAN/1\n1048577\n",
                "STRATA-PLAN/1\n-1\n","STRATA-PLAN/2\n1\nx","STRATA-PLAN/1\n01\nx"}) {
            std::istringstream bad(bytes); require(!read_launch_frame(bad,payload,error),"invalid/truncated frame refused");
        }
        bool overflow=false;
        try { (void) launch_mul(UINT64_MAX,2); } catch (const std::overflow_error&) { overflow=true; }
        require(overflow,"checked multiplication"); overflow=false;
        try { (void) launch_align(UINT64_MAX,256); } catch (const std::overflow_error&) { overflow=true; }
        require(overflow,"checked alignment");
        auto short_resources=resources; short_resources.vram_free=1;
        refusal(r,short_resources,"resource loss refuses");
        short_resources=resources; short_resources.commit_available=1;
        refusal(r,short_resources,"commit shortfall refuses");
        short_resources=resources; short_resources.devices=2;
        refusal(r,short_resources,"unsupported topology refuses");
        auto missing=r; missing.profile=(root/"absent-profile").string(); refusal(missing,resources,"absent metadata refuses");
        missing=r; missing.context=UINT64_MAX; refusal(missing,resources,"overflow request refuses before metadata");
        missing=r; missing.pack=(root/"missing-pack").string(); refusal(missing,resources,"missing pack refuses");
        missing=r; missing.no_prefill_borrow=false; refusal(missing,resources,"unpriced borrowing refuses");
        missing=r; missing.expert_cache=INT64_MAX; refusal(missing,resources,"expert budget overflow refuses");
        r.context=131072; r.kv=2; r.kv_resident=20480; r.vram_reserve=700ull<<20; r.reserve_given=false;
        r.resident_budget=21ull<<30; r.file_cache=9816ull<<20;
        resolved=resolve_launch(r,resources,b,error);
        require(resolved,error.c_str());
        require(b.settings.context==131072 && b.settings.kv==2 && b.settings.kv_resident==20480 &&
                b.settings.spec==2 && b.settings.no_prefill_borrow==1,"128K preserves spec2/q4_0/20480/owned prefill");
        require(b.settings.kv_stage_runs==std::vector<uint64_t>{37748736,37748736},"128K stage has two q4 pools");
        require(b.ram.at("streamed_kv")==905969664,"128K logical host KV priced independently");
        require(admit_launch(r,resources,b.settings,error),"saved 128K streamed q4 spec2 settings admitted exactly");
        for (uint64_t resident : {20481u, 20483u, 131072u, 131073u}) {
            auto bad_request = r; bad_request.kv_resident = resident;
            refusal(bad_request,resources,"nonreplayable KV residency refuses before metadata allocation");
            auto bad_settings = b.settings; bad_settings.kv_resident = resident;
            require(!validate_launch_settings(bad_settings,error),"nonreplayable KV residency refuses codec validation");
            auto bad_payload = b.payload;
            const auto at = bad_payload.find("kv_resident=20480\n");
            require(at != std::string::npos,"streamed fixture carries concrete KV residency");
            bad_payload.replace(at, std::string("kv_resident=20480\n").size(),
                                "kv_resident=" + std::to_string(resident) + '\n');
            require(!decode_launch_settings(bad_payload,decoded,error),"nonreplayable saved KV residency refuses decoding");
        }
        for (uint64_t resident : {0u, 20480u, 20484u}) {
            auto exact_request = r; exact_request.kv_resident = resident;
            LaunchResolution exact_plan;
            require(resolve_launch(exact_request,resources,exact_plan,error),"canonical KV residency resolves");
            require(exact_plan.settings.kv_resident == resident &&
                    admit_launch(exact_request,resources,exact_plan.settings,error),"canonical KV residency replays unchanged");
        }
        core::ModelGeometry g; core::SessionSizing kv;
        uint64_t mapped1=0,mapped2=0;
        const auto verify1=core::Verifier::workspace_bytes(g,8192,10,2,248320,1ull<<20,mapped1);
        const auto verify2=core::Verifier::workspace_bytes(g,8192,10,2,248320,2ull<<20,mapped2);
        require(verify2-verify1==16ull<<20 && mapped1==mapped2,"verifier counts its sixteen expert staging blobs");
        core::qsa_set_kv_int8(true);
        require(core::session_bytes_for(g,8192,10,kv)==core::session_bytes(g,8192,10),"explicit session count matches current owner");
        auto sizing=prefill::Prefill::sizing_inputs(g,131072,64,{core::SessionSizing::Kv::Q4,20480,false});
        core::QsaState q; q.max_cells=131072; q.kv_q4=true; q.kv_mode=1; q.n_slots=20480/4;
        core::SessionState ss; ss.qsa_states=&q; ss.max_cells=131072;
        require(prefill::Prefill::bytes_needed_for(sizing,true)==prefill::Prefill::bytes_needed_owned(g,ss,64),
                "prefill explicit inventory matches current owner");
        auto large = prefill::Prefill::sizing_inputs(g, 131072, 65536, {core::SessionSizing::Kv::Q4,20480,false});
        require(prefill::Prefill::bytes_needed_for(large, true) > prefill::Prefill::bytes_needed_for(sizing, true),
                "ordinary explicit chunk above planned cap remains countable");
        require(prefill::Prefill::bytes_needed_for(large, true) == prefill::Prefill::bytes_needed_owned(g, ss, 65536),
                "ordinary owned pricing accepts the larger chunk");
        auto oversized = a.settings; oversized.prefill = 65536; oversized.context = 131072;
        require(!validate_launch_settings(oversized, error), "planned chunk cap stays in planned validation");
        std::cout << "PASS launch_plan_test checks=" << checks << " slots8k=" << a.settings.slots.size()
                  << " slots128k=" << b.settings.slots.size() << " vram128k=" << b.vram_required << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL launch_plan_test after " << checks << ": " << e.what() << '\n'; return 1; }
}
