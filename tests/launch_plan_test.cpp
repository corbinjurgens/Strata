#include "strata/plan/launch.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/session.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/prefill/moe_mmq.hpp"
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL)
#include <cublas_v2.h>
#endif

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
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
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__)) && \
    !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL)
struct DiscoveryControl {
    std::string events, fail;
    int devices=1;
    bool alive=false, inconsistent=false;
} discovery;
void discovery_event(const char* name) { discovery.events+=std::string(name)+";"; }
cudaError_t CUDARTAPI discovery_devices(int* devices) {
    discovery_event("devices"); *devices=discovery.devices;
    return discovery.fail=="devices" ? cudaErrorInvalidValue : cudaSuccess;
}
cudaError_t CUDARTAPI discovery_sync() {
    require(discovery.alive,"BLAS handle stays alive during synchronization");
    discovery_event("sync");
    return discovery.fail=="sync" ? cudaErrorInvalidValue : cudaSuccess;
}
cudaError_t CUDARTAPI discovery_memory(size_t* free, size_t* total) {
    require(discovery.alive,"BLAS handle stays alive during the resource snapshot");
    require(discovery.events=="devices;create;stream;workspace;math;sync;","required BLAS setup precedes snapshot");
    discovery_event("sample"); *free=8ull<<30; *total=discovery.inconsistent ? 1 : 16ull<<30;
    return discovery.fail=="sample" ? cudaErrorInvalidValue : cudaSuccess;
}
const char* CUDARTAPI discovery_error(cudaError_t) { return "injected CUDA failure"; }
cublasStatus_t CUBLASWINAPI discovery_create(cublasHandle_t* handle) {
    require(discovery.events=="devices;","topology validated before BLAS setup");
    discovery_event("create");
    if (discovery.fail=="create") return CUBLAS_STATUS_ALLOC_FAILED;
    *handle=reinterpret_cast<cublasHandle_t>(1); discovery.alive=true;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t CUBLASWINAPI discovery_destroy(cublasHandle_t handle) {
    require(discovery.alive && handle==reinterpret_cast<cublasHandle_t>(1),"real GEMM owner releases its handle once");
    discovery_event("destroy"); discovery.alive=false;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t CUBLASWINAPI discovery_stream(cublasHandle_t, cudaStream_t stream) {
    require(discovery.alive && stream==nullptr,"library-only discovery uses the default stream");
    discovery_event("stream");
    return discovery.fail=="stream" ? CUBLAS_STATUS_INVALID_VALUE : CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t CUBLASWINAPI discovery_workspace(cublasHandle_t, void* workspace, size_t bytes) {
    require(discovery.alive && workspace==nullptr && bytes==0,"discovery allocates no model workspace");
    discovery_event("workspace");
    return discovery.fail=="workspace" ? CUBLAS_STATUS_INVALID_VALUE : CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t CUBLASWINAPI discovery_math(cublasHandle_t, cublasMath_t mode) {
    require(discovery.alive && mode==CUBLAS_DEFAULT_MATH,"discovery uses production GEMM math setup");
    discovery_event("math");
    return discovery.fail=="math" ? CUBLAS_STATUS_INVALID_VALUE : CUBLAS_STATUS_SUCCESS;
}
void (*discovery_override(const char* name))() {
#define DISCOVERY_OVERRIDE(symbol, replacement) \
    if (std::strstr(name,#symbol)) return reinterpret_cast<void(*)()>(&replacement)
    DISCOVERY_OVERRIDE(cudaGetDeviceCount,discovery_devices);
    DISCOVERY_OVERRIDE(cudaDeviceSynchronize,discovery_sync);
    DISCOVERY_OVERRIDE(cudaMemGetInfo,discovery_memory);
    DISCOVERY_OVERRIDE(cudaGetErrorString,discovery_error);
    DISCOVERY_OVERRIDE(cublasCreate,discovery_create);
    DISCOVERY_OVERRIDE(cublasDestroy,discovery_destroy);
    DISCOVERY_OVERRIDE(cublasSetStream,discovery_stream);
    DISCOVERY_OVERRIDE(cublasSetWorkspace,discovery_workspace);
    DISCOVERY_OVERRIDE(cublasSetMathMode,discovery_math);
#undef DISCOVERY_OVERRIDE
    return nullptr;
}
[[noreturn]] void forbidden_device_call() {
    std::cerr << "FAIL CPU launch test reached a device/runtime call\n";
    std::abort();
}
void forbid_device_calls(bool discovery_mode=false) {
    // This build may link cudart statically, so guard its entry points as well as DLL imports.
    auto guard=[](auto function, void (*failure)()) {
        unsigned char jump[12]={0x48,0xb8}; // mov rax, address; jmp rax (Windows x64)
        const auto destination=reinterpret_cast<uintptr_t>(failure);
        std::memcpy(jump+2,&destination,sizeof destination); jump[10]=0xff; jump[11]=0xe0;
        auto* address=reinterpret_cast<void*>(function); DWORD old=0,ignored=0;
        require(VirtualProtect(address,sizeof jump,PAGE_EXECUTE_READWRITE,&old)!=0,"guard static runtime entry");
        std::memcpy(address,jump,sizeof jump);
        require(VirtualProtect(address,sizeof jump,old,&ignored)!=0,"restore static runtime entry protection");
        require(FlushInstructionCache(GetCurrentProcess(),address,sizeof jump)!=0,"publish static runtime guard");
    };
#define GUARD_DEVICE(function) guard(function,discovery_mode && discovery_override(#function) ? discovery_override(#function) : \
    +[]{ std::cerr << "FAIL CPU launch test reached " #function "\n"; std::abort(); })
    GUARD_DEVICE(&cudaGetDeviceCount); GUARD_DEVICE(&cudaMemGetInfo); GUARD_DEVICE(&cudaGetDevice); GUARD_DEVICE(&cudaSetDevice);
    GUARD_DEVICE(&cudaDeviceSynchronize); GUARD_DEVICE(&cudaGetErrorString);
    GUARD_DEVICE(&cudaGetDeviceProperties);
    GUARD_DEVICE(&cudaDeviceGetAttribute);
    GUARD_DEVICE((static_cast<cudaError_t(CUDARTAPI*)(void**,size_t)>(&cudaMalloc)));
    GUARD_DEVICE((static_cast<cudaError_t(CUDARTAPI*)(void**,size_t)>(&cudaMallocHost)));
    GUARD_DEVICE((static_cast<cudaError_t(CUDARTAPI*)(void**,size_t,unsigned)>(&cudaHostAlloc)));
    GUARD_DEVICE(&cudaHostRegister); GUARD_DEVICE(&cudaFree); GUARD_DEVICE(&cudaFreeHost);
#undef GUARD_DEVICE
    // Static libraries use this executable's imports. Fail before CUDA/BLAS can observe or allocate a device.
    auto* base=reinterpret_cast<unsigned char*>(GetModuleHandleW(nullptr));
    const auto* dos=reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const auto* nt=reinterpret_cast<const IMAGE_NT_HEADERS*>(base+dos->e_lfanew);
    const auto directory=nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    auto* dll=reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base+directory.VirtualAddress);
    size_t guarded=0;
    for (;dll->Name;++dll) {
        if (!dll->OriginalFirstThunk) continue;
        auto* names=reinterpret_cast<IMAGE_THUNK_DATA*>(base+dll->OriginalFirstThunk);
        auto* addresses=reinterpret_cast<IMAGE_THUNK_DATA*>(base+dll->FirstThunk);
        for (;names->u1.AddressOfData;++names,++addresses) {
            if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) continue;
            const auto* entry=reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(base+names->u1.AddressOfData);
            const char* name=reinterpret_cast<const char*>(entry->Name);
            if (std::strncmp(name,"cuda",4) && std::strncmp(name,"cublas",6) && std::strcmp(name,"GlobalMemoryStatusEx")) continue;
            DWORD old=0,ignored=0;
            require(VirtualProtect(&addresses->u1.Function,sizeof addresses->u1.Function,PAGE_READWRITE,&old)!=0,
                    "guard runtime import");
            const auto replacement=discovery_mode ? discovery_override(name) : nullptr;
            addresses->u1.Function=reinterpret_cast<ULONG_PTR>(replacement ? replacement : &forbidden_device_call);
            require(VirtualProtect(&addresses->u1.Function,sizeof addresses->u1.Function,old,&ignored)!=0,
                    "restore runtime import protection");
            ++guarded;
        }
    }
    require(guarded>0,"device/runtime imports guarded for CPU test");
}
void discovery_controls() {
    // Tracing would add extra samples; these controls exercise the admission snapshot alone.
    _putenv_s("STRATA_TRACE","");
    auto resources=[] { return LaunchResources{0,48ull<<30,96ull<<30,1,true}; };
    std::string error;
    discovery={}; auto live=resources();
    require(discover_launch_resources(live,error),"live discovery succeeds through real GEMM owner");
    require(live.vram_free==8ull<<30 && live.vram_total==16ull<<30 && live.devices==1,"live snapshot is used without fixed deduction");
    require(discovery.events=="devices;create;stream;workspace;math;sync;sample;destroy;" && !discovery.alive,
            "snapshot covers complete BLAS setup and owner cleanup follows sample");
    for (const auto& test:std::vector<std::pair<std::string,std::string>>{
            {"devices","devices;"}, {"create","devices;create;"},
            {"stream","devices;create;stream;destroy;"},
            {"workspace","devices;create;stream;workspace;destroy;"},
            {"math","devices;create;stream;workspace;math;destroy;"},
            {"sync","devices;create;stream;workspace;math;sync;destroy;"},
            {"sample","devices;create;stream;workspace;math;sync;sample;destroy;"}}) {
        discovery={}; discovery.fail=test.first; live=resources(); error.clear();
        require(!discover_launch_resources(live,error),"required library/discovery failures refuse admission");
        require(!error.empty() && live.vram_free==0 && !discovery.alive && discovery.events==test.second,
                "failure preserves no snapshot, skips later work and releases the real owner");
        if (test.first=="create") require(error.find("cublasCreate: cuBLAS status 3")!=std::string::npos,"creation status preserved");
        if (test.first=="workspace") require(error.find("cublasSetWorkspace: cuBLAS status 7")!=std::string::npos,"workspace status preserved");
        if (test.first=="math") require(error.find("cublasSetMathMode: cuBLAS status 7")!=std::string::npos,"math status preserved");
    }
    discovery={}; discovery.devices=2; live=resources();
    require(!discover_launch_resources(live,error) && discovery.events=="devices;","unsupported topology never initializes BLAS");
    discovery={}; discovery.inconsistent=true; live=resources();
    require(!discover_launch_resources(live,error) && !discovery.alive && live.vram_free==0,"inconsistent sample still releases handle");
    discovery={}; live={123,456,789,1,true};
    require(discover_launch_resources(live,error) && discovery.events.empty() && live.vram_free==123 &&
            live.ram_available==456 && live.commit_available==789,"injected discovery values stay exact without BLAS");
    std::cout << "PASS launch_discovery_owner checks=" << checks << '\n';
}
#endif
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
void fixture(const fs::path& root, bool gguf_experts = false, bool wrong_offset = false, bool ple = true, int first_type = 18) {
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
        const int type = l==0 ? first_type : l % 2 ? 23 : 18;
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
        const int type = l==0 ? first_type : l%2 ? 23 : 18;
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
    LaunchResolution untouched; untouched.payload="sentinel"; untouched.settings.context=123;
    untouched.vram={{"sentinel",456}}; untouched.ram={{"sentinel",789}};
    untouched.vram_required=1; untouched.ram_required=2; untouched.commit_required=3;
    const auto before=untouched; std::string error;
    require(!resolve_launch(r,resources,untouched,error),label);
    require(untouched.payload==before.payload && untouched.settings==before.settings && untouched.vram==before.vram &&
            untouched.ram==before.ram && untouched.vram_required==before.vram_required &&
            untouched.ram_required==before.ram_required && untouched.commit_required==before.commit_required &&
            untouched.unknown_costs==before.unknown_costs && !error.empty(),"failed resolution preserves complete unpublished state");
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
void runtime_controls(const LaunchSettings& saved, uint64_t context, uint64_t kv, uint64_t resident) {
    uint64_t capacity=0;
    for (const auto& slot : saved.slots) capacity+=(slot.bytes/256+(slot.bytes%256!=0))*256;
    // These are applied-owner values, deliberately no slots, metadata, approved
    // fingerprint, required-memory estimates or validated plan in ordinary mode.
    LaunchSettings ordinary;
    ordinary.context=context; ordinary.prefill=64; ordinary.spec=2; ordinary.kv=kv; ordinary.kv_resident=resident;
    ordinary.batch=0; ordinary.batch_groups=1; ordinary.parallel=1; ordinary.no_prefill_borrow=1;
    ordinary.auto_cache=saved.auto_cache; ordinary.per_layer=saved.per_layer;
    ordinary.resident_budget=saved.resident_budget; ordinary.resident_bytes=saved.resident_bytes;
    ordinary.resident_experts=saved.resident_experts; ordinary.resident_pin=1;
    ordinary.file_cache_effective=saved.file_cache_effective;
    std::string error;
    require(!validate_launch_settings(ordinary,error),"ordinary telemetry specimen cannot be misused as approved payload");
    const auto actual=runtime_launch_settings(ordinary,saved.slots.size(),capacity);
    require(actual.at("context")==context && actual.at("prefill")==64 && actual.at("spec")==2 && actual.at("kv")==kv &&
            actual.at("kv_resident")==resident && actual.at("parallel")==1 && actual.at("batch")==0 &&
            actual.at("resident_bytes")==saved.resident_bytes && actual.at("cache_bytes")==capacity,
            "ordinary generated profile reports concrete applied settings without admission or payload");
    core::RuntimeSnapshot runtime; runtime.settings=actual;
    auto& cache=runtime.vram["expert_cache"]; cache.device_allocation(capacity);
    const uint64_t pages=capacity/2097152+(capacity%2097152!=0);
    require(cache.requested==capacity && cache.rounded==pages*2097152,"runtime component rounds one actual arena independently");
    uint64_t sequence=0; std::string frame;
    require(core::frame_runtime_snapshot(runtime,"cuda",0,"cpu-fixture",sequence,frame,error) && sequence==1 &&
            frame.find("setting\tcontext\t"+std::to_string(context)+"\n")!=std::string::npos,
            "ordinary generated snapshot publishes through real bounded codec");
    const auto strict=runtime_launch_settings(saved,saved.slots.size(),capacity);
    require(strict.at("context")==context && strict.at("kv")==kv && strict.at("kv_resident")==resident &&
            strict.at("prefill")==64 && strict.at("cache_bytes")==capacity,
            "strict generated snapshot uses same applied-settings codec");
    require(frame_effective_launch(saved,1).rfind("STRATA-EFFECTIVE/1\n1\n",0)==0,
            "existing strict effective framing remains compatible");
}
}

int main(int argc, char** argv) {
    try {
        const bool guarded_mode=argc>1 && std::string(argv[1]).starts_with("--");
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__)) && \
    !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL)
        if (guarded_mode) forbid_device_calls(std::string(argv[1])=="--discovery-owner");
        if (argc==2 && std::string(argv[1])=="--discovery-owner") { discovery_controls(); return 0; }
#elif defined(_WIN32)
        if (guarded_mode)
            throw std::runtime_error("CPU runtime guards require Windows x64 CUDA; no unguarded execution on this build");
#else
        (void)guarded_mode;
#endif
        if (argc>1 && std::string(argv[1])=="--resolve-launch") return launch_metadata_main(argc,argv);
        if (argc==2 && std::string(argv[1])=="--codec-roundtrip") {
            std::string payload,error; LaunchSettings saved;
            require(read_launch_stdin(payload,error),error.c_str());
            require(decode_launch_settings(payload,saved,error),error.c_str());
            require(encode_launch_settings(saved)==payload,"frozen payload bytes changed in native codec");
            for (int change=0;change<4;++change) {
                auto invalid=saved;
                if (change==0) invalid.slots.clear();
                if (change==1) invalid.slots[0].offset=256;
                if (change==2) invalid.ring=9;
                if (change==3) invalid.resident_order.push_back(0);
                require(!validate_launch_settings(invalid,error),"frozen codec keeps strict layout and inventory rejection invariants");
            }
            std::cout << "PASS frozen_native_codec exact_roundtrip=1 bytes=" << payload.size() << " checks=" << checks << '\n';
            return 0;
        }
        if (argc>1 && (std::string(argv[1])=="--consume" || std::string(argv[1])=="--consume-ample")) {
            std::string error;
            LaunchSettings saved;
            const bool ample=std::string(argv[1])=="--consume-ample";
            const auto observe_ample=[](LaunchResources& resources,std::string&) {
                resources={24ull<<30,48ull<<30,96ull<<30,1,true}; return true;
            };
            const auto observe=[](LaunchResources& resources,std::string&) {
                resources={6ull<<30,48ull<<30,48ull<<30,1,true}; return true;
            };
            const bool consumed=consume_strict_launch(argc-1,argv+1,saved,error,ample ? +observe_ample : +observe);
            require(consumed,error.c_str());
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
        const bool pure_mode=argc>1 && std::string(argv[1])=="--pure";
        const auto root=pure_mode && argc>2 ? fs::path(argv[2]) : argc>1 && !pure_mode ? fs::path(argv[1])
                           : fs::temp_directory_path()/(pure_mode ? "strata-launch-price-test" : "strata-launch-plan-test");
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
        bool resolved=false;
        if (!pure_mode) {
        resolved=resolve_launch(r,resources,a,error);
        require(resolved,error.c_str());
        require(a.settings.spec==2 && a.settings.no_prefill_borrow==1 && a.settings.prefill==64,"8K preserves generated settings");
        require(a.vram.at("verifier")>16ull<<20,"mandatory verifier is priced");
        require(a.ram.at("streamed_kv")==0,"8K fully resident KV");
        runtime_controls(a.settings,8192,1,0);
        require(a.settings.slots.size()>1 && a.settings.slots[0].bytes!=a.settings.slots[1].bytes,"variable aligned ordered slots");
        require(resolve_launch(r,resources,b,error) && a.payload==b.payload,"fixed inputs resolve deterministically");
        }
        if (pure_mode) {
        auto explicit_request=r; explicit_request.expert_cache=97;
        LaunchPrice explicit_price;
        require(price_launch(explicit_request,nullptr,explicit_price,error),"explicit counts price without observation or budget");
        require(explicit_price.status=="not_evaluated" && explicit_price.configuration_complete &&
                explicit_price.resolution.payload.empty(),"price-only has no approved replay payload");
        LaunchPrice repeat_price;
        require(price_launch(explicit_request,nullptr,repeat_price,error) &&
                repeat_price.resolution.settings==explicit_price.resolution.settings &&
                repeat_price.resolution.vram==explicit_price.resolution.vram && repeat_price.resolution.ram==explicit_price.resolution.ram,
                "pure envelope repeats settings and component prices without observation");
        const auto& priced=explicit_price.resolution;
        LaunchResources exact{priced.vram_required,priced.ram_required,priced.commit_required,1,true};
        LaunchPrice exact_price;
        require(price_launch(explicit_request,&exact,exact_price,error) && exact_price.status=="within_priced_budget" &&
                exact_price.configuration_complete,"exact priced budgets fit explicit counts");
        for (int resource=0;resource<3;++resource) {
            auto short_budget=exact;
            if (resource==0) --short_budget.vram_free;
            if (resource==1) --short_budget.ram_available;
            if (resource==2) --short_budget.commit_available;
            LaunchPrice short_price;
            require(price_launch(explicit_request,&short_budget,short_price,error) && short_price.status=="does_not_fit" &&
                    short_price.vram_shortfall==(resource==0) && short_price.ram_shortfall==(resource==1) &&
                    short_price.commit_shortfall==(resource==2),"independent one-byte resource shortfall is normal pricing");
            require(short_price.resolution.settings==priced.settings && short_price.resolution.vram==priced.vram &&
                    short_price.resolution.ram==priced.ram,"insufficiency retains all explicit component prices");
        }
        for (uint64_t bytes:{0ull,1ull}) {
            LaunchResources tiny{bytes,bytes,bytes,1,true}; LaunchPrice tiny_price;
            require(price_launch(explicit_request,&tiny,tiny_price,error) && tiny_price.status=="does_not_fit" &&
                    tiny_price.vram_shortfall==priced.vram_required-bytes && tiny_price.ram_shortfall==priced.ram_required-bytes &&
                    tiny_price.commit_shortfall==priced.commit_required-bytes,"zero and tiny explicit budgets retain every shortfall");
            if (!bytes) require(!resolve_launch(explicit_request,tiny,b,error),"legacy resolve still refuses missing resource snapshot before sizing");
            for (bool per_layer:{false,true}) {
                auto minimum_request=r; minimum_request.per_layer=per_layer;
                LaunchPrice minimum;
                require(price_launch(minimum_request,&tiny,minimum,error) && !minimum.configuration_complete &&
                        minimum.status=="does_not_fit" && minimum.resolution.payload.empty(),"automatic minimum is priced but incomplete");
                const auto& slots=minimum.resolution.settings.slots;
                require(slots.size()==(per_layer ? 48 : 1) && minimum.resolution.vram.at("expert_cache")>0,
                        "minimum layout respects global or per-layer policy without zero slots");
                for (size_t i=0;i<slots.size();++i)
                    require(slots[i].layer==(per_layer ? i : 0),"minimum preserves profile and per-layer ordering");
            }
        }
        LaunchPrice absent;
        require(!price_launch(r,nullptr,absent,error) && error.starts_with("requires_budget:"),"automatic cache needs budget");
        auto no_budget_auto=explicit_request; no_budget_auto.prefill_auto=true;
        require(!price_launch(no_budget_auto,nullptr,absent,error) && error.starts_with("requires_budget:"),"automatic prefill needs budget");
        auto invalid_price=explicit_request; invalid_price.expert_cache=INT64_MAX;
        require(!price_launch(invalid_price,nullptr,absent,error) && error.find("overflow")!=std::string::npos,
                "explicit cache multiplication overflow is an input error");
        invalid_price=explicit_request; invalid_price.context=UINT64_MAX;
        require(!price_launch(invalid_price,nullptr,absent,error),"maximum unsupported context is an input error");
        invalid_price=explicit_request; invalid_price.expert_cache=0;
        require(!price_launch(invalid_price,nullptr,absent,error),"explicit zero slots remain unsupported");
        invalid_price=r; invalid_price.vram_reserve=1ull<<40;
        LaunchResources zero_inventory{0,0,0,1,true};
        require(!price_launch(invalid_price,&zero_inventory,absent,error),
                "incomplete automatic minimum still enforces inventory bounds without replay validation");
        for (int invalid=0;invalid<4;++invalid) {
            auto request=explicit_request;
            if (invalid==0) request.no_prefill_borrow=false;
            if (invalid==1) request.spec=3;
            if (invalid==2) request.kv=4;
            if (invalid==3) request.resident_budget=0;
            absent.status="sentinel";
            require(!price_launch(request,&resources,absent,error) && absent.status=="sentinel" && !error.empty(),
                    "unsupported requests remain errors without publishing a price");
        }
        auto multi_device=resources; multi_device.devices=2;
        require(!price_launch(explicit_request,&multi_device,absent,error),"pure pricing refuses unsupported device topology");
        LaunchResources maximum{UINT64_MAX,UINT64_MAX,UINT64_MAX,1,true}; LaunchPrice maximum_price;
        require(price_launch(r,&maximum,maximum_price,error) && maximum_price.status=="within_priced_budget" &&
                maximum_price.resolution.settings.slots.size()==48*512,"maximum budgets bound automatic selection by metadata inventory");
        bool overflow=false;
        try { (void)launch_add(UINT64_MAX,1); } catch (const std::overflow_error&) { overflow=true; }
        require(overflow,"pure size addition checks overflow"); overflow=false;
        try { (void)launch_align(UINT64_MAX,256); } catch (const std::overflow_error&) { overflow=true; }
        require(overflow,"pure size alignment checks overflow");
        auto high_budget=resources; high_budget.vram_free=6207569920;
        auto low_budget=resources; low_budget.vram_free=5326766080;
        LaunchPrice high_price,low_price;
        require(price_launch(r,&high_budget,high_price,error) && price_launch(r,&low_budget,low_price,error) &&
                high_price.status=="within_priced_budget" && low_price.status=="within_priced_budget" &&
                high_price.configuration_complete && low_price.configuration_complete,"both recorded MSI budgets resolve automatic layouts");
        require(high_price.resolution.settings.slots.size()>low_price.resolution.settings.slots.size(),
                "automatic layouts may differ with budget");
        require(high_price.resolution.vram_required>low_budget.vram_free &&
                high_budget.vram_free-low_budget.vram_free==(840ull<<20),
                "frozen higher-budget price exceeds lower capacity; budget delta is not an allocation measurement");
        std::cout << "MSI synthetic budgets high=" << high_budget.vram_free << " low=" << low_budget.vram_free
                  << " high_slots=" << high_price.resolution.settings.slots.size()
                  << " low_slots=" << low_price.resolution.settings.slots.size() << '\n';
        std::vector<std::string> price_words={"strata","--resolve-launch","--price-only","--pack",r.pack,
            "--native",r.shards[0],"--expert-profile",r.profile,"--max-context","8192","--prefill","64",
            "--expert-cache","97","--resident-budget-gib","22","--no-prefill-borrow"};
        auto price_cli=[&](std::vector<std::string> words,int expected) {
            std::vector<char*> args; for (auto& word:words) args.push_back(word.data());
            std::ostringstream output,errors;
            auto* old_out=std::cout.rdbuf(output.rdbuf()); auto* old_err=std::cerr.rdbuf(errors.rdbuf());
            const auto code=launch_metadata_main((int)args.size(),args.data());
            std::cout.rdbuf(old_out); std::cerr.rdbuf(old_err);
            require(code==expected,"metadata CLI exit matches normal insufficiency or invalid input");
            require(expected==0 ? errors.str().empty() : output.str().empty(),"CLI errors do not publish partial JSON");
            return expected==0 ? output.str() : errors.str();
        };
        const auto explicit_json=price_cli(price_words,0);
        require(explicit_json.find("\"kind\":\"native_memory_price\"")!=std::string::npos &&
                explicit_json.find("conservative_prefill_envelope")!=std::string::npos &&
                explicit_json.find("\"budget\":null")!=std::string::npos && explicit_json.find("not_evaluated")!=std::string::npos &&
                explicit_json.find("native_payload")==std::string::npos && explicit_json.find("effective_payload")==std::string::npos,
                "price-only CLI publishes frozen transport without approved payloads");
        auto zero_words=price_words;
        zero_words.insert(zero_words.end(),{"--plan-vram-free-bytes","0","--plan-ram-available-bytes","0",
                                           "--plan-commit-available-bytes","0"});
        const char* modules=std::getenv("CUDA_MODULE_LOADING");
        const bool had_modules=modules!=nullptr; const std::string original_modules=modules ? modules : "";
        environment("CUDA_MODULE_LOADING",nullptr);
        const auto zero_json=price_cli(zero_words,0);
        const bool unchanged_modules=std::getenv("CUDA_MODULE_LOADING")==nullptr;
        environment("CUDA_MODULE_LOADING",had_modules ? original_modules.c_str() : nullptr);
        require(unchanged_modules && zero_json.find("does_not_fit")!=std::string::npos &&
                zero_json.find("\"vram_free\":0")!=std::string::npos,"explicit zero bypasses observation fallback and environment defaults");
        auto partial_words=price_words; partial_words.insert(partial_words.end(),{"--plan-vram-free-bytes","0"});
        require(price_cli(partial_words,2).find("all budget fields or none")!=std::string::npos,"partial budget refuses");
        auto auto_words=price_words; auto_words[14]="auto";
        require(price_cli(auto_words,2).find("requires_budget")!=std::string::npos,"auto CLI without budget refuses safely");
        auto minimum_words=zero_words; minimum_words[14]="auto";
        const auto minimum_json=price_cli(minimum_words,0);
        require(minimum_json.find("\"settings\":null")!=std::string::npos &&
                minimum_json.find("\"configuration_complete\":false")!=std::string::npos,
                "incomplete minimum price does not expose a resolved launch configuration");
        for (const char* bad:{"18446744073709551616","-1","01","1x"}) {
            auto words=zero_words; words.back()=bad;
            require(price_cli(words,2).find("invalid metadata number")!=std::string::npos,"malformed and overflowing budget numbers refuse");
        }
        auto maximum_words=zero_words; maximum_words.back()="18446744073709551615";
        require(price_cli(maximum_words,0).find("\"commit_available\":18446744073709551615")!=std::string::npos,
                "maximum budget is represented without signed conversion or overflow");
        auto duplicate_words=price_words; duplicate_words.push_back("--price-only");
        price_cli(duplicate_words,2);
        auto strict_words=price_words; strict_words.insert(strict_words.end(),{"--serve","--suffix-draft","0","--lookup-chain","0"});
        std::vector<char*> strict_args; for (auto& word:strict_words) strict_args.push_back(word.data());
        LaunchRequest strict_price; LaunchResources strict_resources;
        // Remove the metadata dispatch token so this exercises rejection of the flag itself.
        strict_args.erase(strict_args.begin()+1);
        require(!parse_launch_request((int)strict_args.size(),strict_args.data(),strict_price,strict_resources,true,error) &&
                error.find("refuses --price-only")!=std::string::npos,"strict argv rejects price-only as a flag");
        auto injected_words=price_words;
        injected_words.erase(injected_words.begin()+2);
        injected_words.insert(injected_words.end(),{"--plan-vram-free-bytes","6442450944",
                                                   "--plan-ram-available-bytes","51539607552",
                                                   "--plan-commit-available-bytes","51539607552"});
        std::vector<char*> injected_args; for (auto& word:injected_words) injected_args.push_back(word.data());
        LaunchRequest injected_request; LaunchResources injected_resources;
        require(parse_launch_request((int)injected_args.size(),injected_args.data(),injected_request,injected_resources,false,error) &&
                injected_request.budget_given && injected_resources.vram_free==6442450944 &&
                injected_resources.ram_available==51539607552 && injected_resources.commit_available==51539607552 &&
                injected_resources.vram_total==0,
                "complete injected budgets parse exactly without a fabricated observed total");
        for (size_t resource=0;resource<3;++resource) {
            auto zero_injected=injected_words;
            zero_injected[zero_injected.size()-5+2*resource]="0";
            require(price_cli(zero_injected,2).find("resource")!=std::string::npos,
                    "zero injected live budget refuses without device or host discovery");
        }
        core::ModelGeometry geometry; core::SessionSizing kv;
        for (int64_t chunk:{64,2048,8192,32768}) {
            const auto envelope=prefill::Prefill::sizing_envelope(geometry,131072,chunk,kv);
            const auto envelope_bytes=prefill::Prefill::bytes_needed_for(envelope,true);
            for (int mode=0;mode<4;++mode) for (bool fused:{false,true}) {
                if (fused && mode==0) continue;
                prefill::Prefill::KernelPolicy policy;
                policy.mmq_fits=[mode](int type,int64_t) {
                    return mode!=0 && mode!=3 && (mode!=2 || type!=23) && prefill::mmq::supported(type);
                };
                policy.native_fused=[mode](int,int) { return mode==3; };
                policy.ring=envelope.ring; policy.fused_layout=fused;
                const auto known=prefill::Prefill::sizing_inputs_with_policy(geometry,131072,chunk,kv,policy);
                require(envelope.weight_scratch>=known.weight_scratch && envelope.region>=known.region &&
                        envelope.common==known.common && envelope.ring>=known.ring &&
                        envelope_bytes>=prefill::Prefill::bytes_needed_for(known,true),
                        "envelope bounds explicit all-fallback, all-MMQ, mixed and fused-only policy inventories");
                std::cout << "POLICY synthetic chunk=" << chunk << " mode=" << mode << " fused=" << fused
                          << " owned=" << prefill::Prefill::bytes_needed_for(known,true) << " envelope=" << envelope_bytes << '\n';
            }
        }
        const char* original_ring_env=std::getenv("STRATA_PREFILL_RING");
        const bool had_ring=original_ring_env!=nullptr;
        const std::string original_ring=original_ring_env ? original_ring_env : "";
        for (const char* ring:{"8","16","384","512","1024"}) {
            environment("STRATA_PREFILL_RING",ring);
            const auto large=prefill::Prefill::sizing_envelope(geometry,8192,2048,kv);
            const auto small=prefill::Prefill::sizing_envelope(geometry,8192,64,kv);
            require(large.ring==std::min(512,std::atoi(ring)) && small.ring==8,
                    "pure ring policy preserves explicit routed-only, streaming and native capacity bounds");
        }
        environment("STRATA_PREFILL_RING",had_ring ? original_ring.c_str() : nullptr);
        auto auto_prefill=r; auto_prefill.prefill_auto=true; auto_prefill.prefill=0;
        auto ample=resources; ample.vram_free=24ull<<30; ample.commit_available=96ull<<30;
        LaunchPrice auto_price;
        require(price_launch(auto_prefill,&ample,auto_price,error) && auto_price.configuration_complete &&
                auto_price.resolution.settings.prefill==2048,"pure owned auto resolves the existing chunk policy");
        auto streamed=r; streamed.context=131072; streamed.kv=2; streamed.kv_resident=20480;
        LaunchPrice streamed_price;
        require(price_launch(streamed,&ample,streamed_price,error) &&
                streamed_price.resolution.ram.at("streamed_kv")==905969664,"pure envelope preserves streamed KV owner counts");
        const auto threshold_root=root/"policy-threshold";
        fixture(threshold_root,false,false,true,16);
        auto threshold_request=r; threshold_request.pack=threshold_root.string();
        threshold_request.shards={(threshold_root/"model.gguf").string()};
        threshold_request.profile=(threshold_root/"profile.bin").string();
        for (bool per_layer:{false,true}) {
            threshold_request.per_layer=per_layer;
            LaunchResources zero{0,resources.ram_available,resources.commit_available,1,true}; LaunchPrice minimum;
            require(price_launch(threshold_request,&zero,minimum,error),"variable-blob policy minimum prices");
            const auto threshold=minimum.resolution.minimum_policy_vram_free_bytes;
            require(threshold>0,"automatic minimum exposes policy capacity separately from allocations");
            auto edge=zero; edge.vram_free=threshold-1; LaunchPrice below;
            require(price_launch(threshold_request,&edge,below,error) && !below.configuration_complete &&
                    below.status=="does_not_fit" && below.vram_shortfall==1,"one byte below uniform-grant threshold has policy shortfall");
            if (!per_layer) require(below.resolution.vram_required<=edge.vram_free,
                    "global counterexample: allocated minimum fits while uniform grant cannot select a slot");
            edge.vram_free=threshold; LaunchPrice at;
            require(price_launch(threshold_request,&edge,at,error) && at.configuration_complete &&
                    at.status=="within_priced_budget" && !at.vram_shortfall,"exact minimum policy capacity selects a supported layout");
            const auto fixed=minimum.resolution.vram_required-minimum.resolution.vram.at("expert_cache");
            edge.vram_free=fixed+launch_align(minimum.resolution.settings.max_blob,2ull<<20)-1;
            LaunchPrice below_first_grant;
            require(price_launch(threshold_request,&edge,below_first_grant,error) && !below_first_grant.configuration_complete &&
                    below_first_grant.vram_shortfall==threshold-edge.vram_free && below_first_grant.vram_shortfall>0,
                    "both policies refuse below the first page-rounded max-blob grant with an explicit policy deficit");
            std::cout << "POLICY_THRESHOLD per_layer=" << per_layer << " allocated_min=" << minimum.resolution.vram_required
                      << " minimum_vram_free=" << threshold << " below_shortfall=" << below.vram_shortfall << '\n';
        }
        std::cout << "PASS launch_plan_test pure checks=" << checks << '\n';
        return 0;
        }
        require(prefill::owned_auto_chunk(true,true)==2048 && prefill::owned_auto_chunk(true,false)==2048 &&
                prefill::owned_auto_chunk(false,false)==2048 && prefill::owned_auto_chunk(false,true)==0,
                "owned auto keeps ordinary selection and leaves borrowing to its existing owner");
        auto automatic=r; automatic.prefill=0; automatic.prefill_auto=true;
        auto ample=resources; ample.vram_free=24ull<<30; ample.commit_available=96ull<<30;
        LaunchResolution auto_plan,auto_repeat;
        require(resolve_launch(automatic,ample,auto_plan,error),"owned auto resolves with sufficient synthetic VRAM");
        require(auto_plan.settings.prefill==2048 && auto_plan.settings.no_prefill_borrow==1,
                "owned auto is the ordinary 2048 chunk without borrowing");
        require(resolve_launch(automatic,ample,auto_repeat,error) && auto_plan.payload==auto_repeat.payload,
                "automatic resolution repeats exact payload deterministically");
        require(!admit_launch(automatic,ample,auto_plan.settings,error),"strict admission refuses unresolved prefill");
        automatic.prefill_auto=false; automatic.prefill=2048;
        require(admit_launch(automatic,ample,auto_plan.settings,error),"numeric replay consumes saved auto chunk and layout");
        auto auto_loss=ample; auto_loss.vram_free=auto_plan.settings.vram_required-1;
        const auto saved_auto=auto_plan.settings;
        require(!admit_launch(automatic,auto_loss,auto_plan.settings,error) && auto_plan.settings==saved_auto,
                "resource loss refuses without shrinking approved automatic settings");
        automatic.prefill_auto=true; automatic.no_prefill_borrow=false;
        refusal(automatic,ample,"automatic borrowed search remains unsupported");
        automatic.no_prefill_borrow=true;
        auto auto_short=resources; auto_short.vram_free=1;
        refusal(automatic,auto_short,"owned auto refuses below its priced requirement without reserve or chunk reduction");
        const char* headroom_env=std::getenv("STRATA_RESIDENT_HEADROOM_GIB");
        const bool had_headroom=headroom_env!=nullptr;
        const std::string original_headroom=headroom_env ? headroom_env : "";
        auto parse=[&](const std::string& budget,const std::string& chunk,bool strict,LaunchRequest& parsed) {
            std::vector<std::string> words={"strata","--pack",r.pack,"--native",r.shards[0],"--expert-profile",r.profile,
                "--max-context","8192","--prefill",chunk,"--spec","2","--expert-cache","97",
                "--resident-budget-gib",budget,"--no-prefill-borrow","--suffix-draft","0","--lookup-chain","0"};
            if (strict) { words.push_back("--strict-launch-stdin"); words.push_back("--serve"); }
            std::vector<char*> argv; for (auto& word:words) argv.push_back(word.data());
            LaunchResources snapshot;
            return parse_launch_request((int)argv.size(),argv.data(),parsed,snapshot,strict,error);
        };
        environment("STRATA_RESIDENT_HEADROOM_GIB","8.25");
        LaunchRequest fractional;
        require(parse("0.25","64",false,fractional),"fractional resident budget and headroom parse");
        require(fractional.resident_budget==268435456 && fractional.host_headroom==8858370048ull,
                "fractional transport preserves exact ordinary resident byte semantics");
        LaunchResolution fractional_plan;
        require(resolve_launch(fractional,resources,fractional_plan,error),"fractional request resolves");
        require(fractional_plan.settings.resident_budget==268435456 && fractional_plan.settings.host_headroom==8858370048ull,
                "fractional bytes persist in native schema2");
        require(admit_launch(fractional,resources,fractional_plan.settings,error),"fractional numeric request replays exactly");
        LaunchRequest fractional_strict;
        require(parse("0.25","64",true,fractional_strict) &&
                admit_launch(fractional_strict,resources,fractional_plan.settings,error),
                "strict numeric argv parses the same fractional bytes without another selection");
        for (const char* value : {"0.0000000001","2.5e-1","+0.25",".25","00.25"}) {
            LaunchRequest tiny;
            require(parse(value,"64",false,tiny),"finite decimal transport follows ordinary double semantics");
            require(tiny.resident_budget==(std::string(value)=="0.0000000001" ? 1 : 268435456),
                    "sub-byte budget remains positive and decimal spellings retain exact quarter GiB");
        }
        for (const char* value : {"auto","auto:8192","auto:16384","auto:32768"}) {
            LaunchRequest parsed;
            require(parse("0.25",value,false,parsed) && parsed.prefill_auto,"supported owned automatic forms reach native selection");
            require(!parse("0.25",value,true,parsed),"strict argv refuses every unresolved automatic form");
        }
        LaunchRequest empty_budget;
        require(!parse("","64",false,empty_budget),"empty resident budget refuses");
        // _putenv_s removes an empty value on Windows, so nonempty malformed values exercise this fixture's environment seam.
        for (const char* value : {"nan", "inf", "-0", "-0.25", "1e999", "1024.0001", "0x1p0", "1x", " 1", "1 ", "1..2"}) {
            LaunchRequest invalid;
            require(!parse(value,"64",false,invalid),"malformed nonfinite negative overflowing budget refuses");
            environment("STRATA_RESIDENT_HEADROOM_GIB",value);
            require(!parse("0.25","64",false,invalid),"malformed nonfinite negative overflowing headroom refuses");
            environment("STRATA_RESIDENT_HEADROOM_GIB","8.25");
        }
        for (const char* value : {"0","0.0","0e3"}) {
            LaunchRequest invalid;
            require(!parse(value,"64",false,invalid),"zero resident budget refuses");
        }
        for (const char* value : {"auto:","auto:bad","auto:65536","auto:0","064","-1"}) {
            LaunchRequest invalid;
            require(!parse("0.25",value,false,invalid),"unsupported or malformed prefill refuses");
        }
        environment("STRATA_RESIDENT_HEADROOM_GIB",nullptr);
        LaunchRequest defaults;
        require(parse("22","64",true,defaults) && defaults.host_headroom==(4ull<<30) && defaults.vram_reserve==(700ull<<20),
                "default headroom and reserve remain unchanged");
        environment("STRATA_RESIDENT_HEADROOM_GIB",had_headroom ? original_headroom.c_str() : nullptr);
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
        runtime_controls(b.settings,131072,2,20480);
        require(admit_launch(r,resources,b.settings,error),"saved 128K streamed q4 spec2 settings admitted exactly");
        auto short_now=resources; short_now.vram_free=1;
        require(!admit_launch(r,short_now,b.settings,error) && error.rfind("insufficient_resources:",0)==0,
                "actual admission shortage preserves proved resource type");
        require(core::native_error_message("strict launch admission",error,"invalid_launch_plan").rfind("insufficient_resources:",0)==0,
                "admission wrapper cannot relabel resource shortage as invalid payload");
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
