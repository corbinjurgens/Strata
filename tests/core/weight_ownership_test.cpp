#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

namespace strata::core::detail {
struct ExpertCacheTestAccess {
    static void io(ExpertCache& cache, ExpertCacheIo* io) { cache.test_io_ = io; }
    static void exhaust_revision(ExpertCache& cache) { cache.ownership_revision_ = UINT64_MAX; }
};
}
namespace {
using namespace strata::core;
int checks = 0;
void require(bool ok, const char* what) { ++checks; if (!ok) throw std::runtime_error(what); }

struct Source : ExpertSource {
    uint8_t blob_[80]{};
    Source(bool canonical = false) {
        strata::kernels::cpu::ExpertLayout lay;
        lay.native = true; lay.n_layers = 2; lay.n_expert = 4;
        lay.bytes = {56, 80}; lay.offset = {0, 224}; lay.fmt.resize(2);
        lay.fmt[0].up_off = 16; lay.fmt[0].down_off = 32;
        lay.fmt[1].up_off = 24; lay.fmt[1].down_off = 48;
        lay.gguf_off = {1000, 2000, 3000, 4000, 5000, 6000};
        lay.gguf_file = {"a.gguf", "a.gguf", "b.gguf", "a.gguf", "b.gguf", "c.gguf"};
        record_inputs("experts.bin", "a.gguf", lay, !canonical);
        std::memset(blob_, 0x63, sizeof blob_);
    }
    const uint8_t* blob(int64_t, int64_t) override { return blob_; }
};
struct Io {
    bool alloc_ok = true, touch_ok = true, copy_ok = true, sync_ok = true, free_ok = true;
    void* retained = nullptr;
    detail::ExpertCacheIo hooks{this,
        [](void* c, void** p, size_t n) { auto& x=*(Io*)c; if (!x.alloc_ok) return false; *p=std::malloc(n); return *p!=nullptr; },
        [](void* c, void* p, size_t n) { if (!((Io*)c)->touch_ok) return false; std::memset(p, 0, n); return true; },
        [](void* c, void* d, const void* s, size_t n) { auto& x=*(Io*)c; std::memcpy(d, s, x.copy_ok ? n : n/2); return x.copy_ok; },
        [](void* c) { return ((Io*)c)->sync_ok; },
        [](void* c, void* p) { auto& x=*(Io*)c; if (!x.free_ok) { x.retained=p; return false; } std::free(p); return true; }};
    ~Io() { std::free(retained); }
};
WeightSnapshot snapshot(const ExpertCache& c) {
    auto owner = c.weight_ownership();
    WeightSnapshot s; std::string error;
    require(aggregate_weight_ownership({&owner}, {"a.gguf", "b.gguf", "c.gguf"}, s, error), "aggregate cache");
    return s;
}
void cache_cases() {
    Source source;
    auto roles = source.weight_slices(1, 2);
    require(roles.size()==3 && roles[0].offset==0 && roles[0].bytes==24 && roles[0].source_offset==4048,
            "gate source offset uses its own role stride");
    require(roles[1].offset==24 && roles[1].bytes==24 && roles[1].source_offset==5048 && roles[1].shard=="b.gguf",
            "up role has its own shard");
    require(roles[2].offset==48 && roles[2].bytes==32 && roles[2].source_offset==6064 && roles[2].shard=="c.gguf",
            "unequal down role uses its own stride");
    require(source.weight_slices(2,0).empty() && source.weight_slices(0,4).empty(), "source index bounds");
    Io io; ExpertCache c; std::string e;
    detail::ExpertCacheTestAccess::io(c, &io.hooks);
    c.track_ownership(&source);
    require(c.open_sized({56,80,56},2,4,e), "actual cache open_sized via CPU I/O");
    auto s=snapshot(c);
    require(s.known && s.capacity==768 && s.logical==0 && s.padding==768, "aligned allocation holds no model weights before upload");
    require(c.runtime_memory().known && c.runtime_memory().requested==768 && c.runtime_memory().rounded==2097152 &&
            c.runtime_memory().logical==0, "successful allocator request and granule model use independent slot alignment oracle");
    require(c.admit(0,0)==0 && c.fill_slot_blocking(0, source.blob(0,0), e,56), "first blocking upload");
    require(c.ownership_revision()==1, "first confirmed upload advances ownership revision");
    s=snapshot(c);
    require(s.logical==56 && s.shards[0].bytes==32 && s.shards[1].bytes==24 && s.padding==712, "split role upload reconciles");
    require(c.runtime_memory().requested==768 && c.runtime_memory().logical==56, "current logical cache bytes exclude allocation padding");
    require(c.fill_slot_blocking(0,source.blob(0,0),e,56), "repeat upload");
    require(snapshot(c).logical==56, "repeat upload is replacement rather than cumulative bytes");
    require(c.ownership_revision()==2, "same-shard same-size successful upload still advances publication revision");
    require(c.verify_slot(0,source.blob(0,0),e,56), "actual slot verification through injected readback");
    require(c.admit(1,1)==1 && c.fill_slot_blocking(1,source.blob(1,1),e,80), "different layer size upload");
    s=snapshot(c);
    require(s.logical==136 && s.shards[0].bytes==56 && s.shards[1].bytes==48 && s.shards[2].bytes==32, "unequal source roles summed independently");
    require(c.prepare_ownership(0,0,2) && c.fill_slot_queued(0,source.blob(0,2),e,56), "replacement queued");
    require(!snapshot(c).known, "queued copy cannot publish held contents");
    require(!c.runtime_memory().known, "queued upload suppresses fresh numeric component state");
    require(c.sync_queued(e), "replacement synchronized");
    c.replace(0,0,2);
    require(snapshot(c).logical==136 && c.weight_ownership().slices[0].source_offset==1032, "replacement subtracts prior source");
    require(c.prepare_ownership(2,0,2) && c.fill_slot_blocking(2,source.blob(0,2),e,56), "duplicate copy in another slot");
    s=snapshot(c);
    require(s.logical==192 && s.shards[0].bytes==88 && s.shards[0].source_unique==56 &&
            s.shards[1].bytes==72 && s.shards[1].source_unique==48, "separate copies count held bytes but union source spans");
    c.borrow_ownership(1);
    s=snapshot(c);
    require(s.known && s.logical==56 && s.shared==512 && s.padding==200, "borrow removes weights and separates shared scratch");
    require(c.runtime_memory().logical==56 && c.runtime_memory().requested==768, "borrowing changes contents without changing owned allocation");
    require(c.prepare_ownership(1,1,1) && c.fill_slot_queued(1,source.blob(1,1),e,80) &&
            c.prepare_ownership(2,0,2) && c.fill_slot_queued(2,source.blob(0,2),e,56), "restoration issued");
    require(!snapshot(c).known && c.sync_queued(e), "restoration waits for successful sync");
    s=snapshot(c);
    require(s.logical==192 && s.shared==0 && s.padding==576, "restoration exactly reconciles");
    require(c.runtime_memory().logical==192, "confirmed restoration updates current component");
    require(c.prepare_ownership(0,0,2) && c.fill_slot(0,source.blob(0,2),(void*)1,e,56), "nonlegacy asynchronous upload issued");
    require(c.prepare_ownership(1,1,1) && c.fill_slot_blocking(1,source.blob(1,1),e,80) && !snapshot(c).known,
            "legacy copy sync cannot confirm a different stream's pending write");
    c.complete_ownership();
    require(snapshot(c).logical==192, "external completion confirms remaining asynchronous writes once");
    require(!c.fill_slot_blocking(0,nullptr,e,56) && snapshot(c).logical==192, "pre-copy refusal preserves prior holdings");
    io.copy_ok=false;
    require(!c.fill_slot_blocking(0,source.blob(0,2),e,56) && !snapshot(c).known, "partial copy failure is unknown");
    c.complete_ownership();
    require(!snapshot(c).known, "error cannot be healed by unproved completion");
    c.close();
    s=snapshot(c);
    require(s.known && s.capacity==0 && s.logical==0, "confirmed release clears holdings");
    require(c.runtime_memory().known && c.runtime_memory().requested==0 && c.runtime_memory().rounded==0, "confirmed release clears allocation accounting");
    io.copy_ok=true;
    require(c.open_sized({56},2,4,e) && c.admit(0,0)==0, "reopen after confirmed rollback");
    io.sync_ok=false;
    require(!c.fill_slot_blocking(0,source.blob(0,0),e,56) && !snapshot(c).known, "sync failure cannot publish success");
    c.close(); io.sync_ok=true;
    require(c.open_sized({56},2,4,e), "open for uncertain release");
    io.free_ok=false; c.close();
    require(!snapshot(c).known && c.weight_ownership().capacity==256, "unconfirmed release retains uncertain capacity");
    require(!c.runtime_memory().known, "failed release does not claim zero successful storage");
    require(!c.open_sized({56},2,4,e) && !snapshot(c).known, "unknown release cannot be forgotten by reopening");
    // The injected allocator retained the failed free for teardown, as the real
    // owner no longer has a usable pointer after its existing close path.
    Io failed; ExpertCache f; detail::ExpertCacheTestAccess::io(f,&failed.hooks); f.track_ownership(&source);
    failed.alloc_ok=false;
    require(!f.open_sized({56},2,4,e) && snapshot(f).capacity==0, "allocation failure has no successful holdings");
    require(e.find("out_of_memory")==std::string::npos && e.find("insufficient_resources")==std::string::npos,
            "untyped allocator fault cannot prove OOM or a resource shortfall");
    failed.alloc_ok=true; failed.touch_ok=false;
    require(!f.open_sized({56},2,4,e) && snapshot(f).capacity==0, "touch failure followed by confirmed rollback");
    failed.touch_ok=true;
    require(f.open_sized({56},2,4,e) && f.admit(0,0)==0 && f.fill_slot_blocking(0,source.blob(0,0),e,56), "upload before readback mismatch");
    uint8_t wrong[56]{};
    require(!f.verify_slot(0,wrong,e,56) && !snapshot(f).known, "readback mismatch invalidates successful held state");
    f.close();
    require(f.open_sized({56},2,4,e) && f.admit(0,0)==0, "open for ownership revision exhaustion");
    detail::ExpertCacheTestAccess::exhaust_revision(f);
    require(!f.fill_slot_blocking(0,source.blob(0,0),e,56) && !snapshot(f).known && f.ownership_revision()==UINT64_MAX,
            "owner revision exhaustion cannot wrap or publish successful contents");
    Source canonical(true); Io ci; ExpertCache cc; detail::ExpertCacheTestAccess::io(cc,&ci.hooks); cc.track_ownership(&canonical);
    require(cc.open_sized({56},2,4,e) && cc.admit(0,0)==0 && cc.fill_slot_blocking(0,canonical.blob(0,0),e,56), "canonical upload");
    s=snapshot(cc);
    require(s.unattributed==56 && s.shards[0].bytes==0, "canonical pack without proved provenance remains unattributed");
}
void aggregate_cases() {
    WeightOwnership w; w.capacity=512; w.scratch=32; w.mapped_host=4096;
    w.slices={{0,100,"a.gguf",1000},{256,100,"a.gguf",1000}};
    WeightSnapshot s; std::string e;
    require(aggregate_weight_ownership({&w},{"a.gguf"},s,e) && s.logical==200 && s.padding==280 &&
            s.mapped_host==4096 && s.shards[0].source_unique==100, "packed duplicate, scratch and host storage stay separate");
    require(!aggregate_weight_ownership({&w,&w},{"a.gguf"},s,e), "one allocation owner cannot be charged twice");
    w.slices[1].offset=99;
    require(!aggregate_weight_ownership({&w},{"a.gguf"},s,e), "destination overlap refuses");
    w.slices[1].offset=256; w.slices[1].source_offset=UINT64_MAX-3;
    require(!aggregate_weight_ownership({&w},{"a.gguf"},s,e), "source overflow refuses");
    w.slices[1].source_offset=1000;
    require(!aggregate_weight_ownership({&w},{"a.gguf","./a.gguf"},s,e), "duplicate shard identity refuses");
    w.scratch=313;
    require(!aggregate_weight_ownership({&w},{"a.gguf"},s,e), "scratch cannot overlap logical capacity");
    w.slices.clear(); w.scratch=0; w.capacity=UINT64_MAX;
    WeightOwnership other; other.capacity=1;
    require(!aggregate_weight_ownership({&w,&other},{},s,e), "capacity overflow refuses");
}
template<class T> void put(std::ostream& out, T value) { out.write((const char*)&value,sizeof value); }
void string(std::ostream& out, const std::string& s) { put<uint64_t>(out,s.size()); out << s; }
void file_source_case(const std::filesystem::path& root) {
    namespace fs = std::filesystem;
    fs::create_directories(root);
    std::vector<std::string> paths;
    std::vector<uint64_t> data;
    const char* roles[] = {"gate","up","down"};
    // Independent GGUF block arithmetic: 2560*640/32 blocks, Q8_0=34 B,
    // Q4_0=18 B. The down role is a different size and a different shard.
    const uint64_t per[] = {1740800,1740800,921600};
    for (int r=0;r<3;++r) {
        const auto path=root/(std::string(roles[r])+".gguf");
        std::ofstream out(path,std::ios::binary);
        out.write("GGUF",4); put<uint32_t>(out,3); put<uint64_t>(out,1); put<uint64_t>(out,0);
        string(out,"blk.0.ffn_"+std::string(roles[r])+"_exps.weight");
        put<uint32_t>(out,3); put<uint64_t>(out,r==2 ? 640 : 2560); put<uint64_t>(out,r==2 ? 2560 : 640);
        put<uint64_t>(out,4); put<uint32_t>(out,r==2 ? 2 : 8); put<uint64_t>(out,0);
        const uint64_t start=((uint64_t)out.tellp()+31)/32*32;
        out.close(); fs::resize_file(path,start+per[r]*4);
        paths.push_back(path.string()); data.push_back(start);
    }
    auto layout = [&root,&data](bool bad) {
        std::ofstream out(root/"native_experts.txt");
        out << "# strata native experts v4: synthetic (n_expert 4)\n0 8 2 0 4403200 "
            << data[0]+(bad ? 32 : 0) << ' ' << data[1] << ' ' << data[2] << " gate.gguf,up.gguf,down.gguf\n";
    };
    std::string error;
    layout(false);
    require(strata::kernels::cpu::expert_layout_load(root.string(),1,4,error), "real native layout reader accepts split-role fixture");
    FileExpertSource source; source.set_gguf(paths[0]);
    require(source.open(root.string(),1,4,error) && source.gguf_mode(), "actual file source validates and maps independent GGUF roles");
    const auto storage=source.runtime_memory();
    // Header lengths are fixed by the fixture's explicit GGUF encoding above.
    const uint64_t mapped=fs::file_size(root/"gate.gguf")+fs::file_size(root/"up.gguf")+fs::file_size(root/"down.gguf");
    require(storage.at("expert_file_views").mapped==mapped && storage.at("resident_experts").requested==0,
            "file owner reports logical view lengths from independently inspected files, not physical RAM");
    auto slices=source.weight_slices(0,3);
    require(slices.size()==3 && slices[0].bytes==1740800 && slices[1].offset==1740800 &&
            slices[2].offset==3481600 && slices[2].bytes==921600, "actual parsed role sizes and concatenated offsets");
    for (int r=0;r<3;++r)
        require(slices[(size_t)r].shard==paths[(size_t)r] && slices[(size_t)r].source_offset==data[(size_t)r]+3*per[r],
                "parsed role identity and expert source span");
    std::vector<uint8_t> blob(4403200);
    require(source.copy_blob(0,3,blob.data()), "actual source copies split-role bytes without a model or device");
    Io io; ExpertCache cache; detail::ExpertCacheTestAccess::io(cache,&io.hooks); cache.track_ownership(&source);
    require(cache.open_sized({4403200},1,4,error) && cache.admit(0,3)==0 &&
            cache.fill_slot_blocking(0,blob.data(),error,4403200), "actual source through actual cache upload seam");
    auto owner=cache.weight_ownership(); WeightSnapshot s;
    require(aggregate_weight_ownership({&owner},paths,s,error) && s.logical==4403200 && s.capacity==4403200 &&
            s.shards[0].bytes==1740800 && s.shards[1].bytes==1740800 && s.shards[2].bytes==921600,
            "three real reader source spans reconcile with successful cache contents");
    source.close();
    require(source.runtime_memory().at("expert_file_views").mapped==0 &&
            source.runtime_memory().at("file_cache_target").requested==0,"source release updates current storage");
    require(source.weight_slices(0,0).empty(), "source close clears attribution for future uploads");
    layout(true);
    require(strata::kernels::cpu::expert_layout_load(root.string(),1,4,error), "bad span fixture parses layout only");
    FileExpertSource bad; bad.set_gguf(paths[0]);
    require(!bad.open(root.string(),1,4,error) && bad.weight_slices(0,0).empty(), "failed source span validation cannot invent attribution");
}
void frame_cases() {
    WeightSnapshot s; s.capacity=256; s.logical=56; s.padding=200; s.shards={{32,32},{24,24}};
    uint64_t seq=0; std::string frame,e,body;
    require(frame_weight_ownership(s,"cuda",0,"pci:0000:01:00.0",seq,frame,e) && seq==1, "first sequence");
    require(frame.find("shard\t0\t32\t32\t0\nshard\t1\t24\t24\t0\n")!=std::string::npos &&
            frame.find("totals\t256\t56\t200\t0\t0\t0\t0\n")!=std::string::npos, "exact aggregate frame schema");
    std::istringstream in(frame+"READY 8192 stop\n"); uint64_t readseq=0;
    require(read_weight_ownership_frame(in,readseq,body,e) && readseq==1 && in.peek()=='R', "complete boundary leaves following stdout unread");
    for (size_t n=0;n<frame.size();++n) {
        std::istringstream truncated(frame.substr(0,n)); uint64_t old=0; std::string unchanged="last valid";
        require(!read_weight_ownership_frame(truncated,old,unchanged,e) && old==0 && unchanged=="last valid", "every truncated frame retains predecessor");
    }
    std::istringstream repeat(frame);
    require(!read_weight_ownership_frame(repeat,readseq,body,e), "stale sequence refuses");
    s.known=false;
    require(frame_weight_ownership(s,"cuda",0,"pci",seq,frame,e) && seq==2 &&
            frame.find("state\tunknown\n")!=std::string::npos && frame.find("totals\t")==std::string::npos, "unknown frame cannot assert successful numeric holdings");
    seq=UINT64_MAX;
    require(!frame_weight_ownership(s,"cuda",0,"pci",seq,frame,e) && seq==UINT64_MAX, "sequence exhaustion refuses without wrap");
    seq=0; s.known=true; s.logical=57;
    require(!frame_weight_ownership(s,"cuda",0,"pci",seq,frame,e) && seq==0, "unreconciled frame refuses");
    s.logical=56;
    require(!frame_weight_ownership(s,"cuda",0,"",seq,frame,e) && seq==0, "missing device identity cannot publish success");
    s.shards.assign(64,{}); s.shards[0]={56,56};
    require(frame_weight_ownership(s,"cuda",0,"pci",seq,frame,e) && frame.size()<=16384 &&
            frame.find("shard\t63\t0\t0\t0\n")!=std::string::npos, "bounded complete snapshot includes zero shards");
    s.shards.push_back({});
    require(!frame_weight_ownership(s,"cuda",0,"pci",seq,frame,e), "shard frame bound refuses");
    for (const std::string& bad : std::vector<std::string>{"STRATA-OWNERSHIP/2\n1\n0\n",
            "STRATA-OWNERSHIP/1\n01\n0\n", "STRATA-OWNERSHIP/1\n-1\n0\n",
            "STRATA-OWNERSHIP/1\n18446744073709551616\n0\n", "STRATA-OWNERSHIP/1\n1\n16385\n",
            "STRATA-OWNERSHIP/1\n1\n0\n\nSTRATA-OWNERSHIP-END/1 2\n"}) {
        std::istringstream input(bad); uint64_t old=0; std::string valid="last valid";
        require(!read_weight_ownership_frame(input,old,valid,e) && old==0 && valid=="last valid", "malformed envelope preserves last valid snapshot");
    }
}
void runtime_cases(bool emit_fixture = false) {
    RuntimeMemory device; device.device_allocation(768); device.device_allocation(2097153);
    require(device.requested==2097921 && device.rounded==6291456,"separate successful allocations round independently");
    const auto previous=device;
    bool overflow=false;
    try { device.device_allocation(UINT64_MAX); } catch (const std::overflow_error&) { overflow=true; }
    require(overflow && device==previous,"allocation accounting overflow preserves previous complete record");
    RuntimeSnapshot s; s.weights.capacity=768; s.weights.logical=56; s.weights.padding=712; s.weights.shards={{56,56}};
    s.settings={{"context",8192},{"prefill",64},{"batch",0},{"parallel",1},{"spec",2},{"kv",1},{"kv_resident",0}};
    if (emit_fixture) {
        for (const char* name : {"batch_groups", "auto_cache", "per_layer", "no_prefill_borrow", "resident_budget",
                "resident_bytes", "resident_experts", "resident_pin", "file_cache_effective", "file_cache_compact",
                "host_headroom", "vram_reserve", "cache_slots", "cache_bytes", "shared_stage", "ring",
                "stager_threads", "stager_ring", "stager_batch", "prefill_borrowed"}) s.settings[name]=0;
        s.settings["batch_groups"]=1; s.settings["no_prefill_borrow"]=1; s.settings["resident_pin"]=1;
        s.settings["cache_slots"]=3; s.settings["cache_bytes"]=768; s.settings["ring"]=8;
        s.settings["stager_threads"]=1; s.settings["stager_ring"]=2; s.settings["stager_batch"]=1;
    }
    auto& cache=s.vram["expert_cache"]; cache.device_allocation(768); cache.logical=56;
    s.vram["overhead_allowance"].allowance=469762048;
    auto& host=s.ram["streamed_kv"]; host.requested=host.rounded=75498496; host.logical=75497472;
    host.mapped=host.pinned=75498496;
    uint64_t seq=0; std::string frame,error,body;
    require(frame_runtime_snapshot(s,"cuda",0,emit_fixture ? "0000:01:00.0" : "pci",seq,frame,error) && seq==1,"ordinary snapshot needs no approved plan");
    if (emit_fixture) {
        require(std::fwrite(frame.data(),1,frame.size(),stdout)==frame.size(),"runtime fixture output");
        return;
    }
    require(frame.find("setting\tcontext\t8192\n")!=std::string::npos &&
            frame.find("component\tvram\texpert_cache\tknown\t768\t2096384\t56\t0\t0\t0\n")!=std::string::npos &&
            frame.find("component\tram\tstreamed_kv\tknown\t75498496\t0\t75497472\t75498496\t75498496\t0\n")!=std::string::npos &&
            frame.find("component\tvram\toverhead_allowance\tknown\t0\t0\t0\t0\t0\t469762048\n")!=std::string::npos,
            "independent allocation, rounding, logical, mapped, pin and allowance byte oracles");
    require(frame.find("physical\tunavailable\n")!=std::string::npos && frame.find("total_allocated\tunknown\n")!=std::string::npos,
            "no estimator or physical residency masquerades as measured total");
    std::istringstream input(frame+"READY 8192 stop\n"); uint64_t readseq=0;
    require(read_runtime_frame(input,readseq,body,error) && readseq==1 && input.peek()=='R',"runtime frame commits before following stdout");
    for (size_t n=0;n<frame.size();++n) {
        std::istringstream broken(frame.substr(0,n)); uint64_t old=0; std::string previous="predecessor";
        require(!read_runtime_frame(broken,old,previous,error) && old==0 && previous=="predecessor","every interrupted runtime frame retains complete predecessor");
    }
    std::istringstream repeat(frame);
    require(!read_runtime_frame(repeat,readseq,body,error),"runtime stale sequence refuses");
    cache.known=false; s.weights.known=false;
    require(frame_runtime_snapshot(s,"cuda",0,"pci",seq,frame,error) && seq==2 &&
            frame.find("component\tvram\texpert_cache\tunknown\t-\t-\t-\t-\t-\t-\n")!=std::string::npos &&
            frame.find("totals\t")==std::string::npos,"failed or unconfirmed ownership suppresses numeric holdings");
    cache.known=true; cache.logical=769;
    require(!frame_runtime_snapshot(s,"cuda",0,"pci",seq,frame,error) && seq==2 && frame.empty(),"invalid component refuses atomically");
    cache.logical=56; seq=UINT64_MAX;
    require(!frame_runtime_snapshot(s,"cuda",0,"pci",seq,frame,error) && seq==UINT64_MAX,"runtime sequence cannot wrap");
    seq=0; s.settings["bad\nname"]=0;
    require(!frame_runtime_snapshot(s,"cuda",0,"pci",seq,frame,error) && seq==0,"field injection refuses");
    for (const char* code : {"out_of_memory", "insufficient_resources", "invalid_launch_plan"}) {
        const auto cause=std::string(code)+": proved cause";
        const auto wrapped=native_error_message("final initialization",native_error_message("prefill/cache",cause),"invalid_launch_plan");
        char line[512]; std::snprintf(line,sizeof line,"ERROR %s\n",wrapped.c_str());
        require(std::string(line).rfind(std::string("ERROR ")+code+": ",0)==0,"actual wrapper formatting preserves typed cause");
    }
    require(native_error_message("allocation","generic device failure")=="allocation: generic device failure",
            "unproved allocation/device failure is never guessed OOM");
    require(native_error_message("admission","malformed payload","invalid_launch_plan")=="invalid_launch_plan: admission: malformed payload",
            "invalid plan fallback is separate from a proved resource cause");
}
}
int main(int argc, char** argv) {
    try {
        if (argc==2 && std::string(argv[1])=="--runtime-fixture") {
#if defined(_WIN32)
            require(_setmode(_fileno(stdout),_O_BINARY)!=-1,"runtime fixture binary stdout");
#endif
            runtime_cases(true); return 0;
        }
        require(argc==2 && std::filesystem::path(argv[1]).is_absolute(), "explicit absolute fixture directory required");
        cache_cases(); aggregate_cases(); frame_cases(); runtime_cases(); file_source_case(argv[1]);
        std::printf("PASS weight_ownership %d/%d checks (CPU I/O injection; no device execution)\n",checks,checks);
    }
    catch (const std::exception& e) { std::fprintf(stderr,"FAIL after %d checks: %s\n",checks,e.what()); return 1; }
    return 0;
}
