"""Local guarded benchmark wrapper. No CUDA API calls; --self-check never starts a model."""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("watched_benchmark", ROOT / "tools/benchmark_compact_cache.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
GIB = 1024**3
EXPECTED_HEAD = "563c136c1df98ae416022e595e606b963c7f2d22"


class PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in (
        "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
        "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
        (name, wintypes.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]


def windows_memory():
    if os.name != "nt":
        raise RuntimeError("This validation guard requires Windows GetPerformanceInfo")
    info = PERFORMANCE_INFORMATION()
    info.cb = ctypes.sizeof(info)
    api = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
    api.argtypes = [ctypes.POINTER(PERFORMANCE_INFORMATION), wintypes.DWORD]
    api.restype = wintypes.BOOL
    if not api(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    if not info.PageSize or info.CommitTotal > info.CommitLimit or info.PhysicalAvailable > info.PhysicalTotal:
        raise RuntimeError("Invalid GetPerformanceInfo counters")
    return dict(physical_available_bytes=info.PhysicalAvailable * info.PageSize,
                physical_used_bytes=(info.PhysicalTotal - info.PhysicalAvailable) * info.PageSize,
                commit_total_bytes=info.CommitTotal * info.PageSize,
                commit_limit_bytes=info.CommitLimit * info.PageSize,
                commit_available_bytes=(info.CommitLimit - info.CommitTotal) * info.PageSize)


class TraceReader:
    """Incremental bounded stderr parser for generate.cpp mem_mark and serving formats."""
    def __init__(self, path):
        self.path, self.offset, self.pending = Path(path), 0, b""
        self.serving = False
        self.count = 0
        self.last = None
        self.minimum_enforced = None

    def line(self, line):
        value, stage, enforce = None, "startup", False
        m = re.search(r"strata trace: (prefill )?(\d+) MiB free after (.+)", line)
        if m:
            value, stage = int(m[2]), (m[1] or "") + m[3].strip()
            enforce = self.serving or stage == "the completed request"
        m = re.search(r"strata serve: (?:WARNING: )?(\d+) MiB of VRAM free with everything loaded", line)
        if m:
            value, stage, enforce = int(m[1]), "serving", True
            self.serving = True
        m = re.search(r"\bvram_free_mib[=:]\s*(\d+)", line)
        if m:
            value, stage, enforce = int(m[1]), "engine-metric", True
            self.serving = True
        m = re.search(r"strata serve: VRAM .*?, (\d+) MiB free now", line)
        if m:
            value, stage, enforce = int(m[1]), "request-runtime", True
        m = re.search(r"strata generate: CUDA\d+ context ready, ([\d.]+) GiB free", line)
        if m:
            value, stage, enforce = float(m[1]) * 1024, "context-ready", False
        if value is None:
            # A changed memory-bearing trace format must not silently bypass the guard.
            if "strata trace:" in line and "MiB free" in line:
                raise RuntimeError("Unrecognized CUDA memory trace")
            return None
        self.count += 1
        self.last = dict(cuda_free_mib=value, stage=stage, enforced=enforce,
                         observed_unix=time.time(), source="engine-log")
        if enforce:
            self.minimum_enforced = value if self.minimum_enforced is None else min(value, self.minimum_enforced)
        return dict(self.last)

    def read(self):
        if not self.path.exists():
            if self.offset:
                raise RuntimeError("Engine log disappeared")
            return []
        size = self.path.stat().st_size
        if size < self.offset:
            raise RuntimeError("Engine log truncated")
        if size > 2 * bench.LOG_LIMIT:
            raise RuntimeError("Engine log exceeded original diagnostic size limit")
        if size - self.offset > 1024**2:
            raise RuntimeError("Engine log parsing backlog exceeded 1 MiB")
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read(1024**2)
        self.offset += len(data)
        lines = (self.pending + data).split(b"\n")
        self.pending = lines.pop()
        if len(self.pending) > 65536 or any(len(line) > 65536 for line in lines):
            raise RuntimeError("Engine log line exceeded 64 KiB")
        result = []
        for raw in lines:
            sample = self.line(raw.decode("utf-8", errors="strict"))
            if sample:
                result.append(sample)
        return result


class WatchedGuard(bench.MemoryGuard):
    output = None
    start = None

    def __init__(self, owner, engine_log):
        super().__init__(owner, engine_log)
        self.reader = TraceReader(engine_log)
        self.commit_low = 0
        self.min_commit = None
        self.max_owned_rss = self.max_owned_private = 0
        self.lock = threading.RLock()
        self.output = Path(type(self).output)
        self.events = self.output / "telemetry.jsonl"

    def emit(self, event):
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()

    def persist(self):
        pending = self.output / "telemetry-summary.json.tmp"
        pending.write_text(json.dumps(self.summary(), indent=2) + "\n", encoding="utf-8")
        bench.replace_result(pending, self.output / "telemetry-summary.json")

    def fail(self, reason):
        if self.reason is None:
            self.reason = reason
        try:
            self.emit(dict(kind="failure", unix=time.time(), reason=self.reason))
            self.persist()
        finally:
            self.owner.stop()  # The benchmark's retained descendant identities only.

    def traces(self):
        for sample in self.reader.read():
            self.emit(dict(kind="cuda-stage", **sample))
            if sample["enforced"] and sample["cuda_free_mib"] < 256:
                self.fail("Actual engine CUDA-free memory below 256 MiB at " + sample["stage"])
                return

    def thresholds(self, row):
        available = row["commit_available_bytes"]
        self.min_commit = available if self.min_commit is None else min(self.min_commit, available)
        self.commit_low = self.commit_low + 1 if available < 4 * GIB else 0
        if available < 2 * GIB:
            self.fail("Available system commit below 2 GiB")
        elif self.commit_low >= 5:
            self.fail("Available system commit below 4 GiB for five consecutive samples")
        elif row["elapsed_seconds"] >= 5400:
            self.fail("Total wrapper elapsed time reached 5400 seconds")
        # Keep the original physical RAM thresholds and sample accounting.
        self.sample(row["physical_available_bytes"], None)

    def tick(self):
        self.owner.discover()
        row = windows_memory()
        physical = bench.psutil.virtual_memory()
        # Original guard uses psutil.available; record both physical sources explicitly.
        row["gpi_physical_available_bytes"] = row["physical_available_bytes"]
        row["physical_available_bytes"] = physical.available
        row["physical_used_bytes"] = physical.total - physical.available
        members = []
        for process in list(self.owner.members):
            try:
                memory = process.memory_info()
                members.append(dict(pid=process.pid, created=process.create_time(),
                                    rss_bytes=memory.rss, private_bytes=memory.private))
            except bench.psutil.NoSuchProcess:
                continue
        row.update(kind="memory", unix=time.time(), elapsed_seconds=time.monotonic() - self.start,
                   owned_processes=members, owned_rss_bytes=sum(p["rss_bytes"] for p in members),
                   owned_private_bytes=sum(p["private_bytes"] for p in members))
        self.max_owned_rss = max(self.max_owned_rss, row["owned_rss_bytes"])
        self.max_owned_private = max(self.max_owned_private, row["owned_private_bytes"])
        self.emit(row)  # Keep evidence before a threshold can stop descendants.
        self.thresholds(row)
        if not self.reason:
            self.traces()
        self.persist()

    def monitor(self):
        try:
            while not self.done.is_set():
                before = time.monotonic()
                with self.lock:
                    self.tick()
                if self.reason:
                    return
                self.done.wait(max(0, 1 - (time.monotonic() - before)))
        except Exception as exc:
            with self.lock:
                self.fail("Telemetry error: " + type(exc).__name__ + ": " + str(exc))

    def check(self):
        with self.lock:
            try:
                if not self.reason:
                    self.traces()
                    if time.monotonic() - self.start >= 5400:
                        self.fail("Total wrapper elapsed time reached 5400 seconds")
            except Exception as exc:
                self.fail("Telemetry error: " + type(exc).__name__ + ": " + str(exc))
            super().check()

    def summary(self):
        result = super().summary()
        result.update(min_commit_available_gib=self.min_commit / GIB if self.min_commit is not None else None,
                      max_owned_rss_bytes=self.max_owned_rss, max_owned_private_bytes=self.max_owned_private,
                      cuda_stage_samples=self.reader.count, last_cuda_stage=self.reader.last,
                      min_enforced_cuda_free_mib=self.reader.minimum_enforced,
                      cuda_coverage="Engine stage samples only; no continuous VRAM telemetry",
                      commit_source="GetPerformanceInfo CommitLimit minus CommitTotal; not swap usage",
                      elapsed_seconds=time.monotonic() - self.start)
        return result


def short_tests(client, record, save, guard, idle):
    rows = record["watched_short"] = []
    def request(label, messages, tokens, expected=None, sustained=False, cached=False):
        client.label = label
        response = client.chat(messages, tokens, temperature=0,
                               chat_template_kwargs={"enable_thinking": False})
        text = response["choices"][0]["message"].get("content") or ""
        checks = {"nonempty": bool(text.strip())}
        if expected is not None:
            checks["exact_answer"] = bench.exact_answer(text, expected)
        if sustained:
            checks["generated_512_tokens"] = response["usage"]["completion_tokens"] == 512
        if cached:
            checks["prefix_cache_used"] = response["timings"].get("cache_n", 0) > 0
        rows.append(dict(label=label, response=response, checks=checks, passed=all(checks.values())))
        save()
        if not all(checks.values()):
            raise RuntimeError("Watched short check failed: " + label)
        return response
    startup = [{"role": "user", "content": "Calculate 17 times 23. Reply with the integer only."}]
    response = request("short-exact", startup, 32, [391])
    request("short-cached-followup", startup + [response["choices"][0]["message"],
            {"role": "user", "content": "Add 9 to that result. Reply with the integer only."}], 32, [400], cached=True)
    for index in range(3):
        messages = [{"role": "user", "content":
            f"Case {index + 1}: Write a very long numbered tutorial about database transactions, isolation, "
            "indexes and recovery. Continue for at least 1500 words without concluding; give detailed examples."}]
        response = request(f"sustained-{index + 1}", messages, 512, sustained=True)
        request(f"sustained-{index + 1}-followup", messages + [response["choices"][0]["message"],
                {"role": "user", "content": "Calculate 19 times 7. Reply with the integer only."}],
                32, [133], cached=True)
    record["short_idle_started_unix"] = time.time()
    save()
    deadline = time.monotonic() + idle
    while time.monotonic() < deadline:
        guard.check()
        guard.done.wait(min(1, max(0, deadline - time.monotonic())))
    guard.check()
    record["short_idle_finished_unix"] = time.time()
    record["watched_short_passed"] = True
    save()
    return True


def install(output, short, idle):
    WatchedGuard.output, WatchedGuard.start = output, time.monotonic()
    bench.MemoryGuard = WatchedGuard
    bench.SCRATCH = HERE / "scratch"
    original_client, original_load = bench.make_client, bench.load_workload
    original_clean = bench.clean_environment
    def clean(cfg, timing, fetch_timing=False):
        cfg, environment = original_clean(cfg, timing, fetch_timing)
        cfg.setdefault("env", {})["STRATA_TRACE"] = "1"
        return cfg, environment
    bench.clean_environment = clean
    bench.EVIDENCE_ENV = bench.EVIDENCE_ENV | {"STRATA_TRACE"}
    context = {}
    def make_client(validation, cfg, record, save, guard, secret):
        client = original_client(validation, cfg, record, save, guard, secret)
        context.update(record=record, save=save, guard=guard)
        if short:
            original_chat = client.chat
            first = True
            def chat(messages, max_tokens=128, **kwargs):
                nonlocal first
                startup = first
                first = False
                if startup:
                    client.label = "startup-exact"
                    messages = [{"role": "user", "content":
                                 "Calculate 17 times 23. Reply with the integer only."}]
                    max_tokens = 32
                    kwargs.update(temperature=0, chat_template_kwargs={"enable_thinking": False})
                response = original_chat(messages, max_tokens, **kwargs)
                if startup:
                    text = response["choices"][0]["message"].get("content") or ""
                    record["startup_exact_passed"] = bench.exact_answer(text, [391])
                    save()
                    if not record["startup_exact_passed"]:
                        raise RuntimeError("Startup exact answer failed")
                return response
            client.chat = chat
        original_get = client.get
        def get(path):
            result = original_get(path)
            if path == "/health" and result.get("loaded"):
                guard.check()
                if guard.reader.minimum_enforced is None:
                    guard.fail("Loaded engine lacks a parsed serving CUDA-free sample")
                    guard.check()
            return result
        client.get = get
        return client
    def load(name):
        module = original_load(name)
        if short and name == "validation":
            module.smoke = lambda client: short_tests(client, context["record"], context["save"],
                                                       context["guard"], idle)
        return module
    bench.make_client, bench.load_workload = make_client, load


def self_check():
    import tempfile
    class Owner:
        def __init__(self): self.stops = 0
        def stop(self): self.stops += 1
    with tempfile.TemporaryDirectory(prefix="self-check-", dir=HERE) as temp:
        directory = Path(temp)
        WatchedGuard.output, WatchedGuard.start = directory, time.monotonic()
        def guard(): return WatchedGuard(Owner(), directory / "engine.log")
        g = guard()
        for _ in range(4): g.thresholds(dict(commit_available_bytes=3*GIB, physical_available_bytes=8*GIB, elapsed_seconds=1))
        assert not g.reason
        g.thresholds(dict(commit_available_bytes=3*GIB, physical_available_bytes=8*GIB, elapsed_seconds=1))
        assert g.reason and g.owner.stops == 1
        for commit, ram, elapsed in ((GIB,8*GIB,1),(8*GIB,GIB-1,1),(8*GIB,8*GIB,5400)):
            g=guard(); g.thresholds(dict(commit_available_bytes=commit,physical_available_bytes=ram,elapsed_seconds=elapsed))
            assert g.reason and g.owner.stops == 1
        g=guard()
        for _ in range(5): g.thresholds(dict(commit_available_bytes=8*GIB,physical_available_bytes=GIB,elapsed_seconds=1))
        assert g.reason
        reader=TraceReader(directory / "engine.log")
        assert not reader.line("strata trace: 10 MiB free after opening the expert cache")["enforced"]
        assert not reader.line("strata trace: prefill 310 MiB free after entry")["enforced"]
        assert reader.line("strata serve: WARNING: 255 MiB of VRAM free with everything loaded")["enforced"]
        assert reader.line("strata trace: prefill 290 MiB free after buffers")["enforced"]
        assert reader.line("strata trace: 200 MiB free after the completed request")["enforced"]
        assert reader.line("INFO context=131072 vram_free_mib=333")["cuda_free_mib"] == 333
        assert reader.line("strata serve: VRAM 512 MiB kept free: cache, 300 MiB free now")["cuda_free_mib"] == 300
        assert reader.line("strata generate: CUDA0 context ready, 1.50 GiB free before expert arena registration")["cuda_free_mib"] == 1536
        path=directory / "engine.log"
        path.write_bytes(b"strata trace: 10 MiB free after opening the expert cache\nstrata serve: 255 MiB of VRAM free with everything loaded\n")
        g=guard(); g.traces(); assert g.reason and g.owner.stops == 1
        assert g.reader.read() == []
        path.write_bytes(b"x")
        try: g.reader.read()
        except RuntimeError: pass
        else: raise AssertionError("truncation did not fail")
        g=guard()
        g.traces=lambda: (_ for _ in ()).throw(OSError("synthetic telemetry failure"))
        try: g.check()
        except RuntimeError: pass
        else: raise AssertionError("telemetry failure did not fail closed")
        assert g.owner.stops == 1 and "Telemetry error" in g.reason
        # Exercise all short-phase orchestration without HTTP or subprocesses.
        class FakeClient:
            def chat(self, messages, tokens, **kwargs):
                expected = "391" if self.label == "short-exact" else ("400" if self.label == "short-cached-followup" else "133")
                return {"choices": [{"message": {"role": "assistant", "content": expected}}],
                        "usage": {"completion_tokens": tokens}, "timings": {"cache_n": 20}}
        class FakeGuard:
            def check(self): pass
        record = {}
        assert short_tests(FakeClient(), record, lambda: None, FakeGuard(), 0)
        assert len(record["watched_short"]) == 8
        # Partial lines remain pending until completed, with a bounded reader.
        path.write_bytes(b"strata trace: 500 MiB free after the completed")
        reader = TraceReader(path)
        assert reader.read() == []
        with path.open("ab") as stream: stream.write(b" request\n")
        assert reader.read()[0]["enforced"]
    print("Self-check passed: synthetic commit/RAM/time/CUDA thresholds, stage gating, incremental parsing, truncation and telemetry failure; no model started.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--watched-short", action="store_true")
    parser.add_argument("--short-idle-seconds", type=float, default=60)
    extra, remainder=parser.parse_known_args()
    if extra.self_check:
        self_check(); return 0
    if not math.isfinite(extra.short_idle_seconds) or not 0 <= extra.short_idle_seconds <= 600:
        parser.error("--short-idle-seconds must be 0-600")
    # Parse only output here; the imported benchmark validates every original argument.
    locate=argparse.ArgumentParser(add_help=False)
    locate.add_argument("--output", type=Path)
    locate.add_argument("--config", type=Path)
    known,_=locate.parse_known_args(remainder)
    if known.output is None:
        return bench.main(remainder)
    output=known.output.resolve()
    if not output.is_relative_to(HERE) or output == HERE:
        parser.error("--output must be a new child directory under " + str(HERE))
    head=bench.subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    if head != EXPECTED_HEAD:
        parser.error("Expected exact source commit " + EXPECTED_HEAD)
    if known.config is not None:
        cfg = json.loads(known.config.read_text(encoding="utf-8-sig"))
        engine_args = cfg.get("args", [])
        if cfg.get("expert_profile_save") or any(str(arg).split("=", 1)[0] in
                {"--expert-profile-save", "--expert-profile-save-every"} for arg in engine_args):
            parser.error("Validation refuses expert-profile saving; use a read-only profile config")
    if extra.watched_short:
        incompatible={"--long-context","--held-out","--transfer-workload","--smoke-only"}.intersection(remainder)
        if incompatible: parser.error("--watched-short cannot combine with " + ", ".join(sorted(incompatible)))
        remainder += ["--smoke","--validation-only"]
    install(output,extra.watched_short,extra.short_idle_seconds)
    return bench.main(remainder)


if __name__ == "__main__":
    raise SystemExit(main())
