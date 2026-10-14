"""Record desktop/async stalls with other apps closed; --plan never starts a model."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import psutil

import benchmark_compact_cache as benchmark
import prepare_unsloth_q2 as preparation

ROOT = Path(__file__).resolve().parents[1]
CASES = (("desktop", "strata-unsloth-desktop-128k.json"),
         ("async", "strata-unsloth-desktop-async-128k.json"))
GPU_FIELDS = ("index", "memory.used", "memory.free", "utilization.gpu", "utilization.memory",
              "temperature.gpu", "power.draw", "clocks.sm", "clocks.mem", "pstate")


def windows_commit():
    """System commit from GetPerformanceInfo; this is not pagefile residency."""
    if os.name != "nt":
        raise OSError("Windows only")

    class PerformanceInfo(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32)] + [(name, ctypes.c_size_t) for name in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
            "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
            (name, ctypes.c_uint32) for name in ("HandleCount", "ProcessCount", "ThreadCount")]

    info = PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    get_info = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
    get_info.argtypes = [ctypes.POINTER(PerformanceInfo), ctypes.c_uint32]
    get_info.restype = ctypes.c_int
    if not get_info(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"used_bytes": info.CommitTotal * info.PageSize,
            "limit_bytes": info.CommitLimit * info.PageSize}


def collect_sample():
    sample = {"unix": time.time(), "errors": {}}

    def attempt(name, fn):
        try:
            sample[name] = fn()
        except Exception as exc:
            sample[name] = None
            sample["errors"][name] = type(exc).__name__

    attempt("available_ram_bytes", lambda: psutil.virtual_memory().available)
    attempt("commit", windows_commit)
    # Physical-disk counters, when enabled, are cumulative and system-wide.
    def disks():
        counters = psutil.disk_io_counters(perdisk=True)
        if not counters:
            raise OSError("No disk counters available")
        return {name: value._asdict() for name, value in counters.items()}

    attempt("disks", disks)

    def gpu():
        command = shutil.which("nvidia-smi")
        if not command:
            raise FileNotFoundError("nvidia-smi")
        result = subprocess.run([command, "--query-gpu=" + ",".join(GPU_FIELDS),
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                timeout=2, check=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return [dict(zip(GPU_FIELDS, (cell.strip() for cell in line.split(","))))
                for line in result.stdout.strip().splitlines()]

    attempt("gpu", gpu)
    sample["sample_finished_unix"] = time.time()
    return sample


class Telemetry:
    def __init__(self, path):
        self.path = path
        self.done = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.monitor, daemon=True)

    def monitor(self):
        try:
            with self.path.open("x", encoding="utf-8") as stream:
                for _ in range(2400):  # Bounded even if a run fails to finish.
                    if self.done.is_set():
                        return
                    stream.write(json.dumps(collect_sample()) + "\n")
                    stream.flush()
                    if self.done.wait(5):
                        return
                self.error = "Telemetry reached its sample limit."
        except Exception as exc:
            self.error = type(exc).__name__

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.thread.join(timeout=8)
        if self.thread.is_alive():
            self.error = "Telemetry did not stop within eight seconds."


def check_profiles():
    configs = []
    for _, filename in CASES:
        cfg = json.loads((ROOT / filename).read_text(encoding="utf-8-sig"))
        preparation.check_ready(cfg)
        if cfg.get("expert_profile_save") or "--expert-profile-save" in cfg["args"]:
            raise ValueError("Learned-profile saving must be disabled for this comparison.")
        configs.append(cfg)
    base, candidate = configs
    different = {key for key in set(base) | set(candidate) if base.get(key) != candidate.get(key)}
    if different != {"args", "log"} or candidate["args"] != base["args"] + ["--adapt-async", "1"]:
        raise ValueError("The diagnostic profiles must differ only by async swaps and log path.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--sample", action="store_true", help="Read one hardware sample; never starts a model")
    args = parser.parse_args(argv)
    output = None
    try:
        if args.sample:
            print(json.dumps(collect_sample(), indent=2))
            return 0
        check_profiles()
        print("Two diagnostic runs: desktop, then async. Six short responses each, plus warm-up.", flush=True)
        print("This runs Strata and sends prompts automatically. Do not launch another server or client.", flush=True)
        print("Each run includes four minutes idle between rounds to check behavior after a pause.", flush=True)
        print("CPU timing and hardware polling affect timings; use these runs to locate stalls.", flush=True)
        if args.plan:
            print("Readiness passed. No model or monitor was started.")
            return 0
        benchmark.assert_idle()
        output = ROOT / "bench/results" / ("desktop-stalls-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        output.mkdir(parents=True, exist_ok=False)
        print(f"Results: {output}", flush=True)
        notes = ["# Desktop stall diagnostics", "",
                 "Diagnostic timings only; same short workloads at 128K capacity, with a four-minute idle gap.",
                 "Each results.json records request/idle Unix timestamps. Match them to telemetry JSONL timestamps.",
                 "engine.log has CPU decode timing, cache/file counters, and async stats where enabled.",
                 "Commit is system commit, not pagefile usage. Disk counters are cumulative and system-wide.",
                 "GPU polling is sampled, not a per-kernel trace. Missing optional sensors are null plus an error type.",
                 "Async ms/round is cumulative elapsed time and can include idle gaps; it is not per-request stall time.",
                 "No nearly-full-context test, output-quality claim or automatic winning profile selection.", ""]
        (output / "README.md").write_text("\n".join(notes), encoding="utf-8")
        for label, filename in CASES:
            print(f"Starting {label}: loading the model automatically; leave this window open.", flush=True)
            with Telemetry(output / (label + "-telemetry.jsonl")) as monitor:
                result = benchmark.run(argparse.Namespace(config=ROOT / filename, output=output / label,
                    timing=True, smoke=False, smoke_only=False, idle_seconds=240))
            notes += [f"- {label}: {result['status']}; diagnostic decode {result['summary']['tokens_per_second']:.2f} tokens/s."]
            if monitor.error:
                notes += [f"  Telemetry incomplete: {monitor.error}"]
            (output / "README.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
        print(f"Finished. Saved diagnostics to {output}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Diagnostic stopped ({type(exc).__name__}).", file=sys.stderr)
        if output:
            print(f"Partial results: {output}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
