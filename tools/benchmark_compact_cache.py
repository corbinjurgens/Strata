"""Run one guarded, local-only 128K cache benchmark; --timing is diagnostic only."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error


def worker_self_identity():
    """Observe this interpreter itself; the pseudo-handle is never cleanup authority."""
    if os.name != "nt":
        raise RuntimeError("Native bootstrap identity requires Windows.")
    import ctypes as c

    class FileTime(c.Structure):
        _fields_ = [("low", c.c_uint32), ("high", c.c_uint32)]

    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes, kernel.GetCurrentProcess.restype = [], c.c_void_p
    kernel.GetProcessId.argtypes, kernel.GetProcessId.restype = [c.c_void_p], c.c_uint32
    kernel.GetProcessTimes.argtypes = [c.c_void_p] + [c.POINTER(FileTime)] * 4
    kernel.GetProcessTimes.restype = c.c_int32
    handle = kernel.GetCurrentProcess()
    times = [FileTime() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(c.byref(value) for value in times)):
        raise c.WinError(c.get_last_error())
    pid = kernel.GetProcessId(handle)
    if not pid:
        raise c.WinError(c.get_last_error())
    return int(pid), (int(times[0].high) << 32) | int(times[0].low)


def worker_bootstrap(stream, worker, *, pid=None, settings=None):
    """The direct interpreter waits here before importing site or any project code."""
    line = stream.readline(257)
    if len(line) > 256 or not line.endswith("\n"):
        raise RuntimeError("Missing or oversized START envelope.")
    envelope = json.loads(line)
    if envelope.get("command") != "START":
        raise RuntimeError("Bootstrap requires START.")
    identity = envelope.get("identity", {})
    for key, maximum in (("pid", 2**32 - 1), ("creation_filetime", 2**64 - 1)):
        value = identity.get(key)
        if (type(value) is not str or re.fullmatch(r"[1-9][0-9]*", value) is None
                or len(value) > 20 or int(value) > maximum):
            raise ValueError("Invalid native identity transport.")
    if int(identity["pid"]) != (os.getpid() if pid is None else pid):
        raise RuntimeError("START identity does not name this interpreter.")
    observed = worker_self_identity()
    if (type(observed) is not tuple or len(observed) != 2
            or any(type(value) is not int or not 0 < value <= maximum
                   for value, maximum in zip(observed, (2**32 - 1, 2**64 - 1)))
            or observed != (int(identity["pid"]), int(identity["creation_filetime"]))):
        raise RuntimeError("START does not match this interpreter's native creation identity.")
    settings = os.environ.pop("STRATA_BENCHMARK_BOOTSTRAP", "") if settings is None else settings
    if type(settings) is not str or len(settings) > 16384:
        raise ValueError("Invalid bootstrap settings.")
    settings = json.loads(settings)
    envelope.update(args=settings["args"], python_path=settings["python_path"],
                    work_deadline=settings["work_deadline"])
    return worker(envelope)


if __name__ == "__main__" and sys.argv[1:] == ["--internal-acceptance-worker"]:
    def _after_start(envelope):
        # -I -S prevents environment/site startup code before containment.
        # Only the parent-selected dependency paths are installed, after START.
        sys.path[:] = envelope["python_path"]
        module = importlib.util.spec_from_file_location("benchmark_worker", __file__)
        loaded = importlib.util.module_from_spec(module)
        module.loader.exec_module(loaded)
        return loaded.acceptance_worker(envelope)
    raise SystemExit(worker_bootstrap(sys.stdin, _after_start))

import psutil

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ROOT / "bench/results/2026-10-09-unsloth-file-cache"
SCRATCH = ROOT / "build-unsloth"
PORT = 8081
CONTEXT = 131072
LOG_LIMIT = 4 * 1024**2
ACCEPTANCE_LOG_LIMIT = 1024**2
ACCEPTANCE_RECEIPT_LIMIT = 6 * 1024**2
ACCEPTANCE_ARTIFACT_LIMIT = 12 * 1024**2
ACCEPTANCE_SECONDS = 900
CLEANUP_SECONDS = 10
PROFILE_ENV = {"STRATA_VERIFY_PROFILE", "STRATA_DECODE_TIMING", "STRATA_FETCH_PROFILE"}
# Modes follow the actual getenv/environ consumers, not the spelling of a control.
DIAGNOSTIC_ENV = dict.fromkeys((
    "STRATA_VERIFY_PROFILE", "STRATA_DECODE_TIMING", "STRATA_SPLIT_TIMING", "STRATA_TRACE",
    "STRATA_POOL_TRACE", "STRATA_RSS_TRACE", "STRATA_TRACE_ADAPT", "STRATA_VERIFY_DEBUG",
    "STRATA_DRAFT_TIMING", "STRATA_PREFILL_TIMING", "STRATA_LOGPOS", "STRATA_DUMP_FIRST_LOGITS",
    "STRATA_VERIFY_ARENA", "STRATA_DBG_NAN", "STRATA_IDX_FP16_CHECK", "STRATA_QSA_DUMP",
    "STRATA_PREFILL_DUMP_R", "STRATA_PREFILL_DUMP_R_ALL", "STRATA_STATE_HASH", "STRATA_STATE_HASH_GDN",
    "STRATA_SNAPSHOT_VERIFY", "STRATA_VERIFY_NODES", "STRATA_DBG_FEAT",
    "STRATA_DBG_ROUTE_OVERRIDE", "STRATA_DBG_CPU_GATE", "STRATA_LAYER_SPLIT_GATE"), "presence")
DIAGNOSTIC_ENV.update({"STRATA_BATCH_TRACE": "nonempty", "STRATA_DEBUG": "nonempty",
    "STRATA_REQUEST_LINES": "server-log", "STRATA_LOGPOS_EXTRA": "logpos", "STRATA_LOGPOS_TOPK": "logpos",
    "STRATA_QSA_DUMP_LAST": "qsa", "STRATA_HC_UPMIX_CHECK": "positive", "STRATA_CVEC_FUSE_CHECK": "positive",
    "STRATA_FETCH_PROFILE": "one", "STRATA_FILE_COPY_STATS": "one", "STRATA_IO_STATS": "atoi",
    "STRATA_PREFILL_DEMAND_STATS": "first-one", "STRATA_DBG_GDN": "first-one",
    "STRATA_DBG_FORCE_SLOW_LAYER": "first-one", "STRATA_DBG_FEAT_LAYERS": "feature",
    "STRATA_TEST_PAGEABLE": "first-one", "STRATA_TEST_CACHE_FAIL": "positive",
    "STRATA_FS_VERIFY": "positive",
    "STRATA_TEST_VERIFY_STALL": "positive",
    "STRATA_VERIFY_TRACE": "nonzero",
    "STRATA_PIPELINE_LOG": "pipeline", "STRATA_PIPELINE_TRACE": "pipeline",
    "STRATA_PIPELINE_FORCE_MISS": "pipeline-positive", "CUDA_LAUNCH_BLOCKING": "atoi"})
DIAGNOSTIC_GATES = {"STRATA_PIPELINE_DEBUG", "STRATA_VERIFY_DEVICE_PLAN", "STRATA_VERIFY_QDEDUP",
                    "STRATA_VERIFY_COHERENT", "STRATA_VERIFY_ALL_RESIDENT"}  # Preserve optimization selections.
ACCEPTANCE_ENV = set(DIAGNOSTIC_ENV) | {"CUDA_MODULE_LOADING", "STRATA_FILE_COPY_READY"}
EVIDENCE_ENV = PROFILE_ENV | {
    "STRATA_RESIDENT_HEADROOM_GIB", "STRATA_FILE_CACHE_MIB", "STRATA_FILE_CACHE_COMPACT", "STRATA_FILE_DIRECT_UNMAP",
    "STRATA_KV_GROW", "STRATA_ARENA_PIN_GIB", "STRATA_PARTIAL_PIN_GIB", "STRATA_PARTIAL_PIN",
    "STRATA_UNBUFFERED_LOAD", "STRATA_FILE_RELEASE", "STRATA_DEC_BATCH", "STRATA_DEFERRED_REGISTER",
    "STRATA_OWNED_PRICE", "STRATA_FETCH_PVM",
}
# Source-reviewed tuning/treatment names, independently of the evidence projection.
# generate.cpp: host/cache budgets, KV growth, pin cap, pricing, pipeline gate/theta;
# expert_source.cpp: file I/O/release/ready-copy, partial pin, batching/registration;
# verify.cpp: device planning, quantization reuse and coherent/resident placement.
# prefill/gemm.cu: HIP tuning-table path selects GEMM algorithms; table contents are unbound;
# core/pinned.cu: NO_ARENA_THP presence skips the non-Windows arena's THP request;
# program/generate.cpp: RESIDENT_PIN exact "0" selects pageable resident experts.
# Review means tuning rather than diagnostic instrumentation, not runtime admission
# or permission to pool different treatments. New names are unclassified by default.
NORMAL_ENV = {
    "STRATA_RESIDENT_HEADROOM_GIB", "STRATA_FILE_CACHE_MIB", "STRATA_FILE_CACHE_COMPACT",
    "STRATA_FILE_DIRECT_UNMAP", "STRATA_KV_GROW", "STRATA_ARENA_PIN_GIB",
    "STRATA_PARTIAL_PIN_GIB", "STRATA_PARTIAL_PIN", "STRATA_UNBUFFERED_LOAD",
    "STRATA_FILE_RELEASE", "STRATA_DEC_BATCH", "STRATA_DEFERRED_REGISTER", "STRATA_OWNED_PRICE",
    "STRATA_FETCH_PVM", "STRATA_FILE_COPY_READY", "STRATA_PIPELINE_DEBUG", "STRATA_PIPELINE_THETA",
    "STRATA_VERIFY_DEVICE_PLAN", "STRATA_VERIFY_QDEDUP", "STRATA_VERIFY_COHERENT",
    "STRATA_VERIFY_ALL_RESIDENT", "CUDA_MODULE_LOADING",
    "STRATA_HIPBLASLT_TUNING", "STRATA_NO_ARENA_THP", "STRATA_RESIDENT_PIN",
}
NORMAL_CONTROL_LIMIT = 128
NORMAL_ENVIRONMENT_LIMIT = 64 * 1024
NORMAL_SCOPE = "STRATA_*; CUDA_MODULE_LOADING; CUDA_LAUNCH_BLOCKING"
LONG_TARGET = CONTEXT - 192
SUSTAINED_TARGET = 128000
RETRIEVAL_CODES = ["field-cedar-284", "marble-finch-693", "cove-lantern-517"]
HELD_OUT = (
    ("python-aliasing", "Trace this Python code: rows = [[0]] * 3; rows[1].append(7). "
     "Reply only with FINAL: followed by the resulting JSON value of rows. Do not include an explanation.",
     [[0, 7], [0, 7], [0, 7]]),
    ("python-generator", "What list does this Python code print? "
     "values = [2, 4, 6]; stream = (x * 2 for x in values); values[1] = 9; "
     "print(list(stream)). Reply only with FINAL: followed by the resulting JSON list. "
     "Do not include an explanation.", [4, 18, 12]),
    ("dependency-schedule", "Four jobs share two identical workers. A takes 3 minutes, B takes 5, "
     "C takes 4 and must wait for A, D takes 2 and must wait for both B and C. "
     "All available jobs may start at time 0, and jobs cannot be interrupted. What is the earliest "
     "finish time for all jobs? Reply only with FINAL: followed by the integer number of minutes. "
     "Do not include an explanation.", 9),
    ("interval-union", "An integer counter increments once per integer that appears in at least one "
     "of these inclusive ranges: [3,11], [8,16], [20,23], [22,25]. What is the count? "
     "Reply only with FINAL: followed by the integer count. Do not include an explanation.", 20),
)
TRANSFER_WORKLOAD = (
    ("python-run-lengths", "Write a concise complete pure Python function run_lengths(items) that returns "
     "a list of (value, count) tuples for consecutive equal values, preserving run order. Handle an empty "
     "input without special sentinel values or imports, and do not modify items. Expected examples: "
     "run_lengths([3,3,1,3,3]) == [(3,2),(1,1),(3,2)]; run_lengths([]) == []; "
     "run_lengths([None,None,2]) == [(None,2),(2,1)]. Reply with one Python code fence containing a "
     "brief docstring and the complete function only.",
     "Manually review complete syntax, consecutive runs, empty input, None values and unchanged input. "
     "Expected examples: [3,3,1,3,3] -> [(3,2),(1,1),(3,2)]; [] -> []; "
     "[None,None,2] -> [(None,2),(2,1)]. Generated code is not executed by the harness."),
    ("python-mutation-debug", "Debug this Python function: def remove_negatives(values):\n"
     "    for i in range(len(values)):\n        if values[i] < 0:\n            values.pop(i)\n"
     "    return values\nExplain precisely what happens for [2,-1,-3,4], including the skipped element "
     "and eventual exception. Then give a concise corrected pure Python function that returns a new "
     "list without negative numbers while preserving order, and state whether it changes the input. "
     "Keep the answer under 100 words plus the code.",
     "Expected trace: pop -1 at index 1 -> [2,-3,4], index 2 visits 4 and skips -3, index 3 raises "
     "IndexError. A correct new-list implementation returns [2,4] and leaves the input unchanged."),
    ("inventory-arithmetic", "A warehouse starts with 240 usable units. It discovers 12 damaged units "
     "and removes them, ships one third of the remaining units, then receives 5 boxes containing "
     "16 usable units each. It reserves one quarter of the resulting inventory for an event. "
     "How many units remain available for other orders? Show each intermediate quantity, state what "
     "each fraction applies to, and finish by checking that shipped, damaged, reserved and available "
     "units reconcile with the initial stock and deliveries. Use at most 110 words.",
     "Expected arithmetic: 240-12=228; shipped=76; remaining=152; delivered=80; inventory=232; "
     "reserved=58; available=174. Check 12+76+58+174=320=240+80."),
    ("project-note-synthesis", "Turn these fictional project notes into one coherent 85-100 word "
     "status paragraph for a community library, preserving the facts and distinguishing completed "
     "work from plans: Three volunteers scanned 600 pages on Monday. Seven duplicate scans were "
     "removed. The remaining scans still need proofreading. A scanner overheated once; a pause "
     "resolved it, but the cause is unknown. Friday's session is planned, subject to a cooling test. "
     "Mira will run that test Thursday and report the result. Do not invent a completion date, a "
     "diagnosed cause, or additional staffing.",
     "Manual review: 593 scans remain and still need proofreading; overheating occurred once with "
     "unknown cause; Friday is conditional on Thursday's cooling test by Mira; no invented dates, "
     "cause, staffing or completed proofreading. Requested length is 85-100 words."),
)


def load_workload(name):
    spec = importlib.util.spec_from_file_location("compact_cache_" + name, WORKLOADS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def clean_environment(cfg, timing, fetch_timing=False, *, acceptance=False, inherited=None):
    """Disable inherited profiling and make server/client use the same saved API key."""
    cfg = copy.deepcopy(cfg)
    excluded = (ACCEPTANCE_ENV if acceptance else PROFILE_ENV) | {"STRATA_API_KEY"}
    inherited = os.environ if inherited is None else inherited
    env = {k: v for k, v in inherited.items() if k.upper() not in excluded}
    saved_excluded = {"STRATA_API_KEY"} if acceptance else excluded
    cfg["env"] = {k: v for k, v in cfg.get("env", {}).items() if k.upper() not in saved_excluded}
    if acceptance:
        # Explicit treatments remain explicit; only an unselected inherited treatment is removed.
        names = {key.upper() if os.name == "nt" else key for key in cfg["env"]}
        for key, value in (("CUDA_MODULE_LOADING", "EAGER"), ("STRATA_FILE_COPY_READY", "0")):
            if key not in names:
                cfg["env"][key] = value
    if timing:
        cfg["env"]["STRATA_DECODE_TIMING"] = "1"
    if fetch_timing:
        cfg["env"]["STRATA_FETCH_PROFILE"] = "1"
    if acceptance:
        # These controls are consumed by the server itself, before child_env runs.
        for key, value in cfg["env"].items():
            name = key.upper() if os.name == "nt" else key
            if name in {"STRATA_BATCH_TRACE", "STRATA_DEBUG", "STRATA_REQUEST_LINES"}:
                env[name] = str(value)
    return cfg, env


def file_identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def executable_provenance(exe, metadata):
    """A matching receipt records a build claim; current checkout HEAD is never inferred."""
    result = {"executable": file_identity(exe), "source_provenance": "source-unproven",
              "build_receipt": None, "build_claim": None}
    if metadata is not None and metadata.is_file():
        if metadata.stat().st_size > ACCEPTANCE_LOG_LIMIT:
            raise ValueError("Build receipt exceeds its evidence budget.")
        receipt = json.loads(metadata.read_text(encoding="utf-8-sig"))
        if receipt.get("sha256") != result["executable"]["sha256"]:
            raise ValueError("Build receipt does not match the selected executable.")
        result["build_receipt"] = file_identity(metadata)
        result["build_claim"] = {key: receipt.get(key) for key in ("commit", "version", "src", "ggml")}
        result["source_provenance"] = "matching receipt claim; current source/dependency admission still required"
    return result


def protection_snapshot(paths):
    rows = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("A protected input is missing or is a symbolic link.")
        rows.append(file_identity(path))
    return rows


def safe_engine_environment(cfg, inherited, *, acceptance=False):
    """Record allowed controls from both actual environment layers, with config taking precedence."""
    from serve.server import config_environment
    effective = {}
    allowed = EVIDENCE_ENV | ACCEPTANCE_ENV | DIAGNOSTIC_GATES if acceptance else EVIDENCE_ENV
    for key, value in config_environment(cfg, inherited).items():
        name = key.upper() if os.name == "nt" else key
        if name in allowed:
            effective[name] = value
    return effective


def active_diagnostics(environment, *, server_log=False, backend=None):
    def atoi_active(value, *, positive=False):
        match = re.match(r"\s*([+-]?[0-9]+)", value)
        if match is None:
            return False
        number = int(match[1])
        # Out-of-range C atoi is not portable evidence of an inactive diagnostic.
        return not -(2**31) <= number < 2**31 or (number > 0 if positive else number != 0)

    def active(name):
        if name not in environment:
            return False
        value, mode = environment[name], DIAGNOSTIC_ENV[name]
        if mode == "presence":
            return True
        if mode == "one":
            return value == "1"
        if mode == "first-one":
            return bool(value) and value[0] == "1"
        if mode == "nonempty":
            return bool(value)
        if mode == "server-log":
            return bool(value) and server_log
        if mode == "pipeline":
            gate = environment.get("STRATA_PIPELINE_DEBUG", "")
            return bool(gate) and gate[0] != "0"
        if mode == "pipeline-positive":
            gate = environment.get("STRATA_PIPELINE_DEBUG", "")
            return bool(gate) and gate[0] != "0" and atoi_active(value, positive=True)
        if mode in ("logpos", "qsa"):
            return ("STRATA_LOGPOS" if mode == "logpos" else "STRATA_QSA_DUMP") in environment
        if mode == "feature":
            return "STRATA_DBG_FEAT" in environment or "STRATA_DBG_ROUTE_OVERRIDE" in environment
        if mode in ("positive", "atoi"):
            if name == "CUDA_LAUNCH_BLOCKING" and backend == "hip":
                return False
            return atoi_active(value, positive=mode == "positive")
        return bool(value) and value[0] != "0"
    return [name for name in DIAGNOSTIC_ENV if active(name)]


def diagnostic_environment(environment, *, server_log=False, backend=None):
    return bool(active_diagnostics(environment, server_log=server_log, backend=backend))


def normal_comparison_excluded(reason):
    return {"schema": 1, "complete": False, "eligible": False, "controls": [],
            "scope": NORMAL_SCOPE, "name_semantics": "windows" if os.name == "nt" else "posix",
            "environment_sha256": None, "exclusions": [{"reason": reason}]}


def normal_comparison_environment(cfg, inherited, *, server_log=False):
    """Classify the full composed control environment; never export unknown names/values.

    The digest binds a sorted JSON array of original name/value string pairs, not
    semantic review, actual native consumption, allocations or operational admission.
    """
    from serve.server import config_environment
    environment = config_environment(cfg, inherited)
    if len(environment) > 4096:
        return normal_comparison_excluded("environment-limit")
    pairs, effective, size = [], {}, 0
    for key, value in environment.items():
        if type(key) is not str or type(value) is not str:
            return normal_comparison_excluded("invalid-environment")
        name = key.upper() if os.name == "nt" else key
        if not name.startswith("STRATA_") and name not in {"CUDA_MODULE_LOADING", "CUDA_LAUNCH_BLOCKING"}:
            continue  # Ordinary names such as Windows ProgramFiles(x86) are outside the scope.
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", key) is None or "\0" in value:
            return normal_comparison_excluded("invalid-environment")
        # Bound before encoding/parsing, including diagnostic integer prefixes.
        if len(value) > 4096 or len(pairs) >= NORMAL_CONTROL_LIMIT:
            return normal_comparison_excluded("control-inventory-limit")
        size += len(key.encode("utf-8")) + len(value.encode("utf-8", errors="surrogatepass"))
        if size > NORMAL_ENVIRONMENT_LIMIT:
            return normal_comparison_excluded("control-inventory-limit")
        pairs.append([key, value])
        if name in effective and effective[name] != value:
            return normal_comparison_excluded("ambiguous-windows-alias")
        effective[name] = value
    active = set(active_diagnostics(effective, server_log=server_log, backend=cfg.get("backend")))
    controls, exclusions = [], []
    for name in sorted(effective):
        if name in DIAGNOSTIC_ENV:
            item = {"control": name, "kind": "diagnostic", "active": name in active}
            if item["active"]:
                exclusions.append({"reason": "active-diagnostic", "control": name})
        elif name in NORMAL_ENV:
            item = {"control": name, "kind": "reviewed-normal"}
        else:
            identifier = "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest()
            item = {"control": identifier, "kind": "unclassified"}
            exclusions.append({"reason": "unclassified-control", "control": identifier})
        controls.append(item)
    return {"schema": 1, "complete": True, "eligible": not exclusions,
            "scope": NORMAL_SCOPE, "name_semantics": "windows" if os.name == "nt" else "posix",
            "controls": controls, "environment_sha256": json_digest(sorted(pairs)), "exclusions": exclusions}


def normal_comparison_launch_environment(cfg, inherited, *, server_log=False):
    """Retain source classification without claiming a final automatic environment binding."""
    evidence = normal_comparison_environment(cfg, inherited, server_log=server_log)
    return normal_comparison_binding(cfg, evidence)


def normal_comparison_binding(cfg, evidence):
    """Apply launch binding after callers have used the source classification."""
    # The server refreshes policy-owned values after this classification. Its
    # numeric launch metrics cannot recover the exact final name/value bytes.
    if cfg.get("memory_policy"):
        evidence.update(complete=False, eligible=False, environment_sha256=None)
        evidence["exclusions"].append({"reason": "automatic-environment-unbound"})
    return evidence


def consume_normal_comparison(record, cfg, inherited, *, server_log=False):
    """Recompute from independently supplied inputs; old/partial/mutated records fail closed.

    Automatic launches remain unbound even when all prelaunch controls are reviewed.
    Workload and lifecycle outcomes retain their own predicates, including for useful
    diagnostic/UNKNOWN records.
    """
    if type(record) is not dict:
        return normal_comparison_excluded("missing-or-inconsistent-classification")
    expected = normal_comparison_launch_environment(cfg, inherited, server_log=server_log)
    observed = record.get("normal_comparison")
    def same(actual, wanted):
        # Traverse only the bounded expected shape, never serialize untrusted data.
        if type(actual) is not type(wanted):
            return False
        if type(wanted) is dict:
            return actual.keys() == wanted.keys() and all(same(actual[key], value) for key, value in wanted.items())
        if type(wanted) is list:
            return len(actual) == len(wanted) and all(same(a, b) for a, b in zip(actual, wanted))
        return actual == wanted
    if not same(observed, expected):
        return normal_comparison_excluded("missing-or-inconsistent-classification")
    return expected


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def message_identity(message):
    return {"role": message["role"], "characters": len(message["content"]),
            "content_sha256": hashlib.sha256(message["content"].encode("utf-8")).hexdigest(),
            "message_sha256": json_digest(message)}


def source_identity(deadline=None):
    """Working-tree evidence only: this does not claim an old executable used these sources."""
    paths = ["src", "include", "cmake", "CMakeLists.txt", "serve"]

    def git(*args):
        timeout = 30 if deadline is None else min(30, deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("Absolute provenance deadline reached.")
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, timeout=timeout,
                              check=True).stdout

    names = sorted(set(git("ls-files", "-z", "--cached", "--others", "--exclude-standard",
                           "--", *paths).decode("utf-8").split("\0")) - {""})
    files, aggregate = [], hashlib.sha256()
    for name in names:
        path = ROOT / name
        item = file_identity(path) if path.is_file() else {"path": name, "missing": True}
        item["path"] = name
        files.append(item)
        encoded = json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8")
        aggregate.update(len(encoded).to_bytes(8, "big"))
        aggregate.update(encoded)
    return {"meaning": "Working tree at run start, not executable build provenance.",
            "head": git("rev-parse", "HEAD").decode().strip(), "sha256": aggregate.hexdigest(),
            "status": git("status", "--porcelain=v1", "--untracked-files=all", "--", *paths).decode(),
            "files": files}


def summarize(rows):
    if not rows:
        return None
    tokens = sum(row["response"]["usage"]["completion_tokens"] for row in rows)
    decode_ms = sum(row["response"]["timings"]["predicted_ms"] for row in rows)
    prefill = [row["response"]["timings"].get("prompt_ms") for row in rows]
    return {"completion_tokens": tokens, "predicted_ms": decode_ms,
            "tokens_per_second": tokens * 1000 / decode_ms if decode_ms else None,
            "prompt_ms": sum(prefill) if all(value is not None for value in prefill) else None,
            "prompt_tokens": sum(row["response"]["usage"]["prompt_tokens"] for row in rows)}


def held_out_answer(text, expected):
    def same_json(actual, wanted):
        if type(actual) is not type(wanted):
            return False
        if isinstance(wanted, list):
            return len(actual) == len(wanted) and all(same_json(a, b) for a, b in zip(actual, wanted))
        if isinstance(wanted, dict):
            return actual.keys() == wanted.keys() and all(same_json(actual[k], v) for k, v in wanted.items())
        return actual == wanted
    try:
        return same_json(json.loads(text.rsplit("FINAL:", 1)[1].strip()), expected)
    except (IndexError, ValueError):
        return False


def exact_answer(text, values):
    """Allow surrounding/list punctuation, but no extra words, digits or arithmetic expressions."""
    punctuation = r"[\s,;:()\[\]{}\"'`*.!]"
    pattern = punctuation + "*" + (punctuation + "+").join(re.escape(str(value)) for value in values)
    return re.fullmatch(pattern + punctuation + "*", text) is not None


def prompt_counter(cfg):
    # The same files and Service.encode_prompt path used by validation.long_messages and the server.
    from strata_tokenizer import Tokenizer
    from serve.server import Service
    from serve.frontend import ChatTemplate, literal_tags

    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for text, index in vocab.items():
        tokens[index] = text
    tokenizer = Tokenizer(tokens, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                          json.loads((tpath / "token_type.json").read_text()))
    encoder = Service.__new__(Service)
    encoder.tok, encoder.template = tokenizer, ChatTemplate(tpath / "chat_template.jinja")
    encoder.effort_end, encoder.literals = False, literal_tags(tokenizer.control_tokens)
    return lambda messages: len(encoder.encode_prompt(messages, None, {"enable_thinking": False}))


def check_authentication(client, record, save):
    """Prove both missing and wrong credentials fail before sending model work."""
    record["authentication"] = []
    for label, headers in (("missing", {}), ("wrong", {"Authorization": "Bearer benchmark-invalid-key"})):
        request = urllib.request.Request(client.base + "/metrics", headers=headers)
        try:
            with TimeoutOpener(getattr(client, "deadline", None)).open(request, timeout=20) as response:
                code = response.status
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
        record["authentication"].append({"credentials": label, "status": code, "passed": code == 401})
        save()
        if code != 401:
            raise RuntimeError("Authentication validation failed for " + label + " credentials.")


def held_out_tests(client, record, save):
    rows = record["held_out"] = []
    for name, prompt, expected in HELD_OUT:
        client.label = "held-out-" + name
        response = client.chat([{"role": "user", "content": prompt}], 160)
        answer = response["choices"][0]["message"].get("content") or ""
        rows.append({"case": name, "expected": expected, "passed": held_out_answer(answer, expected),
                     "response": response, "metric": record["requests"][-1]["metric"]})
        record["held_out_summary"] = summarize(rows)
        save()
        print("Completed held-out case " + name + ".", flush=True)
    record["held_out_passed"] = all(row["passed"] for row in rows)
    save()
    if not record["held_out_passed"]:
        raise RuntimeError("Held-out validation failed; inspect saved answers and assertions.")


def transfer_workload_tests(client, record, save):
    rows = record["transfer_workload"] = []
    record["transfer_quality"] = "needs_manual_review"
    for name, prompt, review in TRANSFER_WORKLOAD:
        client.label = "transfer-" + name
        response = client.chat([{"role": "user", "content": prompt}], 160,
                               temperature=0, chat_template_kwargs={"enable_thinking": False})
        rows.append({"case": name, "prompt": prompt, "review_expectations": review,
                     "quality": "needs_manual_review", "response": response,
                     "metric": record["requests"][-1]["metric"]})
        record["transfer_summary"] = summarize(rows)
        save()
        print("Completed transfer case " + name + "; quality needs manual review.", flush=True)


def long_context_tests(validation, client, record, save):
    # Reuse the existing deterministic tokenizer/template-aware generator.
    messages, codes, count = validation.long_messages(client.cfg, LONG_TARGET)
    if not LONG_TARGET - 16 <= count <= LONG_TARGET:
        raise RuntimeError("Long prompt generator did not reach its near-128K target.")
    rows = record["long_context"] = []
    client.label = "long-fresh-fit"
    response = client.chat(messages, 256)
    answer = response["choices"][0]["message"].get("content") or ""
    usage, timing = response["usage"], response["timings"]
    checks = {"all_codes_in_order": exact_answer(answer, codes),
              "full_prompt_retained": usage["prompt_tokens"] == count,
              "fresh_prefill": timing.get("prompt_n", 0) >= count - 1024,
              "fit_request_exceeds_capacity": count + 256 > CONTEXT,
              "completion_fits": 0 < usage["completion_tokens"] <= min(256, CONTEXT - count - 8)}
    rows.append({"case": "fresh-near-128k-and-fit", "local_prompt_tokens": count,
                 "requested_max_tokens": 256, "expected": codes, "checks": checks,
                 "passed": all(checks.values()), "response": response,
                 "metric": record["requests"][-1]["metric"]})
    save()
    if not rows[-1]["passed"]:
        raise RuntimeError("Fresh long-context/fit validation failed; inspect saved checks.")
    messages += [response["choices"][0]["message"], {"role": "user", "content":
        "Recall the LABEL3 code and calculate 17 times 23. Reply with the code and integer only."}]
    expected_reuse_tokens = prompt_counter(client.cfg)(messages)
    record["expected_reuse_prompt_tokens"] = expected_reuse_tokens
    save()
    client.label = "long-reuse"
    response = client.chat(messages, 128)
    answer = response["choices"][0]["message"].get("content") or ""
    usage, timing = response["usage"], response["timings"]
    checks = {"retrieval_and_arithmetic": exact_answer(answer, [codes[2], 391]),
              "long_prefix_reused": timing.get("cache_n", 0) >= count - 1024,
              "small_new_prefill": 0 < timing.get("prompt_n", 0) < 1024,
              "full_history_retained": usage["prompt_tokens"] == expected_reuse_tokens,
              "completion_fits": 0 < usage["completion_tokens"] <= 128
              and usage["prompt_tokens"] + usage["completion_tokens"] + 8 <= CONTEXT}
    rows.append({"case": "near-128k-reuse", "expected_prompt_tokens": expected_reuse_tokens,
                 "checks": checks, "passed": all(checks.values()),
                 "response": response, "metric": record["requests"][-1]["metric"]})
    record["long_context_passed"] = all(row["passed"] for row in rows)
    record["long_context_summary"] = summarize(rows)
    save()
    if not record["long_context_passed"]:
        raise RuntimeError("Long-context reuse validation failed; inspect saved checks.")


def acceptance_selection(args):
    """Keep the historical fit test and each acceptance workload separate."""
    context = getattr(args, "expected_context", CONTEXT)
    if type(context) is not int or context not in (65536, CONTEXT):
        raise ValueError("Expected context must be 65536 or 131072.")
    selection = getattr(args, "acceptance_workload", None)
    cohort = getattr(args, "cohort", None)
    if selection is None:
        if (cohort is not None or getattr(args, "candidate_exe", None) is not None
                or getattr(args, "protect_input", [])):
            raise ValueError("Cohort, executable and protection controls require an explicit acceptance workload.")
        if getattr(args, "long_context", False) and context != CONTEXT:
            raise ValueError("The historical clipped-fit test requires 131072 context.")
        return None, context
    if selection not in ("short", "near-full"):
        raise ValueError("Unknown acceptance workload.")
    if not getattr(args, "acceptance_telemetry", False) and getattr(args, "_acceptance_deadline", None) is None:
        raise ValueError("Acceptance workloads require --acceptance-telemetry and the contained worker.")
    if cohort not in ("saved-static", "automatic", "frozen-control"):
        raise ValueError("Acceptance workloads require an explicit cohort.")
    if any(getattr(args, name, False) for name in (
            "smoke", "smoke_only", "held_out", "transfer_workload", "long_context", "validation_only")):
        raise ValueError("Select the short or near-full workload without other workload switches.")
    if getattr(args, "idle_seconds", 0):
        raise ValueError("Acceptance workloads use their own final 30-second resident idle.")
    if selection == "near-full" and (context != CONTEXT or cohort != "automatic"):
        raise ValueError("Near-full acceptance requires the automatic 131072-context cohort.")
    if getattr(args, "_acceptance_deadline", None) is None:
        for name in ("acceptance_physical_gib", "acceptance_commit_gib"):
            value = getattr(args, name, None)
            if not valid_number(value) or value < 4:
                raise ValueError("Selected acceptance workloads require explicit physical and commit floors of at least 4 GiB.")
    return selection, context


def acceptance_case_specs(selection):
    def case(label, tokens, expected=None, minimum=1, fresh=False, cached_from=None, target=None):
        return {"case": label, "tokens": tokens, "expected": copy.deepcopy(expected), "minimum": minimum,
                "fresh": fresh, "cached_from": cached_from, "target": target}
    if selection == "short":
        rows = [case("startup-exact", 32, [391]), case("short-exact", 32, [391]),
                case("short-cached-followup", 32, [400], cached_from="short-exact")]
        for index in range(1, 4):
            label = f"sustained-{index}"
            rows += [case(label, 512, minimum=512), case(label + "-followup", 32, [133], cached_from=label)]
        return rows + [case("2k-fresh", 128, RETRIEVAL_CODES, fresh=True, target=2048),
                       case("2k-cached", 128, [RETRIEVAL_CODES[2], 391], cached_from="2k-fresh")]
    if selection == "near-full":
        return [case("near-full-fresh", 2048, RETRIEVAL_CODES, fresh=True, target=SUSTAINED_TARGET),
                case("near-full-cached", 128, [RETRIEVAL_CODES[2], 391], cached_from="near-full-fresh"),
                case("near-full-sustained", 2048, minimum=512, fresh=True, target=SUSTAINED_TARGET)]
    raise ValueError("Unknown acceptance workload.")


def acceptance_case_checks(spec, count, response, context, prior_count=None, metric=None):
    if type(count) is not int or count <= 0:
        raise ValueError("Invalid encoded prompt count.")
    text = response["choices"][0]["message"].get("content") or ""
    usage, timings = response["usage"], response["timings"]
    generated, cached, read = usage["completion_tokens"], timings.get("cache_n"), timings.get("prompt_n")
    metric = metric or {}
    checks = {"nonempty": bool(text.strip()),
              "successful_finish": metric.get("finish") in ("stop", "length")
              and metric["finish"] == response["choices"][0].get("finish_reason"),
              "output_room": count + spec["tokens"] + 8 <= context,
              "full_history_retained": type(usage["prompt_tokens"]) is int and usage["prompt_tokens"] == count,
              "completion_length": type(generated) is int and spec["minimum"] <= generated <= spec["tokens"],
              "engine_prompt_accounting": type(cached) is int and type(read) is int
              and cached >= 0 and read >= 0 and cached + read == count,
              "native_token_accounting": type(metric.get("prompt_total")) is int
              and metric["prompt_total"] == count and type(metric.get("prompt_read")) is int
              and metric["prompt_read"] == read and type(metric.get("engine_generated")) is int
              and generated <= metric["engine_generated"] <= spec["tokens"]}
    if spec["expected"] is not None:
        checks["exact_answer"] = exact_answer(text, spec["expected"])
    if spec["fresh"]:
        checks["fresh_prefill"] = cached == 0 and read == count
    if spec["target"] is not None:
        checks["target_count"] = abs(count - spec["target"]) <= (256 if spec["target"] == SUSTAINED_TARGET else 16)
    if spec["cached_from"]:
        prefix = max(1, prior_count - 1024) if spec["cached_from"].endswith("-fresh") else 1
        checks["prefix_cache_used"] = type(cached) is int and cached >= prefix
    return checks


def acceptance_request(client, record, save, count_prompt, label, messages):
    spec = next(item for item in acceptance_case_specs(record["acceptance_workload"]) if item["case"] == label)
    tokens, expected, minimum_output = spec["tokens"], spec["expected"], spec["minimum"]
    count = count_prompt(messages)
    context = record["expected_max_context"]
    if type(count) is not int or count <= 0 or count + tokens + 8 > context:
        raise RuntimeError("Acceptance request lacks its requested output room: " + label)
    client.label = label
    response = client.chat(messages, tokens, temperature=0,
                           chat_template_kwargs={"enable_thinking": False})
    rows = record.setdefault("acceptance_cases", [])
    prior_count = next((row["local_prompt_tokens"] for row in rows if row["case"] == spec["cached_from"]), None)
    checks = acceptance_case_checks(spec, count, response, context, prior_count, record["requests"][-1]["metric"])
    rows.append({"case": label, "local_prompt_tokens": count, "requested_max_tokens": tokens,
                 "minimum_completion_tokens": minimum_output, "expected": expected,
                 "checks": checks, "passed": all(checks.values()), "response": response,
                 "metric": record["requests"][-1]["metric"],
                 "messages_sha256": record["requests"][-1]["messages_sha256"]})
    record["acceptance_workload_passed"] = False
    record["acceptance_summary"] = summarize(rows)
    save()
    if not rows[-1]["passed"]:
        raise RuntimeError("Acceptance workload check failed: " + label)
    return response, count


def acceptance_retrieval(validation, client, record, save, count_prompt, target):
    messages, codes, generated_count = validation.long_messages(client.cfg, target)
    count = count_prompt(messages)
    tolerance = 256 if target == SUSTAINED_TARGET else 16
    if (codes != RETRIEVAL_CODES or type(count) is not int or type(generated_count) is not int or count != generated_count
            or abs(count - target) > tolerance):
        raise RuntimeError("Tokenizer/template-aware retrieval did not reach its target.")
    label = "near-full" if target == SUSTAINED_TARGET else "2k"
    response, _ = acceptance_request(client, record, save, count_prompt, label + "-fresh", messages)
    history = copy.deepcopy(messages) + [copy.deepcopy(response["choices"][0]["message"]),
        {"role": "user", "content":
         "Recall the LABEL3 code and calculate 17 times 23. Reply with the code and integer only."}]
    acceptance_request(client, record, save, count_prompt, label + "-cached", history)
    return messages


def acceptance_idle(guard, record, save):
    record["final_idle"] = {"required_seconds": 30, "started_monotonic": time.monotonic(),
                            "completed": False}
    save()
    end = record["final_idle"]["started_monotonic"] + 30
    while time.monotonic() < end:
        guard.check()
        guard.done.wait(min(0.25, max(0, end - time.monotonic())))
    guard.check()
    record["final_idle"]["finished_monotonic"] = time.monotonic()
    record["final_idle"]["completed"] = True
    save()


def acceptance_workload_tests(validation, client, record, save, guard):
    """The original eleven-request sequence or a separate full-context sequence."""
    record["acceptance_workload_passed"] = False
    count_prompt = prompt_counter(client.cfg)
    if record["acceptance_workload"] == "short":
        startup = [{"role": "user", "content": "Calculate 17 times 23. Reply with the integer only."}]
        acceptance_request(client, record, save, count_prompt, "startup-exact", startup)
        response, _ = acceptance_request(client, record, save, count_prompt, "short-exact", startup)
        acceptance_request(client, record, save, count_prompt, "short-cached-followup",
            startup + [response["choices"][0]["message"], {"role": "user", "content":
                "Add 9 to that result. Reply with the integer only."}])
        for index in range(3):
            messages = [{"role": "user", "content":
                f"Case {index + 1}: Write a very long numbered tutorial about database transactions, isolation, "
                "indexes and recovery. Continue for at least 1500 words without concluding; give detailed examples."}]
            response, _ = acceptance_request(client, record, save, count_prompt, f"sustained-{index + 1}",
                                              messages)
            acceptance_request(client, record, save, count_prompt, f"sustained-{index + 1}-followup",
                messages + [response["choices"][0]["message"], {"role": "user", "content":
                    "Calculate 19 times 7. Reply with the integer only."}])
        acceptance_retrieval(validation, client, record, save, count_prompt, 2048)
    else:
        acceptance_retrieval(validation, client, record, save, count_prompt, SUSTAINED_TARGET)
        # A different first system message prevents reuse of the retrieval request. Recount
        # the complete chat after changing the task; never assume the generator's old count.
        messages, _, _ = validation.long_messages(client.cfg, SUSTAINED_TARGET)
        messages = copy.deepcopy(messages)
        messages[0] = {"role": "system", "content":
            "Independent sustained generation test. Treat the reference as background data. "
            "Follow the final writing task, with a detailed answer rather than a list of retrieval codes."}
        messages[-1]["content"] += (
            "\n\nFinal writing task: Write a very long numbered technical tutorial about database transactions, "
            "isolation, indexes and recovery with detailed examples. Continue for at least 1500 words.")
        count = count_prompt(messages)
        if type(count) is not int or abs(count - SUSTAINED_TARGET) > 256:
            raise RuntimeError("Sustained full-chat prompt is outside 128000 +/-256 tokens.")
        acceptance_request(client, record, save, count_prompt, "near-full-sustained", messages)
    acceptance_idle(guard, record, save)
    record["acceptance_workload_passed"] = True
    save()


def validate_selected_workload(record, args, deadline):
    """Independently consume the worker's complete selected evidence, not its aggregate flag."""
    def require(condition, message):
        if not condition:
            raise ValueError("Selected workload evidence: " + message)
    def digest(value):
        return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    def number(value):
        return valid_number(value) and value >= 0
    try:
        selection, context = acceptance_selection(args)
        specs = acceptance_case_specs(selection)
        require(record["acceptance_workload"] == selection and record["cohort"] == args.cohort
                and type(record["expected_max_context"]) is int and record["expected_max_context"] == context,
                "selection/cohort/context differs")
        require(record["acceptance_workload_passed"] is True and record["missing_evidence"] == [], "work remains held")
        cases, requests = record["acceptance_cases"], record["requests"]
        require(type(cases) is list and type(requests) is list and len(cases) == len(specs) == len(requests),
                "case/request sequence is incomplete")
        prior = {}
        last_time = previous_finish = None
        for spec, row, request in zip(specs, cases, requests):
            label = spec["case"]
            require(row["case"] == request["label"] == label, "wrong, duplicated or unordered case")
            require(type(row["requested_max_tokens"]) is int and type(request["max_tokens"]) is int
                    and row["requested_max_tokens"] == request["max_tokens"] == spec["tokens"]
                    and type(row["minimum_completion_tokens"]) is int
                    and row["minimum_completion_tokens"] == spec["minimum"]
                    and row["expected"] == spec["expected"], "case specification differs")
            require(row["response"] == request["response"] and row["metric"] == request["metric"]
                    and not request.get("error"), "request/case payload differs")
            response, metric = row["response"], row["metric"]
            matched_metric(response, {"totals": {"requests": 0}, "requests": []},
                           {"totals": {"requests": 1}, "requests": [metric]})
            require(last_time is None or metric["time"] > last_time, "request association is not ordered/unique")
            last_time = metric["time"]
            previous = prior.get(spec["cached_from"])
            checks = acceptance_case_checks(spec, row["local_prompt_tokens"], response, context,
                                            previous[0]["local_prompt_tokens"] if previous else None, metric)
            require(type(row["checks"]) is dict and all(type(value) is bool for value in row["checks"].values())
                    and row["checks"] == checks and all(checks.values()) and row["passed"] is True,
                    "case checks/data/outcome failed")
            text = response["choices"][0]["message"]["content"]
            require(type(text) is str and len(response["choices"]) == 1
                    and response["choices"][0]["message"]["role"] == "assistant"
                    and response["choices"][0]["finish_reason"] in ("stop", "length"), "malformed returned assistant message")
            require(digest(request["output_sha256"])
                    and request["output_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest(), "output digest differs")
            require(digest(request["messages_sha256"]) and row["messages_sha256"] == request["messages_sha256"],
                    "message identity differs")
            first, wall = request["first_content_seconds"], request["client_wall_seconds"]
            start, finish = request["started_monotonic"], request["finished_monotonic"]
            require(all(number(value) for value in (first, wall, start, finish, request["request_wall_seconds"]))
                    and number(response["first_token_s"]) and number(response["wall_seconds"])
                    and number(request["started_unix"]) and number(request["finished_unix"])
                    and request["started_unix"] <= request["finished_unix"]
                    and wall > 0 and first <= wall + 0.001 and start <= finish <= deadline
                    and (previous_finish is None or start >= previous_finish)
                    and abs(request["request_wall_seconds"] - (finish - start)) <= 0.000001
                    and wall <= finish - start + 0.001
                    and response["first_token_s"] == first and response["wall_seconds"] == wall,
                    "request clocks are incomplete/inconsistent")
            previous_finish = finish
            require(type(request["actual_completion_tokens"]) is int
                    and request["actual_completion_tokens"] == response["usage"]["completion_tokens"]
                    and type(request["cached_tokens"]) is int and request["cached_tokens"] == response["timings"]["cache_n"]
                    and metric["reused"] == request["cached_tokens"]
                    and number(metric["prompt_ms"]) and abs(metric["prompt_ms"] - response["timings"]["prompt_ms"]) <= 0.11,
                    "request counts/timing differ")
            messages = request["message_digests"]
            require(type(messages) is list and len(messages) >= 1, "message evidence missing")
            for message in messages:
                require(message["role"] in ("system", "user", "assistant") and type(message["characters"]) is int
                        and message["characters"] >= 0 and digest(message["content_sha256"])
                        and digest(message["message_sha256"]), "malformed message identity")
            if previous:
                old_row, old_request = previous
                assistant = old_row["response"]["choices"][0]["message"]
                require(len(messages) == len(old_request["message_digests"]) + 2
                        and messages[:-2] == old_request["message_digests"]
                        and messages[-2] == message_identity(assistant)
                        and messages[-1]["role"] == "user", "returned history was not retained")
            if label == "near-full-sustained":
                require(messages[0] != prior["near-full-fresh"][1]["message_digests"][0], "sustained prompt was not separate")
            prior[label] = row, request
        summary = summarize(cases)
        require(record["acceptance_summary"] == summary and all(
            type(record["acceptance_summary"][key]) is type(value) for key, value in summary.items()),
            "aggregate summary differs from actual cases")
        require(type(record["metrics"]["totals"]["requests"]) is int
                and record["metrics"]["totals"]["requests"] == len(requests), "final request total differs")
        idle = record["final_idle"]
        require(type(idle["required_seconds"]) is int and idle["required_seconds"] == 30 and idle["completed"] is True
                and number(idle["started_monotonic"]) and number(idle["finished_monotonic"])
                and requests[-1]["finished_monotonic"] <= idle["started_monotonic"]
                and idle["finished_monotonic"] - idle["started_monotonic"] >= 30
                and idle["finished_monotonic"] <= deadline, "final idle failed")
        protection = record["protection"]
        require(protection["unchanged"] is True and type(protection["before"]) is list
                and len(protection["before"]) > 0 and protection["before"] == protection["after"], "protection differs")
        paths = []
        for item in protection["before"]:
            require(type(item["path"]) is str and Path(item["path"]).is_absolute() and type(item["bytes"]) is int
                    and item["bytes"] >= 0 and digest(item["sha256"]), "protection identity malformed")
            paths.append(item["path"])
        selected_paths = {args.config.resolve(), Path(record["exe"]).resolve()}
        selected_paths.update(Path(path).resolve() for path in getattr(args, "protect_input", []))
        require(len(paths) == len(set(map(Path, paths))) and selected_paths <= set(map(Path, paths)),
                "protected inputs are incomplete/duplicated")
        require(record["config"] == str(args.config.resolve()) and digest(record["exe_sha256"])
                and next(item["sha256"] for item in protection["before"] if item["path"] == record["exe"]) == record["exe_sha256"],
                "consumed executable/config identity differs")
        return True
    except (KeyError, TypeError, ValueError, IndexError, StopIteration, AttributeError, RuntimeError, OverflowError) as exc:
        raise ValueError("Selected workload evidence refused: " + str(exc)) from None


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [redact(x, secret) for x in value]
    if isinstance(value, dict):
        return {redact(k, secret): redact(v, secret) for k, v in value.items()}
    return value


def is_strata_process(info):
    name = (info.get("name") or "").lower()
    if name in ("strata", "strata.exe") or name.startswith("strata-"):
        return True
    command = [str(x).replace("\\", "/").lower() for x in info.get("cmdline") or []]
    server = any(x.endswith("/serve/server.py") or x == "serve/server.py" for x in command)
    module = any(command[i:i + 2] == ["-m", "serve.server"] for i in range(len(command)))
    return server or module


def assert_idle(port=PORT):
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.info["pid"] != os.getpid() and is_strata_process(process.info):
            raise RuntimeError("A Strata engine or server is already running; close it before benchmarking.")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError(f"Port {port} is occupied; no process was started or stopped.") from None


class OwnedProcess:
    """One cancellation owner; opt-in authority is the retained creation handle and Job."""

    def __init__(self, proc, *, native=None, deadline=None):
        self.proc = proc
        self.native, self.deadline = native, deadline
        self.lock = threading.RLock()
        self.errors = []
        self.job = None
        self.identity = None
        self.assigned = self.closed = self.forced = False
        self.exit_code = None
        self.final_accounting = {}
        self.stopped = False
        if native is not None:
            # Retain Popen itself: its creation handle must outlive every wait and query.
            self.handle = getattr(proc, "_handle", None)
            return
        self.root = psutil.Process(proc.pid)
        self.members = {self.root}

    def contain(self):
        with self.lock:
            if not self.handle:
                raise RuntimeError("Popen creation handle is missing.")
            self.identity = native_identity(self.native.identity(self.handle))
            if self.identity[0] != self.proc.pid:
                raise RuntimeError("Creation handle PID differs from Popen PID.")
            self.job = self.native.create_job()
            self.native.configure_job(self.job)
            self.native.assign(self.job, self.handle)
            self.assigned = True

    def discover(self):
        if self.native is not None:
            with self.lock:
                if self.closed:
                    raise RuntimeError("Discovery after handle closure.")
                if native_identity(self.native.identity(self.handle)) != self.identity:
                    raise RuntimeError("Retained process identity changed.")
                return self.native.active(self.job)
        try:
            if self.root.is_running():
                self.members.update(self.root.children(recursive=True))
        except psutil.NoSuchProcess:
            pass

    def stop(self):
        if self.native is not None:
            with self.lock:
                if not self.closed and not self.stopped:
                    self.stopped = self.forced = True
                    # A failed wait must never skip cancellation of our exact handles.
                    try:
                        if self.assigned:
                            self.native.terminate_job(self.job)
                        elif self.handle:
                            self.native.terminate_process(self.handle)
                        else:
                            raise RuntimeError("Cannot cancel without creation handle.")
                    except Exception as exc:
                        self.errors.append("cancel: " + str(exc))
                if self.errors:
                    raise RuntimeError("; ".join(self.errors))
                return
        with self.lock:
            if self.stopped:
                return
            self.stopped = True
            self.discover()
            # psutil's terminate/kill methods check PID reuse against creation time.
            members = list(self.members - {self.root}) + [self.root]
            for process in members:
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(members, timeout=3)
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=3)
            if alive:
                raise RuntimeError("An owned benchmark process did not exit after cleanup.")

    def poll(self):
        with self.lock:
            if self.closed:
                return self.exit_code
            try:
                result = self.native.wait(self.handle, 0)
                if result == 258:  # WAIT_TIMEOUT is ordinary only during nonblocking polling.
                    return None
                if result != 0:
                    raise RuntimeError("Process wait failed: " + str(result))
                self.exit_code = self.native.exit_code(self.handle)
                if type(self.exit_code) is not int or not 0 <= self.exit_code <= 2**32 - 1:
                    raise RuntimeError("Invalid native exit code.")
                self.proc.returncode = self.exit_code
                return self.exit_code
            except Exception as exc:
                self.errors.append("wait: " + str(exc))
                raise

    def finish(self):
        """Account before closing; failures retain authority on the raised exception's owner."""
        with self.lock:
            if self.closed:
                if self.errors:
                    raise RuntimeError("; ".join(self.errors))
                return
            wait_ok = identity_ok = job_ok = False
            self.final_accounting = {}

            def failed(label, exc):
                self.final_accounting[label + "_error"] = str(exc)
                self.errors.append("account: " + str(exc))
                try:
                    self.stop()
                except Exception:
                    pass  # stop retains its own error; independent accounting must continue.

            try:
                remaining = max(0, self.deadline - time.monotonic())
                result = self.native.wait(self.handle, int(remaining * 1000))
                self.final_accounting["wait_result"] = result
                if result != 0:
                    raise RuntimeError("Final owned wait was not signaled: " + str(result))
                self.exit_code = self.native.exit_code(self.handle)
                if type(self.exit_code) is not int or not 0 <= self.exit_code <= 2**32 - 1:
                    raise RuntimeError("Invalid final native exit code.")
                self.proc.returncode = self.exit_code
                self.final_accounting["exit_code"] = self.exit_code
                wait_ok = True
            except Exception as exc:
                failed("wait", exc)
            # These probes are independent of wait success, including after cancellation.
            # They add evidence; neither may turn an uncertain wait into permission to close.
            try:
                observed = native_identity(self.native.identity(self.handle))
                self.final_accounting["identity"] = dict(zip(("pid", "creation_filetime"), map(str, observed)))
                if self.identity is None or observed != self.identity:
                    raise RuntimeError("Final retained identity changed.")
                identity_ok = True
            except Exception as exc:
                failed("identity", exc)
            try:
                if self.job is None:
                    self.final_accounting["job"] = "not created"
                    if self.assigned:
                        raise RuntimeError("Assigned Job handle is missing.")
                else:
                    active = self.native.active(self.job)
                    self.final_accounting["job_active"] = active
                    if type(active) is not int or not 0 <= active <= 2**32 - 1:
                        raise RuntimeError("Unknown Job membership.")
                    if active:
                        raise RuntimeError("Live Job members remain.")
                job_ok = True
            except Exception as exc:
                failed("job", exc)
            if not (wait_ok and identity_ok and job_ok):
                raise RuntimeError("; ".join(self.errors))
            # No wait is pending while handles are closed, including monitor cancellation.
            try:
                if self.job is not None:
                    self.native.close(self.job)
                    self.job = None
                self.native.close_process(self.proc)
                self.closed = True
            except Exception as exc:
                self.errors.append("close: " + str(exc))
            if self.errors:
                raise RuntimeError("; ".join(self.errors))


def native_identity(value):
    if (type(value) is not tuple or len(value) != 2
            or any(type(number) is not int or not 0 < number <= maximum
                   for number, maximum in zip(value, (2**32 - 1, 2**64 - 1)))):
        raise ValueError("Native PID and FILETIME must be exact unsigned integers.")
    return value


class WindowsProcessAPI:
    """Lazy Win32 boundary; CPU composition replaces this boundary before construction.

    ABI: Microsoft GetProcessTimes, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION and WaitForSingleObject contracts.
    Windows LLP64: DWORD=32, LARGE_INTEGER=64, SIZE_T/ULONG_PTR=pointer width.
    No PID lookup, breakaway, process-name exception or epoch conversion is used.
    """

    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("Contained acceptance requires Windows.")
        import ctypes as c
        self.c = c
        self.dll = c.WinDLL("kernel32", use_last_error=True)
        u32, u64, ptr = c.c_uint32, c.c_uint64, c.c_void_p

        class FileTime(c.Structure):
            _fields_ = [("low", u32), ("high", u32)]

        class BasicLimit(c.Structure):
            _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64), ("flags", u32),
                        ("min_ws", c.c_size_t), ("max_ws", c.c_size_t), ("active_limit", u32),
                        ("affinity", c.c_size_t), ("priority", u32), ("scheduling", u32)]

        class ExtendedLimit(c.Structure):
            _fields_ = [("basic", BasicLimit), ("io", u64 * 6),
                        ("process_memory", c.c_size_t), ("job_memory", c.c_size_t),
                        ("peak_process", c.c_size_t), ("peak_job", c.c_size_t)]

        class Accounting(c.Structure):
            _fields_ = [("times", c.c_int64 * 4), ("faults", u32), ("total", u32),
                        ("active", u32), ("terminated", u32)]

        self.FileTime, self.ExtendedLimit, self.Accounting = FileTime, ExtendedLimit, Accounting
        signatures = {
            "GetProcessId": ([ptr], u32),
            "GetProcessTimes": ([ptr] + [c.POINTER(FileTime)] * 4, c.c_int32),
            "CreateJobObjectW": ([ptr, c.c_wchar_p], ptr),
            "SetInformationJobObject": ([ptr, c.c_int32, ptr, u32], c.c_int32),
            "AssignProcessToJobObject": ([ptr, ptr], c.c_int32),
            "QueryInformationJobObject": ([ptr, c.c_int32, ptr, u32, c.POINTER(u32)], c.c_int32),
            "TerminateJobObject": ([ptr, u32], c.c_int32),
            "TerminateProcess": ([ptr, u32], c.c_int32),
            "WaitForSingleObject": ([ptr, u32], u32),
            "GetExitCodeProcess": ([ptr, c.POINTER(u32)], c.c_int32),
            "CloseHandle": ([ptr], c.c_int32),
            "PeekNamedPipe": ([ptr, ptr, u32, c.POINTER(u32), c.POINTER(u32), c.POINTER(u32)], c.c_int32),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.dll, name)
            function.argtypes, function.restype = arguments, result

    def checked(self, value):
        if not value:
            raise self.c.WinError(self.c.get_last_error())
        return value

    def identity(self, handle):
        times = [self.FileTime() for _ in range(4)]
        self.checked(self.dll.GetProcessTimes(int(handle), *(self.c.byref(item) for item in times)))
        return (int(self.checked(self.dll.GetProcessId(int(handle)))),
                (int(times[0].high) << 32) | int(times[0].low))

    def create_job(self):
        return self.checked(self.dll.CreateJobObjectW(None, None))

    def configure_job(self, job):
        limits = self.ExtendedLimit()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        self.checked(self.dll.SetInformationJobObject(job, 9, self.c.byref(limits), self.c.sizeof(limits)))

    def assign(self, job, handle):
        self.checked(self.dll.AssignProcessToJobObject(job, int(handle)))

    def active(self, job):
        accounting = self.Accounting()
        self.checked(self.dll.QueryInformationJobObject(job, 1, self.c.byref(accounting),
                                                       self.c.sizeof(accounting), None))
        return int(accounting.active)

    def terminate_job(self, job):
        self.checked(self.dll.TerminateJobObject(job, 1))

    def terminate_process(self, handle):
        self.checked(self.dll.TerminateProcess(int(handle), 1))

    def wait(self, handle, milliseconds):
        return int(self.dll.WaitForSingleObject(int(handle), milliseconds))

    def exit_code(self, handle):
        code = self.c.c_uint32()
        self.checked(self.dll.GetExitCodeProcess(int(handle), self.c.byref(code)))
        return int(code.value)

    def close(self, handle):
        self.checked(self.dll.CloseHandle(int(handle)))

    def close_process(self, proc):
        # CPython Handle.Close marks itself closed BEFORE CloseHandle can fail. Keep
        # ownership on failure, and detach its destructor only after checked success.
        self.close(proc._handle)
        proc._handle.Detach()

    def capture(self, pipe):
        import msvcrt
        available = self.c.c_uint32()
        success = self.dll.PeekNamedPipe(msvcrt.get_osfhandle(pipe.fileno()), None, 0, None,
                                        self.c.byref(available), None)
        if not success:
            if self.c.get_last_error() == 109:  # ERROR_BROKEN_PIPE, all inherited writers closed.
                return b"", True
            self.checked(success)
        return (os.read(pipe.fileno(), min(8192, available.value)) if available.value else b"", False)


class MemoryGuard:
    def __init__(self, owner, engine_log, *, acceptance_floors=None, deadline=None, sample_limit=4096):
        self.owner, self.engine_log = owner, engine_log
        self.deadline = deadline
        self.sample_limit = sample_limit
        self.done = threading.Event()
        self.reason = None
        self.cleanup_error = None
        self.low_samples = 0
        self.samples = 0
        self.minimum = None
        self.max_pagefile = None
        self.pagefile_unavailable_reason = None
        self.acceptance_floors = acceptance_floors
        if acceptance_floors is not None:
            if len(acceptance_floors) != 2:
                raise ValueError("Acceptance requires physical and commit floors.")
            for value in acceptance_floors:
                if not valid_number(value) or value <= 0 or not valid_number(value * 1024**3):
                    raise ValueError("Acceptance floors must be finite positive GiB values.")
        self.host_samples = []
        self.max_gap_seconds = 0
        self.thread = threading.Thread(target=self.monitor, daemon=True)

    def acceptance_sample(self, snapshot, started, finished):
        """Record unadjusted host availability; pagefile use is never commit availability."""
        if self.reason is not None:
            return
        try:
            if not all(valid_number(value) for value in (started, finished)) or finished < started:
                raise ValueError("Invalid monotonic sample interval.")
            values = [snapshot[key] for key in ("available_gib", "commit_available_gib")]
            if not all(valid_number(value) and value >= 0 and valid_number(value * 1024**3)
                       for value in values):
                raise ValueError("Physical and available-commit telemetry are required.")
            physical, commit = [value * 1024**3 for value in values]
            if not all(value.is_integer() if isinstance(value, float) else True
                       for value in (physical, commit)):
                raise ValueError("Host telemetry must represent whole bytes.")
            gap = finished - self.host_samples[-1]["finished_monotonic"] if self.host_samples else 0
            if self.host_samples and (started < self.host_samples[-1]["finished_monotonic"] or gap <= 0):
                raise ValueError("Host sample clock did not advance monotonically.")
            if len(self.host_samples) >= self.sample_limit:
                raise ValueError("Acceptance raw sample limit reached; evidence is incomplete.")
            self.host_samples.append({"started_monotonic": started, "finished_monotonic": finished,
                                      "physical_available_bytes": int(physical),
                                      "commit_available_bytes": int(commit)})
            self.samples += 1
            self.max_gap_seconds = max(self.max_gap_seconds, gap, finished - started)
            self.low_samples = self.low_samples + 1 if physical < 2.5 * 1024**3 or commit < 4 * 1024**3 else 0
            if physical < 2 * 1024**3 or commit < 2 * 1024**3:
                raise ValueError("Physical or available commit fell below the original 2 GiB emergency floor.")
            if self.low_samples >= 5:
                raise ValueError("Physical below 2.5 GiB or available commit below 4 GiB for five consecutive samples.")
            if self.max_gap_seconds > 1:
                raise ValueError("Host sampling gap or read duration exceeded one second.")
        except (KeyError, TypeError, ValueError) as exc:
            if self.reason is None:
                self.reason = "Acceptance telemetry failed: " + str(exc)
                if self.owner is not None:
                    self.stop_owner()

    def stop_owner(self):
        """Keep the first guard reason even if its cancellation also fails."""
        try:
            if self.owner is not None:
                self.owner.stop()
        except Exception as exc:
            if self.acceptance_floors is None:
                raise
            self.cleanup_error = type(exc).__name__ + ": " + str(exc)

    def acceptance_summary(self, start=None, end=None):
        rows = list(self.host_samples)
        errors = [self.reason] if self.reason else []
        if (not all(valid_number(value) for value in (start, end)) or end <= start
                or len(rows) < 2 or rows[0]["finished_monotonic"] > start
                or rows[-1]["started_monotonic"] < end):
            errors.append("Required monotonic coverage edges are missing or insufficient.")
        minima = {}
        for name, floor in zip(("physical", "commit"), self.acceptance_floors):
            row = min(rows, key=lambda item: item[name + "_available_bytes"]) if rows else None
            minima[name] = ({"available_bytes": row[name + "_available_bytes"],
                             "started_monotonic": row["started_monotonic"],
                             "finished_monotonic": row["finished_monotonic"]} if row else None)
            if row is None or row[name + "_available_bytes"] < floor * 1024**3:
                errors.append(name + " configured headroom evidence failed.")
        return {"host_evidence": "refused" if errors else "satisfied",
                "errors": errors, "floors_gib": dict(zip(("physical", "commit"), self.acceptance_floors)),
                "raw_samples": rows, "minima": minima, "max_gap_seconds": self.max_gap_seconds,
                "coverage_start_monotonic": start, "coverage_end_monotonic": end,
                "sample_limit": self.sample_limit, "cuda_stage_evidence": "missing; later-stage hold",
                "cleanup_error": self.cleanup_error,
                "cuda_floor_mib": 256, "cuda_coverage": "stage evidence required; not collected",
                "lifecycle_acceptance": "pending", "cleanup": "pending", "finalization": "pending"}

    def sample(self, available, pagefile):
        self.samples += 1
        self.minimum = available if self.minimum is None else min(self.minimum, available)
        if pagefile is not None and self.pagefile_unavailable_reason is None:
            self.max_pagefile = pagefile if self.max_pagefile is None else max(self.max_pagefile, pagefile)
        self.low_samples = self.low_samples + 1 if available < 2 * 1024**3 else 0
        if self.reason is None and (available < 1024**3 or self.low_samples >= 5):
            self.reason = ("Available RAM fell below the 1 GiB emergency floor." if available < 1024**3
                           else "Available RAM stayed below 2 GiB for five consecutive samples.")
            self.owner.stop()

    def monitor(self, *, once=False):
        try:
            while not self.done.is_set():
                if self.owner is not None:
                    self.owner.discover()
                if self.acceptance_floors is not None:
                    from tools.prepare_unsloth_q2 import memory_snapshot
                    started = time.monotonic()
                    snapshot = memory_snapshot()
                    self.acceptance_sample(snapshot, started, time.monotonic())
                    if self.reason:
                        return
                    if self.engine_log is not None and self.engine_log.exists() and self.engine_log.stat().st_size > 2 * LOG_LIMIT:
                        raise RuntimeError("Engine log exceeded the diagnostic size limit.")
                    if once:
                        return
                    self.done.wait(0.25)
                    continue
                available = psutil.virtual_memory().available  # Required for the cancellation guard.
                pagefile = None
                if self.pagefile_unavailable_reason is None:
                    try:
                        pagefile = psutil.swap_memory().used
                    except (OSError, RuntimeError, psutil.Error) as exc:
                        # Windows performance counters can be disabled even when physical RAM is readable.
                        self.pagefile_unavailable_reason = type(exc).__name__ + ": " + str(exc)
                        self.max_pagefile = None
                self.sample(available, pagefile)
                if self.reason:
                    return
                if self.engine_log.exists() and self.engine_log.stat().st_size > 2 * LOG_LIMIT:
                    self.reason = "Engine log exceeded the diagnostic size limit."
                    self.owner.stop()
                    return
                self.done.wait(1)
        except Exception as exc:
            if self.acceptance_floors is None:
                self.reason = "Memory/process monitor failed: " + type(exc).__name__
                self.owner.stop()
            else:
                if self.reason is None:
                    self.reason = "Memory/process monitor failed: " + type(exc).__name__
                self.stop_owner()

    def check(self):
        if self.reason:
            raise RuntimeError(self.reason)
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("Absolute worker deadline reached.")

    def summary(self):
        if self.acceptance_floors is not None:
            return self.acceptance_summary()
        return {"samples": self.samples, "min_available_gib": self.minimum / 1024**3
                if self.minimum is not None else None, "max_pagefile_used_gib": self.max_pagefile / 1024**3
                if self.max_pagefile is not None else None,
                "pagefile_available": self.max_pagefile is not None,
                "pagefile_unavailable_reason": self.pagefile_unavailable_reason or
                ("No pagefile sample was collected." if self.max_pagefile is None else None),
                "stop_reason": self.reason}


def valid_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


class BoundedCapture:
    """Keep a bounded head and tail while always draining the subprocess pipe."""

    def __init__(self, limit=LOG_LIMIT):
        self.head, self.tail = bytearray(), bytearray()
        self.total = 0
        self.limit = limit
        self.complete, self.error = False, None

    def write(self, value):
        data = value.encode("utf-8") if isinstance(value, str) else value
        self.total += len(data)
        head_size = min(len(data), self.limit // 2 - len(self.head))
        self.head.extend(data[:head_size])
        self.tail.extend(data[head_size:])
        if len(self.tail) > self.limit // 2:
            del self.tail[:-self.limit // 2]
        return len(value)

    def flush(self):
        # Writes already update the in-memory capture; print(..., flush=True) needs this method.
        pass

    def drain(self, pipe):
        try:
            try:
                while chunk := pipe.read(8192):
                    self.write(chunk)
            finally:
                pipe.close()
            self.complete = True
        except BaseException as exc:
            self.error = type(exc).__name__ + ": " + str(exc)
            raise

    def text(self):
        gap = b"\n[log middle omitted]\n" if self.total > self.limit else b""
        return (self.head + gap + self.tail).decode("utf-8", errors="replace")


def bounded_file(path):
    capture = BoundedCapture()
    if path.exists():
        with path.open("rb") as source:
            while chunk := source.read(65536):
                capture.write(chunk)
    return capture.text()


def replace_result(pending, destination):
    # Windows readers/indexers can briefly deny replacement while holding the old file open.
    # Keep the previous snapshot intact, and still fail on a persistent permission problem.
    for attempt in range(6):
        try:
            pending.replace(destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def matched_metric(response, before, after, *, record=None):
    try:
        totals = [item["totals"]["requests"] for item in (before, after)]
        if any(type(value) is not int or value < 0 for value in totals):
            raise ValueError("Invalid request totals.")
        rows = [row for item in (before, after) for row in item.get("requests", [])]
        if any(not valid_number(row["time"]) or row["time"] < 0 for row in rows):
            raise ValueError("Invalid request association time.")
        old = {row["time"] for row in before.get("requests", [])}
        new = [row for row in after.get("requests", []) if row["time"] not in old]
        if totals[1] - totals[0] != 1 or len(new) != 1:
            raise RuntimeError("Could not uniquely match this response to one new request metric.")
        metric = new[0]
        if record is not None:
            record["metric"] = metric  # Retain uniquely associated invalid evidence on controlled refusal.
        usage, timings = response["usage"], response["timings"]
        for owner, keys in ((usage, ("prompt_tokens", "completion_tokens")),
                            (metric, ("prompt_tokens", "output_tokens"))):
            if any(type(owner[key]) is not int or owner[key] < 0 for key in keys):
                raise ValueError("Invalid request token count.")
        for owner, key in ((metric, "decode_ms"), (timings, "predicted_ms")):
            if not valid_number(owner[key]) or owner[key] < 0:
                raise ValueError("Invalid request timing.")
        for owner, keys, integer in ((metric, ("prompt_ms",), False),
                (timings, ("prompt_ms",), False),
                (metric, ("reused", "engine_generated", "prompt_read", "prompt_total"), True),
                (timings, ("cache_n", "prompt_n", "predicted_n"), True)):
            for key in keys:
                if owner is metric and key in ("reused", "engine_generated", "prompt_read", "prompt_total") \
                        and owner.get(key) is None:
                    continue  # Older engines explicitly report unavailable native counters as null.
                if key in owner and (not valid_number(owner[key]) or owner[key] < 0
                                     or (integer and type(owner[key]) is not int)):
                    raise ValueError("Invalid request count/timing: " + key)
        pairs = [(metric["output_tokens"], usage["completion_tokens"], 0),
                 (metric["prompt_tokens"], usage["prompt_tokens"], 0),
                 (metric["decode_ms"], timings["predicted_ms"], 0.11)]
        for left, right in (("prompt_ms", "prompt_ms"), ("reused", "cache_n")):
            if metric.get(left) is not None and timings.get(right) is not None:
                pairs.append((metric[left], timings[right], 0.11 if left == "prompt_ms" else 0))
        if "predicted_n" in timings:
            pairs.append((timings["predicted_n"], usage["completion_tokens"], 0))
        total = metric.get("prompt_total")
        if total is not None and total < metric["prompt_tokens"]:
            raise ValueError("Native whole-prompt count is smaller than the request prompt.")
        if metric.get("prompt_read") is not None and metric["prompt_read"] > (total if total is not None else usage["prompt_tokens"]):
            raise ValueError("Native prompt-read count exceeds the whole prompt.")
        # Service can inject reasoning wrap-up tokens that are delivered but were
        # not generated by the engine. Only the thinking-disabled selected cases
        # impose the stronger native/delivered count relation.
        prompt = usage["prompt_tokens"]
        for owner, key in ((metric, "reused"), (metric, "prompt_read"),
                           (timings, "cache_n"), (timings, "prompt_n")):
            if owner.get(key) is not None and owner[key] > prompt:
                raise ValueError("Known cache/read count exceeds the observed prompt.")
        for owner, cached, read in ((metric, "reused", "prompt_read"), (timings, "cache_n", "prompt_n")):
            if owner.get(cached) is not None and owner.get(read) is not None and owner[cached] + owner[read] != prompt:
                raise ValueError("Known cache/read counts do not sum to the observed prompt.")
        if metric.get("prompt_read") is not None and timings.get("prompt_n") is not None:
            pairs.append((metric["prompt_read"], timings["prompt_n"], 0))
        if any(abs(left - right) > tolerance for left, right, tolerance in pairs):
            raise RuntimeError("New request metric does not agree with the response usage/timing.")
        return metric
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Invalid request metric evidence: " + str(exc)) from None


class TimeoutOpener:
    def __init__(self, deadline=None):
        self.deadline = deadline
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def open(self, request, timeout=1800):
        if self.deadline is not None:
            timeout = min(timeout, self.deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("Absolute HTTP deadline reached.")
        return self.opener.open(request, timeout=min(timeout, 1800))


def make_client(validation, cfg, record, save, guard, secret):
    class Client(validation.Client):
        label = "warmup"
        deadline = guard.deadline

        def get(self, path):
            guard.check()
            return super().get(path)

        def chat(self, messages, max_tokens=128, **kwargs):
            guard.check()
            measured = bool(record.get("acceptance_workload"))
            if measured:
                kwargs["stream"] = True  # The existing client clocks the first actual content delta.
            started = time.monotonic() if measured else None
            item = {"label": self.label, "max_tokens": max_tokens,
                    "started_unix": time.time()}
            if measured:
                # Preserve request identity without copying each near-full prompt into
                # every snapshot. The HTTP request still receives the complete history.
                item["messages_sha256"] = json_digest(messages)
                item["started_monotonic"] = started
                item["message_digests"] = [message_identity(message) for message in messages]
            else:
                item["messages"] = copy.deepcopy(messages)
            record["requests"].append(item)
            save()
            try:
                before = self.get("/metrics")
                raw_response = super().chat(messages, max_tokens, **kwargs)
                if measured:
                    text = raw_response["choices"][0]["message"].get("content") or ""
                    item["output_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
                response = redact(raw_response, secret)
                item["response"] = response
                if measured:
                    item["first_content_seconds"] = response.get("first_token_s")
                    item["client_wall_seconds"] = response.get("wall_seconds")
                save()  # Keep the answer even if the subsequent metric lookup fails.
                item["metric"] = matched_metric(response, before, self.get("/metrics"), record=item)
                if measured:
                    first, wall = item["first_content_seconds"], item["client_wall_seconds"]
                    timings, usage, metric = response["timings"], response["usage"], item["metric"]
                    if (not all(valid_number(value) for value in (first, wall))
                            or not 0 <= first <= wall + 0.001 or wall <= 0
                            or not all(valid_number(value) and value >= 0 for value in (
                                timings.get("prompt_ms"), timings.get("predicted_ms"), metric.get("prompt_ms")))
                            or abs(metric["prompt_ms"] - timings["prompt_ms"]) > 0.11
                            or type(timings.get("cache_n")) is not int
                            or type(metric.get("reused")) is not int
                            or metric["reused"] != timings["cache_n"]
                            or type(usage["completion_tokens"]) is not int):
                        raise RuntimeError("Missing or inconsistent request-associated streaming evidence.")
                    item["cached_tokens"] = timings["cache_n"]
                    item["actual_completion_tokens"] = usage["completion_tokens"]
                guard.check()
                return response
            except Exception as exc:
                item["error"] = str(exc)
                raise
            finally:
                item["finished_unix"] = time.time()
                if measured:
                    item["finished_monotonic"] = time.monotonic()
                    item["request_wall_seconds"] = item["finished_monotonic"] - started
                save()

    return Client(cfg)


def acceptance_ready():
    # Workload CPU coverage does not supply native, current admission or CUDA proof.
    return False


def acceptance_preflight(args):
    raise RuntimeError("Stage C current admission and protection binding are not implemented.")


def acceptance_protection(args):
    # This independent external protection check must eventually compare admitted protected bytes.
    return False


def direct_interpreter():
    base = Path(sys.base_prefix).resolve()
    interpreter = Path(sys._base_executable).resolve()
    if (interpreter != base / "python.exe" or not interpreter.is_file()
            or (base / "pyvenv.cfg").exists()):
        raise RuntimeError("A verified direct base interpreter is required; venv launchers are refused.")
    return interpreter


def bounded_write(path, data, deadline, limit=ACCEPTANCE_LOG_LIMIT):
    """Write only provisional bytes. Check short writes and close before returning."""
    if time.monotonic() >= deadline or len(data) > limit:
        raise RuntimeError("Artifact exceeds its size or absolute time budget.")
    with path.open("wb") as stream:
        if stream.write(data) != len(data):
            raise OSError("Short artifact write.")
        stream.flush()
    if time.monotonic() >= deadline:
        raise TimeoutError("Artifact finalization exceeded the absolute deadline.")


def acceptance_worker(envelope):
    args = argparse.Namespace(**envelope["args"])
    args.config, args.output = Path(args.config), Path(args.output)
    args.acceptance_telemetry = False
    args.acceptance_physical_gib = args.acceptance_commit_gib = None
    args._acceptance_deadline = envelope["work_deadline"]
    if not valid_number(args._acceptance_deadline) or time.monotonic() >= args._acceptance_deadline:
        raise TimeoutError("START arrived after the worker deadline.")
    # The existing smoke helper owns an unrestricted file writer; retain its hold.
    if args.smoke or getattr(args, "smoke_only", False):
        raise RuntimeError("Contained smoke artifacts require the Stage C bounded writer.")
    try:
        acceptance_preflight(args)  # Heavy admission belongs INSIDE the supervised child.
        _run_workload(args)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        # The external capture does not read private config to learn its secret.
        # Preserve redacted workload details in provisional artifacts, never a traceback.
        print("Contained worker stopped (" + type(exc).__name__ + ").", file=sys.stderr)
        return 1


class AcceptanceFailure(RuntimeError):
    def __init__(self, record, owner):
        super().__init__(record.get("error") or "Contained lifecycle evidence refused.")
        self.record, self.owner = record, owner


def acceptance_final_predicate(record):
    return (record.get("worker_exit") == 0 and record.get("natural_exit") is True
            and record.get("capture_complete") is True and record.get("owned_cleanup") is True
            and record.get("protection") is True and record.get("finalization") is True
            and record.get("within_deadline") is True and record.get("worker_complete") is True
            and record.get("base_host_floors_satisfied") is True
            and record.get("memory", {}).get("host_evidence") == "satisfied"
            and not record.get("error") and not record.get("cleanup_error"))


def acceptance_controller(args, started):
    """Supervise the complete child, including admission and artifact close, from outside it.

    The caller must keep this controller alive. OS scheduling, filesystem and native API
    calls are assumed to return; an arbitrary blocked controller syscall cannot be
    preempted by its own timer. Job containment protects descendants, not that assumption.
    Discovery is periodic accounting, never a proof of each process's birth.
    """
    total_seconds = getattr(args, "acceptance_total_seconds", None)
    total_seconds = ACCEPTANCE_SECONDS if total_seconds is None else total_seconds
    total_deadline = started + total_seconds
    work_deadline = total_deadline - CLEANUP_SECONDS
    floors = tuple(4 if value is None else value for value in (
        getattr(args, "acceptance_physical_gib", None), getattr(args, "acceptance_commit_gib", None)))
    guard = MemoryGuard(None, None, acceptance_floors=floors, sample_limit=4 * total_seconds + 256)
    if not acceptance_ready():
        return _run_workload(args)  # Existing held receipt, with no native/resource entry.
    if args.output.exists():
        raise RuntimeError("Output directory already exists; choose a new directory.")
    native, owner, proc = None, None, None
    capture = BoundedCapture(ACCEPTANCE_LOG_LIMIT - 64)
    worker_environment = dict(os.environ)
    worker_environment.pop("STRATA_BENCHMARK_BOOTSTRAP", None)  # Consumed by bootstrap, never passed to the engine.
    record = {"status": "held", "lifecycle_evidence": "refused", "started_monotonic": started,
              "total_deadline": total_deadline, "work_deadline": work_deadline,
              "pending_stages": ["Stage C", "CUDA", "current admission"],
              "external_limits": "Controller OS calls must return; caller must keep controller alive.",
              "normal_comparison": normal_comparison_excluded("worker-classification-unavailable")}
    coverage_start = coverage_end = None
    eof = False

    def cleanup_failure(error):
        previous = record.get("cleanup_error")
        record["cleanup_error"] = (previous + "; " if previous else "") + str(error)

    try:
        # Host coverage starts before any owned process/native startup work.
        guard.monitor(once=True)
        guard.check()
        coverage_start = time.monotonic()
        interpreter = direct_interpreter()
        native = WindowsProcessAPI()
        settings = json.dumps({"work_deadline": work_deadline, "python_path": [str(ROOT), *sys.path],
                               "args": {key: str(value) if isinstance(value, Path) else value
                                        for key, value in vars(args).items()}})
        if len(settings) > 16384:
            raise RuntimeError("Bootstrap settings exceed the environment budget.")
        environment = dict(os.environ, STRATA_BENCHMARK_BOOTSTRAP=settings)
        if time.monotonic() >= work_deadline:
            raise TimeoutError("Deadline expired before creation.")
        proc = subprocess.Popen([str(interpreter), "-I", "-S", str(Path(__file__).resolve()),
                                 "--internal-acceptance-worker"], cwd=ROOT, env=environment,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                bufsize=0, close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW)
        owner = OwnedProcess(proc, native=native, deadline=total_deadline)
        owner.contain()
        guard.owner = owner
        guard.monitor(once=True)
        guard.check()
        identity = dict(zip(("pid", "creation_filetime"), map(str, owner.identity)))
        record["identity"] = identity
        envelope = {"command": "START", "identity": identity}
        data = (json.dumps(envelope) + "\n").encode("utf-8")
        # The initially empty anonymous pipe receives only this tiny control token,
        # never dependency paths/configuration that could fill its buffer before a read.
        if len(data) > 256 or time.monotonic() >= work_deadline:
            raise RuntimeError("START exceeds its size or time budget.")
        if proc.stdin.write(data) != len(data):
            raise OSError("Short START write.")
        proc.stdin.close()
        while True:
            if time.monotonic() >= work_deadline:
                raise TimeoutError("Absolute worker deadline reached, including capture/finalization.")
            chunk, eof = native.capture(proc.stdout)
            capture.write(chunk)
            if capture.total > ACCEPTANCE_LOG_LIMIT:
                raise RuntimeError("Worker stdout/stderr exceeded the output budget.")
            code = owner.poll()
            guard.monitor(once=True)
            guard.check()
            if code is not None:
                record["worker_exit"] = code
                record.setdefault("worker_observed_exit_monotonic", time.monotonic())
                if code != 0:
                    raise RuntimeError("Worker exited unsuccessfully.")
                if owner.discover() != 0:
                    raise RuntimeError("Worker exited with live Job members.")
                if eof:
                    break
            time.sleep(min(0.25, max(0, work_deadline - time.monotonic())))
        # Explicit final edge after externally observed exit and inherited-pipe EOF.
        time.sleep(min(0.001, max(0, total_deadline - time.monotonic())))
        guard.monitor(once=True)
        guard.check()
        record["natural_exit"] = not owner.forced
        record["capture_complete"] = eof
    except BaseException as exc:
        record["error"] = type(exc).__name__ + ": " + str(exc)
        if owner is not None:
            try:
                owner.stop()
            except Exception as cleanup:
                cleanup_failure(cleanup)
    finally:
        if owner is not None:
            try:
                owner.finish()
                record["owned_cleanup"] = owner.closed
            except Exception as cleanup:
                cleanup_failure(cleanup)
            record["owned_accounting"] = dict(owner.final_accounting)
        if proc is not None:
            for pipe in (proc.stdin, proc.stdout):
                try:
                    pipe.close()
                except Exception as cleanup:
                    cleanup_failure(cleanup)
        # Ownership has finished (or failed and been retained separately). Continue
        # host reads independently, including when owned accounting was unavailable.
        guard.owner = None
        guard.monitor(once=True)
        if guard.reason:
            record.setdefault("error", guard.reason)
        record["memory"] = guard.acceptance_summary(coverage_start, coverage_end)
        minima = record["memory"]["minima"]
        record["base_host_floors_satisfied"] = all(
            minima[name] is not None and minima[name]["available_bytes"] >= 4 * 1024**3
            for name in ("physical", "commit"))
        if guard.cleanup_error:
            cleanup_failure(guard.cleanup_error)
    try:
        if not record.get("error") and not record.get("cleanup_error"):
            paths = list(args.output.iterdir())
            if ({path.name for path in paths} != {"provisional.json", "server.log", "engine.log"}
                    or any(path.is_symlink() or not path.is_file()
                           or path.stat().st_size > ACCEPTANCE_LOG_LIMIT for path in paths)):
                raise RuntimeError("Missing, unexpected or oversized worker artifacts.")
            provisional = args.output / "provisional.json"
            worker_record = json.loads(provisional.read_bytes())
            record["worker_complete"] = worker_record.get("status") == "provisional_complete"
            selection = getattr(args, "acceptance_workload", None)
            if selection:
                record["workload_evidence"] = {key: worker_record.get(key) for key in (
                    "acceptance_workload", "cohort", "expected_max_context", "acceptance_workload_passed",
                    "final_idle", "protection", "missing_evidence")}
                complete = record["worker_complete"]
                record["worker_complete"] = False
                try:
                    record["worker_complete"] = complete and validate_selected_workload(
                        worker_record, args, min(work_deadline, record["worker_observed_exit_monotonic"]))
                except ValueError as exc:
                    record["workload_error"] = str(exc)
                # Consume classification independently of workload/lifecycle success.
                # Never promote the worker's bool, digest or inventory on its own.
                try:
                    with args.config.open("rb") as stream:
                        data = stream.read(ACCEPTANCE_LOG_LIMIT + 1)
                    if len(data) > ACCEPTANCE_LOG_LIMIT:
                        raise ValueError("Configuration exceeds its bounded read budget.")
                    config_identity = next(item for item in worker_record["protection"]["before"]
                                           if item["path"] == str(args.config.resolve()))
                    if config_identity["sha256"] != hashlib.sha256(data).hexdigest():
                        raise ValueError("Consumed configuration binding differs.")
                    cfg = json.loads(data.decode("utf-8-sig"))
                    prepared, inherited = clean_environment(cfg, args.timing, getattr(args, "fetch_timing", False),
                                                           acceptance=True, inherited=worker_environment)
                    record["normal_comparison"] = consume_normal_comparison(worker_record, prepared, inherited)
                except (OSError, ValueError, TypeError, KeyError, StopIteration, AttributeError):
                    record["normal_comparison"] = normal_comparison_excluded("classification-input-unavailable")
            record["protection"] = acceptance_protection(args) is True
            guard.monitor(once=True)
            guard.check()
        args.output.mkdir(parents=True, exist_ok=True)
        bounded_write(args.output / "controller.log", capture.text().encode("utf-8"), total_deadline)
        # A saved receipt always remains HELD. Only the caller can observe this function's
        # post-close result, and must also observe its external exit before qualifying it.
        record["memory_finalization"] = "provisional until post-artifact host edge in returned result"
        payload = (json.dumps(record, indent=2) + "\n").encode("utf-8")
        pending = args.output / "lifecycle.json.tmp"
        bounded_write(pending, payload, total_deadline, limit=ACCEPTANCE_RECEIPT_LIMIT)
        pending.replace(args.output / "lifecycle.json")
        if (args.output / "lifecycle.json").read_bytes() != payload:
            raise OSError("Final receipt readback mismatch.")
        if sum(path.stat().st_size for path in args.output.iterdir()) > ACCEPTANCE_ARTIFACT_LIMIT:
            raise RuntimeError("Total retained artifact budget exceeded.")
        record["finalization"] = True
    except Exception as exc:
        record["finalization_error"] = type(exc).__name__ + ": " + str(exc)
        record["finalization"] = False
    finally:
        # No filesystem/native finalization follows this edge. A slow returning
        # operation is refused by the observed gap; this is not syscall preemption.
        coverage_end = time.monotonic()
        guard.monitor(once=True)
        if guard.reason:
            record.setdefault("error", guard.reason)
        record["memory"] = guard.acceptance_summary(coverage_start, coverage_end)
        minima = record["memory"]["minima"]
        record["base_host_floors_satisfied"] = all(
            minima[name] is not None and minima[name]["available_bytes"] >= 4 * 1024**3
            for name in ("physical", "commit"))
        record["within_deadline"] = time.monotonic() < total_deadline
    if not acceptance_final_predicate(record):
        raise AcceptanceFailure(record, owner)
    record["lifecycle_evidence"] = "satisfied"
    return record


def run(args):
    port = getattr(args, "port", PORT)
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Port must be between 1024 and 65535.")
    total_seconds = getattr(args, "acceptance_total_seconds", None)
    if total_seconds is not None:
        if not getattr(args, "acceptance_telemetry", False):
            raise ValueError("Acceptance total seconds require --acceptance-telemetry.")
        if type(total_seconds) is not int or not 900 <= total_seconds <= 5400:
            raise ValueError("Acceptance total seconds must be between 900 and 5400.")
    acceptance_selection(args)
    if getattr(args, "acceptance_telemetry", False):
        return acceptance_controller(args, time.monotonic())
    return _run_workload(args)


def _run_workload(args):
    worker_deadline = getattr(args, "_acceptance_deadline", None)
    selection, expected_context = acceptance_selection(args)
    idle_seconds = getattr(args, "idle_seconds", 0)
    held_out = getattr(args, "held_out", False)
    transfer_workload = getattr(args, "transfer_workload", False)
    long_context = getattr(args, "long_context", False)
    validation_only = getattr(args, "validation_only", False)
    fetch_timing = getattr(args, "fetch_timing", False)
    acceptance = getattr(args, "acceptance_telemetry", False)
    floors = (getattr(args, "acceptance_physical_gib", None),
              getattr(args, "acceptance_commit_gib", None))
    if not acceptance and any(value is not None for value in floors):
        raise ValueError("Acceptance floors require --acceptance-telemetry.")
    acceptance_guard = (MemoryGuard(None, None, acceptance_floors=tuple(
        4 if value is None else value for value in floors)) if acceptance else None)
    if validation_only and not (held_out or transfer_workload or long_context or args.smoke
                               or getattr(args, "smoke_only", False)):
        raise ValueError("validation_only requires held_out, transfer_workload, long_context or smoke.")
    if not valid_number(idle_seconds) or not 0 <= idle_seconds <= 600:
        raise ValueError("idle_seconds must be between 0 and 600.")
    if args.output.exists():
        raise RuntimeError("Output directory already exists; choose a new directory.")
    config_bytes = args.config.read_bytes() if selection else None
    cfg = json.loads(config_bytes.decode("utf-8-sig") if selection
                     else args.config.read_text(encoding="utf-8-sig"))
    secret = cfg.get("api_key") or ""
    if not isinstance(secret, str) or not secret.strip():
        raise ValueError("An API key is required for the authenticated loopback benchmark.")
    if held_out or transfer_workload or long_context or selection:
        if cfg.get("fit_max_tokens") is not True or cfg.get("open_browser") is not False:
            raise ValueError("Acceptance requires fit_max_tokens=true and open_browser=false in the source config.")
    if acceptance_guard is not None:
        # This is the actual future run's guard and environment composition, not a launch clearance.
        prepared, environment = (clean_environment(cfg, args.timing, fetch_timing, acceptance=True)
                                 if selection else clean_environment(cfg, args.timing, fetch_timing))
        record = {"status": "held", "acceptance_telemetry": True,
                  "acceptance_workload": selection, "cohort": getattr(args, "cohort", None),
                  "expected_max_context": expected_context,
                  "effective_environment": safe_engine_environment(prepared, environment, acceptance=bool(selection)),
                  "normal_comparison": normal_comparison_excluded("launch-held"),
                  "memory": acceptance_guard.summary(),
                  "pending_stages": {"B": "Whole-owner deadline, output, external exit and native cleanup.",
                                     "C": "Workload, protection, provenance and cohort binding."}}
        args.output.mkdir(parents=True, exist_ok=False)
        pending = args.output / "results.json.tmp"
        pending.write_text(json.dumps(redact(record, secret), indent=2) + "\n", encoding="utf-8")
        replace_result(pending, args.output / "results.json")
        raise RuntimeError("Acceptance telemetry preparation held: Stage B/C and CUDA stage evidence are missing.")
    assert_idle(getattr(args, "port", PORT))
    exe = Path(cfg["exe"])
    if not exe.is_absolute():
        exe = Path(cfg.get("cwd") or ROOT) / exe
    exe = exe.resolve()
    saved_exe = exe
    protected = []
    provenance = None
    if selection:
        automatic = bool(cfg.get("memory_policy"))
        if (getattr(args, "cohort", None) == "automatic") != automatic:
            raise ValueError("Selected cohort disagrees with the saved automatic/static policy.")
        from tools.prepare_unsloth_q2 import engine_metadata
        saved_metadata = Path(cfg.get("build_metadata") or engine_metadata(saved_exe))
        if not saved_metadata.is_absolute():
            saved_metadata = ROOT / saved_metadata
        candidate = getattr(args, "candidate_exe", None)
        if candidate is not None:
            exe = Path(candidate).resolve()
            cfg["exe"] = str(exe)
            cfg["build_metadata"] = str(engine_metadata(exe))
        metadata = engine_metadata(exe) if candidate is not None else saved_metadata
        provenance = {"saved": executable_provenance(saved_exe, saved_metadata),
                      "selected": executable_provenance(exe, metadata)}
        paths = [args.config.resolve(), saved_exe, exe]
        paths += [path.resolve() for path in (saved_metadata, metadata) if path.is_file()]
        if cfg.get("tokenizer"):
            tokenizer = Path(cfg["tokenizer"])
            paths += [(tokenizer / name).resolve() for name in (
                "vocab.json", "merges.txt", "token_type.json", "chat_template.jinja")]
        paths += [Path(path).resolve() for path in getattr(args, "protect_input", [])]
        protected = list(dict.fromkeys(paths))
        protection_before = protection_snapshot(protected)
        if protection_before[0]["sha256"] != hashlib.sha256(config_bytes).hexdigest():
            raise ValueError("Configuration changed while binding the consumed inputs.")
    with exe.open("rb") as binary:
        executable_hash = hashlib.file_digest(binary, "sha256").hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"status": "starting", "config": str(args.config.resolve()), "exe": str(exe.resolve()),
              "exe_sha256": executable_hash, "timing_diagnostic": args.timing or fetch_timing,
              "decode_timing": args.timing, "fetch_timing": fetch_timing,
              "expected_max_context": expected_context, "smoke_only": getattr(args, "smoke_only", False),
              "validation_only": validation_only,
              "rows": [], "requests": [],
              "normal_comparison": normal_comparison_excluded("environment-not-classified")}
    if selection:
        record.update(acceptance_workload=selection, cohort=args.cohort,
                      executable_provenance=provenance, harness=file_identity(Path(__file__).resolve()),
                      protection={"scope": "Enumerated inputs only; complete operational admission remains held.",
                                  "before": protection_before, "after": None, "unchanged": False},
                      missing_evidence=["Fresh parent-reviewed one-use operational manifest and complete current input admission.",
                                        "Native lifecycle prerequisite remains unverified.",
                                        "Actual per-request CUDA-stage evidence (minimum 256 MiB).",
                                        "Final recent-cache allocation."] +
                                       (["Server's once-refreshed automatic policy sample."] if automatic else []))
    guard = None

    def save():
        if guard:
            record["memory"] = guard.summary()
        if worker_deadline is not None:
            snapshot = dict(record, status="provisional_" + record["status"])
            pending = args.output / "provisional.json.tmp"
            bounded_write(pending, (json.dumps(redact(snapshot, secret), indent=2) + "\n").encode("utf-8"),
                          worker_deadline)
            pending.replace(args.output / "provisional.json")
            return
        pending = args.output / "results.json.tmp"
        pending.write_text(json.dumps(redact(record, secret), indent=2) + "\n", encoding="utf-8")
        replace_result(pending, args.output / "results.json")

    save()
    record["working_tree"] = source_identity() if worker_deadline is None else source_identity(worker_deadline)
    metadata = exe.with_name("BUILD-" + exe.stem.removeprefix("strata-") + ".json")
    record["build_metadata"] = file_identity(metadata) if metadata.is_file() else None
    SCRATCH.mkdir(exist_ok=True)
    # TemporaryDirectory owns only this unique child; no source config is rewritten.
    with tempfile.TemporaryDirectory(prefix="compact-cache-", dir=SCRATCH) as temporary:
        temp = Path(temporary).resolve()
        if temp.parent != SCRATCH.resolve():
            raise RuntimeError("Temporary config escaped the intended scratch directory.")
        engine_log = temp / "engine.log"
        cfg, environment = (clean_environment(cfg, args.timing, fetch_timing, acceptance=True)
                            if selection else clean_environment(cfg, args.timing, fetch_timing))
        classification = normal_comparison_environment(cfg, environment, server_log=worker_deadline is None)
        # A refused source inventory cannot establish that instrumentation is absent.
        # Final automatic binding alone does not change diagnostic classification.
        record["timing_diagnostic"] = (record["timing_diagnostic"] or not classification["complete"] or any(
            item.get("active") is True for item in classification["controls"]))
        record["normal_comparison"] = normal_comparison_binding(cfg, classification)
        cfg.update(host="127.0.0.1", port=getattr(args, "port", PORT),
                   log=None if worker_deadline is not None else str(engine_log))
        record["effective_settings"] = {key: cfg.get(key) for key in (
            "model_name", "args", "fit_max_tokens", "open_browser",
            "expert_profile_save", "expert_profile_save_every")}
        record["effective_settings"]["env"] = safe_engine_environment(cfg, environment, acceptance=bool(selection))
        config_path = temp / "config.json"
        if worker_deadline is not None:
            bounded_write(config_path, json.dumps(cfg).encode("utf-8"), worker_deadline)
        else:
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
        owner, capture, pump, proc = None, BoundedCapture(
            ACCEPTANCE_LOG_LIMIT - 64 if worker_deadline is not None else LOG_LIMIT), None, None
        try:
            # Recheck immediately before spawning: never attach to an existing service.
            assert_idle(getattr(args, "port", PORT))
            proc = subprocess.Popen([sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata",
                                     "--config", str(config_path), "--host", "127.0.0.1", "--port",
                                     str(getattr(args, "port", PORT))],
                                    cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            owner = OwnedProcess(proc) if worker_deadline is None else None
            pump = threading.Thread(target=capture.drain, args=(proc.stdout,), daemon=True)
            pump.start()
            guard = MemoryGuard(owner, engine_log, deadline=worker_deadline)
            if worker_deadline is None:
                guard.thread.start()
            validation = load_workload("validation")
            validation.OPENER = TimeoutOpener(worker_deadline)
            client = make_client(validation, cfg, record, save, guard, secret)
            deadline = min(time.monotonic() + 600, worker_deadline) if worker_deadline is not None else time.monotonic() + 600
            while time.monotonic() < deadline:
                guard.check()
                if proc.poll() is not None:
                    raise RuntimeError("Server exited during startup; inspect the bounded logs.")
                try:
                    health = client.get("/health")
                    if health.get("loaded"):
                        break
                except OSError:
                    pass
                guard.done.wait(1 if worker_deadline is None else min(1, max(0, deadline - time.monotonic())))
            else:
                raise RuntimeError("Server did not load within 600 seconds.")
            record["health"] = health
            if health.get("max_context") != expected_context:
                raise RuntimeError(f"Loaded server context is not {expected_context}; benchmark stopped.")
            record["engine"] = client.get("/metrics")["engine"]
            record["status"] = "running"
            save()
            if held_out or transfer_workload or long_context or selection:
                check_authentication(client, record, save)
            if selection:
                acceptance_workload_tests(validation, client, record, save, guard)
            else:
                record["warmup"] = client.chat([{"role": "user", "content":
                    "Explain what a database transaction does in a few sentences."}], 64)
            prompts = [] if selection or record["smoke_only"] or validation_only else load_workload("benchmark").PROMPTS
            for round_index in range(2):
                if round_index == 1 and prompts and idle_seconds:
                    record["idle_started_unix"] = time.time()
                    save()
                    print(f"Idle for {idle_seconds:g} seconds before round 2; monitoring continues.", flush=True)
                    idle_end = time.monotonic() + idle_seconds
                    if worker_deadline is not None:
                        idle_end = min(idle_end, worker_deadline)
                    next_notice = time.monotonic() + 30
                    while time.monotonic() < idle_end:
                        guard.check()
                        if time.monotonic() >= next_notice:
                            remaining = max(0, idle_end - time.monotonic())
                            print(f"Intentional idle pause: {remaining:.0f}s remaining; requests resume automatically.",
                                  flush=True)
                            next_notice = time.monotonic() + 30
                        guard.done.wait(min(1, max(0, idle_end - time.monotonic())))
                    guard.check()
                    record["idle_finished_unix"] = time.time()
                    save()
                    print("Idle pause finished. Resuming test requests automatically.", flush=True)
                for index, prompt in enumerate(prompts):
                    client.label = f"round-{round_index}-case-{index}"
                    response = client.chat([
                        {"role": "system", "content": f"Technical explanation case {index}."},
                        {"role": "user", "content": prompt}], 160)
                    record["rows"].append({"round": round_index, "case": index, "prompt": prompt,
                                           "response": response, "metric": record["requests"][-1]["metric"]})
                    save()
                    print(f"Completed round {round_index + 1}, case {index + 1}.", flush=True)
            record["summary"] = summarize(record["rows"])
            if args.smoke or record["smoke_only"]:
                client.label = "smoke"
                validation.OUT = args.output
                smoke_log = BoundedCapture()
                try:
                    with contextlib.redirect_stdout(smoke_log):
                        record["smoke_passed"] = validation.smoke(client)
                finally:
                    (args.output / "smoke.log").write_text(redact(smoke_log.text(), secret), encoding="utf-8")
                if not record["smoke_passed"]:
                    raise RuntimeError("Model smoke validation failed; inspect smoke.json.")
            if held_out:
                held_out_tests(client, record, save)
            if transfer_workload:
                transfer_workload_tests(client, record, save)
            if long_context:
                long_context_tests(validation, client, record, save)
            metrics = client.get("/metrics")
            record["metrics"] = {k: metrics[k] for k in ("engine", "totals")}
            if selection:
                # INFO fields describe actual engine-reported startup allocations, not
                # final host-cache capacity or the server's independent policy resample.
                record["allocation_evidence"] = {
                    "requested_args": cfg.get("args"),
                    "startup_engine_info": record["engine"], "final_engine_info": metrics["engine"],
                    "meaning": "Engine INFO observations; missing final automatic/recent allocations remain explicit."}
            guard.check()
            record["status"] = "complete"
        except BaseException as exc:
            record["status"] = "failed"
            record["error_type"] = type(exc).__name__
            record["error"] = redact(str(exc) or type(exc).__name__, secret)
            raise
        finally:
            if guard and worker_deadline is None:
                guard.done.set()
                guard.thread.join(timeout=10)
            try:
                if owner:
                    owner.stop()
                elif proc is not None and worker_deadline is not None:
                    # The controller's Job owns descendants. Stop only this retained Popen
                    # server handle here; controller independently accounts the complete Job.
                    proc.terminate()
                    proc.wait(timeout=max(0, worker_deadline - time.monotonic()))
                    record["server_cleanup"] = "retained server handle terminated and waited; controller owns descendants"
            except Exception as exc:
                record["status"] = "failed"
                record["cleanup_error" if worker_deadline is not None else "error"] = redact(
                    "Process cleanup failed: " + str(exc), secret)
                raise
            finally:
                if guard and guard.reason:
                    record["status"] = "failed"
                    record["error"] = guard.reason
                if pump:
                    pump.join(timeout=5 if worker_deadline is None else max(0, worker_deadline - time.monotonic()))
                try:
                    if worker_deadline is not None:
                        if pump and pump.is_alive():
                            raise RuntimeError("Worker inherited server capture did not close.")
                        if not capture.complete or capture.error:
                            raise RuntimeError("Worker server capture failed to drain and close.")
                        if capture.total > ACCEPTANCE_LOG_LIMIT:
                            raise RuntimeError("Worker server capture exceeded the output budget.")
                        bounded_write(args.output / "server.log", redact(capture.text(), secret).encode("utf-8"),
                                      worker_deadline)
                        bounded_write(args.output / "engine.log", b"Engine file logging held for bounded Stage C collection.\n",
                                      worker_deadline)
                    else:
                        (args.output / "server.log").write_text(redact(capture.text(), secret), encoding="utf-8")
                        (args.output / "engine.log").write_text(redact(bounded_file(engine_log), secret), encoding="utf-8")
                except OSError as exc:
                    record["status"] = "failed"
                    record["artifact_error"] = redact(type(exc).__name__ + ": " + str(exc), secret)
                    raise
                finally:
                    if protected:
                        try:
                            record["protection"]["after"] = protection_snapshot(protected)
                            record["protection"]["unchanged"] = record["protection"]["after"] == protection_before
                            if not record["protection"]["unchanged"]:
                                raise RuntimeError("Protected benchmark inputs changed during the workload.")
                        except Exception as exc:
                            record["status"] = "failed"
                            record.setdefault("error", redact(str(exc), secret))
                    save()
    if record["status"] != "complete":
        raise RuntimeError("Benchmark was cancelled; inspect results.json.")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="A new directory; existing directories are refused")
    parser.add_argument("--port", type=int, default=PORT, help="Loopback server port, 1024-65535 (default 8081)")
    parser.add_argument("--timing", action="store_true", help="Enable CPU decode stage timing; diagnostic run only")
    parser.add_argument("--fetch-timing", action="store_true", help="Enable expert-fetch profiling; diagnostic run only")
    parser.add_argument("--smoke", action="store_true", help="Also run the existing six model smoke checks")
    parser.add_argument("--smoke-only", action="store_true", help="Run warmup and smoke checks without timed cases")
    parser.add_argument("--held-out", action="store_true", help="Also run four fixed coding/reasoning acceptance checks")
    parser.add_argument("--transfer-workload", action="store_true",
                        help="Also run four broader held-out 160-token cases; quality requires manual review")
    parser.add_argument("--long-context", action="store_true",
                        help="Also validate a fresh near-128K prompt, fit_max_tokens and reuse (may take many minutes)")
    parser.add_argument("--expected-context", type=int, choices=(65536, CONTEXT), default=CONTEXT,
                        help="Required loaded context capacity (default 131072)")
    parser.add_argument("--acceptance-workload", choices=("short", "near-full"),
                        help="Contained opt-in eleven-request short or separate near-full/sustained workload; admission held")
    parser.add_argument("--cohort", choices=("saved-static", "automatic", "frozen-control"),
                        help="Explicit policy cohort for the selected acceptance workload")
    parser.add_argument("--candidate-exe", type=Path,
                        help="Use this executable in a diagnostic clone, preserving the saved executable's provenance")
    parser.add_argument("--protect-input", action="append", default=[], metavar="PATH",
                        help="Additional exact input file to hash before/after (for example the selected launcher)")
    parser.add_argument("--validation-only", action="store_true",
                        help="Run warmup and selected acceptance checks without the six throughput cases")
    parser.add_argument("--idle-seconds", type=float, default=0,
                        help="Idle 0-600 seconds between timed rounds while keeping the memory guard active")
    parser.add_argument("--acceptance-telemetry", action="store_true",
                        help="Prepare opt-in host acceptance telemetry; runtime remains held for Stage B/C")
    parser.add_argument("--acceptance-total-seconds", type=int,
                        help="Opt-in total acceptance budget, 900-5400 seconds (default 900)")
    parser.add_argument("--acceptance-physical-gib", type=float,
                        help="Opt-in physical headroom floor in GiB (default 4)")
    parser.add_argument("--acceptance-commit-gib", type=float,
                        help="Opt-in available-commit headroom floor in GiB (default 4)")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, AcceptanceFailure) and exc.owner is not None and not exc.owner.closed:
            # An embedding caller must receive the retained authority, not only an integer
            # that discards the unresolved handles. A CLI invocation exits nonzero normally.
            raise
        # Arbitrary exception messages can include HTTP credentials; details are redacted on disk.
        print(f"Benchmark stopped ({type(exc).__name__}); see results.json if the run started.", file=sys.stderr)
        return 1
    print(f"Saved benchmark to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
