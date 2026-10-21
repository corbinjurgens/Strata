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

import psutil

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ROOT / "bench/results/2026-10-09-unsloth-file-cache"
SCRATCH = ROOT / "build-unsloth"
PORT = 8081
CONTEXT = 131072
LOG_LIMIT = 4 * 1024**2
PROFILE_ENV = {"STRATA_VERIFY_PROFILE", "STRATA_DECODE_TIMING", "STRATA_FETCH_PROFILE"}
EVIDENCE_ENV = PROFILE_ENV | {
    "STRATA_RESIDENT_HEADROOM_GIB", "STRATA_FILE_CACHE_MIB", "STRATA_FILE_CACHE_COMPACT", "STRATA_FILE_DIRECT_UNMAP",
    "STRATA_KV_GROW", "STRATA_ARENA_PIN_GIB", "STRATA_PARTIAL_PIN_GIB", "STRATA_PARTIAL_PIN",
    "STRATA_UNBUFFERED_LOAD", "STRATA_FILE_RELEASE", "STRATA_DEC_BATCH", "STRATA_DEFERRED_REGISTER",
    "STRATA_OWNED_PRICE", "STRATA_FETCH_PVM",
}
LONG_TARGET = CONTEXT - 192
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


def clean_environment(cfg, timing, fetch_timing=False):
    """Disable inherited profiling and make server/client use the same saved API key."""
    cfg = copy.deepcopy(cfg)
    excluded = PROFILE_ENV | {"STRATA_API_KEY"}
    env = {k: v for k, v in os.environ.items() if k.upper() not in excluded}
    cfg["env"] = {k: v for k, v in cfg.get("env", {}).items() if k.upper() not in excluded}
    if timing:
        cfg["env"]["STRATA_DECODE_TIMING"] = "1"
    if fetch_timing:
        cfg["env"]["STRATA_FETCH_PROFILE"] = "1"
    return cfg, env


def file_identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def safe_engine_environment(cfg, inherited):
    """Record allowed controls from both actual environment layers, with config taking precedence."""
    from serve.server import config_environment
    effective = {}
    for key, value in config_environment(cfg, inherited).items():
        name = key.upper() if os.name == "nt" else key
        if name in EVIDENCE_ENV:
            effective[name] = value
    return effective


def source_identity():
    """Working-tree evidence only: this does not claim an old executable used these sources."""
    paths = ["src", "include", "cmake", "CMakeLists.txt", "serve"]

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, timeout=30,
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
            with TimeoutOpener().open(request, timeout=20) as response:
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


def assert_idle():
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.info["pid"] != os.getpid() and is_strata_process(process.info):
            raise RuntimeError("A Strata engine or server is already running; close it before benchmarking.")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(("127.0.0.1", PORT))
        except OSError:
            raise RuntimeError("Port 8081 is occupied; no process was started or stopped.") from None


class OwnedProcess:
    """Only retain psutil identities descended from the Popen we just created."""

    def __init__(self, proc):
        self.proc = proc
        self.root = psutil.Process(proc.pid)
        self.members = {self.root}
        self.lock = threading.Lock()
        self.stopped = False

    def discover(self):
        try:
            if self.root.is_running():
                self.members.update(self.root.children(recursive=True))
        except psutil.NoSuchProcess:
            pass

    def stop(self):
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


class MemoryGuard:
    def __init__(self, owner, engine_log, *, acceptance_floors=None):
        self.owner, self.engine_log = owner, engine_log
        self.done = threading.Event()
        self.reason = None
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
            if len(self.host_samples) >= 4096:
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
                    self.owner.stop()

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
                "sample_limit": 4096, "cuda_stage_evidence": "missing; later-stage hold",
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

    def monitor(self):
        try:
            while not self.done.is_set():
                self.owner.discover()
                if self.acceptance_floors is not None:
                    from tools.prepare_unsloth_q2 import memory_snapshot
                    started = time.monotonic()
                    snapshot = memory_snapshot()
                    self.acceptance_sample(snapshot, started, time.monotonic())
                    if self.reason:
                        return
                    if self.engine_log.exists() and self.engine_log.stat().st_size > 2 * LOG_LIMIT:
                        raise RuntimeError("Engine log exceeded the diagnostic size limit.")
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
            self.reason = "Memory/process monitor failed: " + type(exc).__name__
            self.owner.stop()

    def check(self):
        if self.reason:
            raise RuntimeError(self.reason)

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

    def __init__(self):
        self.head, self.tail = bytearray(), bytearray()
        self.total = 0

    def write(self, value):
        data = value.encode("utf-8") if isinstance(value, str) else value
        self.total += len(data)
        head_size = min(len(data), LOG_LIMIT // 2 - len(self.head))
        self.head.extend(data[:head_size])
        self.tail.extend(data[head_size:])
        if len(self.tail) > LOG_LIMIT // 2:
            del self.tail[:-LOG_LIMIT // 2]
        return len(value)

    def flush(self):
        # Writes already update the in-memory capture; print(..., flush=True) needs this method.
        pass

    def drain(self, pipe):
        try:
            while chunk := pipe.read(8192):
                self.write(chunk)
        finally:
            pipe.close()

    def text(self):
        gap = b"\n[log middle omitted]\n" if self.total > LOG_LIMIT else b""
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


def matched_metric(response, before, after):
    old = {row["time"] for row in before.get("requests", [])}
    new = [row for row in after.get("requests", []) if row["time"] not in old]
    if after["totals"]["requests"] - before["totals"]["requests"] != 1 or len(new) != 1:
        raise RuntimeError("Could not uniquely match this response to one new request metric.")
    metric = new[0]
    usage, timings = response["usage"], response["timings"]
    if (metric["output_tokens"] != usage["completion_tokens"]
            or metric["prompt_tokens"] != usage["prompt_tokens"]
            or abs(metric["decode_ms"] - timings["predicted_ms"]) > 0.11):
        raise RuntimeError("New request metric does not agree with the response usage/timing.")
    return metric


class TimeoutOpener:
    def __init__(self):
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def open(self, request, timeout=1800):
        return self.opener.open(request, timeout=min(timeout, 1800))


def make_client(validation, cfg, record, save, guard, secret):
    class Client(validation.Client):
        label = "warmup"

        def get(self, path):
            guard.check()
            return super().get(path)

        def chat(self, messages, max_tokens=128, **kwargs):
            guard.check()
            item = {"label": self.label, "messages": copy.deepcopy(messages), "max_tokens": max_tokens,
                    "started_unix": time.time()}
            record["requests"].append(item)
            save()
            try:
                before = self.get("/metrics")
                response = redact(super().chat(messages, max_tokens, **kwargs), secret)
                item["response"] = response
                save()  # Keep the answer even if the subsequent metric lookup fails.
                item["metric"] = matched_metric(response, before, self.get("/metrics"))
                guard.check()
                return response
            except Exception as exc:
                item["error"] = str(exc)
                raise
            finally:
                item["finished_unix"] = time.time()
                save()

    return Client(cfg)


def run(args):
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
    if not 0 <= idle_seconds <= 600:
        raise ValueError("idle_seconds must be between 0 and 600.")
    if args.output.exists():
        raise RuntimeError("Output directory already exists; choose a new directory.")
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    secret = cfg.get("api_key") or ""
    if not isinstance(secret, str) or not secret.strip():
        raise ValueError("An API key is required for the authenticated loopback benchmark.")
    if held_out or transfer_workload or long_context:
        if cfg.get("fit_max_tokens") is not True or cfg.get("open_browser") is not False:
            raise ValueError("Acceptance requires fit_max_tokens=true and open_browser=false in the source config.")
    if acceptance_guard is not None:
        # This is the actual future run's guard and environment composition, not a launch clearance.
        prepared, environment = clean_environment(cfg, args.timing, fetch_timing)
        record = {"status": "held", "acceptance_telemetry": True,
                  "effective_environment": safe_engine_environment(prepared, environment),
                  "memory": acceptance_guard.summary(),
                  "pending_stages": {"B": "Whole-owner deadline, output, external exit and native cleanup.",
                                     "C": "Workload, protection, provenance and cohort binding."}}
        args.output.mkdir(parents=True, exist_ok=False)
        pending = args.output / "results.json.tmp"
        pending.write_text(json.dumps(redact(record, secret), indent=2) + "\n", encoding="utf-8")
        replace_result(pending, args.output / "results.json")
        raise RuntimeError("Acceptance telemetry preparation held: Stage B/C and CUDA stage evidence are missing.")
    assert_idle()
    exe = Path(cfg["exe"])
    if not exe.is_absolute():
        exe = Path(cfg.get("cwd") or ROOT) / exe
    with exe.open("rb") as binary:
        executable_hash = hashlib.file_digest(binary, "sha256").hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"status": "starting", "config": str(args.config.resolve()), "exe": str(exe.resolve()),
              "exe_sha256": executable_hash, "timing_diagnostic": args.timing or fetch_timing,
              "decode_timing": args.timing, "fetch_timing": fetch_timing,
              "expected_max_context": CONTEXT, "smoke_only": getattr(args, "smoke_only", False),
              "validation_only": validation_only,
              "rows": [], "requests": []}
    guard = None

    def save():
        if guard:
            record["memory"] = guard.summary()
        pending = args.output / "results.json.tmp"
        pending.write_text(json.dumps(redact(record, secret), indent=2) + "\n", encoding="utf-8")
        replace_result(pending, args.output / "results.json")

    save()
    record["working_tree"] = source_identity()
    metadata = exe.with_name("BUILD-" + exe.stem.removeprefix("strata-") + ".json")
    record["build_metadata"] = file_identity(metadata) if metadata.is_file() else None
    SCRATCH.mkdir(exist_ok=True)
    # TemporaryDirectory owns only this unique child; no source config is rewritten.
    with tempfile.TemporaryDirectory(prefix="compact-cache-", dir=SCRATCH) as temporary:
        temp = Path(temporary).resolve()
        if temp.parent != SCRATCH.resolve():
            raise RuntimeError("Temporary config escaped the intended scratch directory.")
        engine_log = temp / "engine.log"
        cfg, environment = clean_environment(cfg, args.timing, fetch_timing)
        cfg.update(host="127.0.0.1", port=PORT, log=str(engine_log))
        record["effective_settings"] = {key: cfg.get(key) for key in (
            "model_name", "args", "fit_max_tokens", "open_browser",
            "expert_profile_save", "expert_profile_save_every")}
        record["effective_settings"]["env"] = safe_engine_environment(cfg, environment)
        config_path = temp / "config.json"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
        owner, capture, pump = None, BoundedCapture(), None
        try:
            # Recheck immediately before spawning: never attach to an existing service.
            assert_idle()
            proc = subprocess.Popen([sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata",
                                     "--config", str(config_path), "--host", "127.0.0.1", "--port", str(PORT)],
                                    cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            owner = OwnedProcess(proc)
            pump = threading.Thread(target=capture.drain, args=(proc.stdout,), daemon=True)
            pump.start()
            guard = MemoryGuard(owner, engine_log)
            guard.thread.start()
            validation = load_workload("validation")
            validation.OPENER = TimeoutOpener()
            client = make_client(validation, cfg, record, save, guard, secret)
            deadline = time.monotonic() + 600
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
                guard.done.wait(1)
            else:
                raise RuntimeError("Server did not load within 600 seconds.")
            record["health"] = health
            if health.get("max_context") != CONTEXT:
                raise RuntimeError("Loaded server context is not 131072; benchmark stopped.")
            record["engine"] = client.get("/metrics")["engine"]
            record["status"] = "running"
            save()
            if held_out or transfer_workload or long_context:
                check_authentication(client, record, save)
            record["warmup"] = client.chat([{"role": "user", "content":
                "Explain what a database transaction does in a few sentences."}], 64)
            prompts = [] if record["smoke_only"] or validation_only else load_workload("benchmark").PROMPTS
            for round_index in range(2):
                if round_index == 1 and prompts and idle_seconds:
                    record["idle_started_unix"] = time.time()
                    save()
                    print(f"Idle for {idle_seconds:g} seconds before round 2; monitoring continues.", flush=True)
                    idle_end = time.monotonic() + idle_seconds
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
            guard.check()
            record["status"] = "complete"
        except BaseException as exc:
            record["status"] = "failed"
            record["error_type"] = type(exc).__name__
            record["error"] = redact(str(exc) or type(exc).__name__, secret)
            raise
        finally:
            if guard:
                guard.done.set()
                guard.thread.join(timeout=10)
            try:
                if owner:
                    owner.stop()
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = redact("Process cleanup failed: " + str(exc), secret)
                raise
            finally:
                if guard and guard.reason:
                    record["status"] = "failed"
                    record["error"] = guard.reason
                if pump:
                    pump.join(timeout=5)
                try:
                    (args.output / "server.log").write_text(redact(capture.text(), secret), encoding="utf-8")
                    (args.output / "engine.log").write_text(redact(bounded_file(engine_log), secret), encoding="utf-8")
                except OSError as exc:
                    record["status"] = "failed"
                    record["artifact_error"] = redact(type(exc).__name__ + ": " + str(exc), secret)
                    raise
                finally:
                    save()
    if record["status"] != "complete":
        raise RuntimeError("Benchmark was cancelled; inspect results.json.")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="A new directory; existing directories are refused")
    parser.add_argument("--timing", action="store_true", help="Enable CPU decode stage timing; diagnostic run only")
    parser.add_argument("--fetch-timing", action="store_true", help="Enable expert-fetch profiling; diagnostic run only")
    parser.add_argument("--smoke", action="store_true", help="Also run the existing six model smoke checks")
    parser.add_argument("--smoke-only", action="store_true", help="Run warmup and smoke checks without timed cases")
    parser.add_argument("--held-out", action="store_true", help="Also run four fixed coding/reasoning acceptance checks")
    parser.add_argument("--transfer-workload", action="store_true",
                        help="Also run four broader held-out 160-token cases; quality requires manual review")
    parser.add_argument("--long-context", action="store_true",
                        help="Also validate a fresh near-128K prompt, fit_max_tokens and reuse (may take many minutes)")
    parser.add_argument("--validation-only", action="store_true",
                        help="Run warmup and selected acceptance checks without the six throughput cases")
    parser.add_argument("--idle-seconds", type=float, default=0,
                        help="Idle 0-600 seconds between timed rounds while keeping the memory guard active")
    parser.add_argument("--acceptance-telemetry", action="store_true",
                        help="Prepare opt-in host acceptance telemetry; runtime remains held for Stage B/C")
    parser.add_argument("--acceptance-physical-gib", type=float,
                        help="Opt-in physical headroom floor in GiB (default 4)")
    parser.add_argument("--acceptance-commit-gib", type=float,
                        help="Opt-in available-commit headroom floor in GiB (default 4)")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (Exception, KeyboardInterrupt) as exc:
        # Arbitrary exception messages can include HTTP credentials; details are redacted on disk.
        print(f"Benchmark stopped ({type(exc).__name__}); see results.json if the run started.", file=sys.stderr)
        return 1
    print(f"Saved benchmark to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
