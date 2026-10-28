"""Real pipe/HTTP consumers of synthetic native replay and status records."""
from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import io
import json
import os
import queue
from pathlib import Path
import sys
import socket
import subprocess
from types import SimpleNamespace
import unittest
import urllib.request
from unittest.mock import patch

from serve import launch_plan as lp, server
from serve import launch_runtime as lr
from serve.frontend import ChatTemplate
from serve import test_launch_plan as fixtures
from tools import build_provenance


def ownership_frame(sequence=1, held=256, *, unknown=False, shards=1, capacity=1024):
    payload = "device\t63756461\t0\t303030303a30313a30302e30\nstate\t" + ("unknown" if unknown else "known")
    payload += "\nphysical\tunavailable\n"
    if unknown:
        payload += "shards\t0\n"
    else:
        payload += f"totals\t{capacity}\t{held + 32}\t{1024 - held - 32 - 64 - 128}\t64\t128\t32\t4096\nshards\t{shards}\n"
        payload += f"shard\t0\t{held}\t{held - 16}\t16\n"
        for index in range(1, shards):
            payload += f"shard\t{index}\t0\t0\t0\n"
    return f"STRATA-OWNERSHIP/1\n{sequence}\n{len(payload)}\n{payload}\nSTRATA-OWNERSHIP-END/1 {sequence}\n"


def runtime_frame(sequence=1, held=256, *, plan=None, unknown=False, shards=1, settings=None,
                  components=None, device=None):
    """Independent synthetic row specimen; its sizes are transport oracles, not fit evidence."""
    values = dict(context=8192, prefill=64, spec=2, kv=1, kv_resident=0, batch=0, batch_groups=1,
                  parallel=1, auto_cache=1, per_layer=0, no_prefill_borrow=1, resident_budget=1024,
                  resident_bytes=0, resident_experts=0, resident_pin=1, file_cache_effective=0,
                  file_cache_compact=1, host_headroom=4096, vram_reserve=256, cache_slots=2,
                  cache_bytes=512, shared_stage=0, ring=2, stager_threads=2, stager_ring=16,
                  stager_batch=1, prefill_borrowed=0)
    if plan:
        values.update({k: plan["resolved"][k] for k in values if k in plan["resolved"]})
    values.update(settings or {})
    memory = components if components is not None else {
        "vram": {"expert_cache": [512, 2096640, min(held, 512), 0, 0, 0],
                 "session_including_resident_kv": [2048, 2095104, 1024, 0, 0, 0],
                 "overhead_allowance": [0, 0, 0, 0, 0, values["vram_reserve"]]},
        "ram": {"streamed_kv": [4096, 0, 3072, 4096, 4096, 0],
                "file_cache_target": [128, 0, 64, 0, 0, 512],
                "host_headroom_allowance": [0, 0, 0, 0, 0, values["host_headroom"]]}}
    weights = ownership_frame(held=held, unknown=unknown, shards=shards).split("\n", 3)[3].split("\nSTRATA-OWNERSHIP-END", 1)[0]
    if device:
        weights = device + "\n" + weights.split("\n", 1)[1]
    payload = weights + "total_allocated\tunknown\ndriver_os\tunknown\n"
    payload += "".join(f"setting\t{k}\t{v}\n" for k, v in sorted(values.items()))
    for space in ("vram", "ram"):
        for name, numbers in sorted(memory.get(space, {}).items()):
            state = "unknown" if unknown or numbers is None else "known"
            fields = ["-"] * 6 if state == "unknown" else list(map(str, numbers))
            payload += "\t".join(["component", space, name, state, *fields]) + "\n"
    return f"STRATA-RUNTIME/1\n{sequence}\n{len(payload)}\n{payload}\nSTRATA-RUNTIME-END/1 {sequence}\n"


def fake_runtime_main():
    """A subprocess fixture, including Windows stdout CRLF and raw stdin capture."""
    if "FIXTURE_NATIVE_PATH" in os.environ:
        # The native .cmd shim captures its incoming PATH before Conda's Python
        # helper mutates it. Other native inputs pass through unchanged.
        os.environ["PATH"] = os.environ.pop("FIXTURE_NATIVE_PATH")
    if "--resolve-launch" in sys.argv:
        fixtures.fake_metadata_main()
        return
    cwd = Path.cwd()
    plan = json.loads((cwd / "approved.json").read_text())
    mode = (cwd / "mode.txt").read_text() if (cwd / "mode.txt").exists() else "ok"
    shards = int((cwd / "shards.txt").read_text()) if (cwd / "shards.txt").exists() else 1
    runtime_enabled = (cwd / "runtime.txt").exists()
    if mode == "crlf":
        sys.stdout.reconfigure(newline="\r\n")
    frame = b""
    if "--strict-launch-stdin" in sys.argv:
        header = sys.stdin.buffer.readline()
        length = sys.stdin.buffer.readline()
        payload = sys.stdin.buffer.read(int(length))
        frame = header + length + payload
    capture = {"argv": sys.argv[1:], "cwd": str(cwd), "frame": base64.b64encode(frame).decode(),
               "environment": {k: v for k, v in os.environ.items() if lp._relevant(k)},
               "credential_present": "STRATA_API_KEY" in os.environ or "STRATA_API_KEY_FILE" in os.environ}
    with (cwd / "capture.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(capture) + "\n")
    if mode == "exit-before-ready":
        return
    build = json.loads((cwd / "BUILD.json").read_text())["build"]
    if mode == "bad-build":
        build["commit"] = "mismatch"
    if mode not in ("no-build", "late-build"):
        print("BUILD " + json.dumps(build), flush=True)
    s = plan["resolved"]
    ctx = s["context"] + int(mode == "bad-info")
    print(f"INFO context={ctx} kv=int8 kv_resident=0 expert_slots=2 spec=2 engine=test", flush=True)
    if mode in ("short-extra-info", "oversized-info"):
        print("INFO vram_free_mib=" + ("123" if mode == "short-extra-info" else "9" * 5000), flush=True)
    if mode == "bad-parallel":
        print("INFO parallel=2", flush=True)
    if mode == "late-build":
        print("BUILD " + json.dumps(build), flush=True)
    if frame:
        if mode != "no-effective":
            payload = plan["native"]["effective_payload"]
            if mode == "bad-effective":
                payload = payload.replace("verified=1", "verified=0")
            sys.stdout.write(f"STRATA-EFFECTIVE/1\n1\n{len(payload)}\n{payload}")
        if mode != "no-ownership":
            sys.stdout.write(ownership_frame(shards=shards, unknown=mode == "unknown-start"))
    if runtime_enabled:
        raw = cwd / "native-runtime.bin"
        if raw.exists():
            sys.stdout.flush()
            sys.stdout.buffer.write(raw.read_bytes())
        else:
            sys.stdout.write(runtime_frame(plan=plan, shards=shards, unknown=mode == "unknown-start",
                                           settings={"prefill": 128} if mode == "bad-runtime" else None))
    print(f"READY {s['context'] + int(mode == 'bad-ready')} stop", flush=True)
    sequence = 1
    runtime_sequence = 1
    for command in sys.stdin:
        command = command.strip()
        if command == "QUIT":
            if (cwd / "late.txt").exists() and frame:
                sys.stdout.write(ownership_frame(sequence + 1, 512, shards=shards))
                sys.stdout.flush()
            if (cwd / "late.txt").exists() and runtime_enabled:
                sys.stdout.write(runtime_frame(runtime_sequence + 1, 512, plan=plan, shards=shards))
                sys.stdout.flush()
            return
        if command.startswith("RUNTIME_"):
            runtime_sequence += 2 if command == "RUNTIME_GAP" else 1
            changed = {"prefill": 128} if command == "RUNTIME_CONTRADICT" else None
            record = runtime_frame(runtime_sequence, 256 if command == "RUNTIME_SAME" else 512,
                                   plan=plan, shards=shards, unknown=command == "RUNTIME_UNKNOWN", settings=changed)
            if command == "RUNTIME_BAD":
                record = record.replace("shard\t0\t512", "shard\t0\t513")
            if command == "RUNTIME_TRUNCATED":
                sys.stdout.write(record[:100])
                sys.stdout.flush()
                return
            sys.stdout.write(record)
            print("SYNC " + command, flush=True)
            continue
        if command.startswith("GEN "):
            # ByteTokenizer's fixed answer exercises the real HTTP/parser path.
            for token in b"Hello.":
                print(f"T {token}")
            print("DONE 6 1 1 1 stop", flush=True)
            continue
        if command == "UPDATE":
            sequence += 1
            sys.stdout.write(ownership_frame(sequence, 512, shards=shards))
        elif command == "UNKNOWN":
            sequence += 1
            sys.stdout.write(ownership_frame(sequence, unknown=True))
        elif command == "UTF8":
            sys.stdout.buffer.write(b"\xff\n")
            sys.stdout.buffer.flush()
        elif command in ("BAD", "SEQUENCE", "TRUNCATED", "OVERFLOW", "LENGTH", "BADEND", "VERSION"):
            record = ownership_frame(sequence + 1, 512, shards=shards)
            if command == "BAD":
                record = record.replace("shard\t0\t512", "shard\t0\t513")
            if command == "SEQUENCE":
                record = ownership_frame(sequence, 512, shards=shards)
            if command == "OVERFLOW":
                record = ownership_frame(sequence + 1, 512, shards=shards, capacity=18446744073709551616)
            if command == "LENGTH":
                record = "STRATA-OWNERSHIP/1\n2\n16385\n"
            if command == "BADEND":
                record = record.replace("STRATA-OWNERSHIP-END/1 2", "STRATA-OWNERSHIP-END/1 9")
            if command == "VERSION":
                record = record.replace("STRATA-OWNERSHIP/1", "STRATA-OWNERSHIP/7", 1)
            if command == "TRUNCATED":
                sys.stdout.write(record[:70])
                sys.stdout.flush()
                return
            sys.stdout.write(record)
        print("SYNC " + command, flush=True)


class RuntimeCodec(unittest.TestCase):
    def engine(self, records=None):
        return SimpleNamespace(records=records or lr.Records(), launch_plan=None,
                               spawn=("fixture", ["--native", "model.gguf"], None, None))

    def consume(self, records, frame):
        stream = io.StringIO(frame)
        self.assertTrue(records.consume(stream.readline(), stream))
        return stream.read()

    def stale_unchanged(self, before, after):
        expected = copy.deepcopy(before)
        error = after["current_ownership"]["error"]
        self.assertIsNotNone(error)
        self.assertNotIn("SYNTHETIC-SECRET", error)
        expected["effective_state"] = "stale"
        expected["effective_reason"] = error
        expected["current_ownership"].update(freshness="stale", error=error)
        expected["runtime_memory"].update(freshness="stale", error=error)
        for space in ("vram", "ram"):
            for row in expected["runtime_comparison"][space].values():
                row["freshness"] = "stale"
        self.assertEqual(after, expected)

    def test_runtime_every_interruption_boundary_preserves_whole_publication(self):
        good, update = runtime_frame(), runtime_frame(2, 512, settings={"prefill": 128})
        for boundary in range(len(update)):
            with self.subTest(boundary=boundary):
                engine = self.engine()
                self.consume(engine.records, good)
                before = lr.status(engine)[0]
                engine.proc = SimpleNamespace(stdout=io.StringIO(update[:boundary]))
                engine.lines, engine.slot_q = queue.Queue(), []
                engine._native_lines = server.StrataEngine._native_lines
                server.StrataEngine._pump(engine)
                self.stale_unchanged(before, lr.status(engine)[0])
                self.assertIsNone(engine.lines.get_nowait())
                self.assertTrue(engine.lines.empty())

    def test_runtime_closed_rows_bounds_and_reconciliation_are_atomic(self):
        original = runtime_frame(2, 512).split("\n", 3)[3].split("\nSTRATA-RUNTIME-END", 1)[0]
        mutations = [
            original.replace("setting\tkv\t1", "setting\tkv\t4"),
            original.replace("setting\tauto_cache\t1", "setting\tauto_cache\t2"),
            original.replace("setting\tprefill\t64", "setting\tprefill\t064"),
            original.replace("setting\tprefill\t64", "setting\tprefill\t18446744073709551616"),
            original.replace("setting\tprefill\t64", "setting\tprefill\t+64"),
            original.replace("setting\tprefill\t64", "setting\tprefill\t64\nsetting\tprefill\t64"),
            original.replace("setting\tprefill\t64\n", ""),
            original.replace("setting\tprefill\t64", "setting\tprefill\t64\nsetting\tsecret\t123"),
            original.replace("setting\tprefill\t64", "setting\tprefill\t64\nsetting\tSYNTHETIC-SECRET\t1"),
            original.replace("setting\tprefill\t64", "setting\tprefill\t64\textra"),
            original.replace("setting\tprefill_borrowed\t0", "setting\tprefill_borrowed\t1"),
            original.replace("setting\tparallel\t1", "setting\tparallel\t2"),
            original.replace("component\tvram\texpert_cache\tknown", "component\tvram\texpert_cache\tmaybe"),
            original.replace("component\tvram\texpert_cache", "component\tvram\tunknown_owner"),
            original.replace("component\tvram\texpert_cache", "component\tdevice\texpert_cache"),
            original.replace("known\t512\t2096640\t512", "known\t512\t2096640\t513"),
            original.replace("known\t512\t2096640\t512", "known\t512\t18446744073709551615\t512"),
            original.replace("known\t4096\t0\t3072\t4096\t4096", "known\t4096\t0\t3072\t4097\t4096"),
            original.replace("known\t4096\t0\t3072\t4096\t4096", "known\t4096\t0\t3072\t4096\t4097"),
            original.replace("shard\t0\t512\t496\t16", "shard\t0\t512\t495\t16"),
            original.replace("shards\t1", "shards\t65"),
            original.replace("shard\t0", "shard\t1"),
            original.replace("total_allocated\tunknown", "total_allocated\t0"),
            original.replace("driver_os\tunknown", "driver_os\tknown"),
            original.replace("physical\tunavailable", "physical\t0"),
            original.replace("device\t63756461", "device\t6375646A"),
            original.replace("303030303a30313a30302e30", "756e617661696c61626c65"),
            original + "unrecognized\t1\n", original[:-1],
        ]
        lines = original.splitlines(keepends=True)
        settings = [i for i, line in enumerate(lines) if line.startswith("setting\t")]
        swapped = list(lines)
        swapped[settings[0]], swapped[settings[1]] = swapped[settings[1]], swapped[settings[0]]
        mutations.append("".join(swapped))
        components = [line for line in lines if line.startswith("component\t")]
        mutations += [original + components[-1], original.replace(components[0], components[-1], 1)]
        unknown = runtime_frame(2, unknown=True).split("\n", 3)[3].split("\nSTRATA-RUNTIME-END", 1)[0]
        mutations.append(unknown.replace("unknown\t-\t-\t-\t-\t-\t-", "unknown\t0\t-\t-\t-\t-\t-", 1))
        for payload in mutations:
            with self.subTest(payload=payload):
                engine = self.engine()
                self.consume(engine.records, runtime_frame())
                before = lr.status(engine)[0]
                frame = f"STRATA-RUNTIME/1\n2\n{len(payload)}\n{payload}\nSTRATA-RUNTIME-END/1 2\n"
                self.assertEqual(self.consume(engine.records, frame + "SYNC\n"), "SYNC\n")
                self.stale_unchanged(before, lr.status(engine)[0])
                self.assertEqual(engine.records.runtime_sequence, 1)

    def test_runtime_envelope_sequences_gaps_and_independent_legacy_sequence(self):
        for bad in (runtime_frame(0), runtime_frame(2), runtime_frame(18446744073709551616),
                    runtime_frame().replace("\n1\n", "\n01\n", 1),
                    runtime_frame().replace("STRATA-RUNTIME/1", "STRATA-RUNTIME/2", 1),
                    "STRATA-RUNTIME/1\n1\n16257\n",
                    runtime_frame().replace("STRATA-RUNTIME-END/1 1", "STRATA-RUNTIME-END/1 2")):
            engine = self.engine()
            self.consume(engine.records, bad)
            self.assertIsNone(lr.status(engine)[0]["effective"])
            self.assertEqual(engine.records.runtime_sequence, 0)
            self.assertIsNotNone(engine.records.error)
        engine = self.engine()
        self.consume(engine.records, ownership_frame())
        self.consume(engine.records, runtime_frame())
        self.assertEqual((engine.records.sequence, engine.records.runtime_sequence), (1, 1))
        self.consume(engine.records, runtime_frame(2))  # cache revision with identical byte totals
        self.assertEqual(lr.status(engine)[0]["runtime_memory"]["sequence"], 2)
        self.consume(engine.records, ownership_frame(2, 512))
        self.assertEqual(lr.status(engine)[0]["current_ownership"]["shards"][0]["logical_bytes"], 256)
        self.consume(engine.records, runtime_frame(4, 512))
        launch = lr.status(engine)[0]
        self.assertEqual(launch["current_ownership"]["sequence"], 4)
        self.assertEqual(launch["current_ownership"]["shards"][0]["logical_bytes"], 512)
        self.assertEqual(launch["runtime_memory"]["freshness"], "stale")
        self.assertIn("sequence gap", launch["runtime_memory"]["error"])
        self.consume(engine.records, runtime_frame(5))
        self.assertEqual(lr.status(engine)[0]["runtime_memory"]["freshness"], "stale")
        for bad in (runtime_frame(5), runtime_frame(3), runtime_frame(0)):
            before = lr.status(engine)[0]
            self.consume(engine.records, bad)
            self.stale_unchanged(before, lr.status(engine)[0])
        engine.records = lr.Records()
        self.consume(engine.records, runtime_frame())
        self.consume(engine.records, runtime_frame(lr.U64))
        self.assertEqual(engine.records.runtime_sequence, lr.U64)
        self.consume(engine.records, runtime_frame(0))
        self.assertEqual(engine.records.runtime_sequence, lr.U64)

    def test_runtime_unknown_devices_and_component_overflow(self):
        engine = self.engine()
        self.consume(engine.records, runtime_frame())
        self.consume(engine.records, runtime_frame(2, unknown=True))
        launch = lr.status(engine)[0]
        self.assertEqual(launch["current_ownership"]["freshness"], "unknown")
        self.assertIsNone(launch["current_ownership"]["totals"])
        self.assertIsNone(launch["current_ownership"]["shards"])
        self.assertEqual(launch["current_ownership"]["last_valid_snapshot"]["sequence"], 1)
        self.assertEqual(launch["effective"]["prefill"], 64)
        self.assertIsNone(launch["runtime_memory"]["components"]["vram"]["expert_cache"]["requested_bytes"])
        for device in ("device\t63756461\t0\t756e617661696c61626c65",
                       "device\t686970\t0\t756e617661696c61626c65",
                       "device\t7379636c\t1\t756e617661696c61626c65",
                       "device\t63756461\t1\t303030303a30313a30302e30"):
            records = lr.Records()
            self.consume(records, runtime_frame(unknown=True, device=device))
            self.assertIsNone(records.error)
            self.assertIsNone(records.public()["totals"])
            self.assertIsNone(records.public()["shards"])
        for device in ("device\t6f70656e636c\t0\t756e617661696c61626c65",
                       "device\t63756461\t4294967296\t756e617661696c61626c65",
                       "device\t63756461\t0\t66616b65"):
            records = lr.Records()
            self.consume(records, runtime_frame(unknown=True, device=device))
            self.assertIsNotNone(records.error)
            self.assertIsNone(records.public()["totals"])
        before = lr.status(engine)[0]
        self.consume(engine.records, runtime_frame(3, device="device\t63756461\t0\t303030303a30323a30302e30"))
        self.stale_unchanged(before, lr.status(engine)[0])
        records = lr.Records()
        self.consume(records, runtime_frame(components={"vram": {
            "expert_cache": [lr.U64, 0, 0, 0, 0, 0], "decode_auxiliary": [1, 0, 0, 0, 0, 0]}}))
        self.assertIsNotNone(records.error)
        self.assertIsNone(records.publication()["components"])

    def test_runtime_status_samples_one_publication_even_during_replacement(self):
        first, second = lr.Records(), lr.Records()
        self.consume(first, runtime_frame())
        self.consume(second, runtime_frame(settings={"prefill": 128}, held=512))
        engine = self.engine(first)
        expected = lr.status(engine)
        original = copy.deepcopy
        holdings = first._runtime_published[1]["weights"]
        def replace(value):
            if value is holdings:
                first._runtime_published = second._runtime_published
            return original(value)
        with patch.object(lr.copy, "deepcopy", side_effect=replace):
            self.assertEqual(lr.status(engine), expected)
        self.assertEqual(lr.status(engine)[0]["effective"]["prefill"], 128)

    def test_runtime_component_comparison_bases_unknown_absent_and_extra(self):
        records = lr.Records()
        self.consume(records, runtime_frame())
        estimates = {"vram": {"expert_cache": 2097152, "overhead_allowance": 300, "weights_canonical": 1024},
                     "ram": {"streamed_kv": 3000, "file_cache_target": 512, "host_headroom_allowance": 4096}}
        comparison = lr.component_comparison(estimates, records.publication()["components"], "current")
        cache = comparison["vram"]["expert_cache"]
        self.assertEqual((cache["state"], cache["basis"], cache["actual_comparable_bytes"], cache["delta_bytes"]),
                         ("comparable", "allocator_granule_model", 2097152, 0))
        self.assertEqual(cache["actual"], {"state": "known", "requested_bytes": 512, "rounding_bytes": 2096640,
                                         "rounded_bytes": 2097152, "logical_bytes": 256, "mapped_bytes": 0,
                                         "pinned_bytes": 0, "allowance_bytes": 0})
        self.assertEqual(comparison["vram"]["weights_canonical"]["state"], "absent")
        self.assertIsNone(comparison["vram"]["weights_canonical"]["delta_bytes"])
        self.assertEqual(comparison["vram"]["session_including_resident_kv"]["state"], "extra")
        self.assertIsNone(comparison["vram"]["session_including_resident_kv"]["delta_bytes"])
        self.assertEqual((comparison["vram"]["overhead_allowance"]["basis"],
                          comparison["vram"]["overhead_allowance"]["delta_bytes"]), ("policy_allowance", -44))
        self.assertEqual((comparison["ram"]["streamed_kv"]["basis"], comparison["ram"]["streamed_kv"]["delta_bytes"]),
                         ("logical_payload", 72))
        self.assertEqual(comparison["ram"]["file_cache_target"]["actual_comparable_bytes"], 512)
        self.assertEqual(comparison["ram"]["file_cache_target"]["actual"]["requested_bytes"], 128)
        self.consume(records, runtime_frame(2, unknown=True))
        unknown = lr.component_comparison(estimates, records.publication()["components"], "unknown")
        self.assertEqual(unknown["vram"]["expert_cache"]["state"], "unknown")
        self.assertIsNone(unknown["vram"]["expert_cache"]["delta_bytes"])
        unavailable = lr.component_comparison(estimates, None, "unavailable")
        self.assertEqual(unavailable["ram"]["streamed_kv"]["state"], "unavailable")

    def test_runtime_shard_order_binding_and_unsupported_topology(self):
        disabled = lr.Records()
        self.consume(disabled, runtime_frame(settings={"prefill": 0}))
        self.assertIsNone(disabled.error)
        self.assertEqual(disabled.publication()["effective"]["prefill"], 0)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("payload read")):
            records = lr.Records(argv=["engine", "--serve", "--native", "model-00001-of-00002.gguf"], cwd=str(Path.cwd()))
            self.consume(records, runtime_frame(shards=2))
        self.assertIsNone(records.error)
        self.assertEqual([row["path"] for row in records.public()["shards"]],
                         [str(Path.cwd() / "model-00001-of-00002.gguf"), str(Path.cwd() / "model-00002-of-00002.gguf")])
        before = lr.status(self.engine(records))[0]
        self.consume(records, runtime_frame(2, shards=1))
        self.stale_unchanged(before, lr.status(self.engine(records))[0])
        for argv in (["engine", "--native"], ["engine", "--pack", "pack"],
                     ["engine", "--native", "a", "--native", "b"],
                     ["engine", "--native", "model-00001-of-00065.gguf"]):
            records = lr.Records(argv=argv)
            self.consume(records, runtime_frame())
            self.assertIsNotNone(records.error)
            self.assertIsNone(records.public()["totals"])
        for unknown in (False, True):
            records = lr.Records()
            self.consume(records, runtime_frame(unknown=unknown, settings={"batch": 4, "parallel": 4}))
            self.assertEqual(records.error is None, unknown)
            self.assertIsNone(records.public()["totals"])


class Runtime(unittest.TestCase):
    config = fixtures.Plans.config
    query = fixtures.Plans.query
    plan = fixtures.Plans.plan
    state = fixtures.Plans.state
    rehash = fixtures.Plans.rehash

    def setUp(self):
        fixtures.Plans.setUp(self)
        script = self.root / "runtime.py"
        # Keep the synthetic helper's imports explicit, without site hooks.
        script.write_text("import sys, site\nsys.path.extend(site.getsitepackages())\nsys.path.insert(0," + repr(str(Path(__file__).resolve().parents[1])) +
                          ")\nfrom serve.test_launch_runtime import fake_runtime_main\nfake_runtime_main()\n", encoding="utf-8")
        record = build_provenance.embedded_record(self.exe.read_bytes())
        marker = build_provenance.MARKER + build_provenance.json_bytes(record) + b"\n"
        if os.name == "nt":
            self.exe.write_bytes(b"@echo off\n@rem " + marker +
                                 f'@set "PATH={self.env["PATH"]}"\n'.encode() +
                                 b'@set "FIXTURE_NATIVE_PATH=%PATH%"\n' +
                                 f'"{sys.executable}" -S -B "{script}" %*\n'.encode())
        else:
            self.exe.write_bytes(b"#!/bin/sh\n# " + marker + f'exec "{sys.executable}" -S -B "{script}" "$@"\n'.encode())
            self.exe.chmod(0o700)
        (self.root / "BUILD.json").write_text(json.dumps(build_provenance.metadata_for(self.exe.read_bytes(), self.exe.name)))
        self.saved = self.plan(source_config=self.cfg, config_path=self.config_path)
        self.saved_path = self.root / "approved.json"
        self.saved_path.write_bytes(lp.canonical(self.saved))
        self.hardware = patch.object(lp, "observe_hardware", return_value=copy.deepcopy(fixtures.HARDWARE)).start()
        self.addCleanup(patch.stopall)
        self.original_producer = lp.produce_plan
        self.original_refresh = fixtures.prep.refreshed_config
        self.policy = patch("tools.prepare_unsloth_q2.refreshed_config", side_effect=AssertionError("policy rerun")).start()
        self.producer = patch.object(lp, "produce_plan", side_effect=AssertionError("plan recomputation")).start()
        patch.object(server.Service, "start_telemetry", return_value=None).start()
        self.spawns = []
        real_popen = server.popen
        def capture_spawn(what, argv, **kwargs):
            if what == "the Strata engine":
                self.spawns.append((copy.deepcopy(argv), copy.deepcopy(kwargs.get("env")), kwargs.get("cwd")))
            return real_popen(what, argv, **kwargs)
        patch.object(server, "popen", side_effect=capture_spawn).start()

    def start(self, lazy=False, ordinary=False):
        engine = server.StrataEngine(str(self.exe), self.saved["launch"]["argv"][3:], cwd=str(self.root),
                                     env=lp.replay_environment(self.saved["launch"]["environment"]), lazy=lazy,
                                     launch_plan=None if ordinary else self.saved)
        self.addCleanup(engine.close)
        return engine

    def service(self, engine):
        svc = server.Service(engine, server.ByteTokenizer(), ChatTemplate(server.ROOT / "serve/chat_template.jinja"))
        http = server.serve(svc, "127.0.0.1", 0)
        self.addCleanup(self.close_http, http)
        self.assertEqual(http.server_address[0], "127.0.0.1")
        self.http = http
        return svc, f"http://127.0.0.1:{http.server_address[1]}"

    def close_http(self, http):
        http.shutdown()
        http.server_close()
        self.assertEqual(http.socket.fileno(), -1)

    def get_status(self, base):
        with urllib.request.urlopen(base + "/v1/status", timeout=5) as response:
            return json.load(response)

    def command(self, engine, text):
        engine.proc.stdin.write(text + "\n")
        engine.proc.stdin.flush()
        return engine.lines.get(timeout=5)

    def test_runtime_ordinary_public_http_applied_components_and_process_replacement(self):
        (self.root / "runtime.txt").write_text("enable coherent runtime fixture")
        with patch.dict(os.environ, {"STRATA_API_KEY": "SYNTHETIC-SECRET"}):
            engine = self.start(ordinary=True)
        svc, base = self.service(engine)
        before = self.get_status(base)["launch"]
        self.assertEqual((before["mode"], before["plan_digest"], before["effective_state"]), ("ordinary", None, "observed"))
        self.assertIsNone(before["approved"])
        self.assertIsNone(before["verified_startup"])
        self.assertEqual(before["native_argv"], self.spawns[0][0])
        self.assertEqual({k: before["effective"][k] for k in ("context", "prefill", "kv", "parallel", "cache_slots", "cache_bytes")},
                         {"context": 8192, "prefill": 64, "kv": 1, "parallel": 1, "cache_slots": 2, "cache_bytes": 512})
        self.assertNotIn("verified", before["effective"])
        self.assertEqual(before["runtime_memory"]["protocol"], "STRATA-RUNTIME/1")
        self.assertEqual(before["runtime_memory"]["components"]["ram"]["streamed_kv"]["mapped_bytes"], 4096)
        self.assertEqual(before["current_ownership"]["shards"][0]["path"], str(self.root / "model.gguf"))
        self.assertEqual(self.command(engine, "RUNTIME_UPDATE"), "SYNC RUNTIME_UPDATE\n")
        after = self.get_status(base)["launch"]
        self.assertEqual(after["runtime_memory"]["sequence"], 2)
        self.assertEqual(after["current_ownership"]["shards"][0]["logical_bytes"], 512)
        self.assertEqual(after["runtime_memory"]["components"]["vram"]["expert_cache"]["logical_bytes"], 512)
        self.assertEqual(self.command(engine, "RUNTIME_SAME"), "SYNC RUNTIME_SAME\n")
        self.assertEqual(self.get_status(base)["launch"]["runtime_memory"]["sequence"], 3)
        self.assertEqual(self.command(engine, "RUNTIME_CONTRADICT"), "SYNC RUNTIME_CONTRADICT\n")
        self.assertEqual(self.get_status(base)["launch"]["effective"]["prefill"], 128)
        self.assertEqual(self.get_status(base)["launch"]["effective_state"], "observed")
        self.assertNotIn("SYNTHETIC-SECRET", json.dumps(self.get_status(base)))
        predecessor = engine.records
        old_proc, old_pump = engine.proc, engine.pump
        (self.root / "late.txt").write_text("late runtime publication")
        engine.restart()
        self.assertIsNotNone(old_proc.poll())
        self.assertFalse(old_pump.is_alive())
        self.assertEqual(predecessor.runtime_sequence, 5)
        current = self.get_status(base)["launch"]
        self.consume_runtime(predecessor, runtime_frame(6, 512, settings={"prefill": 256}))
        self.assertEqual(self.get_status(base)["launch"], current)
        self.assertEqual(self.command(engine, "RUNTIME_UNKNOWN"), "SYNC RUNTIME_UNKNOWN\n")
        unknown = self.get_status(base)["launch"]
        self.assertIsNone(unknown["current_ownership"]["totals"])
        self.assertIsNone(unknown["runtime_memory"]["components"]["vram"]["expert_cache"]["requested_bytes"])
        engine.unload()
        unloaded = self.get_status(base)["launch"]
        self.assertIsNone(unloaded["effective"])
        self.assertIsNone(unloaded["runtime_memory"]["components"])
        self.assertFalse(svc.loaded())
        svc.load()
        self.assertEqual(self.get_status(base)["launch"]["runtime_memory"]["sequence"], 1)

    def consume_runtime(self, records, frame):
        stream = io.StringIO(frame)
        self.assertTrue(records.consume(stream.readline(), stream))

    def test_runtime_planned_public_http_strict_startup_and_contradiction_boundary(self):
        (self.root / "runtime.txt").write_text("enable coherent runtime fixture")
        engine = self.start()
        _, base = self.service(engine)
        before = self.get_status(base)["launch"]
        self.assertEqual(before["plan_digest"], self.saved["plan_digest"])
        self.assertEqual({k: before["approved"][k] for k in ("context", "prefill", "cache_slots", "cache_bytes")},
                         {"context": 8192, "prefill": 64, "cache_slots": 2, "cache_bytes": 512})
        self.assertNotIn("slots", before["approved"])
        self.assertNotIn("resident_order", before["approved"])
        self.assertEqual(before["requested"], self.saved["requested"])
        self.assert_effective(before["verified_startup"])
        self.assertEqual(before["effective"]["cache_slots"], 2)
        self.assertNotIn("verified", before["effective"])
        cache = before["runtime_comparison"]["vram"]["expert_cache"]
        self.assertEqual((cache["estimated_bytes"], cache["actual_comparable_bytes"], cache["delta_bytes"]),
                         (512, 2097152, 2096640))
        self.assertEqual(before["runtime_comparison"]["vram"]["weights"]["state"], "absent")
        self.assertEqual(before["runtime_comparison"]["ram"]["streamed_kv"]["state"], "extra")
        self.assertEqual(self.command(engine, "RUNTIME_UPDATE"), "SYNC RUNTIME_UPDATE\n")
        self.assertEqual((engine.records.sequence, engine.records.runtime_sequence), (1, 2))
        before = self.get_status(base)["launch"]
        self.assertEqual(self.command(engine, "RUNTIME_CONTRADICT"), "SYNC RUNTIME_CONTRADICT\n")
        RuntimeCodec.stale_unchanged(self, before, self.get_status(base)["launch"])
        engine.close()
        (self.root / "mode.txt").write_text("bad-runtime")
        with self.startup_handles() as (processes, readers, _), self.assertRaisesRegex(RuntimeError, "invalid_launch_plan"):
            self.start()
        self.assertTrue(processes[0].stdin.closed)
        self.assertTrue(processes[0].stdout.closed)
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(all(not reader.is_alive() for reader in readers))

    def test_runtime_native_cpp_emitter_decoder_and_public_status(self):
        producer = os.environ.get("PAIR_TEST_RUNTIME_CODEC")
        if not producer:
            self.skipTest("PAIR_TEST_RUNTIME_CODEC absent: real native producer is unverified")
        child_env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
        result = subprocess.run([producer, "--runtime-fixture"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=15, env=child_env, cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertEqual(result.stderr, b"")
        self.assertLessEqual(len(result.stdout), 16384)
        text = result.stdout.decode("ascii")
        records = lr.Records()
        stream = io.StringIO(text)
        self.assertTrue(records.consume(stream.readline(), stream))
        self.assertEqual(stream.read(), "")
        self.assertIsNone(records.error)
        self.assertEqual(records.runtime_sequence, 1)
        (self.root / "runtime.txt").write_text("forward exact native producer bytes")
        (self.root / "native-runtime.bin").write_bytes(result.stdout)
        engine = self.start(ordinary=True)
        _, base = self.service(engine)
        launch = self.get_status(base)["launch"]
        self.assertEqual((launch["effective"]["context"], launch["effective"]["cache_slots"], launch["effective"]["cache_bytes"]),
                         (8192, 3, 768))
        self.assertEqual(launch["current_ownership"]["totals"]["capacity_bytes"], 768)
        self.assertEqual(launch["current_ownership"]["shards"][0]["logical_bytes"], 56)
        self.assertEqual(launch["runtime_memory"]["components"]["vram"]["expert_cache"], {
            "state": "known", "requested_bytes": 768, "rounding_bytes": 2096384, "rounded_bytes": 2097152,
            "logical_bytes": 56, "mapped_bytes": 0, "pinned_bytes": 0, "allowance_bytes": 0})
        self.assertEqual(launch["runtime_memory"]["components"]["ram"]["streamed_kv"], {
            "state": "known", "requested_bytes": 75498496, "rounding_bytes": 0, "rounded_bytes": 75498496,
            "logical_bytes": 75497472, "mapped_bytes": 75498496, "pinned_bytes": 75498496, "allowance_bytes": 0})
        self.assertIsNone(launch["runtime_memory"]["total_allocated_bytes"])
        self.assertIsNone(launch["runtime_comparison"]["physical_residency_bytes"])
        print(f"native-runtime-codec executed=True bytes={len(result.stdout)} decoder=True ordinary-public-status=True")

    def test_runtime_process_corruption_unknown_and_gap_public_outcomes(self):
        (self.root / "runtime.txt").write_text("enable coherent runtime fixture")
        for command in ("RUNTIME_BAD", "RUNTIME_TRUNCATED", "RUNTIME_GAP", "RUNTIME_UNKNOWN"):
            with self.subTest(command=command):
                engine = self.start()
                _, base = self.service(engine)
                before = self.get_status(base)["launch"]
                engine.proc.stdin.write(command + "\n")
                engine.proc.stdin.flush()
                if command == "RUNTIME_TRUNCATED":
                    engine.pump.join(5)
                    self.assertFalse(engine.pump.is_alive())
                    self.assertIsNone(engine.lines.get(timeout=5))
                    self.assertTrue(engine.lines.empty())
                else:
                    self.assertEqual(engine.lines.get(timeout=5), "SYNC " + command + "\n")
                after = self.get_status(base)["launch"]
                if command in ("RUNTIME_BAD", "RUNTIME_TRUNCATED"):
                    RuntimeCodec.stale_unchanged(self, before, after)
                elif command == "RUNTIME_GAP":
                    self.assertEqual(after["runtime_memory"]["sequence"], 3)
                    self.assertEqual(after["current_ownership"]["shards"][0]["logical_bytes"], 512)
                    self.assertEqual(after["runtime_memory"]["components"]["vram"]["expert_cache"]["logical_bytes"], 512)
                    self.assertEqual(after["runtime_memory"]["freshness"], "stale")
                    self.assertIn("sequence gap", after["runtime_memory"]["error"])
                else:
                    self.assertEqual(after["runtime_memory"]["sequence"], 2)
                    self.assertEqual(after["current_ownership"]["freshness"], "unknown")
                    self.assertIsNone(after["current_ownership"]["totals"])
                    self.assertIsNone(after["current_ownership"]["shards"])
                    self.assertIsNone(after["runtime_memory"]["components"]["vram"]["expert_cache"]["requested_bytes"])
                    self.assertEqual(after["current_ownership"]["last_valid_snapshot"]["sequence"], 1)
                    self.assertEqual(engine.records.sequence, 1)
                engine.close()
                self.http.shutdown()
                self.http.server_close()

    def captures(self):
        return [json.loads(line) for line in (self.root / "capture.jsonl").read_text().splitlines()]

    def assert_effective(self, actual):
        self.assertEqual((actual["schema"], actual["verified"], actual["context"], actual["parallel"],
                          actual["cache_slots"], actual["cache_bytes"], actual["prefill"]), (1, 1, 8192, 1, 2, 512, 64))
        self.assertEqual({k: actual[k] for k in lp.SCALARS}, {k: self.saved["resolved"][k] for k in lp.SCALARS})
        self.assertNotIn("slots", actual)
        self.assertNotIn("resident_order", actual)

    def replace_plan(self, cfg, query=None):
        self.config_path.write_bytes(lp.canonical(cfg))
        with patch.object(fixtures.prep, "refreshed_config", self.original_refresh):
            self.saved = self.original_producer(cfg, hardware=fixtures.HARDWARE, inherited=self.env,
                                               query=query or self.query, source_config=cfg, config_path=self.config_path)
        self.saved_path.write_bytes(lp.canonical(self.saved))

    def test_documented_first_use_cli_http_and_cleanup(self):
        """PAIR_INTEGRATION.md sequence, with synthetic hardware/artifacts only."""
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        cfg = copy.deepcopy(self.cfg)
        cfg.update(host="127.0.0.1", port=port, lazy_load=True, aliases=["approved-alias"])
        tokenizer = Path(cfg["tokenizer"])
        tokenizer.mkdir()
        for name, value in (("vocab.json", "{}"), ("merges.txt", ""), ("token_type.json", "[]")):
            (tokenizer / name).write_text(value)
        self.config_path.write_bytes(lp.canonical(cfg))
        before = self.state()
        outputs = []
        for mode in ("--print-launch-plan", "--print-launch-plan", "--estimate-memory"):
            output = io.StringIO()
            with patch.dict(os.environ, self.env, clear=True), \
                    patch.object(sys, "argv", ["server", "--engine", "strata", "--config",
                                               str(self.config_path), "--port", str(port), mode]), \
                    patch.object(lp, "produce_plan", self.original_producer), \
                    patch.object(fixtures.prep, "refreshed_config", self.original_refresh), \
                    patch.object(server, "Server", side_effect=AssertionError("dry-run bind")), \
                    patch.object(server, "Service", side_effect=AssertionError("dry-run service")), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(server.main(), 0)
            outputs.append(output.getvalue())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(self.state(), before)
        self.assertEqual(self.spawns, [])
        self.saved = json.loads(outputs[0])
        independent = json.dumps({k: v for k, v in self.saved.items() if k != "plan_digest"},
                                 sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                 allow_nan=False).encode("ascii")
        self.assertEqual(self.saved["schema"], 1)
        self.assertEqual(self.saved["plan_digest"], hashlib.sha256(independent).hexdigest())
        estimate = json.loads(outputs[2])
        self.assertEqual(estimate["kind"], "strata_memory_estimate")
        self.assertEqual(estimate["plan_digest"], self.saved["plan_digest"])
        self.assertEqual(estimate["estimates"], self.saved["estimates"])
        self.saved_path.write_bytes(outputs[0].encode("ascii"))
        guide = (server.ROOT / "docs/PAIR_INTEGRATION.md").read_text(encoding="utf-8")
        recipe = guide.split("@'\n", 1)[1].split("\n'@ | python -B -", 1)[0]
        verified = subprocess.run([sys.executable, "-B", "-", str(self.saved_path)], input=recipe,
                                  text=True, capture_output=True, timeout=10)
        self.assertEqual(verified.returncode, 0, verified.stderr)
        self.assertEqual(verified.stdout.strip(), self.saved["plan_digest"])
        real_serve, services, processes, readers, servers = server.serve, [], [], [], []

        def ready(svc, host, port):
            self.assertEqual((host, port), ("127.0.0.1", self.saved["server"]["port"]))
            services.append(svc)
            self.addCleanup(svc.engine.close)
            http = real_serve(svc, host, port)
            servers.append(http)
            self.addCleanup(self.close_http, http)
            base = f"http://127.0.0.1:{port}"

            def request(path, body=None):
                req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    return json.load(response)

            self.assertEqual(request("/v1/models")["data"][0]["status"], {"value": "unloaded"})
            self.assertIsNone(svc.engine.proc)
            loaded = request("/v1/load", {"model": "approved-alias"})
            self.assertEqual((loaded["status"], loaded["lifecycle"], loaded["loaded"]), ("loaded", "loaded", True))
            processes.append(svc.engine.proc)
            readers.append(svc.engine.pump)
            current = request("/v1/status")
            self.assertEqual(current["launch"]["plan_digest"], self.saved["plan_digest"])
            self.assertEqual(current["launch"]["native_argv"], self.saved["launch"]["argv"])
            self.assert_effective(current["launch"]["effective"])
            self.assertEqual(current["launch"]["current_ownership"]["shards"][0]["logical_bytes"], 256)
            self.assertEqual(current["engine_build"], {"status": "current", "reason": None,
                             "commit": "synthetic", "dirty": True, "built_at": "2026-01-01T00:00:00Z",
                             "cuda": "synthetic", "version": "test", "backend": "cuda"})
            self.assertEqual([m["id"] for m in request("/v1/models")["data"]], ["synthetic", "approved-alias"])
            answer = request("/v1/chat/completions", {"model": "approved-alias", "messages": [
                             {"role": "user", "content": "Hello"}], "max_tokens": 32, "reasoning_effort": "none"})
            self.assertEqual(answer["choices"][0]["message"]["content"], "Hello.")
            self.assertEqual(answer["model"], "approved-alias")
            unloaded = request("/v1/unload", {})
            self.assertEqual((unloaded["status"], unloaded["loaded"], unloaded["lifecycle"]),
                             ("unloaded", False, "unloaded"))
            self.assertEqual(request("/v1/unload", {})["status"], "not loaded")
            self.assertEqual(request("/v1/models")["data"][0]["status"], {"value": "unloaded"})
            self.assertIsNone(unloaded["launch"]["effective"])
            self.assertEqual(unloaded["engine_build"]["status"], "unavailable")
            return http

        with patch.dict(os.environ, self.env, clear=True), \
                patch.object(sys, "argv", ["server", "--launch-plan", str(self.saved_path)]), \
                patch.dict(sys.modules, {"strata_tokenizer": SimpleNamespace(Tokenizer=lambda *a: server.ByteTokenizer())}), \
                patch.object(server, "serve", side_effect=ready), \
                patch.object(server, "engine_args", side_effect=AssertionError("recomposed argv")), \
                patch.object(server.signal, "signal"), \
                patch.object(server.time, "sleep", side_effect=KeyboardInterrupt), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(server.main(), 0)
        self.assertEqual(len(services), 1)
        self.assertEqual(len(self.spawns), 1)
        capture = self.captures()[0]
        self.assertEqual(capture["argv"], self.saved["launch"]["argv"][1:])
        self.assertEqual(capture["cwd"], self.saved["launch"]["cwd"])
        self.assertEqual(capture["environment"], self.saved["launch"]["environment"]["set"])
        self.assertFalse(capture["credential_present"])
        self.assertEqual(base64.b64decode(capture["frame"]), lp.strict_frame(self.saved))
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(processes[0].stdin.closed and processes[0].stdout.closed)
        self.assertFalse(readers[0].is_alive())
        self.assertEqual(servers[0].socket.fileno(), -1)
        with socket.socket() as released:
            released.bind(("127.0.0.1", port))
        # A now-impossible saved plan refuses before any bind or engine spawn.
        self.hardware.return_value["ram_available_bytes"] = 1
        self.assertEqual(self.main_refusal([])["code"], "insufficient_resources")
        self.assertEqual(len(self.spawns), 1)
        print("first-use-fixture schema=1 repeatable=True documented_digest=True exact_replay=True "
              "models/load/status/build/chat/unload=200 impossible_plan=insufficient_resources "
              "child_exited=True pipes_closed=True reader_stopped=True listener_released=True")

    def test_public_main_eager_lazy_frozen_settings_and_original_credential_source(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                tokenizer = self.root / "tokenizer"
                tokenizer.mkdir(exist_ok=True)
                for name, value in (("vocab.json", "{}"), ("merges.txt", ""), ("token_type.json", "[]")):
                    (tokenizer / name).write_text(value)
                cfg = copy.deepcopy(self.cfg)
                cfg.update(host="127.0.0.1", port=port, lazy_load=lazy, fit_max_tokens=True,
                           aliases=["approved-alias"], tokenizer=str(tokenizer),
                           slot_save_path="sessions", sampling={"top_p": 0.9},
                           api_key_file=str(self.root / "SYNTHETIC-KEY-FILE"))
                (self.root / "SYNTHETIC-KEY-FILE").write_text("SYNTHETIC-FILE-KEY\n")
                self.replace_plan(cfg, lp.bounded_query)
                original_files = {p: p.read_bytes() for p in (self.config_path, self.saved_path, self.root / "SYNTHETIC-KEY-FILE")}
                real_serve = server.serve
                services = []
                def ready(svc, host, port):
                    self.addCleanup(svc.engine.close)
                    self.assertEqual((host, port), ("127.0.0.1", self.saved["server"]["port"]))
                    self.assertEqual(svc.api_key, ["SYNTHETIC-FILE-KEY"])
                    self.assertEqual(svc.slot_save_path, str(self.root / "sessions"))
                    self.assertTrue(svc.fit_max_tokens)
                    self.assertEqual(svc.aliases, ["approved-alias"])
                    self.assertEqual(svc.sampling_defaults, {"top_p": 0.9})
                    self.assertEqual(svc.loaded(), not lazy)
                    if lazy:
                        svc.load()
                    http = real_serve(svc, host, port)
                    self.addCleanup(self.close_http, http)
                    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/status",
                                                 headers={"Authorization": "Bearer SYNTHETIC-FILE-KEY"})
                    with urllib.request.urlopen(req, timeout=5) as response:
                        public = json.load(response)
                    self.assert_effective(public["launch"]["effective"])
                    self.assertNotIn("SYNTHETIC-FILE-KEY", json.dumps(public))
                    self.assertNotIn("SYNTHETIC-KEY-FILE", json.dumps(public))
                    services.append(svc)
                    return http
                output = io.StringIO()
                with patch.dict(os.environ, self.env, clear=True), \
                        patch.object(sys, "argv", ["server", "--launch-plan", str(self.saved_path)]), \
                        patch.dict(sys.modules, {"strata_tokenizer": SimpleNamespace(Tokenizer=lambda *a: server.ByteTokenizer())}), \
                        patch("serve.credentials.read_key_file", return_value=["SYNTHETIC-FILE-KEY"]) as read_key, \
                        patch.object(server, "serve", side_effect=ready), \
                        patch.object(server, "engine_args", side_effect=AssertionError("recomposed argv")), \
                        patch.object(server, "effort_end_args", side_effect=AssertionError("recomputed effort")), \
                        patch.object(server.signal, "signal"), \
                        patch.object(server.time, "sleep", side_effect=KeyboardInterrupt), contextlib.redirect_stdout(output):
                    self.assertEqual(server.main(), 0)
                read_key.assert_called_once_with(str(self.root / "SYNTHETIC-KEY-FILE"))
                self.assertEqual(len(services), 1)
                self.assertIsNone(services[0].engine.proc)
                self.assertFalse(services[0].engine.pump.is_alive())
                self.assertNotIn("SYNTHETIC-FILE-KEY", output.getvalue())
                self.assertEqual({p: p.read_bytes() for p in original_files}, original_files)
                capture = self.captures()[-1]
                self.assertEqual(capture["argv"], self.saved["launch"]["argv"][1:])
                self.assertEqual(base64.b64decode(capture["frame"]), lp.strict_frame(self.saved))

    def test_same_count_distinct_ordered_layouts_arrive_unchanged(self):
        payloads = []
        for reverse in (False, True):
            self.replace_plan(self.cfg, lambda argv, cwd, env: json.dumps(fixtures.fake_metadata(argv, env, reverse=reverse)).encode())
            engine = self.start()
            payloads.append(base64.b64decode(self.captures()[-1]["frame"]))
            self.assertEqual(payloads[-1], lp.strict_frame(self.saved))
            self.assertEqual(engine.records.effective["cache_slots"], 2)
            decoded = base64.b64decode(self.saved["native"]["native_payload_base64"])
            self.assertIn(b"slots=1,2,256,0,0,1,256,256\n" if reverse else b"slots=0,1,256,0,1,2,256,256\n", decoded)
            engine.close()
        self.assertNotEqual(payloads[0], payloads[1])

    def test_status_main_shard_order_excludes_additional_role_input_inventory(self):
        cfg = copy.deepcopy(self.cfg)
        first = self.root / "model-00001-of-00002.gguf"
        second = self.root / "model-00002-of-00002.gguf"
        role = self.root / "role-other.gguf"
        data = (self.root / "model.gguf").read_bytes()
        for path in (first, second, role):
            path.write_bytes(data)
        (self.root / "native_experts.txt").write_text("0 0 0 0 0 0 0 0 role-other.gguf\n")
        cfg["args"][cfg["args"].index("--native") + 1] = str(first)
        (self.root / "shards.txt").write_text("2")
        self.replace_plan(cfg)
        self.assertIn(str(role), [entry["path"] for entry in self.saved["inputs"]["files"]])
        engine = self.start()
        _, base = self.service(engine)
        shards = self.get_status(base)["launch"]["current_ownership"]["shards"]
        self.assertEqual([(s["index"], s["path"], s["logical_bytes"]) for s in shards],
                         [(0, str(first), 256), (1, str(second), 0)])

    def test_original_auto_prefill_and_explicit_cache_request_remain_distinct_from_applied_settings(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.pop("memory_policy")
        fixtures.prep.set_flag(cfg["args"], "--prefill", "auto")
        fixtures.prep.set_flag(cfg["args"], "--expert-cache", "97")
        self.replace_plan(cfg)
        engine = self.start()
        _, base = self.service(engine)
        launch = self.get_status(base)["launch"]
        self.assertEqual((launch["effective"]["auto_cache"], launch["effective"]["cache_request"],
                          launch["effective"]["cache_slots"], launch["effective"]["prefill"]), (0, 97, 2, 2048))
        requested = launch["requested"]["args"]
        actual = self.captures()[0]["argv"]
        self.assertEqual(requested[requested.index("--prefill") + 1], "auto")
        self.assertEqual(actual[actual.index("--prefill") + 1], "2048")
        self.assertEqual(actual[actual.index("--expert-cache") + 1], "97")
        self.assertEqual(base64.b64decode(self.captures()[0]["frame"]), lp.strict_frame(self.saved))

    def test_exact_saved_replay_http_dynamic_status_and_windows_text_framing(self):
        (self.root / "mode.txt").write_text("crlf")
        with patch.dict(os.environ, {"STRATA_FILE_CACHE_MIB": "999", "STRATA_UNKNOWN_CONTROL": "changed",
                                     "CUDA_VISIBLE_DEVICES": "7", "STRATA_API_KEY": "SYNTHETIC-SECRET"}):
            engine = self.start()
        svc, base = self.service(engine)
        before = self.get_status(base)
        self.assertEqual(before["launch"]["plan_digest"], self.saved["plan_digest"])
        self.assertEqual(before["launch"]["native_argv"], self.saved["launch"]["argv"])
        self.assert_effective(before["launch"]["effective"])
        self.assertEqual(before["engine_build"], {"status": "current", "reason": None, "commit": "synthetic", "dirty": True,
                                                "built_at": "2026-01-01T00:00:00Z", "cuda": "synthetic", "version": "test", "backend": "cuda"})
        self.assertEqual(before["launch"]["current_ownership"]["totals"], {
            "capacity_bytes": 1024, "logical_bytes": 288, "padding_bytes": 544, "scratch_bytes": 64,
            "shared_bytes": 128, "unattributed_bytes": 32, "mapped_host_bytes": 4096})
        self.assertEqual(self.command(engine, "UPDATE"), "SYNC UPDATE\n")
        after = self.get_status(base)["launch"]["current_ownership"]
        self.assertEqual(after["sequence"], 2)
        self.assertEqual(after["shards"], [{"index": 0, "logical_bytes": 512, "unique_source_bytes": 496,
                                          "duplicate_copy_bytes": 16, "path": str(self.root / "model.gguf")}])
        self.assertEqual(after["device"], {"backend": "cuda", "ordinal": 0, "identity": "0000:01:00.0", "identity_kind": "cuda_pci_bus_id"})
        capture = self.captures()[0]
        self.assertEqual(capture["argv"], self.saved["launch"]["argv"][1:])
        self.assertEqual(capture["cwd"], str(self.root))
        self.assertEqual(base64.b64decode(capture["frame"]), lp.strict_frame(self.saved))
        self.assertEqual(capture["environment"], self.saved["launch"]["environment"]["set"])
        self.assertEqual(self.spawns[0][0], self.saved["launch"]["argv"])
        self.assertEqual({k: v for k, v in self.spawns[0][1].items() if lp._relevant(k)},
                         self.saved["launch"]["environment"]["set"])
        self.assertFalse(capture["credential_present"])
        self.assertNotIn("SYNTHETIC-SECRET", json.dumps(after))
        self.assertIsNone(before["launch"]["runtime_comparison"]["total_allocated_bytes"])
        self.assertIsNone(after["physical_residency_bytes"])
        engine.unload()
        unloaded = self.get_status(base)
        self.assertIsNone(unloaded["launch"]["effective"])
        self.assertIsNone(unloaded["launch"]["current_ownership"]["totals"])
        self.assertEqual(unloaded["engine_build"]["status"], "unavailable")
        self.assertFalse(svc.loaded())
        self.assertFalse(engine.pump.is_alive())
        self.assertIsNone(engine.proc)

    def test_lazy_restart_resource_loss_and_no_stale_predecessor_publication(self):
        engine = self.start(lazy=True)
        svc, base = self.service(engine)
        self.assertFalse(svc.loaded())
        self.hardware.return_value["vram_free_bytes"] = 1
        with self.assertRaisesRegex(lp.PlanError, "insufficient_resources"):
            svc.load()
        self.assertFalse((self.root / "capture.jsonl").exists())
        self.assertIsNone(self.get_status(base)["launch"]["effective"])
        self.hardware.return_value = copy.deepcopy(fixtures.HARDWARE)
        svc.load()
        predecessor = engine.records
        old_proc, old_pump = engine.proc, engine.pump
        (self.root / "late.txt").write_text("publish during owned shutdown")
        engine.restart()
        self.assertIsNotNone(old_proc.poll())
        self.assertFalse(old_pump.is_alive())
        self.assertEqual(predecessor.sequence, 2)
        late = ownership_frame(3, 640)
        stream = io.StringIO(late)
        predecessor.consume(stream.readline(), stream)
        self.assertEqual(self.get_status(base)["launch"]["current_ownership"]["shards"][0]["logical_bytes"], 256)
        self.hardware.return_value["ram_available_bytes"] = 1
        with self.assertRaisesRegex(lp.PlanError, "insufficient_resources"):
            engine.restart()
        self.assertEqual(len(self.captures()), 2)
        self.assertIsNone(self.get_status(base)["launch"]["effective"])
        self.assertIsNone(self.get_status(base)["launch"]["current_ownership"]["totals"])
        self.assertIsNone(engine.proc)
        for capture in self.captures():
            self.assertEqual(capture["argv"], self.saved["launch"]["argv"][1:])
            self.assertEqual(base64.b64decode(capture["frame"]), lp.strict_frame(self.saved))

    def test_startup_missing_mismatched_records_refuse_and_cleanup(self):
        real = server.popen
        for mode in ("no-build", "bad-build", "late-build", "no-effective", "bad-effective", "no-ownership", "bad-ready", "bad-info", "bad-parallel"):
            with self.subTest(mode=mode):
                (self.root / "mode.txt").write_text(mode)
                processes = []
                def spawn(*args, **kwargs):
                    proc = real(*args, **kwargs)
                    processes.append(proc)
                    return proc
                with patch.object(server, "popen", side_effect=spawn), self.assertRaisesRegex(RuntimeError, "invalid_launch_plan"):
                    self.start()
                self.assertEqual(len(processes), 1)
                proc = processes[0]
                self.assertIsNotNone(proc.poll())
                self.assertTrue(proc.stdin.closed)
                self.assertTrue(proc.stdout.closed)

    @contextlib.contextmanager
    def startup_handles(self):
        processes, readers, logs = [], [], []
        real_popen, real_thread, real_open = server.popen, server.threading.Thread, open
        def spawn(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            processes.append(proc)
            # Test cleanup is registered before assertions, including on the
            # broken implementation whose constructor never returns its owner.
            self.addCleanup(server.StrataEngine.close, SimpleNamespace(proc=proc, pump=None, log=None))
            return proc
        def thread(*args, **kwargs):
            result = real_thread(*args, **kwargs)
            if kwargs.get("target") is server.StrataEngine._ready_pump:
                readers.append(result)
            return result
        def opened(*args, **kwargs):
            result = real_open(*args, **kwargs)
            if len(args) > 1 and args[1] == "a":
                logs.append(result)
                self.addCleanup(result.close)
            return result
        with patch.object(server, "popen", side_effect=spawn), \
                patch.object(server.threading, "Thread", side_effect=thread), \
                patch.object(server, "open", side_effect=opened, create=True):
            yield processes, readers, logs

    def test_public_oversized_info_failure_closes_owned_child_and_reader(self):
        (self.root / "mode.txt").write_text("short-extra-info")
        control = self.start()
        self.assertEqual(control.info["vram_free_mib"], 123)
        control.close()
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        for name, value in (("vocab.json", "{}"), ("merges.txt", ""), ("token_type.json", "[]")):
            (tokenizer / name).write_text(value)
        self.replace_plan({**self.cfg, "tokenizer": str(tokenizer)})
        (self.root / "mode.txt").write_text("oversized-info")
        output = io.StringIO()
        with self.startup_handles() as (processes, readers, _), \
                patch.object(sys, "argv", ["server", "--launch-plan", str(self.saved_path)]), \
                patch.object(server, "Server", return_value=SimpleNamespace(server_close=lambda: None)), \
                patch.object(server, "Service", side_effect=AssertionError("service started after refusal")), \
                patch.dict(sys.modules, {"strata_tokenizer": SimpleNamespace(Tokenizer=lambda *a: server.ByteTokenizer())}), \
                contextlib.redirect_stdout(output):
            self.assertEqual(server.main(), 2)
        self.assertEqual(json.loads(output.getvalue().splitlines()[-1])["error"]["code"], "invalid_launch_plan")
        self.assertEqual(len(processes), 1)
        self.assertEqual(len(readers), 1)
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(processes[0].stdin.closed)
        self.assertTrue(processes[0].stdout.closed)
        self.assertFalse(readers[0].is_alive())

    def test_readiness_exception_closes_owned_child_reader_and_log(self):
        from serve.launch_runtime import Records
        with self.startup_handles() as (processes, readers, logs), \
                patch.object(Records, "ready", side_effect=RuntimeError("injected readiness failure")), \
                patch.object(server, "narrate_start"), self.assertRaisesRegex(RuntimeError, "injected readiness failure"):
            server.StrataEngine(str(self.exe), [], log=str(self.root / "startup.log"), launch_plan=self.saved)
        self.assertEqual(len(processes), 1)
        self.assertEqual(len(readers), 1)
        self.assertEqual(len(logs), 1)
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(processes[0].stdin.closed)
        self.assertTrue(processes[0].stdout.closed)
        self.assertFalse(readers[0].is_alive())
        self.assertTrue(logs[0].closed)

    def test_ordinary_restart_retries_after_constructor_cleanup(self):
        for transient in (False, True):
            with self.subTest(transient=transient):
                (self.root / "mode.txt").write_text("ok")
                engine = self.start(ordinary=True)
                predecessor, old_reader = engine.proc, engine.pump
                with self.startup_handles() as (processes, readers, _):
                    spawn = server.popen
                    attempts = []
                    def next_process(*args, **kwargs):
                        (self.root / "mode.txt").write_text("exit-before-ready" if transient and not attempts else "ok")
                        attempts.append(args)
                        return spawn(*args, **kwargs)
                    with patch.object(server, "popen", side_effect=next_process), \
                            patch.object(server.time, "sleep") as delay:
                        engine.restart(tries=2)
                self.assertEqual(len(processes), 2 if transient else 1)
                self.assertEqual(engine.max_context, 8192)
                self.assertTrue(engine.alive())
                self.assertEqual(delay.call_count, int(transient))
                self.assertIsNotNone(predecessor.poll())
                self.assertTrue(predecessor.stdin.closed)
                self.assertTrue(predecessor.stdout.closed)
                self.assertFalse(old_reader.is_alive())
                if transient:
                    self.assertIsNotNone(processes[0].poll())
                    self.assertTrue(processes[0].stdin.closed)
                    self.assertTrue(processes[0].stdout.closed)
                self.assertTrue(all(not reader.is_alive() for reader in readers))
                engine.close()

    def test_ordinary_restart_exhaustion_preserves_startup_exception(self):
        engine = self.start(ordinary=True)
        (self.root / "mode.txt").write_text("exit-before-ready")
        with self.startup_handles() as (processes, readers, _), \
                patch.object(server.time, "sleep") as delay, \
                self.assertRaisesRegex(RuntimeError, "the engine exited before it was ready") as failure:
            engine.restart(tries=1)
        self.assertIs(type(failure.exception), RuntimeError)
        self.assertEqual(len(processes), 1)
        self.assertIsNone(engine.proc)
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(processes[0].stdin.closed)
        self.assertTrue(processes[0].stdout.closed)
        self.assertTrue(all(not reader.is_alive() for reader in readers))
        delay.assert_not_called()

    def test_planned_restart_failure_does_not_retry(self):
        engine = self.start()
        (self.root / "mode.txt").write_text("exit-before-ready")
        with self.startup_handles() as (processes, readers, _), \
                patch.object(server.time, "sleep") as delay, \
                self.assertRaisesRegex(RuntimeError, "invalid_launch_plan"):
            engine.restart(tries=3)
        self.assertEqual(len(processes), 1)
        self.assertIsNone(engine.proc)
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(processes[0].stdin.closed)
        self.assertTrue(processes[0].stdout.closed)
        self.assertTrue(all(not reader.is_alive() for reader in readers))
        delay.assert_not_called()

    def test_restart_does_not_retry_uncertain_startup_cleanup(self):
        from serve import launch_runtime
        class Process:
            def __init__(self, text, mode="exit"):
                self.stdout, self.stdin = io.StringIO(text), io.StringIO()
                self.mode, self.returncode = mode, None
                self.waits, self.exit_pending = 0, False
            def poll(self):
                if self.mode == "poll-error":
                    raise PermissionError("synthetic exit query denied")
                if self.exit_pending:
                    self.returncode = 0
                return self.returncode
            def wait(self, timeout=None):
                self.waits += 1
                if self.mode == "wait-error":
                    raise PermissionError("synthetic wait denied")
                if self.returncode is not None:
                    return self.returncode
                if self.mode != "exit":
                    if self.mode == "exit-before-final-poll" and self.waits == 4:
                        self.exit_pending = True
                    raise subprocess.TimeoutExpired("synthetic", timeout)
                self.returncode = 0
                return self.returncode
            def terminate(self):
                if self.mode == "exit-on-denial":
                    self.returncode = -1
                if self.mode in ("denied", "exit-on-denial"):
                    raise PermissionError("synthetic terminate denied")
                if self.mode == "exit":
                    self.returncode = -1
            def kill(self):
                if self.mode == "exit-on-kill-error":
                    self.exit_pending = True
                    raise PermissionError("synthetic kill denied")
                if self.mode == "kill-denied":
                    raise PermissionError("synthetic kill denied")
                if self.mode == "exit":
                    self.returncode = -9
        confirmed_modes = ("exit", "exit-on-denial", "exit-before-final-poll", "exit-on-kill-error")
        for mode in (*confirmed_modes, "denied", "timeout", "kill-denied", "wait-error", "poll-error"):
            for tries in (1, 2):
                with self.subTest(mode=mode, tries=tries):
                    processes = [Process("READY 4096 stop\n"), Process("", mode), Process("READY 4096 stop\n")]
                    spawns = []
                    def spawn(*args, **kwargs):
                        proc = processes[len(spawns)]
                        spawns.append(proc)
                        return proc
                    uncertain = mode not in confirmed_modes
                    expected = server.EngineStuck if uncertain else (RuntimeError if tries == 1 else None)
                    with patch.object(server, "popen", side_effect=spawn), \
                            patch.object(server, "contain"), \
                            patch.object(launch_runtime, "build_record", return_value=None), \
                            patch.object(server.time, "sleep") as delay:
                        engine = server.StrataEngine("synthetic", [])
                        cv, ctl = engine.slot_cv, engine.ctl
                        try:
                            with patch.object(cv, "notify_all", wraps=cv.notify_all) as notify, \
                                    (self.assertRaises(expected) if expected else contextlib.nullcontext()) as failure:
                                engine.restart(tries=tries)
                            if expected:
                                self.assertIs(type(failure.exception), expected)
                            self.assertEqual(notify.call_count, 1)
                            self.assertIs(engine.slot_cv, cv)
                            self.assertIs(engine.ctl, ctl)
                            self.assertFalse(engine.starting)
                            failed = processes[1]
                            if uncertain:
                                self.assertEqual(len(spawns), 2)
                                self.assertIs(engine.proc, failed)
                                self.assertIsNone(failed.returncode)
                                self.assertFalse(failed.stdout.closed)
                                delay.assert_not_called()
                            else:
                                self.assertEqual(len(spawns), 2 if tries == 1 else 3)
                                self.assertIsNotNone(failed.poll())
                                self.assertTrue(failed.stdin.closed)
                                self.assertTrue(failed.stdout.closed)
                                self.assertEqual(delay.call_count, int(tries > 1))
                                if tries > 1:
                                    self.assertEqual(engine.max_context, 4096)
                        finally:
                            # These are in-memory process specimens, restored so
                            # the real owner can exercise confirmed-exit cleanup.
                            for proc in processes:
                                proc.mode = "exit"
                            engine.close()
                            for proc in spawns:
                                proc.stdin.close()
                                proc.stdout.close()

    def test_runtime_malformed_truncated_and_sequence_keep_last_snapshot_out_of_token_queue(self):
        for command in ("BAD", "SEQUENCE", "TRUNCATED", "OVERFLOW", "LENGTH", "BADEND", "VERSION", "UTF8"):
            with self.subTest(command=command):
                engine = self.start()
                _, base = self.service(engine)
                engine.proc.stdin.write(command + "\n")
                engine.proc.stdin.flush()
                # A complete rejected envelope leaves later command traffic
                # readable. An incomplete envelope quarantines the remaining bytes.
                if command != "TRUNCATED":
                    engine.proc.stdin.write("QUIT\n")
                    engine.proc.stdin.flush()
                engine.pump.join(timeout=5)
                self.assertFalse(engine.pump.is_alive())
                if command not in ("TRUNCATED", "LENGTH", "BADEND", "VERSION", "UTF8"):
                    self.assertEqual(engine.lines.get(timeout=5), "SYNC " + command + "\n")
                self.assertIsNone(engine.lines.get(timeout=5))
                self.assertTrue(engine.lines.empty())
                current = self.get_status(base)["launch"]["current_ownership"]
                self.assertEqual(current["freshness"], "stale")
                self.assertIsNotNone(current["error"])
                self.assertEqual(current["shards"][0]["logical_bytes"], 256)
                self.assertEqual(current["sequence"], 1)
                engine.close()
                self.http.shutdown()
                self.http.server_close()

    def test_unknown_is_not_successful_numeric_zero(self):
        engine = self.start()
        _, base = self.service(engine)
        self.assertEqual(self.command(engine, "UNKNOWN"), "SYNC UNKNOWN\n")
        current = self.get_status(base)["launch"]["current_ownership"]
        self.assertEqual(current["state"], "unknown")
        self.assertIsNone(current["totals"])
        self.assertIsNone(current["shards"])
        self.assertEqual(current["freshness"], "unknown")
        self.assertEqual(current["last_valid_snapshot"]["shards"][0]["logical_bytes"], 256)
        engine.close()
        (self.root / "mode.txt").write_text("unknown-start")
        engine = self.start()
        _, base = self.service(engine)
        status = self.get_status(base)
        self.assertTrue(status["loaded"])
        self.assertIsNone(status["launch"]["current_ownership"]["totals"])
        self.assertEqual(status["launch"]["current_ownership"]["state"], "unknown")

    def test_ordinary_native_build_and_legacy_info(self):
        engine = self.start(ordinary=True)
        _, base = self.service(engine)
        status = self.get_status(base)
        self.assertEqual(engine.info["version"], "test")
        self.assertEqual(status["engine_build"]["commit"], "synthetic")
        self.assertIsNone(status["launch"]["plan_digest"])
        self.assertEqual(self.captures()[0]["argv"], ["--serve", *self.saved["launch"]["argv"][3:]])
        self.assertEqual(self.captures()[0]["frame"], "")
        engine.close()
        (self.root / "mode.txt").write_text("no-build")
        engine = self.start(ordinary=True)
        _, base = self.service(engine)
        self.assertEqual(self.get_status(base)["engine_build"]["status"], "unavailable")
        self.assertEqual(engine.info["version"], "test")

    def main_refusal(self, argv):
        before = self.state()
        output = io.StringIO()
        with patch.object(sys, "argv", ["server", "--launch-plan", str(self.saved_path), *argv]), \
                patch.object(server, "Server", side_effect=AssertionError("bind")), \
                patch.object(server, "Service", side_effect=AssertionError("service")), \
                patch.object(server, "StrataEngine", side_effect=AssertionError("spawn")), \
                contextlib.redirect_stdout(output):
            try:
                code = server.main()
            except SystemExit as error:
                code = error.code
        self.assertEqual(code, 2)
        error = json.loads(output.getvalue())["error"]
        self.assertIn(error["code"], ("invalid_launch_plan", "insufficient_resources"))
        self.assertNotIn("SECRET", output.getvalue())
        self.assertEqual(self.state(), before)
        return error

    def test_public_conflict_malformed_drift_and_resource_refusal_before_bind(self):
        for argv in (["--port", "1234"], ["--engine", "mock"], ["--gpu", "3"], ["--lazy"],
                     ["--slot-save-path", str(self.root / "different")], ["--before-load", "SYNTHETIC-SECRET"],
                     ["--port", "SYNTHETIC-SECRET"], ["--config", str(self.root / "different.json")]):
            with self.subTest(argv=argv):
                self.main_refusal(argv)
        self.hardware.return_value["gpu_identity"] = "other GPU"
        self.main_refusal([])
        self.hardware.return_value = copy.deepcopy(fixtures.HARDWARE)
        self.hardware.return_value["commit_available_bytes"] = 1
        self.assertEqual(self.main_refusal([])["code"], "insufficient_resources")
        self.hardware.return_value = copy.deepcopy(fixtures.HARDWARE)
        (self.root / "profile.bin").write_bytes(b"changed metadata")
        self.main_refusal([])
        self.saved_path.write_text('{"api_key":"SYNTHETIC-SECRET"}')
        self.main_refusal([])

    def test_saved_integrity_capability_build_and_config_drift_refuse_before_spawn(self):
        variants = [lambda p: p.update(schema=77), lambda p: p["native"]["capabilities"].update(strict_consumer=False),
                    lambda p: p["launch"]["argv"].append("--serve"),
                    lambda p: p["native"].update(effective_payload="schema=77\n"),
                    lambda p: p["native"]["build"].update(commit="other")]
        for mutate in variants:
            changed = copy.deepcopy(self.saved)
            mutate(changed)
            changed["plan_digest"] = lp.digest(changed)
            self.saved_path.write_bytes(lp.canonical(changed))
            self.main_refusal([])
        changed = copy.deepcopy(self.saved)
        changed["plan_digest"] = "0" * 64
        self.saved_path.write_bytes(lp.canonical(changed))
        self.main_refusal([])
        self.saved_path.write_bytes(lp.canonical(self.saved))
        for path in (self.exe, self.root / "BUILD.json", self.config_path):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                if path == self.config_path:
                    changed = copy.deepcopy(self.cfg)
                    changed["model_name"] = "changed"
                    path.write_bytes(lp.canonical(changed))
                else:
                    path.write_bytes(original + b"changed")
                self.main_refusal([])
                path.write_bytes(original)
        with patch.object(server, "popen", side_effect=AssertionError("spawn")):
            self.hardware.return_value["vram_free_bytes"] = 0
            with self.assertRaisesRegex(lp.PlanError, "insufficient_resources"):
                self.start()

    def test_credentials_change_independently_with_shared_precedence(self):
        from serve.launch_runtime import replay_cli
        cfg = copy.deepcopy(self.cfg)
        cfg.update(api_key="SYNTHETIC-INLINE-KEY", api_key_file=str(self.root / "SYNTHETIC-KEY-PATH"))
        # Credential-only changes preserve the plan's original config identity.
        self.config_path.write_bytes(lp.canonical(cfg))
        args = lambda: SimpleNamespace(launch_plan=str(self.saved_path), config=None, engine="mock", api_key=None,
                                       tokenizer="default")
        with patch.dict(os.environ, self.env, clear=True), \
                patch("serve.credentials.read_key_file", return_value=["SYNTHETIC-FILE-KEY"]) as read:
            self.assertEqual(replay_cli(args(), set())[2], ["SYNTHETIC-FILE-KEY"])
            read.assert_called_once_with(str(self.root / "SYNTHETIC-KEY-PATH"))
            with patch.dict(os.environ, {"STRATA_API_KEY": "SYNTHETIC-ENV-KEY"}):
                self.assertEqual(replay_cli(args(), set())[2], "SYNTHETIC-ENV-KEY")
                cli = args()
                cli.api_key = "SYNTHETIC-CLI-KEY"
                self.assertEqual(replay_cli(cli, {"api_key"})[2], "SYNTHETIC-CLI-KEY")
                cli.api_key = ""
                with self.assertRaises(ValueError):
                    replay_cli(cli, {"api_key"})
            self.assertEqual(read.call_count, 1)

    def test_public_abbreviated_replay_bad_arguments_are_bounded_json(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["server", "--launch-p", str(self.saved_path), "--port", "SYNTHETIC-SECRET"]), \
                contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as error:
            server.main()
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(json.loads(output.getvalue())["error"]["code"], "invalid_launch_plan")
        self.assertNotIn("SECRET", output.getvalue())


if __name__ == "__main__":
    unittest.main()
