"""Public read-only planning with synthetic artifacts and a leaf metadata protocol."""
from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from serve import launch_plan as lp, server
from tools import build_provenance
from tools import prepare_unsloth_q2 as prep


HARDWARE = {"gpu_index": 0, "gpu_identity": "synthetic-GPU-0", "gpu_name": "fixture CUDA 6 GiB",
            "vram_total_bytes": 6 << 30, "vram_free_bytes": 6 << 30, "ram_total_bytes": 64 << 30,
            "ram_available_bytes": 48 << 30, "commit_available_bytes": 48 << 30,
            "windows_commit": True, "devices": 1, "commit_basis": "available_commit"}


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = value * 1099511628211 % 2**64
    return value


def fake_metadata(argv, env, *, reverse=False):
    """Independent protocol fixture, not native arithmetic or evidence of real fit."""
    assert argv[1] == "--resolve-launch" and "--serve" not in argv
    values = {}
    for flag in ("--max-context", "--prefill", "--spec", "--kv-resident", "--expert-cache",
                 "--resident-budget-gib", "--vram-reserve-mib", "--kv"):
        if flag in argv:
            values[flag] = argv[argv.index(flag) + 1]
    environment = {}
    for key, value in env.items():
        name = key.upper() if os.name == "nt" else key
        if (name.startswith(("STRATA_", "CUDA_", "GGML_", "CUBLAS_", "OMP_")) or
                name in ("PATH", "SYSTEMROOT", "LD_PRELOAD", "LD_LIBRARY_PATH")) and name not in (
                    "STRATA_API_KEY", "STRATA_API_KEY_FILE"):
            environment[name] = value
    env_bytes = b""
    for name, value in sorted(environment.items()):
        a, b = name.encode(), value.encode()
        env_bytes += str(len(a)).encode() + b":" + a + str(len(b)).encode() + b":" + b
    cache = values.get("--expert-cache", "auto")
    requested_prefill = values["--prefill"]
    settings = dict(context=int(values["--max-context"]),
                    prefill=2048 if requested_prefill.startswith("auto") else int(requested_prefill), spec=2,
                    kv={"fp16": 0, "int8": 1, "q4_0": 2, "k8v4": 3}[values.get("--kv", "int8")],
                    kv_resident=int(values.get("--kv-resident", 0)), batch=0, batch_groups=1, parallel=1,
                    auto_cache=int(cache == "auto"), per_layer=0, no_prefill_borrow=1,
                    resident_budget=max(1, int(float(values["--resident-budget-gib"]) * 1073741824)),
                    resident_bytes=0, resident_experts=0, resident_pin=1,
                    file_cache=int(env.get("STRATA_FILE_CACHE_MIB", "0")) << 20,
                    host_headroom=int(float(env.get("STRATA_RESIDENT_HEADROOM_GIB", "4")) * 1073741824),
                    vram_reserve=int(values.get("--vram-reserve-mib", "700")) << 20,
                    shared_stage=0, ring=2, workspace=32 << 20, weight_scratch=2 << 20,
                    common=2 << 20, region=2 << 20, prefill_mmq=0, prefill_stage_own=0,
                    prefill_reserve_bo=0, prefill_ring_enabled=1, experts=512, top_k=10, vocab=248320,
                    max_blob=256, cache_request=0 if cache == "auto" else int(cache), environment=fnv(env_bytes),
                    file_cache_effective=0, file_cache_compact=1, vram_required=1024,
                    ram_required=2048, commit_required=3072, stager_threads=2, stager_ring=16, stager_batch=1)
    payload = "schema=2\n" + "".join(f"{k}={v}\n" for k, v in settings.items())
    runs = []
    if settings["kv_resident"]:
        cells = (settings["context"] + 3) // 4 * 4
        runs = [cells * 288, cells * 288]
    slots = [[1, 2, 256, 0], [0, 1, 256, 256]] if reverse else [[0, 1, 256, 0], [1, 2, 256, 256]]
    payload += "layer_blobs=" + ",".join(["256"] * 48) + "\nkv_stage_runs=" + ",".join(map(str, runs)) + "\n"
    payload += "resident_order=\nslots=" + ",".join(str(x) for row in slots for x in row) + "\n"
    effective = "schema=1\nverified=1\n" + "".join(f"{k}={v}\n" for k, v in settings.items())
    effective += "cache_slots=2\ncache_bytes=512\nkv_stage_runs=" + ",".join(map(str, runs)) + "\n"
    effective += "native_payload_fingerprint=" + str(fnv(payload.encode())) + "\n"
    settings.update(layer_blobs=[256] * 48, kv_stage_runs=runs, resident_order=[], slots=slots)
    return dict(schema=1, native_schema=2, kind="native_metadata", strict_consumer=True, ownership_telemetry=True,
                ownership_protocol="STRATA-OWNERSHIP/1", ownership_frame_limit=1 << 20,
                ownership_shard_limit=256, effective_protocol="STRATA-EFFECTIVE/1",
                native_payload=payload, effective_payload=effective, settings=settings,
                estimates=dict(vram={"weights": 256, "expert_cache": 512, "overhead_allowance": 256},
                               ram={"weights": 1024, "host_headroom_allowance": 1024},
                               vram_required=1024, ram_required=2048, commit_required=3072,
                               unknown_costs="fixture: unmeasured driver/library/runtime overhead"),
                resources={"vram_free": int(argv[argv.index("--plan-vram-free-bytes") + 1]),
                           "ram_available": int(argv[argv.index("--plan-ram-available-bytes") + 1]),
                           "commit_available": int(argv[argv.index("--plan-commit-available-bytes") + 1]),
                           "devices": 1})


def fake_metadata_main():
    print(json.dumps(fake_metadata([sys.argv[0], *sys.argv[1:]], dict(os.environ))))


class Plans(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-launch-plan-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {k.upper() if os.name == "nt" else k: v for k, v in os.environ.items()
                    if k.upper() in ("SYSTEMROOT", "PATH", "TEMP", "TMP")}
        self.env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + str(Path(os.environ.get("SYSTEMROOT", "/")) / "System32")
        record = {"schema_version": 1, "version": "test", "commit": "synthetic", "dirty": True,
                  "built_at": "2026-01-01T00:00:00Z", "cuda": "synthetic", "backend": "cuda"}
        self.exe = self.root / ("fixture.cmd" if os.name == "nt" else "fixture")
        script = self.root / "metadata.py"
        script.write_text("import sys\nsys.path.insert(0," + repr(str(Path(__file__).resolve().parents[1])) +
                          ")\nfrom serve.test_launch_plan import fake_metadata_main\nfake_metadata_main()\n", encoding="utf-8")
        marker = build_provenance.MARKER + build_provenance.json_bytes(record) + b"\n"
        if os.name == "nt":
            self.exe.write_bytes(b"@echo off\n@rem " + marker +
                                 f'@set "PATH={self.env["PATH"]}"\n'.encode() +
                                 f'"{sys.executable}" -B "{script}" %*\n'.encode())
        else:
            self.exe.write_bytes(b"#!/bin/sh\n# " + marker +
                                 f'exec "{sys.executable}" -B "{script}" "$@"\n'.encode())
            self.exe.chmod(0o700)
        receipt = build_provenance.metadata_for(self.exe.read_bytes(), self.exe.name)
        (self.root / "BUILD.json").write_text(json.dumps(receipt), encoding="utf-8")
        for name in ("index.txt", "native_experts.txt", "profile.bin"):
            (self.root / name).write_bytes(b"synthetic allocation metadata\n")
        (self.root / "dense.bin").write_bytes(b"synthetic payload")
        (self.root / "model.gguf").write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 0) + b"payload not read")
        self.cfg = self.config(8192)
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.cfg), encoding="utf-8")

    def config(self, context):
        generated = prep.memory_plan(64, 6, context, available_gib=48, commit_available_gib=48)
        return {"exe": str(self.exe), "cwd": str(self.root), "gpu": 0, "parallel": 1,
                "args": ["--pack", str(self.root), "--native", str(self.root / "model.gguf"),
                         "--expert-profile", str(self.root / "profile.bin"), *generated["args"]],
                "env": generated["env"], "model_name": "synthetic", "tokenizer": str(self.root / "missing-tokenizer"),
                "memory_policy": {"version": 1, "generated_args": generated["args"],
                                  "generated_env": generated["env"], "overrides": {}}}

    def query(self, argv, *, cwd, env):
        self.assertEqual(cwd, str(self.root.resolve()))
        return json.dumps(fake_metadata(argv, env)).encode()

    def plan(self, cfg=None, **kwargs):
        return lp.produce_plan(cfg or self.cfg, hardware=kwargs.pop("hardware", HARDWARE),
                               inherited=kwargs.pop("inherited", self.env), query=kwargs.pop("query", self.query), **kwargs)

    def state(self):
        return {str(p.relative_to(self.root)): (p.stat().st_size, p.read_bytes())
                for p in self.root.iterdir() if p.is_file()}

    def rehash(self, plan):
        plan["plan_digest"] = lp.digest(plan)
        return plan

    def test_deterministic_digest_and_original_payload_round_trip(self):
        before = self.state()
        a, b = self.plan(), self.plan()
        self.assertEqual(lp.canonical(a), lp.canonical(b))
        independent = json.dumps({k: v for k, v in a.items() if k != "plan_digest"}, sort_keys=True,
                                 separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
        self.assertEqual(a["plan_digest"], hashlib.sha256(independent).hexdigest())
        payload = base64.b64decode(a["native"]["native_payload_base64"])
        self.assertEqual(a["native"]["native_payload_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(lp.strict_frame(a), b"STRATA-PLAN/1\n" + str(len(payload)).encode() + b"\n" + payload)
        self.assertNotIn("built_at", a["native"]["build"])
        self.assertEqual(self.state(), before)
        saved = self.root / "plan.json"
        saved.write_bytes(lp.canonical(a))
        self.assertEqual(lp.read_plan(saved), a)
        self.assertEqual(lp.validate_inputs(a), a)

    def test_queue_capacity_is_validated_and_frozen_exactly(self):
        from types import SimpleNamespace
        from serve import launch_runtime
        for value in (None, 0, 32, 65536):
            cfg = {**self.cfg, "max_queued_requests": value}
            plan = self.plan(cfg)
            self.assertIn("max_queued_requests", plan["server"])
            self.assertEqual(plan["server"]["max_queued_requests"], value)
            self.assertEqual(lp.validate_plan(plan)["server"]["max_queued_requests"], value)
            saved = self.root / "queue-plan.json"
            saved.write_bytes(lp.canonical(plan))
            args = SimpleNamespace(launch_plan=str(saved), config=None, api_key=None)
            with patch.object(launch_runtime, "admit"):
                _, replayed, _ = launch_runtime.replay_cli(args, set())
            self.assertEqual(replayed["max_queued_requests"], value)
        self.assertNotIn("max_queued_requests", self.plan()["server"])
        for value in (True, False, -1, 1.5, "32", 65537, [], {}):
            with self.subTest(value=value), self.assertRaises(lp.PlanError):
                self.plan({**self.cfg, "max_queued_requests": value})

    def test_generated_small_gpu_profiles_use_complete_policy(self):
        for context, kv, resident, budget, recent in ((8192, 1, 0, 22, 10176), (131072, 2, 20480, 21, 9816)):
            with self.subTest(context=context):
                plan = self.plan(self.config(context))
                s = plan["resolved"]
                self.assertEqual((s["context"], s["prefill"], s["spec"], s["kv"], s["kv_resident"]),
                                 (context, 64, 2, kv, resident))
                self.assertEqual((s["no_prefill_borrow"], s["auto_cache"], s["resident_budget"]), (1, 1, budget << 30))
                self.assertEqual(s["file_cache"], recent << 20)
                self.assertEqual(plan["launch"]["argv"][1:3], ["--strict-launch-stdin", "--serve"])
                self.assertIn("--no-prefill-borrow", plan["launch"]["argv"])
                self.assertEqual(plan["estimates"]["units"], "bytes")
                self.assertIsNone(plan["estimates"]["runtime_comparison"]["physical_residency_bytes"])
                self.assertEqual(bool(s["kv_stage_runs"]), context == 131072)

    def test_explicit_cache_preserves_request_budget_not_applied_slot_count(self):
        cfg = copy.deepcopy(self.cfg)
        prep.set_flag(cfg["args"], "--expert-cache", "97")
        prep.set_flag(cfg["args"], "--prefill", "32")
        plan = self.plan(cfg)
        self.assertEqual(plan["resolved"]["cache_request"], 97)
        self.assertEqual(plan["resolved"]["auto_cache"], 0)
        self.assertEqual(plan["resolved"]["prefill"], 32)
        self.assertEqual(len(plan["resolved"]["slots"]), 2)
        self.assertEqual(plan["launch"]["argv"][plan["launch"]["argv"].index("--expert-cache") + 1], "97")

    def test_ordered_native_layouts_with_equal_slot_count_are_preserved(self):
        a = self.plan()
        b = self.plan(query=lambda argv, cwd, env: json.dumps(fake_metadata(argv, env, reverse=True)).encode())
        self.assertEqual(len(a["resolved"]["slots"]), len(b["resolved"]["slots"]))
        self.assertEqual(a["resolved"]["slots"], [[0, 1, 256, 0], [1, 2, 256, 256]])
        self.assertEqual(b["resolved"]["slots"], [[1, 2, 256, 0], [0, 1, 256, 256]])
        self.assertNotEqual(lp.strict_frame(a), lp.strict_frame(b))
        self.assertNotEqual(a["plan_digest"], b["plan_digest"])

    def test_no_credential_reads_or_serialization_and_credential_independent_identity(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(api_key="SYNTHETIC-INLINE-SECRET", api_key_file=str(self.root / "absent-secret-file"))
        cfg["env"]["STRATA_API_KEY"] = "SYNTHETIC-CONFIG-SECRET"
        env = {**self.env, "STRATA_API_KEY": "SYNTHETIC-ENV-SECRET", "UNRELATED_SECRET": "SYNTHETIC-AMBIENT-SECRET"}
        with patch("serve.credentials.read_key_file") as read, patch.object(prep, "check_ready") as ready:
            plan = self.plan(cfg, inherited=env)
        read.assert_not_called()
        ready.assert_not_called()
        for value in ("SYNTHETIC-INLINE-SECRET", "absent-secret-file", "SYNTHETIC-ENV-SECRET",
                      "SYNTHETIC-AMBIENT-SECRET", "SYNTHETIC-CONFIG-SECRET"):
            self.assertNotIn(value, lp.canonical(plan).decode())
        self.assertEqual(plan["inputs"]["config_sha256"], self.plan()["inputs"]["config_sha256"])
        self.assertNotIn("STRATA_API_KEY", plan["launch"]["environment"]["set"])

    def test_replay_environment_freezes_native_controls_and_search_paths(self):
        plan = self.plan(inherited={**self.env, "CUDA_PATH": "/synthetic/cuda", "CUDA_PATH_V99_0": "/synthetic/cuda99"})
        inherited = {"PATH": "changed", "STRATA_SECRET": "secret", "CUDA_NEW_CONTROL": "changed",
                     "STRATA_API_KEY": "server-only", "SYSTEMROOT": "changed", "UNRELATED": "keep"}
        actual = lp.replay_environment(plan["launch"]["environment"], inherited)
        self.assertEqual(actual["PATH"], self.env["PATH"])
        self.assertEqual(actual["CUDA_MODULE_LOADING"], "EAGER")
        self.assertEqual(actual["CUDA_PATH"], "/synthetic/cuda")
        self.assertEqual(actual["CUDA_PATH_V99_0"], "/synthetic/cuda99")
        self.assertEqual(actual["UNRELATED"], "keep")
        self.assertNotIn("STRATA_SECRET", actual)
        self.assertNotIn("CUDA_NEW_CONTROL", actual)
        self.assertNotIn("STRATA_API_KEY", actual)

    def test_unknown_native_environment_refuses_without_echo(self):
        for env in ({**self.env, "STRATA_UNKNOWN": "do-not-print"}, {**self.env, "GGML_SECRET": "do-not-print"}):
            with self.assertRaises(lp.PlanError) as error:
                self.plan(inherited=env)
            self.assertNotIn("do-not-print", str(error.exception))

    def test_resource_changes_are_visible_and_not_resampled(self):
        with patch.object(lp, "observe_hardware", return_value=HARDWARE) as observed, \
                patch.object(prep, "detected_memory") as memory, patch.object(prep, "detected_gpu") as gpu:
            a = lp.produce_plan(self.cfg, inherited=self.env, query=self.query)
        observed.assert_called_once_with(0)
        memory.assert_not_called()
        gpu.assert_not_called()
        b = self.plan(hardware={**HARDWARE, "vram_free_bytes": (6 << 30) - 1})
        self.assertNotEqual(a["plan_digest"], b["plan_digest"])
        self.assertEqual(b["hardware"]["vram_free_bytes"], (6 << 30) - 1)

    def test_insufficient_observed_resources_refuse_before_any_launch(self):
        with self.assertRaisesRegex(lp.PlanError, "insufficient_resources"):
            self.plan(hardware={**HARDWARE, "ram_available_bytes": 1})

    def test_automatic_prefill_delegates_once_and_replays_concrete_native_chunk(self):
        before = self.state()
        for value in ("auto", "auto:8192", "auto:16384", "auto:32768"):
            cfg = copy.deepcopy(self.cfg)
            prep.set_flag(cfg["args"], "--prefill", value)
            query = Mock(side_effect=self.query)
            a = self.plan(cfg, query=query)
            query.assert_called_once()
            sent = query.call_args.args[0]
            self.assertEqual(sent[sent.index("--prefill") + 1], value)
            self.assertEqual(a["resolved"]["prefill"], 2048)
            replay = a["launch"]["argv"]
            self.assertEqual(replay[replay.index("--prefill") + 1], "2048")
            for original in (a["requested"]["args"], a["requested"]["native_args"]):
                self.assertEqual(original[original.index("--prefill") + 1], value)
            self.assertEqual(lp.canonical(a), lp.canonical(self.plan(cfg)))
            payload = json.loads(self.query(sent, **query.call_args.kwargs))["native_payload"].encode()
            self.assertTrue(lp.strict_frame(a).endswith(payload))
            for field in ("launch", "requested"):
                changed = copy.deepcopy(a)
                args = changed[field]["argv" if field == "launch" else "native_args"]
                args[args.index("--prefill") + 1] = "auto" if field == "launch" else "64"
                with self.assertRaises(lp.PlanError):
                    lp.validate_plan(self.rehash(changed))
        self.assertEqual(self.state(), before)

    def test_fractional_resident_budget_and_headroom_keep_exact_bytes(self):
        before = self.state()
        cfg = copy.deepcopy(self.cfg)
        prep.set_flag(cfg["args"], "--resident-budget-gib", "0.25")
        cfg["env"] = {**cfg["env"], "STRATA_RESIDENT_HEADROOM_GIB": "8.25"}
        a = self.plan(cfg)
        self.assertEqual(a["resolved"]["resident_budget"], 268435456)
        self.assertEqual(a["resolved"]["host_headroom"], 8858370048)
        self.assertEqual(a["launch"]["environment"]["set"]["STRATA_RESIDENT_HEADROOM_GIB"], "8.25")
        self.assertEqual(a["launch"]["argv"][a["launch"]["argv"].index("--resident-budget-gib") + 1], "0.25")
        self.assertEqual(lp.canonical(a), lp.canonical(self.plan(cfg)))
        self.assertIn(b"resident_budget=268435456\n", lp.strict_frame(a))
        self.assertIn(b"host_headroom=8858370048\n", lp.strict_frame(a))
        self.assertEqual(self.state(), before)
        changed = copy.deepcopy(a)
        changed["launch"]["environment"]["set"]["STRATA_RESIDENT_HEADROOM_GIB"] = "8.5"
        with self.assertRaisesRegex(lp.PlanError, "headroom disagreement"):
            lp.validate_plan(self.rehash(changed))

    def test_malformed_fractional_transport_and_unsupported_auto_are_read_only(self):
        before = self.state()
        for value in ("", "nan", "inf", "-0", "-0.25", "1e999", "1024.0001", "0x1p0", "1x", " 1", "1 ", "1..2"):
            for owner in ("budget", "headroom"):
                cfg = copy.deepcopy(self.cfg)
                if owner == "budget":
                    prep.set_flag(cfg["args"], "--resident-budget-gib", value)
                else:
                    cfg["env"] = {**cfg["env"], "STRATA_RESIDENT_HEADROOM_GIB": value}
                query = Mock()
                with self.subTest(value=value, owner=owner), self.assertRaises(lp.PlanError):
                    self.plan(cfg, query=query)
                query.assert_not_called()
        for value in ("auto:", "auto:bad", "auto:65536", "auto:0"):
            cfg = copy.deepcopy(self.cfg)
            prep.set_flag(cfg["args"], "--prefill", value)
            with self.assertRaises(lp.PlanError):
                self.plan(cfg)
        self.assertEqual(self.state(), before)

    def test_fractional_transport_matches_native_double_and_default_headroom(self):
        for text, expected in (("0.25", 268435456), (".25", 268435456), ("+0.25", 268435456),
                               ("2.5e-1", 268435456), ("00.25", 268435456), ("0.0000000001", 1)):
            self.assertEqual(lp._gib_bytes(text, positive=True), expected)
        self.assertEqual(lp._gib_bytes("0.0000000001"), 0)
        self.assertEqual(lp._gib_bytes("0"), 0)
        cfg = copy.deepcopy(self.cfg)
        cfg.pop("memory_policy")
        cfg["env"] = {k: v for k, v in cfg["env"].items() if k != "STRATA_RESIDENT_HEADROOM_GIB"}
        self.assertEqual(self.plan(cfg)["resolved"]["host_headroom"], 4294967296)

    def test_public_automatic_fractional_planning_uses_leaf_protocol_and_starts_nothing(self):
        cfg = copy.deepcopy(self.cfg)
        prep.set_flag(cfg["args"], "--prefill", "auto")
        prep.set_flag(cfg["args"], "--resident-budget-gib", "0.25")
        cfg["env"] = {**cfg["env"], "STRATA_RESIDENT_HEADROOM_GIB": "8.25"}
        self.config_path.write_text(json.dumps(cfg), encoding="utf-8")
        before = self.state()
        code_a, a = self.cli("--print-launch-plan")
        code_b, b = self.cli("--print-launch-plan")
        self.assertEqual((code_a, code_b), (0, 0))
        self.assertEqual(a, b)
        plan = json.loads(a)
        self.assertEqual(plan["resolved"]["prefill"], 2048)
        self.assertEqual(plan["resolved"]["resident_budget"], 268435456)
        self.assertEqual(plan["resolved"]["host_headroom"], 8858370048)
        self.assertEqual(plan["requested"]["args"][plan["requested"]["args"].index("--prefill") + 1], "auto")
        self.assertIn(b"prefill=2048\n", lp.strict_frame(plan))
        self.assertEqual(self.state(), before)

    def test_unsupported_topology_flags_and_malformed_numbers(self):
        for flag, value in (("--prefill", "64x"), ("--spec", "3"), ("--max-context", "NaN"),
                            ("--batch", "2"), ("--prefill", "064")):
            cfg = copy.deepcopy(self.cfg)
            if flag in cfg["args"]:
                prep.set_flag(cfg["args"], flag, value)
            else:
                cfg["args"] += [flag, value]
            with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                self.plan(cfg)
        for extra in ({"parallel": 2}, {"gpu": "0,1"}, {"backend": "hip"}, {"vision": {}},
                      {"effort_position": "end"}, {"before_load": "command"}):
            with self.subTest(extra=extra), self.assertRaises(lp.PlanError):
                self.plan({**self.cfg, **extra})

    def test_duplicate_args_and_mismatched_resource_reply_refuse(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["args"] += ["--prefill", "64"]
        with self.assertRaises(lp.PlanError):
            self.plan(cfg)
        def wrong(argv, cwd, env):
            reply = fake_metadata(argv, env)
            reply["resources"]["vram_free"] -= 1
            return json.dumps(reply).encode()
        with self.assertRaisesRegex(lp.PlanError, "different resource sample"):
            self.plan(query=wrong)

    def test_bad_schema_digest_payload_effective_and_contradictions_refuse(self):
        base = self.plan()
        for key, value in (("schema", 2), ("schema", True), ("kind", "other"), ("unknown", 1)):
            bad = copy.deepcopy(base)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(lp.PlanError):
                lp.validate_plan(self.rehash(bad))
        bad = copy.deepcopy(base)
        bad["plan_digest"] = "0" * 64
        with self.assertRaisesRegex(lp.PlanError, "public digest mismatch"):
            lp.validate_plan(bad)
        for mutate in (lambda p: p["resolved"].update(prefill=32),
                       lambda p: p["native"].update(native_payload_base64="!"),
                       lambda p: p["native"].update(effective_payload="schema=1\nverified=0\n"),
                       lambda p: p["native"]["capabilities"].update(strict_consumer=False),
                       lambda p: p["estimates"]["vram"].update(weights=257),
                       lambda p: p["launch"]["environment"]["set"].update(STRATA_API_KEY="secret"),
                       lambda p: p["launch"]["environment"]["set"].update(CUDA_MODULE_LOADING="LAZY")):
            bad = copy.deepcopy(base)
            mutate(bad)
            with self.assertRaises(lp.PlanError):
                lp.validate_plan(self.rehash(bad))

    def test_malformed_nonfinite_duplicate_json(self):
        for data in (b'{"a":1,"a":2}', b'{"schema":NaN}', b'{"a":1e999}', b'[]x', b'\xff'):
            with self.subTest(data=data), self.assertRaises(lp.PlanError):
                lp.parse_json(data)
        with self.assertRaises(lp.PlanError):
            lp.canonical({"n": float("inf")})
        with self.assertRaises(lp.PlanError) as error:
            lp.read_plan(self.root / "synthetic-secret-missing-plan")
        self.assertNotIn("synthetic-secret", str(error.exception))
        with self.assertRaises(lp.PlanError):
            lp._codec_fields(b"schema=2\r\n")

    def test_metadata_and_executable_identity_drift(self):
        a = self.plan()
        path = self.root / "index.txt"
        path.write_bytes(b"changed metadata")
        with self.assertRaisesRegex(lp.PlanError, "input identity changed"):
            lp.validate_inputs(a)
        path.write_bytes(b"synthetic allocation metadata\n")
        self.exe.write_bytes(self.exe.read_bytes() + b"\n")
        with self.assertRaisesRegex(lp.PlanError, "executable identity changed"):
            lp.validate_inputs(a)

    def test_role_shards_and_new_pack_payloads_are_bound(self):
        path = self.root / "other.gguf"
        path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 0))
        (self.root / "native_experts.txt").write_text("0 18 20 0 256 32 64 96 ,other.gguf,\n")
        plan = self.plan()
        self.assertIn(str(path), [row["path"] for row in plan["inputs"]["files"]])
        (self.root / "experts.bin").write_bytes(b"new allocation source")
        with self.assertRaisesRegex(lp.PlanError, "input identity changed"):
            lp.validate_inputs(plan)

    def test_model_header_identity_does_not_read_tensor_payload(self):
        a = self.plan()
        model = next(x for x in a["inputs"]["files"] if x["metadata_sha256"] is not None)
        self.assertEqual(model["metadata_bytes"], 24)
        self.assertEqual(model["metadata_sha256"], hashlib.sha256(struct.pack("<IIQQ", 0x46554747, 3, 0, 0)).hexdigest())
        self.assertIsNone(model["content_sha256"])
        self.assertIn("complete model/library content", a["approval"])

    def test_fault_boundaries_leave_complete_synthetic_state_unchanged(self):
        for stage in ("config", "sample", "metadata", "serialization"):
            with self.subTest(stage=stage):
                before = self.state()
                if stage == "config":
                    with self.assertRaises(lp.PlanError):
                        lp.parse_json(b'{"api_key":"synthetic-secret",')
                elif stage == "sample":
                    with patch.object(lp, "observe_hardware", side_effect=OSError("synthetic-secret")), self.assertRaises(lp.PlanError):
                        lp.produce_plan(self.cfg, query=self.query, inherited=self.env)
                elif stage == "metadata":
                    with self.assertRaises(lp.PlanError):
                        self.plan(query=lambda *a, **kw: b'{"key":"synthetic-secret",')
                else:
                    real_canonical = lp.canonical
                    def fail_plan(value):
                        if isinstance(value, dict) and value.get("kind") == "strata_launch_plan":
                            raise lp.PlanError("invalid_launch_plan: injected serialization failure")
                        return real_canonical(value)
                    with patch.object(lp, "canonical", side_effect=fail_plan), self.assertRaises(lp.PlanError):
                        self.plan()
                self.assertEqual(self.state(), before)

    def cli(self, *options):
        output, errors = io.StringIO(), io.StringIO()
        argv = ["strata", "--engine", "strata", "--config", str(self.config_path), *options]
        with patch.object(sys, "argv", argv), patch.object(lp, "observe_hardware", return_value=HARDWARE), \
                patch.dict(os.environ, self.env, clear=True), contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), \
                patch.object(server, "Server") as bind, patch.object(server, "ByteTokenizer") as tokenizer, \
                patch.object(server, "Service") as service, patch.object(server, "StrataEngine") as engine, \
                patch.object(server, "hub_from_config") as hub, patch.object(server, "resolve_credentials") as credentials, \
                patch("webbrowser.open") as browser:
            result = server.main()
        for seam in (bind, tokenizer, service, engine, hub, credentials, browser):
            seam.assert_not_called()
        self.assertEqual(errors.getvalue(), "")
        return result, output.getvalue()

    def test_public_main_uses_fake_native_executable_and_outputs_only_repeatable_json(self):
        before = self.state()
        code_a, a = self.cli("--print-launch-plan", "--api-key", "synthetic-cli-secret")
        code_b, b = self.cli("--print-launch-plan", "--api-key", "synthetic-cli-secret")
        self.assertEqual((code_a, code_b), (0, 0))
        self.assertEqual(a, b)
        plan = lp.validate_plan(json.loads(a))
        self.assertEqual(plan["server"]["port"], 8095)
        self.assertNotIn("synthetic-cli-secret", a)
        self.assertEqual(self.state(), before)
        self.assertEqual(lp.validate_inputs(plan), plan)

    def test_public_estimate_matches_plan_and_preserves_server_precedence(self):
        options = ("--host", "localhost", "--port", "9012", "--gpu", "0", "--lazy", "--fit-max-tokens")
        code, data = self.cli("--print-launch-plan", *options)
        self.assertEqual(code, 0, data)
        plan = json.loads(data)
        self.assertEqual(plan["server"]["host"], "localhost")
        self.assertEqual(plan["server"]["port"], 9012)
        self.assertIs(plan["server"]["lazy_load"], True)
        code, data = self.cli("--estimate-memory", *options)
        self.assertEqual(code, 0, data)
        estimate = json.loads(data)
        self.assertEqual(estimate["estimates"], plan["estimates"])
        self.assertEqual(estimate["admission"], plan["admission"])
        self.assertEqual(estimate["plan_digest"], plan["plan_digest"])

    def test_public_plan_freezes_the_actual_browser_action(self):
        for configured, cli_open, expected in ((True, False, False), (False, True, False),
                                                (True, True, True), (None, True, True)):
            with self.subTest(configured=configured, cli_open=cli_open):
                cfg = copy.deepcopy(self.cfg)
                if configured is not None:
                    cfg["open_browser"] = configured
                self.config_path.write_bytes(lp.canonical(cfg))
                code, output = self.cli("--print-launch-plan", *(["--open"] if cli_open else []))
                self.assertEqual(code, 0, output)
                self.assertIs(json.loads(output)["server"]["open_browser"], expected)

    def test_public_faults_are_secret_free_read_only_and_never_start(self):
        for contents in (b'{"api_key":"synthetic-secret",', b'[]', b'{"exe":"synthetic-secret"}'):
            self.config_path.write_bytes(contents)
            before = self.state()
            code, output = self.cli("--print-launch-plan")
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output)["error"]["code"], "invalid_launch_plan")
            self.assertNotIn("synthetic-secret", output)
            self.assertEqual(self.state(), before)

    def test_server_settings_refuse_before_query_and_in_saved_plans(self):
        baseline = self.plan()
        invalid = ({"engine_silence_s": -1}, {"sampling": {"top_p": 2}},
                   {"sampling": {"temperature": "synthetic-secret"}}, {"aliases": [1]},
                   {"allowed_hosts": [1]}, {"cors_origins": "synthetic-secret"},
                   {"trusted_origins": "*"}, {"tool_call_recovery": "synthetic-secret"},
                   {"anthropic_thinking": "synthetic-secret"}, {"port": 0},
                   {"idle_unload_s": -1}, {"min_free_vram_mib": -1})
        before = self.state()
        for settings in invalid:
            query = Mock()
            with self.subTest(settings=settings), self.assertRaises(lp.PlanError) as error:
                self.plan({**self.cfg, **settings}, query=query)
            self.assertNotIn("synthetic-secret", str(error.exception))
            query.assert_not_called()
            changed = copy.deepcopy(baseline)
            changed["server"].update(settings)
            with self.assertRaises(lp.PlanError):
                lp.validate_plan(self.rehash(changed))
        self.assertEqual(self.state(), before)

    def test_public_invalid_server_settings_are_secret_free_json(self):
        for settings in ({"engine_silence_s": -1}, {"sampling": {"top_p": 2}},
                         {"sampling": {"temperature": "synthetic-secret"}}):
            self.config_path.write_text(json.dumps({**self.cfg, **settings}), encoding="utf-8")
            before = self.state()
            code, output = self.cli("--print-launch-plan")
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output)["error"]["code"], "invalid_launch_plan")
            self.assertNotIn("synthetic-secret", output)
            self.assertEqual(self.state(), before)

    def test_server_normalization_reuses_ordinary_validation_without_diagnostics(self):
        settings = {"aliases": "other, synthetic, other", "sampling": {"top_p": 0.5, "unknown": "synthetic-secret"},
                    "allowed_hosts": "https://EXAMPLE.com:8443/path", "engine_silence_s": 0}
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            plan = self.plan({**self.cfg, **settings})
        self.assertEqual(plan["server"]["aliases"], ["other"])
        self.assertEqual(plan["server"]["sampling"], {"top_p": 0.5})
        self.assertEqual(plan["server"]["allowed_hosts"], ["example.com"])
        self.assertEqual(plan["server"]["engine_silence_s"], 0)
        self.assertEqual((output.getvalue(), errors.getvalue()), ("", ""))
        self.assertNotIn("synthetic-secret", lp.canonical(plan).decode())

    def test_session_directory_freezes_config_and_cli_bases_without_creation(self):
        cfg = {**self.cfg, "slot_save_path": "relative-sessions"}
        self.config_path.write_text(json.dumps(cfg), encoding="utf-8")
        before = self.state()
        with patch.object(os, "makedirs", side_effect=AssertionError("planning created directory")):
            code_a, raw_a = self.cli("--print-launch-plan")
            code_b, raw_b = self.cli("--print-launch-plan", "--slot-save-path", "relative-sessions")
            code_empty, raw_empty = self.cli("--print-launch-plan", "--slot-save-path", "")
        self.assertEqual((code_a, code_b), (0, 0))
        self.assertEqual((code_empty, raw_empty), (0, raw_a))
        a, b = json.loads(raw_a), json.loads(raw_b)
        self.assertEqual(a["server"]["slot_save_path"], os.path.abspath(self.root / "relative-sessions"))
        self.assertEqual(b["server"]["slot_save_path"], os.path.abspath("relative-sessions"))
        self.assertNotEqual(a["plan_digest"], b["plan_digest"])
        changed = copy.deepcopy(a)
        changed["server"]["slot_save_path"] = "relative-sessions"
        with self.assertRaises(lp.PlanError):
            lp.validate_plan(self.rehash(changed))
        with patch.object(os, "makedirs") as create, patch.object(os.path, "isdir", return_value=True):
            self.assertEqual(server.slot_save_dir("relative-sessions", str(self.root)), a["server"]["slot_save_path"])
        create.assert_called_once_with(a["server"]["slot_save_path"], mode=0o700, exist_ok=True)
        self.assertEqual(self.state(), before)

    def test_saved_config_drift_is_validated_without_reading_credentials(self):
        plan = self.plan(source_config=self.cfg, config_path=self.config_path)
        cfg = {**self.cfg, "api_key_file": str(self.root / "not-present")}
        self.config_path.write_text(json.dumps(cfg), encoding="utf-8")
        lp.validate_inputs(plan)
        cfg["model_name"] = "changed"
        self.config_path.write_text(json.dumps(cfg), encoding="utf-8")
        with self.assertRaisesRegex(lp.PlanError, "configuration identity changed"):
            lp.validate_inputs(plan)


class Queries(unittest.TestCase):
    def test_gpu_identity_total_and_free_share_one_selected_sample(self):
        sample = b"0, GPU-zero, fixture zero, 4096, 3000\n2, GPU-two, fixture two, 6144, 5000\n"
        memory = {"ram_gib": 64, "available_gib": 48, "commit_available_gib": 40, "windows_commit": True}
        with patch.object(lp, "bounded_query", return_value=sample) as query, \
                patch.object(prep, "memory_snapshot", return_value=memory) as host:
            observed = lp.observe_hardware(2)
        query.assert_called_once()
        host.assert_called_once_with()
        self.assertEqual((observed["gpu_index"], observed["gpu_identity"], observed["gpu_name"]), (2, "GPU-two", "fixture two"))
        self.assertEqual((observed["vram_total_bytes"], observed["vram_free_bytes"]), (6144 << 20, 5000 << 20))
        self.assertEqual((observed["ram_total_bytes"], observed["ram_available_bytes"], observed["commit_available_bytes"]),
                         (64 << 30, 48 << 30, 40 << 30))

    def test_planning_parser_errors_are_json_and_do_not_echo_values(self):
        for extra in (("--port", "synthetic-secret"), ("--unknown", "synthetic-secret"), ("--estimate-memory",),
                      ("--print-launch-plan=synthetic-secret",)):
            output, errors = io.StringIO(), io.StringIO()
            prefix = [] if extra[0].startswith("--print-launch-plan=") else ["--print-launch-plan"]
            with patch.object(sys, "argv", ["strata", *prefix, *extra]), \
                    contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as exit:
                server.main()
            self.assertEqual(exit.exception.code, 2)
            self.assertEqual(json.loads(output.getvalue())["error"]["code"], "invalid_launch_plan")
            self.assertEqual(errors.getvalue(), "")
            self.assertNotIn("synthetic-secret", output.getvalue())

    def test_abbreviated_planning_parser_errors_do_not_echo_argument_values(self):
        for option in ("--print-launch-pl", "--print", "--estimate-mem", "--estimate"):
            output, errors = io.StringIO(), io.StringIO()
            with patch.object(sys, "argv", ["strata", option, "--port", "synthetic-secret"]), \
                    contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as exit:
                server.main()
            self.assertEqual(exit.exception.code, 2)
            self.assertEqual(json.loads(output.getvalue())["error"]["code"], "invalid_launch_plan")
            self.assertEqual(errors.getvalue(), "")
            self.assertNotIn("synthetic-secret", output.getvalue())

    def test_timeout_and_overlong_output_confirm_owned_leaf_exit(self):
        original = server.popen
        processes = []
        def capture(*a, **kw):
            process = original(*a, **kw)
            processes.append(process)
            return process
        for script, timeout, limit in (("import time;time.sleep(20)", 0.05, 1024),
                                        ("print('x'*100000)", 5, 1024)):
            with patch.object(server, "popen", side_effect=capture), self.assertRaises(lp.PlanError):
                lp.bounded_query([sys.executable, "-B", "-c", script], cwd=tempfile.gettempdir(),
                                 env=dict(os.environ), timeout=timeout, limit=limit)
            self.assertIsNotNone(processes[-1].poll())

    def test_query_refuses_ready_and_does_not_echo_stderr(self):
        for script in ("print('READY')", "import sys;sys.stderr.write('synthetic-secret');sys.exit(2)"):
            with self.assertRaises(lp.PlanError) as error:
                lp.bounded_query([sys.executable, "-B", "-c", script], cwd=tempfile.gettempdir(), env=dict(os.environ))
            self.assertNotIn("synthetic-secret", str(error.exception))


if __name__ == "__main__":
    unittest.main()
