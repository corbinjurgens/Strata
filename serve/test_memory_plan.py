"""Public pure pricing tests, synthetic files and injected native transport only."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from serve import launch_plan as lp, memory_plan as mp, server
from serve.test_launch_plan import fake_metadata
from tools import build_provenance


PUBLIC_TEMP = Path(tempfile.gettempdir()) / "strata-memory-plan-tests"


class MemoryPlans(unittest.TestCase):
    def setUp(self):
        PUBLIC_TEMP.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-", dir=PUBLIC_TEMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.exe = self.root / "synthetic.exe"
        record = {"schema_version": 1, "version": "fixture", "commit": "synthetic", "dirty": True,
                  "built_at": "2026-10-29T00:00:00Z", "cuda": "synthetic", "backend": "cuda"}
        self.exe.write_bytes(b"Not executable: " + build_provenance.MARKER + build_provenance.json_bytes(record) + b"\n")
        for name in ("index.txt", "native_experts.txt", "profile.bin"):
            (self.root / name).write_bytes(b"synthetic metadata\n")
        (self.root / "dense.bin").write_bytes(b"unread tensor payload")
        (self.root / "model.gguf").write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 0) + b"unread tensors")
        self.request = {"schema_version": 1, "model": {"id": "synthetic"},
            "config": {"context": 8192, "kv_type": "int8", "batch_size": 1, "microbatch_size": 1024,
                       "parallel_slots": 1, "devices": [{"id": 0, "backend": "cuda"}], "offload": {"mode": "hybrid"}},
            "budget": {"basis": "before_runtime_initialization", "devices": [{"id": 0, "bytes": 1 << 30}],
                       "host_ram_bytes": 8 << 30, "host_commit_bytes": 16 << 30},
            "extensions": {"strata": {"executable": str(self.exe), "working_directory": str(self.root),
                "pack": str(self.root), "native_model": str(self.root / "model.gguf"),
                "expert_profile": str(self.root / "profile.bin"), "resident_budget_bytes": 1 << 30,
                "file_cache_bytes": 0, "host_headroom_bytes": 0, "vram_reserve_bytes": 1 << 20,
                "cache_slots": 2, "kv_resident_cells": 0, "spec": 2, "no_prefill_borrow": True}}}
        self.calls = []

    def native(self, argv, *, cwd, env):
        self.calls.append((list(argv), cwd, dict(env)))
        # Existing read-only fixture supplies native settings, never a second
        # implementation of allocation arithmetic. Prices below are protocol data.
        fixture_argv = list(argv)
        if "--plan-vram-free-bytes" not in argv:
            fixture_argv += ["--plan-vram-free-bytes", "0", "--plan-ram-available-bytes", "0",
                             "--plan-commit-available-bytes", "0"]
        old = fake_metadata(fixture_argv, env)
        vram = dict.fromkeys(mp.DEVICE_COMPONENTS, 10)
        vram["overhead_allowance"] = int(argv[argv.index("--vram-reserve-mib") + 1]) << 20
        ram = dict.fromkeys(mp.HOST_COMPONENTS, 10)
        ram["host_headroom_allowance"] = int(float(env["STRATA_RESIDENT_HEADROOM_GIB"]) * (1 << 30))
        ram["file_cache_target"] = int(env["STRATA_FILE_CACHE_MIB"]) << 20
        vtotal, rtotal = sum(vram.values()), sum(ram.values())
        estimates = {"vram": vram, "ram": ram, "vram_required": vtotal, "ram_required": rtotal,
                     "commit_required": vtotal + rtotal, "unknown_costs": "CUDA context/modules, graph capture, CPU TLS/request scratch, driver and OS residency"}
        budget = old["resources"] if "--plan-vram-free-bytes" in argv else None
        short = {name: max(0, total - budget[key]) if budget else 0 for name, total, key in (
            ("vram", vtotal, "vram_free"), ("ram", rtotal, "ram_available"),
            ("commit", vtotal + rtotal, "commit_available"))}
        status = "not_evaluated" if budget is None else "does_not_fit" if any(short.values()) else "within_priced_budget"
        return lp.canonical({"schema": 1, "kind": "native_memory_price", "configuration_complete": True,
            "settings": old["settings"], "estimates": estimates, "budget": budget,
            "fit": {"status": status, "shortfall_bytes": short}})

    def plan(self, request=None, query=None):
        return mp.memory_plan(self.request if request is None else request, query=query or self.native)

    def cli(self, extra=(), *, query=None, request=None, options=None):
        path = self.root / "request.json"
        path.write_bytes(lp.canonical(self.request if request is None else request))
        args = options if options is not None else ["--estimate-memory", "--memory-plan-request", str(path), *extra]
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", ["server", *args]), patch.object(lp, "bounded_query", query or self.native), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = server.main()
            except SystemExit as error:
                code = error.code
        self.assertEqual(err.getvalue(), "")
        return code, json.loads(out.getvalue()), out.getvalue()

    def test_direct_deterministic_indeterminate_and_all_groups(self):
        original = copy.deepcopy(self.request)
        a, b = self.plan(), self.plan()
        self.assertEqual(lp.canonical(a), lp.canonical(b))
        self.assertEqual(self.request, original)
        self.assertEqual(a["kind"], "engine_memory_plan")
        self.assertEqual(a["fit"]["status"], "indeterminate")
        self.assertEqual(a["resolved_config"], a["requested_config"])
        for resource in [*a["resources"]["devices"], a["resources"]["host_ram"], a["resources"]["host_commit"]]:
            self.assertEqual(set(resource["groups"]), set(mp.GROUPS))
            for group in resource["groups"].values():
                self.assertIn(group["quality"], ("exact", "estimated", "unknown"))
                self.assertIn("basis", group)
            self.assertFalse(resource["groups"]["transient_peak"]["additive"])

    def test_only_explicit_applicable_allowances_qualify_fit(self):
        self.request["extensions"]["strata"]["allowances"] = dict.fromkeys(mp.ALLOWANCES, 1000)
        p = self.plan()
        self.assertEqual(p["fit"]["status"], "fits_within_allowances")
        dev = p["resources"]["devices"][0]
        self.assertEqual(dev["required_with_allowances_bytes"], dev["priced_total_bytes"] + 3000)
        del self.request["extensions"]["strata"]["allowances"]["host_commit_runtime"]
        self.assertEqual(self.plan()["fit"]["status"], "indeterminate")

    def test_after_runtime_still_requires_late_and_host_bounds(self):
        self.request["budget"]["basis"] = "after_runtime_initialization"
        p = self.plan()
        costs = p["resources"]["devices"][0]["uncovered_costs"]
        self.assertNotIn("device_runtime_context", costs)
        self.assertIn("device_late_libraries", costs)
        self.assertEqual(p["fit"]["status"], "indeterminate")
        self.request["extensions"]["strata"]["allowances"] = {k: 0 for k in mp.ALLOWANCES if k != "device_runtime_context"}
        self.assertEqual(self.plan()["fit"]["status"], "fits_within_allowances")

    def test_no_budget_explicit_is_not_evaluated(self):
        del self.request["budget"]
        self.assertEqual(self.plan()["fit"]["status"], "not_evaluated")
        self.assertNotIn("--plan-vram-free-bytes", self.calls[0][0])

    def test_cli_normal_not_fit_zero_tiny_and_commit_independent(self):
        for size in (0, 1):
            with self.subTest(size=size):
                req = copy.deepcopy(self.request)
                req["budget"]["devices"][0]["bytes"] = size
                req["budget"]["host_ram_bytes"] = size
                req["budget"]["host_commit_bytes"] = size
                code, p, _ = self.cli(request=req)
                self.assertEqual(code, 0)
                self.assertEqual(p["fit"]["status"], "does_not_fit")
                for resource in [*p["resources"]["devices"], p["resources"]["host_ram"], p["resources"]["host_commit"]]:
                    self.assertGreater(resource["shortfall_bytes"], 0)
        self.request["budget"]["host_commit_bytes"] = 0
        p = self.plan()
        self.assertEqual(p["resources"]["devices"][0]["shortfall_bytes"], 0)
        self.assertGreater(p["resources"]["host_commit"]["shortfall_bytes"], 0)

    def test_auto_selection_preserves_caller_allowances(self):
        self.request["extensions"]["strata"]["cache_slots"] = "auto"
        self.request["extensions"]["strata"]["allowances"] = dict.fromkeys(mp.ALLOWANCES, 1000)
        self.request["budget"]["basis"] = "after_runtime_initialization"
        original = copy.deepcopy(self.request)
        result = self.plan()
        argv = self.calls[0][0]
        for flag, expected in (("vram-free", original["budget"]["devices"][0]["bytes"] - 2000),
                               ("ram-available", original["budget"]["host_ram_bytes"] - 2000),
                               ("commit-available", original["budget"]["host_commit_bytes"] - 2000)):
            self.assertEqual(int(argv[argv.index("--plan-" + flag + "-bytes") + 1]), expected)
        self.assertEqual(result["fit"]["status"], "fits_within_allowances")
        self.assertEqual(result["resources"]["devices"][0]["budget_bytes"],
                         original["budget"]["devices"][0]["bytes"])
        self.assertEqual(self.request, original)

    def test_allowances_larger_than_budget_preserve_normal_shortfall(self):
        self.request["extensions"]["strata"]["allowances"] = dict.fromkeys(mp.ALLOWANCES, 1000)
        self.request["budget"]["devices"][0]["bytes"] = 1
        self.request["budget"]["host_ram_bytes"] = 1
        self.request["budget"]["host_commit_bytes"] = 1
        result = self.plan()
        argv = self.calls[0][0]
        for name in ("vram-free", "ram-available", "commit-available"):
            self.assertEqual(argv[argv.index("--plan-" + name + "-bytes") + 1], "0")
        self.assertEqual(result["fit"]["status"], "does_not_fit")
        device = result["resources"]["devices"][0]
        self.assertEqual(device["shortfall_bytes"], device["priced_total_bytes"] + 3000 - 1)

    def test_native_ring_domain_is_not_an_unrelated_small_integer_range(self):
        for value in ("8", "16", "512"):
            with self.subTest(value=value):
                self.request["extensions"]["strata"]["environment"] = {"STRATA_PREFILL_RING": value}
                self.plan()
                self.assertEqual(self.calls[-1][2]["STRATA_PREFILL_RING"], value)
        for value in ("4", "9", "15", "513"):
            with self.subTest(value=value):
                self.request["extensions"]["strata"]["environment"] = {"STRATA_PREFILL_RING": value}
                with self.assertRaises(mp.MemoryPlanError):
                    self.plan()

    def test_auto_without_budget_refuses_before_query(self):
        for where, field in (("config", "microbatch_size"), ("strata", "cache_slots")):
            req = copy.deepcopy(self.request)
            del req["budget"]
            target = req["config"] if where == "config" else req["extensions"]["strata"]
            target[field] = "auto"
            query = Mock(side_effect=AssertionError("must not query"))
            with self.assertRaisesRegex(mp.MemoryPlanError, "requires_budget"):
                self.plan(req, query)
            query.assert_not_called()

    def test_native_resolves_autos_and_reports_incomplete_minimum(self):
        self.request["config"]["microbatch_size"] = "auto:8192"
        self.request["extensions"]["strata"]["cache_slots"] = "auto"
        p = self.plan()
        self.assertEqual(p["resolved_config"]["microbatch_size"], 2048)
        self.assertEqual(p["requested_config"]["microbatch_size"], "auto:8192")
        self.request["budget"]["devices"][0]["bytes"] = 0
        def incomplete(*args, **kwargs):
            response = json.loads(self.native(*args, **kwargs))
            response.update(configuration_complete=False, settings=None)
            return lp.canonical(response)
        code, p, _ = self.cli(query=incomplete)
        self.assertEqual(code, 0)
        self.assertIsNone(p["resolved_config"])
        self.assertFalse(p["configuration_complete"])
        self.assertEqual(p["pricing_scope"], "minimum_supported_layout")
        self.assertGreater(p["resources"]["devices"][0]["priced_total_bytes"], 0)

    def test_known_conservative_envelope_preserves_allowance_qualification(self):
        self.request["extensions"]["strata"]["allowances"] = dict.fromkeys(mp.ALLOWANCES, 0)
        def envelope(*args, **kwargs):
            response = json.loads(self.native(*args, **kwargs))
            response["estimates"]["unknown_costs"] = mp.NATIVE_ENVELOPE_UNKNOWNS
            response["estimate_basis"] = "resolved_cache_layout_with_conservative_prefill_envelope"
            return lp.canonical(response)
        result = self.plan(query=envelope)
        self.assertEqual(result["fit"]["status"], "fits_within_allowances")
        device = result["resources"]["devices"][0]
        self.assertEqual(device["combined_allocations"][-1]["quality"], "estimated")
        self.assertIn("conservative_prefill_envelope", result["extensions"]["strata"]["estimate_basis"])

    def test_minimum_selection_policy_is_not_an_allocation(self):
        self.request["extensions"]["strata"]["cache_slots"] = "auto"
        def policy_limited(*args, **kwargs):
            response = json.loads(self.native(*args, **kwargs))
            capacity = response["budget"]["vram_free"]
            self.assertLess(response["estimates"]["vram_required"], capacity)
            response["estimates"]["minimum_policy_vram_free_bytes"] = capacity + 1
            response.update(configuration_complete=False, settings=None)
            response["fit"]["status"] = "does_not_fit"
            response["fit"]["shortfall_bytes"]["vram"] = 1
            return lp.canonical(response)
        code, result, _ = self.cli(query=policy_limited)
        self.assertEqual(code, 0)
        self.assertEqual(result["fit"]["status"], "does_not_fit")
        device = result["resources"]["devices"][0]
        self.assertEqual(device["shortfall_bytes"], 1)
        self.assertEqual(device["minimum_selection_capacity_bytes"], device["budget_bytes"] + 1)
        native = result["extensions"]["strata"]["native_components"]
        self.assertEqual(device["priced_total_bytes"], sum(native["vram"].values()))
        self.assertLess(device["priced_total_bytes"], device["budget_bytes"])

    def test_unknown_partitions_are_not_guessed_or_double_counted(self):
        p = self.plan()
        dev = p["resources"]["devices"][0]
        for name in ("kv_cache", "compute_scratch", "output_buffers"):
            self.assertIsNone(dev["groups"][name]["bytes"])
            self.assertEqual(dev["groups"][name]["quality"], "unknown")
        e = p["extensions"]["strata"]["native_components"]
        self.assertEqual(dev["priced_total_bytes"], sum(e["vram"].values()))
        self.assertEqual(dev["allocated_bytes"] + self.request["extensions"]["strata"]["vram_reserve_bytes"], dev["priced_total_bytes"])
        ram = p["resources"]["host_ram"]
        self.assertEqual(ram["mapped_view_bytes"], 20)
        self.assertEqual(ram["allocated_bytes"] + ram["mapped_view_bytes"] + sum(x["bytes"] for x in ram["assumptions"]), ram["priced_total_bytes"])

    def test_metadata_identity_and_caller_digest_never_claim_content_verified(self):
        self.request["model"]["content_sha256"] = "ab" * 32
        p = self.plan()
        self.assertFalse(p["cacheable"])
        self.assertIsNone(p["model"]["content_sha256"])
        self.assertEqual(p["model"]["asserted_content_sha256"], "ab" * 32)
        files = p["extensions"]["strata"]["inputs"]
        gguf = next(f for f in files if f["path"].endswith("model.gguf"))
        self.assertIsNone(gguf["content_sha256"])
        self.assertEqual(gguf["metadata_bytes"], 24)
        self.assertIsNotNone(gguf["metadata_sha256"])
        self.assertIsNotNone(p["engine"]["executable_sha256"])

    def test_exact_query_argv_and_isolated_environment(self):
        self.request["config"]["devices"][0]["id"] = 2
        self.request["budget"]["devices"][0]["id"] = 2
        self.request["extensions"]["strata"]["environment"] = {"STRATA_STAGER_RING": "32"}
        with patch.dict(os.environ, {"STRATA_API_KEY": "secret", "CUDA_VISIBLE_DEVICES": "8", "STRATA_UNPRICED": "1"}):
            self.plan()
        argv, cwd, env = self.calls[0]
        self.assertEqual(argv, [str(self.exe), "--resolve-launch", "--price-only", "--pack", str(self.root),
            "--native", str(self.root / "model.gguf"), "--expert-profile", str(self.root / "profile.bin"),
            "--max-context", "8192", "--prefill", "1024", "--kv", "int8", "--spec", "2", "--kv-resident", "0",
            "--resident-budget-gib", "1", "--expert-cache", "2", "--vram-reserve-mib", "1", "--no-prefill-borrow",
            "--plan-vram-free-bytes", str(1 << 30), "--plan-ram-available-bytes", str(8 << 30),
            "--plan-commit-available-bytes", str(16 << 30)])
        self.assertEqual(cwd, str(self.root))
        self.assertEqual({k: v for k, v in env.items() if k not in ("PATH", "SYSTEMROOT", "TEMP", "TMP")},
                         {"CUDA_VISIBLE_DEVICES": "2", "STRATA_FILE_CACHE_MIB": "0", "STRATA_RESIDENT_HEADROOM_GIB": "0",
                          "STRATA_STAGER_RING": "32"})

    def test_bad_numeric_unknown_device_batch_and_unsupported_fields(self):
        cases = [("config", "context", True), ("config", "context", -1), ("config", "context", 2**64),
            ("config", "context", 1.5), ("config", "microbatch_size", 0), ("config", "microbatch_size", 8193),
            ("config", "parallel_slots", 2), ("config", "batch_size", 2), ("config", "kv_type", "bad"),
            ("config", "devices", []), ("config", "devices", [{"id": 0, "backend": "hip"}]),
            ("config", "devices", [{"id": "auto", "backend": "cuda"}]),
            ("config", "offload", {"mode": "hybrid", "layers": 2}),
            ("config", "unsupported", 1), ("strata", "cache_slots", 0), ("strata", "spec", 3),
            ("strata", "no_prefill_borrow", 1), ("strata", "resident_budget_bytes", 2**64),
            ("strata", "vram_reserve_bytes", 1), ("strata", "environment", {"STRATA_API_KEY": "secret"}),
            ("strata", "environment", {"STRATA_STAGER_RING": "0"}),
            ("strata", "allowances", {"unknown": 1})]
        for where, key, value in cases:
            with self.subTest(where=where, key=key, value=value):
                req = copy.deepcopy(self.request)
                target = req["config"] if where == "config" else req["extensions"]["strata"]
                target[key] = value
                query = Mock(side_effect=AssertionError("must not query"))
                with self.assertRaises(mp.MemoryPlanError):
                    self.plan(req, query)
                query.assert_not_called()
        for key, value in (("host_ram_bytes", -1), ("host_commit_bytes", 2**64), ("basis", "auto")):
            req = copy.deepcopy(self.request)
            req["budget"][key] = value
            with self.assertRaises(mp.MemoryPlanError):
                self.plan(req)

    def test_defaults_and_alias_are_separate_from_request(self):
        self.request["config"]["kv_type"] = "q8_0"
        del self.request["config"]["batch_size"]
        p = self.plan()
        self.assertEqual(p["resolved_config"]["kv_type"], "int8")
        self.assertEqual(p["requested_config"]["kv_type"], "q8_0")
        self.assertEqual(p["resolved_config"]["batch_size"], 1)
        self.assertNotIn("batch_size", p["requested_config"])

    def test_cli_errors_are_json_sanitized_with_abbreviations_and_conflicts(self):
        for extra in (("--config", "private-secret"), ("--api-key", "private-secret"), ("--engine", "mock"),
                      ("--port", "private-secret"), ("--launch-plan", "private-secret"), ("--check-install",),
                      ("--print-launch-plan",), ("--unknown", "private-secret")):
            code, p, raw = self.cli(extra)
            self.assertEqual(code, 2)
            self.assertIn("error", p)
            self.assertNotIn("private-secret", raw)
        path = self.root / "request.json"
        for options in (["--memory-plan-request", str(path)],
                        ["--estimate-mem", "--memory-plan-req", str(path), "--port", "private-secret"],
                        ["--memory-plan-req=private-secret", "--estimate-memory", "--bad"]):
            code, p, raw = self.cli(options=options)
            self.assertEqual(code, 2)
            self.assertIn("error", p)
            self.assertNotIn("private-secret", raw)
        code, p, _ = self.cli(options=["--estimate-mem", "--memory-plan-req", str(path)])
        self.assertEqual(code, 0)
        self.assertEqual(p["kind"], "engine_memory_plan")

    def test_safe_errors_from_missing_files_native_text_and_unknown_field_names(self):
        for query in (Mock(side_effect=OSError("private-secret/path")),
                      Mock(side_effect=lp.PlanError("native private-secret --argv")),
                      Mock(return_value=b"not-json private-secret")):
            code, _, raw = self.cli(query=query)
            self.assertEqual(code, 2)
            self.assertNotIn("private-secret", raw)
        self.request["extensions"]["strata"]["native_model"] = str(self.root / "private-secret")
        code, _, raw = self.cli()
        self.assertEqual(code, 2)
        self.assertNotIn("private-secret", raw)
        req = copy.deepcopy(self.request)
        req["config"]["unsafe/private-secret"] = 1
        code, p, raw = self.cli(request=req)
        self.assertEqual(code, 2)
        self.assertEqual(p["error"]["field"], "config.unknown_field")
        self.assertNotIn("private-secret", raw)

    def test_changed_identity_and_bad_native_totals_refuse(self):
        def changed(*args, **kwargs):
            result = self.native(*args, **kwargs)
            (self.root / "index.txt").write_bytes(b"changed metadata")
            return result
        with self.assertRaisesRegex(mp.MemoryPlanError, "inputs_changed"):
            self.plan(query=changed)
        def bad(*args, **kwargs):
            result = json.loads(self.native(*args, **kwargs))
            result["estimates"]["vram_required"] = 0
            return lp.canonical(result)
        with self.assertRaises(mp.MemoryPlanError):
            self.plan(query=bad)

    def test_unrecognized_native_unknown_costs_cannot_be_covered(self):
        self.request["extensions"]["strata"]["allowances"] = dict.fromkeys(mp.ALLOWANCES, 0)
        def extra_unknown(*args, **kwargs):
            result = json.loads(self.native(*args, **kwargs))
            result["estimates"]["unknown_costs"] += ", new cost"
            return lp.canonical(result)
        p = self.plan(query=extra_unknown)
        self.assertEqual(p["fit"]["status"], "indeterminate")
        self.assertIn("unclassified_native_costs", p["resources"]["host_commit"]["uncovered_costs"])

    def test_pure_routing_cannot_touch_live_or_startup_owners(self):
        with contextlib.ExitStack() as stack:
            for obj, name in ((lp, "produce_plan"), (lp, "observe_hardware"),
                              (server, "resolve_credentials"), (server, "api_key_of"),
                              (server, "StrataEngine"), (server, "MockEngine"), (server, "ChatTemplate"),
                              (server, "ByteTokenizer"), (server, "Server"), (server, "Service"),
                              (server, "ThreadingHTTPServer"), (server, "hub_from_config")):
                stack.enter_context(patch.object(obj, name, side_effect=AssertionError("forbidden " + name)))
            stack.enter_context(patch.object(server, "popen", side_effect=AssertionError("no real process")))
            code, p, raw = self.cli()
        self.assertEqual(code, 0)
        self.assertEqual(p["kind"], "engine_memory_plan")
        self.assertEqual(raw, lp.canonical(p).decode() + "\n")


if __name__ == "__main__":
    unittest.main()
