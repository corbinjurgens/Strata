"""Offline safety/measurement checks; never loads a model or contacts a server."""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import benchmark_compact_cache as bench


class HelperTests(unittest.TestCase):
    def test_snapshot_replacement_survives_a_temporary_reader_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            pending, destination = Path(folder) / "pending.json", Path(folder) / "result.json"
            pending.write_text('{"state":"new"}')
            destination.write_text('{"state":"old"}')
            real_replace = Path.replace
            calls = []

            def locked_once(source, target):
                calls.append(source)
                self.assertEqual(json.loads(destination.read_text()), {"state": "old"})
                if len(calls) == 1:
                    raise PermissionError("simulated reader sharing lock")
                return real_replace(source, target)

            with patch.object(Path, "replace", locked_once), patch.object(bench.time, "sleep"):
                bench.replace_result(pending, destination)
            self.assertEqual(json.loads(destination.read_text()), {"state": "new"})
            self.assertFalse(pending.exists())
            self.assertEqual(len(calls), 2)

    def test_persistent_snapshot_denial_keeps_prior_data_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            pending, destination = Path(folder) / "pending.json", Path(folder) / "result.json"
            pending.write_text("new")
            destination.write_text("old")
            with patch.object(Path, "replace", side_effect=PermissionError("persistent denial")) as replace, \
                    patch.object(bench.time, "sleep") as sleep:
                with self.assertRaisesRegex(PermissionError, "persistent denial"):
                    bench.replace_result(pending, destination)
            self.assertEqual(destination.read_text(), "old")
            self.assertEqual(pending.read_text(), "new")
            self.assertLessEqual(replace.call_count, 10)
            self.assertLessEqual(sum(call.args[0] for call in sleep.call_args_list), 1)

    def test_profiling_and_conflicting_auth_are_removed_from_environment(self):
        cfg = {"env": {"STRATA_VERIFY_PROFILE": "1", "strata_decode_timing": "9",
                       "STRATA_FETCH_PROFILE": "1",
                       "STRATA_API_KEY": "engine-only-key", "KEEP": "yes"},
               "args": ["--max-context", "131072"], "api_key": "example-secret"}
        with patch.dict(bench.os.environ, {"STRATA_VERIFY_PROFILE": "1", "STRATA_DECODE_TIMING": "1",
                                          "STRATA_FETCH_PROFILE": "1",
                                          "STRATA_API_KEY": "different-inherited-key"}):
            clean, env = bench.clean_environment(cfg, False)
            timed, timed_env = bench.clean_environment(cfg, True)
            self.assertEqual(bench.os.environ["STRATA_API_KEY"], "different-inherited-key")
        self.assertEqual(clean["env"], {"KEEP": "yes"})
        self.assertNotIn("STRATA_VERIFY_PROFILE", env)
        self.assertNotIn("STRATA_DECODE_TIMING", env)
        self.assertNotIn("STRATA_FETCH_PROFILE", env)
        self.assertEqual(timed["env"], {"KEEP": "yes", "STRATA_DECODE_TIMING": "1"})
        self.assertNotIn("STRATA_VERIFY_PROFILE", timed_env)
        self.assertNotIn("STRATA_DECODE_TIMING", timed_env)
        self.assertNotIn("STRATA_API_KEY", env)
        self.assertNotIn("STRATA_API_KEY", timed_env)
        self.assertEqual(clean["args"], ["--max-context", "131072"])
        self.assertEqual(clean["api_key"], "example-secret")
        self.assertEqual(cfg["env"]["strata_decode_timing"], "9")

    def test_fetch_profiling_is_explicit_and_does_not_enable_decode_timers(self):
        cfg, env = bench.clean_environment({"env": {"STRATA_DECODE_TIMING": "1"}}, False, True)
        self.assertEqual(cfg["env"], {"STRATA_FETCH_PROFILE": "1"})
        self.assertNotIn("STRATA_FETCH_PROFILE", env)
        self.assertNotIn("STRATA_DECODE_TIMING", env)

    def test_safe_evidence_merges_inherited_controls_with_config_precedence(self):
        inherited = {"STRATA_FILE_CACHE_MIB": "16384", "STRATA_FILE_DIRECT_UNMAP": "1",
                     "STRATA_UNBUFFERED_LOAD": "0", "STRATA_FETCH_PVM": "1",
                     "HF_TOKEN": "secret", "STRATA_API_KEY": "secret"}
        cfg = {"env": {"STRATA_FILE_CACHE_MIB": "4096", "STRATA_OWNED_PRICE": "exact"}}
        evidence = bench.safe_engine_environment(cfg, inherited)
        self.assertEqual(evidence, {"STRATA_FILE_CACHE_MIB": "4096", "STRATA_FILE_DIRECT_UNMAP": "1",
                                   "STRATA_UNBUFFERED_LOAD": "0", "STRATA_FETCH_PVM": "1",
                                   "STRATA_OWNED_PRICE": "exact"})

    def test_safe_evidence_obeys_platform_environment_case_rules(self):
        inherited = {"STRATA_FILE_CACHE_MIB": "64"}
        cfg = {"env": {"strata_file_cache_mib": "128"}}
        with patch.object(bench.os, "name", "nt"):
            self.assertEqual(bench.safe_engine_environment(cfg, inherited), {"STRATA_FILE_CACHE_MIB": "128"})
        with patch.object(bench.os, "name", "posix"):
            self.assertEqual(bench.safe_engine_environment(cfg, inherited), {"STRATA_FILE_CACHE_MIB": "64"})

    def test_safe_evidence_honors_saved_and_prepared_removals_without_child_side_effects(self):
        from serve import server
        key = "STRATA_RESIDENT_HEADROOM_GIB"
        for removal in ("null", "delete", "prepared"):
            with self.subTest(removal=removal):
                cfg = {"backend": "hip", "lib_dirs": ["unused"], "env": {}, "memory_policy": {
                    "version": 1, "generated_env": {key: "4"}, "overrides": {"env": {}}}}
                if removal == "null":
                    cfg["env"][key] = None
                elif removal == "prepared":
                    cfg["memory_policy"]["generated_env"] = {}
                    cfg["memory_policy"]["overrides"]["env"][key] = None
                original = json.dumps(cfg, sort_keys=True)
                with patch.object(server, "child_env", side_effect=AssertionError("full child setup")), \
                        patch.object(server, "hip_env_guard", side_effect=AssertionError("HIP directory probes")), \
                        patch.object(server, "gpu_list", side_effect=AssertionError("GPU composition")), \
                        patch.object(server, "_dir_writable", side_effect=AssertionError("directory probe")):
                    self.assertEqual(bench.safe_engine_environment(cfg, {key: "8", "STRATA_FILE_DIRECT_UNMAP": "1"}),
                                     {"STRATA_FILE_DIRECT_UNMAP": "1"})
                self.assertEqual(json.dumps(cfg, sort_keys=True), original)

    def test_summary_separates_prefill_and_uses_total_decode_time(self):
        rows = [{"response": {"usage": {"completion_tokens": count, "prompt_tokens": 80},
                             "timings": {"predicted_ms": ms, "prompt_ms": prefill}}}
                for count, ms, prefill in ((10, 1000, 4000), (90, 3000, 5000))]
        self.assertEqual(bench.summarize(rows), {"completion_tokens": 100, "predicted_ms": 4000,
                         "tokens_per_second": 25, "prompt_ms": 9000, "prompt_tokens": 160})

    def test_held_out_checks_reject_incorrect_missing_and_extra_answers(self):
        self.assertTrue(bench.held_out_answer("Explanation.\nFINAL: [4,18,12]", [4, 18, 12]))
        for text in ("FINAL: [4,8,12]", "[4,18,12]", "FINAL: [4,18,12] more", "FINAL: [4,18]"):
            self.assertFalse(bench.held_out_answer(text, [4, 18, 12]))
        self.assertFalse(bench.held_out_answer("FINAL: 8", 9))

    def test_held_out_checks_require_exact_json_types_including_nested_values(self):
        for text, expected in (("FINAL: false", 0), ("FINAL: 9.0", 9),
                               ('FINAL: {"nested": [false, 7]}', {"nested": [0, 7]})):
            self.assertFalse(bench.held_out_answer(text, expected))
        self.assertTrue(bench.held_out_answer('FINAL: {"nested": [0, 7]}', {"nested": [0, 7]}))

    def test_exact_answers_allow_presentation_punctuation_and_reject_numeric_or_code_extensions(self):
        for text in ("cove-lantern-517 391", "`cove-lantern-517`, 391.", "[\"cove-lantern-517\", 391]"):
            with self.subTest(text=text):
                self.assertTrue(bench.exact_answer(text, ["cove-lantern-517", 391]))
        for text in ("cove-lantern-517 391.5", "cove-lantern-517 391/2", "cove-lantern-5170 391",
                     "prefix-cove-lantern-517 391", "cove-lantern-517 1391", "cove-lantern-517 391e2",
                     "cove-lantern-517 +391", "cove-lantern-517 391 extra", "391 cove-lantern-517"):
            with self.subTest(text=text):
                self.assertFalse(bench.exact_answer(text, ["cove-lantern-517", 391]))
        codes = ["field-cedar-284", "marble-finch-693", "cove-lantern-517"]
        self.assertTrue(bench.exact_answer("field-cedar-284, marble-finch-693, cove-lantern-517.", codes))
        for text in ("field-cedar-2840, marble-finch-693, cove-lantern-517",
                     "field-cedar-284, cove-lantern-517, marble-finch-693",
                     "field-cedar-284, marble-finch-693, cove-lantern-5170",
                     "field-cedar-284, marble-finch-693, cove-lantern-517, cove-lantern-517"):
            with self.subTest(text=text):
                self.assertFalse(bench.exact_answer(text, codes))

    def test_prompt_counter_uses_complete_chat_template_and_thinking_disabled(self):
        from serve import server, frontend
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "vocab.json").write_text('{"second": 1, "first": 0}')
            (root / "merges.txt").write_text("first second\n")
            (root / "token_type.json").write_text("[1, 2]")
            tokenizer = types.SimpleNamespace(control_tokens=[])
            factory = Mock(return_value=tokenizer)
            module = types.SimpleNamespace(Tokenizer=factory)
            messages = [{"role": "system", "content": "reference"}, {"role": "user", "content": "long text"},
                        {"role": "assistant", "content": "three codes"}, {"role": "user", "content": "followup"}]
            with patch.dict(sys.modules, {"strata_tokenizer": module}), \
                    patch.object(frontend, "ChatTemplate") as template, \
                    patch.object(server.Service, "encode_prompt", autospec=True, return_value=[0] * 37) as encode:
                count = bench.prompt_counter({"tokenizer": str(root)})(messages)
            self.assertEqual(count, 37)
            factory.assert_called_once_with(["first", "second"], ["first second", ""], [1, 2])
            template.assert_called_once_with(root / "chat_template.jinja")
            self.assertEqual(encode.call_args.args[1:], (messages, None, {"enable_thinking": False}))
            self.assertIs(encode.call_args.args[0].tok, tokenizer)
            self.assertFalse(encode.call_args.args[0].effort_end)

    def test_authentication_checks_both_missing_and_wrong_keys(self):
        record, save = {}, Mock()
        client = types.SimpleNamespace(base="http://127.0.0.1:8081")
        errors = [bench.urllib.error.HTTPError(client.base, 401, "Unauthorized", {}, io.BytesIO())
                  for _ in range(2)]
        with patch.object(bench, "TimeoutOpener") as opener:
            opener.return_value.open.side_effect = errors
            bench.check_authentication(client, record, save)
        self.assertEqual(record["authentication"], [
            {"credentials": "missing", "status": 401, "passed": True},
            {"credentials": "wrong", "status": 401, "passed": True}])
        calls = opener.return_value.open.call_args_list
        self.assertEqual(calls[0].args[0].headers, {})
        self.assertEqual(calls[1].args[0].headers, {"Authorization": "Bearer benchmark-invalid-key"})
        self.assertEqual(save.call_count, 2)

    def test_authentication_acceptance_and_wrong_error_code_fail_with_evidence(self):
        client = types.SimpleNamespace(base="http://127.0.0.1:8081")
        for status in (200, 403, 500):
            record, save = {}, Mock()
            with self.subTest(status=status), patch.object(bench, "TimeoutOpener") as opener:
                if status == 200:
                    opener.return_value.open.return_value.__enter__.return_value.status = 200
                else:
                    opener.return_value.open.side_effect = bench.urllib.error.HTTPError(
                        client.base, status, "wrong status", {}, io.BytesIO())
                with self.assertRaisesRegex(RuntimeError, "Authentication validation failed"):
                    bench.check_authentication(client, record, save)
            self.assertFalse(record["authentication"][0]["passed"])
            self.assertEqual(record["authentication"][0]["status"], status)
            save.assert_called_once_with()

    def test_working_tree_identity_detects_dirty_source_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "src").mkdir()
            source = root / "src/cache.cpp"
            source.write_text("old")

            def git(command, **kwargs):
                if "ls-files" in command:
                    return types.SimpleNamespace(stdout=b"src/cache.cpp\0")
                if "rev-parse" in command:
                    return types.SimpleNamespace(stdout=b"abc123\n")
                return types.SimpleNamespace(stdout=b" M src/cache.cpp\n")

            with patch.object(bench, "ROOT", root), patch.object(bench.subprocess, "run", side_effect=git):
                first = bench.source_identity()
                source.write_text("new")
                second = bench.source_identity()
            self.assertEqual(first["head"], "abc123")
            self.assertNotEqual(first["sha256"], second["sha256"])
            self.assertNotEqual(first["files"][0]["sha256"], second["files"][0]["sha256"])
            self.assertIn("not executable build provenance", first["meaning"])

    def test_existing_engine_or_python_server_is_refused_without_termination(self):
        for info in ({"pid": -1, "name": "strata-unsloth.exe", "cmdline": []},
                     {"pid": -2, "name": "python.exe", "cmdline": ["python", "C:\\Tools\\Strata\\serve\\server.py"]},
                     {"pid": -3, "name": "python", "cmdline": ["python", "-m", "serve.server"]}):
            process = Mock(info=info)
            with self.subTest(info=info), patch.object(bench.psutil, "process_iter", return_value=[process]), \
                    patch.object(bench.socket, "socket") as socket_factory:
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    bench.assert_idle()
                process.terminate.assert_not_called()
                process.kill.assert_not_called()
                socket_factory.assert_not_called()

    def test_occupied_port_is_refused(self):
        with patch.object(bench.psutil, "process_iter", return_value=[]), \
                patch.object(bench.socket, "socket") as socket_factory:
            probe = socket_factory.return_value.__enter__.return_value
            probe.bind.side_effect = OSError("occupied")
            with self.assertRaisesRegex(RuntimeError, "Port 8081 is occupied"):
                bench.assert_idle()
            probe.bind.assert_called_once_with(("127.0.0.1", 8081))

    def test_memory_guard_stops_immediately_below_emergency_floor(self):
        owner = Mock()
        guard = bench.MemoryGuard(owner, Path("unused"))
        guard.sample(1024**3 - 1, 0)
        owner.stop.assert_called_once()
        self.assertEqual(guard.samples, 1)
        with self.assertRaisesRegex(RuntimeError, "1 GiB emergency"):
            guard.check()
        guard.sample(0, 0)
        owner.stop.assert_called_once()

    def test_memory_guard_requires_five_consecutive_low_samples(self):
        owner = Mock()
        guard = bench.MemoryGuard(owner, Path("unused"))
        for available in [1, 1, 1, 1, 3, 1, 1, 1, 1]:
            guard.sample(available * 1024**3, 0)
        owner.stop.assert_not_called()
        guard.sample(1 * 1024**3, 0)
        owner.stop.assert_called_once_with()
        self.assertEqual(guard.summary()["samples"], 10)
        self.assertEqual(guard.summary()["min_available_gib"], 1)
        with self.assertRaisesRegex(RuntimeError, "five consecutive"):
            guard.check()

    def test_unavailable_pagefile_does_not_stop_healthy_ram_monitoring(self):
        owner = Mock()
        guard = bench.MemoryGuard(owner, Path("unused"))
        guard.done = Mock()
        guard.done.is_set.side_effect = [False, True]
        with patch.object(bench.psutil, "virtual_memory", return_value=types.SimpleNamespace(available=3 * 1024**3)), \
                patch.object(bench.psutil, "swap_memory", side_effect=RuntimeError(
                    "PdhAddEnglishCounterW failed. Performance counters may be disabled.")):
            guard.monitor()
        owner.stop.assert_not_called()
        guard.check()
        self.assertEqual(guard.summary()["samples"], 1)
        self.assertEqual(guard.summary()["min_available_gib"], 3)
        self.assertIsNone(guard.summary()["max_pagefile_used_gib"])
        self.assertFalse(guard.summary()["pagefile_available"])
        self.assertIn("PdhAddEnglishCounterW failed", guard.summary()["pagefile_unavailable_reason"])

    def test_unavailable_pagefile_preserves_five_sample_low_ram_cancellation(self):
        owner = Mock()
        guard = bench.MemoryGuard(owner, Path("unused"))
        guard.done = Mock()
        guard.done.is_set.return_value = False
        samples = [types.SimpleNamespace(available=n * 1024**3) for n in (3, 1, 1, 1, 1, 1)]
        with patch.object(bench.psutil, "virtual_memory", side_effect=samples) as ram, \
                patch.object(bench.psutil, "swap_memory", side_effect=RuntimeError("Counters disabled")) as swap:
            guard.monitor()
        owner.stop.assert_called_once_with()
        self.assertEqual(ram.call_count, 6)
        swap.assert_called_once_with()
        self.assertEqual(guard.summary()["samples"], 6)
        self.assertEqual(guard.summary()["min_available_gib"], 1)
        self.assertIsNone(guard.summary()["max_pagefile_used_gib"])
        with self.assertRaisesRegex(RuntimeError, "five consecutive"):
            guard.check()

    def test_unavailable_required_ram_telemetry_still_stops_run(self):
        owner = Mock()
        guard = bench.MemoryGuard(owner, Path("unused"))
        with patch.object(bench.psutil, "virtual_memory", side_effect=RuntimeError("RAM failed")):
            guard.monitor()
        owner.stop.assert_called_once_with()
        self.assertEqual(guard.summary()["samples"], 0)
        self.assertIsNone(guard.summary()["min_available_gib"])
        with self.assertRaisesRegex(RuntimeError, "Memory/process monitor failed"):
            guard.check()

    def test_cleanup_only_terminates_owned_descendants_and_is_idempotent(self):
        root, child, unrelated = Mock(), Mock(), Mock()
        root.children.return_value = [child]
        root.is_running.return_value = True
        with patch.object(bench.psutil, "Process", return_value=root), \
                patch.object(bench.psutil, "wait_procs", side_effect=[([root], [child]), ([child], [])]) as wait:
            owner = bench.OwnedProcess(Mock(pid=123))
            owner.stop()
            owner.stop()
        root.terminate.assert_called_once_with()
        child.terminate.assert_called_once_with()
        child.kill.assert_called_once_with()
        unrelated.terminate.assert_not_called()
        unrelated.kill.assert_not_called()
        self.assertEqual(set(wait.call_args_list[0].args[0]), {root, child})

    def test_metrics_match_new_record_not_list_position(self):
        old = {"time": 1, "output_tokens": 160, "prompt_tokens": 12, "decode_ms": 1000}
        new = {"time": 2, "output_tokens": 160, "prompt_tokens": 12, "decode_ms": 2000}
        response = {"usage": {"completion_tokens": 160, "prompt_tokens": 12},
                    "timings": {"predicted_ms": 2000}}
        before = {"requests": [old], "totals": {"requests": 1}}
        after = {"requests": [old, new], "totals": {"requests": 2}}
        self.assertEqual(bench.matched_metric(response, before, after), new)
        after["totals"]["requests"] = 3
        with self.assertRaisesRegex(RuntimeError, "uniquely match"):
            bench.matched_metric(response, before, after)
        after["totals"]["requests"] = 2
        new["decode_ms"] = 9000
        with self.assertRaisesRegex(RuntimeError, "does not agree"):
            bench.matched_metric(response, before, after)

    def test_capture_retains_bounded_head_and_tail(self):
        capture = bench.BoundedCapture()
        capture.write(b"start" + b"a" * (bench.LOG_LIMIT * 2) + b"end")
        text = capture.text()
        self.assertTrue(text.startswith("start"))
        self.assertTrue(text.endswith("end"))
        self.assertIn("[log middle omitted]", text)
        self.assertLess(len(text), bench.LOG_LIMIT + 100)

    def test_no_proxy_opener_caps_validation_timeout(self):
        with patch.object(bench.urllib.request, "build_opener") as factory:
            opener = bench.TimeoutOpener()
            opener.open("fake-request", timeout=7200)
        self.assertEqual(factory.call_args.args[0].proxies, {})
        factory.return_value.open.assert_called_once_with("fake-request", timeout=1800)


class AcceptanceTelemetryTests(unittest.TestCase):
    def test_actual_monitor_keeps_emergency_reason_when_owned_cleanup_fails(self):
        from tools import prepare_unsloth_q2 as preparation
        guard = self.guard()
        guard.owner.stop.side_effect = RuntimeError("owned survivor")
        with patch.object(preparation, "memory_snapshot", return_value={
                "available_gib": 1, "commit_available_gib": 9}), \
                patch.object(bench.time, "monotonic", side_effect=[1, 1.1]):
            guard.monitor()
        self.assertIn("2 GiB emergency", guard.reason)
        self.assertEqual(guard.cleanup_error, "RuntimeError: owned survivor")
        result = guard.acceptance_summary(1, 2)
        self.assertEqual(result["host_evidence"], "refused")
        self.assertEqual(result["cleanup_error"], guard.cleanup_error)
        self.assertEqual(result["cleanup"], "pending")
        with self.assertRaisesRegex(RuntimeError, "2 GiB emergency"):
            guard.check()
        self.sample(guard, 8, 8, 2)
        guard.owner.stop.assert_called_once_with()

    def test_actual_monitor_read_error_survives_owned_cleanup_failure(self):
        from tools import prepare_unsloth_q2 as preparation
        guard = self.guard()
        guard.owner.stop.side_effect = RuntimeError("cleanup failed")
        with patch.object(preparation, "memory_snapshot", side_effect=OSError("unreadable")), \
                patch.object(bench.time, "monotonic", return_value=1):
            guard.monitor()
        self.assertEqual(guard.reason, "Memory/process monitor failed: OSError")
        self.assertEqual(guard.cleanup_error, "RuntimeError: cleanup failed")
        guard.owner.stop.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "monitor failed: OSError"):
            guard.check()

    def test_legacy_guard_cleanup_failure_still_propagates(self):
        guard = bench.MemoryGuard(Mock(), Path("unused"))
        guard.owner.stop.side_effect = RuntimeError("legacy cleanup failure")
        with self.assertRaisesRegex(RuntimeError, "legacy cleanup failure"):
            guard.sample(512 * 1024**2, None)
        self.assertEqual(guard.reason, "Available RAM fell below the 1 GiB emergency floor.")
        self.assertIsNone(guard.cleanup_error)
        guard.owner.stop.assert_called_once_with()

    def guard(self, floors=(4, 4)):
        return bench.MemoryGuard(Mock(), Path("unused"), acceptance_floors=floors)

    def sample(self, guard, physical=4, commit=4, start=1, end=None):
        guard.acceptance_sample({"available_gib": physical, "commit_available_gib": commit},
                                start, start if end is None else end)

    def test_exact_floors_raw_bytes_independent_minimum_timestamps_and_pending_lifecycle(self):
        guard = self.guard()
        self.sample(guard, 4, 7, 1)
        self.sample(guard, 6, 4, 2)
        result = guard.acceptance_summary(1, 2)
        self.assertEqual(result["host_evidence"], "satisfied")
        self.assertEqual(result["minima"]["physical"], {
            "available_bytes": 4 * 1024**3, "started_monotonic": 1, "finished_monotonic": 1})
        self.assertEqual(result["minima"]["commit"]["finished_monotonic"], 2)
        self.assertEqual(result["max_gap_seconds"], 1)
        self.assertEqual(result["raw_samples"][1]["commit_available_bytes"], 4 * 1024**3)
        self.assertEqual(result["lifecycle_acceptance"], "pending")
        self.assertEqual(result["cleanup"], "pending")
        self.assertEqual(result["finalization"], "pending")
        self.assertIn("missing", result["cuda_stage_evidence"])
        self.assertEqual(result["cuda_floor_mib"], 256)
        guard.owner.stop.assert_not_called()

    def test_configured_floors_are_separate_from_emergency(self):
        for physical, commit in ((4 - 1 / 1024**3, 8), (8, 4 - 1 / 1024**3)):
            with self.subTest(physical=physical, commit=commit):
                guard = self.guard()
                self.sample(guard, physical, commit, 1)
                self.sample(guard, physical, commit, 2)
                self.assertEqual(guard.acceptance_summary(1, 2)["host_evidence"], "refused")
                guard.check()  # Acceptance refusal is distinct from the original emergency cancellation.
                guard.owner.stop.assert_not_called()

    def test_each_immediate_emergency_dimension_and_exact_boundary(self):
        for physical, commit in ((2 - 1 / 1024**3, 9), (9, 2 - 1 / 1024**3)):
            guard = self.guard()
            self.sample(guard, physical, commit)
            guard.owner.stop.assert_called_once_with()
            with self.assertRaisesRegex(RuntimeError, "2 GiB emergency"):
                guard.check()
        guard = self.guard()
        self.sample(guard, 2, 2)
        guard.owner.stop.assert_not_called()

    def test_original_five_sample_rule_and_reset(self):
        for physical, commit in ((2.5 - 1 / 1024**3, 9), (9, 4 - 1 / 1024**3)):
            guard = self.guard()
            for index in range(4):
                self.sample(guard, physical, commit, index)
            self.sample(guard, 2.5, 4, 4)  # Exact boundaries reset both dimensions.
            for index in range(5, 9):
                self.sample(guard, physical, commit, index)
            guard.owner.stop.assert_not_called()
            self.sample(guard, physical, commit, 9)
            guard.owner.stop.assert_called_once_with()
            with self.assertRaisesRegex(RuntimeError, "five consecutive"):
                guard.check()

    def test_missing_invalid_nonfinite_and_bool_telemetry_fail_closed(self):
        cases = ({}, {"available_gib": 8}, {"available_gib": None, "commit_available_gib": 8})
        cases += tuple({"available_gib": value, "commit_available_gib": 8}
                       for value in (True, "8", -1, float("nan"), float("inf"), 0.1 / 1024**3))
        cases += tuple({"available_gib": 8, "commit_available_gib": value}
                       for value in (None, False, float("nan"), float("inf")))
        for snapshot in cases:
            with self.subTest(snapshot=snapshot):
                guard = self.guard()
                guard.acceptance_sample(snapshot, 1, 1)
                guard.owner.stop.assert_called_once_with()
                self.assertEqual(guard.acceptance_summary(1, 2)["host_evidence"], "refused")
                self.assertEqual(guard.host_samples, [])

    def test_clock_gap_read_duration_and_coverage_edges_refuse(self):
        for start, end in ((float("nan"), 1), (1, float("inf")), (True, 1), (2, 1)):
            guard = self.guard()
            self.sample(guard, start=start, end=end)
            guard.owner.stop.assert_called_once_with()
        for start, end in ((2.0001, 2.0001), (0.9, 1.1), (1, 1), (1, 2.0001)):
            guard = self.guard()
            self.sample(guard)
            self.sample(guard, start=start, end=end)
            self.assertEqual(guard.acceptance_summary(1, 2)["host_evidence"], "refused")
            guard.owner.stop.assert_called_once_with()
        guard = self.guard()
        self.sample(guard, start=1, end=1.1)
        self.sample(guard, start=2, end=2.1)
        for start, end in ((None, None), (1, 2), (1.1, 2.1), (2, 1), (True, 2), (1, float("nan"))):
            self.assertEqual(guard.acceptance_summary(start, end)["host_evidence"], "refused")
        self.assertEqual(guard.acceptance_summary(1.1, 2)["host_evidence"], "satisfied")

    def test_missing_samples_and_bounded_storage_refuse(self):
        guard = self.guard()
        self.assertEqual(guard.summary()["host_evidence"], "refused")
        self.sample(guard)
        self.assertEqual(guard.acceptance_summary(1, 2)["host_evidence"], "refused")
        guard.host_samples *= 4096
        self.sample(guard, start=2)
        self.assertEqual(len(guard.host_samples), 4096)
        guard.owner.stop.assert_called_once_with()

    def test_actual_monitor_uses_repository_commit_snapshot_not_swap_usage(self):
        from tools import prepare_unsloth_q2 as preparation
        guard = self.guard()
        guard.done = Mock()
        guard.done.is_set.side_effect = [False, False, True]
        with patch.object(preparation, "memory_snapshot", side_effect=[
                {"available_gib": 4, "commit_available_gib": 5},
                {"available_gib": 5, "commit_available_gib": 4}]) as snapshot, \
                patch.object(bench.time, "monotonic", side_effect=[1, 1.1, 1.5, 1.6]), \
                patch.object(bench.psutil, "swap_memory", side_effect=AssertionError("not commit")):
            guard.monitor()
        self.assertEqual(snapshot.call_count, 2)
        self.assertEqual(guard.acceptance_summary(1.1, 1.5)["host_evidence"], "satisfied")
        guard.owner.stop.assert_not_called()
        self.assertEqual(guard.done.wait.call_args.args, (0.25,))

    def test_actual_monitor_read_failure_is_fatal_and_retains_prior_rows(self):
        from tools import prepare_unsloth_q2 as preparation
        guard = self.guard()
        guard.done = Mock()
        guard.done.is_set.side_effect = [False, False]
        with patch.object(preparation, "memory_snapshot", side_effect=[
                {"available_gib": 8, "commit_available_gib": 8}, OSError("unreadable")]), \
                patch.object(bench.time, "monotonic", side_effect=[1, 1.1, 1.5]):
            guard.monitor()
        self.assertEqual(len(guard.host_samples), 1)
        guard.owner.stop.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "monitor failed: OSError"):
            guard.check()

    def test_fatal_reason_and_raw_failure_are_sticky(self):
        guard = self.guard()
        self.sample(guard, 1, 8)
        failed_rows, reason = list(guard.host_samples), guard.reason
        self.sample(guard, 8, 8, 2)
        self.assertEqual(guard.host_samples, failed_rows)
        self.assertEqual(guard.reason, reason)
        guard.owner.stop.assert_called_once_with()


class RunTests(unittest.TestCase):
    def setUp(self):
        scratch = bench.ROOT / "build-unsloth"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="p1-u2-fixture-", dir=scratch)
        self.root = Path(self.temporary.name).resolve()
        self.assertEqual(self.root.parent, scratch.resolve())
        self.binary = self.root / "fake-engine.exe"
        self.binary.write_bytes(b"fake engine for hash only")
        self.config = self.root / "source.json"
        self.secret = "TEST-DO-NOT-SERIALIZE-API-KEY"
        self.config.write_text(json.dumps({"exe": str(self.binary), "api_key": self.secret,
            "model_name": "fake", "host": "0.0.0.0", "port": 8080,
            "fit_max_tokens": True, "open_browser": False,
            "args": ["--max-context", "131072", "--resident-budget-gib", "8"],
            "env": {"STRATA_VERIFY_PROFILE": "1", "STRATA_FILE_CACHE_MIB": "4096"}}))
        self.original = self.config.read_bytes()
        self.output = self.root / "output"
        self.args = argparse.Namespace(config=self.config, output=self.output, timing=False, smoke=False)

    def tearDown(self):
        # The resolved unique fixture directory was checked against build-unsloth in setUp.
        self.temporary.cleanup()

    def test_output_collision_refuses_before_reading_config_or_starting_process(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("untouched")
        self.args.config = self.root / "does-not-exist.json"
        with patch.object(bench.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                bench.run(self.args)
        popen.assert_not_called()
        self.assertEqual(sentinel.read_text(), "untouched")

    def test_acceptance_public_preparation_uses_real_guard_and_environment_but_refuses_launch(self):
        self.args.acceptance_telemetry = True
        self.args.acceptance_physical_gib = 6
        self.args.acceptance_commit_gib = 8
        from tools import prepare_unsloth_q2 as preparation
        with patch.dict(bench.os.environ, {}, clear=True), \
                patch.object(bench.subprocess, "Popen") as popen, \
                patch.object(bench, "assert_idle") as idle, \
                patch.object(preparation, "memory_snapshot") as snapshot:
            with self.assertRaisesRegex(RuntimeError, "Stage B/C"):
                bench.run(self.args)
        popen.assert_not_called()
        idle.assert_not_called()
        snapshot.assert_not_called()
        saved = json.loads((self.output / "results.json").read_text())
        self.assertEqual(saved["status"], "held")
        self.assertEqual(saved["memory"]["floors_gib"], {"physical": 6, "commit": 8})
        self.assertEqual(saved["memory"]["host_evidence"], "refused")
        self.assertEqual(saved["memory"]["raw_samples"], [])
        self.assertEqual(saved["memory"]["cleanup"], "pending")
        self.assertEqual(saved["memory"]["finalization"], "pending")
        self.assertEqual(saved["effective_environment"], {"STRATA_FILE_CACHE_MIB": "4096"})
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertNotIn(self.secret, (self.output / "results.json").read_text())

    def test_invalid_acceptance_floors_refuse_before_config_or_resources(self):
        for value in (True, 0, -1, "4", float("nan"), float("inf")):
            with self.subTest(value=value):
                self.args.acceptance_telemetry = True
                self.args.acceptance_physical_gib = value
                self.args.config = self.root / "missing.json"
                with patch.object(bench.subprocess, "Popen") as popen, \
                        patch.object(bench, "assert_idle") as idle:
                    with self.assertRaises(ValueError):
                        bench.run(self.args)
                popen.assert_not_called()
                idle.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_acceptance_floor_flags_require_opt_in(self):
        self.args.acceptance_commit_gib = 4
        with self.assertRaisesRegex(ValueError, "require --acceptance-telemetry"):
            bench.run(self.args)
        self.assertFalse(self.output.exists())

    def test_startup_refusal_creates_no_output_and_launches_nothing(self):
        with patch.object(bench, "assert_idle", side_effect=RuntimeError("A Strata engine is already running")), \
                patch.object(bench.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                bench.run(self.args)
        popen.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.config.read_bytes(), self.original)

    def fake_client(self, fail=False, context=131072):
        secret = self.secret

        class Client:
            def __init__(self, cfg):
                self.cfg, self.count, self.history = cfg, 0, []

            def get(self, path):
                if path == "/health":
                    return {"loaded": True, "max_context": context}
                return {"engine": {"cache_slots": 4990}, "totals": {"requests": self.count},
                        "requests": list(reversed(self.history))}

            def chat(self, messages, max_tokens=128, **kwargs):
                if fail:
                    raise OSError("mock request failed with Bearer " + secret)
                self.count += 1
                ms = 1000 if max_tokens == 64 else (1000 if self.count % 2 else 2000)
                self.history.append({"time": self.count, "output_tokens": max_tokens,
                                     "prompt_tokens": 20, "decode_ms": ms})
                return {"choices": [{"message": {"content": "answer " + secret}}],
                        "usage": {"completion_tokens": max_tokens, "prompt_tokens": 20},
                        "timings": {"predicted_ms": ms}, "wall_seconds": 2}

        return Client

    def invoke(self, fail=False, context=131072, smoke=None, expected_env=None):
        validation = types.SimpleNamespace(Client=self.fake_client(fail, context), smoke=smoke)
        benchmark = types.SimpleNamespace(PROMPTS=["one", "two", "three"])
        owner = Mock()
        proc = Mock(pid=123, stdout=io.BytesIO(("server " + self.secret).encode()))
        proc.poll.return_value = None

        def spawn(argv, **kwargs):
            temporary_cfg = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
            self.assertEqual(temporary_cfg["host"], "127.0.0.1")
            self.assertEqual(temporary_cfg["port"], 8081)
            self.assertEqual(temporary_cfg["api_key"], self.secret)
            self.assertEqual(temporary_cfg["args"], ["--max-context", "131072", "--resident-budget-gib", "8"])
            self.assertEqual(temporary_cfg["env"], expected_env or {"STRATA_FILE_CACHE_MIB": "4096"})
            self.assertNotIn("STRATA_VERIFY_PROFILE", kwargs["env"])
            Path(temporary_cfg["log"]).write_text("engine " + self.secret)
            return proc

        self.owner = owner
        with patch.object(bench, "SCRATCH", self.root), patch.object(bench, "assert_idle"), \
                patch.object(bench, "source_identity", return_value={"head": "fake", "sha256": "test-only"}), \
                patch.object(bench, "OwnedProcess", return_value=owner), \
                patch.object(bench.subprocess, "Popen", side_effect=spawn), \
                patch.object(bench, "load_workload", side_effect=lambda name: validation if name == "validation" else benchmark), \
                patch.object(bench.psutil, "virtual_memory", return_value=types.SimpleNamespace(available=4 * 1024**3)), \
                patch.object(bench.psutil, "swap_memory", return_value=types.SimpleNamespace(used=0)):
            return bench.run(self.args)

    def assert_safe_artifacts(self):
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertEqual(list(self.root.glob("compact-cache-*")), [])
        for path in self.output.iterdir():
            self.assertNotIn(self.secret, path.read_text(), path.name)

    def test_mid_request_failure_keeps_redacted_partial_results_and_cleans_up(self):
        with self.assertRaisesRegex(OSError, "mock request failed"):
            self.invoke(fail=True)
        self.owner.stop.assert_called_once_with()
        saved = json.loads((self.output / "results.json").read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(len(saved["requests"]), 1)
        self.assertIn("[REDACTED]", saved["error"])
        self.assert_safe_artifacts()

    def test_context_mismatch_prevents_first_request(self):
        with self.assertRaisesRegex(RuntimeError, "context is not 131072"):
            self.invoke(context=65536)
        saved = json.loads((self.output / "results.json").read_text())
        self.assertEqual(saved["requests"], [])
        self.assertEqual(saved["status"], "failed")
        self.owner.stop.assert_called_once_with()
        self.assert_safe_artifacts()

    def test_success_records_six_answers_and_weighted_decode_rate_without_secrets(self):
        record = self.invoke()
        self.assertEqual(record["status"], "complete")
        self.assertEqual(len(record["rows"]), 6)
        self.assertEqual(len(record["requests"]), 7)
        self.assertEqual(record["summary"]["completion_tokens"], 960)
        self.assertEqual(record["summary"]["predicted_ms"], 9000)
        self.assertAlmostEqual(record["summary"]["tokens_per_second"], 960 / 9)
        self.assertEqual(record["engine"], {"cache_slots": 4990})
        self.assertEqual(record["memory"]["min_available_gib"], 4)
        self.assertEqual(len(record["exe_sha256"]), 64)
        self.owner.stop.assert_called_once_with()
        self.assert_safe_artifacts()

    def test_environment_secrets_are_not_persisted_but_child_configuration_is_preserved(self):
        cfg = json.loads(self.config.read_text())
        other_secret = "TEST-HF-TOKEN-DIFFERENT-FROM-API-KEY"
        cfg["env"]["HF_TOKEN"] = other_secret
        cfg["env"]["STRATA_CUSTOM_SECRET"] = "TEST-UNLISTED-STRATA-SECRET"
        cfg["env"]["STRATA_FILE_DIRECT_UNMAP"] = "1"
        self.config.write_text(json.dumps(cfg))
        self.original = self.config.read_bytes()
        expected_env = {"STRATA_FILE_CACHE_MIB": "4096", "HF_TOKEN": other_secret,
                        "STRATA_CUSTOM_SECRET": "TEST-UNLISTED-STRATA-SECRET", "STRATA_FILE_DIRECT_UNMAP": "1"}
        result = self.invoke(expected_env=expected_env)
        self.assertEqual(result["effective_settings"]["env"],
                         {"STRATA_FILE_CACHE_MIB": "4096", "STRATA_FILE_DIRECT_UNMAP": "1"})
        for path in self.output.iterdir():
            self.assertNotIn(other_secret, path.read_text(), path.name)
            self.assertNotIn("TEST-UNLISTED-STRATA-SECRET", path.read_text(), path.name)
        self.assert_safe_artifacts()

    def test_run_smoke_accepts_flushed_output_and_saves_results(self):
        self.args.smoke = True

        def smoke(client):
            self.assertEqual(client.cfg["port"], 8081)
            response = client.chat([{"role": "user", "content": "mock smoke request"}])
            print("smoke progress " + json.dumps(response), flush=True)
            (self.output / "smoke.json").write_text(json.dumps([{"passed": True, "response": response}]))
            return True

        record = self.invoke(smoke=smoke)
        saved = json.loads((self.output / "results.json").read_text())
        self.assertEqual(record["status"], "complete")
        self.assertTrue(saved["smoke_passed"])
        self.assertEqual(len(saved["rows"]), 6)
        self.assertEqual(len(saved["requests"]), 8)
        self.assertEqual(saved["requests"][-1]["label"], "smoke")
        self.assertEqual(saved["requests"][-1]["metric"]["output_tokens"], 128)
        self.assertIn("smoke progress", (self.output / "smoke.log").read_text())
        self.assertTrue(json.loads((self.output / "smoke.json").read_text())[0]["passed"])
        self.owner.stop.assert_called_once_with()
        self.assert_safe_artifacts()

    def test_run_smoke_failure_keeps_flushed_partial_log_and_request_results(self):
        self.args.smoke = True

        def smoke(client):
            response = client.chat([{"role": "user", "content": "mock smoke request"}])
            print("partial smoke progress " + json.dumps(response), flush=True)
            raise RuntimeError("mock smoke failure " + self.secret)

        with self.assertRaisesRegex(RuntimeError, "mock smoke failure"):
            self.invoke(smoke=smoke)
        saved = json.loads((self.output / "results.json").read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(len(saved["rows"]), 6)
        self.assertEqual(len(saved["requests"]), 8)
        self.assertEqual(saved["requests"][-1]["label"], "smoke")
        self.assertIn("partial smoke progress", (self.output / "smoke.log").read_text())
        self.assertIn("[REDACTED]", saved["error"])
        self.owner.stop.assert_called_once_with()
        self.assert_safe_artifacts()

    def test_smoke_only_exercises_validation_without_claiming_timed_results(self):
        self.args.smoke_only = True

        def smoke(client):
            client.chat([{"role": "user", "content": "mock smoke request"}])
            print("smoke-only complete", flush=True)
            return True

        record = self.invoke(smoke=smoke)
        self.assertEqual(record["status"], "complete")
        self.assertTrue(record["smoke_passed"])
        self.assertTrue(record["smoke_only"])
        self.assertEqual(record["rows"], [])
        self.assertIsNone(record["summary"])
        self.assertEqual(len(record["requests"]), 2)  # Warmup plus validation request.
        self.owner.stop.assert_called_once_with()
        self.assert_safe_artifacts()

    def test_idle_gap_and_requests_have_correlatable_timestamps(self):
        self.args.idle_seconds = 0.01
        record = self.invoke()
        self.assertLessEqual(record["idle_started_unix"], record["idle_finished_unix"])
        self.assertLessEqual(record["requests"][3]["finished_unix"], record["idle_started_unix"])
        self.assertGreaterEqual(record["requests"][4]["started_unix"], record["idle_finished_unix"])
        for request in record["requests"]:
            self.assertLessEqual(request["started_unix"], request["finished_unix"])
        self.assert_safe_artifacts()

    def test_invalid_idle_time_starts_nothing(self):
        self.args.idle_seconds = -1
        with patch.object(bench, "assert_idle") as idle:
            with self.assertRaisesRegex(ValueError, "idle_seconds"):
                bench.run(self.args)
            idle.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_empty_api_key_refuses_before_starting(self):
        cfg = json.loads(self.config.read_text())
        cfg["api_key"] = ""
        self.config.write_text(json.dumps(cfg))
        with patch.object(bench, "assert_idle") as idle, patch.object(bench.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "API key is required"):
                bench.run(self.args)
        idle.assert_not_called()
        popen.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_acceptance_requires_explicit_fit_and_disabled_browser(self):
        self.args.held_out = True
        for key, value in (("fit_max_tokens", False), ("open_browser", True)):
            cfg = json.loads(self.original)
            cfg[key] = value
            self.config.write_text(json.dumps(cfg))
            with self.subTest(key=key), patch.object(bench.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(ValueError, "fit_max_tokens=true and open_browser=false"):
                    bench.run(self.args)
            popen.assert_not_called()

    def test_validation_only_requires_an_acceptance_mode(self):
        self.args.validation_only = True
        with patch.object(bench.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "validation_only requires"):
                bench.run(self.args)
        popen.assert_not_called()

    def test_validation_only_excludes_standard_cases_and_calls_acceptance(self):
        self.args.validation_only = True
        self.args.held_out = True
        with patch.object(bench, "check_authentication") as authentication, \
                patch.object(bench, "held_out_tests") as held_out:
            record = self.invoke()
        self.assertEqual(record["rows"], [])
        self.assertIsNone(record["summary"])
        self.assertEqual(len(record["requests"]), 1)
        authentication.assert_called_once()
        held_out.assert_called_once()
        self.owner.stop.assert_called_once_with()

    def test_transfer_validation_only_runs_four_cases_after_warmup_with_authentication(self):
        self.args.validation_only = True
        self.args.transfer_workload = True
        with patch.object(bench, "check_authentication") as authentication:
            record = self.invoke()
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["rows"], [])
        self.assertIsNone(record["summary"])
        self.assertEqual(len(record["requests"]), 5)
        self.assertEqual(len(record["transfer_workload"]), 4)
        self.assertEqual(record["transfer_quality"], "needs_manual_review")
        self.assertEqual(record["transfer_summary"]["completion_tokens"], 640)
        authentication.assert_called_once()
        self.owner.stop.assert_called_once_with()
        self.assert_safe_artifacts()

    def test_transfer_cli_sets_optional_workload_and_validation_only(self):
        with patch.object(bench, "run") as run, patch.object(sys, "stdout", io.StringIO()):
            status = bench.main(["--config", str(self.config), "--output", str(self.output),
                                 "--transfer-workload", "--validation-only"])
        self.assertEqual(status, 0)
        self.assertTrue(run.call_args.args[0].transfer_workload)
        self.assertTrue(run.call_args.args[0].validation_only)
        self.assertFalse(run.call_args.args[0].held_out)
        self.assertFalse(run.call_args.args[0].timing)

    def test_long_acceptance_failure_keeps_evidence_and_stops_owned_process(self):
        self.args.validation_only = True
        self.args.long_context = True

        def fail(validation, client, record, save):
            record["long_context"] = [{"passed": False, "checks": {"full_prompt_retained": False}}]
            save()
            raise RuntimeError("mock full prompt retention failure")

        with patch.object(bench, "check_authentication"), \
                patch.object(bench, "long_context_tests", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "full prompt retention"):
                self.invoke()
        self.owner.stop.assert_called_once_with()
        saved = json.loads((self.output / "results.json").read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["rows"], [])
        self.assertFalse(saved["long_context"][0]["checks"]["full_prompt_retained"])
        self.assert_safe_artifacts()

    def test_mid_request_snapshot_failure_preserves_answer_and_stops_owned_process(self):
        replace = bench.replace_result
        count = 0

        def fail_once(pending, destination):
            nonlocal count
            count += 1
            if count == 4:
                raise PermissionError("mock result snapshot failure")
            return replace(pending, destination)

        with patch.object(bench, "replace_result", side_effect=fail_once):
            with self.assertRaisesRegex(PermissionError, "mock result snapshot failure"):
                self.invoke()
        self.owner.stop.assert_called_once_with()
        saved = json.loads((self.output / "results.json").read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(len(saved["requests"]), 1)
        self.assertEqual(saved["requests"][0]["response"]["usage"]["completion_tokens"], 64)
        self.assertIn("mock result snapshot failure", saved["requests"][0]["error"])
        self.assert_safe_artifacts()

    def test_log_write_failure_still_saves_failure_and_stops_owned_process(self):
        write_text = Path.write_text

        def write(path, data, *args, **kwargs):
            if path == self.output / "engine.log":
                raise OSError("mock output disk failure")
            return write_text(path, data, *args, **kwargs)

        with patch.object(Path, "write_text", write):
            with self.assertRaisesRegex(OSError, "mock output disk failure"):
                self.invoke()
        self.owner.stop.assert_called_once_with()
        saved = json.loads((self.output / "results.json").read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(len(saved["rows"]), 6)
        self.assertIn("mock output disk failure", saved["artifact_error"])
        self.assert_safe_artifacts()


class LongContextTests(unittest.TestCase):
    def invoke(self, fresh_change=None, reuse_change=None):
        count = 130880
        self.record = {"requests": []}
        self.save = Mock()
        messages = [{"role": "user", "content": "large deterministic fixture"}]
        validation = types.SimpleNamespace(long_messages=Mock(return_value=(
            messages, ["cedar-284", "finch-693", "lantern-517"], count)))
        client = Mock(cfg={})
        responses = [
            {"choices": [{"message": {"role": "assistant", "content": "cedar-284 finch-693 lantern-517"}}],
             "usage": {"prompt_tokens": count, "completion_tokens": 20},
             "timings": {"cache_n": 0, "prompt_n": count, "prompt_ms": 900000, "predicted_ms": 1000}},
            {"choices": [{"message": {"role": "assistant", "content": "lantern-517 391"}}],
             "usage": {"prompt_tokens": count + 50, "completion_tokens": 10},
             "timings": {"cache_n": count, "prompt_n": 50, "prompt_ms": 4000, "predicted_ms": 500}},
        ]
        if fresh_change:
            fresh_change(responses[0])
        if reuse_change:
            reuse_change(responses[1])

        def chat(*args, **kwargs):
            self.record["requests"].append({"metric": {"file_mb": 32}})
            return responses.pop(0)

        client.chat.side_effect = chat
        self.client = client
        self.counter = Mock(return_value=130930)
        with patch.object(bench, "prompt_counter", return_value=self.counter):
            bench.long_context_tests(validation, client, self.record, self.save)
        self.assertEqual(validation.long_messages.call_args.args[1], 130880)
        return self.record

    def test_fresh_near_128k_fit_and_real_prefix_reuse(self):
        record = self.invoke()
        self.assertTrue(record["long_context_passed"])
        self.assertEqual([call.args[1] for call in self.client.chat.call_args_list], [256, 128])
        self.assertEqual(record["long_context"][0]["local_prompt_tokens"], 130880)
        self.assertEqual(record["long_context_summary"]["completion_tokens"], 30)
        self.assertEqual(record["long_context_summary"]["predicted_ms"], 1500)
        self.assertEqual(record["long_context_summary"]["prompt_ms"], 904000)
        self.assertEqual(record["long_context"][1]["expected_prompt_tokens"], 130930)
        self.counter.assert_called_once()
        counted_messages = self.counter.call_args.args[0]
        self.assertEqual(len(counted_messages), 3)
        self.assertEqual(counted_messages[0]["content"], "large deterministic fixture")
        self.assertEqual(counted_messages[1]["content"], "cedar-284 finch-693 lantern-517")
        self.assertIn("17 times 23", counted_messages[2]["content"])

    def test_truncated_fresh_prompt_is_rejected_and_saved_before_reuse(self):
        with self.assertRaisesRegex(RuntimeError, "Fresh long-context/fit"):
            self.invoke(fresh_change=lambda row: row["usage"].update(prompt_tokens=32768))
        self.assertEqual(self.client.chat.call_count, 1)
        self.assertFalse(self.record["long_context"][0]["checks"]["full_prompt_retained"])
        self.save.assert_called_once_with()

    def test_incorrect_retrieval_or_overlong_completion_fails(self):
        changes = [lambda row: row["choices"][0]["message"].update(content="cedar-284 only"),
                   lambda row: row["choices"][0]["message"].update(content="cedar-2840 finch-693 lantern-517"),
                   lambda row: row["choices"][0]["message"].update(content="finch-693 cedar-284 lantern-517"),
                   lambda row: row["usage"].update(completion_tokens=200)]
        for change in changes:
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, "Fresh long-context/fit"):
                self.invoke(fresh_change=change)
            self.assertFalse(self.record["long_context"][0]["passed"])

    def test_reuse_must_reuse_prefix_and_answer_correctly(self):
        changes = [lambda row: row["timings"].update(cache_n=0, prompt_n=130930),
                   lambda row: row["choices"][0]["message"].update(content="lantern-517 390"),
                   lambda row: row["choices"][0]["message"].update(content="lantern-517 3910"),
                   lambda row: row["choices"][0]["message"].update(content="lantern-517 391.5"),
                   lambda row: row["choices"][0]["message"].update(content="lantern-517 391/2"),
                   lambda row: row["choices"][0]["message"].update(content="lantern-5170 391"),
                   lambda row: row["usage"].update(prompt_tokens=130900)]
        for change in changes:
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, "reuse validation failed"):
                self.invoke(reuse_change=change)
            self.assertFalse(self.record["long_context_passed"])
            self.assertEqual(len(self.record["long_context"]), 2)
            self.assertEqual(self.save.call_count, 3)


class HeldOutTests(unittest.TestCase):
    def test_all_four_checks_and_separate_summary_are_saved_even_if_one_answer_fails(self):
        record, save = {"requests": []}, Mock()
        client = Mock()
        answers = iter(("FINAL: [[0,7],[0,7],[0,7]]", "FINAL: [4,8,12]", "FINAL: 9", "FINAL: 20"))

        def chat(messages, max_tokens):
            self.assertEqual(max_tokens, 160)
            record["requests"].append({"metric": {"file_mb": 15}})
            return {"choices": [{"message": {"content": next(answers)}}],
                    "usage": {"completion_tokens": 40, "prompt_tokens": 80},
                    "timings": {"predicted_ms": 2000, "prompt_ms": 1000}}

        client.chat.side_effect = chat
        with self.assertRaisesRegex(RuntimeError, "Held-out validation failed"):
            bench.held_out_tests(client, record, save)
        self.assertEqual(client.chat.call_count, 4)
        self.assertEqual([row["passed"] for row in record["held_out"]], [True, False, True, True])
        self.assertFalse(record["held_out_passed"])
        self.assertEqual(record["held_out_summary"]["tokens_per_second"], 20)
        self.assertEqual(record["held_out_summary"]["prompt_ms"], 4000)
        self.assertEqual(record["held_out"][1]["expected"], [4, 18, 12])


class TransferWorkloadTests(unittest.TestCase):
    def test_transfer_aggregates_decode_and_prefill_and_requires_manual_review(self):
        record, save, client = {"requests": []}, Mock(), Mock()
        timings = iter(((20, 1000, 100), (60, 2000, 200), (160, 4000, 300), (80, 1000, 400)))

        def chat(messages, max_tokens, **kwargs):
            self.assertEqual(max_tokens, 160)
            self.assertEqual(kwargs, {"temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
            count, decode, prefill = next(timings)
            record["requests"].append({"metric": {"file_mb": 42}})
            return {"choices": [{"message": {"content": "```python\nraise AssertionError('never execute')\n```"},
                                 "finish_reason": "length"}],
                    "usage": {"completion_tokens": count, "prompt_tokens": 100},
                    "timings": {"predicted_ms": decode, "prompt_ms": prefill}}

        client.chat.side_effect = chat
        bench.transfer_workload_tests(client, record, save)
        self.assertEqual(client.chat.call_count, 4)
        self.assertEqual(save.call_count, 4)
        self.assertEqual(record["transfer_summary"], {"completion_tokens": 320, "predicted_ms": 8000,
                         "tokens_per_second": 40, "prompt_ms": 1000, "prompt_tokens": 400})
        self.assertEqual(record["transfer_quality"], "needs_manual_review")
        self.assertEqual([row["case"] for row in record["transfer_workload"]],
                         ["python-run-lengths", "python-mutation-debug", "inventory-arithmetic", "project-note-synthesis"])
        for row in record["transfer_workload"]:
            self.assertEqual(row["quality"], "needs_manual_review")
            self.assertNotIn("passed", row)
            self.assertEqual(row["metric"], {"file_mb": 42})
            self.assertEqual(row["response"]["choices"][0]["finish_reason"], "length")
            self.assertIn("never execute", row["response"]["choices"][0]["message"]["content"])
        self.assertIn("[None,None,2] -> [(None,2),(2,1)]",
                      record["transfer_workload"][0]["review_expectations"])
        self.assertIn("available=174", record["transfer_workload"][2]["review_expectations"])

    def test_transfer_failure_retains_completed_cases_and_partial_aggregate(self):
        record, save, client = {"requests": []}, Mock(), Mock()

        def chat(*args, **kwargs):
            if len(record["requests"]) == 2:
                raise OSError("simulated third request failure")
            record["requests"].append({"metric": {"file_mb": 4}})
            return {"choices": [{"message": {"content": "unreviewed answer"}}],
                    "usage": {"completion_tokens": 40, "prompt_tokens": 100},
                    "timings": {"predicted_ms": 2000, "prompt_ms": 500}}

        client.chat.side_effect = chat
        with self.assertRaisesRegex(OSError, "simulated third request failure"):
            bench.transfer_workload_tests(client, record, save)
        self.assertEqual(len(record["transfer_workload"]), 2)
        self.assertEqual(record["transfer_summary"]["completion_tokens"], 80)
        self.assertEqual(record["transfer_summary"]["tokens_per_second"], 20)
        self.assertEqual(save.call_count, 2)


class ContainedLifecycleTests(unittest.TestCase):
    """Real public composition; only OS, workload, filesystem and missing Stage C seams are fake."""

    class BlockedChild(BaseException):
        pass

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="contained-cpu-")
        self.root = Path(self.temporary.name)
        self.binary = self.root / "python.exe"
        self.binary.write_bytes(b"synthetic direct interpreter, never executed")
        self.config = self.root / "synthetic.json"
        self.config.write_text(json.dumps({"exe": str(self.binary), "api_key": "synthetic-key",
                                          "fit_max_tokens": True, "open_browser": False}))
        self.output = self.root / "output"
        self.args = argparse.Namespace(config=self.config, output=self.output, timing=False, smoke=False,
                                       acceptance_telemetry=True)
        self.now, self.mode, self.started, self.samples = 100.0, "green", False, 0
        self.events, self.requests = [], []
        self.root_proc = types.SimpleNamespace(pid=321, _handle=1234, stdin=io.BytesIO(), stdout=io.BytesIO())
        # Keep the START bytes inspectable after close, as a fake pipe transport would.
        class StartPipe(io.BytesIO):
            def close(pipe):
                pipe.envelope = pipe.getvalue()
        self.root_proc.stdin = StartPipe()
        self.code, self.assigned, self.killed, self.closed = None, False, False, False
        self.stage = None
        self.cleanup = contextlib.ExitStack()
        self.addCleanup(self.cleanup.close)
        self.addCleanup(self.temporary.cleanup)
        self.native = self.Native(self)
        self.NativeBinding = bench.WindowsProcessAPI
        self.SelfIdentityBinding = bench.worker_self_identity
        from tools import prepare_unsloth_q2 as preparation
        patches = [
            patch.object(bench, "acceptance_ready", return_value=True),
            patch.object(bench, "acceptance_preflight", side_effect=self.preflight),
            patch.object(bench, "acceptance_protection", side_effect=lambda args: self.mode != "protection"),
            patch.object(bench, "WindowsProcessAPI", side_effect=lambda: self.native),
            patch.object(bench, "worker_self_identity", return_value=(321, 2**63 + 7)),
            patch.object(ctypes, "WinDLL", side_effect=AssertionError("unintercepted native DLL")),
            patch.dict(sys.modules, {"serve.winjob": types.SimpleNamespace(
                contain=Mock(side_effect=AssertionError("server native containment boundary")))}),
            patch.object(bench.sys, "base_prefix", str(self.root)),
            patch.object(bench.sys, "_base_executable", str(self.binary)),
            patch.object(bench, "SCRATCH", self.root),
            patch.object(bench.time, "monotonic", side_effect=self.clock),
            patch.object(bench.time, "sleep", side_effect=self.sleep),
            patch.object(bench.subprocess, "Popen", side_effect=self.spawn),
            patch.object(bench.subprocess, "run", side_effect=self.git),
            patch.object(bench.psutil, "Process", side_effect=AssertionError("PID lookup forbidden")),
            patch.object(bench.psutil, "process_iter", return_value=[]),
            patch.object(bench.psutil, "virtual_memory", side_effect=AssertionError("real host read")),
            patch.object(bench.psutil, "swap_memory", side_effect=AssertionError("real commit read")),
            patch.object(preparation, "memory_snapshot", side_effect=self.sample),
            patch.object(bench.socket, "socket", return_value=Mock(__enter__=Mock(return_value=Mock()),
                                                                  __exit__=Mock(return_value=False))),
            patch.object(bench, "load_workload", side_effect=self.workload),
        ]
        for item in patches:
            self.cleanup.enter_context(item)

    def clock(self):
        self.now += 0.001
        return self.now

    def sleep(self, seconds):
        self.assertGreaterEqual(seconds, 0)
        self.now += seconds

    def git(self, argv, **kwargs):
        self.assertEqual(argv[0], "git", "Unintercepted external command")
        data = {"ls-files": b"", "rev-parse": b"synthetic\n", "status": b""}[argv[1]]
        return types.SimpleNamespace(stdout=data)

    def preflight(self, args):
        self.events.append("admission")
        self.assertTrue(self.assigned)
        self.assertTrue(self.started)
        self.assertFalse(self.output.exists())
        if self.mode == "blocked-admission":
            self.stage = "admission"
            raise self.BlockedChild()

    def sample(self):
        self.samples += 1
        if self.mode == "missing-telemetry":
            return {"available_gib": 8}
        if self.mode == "missing-final-edge" and self.closed:
            return {"available_gib": 8}
        if self.mode == "gap" and self.samples == 2:
            self.now += 1.1
        if self.mode == "guard-cleanup" and self.samples >= 2:
            return {"available_gib": 1, "commit_available_gib": 8}
        if self.mode == "floor-fail":
            return {"available_gib": 3, "commit_available_gib": 9}
        return {"available_gib": 8, "commit_available_gib": 9}

    def spawn(self, argv, **kwargs):
        if argv[-1] == "--internal-acceptance-worker":
            self.assertEqual(argv[:3], [str(self.binary), "-I", "-S"])
            self.assertFalse(self.output.exists())
            self.assertGreaterEqual(self.samples, 1, "Host coverage must precede owned startup")
            self.events.append("create")
            self.bootstrap_settings = kwargs["env"]["STRATA_BENCHMARK_BOOTSTRAP"]
            if self.mode == "missing-handle":
                self.root_proc._handle = None
            return self.root_proc
        self.assertIn("--engine", argv, "Unexpected process boundary")
        self.assertTrue(self.assigned)
        cfg = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
        self.assertIsNone(cfg["log"])
        self.events.append("server")
        return types.SimpleNamespace(stdout=io.BytesIO(b"synthetic server output\n"),
                                     poll=lambda: None, terminate=lambda: self.events.append("server-stop"),
                                     wait=lambda timeout: self.events.append("server-wait") or 0)

    def workload(self, name):
        test = self
        class Client:
            def __init__(self, cfg):
                self.cfg, self.history = cfg, []

            def get(self, path):
                if path == "/health":
                    return {"loaded": True, "max_context": 131072}
                return {"engine": {}, "totals": {"requests": len(self.history)}, "requests": self.history[:]}

            def chat(self, messages, max_tokens=128, **kwargs):
                test.requests.append(messages)
                if test.mode == "request-secret":
                    raise OSError("synthetic-key from an HTTP failure")
                index = len(self.history) + 1
                self.history.append({"time": index, "output_tokens": max_tokens,
                                     "prompt_tokens": 20, "decode_ms": 1000})
                return {"choices": [{"message": {"content": "synthetic answer"}}],
                        "usage": {"completion_tokens": max_tokens, "prompt_tokens": 20},
                        "timings": {"predicted_ms": 1000}}
        return types.SimpleNamespace(Client=Client, PROMPTS=["synthetic workload"])

    class Native:
        def __init__(self, test):
            self.t = test

        def identity(self, handle):
            t = self.t
            t.assertEqual(handle, 1234)
            if t.mode == "bad-identity":
                return 321, 123.0
            if t.mode == "out-of-range-identity":
                return 321, 2**64
            if t.mode == "reused-identity" and t.started:
                return 321, 2**63 + 8
            return 321, 2**63 + 7

        def create_job(self):
            self.t.events.append("create-job")
            return 5678

        def configure_job(self, job):
            self.t.events.append("kill-on-close")
            if self.t.mode == "configure-fail":
                raise OSError("Job configuration failed")

        def assign(self, job, handle):
            t = self.t
            t.events.append("assign")
            if t.mode == "assignment":
                raise OSError("nested assignment denied")
            t.assigned = True

        def active(self, job):
            t = self.t
            t.assertEqual(job, 5678)
            if t.mode == "unknown-job":
                raise OSError("Job membership unavailable")
            if t.mode in ("live-job", "survivor"):
                return 1
            return int(t.code is None)

        def terminate_job(self, job):
            t = self.t
            t.events.append("cancel-job")
            t.killed = True
            if t.mode == "guard-cleanup":
                raise OSError("synthetic cancel failure")
            if t.mode != "survivor":
                t.code = 1

        def terminate_process(self, handle):
            self.t.events.append("cancel-exact-handle")
            self.t.killed, self.t.code = True, 1

        def wait(self, handle, milliseconds):
            t = self.t
            t.events.append("wait")
            if handle is None:
                raise OSError("missing handle")
            if t.mode == "wait-failed":
                return 0xffffffff
            if t.mode == "wait-error" and not t.killed:
                raise OSError("synthetic wait exception")
            if t.mode in ("wait-timeout", "survivor") and t.killed:
                t.now += milliseconds / 1000
                return 258
            return 258 if t.code is None else 0

        def exit_code(self, handle):
            return self.t.code

        def close(self, job):
            t = self.t
            t.events.append("close-job")
            if t.mode == "close-fail":
                raise OSError("close failed")

        def close_process(self, proc):
            if self.t.mode == "process-close-fail":
                raise OSError("process close failed")
            self.t.events.append("close-process")
            self.t.closed = True

        def capture(self, pipe):
            t = self.t
            if not t.started and not t.killed:
                t.started = True
                t.events.append("START")
                if t.mode in ("hung", "late-worker", "wait-timeout", "survivor"):
                    t.stage = "parked worker"
                else:
                    envelope = t.root_proc.stdin.envelope.decode("utf-8")
                    if t.mode == "malformed-start":
                        item = json.loads(envelope)
                        item["identity"]["creation_filetime"] = "01"
                        envelope = json.dumps(item) + "\n"
                    try:
                        t.code = bench.worker_bootstrap(io.StringIO(envelope), bench.acceptance_worker, pid=321,
                                                       settings=t.bootstrap_settings)
                    except t.BlockedChild:
                        pass  # A parked external syscall: the process, not the controller, is blocked.
                    except BaseException as exc:
                        t.child_error = exc
                        t.code = 1
            return (b"X" * (bench.ACCEPTANCE_LOG_LIMIT + 1) if t.mode == "capture-overflow" else b"",
                    t.code is not None and t.mode != "blocked-capture")

    def invoke(self, mode="green"):
        self.mode = mode
        with contextlib.redirect_stdout(io.StringIO()):
            return bench.run(self.args)

    def refused(self, mode):
        with self.assertRaises(bench.AcceptanceFailure) as caught:
            self.invoke(mode)
        record = caught.exception.record
        self.assertFalse(bench.acceptance_final_predicate(record))
        self.assertEqual(record["status"], "held")
        self.assertEqual(record["lifecycle_evidence"], "refused")
        for path in self.output.glob("*.json"):
            self.assertNotEqual(json.loads(path.read_text()).get("status"), "PASS")
        return caught.exception

    def test_public_run_composes_bootstrap_existing_workload_monitor_owner_and_finalizer(self):
        result = self.invoke()
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertEqual(result["lifecycle_evidence"], "satisfied")
        self.assertEqual(result["identity"], {"pid": "321", "creation_filetime": str(2**63 + 7)})
        self.assertEqual(len(self.requests), 3)
        self.assertGreaterEqual(self.samples, 3)
        self.assertLess(self.events.index("kill-on-close"), self.events.index("assign"))
        self.assertLess(self.events.index("assign"), self.events.index("START"))
        self.assertLess(self.events.index("START"), self.events.index("admission"))
        self.assertLess(self.events.index("admission"), self.events.index("server"))
        self.assertTrue(self.closed)
        self.assertFalse(self.killed)
        self.assertEqual(json.loads((self.output / "provisional.json").read_text())["status"], "provisional_complete")
        self.assertEqual(json.loads((self.output / "lifecycle.json").read_text())["status"], "held")
        self.assertFalse((self.output / "results.json").exists())
        self.assertLess(sum(path.stat().st_size for path in self.output.iterdir()), bench.ACCEPTANCE_ARTIFACT_LIMIT)

    def test_main_uses_the_same_complete_public_composition(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = bench.main(["--config", str(self.config), "--output", str(self.output), "--acceptance-telemetry"])
        self.assertEqual(code, 0)
        self.assertEqual(len(self.requests), 3)
        self.assertTrue(self.closed)

    def test_assignment_failure_rolls_back_only_creation_handle_and_never_sends_start(self):
        self.refused("assignment")
        self.assertFalse(self.started)
        self.assertNotIn("admission", self.events)
        self.assertEqual(self.root_proc.stdin.envelope, b"")
        self.assertIn("cancel-exact-handle", self.events)
        self.assertNotIn("cancel-job", self.events)
        self.assertTrue(self.closed)

    def test_job_configuration_failure_retains_then_closes_owned_job(self):
        self.refused("configure-fail")
        self.assertFalse(self.started)
        self.assertNotIn("assign", self.events)
        self.assertIn("cancel-exact-handle", self.events)
        self.assertIn("close-job", self.events)

    def test_hung_worker_uses_reserved_absolute_cleanup_budget(self):
        failure = self.refused("hung")
        self.assertTrue(self.killed)
        self.assertTrue(self.closed)
        self.assertLess(self.now, failure.record["total_deadline"])
        self.assertIn("Absolute worker deadline", failure.record["error"])

    def test_blocked_admission_is_inside_independently_cancelled_process(self):
        self.refused("blocked-admission")
        self.assertEqual(self.stage, "admission")
        self.assertTrue(self.killed)
        self.assertNotIn("server", self.events)

    def test_late_worker_cannot_resume_after_owned_job_cancellation(self):
        self.refused("late-worker")
        self.now += 1000
        self.native.capture(self.root_proc.stdout)
        self.assertTrue(self.killed)
        self.assertNotIn("admission", self.events)
        self.assertFalse((self.output / "provisional.json").exists())

    def test_blocked_worker_artifact_replace_is_supervised_after_requests_and_cleanup(self):
        real_replace = Path.replace
        def replace(path, target):
            if (path.name == "provisional.json.tmp"
                    and json.loads(path.read_text()).get("status") == "provisional_complete"):
                self.stage = "worker artifact replace"
                raise self.BlockedChild()
            return real_replace(path, target)
        with patch.object(Path, "replace", replace):
            failure = self.refused("green")
        self.assertEqual(len(self.requests), 3)
        self.assertIn("server-stop", self.events)
        self.assertEqual(self.stage, "worker artifact replace")
        self.assertTrue(self.killed)
        self.assertTrue(self.closed)
        self.assertIn("Absolute worker deadline", failure.record["error"])

    def test_inherited_capture_without_eof_refuses_despite_worker_exit_zero(self):
        failure = self.refused("blocked-capture")
        self.assertEqual(failure.record["worker_exit"], 0)
        self.assertFalse(failure.record.get("capture_complete", False))
        self.assertTrue(self.killed)

    def test_wait_failed_still_cancels_and_retains_uncertain_handles(self):
        probes = []
        identity, active = self.native.identity, self.native.active
        def observe_identity(handle):
            probes.append(("identity", self.killed))
            return identity(handle)
        def observe_job(job):
            probes.append(("job", self.killed))
            return active(job)
        with patch.object(self.native, "identity", side_effect=observe_identity), \
                patch.object(self.native, "active", side_effect=observe_job):
            failure = self.refused("wait-failed")
        self.assertTrue(self.killed)
        self.assertFalse(self.closed)
        self.assertIs(failure.owner.proc, self.root_proc)
        self.assertNotIn("close-process", self.events)
        self.assertIn(("identity", True), probes)
        self.assertIn(("job", True), probes)
        self.assertEqual(failure.record["owned_accounting"]["wait_result"], 0xffffffff)
        self.assertEqual(failure.record["owned_accounting"]["identity"],
                         {"pid": "321", "creation_filetime": str(2**63 + 7)})
        self.assertEqual(failure.record["owned_accounting"]["job_active"], 0)
        self.assertIn("Final owned wait was not signaled", failure.record["cleanup_error"])
        with self.assertRaises(RuntimeError):
            failure.owner.stop()

    def test_failed_wait_and_identity_still_attempt_post_cancel_job_accounting(self):
        identity, active = self.native.identity, self.native.active
        probes = []
        def observe_identity(handle):
            if self.killed:
                raise OSError("synthetic post-cancel identity failure")
            return identity(handle)
        def observe_job(job):
            probes.append(self.killed)
            return active(job)
        with patch.object(self.native, "identity", side_effect=observe_identity), \
                patch.object(self.native, "active", side_effect=observe_job):
            failure = self.refused("wait-failed")
        self.assertIn(True, probes)
        self.assertIn("post-cancel identity failure", failure.record["owned_accounting"]["identity_error"])
        self.assertEqual(failure.record["owned_accounting"]["job_active"], 0)
        self.assertFalse(self.closed)

    def test_wait_exception_cancels_then_accounts_without_erasing_the_failure(self):
        failure = self.refused("wait-error")
        self.assertTrue(self.killed)
        self.assertTrue(self.closed)
        self.assertIn("wait exception", failure.record["error"])
        self.assertIn("wait exception", failure.record["cleanup_error"])

    def test_final_wait_timeout_does_not_close_the_sole_authority(self):
        failure = self.refused("wait-timeout")
        self.assertTrue(self.killed)
        self.assertFalse(self.closed)
        self.assertIn("258", failure.record["cleanup_error"])
        self.assertIs(failure.owner.proc, self.root_proc)

    def test_close_failure_refuses_even_after_natural_exit(self):
        failure = self.refused("close-fail")
        self.assertIn("close failed", failure.record["cleanup_error"])
        self.assertFalse(self.closed)

    def test_process_handle_close_failure_retains_authority(self):
        failure = self.refused("process-close-fail")
        self.assertIn("process close failed", failure.record["cleanup_error"])
        self.assertFalse(self.closed)
        self.assertIs(failure.owner.proc, self.root_proc)

    def test_main_preserves_unresolved_authority_for_an_embedding_caller(self):
        self.mode = "wait-failed"
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(bench.AcceptanceFailure) as failure:
            bench.main(["--config", str(self.config), "--output", str(self.output), "--acceptance-telemetry"])
        self.assertIs(failure.exception.owner.proc, self.root_proc)
        self.assertFalse(self.closed)

    def test_live_job_member_is_not_natural_success(self):
        self.refused("live-job")
        self.assertTrue(self.killed)
        self.assertFalse(self.closed)

    def test_unknown_job_membership_refuses_and_retains_authority(self):
        failure = self.refused("unknown-job")
        self.assertTrue(self.killed)
        self.assertFalse(self.closed)
        self.assertIs(failure.owner.proc, self.root_proc)

    def test_survivor_retains_exact_authority(self):
        failure = self.refused("survivor")
        self.assertTrue(self.killed)
        self.assertFalse(self.closed)
        self.assertIs(failure.owner.proc, self.root_proc)

    def test_guard_primary_reason_and_cleanup_failure_both_survive(self):
        failure = self.refused("guard-cleanup")
        self.assertIn("2 GiB", failure.record["error"])
        self.assertIn("cancel failure", failure.record["cleanup_error"])
        self.assertIn("Final owned wait was not signaled", failure.record["cleanup_error"])
        self.assertEqual(failure.record["memory"]["minima"]["physical"]["available_bytes"], 1024**3)

    def test_missing_telemetry_prevents_start(self):
        self.refused("missing-telemetry")
        self.assertFalse(self.started)

    def test_missing_final_coverage_edge_refuses_after_successful_work(self):
        failure = self.refused("missing-final-edge")
        self.assertEqual(len(self.requests), 3)
        self.assertIn("Required monotonic coverage edges", " ".join(failure.record["memory"]["errors"]))

    def test_configured_floor_refuses_without_rewriting_original_emergency_threshold(self):
        failure = self.refused("floor-fail")
        self.assertEqual(len(self.requests), 3)
        self.assertFalse(self.killed)
        self.assertEqual(failure.record["memory"]["minima"]["physical"]["available_bytes"], 3 * 1024**3)

    def test_lower_configured_allowance_does_not_remove_independent_four_gib_floor(self):
        self.args.acceptance_physical_gib = 2
        self.args.acceptance_commit_gib = 2
        failure = self.refused("floor-fail")
        self.assertEqual(failure.record["memory"]["host_evidence"], "satisfied")
        self.assertFalse(failure.record["base_host_floors_satisfied"])

    def test_higher_configured_allowance_is_also_required(self):
        self.args.acceptance_physical_gib = 10
        failure = self.refused("green")
        self.assertEqual(failure.record["memory"]["host_evidence"], "refused")
        self.assertTrue(failure.record["base_host_floors_satisfied"])

    def test_capture_volume_is_finite_and_overflow_refuses(self):
        failure = self.refused("capture-overflow")
        self.assertIn("output budget", failure.record["error"])
        self.assertTrue(self.killed)
        self.assertLessEqual((self.output / "controller.log").stat().st_size, bench.ACCEPTANCE_LOG_LIMIT)

    def test_gap_refuses_actual_controller_monitor_composition(self):
        failure = self.refused("gap")
        self.assertGreater(failure.record["memory"]["max_gap_seconds"], 1)

    def test_changed_retained_identity_refuses(self):
        self.refused("reused-identity")
        self.assertTrue(self.killed)
        self.assertFalse(self.closed)

    def test_malformed_native_identity_refuses_before_start(self):
        self.refused("bad-identity")
        self.assertFalse(self.started)

    def test_out_of_range_native_identity_refuses_actual_public_path(self):
        self.refused("out-of-range-identity")
        self.assertFalse(self.started)
        self.assertIn("cancel-exact-handle", self.events)

    def test_venv_configuration_refuses_before_any_process_creation(self):
        (self.root / "pyvenv.cfg").write_text("synthetic venv launcher marker")
        failure = self.refused("green")
        self.assertIn("direct base interpreter", failure.record["error"])
        self.assertNotIn("create", self.events)

    def test_malformed_transport_cannot_enter_worker_admission(self):
        self.refused("malformed-start")
        self.assertNotIn("admission", self.events)

    def test_canonical_wrong_filetime_in_external_start_pipe_refuses_before_admission(self):
        write = self.root_proc.stdin.write
        def corrupt_token(data):
            token = json.loads(data)
            token["identity"]["creation_filetime"] = "1"
            write((json.dumps(token) + "\n").encode("utf-8"))
            return len(data)
        with patch.object(self.root_proc.stdin, "write", side_effect=corrupt_token):
            failure = self.refused("green")
        self.assertEqual(failure.record["identity"]["creation_filetime"], str(2**63 + 7))
        self.assertIn("native creation identity", str(self.child_error))
        self.assertNotIn("admission", self.events)
        self.assertEqual(self.requests, [])
        self.assertTrue(self.closed)

    def test_independent_worker_self_identity_mismatch_refuses_before_admission(self):
        with patch.object(bench, "worker_self_identity", return_value=(321, 2**63 + 8)):
            self.refused("green")
        self.assertNotIn("admission", self.events)
        self.assertEqual(self.requests, [])
        self.assertIn("native creation identity", str(self.child_error))

    def test_independent_worker_self_pid_mismatch_refuses_before_admission(self):
        with patch.object(bench, "worker_self_identity", return_value=(322, 2**63 + 7)):
            self.refused("green")
        self.assertNotIn("admission", self.events)
        self.assertEqual(self.requests, [])

    def test_missing_creation_handle_refuses_without_pid_fallback(self):
        self.refused("missing-handle")
        self.assertFalse(self.started)
        self.assertFalse(self.closed)

    def test_protection_failure_cannot_qualify_lifecycle(self):
        self.refused("protection")

    def test_partial_final_receipt_has_no_authoritative_pass(self):
        real_open = Path.open
        class ShortWriter(io.BytesIO):
            def write(self, data):
                return super().write(data[:7])
        def open_file(path, *args, **kwargs):
            if path.name == "lifecycle.json.tmp":
                return ShortWriter()
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", open_file):
            failure = self.refused("green")
        self.assertIn("Short artifact write", failure.record["finalization_error"])
        self.assertFalse((self.output / "lifecycle.json").exists())

    def test_partial_worker_artifact_cannot_return_success(self):
        real_open = Path.open
        class ShortWriter(io.BytesIO):
            def write(self, data):
                return super().write(data[:7])
        def open_file(path, *args, **kwargs):
            return ShortWriter() if path.name == "provisional.json.tmp" else real_open(path, *args, **kwargs)
        with patch.object(Path, "open", open_file):
            failure = self.refused("green")
        self.assertEqual(failure.record["worker_exit"], 1)
        self.assertFalse((self.output / "provisional.json").exists())

    def test_worker_failure_keeps_credentials_out_of_external_capture(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            failure = self.refused("request-secret")
        self.assertEqual(failure.record["worker_exit"], 1)
        self.assertIn("Contained worker stopped (OSError)", stderr.getvalue())
        self.assertNotIn("synthetic-key", stderr.getvalue())
        provisional = (self.output / "provisional.json").read_text()
        self.assertNotIn("synthetic-key", provisional)
        self.assertIn("[REDACTED]", provisional)

    def test_replace_failure_has_no_authoritative_pass(self):
        real_replace = Path.replace
        def replace(path, target):
            if path.name == "lifecycle.json.tmp":
                raise OSError("synthetic replace failure")
            return real_replace(path, target)
        with patch.object(Path, "replace", replace):
            failure = self.refused("green")
        self.assertIn("replace failure", failure.record["finalization_error"])

    def test_artifact_close_failure_has_no_authoritative_pass(self):
        real_open = Path.open
        class BadClose(io.BytesIO):
            def __exit__(self, *args):
                super().__exit__(*args)
                raise OSError("synthetic artifact close failure")
        def open_file(path, *args, **kwargs):
            return BadClose() if path.name == "lifecycle.json.tmp" else real_open(path, *args, **kwargs)
        with patch.object(Path, "open", open_file):
            failure = self.refused("green")
        self.assertIn("artifact close failure", failure.record["finalization_error"])

    def test_finalization_that_returns_late_still_refuses(self):
        real_replace = Path.replace
        def replace(path, target):
            result = real_replace(path, target)
            if path.name == "lifecycle.json.tmp":
                self.now += bench.ACCEPTANCE_SECONDS
            return result
        with patch.object(Path, "replace", replace):
            failure = self.refused("green")
        self.assertFalse(failure.record["within_deadline"])
        self.assertEqual(json.loads((self.output / "lifecycle.json").read_text())["status"], "held")

    def assert_gap_within_total_deadline(self, failure):
        self.assertTrue(failure.record["within_deadline"])
        self.assertEqual(failure.record["memory"]["host_evidence"], "refused")
        self.assertGreater(failure.record["memory"]["max_gap_seconds"], 1)
        self.assertTrue(failure.record["base_host_floors_satisfied"])

    def test_slow_returning_process_creation_is_covered_and_refuses_before_start(self):
        spawn = self.spawn
        def slow_spawn(argv, **kwargs):
            proc = spawn(argv, **kwargs)
            self.now += 5
            return proc
        with patch.object(bench.subprocess, "Popen", side_effect=slow_spawn):
            failure = self.refused("green")
        self.assert_gap_within_total_deadline(failure)
        self.assertFalse(self.started)

    def test_slow_returning_final_wait_cannot_escape_host_coverage(self):
        wait = self.native.wait
        def slow_final_wait(handle, milliseconds):
            result = wait(handle, milliseconds)
            if milliseconds > 0:
                self.now += 5
            return result
        with patch.object(self.native, "wait", side_effect=slow_final_wait):
            failure = self.refused("green")
        self.assert_gap_within_total_deadline(failure)
        self.assertEqual(len(self.requests), 3)
        self.assertTrue(self.closed)

    def test_slow_returning_protection_cannot_escape_host_coverage(self):
        def slow_protection(args):
            self.now += 5
            return True
        with patch.object(bench, "acceptance_protection", side_effect=slow_protection):
            failure = self.refused("green")
        self.assert_gap_within_total_deadline(failure)
        self.assertTrue(failure.record["protection"])
        self.assertTrue(self.closed)

    def test_slow_returning_artifact_tail_cannot_escape_host_coverage(self):
        read = Path.read_bytes
        def slow_read(path):
            data = read(path)
            if path.name == "lifecycle.json":
                self.now += 5
            return data
        with patch.object(Path, "read_bytes", slow_read):
            failure = self.refused("green")
        self.assert_gap_within_total_deadline(failure)
        self.assertTrue(failure.record["finalization"])
        self.assertEqual(json.loads((self.output / "lifecycle.json").read_text())["status"], "held")

    def test_native_bindings_marshal_exact_widths_against_fake_dll_only(self):
        calls = []
        class Function:
            def __init__(self, name):
                self.name = name

            def __call__(self, *args):
                calls.append((self.name, args))
                if self.name == "GetProcessTimes":
                    args[1]._obj.low, args[1]._obj.high = 7, 2**31
                if self.name == "GetProcessId":
                    return 321
                if self.name == "GetCurrentProcess":
                    return -1
                if self.name == "CreateJobObjectW":
                    return 5678
                if self.name == "WaitForSingleObject":
                    return 0
                if self.name == "GetExitCodeProcess":
                    args[1]._obj.value = 0
                return 1
        class DLL:
            def __init__(self):
                self.functions = {}

            def __getattr__(self, name):
                return self.functions.setdefault(name, Function(name))
        dll = DLL()
        with patch.object(ctypes, "WinDLL", return_value=dll):
            api = self.NativeBinding()
            self.assertEqual(api.identity(1234), (321, 2**63 + 7))
            job = api.create_job()
            api.configure_job(job)
            api.assign(job, 1234)
            self.assertEqual(api.active(job), 0)
            self.assertEqual(api.wait(1234, 0), 0)
            self.assertEqual(api.exit_code(1234), 0)
            api.terminate_job(job)
            api.terminate_process(1234)
            api.close(job)
            previous_calls = len(calls)
            self.assertEqual(self.SelfIdentityBinding(), (321, 2**63 + 7))
            self.assertEqual([name for name, _ in calls[previous_calls:]],
                             ["GetCurrentProcess", "GetProcessTimes", "GetProcessId"])
        self.assertEqual(ctypes.sizeof(api.FileTime), 8)
        self.assertEqual(ctypes.sizeof(api.Accounting), 48)
        self.assertEqual(ctypes.sizeof(api.ExtendedLimit), 144 if ctypes.sizeof(ctypes.c_void_p) == 8 else 112)
        limits = next(args for name, args in calls if name == "SetInformationJobObject")
        self.assertEqual(limits[1], 9)
        self.assertEqual(limits[2]._obj.basic.flags, 0x2000)
        self.assertEqual(dll.functions["GetProcessId"].restype, ctypes.c_uint32)
        self.assertEqual(dll.functions["CreateJobObjectW"].restype, ctypes.c_void_p)

    def test_failed_native_close_does_not_detach_popen_authority(self):
        api = object.__new__(self.NativeBinding)
        handle = Mock()
        api.close = Mock(side_effect=OSError("synthetic native close"))
        with self.assertRaises(OSError):
            api.close_process(types.SimpleNamespace(_handle=handle))
        handle.Detach.assert_not_called()

    def test_identity_rejects_float_bool_zero_negative_and_overflow(self):
        for value in ((True, 1), (1, False), (0, 1), (-1, 1), (2**32, 1), (1, 2**64), (1, 1.0)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bench.native_identity(value)
        for value in ("01", "0", "-1", "1.0", str(2**64), True, 1):
            item = {"command": "START", "identity": {"pid": "321", "creation_filetime": value}}
            with self.subTest(transport=value), self.assertRaises(ValueError):
                bench.worker_bootstrap(io.StringIO(json.dumps(item) + "\n"),
                                       lambda _: self.fail("worker entered"), pid=321)

    def test_duration_and_edge_samples_fit_raw_sample_cap(self):
        self.assertLessEqual(bench.ACCEPTANCE_SECONDS / 0.25 + 8, 4096)


if __name__ == "__main__":
    unittest.main()
