"""Offline safety/measurement checks; never loads a model or contacts a server."""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
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

    def test_occupied_port_message_preserves_default_and_names_selected_port(self):
        for port in (8081, 18081):
            with self.subTest(port=port), patch.object(bench.psutil, "process_iter", return_value=[]), \
                    patch.object(bench.socket, "socket") as socket_factory:
                probe = socket_factory.return_value.__enter__.return_value
                probe.bind.side_effect = OSError("occupied")
                with self.assertRaises(RuntimeError) as caught:
                    if port == 8081:
                        bench.assert_idle()
                    else:
                        bench.assert_idle(port)
                expected = {8081: "Port 8081 is occupied; no process was started or stopped.",
                            18081: "Port 18081 is occupied; no process was started or stopped."}
                self.assertEqual(str(caught.exception), expected[port])
                probe.bind.assert_called_once_with(("127.0.0.1", port))

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

    def test_default_sample_limit_remains_4096(self):
        guard = bench.MemoryGuard(Mock(), Path("unused"))
        self.assertEqual(guard.sample_limit, 4096)
        self.assertEqual(self.guard().acceptance_summary()["sample_limit"], 4096)

    def test_explicit_sample_limit_is_enforced_and_reported(self):
        guard = bench.MemoryGuard(Mock(), Path("unused"), acceptance_floors=(4, 4), sample_limit=2)
        self.sample(guard, start=1)
        self.sample(guard, start=2)
        self.assertEqual(guard.acceptance_summary(1, 2)["sample_limit"], 2)
        self.assertEqual(guard.acceptance_summary(1, 2)["host_evidence"], "satisfied")
        self.sample(guard, start=3)
        self.assertEqual(len(guard.host_samples), 2)
        self.assertIn("raw sample limit", guard.reason)
        guard.owner.stop.assert_called_once_with()

    def test_maximum_duration_host_receipt_fits_six_mib(self):
        guard = bench.MemoryGuard(None, None, acceptance_floors=(4, 4), sample_limit=4 * 5400 + 256)
        for index in range(4 * 5400 + 256):
            start = 12345678.123456789 + index / 4
            self.sample(guard, physical=32, commit=64, start=start, end=start + 0.000123456)
        summary = guard.acceptance_summary(guard.host_samples[0]["finished_monotonic"],
                                           guard.host_samples[-1]["started_monotonic"])
        self.assertEqual(summary["host_evidence"], "satisfied")
        self.assertEqual(len(summary["raw_samples"]), 21856)
        self.assertEqual(set(summary["raw_samples"][0]), {"started_monotonic", "finished_monotonic",
                                                       "physical_available_bytes", "commit_available_bytes"})
        record = {"status": "held", "memory": summary,
                  "memory_finalization": "provisional until post-artifact host edge in returned result"}
        payload = (json.dumps(record, indent=2) + "\n").encode("utf-8")
        self.assertEqual(bench.ACCEPTANCE_RECEIPT_LIMIT, 6 * 1024**2)
        self.assertLessEqual(len(payload), bench.ACCEPTANCE_RECEIPT_LIMIT)

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
        scratch = Path(tempfile.gettempdir())
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
        # The resolved unique fixture directory was checked against system temp in setUp.
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

    def test_cli_port_and_acceptance_duration_defaults_and_overrides(self):
        for options, expected in (([], (8081, None)),
                                  (["--port", "18081", "--acceptance-telemetry",
                                    "--acceptance-total-seconds", "5400"], (18081, 5400))):
            with self.subTest(options=options), patch.object(bench, "run") as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(bench.main(["--config", str(self.config), "--output", str(self.output),
                                             *options]), 0)
            args = run.call_args.args[0]
            self.assertEqual((args.port, args.acceptance_total_seconds), expected)

    def test_invalid_ports_refuse_before_reading_config_or_starting_process(self):
        self.args.config = self.root / "missing.json"
        for value in (0, 1023, 65536, -1, True, 8081.5, "8081"):
            self.args.port = value
            with self.subTest(port=value), patch.object(bench.subprocess, "Popen") as popen, \
                    patch.object(bench, "assert_idle") as idle:
                with self.assertRaisesRegex(ValueError, "Port must be between 1024 and 65535"):
                    bench.run(self.args)
            popen.assert_not_called()
            idle.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_invalid_acceptance_durations_refuse_before_reading_config_or_starting_process(self):
        self.args.config = self.root / "missing.json"
        self.args.acceptance_telemetry = True
        for value in (0, 899, 5401, -1, True, 900.5, "900"):
            self.args.acceptance_total_seconds = value
            with self.subTest(seconds=value), patch.object(bench.subprocess, "Popen") as popen, \
                    patch.object(bench, "acceptance_controller") as controller:
                with self.assertRaisesRegex(ValueError, "Acceptance total seconds must be between 900 and 5400"):
                    bench.run(self.args)
            popen.assert_not_called()
            controller.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_acceptance_duration_requires_telemetry_opt_in(self):
        self.args.acceptance_total_seconds = 900
        with patch.object(bench.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "require --acceptance-telemetry"):
                bench.run(self.args)
        popen.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_selected_public_workload_requires_both_explicit_floors_of_at_least_four(self):
        self.args.acceptance_telemetry = True
        self.args.acceptance_workload, self.args.cohort = "short", "saved-static"
        self.args.config = self.root / "missing.json"
        cases = [(None, None), (None, 4), (4, None), (3.9, 4), (4, 3.9)]
        cases += [(value, 4) for value in (True, "4", float("nan"), float("inf"))]
        cases += [(4, value) for value in (True, "4", float("nan"), float("inf"))]
        for physical, commit in cases:
            self.args.acceptance_physical_gib, self.args.acceptance_commit_gib = physical, commit
            with self.subTest(physical=physical, commit=commit), \
                    patch.object(bench.subprocess, "Popen") as popen, \
                    patch.object(bench, "acceptance_controller") as controller:
                with self.assertRaisesRegex(ValueError, "explicit physical and commit floors of at least 4 GiB"):
                    bench.run(self.args)
            popen.assert_not_called()
            controller.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_selected_worker_still_accepts_none_floors(self):
        self.args.acceptance_workload, self.args.cohort = "short", "saved-static"
        self.args._acceptance_deadline = 12345
        self.args.acceptance_telemetry = False
        self.args.acceptance_physical_gib = self.args.acceptance_commit_gib = None
        self.assertEqual(bench.acceptance_selection(self.args), ("short", 131072))

    def test_telemetry_only_preparation_keeps_default_four_gib_floors(self):
        self.args.acceptance_telemetry = True
        with patch.object(bench.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "Stage B/C"):
                bench.run(self.args)
        popen.assert_not_called()
        saved = json.loads((self.output / "results.json").read_text())
        self.assertEqual(saved["memory"]["floors_gib"], {"physical": 4, "commit": 4})
        self.assertEqual(saved["memory"]["sample_limit"], 4096)

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

    def test_controller_default_duration_is_900_seconds_with_cleanup_reserved(self):
        self.args.acceptance_total_seconds = None
        result = self.invoke()
        self.assertAlmostEqual(result["total_deadline"] - result["started_monotonic"], 900)
        self.assertAlmostEqual(result["work_deadline"] - result["started_monotonic"], 890)
        self.assertEqual(result["memory"]["sample_limit"], 3856)

    def test_controller_maximum_duration_scales_deadlines_and_sample_cap(self):
        self.args.acceptance_total_seconds = 5400
        result = self.invoke()
        self.assertAlmostEqual(result["total_deadline"] - result["started_monotonic"], 5400)
        self.assertAlmostEqual(result["work_deadline"] - result["started_monotonic"], 5390)
        self.assertEqual(result["memory"]["sample_limit"], 21856)

    def test_selected_port_reaches_both_idle_checks_clone_and_server_command(self):
        self.args.port = 18081
        commands, clones = [], []
        def spawn(argv, **kwargs):
            if "--engine" in argv:
                commands.append(argv)
                clones.append(json.loads(Path(argv[argv.index("--config") + 1]).read_text()))
            return self.spawn(argv, **kwargs)
        with patch.object(bench, "assert_idle") as idle, \
                patch.object(bench.subprocess, "Popen", side_effect=spawn):
            result = self.invoke()
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertEqual([call.args for call in idle.call_args_list], [(18081,), (18081,)])
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][commands[0].index("--port") + 1], "18081")
        self.assertEqual(clones[0]["port"], 18081)

    def test_controller_uses_six_mib_receipt_cap_and_keeps_other_artifact_caps(self):
        with patch.object(bench, "bounded_write", wraps=bench.bounded_write) as write:
            self.invoke()
        receipt_calls = [call for call in write.call_args_list if call.args[0].name == "lifecycle.json.tmp"]
        self.assertEqual(len(receipt_calls), 1)
        self.assertEqual(receipt_calls[0].kwargs, {"limit": 6 * 1024**2})
        other_calls = [call for call in write.call_args_list if call.args[0].name != "lifecycle.json.tmp"]
        self.assertTrue(other_calls)
        self.assertTrue(all(call.kwargs.get("limit", bench.ACCEPTANCE_LOG_LIMIT) == 1024**2
                            for call in other_calls))
        self.assertEqual(bench.ACCEPTANCE_ARTIFACT_LIMIT, 12 * 1024**2)

    def test_selected_workload_cannot_qualify_a_complete_cpu_receipt_without_runtime_evidence(self):
        self.args.acceptance_workload, self.args.cohort = "short", "saved-static"
        self.args.acceptance_physical_gib = self.args.acceptance_commit_gib = 4
        def workload(validation, client, record, save, guard):
            record.update(acceptance_workload_passed=True, acceptance_cases=[{"passed": True}] * 11,
                          final_idle={"completed": True, "started_monotonic": 10, "finished_monotonic": 40})
            save()
        with patch.object(bench, "acceptance_workload_tests", side_effect=workload), \
                patch.object(bench, "check_authentication"):
            error = self.refused("green")
        result = error.record
        self.assertEqual(result["worker_exit"], 0)
        self.assertTrue(result["natural_exit"])
        self.assertTrue(result["capture_complete"])
        self.assertTrue(result["owned_cleanup"])
        self.assertFalse(result["worker_complete"])
        self.assertTrue(result["workload_evidence"]["acceptance_workload_passed"])
        self.assertTrue(result["workload_evidence"]["missing_evidence"])
        self.assertTrue(self.closed)

    def test_selected_workload_controller_refuses_a_different_cohort_in_the_worker_receipt(self):
        self.args.acceptance_workload, self.args.cohort = "short", "saved-static"
        self.args.acceptance_physical_gib = self.args.acceptance_commit_gib = 4
        def workload(validation, client, record, save, guard):
            record.update(acceptance_workload_passed=True, acceptance_cases=[{"passed": True}] * 11,
                          final_idle={"completed": True, "started_monotonic": 10, "finished_monotonic": 40},
                          cohort="frozen-control", missing_evidence=[])
            save()
        with patch.object(bench, "acceptance_workload_tests", side_effect=workload), \
                patch.object(bench, "check_authentication"):
            error = self.refused("green")
        self.assertFalse(error.record["worker_complete"])
        self.assertEqual(error.record["workload_evidence"]["cohort"], "frozen-control")

    def invoke_selected(self, mutate=None, selection="short", missing=False):
        self.args.acceptance_workload = selection
        self.args.acceptance_physical_gib = self.args.acceptance_commit_gib = 4
        self.args.expected_context = 131072
        self.args.cohort = "saved-static" if selection == "short" else "automatic"
        if selection == "near-full":
            cfg = json.loads(self.config.read_text())
            cfg.setdefault("memory_policy", {"version": 1})
            self.config.write_text(json.dumps(cfg))
        adapter = types.SimpleNamespace(args=self.args, count=StageCSourceTests.count, calls=[], change=None,
            metric_change=None, reference_text="synthetic reference with planted codes")
        base_client = StageCSourceTests.client_type(adapter)
        guards = []
        real_guard, real_workload, real_idle, real_read = (bench.MemoryGuard, bench.acceptance_workload_tests,
                                                         bench.acceptance_idle, Path.read_bytes)
        def make_guard(*args, **kwargs):
            guard = real_guard(*args, **kwargs)
            guards.append(guard)
            return guard
        def advance(seconds):
            while seconds > 0:
                step = min(0.25, seconds)
                self.now += step
                seconds -= step
                guards[0].monitor(once=True)  # Simulate the independent controller sampler during worker time.
        class Client(base_client):
            def chat(client, *args, **kwargs):
                response = super().chat(*args, **kwargs)
                advance(3.6)
                return response
        validation = types.SimpleNamespace(Client=Client,
            long_messages=lambda cfg, target: StageCSourceTests.long_messages(adapter, cfg, target))
        def idle(guard, record, save):
            with patch.object(guard.done, "wait", side_effect=advance):
                real_idle(guard, record, save)
        def workload(*args):
            real_workload(*args)
            record, save = args[2:4]
            if not missing:
                record["missing_evidence"] = []  # Only this CPU fixture supplies the remaining producer seams.
            save()
        def read(path):
            data = real_read(path)
            if mutate and path == self.output / "provisional.json" and self.code == 0:
                item = json.loads(data)
                mutate(item)
                return json.dumps(item).encode()
            return data
        with patch.object(bench, "MemoryGuard", side_effect=make_guard), \
                patch.object(bench, "load_workload", return_value=validation), \
                patch.object(bench, "prompt_counter", return_value=StageCSourceTests.count), \
                patch.object(bench, "check_authentication"), \
                patch.object(bench, "acceptance_workload_tests", side_effect=workload), \
                patch.object(bench, "acceptance_idle", side_effect=idle), patch.object(Path, "read_bytes", read), \
                patch.object(bench.time, "time", side_effect=lambda: 1800000000 + self.now):
            return self.invoke()


class StageCSourceTests(unittest.TestCase):
    """Reach the existing worker body with CPU transports; native/admission stay held."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="stage-c-source-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.config, self.binary = self.root / "saved.json", self.root / "saved.exe"
        self.binary.write_bytes(b"synthetic saved executable, never executed")
        self.cfg = {"exe": str(self.binary), "api_key": "stage-c-private-test-key", "model_name": "synthetic",
                    "fit_max_tokens": True, "open_browser": False, "args": ["--max-context", "131072"],
                    "memory_policy": {"version": 1}, "env": {}}
        self.config.write_text(json.dumps(self.cfg))
        self.args = argparse.Namespace(config=self.config, output=self.root / "output", timing=False,
            smoke=False, acceptance_telemetry=False, acceptance_workload="near-full", cohort="automatic",
            expected_context=131072, _acceptance_deadline=bench.time.monotonic() + 120)
        self.original = self.config.read_bytes()
        self.change = None
        self.metric_change = None
        self.reference_text = "synthetic reference with planted codes"
        self.calls, self.clones, self.processes = [], [], []
        self.server_environments = []

    @staticmethod
    def count(messages):
        system = messages[0].get("content", "")
        if system.startswith("reference-"):
            return int(system.split("-")[1]) + (80 if len(messages) > 2 else 0)
        if system.startswith("Independent sustained"):
            return 128100
        return 60 + 40 * (len(messages) - 1)

    def long_messages(self, cfg, target):
        return ([{"role": "system", "content": "reference-" + str(target)},
                 {"role": "user", "content": self.reference_text}],
                ["field-cedar-284", "marble-finch-693", "cove-lantern-517"], target)

    def client_type(self):
        test = self
        class Client:
            def __init__(self, cfg):
                self.cfg, self.history = cfg, []

            def get(self, path):
                if path == "/health":
                    return {"loaded": True, "max_context": test.args.expected_context}
                return {"engine": {"arena_mib": 4096, "expert_cache_mib": 1024},
                        "totals": {"requests": len(self.history)}, "requests": self.history[:]}

            def chat(self, messages, max_tokens=128, **kwargs):
                test.calls.append((self.label, bench.copy.deepcopy(messages), max_tokens, kwargs))
                count = test.count(messages)
                cached = max(1, count - 80) if "cached" in self.label or "followup" in self.label else 0
                if self.label in ("startup-exact", "short-exact"):
                    text = "391"
                elif self.label == "short-cached-followup":
                    text = "400"
                elif "followup" in self.label:
                    text = "133"
                elif self.label.endswith("-fresh"):
                    text = "field-cedar-284 marble-finch-693 cove-lantern-517"
                elif self.label.endswith("-cached"):
                    text = "cove-lantern-517 391"
                else:
                    text = "A detailed synthetic tutorial."
                generated = 512 if "sustained" in self.label and "followup" not in self.label else 20
                response = {"choices": [{"message": {"role": "assistant", "content": text},
                                          "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": count, "completion_tokens": generated},
                            "timings": {"cache_n": cached, "prompt_n": count - cached,
                                        "prompt_ms": 1200, "predicted_ms": 2400},
                            "first_token_s": 0.25, "wall_seconds": 3.6}
                if test.change:
                    test.change(self.label, response)
                self.history.append({"time": len(self.history) + 1, "prompt_tokens": response["usage"]["prompt_tokens"],
                    "output_tokens": response["usage"]["completion_tokens"], "decode_ms": 2400,
                    "prompt_ms": response["timings"]["prompt_ms"], "reused": response["timings"]["cache_n"],
                    "prompt_total": count, "prompt_read": count - cached,
                    "engine_generated": response["usage"]["completion_tokens"],
                    "finish": response["choices"][0]["finish_reason"]})
                if test.metric_change:
                    test.metric_change(self.history[-1])
                return response
        return Client

    def invoke(self):
        validation = types.SimpleNamespace(Client=self.client_type(), long_messages=self.long_messages)
        def spawn(argv, **kwargs):
            self.assertIn("--config", argv, "Only the intercepted server boundary may be reached")
            clone = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
            self.clones.append(clone)
            self.server_environments.append(dict(kwargs["env"]))
            if self.args.cohort == "automatic":
                self.assertIn("memory_policy", clone, "Automatic policy must reach the real server consumer unchanged")
            else:
                self.assertNotIn("memory_policy", clone, "Static/frozen cohorts must not acquire automatic policy")
            proc = types.SimpleNamespace(stdout=io.BytesIO(b"synthetic server output\n"), poll=lambda: None,
                                         terminate=Mock(), wait=Mock(return_value=0))
            self.processes.append(proc)
            return proc
        real_idle = bench.acceptance_idle
        def bounded_idle(guard, record, save):
            now = [bench.time.monotonic()]
            def wait(seconds):
                now[0] += seconds
            with patch.object(bench.time, "monotonic", side_effect=lambda: now[0]), \
                    patch.object(guard.done, "wait", side_effect=wait):
                real_idle(guard, record, save)
        from tools import prepare_unsloth_q2 as prep
        with patch.object(bench, "SCRATCH", self.root), patch.object(bench, "assert_idle"), \
                patch.object(bench, "source_identity", return_value={"head": "synthetic-current-head"}), \
                patch.object(bench, "load_workload", return_value=validation), \
                patch.object(bench, "prompt_counter", return_value=self.count), \
                patch.object(bench, "check_authentication"), \
                patch.object(bench, "acceptance_idle", side_effect=bounded_idle), \
                patch.object(bench.subprocess, "Popen", side_effect=spawn), \
                patch.object(bench.subprocess, "run", side_effect=AssertionError("external command forbidden")), \
                patch.object(bench, "WindowsProcessAPI", side_effect=AssertionError("native forbidden")), \
                patch.object(bench.psutil, "virtual_memory", side_effect=AssertionError("host read forbidden")), \
                patch.object(bench.psutil, "swap_memory", side_effect=AssertionError("commit read forbidden")), \
                patch.object(prep, "memory_snapshot", side_effect=AssertionError("resource sample forbidden")):
            return bench.run(self.args)

    def saved(self):
        return json.loads((self.args.output / "provisional.json").read_text())

    def test_public_near_full_body_records_separate_fresh_sustained_request_and_actual_history(self):
        result = self.invoke()
        self.assertTrue(result["acceptance_workload_passed"])
        self.assertEqual([row[0] for row in self.calls], ["near-full-fresh", "near-full-cached", "near-full-sustained"])
        self.assertEqual([row[2] for row in self.calls], [2048, 128, 2048])
        self.assertEqual([row["local_prompt_tokens"] for row in result["acceptance_cases"]], [128000, 128080, 128100])
        self.assertEqual(result["acceptance_cases"][2]["response"]["usage"]["completion_tokens"], 512)
        first_answer = result["requests"][0]["response"]["choices"][0]["message"]
        self.assertEqual(self.calls[1][1][2], first_answer)
        self.assertEqual(self.calls[1][1][:2], self.calls[0][1])
        self.assertNotEqual(self.calls[2][1][0], self.calls[0][1][0])
        self.assertTrue(all(row[3]["stream"] for row in self.calls))
        for row in result["requests"]:
            self.assertEqual(row["first_content_seconds"], 0.25)
            self.assertEqual(row["client_wall_seconds"], 3.6)
            content = row["response"]["choices"][0]["message"]["content"]
            self.assertEqual(row["output_sha256"], hashlib.sha256(content.encode()).hexdigest())
            self.assertEqual(row["cached_tokens"], row["metric"]["reused"])
        self.assertTrue(result["protection"]["unchanged"])
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertTrue(result["final_idle"]["completed"])
        self.assertGreaterEqual(result["final_idle"]["finished_monotonic"] - result["final_idle"]["started_monotonic"], 30)
        self.assertEqual(result["executable_provenance"]["saved"]["source_provenance"], "source-unproven")
        self.assertEqual(result["rows"], [])
        self.assertTrue(result["missing_evidence"])
        self.assertFalse(bench.acceptance_ready())
        self.assertEqual(self.saved()["status"], "provisional_complete")
        self.processes[0].terminate.assert_called_once()
        self.processes[0].wait.assert_called_once()

    def test_original_eleven_request_short_sequence_is_distinct_from_near_full(self):
        self.args.acceptance_workload = "short"
        self.args.expected_context = 65536
        self.cfg["args"] = ["--max-context", "65536"]
        self.config.write_text(json.dumps(self.cfg))
        self.original = self.config.read_bytes()
        result = self.invoke()
        self.assertEqual([row[0] for row in self.calls], ["startup-exact", "short-exact", "short-cached-followup",
            "sustained-1", "sustained-1-followup", "sustained-2", "sustained-2-followup",
            "sustained-3", "sustained-3-followup", "2k-fresh", "2k-cached"])
        self.assertEqual([row[2] for row in self.calls], [32, 32, 32, 512, 32, 512, 32, 512, 32, 128, 128])
        self.assertEqual(result["expected_max_context"], 65536)
        self.assertTrue(result["acceptance_workload_passed"])
        self.assertNotIn("long_context", result)
        self.assertEqual(result["final_idle"]["required_seconds"], 30)

    def test_near_full_history_remains_on_wire_while_provisional_artifacts_remain_bounded(self):
        self.reference_text = "synthetic reference data " * 30000
        result = self.invoke()
        self.assertEqual(self.calls[0][1][1]["content"], self.reference_text)
        self.assertEqual(self.calls[1][1][1]["content"], self.reference_text)
        for request, wire in zip(result["requests"], self.calls):
            expected = json.dumps(wire[1], sort_keys=True, separators=(",", ":")).encode()
            self.assertEqual(request["messages_sha256"], hashlib.sha256(expected).hexdigest())
            self.assertNotIn("messages", request)
        self.assertLess((self.args.output / "provisional.json").stat().st_size, 1024**2)
        self.assertTrue(result["acceptance_workload_passed"])

    def test_short_natural_early_stop_keeps_partial_checks_and_owned_cleanup(self):
        self.args.acceptance_workload = "short"
        self.change = lambda label, response: response["usage"].update(completion_tokens=511) if label == "sustained-1" else None
        with self.assertRaisesRegex(RuntimeError, "sustained-1"):
            self.invoke()
        result = self.saved()
        self.assertEqual(result["status"], "provisional_failed")
        self.assertEqual(len(result["acceptance_cases"]), 4)
        self.assertFalse(result["acceptance_cases"][-1]["checks"]["completion_length"])
        self.assertFalse(result["acceptance_workload_passed"])
        self.assertNotIn("final_idle", result)
        self.processes[0].terminate.assert_called_once()

    def test_near_full_refuses_wrong_code_cache_history_count_and_short_sustained_output(self):
        changes = [
            ("near-full-fresh", lambda r: r["choices"][0]["message"].update(content="field-cedar-2840 marble-finch-693 cove-lantern-517")),
            ("near-full-fresh", lambda r: r["usage"].update(prompt_tokens=127999)),
            ("near-full-fresh", lambda r: r["timings"].update(cache_n=1, prompt_n=127999)),
            ("near-full-cached", lambda r: r["choices"][0]["message"].update(content="cove-lantern-517 390")),
            ("near-full-cached", lambda r: r["timings"].update(cache_n=0, prompt_n=128080)),
            ("near-full-cached", lambda r: r["usage"].update(prompt_tokens=128079)),
            ("near-full-sustained", lambda r: r["usage"].update(completion_tokens=511)),
            ("near-full-sustained", lambda r: r["usage"].update(completion_tokens=2049)),
            ("near-full-sustained", lambda r: r["timings"].update(cache_n=128000, prompt_n=100)),
        ]
        for index, (label, change) in enumerate(changes):
            with self.subTest(label=label, index=index):
                self.args.output = self.root / ("output-" + str(index))
                self.change = lambda name, row: change(row) if name == label else None
                with self.assertRaisesRegex(RuntimeError, "Acceptance workload check|Invalid request metric evidence"):
                    self.invoke()
                self.assertFalse(self.saved()["acceptance_workload_passed"])
                self.assertEqual(self.saved()["status"], "provisional_failed")
                self.assertTrue(self.saved()["protection"]["unchanged"])

    def test_missing_stream_latency_is_a_failure_with_answer_and_redacted_receipt(self):
        def change(label, response):
            response.pop("first_token_s")
            response["choices"][0]["message"]["content"] = self.cfg["api_key"]
        self.change = change
        with self.assertRaisesRegex(RuntimeError, "streaming evidence"):
            self.invoke()
        result = self.saved()
        self.assertEqual(result["requests"][0]["response"]["choices"][0]["message"]["content"], "[REDACTED]")
        self.assertEqual(result["requests"][0]["output_sha256"], hashlib.sha256(self.cfg["api_key"].encode()).hexdigest())
        for path in self.args.output.iterdir():
            self.assertNotIn(self.cfg["api_key"], path.read_text())

    def test_prefill_and_cached_token_metrics_must_belong_to_the_returned_response(self):
        for field, value in (("prompt_ms", 1201), ("reused", 1)):
            self.args.output = self.root / field
            self.metric_change = lambda metric: metric.update({field: value})
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "does not agree|streaming evidence|Invalid request metric evidence"):
                self.invoke()
            result = self.saved()
            self.assertEqual(result["status"], "provisional_failed")
            self.assertEqual(result["requests"][0]["response"]["usage"]["completion_tokens"], 20)
            self.assertEqual(result["requests"][0]["metric"][field], value)

    def test_protected_input_mutation_refuses_after_successful_requests_and_preserves_original_receipt(self):
        def change(label, response):
            if label == "near-full-sustained":
                self.binary.write_bytes(b"changed synthetic executable")
        self.change = change
        with self.assertRaisesRegex(RuntimeError, "Benchmark was cancelled"):
            self.invoke()
        result = self.saved()
        self.assertTrue(result["acceptance_workload_passed"])
        self.assertFalse(result["protection"]["unchanged"])
        self.assertEqual(result["status"], "provisional_failed")
        self.assertIn("Protected benchmark inputs changed", result["error"])

    def test_consumed_configuration_must_match_the_before_hash_before_server_creation(self):
        snapshot = bench.protection_snapshot
        def changed(paths):
            self.config.write_bytes(self.original + b"\n")
            return snapshot(paths)
        with patch.object(bench, "assert_idle"), patch.object(bench, "protection_snapshot", side_effect=changed), \
                patch.object(bench.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "changed while binding"):
                bench.run(self.args)
        popen.assert_not_called()
        self.assertFalse(self.args.output.exists())

    def test_selected_cohort_cannot_silently_turn_automatic_into_frozen_budgets(self):
        self.args.acceptance_workload = "short"
        for cohort in ("saved-static", "frozen-control"):
            with self.subTest(cohort=cohort), patch.object(bench, "assert_idle"), \
                    patch.object(bench.subprocess, "Popen", side_effect=AssertionError("process forbidden")):
                self.args.cohort = cohort
                with self.assertRaisesRegex(ValueError, "cohort disagrees"):
                    bench.run(self.args)
                self.assertFalse(self.args.output.exists())

    def test_saved_static_and_frozen_cohorts_preserve_their_explicit_budgets(self):
        self.args.acceptance_workload = "short"
        self.cfg.pop("memory_policy")
        self.cfg["args"] += ["--resident-budget-gib", "39", "--expert-cache", "1"]
        self.cfg["env"] = {"STRATA_FILE_CACHE_MIB": "4096"}
        self.config.write_text(json.dumps(self.cfg))
        self.original = self.config.read_bytes()
        for cohort in ("saved-static", "frozen-control"):
            self.args.cohort = cohort
            self.args.output = self.root / cohort
            result = self.invoke()
            self.assertEqual(result["cohort"], cohort)
            self.assertEqual(self.clones[-1]["args"], ["--max-context", "131072", "--resident-budget-gib", "39", "--expert-cache", "1"])
            self.assertEqual(self.clones[-1]["env"]["STRATA_FILE_CACHE_MIB"], "4096")
            self.assertEqual(self.config.read_bytes(), self.original)

    def test_public_opt_in_is_held_before_native_resource_or_model_entry(self):
        self.args.acceptance_telemetry = True
        self.args.acceptance_physical_gib = self.args.acceptance_commit_gib = 4
        del self.args._acceptance_deadline
        from tools import prepare_unsloth_q2 as prep
        with patch.object(bench.subprocess, "Popen", side_effect=AssertionError("process forbidden")), \
                patch.object(bench, "WindowsProcessAPI", side_effect=AssertionError("native forbidden")), \
                patch.object(bench, "assert_idle", side_effect=AssertionError("inventory forbidden")), \
                patch.object(prep, "memory_snapshot", side_effect=AssertionError("resource sample forbidden")):
            with self.assertRaisesRegex(RuntimeError, "held"):
                bench.run(self.args)
        result = json.loads((self.args.output / "results.json").read_text())
        self.assertEqual(result["acceptance_workload"], "near-full")
        self.assertEqual(result["cohort"], "automatic")
        self.assertEqual(result["status"], "held")
        self.assertEqual(result["effective_environment"], {"CUDA_MODULE_LOADING": "EAGER", "STRATA_FILE_COPY_READY": "0"})

    def test_cli_selects_workload_context_cohort_candidate_and_exact_protected_files(self):
        with patch.object(bench, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bench.main(["--config", str(self.config), "--output", str(self.args.output),
                "--acceptance-telemetry", "--acceptance-workload", "short", "--expected-context", "65536",
                "--acceptance-physical-gib", "4", "--acceptance-commit-gib", "4",
                "--cohort", "saved-static", "--candidate-exe", str(self.binary),
                "--protect-input", str(self.config)]), 0)
        args = run.call_args.args[0]
        self.assertEqual((args.acceptance_workload, args.expected_context, args.cohort), ("short", 65536, "saved-static"))
        self.assertEqual(args.protect_input, [str(self.config)])
        self.assertEqual(args.candidate_exe, self.binary)

    def test_conflicting_or_uncontained_selections_refuse_before_reading_config(self):
        changes = [{"_acceptance_deadline": None}, {"cohort": None}, {"cohort": "saved-static"},
                   {"expected_context": 65536}, {"long_context": True}, {"smoke": True},
                   {"idle_seconds": 1}, {"expected_context": True}, {"acceptance_workload": "unknown"}]
        for changed in changes:
            args = argparse.Namespace(**vars(self.args))
            args.config = self.root / "absent.json"
            vars(args).update(changed)
            with self.subTest(changed=changed), patch.object(bench.subprocess, "Popen") as popen:
                with self.assertRaises(ValueError):
                    bench.run(args)
                popen.assert_not_called()
                self.assertFalse(args.output.exists())

    def test_clean_acceptance_environment_removes_inherited_treatments_and_preserves_explicit_ones(self):
        from serve.server import config_environment
        inherited = {"CUDA_MODULE_LOADING": "LAZY", "STRATA_FILE_COPY_READY": "1",
                     "STRATA_VERIFY_PROFILE": "1", "STRATA_SPLIT_TIMING": "1", "STRATA_TRACE": "1",
                     "CUDA_LAUNCH_BLOCKING": "1", "STRATA_API_KEY": "inherited-private", "KEEP": "value"}
        cfg = {"env": {"STRATA_FILE_CACHE_MIB": "4096"}}
        with patch.dict(bench.os.environ, inherited, clear=True):
            clean, environment = bench.clean_environment(cfg, False, acceptance=True)
        self.assertEqual(environment, {"KEEP": "value"})
        self.assertEqual(config_environment(clean, environment), {"KEEP": "value", "STRATA_FILE_CACHE_MIB": "4096",
            "CUDA_MODULE_LOADING": "EAGER", "STRATA_FILE_COPY_READY": "0"})
        self.assertEqual(cfg, {"env": {"STRATA_FILE_CACHE_MIB": "4096"}})
        explicit = {"env": {"CUDA_MODULE_LOADING": "LAZY", "STRATA_FILE_COPY_READY": "1",
                            "STRATA_VERIFY_PROFILE": "1", "STRATA_API_KEY": "another-private-key"}}
        with patch.dict(bench.os.environ, inherited, clear=True):
            treatment, environment = bench.clean_environment(explicit, False, acceptance=True)
        self.assertEqual(config_environment(treatment, environment), {"KEEP": "value", "CUDA_MODULE_LOADING": "LAZY",
            "STRATA_FILE_COPY_READY": "1", "STRATA_VERIFY_PROFILE": "1"})

    def test_saved_and_candidate_executable_provenance_are_separate_and_receipts_are_claims(self):
        candidate = self.root / "strata-candidate.exe"
        candidate.write_bytes(b"synthetic candidate executable")
        receipt = self.root / "BUILD-candidate.json"
        receipt.write_text(json.dumps({"commit": "older-receipt-commit", "version": "synthetic-version",
            "src": "synthetic-source", "ggml": "synthetic-dependencies",
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}))
        launcher = self.root / "saved-launcher.bat"
        launcher.write_bytes(b"synthetic launcher, never executed")
        self.args.candidate_exe, self.args.protect_input = candidate, [str(launcher)]
        result = self.invoke()
        saved, selected = (result["executable_provenance"][key] for key in ("saved", "selected"))
        self.assertEqual(saved["executable"]["path"], str(self.binary))
        self.assertEqual(saved["source_provenance"], "source-unproven")
        self.assertIsNone(saved["build_claim"])
        self.assertEqual(selected["executable"]["path"], str(candidate))
        self.assertEqual(selected["build_claim"]["commit"], "older-receipt-commit")
        self.assertIn("admission still required", selected["source_provenance"])
        self.assertNotEqual(selected["build_claim"]["commit"], result["working_tree"]["head"])
        self.assertEqual(self.clones[0]["exe"], str(candidate))
        self.assertEqual(self.clones[0]["build_metadata"], str(receipt))
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertIn(str(launcher), [row["path"] for row in result["protection"]["after"]])

    def test_mismatched_build_receipt_refuses_before_the_server_boundary(self):
        receipt = self.root / "BUILD-saved.json"
        receipt.write_text(json.dumps({"commit": "invented", "sha256": "0" * 64}))
        with patch.object(bench, "assert_idle"), patch.object(bench.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "receipt does not match"):
                bench.run(self.args)
        popen.assert_not_called()
        self.assertFalse(self.args.output.exists())

    def test_full_chat_output_room_refuses_before_request_and_generator_count_is_independently_checked(self):
        validation = types.SimpleNamespace(long_messages=Mock(return_value=(
            [{"role": "user", "content": "synthetic"}], ["a", "b", "c"], 128000)))
        client = Mock(cfg={})
        record = {"expected_max_context": 131072, "acceptance_workload": "near-full", "requests": []}
        for count in (128257, 127743, 128001):
            with self.subTest(count=count), self.assertRaisesRegex(RuntimeError, "did not reach"):
                bench.acceptance_retrieval(validation, client, record, Mock(), lambda _: count, 128000)
        client.chat.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "output room"):
            bench.acceptance_request(client, record, Mock(), lambda _: 129020,
                                     "near-full-fresh", [])
        client.chat.assert_not_called()

    def test_actual_readonly_generator_and_server_template_count_leave_unclipped_2048_room(self):
        from serve.server import ByteTokenizer, MockEngine, Service
        from serve.frontend import ChatTemplate
        from tools import needle_bench
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        (tokenizer / "vocab.json").write_text("{}")
        (tokenizer / "merges.txt").write_text("")
        (tokenizer / "token_type.json").write_text("[]")
        (tokenizer / "chat_template.jinja").write_bytes((bench.ROOT / "serve/chat_template.jinja").read_bytes())
        cfg = {"tokenizer": str(tokenizer)}
        validation = bench.load_workload("validation")
        with patch.dict(sys.modules, {"strata_tokenizer": types.SimpleNamespace(Tokenizer=lambda *args: ByteTokenizer())}), \
                patch.object(needle_bench, "haystack", side_effect=lambda n: ("synthetic reference data " * (n // 25 + 1))[:n]):
            messages, codes, count = validation.long_messages(cfg, 128000)
            measured = bench.prompt_counter(cfg)(messages)
        self.assertEqual(count, 128000)
        self.assertEqual(measured, 128000)
        tok = ByteTokenizer()
        engine = MockEngine(tok, " ".join(codes), max_context=131072)
        service = Service(engine, tok, ChatTemplate(tokenizer / "chat_template.jinja"), fit_max_tokens=True)
        ids, _, max_new = service.prepare(messages, None, {"enable_thinking": False}, 2048)
        self.assertEqual(len(ids), 128000)
        self.assertEqual(max_new, 2048)
        self.assertLessEqual(len(ids) + max_new + 8, 131072)
        generated = list(engine.generate(ids, max_new, {}, bench.threading.Event()))
        self.assertEqual(tok.decode(generated[:-1]), "field-cedar-284 marble-finch-693 cove-lantern-517")
        self.assertEqual(engine.last_prompt, ids)

    def test_existing_stream_client_clocks_first_content_and_matches_the_same_request_metric(self):
        validation = bench.load_workload("validation")
        now, total = [100.0], [0]
        metric = {"time": 1, "prompt_tokens": 20, "output_tokens": 2,
                  "prompt_ms": 200, "decode_ms": 1000, "reused": 3}
        class Stream:
            def __enter__(stream): return stream
            def __exit__(stream, *args): return False
            def __iter__(stream):
                frames = [(101.0, {"choices": [{"delta": {"role": "assistant"}}]}),
                          (102.5, {"choices": [{"delta": {"content": "OK"}}]}),
                          (104.0, {"choices": [{"delta": {}, "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 20, "completion_tokens": 2},
                                   "timings": {"prompt_ms": 200, "predicted_ms": 1000, "cache_n": 3, "prompt_n": 17}})]
                for at, frame in frames:
                    now[0] = at
                    yield b"data: " + json.dumps(frame).encode() + b"\n\n"
                yield b"data: [DONE]\n\n"
        def open_request(request, **kwargs):
            if request.full_url.endswith("/metrics"):
                return io.BytesIO(json.dumps({"totals": {"requests": total[0]},
                                              "requests": [metric] if total[0] else []}).encode())
            body = json.loads(request.data)
            self.assertIs(body["stream"], True)
            total[0] += 1
            return Stream()
        record = {"requests": [], "acceptance_workload": "short"}
        guard = bench.MemoryGuard(None, None, deadline=200)
        cfg = {"model_name": "synthetic", "api_key": "synthetic-key", "port": 8081}
        validation.OPENER = types.SimpleNamespace(open=open_request)
        with patch.object(bench.time, "monotonic", side_effect=lambda: now[0]):
            client = bench.make_client(validation, cfg, record, Mock(), guard, "synthetic-key")
            response = client.chat([{"role": "user", "content": "hello"}], 32)
        self.assertEqual(response["choices"][0]["message"]["content"], "OK")
        self.assertEqual(record["requests"][0]["first_content_seconds"], 2.5)
        self.assertEqual(record["requests"][0]["client_wall_seconds"], 4)
        self.assertEqual(record["requests"][0]["metric"], metric)
        self.assertEqual(record["requests"][0]["actual_completion_tokens"], 2)
        self.assertEqual(record["requests"][0]["cached_tokens"], 3)


class StageCRepairTests(unittest.TestCase):
    def source(self):
        case = StageCSourceTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def contained(self):
        case = ContainedLifecycleTests()
        case.setUp()
        return case

    def test_shared_matcher_rejects_invalid_counts_and_timings_before_comparison(self):
        good = {"choices": [], "usage": {"prompt_tokens": 60, "completion_tokens": 20},
                "timings": {"predicted_ms": 2400, "prompt_ms": 1200, "cache_n": 0, "prompt_n": 60}}
        metric = {"time": 1, "prompt_tokens": 60, "output_tokens": 20, "decode_ms": 2400,
                  "prompt_ms": 1200, "reused": 0}
        before = {"totals": {"requests": 0}, "requests": []}
        fields = [("response", "usage", "prompt_tokens"), ("response", "usage", "completion_tokens"),
                  ("response", "timings", "predicted_ms"), ("response", "timings", "prompt_ms"),
                  ("response", "timings", "cache_n"), ("response", "timings", "prompt_n"),
                  *[("metric", None, name) for name in ("time", "prompt_tokens", "output_tokens", "decode_ms", "prompt_ms", "reused")]]
        for owner, section, key in fields:
            for bad in (None, float("nan"), float("inf"), -1, True, "2400"):
                if owner == "metric" and key == "reused" and bad is None:
                    continue  # The actual server uses null for older native telemetry.
                response, changed = bench.copy.deepcopy(good), dict(metric)
                target = response[section] if owner == "response" else changed
                target[key] = bad
                with self.subTest(owner=owner, key=key, bad=bad), self.assertRaises(RuntimeError):
                    bench.matched_metric(response, before, {"totals": {"requests": 1}, "requests": [changed]})
        self.assertEqual(bench.matched_metric(good, before, {"totals": {"requests": 1}, "requests": [metric]}), metric)

    def test_legacy_null_native_counts_survive_but_invalid_known_counts_refuse(self):
        response = {"usage": {"prompt_tokens": 60, "completion_tokens": 20},
                    "timings": {"predicted_ms": 2400, "cache_n": 0}}
        metric = {"time": 1, "prompt_tokens": 60, "output_tokens": 20, "decode_ms": 2400,
                  "prompt_total": 60, "prompt_read": None, "engine_generated": None, "reused": None}
        before = {"totals": {"requests": 0}, "requests": []}
        def match(row):
            return bench.matched_metric(response, before, {"totals": {"requests": 1}, "requests": [row]})
        self.assertEqual(match(metric), metric)
        for key, bad in (("prompt_total", float("nan")), ("prompt_total", 59),
                         ("prompt_read", 61), ("prompt_read", True), ("engine_generated", -1)):
            with self.subTest(key=key, bad=bad), self.assertRaises(RuntimeError):
                match(dict(metric, **{key: bad}))
        class Client:
            def __init__(self, cfg): self.done = False
            def get(self, path):
                return {"totals": {"requests": int(self.done)}, "requests": [metric] if self.done else []}
            def chat(self, *args, **kwargs): self.done = True; return response
        record = {"requests": []}
        client = bench.make_client(types.SimpleNamespace(Client=Client), {}, record, lambda: None,
                                   types.SimpleNamespace(deadline=None, check=lambda: None), "key")
        self.assertEqual(client.chat([{"role": "user", "content": "test"}]), response)

    def test_public_reasoning_wrap_up_preserves_distinct_delivered_and_native_counts(self):
        import threading
        from serve.test_server import ThinkingEngine, ByteTokenizer, ChatTemplate, Service
        from serve.server import openai_chunks, openai_collect
        class MeteredThinking(ThinkingEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                count = 0
                try:
                    for token in super().generate(ids, max_new, sampling, cancel, embeddings):
                        count += 1
                        yield token
                finally:
                    self.last = {"generated": count, "prompt_ms": 100., "decode_ms": 200.,
                                 "reused": 0, "prompt_read": len(ids), "finish": "stop"}
        for budget in (0, 20):
            tok = ByteTokenizer()
            engine = MeteredThinking(tok)
            service = Service(engine, tok, ChatTemplate(Path(__file__).resolve().parents[1] / "serve/chat_template.jinja"))
            ids, thinking, limit = service.prepare([{"role": "user", "content": "2+2?"}], [], {}, 400)
            sampling = {"reasoning_budget_tokens": budget}
            with contextlib.redirect_stdout(io.StringIO()):
                events = list(service.run(ids, thinking, [], limit, sampling, threading.Event()))
            response = openai_collect(openai_chunks(service, sampling, ids, thinking, [], limit,
                                                   threading.Event(), run=iter(events)))
            metric = dict(service.history[-1])
            matched = bench.matched_metric(response, {"totals": {"requests": 0}, "requests": []},
                                            {"totals": {"requests": 1}, "requests": [metric]})
            self.assertEqual(matched, metric)
            self.assertEqual(response["choices"][0]["finish_reason"], "stop")
            self.assertEqual(response["choices"][0]["message"]["content"], "The answer is 4.")
            if budget:
                self.assertGreater(response["usage"]["completion_tokens"], metric["engine_generated"])

    def test_selected_native_counters_cannot_be_missing_or_exceed_the_request(self):
        for key, bad in (("prompt_total", float("nan")), ("prompt_read", 999999),
                         ("engine_generated", 999999), ("prompt_read", None), ("engine_generated", None)):
            case = self.source()
            case.metric_change = lambda metric, key=key, bad=bad: metric.update({key: bad})
            with self.subTest(key=key, bad=bad), self.assertRaises(RuntimeError):
                case.invoke()
            self.assertNotEqual(case.saved()["status"], "complete")

    def test_selected_finish_must_be_successful_and_match_the_response(self):
        for state in (None, "error", "cancel", "disconnect", "length", False):
            case = self.source()
            case.metric_change = lambda metric, state=state: metric.update(finish=state)
            with self.subTest(producer=state), self.assertRaises(RuntimeError):
                case.invoke()
            case = self.contained()
            try:
                def mutation(record):
                    for row in (record["acceptance_cases"][0], record["requests"][0]):
                        row["metric"]["finish"] = state
                with self.subTest(consumer=state), self.assertRaises(bench.AcceptanceFailure):
                    case.invoke_selected(mutation)
            finally:
                case.doCleanups()

    def test_known_cache_read_counts_must_agree_without_requiring_legacy_nulls(self):
        response = {"usage": {"prompt_tokens": 60, "completion_tokens": 20},
                    "timings": {"predicted_ms": 2400, "cache_n": 10, "prompt_n": 50}}
        metric = {"time": 1, "prompt_tokens": 60, "output_tokens": 20, "decode_ms": 2400,
                  "prompt_total": 60, "prompt_read": 50, "reused": 10}
        before = {"totals": {"requests": 0}, "requests": []}
        def match(reply, row):
            return bench.matched_metric(reply, before, {"totals": {"requests": 1}, "requests": [row]})
        self.assertEqual(match(response, metric), metric)
        for cached, read in ((61, 0), (10, 60), (0, 59)):
            changed = bench.copy.deepcopy(response)
            changed["timings"].update(cache_n=cached, prompt_n=read)
            with self.subTest(cached=cached, read=read), self.assertRaises(RuntimeError):
                match(changed, dict(metric, reused=cached, prompt_read=read))
        with self.assertRaises(RuntimeError):
            match(response, dict(metric, reused=None, prompt_read=60))
        legacy = dict(metric, reused=None, prompt_read=None)
        self.assertEqual(match(response, legacy), legacy)

    def test_selected_protection_must_include_every_explicit_input(self):
        for omitted in (False, True):
            case = self.contained()
            try:
                required = case.root / "required.bin"
                required.write_bytes(b"protected input")
                case.args.protect_input = [str(required)]
                def mutation(record):
                    for field in ("before", "after"):
                        record["protection"][field] = [row for row in record["protection"][field]
                                                         if row["path"] != str(required)]
                if omitted:
                    with self.assertRaises(bench.AcceptanceFailure) as caught:
                        case.invoke_selected(mutation)
                    self.assertFalse(caught.exception.record["worker_complete"])
                    self.assertIn("workload_error", caught.exception.record)
                else:
                    self.assertTrue(bench.acceptance_final_predicate(case.invoke_selected()))
            finally:
                case.doCleanups()

    def test_remaining_source_diagnostics_are_cleaned_recorded_and_classified(self):
        presence = {"STRATA_STATE_HASH", "STRATA_VERIFY_NODES", "STRATA_DBG_FEAT",
                    "STRATA_DBG_ROUTE_OVERRIDE", "STRATA_DBG_CPU_GATE"}
        for key in (*sorted(presence), "STRATA_FILE_COPY_STATS", "STRATA_IO_STATS", "STRATA_FS_VERIFY",
                    "STRATA_TEST_VERIFY_STALL", "STRATA_PREFILL_DEMAND_STATS", "STRATA_DBG_GDN",
                    "STRATA_DBG_FORCE_SLOW_LAYER", "STRATA_TEST_PAGEABLE", "STRATA_TEST_CACHE_FAIL"):
            for value in ("", "0", "1"):
                for explicit in (False, True):
                    case = self.source()
                    if explicit:
                        case.cfg["env"] = {key: value}
                        case.config.write_text(json.dumps(case.cfg))
                    with self.subTest(key=key, value=value, explicit=explicit), \
                            patch.dict(bench.os.environ, {key: "1"}, clear=True):
                        record = case.invoke()
                    self.assertEqual(record["timing_diagnostic"], explicit and (key in presence or value == "1"))
                    if explicit:
                        self.assertEqual(record["effective_settings"]["env"][key], value)
                    else:
                        self.assertNotIn(key, record["effective_settings"]["env"])
                        self.assertNotIn(key, case.server_environments[0])
        self.assertTrue(bench.diagnostic_environment({"STRATA_PREFILL_DEMAND_STATS": "10"}))
        self.assertTrue(bench.diagnostic_environment({"STRATA_DBG_GDN": "10"}))
        self.assertTrue(bench.diagnostic_environment({"STRATA_TEST_PAGEABLE": "10"}))
        self.assertFalse(bench.diagnostic_environment({"STRATA_TEST_PAGEABLE": "01"}))
        self.assertTrue(bench.diagnostic_environment({"STRATA_TEST_CACHE_FAIL": "2"}))
        self.assertFalse(bench.diagnostic_environment({"STRATA_TEST_CACHE_FAIL": "-1"}))
        self.assertFalse(bench.diagnostic_environment({"STRATA_DBG_GDN": "01"}))
        self.assertFalse(bench.diagnostic_environment({"STRATA_DBG_FEAT_LAYERS": "3"}))
        self.assertTrue(bench.diagnostic_environment({"STRATA_DBG_FEAT_LAYERS": "3", "STRATA_DBG_FEAT": ""}))
        for key in ("STRATA_VERIFY_DEVICE_PLAN", "STRATA_VERIFY_QDEDUP", "STRATA_VERIFY_COHERENT",
                    "STRATA_VERIFY_ALL_RESIDENT"):
            with patch.dict(bench.os.environ, {key: "1"}, clear=True):
                cfg, inherited = bench.clean_environment({"env": {}}, False, acceptance=True)
            self.assertEqual(inherited[key], "1")
            self.assertEqual(bench.safe_engine_environment(cfg, inherited, acceptance=True)[key], "1")
            self.assertFalse(bench.diagnostic_environment({key: "1"}))

    def test_selected_public_body_refuses_nonfinite_decode_metric_with_partial_answer(self):
        for bad in (None, float("nan"), float("inf"), -1, True):
            case = self.source()
            case.metric_change = lambda metric: metric.update(decode_ms=bad)
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                case.invoke()
            self.assertEqual(case.saved()["status"], "provisional_failed")
            self.assertIn("response", case.saved()["requests"][0])
            case.processes[0].terminate.assert_called_once()

    def test_complete_selected_public_composition_accepts_only_the_complete_synthetic_evidence(self):
        for selection in ("short", "near-full"):
            case = self.contained()
            try:
                result = case.invoke_selected(selection=selection)
                self.assertTrue(bench.acceptance_final_predicate(result))
                self.assertTrue(result["worker_complete"])
                saved = json.loads((case.output / "provisional.json").read_bytes())
                self.assertEqual(len(saved["acceptance_cases"]), 11 if selection == "short" else 3)
                self.assertTrue(all(row["passed"] for row in saved["acceptance_cases"]))
                self.assertTrue(all("response" in row and "metric" in row for row in saved["requests"]))
            finally:
                case.doCleanups()

    def test_complete_public_composition_rejects_malformed_failed_or_mismatched_case_and_request_evidence(self):
        def field(path, value):
            def mutate(record):
                owner = record
                for key in path[:-1]: owner = owner[key]
                owner[path[-1]] = value
            return mutate
        def both_response(row_index, section, key, value):
            def mutate(record):
                for row in (record["acceptance_cases"][row_index], record["requests"][row_index]):
                    row["response"][section][key] = value
            return mutate
        def short_output(record):
            for item in (record["acceptance_cases"][3], record["requests"][3]):
                item["response"]["usage"]["completion_tokens"] = 511
                item["metric"]["output_tokens"] = 511
            record["requests"][3]["actual_completion_tokens"] = 511
        def no_cache(record):
            for item in (record["acceptance_cases"][2], record["requests"][2]):
                item["response"]["timings"].update(cache_n=0, prompt_n=item["response"]["usage"]["prompt_tokens"])
                item["metric"]["reused"] = 0
            record["requests"][2]["cached_tokens"] = 0
        def wrong_answer(record):
            for item in (record["acceptance_cases"][0], record["requests"][0]):
                item["response"]["choices"][0]["message"]["content"] = "390"
            record["requests"][0]["output_sha256"] = hashlib.sha256(b"390").hexdigest()
        mutations = {
            "empty_cases": field(("acceptance_cases",), [{}] * 11),
            "missing_case": lambda r: r["acceptance_cases"].pop(),
            "malformed_case": field(("acceptance_cases", 0), None),
            "false_row": field(("acceptance_cases", 3, "passed"), False),
            "nonbool_passed": field(("acceptance_cases", 3, "passed"), 1),
            "false_check": field(("acceptance_cases", 3, "checks", "completion_length"), False),
            "missing_check": lambda r: r["acceptance_cases"][0]["checks"].pop("exact_answer"),
            "wrong_label": field(("acceptance_cases", 0, "case"), "near-full-sustained"),
            "duplicate_case": lambda r: r["acceptance_cases"].__setitem__(1, r["acceptance_cases"][0]),
            "case_order": lambda r: r["acceptance_cases"].reverse(),
            "no_requests": field(("requests",), []),
            "missing_request": lambda r: r["requests"].pop(),
            "malformed_request": field(("requests", 0), {}),
            "duplicate_request": lambda r: r["requests"].__setitem__(1, r["requests"][0]),
            "request_order": lambda r: r["requests"].reverse(),
            "wrong_request_label": field(("requests", 0, "label"), "invented"),
            "mismatched_response": field(("requests", 0, "response", "usage", "completion_tokens"), 19),
            "wrong_output_digest": field(("requests", 0, "output_sha256"), "0" * 64),
            "missing_first_content": lambda r: r["requests"][0].pop("first_content_seconds"),
            "malformed_wall_clock": field(("requests", 0, "started_unix"), float("nan")),
            "false_completion_count": field(("requests", 0, "actual_completion_tokens"), 21),
            "wrong_max_tokens": field(("requests", 0, "max_tokens"), 31),
            "false_actual_length": both_response(3, "usage", "completion_tokens", 511),
            "false_actual_cache": both_response(2, "timings", "cache_n", 0),
            "coherent_short_output": short_output,
            "coherent_no_cache": no_cache,
            "coherent_wrong_answer": wrong_answer,
            "nonfinite_actual_timing": both_response(0, "timings", "predicted_ms", float("nan")),
            "wrong_target_count": field(("acceptance_cases", 9, "local_prompt_tokens"), 1024),
            "wrong_expected_answer": field(("acceptance_cases", 0, "expected"), [390]),
            "false_aggregate": field(("acceptance_summary", "completion_tokens"), 1),
            "history_missing": lambda r: r["requests"][2]["message_digests"].pop(1),
            "history_wrong_answer": field(("requests", 2, "message_digests", 1, "content_sha256"), "0" * 64),
            "history_wrong_message": field(("requests", 2, "message_digests", 1, "message_sha256"), "0" * 64),
            "false_idle": field(("final_idle", "completed"), False),
            "short_idle": lambda r: r["final_idle"].update(finished_monotonic=r["final_idle"]["started_monotonic"] + 29),
            "future_idle": field(("final_idle", "finished_monotonic"), 900),
            "missing_protection": field(("protection", "before"), []),
            "changed_protection": field(("protection", "after", 0, "sha256"), "0" * 64),
            "malformed_protection": field(("protection", "before", 0, "bytes"), True),
        }
        for name, mutate in mutations.items():
            case = self.contained()
            try:
                with self.subTest(name=name), self.assertRaises(bench.AcceptanceFailure) as caught:
                    case.invoke_selected(mutate)
                result = caught.exception.record
                self.assertEqual(result["worker_exit"], 0, "Mutation must reach the external consumer")
                self.assertFalse(result["worker_complete"])
                self.assertIn("workload_error", result)
                self.assertFalse(bench.acceptance_final_predicate(result))
            finally:
                case.doCleanups()

    def test_near_full_public_consumer_rechecks_fresh_output_room_and_sustained_length(self):
        def wrong(record, field, value):
            row, request = record["acceptance_cases"][2], record["requests"][2]
            for item in (row, request): item["response"]["usage"][field] = value
        mutations = [lambda r: wrong(r, "completion_tokens", 511),
                     lambda r: r["acceptance_cases"][2].update(local_prompt_tokens=129100),
                     lambda r: r["requests"][2]["message_digests"].__setitem__(0, r["requests"][0]["message_digests"][0])]
        for mutate in mutations:
            case = self.contained()
            try:
                with self.assertRaises(bench.AcceptanceFailure) as caught:
                    case.invoke_selected(mutate, selection="near-full")
                self.assertEqual(caught.exception.record["worker_exit"], 0)
                self.assertFalse(caught.exception.record["worker_complete"])
            finally:
                case.doCleanups()

    def test_complete_selected_evidence_still_refuses_real_missing_producer_holds(self):
        case = self.contained()
        try:
            with self.assertRaises(bench.AcceptanceFailure) as caught:
                case.invoke_selected(missing=True)
            self.assertFalse(caught.exception.record["worker_complete"])
            self.assertTrue(caught.exception.record["workload_evidence"]["missing_evidence"])
        finally:
            case.doCleanups()

    def test_source_established_diagnostic_modes_layers_and_effective_values(self):
        presence = ("STRATA_POOL_TRACE", "STRATA_RSS_TRACE", "STRATA_TRACE_ADAPT", "STRATA_VERIFY_DEBUG",
                    "STRATA_DRAFT_TIMING", "STRATA_PREFILL_TIMING", "STRATA_VERIFY_PROFILE",
                    "STRATA_DECODE_TIMING", "STRATA_SPLIT_TIMING", "STRATA_TRACE", "STRATA_LOGPOS", "STRATA_DUMP_FIRST_LOGITS",
                    "STRATA_VERIFY_ARENA", "STRATA_DBG_NAN", "STRATA_IDX_FP16_CHECK", "STRATA_QSA_DUMP",
                    "STRATA_PREFILL_DUMP_R", "STRATA_PREFILL_DUMP_R_ALL", "STRATA_STATE_HASH_GDN", "STRATA_SNAPSHOT_VERIFY")
        for key in presence + ("STRATA_BATCH_TRACE", "STRATA_DEBUG", "STRATA_FETCH_PROFILE", "STRATA_VERIFY_TRACE"):
            for value in ("", "0", "1"):
                with self.subTest(key=key, value=value):
                    with patch.dict(bench.os.environ, {key: value, "STRATA_FILE_CACHE_MIB": "4096", "STRATA_PIPELINE_THETA": "0.8"}, clear=True):
                        cfg, inherited = bench.clean_environment({"env": {}}, False, acceptance=True)
                        self.assertNotIn(key, inherited)
                        self.assertEqual(inherited["STRATA_PIPELINE_THETA"], "0.8")
                        self.assertEqual(inherited["STRATA_FILE_CACHE_MIB"], "4096")
                        legacy_cfg, legacy = bench.clean_environment({"env": {key: value}}, False)
                        if key not in ("STRATA_VERIFY_PROFILE", "STRATA_DECODE_TIMING", "STRATA_FETCH_PROFILE"):
                            self.assertEqual(legacy[key], value)
                            self.assertEqual(legacy_cfg["env"][key], value)
                    with patch.dict(bench.os.environ, {}, clear=True):
                        cfg, inherited = bench.clean_environment({"env": {key: value}}, False, acceptance=True)
                    evidence = bench.safe_engine_environment(cfg, inherited, acceptance=True)
                    self.assertEqual(evidence[key], value)
                    expected = key in presence or (bool(value) if key in ("STRATA_BATCH_TRACE", "STRATA_DEBUG") else value == "1")
                    self.assertEqual(bench.diagnostic_environment(evidence), expected)
                    if key in ("STRATA_BATCH_TRACE", "STRATA_DEBUG"):
                        self.assertEqual(inherited[key], value)
        for gate in ("0", "1"):
            with patch.dict(bench.os.environ, {"STRATA_PIPELINE_DEBUG": gate, "STRATA_PIPELINE_LOG": "0",
                                             "STRATA_PIPELINE_FORCE_MISS": "1"}, clear=True):
                cfg, inherited = bench.clean_environment({"env": {}}, False, acceptance=True)
                self.assertEqual(inherited["STRATA_PIPELINE_DEBUG"], gate)
                self.assertNotIn("STRATA_PIPELINE_FORCE_MISS", inherited)
                self.assertNotIn("STRATA_PIPELINE_LOG", inherited)
                cfg, inherited = bench.clean_environment({"env": {"STRATA_PIPELINE_LOG": "0"}}, False, acceptance=True)
                self.assertEqual(bench.diagnostic_environment(bench.safe_engine_environment(cfg, inherited, acceptance=True)), gate == "1")
        for name in ("STRATA_LOGPOS_EXTRA", "STRATA_LOGPOS_TOPK", "STRATA_QSA_DUMP_LAST"):
            parent = "STRATA_QSA_DUMP" if name == "STRATA_QSA_DUMP_LAST" else "STRATA_LOGPOS"
            self.assertFalse(bench.diagnostic_environment({name: "1"}))
            self.assertTrue(bench.diagnostic_environment({name: "1", parent: ""}))
        for name in ("STRATA_HC_UPMIX_CHECK", "STRATA_CVEC_FUSE_CHECK"):
            for value, expected in (("0", False), ("-1", False), ("2", True), (" 2suffix", True)):
                self.assertEqual(bench.diagnostic_environment({name: value}), expected)
        self.assertFalse(bench.diagnostic_environment({"STRATA_REQUEST_LINES": "0"}, server_log=False))
        self.assertTrue(bench.diagnostic_environment({"STRATA_REQUEST_LINES": "0"}, server_log=True))
        for value, expected in (("00", False), ("foo", False), ("-1", True), (" 01suffix", True)):
            self.assertEqual(bench.diagnostic_environment({"CUDA_LAUNCH_BLOCKING": value}), expected)
        self.assertFalse(bench.diagnostic_environment({"CUDA_LAUNCH_BLOCKING": "1"}, backend="hip"))

    def test_selected_launch_removes_inherited_aliases_and_records_explicit_presence_zero(self):
        keys = ("STRATA_BATCH_TRACE", "STRATA_POOL_TRACE", "STRATA_RSS_TRACE", "STRATA_TRACE_ADAPT")
        for explicit in (False, True):
            case = self.source()
            if explicit:
                case.cfg["env"] = {key: "0" for key in keys}
                case.config.write_text(json.dumps(case.cfg))
            with patch.dict(bench.os.environ, {key: "1" for key in keys}, clear=True):
                result = case.invoke()
            self.assertEqual(result["timing_diagnostic"], explicit)
            for key in keys:
                if explicit:
                    self.assertEqual(result["effective_settings"]["env"][key], "0")
                else:
                    self.assertNotIn(key, result["effective_settings"]["env"])
                    self.assertNotIn(key, case.server_environments[0])
            if explicit:
                self.assertEqual(case.server_environments[0]["STRATA_BATCH_TRACE"], "0")


class NormalComparisonTests(unittest.TestCase):
    def source(self, *, automatic=False):
        case = StageCSourceTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        if not automatic:
            # Control-classification tests use a launch whose controls cannot refresh.
            case.cfg.pop("memory_policy")
            case.args.cohort = "saved-static"
            case.args.acceptance_workload = "short"
            case.config.write_text(json.dumps(case.cfg))
        return case

    @staticmethod
    def effective(case):
        from serve.server import config_environment
        return config_environment(case.clones[0], case.server_environments[0])

    def test_public_active_profile_with_invalid_name_retains_diagnostic_label_on_failure(self):
        for cohort in ("automatic", "saved-static", "frozen-control"):
            for failed in (False, True):
                with self.subTest(cohort=cohort, failed=failed):
                    case = self.source(automatic=cohort == "automatic")
                    case.args.cohort = cohort
                    case.cfg["env"] = {"STRATA_VERIFY_PROFILE": "0", "STRATA_BAD-NAME": "0"}
                    case.config.write_text(json.dumps(case.cfg))
                    if failed:
                        target = "near-full-sustained" if cohort == "automatic" else "sustained-1"
                        case.change = lambda label, response: response["usage"].update(
                            completion_tokens=511) if label == target else None
                    with patch.dict(bench.os.environ, {}, clear=True):
                        if failed:
                            with self.assertRaisesRegex(RuntimeError, "Acceptance workload check"):
                                case.invoke()
                            result = case.saved()
                        else:
                            result = case.invoke()
                    self.assertEqual(bench.active_diagnostics(self.effective(case)), ["STRATA_VERIFY_PROFILE"])
                    self.assertTrue(result["timing_diagnostic"])
                    self.assertEqual(result["acceptance_workload_passed"], not failed)
                    self.assertEqual(result["status"], "provisional_failed" if failed else "complete")
                    self.assertEqual(result["normal_comparison"]["controls"], [])
                    self.assertFalse(result["normal_comparison"]["complete"])
                    self.assertFalse(result["normal_comparison"]["eligible"])
                    self.assertIsNone(result["normal_comparison"]["environment_sha256"])
                    self.assertEqual(bench.consume_normal_comparison(
                        result, case.clones[0], case.server_environments[0]), result["normal_comparison"])

    def test_public_early_inventory_refusal_is_diagnostically_uncertain_without_unbounded_parsing(self):
        specimens = [
            ({"STRATA_IO_STATS": "9" * 4097}, {}, "control-inventory-limit"),
            ({"STRATA_PRIVATE_" + str(i): "0" for i in range(129)}, {}, "control-inventory-limit"),
            ({"STRATA_PRIVATE_" + str(i): "x" * 4096 for i in range(17)}, {}, "control-inventory-limit"),
            ({}, {"OTHER_" + str(i): "0" for i in range(4097)}, "environment-limit"),
            ({"STRATA_BAD-NAME": "PRIVATE-VALUE-SENTINEL"}, {}, "invalid-environment"),
            ({"strata_file_copy_ready": "1", "STRATA_FILE_COPY_READY": "0"}, {}, "ambiguous-windows-alias"),
        ]
        for automatic in (False, True):
            for configured, inherited, reason in specimens:
                with self.subTest(automatic=automatic, reason=reason):
                    case = self.source(automatic=automatic)
                    case.cfg["env"] = configured
                    case.config.write_text(json.dumps(case.cfg))
                    with patch.dict(bench.os.environ, inherited, clear=True), \
                            patch.object(bench, "active_diagnostics", side_effect=AssertionError("unbounded diagnostic scan")):
                        result = case.invoke()
                    self.assertTrue(result["timing_diagnostic"])
                    self.assertTrue(result["acceptance_workload_passed"])
                    self.assertEqual(result["normal_comparison"]["exclusions"][0], {"reason": reason})
                    self.assertFalse(result["normal_comparison"]["complete"])
                    self.assertFalse(result["normal_comparison"]["eligible"])
                    self.assertIsNone(result["normal_comparison"]["environment_sha256"])
                    self.assertNotIn("PRIVATE", json.dumps(result["normal_comparison"]))

    def test_public_complete_source_inventory_keeps_diagnostic_gates_separate_from_binding(self):
        specimens = [({}, "cuda", False), ({"STRATA_PRIVATE_UNKNOWN": "0"}, "cuda", False),
            ({"STRATA_VERIFY_PROFILE": ""}, "cuda", True), ({"STRATA_VERIFY_PROFILE": "0"}, "cuda", True),
            ({"STRATA_FETCH_PROFILE": "0"}, "cuda", False), ({"STRATA_FETCH_PROFILE": "1"}, "cuda", True),
            ({"CUDA_LAUNCH_BLOCKING": "1"}, "hip", False), ({"CUDA_LAUNCH_BLOCKING": "1"}, "cuda", True)]
        for cohort in ("automatic", "saved-static", "frozen-control"):
            for controls, backend, diagnostic in specimens:
                with self.subTest(cohort=cohort, controls=controls, backend=backend):
                    case = self.source(automatic=cohort == "automatic")
                    case.args.cohort = cohort
                    case.cfg.update(env=controls, backend=backend)
                    case.config.write_text(json.dumps(case.cfg))
                    with patch.dict(bench.os.environ, {}, clear=True):
                        result = case.invoke()
                    self.assertEqual(result["timing_diagnostic"], diagnostic)
                    self.assertTrue(result["acceptance_workload_passed"])
                    classified = bench.normal_comparison_environment(case.clones[0], case.server_environments[0])
                    self.assertTrue(classified["complete"])
                    self.assertEqual(result["normal_comparison"]["complete"], cohort != "automatic")

    def test_automatic_refresh_public_worker_and_outer_consumer_have_no_final_digest(self):
        from tools import prepare_unsloth_q2 as prep
        case = ContainedLifecycleTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        plan = prep.memory_plan(64, 6, 131072, available_gib=56, commit_available_gib=60)
        cfg = json.loads(case.config.read_text())
        cfg.update(args=plan["args"], env=plan["env"], memory_policy={
            "version": 1, "generated_args": plan["args"], "generated_env": plan["env"],
            "overrides": {"args": {}, "env": {}}})
        case.config.write_text(json.dumps(cfg))
        observed = []
        original_spawn = case.spawn

        def spawn(argv, **kwargs):
            if "--engine" in argv:
                path = Path(argv[argv.index("--config") + 1])
                template = json.loads(path.read_text())
                before = bench.normal_comparison_environment(template, kwargs["env"])
                # Execute the server's actual refresh/planner/admission owners;
                # GPU/host discovery and model-file readiness are external fakes.
                with patch.dict(bench.os.environ, kwargs["env"], clear=True), \
                        patch.object(prep, "check_ready", return_value=None), \
                        contextlib.redirect_stdout(io.StringIO()):
                    refreshed = prep.prepare_launch(path)
                after = bench.normal_comparison_environment(refreshed, kwargs["env"])
                self.assertTrue(before["eligible"])
                self.assertTrue(after["eligible"])
                self.assertNotEqual(template["env"]["STRATA_FILE_CACHE_MIB"],
                                    refreshed["env"]["STRATA_FILE_CACHE_MIB"])
                self.assertNotEqual(before["environment_sha256"], after["environment_sha256"])
                self.assertEqual(json.loads(path.read_text()), template)
                observed.append((before, after))
            return original_spawn(argv, **kwargs)

        sample = {"ram_gib": 64, "available_gib": 40, "commit_available_gib": 50, "windows_commit": True}
        with patch.dict(bench.os.environ, {}, clear=True), \
                patch.object(bench.subprocess, "Popen", side_effect=spawn), \
                patch.object(prep, "memory_snapshot", side_effect=lambda: dict(case.sample(), **sample)), \
                patch.object(prep.setup, "gpu_info", return_value={"index": 0, "vram_gb": 6}):
            result = case.invoke_selected(selection="near-full")
        self.assertEqual(len(observed), 1)
        worker = json.loads((case.output / "provisional.json").read_text())
        for record in (worker, result):
            evidence = record["normal_comparison"]
            self.assertFalse(evidence["eligible"])
            self.assertFalse(evidence["complete"])
            self.assertIsNone(evidence["environment_sha256"])
            self.assertEqual(evidence["controls"], observed[0][0]["controls"])
            self.assertEqual(evidence["exclusions"], [{"reason": "automatic-environment-unbound"}])
        self.assertEqual(worker["status"], "provisional_complete")
        self.assertTrue(worker["acceptance_workload_passed"])
        self.assertFalse(worker["timing_diagnostic"])
        self.assertTrue(result["worker_complete"])
        self.assertTrue(bench.acceptance_final_predicate(result))  # Existing CPU lifecycle predicate only.
        self.assertEqual(json.loads((case.output / "lifecycle.json").read_text())["status"], "held")
        case.doCleanups()
        self.assertFalse(bench.acceptance_ready())

    def test_automatic_unbound_retains_diagnostics_unknowns_and_failed_workload_evidence(self):
        unknown = "STRATA_PRIVATE_UNKNOWN_SENTINEL"
        for controls, reason, diagnostic in (({}, None, False),
                ({unknown: ""}, "unclassified-control", False),
                ({unknown: "0"}, "unclassified-control", False),
                ({"STRATA_VERIFY_PROFILE": "0"}, "active-diagnostic", True)):
            for failed in (False, True):
                with self.subTest(controls=controls, failed=failed):
                    case = self.source(automatic=True)
                    case.cfg["env"] = controls
                    case.config.write_text(json.dumps(case.cfg))
                    if failed:
                        case.change = lambda label, response: response["usage"].update(
                            completion_tokens=511) if label == "near-full-sustained" else None
                    with patch.dict(bench.os.environ, {}, clear=True):
                        if failed:
                            with self.assertRaisesRegex(RuntimeError, "Acceptance workload check"):
                                case.invoke()
                            result = case.saved()
                        else:
                            result = case.invoke()
                    classified = bench.normal_comparison_environment(case.clones[0], case.server_environments[0])
                    evidence = result["normal_comparison"]
                    self.assertEqual(classified["eligible"], reason is None)
                    self.assertTrue(classified["complete"])
                    self.assertIsNotNone(classified["environment_sha256"])
                    self.assertFalse(evidence["complete"])
                    self.assertFalse(evidence["eligible"])
                    self.assertIsNone(evidence["environment_sha256"])
                    self.assertEqual(evidence["controls"], classified["controls"])
                    self.assertEqual(evidence["exclusions"], classified["exclusions"] +
                                     [{"reason": "automatic-environment-unbound"}])
                    self.assertEqual(result["timing_diagnostic"], diagnostic)
                    self.assertEqual(result["acceptance_workload_passed"], not failed)
                    self.assertEqual(result["status"], "provisional_failed" if failed else "complete")
                    self.assertEqual(len(result["requests"]), 3)
                    self.assertNotIn(unknown, json.dumps(evidence))
                    self.assertEqual(bench.consume_normal_comparison(
                        result, case.clones[0], case.server_environments[0]), evidence)

    def test_policy_removed_frozen_and_static_launches_keep_exact_classification(self):
        from tools import prepare_unsloth_q2 as prep
        plan = prep.memory_plan(64, 6, 131072, available_gib=56, commit_available_gib=60)
        for cohort in ("saved-static", "frozen-control"):
            with self.subTest(cohort=cohort):
                case = self.source()
                case.args.cohort = cohort
                case.cfg.update(args=plan["args"], env=plan["env"])
                case.config.write_text(json.dumps(case.cfg))
                with patch.dict(bench.os.environ, {}, clear=True):
                    result = case.invoke()
                self.assertNotIn("memory_policy", case.clones[0])
                classified = bench.normal_comparison_environment(case.clones[0], case.server_environments[0])
                self.assertEqual(result["normal_comparison"], classified)
                self.assertTrue(classified["complete"])
                self.assertTrue(classified["eligible"])
                pairs = sorted([name, value] for name, value in self.effective(case).items()
                               if name.startswith("STRATA_") or name == "CUDA_MODULE_LOADING")
                expected = hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()
                self.assertEqual(classified["environment_sha256"], expected)
                self.assertEqual(bench.consume_normal_comparison(
                    result, case.clones[0], case.server_environments[0]), classified)
                self.assertTrue(result["acceptance_workload_passed"])

    def test_automatic_consumer_refuses_legacy_forged_and_typed_binding_claims(self):
        cfg = {"memory_policy": {"version": 1}, "env": {"STRATA_FILE_CACHE_MIB": "4096"}}
        classified = bench.normal_comparison_environment(cfg, {})
        self.assertTrue(classified["eligible"])  # The pure classifier retains its original scope.
        unbound = dict(classified, complete=False, eligible=False, environment_sha256=None,
                       exclusions=[{"reason": "automatic-environment-unbound"}])
        self.assertEqual(bench.consume_normal_comparison({"normal_comparison": unbound}, cfg, {}), unbound)
        for evidence in (None, classified, dict(unbound, eligible=True), dict(unbound, eligible=0),
                         dict(unbound, complete=0), dict(unbound, controls=[]), dict(unbound, exclusions=[]),
                         dict(unbound, environment_sha256=classified["environment_sha256"]),
                         dict(unbound, exclusions=[{"reason": "PRIVATE-FORGED-SENTINEL"}])):
            result = bench.consume_normal_comparison({"normal_comparison": evidence}, cfg, {})
            self.assertFalse(result["eligible"])
            self.assertFalse(result["complete"])
            self.assertIsNone(result["environment_sha256"])
            self.assertEqual(result["exclusions"], [{"reason": "missing-or-inconsistent-classification"}])
            self.assertNotIn("PRIVATE", json.dumps(result))

    def test_legacy_automatic_public_flow_retains_results_without_a_final_binding(self):
        case = RunTests()
        case.setUp()
        self.addCleanup(case.tearDown)
        cfg = json.loads(case.config.read_text())
        cfg["memory_policy"] = {"version": 1}
        case.config.write_text(json.dumps(cfg))
        case.original = case.config.read_bytes()
        with patch.dict(bench.os.environ, {}, clear=True):
            result = case.invoke()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["rows"]), 6)
        self.assertEqual(len(result["requests"]), 7)
        self.assertEqual(result["summary"]["completion_tokens"], 960)
        self.assertFalse(result["timing_diagnostic"])
        self.assertFalse(result["normal_comparison"]["complete"])
        self.assertFalse(result["normal_comparison"]["eligible"])
        self.assertIsNone(result["normal_comparison"]["environment_sha256"])
        self.assertEqual(result["normal_comparison"]["exclusions"], [{"reason": "automatic-environment-unbound"}])
        case.assert_safe_artifacts()

    def test_public_known_tuning_controls_are_preserved_and_explicitly_normal(self):
        controls = {"STRATA_FILE_CACHE_MIB": "4096", "STRATA_FILE_COPY_READY": "1",
                    "STRATA_PIPELINE_DEBUG": "1", "STRATA_PIPELINE_THETA": "0.8",
                    "STRATA_VERIFY_DEVICE_PLAN": "1", "STRATA_VERIFY_QDEDUP": "1",
                    "STRATA_VERIFY_COHERENT": "1", "STRATA_VERIFY_ALL_RESIDENT": "0"}
        case = self.source()
        case.cfg["env"] = dict(controls)
        case.config.write_text(json.dumps(case.cfg))
        with patch.dict(bench.os.environ, {}, clear=True):
            result = case.invoke()
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["acceptance_workload_passed"])
        self.assertTrue(result["normal_comparison"]["complete"])
        self.assertTrue(result["normal_comparison"]["eligible"])
        self.assertEqual(result["normal_comparison"]["exclusions"], [])
        self.assertFalse(result["timing_diagnostic"])
        effective = self.effective(case)
        for key, value in controls.items():
            self.assertEqual(effective[key], value)
        self.assertEqual({item["kind"] for item in result["normal_comparison"]["controls"]}, {"reviewed-normal"})

    def test_public_unseen_controls_survive_both_layers_even_empty_or_zero_without_value_leaks(self):
        key, secret = "STRATA_FUTURE_TREATMENT_74B9", "UNCLASSIFIED-SECRET-SENTINEL-20261022"
        for value in ("", "0", secret):
            for explicit in (False, True):
                case = self.source()
                if explicit:
                    case.cfg["env"] = {key: value}
                    case.config.write_text(json.dumps(case.cfg))
                with self.subTest(value_empty=value == "", explicit=explicit), \
                        patch.dict(bench.os.environ, {key: value}, clear=True):
                    result = case.invoke()
                self.assertEqual(self.effective(case)[key], value)
                self.assertTrue(result["normal_comparison"]["complete"])
                self.assertFalse(result["normal_comparison"]["eligible"])
                self.assertEqual(result["normal_comparison"]["exclusions"], [{"reason": "unclassified-control",
                    "control": "sha256:" + hashlib.sha256(key.encode()).hexdigest()}])
                self.assertTrue(result["acceptance_workload_passed"])
                self.assertEqual(result["status"], "complete")
                self.assertFalse(result["timing_diagnostic"])
                self.assertNotIn(key, json.dumps(result["normal_comparison"]))
                for path in case.args.output.iterdir():
                    self.assertNotIn(secret, path.read_text())
                    self.assertNotIn(case.cfg["api_key"], path.read_text())

    def test_amd_setup_controls_are_reviewed_treatments_and_bind_exact_values(self):
        controls = {
            "STRATA_HIPBLASLT_TUNING": ("", "0", "C:/PRIVATE-TUNING-PATH-SENTINEL/table.txt"),
            "STRATA_NO_ARENA_THP": ("", "0", "1"),
            "STRATA_RESIDENT_PIN": ("", "0", "1", "00"),
        }
        absent = bench.normal_comparison_environment({"backend": "hip", "env": {}}, {})
        self.assertTrue(absent["eligible"])
        self.assertEqual(absent["controls"], [])
        for key, values in controls.items():
            for value in values:
                with self.subTest(control=key, value=value):
                    cfg = {"backend": "hip", "env": {key: value}}
                    evidence = bench.normal_comparison_environment(cfg, {})
                    self.assertTrue(evidence["complete"])
                    self.assertTrue(evidence["eligible"])
                    self.assertEqual(evidence["exclusions"], [])
                    self.assertEqual(evidence["controls"], [{"control": key, "kind": "reviewed-normal"}])
                    encoded = json.dumps([[key, value]], separators=(",", ":")).encode()
                    self.assertEqual(evidence["environment_sha256"], hashlib.sha256(encoded).hexdigest())
                    self.assertEqual(bench.consume_normal_comparison({"normal_comparison": evidence}, cfg, {}), evidence)
                    changed = {"backend": "hip", "env": {key: value + "-changed"}}
                    rejected = bench.consume_normal_comparison({"normal_comparison": evidence}, changed, {})
                    self.assertFalse(rejected["eligible"])
                    self.assertFalse(rejected["complete"])
                    self.assertNotIn("PRIVATE-TUNING-PATH-SENTINEL", json.dumps(evidence))

    def test_public_combined_amd_setup_environment_preserves_layers_and_private_paths(self):
        controls = {"STRATA_HIPBLASLT_TUNING": "C:/PRIVATE-TUNING-PATH-SENTINEL/table.txt",
                    "STRATA_NO_ARENA_THP": "1", "STRATA_RESIDENT_PIN": "0"}
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                case = self.source()
                case.cfg["backend"] = "hip"
                if explicit:
                    case.cfg["env"] = dict(controls)
                case.config.write_text(json.dumps(case.cfg))
                inherited = {key: "INHERITED-PRIVATE-PATH-SENTINEL" for key in controls} if explicit else controls
                with patch.dict(bench.os.environ, inherited, clear=True):
                    result = case.invoke()
                effective = self.effective(case)
                for key, value in controls.items():
                    self.assertEqual(effective[key], value)
                evidence = result["normal_comparison"]
                self.assertEqual(result["status"], "complete")
                self.assertTrue(result["acceptance_workload_passed"])
                self.assertFalse(result["timing_diagnostic"])
                self.assertTrue(evidence["complete"])
                self.assertTrue(evidence["eligible"])
                self.assertEqual(evidence["exclusions"], [])
                selected = [item for item in evidence["controls"] if item["control"] in controls]
                self.assertEqual(selected, [{"control": key, "kind": "reviewed-normal"} for key in sorted(controls)])
                self.assertEqual(bench.consume_normal_comparison(result, case.clones[0], case.server_environments[0]), evidence)
                for path in case.args.output.iterdir():
                    text = path.read_text()
                    for secret in ("PRIVATE-TUNING-PATH-SENTINEL", "INHERITED-PRIVATE-PATH-SENTINEL", case.cfg["api_key"]):
                        self.assertNotIn(secret, text)

    def test_amd_setup_removal_and_null_follow_actual_nonmanaged_composition(self):
        from serve.server import config_environment
        controls = {"STRATA_HIPBLASLT_TUNING": "PRIVATE-REMOVED-TUNING-SENTINEL",
                    "STRATA_NO_ARENA_THP": "1", "STRATA_RESIDENT_PIN": "0"}
        for key in controls:
            for inherited in ({}, {key: controls[key]}):
                for configured, expected in (({}, inherited), ({key: None}, {key: "None"}), ({key: ""}, {key: ""})):
                    with self.subTest(control=key, inherited=bool(inherited), configured=configured):
                        # Setup controls are not managed memory-policy removals: None is the literal "None".
                        cfg = {"memory_policy": {"version": 1}, "env": configured}
                        self.assertEqual(config_environment(cfg, inherited), expected)
                        evidence = bench.normal_comparison_environment(cfg, inherited)
                        self.assertTrue(evidence["eligible"])
                        self.assertEqual(evidence["controls"], [{"control": name, "kind": "reviewed-normal"}
                                                               for name in sorted(expected)])
                        encoded = json.dumps(sorted([name, value] for name, value in expected.items()),
                                             separators=(",", ":")).encode()
                        self.assertEqual(evidence["environment_sha256"], hashlib.sha256(encoded).hexdigest())
                        consumed = bench.consume_normal_comparison({"normal_comparison": evidence}, cfg, inherited)
                        self.assertFalse(consumed["complete"])
                        self.assertFalse(consumed["eligible"])
                        self.assertIsNone(consumed["environment_sha256"])
                        self.assertNotIn("PRIVATE-REMOVED-TUNING-SENTINEL", json.dumps(evidence))

    def test_amd_setup_case_aliases_preserve_platform_and_conflict_rules(self):
        from serve.server import config_environment
        for key in ("STRATA_HIPBLASLT_TUNING", "STRATA_NO_ARENA_THP", "STRATA_RESIDENT_PIN"):
            for platform in ("nt", "posix"):
                with self.subTest(control=key, platform=platform), patch.object(bench.os, "name", platform):
                    cfg = {"env": {key.lower(): "0"}}
                    evidence = bench.normal_comparison_environment(cfg, {})
                    self.assertTrue(evidence["eligible"])
                    self.assertEqual(evidence["controls"], [{"control": key, "kind": "reviewed-normal"}]
                                     if platform == "nt" else [])
                    self.assertEqual(config_environment(cfg, {key: "1"}), {key: "1", key.lower(): "0"})
                    conflict = bench.normal_comparison_environment(cfg, {key: "1"})
                    self.assertEqual(conflict["eligible"], platform == "posix")
                    self.assertEqual(conflict["complete"], platform == "posix")
                    same = bench.normal_comparison_environment(cfg, {key: "0"})
                    self.assertTrue(same["eligible"])
                    self.assertEqual(same["controls"], [{"control": key, "kind": "reviewed-normal"}])
                    self.assertEqual(bench.consume_normal_comparison({"normal_comparison": same}, cfg, {key: "0"}), same)

    def test_amd_setup_controls_do_not_admit_unknown_or_diagnostic_treatments(self):
        controls = {"STRATA_HIPBLASLT_TUNING": "C:/PRIVATE-TUNING-PATH-SENTINEL/table.txt",
                    "STRATA_NO_ARENA_THP": "1", "STRATA_RESIDENT_PIN": "0"}
        unknown = "STRATA_PRIVATE_UNKNOWN_NAME_SENTINEL"
        for key, values, reason in ((unknown, ("", "0"), "unclassified-control"),
                                    ("STRATA_VERIFY_PROFILE", ("", "0"), "active-diagnostic")):
            for value in values:
                cfg = {"backend": "hip", "env": dict(controls, **{key: value})}
                evidence = bench.normal_comparison_environment(cfg, {})
                identifier = "sha256:" + hashlib.sha256(unknown.encode()).hexdigest() if key == unknown else key
                self.assertTrue(evidence["complete"])
                self.assertFalse(evidence["eligible"])
                self.assertEqual(evidence["exclusions"], [{"reason": reason, "control": identifier}])
                self.assertEqual(bench.consume_normal_comparison({"normal_comparison": evidence}, cfg, {}), evidence)
                serialized = json.dumps(evidence)
                self.assertNotIn(unknown, serialized)
                self.assertNotIn("PRIVATE-TUNING-PATH-SENTINEL", serialized)

    def test_r5_exact_public_specimens_follow_actual_cleaning_and_source_gates(self):
        controls = {"STRATA_PIPELINE_DEBUG": "1", "STRATA_PIPELINE_FORCE_MISS": "1", "STRATA_LAYER_SPLIT_GATE": "0"}
        for explicit in (False, True):
            case = self.source()
            if explicit:
                case.cfg["env"] = dict(controls)
                case.config.write_text(json.dumps(case.cfg))
            with patch.dict(bench.os.environ, controls, clear=True):
                result = case.invoke()
            effective = self.effective(case)
            self.assertEqual(effective["STRATA_PIPELINE_DEBUG"], "1")
            self.assertEqual(result["normal_comparison"]["eligible"], not explicit)
            self.assertEqual(result["timing_diagnostic"], explicit)
            self.assertEqual(result["status"], "complete")
            for key in ("STRATA_PIPELINE_FORCE_MISS", "STRATA_LAYER_SPLIT_GATE"):
                if explicit:
                    self.assertEqual(effective[key], controls[key])
                else:
                    self.assertNotIn(key, effective)
        for gate in ("", "0", "1"):
            for value in ("", "0", "-1", "1", " 2suffix", "-4294967295"):
                cfg = {"env": {"STRATA_PIPELINE_DEBUG": gate, "STRATA_PIPELINE_FORCE_MISS": value}}
                evidence = bench.normal_comparison_environment(cfg, {})
                self.assertEqual(evidence["eligible"], not (gate == "1" and value in ("1", " 2suffix", "-4294967295")))
        self.assertFalse(bench.normal_comparison_environment(
            {"env": {"STRATA_TEST_CACHE_FAIL": "-4294967295"}}, {})["eligible"])
        for value in ("", "0"):
            evidence = bench.normal_comparison_environment({"env": {"STRATA_LAYER_SPLIT_GATE": value}}, {})
            self.assertFalse(evidence["eligible"])
            self.assertEqual(evidence["exclusions"], [{"reason": "active-diagnostic", "control": "STRATA_LAYER_SPLIT_GATE"}])

    def test_public_removals_classify_effective_state_without_stripping_unknown_nulls(self):
        for unknown in (False, True):
            case = self.source(automatic=True)
            key = "STRATA_RESIDENT_HEADROOM_GIB"
            secret = "REMOVED-PRIVATE-CONTROL-SENTINEL"
            case.cfg["memory_policy"] = {"version": 1, "generated_env": {key: "8"}, "overrides": {"env": {key: None}}}
            case.cfg["env"] = {key: None}
            if unknown:
                case.cfg["env"]["STRATA_NOT_A_MANAGED_REMOVAL"] = None
            case.config.write_text(json.dumps(case.cfg))
            with patch.dict(bench.os.environ, {key: secret}, clear=True):
                result = case.invoke()
            effective = self.effective(case)
            self.assertNotIn(key, effective)
            classified = bench.normal_comparison_environment(case.clones[0], case.server_environments[0])
            self.assertEqual(classified["eligible"], not unknown)
            self.assertFalse(result["normal_comparison"]["eligible"])
            self.assertFalse(result["normal_comparison"]["complete"])
            self.assertIsNone(result["normal_comparison"]["environment_sha256"])
            if unknown:
                self.assertEqual(effective["STRATA_NOT_A_MANAGED_REMOVAL"], "None")
            self.assertNotIn(secret, json.dumps(result))
            self.assertTrue(result["acceptance_workload_passed"])

    def test_environment_case_and_typed_conflicts_match_actual_server_composition(self):
        from serve.server import config_environment
        for platform in ("nt", "posix"):
            with patch.object(bench.os, "name", platform):
                cfg = {"env": {"strata_fresh_unreviewed": "0"}}
                self.assertEqual(config_environment(cfg, {}), {"strata_fresh_unreviewed": "0"})
                self.assertEqual(bench.normal_comparison_environment(cfg, {})["eligible"], platform == "posix")
                cfg = {"memory_policy": {"version": 1}, "env": {"strata_file_cache_mib": "128"}}
                inherited = {"STRATA_FILE_CACHE_MIB": "64"}
                expected = ({"STRATA_FILE_CACHE_MIB": "128"} if platform == "nt" else
                            {"STRATA_FILE_CACHE_MIB": "64", "strata_file_cache_mib": "128"})
                self.assertEqual(config_environment(cfg, inherited), expected)
                self.assertTrue(bench.normal_comparison_environment(cfg, inherited)["eligible"])
                conflict = {"memory_policy": {"version": 1},
                            "env": {"STRATA_FILE_CACHE_MIB": 1, "strata_file_cache_mib": True}}
                if platform == "nt":
                    for operation in (config_environment, bench.normal_comparison_environment):
                        with self.assertRaisesRegex(ValueError, "Conflicting managed environment aliases"):
                            operation(conflict, {})
                else:
                    self.assertEqual(config_environment(conflict, {}), {"STRATA_FILE_CACHE_MIB": "1", "strata_file_cache_mib": "True"})
        with patch.object(bench.os, "name", "nt"):
            evidence = bench.normal_comparison_environment({"env": {"strata_file_copy_ready": "1"}},
                                                          {"STRATA_FILE_COPY_READY": "0"})
            self.assertFalse(evidence["eligible"])
            self.assertFalse(evidence["complete"])

    def test_invalid_or_oversized_effective_inventories_refuse_without_false_completeness(self):
        specimens = [
            {"STRATA_SECRET=PRIVATE-NAME-SENTINEL": "PRIVATE-VALUE-SENTINEL"},
            {"STRATA_\nPRIVATE-NAME-SENTINEL": "PRIVATE-VALUE-SENTINEL"},
            {"STRATA_" + "X" * 129: "PRIVATE-VALUE-SENTINEL"},
            {"STRATA_FUTURE": "PRIVATE-VALUE-SENTINEL\0"},
            {"STRATA_FUTURE": "X" * 4097},
            {"STRATA_FUTURE_" + str(index): "0" for index in range(129)},
            {"STRATA_FUTURE_" + str(index): "X" * 4096 for index in range(17)},
            {"OTHER_" + str(index): "0" for index in range(4097)},
        ]
        for environment in specimens:
            result = bench.normal_comparison_environment({"env": environment}, {})
            self.assertFalse(result["complete"])
            self.assertFalse(result["eligible"])
            self.assertIsNone(result["environment_sha256"])
            self.assertEqual(result["controls"], [])
            self.assertLess(len(json.dumps(result)), 512)
            self.assertNotIn("PRIVATE", json.dumps(result))

    def test_ordinary_windows_names_outside_scope_do_not_refuse_classification(self):
        inherited = {"ProgramFiles(x86)": "C:\\Program Files (x86)", "CommonProgramFiles(x86)": "C:\\Common",
                     "ASL.LOG": "Destination=file", "=C:": "C:\\Users", "OTHER": "PRIVATE\0VALUE"}
        for platform in ("nt", "posix"):
            with patch.object(bench.os, "name", platform):
                for extra, eligible in (({}, True), ({"STRATA_FUTURE_74B9": "0"}, False)):
                    result = bench.normal_comparison_environment(
                        {"env": {"STRATA_FILE_CACHE_MIB": "4096"}}, dict(inherited, **extra))
                    self.assertTrue(result["complete"])
                    self.assertEqual(result["eligible"], eligible)
                    self.assertNotIn("PRIVATE", json.dumps(result))
                    self.assertNotIn("Program", json.dumps(result))
        case = self.source()
        with patch.dict(bench.os.environ, {"ProgramFiles(x86)": "C:\\Program Files (x86)"}, clear=True):
            result = case.invoke()
        self.assertTrue(result["normal_comparison"]["complete"])
        self.assertTrue(result["normal_comparison"]["eligible"])

    def test_public_invalid_unknown_name_keeps_workload_outcome_and_redacts_inventory(self):
        case = self.source()
        case.cfg["env"] = {"STRATA_BAD=PRIVATE-NAME-SENTINEL": "PRIVATE-VALUE-SENTINEL"}
        case.config.write_text(json.dumps(case.cfg))
        with patch.dict(bench.os.environ, {}, clear=True):
            result = case.invoke()
        self.assertEqual(self.effective(case)["STRATA_BAD=PRIVATE-NAME-SENTINEL"], "PRIVATE-VALUE-SENTINEL")
        self.assertTrue(result["acceptance_workload_passed"])
        self.assertFalse(result["normal_comparison"]["eligible"])
        self.assertFalse(result["normal_comparison"]["complete"])
        for path in case.args.output.iterdir():
            self.assertNotIn("PRIVATE-NAME-SENTINEL", path.read_text())
            self.assertNotIn("PRIVATE-VALUE-SENTINEL", path.read_text())

    def test_byte_binding_is_unambiguous_but_does_not_review_an_unknown_control(self):
        # These collide under concatenated name/value encoding, but not JSON pairs.
        first = bench.normal_comparison_environment({"env": {"STRATA_A": "bSTRATA_Bc", "STRATA_B": "d"}}, {})
        second = bench.normal_comparison_environment({"env": {"STRATA_A": "b", "STRATA_B": "cSTRATA_Bd"}}, {})
        self.assertEqual(first["environment_sha256"], hashlib.sha256(
            b'[["STRATA_A","bSTRATA_Bc"],["STRATA_B","d"]]').hexdigest())
        self.assertNotEqual(first["environment_sha256"], second["environment_sha256"])
        self.assertFalse(first["eligible"])
        self.assertFalse(second["eligible"])
        reordered = bench.normal_comparison_environment({"env": {"STRATA_B": "d", "STRATA_A": "bSTRATA_Bc"}}, {})
        self.assertEqual(first, reordered)

    def test_legacy_public_flow_retains_counts_and_outputs_but_excludes_unknown_controls(self):
        for unknown in (False, True):
            case = RunTests()
            case.setUp()
            try:
                cfg = json.loads(case.config.read_text())
                if unknown:
                    cfg["env"]["STRATA_LEGACY_UNREVIEWED"] = "LEGACY-PRIVATE-SENTINEL"
                    case.config.write_text(json.dumps(cfg))
                    case.original = case.config.read_bytes()
                expected = {key: value for key, value in cfg["env"].items() if key != "STRATA_VERIFY_PROFILE"}
                with patch.dict(bench.os.environ, {}, clear=True):
                    result = case.invoke(expected_env=expected)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(len(result["rows"]), 6)
                self.assertEqual(len(result["requests"]), 7)
                self.assertEqual(result["summary"]["completion_tokens"], 960)
                self.assertEqual(result["summary"]["predicted_ms"], 9000)
                self.assertEqual(result["normal_comparison"]["eligible"], not unknown)
                self.assertTrue(result["normal_comparison"]["complete"])
                case.assert_safe_artifacts()
                for path in case.output.iterdir():
                    self.assertNotIn("LEGACY-PRIVATE-SENTINEL", path.read_text())
            finally:
                case.tearDown()

    def test_consumer_excludes_legacy_missing_partial_and_malformed_classifications(self):
        good = bench.normal_comparison_environment({"env": {}}, {})
        specimens = [None, [], {}, {"timing_diagnostic": False}, {"normal_comparison": True},
                     {"normal_comparison": {}}, {"normal_comparison": dict(good, eligible=1)},
                     {"normal_comparison": dict(good, complete=False)},
                     {"normal_comparison": dict(good, controls=[{}] * 129)},
                     {"normal_comparison": dict(good, environment_sha256="PRIVATE-VALUE-SENTINEL")},
                     {"normal_comparison": dict(good, exclusions=[{"reason": "PRIVATE-VALUE-SENTINEL"}])}]
        for record in specimens:
            result = bench.consume_normal_comparison(record, {"env": {}}, {})
            self.assertFalse(result["eligible"])
            self.assertFalse(result["complete"])
            self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertTrue(bench.consume_normal_comparison({"normal_comparison": good}, {"env": {}}, {})["eligible"])

    def test_actual_selected_consumer_never_promotes_missing_or_mutated_worker_classification(self):
        for mutation in ("missing", "eligible", "complete", "controls", "digest", "unsafe-reason", "all-green"):
            case = ContainedLifecycleTests()
            case.setUp()
            try:
                def mutate(record):
                    evidence = record["normal_comparison"]
                    if mutation == "missing":
                        record.pop("normal_comparison")
                    elif mutation == "eligible":
                        evidence["eligible"] = True
                    elif mutation == "complete":
                        evidence["complete"] = 1
                    elif mutation == "controls":
                        evidence["controls"] = []
                    elif mutation == "digest":
                        evidence["environment_sha256"] = "0" * 64
                    elif mutation == "unsafe-reason":
                        evidence["exclusions"] = [{"reason": "PRIVATE-WORKER-SENTINEL"}]
                    else:
                        record["normal_comparison"] = bench.normal_comparison_environment(
                            {"env": {"CUDA_MODULE_LOADING": "EAGER", "STRATA_FILE_COPY_READY": "0"}}, {})
                with self.subTest(mutation=mutation), \
                        patch.dict(bench.os.environ, {"STRATA_UNSEEN_CONSUMER_TREATMENT": "PRIVATE-UNKNOWN-SENTINEL"}, clear=True):
                    result = case.invoke_selected(mutate)
                self.assertTrue(bench.acceptance_final_predicate(result))
                self.assertTrue(result["worker_complete"])
                self.assertFalse(result["normal_comparison"]["eligible"])
                self.assertFalse(result["normal_comparison"]["complete"])
                self.assertNotIn("PRIVATE", json.dumps(result))
            finally:
                case.doCleanups()

    def test_selected_consumer_retains_successful_unknown_and_normal_environment_outcomes(self):
        for mode in ("normal", "inherited-unknown", "explicit-unknown"):
            case = ContainedLifecycleTests()
            case.setUp()
            try:
                unknown = mode != "normal"
                environment = {"STRATA_VERIFY_QDEDUP": "1"}
                if mode == "inherited-unknown":
                    environment["STRATA_UNSEEN_LATER_CONTROL"] = "0"
                elif mode == "explicit-unknown":
                    cfg = json.loads(case.config.read_text())
                    cfg["env"] = {"STRATA_UNSEEN_LATER_CONTROL": ""}
                    case.config.write_text(json.dumps(cfg))
                with patch.dict(bench.os.environ, environment, clear=True):
                    result = case.invoke_selected()
                self.assertTrue(bench.acceptance_final_predicate(result))
                self.assertTrue(result["worker_complete"])
                self.assertTrue(result["normal_comparison"]["complete"])
                self.assertEqual(result["normal_comparison"]["eligible"], not unknown)
                self.assertEqual(json.loads((case.output / "lifecycle.json").read_text())["status"], "held")
            finally:
                case.doCleanups()
        self.assertFalse(bench.acceptance_ready())
        self.assertFalse(bench.acceptance_protection(None))
        with self.assertRaisesRegex(RuntimeError, "not implemented"):
            bench.acceptance_preflight(None)


class DesktopComparisonConsumerTests(unittest.TestCase):
    """CPU-only public desktop report/caller controls, with fatal external seams."""

    def setUp(self):
        from tools import prepare_unsloth_q2 as preparation
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        def forbidden(*args, **kwargs):
            raise AssertionError("real external seam forbidden in desktop consumer tests")
        winjob = types.ModuleType("serve.winjob")
        winjob.contain = winjob.no_throttle = forbidden
        self.stack.enter_context(patch.dict(sys.modules, {
            "serve.winjob": winjob, "benchmark_compact_cache": bench,
            "prepare_unsloth_q2": preparation}))
        for target, name in ((bench.subprocess, "Popen"), (bench.subprocess, "run"),
                             (bench.socket, "socket"), (bench.urllib.request, "urlopen"),
                             (bench.psutil, "process_iter"), (bench.psutil, "Process"),
                             (bench.psutil, "virtual_memory"),
                             (bench.psutil, "swap_memory"), (ctypes, "WinDLL"),
                             (bench, "assert_idle"), (preparation, "check_ready"),
                             (preparation, "memory_snapshot"), (preparation.setup, "gpu_info")):
            self.stack.enter_context(patch.object(target, name, side_effect=forbidden, create=True))
        spec = bench.importlib.util.spec_from_file_location(
            "desktop_consumer_test", Path(__file__).with_name("benchmark_desktop_profiles.py"))
        self.desktop = bench.importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.desktop)

    def specimen(self, env=None, *, automatic=False, answer="clean", speed=23.5):
        cfg = {"args": ["--max-context", "131072"], "env": dict(env or {}),
               "api_key": "PRIVATE-CREDENTIAL", "model_name": "PRIVATE-MODEL"}
        if automatic:
            cfg["memory_policy"] = {"version": 1}
        context = bench.clean_environment(cfg, False, inherited={"PRIVATE-ORDINARY": "PRIVATE-VALUE"})
        result = {"status": "complete", "timing_diagnostic": False,
                  "rows": [{"response": {"choices": [{"message": {"role": "assistant", "content": answer},
                                                       "finish_reason": "stop"}]}}],
                  "summary": {"tokens_per_second": speed, "completion_tokens": 160},
                  "normal_comparison": bench.normal_comparison_launch_environment(*context, server_log=True)}
        return result, context

    def test_legacy_public_report_without_context_excludes_old_performance_row(self):
        result, _ = self.specimen(speed=987.65)
        result.pop("normal_comparison")
        text = self.desktop.report([("legacy", result)])
        self.assertNotIn("987.65", text)
        self.assertNotIn("1/1", text)
        self.assertIn("independent launch context unavailable", text)
        self.assertIn("No eligible complete runs", text)
        # Even a valid modern record cannot supply its own independent inputs.
        modern, _ = self.specimen()
        self.assertNotIn("23.50", self.desktop.report([("modern", modern)]))

    def test_mixed_public_report_uses_first_eligible_baseline_and_preserves_inputs(self):
        excluded, bad_context = self.specimen({"STRATA_PRIVATE_UNKNOWN": ""}, answer="excluded", speed=987.65)
        first, context = self.specimen()
        last, last_context = self.specimen(answer="different", speed=31)
        records = [("excluded", excluded), ("first-clean", first), ("last-clean", last)]
        contexts = {"excluded": bad_context, "first-clean": context, "last-clean": last_context}
        before = bench.copy.deepcopy((records, contexts))
        text = self.desktop.report(records, contexts=contexts)
        self.assertIn("| first-clean | 23.50 | 160 | 1/1 |", text)
        self.assertIn("| last-clean | 31.00 | 160 | 0/1 |", text)
        self.assertNotIn("987.65", text)
        for private in ("STRATA_PRIVATE_UNKNOWN", "PRIVATE-CREDENTIAL", "PRIVATE-MODEL", "PRIVATE-VALUE", "PRIVATE-ORDINARY"):
            self.assertNotIn(private, text)
        self.assertEqual((records, contexts), before)
        self.assertIn("build, model, workload, allocation, cohort or performance", text)
        self.assertIn("unsupported for new normal-comparison conclusions", text)

    def test_unknown_diagnostic_automatic_and_forged_classifications_cannot_rank(self):
        for env, automatic in (({"STRATA_PRIVATE_UNKNOWN": ""}, False),
                               ({"STRATA_PRIVATE_UNKNOWN": "0"}, False),
                               ({"STRATA_DEBUG": "1"}, False),
                               ({"STRATA_REQUEST_LINES": "1"}, False), ({}, True)):
            with self.subTest(env=env, automatic=automatic):
                result, context = self.specimen(env, automatic=automatic, speed=987.65)
                self.assertNotIn("987.65", self.desktop.report([("bad", result)], contexts={"bad": context}))
                # A forged true flag cannot override actual source classification/binding.
                result["normal_comparison"]["eligible"] = True
                result["normal_comparison"]["complete"] = True
                self.assertNotIn("987.65", self.desktop.report([("bad", result)], contexts={"bad": context}))
        result, context = self.specimen(speed=987.65)
        for field, value in (("eligible", 1), ("complete", 1), ("schema", True),
                             ("environment_sha256", "forged"), ("controls", [{}] * 129)):
            with self.subTest(field=field):
                changed = bench.copy.deepcopy(result)
                changed["normal_comparison"][field] = value
                self.assertNotIn("987.65", self.desktop.report([("bad", changed)], contexts={"bad": context}))

    def test_missing_malformed_and_mismatched_independent_contexts_fail_closed(self):
        result, context = self.specimen(speed=987.65)
        for contexts in (None, [], {}, {"bad": None}, {"bad": []}, {"bad": ({},)},
                         {"bad": ({"env": []}, {})}, {"bad": ({"env": {}}, [])},
                         {"bad": ({"env": {"STRATA_PRIVATE_UNKNOWN": "0"}}, {})}):
            with self.subTest(contexts=contexts):
                text = self.desktop.report([("bad", result)], contexts=contexts)
                self.assertNotIn("987.65", text)
                self.assertIn("No eligible complete runs", text)
        result.pop("normal_comparison")
        self.assertNotIn("987.65", self.desktop.report([("bad", result)], contexts={"bad": context}))

    def test_incomplete_failed_diagnostic_and_malformed_results_cannot_set_baseline(self):
        clean, context = self.specimen()
        changes = [{"status": value} for value in (None, "failed", "starting", "provisional_complete")]
        changes += [{"timing_diagnostic": value} for value in (None, True, 0)]
        changes += [{key: value} for key, value in (("rows", []), ("rows", [{}]), ("summary", {}),
                    ("smoke_passed", False), ("decode_timing", True), ("fetch_timing", True),
                    ("validation_only", True), ("smoke_only", True), ("error", "PRIVATE-FAILURE"),
                    ("cleanup_error", "PRIVATE-FAILURE"), ("artifact_error", "PRIVATE-FAILURE"))]
        for change in changes:
            with self.subTest(change=change):
                bad, _ = self.specimen(answer="excluded", speed=987.65)
                bad.update(change)
                text = self.desktop.report([("bad", bad), ("clean", clean)],
                                           contexts={"bad": context, "clean": context})
                self.assertNotIn("987.65", text)
                self.assertNotIn("PRIVATE-FAILURE", text)
                self.assertIn("| clean | 23.50 | 160 | 1/1 |", text)

    def test_all_excluded_empty_and_nonfinite_results_are_valid_reports(self):
        result, context = self.specimen()
        for speed in (0, -1, True, float("nan"), float("inf"), "987.65"):
            with self.subTest(speed=speed):
                changed = bench.copy.deepcopy(result)
                changed["summary"]["tokens_per_second"] = speed
                text = self.desktop.report([("bad", changed)], contexts={"bad": context})
                self.assertIn("No eligible complete runs", text)
                self.assertNotIn("1/1", text)
        self.assertIn("No eligible complete runs", self.desktop.report([]))
        for bad in (None, [], {}, {"normal_comparison": True}):
            self.assertIn("No eligible complete runs", self.desktop.report([("bad", bad)], contexts={"bad": context}))

    def test_malformed_first_response_cannot_rank_or_choose_baseline(self):
        bad, context = self.specimen(speed=987.65)
        bad["rows"] = [{"response": {"choices": [None]}}]
        clean, clean_context = self.specimen(answer="valid")
        records = [("malformed", bad), ("clean", clean)]
        contexts = {"malformed": context, "clean": clean_context}
        before = bench.copy.deepcopy((records, contexts))
        text = self.desktop.report(records, contexts=contexts)
        self.assertIn("| malformed | complete assistant responses unavailable; inspect raw results |", text)
        self.assertNotIn("987.65", text)
        self.assertIn("| clean | 23.50 | 160 | 1/1 |", text)
        self.assertEqual((records, contexts), before)
        for private in ("PRIVATE-CREDENTIAL", "PRIVATE-MODEL", "PRIVATE-ORDINARY", "PRIVATE-VALUE"):
            self.assertNotIn(private, text)

    def malformed_choices(self):
        return [None, True, 1, "PRIVATE-ANSWER", [], {},
                {"message": None}, {"message": []}, {"message": "PRIVATE-ANSWER"},
                {"message": {}}, {"message": {"role": "assistant"}, "finish_reason": "stop"},
                *({"message": {"role": "assistant", "content": value}, "finish_reason": "stop"}
                  for value in (None, 0, True, [], {"PRIVATE-KEY": "PRIVATE-ANSWER"})),
                *({"message": {"role": value, "content": "PRIVATE-ANSWER"}, "finish_reason": "stop"}
                  for value in (None, "user", "", 0, [])),
                {"message": {"content": "PRIVATE-ANSWER"}, "finish_reason": "stop"},
                {"message": {"role": "assistant", "content": "PRIVATE-ANSWER"}},
                *({"message": {"role": "assistant", "content": "PRIVATE-ANSWER"}, "finish_reason": value}
                  for value in (None, "", "tool_calls", "error", 0, True, [], {}))]

    def test_malformed_choices_in_any_element_or_later_row_exclude_whole_record(self):
        clean, context = self.specimen()
        valid = clean["rows"][0]["response"]["choices"][0]
        for invalid in self.malformed_choices():
            for choices in ([invalid], [valid, invalid], [invalid, valid]):
                for later in (False, True):
                    with self.subTest(invalid=invalid, choices=len(choices), later=later):
                        bad, _ = self.specimen(speed=987.65)
                        bad["rows"] = ([bench.copy.deepcopy(clean["rows"][0])] if later else [])
                        bad["rows"].append({"response": {"choices": bench.copy.deepcopy(choices)}})
                        before = bench.copy.deepcopy(bad)
                        text = self.desktop.report([("bad", bad), ("clean", clean)],
                                                   contexts={"bad": context, "clean": context})
                        self.assertNotIn("987.65", text)
                        self.assertIn("| clean | 23.50 | 160 | 1/1 |", text)
                        self.assertIn("| bad | complete assistant responses unavailable; inspect raw results |", text)
                        self.assertEqual(bad, before)
                        self.assertNotIn("PRIVATE-ANSWER", text)
                        self.assertNotIn("PRIVATE-KEY", text)

    def test_complete_assistant_text_stop_length_and_multiple_valid_choices_compare(self):
        for content in ("", "ordinary answer", "line one\nline two", "答え"):
            for finish in ("stop", "length"):
                for multiple in (False, True):
                    with self.subTest(content=content, finish=finish, multiple=multiple):
                        result, context = self.specimen(answer=content)
                        result["rows"][0]["response"]["choices"][0]["finish_reason"] = finish
                        if multiple:
                            result["rows"][0]["response"]["choices"].append({
                                "message": {"role": "assistant", "content": "second answer"}, "finish_reason": "stop"})
                        before = bench.copy.deepcopy((result, context))
                        text = self.desktop.report([("valid", result)], contexts={"valid": context})
                        self.assertIn("| valid | 23.50 | 160 | 1/1 |", text)
                        self.assertNotIn("No eligible complete runs", text)
                        self.assertEqual((result, context), before)

    def test_actual_ordinary_runner_fixture_still_produces_comparable_responses(self):
        case = RunTests()
        case.setUp()
        self.addCleanup(case.tearDown)
        source = json.loads(case.config.read_text())
        context = bench.clean_environment(source, False, inherited={})
        base_client = case.fake_client()
        # Strengthen only this fixture's wire response with fields emitted by the
        # ordinary API and required by existing assistant-response validation.
        class CompleteClient(base_client):
            def chat(self, *args, **kwargs):
                response = super().chat(*args, **kwargs)
                response["choices"][0]["message"]["role"] = "assistant"
                response["choices"][0]["finish_reason"] = "stop"
                return response
        with patch.object(case, "fake_client", return_value=CompleteClient), \
                patch.dict(bench.os.environ, {}, clear=True):
            result = case.invoke()
        before = bench.copy.deepcopy(result)
        text = self.desktop.report([("ordinary", result)], contexts={"ordinary": context})
        self.assertEqual(len(result["rows"]), 6)
        self.assertIn("| ordinary | 106.67 | 960 | 6/6 | not requested |", text)
        self.assertEqual(result, before)
        self.assertNotIn(case.secret, text)
        case.owner.stop.assert_called_once_with()
        case.assert_safe_artifacts()

    def test_main_malformed_responses_share_table_and_console_exclusion(self):
        for later in (False, True):
            for invalid in (None, {"message": {}}, {"message": {"role": "assistant", "content": None}}):
                with self.subTest(later=later, invalid=invalid):
                    def corrupt(index, record):
                        if index == 0:
                            record["summary"]["tokens_per_second"] = 987.65
                            if later:
                                record["rows"].append(bench.copy.deepcopy(record["rows"][0]))
                            if not later and invalid is None:
                                record["rows"][-1]["response"]["choices"] = [None]
                            else:
                                record["rows"][-1]["response"]["choices"].append(invalid)
                    code, stdout, stderr = self.invoke_main(mutate=corrupt)
                    self.assertEqual((code, stderr), (0, ""))
                    self.assertNotIn("987.65", stdout + self.writes[-1][1])
                    self.assertIn("compact-before: excluded from normal comparison (complete assistant responses unavailable)", stdout)
                    self.assertIn("| desktop | 23.50 | 160 | 1/1 | passed |", self.writes[-1][1])
                    self.assertEqual(stdout.count("tokens/s (environment-eligible candidate)"), 3)
                    self.assertIs(self.returned[0]["rows"][-1]["response"]["choices"][-1], invalid)

    def invoke_main(self, *, plan=False, excluded=(), fail_at=None, saving=None, env=None, automatic=False, mutate=None):
        """All files/workloads/readiness here are returning in-memory collaborators."""
        root = Path(tempfile.gettempdir()).resolve() / "desktop-consumer-memory-only"
        self.configs = {root / filename: {"args": ["--max-context", "131072"],
            "api_key": "PRIVATE-CREDENTIAL", "env": {"STRATA_FILE_CACHE_MIB": "4096", "STRATA_VERIFY_PROFILE": "1"}}
            for _, filename, _ in self.desktop.CASES}
        if saving:
            self.configs[root / self.desktop.CASES[0][1]].update(saving)
        for cfg in self.configs.values():
            cfg["env"].update(env or {})
            if automatic:
                cfg["memory_policy"] = {"version": 1}
        self.events, self.writes, self.returned, self.launch_contexts = [], [], [], []
        self.consumptions = []
        def read(path, *args, **kwargs):
            self.events.append(("read", path.name))
            return json.dumps(self.configs[path])
        def ready(cfg):
            self.events.append(("ready", bench.copy.deepcopy(cfg)))
        def idle():
            self.events.append(("idle",))
        def mkdir(path, *args, **kwargs):
            self.events.append(("mkdir", path, kwargs))
        def write(path, text, **kwargs):
            self.writes.append((path, text))
            return len(text)
        def workload(args):
            index = len(self.returned)
            label, filename, smoke = self.desktop.CASES[index]
            self.assertEqual(vars(args), {"config": root / filename, "output": args.output,
                                         "timing": False, "smoke": smoke, "smoke_only": False})
            self.assertEqual(args.output.name, label)
            self.events.append(("run", label))
            if fail_at == index:
                raise RuntimeError("PRIVATE-CREDENTIAL")
            cfg = json.loads(args.config.read_text())
            context = bench.clean_environment(cfg, False, inherited=dict(bench.os.environ))
            self.launch_contexts.append(bench.copy.deepcopy(context))
            result, _ = self.specimen(answer="excluded" if index in excluded else "clean", speed=987.65 if index in excluded else 23.5)
            result["normal_comparison"] = bench.normal_comparison_launch_environment(*context, server_log=True)
            if smoke:
                result["smoke_passed"] = True
            if index in excluded:
                result.pop("normal_comparison")
            if mutate:
                mutate(index, result)
            # Returned metadata and subsequent process environment are not independent inputs.
            result["effective_settings"] = {"env": {"STRATA_PRIVATE_UNKNOWN": "PRIVATE-POISON"}}
            self.returned.append(result)
            bench.os.environ["STRATA_RESIDENT_PIN"] = str(index % 2)
            return result
        real_consume = bench.consume_normal_comparison
        def consume(record, cfg, inherited, **kwargs):
            index = next(index for index, value in enumerate(self.returned) if value is record)
            self.consumptions.append((index, bench.copy.deepcopy((cfg, inherited)), kwargs))
            return real_consume(record, cfg, inherited, **kwargs)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(self.desktop, "ROOT", root), \
                patch.object(Path, "read_text", read), patch.object(Path, "mkdir", mkdir), \
                patch.object(Path, "write_text", write), \
                patch.object(self.desktop.preparation, "check_ready", side_effect=ready), \
                patch.object(bench, "assert_idle", side_effect=idle), \
                patch.object(bench, "run", side_effect=workload), \
                patch.object(bench, "consume_normal_comparison", side_effect=consume), \
                patch.dict(bench.os.environ, {"STRATA_API_KEY": "PRIVATE-INHERITED-KEY", "STRATA_VERIFY_PROFILE": "1"}, clear=True), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = self.desktop.main(["--plan"] if plan else [])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_main_retains_independent_clean_context_and_original_run_order_settings(self):
        code, stdout, stderr = self.invoke_main(excluded=(0,))
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual([event[1] for event in self.events if event[0] == "run"],
                         ["compact-before", "desktop", "desktop-gpu", "compact-after"])
        self.assertEqual([event[0] for event in self.events[:8]], ["read", "ready"] * 4)
        self.assertEqual([event[0] for event in self.events[8:10]], ["idle", "mkdir"])
        self.assertEqual(len(self.writes), 4)
        for index, context, kwargs in self.consumptions:
            self.assertEqual(context, self.launch_contexts[index])
            self.assertEqual(kwargs, {"server_log": True})
            self.assertNotIn("STRATA_API_KEY", context[1])
            self.assertNotIn("STRATA_VERIFY_PROFILE", context[0]["env"])
            self.assertNotIn("STRATA_VERIFY_PROFILE", context[1])
        self.assertEqual({index for index, _, _ in self.consumptions}, {0, 1, 2, 3})
        self.assertIn("compact-before: excluded from normal comparison", stdout)
        self.assertEqual(stdout.count("23.50 tokens/s (environment-eligible candidate)"), 3)
        self.assertNotIn("987.65", stdout + self.writes[-1][1])
        self.assertIn("| desktop | 23.50 | 160 | 1/1 | passed |", self.writes[-1][1])
        self.assertNotIn("| compact-before | 987.65", self.writes[-1][1])
        for private in ("PRIVATE-CREDENTIAL", "PRIVATE-INHERITED-KEY", "PRIVATE-POISON", "STRATA_PRIVATE_UNKNOWN"):
            self.assertNotIn(private, stdout + stderr + "".join(text for _, text in self.writes))
        self.assertNotIn("normal_comparison", self.returned[0])

    def test_main_all_excluded_never_advertises_throughput(self):
        code, stdout, stderr = self.invoke_main(excluded=(0, 1, 2, 3))
        self.assertEqual((code, stderr), (0, ""))
        self.assertNotIn("tokens/s", stdout)
        self.assertEqual(stdout.count("excluded from normal comparison"), 4)
        self.assertIn("No eligible complete runs", self.writes[-1][1])
        self.assertNotIn("987.65", self.writes[-1][1])

    def test_main_actual_unknown_diagnostic_and_automatic_classification_excludes_console_and_table(self):
        for env, automatic in (({"STRATA_PRIVATE_UNKNOWN": ""}, False),
                               ({"STRATA_PRIVATE_UNKNOWN": "0"}, False),
                               ({"STRATA_DEBUG": "1"}, False),
                               ({"STRATA_REQUEST_LINES": "1"}, False), ({}, True)):
            with self.subTest(env=env, automatic=automatic):
                code, stdout, stderr = self.invoke_main(env=env, automatic=automatic)
                self.assertEqual((code, stderr), (0, ""))
                self.assertNotIn("tokens/s", stdout)
                self.assertIn("No eligible complete runs", self.writes[-1][1])
                self.assertNotIn("23.50", self.writes[-1][1])
                self.assertEqual(stdout.count("excluded from normal comparison"), 4)

    def test_main_plan_starts_nothing_and_preserves_learned_profile_refusal(self):
        code, stdout, stderr = self.invoke_main(plan=True)
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Readiness passed. No model was started.", stdout)
        self.assertEqual([event[0] for event in self.events], ["read", "ready"] * 4)
        self.assertEqual(self.writes, [])
        self.assertEqual(self.returned, [])
        for saving in ({"expert_profile_save": "private.bin"}, {"args": ["--expert-profile-save", "private.bin"]}):
            code, stdout, stderr = self.invoke_main(plan=True, saving=saving)
            self.assertEqual(code, 1)
            self.assertIn("Comparison stopped (ValueError)", stderr)
            self.assertEqual([event[0] for event in self.events], ["read", "ready"])
            self.assertEqual(self.writes, [])

    def test_main_failure_stops_sequence_and_retains_prior_report_without_secret(self):
        code, stdout, stderr = self.invoke_main(fail_at=1)
        self.assertEqual(code, 1)
        self.assertIn("Comparison stopped (RuntimeError)", stderr)
        self.assertNotIn("PRIVATE-CREDENTIAL", stderr)
        self.assertEqual([event[1] for event in self.events if event[0] == "run"], ["compact-before", "desktop"])
        self.assertEqual(len(self.writes), 1)
        self.assertIn("| compact-before | 23.50 | 160 | 1/1 |", self.writes[0][1])


if __name__ == "__main__":
    unittest.main()
