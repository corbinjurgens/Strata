"""Offline safety/measurement checks; never loads a model or contacts a server."""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import hmac
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import benchmark_compact_cache as bench

# Every selected acceptance clone carries STRATA_TRACE=1 for its CUDA-stage rows.
TRACE_CONTROL = {"control": "STRATA_TRACE", "kind": "diagnostic", "active": True}
TRACE_EXCLUSION = {"reason": "active-diagnostic", "control": "STRATA_TRACE"}


def other_active_diagnostics(evidence):
    """Active diagnostic controls of a classification besides the acceptance trace."""
    return [item["control"] for item in evidence["controls"]
            if item.get("active") is True and item != TRACE_CONTROL]


class ControllerTimingTests(unittest.TestCase):
    def test_wall_intervals_sleep_request_and_exception_without_error_text(self):
        timing = bench.ControllerTiming()
        with patch.object(bench.time, "monotonic", side_effect=[10.0, 10.85, 11.0, 12.1, 12.2]):
            with timing.phase("sleep", requested_seconds=0.25):
                pass
            error = OSError("PRIVATE-KEY C:/private/profile.json")
            with self.assertRaises(OSError) as caught:
                with timing.phase("capture"):
                    raise error
            timing.freeze()
        self.assertIs(caught.exception, error)
        result = timing.summary()
        sleep, capture = result["recent_operations"]
        self.assertEqual((sleep["operation"], sleep["started_monotonic"], sleep["finished_monotonic"],
                          sleep["requested_seconds"], sleep["outcome"]), ("sleep", 10, 10.85, 0.25, "completed"))
        self.assertAlmostEqual(sleep["elapsed_seconds"], 0.85)
        self.assertEqual(capture["outcome"], "raised")
        self.assertAlmostEqual(capture["elapsed_seconds"], 1.1)
        self.assertEqual(result["last_completed_operation"], capture)
        self.assertEqual(result["refusal"]["last_completed_operation"], capture)
        self.assertIsNone(result["current_operation"])
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertNotIn("profile.json", json.dumps(result))

    def test_ring_and_inflight_refusal_survive_later_cleanup_and_summary_mutation(self):
        timing = bench.ControllerTiming()
        with patch.object(bench.time, "monotonic", return_value=100.0):
            for _ in range(100):
                with timing.phase("poll"):
                    pass
            with timing.phase("log_parse"):
                timing.freeze()
                frozen = timing.summary()
            after = timing.summary()
            for _ in range(100):
                with timing.phase("discovery"):
                    pass
                timing.freeze()
        self.assertEqual(len(after["recent_operations"]), 32)
        self.assertEqual(after["recent_operations"][0]["operation"], "poll")
        self.assertEqual(after["recent_operations"][-1]["operation"], "log_parse")
        self.assertEqual(frozen["current_operation"]["outcome"], "running")
        self.assertEqual(after["refusal"]["current_operation"], frozen["current_operation"])
        self.assertEqual(after, timing.summary())
        after["recent_operations"].clear()
        after["refusal"]["current_operation"]["operation"] = "mutated"
        self.assertEqual(len(timing.summary()["recent_operations"]), 32)
        self.assertEqual(timing.summary()["refusal"]["current_operation"]["operation"], "log_parse")

    def test_only_fixed_labels_and_bounded_sleep_values_are_accepted(self):
        timing = bench.ControllerTiming()
        for label in ("PRIVATE-KEY", "C:/private/profile.json", "capture\n", "x" * 100000):
            with self.subTest(label=label[:20]), self.assertRaisesRegex(ValueError, "Unknown controller timing"):
                with timing.phase(label):
                    self.fail("Invalid label reached operation")
        for request in (True, -1, 0.251, float("inf"), float("nan"), "0.25"):
            with self.subTest(request=request), self.assertRaisesRegex(ValueError, "Invalid controller timing sleep"):
                with timing.phase("sleep", requested_seconds=request):
                    self.fail("Invalid sleep request reached operation")
        self.assertEqual(timing.summary()["recent_operations"], [])

    def test_invalid_or_backwards_clocks_never_publish_nonfinite_times(self):
        for started, finished in ((float("nan"), 10), (10, float("inf")), (True, 10), (-1, 10), (10, 9)):
            with self.subTest(started=started, finished=finished):
                timing = bench.ControllerTiming()
                with patch.object(bench.time, "monotonic", side_effect=[started, finished, float("nan")]):
                    with timing.phase("host_read"):
                        pass
                    timing.freeze()
                result = timing.summary()
                self.assertFalse(result["clock_valid"])
                row, = result["recent_operations"]
                self.assertIsNone(row["finished_monotonic"])
                self.assertIsNone(row["elapsed_seconds"])
                self.assertIsNone(result["refusal"]["observed_monotonic"])
                json.dumps(result, allow_nan=False)

    def test_unselected_guard_uses_no_diagnostic_clock_or_summary_fields(self):
        guard = bench.MemoryGuard(None, None, acceptance_floors=(4, 4))
        with patch.object(bench.time, "monotonic", side_effect=AssertionError("Diagnostic clock on unselected route")):
            with guard.timing_phase("capture"):
                pass
            guard.timing_refusal()
            self.assertNotIn("controller_timing", guard.acceptance_summary())
        self.assertFalse(guard.thread.is_alive())

    def test_serial_gap_bounds_and_refusal_survive_cleanup_without_attributing_unmeasured_time(self):
        from tools import prepare_unsloth_q2 as preparation
        for timed in (False, True):
            with self.subTest(timed=timed):
                now = [10.0]
                def snapshot():
                    now[0] += 0.0001191
                    return {"available_gib": 8, "commit_available_gib": 9}
                def stop():
                    now[0] += 100
                    raise OSError("synthetic cleanup failure")
                owner = Mock()
                owner.stop.side_effect = stop
                guard = bench.MemoryGuard(owner, None, acceptance_floors=(4, 4))
                guard.controller_timing = bench.ControllerTiming()
                with patch.object(bench.time, "monotonic", side_effect=lambda: now[0]), \
                        patch.object(preparation, "memory_snapshot", side_effect=snapshot):
                    guard.monitor(once=True)
                    with guard.timing_phase("capture") if timed else contextlib.nullcontext():
                        now[0] += 0.8485836
                    with guard.timing_phase("sleep", requested_seconds=0.25):
                        now[0] += 0.25
                    guard.monitor(once=True)
                    result = guard.acceptance_summary()
                    diagnostic = result["controller_timing"]
                    guard.owner = None
                    guard.monitor(once=True)
                    self.assertEqual(guard.acceptance_summary()["controller_timing"], diagnostic)
                self.assertEqual(guard.reason,
                    "Acceptance telemetry failed: Host sampling gap or read duration exceeded one second.")
                self.assertAlmostEqual(result["max_gap_seconds"], 1.0987027)
                self.assertEqual(guard.cleanup_error, "OSError: synthetic cleanup failure")
                self.assertLess(diagnostic["refusal"]["observed_monotonic"], 12)
                samples = result["raw_samples"]
                reads = [row for row in diagnostic["recent_operations"] if row["operation"] == "host_read"]
                self.assertEqual(len(reads), 2)
                for row, sample in zip(reads, samples):
                    self.assertEqual(row["started_monotonic"], sample["started_monotonic"])
                    self.assertEqual(row["finished_monotonic"], sample["finished_monotonic"])
                    self.assertAlmostEqual(row["elapsed_seconds"], 0.0001191)
                capture = [row for row in diagnostic["recent_operations"] if row["operation"] == "capture"]
                self.assertEqual(len(capture), int(timed))
                if timed:
                    self.assertAlmostEqual(capture[0]["elapsed_seconds"], 0.8485836)


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
        self.assertEqual(result["cuda"], {"rows": [], "minimum_enforced_mib": None, "serving_rows": 0,
            "completed_request_rows": 0, "status": "refused", "errors": [
                "Engine log was not bound to its authenticated final bytes.",
                "No enforced serving CUDA-free row was observed.",
                "Required completed-request marks are not configured."]})
        self.assertNotIn("cuda_stage_evidence", result)
        self.assertNotIn("cuda_coverage", result)
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
        composition = StageCCompositionTests("test_short_static_full_public_composition_passes")
        composition.setUp()
        self.addCleanup(composition.doCleanups)
        complete_record = composition.invoke()
        self.assertTrue(bench.acceptance_final_predicate(complete_record))
        timing = bench.ControllerTiming()
        # Saturate the fixed ring and refusal/current/last fields with long finite floats.
        with patch.object(bench.time, "monotonic", side_effect=lambda: next(clock)):
            clock = iter([value for _ in range(32) for value in (1.2345678901234567e+200,
                                                                    1.2345678901234567e+201)]
                         + [1.2345678901234567e+200, 1.2345678901234567e+201,
                            1.7976931348623157e+308])
            for _ in range(32):
                with timing.phase("sleep", requested_seconds=0.25):
                    pass
            with timing.phase("sleep", requested_seconds=0.25):
                timing.freeze()
        self.assertEqual(len(timing.summary()["recent_operations"]), 32)
        self.assertLessEqual(len(json.dumps(timing.summary(), indent=4).encode()), 16384)
        temporary = tempfile.TemporaryDirectory(prefix="receipt-cap-")
        self.addCleanup(temporary.cleanup)
        log = Path(temporary.name) / "engine.log"
        guard = bench.MemoryGuard(None, log, acceptance_floors=(4, 4), sample_limit=4 * 5400 + 256,
                                  engine_log_limit=bench.ACCEPTANCE_LOG_LIMIT, required_request_marks=11)
        for index in range(4 * 5400 + 256):
            start = 12345678.123456789 + index / 4
            self.sample(guard, physical=32, commit=64, start=start, end=start + 0.000123456)
        # Maximal encoded stages, largest admitted integer MiB and a long finite observation clock.
        for stage in ("s" * 128, "\x01" * 21 + "ss", "\u00e9" * 21 + "ss",
                      "\U0001f600" * 10 + "s" * 8, '"' * 64, "\\" * 64):
            with self.subTest(stage=ascii(stage)):
                self.assertLessEqual(len(stage), 128)
                self.assertEqual(len(json.dumps(stage).encode("utf-8")) - 2, 128)
                # Each specimen gets a fresh reader; the full-duration host evidence is shared.
                reader = bench.MemoryGuard(None, log, acceptance_floors=(4, 4),
                                           engine_log_limit=bench.ACCEPTANCE_LOG_LIMIT, required_request_marks=11)
                reader.host_samples = guard.host_samples
                reader.samples, reader.max_gap_seconds = guard.samples, guard.max_gap_seconds
                reader.sample_limit = guard.sample_limit
                lines = ["strata serve: 9007199254740991 MiB of VRAM free with everything loaded"]
                lines += [f"strata trace: 9007199254740991 MiB free after {stage}"] * 4095
                data = ("\r\n".join(lines) + "\r\n").encode("utf-8")
                self.assertLessEqual(len(data), bench.ACCEPTANCE_LOG_LIMIT)
                log.write_bytes(data)
                with patch.object(bench.time, "monotonic", return_value=sys.float_info.max):
                    reader.read_engine_log()
                    with log.open("rb") as stream:
                        opened = bench.os.fstat(stream.fileno())
                    reader.finish_engine_log(data, (opened.st_dev, opened.st_ino))
                self.assertIsNone(reader.reason)
                summary = reader.acceptance_summary(reader.host_samples[0]["finished_monotonic"],
                                                    reader.host_samples[-1]["started_monotonic"])
                self.assertEqual(summary["host_evidence"], "satisfied")
                self.assertEqual(len(summary["raw_samples"]), 21856)
                self.assertEqual(set(summary["raw_samples"][0]), {"started_monotonic", "finished_monotonic",
                                                               "physical_available_bytes", "commit_available_bytes"})
                self.assertEqual(len(summary["cuda"]["rows"]), 4096)
                self.assertEqual(summary["cuda"]["rows"][-1], {
                    "cuda_free_mib": 9007199254740991, "stage": stage, "enforced": True,
                    "observed_monotonic": sys.float_info.max})
                self.assertEqual(summary["cuda"]["errors"], ["Completed-request CUDA marks: 0 of 11 required."])
                summary["controller_timing"] = timing.summary()
                record = dict(complete_record, memory=summary)
                payload = (json.dumps(record, indent=2) + "\n").encode("utf-8")
                self.assertEqual(bench.ACCEPTANCE_RECEIPT_LIMIT, 6 * 1024**2)
                self.assertLessEqual(len(payload), bench.ACCEPTANCE_RECEIPT_LIMIT)
                self.assertEqual(json.loads(payload)["memory"]["controller_timing"], timing.summary())

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


class EngineLogEvidenceTests(unittest.TestCase):
    """The acceptance guard's incremental engine-log reader on synthetic r8-format logs."""

    SERVING = "strata serve: 761 MiB of VRAM free with everything loaded"
    MARKS = (731, 729, 715, 702, 698, 690, 688, 686, 685, 684, 683)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="engine-log-")
        self.addCleanup(temporary.cleanup)
        self.log = Path(temporary.name) / "engine.log"

    def guard(self, marks=11, **kwargs):
        kwargs.setdefault("engine_log_limit", bench.ACCEPTANCE_LOG_LIMIT)
        return bench.MemoryGuard(Mock(), self.log, acceptance_floors=(4, 4), required_request_marks=marks, **kwargs)

    @staticmethod
    def text(lines):
        return "".join(line + "\r\n" for line in lines).encode("utf-8")  # Windows text-mode stderr.

    def lines(self, marks):
        return (["[strata] 2026-10-23 12:00:00 engine started: strata.exe --serve",
                 "strata generate: CUDA0 context ready, 10.21 GiB free before expert arena registration",
                 "strata trace: 2310 MiB free after opening the expert cache", self.SERVING]
                + [f"strata trace: prefill 900 MiB free after entry\nstrata trace: {mib} MiB free after the completed request"
                   for mib in marks])

    def append(self, data):
        with self.log.open("ab") as stream:
            stream.write(data)

    def finish(self, guard):
        data = self.log.read_bytes()
        with self.log.open("rb") as stream:
            opened = bench.os.fstat(stream.fileno())
        guard.finish_engine_log(data, (opened.st_dev, opened.st_ino))
        return guard.cuda_summary()

    def complete(self, guard, lines):
        data = self.text(lines)
        middle = len(data) // 2  # A line split across reads stays pending until completed.
        self.log.write_bytes(data[:middle])
        guard.read_engine_log()
        self.append(data[middle:])
        guard.read_engine_log()
        return self.finish(guard)

    def assert_refused(self, guard, reason, error=None):
        summary = guard.cuda_summary()
        self.assertEqual(summary["status"], "refused")
        self.assertEqual(guard.reason, reason)
        guard.owner.stop.assert_called_once_with()
        if error is not None:
            self.assertEqual(summary["errors"][0], error)
        with self.assertRaises(RuntimeError) as caught:
            guard.check()
        self.assertEqual(str(caught.exception), reason)
        return summary

    def test_short_sequence_with_eleven_marks_is_satisfied(self):
        guard = self.guard(11)
        summary = self.complete(guard, self.lines(self.MARKS))
        self.assertEqual(summary["status"], "satisfied")
        self.assertEqual(summary["errors"], [])
        self.assertEqual(summary["serving_rows"], 1)
        self.assertEqual(summary["completed_request_rows"], 11)
        self.assertEqual(summary["minimum_enforced_mib"], 683)
        # context-ready and the pre-serving trace are recorded but not enforced.
        self.assertEqual([(row["stage"], row["cuda_free_mib"], row["enforced"]) for row in summary["rows"][:5]],
                         [("context-ready", 10.21 * 1024, False), ("opening the expert cache", 2310, False),
                          ("serving", 761, True), ("prefill entry", 900, True), ("the completed request", 731, True)])
        self.assertEqual(len(summary["rows"]), 3 + 2 * 11)
        self.assertIsNone(guard.reason)
        guard.owner.stop.assert_not_called()

    def test_near_full_sequence_with_three_marks_is_satisfied(self):
        guard = self.guard(3)
        summary = self.complete(guard, self.lines((1210, 940, 812)))
        self.assertEqual((summary["status"], summary["completed_request_rows"], summary["minimum_enforced_mib"]),
                         ("satisfied", 3, 761))

    def test_no_serving_row_is_refused(self):
        guard = self.guard(11)
        summary = self.complete(guard, [line for line in self.lines(self.MARKS) if line != self.SERVING])
        self.assertEqual(summary["status"], "refused")
        self.assertEqual(summary["serving_rows"], 0)
        self.assertEqual(summary["completed_request_rows"], 11)
        self.assertEqual(summary["errors"], ["No enforced serving CUDA-free row was observed."])
        guard.owner.stop.assert_not_called()

    def test_one_missing_request_mark_is_refused(self):
        guard = self.guard(11)
        summary = self.complete(guard, self.lines(self.MARKS[:-1]))
        self.assertEqual(summary["errors"], ["Completed-request CUDA marks: 10 of 11 required."])
        self.assertEqual(summary["status"], "refused")

    def test_unconfigured_marks_and_unbound_stream_are_refused(self):
        guard = self.guard(None)
        self.log.write_bytes(self.text(self.lines(self.MARKS)))
        guard.read_engine_log()
        self.assertEqual(guard.cuda_summary()["errors"], ["Engine log was not bound to its authenticated final bytes.",
                                                          "Required completed-request marks are not configured."])

    def test_255_mib_mark_stops_owner_and_keeps_first_reason(self):
        guard = self.guard(11)
        marks = (731, 255) + self.MARKS[2:]
        self.log.write_bytes(self.text(self.lines(marks)))
        guard.read_engine_log()
        summary = self.assert_refused(guard, "Actual engine CUDA-free memory below 256 MiB at the completed request")
        self.assertEqual(summary["minimum_enforced_mib"], 255)
        self.assertEqual(summary["completed_request_rows"], 2)  # Reading stops at the failing row.
        self.assertIn("An enforced CUDA-free row is below 256 MiB.", summary["errors"])
        guard.read_engine_log()
        self.assertEqual(len(guard.cuda_rows), len(summary["rows"]))
        self.assertEqual(self.finish(guard)["status"], "refused")

    def test_below_floor_cleanup_failure_is_separate_and_earlier_reason_survives(self):
        guard = self.guard(11)
        guard.owner.stop.side_effect = RuntimeError("synthetic owned survivor")
        self.log.write_bytes(self.text([self.SERVING, "strata trace: 100 MiB free after the completed request"]))
        guard.read_engine_log()
        self.assertEqual(guard.reason, "Actual engine CUDA-free memory below 256 MiB at the completed request")
        self.assertEqual(guard.cleanup_error, "RuntimeError: synthetic owned survivor")
        guard = self.guard(11)
        guard.reason = "Physical or available commit fell below the original 2 GiB emergency floor."
        guard.read_engine_log()
        self.assertEqual(guard.reason, "Physical or available commit fell below the original 2 GiB emergency floor.")
        self.assertEqual(guard.cuda_summary()["minimum_enforced_mib"], 100)

    def test_pre_serving_trace_below_floor_is_recorded_without_enforcement(self):
        guard = self.guard(1)
        self.complete(guard, ["strata trace: 10 MiB free after opening the expert cache",
                              "strata trace: prefill 200 MiB free after entry",
                              "strata generate: CUDA0 context ready, 0.10 GiB free before expert arena registration",
                              self.SERVING, "strata trace: 700 MiB free after the completed request"])
        self.assertIsNone(guard.reason)
        self.assertEqual(guard.cuda_summary()["status"], "satisfied")
        guard = self.guard(1)
        self.log.write_bytes(self.text([self.SERVING, "strata trace: prefill 200 MiB free after entry"]))
        guard.read_engine_log()
        self.assertEqual(guard.reason, "Actual engine CUDA-free memory below 256 MiB at prefill entry")

    def test_other_guard_formats_are_enforced_like_the_reference_reader(self):
        cases = (("INFO context=131072 vram_free_mib=333", "engine-metric", 333),
                 ("strata serve: VRAM 512 MiB kept free: cache, 300 MiB free now", "request-runtime", 300),
                 ("strata serve: WARNING: 600 MiB of VRAM free with everything loaded - LOW", "serving", 600))
        for line, stage, value in cases:
            with self.subTest(stage=stage):
                self.log.write_bytes(self.text([line]))
                guard = self.guard(1)
                guard.read_engine_log()
                self.assertEqual([(row["stage"], row["cuda_free_mib"], row["enforced"]) for row in guard.cuda_rows],
                                 [(stage, value, True)])

    def reader_refused(self, error, data=None, *, setup=None, guard=None):
        guard = guard or self.guard(11)
        if setup is not None:
            setup(guard)
        else:
            self.log.write_bytes(data)
            guard.read_engine_log()
        summary = self.assert_refused(guard, "Engine CUDA-stage evidence failed: " + error, error)
        self.assertNotIn("PRIVATE", json.dumps(summary))
        return summary

    def test_unrecognized_free_memory_trace_is_refused(self):
        for line in ("strata trace: 900 MiB free before the request", "strata trace: MiB free after PRIVATE"):
            with self.subTest(line=line):
                self.reader_refused("Unrecognized CUDA memory trace.", self.text([self.SERVING, line]))

    def test_strict_utf8_is_required(self):
        self.reader_refused("Engine log is not strict UTF-8.", self.text([self.SERVING]) + b"PRIVATE \xff\r\n")

    def test_line_and_row_caps_and_value_bounds_refuse(self):
        self.reader_refused("Engine log line exceeded 64 KiB.", b"PRIVATE" * 10000)
        self.reader_refused("Engine log line exceeded 64 KiB.", b"x" * 65537 + b"\n")
        rows = [self.SERVING] + ["strata trace: 900 MiB free after the completed request"] * 4095
        guard = self.guard(11)
        self.log.write_bytes(self.text(rows))
        guard.read_engine_log()
        self.assertIsNone(guard.reason)
        self.assertEqual(len(guard.cuda_rows), 4096)
        self.append(self.text(["strata trace: 900 MiB free after the completed request"]))
        self.reader_refused("CUDA-stage row limit reached; evidence is incomplete.",
                            setup=lambda guard: guard.read_engine_log(), guard=guard)
        self.reader_refused("CUDA memory trace exceeds its evidence bounds.",
                            self.text(["strata trace: 900 MiB free after " + "s" * 129]))
        self.reader_refused("CUDA memory trace exceeds its evidence bounds.",
                            self.text(["strata trace: " + "9" * 16 + " MiB free after the completed request"]))
        summary = self.reader_refused(
            "CUDA memory trace has an unreadable value.",
            self.text(["strata trace: " + "9" * 5000 + " MiB free after the completed request"]))
        self.assertEqual(summary["rows"], [])
        self.reader_refused("CUDA memory trace exceeds its evidence bounds.",
                            self.text(["strata generate: CUDA0 context ready, " + "9" * 400 + " GiB free"]))

    def test_overlong_numeric_values_refuse_with_a_fixed_message(self):
        # The review probe's float() error quoted 64 KiB of digits into every reason copy.
        for line in ("strata generate: CUDA0 context ready, " + "1." * 32000 + " GiB free",
                     "strata trace: " + "7" * 60000 + " MiB free after the completed request"):
            with self.subTest(line=line[:40]):
                self.assertLess(len(line), 65536)
                summary = self.reader_refused("CUDA memory trace has an unreadable value.",
                                              self.text([self.SERVING, line]))
                guard_text = json.dumps(summary)
                self.assertNotIn("1.1.1", guard_text)
                self.assertNotIn("7777", guard_text)
                self.assertLess(len(guard_text), 1024)

    def test_stage_json_payload_boundary_preserves_ascii_and_counts_escape_expansion(self):
        # Non-BMP characters encode as surrogate pairs. Lone surrogates cannot enter a strict UTF-8 log,
        # but the row owner must still count their JSON expansion when called directly.
        stages = ("s" * 128, "\x01" * 21 + "ss", "\u00e9" * 21 + "ss",
                  "\U0001f600" * 10 + "s" * 8, "\ud800" * 21 + "ss", "\udfff" * 21 + "ss",
                  '"' * 64, "\\" * 64, "s" + "\t" * 63 + "s", "s" + "\b" * 63 + "s")
        for stage in stages:
            with self.subTest(stage=ascii(stage)):
                guard = self.guard(1)
                self.assertEqual(len(json.dumps(stage).encode("utf-8")) - 2, 128)
                row = guard.engine_row("strata trace: 731 MiB free after " + stage)
                self.assertEqual(row["stage"], stage)
                self.assertEqual(guard.cuda_rows, [row])
                with self.assertRaisesRegex(ValueError, r"^CUDA memory trace exceeds its evidence bounds\.$"):
                    guard.engine_row("strata trace: 731 MiB free after " + stage + "s")
                self.assertEqual(guard.cuda_rows, [row])  # Refusal precedes insertion.

    def test_over_budget_stage_fails_closed_without_recording_offending_text(self):
        stages = ("\x01" * 128,) + tuple("PRIVATE-STAGE-SENTINEL" + suffix
                   for suffix in ("\x01" * 21, "\u00e9" * 21, "\U0001f600" * 10, '"' * 64, "\\" * 64))
        for stage in stages:
            with self.subTest(stage=ascii(stage)):
                guard = self.guard(1)
                data = self.text([self.SERVING, "strata trace: 731 MiB free after the completed request",
                                  "strata trace: 731 MiB free after " + stage,
                                  "strata trace: 732 MiB free after later row"])
                summary = self.reader_refused("CUDA memory trace exceeds its evidence bounds.", data, guard=guard)
                self.assertEqual([row["stage"] for row in summary["rows"]], ["serving", "the completed request"])
                self.assertEqual(summary["completed_request_rows"], 1)
                self.assertNotIn(json.dumps(stage)[1:-1], json.dumps(summary))
                guard.read_engine_log()
                self.assertEqual(self.finish(guard), summary)
                self.assertEqual(self.log.read_bytes(), data)

    def test_admitted_numeric_shapes_keep_their_actual_bounds(self):
        guard = self.guard(1)
        cases = (("strata trace: 9007199254740991 MiB free after widest integer", 9007199254740991),
                 ("strata generate: CUDA0 context ready, 8796093022207.999 GiB free", 9007199254740991.0),
                 ("strata generate: CUDA0 context ready, 0.00000000000000000001 GiB free", 1.024e-17),
                 ("strata generate: CUDA0 context ready, 0 GiB free", 0.0))
        for line, expected in cases:
            with self.subTest(line=line):
                row = guard.engine_row(line)
                self.assertEqual(type(row["cuda_free_mib"]), type(expected))
                self.assertEqual(row["cuda_free_mib"], expected)
                self.assertLessEqual(len(json.dumps(row["cuda_free_mib"])), 24)
        rows = list(guard.cuda_rows)
        for line in ("strata trace: 9007199254740992 MiB free after too large",
                     "strata generate: CUDA0 context ready, 8796093022208 GiB free",
                     "strata generate: CUDA0 context ready, " + "9" * 400 + " GiB free"):
            with self.subTest(line=line[:80]):
                with self.assertRaisesRegex(ValueError, r"^CUDA memory trace exceeds its evidence bounds\.$"):
                    guard.engine_row(line)
                self.assertEqual(guard.cuda_rows, rows)

    def test_private_value_in_a_recognized_row_is_refused_and_never_recorded(self):
        for secret in ("synthetic-api-key", "ab" * 32):
            for line in ("strata trace: 731 MiB free after " + secret,
                         "strata serve: 761 MiB of VRAM free with everything loaded " + secret):
                with self.subTest(secret=secret, line=line[:20]):
                    guard = self.guard(11)
                    guard.engine_log_secrets = ("synthetic-api-key", "ab" * 32)
                    summary = self.reader_refused("CUDA memory trace contains a private value.",
                                                  self.text([self.SERVING, line]), guard=guard)
                    self.assertEqual([row["stage"] for row in summary["rows"]], ["serving"])
                    self.assertNotIn(secret, json.dumps(summary) + guard.reason)

    def test_backlog_and_size_limits_refuse(self):
        data = self.text(["strata generate: idle"] * 50000)
        self.assertGreater(len(data), 1024**2)
        self.reader_refused("Engine log exceeded its size limit.", data)
        self.reader_refused("Engine log parsing backlog exceeded 1 MiB.", data,
                            guard=self.guard(11, engine_log_limit=2 * bench.LOG_LIMIT))

    def test_truncated_disappearing_replaced_and_linked_logs_refuse(self):
        first = self.text(self.lines(self.MARKS[:3]))
        def truncated(guard):
            self.log.write_bytes(first)
            guard.read_engine_log()
            self.log.write_bytes(first[:10])
            guard.read_engine_log()
        self.reader_refused("Engine log truncated.", setup=truncated)
        def disappeared(guard):
            self.log.write_bytes(b"")
            guard.read_engine_log()
            self.log.unlink()
            guard.read_engine_log()
        self.reader_refused("Engine log disappeared.", setup=disappeared)
        def replaced(guard):
            self.log.write_bytes(first)
            guard.read_engine_log()
            self.log.unlink()
            self.log.write_bytes(first)  # Identical bytes in a different object.
            guard.read_engine_log()
        self.reader_refused("Engine log identity changed.", setup=replaced)
        def linked(guard):
            self.log.unlink()
            target = self.log.with_name("target.log")
            target.write_bytes(first)
            self.log.symlink_to(target)
            guard.read_engine_log()
        self.reader_refused("Engine log is linked.", setup=linked)

    def test_absent_log_before_creation_is_not_an_error(self):
        guard = self.guard(11)
        guard.read_engine_log()
        self.assertIsNone(guard.reason)
        self.assertEqual(guard.cuda_rows, [])

    def test_final_bytes_must_extend_the_consumed_stream(self):
        data = self.text(self.lines(self.MARKS))
        def rewritten(guard):
            self.log.write_bytes(data)
            guard.read_engine_log()
            self.log.write_bytes(data.replace(b"731", b"999"))  # Same size, consumed bytes changed.
            self.finish(guard)
        self.reader_refused("Engine log changed after guard consumption.", setup=rewritten)
        def other_object(guard):
            self.log.write_bytes(data[:100])
            guard.read_engine_log()
            guard.finish_engine_log(data, (0, 0))
        self.reader_refused("Engine log changed after guard consumption.", setup=other_object)
        def shorter(guard):
            self.log.write_bytes(data)
            guard.read_engine_log()
            guard.finish_engine_log(data[:-10], guard.engine_identity)
        self.reader_refused("Engine log changed after guard consumption.", setup=shorter)
        def incomplete(guard):
            self.log.write_bytes(data + b"strata trace: 700 MiB free after the completed")
            guard.read_engine_log()
            self.finish(guard)
        self.reader_refused("Engine log ends with an incomplete line.", setup=incomplete)

    def test_legacy_guard_keeps_its_limit_and_never_parses_the_log(self):
        guard = bench.MemoryGuard(Mock(), self.log)
        self.assertEqual((guard.engine_log_limit, guard.required_request_marks), (2 * bench.LOG_LIMIT, None))
        self.log.write_bytes(self.text([self.SERVING, "strata trace: 10 MiB free after the completed request",
                                        "strata trace: 900 MiB free before PRIVATE"]))
        guard.done = Mock()
        guard.done.is_set.side_effect = [False, True]
        with patch.object(bench.psutil, "virtual_memory", return_value=types.SimpleNamespace(available=8 * 1024**3)), \
                patch.object(bench.psutil, "swap_memory", return_value=types.SimpleNamespace(used=0)):
            guard.monitor()
        self.assertIsNone(guard.reason)
        self.assertEqual(guard.cuda_rows, [])
        guard.owner.stop.assert_not_called()
        self.assertEqual(set(guard.summary()), {"samples", "min_available_gib", "max_pagefile_used_gib",
                                                "pagefile_available", "pagefile_unavailable_reason", "stop_reason"})
        with self.log.open("r+b") as stream:
            stream.truncate(2 * bench.LOG_LIMIT + 1)
        guard.done.is_set.side_effect = [False, True]
        with patch.object(bench.psutil, "virtual_memory", return_value=types.SimpleNamespace(available=8 * 1024**3)), \
                patch.object(bench.psutil, "swap_memory", return_value=types.SimpleNamespace(used=0)):
            guard.monitor()
        self.assertEqual(guard.reason, "Engine log exceeded the diagnostic size limit.")
        guard.owner.stop.assert_called_once_with()

    def test_worker_engine_log_check_is_bounded_strict_and_secret_free(self):
        class File:
            handle = 7
            def __init__(self, data):
                self.data, self.limits = data, []
            def read(self, limit):
                self.limits.append(limit)
                return self.data[:limit]
        secret = "private-api-key"
        data = self.text(self.lines(self.MARKS))
        file = File(data)
        self.assertEqual(bench.acceptance_engine_log(file, secret), data)
        self.assertEqual(file.limits, [bench.ACCEPTANCE_LOG_LIMIT + 1])
        for value, error in ((b"x" * (bench.ACCEPTANCE_LOG_LIMIT + 1), "Engine log exceeds its size budget."),
                             (data + b"\xff\r\n", "Engine log is not strict UTF-8."),
                             (data + b"Authorization: Bearer private-api-key\r\n", "Engine log contains the API key.")):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, "^" + error.replace(".", r"\.") + "$"):
                bench.acceptance_engine_log(File(value), secret)
        for missing in (None, types.SimpleNamespace(handle=None)):
            with self.assertRaisesRegex(RuntimeError, "^Engine log is missing\\.$"):
                bench.acceptance_engine_log(missing, secret)

    def test_real_worker_engine_log_handle_admits_only_the_engine_append_open(self):
        import _winapi
        api = bench.WindowsEngineLogAPI()
        directory = self.log.parent
        handle = api.open(self.log)
        try:
            other = directory / "other.log"
            other.write_bytes(b"external replacement bytes")
            with open(self.log, "a", encoding="utf-8") as server:  # serve/server.py StrataEngine's open.
                server.write(self.SERVING + "\n")
                server.flush()
                for attack in (self.log.unlink, lambda: self.log.rename(directory / "moved.log"),
                               lambda: other.replace(self.log)):
                    with self.assertRaises(PermissionError):
                        attack()
            self.assertEqual(other.read_bytes(), b"external replacement bytes")
            with self.assertRaises(PermissionError):
                self.log.unlink()  # The worker handle alone still denies deletion.
            bench.os.link(self.log, directory / "alias.log")  # A hard link names the same object.
            self.assertTrue(bench.os.path.samefile(self.log, directory / "alias.log"))
            api.seek(handle, 0)
            self.assertEqual(api.read(handle, 4096), (self.SERVING + "\r\n").encode())
        finally:
            api.close(handle)
        self.log.unlink()
        (directory / "alias.log").unlink()
        # Planted names refuse the create-once open and are never followed or truncated.
        foreign = directory / "foreign.txt"
        foreign.write_bytes(b"external foreign bytes")
        dangling = directory / "dangling-target.txt"
        plants = {"file": lambda: self.log.write_bytes(b"planted"),
                  "hardlink": lambda: bench.os.link(foreign, self.log),
                  "symlink": lambda: self.log.symlink_to(foreign),
                  "dangling-symlink": lambda: self.log.symlink_to(dangling),
                  "junction": lambda: _winapi.CreateJunction(str(directory / "jdir"), str(self.log))}
        (directory / "jdir").mkdir()
        for kind, plant in plants.items():
            with self.subTest(kind=kind):
                plant()
                with self.assertRaises(OSError):
                    api.close(api.open(self.log))
                self.assertEqual(foreign.read_bytes(), b"external foreign bytes")
                self.assertFalse(dangling.exists())
                if kind == "junction":
                    bench.os.rmdir(self.log)
                else:
                    self.log.unlink()


class LaunchEvidenceTests(unittest.TestCase):
    # Source formats: generate.cpp configure_file_cache/auto_cache and expert_source.cpp complement_ready_.
    RECENT = ("strata generate: recent file expert cache 4525.0 MiB "
              "(requested 4525 MiB; compact buffers, lazy, 8 GiB headroom)\n")
    AUTO = ("strata generate: expert cache auto: 10.21 GiB free, 1024 MiB reserved "
            "(+0 MiB for the draft head) -> 4542 slots\n")
    PINNED = ("FileExpertSource: mapped pinned cache complement ready: "
              "resident 9.00 GiB, pinned 9.00 GiB in 12 s\n")
    SERVING = "strata serve: 761 MiB of VRAM free with everything loaded\n"

    def setUp(self):
        self.launch = {"cohort": "automatic", "sample_status": "available",
                       "sample": {"headroom_requested_gib": 8}}
        self.allocations = bench.acceptance_allocations(self.AUTO + self.PINNED + self.RECENT + self.SERVING)

    def evidence(self, *, launch=None, cohort="automatic", floors=(8, 8), allocations=None):
        return bench.acceptance_launch_evidence(self.launch if launch is None else launch, cohort, floors,
                                                self.allocations if allocations is None else allocations)

    def test_automatic_eight_gib_policy_satisfies_both_eight_gib_floors(self):
        self.assertEqual(self.evidence(), {"status": "satisfied"})
        self.launch["sample"]["headroom_requested_gib"] = 8.0
        self.assertEqual(self.evidence(), {"status": "satisfied"})

    def test_eight_gib_headroom_refuses_four_gib_floors_and_each_asymmetric_floor(self):
        for floors in ((4, 4), (8, 4), (4, 8), (8, 7.999), (7.999, 8)):
            with self.subTest(floors=floors):
                self.assertEqual(self.evidence(floors=floors), {
                    "status": "refused", "reason": "Automatic launch headroom exceeds the physical or commit floor."})

    def test_launch_must_be_a_dictionary(self):
        for launch in (None, [], "automatic", True, 8):
            with self.subTest(launch=launch):
                self.assertEqual(bench.acceptance_launch_evidence(launch, "automatic", (8, 8), self.allocations), {
                    "status": "refused", "reason": "Launch memory evidence is unavailable."})

    def test_saved_static_requires_static_and_automatic_requires_automatic(self):
        for selected, realized in (("saved-static", "automatic"), ("automatic", "static"),
                                    ("saved-static", "saved-static"), ("automatic", None)):
            with self.subTest(selected=selected, realized=realized):
                self.launch["cohort"] = realized
                self.assertEqual(self.evidence(cohort=selected), {
                    "status": "refused", "reason": "Realized launch cohort differs from the selected cohort."})

    def test_static_sample_does_not_require_an_automatic_sample_or_recent_cache(self):
        self.assertEqual(self.evidence(launch={"cohort": "static", "sample_status": "unavailable", "sample": None},
                                      cohort="saved-static", allocations=bench.acceptance_allocations(self.SERVING)),
                         {"status": "satisfied"})

    def test_frozen_and_unknown_selected_cohorts_refuse(self):
        for cohort in ("frozen-control", "static", None):
            with self.subTest(cohort=cohort):
                self.assertEqual(self.evidence(cohort=cohort), {
                    "status": "refused", "reason": "Launch cohort is unsupported."})

    def test_unavailable_or_missing_automatic_sample_status_refuses(self):
        for status in ("unavailable", None, True, "AVAILABLE"):
            with self.subTest(status=status):
                self.launch["sample_status"] = status
                self.assertEqual(self.evidence(), {
                    "status": "refused", "reason": "Automatic launch sample is unavailable."})

    def test_automatic_sample_shape_and_headroom_must_be_finite_nonnegative_numbers(self):
        for sample in (None, [], True, {}, {"headroom_requested_gib": None},
                       *[{"headroom_requested_gib": value} for value in
                         (True, False, "8", float("nan"), float("inf"), -float("inf"), -0.1, 10**400)]):
            with self.subTest(sample=sample):
                self.launch["sample"] = sample
                self.assertEqual(self.evidence(), {
                    "status": "refused", "reason": "Automatic launch headroom must be a finite nonnegative number."})

    def test_missing_recent_startup_line_refuses_even_with_available_automatic_sample(self):
        allocations = bench.acceptance_allocations(self.AUTO + self.PINNED + self.SERVING)
        self.assertEqual(self.evidence(allocations=allocations), {
            "status": "refused", "reason": "Automatic launch recent-cache startup allocation is missing."})

    def test_exact_source_format_allocation_values_and_capacity_qualification(self):
        self.assertEqual(self.allocations, {
            "recent_cache": {"actual_mib": 4525.0, "requested_mib": 4525},
            "expert_cache_auto": {"free_gib": 10.21, "reserved_mib": 1024, "draft_head_mib": 0, "slots": 4542},
            "mapped_pinned_complement": {"resident_gib": 9.0, "pinned_gib": 9.0}, "errors": [],
            "end_of_run_cache_capacity":
                "not collected; printed only under STRATA_FETCH_PROFILE, which acceptance arms exclude"})
        self.assertIs(type(self.allocations["recent_cache"]["actual_mib"]), float)
        self.assertIs(type(self.allocations["recent_cache"]["requested_mib"]), int)

    def test_last_startup_values_win_and_post_serving_values_are_ignored(self):
        early = (self.RECENT.replace("4525.0", "100.0").replace("4525 MiB", "100 MiB")
                 + self.AUTO.replace("4542", "10") + self.PINNED.replace("9.00", "1.00"))
        late = self.RECENT.replace("4525.0", "2.0") + self.AUTO.replace("4542", "2")
        for serving in (self.SERVING, self.SERVING.replace("761 MiB", "WARNING: 761 MiB"),
                        self.SERVING.rstrip() + " - LOW: requests may stall\n"):
            with self.subTest(serving=serving):
                self.assertEqual(bench.acceptance_allocations(early + self.AUTO + self.PINNED + self.RECENT
                                                              + serving + late), self.allocations)

    def test_recent_line_only_after_serving_cannot_satisfy_automatic_launch(self):
        allocations = bench.acceptance_allocations(self.SERVING + self.RECENT)
        self.assertIsNone(allocations["recent_cache"])
        self.assertEqual(self.evidence(allocations=allocations)["status"], "refused")

    def test_no_serving_boundary_refuses_startup_evidence(self):
        allocations = bench.acceptance_allocations(self.RECENT)
        self.assertEqual(allocations["errors"], ["Engine serving startup boundary is missing."])
        self.assertEqual(self.evidence(allocations=allocations), {
            "status": "refused", "reason": "Engine startup allocation evidence is unreadable or incomplete."})

    def test_malformed_and_nonfinite_allocation_values_refuse_without_echoing_text(self):
        for key, text in (("recent_cache", self.RECENT.replace("4525.0", "1..2")),
                          ("recent_cache", self.RECENT.replace("4525.0", "9" * 400)),
                          ("recent_cache", self.RECENT.replace("4525.0", "-1.0")),
                          ("recent_cache", self.RECENT.replace("requested 4525", "requested " + "9" * 5000)),
                          ("expert_cache_auto", self.AUTO.replace("10.21", "1..2")),
                          ("expert_cache_auto", self.AUTO.replace("10.21", "9" * 400)),
                          ("expert_cache_auto", self.AUTO.replace("1024", "-1")),
                          ("expert_cache_auto", self.AUTO.replace("4542", "9" * 5000)),
                          ("mapped_pinned_complement", self.PINNED.replace("9.00", "1..2")),
                          ("mapped_pinned_complement", self.PINNED.replace("9.00", "9" * 400))):
            with self.subTest(key=key, text=text[:100]):
                allocations = bench.acceptance_allocations(self.RECENT + text + self.SERVING)
                self.assertEqual(allocations["errors"], ["Unreadable startup allocation: " + key])
                self.assertEqual(self.evidence(allocations=allocations)["status"], "refused")

    def test_unrelated_auto_diagnostics_and_missing_optional_allocations_are_allowed(self):
        text = ("strata generate: expert cache auto: Windows keeps 1024 MiB free after the cache\n"
                + "strata generate: expert cache auto: the 1024 MiB reserve leaves too few slots\n"
                + self.RECENT + self.SERVING)
        allocations = bench.acceptance_allocations(text)
        self.assertEqual(allocations["errors"], [])
        self.assertIsNone(allocations["expert_cache_auto"])
        self.assertIsNone(allocations["mapped_pinned_complement"])
        self.assertEqual(self.evidence(allocations=allocations), {"status": "satisfied"})


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
            self.assertNotIn("launch_memory_metrics", temporary_cfg)
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
        self.assertNotIn("launch_memory", record)
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
        self.worker_thread = None
        self.clock_lock = threading.RLock()
        self.stage = None
        # A manifest run's fake engine appends r8-format CUDA rows to the configured log.
        self.engine_stream, self.engine_marks = None, 0
        self.engine_startup = ""
        self.engine_serving = "strata serve: 761 MiB of VRAM free with everything loaded\n"
        self.engine_mark = lambda index: "strata trace: 731 MiB free after the completed request\n"
        self.cleanup = contextlib.ExitStack()
        self.addCleanup(self.cleanup.close)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.engine_close)
        self.native = self.Native(self)
        self.NativeBinding = bench.WindowsProcessAPI
        self.SelfIdentityBinding = bench.worker_self_identity
        from tools import prepare_unsloth_q2 as preparation
        patches = [
            patch.object(bench, "acceptance_ready", return_value=True),
            patch.object(bench, "acceptance_preflight", side_effect=self.preflight),
            patch.object(bench, "acceptance_protection", side_effect=lambda args, worker_record: self.mode != "protection"),
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
        with self.clock_lock:
            self.now += 0.001
            return self.now

    def sleep(self, seconds):
        self.assertGreaterEqual(seconds, 0)
        with self.clock_lock:
            self.now += min(seconds, 0.001) if self.worker_thread is not None else seconds
        if self.worker_thread is not None:
            threading.Event().wait(0.001)  # Let the contained CPU worker and controller exchange artifacts.

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
        return {"status": "admitted"}

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
        if getattr(self.args, "acceptance_manifest", None) is None:
            self.assertIsNone(cfg["log"])
        else:
            self.assertEqual(cfg["log"], str(self.output.resolve() / "engine.log"))
            self.assertTrue(Path(cfg["log"]).is_file(), "The worker creates the engine log before the server")
            # The same append open as serve/server.py StrataEngine, into the worker's object.
            self.engine_stream = open(cfg["log"], "a", encoding="utf-8")
            self.engine_write(self.engine_startup)
            self.engine_write(self.engine_serving)
        self.events.append("server")
        def terminate():
            self.events.append("server-stop")
            self.engine_close()
        return types.SimpleNamespace(stdout=io.BytesIO(b"synthetic server output\n"),
                                     poll=lambda: None, terminate=terminate,
                                     wait=lambda timeout: self.events.append("server-wait") or 0)

    def engine_write(self, text):
        if self.engine_stream is not None and text:
            self.engine_stream.write(text)
            self.engine_stream.flush()

    def engine_close(self):
        if self.engine_stream is not None:
            self.engine_stream.close()
            self.engine_stream = None

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
            if t.worker_thread is not None:
                t.now = max(t.now, json.loads(t.bootstrap_settings)["work_deadline"] + 0.01)
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
            if self.t.worker_thread is not None:
                self.t.worker_thread.join(5)
                self.t.assertFalse(self.t.worker_thread.is_alive(), "Contained CPU worker did not finish")
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
                    def execute():
                        try:
                            t.code = bench.worker_bootstrap(io.StringIO(envelope), bench.acceptance_worker, pid=321,
                                                           settings=t.bootstrap_settings)
                        except t.BlockedChild:
                            pass  # A parked external syscall: the process, not the controller, is blocked.
                        except BaseException as exc:
                            t.child_error = exc
                            t.code = 1
                    if getattr(t.args, "acceptance_manifest", None) is None:
                        execute()
                    else:
                        t.worker_thread = threading.Thread(target=execute, daemon=True)
                        t.worker_thread.start()
            if t.worker_thread is not None:
                threading.Event().wait(0.001)
            return (b"X" * (bench.ACCEPTANCE_LOG_LIMIT + 1) if t.mode == "capture-overflow" else b"",
                    t.code is not None and t.mode != "blocked-capture")

    def invoke(self, mode="green"):
        self.mode = mode
        with contextlib.redirect_stdout(io.StringIO()):
            return bench.run(self.args)

    def invoke_held(self):
        # This isolated lifecycle fixture forces readiness without a manifest. Its
        # old lifecycle qualifiers still complete, but cannot supply Stage C proof.
        with self.assertRaises(bench.AcceptanceFailure) as caught:
            self.invoke()
        record = caught.exception.record
        self.assertEqual(record["worker_exit"], 0)
        for key in ("natural_exit", "capture_complete", "owned_cleanup", "protection", "finalization",
                    "within_deadline", "worker_complete", "base_host_floors_satisfied"):
            self.assertIs(record[key], True, key)
        self.assertEqual(record["memory"]["host_evidence"], "satisfied")
        self.assertNotIn("admission", record)
        self.assertNotIn("error", record)
        self.assertNotIn("cleanup_error", record)
        self.assertFalse(bench.acceptance_final_predicate(record))
        return record

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
        result = self.invoke_held()
        self.assertFalse(bench.acceptance_final_predicate(result))
        self.assertEqual(result["lifecycle_evidence"], "refused")
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
        self.assertEqual(code, 1)  # Forced readiness alone cannot qualify missing Stage C evidence.
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
        def slow_protection(args, worker_record):
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
        result = self.invoke_held()
        self.assertAlmostEqual(result["total_deadline"] - result["started_monotonic"], 900)
        self.assertAlmostEqual(result["work_deadline"] - result["started_monotonic"], 890)
        self.assertEqual(result["memory"]["sample_limit"], 3856)

    def test_controller_maximum_duration_scales_deadlines_and_sample_cap(self):
        self.args.acceptance_total_seconds = 5400
        result = self.invoke_held()
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
            result = self.invoke_held()
        self.assertFalse(bench.acceptance_final_predicate(result))
        self.assertEqual([call.args for call in idle.call_args_list], [(18081,), (18081,)])
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][commands[0].index("--port") + 1], "18081")
        self.assertEqual(clones[0]["port"], 18081)

    def test_controller_uses_six_mib_receipt_cap_and_keeps_other_artifact_caps(self):
        with patch.object(bench, "bounded_write", wraps=bench.bounded_write) as write:
            self.invoke_held()
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

    def invoke_selected(self, mutate=None, selection="short", missing=False, floors=(4, 4), held=False,
                        worker_evidence=False, cohort=None):
        self.args.acceptance_workload = selection
        self.args.acceptance_physical_gib, self.args.acceptance_commit_gib = floors
        self.args.expected_context = 131072
        self.args.cohort = cohort or ("saved-static" if selection == "short" else "automatic")
        if selection == "near-full":
            cfg = json.loads(self.config.read_text())
            cfg.setdefault("memory_policy", {"version": 1})
            self.config.write_text(json.dumps(cfg))
        adapter = types.SimpleNamespace(args=self.args, count=StageCSourceTests.count, calls=[], change=None,
            metric_change=None, reference_text="synthetic reference with planted codes",
            launch_memory=getattr(self, "launch_memory", {"cohort": "static", "sample_status": "unavailable"}))
        base_client = StageCSourceTests.client_type(adapter)
        guards = []
        controller_thread = threading.current_thread()
        real_guard, real_workload, real_idle, real_open = (bench.MemoryGuard, bench.acceptance_workload_tests,
                                                         bench.acceptance_idle, Path.open)
        def make_guard(*args, **kwargs):
            guard = real_guard(*args, **kwargs)
            phase = guard.timing_phase
            # advance() synthesizes host observations on the fake worker thread.
            # Production monitor calls are serial on the controller; only those
            # calls participate in its operation timing, not this fixture's clock driver.
            guard.timing_phase = lambda *args, **kwargs: (phase(*args, **kwargs)
                if threading.current_thread() is controller_thread else contextlib.nullcontext())
            monitor = guard.monitor
            def sample(*args, **kwargs):
                # One lock order on every thread: the owner's lock, then the clock, as in
                # OwnedProcess.finish(), which reads the clock while it holds its own lock.
                owner = guard.owner
                with owner.lock if owner is not None else contextlib.nullcontext(), self.clock_lock:
                    return monitor(*args, **kwargs)
            guard.monitor = sample
            guards.append(guard)
            return guard
        def advance(seconds):
            while seconds > 0:
                step = min(0.25, seconds)
                with self.clock_lock:
                    self.now += step
                    seconds -= step
                # Advance the one external sampler without overlapping a read. After the fake
                # Job is cancelled the controller may hold the owner's lock while it accounts
                # for and joins this worker thread, so this thread then stops waiting for it.
                owner = guards[0].owner
                locked = owner is None
                while not locked and not self.killed:
                    locked = owner.lock.acquire(timeout=0.001)
                if locked:
                    try:
                        guards[0].monitor(once=True)
                    finally:
                        if owner is not None:
                            owner.lock.release()
        class Client(base_client):
            def chat(client, *args, **kwargs):
                response = super().chat(*args, **kwargs)
                self.engine_marks += 1
                self.engine_write(self.engine_mark(self.engine_marks))
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
            if not missing and not worker_evidence:
                record["missing_evidence"] = []  # Only this CPU fixture supplies the remaining producer seams.
            save()
        def open_file(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            if mutate and path == self.output / "provisional.json" and self.code == 0 and args == ("rb",):
                with stream:
                    item = json.loads(stream.read(bench.ACCEPTANCE_LOG_LIMIT + 1))
                mutate(item)
                return io.BytesIO(json.dumps(item).encode())
            return stream
        with patch.object(bench, "MemoryGuard", side_effect=make_guard), \
                patch.object(bench, "load_workload", return_value=validation), \
                patch.object(bench, "prompt_counter", return_value=StageCSourceTests.count), \
                patch.object(bench, "check_authentication"), \
                patch.object(bench, "acceptance_workload_tests", side_effect=workload), \
                patch.object(bench, "acceptance_idle", side_effect=idle), patch.object(Path, "open", open_file), \
                patch.object(bench.time, "time", side_effect=lambda: 1800000000 + self.now):
            return self.invoke_held() if held else self.invoke()


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
        self.get_calls = []
        self.launch_memory = {"cohort": "automatic", "sample_status": "available",
                              "sample": {"headroom_requested_gib": 8}}

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
                if hasattr(test, "get_calls"):
                    test.get_calls.append(path)
                if path == "/health":
                    return {"loaded": True, "max_context": test.args.expected_context}
                return {"engine": {"arena_mib": 4096, "expert_cache_mib": 1024},
                        "launch_memory": getattr(test, "launch_memory", None),
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

    def test_selected_clone_enables_launch_metrics_and_records_after_health(self):
        self.cfg["launch_memory_metrics"] = False
        self.config.write_text(json.dumps(self.cfg))
        result = self.invoke()
        self.assertIs(self.clones[0]["launch_memory_metrics"], True)
        self.assertEqual(self.get_calls[:3], ["/health", "/metrics", "/metrics"])
        self.assertEqual(result["launch_memory"], {"cohort": "automatic", "sample_status": "available",
                                                 "sample": {"headroom_requested_gib": 8}})
        self.assertEqual(self.saved()["launch_memory"], result["launch_memory"])
        self.assertEqual(result["engine"], {"arena_mib": 4096, "expert_cache_mib": 1024})
        self.assertIs(json.loads(self.config.read_text())["launch_memory_metrics"], False)

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
        self.assertFalse(bench.acceptance_ready(self.args))
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
        self.assertEqual(result["effective_environment"], {"CUDA_MODULE_LOADING": "EAGER", "STRATA_FILE_COPY_READY": "0",
                                                           "STRATA_TRACE": "1"})

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
            "CUDA_MODULE_LOADING": "EAGER", "STRATA_FILE_COPY_READY": "0", "STRATA_TRACE": "1"})
        self.assertEqual(cfg, {"env": {"STRATA_FILE_CACHE_MIB": "4096"}})
        explicit = {"env": {"CUDA_MODULE_LOADING": "LAZY", "STRATA_FILE_COPY_READY": "1",
                            "STRATA_VERIFY_PROFILE": "1", "STRATA_API_KEY": "another-private-key"}}
        with patch.dict(bench.os.environ, inherited, clear=True):
            treatment, environment = bench.clean_environment(explicit, False, acceptance=True)
        self.assertEqual(config_environment(treatment, environment), {"KEEP": "value", "CUDA_MODULE_LOADING": "LAZY",
            "STRATA_FILE_COPY_READY": "1", "STRATA_VERIFY_PROFILE": "1", "STRATA_TRACE": "1"})

    def test_acceptance_clone_sets_one_exact_trace_and_legacy_cleaning_does_not(self):
        saved = {"env": {"strata_trace": "0", "Strata_Trace": "", "STRATA_TRACE": "", "KEEP_ME": "1"}}
        with patch.object(bench.os, "name", "nt"), patch.dict(bench.os.environ, {"STRATA_TRACE": "0"}, clear=True):
            clean, environment = bench.clean_environment(saved, False, acceptance=True)
        self.assertEqual(clean["env"], {"KEEP_ME": "1", "CUDA_MODULE_LOADING": "EAGER", "STRATA_FILE_COPY_READY": "0",
                                        "STRATA_TRACE": "1"})
        self.assertEqual(environment, {})
        with patch.object(bench.os, "name", "posix"), patch.dict(bench.os.environ, {}, clear=True):
            clean, _ = bench.clean_environment(saved, False, acceptance=True)
        self.assertEqual({key: clean["env"][key] for key in ("strata_trace", "Strata_Trace", "STRATA_TRACE")},
                         {"strata_trace": "0", "Strata_Trace": "", "STRATA_TRACE": "1"})
        with patch.dict(bench.os.environ, {}, clear=True):
            legacy, _ = bench.clean_environment({"env": {"STRATA_TRACE": "0"}}, False)
            unset, _ = bench.clean_environment({"env": {}}, False)
        self.assertEqual(legacy["env"], {"STRATA_TRACE": "0"})
        self.assertEqual(unset["env"], {})
        self.assertEqual(saved["env"]["STRATA_TRACE"], "")  # The saved config is never changed.

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


class AcceptanceManifestTests(unittest.TestCase):
    """Synthetic inventory and Git transports; no model, native API or child workload."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stage-c-manifest-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workloads = self.root / "bench/results/fixture"
        self.harness = [self.root / "tools/benchmark_compact_cache.py", self.root / "serve/server.py",
                        self.root / "tools/prepare_unsloth_q2.py", self.root / "setup.py",
                        self.workloads / "validation.py", self.workloads / "benchmark.py"]
        self.saved = self.root / "engine/saved.exe"
        self.candidate = self.root / "strata-candidate.exe"
        self.saved_receipt = self.root / "receipts/saved.json"
        self.selected_receipt = self.root / "BUILD-candidate.json"
        self.tokenizer = self.root / "tokenizer"
        self.pack = self.root / "pack"
        self.profile = self.root / "profile.bin"
        self.interpreter = self.root / "python.exe"
        self.protected = [self.root / "launcher.bat", self.root / "extra-input.txt"]
        self.shards = [self.root / "model-01.gguf", self.root / "model-02.gguf"]
        self.pack_data = [self.pack / "a-data.bin", self.pack / "z-data.bin"]
        self.tokenizer_files = [self.tokenizer / name for name in
                                ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja")]
        for index, path in enumerate(self.harness + [self.saved, self.candidate, self.saved_receipt,
                self.selected_receipt, self.profile, self.interpreter, self.pack / "native_experts.txt"]
                + self.protected + self.shards + self.pack_data + self.tokenizer_files):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic file " + str(index)).encode())
        (self.pack / "ignored-directory").mkdir()
        (self.pack / "ignored-directory/nested.bin").write_bytes(b"not enumerated recursively")
        receipt = {path.name: {"path": str(path), "size": path.stat().st_size,
                              "mtime_ns": path.stat().st_mtime_ns,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in reversed(self.shards)}
        self.source_receipt = self.pack / "source-verification.json"
        self.source_receipt.write_text(json.dumps(receipt))
        self.config = self.root / "saved.json"
        self.cfg = {"exe": "saved.exe", "cwd": str(self.saved.parent), "build_metadata": "receipts/saved.json",
                    "tokenizer": str(self.tokenizer), "api_key": "synthetic-private-key",
                    "fit_max_tokens": True, "open_browser": False, "memory_policy": {"version": 1},
                    "args": ["--pack", str(self.pack), "--native", str(self.shards[0]),
                             "--expert-profile", str(self.profile)]}
        self.config.write_text(json.dumps(self.cfg))
        self.args = argparse.Namespace(config=self.config, output=self.root / "new-output", timing=False,
            acceptance_telemetry=True, acceptance_workload="short", cohort="automatic", expected_context=65536,
            acceptance_physical_gib=8.0, acceptance_commit_gib=9.0, port=18081, acceptance_total_seconds=1200,
            candidate_exe=self.candidate, protect_input=[str(path) for path in self.protected],
            write_acceptance_manifest=self.root / "manifest.json", manifest_not_after_unix=10000.0)
        self.git = Mock(side_effect=lambda argv, **kwargs: types.SimpleNamespace(
            stdout=b"" if argv[1] == "status" else b"0123456789abcdef0123456789abcdef01234567\n"))
        for name, value in (("ROOT", self.root), ("WORKLOADS", self.workloads)):
            self.enterContext(patch.object(bench, name, value))
        self.enterContext(patch.object(bench, "direct_interpreter", return_value=self.interpreter))
        self.enterContext(patch.object(bench.time, "time", return_value=1000.0))
        self.enterContext(patch.object(bench.subprocess, "run", self.git))
        self.popen = self.enterContext(patch.object(bench.subprocess, "Popen",
                                                   side_effect=AssertionError("No workload process permitted")))
        self.enterContext(patch.object(bench, "acceptance_controller",
                                       side_effect=AssertionError("No controller permitted")))
        self.enterContext(patch.object(bench, "_run_workload", side_effect=AssertionError("No workload permitted")))

    def write(self):
        self.assertEqual(bench.run(self.args), 0)
        self.popen.assert_not_called()
        self.assertFalse(self.args.output.exists())
        return bench.load_acceptance_manifest(self.args.write_acceptance_manifest)

    def refused(self, reason):
        with self.assertRaisesRegex(ValueError, reason):
            bench.run(self.args)
        self.assertFalse(self.args.write_acceptance_manifest.exists())
        self.assertFalse(self.args.output.exists())
        self.popen.assert_not_called()

    def load_changed(self, manifest):
        path = self.root / "malformed.json"
        path.write_text(json.dumps(manifest))
        return bench.load_acceptance_manifest(path)

    def test_writer_exact_manifest_roles_paths_identities_and_loader_round_trip(self):
        manifest = self.write()
        roles_and_paths = [("config", self.config), ("saved-exe", self.saved), ("selected-exe", self.candidate),
            ("saved-build-receipt", self.saved_receipt), ("selected-build-receipt", self.selected_receipt),
            *[("tokenizer", path) for path in self.tokenizer_files],
            ("pack-descriptor", self.pack / "native_experts.txt"), ("pack-descriptor", self.source_receipt),
            ("expert-profile", self.profile), *[("harness", path) for path in self.harness],
            ("interpreter", self.interpreter), *[("protect-input", path) for path in self.protected]]
        expected_inputs = [{"role": role, "path": str(path), "bytes": len(path.read_bytes()),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                           for role, path in roles_and_paths]
        expected_sizes = [{"role": role, "path": str(path), "bytes": path.stat().st_size,
                           "mtime_ns": path.stat().st_mtime_ns}
                          for role, path in [("model-shard", path) for path in self.shards]
                          + [("pack-data", path) for path in self.pack_data]]
        expected = {"schema": "strata-acceptance-manifest/1", "created_unix": 1000.0,
            "not_after_unix": 10000.0, "source_head": "0123456789abcdef0123456789abcdef01234567",
            "output": str(self.args.output), "workload": "short", "cohort": "automatic",
            "expected_context": 65536, "port": 18081, "total_seconds": 1200,
            "floors_gib": {"physical": 8.0, "commit": 9.0}, "candidate_exe": str(self.candidate),
            "protect_input": [str(path) for path in self.protected], "inputs": expected_inputs,
            "size_bound": expected_sizes, "unbound": [{"role": "python-dependencies",
                "reason": "Interpreter file is hashed; imported packages and DLLs are not bound."}]}
        self.assertEqual(manifest, expected)
        self.assertEqual(self.args.write_acceptance_manifest.read_bytes(),
                         (json.dumps(expected, indent=2) + "\n").encode())
        self.assertNotIn(b"synthetic-private-key", self.args.write_acceptance_manifest.read_bytes())
        self.assertEqual(self.git.call_count, 2)
        self.assertEqual([call.args[0] for call in self.git.call_args_list], [
            ["git", "status", "--porcelain=v1", "--untracked-files=no", "--",
             *[str(path.relative_to(self.root)) for path in self.harness]], ["git", "rev-parse", "HEAD"]])
        for call in self.git.call_args_list:
            self.assertEqual(call.kwargs, {"cwd": self.root, "capture_output": True, "timeout": 30, "check": True})
        self.assertFalse(bench.acceptance_ready(self.args))

    def test_cli_writes_prints_manifest_path_and_returns_zero(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = bench.main(["--config", str(self.config), "--output", str(self.args.output),
                "--acceptance-telemetry", "--acceptance-workload", "short", "--cohort", "automatic",
                "--acceptance-physical-gib", "8", "--acceptance-commit-gib", "8", "--port", "18081",
                "--write-acceptance-manifest", str(self.args.write_acceptance_manifest),
                "--manifest-not-after-unix", "10000"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "Saved acceptance manifest to " + str(self.args.write_acceptance_manifest) + "\n")
        manifest = bench.load_acceptance_manifest(self.args.write_acceptance_manifest)
        self.assertIsNone(manifest["candidate_exe"])
        self.assertEqual(manifest["total_seconds"], 900)
        self.assertEqual(manifest["port"], 18081)
        self.assertEqual([row["role"] for row in manifest["inputs"]].count("saved-build-receipt"), 1)
        self.assertNotIn("selected-build-receipt", [row["role"] for row in manifest["inputs"]])
        self.popen.assert_not_called()

    def test_absent_receipts_remain_explicitly_source_unproven(self):
        self.saved_receipt.unlink()
        self.selected_receipt.unlink()
        manifest = self.write()
        self.assertEqual(manifest["unbound"], [
            {"role": "python-dependencies", "reason": "Interpreter file is hashed; imported packages and DLLs are not bound."},
            {"role": "saved-build-receipt", "reason": "absent; executable is source-unproven"},
            {"role": "selected-build-receipt", "reason": "absent; executable is source-unproven"}])
        self.assertFalse(any(row["role"].endswith("build-receipt") for row in manifest["inputs"]))

    def test_saved_executable_and_default_receipt_use_root_when_cwd_is_absent(self):
        self.cfg.pop("cwd")
        self.cfg.pop("build_metadata")
        self.cfg["exe"] = "engine/saved.exe"
        receipt = self.saved.parent / "BUILD-saved.json"
        receipt.write_bytes(b"synthetic default receipt")
        self.config.write_text(json.dumps(self.cfg))
        manifest = self.write()
        self.assertEqual(manifest["inputs"][1]["path"], str(self.saved))
        self.assertEqual(manifest["inputs"][3]["path"], str(receipt))

    def test_relative_engine_arguments_bind_engine_cwd_despite_caller_cwd(self):
        caller = self.root / "caller/engine"
        caller.mkdir(parents=True)
        decoy_pack = caller.parent / "pack"
        decoy_pack.mkdir()
        for path in [self.pack / "native_experts.txt", self.source_receipt, *self.pack_data]:
            (decoy_pack / path.name).write_bytes(path.read_bytes())
        (caller.parent / "profile.bin").write_bytes(b"wrong caller profile")
        self.cfg["exe"] = str(self.saved)
        self.cfg["tokenizer"] = "../../tokenizer"
        for cwd in (str(self.saved.parent), "engine", None, ""):
            self.cfg["cwd"] = cwd
            prefix = "../" if cwd else ""
            self.cfg["args"] = ["--pack", prefix + "pack", "--native", prefix + "model-01.gguf",
                                "--expert-profile", prefix + "profile.bin"]
            self.config.write_text(json.dumps(self.cfg))
            with self.subTest(cwd=cwd), contextlib.chdir(caller):
                manifest = self.write()
            bound = [(row["role"], row["path"]) for row in manifest["inputs"]
                     if row["role"] in ("pack-descriptor", "expert-profile", "tokenizer")]
            self.assertEqual(bound, [*[("tokenizer", str(path)) for path in self.tokenizer_files],
                ("pack-descriptor", str(self.pack / "native_experts.txt")),
                ("pack-descriptor", str(self.source_receipt)), ("expert-profile", str(self.profile))])
            self.assertEqual([row["path"] for row in manifest["size_bound"]],
                             [str(path) for path in self.shards + self.pack_data])
            self.args.write_acceptance_manifest.unlink()
        self.popen.assert_not_called()

    def test_existing_destination_is_preserved(self):
        self.args.write_acceptance_manifest.write_bytes(b"prior evidence")
        with self.assertRaisesRegex(ValueError, "already exists"):
            bench.run(self.args)
        self.assertEqual(self.args.write_acceptance_manifest.read_bytes(), b"prior evidence")
        self.git.assert_not_called()
        self.popen.assert_not_called()

    def test_output_contained_or_equal_destination_is_refused_before_creation(self):
        for path in (self.args.output, self.args.output / "manifest.json"):
            with self.subTest(path=path):
                self.args.write_acceptance_manifest = path
                self.refused("outside output")
        self.git.assert_not_called()

    def test_frozen_control_is_refused(self):
        self.args.cohort = "frozen-control"
        self.refused("Frozen control")
        self.git.assert_not_called()

    def test_missing_pack_or_native_flag_is_refused(self):
        for flag in ("--pack", "--native"):
            cfg = bench.copy.deepcopy(self.cfg)
            index = cfg["args"].index(flag)
            del cfg["args"][index:index + 2]
            self.config.write_text(json.dumps(cfg))
            with self.subTest(flag=flag):
                self.refused("Acceptance inputs require --pack and --native")
        self.git.assert_not_called()

    def test_writer_options_must_be_supplied_together(self):
        for key in ("write_acceptance_manifest", "manifest_not_after_unix"):
            args = argparse.Namespace(**vars(self.args))
            setattr(args, key, None)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "required together"):
                bench.run(args)
        self.assertFalse(self.args.write_acceptance_manifest.exists())
        self.git.assert_not_called()

    def test_writer_requires_telemetry_selection_and_explicit_floors(self):
        changes = [{"acceptance_telemetry": False}, {"acceptance_workload": None, "cohort": None,
            "candidate_exe": None, "protect_input": []}, {"acceptance_physical_gib": None},
            {"acceptance_commit_gib": None}, {"acceptance_physical_gib": 3.9}]
        for changed in changes:
            args = argparse.Namespace(**vars(self.args))
            vars(args).update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                bench.run(args)
        self.assertFalse(self.args.write_acceptance_manifest.exists())
        self.git.assert_not_called()

    def test_dirty_tracked_harness_refuses_before_write(self):
        self.git.side_effect = lambda *args, **kwargs: types.SimpleNamespace(stdout=b" M serve/server.py\n")
        self.refused("harness source is dirty")
        self.assertEqual(self.git.call_count, 1)

    def test_changed_source_shard_mtime_refuses_before_write(self):
        path = self.shards[0]
        stat = path.stat()
        bench.os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        self.refused("Source shard stat differs")
        self.git.assert_not_called()

    def test_changed_source_shard_size_refuses_before_write(self):
        self.shards[0].write_bytes(b"changed shard size")
        self.refused("Source shard stat differs")

    def test_duplicate_resolved_protected_path_is_refused(self):
        self.args.protect_input.append(str(self.protected[0].parent / "engine/../launcher.bat"))
        self.refused("Duplicate resolved input path")

    def test_candidate_resolving_to_saved_executable_is_refused(self):
        self.args.candidate_exe = self.saved.parent / "../engine/saved.exe"
        self.refused("Duplicate resolved input path")

    def test_missing_protected_input_is_refused(self):
        self.protected[0].unlink()
        self.refused("protected input is missing")

    def test_symbolic_link_input_or_pack_entry_is_refused(self):
        # Model the filesystem seam so this refusal does not need Windows symlink privileges.
        real_is_symlink = Path.is_symlink
        for target in (self.config, self.candidate, self.saved_receipt, self.pack_data[0], self.shards[0]):
            with self.subTest(target=target), patch.object(Path, "is_symlink",
                    lambda path: path == target or real_is_symlink(path)):
                self.refused("symbolic link")

    def test_saved_executable_direct_symbolic_link_is_refused(self):
        link = self.root / "saved-link.exe"
        link.symlink_to(self.saved)
        self.assertTrue(link.is_symlink())
        self.assertTrue(link.is_file())
        self.cfg["exe"] = str(link)
        self.config.write_text(json.dumps(self.cfg))
        self.refused("symbolic link")
        self.git.assert_not_called()

    def test_saved_executable_symbolic_link_ancestor_is_refused(self):
        link = self.root / "engine-link"
        link.symlink_to(self.saved.parent, target_is_directory=True)
        self.assertTrue(link.is_symlink())
        supplied = link / self.saved.name
        self.assertFalse(supplied.is_symlink())
        self.assertTrue(supplied.is_file())
        self.cfg["exe"] = str(supplied)
        self.config.write_text(json.dumps(self.cfg))
        self.refused("symbolic link")
        self.git.assert_not_called()

    def test_pack_directory_symbolic_link_ancestor_is_refused(self):
        link = self.root / "pack-parent-link"
        link.symlink_to(self.pack.parent, target_is_directory=True)
        self.assertTrue(link.is_symlink())
        supplied = link / self.pack.name
        self.assertFalse(supplied.is_symlink())
        self.assertTrue(supplied.is_dir())
        self.cfg["args"][self.cfg["args"].index("--pack") + 1] = str(supplied)
        self.config.write_text(json.dumps(self.cfg))
        self.refused("symbolic link")
        self.git.assert_not_called()

    def test_symbolic_link_ancestor_before_parent_traversal_is_refused(self):
        link = self.root / "engine-link"
        link.symlink_to(self.saved.parent, target_is_directory=True)
        self.assertTrue(link.is_symlink())
        supplied = link / ".." / "engine" / self.saved.name
        self.assertFalse(supplied.is_symlink())
        self.assertTrue(supplied.is_file())
        self.cfg.update(cwd=str(self.root), exe="engine-link/../engine/saved.exe")
        self.config.write_text(json.dumps(self.cfg))
        self.refused("symbolic link")
        self.git.assert_not_called()

    def test_invalid_source_head_or_expiry_refuses_before_write(self):
        for head in (b"ABCDEF0123456789abcdef0123456789abcdef0123\n", b"short\n"):
            self.git.side_effect = lambda argv, head=head, **kwargs: types.SimpleNamespace(
                stdout=b"" if argv[1] == "status" else head)
            with self.subTest(head=head):
                self.refused("digest")
        self.git.side_effect = lambda argv, **kwargs: types.SimpleNamespace(
            stdout=b"" if argv[1] == "status" else b"0123456789abcdef0123456789abcdef01234567\n")
        self.args.manifest_not_after_unix = 2199.0
        self.refused("window is too short")

    def test_loader_rejects_duplicate_keys_at_top_level_and_inside_rows(self):
        manifest = self.write()
        raw = json.dumps(manifest)
        path = self.root / "duplicate.json"
        for changed in ('{"schema":"strata-acceptance-manifest/1",' + raw[1:],
                        raw.replace('"physical": 8.0', '"physical": 8.0, "physical": 8.0'),
                        raw.replace('"role": "config"', '"role": "config", "role": "config"')):
            path.write_text(changed)
            with self.subTest(changed=changed[:80]), self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                bench.load_acceptance_manifest(path)

    def test_loader_rejects_missing_extra_keys_in_every_container(self):
        manifest = self.write()
        for container in ((), ("floors_gib",), ("inputs", 0), ("size_bound", 0), ("unbound", 0)):
            for operation in ("missing", "extra"):
                changed = bench.copy.deepcopy(manifest)
                target = changed
                for key in container:
                    target = target[key]
                if operation == "missing":
                    del target[next(iter(target))]
                else:
                    target["unexpected"] = None
                with self.subTest(container=container, operation=operation), self.assertRaises(ValueError):
                    self.load_changed(changed)

    def test_loader_rejects_wrong_exact_types_without_coercion(self):
        manifest = self.write()
        changes = {("schema",): [1, None], ("created_unix",): [True, "1000", None],
            ("not_after_unix",): [False, "10000", None], ("source_head",): [40, None],
            ("output",): [1, None], ("workload",): [True, []], ("cohort",): [1, []],
            ("expected_context",): [True, 65536.0, "65536"], ("port",): [True, 18081.0, "18081"],
            ("total_seconds",): [True, 1200.0, "1200"], ("floors_gib",): [[], None],
            ("floors_gib", "physical"): [True, "8", None], ("floors_gib", "commit"): [False, "9", None],
            ("candidate_exe",): [1, []], ("protect_input",): [{}, "path"], ("protect_input", 0): [None, 1],
            ("inputs",): [{}, None], ("inputs", 0): [[], None], ("inputs", 0, "role"): [[], 1],
            ("inputs", 0, "path"): [None, 1], ("inputs", 0, "sha256"): [None, 1],
            ("inputs", 0, "bytes"): [True, 1.0, "1"], ("size_bound",): [{}, None],
            ("size_bound", 0, "bytes"): [True, 1.0, "1"],
            ("size_bound", 0, "mtime_ns"): [True, 1.0, "1"], ("unbound",): [{}, None],
            ("unbound", 0, "role"): [[], 1], ("unbound", 0, "reason"): [False, 1]}
        for keys, values in changes.items():
            for value in values:
                changed = bench.copy.deepcopy(manifest)
                target = changed
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.subTest(keys=keys, value=value), self.assertRaises(ValueError):
                    self.load_changed(changed)

    def test_loader_rejects_uppercase_or_wrong_length_digests(self):
        manifest = self.write()
        for key in ("source_head", "sha256"):
            for value in ("A" * (40 if key == "source_head" else 64), "a" * 39, "g" * 64):
                changed = bench.copy.deepcopy(manifest)
                (changed if key == "source_head" else changed["inputs"][0])[key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "digest"):
                    self.load_changed(changed)

    def test_loader_rejects_relative_paths_in_all_path_fields(self):
        manifest = self.write()
        for keys in (("output",), ("candidate_exe",), ("protect_input", 0),
                     ("inputs", 0, "path"), ("size_bound", 0, "path")):
            changed = bench.copy.deepcopy(manifest)
            target = changed
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = "relative/path"
            with self.subTest(keys=keys), self.assertRaisesRegex(ValueError, "absolute"):
                self.load_changed(changed)

    def test_loader_rejects_invalid_values_and_duplicate_resolved_inputs(self):
        manifest = self.write()
        for keys, value in ((("created_unix",), float("nan")), (("not_after_unix",), float("inf")),
                (("floors_gib", "physical"), 3.9), (("inputs", 0, "bytes"), -1),
                (("size_bound", 0, "mtime_ns"), -1), (("port",), 65536), (("total_seconds",), 899),
                (("expected_context",), 65537), (("cohort",), "frozen-control"), (("workload",), "unknown")):
            changed = bench.copy.deepcopy(manifest)
            target = changed
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            with self.subTest(keys=keys, value=value), self.assertRaises(ValueError):
                self.load_changed(changed)
        changed = bench.copy.deepcopy(manifest)
        changed["inputs"].append(dict(changed["inputs"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate resolved input"):
            self.load_changed(changed)

    def test_loader_reads_only_limit_plus_one_and_refuses_oversized_bytes(self):
        stream = io.BytesIO(b" " * (1024**2 + 2))
        read = Mock(wraps=stream.read)
        stream.read = read
        with patch.object(Path, "open", return_value=stream):
            with self.assertRaisesRegex(ValueError, "exceeds its evidence budget"):
                bench.load_acceptance_manifest(self.root / "oversized.json")
        read.assert_called_once_with(1024**2 + 1)

    def test_loader_reports_malformed_json_encoding_or_missing_file_as_value_error(self):
        path = self.root / "invalid.json"
        for data in (b"{", b"\xff", b"[]", b"null"):
            path.write_bytes(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                bench.load_acceptance_manifest(path)
        with self.assertRaises(ValueError):
            bench.load_acceptance_manifest(self.root / "missing.json")

    def test_short_or_failed_manifest_write_removes_its_partial_destination(self):
        original_open = Path.open
        before = set(self.root.iterdir())
        for failure in ("short", "write", "flush", "close"):
            writes = []
            def open_stream(path, mode="r", *args, **kwargs):
                stream = original_open(path, mode, *args, **kwargs)
                if mode != "xb":
                    return stream
                writes.append(path)
                class FailedStream:
                    def __enter__(inner): return inner
                    def __exit__(inner, *exc):
                        stream.close()
                        if failure == "close":
                            raise OSError("simulated close failure")
                    def write(inner, data):
                        if failure == "write":
                            stream.write(data[:10])
                            raise OSError("simulated write failure")
                        return stream.write(data[:-1] if failure == "short" else data)
                    def flush(inner):
                        if failure == "flush":
                            raise OSError("simulated flush failure")
                        stream.flush()
                return FailedStream()
            with self.subTest(failure=failure), patch.object(Path, "open", open_stream):
                with self.assertRaisesRegex(OSError, "Short artifact write|simulated .* failure"):
                    bench.run(self.args)
            self.assertFalse(self.args.write_acceptance_manifest.exists())
            self.assertFalse(self.args.output.exists())
            self.assertEqual(len(writes), 1)
            self.assertEqual(set(self.root.iterdir()), before)
        self.popen.assert_not_called()

    def test_exclusive_write_race_preserves_the_other_writers_destination(self):
        real_write = bench.bounded_write
        def race(path, *args, **kwargs):
            path.write_bytes(b"another writer's evidence")
            return real_write(path, *args, **kwargs)
        with patch.object(bench, "bounded_write", side_effect=race), self.assertRaises(FileExistsError):
            bench.run(self.args)
        self.assertEqual(self.args.write_acceptance_manifest.read_bytes(), b"another writer's evidence")
        self.assertFalse(self.args.output.exists())

    def test_failed_exclusive_write_preserves_replacement_created_after_close(self):
        original_open = Path.open
        destination = self.args.write_acceptance_manifest
        replacement = b"another writer's replacement after close"
        chronology = []

        def open_stream(path, mode="r", *args, **kwargs):
            stream = original_open(path, mode, *args, **kwargs)
            if mode not in ("wb", "xb"):
                return stream
            class FailedStream:
                def __enter__(inner): return inner
                def __exit__(inner, *exc):
                    stream.close()
                    self.assertTrue(stream.closed)
                    chronology.append("closed")
                    destination.unlink(missing_ok=True)
                    with original_open(destination, "xb") as other:
                        other.write(replacement)
                    chronology.append("replaced")
                def write(inner, data): return stream.write(data)
                def flush(inner):
                    chronology.append("flush failed")
                    raise OSError("simulated flush failure")
            return FailedStream()

        with patch.object(Path, "open", open_stream), self.assertRaisesRegex(OSError, "simulated flush failure"):
            bench.run(self.args)
        self.assertEqual(chronology, ["flush failed", "closed", "replaced"])
        self.assertTrue(destination.exists())
        self.assertEqual(destination.read_bytes(), replacement)
        self.assertFalse(self.args.output.exists())
        self.popen.assert_not_called()

    def test_exclusive_publication_race_after_close_preserves_other_writer(self):
        original_open = Path.open
        destination = self.args.write_acceptance_manifest
        replacement = b"another writer published before the final link"
        before = set(self.root.iterdir())

        def open_stream(path, mode="r", *args, **kwargs):
            stream = original_open(path, mode, *args, **kwargs)
            if mode != "xb":
                return stream
            class RacingStream:
                def __enter__(inner): return inner
                def __exit__(inner, *exc):
                    stream.close()
                    with original_open(destination, "xb") as other:
                        other.write(replacement)
                def write(inner, data): return stream.write(data)
                def flush(inner): stream.flush()
            return RacingStream()

        with patch.object(Path, "open", open_stream), self.assertRaises(FileExistsError):
            bench.run(self.args)
        self.assertEqual(destination.read_bytes(), replacement)
        self.assertEqual(set(self.root.iterdir()), before | {destination})
        self.assertFalse(self.args.output.exists())
        self.popen.assert_not_called()

    def test_exclusive_publication_failure_removes_staging_and_leaves_no_manifest(self):
        before = set(self.root.iterdir())
        with patch.object(bench.os, "link", side_effect=OSError("simulated publication failure")) as link:
            with self.assertRaisesRegex(OSError, "simulated publication failure"):
                bench.run(self.args)
        link.assert_called_once()
        self.assertFalse(self.args.write_acceptance_manifest.exists())
        self.assertFalse(self.args.output.exists())
        self.assertEqual(set(self.root.iterdir()), before)
        self.popen.assert_not_called()

    def test_exclusive_deadline_after_close_refuses_before_publication(self):
        before = set(self.root.iterdir())
        with patch.object(bench.time, "monotonic", side_effect=[0, 1]), patch.object(bench.os, "link") as link:
            with self.assertRaisesRegex(TimeoutError, "finalization exceeded"):
                bench.bounded_write(self.args.write_acceptance_manifest, b"{}", deadline=1, exclusive=True)
        link.assert_not_called()
        self.assertFalse(self.args.write_acceptance_manifest.exists())
        self.assertEqual(set(self.root.iterdir()), before)


class AcceptanceAdmissionTests(unittest.TestCase):
    """Real manifest writer and preflight through the contained CPU lifecycle."""

    def setUp(self):
        # Imports performed on synthetic trees must not accumulate their paths
        # across otherwise independent controller bootstrap envelopes.
        self.enterContext(patch.object(sys, "path", list(sys.path)))
        self.real_ready = bench.acceptance_ready
        real_preflight = bench.acceptance_preflight
        self.real_preflight = real_preflight
        self.directory_api = bench.WindowsDirectoryAPI()
        self.provisional_api = bench.WindowsProvisionalAPI()
        self.receipt_api = bench.WindowsReceiptAPI()
        self.engine_log_api = bench.WindowsEngineLogAPI()
        self.case = case = ContainedLifecycleTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.enterContext(patch.object(bench, "WindowsDirectoryAPI", return_value=self.directory_api))
        self.enterContext(patch.object(bench, "WindowsProvisionalAPI", return_value=self.provisional_api))
        self.enterContext(patch.object(bench, "WindowsReceiptAPI", return_value=self.receipt_api))
        self.enterContext(patch.object(bench, "WindowsEngineLogAPI", return_value=self.engine_log_api))
        self.root, self.args = case.root, case.args
        self.enterContext(patch.object(bench, "ROOT", self.root))
        self.enterContext(patch.object(bench, "WORKLOADS", self.root / "workloads"))
        self.enterContext(patch.object(bench.time, "time", side_effect=lambda: 1800000000 + case.now))
        self.head = "0123456789abcdef0123456789abcdef01234567"
        self.git_calls = []

        def git(argv, **kwargs):
            self.assertEqual(argv[0], "git")
            self.git_calls.append((argv, kwargs))
            return types.SimpleNamespace(stdout=(self.head + "\n").encode() if argv[1] == "rev-parse" else b"")

        self.enterContext(patch.object(bench.subprocess, "run", side_effect=git))
        self.before_preflight = None
        self.worker_args = None

        def preflight(args):
            self.worker_args = args
            case.events.append("admission")
            self.assertTrue(case.assigned)
            self.assertTrue(case.started)
            if self.before_preflight:
                self.before_preflight(args)
            from tools import prepare_unsloth_q2 as preparation
            snapshot = preparation.memory_snapshot
            samples = []
            def observed():
                if threading.current_thread() is case.worker_thread:
                    samples.append(True)
                return snapshot()
            try:
                with patch.object(preparation, "memory_snapshot", side_effect=observed):
                    return real_preflight(args)
            finally:
                self.preflight_samples = len(samples)
                self.preflight_output_exists = args.output.exists()

        self.enterContext(patch.object(bench, "acceptance_preflight", side_effect=preflight))
        self.saved = self.root / "saved.exe"
        self.shard, self.pack = self.root / "shard.gguf", self.root / "pack"
        self.pack_data = self.pack / "experts.bin"
        self.extra = [self.root / "launcher.bat", self.root / "other.txt"]
        tokenizer = self.root / "tokenizer"
        files = [self.saved, self.shard, self.pack_data, self.pack / "native_experts.txt", *self.extra,
                 *bench.acceptance_harness_paths(),
                 *[tokenizer / name for name in ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja")]]
        for path in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic input, never executed")
        stat = self.shard.stat()
        (self.pack / "source-verification.json").write_text(json.dumps({"shard.gguf": {
            "path": str(self.shard), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib.sha256(self.shard.read_bytes()).hexdigest()}}))
        cfg = json.loads(case.config.read_text())
        cfg.update(exe=str(self.saved), tokenizer=str(tokenizer),
                   args=["--pack", str(self.pack), "--native", str(self.shard), "--max-context", "131072"])
        case.config.write_text(json.dumps(cfg))
        self.args.acceptance_workload, self.args.cohort = "short", "saved-static"
        self.args.acceptance_physical_gib = self.args.acceptance_commit_gib = 4
        self.args.expected_context, self.args.port = 131072, 8081
        self.args.candidate_exe, self.args.protect_input = None, [str(path) for path in self.extra]
        self.args.acceptance_total_seconds = None
        self.args.write_acceptance_manifest = self.root / "manifest.json"
        self.args.manifest_not_after_unix = 1800010000.0
        self.assertEqual(bench.run(self.args), 0)
        self.assertEqual(case.events, [])
        self.assertEqual(case.samples, 0)
        self.assertFalse(case.output.exists())
        self.args.acceptance_manifest = self.args.write_acceptance_manifest
        self.args.write_acceptance_manifest = self.args.manifest_not_after_unix = None
        self.manifest = bench.load_acceptance_manifest(self.args.acceptance_manifest)
        self.digest = hashlib.sha256(self.args.acceptance_manifest.read_bytes()).hexdigest()
        self.git_calls.clear()
        self.directory_handles = set()
        handles_lock = threading.Lock()
        native_open, native_close = self.directory_api.open, self.directory_api.close
        def opened(*args, **kwargs):
            with handles_lock:
                handle = native_open(*args, **kwargs)
                self.directory_handles.add(handle)
                return handle
        def closed(handle):
            with handles_lock:
                native_close(handle)
                self.directory_handles.remove(handle)
        self.enterContext(patch.object(self.directory_api, "open", side_effect=opened))
        self.enterContext(patch.object(self.directory_api, "close", side_effect=closed))
        self.addCleanup(lambda: self.assertEqual(self.directory_handles, set(), "Unclosed directory authorities"))
        self.provisional_handles = set()
        provisional_open, provisional_close = self.provisional_api.open, self.provisional_api.close
        def opened_provisional(path):
            handle = provisional_open(path)
            self.provisional_handles.add(handle)
            return handle
        def closed_provisional(handle):
            provisional_close(handle)
            self.provisional_handles.remove(handle)
        self.enterContext(patch.object(self.provisional_api, "open", side_effect=opened_provisional))
        self.enterContext(patch.object(self.provisional_api, "close", side_effect=closed_provisional))
        self.addCleanup(lambda: self.assertEqual(self.provisional_handles, set(), "Unclosed provisional handles"))
        self.receipt_handles, self.receipt_opens, self.receipt_events = set(), [], []
        receipt_open, receipt_close = self.receipt_api.open, self.receipt_api.close
        def opened_receipt(path):
            handle = receipt_open(path)
            self.receipt_handles.add(handle)
            self.receipt_opens.append(handle)
            self.receipt_events.append(("open", handle))
            return handle
        def closed_receipt(handle):
            receipt_close(handle)
            self.receipt_handles.remove(handle)
            self.receipt_events.append(("close", handle))
        self.enterContext(patch.object(self.receipt_api, "open", side_effect=opened_receipt))
        self.enterContext(patch.object(self.receipt_api, "close", side_effect=closed_receipt))
        self.addCleanup(lambda: self.assertEqual(self.receipt_handles, set(), "Unclosed controller receipt handles"))
        self.engine_log_handles, self.engine_log_events = set(), []
        engine_open, engine_close = self.engine_log_api.open, self.engine_log_api.close
        engine_final = self.engine_log_api.open_final
        def opened_engine_log(path):
            handle = engine_open(path)
            self.engine_log_handles.add(handle)
            self.engine_log_events.append(("open", handle))
            return handle
        def opened_final_engine_log(path):
            handle = engine_final(path)
            self.engine_log_handles.add(handle)
            self.engine_log_events.append(("final", handle))
            return handle
        def closed_engine_log(handle):
            engine_close(handle)
            self.engine_log_handles.remove(handle)
            self.engine_log_events.append(("close", handle))
        self.enterContext(patch.object(self.engine_log_api, "open", side_effect=opened_engine_log))
        self.enterContext(patch.object(self.engine_log_api, "open_final", side_effect=opened_final_engine_log))
        self.enterContext(patch.object(self.engine_log_api, "close", side_effect=closed_engine_log))
        self.addCleanup(lambda: self.assertEqual(self.engine_log_handles, set(), "Unclosed worker engine-log handles"))

    def test_windows_atomic_birth_and_overlapping_handles_prevent_replacement_until_both_close(self):
        output, moved = self.root / "atomic-output", self.root / "moved-output"
        original_mkdir = Path.mkdir
        def no_mkdir(path, *args, **kwargs):
            self.assertNotEqual(path, output, "Birth authority must come from the native create operation")
            return original_mkdir(path, *args, **kwargs)
        with patch.object(Path, "mkdir", no_mkdir):
            worker = bench.AcceptanceOutputOwner(output, str(output), create=True)
        controller = None
        try:
            self.assertIsNotNone(worker.handle)
            self.assertTrue(output.is_dir())
            self.assertNotEqual(worker.identity[1], "0" * 32)
            with self.assertRaises(PermissionError):
                output.rename(moved)
            with self.assertRaises(PermissionError):
                output.rmdir()
            with self.assertRaises(FileExistsError):
                bench.AcceptanceOutputOwner(output, str(output), create=True)
            controller = bench.AcceptanceOutputOwner(output, str(output))
            self.assertEqual(controller.identity, worker.identity)
            self.assertNotEqual(controller.handle, worker.handle)
            worker.close()
            with self.assertRaises(PermissionError):
                output.rename(moved)
            with self.assertRaises(PermissionError):
                output.rmdir()
            controller.validate()
        finally:
            worker.close()
            if controller is not None:
                controller.close()
        output.rename(moved)
        self.assertTrue(moved.is_dir())
        self.assertFalse(output.exists())
        output.mkdir()
        self.assertTrue(output.is_dir())

    def test_worker_post_native_create_pre_identity_replacement_is_denied(self):
        original = self.directory_api.open
        moved, attempts = self.root / "post-create-owned", []
        foreign = self.root / "pre-claim-foreign"
        contents = self.foreign_artifacts(foreign)
        def open_directory(path, **kwargs):
            handle = original(path, **kwargs)
            if kwargs.get("create"):
                self.assertFalse((path / "provisional.json").exists())
                with self.assertRaises(PermissionError):
                    path.rename(moved)
                with self.assertRaises(PermissionError):
                    foreign.replace(path)
                attempts.append(handle)
            return handle
        with patch.object(self.directory_api, "open", side_effect=open_directory):
            result = self.case.invoke_selected()
        self.assertEqual(len(attempts), 1)
        self.assertFalse(moved.exists())
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertEqual({p.name: p.read_bytes() for p in foreign.iterdir()}, contents)
        self.assertIsNone(self.worker_args._output_owner.handle)

    def publication_deadline_refused(self, offset, *, acknowledgment=False):
        flush, wait = self.provisional_api.flush, bench.acceptance_wait_for_controller
        crossed = []
        def flushed(handle):
            result = flush(handle)
            if not acknowledgment and not crossed:
                self.case.now = self.worker_args._acceptance_deadline + offset
                crossed.append(self.case.now - self.worker_args._acceptance_deadline)
            return result
        def acquired(*args):
            wait(*args)
            if acknowledgment:
                self.case.now = self.worker_args._acceptance_deadline + offset
                crossed.append(self.case.now - self.worker_args._acceptance_deadline)
        with patch.object(self.provisional_api, "flush", side_effect=flushed), \
                patch.object(bench, "acceptance_wait_for_controller", side_effect=acquired), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(len(crossed), 1)
        self.assertAlmostEqual(crossed[0], offset)
        self.assertNotIn("server", self.case.events)
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        self.assertIsNone(self.worker_args._output_owner.handle)

    def test_claim_publication_at_deadline_equality_never_starts_server(self):
        self.publication_deadline_refused(0.0)

    def test_claim_publication_one_ms_after_deadline_never_starts_server(self):
        self.publication_deadline_refused(0.001)

    def test_acknowledgment_at_deadline_equality_never_starts_server(self):
        self.publication_deadline_refused(0.0, acknowledgment=True)

    def test_acknowledgment_one_ms_after_deadline_never_starts_server(self):
        self.publication_deadline_refused(0.001, acknowledgment=True)

    def post_second_proof_attack(self, *, populated=False, symlink=False):
        foreign, moved = self.root / "post-proof-foreign", self.root / "post-proof-owned"
        contents = self.foreign_artifacts(foreign) if populated else {}
        if not populated:
            foreign.mkdir()
        claimed = bench.acceptance_claimed_output
        proofs = []
        def prove(*args, **kwargs):
            record = claimed(*args, **kwargs)
            if self.case.closed:
                proofs.append(True)
                if len(proofs) == 2:
                    self.assertIsNotNone(kwargs["owner"].handle)
                    with self.assertRaises(PermissionError):
                        self.case.output.rename(moved)
                    if symlink:
                        with self.assertRaises(FileExistsError):
                            self.case.output.symlink_to(foreign, target_is_directory=True)
                    else:
                        with self.assertRaises(PermissionError):
                            foreign.replace(self.case.output)
            return record
        with patch.object(bench, "acceptance_claimed_output", side_effect=prove):
            result = self.case.invoke_selected()
        self.assertEqual(proofs, [True, True])
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertFalse(moved.exists())
        self.assertFalse(self.case.output.is_symlink())
        self.assertEqual({p.name: p.read_bytes() for p in foreign.iterdir()}, contents)
        self.assertTrue((self.case.output / "lifecycle.json").is_file())

    def test_post_second_proof_empty_directory_replacement_is_denied(self):
        self.post_second_proof_attack()

    def test_post_second_proof_populated_directory_replacement_preserves_foreign_bytes(self):
        self.post_second_proof_attack(populated=True)

    def test_post_second_proof_symlink_replacement_preserves_foreign_bytes(self):
        self.post_second_proof_attack(populated=True, symlink=True)

    def controller_name_refused(self, name):
        contents = {}
        def protection(args, worker_record):
            (args.output / name).write_bytes(b"pre-existing foreign controller bytes")
            contents.update({p.name: p.read_bytes() for p in args.output.iterdir()})
            return True
        with patch.object(bench, "acceptance_protection", side_effect=protection), \
                self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(caught.exception.record["finalization_error"], "FileExistsError: Controller artifact already exists.")
        self.assertFalse(caught.exception.record["finalization"])
        self.assertEqual({p.name: p.read_bytes() for p in self.case.output.iterdir()}, contents)

    def test_preexisting_controller_log_refuses_without_replacement(self):
        self.controller_name_refused("controller.log")

    def test_preexisting_former_lifecycle_staging_name_is_foreign_and_refuses_untouched(self):
        # The controller no longer stages lifecycle.json, so this name is only a foreign entry.
        contents = {}
        def protection(args, worker_record):
            (args.output / "lifecycle.json.tmp").write_bytes(b"pre-existing foreign controller bytes")
            contents.update({p.name: p.read_bytes() for p in args.output.iterdir()})
            return True
        with patch.object(bench, "acceptance_protection", side_effect=protection), \
                self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(caught.exception.record["finalization_error"],
                         "RuntimeError: Unexpected or oversized retained artifacts.")
        self.assertFalse(caught.exception.record["finalization"])
        after = {p.name: p.read_bytes() for p in self.case.output.iterdir()}
        self.assertEqual({name: after[name] for name in contents}, contents)
        self.assertEqual(set(after) - set(contents), {"controller.log", "lifecycle.json"})

    def test_preexisting_lifecycle_destination_refuses_without_replacement(self):
        self.controller_name_refused("lifecycle.json")

    def test_controller_acquisition_failure_never_acknowledges_or_starts_server(self):
        original = self.directory_api.open
        failures = []
        def open_directory(*args, **kwargs):
            if threading.current_thread() is not self.case.worker_thread and self.case.output.exists():
                failures.append(True)
                raise PermissionError("synthetic controller acquisition failure")
            return original(*args, **kwargs)
        with patch.object(self.directory_api, "open", side_effect=open_directory):
            result, writes = self.failed_with_controller_writes()
        self.assertTrue(failures)
        self.assertEqual(writes, [])
        self.assertNotIn("server", self.case.events)
        self.assertIn("synthetic controller acquisition failure", result["error"])
        self.assertFalse((self.case.output / "controller-ack.json").exists())
        self.assertIsNone(self.worker_args._output_owner.handle)

    def test_controller_handle_identity_mismatch_never_acknowledges_or_starts_server(self):
        original, calls = self.directory_api.identity, []
        def identity(handle):
            result = original(handle)
            if threading.current_thread() is not self.case.worker_thread:
                calls.append(handle)
                if len(calls) == 1:
                    return result[0] + 1, result[1]
            return result
        with patch.object(self.directory_api, "identity", side_effect=identity):
            result, writes = self.failed_with_controller_writes()
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(writes, [])
        self.assertNotIn("server", self.case.events)
        self.assertIn("Worker output identity differs.", result["error"])
        self.assertFalse((self.case.output / "controller-ack.json").exists())

    def test_partial_claim_is_retried_without_ack_until_strict_parse_succeeds(self):
        # The controller reads only through its retained receipt handle, never by path.
        original, partial = self.receipt_api.read, []
        def read(handle, limit):
            if not partial:
                partial.append(handle)
                self.assertNotIn("server", self.case.events)
                self.assertFalse((self.case.output / "controller-ack.json").exists())
                return b'{"output_claim":'
            return original(handle, limit)
        with patch.object(self.receipt_api, "read", side_effect=read):
            result = self.case.invoke_selected()
        self.assertEqual(len(partial), 1)
        self.assertEqual(self.receipt_opens[0], partial[0])
        self.assertGreater(len(self.receipt_opens), 1, "A partial claim must be retried with a fresh handle")
        self.assertTrue(bench.acceptance_final_predicate(result))

    def test_malformed_acquisition_claim_is_never_acknowledged(self):
        reads = []
        def read(handle, limit):
            reads.append(True)
            return b'{"output_claim":false}'
        with patch.object(self.receipt_api, "read", side_effect=read):
            result, writes = self.failed_with_controller_writes()
        self.assertTrue(reads)
        self.assertEqual(writes, [])
        self.assertNotIn("server", self.case.events)
        # Authentication precedes every field, including the claim.
        self.assertIn("Worker receipt authentication is missing or malformed.", result["error"])
        self.assertFalse((self.case.output / "controller-ack.json").exists())

    def test_acknowledgment_mismatch_refuses_before_server(self):
        create = bench.bounded_create
        def changed(path, data, *args, **kwargs):
            if path.name == "controller-ack.json":
                data = b'{"acquired_sha256":"' + b"0" * 64 + b'"}'
            return create(path, data, *args, **kwargs)
        with patch.object(bench, "bounded_create", side_effect=changed), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertNotIn("server", self.case.events)
        self.assertEqual(self.worker_args._acceptance_admission["reason"], "Controller acknowledgment differs.")
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))

    def test_acknowledgment_close_failure_is_terminal_and_keeps_the_held_receipt(self):
        # The token is visible once its exclusive create has written it, so a later
        # failure cannot unpublish it: it stops the run without any retry instead.
        record = self.acknowledgment_failure_after_publication()
        self.assertIn("OSError: synthetic acknowledgment failure after publication", record["error"])
        self.assertTrue((self.case.output / "controller-ack.json").is_file())

    def test_controller_publish_race_preserves_foreign_destination(self):
        create, data, planted = bench.bounded_create, b"foreign lifecycle at publication", []
        def publish(path, *args, **kwargs):
            if path.name == "lifecycle.json":
                path.write_bytes(data)
                planted.append(True)
            return create(path, *args, **kwargs)
        with patch.object(bench, "bounded_create", side_effect=publish), \
                self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(planted, [True])
        self.assertFalse(caught.exception.record["finalization"])
        self.assertTrue(caught.exception.record["finalization_error"].startswith("FileExistsError: "))
        self.assertEqual((self.case.output / "lifecycle.json").read_bytes(), data)
        self.assertFalse((self.case.output / "lifecycle.json.tmp").exists())

    def test_controller_readback_failure_refuses_with_closed_authorities(self):
        original, reads = Path.open, []
        def open_file(path, *args, **kwargs):
            if path.name == "lifecycle.json" and args == ("rb",):
                reads.append(True)
                return io.BytesIO(b"corrupted receipt")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", open_file), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(reads, [True])
        self.assertEqual(caught.exception.record["finalization_error"], "OSError: Final receipt readback mismatch.")
        self.assertFalse(caught.exception.record["finalization"])
        self.assertIsNone(self.worker_args._output_owner.handle)

    def output_close_failure(self, *, worker):
        close, failures = bench.AcceptanceOutputOwner.close, []
        def failed(owner):
            active = owner.handle is not None
            close(owner)
            if active and (threading.current_thread() is self.case.worker_thread) == worker:
                if worker or self.case.closed:
                    failures.append(True)
                    raise OSError("synthetic output authority close failure")
        stderr = io.StringIO()
        with patch.object(bench.AcceptanceOutputOwner, "close", failed), \
                contextlib.redirect_stderr(stderr), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(failures, [True])
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        if worker:
            self.assertEqual(caught.exception.record["worker_exit"], 1)
            self.assertIn("Worker output authority close failed (OSError).", stderr.getvalue())
        else:
            self.assertFalse(caught.exception.record["finalization"])
            self.assertIn("Controller output authority close failed", caught.exception.record["cleanup_error"])

    def test_worker_authority_close_failure_cannot_pass(self):
        self.output_close_failure(worker=True)

    def test_controller_authority_close_failure_cannot_pass(self):
        self.output_close_failure(worker=False)

    def test_non_windows_manifest_route_refuses_before_host_worker_or_native_startup(self):
        args = argparse.Namespace(**vars(self.args), _acceptance_deadline=990)
        samples = self.case.samples
        with patch.object(bench.os, "name", "posix"), \
                patch.object(bench, "WindowsDirectoryAPI") as directory, \
                patch.object(bench, "WindowsProcessAPI") as native, \
                patch.object(bench.subprocess, "Popen") as spawn, \
                patch.object(bench, "load_acceptance_manifest") as load:
            with self.assertRaises(bench.AcceptanceFailure) as caught:
                self.case.invoke()
            with self.assertRaisesRegex(ValueError, "^Manifest containment requires Windows directory authority\\.$"):
                self.real_preflight(args)
        self.assertEqual(caught.exception.record["admission"], {
            "status": "refused", "reason": "Manifest containment requires Windows directory authority."})
        self.assertEqual(args._acceptance_admission["reason"], "Manifest containment requires Windows directory authority.")
        self.assertEqual(self.case.samples, samples)
        self.assertFalse(self.case.output.exists())
        directory.assert_not_called()
        native.assert_not_called()
        spawn.assert_not_called()
        load.assert_not_called()

    def test_resolved_argument_alias_uses_the_canonical_manifest_object_through_finalization(self):
        alias = self.root / "alias-component"
        alias.mkdir()
        self.args.output = alias / ".." / "output"
        result = self.case.invoke_selected()
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertEqual(self.args.output, Path(self.manifest["output"]))
        self.assertEqual(self.worker_args._output_owner.path, Path(self.manifest["output"]))
        self.assertIn("server", self.case.events)

    def test_no_output_filesystem_or_native_finalization_follows_controller_close(self):
        close, native_open, open_file = bench.AcceptanceOutputOwner.close, self.directory_api.open, Path.open
        closed = []
        def release(owner):
            close(owner)
            if self.case.closed:
                closed.append(True)
        def directory(*args, **kwargs):
            self.assertFalse(closed, "Native finalization after controller authority close")
            return native_open(*args, **kwargs)
        def artifact(path, *args, **kwargs):
            self.assertFalse(closed, "Filesystem finalization after controller authority close")
            return open_file(path, *args, **kwargs)
        with patch.object(bench.AcceptanceOutputOwner, "close", release), \
                patch.object(self.directory_api, "open", side_effect=directory), patch.object(Path, "open", artifact):
            result = self.case.invoke_selected()
        self.assertEqual(closed, [True])
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertEqual(self.directory_handles, set())

    def test_controller_primary_finalization_failure_and_close_failure_are_separate(self):
        close = bench.AcceptanceOutputOwner.close
        def failed(owner):
            close(owner)
            if self.case.closed:
                raise OSError("synthetic secondary close failure")
        with patch.object(bench.AcceptanceOutputOwner, "close", failed), \
                patch.object(bench, "acceptance_protection", side_effect=ValueError("synthetic primary protection failure")), \
                self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(caught.exception.record["finalization_error"], "ValueError: synthetic primary protection failure")
        self.assertIn("synthetic secondary close failure", caught.exception.record["cleanup_error"])
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        self.assertEqual(self.directory_handles, set())

    def test_private_handshake_artifacts_are_included_in_aggregate_budget(self):
        create, budgets = bench.bounded_create, []
        def publish(path, *args, **kwargs):
            result = create(path, *args, **kwargs)
            if path.name == "lifecycle.json":
                paths = list(self.case.output.iterdir())
                self.assertIn(self.case.output / "controller-ack.json", paths)
                without_ack = sum(p.stat().st_size for p in paths if p.name != "controller-ack.json")
                bench.ACCEPTANCE_ARTIFACT_LIMIT = without_ack
                budgets.append(without_ack)
            return result
        with patch.object(bench, "ACCEPTANCE_ARTIFACT_LIMIT", bench.ACCEPTANCE_ARTIFACT_LIMIT), \
                patch.object(bench, "bounded_create", side_effect=publish), \
                self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(len(budgets), 1)
        self.assertGreater(sum(p.stat().st_size for p in self.case.output.iterdir()), budgets[0])
        self.assertEqual(caught.exception.record["finalization_error"], "RuntimeError: Total retained artifact budget exceeded.")
        self.assertFalse(caught.exception.record["finalization"])

    def test_owned_request_failure_retains_bounded_primary_failure_receipt(self):
        with patch.object(bench, "acceptance_workload_tests", side_effect=OSError("synthetic request failure")), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        result = caught.exception.record
        self.assertEqual(result["worker_exit"], 1)
        self.assertTrue(result["finalization"])
        worker = json.loads((self.case.output / "provisional.json").read_bytes())
        self.assertEqual(worker["error"], "synthetic request failure")
        self.assertEqual(worker["status"], "provisional_failed")
        self.assertIn("server-stop", self.case.events)
        self.assertLess(sum(p.stat().st_size for p in self.case.output.iterdir()), bench.ACCEPTANCE_ARTIFACT_LIMIT)

    def test_failed_controller_native_close_retains_authority_for_bounded_caller_cleanup(self):
        close, failures = self.directory_api.close, []
        def fail(handle):
            if self.case.closed and not failures:
                failures.append(True)
                raise OSError("synthetic retained controller close failure")
            return close(handle)
        with patch.object(self.directory_api, "close", side_effect=fail), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        owner = caught.exception.output_owner
        self.assertEqual(failures, [True])
        self.assertIn(owner.handle, self.directory_handles)
        self.assertIn("synthetic retained controller close failure", caught.exception.record["cleanup_error"])
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        try:
            with self.assertRaises(PermissionError):
                self.case.output.rename(self.root / "unclosed-controller-output")
        finally:
            owner.close()
        self.assertEqual(self.directory_handles, set())

    def test_failed_worker_native_close_is_reported_and_cannot_pass(self):
        close, failures = self.directory_api.close, []
        def fail(handle):
            if threading.current_thread() is self.case.worker_thread and not failures:
                failures.append(True)
                raise OSError("synthetic retained worker close failure")
            return close(handle)
        stderr = io.StringIO()
        with patch.object(self.directory_api, "close", side_effect=fail), contextlib.redirect_stderr(stderr), \
                self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        owner = self.worker_args._output_owner
        try:
            self.assertEqual(failures, [True])
            self.assertIn(owner.handle, self.directory_handles)
            self.assertEqual(caught.exception.record["worker_exit"], 1)
            self.assertIn("Worker output authority close failed (OSError).", stderr.getvalue())
            self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        finally:
            owner.close()  # A real worker's exit closes this process-local handle; the CPU fake must do so.
        self.assertEqual(self.directory_handles, set())

    def test_partial_claim_close_failure_is_not_retried_as_unacknowledged_success(self):
        close, failures = self.directory_api.close, []
        def fail(handle):
            if threading.current_thread() is not self.case.worker_thread and receipt_read.called and not failures:
                failures.append(True)
                raise OSError("synthetic partial acquisition close failure")
            return close(handle)
        with patch.object(self.receipt_api, "read", return_value=b'{"output_claim":') as receipt_read, \
                patch.object(self.directory_api, "close", side_effect=fail):
            result, writes = self.failed_with_controller_writes()
        self.assertEqual(failures, [True])
        receipt_read.assert_called_once()
        self.assertTrue(result["error"].startswith("JSONDecodeError:"))
        self.assertIn("synthetic partial acquisition close failure", result["cleanup_error"])
        self.assertEqual(writes, [])
        self.assertNotIn("server", self.case.events)
        self.assertFalse((self.case.output / "controller-ack.json").exists())
        self.assertEqual(self.directory_handles, set())

    def test_missing_receipt_close_failure_is_not_retried_as_unacknowledged_success(self):
        ready, owners, failures, controller_writes = threading.Event(), [], [], []
        acquire, wait = bench.acceptance_acquire_output, bench.acceptance_wait_for_controller
        close, create = self.directory_api.close, bench.bounded_create
        def worker_wait(*args):
            ready.set()  # The worker has published and is waiting for acknowledgment.
            return wait(*args)
        def acquired(*args):
            self.assertTrue(ready.wait(5), "Worker did not publish its initial receipt")
            return acquire(*args)
        def missing(owner):
            self.assertIn(owner.handle, self.directory_handles)
            owners.append(owner)
            raise FileNotFoundError("synthetic missing receipt after directory acquisition")
        def fail(handle):
            if threading.current_thread() is not self.case.worker_thread and not failures:
                self.assertEqual(len(owners), 1)
                self.assertEqual(handle, owners[0].handle)
                failures.append(True)
                raise OSError("synthetic missing receipt acquisition close failure")
            return close(handle)
        def created(path, *args, **kwargs):
            if threading.current_thread() is not self.case.worker_thread and path.parent == self.case.output:
                controller_writes.append(path)
            return create(path, *args, **kwargs)
        with patch.object(bench, "acceptance_wait_for_controller", side_effect=worker_wait), \
                patch.object(bench, "acceptance_acquire_output", side_effect=acquired) as acquisition, \
                patch.object(bench.AcceptanceOutputOwner, "open_receipt", missing), \
                patch.object(self.receipt_api, "read", wraps=self.receipt_api.read) as receipt_read, \
                patch.object(self.directory_api, "close", side_effect=fail), \
                patch.object(bench, "bounded_create", side_effect=created), \
                patch.object(bench, "acceptance_publish_acknowledgment", wraps=bench.acceptance_publish_acknowledgment) as ack:
            result, writes = self.failed_with_controller_writes()
        self.assertEqual(failures, [True])
        acquisition.assert_called_once()
        receipt_read.assert_not_called()
        ack.assert_not_called()
        self.assertEqual(result["error"], "FileNotFoundError: synthetic missing receipt after directory acquisition")
        self.assertEqual(result["cleanup_error"], "OSError: synthetic missing receipt acquisition close failure")
        self.assertEqual(writes, [])
        self.assertEqual(controller_writes, [])
        self.assertNotIn("server", self.case.events)
        self.assertEqual(self.case.requests, [])
        for name in ("controller-ack.json", "controller.log", "lifecycle.json", "lifecycle.json.tmp", "server.log", "engine.log"):
            self.assertFalse((self.case.output / name).exists(), name)
        self.assertFalse(result["finalization"])
        self.assertFalse(bench.acceptance_final_predicate(result))
        self.assertIs(result["owned_cleanup"], True)
        self.assertEqual(result["owned_accounting"]["job_active"], 0)
        self.assertEqual(result["owned_accounting"]["wait_result"], 0)
        self.assertTrue(self.case.closed)
        self.assertFalse(self.case.worker_thread.is_alive())
        self.assertFalse(hasattr(self.case, "child_error"))
        self.assertIn("cancel-job", self.case.events)
        self.assertIn("close-job", self.case.events)
        self.assertIn("close-process", self.case.events)
        owner, = owners
        self.assertIsNone(owner.handle)
        self.assertIsNone(owner.receipt)
        self.assertFalse(owner.acknowledged)
        self.assertIsNone(self.worker_args._output_owner.handle)
        self.assertIsNone(self.worker_args._provisional_file.handle)
        self.assertIsNone(self.worker_args._engine_log)
        self.assertEqual(self.directory_handles, set())
        self.assertEqual(self.receipt_handles, set())
        self.assertEqual(self.provisional_handles, set())
        self.assertEqual(self.engine_log_handles, set())

    def test_worker_identity_failure_and_failed_acquisition_close_keep_primary_reason(self):
        close, identity, calls, failures = self.directory_api.close, self.directory_api.identity, [], []
        def identify(handle):
            if threading.current_thread() is self.case.worker_thread:
                calls.append(handle)
                if len(calls) == 2:
                    raise ValueError("synthetic primary birth identity failure")
            return identity(handle)
        def fail(handle):
            if threading.current_thread() is self.case.worker_thread and not failures:
                failures.append(True)
                raise OSError("synthetic birth close failure")
            return close(handle)
        with patch.object(self.directory_api, "identity", side_effect=identify), \
                patch.object(self.directory_api, "close", side_effect=fail):
            result, writes = self.failed_with_controller_writes()
        self.assertEqual(failures, [True])
        self.assertEqual(self.worker_args._acceptance_admission["reason"], "synthetic primary birth identity failure")
        self.assertEqual(writes, [])
        self.assertNotIn("server", self.case.events)
        self.assertFalse(bench.acceptance_final_predicate(result))
        self.assertIsNone(self.worker_args._output_owner.handle)
        self.assertEqual(self.directory_handles, set())

    def test_final_handle_path_identity_failure_refuses_without_foreign_writes(self):
        native_path, failures = self.directory_api.path, []
        foreign = self.root / "final-identity-foreign"
        contents = self.foreign_artifacts(foreign)
        def identity(handle):
            if self.case.closed:
                failures.append(True)
                return str(foreign)
            return native_path(handle)
        with patch.object(self.directory_api, "path", side_effect=identity):
            result, writes = self.failed_with_controller_writes()
        self.assertTrue(failures)
        self.assertEqual(writes, [])
        self.assertEqual(result["finalization_error"], "ValueError: Worker output identity differs.")
        self.assertEqual({p.name: p.read_bytes() for p in foreign.iterdir()}, contents)

    def test_exited_worker_claim_is_never_acquired_or_acknowledged(self):
        capture = self.case.native.capture
        def exited(pipe):
            result = capture(pipe)
            self.case.worker_thread.join(5)
            self.assertFalse(self.case.worker_thread.is_alive())
            return result[0], True
        with patch.object(self.case.native, "capture", side_effect=exited), \
                patch.object(bench, "acceptance_wait_for_controller", side_effect=ValueError("synthetic worker claim failure")), \
                patch.object(bench, "acceptance_acquire_output", wraps=bench.acceptance_acquire_output) as acquire, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        acquire.assert_not_called()
        self.assertNotIn("server", self.case.events)
        self.assertEqual(caught.exception.record["worker_exit"], 1)
        self.assertFalse(caught.exception.record["finalization"])
        self.assertFalse((self.case.output / "controller-ack.json").exists())
        self.assertIsNone(self.worker_args._output_owner.handle)

    def changed_manifest(self, change):
        change(self.manifest)
        self.args.acceptance_manifest.write_text(json.dumps(self.manifest))
        self.digest = hashlib.sha256(self.args.acceptance_manifest.read_bytes()).hexdigest()

    def controller_refused(self, reason):
        samples = self.case.samples
        with self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke()
        self.assertEqual(caught.exception.record["admission"], {
            "manifest_sha256": self.digest, "status": "refused", "reason": reason})
        self.assertEqual(caught.exception.record["error"], "ValueError: " + reason)
        self.assertNotIn("create", self.case.events)
        self.assertNotIn("server", self.case.events)
        self.assertIsNone(caught.exception.owner)
        self.assertEqual(self.case.samples, samples)
        return caught.exception.record

    def worker_refused(self, reason, *, consumed_digest=None):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke()
        self.assertEqual(caught.exception.record["admission"], {
            "manifest_sha256": self.digest, "status": "admitted"})
        self.assertIn("create", self.case.events, caught.exception.record)
        self.assertIn("admission", self.case.events)
        self.assertNotIn("server", self.case.events)
        self.assertEqual(self.worker_args._acceptance_admission, {
            "manifest_sha256": self.digest if consumed_digest is None else consumed_digest,
            "status": "refused", "reason": reason})
        self.assertIn("Acceptance admission refused: " + reason + "\n", stderr.getvalue())
        self.assertFalse((self.case.output / "provisional.json").exists())
        self.assertEqual(self.case.code, 1)
        return caught.exception.record

    def test_green_real_preflight_and_manifest_order_survive_public_containment(self):
        # The writer's extended inventory owns protection, including its supplied order.
        self.changed_manifest(lambda value: value["inputs"].insert(3, value["inputs"].pop(2)))
        result = self.case.invoke_selected()
        self.assertTrue(bench.acceptance_final_predicate(result))
        worker = json.loads((self.case.output / "provisional.json").read_text())
        expected = {"manifest_sha256": self.digest, "status": "admitted"}
        self.assertEqual(result["admission"], expected)
        self.assertEqual(worker["admission"], expected)
        identities = [{key: row[key] for key in ("path", "bytes", "sha256")} for row in self.manifest["inputs"]]
        self.assertEqual(worker["protection"]["before"], identities)
        self.assertEqual(worker["protection"]["after"], identities)
        self.assertTrue(worker["protection"]["unchanged"])
        self.assertEqual(len(worker["acceptance_cases"]), 11)
        self.assertEqual([row["path"] for row in identities][-2:], [str(path) for path in self.extra])
        head = next(call for call in self.git_calls if call[0] == ["git", "rev-parse", "HEAD"])
        self.assertEqual(head[1]["cwd"], self.root)
        self.assertTrue(head[1]["check"])
        self.assertTrue(head[1]["capture_output"])
        self.assertGreater(head[1]["timeout"], 0)
        self.assertLessEqual(head[1]["timeout"], 30)

    def test_valid_manifest_replacement_after_controller_admission_refuses_before_server(self):
        replacement = bench.copy.deepcopy(self.manifest)
        replacement["created_unix"] += 1
        replacement_bytes = (json.dumps(replacement, indent=2) + "\n").encode()
        replacement_digest = hashlib.sha256(replacement_bytes).hexdigest()
        self.assertNotEqual(self.digest, replacement_digest)
        self.assertEqual({key for key in replacement if replacement[key] != self.manifest[key]}, {"created_unix"})

        def replace(args):
            Path(args.acceptance_manifest).write_bytes(replacement_bytes)
            self.assertEqual(bench.load_acceptance_manifest(args.acceptance_manifest), replacement)

        self.before_preflight = replace
        with patch.object(bench, "file_identity", wraps=bench.file_identity) as identity:
            result = self.worker_refused("Controller manifest digest differs.", consumed_digest=replacement_digest)
        identity.assert_not_called()
        self.assertEqual(result["admission"], {"status": "admitted", "manifest_sha256": self.digest})
        self.assertFalse(bench.acceptance_final_predicate(result))
        self.assertEqual(self.git_calls, [])
        self.assertEqual(self.preflight_samples, 0)
        self.assertFalse(self.preflight_output_exists)

    def test_controller_digest_transport_overrides_public_private_arguments(self):
        self.args._admitted_manifest_sha256 = "f" * 64
        result = self.case.invoke_selected()
        self.assertTrue(bench.acceptance_final_predicate(result))
        settings = json.loads(self.case.bootstrap_settings)
        self.assertEqual(settings["admitted_manifest_sha256"], self.digest)
        self.assertEqual(self.worker_args._admitted_manifest_sha256, self.digest)

    def test_missing_controller_digest_refuses_before_identity_git_memory_or_output(self):
        def remove(args):
            if hasattr(args, "_admitted_manifest_sha256"):
                delattr(args, "_admitted_manifest_sha256")
        self.before_preflight = remove
        with patch.object(bench, "file_identity", wraps=bench.file_identity) as identity:
            self.worker_refused("Controller manifest digest is required.")
        identity.assert_not_called()
        self.assertEqual(self.git_calls, [])
        self.assertEqual(self.preflight_samples, 0)
        self.assertFalse(self.preflight_output_exists)

    def test_missing_bootstrap_digest_cannot_fall_back_to_public_private_value(self):
        self.args._admitted_manifest_sha256 = self.digest
        bootstrap = bench.worker_bootstrap
        def missing(stream, worker, **kwargs):
            settings = json.loads(kwargs["settings"])
            settings.pop("admitted_manifest_sha256")
            kwargs["settings"] = json.dumps(settings)
            return bootstrap(stream, worker, **kwargs)
        with patch.object(bench, "worker_bootstrap", side_effect=missing), \
                patch.object(bench, "file_identity", wraps=bench.file_identity) as identity:
            self.worker_refused("Controller manifest digest is required.")
        identity.assert_not_called()
        self.assertEqual(self.git_calls, [])
        self.assertEqual(self.preflight_samples, 0)
        self.assertFalse(self.preflight_output_exists)

    def test_malformed_controller_digest_refuses_before_identity_git_memory_or_output(self):
        for digest in (False, 64, "", "A" * 64, "a" * 63, "a" * 65, "g" * 64):
            args = argparse.Namespace(**vars(self.args), _acceptance_deadline=990,
                                      _admitted_manifest_sha256=digest)
            samples = self.case.samples
            with self.subTest(digest=digest), patch.object(bench, "file_identity") as identity, \
                    self.assertRaisesRegex(ValueError, "^Invalid controller manifest digest\\.$"):
                self.real_preflight(args)
            identity.assert_not_called()
            self.assertEqual(args._acceptance_admission, {"status": "refused", "manifest_sha256": self.digest,
                                                         "reason": "Invalid controller manifest digest."})
            self.assertEqual(self.git_calls, [])
            self.assertEqual(self.case.samples, samples)
            self.assertFalse(args.output.exists())

    def test_late_foreign_output_directory_refuses_before_server_and_preserves_marker(self):
        checks = []
        marker = self.case.output / "foreign.txt"
        def idle(port):
            checks.append(port)
            if len(checks) == 2:
                self.case.output.mkdir()
                marker.write_bytes(b"foreign output must survive")
        with patch.object(bench, "assert_idle", side_effect=idle):
            self.worker_refused("Output directory already exists; choose a new directory.")
        self.assertEqual(checks, [8081, 8081])
        self.assertEqual(marker.read_bytes(), b"foreign output must survive")
        self.assertFalse(self.preflight_output_exists)
        self.assertNotIn("server-stop", self.case.events)

    def foreign_artifacts(self, target):
        target.mkdir()
        contents = {name: ("foreign bytes: " + name).encode() for name in (
            "controller.log", "lifecycle.json", "lifecycle.json.tmp", "provisional.json")}
        for name, data in contents.items():
            (target / name).write_bytes(data)
        return contents

    def failed_with_controller_writes(self):
        writes = []
        write, mkdir, replace = bench.bounded_write, Path.mkdir, Path.replace
        def bounded(path, *args, **kwargs):
            if self.case.closed:
                writes.append(("write", path))
            return write(path, *args, **kwargs)
        def directory(path, *args, **kwargs):
            if self.case.closed:
                writes.append(("mkdir", path))
            return mkdir(path, *args, **kwargs)
        def publish(path, *args, **kwargs):
            if self.case.closed:
                writes.append(("replace", path))
            return replace(path, *args, **kwargs)
        with patch.object(bench, "bounded_write", side_effect=bounded), \
                patch.object(Path, "mkdir", directory), patch.object(Path, "replace", publish), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke()
        self.assertFalse(caught.exception.record["finalization"])
        self.assertTrue(self.case.closed)
        return caught.exception.record, writes

    def assert_foreign_artifacts(self, target, contents, writes):
        self.assertNotIn("server", self.case.events)
        self.assertEqual(writes, [])
        self.assertEqual({path.name: path.read_bytes() for path in target.iterdir()}, contents)

    def test_final_idle_foreign_artifacts_are_never_adopted_by_controller(self):
        checks, contents = [], {}
        def idle(port):
            checks.append(port)
            if len(checks) == 2:
                contents.update(self.foreign_artifacts(self.case.output))
        with patch.object(bench, "assert_idle", side_effect=idle):
            result, writes = self.failed_with_controller_writes()
        self.assertEqual(checks, [8081, 8081])
        self.assert_foreign_artifacts(self.case.output, contents, writes)
        self.assertEqual(result["worker_exit"], 1)

    def test_final_idle_foreign_symlink_and_target_artifacts_survive_controller(self):
        target = self.root / "foreign-target"
        contents = self.foreign_artifacts(target)
        checks, links = [], []
        def idle(port):
            checks.append(port)
            if len(checks) == 2:
                self.case.output.symlink_to(target, target_is_directory=True)
                links.append(self.case.output.readlink())
        with patch.object(bench, "assert_idle", side_effect=idle):
            _, writes = self.failed_with_controller_writes()
        self.assertEqual(checks, [8081, 8081])
        self.assertTrue(self.case.output.is_symlink())
        self.assertEqual([self.case.output.readlink()], links)
        self.assertEqual(self.case.output.resolve(), target)
        self.assert_foreign_artifacts(target, contents, writes)

    def test_foreign_output_at_final_deadline_sample_refuses_before_server(self):
        checks, contents, timeline = [], {}, []
        def idle(port):
            checks.append(port)
        def clock():
            if (threading.current_thread() is self.case.worker_thread
                    and len(checks) == 2 and not contents):
                contents.update(self.foreign_artifacts(self.case.output))
                timeline.append("foreign-created-at-deadline-sample")
            return self.case.clock()
        with patch.object(bench, "assert_idle", side_effect=idle), \
                patch.object(bench.time, "monotonic", side_effect=clock):
            _, writes = self.failed_with_controller_writes()
        self.assertEqual(timeline, ["foreign-created-at-deadline-sample"])
        self.assert_foreign_artifacts(self.case.output, contents, writes)

    def test_foreign_output_after_last_absence_check_loses_exclusive_claim_before_server(self):
        checks, contents = [], {}
        output_check = bench.acceptance_output
        def idle(port):
            checks.append(port)
        def checked(args, manifest):
            result = output_check(args, manifest)
            if len(checks) == 2 and not contents:
                contents.update(self.foreign_artifacts(self.case.output))
            return result
        with patch.object(bench, "assert_idle", side_effect=idle), \
                patch.object(bench, "acceptance_output", side_effect=checked):
            _, writes = self.failed_with_controller_writes()
        self.assertEqual(checks, [8081, 8081])
        self.assertTrue(contents)
        self.assert_foreign_artifacts(self.case.output, contents, writes)

    def test_private_output_authority_overrides_public_args_and_is_proved_before_server(self):
        self.args._output_authority = "f" * 64
        spawn, native_open = self.case.spawn, self.directory_api.open
        claimed, server = [], []
        def directory(path, **kwargs):
            result = native_open(path, **kwargs)
            if path == self.case.output and kwargs.get("create"):
                claimed.append(True)
            return result
        def popen(argv, **kwargs):
            if "--engine" in argv:
                server.append(True)
                self.assertEqual(claimed, [True])
                data = (self.case.output / "provisional.json").read_bytes()
                first = json.loads(data)
                settings = json.loads(self.case.bootstrap_settings)
                authority = settings["output_authority"]
                self.assertRegex(authority, r"^[0-9a-f]{64}$")
                self.assertNotEqual(authority, self.args._output_authority)
                self.assertEqual(self.worker_args._output_authority, authority)
                owner = self.worker_args._output_owner
                self.assertIsNotNone(owner.handle)
                self.assertTrue((self.case.output / "controller-ack.json").is_file())
                identity = hashlib.sha256(json.dumps([
                    "strata-acceptance-output/2", authority, self.manifest["output"], *owner.identity
                ], separators=(",", ":")).encode()).hexdigest()
                unsigned = {"status": "provisional_starting", "output_claim": {
                    "path": self.manifest["output"], "identity_sha256": identity}}
                mac = hmac.new(bytes.fromhex(authority), json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
                               .encode(), hashlib.sha256).hexdigest()
                self.assertEqual(first, dict(unsigned, receipt_hmac_sha256=mac))
                # The receipt's keys are sorted; the token keeps the claim's own field order.
                acquired = hashlib.sha256(json.dumps([
                    "strata-acceptance-acquired/1", authority, unsigned["output_claim"]
                ], separators=(",", ":")).encode()).hexdigest()
                ack = (self.case.output / "controller-ack.json").read_bytes()
                self.assertEqual(json.loads(ack), {"acquired_sha256": acquired})
                self.assertFalse((self.case.output / "controller-ack.json.tmp").exists())
                self.assertNotIn(authority.encode(), data)
                self.assertNotIn(b"synthetic-key", data)
            return spawn(argv, **kwargs)
        with patch.object(self.directory_api, "open", side_effect=directory), \
                patch.object(bench.subprocess, "Popen", side_effect=popen):
            result = self.case.invoke_selected()
        self.assertEqual(server, [True])
        self.assertTrue(bench.acceptance_final_predicate(result))
        authority = json.loads(self.case.bootstrap_settings)["output_authority"]
        for path in self.case.output.iterdir():
            self.assertNotIn(authority.encode(), path.read_bytes())

    def output_authority_refused(self, value, *, missing=False):
        self.args._output_authority = "f" * 64
        bootstrap = bench.worker_bootstrap
        def changed(stream, worker, **kwargs):
            settings = json.loads(kwargs["settings"])
            if missing:
                settings.pop("output_authority", None)
            else:
                settings["output_authority"] = value
            kwargs["settings"] = json.dumps(settings)
            return bootstrap(stream, worker, **kwargs)
        with patch.object(bench, "worker_bootstrap", side_effect=changed):
            result, writes = self.failed_with_controller_writes()
        self.assertEqual(writes, [])
        self.assertFalse(bench.acceptance_final_predicate(result))
        if missing or value is None or type(value) is not str or bench.re.fullmatch(r"[0-9a-f]{64}", value) is None:
            self.assertNotIn("server", self.case.events)
            self.assertFalse(self.case.output.exists())
        return result

    def test_missing_private_output_authority_cannot_use_public_private_argument(self):
        self.output_authority_refused(None, missing=True)

    def test_malformed_private_output_authority_cannot_use_public_private_argument(self):
        self.output_authority_refused(False)

    def test_wrong_private_output_authority_prevents_controller_artifact_writes(self):
        self.output_authority_refused("e" * 64)

    def changed_claim_refused(self, change):
        close = self.case.native.close_process
        contents = {}
        def release(proc):
            change()
            contents.update({path.name: path.read_bytes() for path in self.case.output.iterdir()})
            close(proc)
        with patch.object(self.case.native, "close_process", side_effect=release):
            result, writes = self.failed_with_controller_writes()
        self.assertIn("server", self.case.events)
        self.assertEqual(writes, [])
        self.assertEqual({path.name: path.read_bytes() for path in self.case.output.iterdir()}, contents)
        self.assertEqual(result["normal_comparison"]["exclusions"], [{"reason": "worker-classification-unavailable"}])
        return result

    def test_worker_output_claim_cannot_go_missing_while_controller_receipt_is_held(self):
        # The former post-exit unlink now executes and is denied by the controller handle.
        close, attempts = self.case.native.close_process, []
        provisional = self.case.output / "provisional.json"
        def release(proc):
            close(proc)
            self.assertIsNone(self.worker_args._provisional_file.handle)
            before = provisional.read_bytes()
            with self.assertRaises(PermissionError):
                provisional.unlink()
            attempts.append(before)
        with patch.object(self.case.native, "close_process", side_effect=release):
            result = self.case.invoke_selected()
        self.assertEqual(len(attempts), 1)
        self.assertEqual(provisional.read_bytes(), attempts[0])
        self.assertEqual(result["artifact_sha256"]["provisional.json"], hashlib.sha256(attempts[0]).hexdigest())
        self.assertTrue(bench.acceptance_final_predicate(result))

    def test_malformed_worker_output_claim_refuses_controller_without_writes(self):
        self.changed_claim_refused(lambda: (self.case.output / "provisional.json").write_bytes(b'{"output_claim":false}'))

    def test_duplicate_worker_output_claim_refuses_controller_without_writes(self):
        def change():
            path = self.case.output / "provisional.json"
            record = json.loads(path.read_bytes())
            path.write_text(json.dumps(record)[:-1] + ', "output_claim":' + json.dumps(record["output_claim"]) + '}')
        self.changed_claim_refused(change)

    def test_oversized_worker_output_claim_refuses_controller_without_writes(self):
        self.changed_claim_refused(lambda: (self.case.output / "provisional.json").write_bytes(
            b" " * (bench.ACCEPTANCE_LOG_LIMIT + 1)))

    def test_mismatched_worker_output_claim_refuses_controller_without_writes(self):
        def change():
            path = self.case.output / "provisional.json"
            record = json.loads(path.read_bytes())
            record["output_claim"]["identity_sha256"] = "0" * 64
            path.write_text(json.dumps(record))
        self.changed_claim_refused(change)

    def test_changed_claim_path_refuses_controller_without_writes(self):
        def change():
            path = self.case.output / "provisional.json"
            record = json.loads(path.read_bytes())
            record["output_claim"]["path"] = str(self.root / "other-output")
            path.write_text(json.dumps(record))
        self.changed_claim_refused(change)

    def test_foreign_directory_with_copied_valid_claim_cannot_be_adopted(self):
        # A copied valid claim on a different object still fails, independently of
        # Windows denying replacement while the acquired controller handle lives.
        retained = self.root / "foreign-output"
        contents = self.foreign_artifacts(retained)
        result = self.case.invoke_selected()
        data = (self.case.output / "provisional.json").read_bytes()
        (retained / "provisional.json").write_bytes(data)
        contents["provisional.json"] = data
        authority = json.loads(self.case.bootstrap_settings)["output_authority"]
        owner = bench.AcceptanceOutputOwner(retained, str(retained))
        try:
            owner.open_receipt()
            # The copied bytes still authenticate; the claim binds the other directory.
            with self.assertRaisesRegex(ValueError, "Worker output claim differs"):
                bench.acceptance_claimed_output(retained, str(retained), authority, owner=owner)
            self.assertEqual(owner.receipt_bytes, data)
        finally:
            owner.close()
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertEqual({path.name: path.read_bytes() for path in retained.iterdir()}, contents)

    def test_controller_refuses_output_ancestor_symlink_before_reading_claim(self):
        parent = self.root / "container"
        parent.mkdir()
        self.args.output = self.case.output = parent / "output"
        self.changed_manifest(lambda value: value.update(output=str(self.case.output)))
        retained = self.root / "retained-container"
        def change():
            parent.rename(retained)
            parent.symlink_to(retained, target_is_directory=True)
        close = self.case.native.close_process
        attempts = []
        def release(proc):
            with self.assertRaises(PermissionError):
                change()
            attempts.append(True)
            close(proc)
        with patch.object(self.case.native, "close_process", side_effect=release):
            result = self.case.invoke_selected()
        self.assertEqual(attempts, [True])
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertFalse(parent.is_symlink())
        self.assertFalse(retained.exists())

    def test_inaccessible_worker_output_claim_refuses_controller_without_writes(self):
        read = self.receipt_api.read
        def inaccessible(handle, limit):
            if self.case.closed:
                raise PermissionError("synthetic inaccessible claim")
            return read(handle, limit)
        with patch.object(self.receipt_api, "read", side_effect=inaccessible):
            result, writes = self.failed_with_controller_writes()
        self.assertEqual(writes, [])
        self.assertIn("synthetic inaccessible claim", result["finalization_error"])
        self.assertFalse((self.case.output / "controller.log").exists())
        self.assertFalse((self.case.output / "lifecycle.json").exists())

    def test_legitimate_server_spawn_failure_after_claim_keeps_bounded_failure_receipt(self):
        spawn = self.case.spawn
        def fail(argv, **kwargs):
            if "--engine" in argv:
                self.assertEqual(json.loads((self.case.output / "provisional.json").read_bytes())["status"],
                                 "provisional_starting")
                raise OSError("synthetic spawn failure")
            return spawn(argv, **kwargs)
        with patch.object(bench.subprocess, "Popen", side_effect=fail), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke()
        result = caught.exception.record
        self.assertEqual(result["worker_exit"], 1)
        self.assertTrue(result["finalization"])
        self.assertEqual(json.loads((self.case.output / "lifecycle.json").read_bytes())["status"], "held")
        worker = json.loads((self.case.output / "provisional.json").read_bytes())
        self.assertEqual(worker["status"], "provisional_failed")
        self.assertEqual(worker["error"], "synthetic spawn failure")
        self.assertIn("output_claim", worker)
        self.assertLess(sum(path.stat().st_size for path in self.case.output.iterdir()), bench.ACCEPTANCE_ARTIFACT_LIMIT)

    def test_late_resolved_output_alias_refuses_before_server_without_creating_target(self):
        checks = []
        target = self.root / "other-output"
        def idle(port):
            checks.append(port)
            if len(checks) == 2:
                self.worker_args.output = self.root / "absent/../other-output"
        with patch.object(bench, "assert_idle", side_effect=idle):
            self.worker_refused("Manifest output differs.")
        self.assertEqual(checks, [8081, 8081])
        self.assertFalse(target.exists())
        self.assertFalse((self.root / "absent").exists())

    def test_late_output_symlink_refuses_before_server_and_preserves_foreign_target(self):
        checks = []
        target = self.root / "foreign"
        target.mkdir()
        marker = target / "foreign.txt"
        marker.write_bytes(b"foreign target must survive")
        def idle(port):
            checks.append(port)
            if len(checks) == 2:
                self.case.output.symlink_to(target, target_is_directory=True)
        with patch.object(bench, "assert_idle", side_effect=idle):
            self.worker_refused("Output directory already exists; choose a new directory.")
        self.assertEqual(checks, [8081, 8081])
        self.assertTrue(self.case.output.is_symlink())
        self.assertEqual(marker.read_bytes(), b"foreign target must survive")
        self.assertFalse((target / "provisional.json").exists())

    def test_late_parent_symlink_to_same_admitted_output_refuses_before_server(self):
        checks = []
        alias = self.root / "alias"
        def idle(port):
            checks.append(port)
            if len(checks) == 2:
                alias.symlink_to(self.root, target_is_directory=True)
                self.worker_args.output = alias / "output"
                self.assertEqual(str(self.worker_args.output.resolve()), self.manifest["output"])
        with patch.object(bench, "assert_idle", side_effect=idle):
            self.worker_refused("Manifest output path is linked.")
        self.assertEqual(checks, [8081, 8081])
        self.assertTrue(alias.is_symlink())
        self.assertEqual(alias.resolve(), self.root)

    def test_reordered_config_and_executable_rows_preserve_order_and_reach_workload(self):
        self.changed_manifest(lambda value: value["inputs"].insert(1, value["inputs"].pop(0)))
        self.assertEqual([row["role"] for row in self.manifest["inputs"][:2]], ["saved-exe", "config"])
        result = self.case.invoke_selected()
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertIn("server", self.case.events)
        worker = json.loads((self.case.output / "provisional.json").read_text())
        self.assertEqual(worker["admission"], {"status": "admitted", "manifest_sha256": self.digest})
        self.assertEqual([row["path"] for row in worker["protection"]["before"]],
                         [row["path"] for row in self.manifest["inputs"]])
        self.assertTrue(worker["acceptance_workload_passed"])

    def alternate_protected_config(self):
        alternate = self.root / "alternate.json"
        executable = self.root / "unmanifested.exe"
        executable.write_bytes(b"foreign executable, never executed")
        cfg = json.loads(self.args.config.read_text())
        cfg["exe"] = str(executable)
        alternate.write_text(json.dumps(cfg))
        self.args.protect_input.append(str(alternate))
        def change(manifest):
            manifest["inputs"].insert(1, manifest["inputs"].pop(0))
            manifest["inputs"].append(dict(role="protect-input", **bench.file_identity(alternate)))
            manifest["protect_input"].append(str(alternate))
        self.changed_manifest(change)
        self.assertEqual(self.manifest["inputs"][1]["role"], "config")
        self.assertNotIn(str(executable), [row["path"] for row in self.manifest["inputs"]])
        return alternate, executable

    def test_alternate_protect_input_config_refuses_controller_before_worker(self):
        alternate, executable = self.alternate_protected_config()
        before = {path: path.read_bytes() for path in (self.args.config, alternate, executable)}
        self.args.config = alternate
        self.controller_refused("Manifest config differs.")
        self.assertFalse(self.case.output.exists())
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_alternate_protect_input_config_refuses_worker_before_server(self):
        alternate, executable = self.alternate_protected_config()
        before = {path: path.read_bytes() for path in (self.args.config, alternate, executable)}
        self.before_preflight = lambda args: setattr(args, "config", alternate)
        self.worker_refused("Manifest config differs.")
        self.assertEqual(self.preflight_samples, 0)
        self.assertEqual(self.git_calls, [])
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_config_role_cardinality_types_and_canonical_path_refuse_both_admissions(self):
        mutations = [
            (lambda rows: rows.pop(0), "Manifest requires exactly one config input."),
            (lambda rows: rows[1].update(role="config"), "Manifest requires exactly one config input."),
            (lambda rows: rows[0].update(role="protect-input"), "Manifest requires exactly one config input."),
            (lambda rows: rows[0].update(role=True), "Invalid manifest input role."),
            (lambda rows: rows[0].update(role=["config"]), "Invalid manifest input role."),
            (lambda rows: rows[0].update(path=True), "Manifest paths must be absolute strings."),
            (lambda rows: rows[0].update(path=[str(self.args.config)]), "Manifest paths must be absolute strings."),
            (lambda rows: rows[0].pop("path"), "Invalid manifest input keys."),
            (lambda rows: rows[0].update(path=str(self.root / "absent/../synthetic.json")),
             "Manifest config differs."),
        ]
        for change, reason in mutations:
            manifest = bench.copy.deepcopy(self.manifest)
            change(manifest["inputs"])
            with self.subTest(inputs=manifest["inputs"][:2]):
                with self.assertRaises(ValueError) as controller:
                    bench.acceptance_admission(self.args, manifest, 1800000100.0)
                self.assertEqual(str(controller.exception), reason)
                args = argparse.Namespace(**vars(self.args), _acceptance_deadline=990,
                                          _admitted_manifest_sha256=self.digest)
                with patch.object(bench, "load_acceptance_manifest", return_value=(manifest, self.digest)), \
                        patch.object(bench, "file_identity", side_effect=AssertionError("input read before admission")), \
                        self.assertRaises(ValueError) as worker:
                    self.real_preflight(args)
                self.assertEqual(str(worker.exception), reason)
                self.assertEqual(args._acceptance_admission["status"], "refused")
                self.assertEqual(args._acceptance_admission["reason"], reason)
                self.assertEqual(self.git_calls, [])
                self.assertEqual(self.case.samples, 0)
                self.assertEqual(self.case.events, [])
                self.assertFalse(self.case.output.exists())

    def test_invalid_selected_config_types_cannot_alias_the_role_path(self):
        for config in (None, True, 1, 1.0, [str(self.args.config)]):
            args = argparse.Namespace(**vars(self.args))
            args.config = config
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "^Invalid manifest config argument\\.$"):
                bench.acceptance_admission(args, self.manifest, 1800000100.0)
        self.assertEqual(self.case.events, [])
        self.assertFalse(self.case.output.exists())

    def worker_artifact_collision(self, name, kind, *, save_number=1):
        target = self.case.output / name
        foreign = self.root / "foreign-artifact"
        original = b"external foreign artifact bytes must survive"
        foreign.write_bytes(original)
        original_open, attempts, injected = Path.open, [], []
        native_open, publish = self.provisional_api.open, bench.AcceptanceProvisionalFile.publish

        def insert():
            attempts.append("xb")
            if len(attempts) != save_number:
                return
            if kind == "hardlink":
                bench.os.link(foreign, target)
            elif kind == "symlink":
                target.symlink_to(foreign)
            elif kind == "directory":
                target.mkdir()
                (target / "keep").write_bytes(original)
            elif kind == "junction":
                import _winapi
                directory = self.root / "foreign-directory"
                directory.mkdir()
                (directory / "keep").write_bytes(original)
                _winapi.CreateJunction(str(directory), str(target))
            else:
                with original_open(target, "xb") as stream:
                    stream.write(original)
            injected.append(True)

        def open_stream(path, mode="r", *args, **kwargs):
            if path == target and mode in ("wb", "xb"):
                self.assertNotEqual(name, "provisional.json.tmp", "Manifest staging must never be opened")
                self.assertEqual(mode, "xb")
                insert()
            return original_open(path, mode, *args, **kwargs)

        def claim(path):
            if path == target:
                insert()
            return native_open(path)

        def engine_log(path):
            if path == target:
                insert()  # The planted name precedes the worker's create-once engine log.
            return engine_open(path)

        def prepared(file, data):
            if name == "provisional.json.tmp" and "requests" in json.loads(data):
                insert()
            return publish(file, data)

        engine_open = self.engine_log_api.open
        with patch.object(Path, "open", open_stream), \
                patch.object(self.provisional_api, "open", side_effect=claim), \
                patch.object(self.engine_log_api, "open", side_effect=engine_log), \
                patch.object(bench.AcceptanceProvisionalFile, "publish", prepared), \
                contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(injected, [True])
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        self.assertEqual(self.case.code, 0 if name == "provisional.json.tmp" else 1)
        self.assertEqual(foreign.read_bytes(), original)
        if kind in ("directory", "junction"):
            self.assertEqual((target / "keep").read_bytes(), original)
            if kind == "junction":
                self.assertTrue(target.is_junction())
        else:
            self.assertEqual(target.read_bytes(), original)
            if kind == "hardlink":
                self.assertTrue(bench.os.path.samefile(target, foreign))
            elif kind == "symlink":
                self.assertTrue(target.is_symlink())
        self.assertEqual(set(attempts), {"xb"})
        if name == "provisional.json":
            self.assertNotIn("server", self.case.events)
            self.assertFalse(caught.exception.record["finalization"])
        elif name == "engine.log":
            # The engine's log is created once before the server, so a planted name refuses the server.
            self.assertNotIn("server", self.case.events)
            receipt = json.loads((self.case.output / "provisional.json").read_bytes())
            # The controller's guard may refuse the planted object and stop the worker first.
            self.assertIn(receipt["status"], ("provisional_failed", "provisional_starting"))
            if receipt["status"] == "provisional_failed":
                self.assertEqual(receipt["artifact_error"], "RuntimeError: Engine log is missing.")
            self.assertIsNone(self.worker_args._engine_log.handle)
            self.assertEqual(self.engine_log_handles, set())
        else:
            self.assertIn("server", self.case.events)
            receipt = json.loads((self.case.output / "provisional.json").read_bytes())
            if name == "provisional.json.tmp":
                self.assertEqual(receipt["status"], "provisional_complete")
                self.assertIn("unexpected", caught.exception.record["finalization_error"])
            else:
                self.assertNotEqual(receipt["status"], "provisional_complete")
            if name in ("server.log", "engine.log"):
                self.assertEqual(receipt["status"], "provisional_failed")
                self.assertIn("artifact_error", receipt)
        self.assertEqual(self.directory_handles, set())

    def test_worker_server_log_hardlink_collision_preserves_external_bytes(self):
        self.worker_artifact_collision("server.log", "hardlink")

    def test_worker_engine_log_hardlink_collision_preserves_external_bytes(self):
        self.worker_artifact_collision("engine.log", "hardlink")

    def test_worker_provisional_staging_hardlink_collision_preserves_external_bytes(self):
        self.worker_artifact_collision("provisional.json.tmp", "hardlink")

    def test_worker_server_log_file_collision_preserves_bytes(self):
        self.worker_artifact_collision("server.log", "file")

    def test_worker_server_log_symlink_collision_preserves_bytes(self):
        self.worker_artifact_collision("server.log", "symlink")

    def test_worker_server_log_directory_collision_preserves_bytes(self):
        self.worker_artifact_collision("server.log", "directory")

    def test_worker_server_log_junction_collision_preserves_bytes(self):
        self.worker_artifact_collision("server.log", "junction")

    def test_worker_engine_log_file_collision_preserves_bytes(self):
        self.worker_artifact_collision("engine.log", "file")

    def test_worker_engine_log_symlink_collision_preserves_bytes(self):
        self.worker_artifact_collision("engine.log", "symlink")

    def test_worker_engine_log_directory_collision_preserves_bytes(self):
        self.worker_artifact_collision("engine.log", "directory")

    def test_worker_engine_log_junction_collision_preserves_bytes(self):
        self.worker_artifact_collision("engine.log", "junction")

    def test_worker_provisional_staging_file_collision_preserves_bytes(self):
        self.worker_artifact_collision("provisional.json.tmp", "file")

    def test_worker_provisional_staging_symlink_collision_preserves_bytes(self):
        self.worker_artifact_collision("provisional.json.tmp", "symlink")

    def test_worker_provisional_staging_directory_collision_preserves_bytes(self):
        self.worker_artifact_collision("provisional.json.tmp", "directory")

    def test_worker_provisional_staging_junction_collision_preserves_bytes(self):
        self.worker_artifact_collision("provisional.json.tmp", "junction")

    def test_worker_initial_provisional_file_collision_never_acquires_replacement_rights(self):
        self.worker_artifact_collision("provisional.json", "file")

    def test_worker_initial_provisional_hardlink_collision_never_acquires_replacement_rights(self):
        self.worker_artifact_collision("provisional.json", "hardlink")

    def test_worker_initial_provisional_symlink_collision_never_acquires_replacement_rights(self):
        self.worker_artifact_collision("provisional.json", "symlink")

    def test_worker_initial_provisional_directory_collision_never_acquires_replacement_rights(self):
        self.worker_artifact_collision("provisional.json", "directory")

    def test_worker_initial_provisional_junction_collision_never_acquires_replacement_rights(self):
        self.worker_artifact_collision("provisional.json", "junction")

    def test_worker_repeated_provisional_staging_hardlink_collision_preserves_bytes(self):
        self.worker_artifact_collision("provisional.json.tmp", "hardlink", save_number=4)

    def worker_write_failure(self, name, failure, *, save_number=1, primary=False):
        original_open = Path.open
        publish = bench.AcceptanceProvisionalFile.publish
        attempts, injected, snapshots = [], [], []
        provisional = self.case.output / "provisional.json"
        target = self.case.output / name
        message = "synthetic exclusive " + failure + " failure"

        def observed(file, data):
            snapshots.append(json.loads(data))
            return publish(file, data)

        def open_stream(path, mode="r", *args, **kwargs):
            if path != target or mode not in ("wb", "xb"):
                return original_open(path, mode, *args, **kwargs)
            self.assertEqual(mode, "xb")
            attempts.append(True)
            if len(attempts) != save_number:
                return original_open(path, mode, *args, **kwargs)
            injected.append(True)
            if failure == "create":
                raise OSError(message)
            stream = original_open(path, mode, *args, **kwargs)
            class FailedStream:
                def __enter__(inner): return inner
                def __exit__(inner, *exc):
                    stream.close()
                    self.assertTrue(stream.closed)
                    if failure == "close":
                        raise OSError(message)
                def write(inner, data):
                    if failure == "write":
                        raise OSError(message)
                    return stream.write(data[:-1] if failure == "short-write" else data)
                def flush(inner):
                    if failure == "flush":
                        raise OSError(message)
                    stream.flush()
            return FailedStream()

        stderr = io.StringIO()
        workload = (patch.object(bench, "acceptance_workload_tests", side_effect=ValueError("primary request failure"))
                    if primary else contextlib.nullcontext())
        with patch.object(Path, "open", open_stream), patch.object(bench.AcceptanceProvisionalFile, "publish", observed), \
                workload, contextlib.redirect_stderr(stderr), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(injected, [True])
        self.assertEqual(self.case.code, 1)
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        self.assertEqual(self.directory_handles, set())
        self.assertIn("server-stop", self.case.events)
        receipt = json.loads(provisional.read_bytes())
        self.assertEqual(receipt["status"], "provisional_failed")
        final = snapshots[-1]
        self.assertEqual(final["status"], "provisional_failed")
        if primary:
            self.assertEqual(final["error_type"], "ValueError")
            self.assertEqual(final["error"], "primary request failure")
            self.assertIn("Contained worker stopped (ValueError).", stderr.getvalue())
            self.assertEqual(final["artifact_error"], "OSError: " + message)
        else:
            expected = "OSError: Short artifact write." if failure == "short-write" else "OSError: " + message
            self.assertEqual(final["artifact_error"], expected)

    def retained_provisional_failure(self, operation, *, publish_number=1, primary=False):
        publish, snapshots, injected = bench.AcceptanceProvisionalFile.publish, [], []
        active = False
        native_name = "path" if operation == "final-path" else operation
        native_name = "write" if operation == "short-write" else native_name
        native_name = "open" if operation == "create" else native_name
        native = getattr(self.provisional_api, native_name)
        message = "synthetic retained " + operation + " failure"
        paths = []

        def prepared(file, data):
            nonlocal active
            snapshots.append(json.loads(data))
            active = len(snapshots) == publish_number
            try:
                return publish(file, data)
            finally:
                active = False

        def failed(*args):
            if not injected and (active or operation in ("create", "close")):
                if operation == "final-path":
                    paths.append(True)
                    if len(paths) == 1:
                        return native(*args)
                injected.append(True)
                if operation == "short-write":
                    return native(args[0], args[1][:-1])
                if operation == "close":
                    native(*args)  # Returning close error still cannot qualify a complete receipt.
                    raise OSError(message)
                raise OSError(message)
            return native(*args)

        stderr = io.StringIO()
        workload = (patch.object(bench, "acceptance_workload_tests", side_effect=ValueError("primary request failure"))
                    if primary else contextlib.nullcontext())
        with patch.object(bench.AcceptanceProvisionalFile, "publish", prepared), \
                patch.object(self.provisional_api, native_name, side_effect=failed), workload, \
                contextlib.redirect_stderr(stderr), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(injected, [True], "The claimed native failure must actually execute")
        self.assertEqual(self.case.code, 1)
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        self.assertEqual(self.provisional_handles, set())
        self.assertEqual(self.directory_handles, set())
        if operation == "create" or (publish_number == 1 and operation != "close"):
            self.assertNotIn("server", self.case.events)
        else:
            self.assertIn("server-stop", self.case.events)
        if primary:
            self.assertEqual(snapshots[-1]["error_type"], "ValueError")
            self.assertEqual(snapshots[-1]["error"], "primary request failure")
            self.assertIn("Contained worker stopped (ValueError).", stderr.getvalue())
        self.assertFalse((self.case.output / "provisional.json.tmp").exists())

    def test_initial_claim_exclusive_create_failure_never_starts_server(self):
        self.retained_provisional_failure("create")

    def test_initial_claim_write_failure_never_starts_server(self):
        self.retained_provisional_failure("write")

    def test_initial_claim_short_write_never_starts_server(self):
        self.retained_provisional_failure("short-write")

    def test_initial_claim_flush_failure_never_starts_server(self):
        self.retained_provisional_failure("flush")

    def test_retained_claim_final_close_failure_cannot_pass(self):
        self.retained_provisional_failure("close")

    def test_provisional_update_path_validation_failure_cannot_pass(self):
        self.retained_provisional_failure("path", publish_number=2)

    def test_provisional_update_write_failure_cannot_pass(self):
        self.retained_provisional_failure("write", publish_number=2)

    def test_provisional_update_short_write_cannot_pass(self):
        self.retained_provisional_failure("short-write", publish_number=2)

    def test_provisional_update_flush_failure_cannot_pass(self):
        self.retained_provisional_failure("flush", publish_number=2)

    def test_provisional_update_final_validation_failure_cannot_pass(self):
        self.retained_provisional_failure("final-path", publish_number=2)

    def test_server_artifact_create_failure_is_not_pass(self):
        self.worker_write_failure("server.log", "create")

    def test_server_artifact_write_failure_is_not_pass(self):
        self.worker_write_failure("server.log", "write")

    def test_server_artifact_short_write_is_not_pass(self):
        self.worker_write_failure("server.log", "short-write")

    def test_server_artifact_flush_failure_is_not_pass(self):
        self.worker_write_failure("server.log", "flush")

    def test_server_artifact_close_failure_is_not_pass(self):
        self.worker_write_failure("server.log", "close")

    def engine_log_handle_failure(self, operation, *, short=False):
        # The worker no longer writes engine.log: the engine appends to the worker's
        # create-once object, which the worker reads once and closes at exit.
        native = getattr(self.engine_log_api, operation)
        message, injected = "synthetic engine-log " + operation + " failure", []

        def failed(*args):
            if injected or (operation == "close" and args[0] != self.worker_args._engine_log.handle):
                return native(*args)  # A close failure targets the retained handle, not the final reader.
            injected.append(True)
            if operation == "read":
                if short:
                    return native(*args)[:-1]
                raise OSError(message)
            if operation == "close":
                raise OSError(message)  # The handle stays retained, as for a failed CloseHandle.
            raise OSError(message)

        stderr = io.StringIO()
        with patch.object(self.engine_log_api, operation, side_effect=failed), \
                contextlib.redirect_stderr(stderr), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        record = caught.exception.record
        try:
            self.assertEqual(injected, [True], "The claimed engine-log failure must actually execute")
            self.assertFalse(bench.acceptance_final_predicate(record))
            receipt = json.loads((self.case.output / "provisional.json").read_bytes())
            if operation == "open":
                self.assertNotIn("server", self.case.events)
                self.assertEqual(self.case.code, 1)
                self.assertEqual(receipt["error"], message)
            elif short:
                self.assertEqual(self.case.code, 0)
                self.assertEqual(receipt["status"], "provisional_complete")
                self.assertEqual(record["finalization_error"],
                                 "ValueError: Worker engine.log differs from its authenticated digest.")
                self.assertNotEqual(record["memory"]["cuda"]["status"], "satisfied")
            elif operation == "close":
                self.assertEqual(self.case.code, 1)
                self.assertEqual(receipt["status"], "provisional_complete")
                self.assertIn("Worker engine log close failed (OSError).", stderr.getvalue())
                self.assertEqual(len(self.engine_log_handles), 1)
            else:
                self.assertIn("server-stop", self.case.events)
                self.assertEqual(self.case.code, 1)
                self.assertEqual(receipt["status"], "provisional_failed")
                self.assertEqual(receipt["artifact_error"], "OSError: " + message)
                self.assertNotIn("engine.log", receipt.get("artifact_sha256", {}))
        finally:
            engine_log = self.worker_args._engine_log
            if engine_log is not None and engine_log.handle is not None:
                engine_log.deadline = float("inf")  # Caller cleanup of the retained handle.
                engine_log.close()
        self.assertEqual(self.engine_log_handles, set())

    def test_engine_log_create_failure_never_starts_server(self):
        self.engine_log_handle_failure("open")

    def test_engine_log_read_failure_is_not_pass(self):
        self.engine_log_handle_failure("read")

    def test_engine_log_short_read_is_not_pass(self):
        self.engine_log_handle_failure("read", short=True)

    def test_engine_log_final_open_failure_is_not_pass(self):
        # The final read opens a fresh deny-writers handle (no seek); a non-sharing failure is not retried.
        self.engine_log_handle_failure("open_final")

    def test_engine_log_close_failure_retains_handle_and_is_not_pass(self):
        self.engine_log_handle_failure("close")

    def test_primary_request_error_survives_separate_worker_artifact_error(self):
        self.worker_write_failure("server.log", "flush", primary=True)

    def test_late_provisional_final_validation_failure_cannot_pass(self):
        self.retained_provisional_failure("final-path", publish_number=5)

    def provisional_truncate_failure(self, save_number, *, primary=False):
        # Replacement publication is gone; its first/later/primary-error boundaries
        # now fail the retained object's explicit exact-length truncation instead.
        self.retained_provisional_failure("truncate", publish_number=save_number + 1, primary=primary)

    def test_first_provisional_update_truncate_failure_cannot_pass(self):
        self.provisional_truncate_failure(1)

    def test_repeated_provisional_update_truncate_failure_cannot_pass(self):
        self.provisional_truncate_failure(4)

    def test_primary_request_error_survives_failure_receipt_truncate_failure(self):
        self.provisional_truncate_failure(3, primary=True)

    def test_primary_request_error_cleanup_and_artifact_errors_remain_separate(self):
        spawn, create = self.case.spawn, bench.bounded_create
        def spawning(argv, **kwargs):
            proc = spawn(argv, **kwargs)
            if "--engine" in argv:
                def terminate():
                    self.case.events.append("server-stop")
                    raise OSError("synthetic cleanup failure")
                proc.terminate = terminate
            return proc
        def creating(path, *args, **kwargs):
            if path == self.case.output / "server.log":
                raise OSError("synthetic artifact failure")
            return create(path, *args, **kwargs)
        stderr = io.StringIO()
        with patch.object(bench.subprocess, "Popen", side_effect=spawning), \
                patch.object(bench, "bounded_create", side_effect=creating), \
                patch.object(bench, "acceptance_workload_tests", side_effect=ValueError("primary request failure")), \
                contextlib.redirect_stderr(stderr), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        receipt = json.loads((self.case.output / "provisional.json").read_bytes())
        self.assertEqual(receipt["status"], "provisional_failed")
        self.assertEqual(receipt["error_type"], "ValueError")
        self.assertEqual(receipt["error"], "primary request failure")
        self.assertEqual(receipt["cleanup_error"], "Process cleanup failed: synthetic cleanup failure")
        self.assertEqual(receipt["artifact_error"], "OSError: synthetic artifact failure")
        self.assertIn("Contained worker stopped (ValueError).", stderr.getvalue())
        self.assertIn("server-stop", self.case.events)
        self.assertEqual(self.case.code, 1)
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        self.assertEqual(self.directory_handles, set())

    def test_manifest_artifact_size_and_expired_deadline_refuse_before_create(self):
        path = self.root / "bounded-artifact"
        for data, deadline in ((b"12345", 990), (b"1234", 0)):
            with self.subTest(data=data, deadline=deadline), patch.object(Path, "open") as opened, \
                    self.assertRaisesRegex(RuntimeError, "Artifact exceeds its size or absolute time budget"):
                bench.bounded_create(path, data, deadline, limit=4)
            opened.assert_not_called()
            self.assertFalse(path.exists())

    def test_manifest_artifact_post_close_deadline_refuses_and_leaves_closed_created_bytes(self):
        path = self.root / "bounded-artifact"
        with patch.object(bench.time, "monotonic", side_effect=[0, 1]), \
                self.assertRaisesRegex(TimeoutError, "Artifact finalization exceeded the absolute deadline"):
            bench.bounded_create(path, b"1234", deadline=1, limit=4)
        self.assertEqual(path.read_bytes(), b"1234")
        path.rename(self.root / "closed-artifact")

    def test_provisional_publication_returning_at_deadline_cannot_pass(self):
        flush, injected, calls = self.provisional_api.flush, [], []
        def publish(handle):
            result = flush(handle)
            calls.append(True)
            if len(calls) == 2:
                injected.append(True)
                self.case.now = self.worker_args._acceptance_deadline
            return result
        with patch.object(self.provisional_api, "flush", side_effect=publish), contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(injected, [True])
        self.assertEqual(self.case.code, 1)
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        self.assertEqual(self.directory_handles, set())

    def test_repeated_provisional_publication_retains_one_exclusive_handle_until_final_save(self):
        claim, close = self.provisional_api.open, self.provisional_api.close
        publish, original_open, replace = bench.AcceptanceProvisionalFile.publish, Path.open, Path.replace
        pending = self.case.output / "provisional.json.tmp"
        provisional = self.case.output / "provisional.json"
        claimed, closed, published = [], [], []
        def created(path):
            handle = claim(path)
            claimed.append(handle)
            self.assertEqual(len(claimed), 1)
            return handle
        def saved(file, data):
            self.assertEqual(closed, [])
            self.assertEqual(file.handle, claimed[0])
            identity = file.api.identity(file.handle)
            result = publish(file, data)
            self.assertEqual(file.api.identity(file.handle), identity)
            published.append(data)
            self.assertEqual(provisional.read_bytes(), data)
            self.assertFalse(pending.exists())
            return result
        def closing(handle):
            self.assertEqual(json.loads(published[-1])["status"], "provisional_complete")
            closed.append(handle)
            return close(handle)
        def open_file(path, mode="r", *args, **kwargs):
            if path in (provisional, pending):
                self.assertEqual(mode, "rb", "Receipt writes must use only the retained handle")
            return original_open(path, mode, *args, **kwargs)
        def replacing(path, destination):
            self.assertNotIn(path, (provisional, pending))
            self.assertNotEqual(Path(destination), provisional)
            return replace(path, destination)
        with patch.object(self.provisional_api, "open", side_effect=created), \
                patch.object(self.provisional_api, "close", side_effect=closing), \
                patch.object(bench.AcceptanceProvisionalFile, "publish", saved), \
                patch.object(Path, "open", open_file), patch.object(Path, "replace", replacing):
            result = self.case.invoke_selected()
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertGreater(len(published), 4)
        self.assertEqual(closed, claimed)
        self.assertEqual(json.loads(published[-1])["status"], "provisional_complete")
        self.assertFalse(pending.exists())

    def test_final_worker_artifact_read_failure_cannot_pass(self):
        read, injected = self.receipt_api.read, []
        def failed(handle, limit):
            if self.case.code == 0:
                injected.append(True)
                raise OSError("synthetic final artifact read failure")
            return read(handle, limit)
        with patch.object(self.receipt_api, "read", side_effect=failed),                 self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(injected, [True])
        self.assertEqual(caught.exception.record["finalization_error"], "OSError: synthetic final artifact read failure")
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        self.assertEqual(self.directory_handles, set())

    def test_retained_provisional_allows_reads_denies_writes_and_delete_until_close(self):
        output = self.root / "retained-file"
        owner = bench.AcceptanceOutputOwner(output, str(output), create=True)
        receipt = bench.AcceptanceProvisionalFile(owner, 990)
        try:
            receipt.create()
            handle, identity = receipt.handle, receipt.identity
            receipt.publish(b'{"snapshot":"long first value"}')
            receipt.publish(b'{"snapshot":2}')
            self.assertEqual(receipt.path.read_bytes(), b'{"snapshot":2}')
            self.assertEqual(receipt.path.stat().st_size, 14)
            self.assertEqual(receipt.handle, handle)
            self.assertEqual(receipt.api.identity(handle), identity)
            with self.assertRaisesRegex(RuntimeError, "already claimed"):
                receipt.create()
            with self.assertRaises(PermissionError):
                receipt.path.unlink()
            with self.assertRaises(PermissionError):
                receipt.path.rename(output / "renamed.json")
            for mode in ("wb", "r+b"):
                with self.subTest(mode=mode), self.assertRaises(PermissionError):
                    with receipt.path.open(mode):
                        self.fail("A competing writer acquired the live receipt")
            self.assertEqual(receipt.path.read_bytes(), b'{"snapshot":2}')
        finally:
            receipt.close()
            owner.close()
        receipt.path.rename(output / "renamed.json")
        self.assertEqual((output / "renamed.json").read_bytes(), b'{"snapshot":2}')
        with self.assertRaisesRegex(RuntimeError, "already claimed"):
            receipt.create()

    def provisional_replacement_attempts(self, kind):
        publish, snapshots = bench.AcceptanceProvisionalFile.publish, []
        foreign = self.root / "replacement-source"
        original = b"external replacement bytes must survive"
        if kind in ("directory", "junction"):
            foreign.mkdir()
            (foreign / "keep").write_bytes(original)
        else:
            foreign.write_bytes(original)
        attempts, identities = [], []

        def prepared(file, data):
            # The serialized snapshot is ready, at the former closed-stage /
            # Path.replace edge. Both attack operations must actually execute.
            before = file.path.read_bytes()
            identities.append(file.api.identity(file.handle))
            attempts.append("unlink")
            with self.assertRaises(PermissionError):
                file.path.unlink()
            attempts.append("insert")
            with self.assertRaises(OSError):
                if kind == "hardlink":
                    bench.os.link(foreign, file.path)
                elif kind == "file":
                    foreign.replace(file.path)
                elif kind == "symlink":
                    file.path.symlink_to(foreign)
                elif kind == "directory":
                    file.path.mkdir()
                else:
                    import _winapi
                    _winapi.CreateJunction(str(foreign), str(file.path))
            self.assertEqual(file.path.read_bytes(), before)
            self.assertEqual(file.api.identity(file.handle), identities[-1])
            self.assertEqual((foreign / "keep" if foreign.is_dir() else foreign).read_bytes(), original)
            result = publish(file, data)
            self.assertEqual(file.path.read_bytes(), data)
            snapshots.append(json.loads(data))
            return result

        with patch.object(bench.AcceptanceProvisionalFile, "publish", prepared):
            result = self.case.invoke_selected()
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertGreater(len(snapshots), 4)
        self.assertEqual(attempts, ["unlink", "insert"] * len(snapshots))
        self.assertEqual(snapshots[0]["status"], "provisional_starting")
        self.assertEqual(snapshots[-1]["status"], "provisional_complete")
        self.assertEqual(set(identities), {identities[0]})
        self.assertIsNone(self.worker_args._provisional_file.handle)
        self.assertEqual(self.provisional_handles, set())
        self.assertFalse((self.case.output / "provisional.json.tmp").exists())

    def test_r5b_hardlink_swap_attempt_executes_and_is_denied_at_every_publication(self):
        self.provisional_replacement_attempts("hardlink")

    def test_provisional_file_swap_attempt_is_denied_at_initial_repeated_and_final_publication(self):
        self.provisional_replacement_attempts("file")

    def test_provisional_symlink_insertion_is_denied_at_initial_repeated_and_final_publication(self):
        self.provisional_replacement_attempts("symlink")

    def test_provisional_directory_insertion_is_denied_at_initial_repeated_and_final_publication(self):
        self.provisional_replacement_attempts("directory")

    def test_provisional_junction_insertion_is_denied_at_initial_repeated_and_final_publication(self):
        self.provisional_replacement_attempts("junction")

    def test_initial_retained_seek_failure_never_starts_server(self):
        self.retained_provisional_failure("seek")

    def test_repeated_retained_seek_failure_cannot_pass(self):
        self.retained_provisional_failure("seek", publish_number=4)

    def test_initial_retained_truncate_failure_never_starts_server(self):
        self.retained_provisional_failure("truncate")

    def test_initial_retained_identity_failure_never_starts_server(self):
        self.retained_provisional_failure("identity")

    def test_repeated_retained_identity_failure_cannot_pass(self):
        self.retained_provisional_failure("identity", publish_number=4)

    def test_initial_retained_path_failure_never_starts_server(self):
        self.retained_provisional_failure("path")

    def test_primary_request_error_survives_retained_write_failure(self):
        self.retained_provisional_failure("write", publish_number=4, primary=True)

    def test_primary_request_error_survives_retained_close_failure(self):
        self.retained_provisional_failure("close", primary=True)

    def retained_deadline_failure(self, boundary, edge, offset):
        publish, step = bench.AcceptanceProvisionalFile.publish, bench.AcceptanceProvisionalFile.step
        calls, executed, crossed, snapshots = [], [], [], []
        active = False

        def clock():
            if crossed and threading.current_thread() is self.case.worker_thread:
                return self.worker_args._acceptance_deadline + offset
            return self.case.clock()

        def prepared(file, data):
            nonlocal active
            snapshots.append(json.loads(data))
            active = len(snapshots) == 2
            try:
                return publish(file, data)
            finally:
                active = False

        def stepping(file, operation, *args):
            if not active:
                return step(file, operation, *args)
            calls.append(True)
            if len(calls) != boundary:
                return step(file, operation, *args)
            if edge == "before":
                crossed.append(offset)
            def performed(*values):
                executed.append(True)
                result = operation(*values)
                if edge == "after":
                    crossed.append(offset)
                return result
            return step(file, performed, *args)

        with patch.object(bench.time, "monotonic", side_effect=clock), \
                patch.object(bench.AcceptanceProvisionalFile, "publish", prepared), \
                patch.object(bench.AcceptanceProvisionalFile, "step", stepping), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(crossed, [offset])
        self.assertEqual(len(calls), boundary)
        self.assertEqual(executed, [True] if edge == "after" else [])
        self.assertEqual(self.case.code, 1)
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        self.assertEqual(self.provisional_handles, set())
        self.assertIsNone(self.worker_args._provisional_file.handle)
        self.assertEqual(snapshots[-1]["status"], "provisional_failed")

    def retained_create_or_close_deadline_failure(self, phase, edge, offset):
        create, close = bench.AcceptanceProvisionalFile.create, bench.AcceptanceProvisionalFile.close
        check = bench.AcceptanceProvisionalFile.check_deadline
        native_open, native_close = self.provisional_api.open, self.provisional_api.close
        active, crossed, checks, opened, closed = [], [], [], [], []

        def clock():
            if crossed and threading.current_thread() is self.case.worker_thread:
                return self.worker_args._acceptance_deadline + offset
            return self.case.clock()

        def creating(file):
            if file.path.name == "provisional.json":  # The engine-log handle has its own create.
                active.append(True)
            try:
                return create(file)
            finally:
                active.clear()

        def checked(file):
            if phase == "create" and active:
                checks.append(True)
                if len(checks) == (3 if edge == "before" else 4):
                    crossed.append(offset)
            return check(file)

        def opening(path):
            opened.append(True)
            return native_open(path)

        def closing(file):
            if phase == "close" and edge == "before" and file.path.name == "provisional.json":
                crossed.append(offset)
            return close(file)

        def native_closing(handle):
            result = native_close(handle)
            closed.append(True)
            if phase == "close" and edge == "after":
                crossed.append(offset)
            return result

        with patch.object(bench.time, "monotonic", side_effect=clock), \
                patch.object(bench.AcceptanceProvisionalFile, "create", creating), \
                patch.object(bench.AcceptanceProvisionalFile, "check_deadline", checked), \
                patch.object(bench.AcceptanceProvisionalFile, "close", closing), \
                patch.object(self.provisional_api, "open", side_effect=opening), \
                patch.object(self.provisional_api, "close", side_effect=native_closing), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(crossed, [offset])
        self.assertEqual(opened, [] if phase == "create" and edge == "before" else [True])
        self.assertEqual(closed, opened, "Expired cleanup must still release its handle")
        self.assertEqual(self.case.code, 1)
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        self.assertEqual(self.provisional_handles, set())
        if phase == "create":
            self.assertNotIn("server", self.case.events)

    def test_real_provisional_close_failure_retains_handle_and_cannot_pass(self):
        close, failures = self.provisional_api.close, []
        def failed(handle):
            failures.append(handle)
            raise OSError("synthetic native provisional close failure")
        stderr = io.StringIO()
        with patch.object(self.provisional_api, "close", side_effect=failed), \
                contextlib.redirect_stderr(stderr), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        try:
            self.assertEqual(failures, [self.worker_args._provisional_file.handle])
            self.assertEqual(self.provisional_handles, set(failures))
            self.assertEqual(self.case.code, 1)
            self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
            self.assertIn("Worker provisional close failed (OSError).", stderr.getvalue())
        finally:
            receipt = self.worker_args._provisional_file
            receipt.deadline = 0  # Caller cleanup is still required after expiration.
            with self.assertRaisesRegex(TimeoutError, "Provisional close reached the absolute deadline"):
                receipt.close()
        self.assertEqual(self.provisional_handles, set())

    def provisional_identity_mismatch(self, *, path=False, initial=False):
        native = self.provisional_api.path if path else self.provisional_api.identity
        publish = bench.AcceptanceProvisionalFile.publish
        snapshots, injected, identities = [], [], []

        def prepared(file, data):
            snapshots.append(json.loads(data))
            return publish(file, data)

        def changed(handle):
            actual = native(handle)
            if not injected and (initial or len(snapshots) == 2):
                if not path:
                    identities.append(actual)
                    if initial and len(identities) == 1:
                        return actual  # Capture the true birth identity, then contradict it.
                injected.append(True)
                return str(self.root / "different.json") if path else (actual[0] + 1, actual[1])
            return actual

        with patch.object(bench.AcceptanceProvisionalFile, "publish", prepared), \
                patch.object(self.provisional_api, "path" if path else "identity", side_effect=changed), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(injected, [True])
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        self.assertEqual(self.case.code, 1)
        self.assertEqual(self.provisional_handles, set())
        if initial:
            self.assertNotIn("server", self.case.events)
        else:
            self.assertEqual(snapshots[-1]["error"], "Provisional file identity differs.")

    def test_initial_provisional_handle_identity_mismatch_never_starts_server(self):
        self.provisional_identity_mismatch(initial=True)

    def test_initial_provisional_handle_path_mismatch_never_starts_server(self):
        self.provisional_identity_mismatch(path=True, initial=True)

    def test_repeated_provisional_handle_identity_mismatch_cannot_pass(self):
        self.provisional_identity_mismatch()

    def test_repeated_provisional_handle_path_mismatch_cannot_pass(self):
        self.provisional_identity_mismatch(path=True)

    def test_final_partial_provisional_write_is_never_accepted_and_releases_handle(self):
        write, injected = self.provisional_api.write, []
        def partial(handle, data):
            if json.loads(data).get("status") == "provisional_complete":
                injected.append(True)
                return write(handle, b'{"status":')
            return write(handle, data)
        with patch.object(self.provisional_api, "write", side_effect=partial), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(injected, [True])
        self.assertEqual(self.case.code, 1)
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        self.assertFalse(caught.exception.record["finalization"])
        with self.assertRaises(json.JSONDecodeError):
            json.loads((self.case.output / "provisional.json").read_bytes())
        self.assertEqual(self.provisional_handles, set())

    def test_retained_snapshot_size_refusal_is_latched_and_releases_handle(self):
        owner = bench.AcceptanceOutputOwner(self.root / "size-refusal", str(self.root / "size-refusal"), create=True)
        receipt = bench.AcceptanceProvisionalFile(owner, 990)
        try:
            receipt.create()
            receipt.publish(b'{"status":"provisional_starting"}')
            before = receipt.path.read_bytes()
            with self.assertRaisesRegex(ValueError, "exceeds its size budget"):
                receipt.publish(b"x" * (bench.ACCEPTANCE_LOG_LIMIT + 1))
            with self.assertRaisesRegex(RuntimeError, "previously failed"):
                receipt.publish(b'{"status":"provisional_complete"}')
            self.assertEqual(receipt.path.read_bytes(), before)
        finally:
            receipt.close()
            owner.close()
        self.assertEqual(self.provisional_handles, set())

    def selected_refused(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertTrue(self.case.closed)
        return caught.exception.record

    def receipt_mac(self, record):
        authority = json.loads(self.case.bootstrap_settings)["output_authority"]
        unsigned = {key: value for key, value in record.items() if key != "receipt_hmac_sha256"}
        message = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hmac.new(bytes.fromhex(authority), message, hashlib.sha256).hexdigest()

    def assert_retained_receipt_is_last_opened(self, final_reads):
        # Acquisition may legitimately retry while the claim is incomplete, so the
        # count of opens is timing-dependent; the order and closure are not.
        self.assertTrue(final_reads)
        self.assertEqual(set(final_reads), {self.receipt_opens[-1]})
        self.assertEqual(self.receipt_events, [(kind, handle) for handle in self.receipt_opens
                                               for kind in ("open", "close")])

    def test_lifecycle_records_digests_of_every_authenticated_retained_artifact(self):
        read, final_reads = self.receipt_api.read, []
        def reading(handle, limit):
            if self.case.closed:
                final_reads.append(handle)
            return read(handle, limit)
        with patch.object(self.receipt_api, "read", side_effect=reading):
            result = self.case.invoke_selected()
        self.assertTrue(bench.acceptance_final_predicate(result))
        output = self.case.output
        expected = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                    for name in ("provisional.json", "server.log", "engine.log", "controller.log")}
        self.assertEqual(result["artifact_sha256"], expected)
        self.assertEqual(json.loads((output / "lifecycle.json").read_bytes())["artifact_sha256"], expected)
        worker = json.loads((output / "provisional.json").read_bytes())
        self.assertEqual(worker["status"], "provisional_complete")
        self.assertEqual(worker["artifact_sha256"], {name: expected[name] for name in ("server.log", "engine.log")})
        self.assertEqual(worker["receipt_hmac_sha256"], self.receipt_mac(worker))
        authority = json.loads(self.case.bootstrap_settings)["output_authority"]
        for path in output.iterdir():
            self.assertNotIn(authority.encode(), path.read_bytes())
        self.assertEqual(len(final_reads), 2)
        self.assert_retained_receipt_is_last_opened(final_reads)

    def post_close_receipt_swap(self, kind):
        # Parent probe r6 sequence: the worker's final close has returned and the
        # controller has not yet made its first finalization read.
        close, publish = bench.AcceptanceProvisionalFile.close, bench.AcceptanceProvisionalFile.publish
        claimed = bench.acceptance_claimed_output
        foreign = self.root / "post-close-source"
        original = b"external post-close bytes must survive"
        if kind == "junction":
            foreign.mkdir()
            (foreign / "keep").write_bytes(original)
        else:
            foreign.write_bytes(original)
        published, attempts, identities = [], [], []

        def observed(file, data):
            result = publish(file, data)
            published.append(data)
            return result

        def closing(file):
            identity = file.identity if file.handle is not None else None
            close(file)
            if identity is None or attempts or file.path.name != "provisional.json":
                return
            self.assertIsNone(file.handle)
            self.assertEqual(len(self.receipt_handles), 1, "The controller receipt handle must be live")
            identities.extend([identity, self.receipt_api.identity(next(iter(self.receipt_handles)))])
            before = file.path.read_bytes()
            attempts.append("unlink")
            with self.assertRaises(PermissionError):
                file.path.unlink()
            attempts.append(kind)
            with self.assertRaises(OSError):
                if kind == "file":
                    with open(file.path, "xb") as stream:
                        stream.write(original)
                elif kind == "hardlink":
                    bench.os.link(foreign, file.path)
                elif kind == "rename":
                    foreign.replace(file.path)
                elif kind == "symlink":
                    file.path.symlink_to(foreign)
                else:
                    import _winapi
                    _winapi.CreateJunction(str(foreign), str(file.path))
            self.assertEqual(file.path.read_bytes(), before)

        def first_final_read(*args, **kwargs):
            if self.case.closed:
                self.assertEqual(attempts, ["unlink", kind], "The attack must precede the first final read")
            return claimed(*args, **kwargs)

        with patch.object(bench.AcceptanceProvisionalFile, "close", closing), \
                patch.object(bench.AcceptanceProvisionalFile, "publish", observed), \
                patch.object(bench, "acceptance_claimed_output", side_effect=first_final_read):
            result = self.case.invoke_selected()
        self.assertEqual(attempts, ["unlink", kind])
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertEqual(identities[0], identities[1])
        provisional = self.case.output / "provisional.json"
        self.assertFalse(provisional.is_symlink())
        self.assertEqual(provisional.read_bytes(), published[-1])
        self.assertEqual(json.loads(published[-1])["status"], "provisional_complete")
        self.assertEqual(result["artifact_sha256"]["provisional.json"], hashlib.sha256(published[-1]).hexdigest())
        self.assertEqual((foreign / "keep" if kind == "junction" else foreign).read_bytes(), original)
        self.assertEqual(self.receipt_handles, set())

    def test_post_close_unlink_and_new_file_cannot_replace_worker_receipt(self):
        self.post_close_receipt_swap("file")

    def test_post_close_hardlink_cannot_replace_worker_receipt(self):
        self.post_close_receipt_swap("hardlink")

    def test_post_close_rename_over_cannot_replace_worker_receipt(self):
        self.post_close_receipt_swap("rename")

    def test_post_close_symlink_cannot_replace_worker_receipt(self):
        self.post_close_receipt_swap("symlink")

    def test_post_close_junction_cannot_replace_worker_receipt(self):
        self.post_close_receipt_swap("junction")

    def post_close_receipt_rewrite(self, change, error=None):
        close, publish = bench.AcceptanceProvisionalFile.close, bench.AcceptanceProvisionalFile.publish
        published, rewritten = [], []

        def observed(file, data):
            result = publish(file, data)
            published.append(data)
            return result

        def closing(file):
            active = file.handle is not None and file.path.name == "provisional.json"
            close(file)
            if active and not rewritten:
                data = change(published)
                # The worker writer is gone; the controller handle shares writes by necessity.
                with open(file.path, "r+b") as stream:
                    stream.write(data)
                    stream.truncate()
                rewritten.append(data)
                self.assertEqual(file.path.read_bytes(), data)

        with patch.object(bench.AcceptanceProvisionalFile, "close", closing), \
                patch.object(bench.AcceptanceProvisionalFile, "publish", observed), \
                self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        record = caught.exception.record
        self.assertEqual(len(rewritten), 1)
        self.assertEqual(record["worker_exit"], 0)
        self.assertFalse(record["finalization"])
        if error is not None:
            self.assertEqual(record["finalization_error"], error)
        self.assertFalse(bench.acceptance_final_predicate(record))
        self.assertEqual((self.case.output / "provisional.json").read_bytes(), rewritten[0])
        self.assertFalse((self.case.output / "lifecycle.json").exists())
        return published, record

    def test_post_close_in_place_different_record_refuses(self):
        def different(published):
            record = json.loads(published[-1])
            record["foreign_marker"] = True
            return json.dumps(record).encode()
        self.post_close_receipt_rewrite(different, "ValueError: Worker receipt authentication differs.")

    def test_post_close_in_place_replay_of_earlier_worker_snapshot_refuses(self):
        def replay(published):
            earlier = json.loads(published[-2])
            self.assertEqual(earlier["receipt_hmac_sha256"], self.receipt_mac(earlier))
            self.assertNotIn("artifact_sha256", earlier)
            return published[-2]
        self.post_close_receipt_rewrite(replay, "ValueError: Worker server.log differs from its authenticated digest.")

    def test_post_close_in_place_truncation_refuses(self):
        published, record = self.post_close_receipt_rewrite(lambda published: published[-1][:len(published[-1]) // 2])
        self.assertGreater(len(published[-1]), 2)
        self.assertTrue(record["finalization_error"].startswith("JSONDecodeError: "))

    def test_in_place_replay_between_controller_final_reads_refuses(self):
        publish, claimed = bench.AcceptanceProvisionalFile.publish, bench.acceptance_claimed_output
        published, reads = [], []

        def observed(file, data):
            result = publish(file, data)
            published.append(data)
            return result

        def read(*args, **kwargs):
            record = claimed(*args, **kwargs)
            if self.case.closed:
                reads.append(record)
                if len(reads) == 1:
                    # A valid earlier worker snapshot still authenticates and carries the claim.
                    with open(self.case.output / "provisional.json", "r+b") as stream:
                        stream.write(published[-2])
                        stream.truncate()
            return record

        with patch.object(bench.AcceptanceProvisionalFile, "publish", observed), \
                patch.object(bench, "acceptance_claimed_output", side_effect=read):
            record = self.selected_refused()
        self.assertEqual(len(reads), 2)
        self.assertEqual(reads[0]["status"], "provisional_complete")
        self.assertNotEqual(reads[1]["status"], "provisional_complete")
        self.assertFalse(record["finalization"])
        self.assertEqual(record["finalization_error"], "ValueError: Worker receipt changed after controller consumption.")
        self.assertFalse(bench.acceptance_final_predicate(record))
        self.assertEqual((self.case.output / "provisional.json").read_bytes(), published[-2])

    def worker_log_changed_after_close(self, name, mode):
        create, close, changed = bench.bounded_create, bench.AcceptanceProvisionalFile.close, []
        target = self.case.output / name

        def change(path, data):
            if mode == "rewritten":
                with open(path, "r+b") as stream:
                    stream.write(b"X")
            else:
                path.unlink()
                path.write_bytes(data if mode == "identical" else data + b"foreign suffix\n")
            changed.append(data)

        def creating(path, data, *args, **kwargs):
            result = create(path, data, *args, **kwargs)
            if path == target:
                change(path, data)
            return result

        def closing(file):
            active = file.handle is not None and file.path == target
            close(file)
            if active:
                # The engine and the worker have closed their handles; the guard is not mid-read.
                with self.case.clock_lock:
                    change(target, target.read_bytes())

        hook = (patch.object(bench.AcceptanceProvisionalFile, "close", closing) if name == "engine.log"
                else patch.object(bench, "bounded_create", side_effect=creating))
        if mode == "identical" and name != "engine.log":
            with hook:
                result = self.case.invoke_selected()
        else:
            with hook, self.assertRaises(bench.AcceptanceFailure) as caught:
                self.case.invoke_selected()
            result = caught.exception.record
        self.assertEqual(len(changed), 1)
        digest = hashlib.sha256(changed[0]).hexdigest()
        worker = json.loads((self.case.output / "provisional.json").read_bytes())
        self.assertEqual(worker["artifact_sha256"][name], digest)
        self.assertEqual(worker["status"], "provisional_complete")
        if name != "engine.log" or mode == "rewritten":
            self.assertEqual(result["worker_exit"], 0)  # A guard refusal can precede the recorded exit.
        if name == "engine.log" and mode != "rewritten":
            # A replaced object, even with identical bytes, is refused by the guard's bound identity.
            self.assertIn(result["memory"]["cuda"]["errors"][0], ("Engine log identity changed.",
                                                                  "Engine log disappeared."))
            self.assertEqual(result["error"], "RuntimeError: Engine CUDA-stage evidence failed: "
                             + result["memory"]["cuda"]["errors"][0])
            self.assertEqual(result["memory"]["cuda"]["status"], "refused")
            self.assertFalse(bench.acceptance_final_predicate(result))
            self.assertNotIn(name, result.get("artifact_sha256", {}))
        elif mode == "identical":
            self.assertTrue(bench.acceptance_final_predicate(result))
            self.assertEqual(target.read_bytes(), changed[0])
            self.assertEqual(result["artifact_sha256"][name], digest)
            lifecycle = json.loads((self.case.output / "lifecycle.json").read_bytes())
            self.assertEqual(lifecycle["artifact_sha256"][name], digest)
        else:
            self.assertNotEqual(hashlib.sha256(target.read_bytes()).hexdigest(), digest)
            self.assertFalse(result["finalization"])
            self.assertEqual(result["finalization_error"],
                             "ValueError: Worker " + name + " differs from its authenticated digest.")
            self.assertFalse(bench.acceptance_final_predicate(result))
            self.assertNotIn(name, result["artifact_sha256"])

    def test_byte_identical_server_log_replacement_passes_with_recorded_digest(self):
        self.worker_log_changed_after_close("server.log", "identical")

    def test_byte_identical_engine_log_replacement_refuses_by_guard_identity(self):
        self.worker_log_changed_after_close("engine.log", "identical")

    def test_modified_server_log_replacement_refuses(self):
        self.worker_log_changed_after_close("server.log", "replaced")

    def test_modified_engine_log_replacement_refuses(self):
        self.worker_log_changed_after_close("engine.log", "replaced")

    def test_in_place_server_log_rewrite_refuses(self):
        self.worker_log_changed_after_close("server.log", "rewritten")

    def test_in_place_engine_log_rewrite_refuses(self):
        self.worker_log_changed_after_close("engine.log", "rewritten")

    def engine_run(self, mark=None, *, selection="short", refused=False, floors=(4, 4)):
        clones = []
        spawn = self.case.spawn
        def spawning(argv, **kwargs):
            if "--engine" in argv:
                clones.append((json.loads(Path(argv[argv.index("--config") + 1]).read_text()), dict(kwargs["env"])))
            return spawn(argv, **kwargs)
        if mark is not None:
            self.case.engine_mark = mark
        with patch.object(bench.subprocess, "Popen", side_effect=spawning):
            if refused:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(bench.AcceptanceFailure) as caught:
                    self.case.invoke_selected(selection=selection, floors=floors)
                result = caught.exception.record
            else:
                result = self.case.invoke_selected(selection=selection, floors=floors)
        self.assertEqual(self.engine_log_handles, set())
        return result, clones

    def automatic_launch_run(self, *, floors=(8, 8), startup=None, launch=None, refused=False):
        cfg = json.loads(self.case.config.read_text())
        cfg["memory_policy"] = {"version": 1}
        self.case.config.write_text(json.dumps(cfg))
        self.args.acceptance_workload, self.args.cohort = "near-full", "automatic"
        self.args.acceptance_physical_gib, self.args.acceptance_commit_gib = floors
        manifest = self.root / "automatic-launch-manifest.json"
        self.args.acceptance_manifest = None
        self.args.write_acceptance_manifest, self.args.manifest_not_after_unix = manifest, 1800010000.0
        self.assertEqual(bench.run(self.args), 0)
        self.args.acceptance_manifest = manifest
        self.args.write_acceptance_manifest = self.args.manifest_not_after_unix = None
        self.case.launch_memory = ({"cohort": "automatic", "sample_status": "available",
                                    "sample": {"headroom_requested_gib": 8}} if launch is None else launch)
        self.case.engine_startup = (LaunchEvidenceTests.AUTO + LaunchEvidenceTests.PINNED + LaunchEvidenceTests.RECENT
                                   if startup is None else startup)
        return self.engine_run(selection="near-full", floors=floors, refused=refused)

    def test_controller_checks_realized_automatic_launch_and_authenticated_startup_allocations(self):
        result, clones = self.automatic_launch_run()
        self.assertEqual(result["launch_evidence"], {"status": "satisfied"})
        self.assertEqual(result["memory"]["floors_gib"], {"physical": 8, "commit": 8})
        self.assertEqual(result["allocation_evidence"], {
            "recent_cache": {"actual_mib": 4525.0, "requested_mib": 4525},
            "expert_cache_auto": {"free_gib": 10.21, "reserved_mib": 1024, "draft_head_mib": 0, "slots": 4542},
            "mapped_pinned_complement": {"resident_gib": 9.0, "pinned_gib": 9.0}, "errors": [],
            "end_of_run_cache_capacity":
                "not collected; printed only under STRATA_FETCH_PROFILE, which acceptance arms exclude"})
        worker = json.loads((self.case.output / "provisional.json").read_bytes())
        self.assertEqual(worker["launch_memory"], self.case.launch_memory)
        self.assertEqual(worker["engine"], {"arena_mib": 4096, "expert_cache_mib": 1024})
        self.assertIs(clones[0][0]["launch_memory_metrics"], True)
        lifecycle = json.loads((self.case.output / "lifecycle.json").read_bytes())
        self.assertEqual(lifecycle["launch_evidence"], {"status": "satisfied"})
        self.assertEqual(lifecycle["allocation_evidence"], result["allocation_evidence"])

    def test_controller_compares_automatic_headroom_with_its_commit_floor_independently(self):
        result, _ = self.automatic_launch_run(floors=(8, 4), refused=True)
        self.assertEqual(result["launch_evidence"], {
            "status": "refused", "reason": "Automatic launch headroom exceeds the physical or commit floor."})
        self.assertEqual(result["memory"]["floors_gib"], {"physical": 8, "commit": 4})

    def test_controller_missing_recent_allocation_refuses_an_otherwise_available_automatic_launch(self):
        result, _ = self.automatic_launch_run(startup=LaunchEvidenceTests.AUTO + LaunchEvidenceTests.PINNED,
                                             refused=True)
        self.assertEqual(result["launch_evidence"], {
            "status": "refused", "reason": "Automatic launch recent-cache startup allocation is missing."})
        self.assertIsNone(result["allocation_evidence"]["recent_cache"])

    def test_controller_unavailable_automatic_sample_refuses_despite_startup_allocations(self):
        result, _ = self.automatic_launch_run(launch={"cohort": "automatic", "sample_status": "unavailable",
                                                    "sample": {"headroom_requested_gib": 8}}, refused=True)
        self.assertEqual(result["launch_evidence"], {
            "status": "refused", "reason": "Automatic launch sample is unavailable."})
        self.assertEqual(result["allocation_evidence"]["recent_cache"], {"actual_mib": 4525.0, "requested_mib": 4525})

    def test_controller_refuses_a_realized_automatic_sample_in_a_saved_static_arm(self):
        self.case.launch_memory = {"cohort": "automatic", "sample_status": "available",
                                   "sample": {"headroom_requested_gib": 8}}
        result, _ = self.engine_run(refused=True)
        self.assertEqual(result["launch_evidence"], {
            "status": "refused", "reason": "Realized launch cohort differs from the selected cohort."})
        self.assertEqual(json.loads((self.case.output / "lifecycle.json").read_bytes())["launch_evidence"],
                         result["launch_evidence"])

    def test_controller_static_launch_needs_no_automatic_sample(self):
        self.case.launch_memory = {"cohort": "static", "sample_status": "unavailable", "sample": None}
        result, _ = self.engine_run()
        self.assertEqual(result["launch_evidence"], {"status": "satisfied"})
        self.assertIsNone(result["allocation_evidence"]["recent_cache"])

    def test_manifest_short_run_enforces_eleven_marks_from_the_worker_owned_engine_log(self):
        result, clones = self.engine_run()
        self.assertTrue(bench.acceptance_final_predicate(result))
        cuda = result["memory"]["cuda"]
        self.assertEqual((cuda["status"], cuda["errors"], cuda["serving_rows"], cuda["completed_request_rows"],
                          cuda["minimum_enforced_mib"]), ("satisfied", [], 1, 11, 731))
        self.assertEqual([row["stage"] for row in cuda["rows"]], ["serving"] + ["the completed request"] * 11)
        self.assertEqual(result["memory"]["cuda_floor_mib"], 256)
        lifecycle = json.loads((self.case.output / "lifecycle.json").read_bytes())
        self.assertEqual(lifecycle["memory"]["cuda"]["status"], "satisfied")
        self.assertEqual(len(lifecycle["memory"]["cuda"]["rows"]), 12)
        # The engine, not the worker, wrote the retained log; it is never rewritten.
        engine = (self.case.output / "engine.log").read_bytes()
        self.assertEqual(engine, ("strata serve: 761 MiB of VRAM free with everything loaded\r\n"
                                  + "strata trace: 731 MiB free after the completed request\r\n" * 11).encode())
        self.assertEqual(result["artifact_sha256"]["engine.log"], hashlib.sha256(engine).hexdigest())
        self.assertNotIn(b"Engine file logging held", engine)
        # The retained handle, then the final read's deny-writers handle, closed before the retained one.
        (_, retained), (_, final), *closes = self.engine_log_events
        self.assertEqual([kind for kind, _ in self.engine_log_events], ["open", "final", "close", "close"])
        self.assertEqual([handle for _, handle in closes], [final, retained])
        # The acceptance clone carries the trace, so the run is diagnostic and never normal-comparable.
        (clone, environment), = clones
        self.assertEqual(clone["env"]["STRATA_TRACE"], "1")
        self.assertEqual(clone["log"], str(self.case.output.resolve() / "engine.log"))
        self.assertNotIn("STRATA_TRACE", environment)
        for record in (result, json.loads((self.case.output / "provisional.json").read_bytes())):
            self.assertFalse(record["normal_comparison"]["eligible"])
            self.assertIn(TRACE_CONTROL, record["normal_comparison"]["controls"])
            self.assertIn(TRACE_EXCLUSION, record["normal_comparison"]["exclusions"])
        self.assertTrue(json.loads((self.case.output / "provisional.json").read_bytes())["timing_diagnostic"])

    def test_manifest_near_full_run_enforces_three_marks(self):
        cfg = json.loads(self.case.config.read_text())
        cfg["memory_policy"] = {"version": 1}
        self.case.config.write_text(json.dumps(cfg))
        self.args.acceptance_workload, self.args.cohort = "near-full", "automatic"
        manifest = self.root / "near-full-manifest.json"
        self.args.acceptance_manifest = None
        self.args.write_acceptance_manifest, self.args.manifest_not_after_unix = manifest, 1800010000.0
        self.assertEqual(bench.run(self.args), 0)
        self.args.acceptance_manifest = manifest
        self.args.write_acceptance_manifest = self.args.manifest_not_after_unix = None
        self.case.launch_memory = {"cohort": "automatic", "sample_status": "available",
                                   "sample": {"headroom_requested_gib": 4}}
        self.case.engine_startup = LaunchEvidenceTests.RECENT
        result, _ = self.engine_run(selection="near-full")
        self.assertTrue(bench.acceptance_final_predicate(result))
        cuda = result["memory"]["cuda"]
        self.assertEqual((cuda["status"], cuda["serving_rows"], cuda["completed_request_rows"]), ("satisfied", 1, 3))
        self.assertEqual(json.loads(manifest.read_text())["workload"], "near-full")

    def test_manifest_run_missing_one_request_mark_is_refused_evidence(self):
        result, _ = self.engine_run(lambda index: "" if index == 7 else
                                    "strata trace: 731 MiB free after the completed request\n", refused=True)
        cuda = result["memory"]["cuda"]
        self.assertEqual(cuda["status"], "refused")
        self.assertEqual(cuda["errors"], ["Completed-request CUDA marks: 10 of 11 required."])
        self.assertEqual(json.loads((self.case.output / "lifecycle.json").read_bytes())["memory"]["cuda"]["status"],
                         "refused")

    def test_manifest_run_without_serving_row_is_refused_evidence(self):
        self.case.engine_serving = "strata generate: loading\n"
        result, _ = self.engine_run(refused=True)
        self.assertEqual(result["memory"]["cuda"]["errors"], ["No enforced serving CUDA-free row was observed."])

    def assert_guard_stop(self, result, reason):
        self.assertEqual(result["memory"]["errors"][0], reason)  # The guard's first reason survives.
        self.assertIn("cancel-job", self.case.events)
        # The fake Job cancellation also advances the clock past the work deadline, so the
        # controller loop may record that edge before it re-checks the guard.
        self.assertIn(result["error"], ("RuntimeError: " + reason,
                                        "TimeoutError: Absolute worker deadline reached, including capture/finalization."))
        self.assertFalse(bench.acceptance_final_predicate(result))

    def test_manifest_run_255_mib_mark_stops_owner_and_keeps_the_primary_reason(self):
        result, _ = self.engine_run(lambda index: f"strata trace: {255 if index == 2 else 731} MiB free after "
                                    "the completed request\n", refused=True)
        reason = "Actual engine CUDA-free memory below 256 MiB at the completed request"
        self.assert_guard_stop(result, reason)
        self.assertFalse(bench.acceptance_final_predicate(result))
        cuda = result["memory"]["cuda"]
        self.assertEqual((cuda["status"], cuda["minimum_enforced_mib"], cuda["completed_request_rows"]),
                         ("refused", 255, 2))
        self.assertEqual(result["memory"]["errors"][0], reason)

    def test_manifest_run_unrecognized_free_memory_trace_stops_owner(self):
        result, _ = self.engine_run(lambda index: "strata trace: 731 MiB free before the request\n"
                                    if index == 3 else "strata trace: 731 MiB free after the completed request\n",
                                    refused=True)
        self.assert_guard_stop(result, "Engine CUDA-stage evidence failed: Unrecognized CUDA memory trace.")
        self.assertEqual(result["memory"]["cuda"]["errors"][0], "Unrecognized CUDA memory trace.")

    def test_manifest_run_api_key_in_engine_log_fails_worker_without_copying_it(self):
        secret = json.loads(self.case.config.read_text())["api_key"]
        result, _ = self.engine_run(lambda index: "strata trace: 731 MiB free after the completed request\n"
                                    + ("strata generate: echoed " + secret + "\n" if index == 4 else ""), refused=True)
        receipt = json.loads((self.case.output / "provisional.json").read_bytes())
        self.assertEqual(receipt["status"], "provisional_failed")
        self.assertEqual(receipt["artifact_error"], "ValueError: Engine log contains the API key.")
        self.assertNotIn("artifact_sha256", receipt)  # No digest is published for a refused log.
        self.assertEqual(result["worker_exit"], 1)
        self.assertFalse(bench.acceptance_final_predicate(result))
        self.assertIn(secret.encode(), (self.case.output / "engine.log").read_bytes())  # Retained, never rewritten.
        for name in ("provisional.json", "lifecycle.json", "controller.log", "server.log"):
            self.assertNotIn(secret.encode(), (self.case.output / name).read_bytes())

    def private_value_in_trace_stage(self, value):
        written = []
        def mark(index):
            line = "strata trace: 731 MiB free after the completed request\n"
            if index == 4:
                written.append(value())
                line += "strata trace: 731 MiB free after " + written[0] + "\n"  # A recognized stage.
            return line
        result, _ = self.engine_run(mark, refused=True)
        secret, = written
        self.assertIn(secret.encode(), (self.case.output / "engine.log").read_bytes())  # Retained, never rewritten.
        self.assert_guard_stop(result, "Engine CUDA-stage evidence failed: CUDA memory trace contains a private value.")
        cuda = result["memory"]["cuda"]
        self.assertEqual((cuda["status"], cuda["errors"][0]), ("refused", "CUDA memory trace contains a private value."))
        self.assertEqual(cuda["completed_request_rows"], 4)  # Rows before the refused one only.
        self.assertNotIn(secret, json.dumps(result))
        self.assertTrue((self.case.output / "lifecycle.json").is_file())
        for path in self.case.output.iterdir():
            if path.name != "engine.log":
                self.assertNotIn(secret.encode(), path.read_bytes(), path.name)

    def test_manifest_run_api_key_in_a_recognized_trace_stage_refuses_cuda_evidence(self):
        self.private_value_in_trace_stage(lambda: json.loads(self.case.config.read_text())["api_key"])

    def test_manifest_run_output_authority_in_a_recognized_trace_stage_refuses_cuda_evidence(self):
        self.private_value_in_trace_stage(lambda: json.loads(self.case.bootstrap_settings)["output_authority"])

    def test_manifest_run_escaped_over_budget_stage_refuses_without_copying_it(self):
        stage = "\x01" * 128  # The parent's 4096-row receipt counterexample, refused at its first row.
        result, _ = self.engine_run(lambda index: "strata trace: 731 MiB free after the completed request\n"
                                    + ("strata trace: 731 MiB free after " + stage + "\n" if index == 4 else ""),
                                    refused=True)
        self.assert_guard_stop(result, "Engine CUDA-stage evidence failed: CUDA memory trace exceeds its evidence bounds.")
        cuda = result["memory"]["cuda"]
        self.assertEqual((cuda["status"], cuda["errors"][0], cuda["completed_request_rows"]),
                         ("refused", "CUDA memory trace exceeds its evidence bounds.", 4))
        self.assertFalse(bench.acceptance_final_predicate(result))
        self.assertNotIn("\\u0001", json.dumps(result))
        lifecycle = (self.case.output / "lifecycle.json").read_bytes()
        self.assertNotIn(b"\\u0001", lifecycle)
        self.assertEqual(json.loads(lifecycle)["memory"]["cuda"]["status"], "refused")
        self.assertIn(stage.encode(), (self.case.output / "engine.log").read_bytes())

    def late_engine_writer(self, closes):
        # The engine is killed after the server exits; its append handle can outlive the wait.
        late, attempts, raised = [], [], []
        final, check = self.engine_log_api.open_final, bench.acceptance_engine_log
        def stopped():
            late.append(self.case.engine_stream)
            self.case.engine_stream = None
        def opening(path):
            try:
                return final(path)
            except OSError as exc:
                attempts.append(exc.winerror)
                if closes:
                    late[0].write("strata generate: late engine line\n")
                    late[0].close()  # The engine exits before the deadline.
                else:
                    with self.case.clock_lock:  # The engine outlives the worker's absolute deadline.
                        self.case.now = json.loads(self.case.bootstrap_settings)["work_deadline"] + 0.01
                raise
        def checked(*args):
            try:
                return check(*args)
            except BaseException as exc:
                raised.append(exc)
                raise
        try:
            with patch.object(self.case, "engine_close", side_effect=stopped), \
                    patch.object(self.engine_log_api, "open_final", side_effect=opening), \
                    patch.object(bench, "acceptance_engine_log", side_effect=checked):
                result, _ = self.engine_run(refused=not closes)
        finally:
            for stream in late:
                stream.close()
        self.assertEqual(attempts[0], 32, "A live writer must make the deny-writers open fail")
        return result, attempts, raised

    def test_late_engine_writer_is_awaited_and_its_bytes_are_bound(self):
        result, attempts, raised = self.late_engine_writer(closes=True)
        self.assertEqual((attempts, raised), ([32], []))
        engine = (self.case.output / "engine.log").read_bytes()
        self.assertTrue(engine.endswith(b"strata generate: late engine line\r\n"))
        self.assertEqual(result["artifact_sha256"]["engine.log"], hashlib.sha256(engine).hexdigest())
        self.assertEqual(result["memory"]["cuda"]["status"], "satisfied")
        self.assertTrue(bench.acceptance_final_predicate(result))

    def test_late_engine_writer_past_the_deadline_refuses(self):
        result, attempts, raised = self.late_engine_writer(closes=False)
        self.assertEqual(attempts, [32])
        self.assertEqual([(type(exc), str(exc)) for exc in raised],
                         [(TimeoutError, "Engine log writers remained open at the absolute deadline.")])
        self.assertNotIn("engine.log", result.get("artifact_sha256", {}))
        self.assertNotEqual(result["memory"]["cuda"]["status"], "satisfied")
        self.assertFalse(bench.acceptance_final_predicate(result))

    def mid_run(self, attack, index=3):
        target, done = self.case.output / "engine.log", []
        def mark(current):
            if current == index:
                with self.case.clock_lock:  # A third process acts between the guard's reads.
                    attack(target)
                done.append(True)
            return "strata trace: 731 MiB free after the completed request\n"
        return target, done, mark

    def test_engine_log_names_cannot_be_replaced_while_the_run_owns_them(self):
        import _winapi
        alias, foreign = self.root / "engine-alias.log", self.root / "engine-foreign.log"
        foreign.write_bytes(b"external foreign bytes")
        (self.root / "engine-foreign-directory").mkdir()
        outcomes = []
        def attack(target):
            for name, operation in (
                    ("unlink", target.unlink), ("rename", lambda: target.rename(self.root / "moved.log")),
                    ("replace", lambda: foreign.replace(target)), ("symlink", lambda: target.symlink_to(foreign)),
                    ("junction", lambda: _winapi.CreateJunction(str(self.root / "engine-foreign-directory"),
                                                                str(target)))):
                with self.assertRaises(OSError):
                    operation()
                outcomes.append(name)
            bench.os.link(target, alias)  # Allowed, but it names the same retained object.
            self.assertTrue(bench.os.path.samefile(target, alias))
            outcomes.append("hardlink")
        target, done, mark = self.mid_run(attack)
        result, _ = self.engine_run(mark)
        self.assertEqual(done, [True])
        self.assertEqual(outcomes, ["unlink", "rename", "replace", "symlink", "junction", "hardlink"])
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertEqual(result["memory"]["cuda"]["status"], "satisfied")
        self.assertEqual(foreign.read_bytes(), b"external foreign bytes")
        self.assertFalse(target.is_symlink())
        self.assertTrue(bench.os.path.samefile(target, alias))

    def test_mid_run_engine_log_truncation_stops_owner(self):
        def attack(target):
            with open(target, "r+b") as stream:
                stream.truncate(10)
        target, done, mark = self.mid_run(attack)
        result, _ = self.engine_run(mark, refused=True)
        self.assertEqual(done, [True])
        self.assert_guard_stop(result, "Engine CUDA-stage evidence failed: Engine log truncated.")
        self.assertEqual(result["memory"]["cuda"]["status"], "refused")

    def test_mid_run_rewrite_of_consumed_rows_refuses_at_final_binding(self):
        def attack(target):
            with open(target, "r+b") as stream:  # Same size; the worker's own final digest includes it.
                data = stream.read()
                stream.seek(0)
                stream.write(data.replace(b"761 MiB", b"999 MiB"))
        target, done, mark = self.mid_run(attack)
        result, _ = self.engine_run(mark, refused=True)
        self.assertEqual(done, [True])
        error = "Engine CUDA-stage evidence failed: Engine log changed after guard consumption."
        self.assertFalse(result["finalization"])
        self.assertEqual(result["finalization_error"], "RuntimeError: " + error)
        self.assertEqual(result["error"], error)
        self.assertEqual(result["memory"]["cuda"]["status"], "refused")
        worker = json.loads((self.case.output / "provisional.json").read_bytes())
        self.assertEqual(worker["artifact_sha256"]["engine.log"], hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertIn(b"999 MiB", target.read_bytes())

    def receipt_mac_refused(self, change, error, *, claim=False):
        receipt, injected = bench.acceptance_receipt, []
        status = "provisional_starting" if claim else "provisional_complete"

        def signed(snapshot, authority):
            data = receipt(snapshot, authority)
            if snapshot.get("status") == status:
                record = json.loads(data)
                change(record)
                injected.append(record)
                data = (json.dumps(record, indent=2) + "\n").encode("utf-8")
            return data

        with patch.object(bench, "acceptance_receipt", side_effect=signed):
            result = self.failed_with_controller_writes()[0] if claim else self.selected_refused()
        self.assertEqual(len(injected), 1)
        self.assertFalse(result["finalization"])
        self.assertFalse(bench.acceptance_final_predicate(result))
        if claim:
            self.assertIn(error, result["error"])
            self.assertNotIn("server", self.case.events)
            self.assertFalse((self.case.output / "controller-ack.json").exists())
        else:
            self.assertEqual(result["worker_exit"], 0)
            self.assertEqual(result["finalization_error"], "ValueError: " + error)
            self.assertFalse((self.case.output / "lifecycle.json").exists())

    def test_missing_final_receipt_mac_refuses(self):
        self.receipt_mac_refused(lambda record: record.pop("receipt_hmac_sha256"),
                                 "Worker receipt authentication is missing or malformed.")

    def test_malformed_final_receipt_mac_refuses(self):
        for value in ("Z" * 64, "A" * 64, "0" * 63, 1, None):
            with self.subTest(value=value):
                record = {"status": "provisional_complete", "receipt_hmac_sha256": value}
                with self.assertRaisesRegex(ValueError, "^Worker receipt authentication is missing or malformed\\.$"):
                    bench.acceptance_receipt_verify(record, "f" * 64)
        self.receipt_mac_refused(lambda record: record.update(receipt_hmac_sha256="not-a-mac"),
                                 "Worker receipt authentication is missing or malformed.")

    def test_mismatched_final_receipt_mac_refuses(self):
        def foreign_key(record):
            unsigned = {key: value for key, value in record.items() if key != "receipt_hmac_sha256"}
            record["receipt_hmac_sha256"] = hmac.new(b"\xee" * 32, json.dumps(
                unsigned, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256).hexdigest()
        self.receipt_mac_refused(foreign_key, "Worker receipt authentication differs.")

    def test_mismatched_claim_mac_is_never_acknowledged(self):
        self.receipt_mac_refused(lambda record: record.update(receipt_hmac_sha256="0" * 64),
                                 "Worker receipt authentication differs.", claim=True)

    def test_claim_read_before_controller_receipt_handle_refuses_without_path_read(self):
        output = self.root / "unheld-receipt"
        output.mkdir()
        (output / "provisional.json").write_bytes(b'{"status":"provisional_starting"}')
        owner = bench.AcceptanceOutputOwner(output, str(output))
        try:
            with patch.object(Path, "open", side_effect=AssertionError("receipt reopened by path")), \
                    self.assertRaisesRegex(RuntimeError, "^Controller receipt handle is not open\\.$"):
                bench.acceptance_claimed_output(output, str(output), "f" * 64, owner=owner)
        finally:
            owner.close()
        self.assertEqual(self.receipt_opens, [])
        with patch.object(bench.AcceptanceOutputOwner, "open_receipt", autospec=True) as skipped:
            result, writes = self.failed_with_controller_writes()
        self.assertTrue(skipped.called)
        self.assertEqual(writes, [])
        self.assertIn("Controller receipt handle is not open.", result["error"])
        self.assertNotIn("server", self.case.events)
        self.assertFalse((self.case.output / "controller-ack.json").exists())
        self.assertEqual(self.receipt_opens, [])

    def controller_receipt_failure(self, operation, *, final):
        native, injected = getattr(self.receipt_api, operation), []
        message = "synthetic controller receipt " + operation + " failure"

        def failed(*args):
            if not injected and (self.case.closed or not final):
                injected.append(True)
                if operation == "path":
                    return str(self.root / "different.json")
                if operation == "identity":
                    actual = native(*args)
                    return actual[0] + 1, actual[1]
                if operation == "attributes":
                    return native(*args) | 0x400  # FILE_ATTRIBUTE_REPARSE_POINT
                raise OSError(message)
            return native(*args)

        with patch.object(self.receipt_api, operation, side_effect=failed):
            result = self.selected_refused() if final else self.failed_with_controller_writes()[0]
        self.assertEqual(injected, [True], "The controller receipt failure must actually execute")
        self.assertFalse(result["finalization"])
        expected = message if operation in ("open", "read") else "Worker receipt identity differs."
        self.assertFalse(bench.acceptance_final_predicate(result))
        if final:
            self.assertIn("server", self.case.events)
            self.assertEqual(result["worker_exit"], 0)
            self.assertIn(expected, result["finalization_error"])
            self.assertFalse((self.case.output / "controller.log").exists())
        else:
            self.assertIn(expected, result["error"])
            self.assertNotIn("server", self.case.events)
            self.assertFalse((self.case.output / "controller-ack.json").exists())
        self.assertEqual(self.receipt_handles, set())

    def test_controller_receipt_open_failure_never_acknowledges(self):
        self.controller_receipt_failure("open", final=False)

    def test_controller_receipt_acquisition_read_failure_never_acknowledges(self):
        self.controller_receipt_failure("read", final=False)

    def test_controller_receipt_final_read_failure_cannot_pass(self):
        self.controller_receipt_failure("read", final=True)

    def test_controller_receipt_acquisition_path_mismatch_never_acknowledges(self):
        self.controller_receipt_failure("path", final=False)

    def test_controller_receipt_final_path_mismatch_cannot_pass(self):
        self.controller_receipt_failure("path", final=True)

    def test_controller_receipt_final_identity_mismatch_cannot_pass(self):
        self.controller_receipt_failure("identity", final=True)

    def test_controller_receipt_final_reparse_attribute_cannot_pass(self):
        self.controller_receipt_failure("attributes", final=True)

    def test_controller_receipt_close_failure_cannot_pass_and_keeps_errors_separate(self):
        close, failures = self.receipt_api.close, []

        def failed(handle):
            if self.case.closed and not failures:
                failures.append(handle)
                raise OSError("synthetic controller receipt close failure")
            return close(handle)

        with patch.object(self.receipt_api, "close", side_effect=failed), \
                self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        record, owner = caught.exception.record, caught.exception.output_owner
        try:
            self.assertEqual(len(failures), 1)
            self.assertEqual(self.receipt_handles, set(failures))
            self.assertEqual(owner.receipt, failures[0])
            self.assertIsNone(owner.handle, "The directory authority still closes")
            self.assertFalse(record["finalization"])
            self.assertNotIn("finalization_error", record)
            self.assertEqual(record["cleanup_error"],
                             "Controller output authority close failed: synthetic controller receipt close failure")
            self.assertFalse(bench.acceptance_final_predicate(record))
            self.assertTrue((self.case.output / "lifecycle.json").is_file())
        finally:
            owner.close()
        self.assertEqual(self.receipt_handles, set())

    def test_real_controller_receipt_follows_live_writer_and_denies_replacement_until_close(self):
        output = self.root / "controller-receipt"
        worker = bench.AcceptanceOutputOwner(output, str(output), create=True)
        receipt = bench.AcceptanceProvisionalFile(worker, 990)
        controller, moved, other = None, output / "moved.json", output / "other.json"
        try:
            receipt.create()
            receipt.publish(b'{"snapshot":"long first value"}')
            controller = bench.AcceptanceOutputOwner(output, str(output))
            controller.open_receipt()
            self.assertEqual(controller.read_receipt(), b'{"snapshot":"long first value"}')
            for data in (b'{"snapshot":2}', b'{"snapshot":"third"}'):
                receipt.publish(data)
                self.assertEqual(controller.read_receipt(), data)
            with self.assertRaises(PermissionError):
                receipt.path.open("r+b")  # The live worker writer still denies other writers.
            receipt.close()
            self.assertIsNone(receipt.handle)
            other.write_bytes(b"other bytes must survive")
            for attempt in (receipt.path.unlink, lambda: receipt.path.rename(moved), lambda: other.replace(receipt.path)):
                with self.assertRaises(PermissionError):
                    attempt()
            self.assertEqual(other.read_bytes(), b"other bytes must survive")
            self.assertFalse(moved.exists())
            self.assertEqual(controller.read_receipt(), b'{"snapshot":"third"}')
            self.assertEqual(controller.receipt_api.identity(controller.receipt), receipt.identity)
        finally:
            receipt.close()
            if controller is not None:
                controller.close()
            worker.close()
        self.assertIsNone(controller.receipt)
        receipt.path.rename(moved)
        self.assertEqual(moved.read_bytes(), b'{"snapshot":"third"}')
        self.assertEqual(self.receipt_handles, set())

    def server_under_held_receipt(self):
        # At each server start: does the controller hold exactly one receipt, on the live worker object?
        spawn, held = self.case.spawn, []
        def popen(argv, **kwargs):
            if "--engine" in argv:
                worker, handles = self.worker_args._provisional_file, list(self.receipt_handles)
                held.append(len(handles) == 1 and worker.handle is not None
                            and self.receipt_api.identity(handles[0]) == worker.identity)
            return spawn(argv, **kwargs)
        self.enterContext(patch.object(bench.subprocess, "Popen", side_effect=popen))
        return held

    def acquisitions(self):
        acquire, outcomes = bench.acceptance_acquire_output, []
        def acquiring(*args):
            try:
                owner = acquire(*args)
            except BaseException as exc:
                outcomes.append(type(exc).__name__)
                raise
            outcomes.append("held")
            return owner
        self.enterContext(patch.object(bench, "acceptance_acquire_output", side_effect=acquiring))
        return outcomes

    def assert_one_held_acquisition(self, outcomes):
        self.assertEqual(outcomes.count("held"), 1)
        self.assertEqual(outcomes[-1], "held", "No acquisition may follow acknowledgment publication")
        self.assertLessEqual(set(outcomes[:-1]), {"FileNotFoundError", "JSONDecodeError"})

    def test_acknowledgment_is_one_exclusive_create_after_the_receipt_is_held(self):
        held, outcomes = self.server_under_held_receipt(), self.acquisitions()
        create, created = bench.bounded_create, []
        def creating(path, *args, **kwargs):
            if threading.current_thread() is not self.case.worker_thread:
                created.append((path.name, len(self.receipt_handles)))
            return create(path, *args, **kwargs)
        with patch.object(bench, "bounded_create", side_effect=creating):
            result = self.case.invoke_selected()
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertEqual(created, [("controller-ack.json", 1), ("controller.log", 1), ("lifecycle.json", 1)])
        self.assertEqual(held, [True])
        self.assert_one_held_acquisition(outcomes)
        self.assertEqual(sorted(p.name for p in self.case.output.iterdir()), [
            "controller-ack.json", "controller.log", "engine.log", "lifecycle.json", "provisional.json", "server.log"])

    def acknowledgment_planted_before_create(self, content):
        # A foreign process creates controller-ack.json immediately before the controller's create.
        held, outcomes = self.server_under_held_receipt(), self.acquisitions()
        create, planted = bench.bounded_create, []
        def publish(path, data, *args, **kwargs):
            if path.name == "controller-ack.json" and not planted:
                planted.append(data if content is None else content)
                path.write_bytes(planted[0])
            return create(path, data, *args, **kwargs)
        with patch.object(bench, "bounded_create", side_effect=publish):
            result, writes = self.failed_with_controller_writes()
        self.assertEqual(len(planted), 1, "The foreign create must actually execute")
        self.assertTrue(result["error"].startswith("FileExistsError: "))
        self.assertEqual(result["finalization_error"], "ValueError: Controller output authority was not acquired.")
        self.assertFalse(bench.acceptance_final_predicate(result))
        self.assertEqual((self.case.output / "controller-ack.json").read_bytes(), planted[0])
        self.assertEqual(writes, [])
        self.assert_one_held_acquisition(outcomes)
        return held

    def test_preexisting_garbage_acknowledgment_is_terminal_and_never_starts_server(self):
        self.assertEqual(self.acknowledgment_planted_before_create(b"not json"), [])
        self.assertNotIn("server", self.case.events)

    def test_preexisting_wrong_acknowledgment_is_terminal_and_never_starts_server(self):
        self.assertEqual(self.acknowledgment_planted_before_create(b'{"acquired_sha256":"' + b"0" * 64 + b'"}'), [])
        self.assertNotIn("server", self.case.events)

    def test_preexisting_copied_acknowledgment_cannot_start_server_without_held_receipt(self):
        # Worst case beyond the boundary: the exact token, which no artifact exposes before this create.
        held = self.acknowledgment_planted_before_create(None)
        self.assertNotIn(False, held)

    def test_acknowledgment_cannot_be_replaced_or_unlinked_while_the_controller_writes_it(self):
        held, outcomes = self.server_under_held_receipt(), self.acquisitions()
        real_open, gate, attempts = Path.open, threading.Event(), []
        target, other = self.case.output / "acknowledgment-other", self.root / "acknowledgment-other"
        wrong = b'{"acquired_sha256":"' + b"0" * 64 + b'"}'
        other.write_bytes(b"other bytes must survive")

        def attack(path, point):
            for name, operation, error in (
                    ("unlink", path.unlink, PermissionError),
                    ("replace", lambda: other.replace(path), PermissionError),
                    ("rename", lambda: path.rename(target), PermissionError),
                    ("link", lambda: bench.os.link(other, path), FileExistsError)):
                with self.assertRaises(error):
                    operation()
                attempts.append((point, name))
            if point == "flushed":
                with open(path, "r+b") as stream:  # The controller's create shares writes.
                    stream.write(wrong)
                    stream.truncate()
                attempts.append((point, "overwrite"))
                gate.set()

        class Stream:
            def __init__(stream, path, inner):
                stream.path, stream.inner = path, inner
            def __enter__(stream):
                return stream
            def __exit__(stream, *exc):
                return stream.inner.__exit__(*exc)
            def write(stream, data):
                attack(stream.path, "opened")
                return stream.inner.write(data)
            def flush(stream):
                stream.inner.flush()
                attack(stream.path, "flushed")

        def opening(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if path == self.case.output / "controller-ack.json":
                if threading.current_thread() is self.case.worker_thread:
                    gate.wait(10)  # Scheduling only: the worker reads after the foreign overwrite.
                elif mode == "xb":
                    return Stream(path, real_open(path, *args, **kwargs))
            return real_open(path, *args, **kwargs)

        with patch.object(Path, "open", opening), contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        record = caught.exception.record
        self.assertEqual(attempts, [(point, name) for point in ("opened", "flushed")
                                    for name in ("unlink", "replace", "rename", "link")] + [("flushed", "overwrite")])
        self.assertEqual(other.read_bytes(), b"other bytes must survive")
        self.assertFalse(target.exists())
        self.assertEqual(record["worker_exit"], 1)
        self.assertEqual(self.worker_args._acceptance_admission["reason"], "Controller acknowledgment differs.")
        self.assertNotIn("server", self.case.events)
        self.assertEqual(held, [])
        self.assertFalse(bench.acceptance_final_predicate(record))
        self.assert_one_held_acquisition(outcomes)

    def acknowledgment_failure_after_publication(self, foreign=None, *, swap=False):
        # Review r7 P3/P5 adapted: the token is published and reaches the worker, a foreign
        # process optionally copies it ("left") or removes it ("cleared"), then the
        # controller's publication step fails.
        held, outcomes = self.server_under_held_receipt(), self.acquisitions()
        create, close, seen = bench.bounded_create, bench.AcceptanceProvisionalFile.close, {}

        def publish(path, data, *args, **kwargs):
            result = create(path, data, *args, **kwargs)
            if path.name == "controller-ack.json" and "copied" not in seen:
                seen["copied"] = path.read_bytes()
                for _ in range(2000):
                    if "server" in self.case.events:
                        break
                    threading.Event().wait(0.005)
                if foreign is not None:
                    path.unlink()  # The controller's create has closed; its name is now free.
                if foreign == "left":
                    path.write_bytes(seen["copied"])  # A foreign copy of the token stays.
                seen["outcomes"] = len(outcomes)
                raise OSError("synthetic acknowledgment failure after publication")
            return result

        def closing(file):
            active = file.handle is not None and file.path.name == "provisional.json"
            try:
                close(file)  # After the stop this close also reports the passed deadline.
            finally:
                if active and swap:
                    # The worker writer is gone; the controller still holds its receipt.
                    self.assertIsNone(file.handle)
                    seen["opens_at_worker_close"] = len(self.receipt_opens)
                    data = file.path.read_bytes()
                    with self.assertRaises(PermissionError):
                        file.path.unlink()
                    with self.assertRaises(FileExistsError):
                        with open(file.path, "xb") as stream:
                            stream.write(data)
                    seen["swap_denied"] = True

        with patch.object(bench, "bounded_create", side_effect=publish), \
                patch.object(bench.AcceptanceProvisionalFile, "close", closing):
            record, writes = self.failed_with_controller_writes()
        authority = json.loads(self.case.bootstrap_settings)["output_authority"]
        claim = json.loads((self.case.output / "provisional.json").read_bytes())["output_claim"]
        claim = {"path": claim["path"], "identity_sha256": claim["identity_sha256"]}  # Token field order.
        acquired = hashlib.sha256(json.dumps(["strata-acceptance-acquired/1", authority, claim],
                                             separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(json.loads(seen["copied"]), {"acquired_sha256": acquired})
        self.assertIn("server", self.case.events, "The copied token must actually reach the worker")
        self.assertEqual(held, [True])
        self.assertEqual(len(outcomes), seen["outcomes"], "Publication failure is never retried")
        self.assert_one_held_acquisition(outcomes)
        self.assertIn("OSError: synthetic acknowledgment failure after publication", record["error"])
        self.assertEqual(record["finalization_error"], "ValueError: Controller output authority was not acquired.")
        self.assertFalse(bench.acceptance_final_predicate(record))
        self.assertEqual(writes, [])
        self.assertEqual((self.case.output / "controller-ack.json").exists(), foreign != "cleared")
        if swap:
            self.assertTrue(seen["swap_denied"])
            self.assertEqual(len(self.receipt_opens), seen["opens_at_worker_close"])
        return record

    def test_copied_acknowledgment_left_after_publication_failure_cannot_pass(self):
        self.acknowledgment_failure_after_publication("left")

    def test_copied_acknowledgment_cleared_after_publication_failure_is_never_reacquired(self):
        self.acknowledgment_failure_after_publication("cleared")

    def test_publication_failure_keeps_receipt_on_worker_object_after_worker_close(self):
        # Review r7 P5 adapted: no reacquisition after the worker closes, and no swap while held.
        self.acknowledgment_failure_after_publication("cleared", swap=True)

    def receipt_reencoded(self, encode):
        # Review r7 P2: after the worker closes, rewrite the same record with different bytes.
        read_receipt, seen = bench.AcceptanceOutputOwner.read_receipt, {}
        def reading(owner):
            if self.case.closed and "data" not in seen:
                path = owner.path / "provisional.json"
                original = path.read_bytes()
                data = encode(original)
                self.assertNotEqual(data, original)
                self.assertEqual(json.loads(data), json.loads(original))
                with open(path, "r+b") as stream:  # The controller handle shares writes by necessity.
                    stream.write(data)
                    stream.truncate()
                seen.update(original=original, data=data)
            return read_receipt(owner)
        with patch.object(bench.AcceptanceOutputOwner, "read_receipt", reading):
            record = self.selected_refused()
        self.assertIn("data", seen, "The rewrite must actually execute")
        self.assertEqual(record["worker_exit"], 0)
        self.assertFalse(record["finalization"])
        self.assertEqual(record["finalization_error"], "ValueError: Worker receipt encoding differs.")
        self.assertFalse(bench.acceptance_final_predicate(record))
        self.assertNotIn("artifact_sha256", record)
        self.assertFalse((self.case.output / "lifecycle.json").exists())
        self.assertEqual((self.case.output / "provisional.json").read_bytes(), seen["data"])

    def test_compact_reencoded_receipt_refuses(self):
        self.receipt_reencoded(lambda data: json.dumps(json.loads(data), separators=(",", ":")).encode())

    def test_reordered_key_reencoded_receipt_refuses(self):
        # The worker's encoding sorts keys; any other order with the same layout must refuse.
        self.receipt_reencoded(lambda data: (json.dumps(dict(reversed(json.loads(data).items())), indent=2)
                                             + "\n").encode())

    def test_worker_receipt_bytes_are_the_sorted_indented_encoding_of_their_value(self):
        result = self.case.invoke_selected()
        self.assertTrue(bench.acceptance_final_predicate(result))
        data = (self.case.output / "provisional.json").read_bytes()
        self.assertEqual(data, (json.dumps(json.loads(data), indent=2, sort_keys=True) + "\n").encode())

    def test_escaped_character_reencoded_receipt_refuses(self):
        self.receipt_reencoded(lambda data: data.replace(b'"provisional_complete"', b'"provisional\\u005fcomplete"'))

    def test_trailing_whitespace_reencoded_receipt_refuses(self):
        self.receipt_reencoded(lambda data: data + b" \n")

    def lifecycle_foreign_entry(self, kind):
        # Review r7 P1 adapted: a foreign entry appears right after the controller's lifecycle create.
        create, removals, planted = bench.bounded_create, [], []
        external, foreign = self.root / "external-lifecycle.bin", self.case.output / "lifecycle.json.tmp"
        external_bytes = b"external foreign bytes, must survive"

        def publish(path, *args, **kwargs):
            result = create(path, *args, **kwargs)
            if path.name == "lifecycle.json" and kind is not None:
                if kind == "hardlink":
                    external.write_bytes(external_bytes)
                    bench.os.link(external, foreign)
                else:
                    foreign.write_bytes(b"attacker-created sole-link foreign file")
                planted.append(foreign.read_bytes())
            return result

        def removal(name):
            original = getattr(bench.os, name)
            def removing(*args, **kwargs):
                if threading.current_thread() is not self.case.worker_thread and any(
                        isinstance(path, (str, bench.os.PathLike)) and Path(path).parent == self.case.output
                        for path in args[:2]):
                    removals.append((name, *(Path(path).name for path in args[:2])))
                return original(*args, **kwargs)
            return patch.object(bench.os, name, side_effect=removing)

        with patch.object(bench, "bounded_create", side_effect=publish), \
                removal("unlink"), removal("remove"), removal("replace"), removal("rename"):
            if kind is None:
                result = self.case.invoke_selected()
            else:
                with self.assertRaises(bench.AcceptanceFailure) as caught:
                    self.case.invoke_selected()
                result = caught.exception.record
        self.assertEqual(removals, [], "The controller never deletes or replaces an output entry")
        lifecycle = json.loads((self.case.output / "lifecycle.json").read_bytes())
        self.assertEqual(lifecycle["artifact_sha256"], result["artifact_sha256"])
        if kind is None:
            self.assertTrue(bench.acceptance_final_predicate(result))
            self.assertFalse(foreign.exists())
            return
        self.assertEqual(len(planted), 1, "The foreign entry must actually be created")
        self.assertFalse(result["finalization"])
        self.assertEqual(result["finalization_error"], "RuntimeError: Unexpected or oversized retained artifacts.")
        self.assertFalse(bench.acceptance_final_predicate(result))
        self.assertEqual(foreign.read_bytes(), planted[0])
        if kind == "hardlink":
            self.assertEqual(external.read_bytes(), external_bytes)
            self.assertEqual(bench.os.stat(external).st_ino, bench.os.stat(foreign).st_ino)

    def test_passing_run_never_deletes_or_replaces_an_output_entry(self):
        self.lifecycle_foreign_entry(None)

    def test_foreign_file_after_lifecycle_create_survives_and_refuses(self):
        self.lifecycle_foreign_entry("file")

    def test_foreign_hard_link_after_lifecycle_create_survives_and_refuses(self):
        self.lifecycle_foreign_entry("hardlink")

    def test_final_worker_artifact_accounting_failure_cannot_pass(self):
        original_stat, injected = Path.stat, []
        def stat(path, *args, **kwargs):
            if path == self.case.output / "server.log" and self.case.code == 0:
                injected.append(True)
                raise OSError("synthetic final artifact accounting failure")
            return original_stat(path, *args, **kwargs)
        with patch.object(Path, "stat", stat), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        self.assertEqual(injected, [True])
        self.assertEqual(caught.exception.record["finalization_error"],
                         "OSError: synthetic final artifact accounting failure")
        self.assertFalse(bench.acceptance_final_predicate(caught.exception.record))
        self.assertEqual(self.directory_handles, set())

    def test_missing_or_changed_consumed_config_identity_still_refuses_before_server(self):
        self.args.acceptance_telemetry = False
        self.args.acceptance_physical_gib = self.args.acceptance_commit_gib = None
        self.args._acceptance_deadline = 990
        self.args._acceptance_manifest = self.manifest
        snapshot = bench.protection_snapshot
        def changed(paths):
            rows = snapshot(paths)
            if missing:
                return [row for row in rows if row["path"] != str(self.args.config.resolve())]
            next(row for row in rows if row["path"] == str(self.args.config.resolve()))["sha256"] = "f" * 64
            return rows
        for missing in (False, True):
            with self.subTest(missing=missing), patch.object(bench, "protection_snapshot", side_effect=changed), \
                    self.assertRaisesRegex(ValueError, "^Configuration changed while binding the consumed inputs\\.$"):
                bench._run_workload(self.args)
            self.assertNotIn("server", self.case.events)
            self.assertFalse(self.case.output.exists())

    def test_controller_output_mismatch_refuses_before_any_host_sample_or_child(self):
        self.args.output = self.root / "different-output"
        self.controller_refused("Manifest output differs.")
        self.assertFalse(self.args.output.exists())

    def test_controller_existing_output_is_preserved_without_child_or_evidence_writes(self):
        self.case.output.mkdir()
        marker = self.case.output / "keep.txt"
        marker.write_bytes(b"foreign output")
        self.controller_refused("Output directory already exists; choose a new directory.")
        self.assertEqual(list(self.case.output.iterdir()), [marker])
        self.assertEqual(marker.read_bytes(), b"foreign output")

    def test_controller_existing_resolved_output_alias_refuses_without_worker(self):
        self.case.output.mkdir()
        self.args.output = self.root / "absent/../output"
        self.controller_refused("Output directory already exists; choose a new directory.")
        self.assertEqual(list(self.case.output.iterdir()), [])
        self.assertFalse((self.root / "absent").exists())

    def test_controller_late_window_refuses_without_worker(self):
        self.changed_manifest(lambda value: value.update(not_after_unix=1800001000.0))
        self.case.now = 100.001
        self.controller_refused("Manifest window is too short.")

    def test_exact_expiry_equality_is_admitted_but_next_float_is_refused(self):
        self.changed_manifest(lambda value: value.update(not_after_unix=1800001000.0))
        self.assertIsNone(bench.acceptance_admission(self.args, self.manifest, 1800000100.0))
        with self.assertRaisesRegex(ValueError, "^Manifest window is too short\\.$"):
            bench.acceptance_admission(self.args, self.manifest, 1800000100.0000002)

    def test_controller_floor_mismatch_refuses_without_worker(self):
        self.args.acceptance_commit_gib = 8
        self.controller_refused("Manifest commit floor differs.")

    def test_controller_frozen_control_refuses_without_worker(self):
        self.args.cohort = "frozen-control"
        self.controller_refused("Frozen control cannot use an acceptance manifest.")

    def test_resolved_output_candidate_and_protect_aliases_match_writer_values(self):
        self.args.output = self.root / "missing/../output"
        self.args.protect_input = [str(self.root / "missing/.." / path.name) for path in self.extra]
        self.assertIsNone(bench.acceptance_admission(self.args, self.manifest, 1800000100.0))
        self.manifest["candidate_exe"] = str(self.saved)
        self.args.candidate_exe = self.root / "missing/../saved.exe"
        self.assertIsNone(bench.acceptance_admission(self.args, self.manifest, 1800000100.0))

    def test_noncanonical_manifest_output_alias_cannot_equal_resolved_argument(self):
        self.changed_manifest(lambda value: value.update(output=str(self.root / "missing/../output")))
        self.controller_refused("Manifest output differs.")

    def test_controller_exact_scalar_and_order_comparisons(self):
        cases = [("acceptance_workload", "near-full", "workload"),
                 ("cohort", "automatic", "cohort"), ("expected_context", 65536, "expected_context"),
                 ("port", 18081, "port"), ("acceptance_total_seconds", 1200, "total_seconds"),
                 ("acceptance_physical_gib", 4.0, "physical floor"),
                 ("acceptance_commit_gib", 4.0, "commit floor"),
                 ("candidate_exe", self.saved, "candidate_exe"),
                 ("protect_input", list(reversed(self.args.protect_input)), "protect_input")]
        for name, value, reason in cases:
            with self.subTest(name=name):
                original = getattr(self.args, name)
                setattr(self.args, name, value)
                try:
                    # Near-full requires automatic; test its workload equality directly.
                    if name == "acceptance_workload":
                        self.args.cohort = "automatic"
                    self.controller_refused("Manifest " + reason + " differs.")
                finally:
                    setattr(self.args, name, original)
                    self.args.cohort = "saved-static"

    def test_boolean_and_numeric_aliases_never_establish_boundary_equality(self):
        cases = [("port", 8081.0), ("expected_context", 131072.0),
                 ("acceptance_total_seconds", 900.0), ("acceptance_physical_gib", True),
                 ("acceptance_commit_gib", True), ("candidate_exe", False),
                 ("protect_input", tuple(self.args.protect_input))]
        for name, value in cases:
            with self.subTest(name=name):
                args = argparse.Namespace(**vars(self.args))
                setattr(args, name, value)
                with self.assertRaises(ValueError):
                    bench.acceptance_admission(args, self.manifest, 1800000100.0)
        for key, value in (("port", 8081.0), ("total_seconds", 900.0), ("expected_context", 131072.0)):
            manifest = bench.copy.deepcopy(self.manifest)
            manifest[key] = value
            with self.subTest(manifest=key), self.assertRaises(ValueError):
                bench.acceptance_admission(self.args, manifest, 1800000100.0)

    def test_first_controller_physical_sample_one_byte_below_floor_refuses_worker(self):
        self.first_sample_refused("available_gib", "physical")

    def test_first_controller_commit_sample_one_byte_below_floor_refuses_worker(self):
        self.first_sample_refused("commit_available_gib", "commit")

    def first_sample_refused(self, key, name):
        from tools import prepare_unsloth_q2 as prep
        snapshot = {"available_gib": 4, "commit_available_gib": 4}
        snapshot[key] -= 1 / 1024**3
        with patch.object(prep, "memory_snapshot", return_value=snapshot), \
                self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke()
        self.assertEqual(caught.exception.record["admission"], {
            "manifest_sha256": self.digest, "status": "refused",
            "reason": "Fresh " + name + " memory is below the manifest floor."})
        self.assertNotIn("create", self.case.events)
        self.assertFalse(self.case.output.exists())

    def test_worker_rejects_input_hash_changed_after_manifest_creation(self):
        self.extra[0].write_bytes(b"changed input, same reviewed path")
        self.worker_refused("Manifest inputs identity changed.")

    def test_worker_rejects_shard_mtime_changed_after_manifest_creation(self):
        stat = self.shard.stat()
        bench.os.utime(self.shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        self.worker_refused("Manifest size_bound identity changed.")

    def test_worker_rejects_pack_data_size_changed_after_manifest_creation(self):
        self.pack_data.write_bytes(b"different size")
        self.worker_refused("Manifest size_bound identity changed.")

    def test_worker_rejects_source_head_change(self):
        self.head = "f" * 40
        self.worker_refused("Manifest source HEAD differs.")

    def test_worker_fresh_physical_sample_one_byte_below_floor_refuses_server(self):
        self.worker_sample_refused("available_gib", "physical")

    def test_worker_fresh_commit_sample_one_byte_below_floor_refuses_server(self):
        self.worker_sample_refused("commit_available_gib", "commit")

    def worker_sample_refused(self, key, name):
        from tools import prepare_unsloth_q2 as prep
        def snapshot():
            self.case.samples += 1
            result = {"available_gib": 4, "commit_available_gib": 4}
            if threading.current_thread() is self.case.worker_thread:
                result[key] -= 1 / 1024**3
            return result
        with patch.object(prep, "memory_snapshot", side_effect=snapshot):
            self.worker_refused("Fresh " + name + " memory is below the manifest floor.")

    def test_worker_repeats_output_identity_check_independently(self):
        self.before_preflight = lambda args: setattr(args, "output", self.root / "other-output")
        self.worker_refused("Manifest output differs.")
        self.assertFalse((self.root / "other-output").exists())

    def test_worker_repeats_existing_output_check_independently(self):
        self.before_preflight = lambda args: args.output.mkdir()
        self.worker_refused("Output directory already exists; choose a new directory.")

    def test_worker_repeats_frozen_control_check_independently(self):
        self.before_preflight = lambda args: setattr(args, "cohort", "frozen-control")
        self.worker_refused("Frozen control cannot use an acceptance manifest.")

    def test_worker_repeats_window_check_independently(self):
        self.changed_manifest(lambda value: value.update(not_after_unix=1800001001.0))
        self.before_preflight = lambda args: setattr(self.case, "now", 101.001)
        self.worker_refused("Manifest window is too short.")

    def test_worker_repeats_port_check_independently(self):
        self.before_preflight = lambda args: setattr(args, "port", 18081)
        self.worker_refused("Manifest port differs.")

    def test_worker_repeats_cohort_check_independently(self):
        self.before_preflight = lambda args: setattr(args, "cohort", "automatic")
        self.worker_refused("Manifest cohort differs.")

    def test_worker_repeats_workload_check_independently(self):
        def change(args):
            args.acceptance_workload, args.cohort = "near-full", "automatic"
        self.before_preflight = change
        self.worker_refused("Manifest workload differs.")

    def test_worker_repeats_context_check_independently(self):
        self.before_preflight = lambda args: setattr(args, "expected_context", 65536)
        self.worker_refused("Manifest expected_context differs.")

    def test_worker_repeats_total_seconds_check_independently(self):
        self.before_preflight = lambda args: setattr(args, "acceptance_total_seconds", 1200)
        self.worker_refused("Manifest total_seconds differs.")

    def test_worker_repeats_candidate_check_independently(self):
        self.before_preflight = lambda args: setattr(args, "candidate_exe", str(self.saved))
        self.worker_refused("Manifest candidate_exe differs.")

    def test_worker_repeats_protect_order_check_independently(self):
        self.before_preflight = lambda args: args.protect_input.reverse()
        self.worker_refused("Manifest protect_input differs.")

    def test_worker_refuses_numeric_port_alias_even_when_value_matches(self):
        self.before_preflight = lambda args: setattr(args, "port", 8081.0)
        self.worker_refused("Manifest port differs.")

    def test_worker_refuses_numeric_duration_alias_even_when_value_matches(self):
        self.before_preflight = lambda args: setattr(args, "acceptance_total_seconds", 900.0)
        self.worker_refused("Manifest total_seconds differs.")

    def test_worker_refuses_numeric_input_size_alias_even_when_value_matches(self):
        real_identity = bench.file_identity
        def identity(path):
            result = real_identity(path)
            if path == self.extra[0] and self.case.started:
                result["bytes"] = float(result["bytes"])
            return result
        with patch.object(bench, "file_identity", side_effect=identity):
            self.worker_refused("Manifest inputs identity changed.")

    def test_worker_missing_memory_telemetry_refuses_server(self):
        from tools import prepare_unsloth_q2 as prep
        original = self.case.sample
        def snapshot():
            result = original()
            if threading.current_thread() is self.case.worker_thread:
                result.pop("commit_available_gib")
            return result
        with patch.object(prep, "memory_snapshot", side_effect=snapshot):
            self.worker_refused("Fresh commit memory is below the manifest floor.")

    def test_worker_expired_git_budget_never_starts_git_or_server(self):
        self.before_preflight = lambda args: setattr(args, "_acceptance_deadline", self.case.now)
        self.worker_refused("Absolute worker deadline reached during admission.")
        self.assertEqual(self.git_calls, [])

    def test_worker_git_timeout_refuses_before_server_start(self):
        with patch.object(bench.subprocess, "run", side_effect=bench.subprocess.TimeoutExpired(
                ["git", "rev-parse", "HEAD"], 0.5)):
            self.worker_refused("Command '['git', 'rev-parse', 'HEAD']' timed out after 0.5 seconds")

    def test_worker_git_timeout_uses_remaining_deadline(self):
        def shorten(args):
            args._acceptance_deadline = self.case.now + 0.5
        self.before_preflight = shorten
        # Reach the server seam only after the real preflight returns.
        with patch.object(bench, "assert_idle", side_effect=RuntimeError("stop after preflight")):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(bench.AcceptanceFailure):
                self.case.invoke()
        call = next(call for call in self.git_calls if call[0] == ["git", "rev-parse", "HEAD"])
        self.assertGreater(call[1]["timeout"], 0)
        self.assertLess(call[1]["timeout"], 0.5)
        self.assertNotIn("server", self.case.events)

    def test_deadline_crossing_after_preflight_refuses_immediately_before_server_popen(self):
        self.server_deadline_refused(0.001)

    def test_exact_deadline_after_preflight_refuses_immediately_before_server_popen(self):
        self.server_deadline_refused(0)

    def server_deadline_refused(self, offset):
        checks = []
        def idle(port):
            checks.append(port)
            if len(checks) == 2:
                deadline = self.worker_args._acceptance_deadline + offset
                self.case.now = deadline
                self.enterContext(patch.object(bench.time, "monotonic", return_value=deadline))
        with patch.object(bench, "assert_idle", side_effect=idle):
            self.worker_refused("Absolute worker deadline reached before server start.")
        self.assertEqual(checks, [8081, 8081])

    def test_missing_manifest_fails_worker_before_workload_or_output(self):
        self.before_preflight = lambda args: setattr(args, "acceptance_manifest", None)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(bench.AcceptanceFailure):
            self.case.invoke()
        self.assertEqual(self.worker_args._acceptance_admission, {
            "status": "refused", "reason": "Acceptance manifest is required."})
        self.assertIn("Acceptance admission refused: Acceptance manifest is required.", stderr.getvalue())
        self.assertNotIn("server", self.case.events)

    def test_manifest_cli_requires_telemetry_and_selected_workload(self):
        for telemetry, workload in ((False, None), (True, None), (False, "short")):
            with self.subTest(telemetry=telemetry, workload=workload):
                args = argparse.Namespace(**vars(self.args))
                args.acceptance_telemetry, args.acceptance_workload = telemetry, workload
                with self.assertRaises(ValueError):
                    bench.run(args)
        with patch.object(bench, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bench.main(["--config", str(self.args.config), "--output", str(self.args.output),
                "--acceptance-telemetry", "--acceptance-workload", "short", "--cohort", "saved-static",
                "--acceptance-physical-gib", "4", "--acceptance-commit-gib", "4",
                "--acceptance-manifest", str(self.args.acceptance_manifest)]), 0)
        self.assertEqual(run.call_args.args[0].acceptance_manifest, self.args.acceptance_manifest)
        self.assertNotIn("create", self.case.events)

    def test_real_readiness_without_manifest_avoids_all_host_and_process_work(self):
        self.args.acceptance_manifest = None
        with patch.object(bench, "acceptance_ready", self.real_ready), \
                self.assertRaisesRegex(RuntimeError, "Acceptance telemetry preparation held"):
            self.case.invoke()
        self.assertEqual(self.case.events, [])
        self.assertEqual(self.case.samples, 0)
        self.assertEqual(json.loads((self.case.output / "results.json").read_text())["status"], "held")


class StageCCompositionTests(unittest.TestCase):
    """Activate the public route with real Stage C owners and synthetic external effects."""

    def setUp(self):
        ready, preflight, protection = bench.acceptance_ready, bench.acceptance_preflight, bench.acceptance_protection
        admission = AcceptanceAdmissionTests()
        admission.setUp()
        self.addCleanup(admission.doCleanups)
        self.admission, self.case, self.args = admission, admission.case, admission.args
        # Restore the actual owners over the older isolated lifecycle fixture's stubs.
        self.enterContext(patch.object(bench, "acceptance_ready", ready))
        self.enterContext(patch.object(bench, "acceptance_preflight", preflight))
        self.enterContext(patch.object(bench, "acceptance_protection", protection))
        self.selection, self.floors = "short", (4, 4)

    def automatic(self, selection="near-full"):
        self.selection, self.floors = selection, (8, 8)
        cfg = json.loads(self.args.config.read_text())
        cfg["memory_policy"] = {"version": 1}
        self.args.config.write_text(json.dumps(cfg))
        self.args.acceptance_workload, self.args.cohort = selection, "automatic"
        self.args.acceptance_physical_gib = self.args.acceptance_commit_gib = 8
        manifest = self.case.root / "composition-manifest.json"
        self.args.acceptance_manifest = None
        self.args.write_acceptance_manifest, self.args.manifest_not_after_unix = manifest, 1800010000.0
        self.assertEqual(bench.run(self.args), 0)
        self.args.acceptance_manifest = manifest
        self.args.write_acceptance_manifest = self.args.manifest_not_after_unix = None
        self.case.launch_memory = {"cohort": "automatic", "sample_status": "available",
                                   "sample": {"headroom_requested_gib": 8}}
        self.case.engine_startup = LaunchEvidenceTests.AUTO + LaunchEvidenceTests.PINNED + LaunchEvidenceTests.RECENT

    def invoke(self):
        # The helper drives the real workload and guard; it must not fill missing_evidence.
        return self.case.invoke_selected(selection=self.selection, floors=self.floors, worker_evidence=True,
                                         cohort=self.args.cohort)

    def refused(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(bench.AcceptanceFailure) as caught:
            self.invoke()
        result = caught.exception.record
        self.assertFalse(bench.acceptance_final_predicate(result))
        self.assertNotIn("pending_stages", result)
        return result

    def assert_complete(self, result, requests):
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertEqual(result["admission"]["status"], "admitted")
        self.assertIs(result["protection"], True)
        self.assertEqual(result["protection_evidence"]["status"], "protected")
        self.assertEqual(result["memory"]["cuda"]["status"], "satisfied")
        self.assertEqual(result["memory"]["cuda"]["completed_request_rows"], requests)
        self.assertEqual(result["launch_evidence"], {"status": "satisfied"})
        self.assertGreaterEqual(result["job_drain_seconds"], 0)
        self.assertNotIn("pending_stages", result)
        worker = json.loads((self.case.output / "provisional.json").read_bytes())
        self.assertEqual(worker["admission"]["status"], "admitted")
        self.assertEqual(worker["missing_evidence"], [])
        self.assertEqual(len(worker["acceptance_cases"]), requests)
        lifecycle = json.loads((self.case.output / "lifecycle.json").read_bytes())
        self.assertNotIn("pending_stages", lifecycle)
        self.assertEqual(lifecycle["job_drain_seconds"], result["job_drain_seconds"])
        self.assertEqual(lifecycle["status"], "held")  # A saved receipt still cannot assert its own external exit.

    def test_readiness_requires_both_selected_workload_and_manifest(self):
        for selection, manifest, expected in ((None, None, False), ("short", None, False),
                (None, self.args.acceptance_manifest, False), ("short", self.args.acceptance_manifest, True),
                ("near-full", self.args.acceptance_manifest, True)):
            with self.subTest(selection=selection, manifest=manifest):
                self.assertIs(bench.acceptance_ready(argparse.Namespace(
                    acceptance_workload=selection, acceptance_manifest=manifest)), expected)
        self.assertFalse(bench.acceptance_ready(argparse.Namespace()))
        with self.assertRaisesRegex(RuntimeError, "^Acceptance manifest is required\\.$"):
            bench.acceptance_preflight(argparse.Namespace())

    def test_short_static_full_public_composition_passes(self):
        self.assert_complete(self.invoke(), 11)

    def test_near_full_automatic_full_public_composition_passes(self):
        self.automatic()
        self.assert_complete(self.invoke(), 3)

    def test_short_automatic_full_public_composition_passes(self):
        self.automatic("short")
        self.assert_complete(self.invoke(), 11)

    def test_final_predicate_requires_each_stage_c_evidence_field(self):
        complete = self.invoke()
        self.assert_complete(complete, 11)
        for key, path in (("admission", ("admission",)), ("protection", ("protection",)),
                          ("CUDA", ("memory", "cuda")), ("launch", ("launch_evidence",))):
            for replacement in (None, {"status": "refused"}):
                changed = bench.copy.deepcopy(complete)
                target = changed
                for part in path[:-1]:
                    target = target[part]
                if replacement is None:
                    target.pop(path[-1])
                else:
                    target[path[-1]] = replacement
                with self.subTest(evidence=key, replacement=replacement):
                    self.assertFalse(bench.acceptance_final_predicate(changed))

    def test_missing_controller_admission_refuses_before_resource_or_process_entry(self):
        manifest = json.loads(self.args.acceptance_manifest.read_bytes())
        manifest["not_after_unix"] = 1800000500.0
        self.args.acceptance_manifest.write_text(json.dumps(manifest))
        result = self.refused()
        self.assertEqual(result["admission"]["status"], "refused")
        self.assertEqual(self.case.samples, 0)
        self.assertEqual(self.case.events, [])

    def test_missing_worker_admission_refuses_before_server(self):
        assign = self.case.native.assign
        def changed(job, handle):
            assign(job, handle)
            self.admission.extra[0].write_bytes(b"changed after controller admission")
        with patch.object(self.case.native, "assign", side_effect=changed):
            result = self.refused()
        self.assertEqual(result["admission"]["status"], "admitted")
        self.assertEqual(result["worker_exit"], 1)
        self.assertEqual(result["error"], "RuntimeError: Worker exited unsuccessfully.")
        self.assertNotIn("server", self.case.events)

    def test_missing_post_exit_protection_refuses(self):
        capture, changed = self.case.native.capture, []
        def mutate(pipe):
            result = capture(pipe)
            if self.case.code == 0 and not changed:
                changed.append(True)
                self.admission.extra[0].write_bytes(b"changed after worker exit")
            return result
        with patch.object(self.case.native, "capture", side_effect=mutate):
            result = self.refused()
        self.assertEqual(changed, [True])
        self.assertIs(result["protection"], False)
        self.assertTrue(result["worker_complete"])
        self.assertEqual(result["memory"]["cuda"]["status"], "satisfied")
        self.assertEqual(result["launch_evidence"]["status"], "satisfied")

    def test_missing_cuda_request_row_refuses_an_otherwise_complete_run(self):
        self.case.engine_mark = lambda index: ("" if index == 11 else
            "strata trace: 731 MiB free after the completed request\n")
        result = self.refused()
        self.assertEqual(result["memory"]["cuda"]["errors"], ["Completed-request CUDA marks: 10 of 11 required."])
        self.assertTrue(result["worker_complete"])
        self.assertTrue(result["protection"])
        self.assertEqual(result["launch_evidence"]["status"], "satisfied")

    def test_missing_launch_record_refuses_static_worker_with_no_worker_owned_evidence_gap(self):
        self.case.launch_memory = None
        result = self.refused()
        self.assertEqual(result["launch_evidence"], {"status": "refused", "reason": "Launch memory evidence is unavailable."})
        self.assertTrue(result["worker_complete"])
        self.assertEqual(result["workload_evidence"]["missing_evidence"], [])

    def test_unavailable_automatic_sample_is_the_only_worker_owned_missing_evidence(self):
        self.automatic()
        self.case.launch_memory = {"cohort": "automatic", "sample_status": "unavailable", "sample": None}
        result = self.refused()
        self.assertEqual(result["workload_evidence"]["missing_evidence"], ["Server's once-refreshed automatic policy sample."])
        self.assertFalse(result["worker_complete"])
        self.assertEqual(result["launch_evidence"], {"status": "refused", "reason": "Automatic launch sample is unavailable."})

    def test_missing_automatic_startup_allocation_is_controller_owned(self):
        self.automatic()
        self.case.engine_startup = LaunchEvidenceTests.AUTO + LaunchEvidenceTests.PINNED
        result = self.refused()
        self.assertTrue(result["worker_complete"])
        self.assertEqual(result["workload_evidence"]["missing_evidence"], [])
        self.assertEqual(result["launch_evidence"], {
            "status": "refused", "reason": "Automatic launch recent-cache startup allocation is missing."})

    def drain(self, *, never=False):
        capture, active, sleep = self.case.native.capture, self.case.native.active, self.case.sleep
        polls, counts = [], []
        def captured(pipe):
            result = capture(pipe)
            if self.case.code == 0:
                polls.append(self.case.now)
            return result
        def counted(job):
            if self.case.code == 0 and not self.case.killed:
                value = int(never or len(polls) <= 3)
                counts.append((len(polls), value))
                return value
            return active(job)
        def slept(seconds):
            if self.case.code == 0:
                with self.case.clock_lock:
                    self.case.now += seconds
            else:
                sleep(seconds)
        self.enterContext(patch.object(self.case.native, "capture", side_effect=captured))
        self.enterContext(patch.object(self.case.native, "active", side_effect=counted))
        self.enterContext(patch.object(bench.time, "sleep", side_effect=slept))
        return polls, counts

    def test_job_count_one_for_three_post_exit_polls_then_zero_passes(self):
        polls, counts = self.drain()
        result = self.invoke()
        self.assert_complete(result, 11)
        self.assertEqual(len(polls), 4)
        for poll in (1, 2, 3):
            self.assertEqual({count for index, count in counts if index == poll}, {1})
        self.assertEqual({count for index, count in counts if index == 4}, {0})
        self.assertGreaterEqual(result["job_drain_seconds"], 0.75)
        self.assertLess(result["job_drain_seconds"], 2)
        self.assertFalse(self.case.killed)

    def test_job_never_empty_refuses_at_work_deadline_with_owned_cleanup(self):
        polls, counts = self.drain(never=True)
        result = self.refused()
        self.assertEqual(result["error"], "TimeoutError: Job members remained after worker exit.")
        self.assertEqual(result["worker_exit"], 0)
        self.assertTrue(all(count == 1 for _, count in counts))
        self.assertGreater(len(polls), 3)
        self.assertGreaterEqual(result["job_drain_seconds"],
                                result["work_deadline"] - result["worker_observed_exit_monotonic"])
        self.assertTrue(result["owned_cleanup"])
        self.assertTrue(result["within_deadline"])
        self.assertTrue(self.case.killed)
        self.assertLessEqual(result["memory"]["max_gap_seconds"], 1)

    def test_nonzero_exit_refuses_immediately_without_waiting_for_job_drain(self):
        polls, _ = self.drain(never=True)
        with patch.object(self.case.native, "exit_code", return_value=7):
            result = self.refused()
        self.assertEqual(result["worker_exit"], 7)
        self.assertEqual(result["error"], "RuntimeError: Worker exited unsuccessfully.")
        self.assertNotIn("job_drain_seconds", result)
        self.assertEqual(len(polls), 1)
        self.assertLess(result["worker_observed_exit_monotonic"], result["work_deadline"])


class ControllerTimingCompositionTests(unittest.TestCase):
    setUp = StageCCompositionTests.setUp
    invoke = StageCCompositionTests.invoke
    refused = StageCCompositionTests.refused

    def delayed(self, target, attribute, operation, *, raised=False, cleanup_error=False):
        timing = bench.ControllerTiming()
        original = getattr(target, attribute)
        controller_thread = threading.current_thread()
        injected = []
        def delayed(*args, **kwargs):
            if (not injected and threading.current_thread() is controller_thread and self.case.samples >= 2
                    and timing.current is not None and timing.current["operation"] == operation):
                injected.append(self.case.now)
                with self.case.clock_lock:
                    self.case.now += 1.1
                if raised:
                    raise OSError("synthetic controller operation failure")
            return original(*args, **kwargs)
        if cleanup_error:
            terminate = self.case.native.terminate_job
            def fail_cleanup(job):
                terminate(job)
                raise OSError("synthetic controller cleanup failure")
            self.enterContext(patch.object(self.case.native, "terminate_job", side_effect=fail_cleanup))
        with patch.object(bench, "ControllerTiming", return_value=timing), patch.object(target, attribute, delayed):
            result = self.refused()
        self.assertEqual(len(injected), 1, "The selected controller operation must actually execute")
        diagnostic = result["memory"]["controller_timing"]
        rows = diagnostic["recent_operations"]
        delayed_row = next(row for row in rows if row["operation"] == operation and row["elapsed_seconds"] >= 1.1)
        self.assertEqual(delayed_row["outcome"], "raised" if raised else "completed")
        self.assertIsNone(diagnostic["current_operation"])
        self.assertIsNotNone(diagnostic["refusal"])
        self.assertLessEqual(len(rows), 32)
        self.assertEqual(diagnostic, timing.summary())  # Stopped/final probes cannot erase the refusal interval.
        if raised:
            expected = ("Memory/process monitor failed: OSError" if operation == "discovery" else
                        "Engine CUDA-stage evidence failed: OSError" if operation in ("log_read", "log_parse") else
                        "OSError: synthetic controller operation failure")
            self.assertIn(expected, result["error"])
            self.assertEqual(diagnostic["refusal"]["last_completed_operation"], delayed_row)
        else:
            self.assertEqual(result["memory"]["errors"][0],
                             "Acceptance telemetry failed: Host sampling gap or read duration exceeded one second.")
            self.assertGreater(result["memory"]["max_gap_seconds"], 1)
            if operation != "host_read":
                row = result["memory"]["raw_samples"][-1]
                self.assertLess(row["finished_monotonic"] - row["started_monotonic"], 0.1)
            self.assertIn("host_read", {row["operation"] for row in rows})
        if operation == "sleep":
            self.assertEqual(delayed_row["requested_seconds"], 0.25)
            self.assertGreater(delayed_row["finished_monotonic"] - delayed_row["started_monotonic"], 1.1)
        if cleanup_error:
            self.assertIn("synthetic controller cleanup failure", result["cleanup_error"])
        return result

    def test_capture_delay_is_retained_at_unchanged_gap_refusal(self):
        self.delayed(self.case.native, "capture", "capture")

    def test_capture_bookkeeping_delay_is_distinct_from_memory_read(self):
        self.delayed(bench.BoundedCapture, "write", "capture")

    def test_poll_delay_is_retained_at_unchanged_gap_refusal(self):
        self.delayed(bench.OwnedProcess, "poll", "poll")

    def test_discovery_delay_is_retained_before_memory_read_start(self):
        self.delayed(bench.OwnedProcess, "discover", "discovery")

    def test_output_acquisition_delay_is_retained(self):
        self.delayed(bench, "acceptance_acquire_output", "output_acquisition")

    def test_log_filesystem_delay_is_separate_from_parsing(self):
        self.delayed(Path, "open", "log_read")

    def test_log_parsing_delay_is_separate_from_filesystem_read(self):
        self.delayed(bench.MemoryGuard, "consume_engine_log", "log_parse")

    def test_host_read_delay_remains_a_read_duration_refusal(self):
        from tools import prepare_unsloth_q2 as preparation
        self.delayed(preparation, "memory_snapshot", "host_read")

    def test_sleep_requested_duration_and_actual_resume_are_retained(self):
        self.delayed(bench.time, "sleep", "sleep")

    def test_controller_exception_retains_last_raised_operation_and_primary_error(self):
        self.delayed(self.case.native, "capture", "capture", raised=True)

    def test_discovery_exception_retains_guard_primary_error_and_operation(self):
        self.delayed(bench.OwnedProcess, "discover", "discovery", raised=True)

    def test_log_read_exception_retains_fixed_error_and_operation(self):
        self.delayed(Path, "open", "log_read", raised=True)

    def test_log_parse_exception_retains_fixed_error_and_operation(self):
        self.delayed(bench.MemoryGuard, "consume_engine_log", "log_parse", raised=True)

    def test_gap_refusal_diagnostics_survive_failing_owned_cleanup(self):
        self.delayed(self.case.native, "capture", "capture", cleanup_error=True)

    def test_complete_run_has_bounded_diagnostics_only_in_controller_receipts(self):
        result = self.invoke()
        self.assertTrue(bench.acceptance_final_predicate(result))
        timing = result["memory"]["controller_timing"]
        self.assertTrue(timing["clock_valid"])
        self.assertIsNone(timing["refusal"])
        self.assertLessEqual(len(timing["recent_operations"]), 32)
        self.assertIn("sleep", {row["operation"] for row in timing["recent_operations"]})
        lifecycle = json.loads((self.case.output / "lifecycle.json").read_bytes())
        self.assertIn("controller_timing", lifecycle["memory"])
        worker = json.loads((self.case.output / "provisional.json").read_bytes())
        self.assertNotIn("controller_timing", json.dumps(worker))
        self.assertFalse(any("controller_timing" in row for row in result["memory"]["raw_samples"]))


class AcceptanceProtectionTests(unittest.TestCase):
    """The controller's real post-exit protection check against the admitted manifest."""

    def setUp(self):
        real_protection = bench.acceptance_protection
        self.real_protection = real_protection
        self.admission = admission = AcceptanceAdmissionTests()
        admission.setUp()
        self.addCleanup(admission.doCleanups)
        self.case, self.args, self.manifest = admission.case, admission.args, admission.manifest
        self.replacement = b"SYNTHETIC INPUT, NEVER EXECUTED"
        self.assertEqual(len(self.replacement), len(b"synthetic input, never executed"))
        self.before_protection, self.results = None, []

        def protection(args, worker_record):
            if self.before_protection:
                self.before_protection(args, worker_record)
            result = real_protection(args, worker_record)
            self.results.append(result)
            return result

        self.enterContext(patch.object(bench, "acceptance_protection", side_effect=protection))

    def row(self, role, index=0):
        rows = [row for key in ("inputs", "size_bound") for row in self.manifest[key] if row["role"] == role]
        return rows[index]

    def worker_rows(self):
        # Spelled independently of the code under test: the admitted identity without its role.
        return [{"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]} for row in self.manifest["inputs"]]

    def direct(self, worker_record=None):
        args = argparse.Namespace(**vars(self.args), _controller_manifest_sha256=self.admission.digest)
        if worker_record is None:
            worker_record = {"protection": {"before": self.worker_rows(), "after": self.worker_rows()}}
        return self.real_protection(args, worker_record), args._acceptance_protection_evidence

    def refused(self, *mismatches):
        with self.assertRaises(bench.AcceptanceFailure) as caught:
            self.case.invoke_selected()
        record = caught.exception.record
        self.assertEqual(self.results, [False])
        self.assertIs(record["protection"], False)
        self.assertFalse(bench.acceptance_final_predicate(record))
        # Protection is the only failing qualifier, and it refused without an exception.
        self.assertTrue(bench.acceptance_final_predicate(dict(record, protection=True)))
        self.assertNotIn("finalization_error", record)
        evidence = record["protection_evidence"]
        self.assertEqual(evidence["status"], "refused")
        self.assertEqual(evidence["mismatches"], list(mismatches))
        saved = json.loads((self.case.output / "lifecycle.json").read_text())
        self.assertEqual(saved["protection_evidence"], evidence)
        self.assertFalse(saved["protection"])
        self.assertNotIn(self.replacement.decode(), json.dumps(evidence))
        self.assertNotIn("synthetic input", json.dumps(evidence))
        return record

    def test_green_exact_current_and_worker_evidence_protects(self):
        result = self.case.invoke_selected()
        self.assertTrue(bench.acceptance_final_predicate(result))
        self.assertEqual(self.results, [True])
        self.assertIs(result["protection"], True)
        evidence = {"status": "protected", "mismatches": [], "manifest_sha256": self.admission.digest}
        self.assertEqual(result["protection_evidence"], evidence)
        self.assertEqual(json.loads((self.case.output / "lifecycle.json").read_text())["protection_evidence"], evidence)

    def test_admitted_input_changed_after_worker_exit_refuses(self):
        target = self.admission.extra[0]
        self.before_protection = lambda args, worker_record: target.write_bytes(self.replacement)
        self.refused({"comparison": "current-identity", "role": "protect-input", "path": str(target.resolve())})

    def test_size_bound_shard_mtime_changed_after_worker_exit_refuses(self):
        shard = self.admission.shard
        def touch(args, worker_record):
            stat = shard.stat()
            bench.os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
        self.before_protection = touch
        self.refused({"comparison": "current-stat", "role": "model-shard", "path": str(shard.resolve())})

    def test_worker_before_list_mismatch_refuses(self):
        def claim(args, worker_record):
            worker_record["protection"]["before"][1]["sha256"] = "0" * 64
        self.before_protection = claim
        row = self.manifest["inputs"][1]
        self.refused({"comparison": "worker-before", "role": row["role"], "path": row["path"]})

    def test_worker_after_list_mismatch_refuses(self):
        def claim(args, worker_record):
            worker_record["protection"]["after"].pop()
        self.before_protection = claim
        row = self.manifest["inputs"][-1]
        self.refused({"comparison": "worker-after", "role": row["role"], "path": row["path"]})

    def test_worker_reordered_snapshots_pass_worker_checks_but_refuse_manifest_order(self):
        snapshot = bench.protection_snapshot
        def reordered(paths):
            rows = snapshot(paths)
            if threading.current_thread() is self.case.worker_thread:
                rows[0], rows[1] = rows[1], rows[0]
            return rows
        with patch.object(bench, "protection_snapshot", side_effect=reordered):
            first, second = self.manifest["inputs"][:2]
            self.refused(*[{"comparison": side, "role": row["role"], "path": row["path"]}
                           for side in ("worker-before", "worker-after") for row in (first, second)])

    def test_missing_manifest_after_worker_exit_refuses(self):
        self.before_protection = lambda args, worker_record: args.acceptance_manifest.unlink()
        record = self.refused()
        self.assertEqual(record["protection_evidence"]["reason"], "Cannot read a valid acceptance manifest.")

    def test_rebaselined_manifest_after_worker_exit_refuses(self):
        target = self.admission.extra[0]
        def rebaseline(args, worker_record):
            target.write_bytes(self.replacement)
            manifest = json.loads(args.acceptance_manifest.read_text())
            for row in manifest["inputs"]:
                if row["path"] == str(target.resolve()):
                    row["sha256"] = hashlib.sha256(self.replacement).hexdigest()
            args.acceptance_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        self.before_protection = rebaseline
        record = self.refused()
        self.assertEqual(record["protection_evidence"]["reason"],
                         "Manifest differs from the controller's admitted manifest.")

    def test_direct_green_and_legacy_no_manifest_control(self):
        self.assertEqual(self.direct(), (True, {"status": "protected", "mismatches": [],
                                                "manifest_sha256": self.admission.digest}))
        exact = {"protection": {"before": self.worker_rows(), "after": self.worker_rows()}}
        legacy = argparse.Namespace(**dict(vars(self.args), acceptance_manifest=None))
        self.assertIs(self.real_protection(legacy, exact), False)
        self.assertEqual(legacy._acceptance_protection_evidence,
                         {"status": "refused", "mismatches": [], "reason": "Acceptance manifest is required."})
        self.assertIs(self.real_protection(None, None), False)
        unadmitted = argparse.Namespace(**vars(self.args))
        self.assertIs(self.real_protection(unadmitted, exact), False)
        self.assertEqual(unadmitted._acceptance_protection_evidence["reason"],
                         "Manifest differs from the controller's admitted manifest.")

    def test_direct_malformed_worker_structure_and_type_aliases_refuse(self):
        first = self.manifest["inputs"][0]
        def every(side):
            return [{"comparison": "worker-" + side, "role": None, "path": None}]
        def at(side, row=first):
            return [{"comparison": "worker-" + side, "role": row["role"], "path": row["path"]}]
        cases = {
            "missing protection": ({}, every("before") + every("after")),
            "protection not object": ({"protection": [self.worker_rows()]}, every("before") + every("after")),
            "record not object": ([], every("before") + every("after")),
            "before missing": ({"protection": {"after": self.worker_rows()}}, every("before")),
            "after not list": ({"protection": {"before": self.worker_rows(), "after": {}}}, every("after")),
        }
        for name, (record, expected) in cases.items():
            with self.subTest(name):
                result, evidence = self.direct(record)
                self.assertIs(result, False)
                self.assertEqual(evidence["mismatches"], expected)
        mutations = {
            "float bytes alias": lambda row: row.update(bytes=float(row["bytes"])),
            "role retained": lambda row: row.update(role=first["role"]),
            "uppercase digest": lambda row: row.update(sha256=row["sha256"].upper()),
            "missing digest": lambda row: row.pop("sha256"),
            "not an object": lambda row: row.clear(),
        }
        for side in ("before", "after"):
            for name, mutate in mutations.items():
                with self.subTest(side=side, mutation=name):
                    rows = self.worker_rows()
                    mutate(rows[0])
                    protection = {"before": self.worker_rows(), "after": self.worker_rows(), side: rows}
                    result, evidence = self.direct({"protection": protection})
                    self.assertIs(result, False)
                    self.assertEqual(evidence["mismatches"], at(side))
            with self.subTest(side=side, mutation="foreign extra row"):
                foreign = str(self.admission.root / "foreign-unrelated.bin")
                rows = self.worker_rows() + [{"path": foreign, "bytes": 1, "sha256": "0" * 64}]
                protection = {"before": self.worker_rows(), "after": self.worker_rows(), side: rows}
                result, evidence = self.direct({"protection": protection})
                self.assertIs(result, False)
                self.assertEqual(evidence["mismatches"], every(side))
                self.assertNotIn("foreign-unrelated", json.dumps(evidence))

    def test_direct_missing_linked_unreadable_and_grown_inputs_refuse(self):
        target = self.admission.extra[1]
        identity = {"comparison": "current-identity", "role": "protect-input", "path": str(target.resolve())}
        real_is_symlink, real_open = Path.is_symlink, Path.open
        def unreadable(path, *args, **kwargs):
            if path == target:
                raise PermissionError("synthetic sharing violation")
            return real_open(path, *args, **kwargs)
        with self.subTest("linked"), patch.object(Path, "is_symlink", lambda path: path == target or real_is_symlink(path)):
            self.assertEqual(self.direct(), (False, {"status": "refused", "mismatches": [identity],
                                                      "manifest_sha256": self.admission.digest}))
        with self.subTest("unreadable"), patch.object(Path, "open", unreadable):
            self.assertEqual(self.direct()[1]["mismatches"], [identity])
        original = target.read_bytes()
        copy = self.admission.root / "identical-copy.txt"
        copy.write_bytes(original)
        def swapped(path, *args, **kwargs):
            return real_open(copy if path == target else path, *args, **kwargs)
        with self.subTest("swapped object"), patch.object(Path, "open", swapped):
            self.assertEqual(self.direct()[1]["mismatches"], [identity])
        reads = []
        def counted(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            if path == target:
                read = stream.read
                stream.read = lambda size=-1: reads.append(read(size)) or reads[-1]
            return stream
        with self.subTest("grown"), patch.object(Path, "open", counted):
            target.write_bytes(original + b"x" * 1024**2)
            self.assertEqual(self.direct()[1]["mismatches"], [identity])
            self.assertEqual(sum(map(len, reads)), len(original) + 1)
        with self.subTest("missing"):
            target.unlink()
            self.assertEqual(self.direct()[1]["mismatches"], [identity])
        target.write_bytes(original)
        self.assertEqual(self.direct()[0], True)

    def test_direct_linked_pack_parent_refuses_every_pack_row(self):
        pack, moved = self.admission.pack, self.admission.root / "pack-moved"
        pack.rename(moved)
        pack.symlink_to(moved, target_is_directory=True)
        result, evidence = self.direct()
        self.assertIs(result, False)
        expected = [{"comparison": "current-identity" if key == "inputs" else "current-stat",
                     "role": row["role"], "path": row["path"]}
                    for key in ("inputs", "size_bound") for row in self.manifest[key]
                    if Path(row["path"]).parent == pack]
        self.assertEqual(len(expected), 3)
        self.assertEqual(evidence["mismatches"], expected)


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
                    self.assertFalse(bench.acceptance_final_predicate(case.invoke_selected(held=True)))
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
                    self.assertTrue(record["timing_diagnostic"])  # The acceptance trace is always active.
                    self.assertEqual(other_active_diagnostics(record["normal_comparison"]),
                                     [key] if explicit and (key in presence or value == "1") else [])
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
                result = case.invoke_selected(selection=selection, held=True)
                self.assertFalse(bench.acceptance_final_predicate(result))
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
                    # The acceptance clone's own trace replaces any saved STRATA_TRACE value.
                    self.assertEqual(evidence.pop("STRATA_TRACE"), "1")
                    if key != "STRATA_TRACE":
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
                evidence = bench.safe_engine_environment(cfg, inherited, acceptance=True)
                self.assertEqual(evidence.pop("STRATA_TRACE"), "1")
                self.assertEqual(bench.diagnostic_environment(evidence), gate == "1")
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
            self.assertTrue(result["timing_diagnostic"])  # The acceptance trace is always active.
            self.assertEqual(other_active_diagnostics(result["normal_comparison"]),
                             sorted(keys) if explicit else [])
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
                    self.assertEqual(bench.active_diagnostics(self.effective(case)),
                                     ["STRATA_VERIFY_PROFILE", "STRATA_TRACE"])
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
                    self.assertTrue(result["timing_diagnostic"])  # The acceptance trace is always active.
                    self.assertTrue(result["acceptance_workload_passed"])
                    classified = bench.normal_comparison_environment(case.clones[0], case.server_environments[0])
                    self.assertTrue(classified["complete"])
                    self.assertIn(TRACE_CONTROL, classified["controls"])
                    self.assertEqual(bool(other_active_diagnostics(classified)), diagnostic)
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
                for evidence in (before, after):  # Only the acceptance trace excludes the clone.
                    self.assertFalse(evidence["eligible"])
                    self.assertEqual(evidence["exclusions"], [TRACE_EXCLUSION])
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
            result = case.invoke_selected(selection="near-full", held=True)
        self.assertEqual(len(observed), 1)
        worker = json.loads((case.output / "provisional.json").read_text())
        for record in (worker, result):
            evidence = record["normal_comparison"]
            self.assertFalse(evidence["eligible"])
            self.assertFalse(evidence["complete"])
            self.assertIsNone(evidence["environment_sha256"])
            self.assertEqual(evidence["controls"], observed[0][0]["controls"])
            self.assertEqual(evidence["exclusions"], [TRACE_EXCLUSION, {"reason": "automatic-environment-unbound"}])
        self.assertEqual(worker["status"], "provisional_complete")
        self.assertTrue(worker["acceptance_workload_passed"])
        self.assertTrue(worker["timing_diagnostic"])
        self.assertTrue(result["worker_complete"])
        self.assertFalse(bench.acceptance_final_predicate(result))  # The isolated fixture has no manifest.
        self.assertEqual(json.loads((case.output / "lifecycle.json").read_text())["status"], "held")
        case.doCleanups()
        self.assertFalse(bench.acceptance_ready(argparse.Namespace()))

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
                    self.assertFalse(classified["eligible"])  # The acceptance trace always excludes it.
                    self.assertIn(TRACE_EXCLUSION, classified["exclusions"])
                    self.assertEqual([item["reason"] for item in classified["exclusions"] if item != TRACE_EXCLUSION],
                                     [] if reason is None else [reason])
                    self.assertTrue(classified["complete"])
                    self.assertIsNotNone(classified["environment_sha256"])
                    self.assertFalse(evidence["complete"])
                    self.assertFalse(evidence["eligible"])
                    self.assertIsNone(evidence["environment_sha256"])
                    self.assertEqual(evidence["controls"], classified["controls"])
                    self.assertEqual(evidence["exclusions"], classified["exclusions"] +
                                     [{"reason": "automatic-environment-unbound"}])
                    self.assertTrue(result["timing_diagnostic"])
                    self.assertEqual(bool(other_active_diagnostics(classified)), diagnostic)
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
                self.assertFalse(classified["eligible"])
                self.assertEqual(classified["exclusions"], [TRACE_EXCLUSION])
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
        self.assertFalse(result["normal_comparison"]["eligible"])
        self.assertEqual(result["normal_comparison"]["exclusions"], [TRACE_EXCLUSION])
        self.assertTrue(result["timing_diagnostic"])
        effective = self.effective(case)
        for key, value in controls.items():
            self.assertEqual(effective[key], value)
        self.assertIn(TRACE_CONTROL, result["normal_comparison"]["controls"])
        self.assertEqual({item["kind"] for item in result["normal_comparison"]["controls"]
                          if item != TRACE_CONTROL}, {"reviewed-normal"})

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
                    "control": "sha256:" + hashlib.sha256(key.encode()).hexdigest()}, TRACE_EXCLUSION])
                self.assertTrue(result["acceptance_workload_passed"])
                self.assertEqual(result["status"], "complete")
                self.assertTrue(result["timing_diagnostic"])
                self.assertEqual(other_active_diagnostics(result["normal_comparison"]), [])
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
                self.assertTrue(result["timing_diagnostic"])
                self.assertTrue(evidence["complete"])
                self.assertFalse(evidence["eligible"])
                self.assertEqual(evidence["exclusions"], [TRACE_EXCLUSION])
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
            self.assertFalse(result["normal_comparison"]["eligible"])
            self.assertEqual(result["normal_comparison"]["exclusions"] == [TRACE_EXCLUSION], not explicit)
            self.assertTrue(result["timing_diagnostic"])
            self.assertEqual(bool(other_active_diagnostics(result["normal_comparison"])), explicit)
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
            self.assertFalse(classified["eligible"])
            self.assertEqual(classified["exclusions"] == [TRACE_EXCLUSION], not unknown)
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
        self.assertFalse(result["normal_comparison"]["eligible"])
        self.assertEqual(result["normal_comparison"]["exclusions"], [TRACE_EXCLUSION])

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
                    result = case.invoke_selected(mutate, held=True)
                self.assertFalse(bench.acceptance_final_predicate(result))
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
                    result = case.invoke_selected(held=True)
                self.assertFalse(bench.acceptance_final_predicate(result))
                self.assertTrue(result["worker_complete"])
                self.assertTrue(result["normal_comparison"]["complete"])
                self.assertFalse(result["normal_comparison"]["eligible"])
                self.assertIn(TRACE_EXCLUSION, result["normal_comparison"]["exclusions"])
                self.assertEqual(result["normal_comparison"]["exclusions"] == [TRACE_EXCLUSION], not unknown)
                self.assertEqual(json.loads((case.output / "lifecycle.json").read_text())["status"], "held")
            finally:
                case.doCleanups()
        self.assertFalse(bench.acceptance_ready(argparse.Namespace()))
        self.assertFalse(bench.acceptance_protection(None, None))
        with self.assertRaisesRegex(RuntimeError, "^Acceptance manifest is required\\.$"):
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


def retained_publication_deadline_test(boundary, edge, offset):
    def test(self):
        self.retained_deadline_failure(boundary, edge, offset)
    return test


def retained_lifetime_deadline_test(phase, edge, offset):
    def test(self):
        self.retained_create_or_close_deadline_failure(phase, edge, offset)
    return test


# Each method gets fresh real Windows files and the complete controller/worker
# composition. Equality and late return are independent cases at both edges.
for _boundary, _step in enumerate((
        "owner_before", "symlink_before", "junction_before", "resolve_before", "path_before", "identity_before",
        "seek_start", "write", "seek_length", "truncate", "flush",
        "owner_after", "symlink_after", "junction_after", "resolve_after", "path_after", "identity_after"), 1):
    for _edge in ("before", "after"):
        for _time, _offset in (("equality", 0.0), ("late", 0.001)):
            setattr(AcceptanceAdmissionTests, "test_retained_deadline_" + _step + "_" + _edge + "_" + _time,
                    retained_publication_deadline_test(_boundary, _edge, _offset))
for _phase in ("create", "close"):
    for _edge in ("before", "after"):
        for _time, _offset in (("equality", 0.0), ("late", 0.001)):
            setattr(AcceptanceAdmissionTests, "test_retained_deadline_" + _phase + "_" + _edge + "_" + _time,
                    retained_lifetime_deadline_test(_phase, _edge, _offset))


if __name__ == "__main__":
    unittest.main()
