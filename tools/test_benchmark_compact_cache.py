"""Offline safety/measurement checks; never loads a model or contacts a server."""
from __future__ import annotations

import argparse
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
        with tempfile.TemporaryDirectory(dir=bench.SCRATCH) as folder:
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
        with tempfile.TemporaryDirectory(dir=bench.SCRATCH) as folder:
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
        with tempfile.TemporaryDirectory(dir=bench.SCRATCH) as folder:
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
        with tempfile.TemporaryDirectory(dir=bench.SCRATCH) as folder:
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


if __name__ == "__main__":
    unittest.main()
