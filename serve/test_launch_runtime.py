"""Real pipe/HTTP consumers of synthetic native replay and status records."""
from __future__ import annotations

import base64
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import socket
import subprocess
from types import SimpleNamespace
import unittest
import urllib.request
from unittest.mock import patch

from serve import launch_plan as lp, server
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
    print(f"READY {s['context'] + int(mode == 'bad-ready')} stop", flush=True)
    sequence = 1
    for command in sys.stdin:
        command = command.strip()
        if command == "QUIT":
            if (cwd / "late.txt").exists() and frame:
                sys.stdout.write(ownership_frame(sequence + 1, 512, shards=shards))
                sys.stdout.flush()
            return
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
