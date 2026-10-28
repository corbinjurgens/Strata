"""No-model installation CLI and bounded leaf-probe failure controls."""
from __future__ import annotations

import contextlib
import copy
import ctypes
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from serve import install_check as ic, launch_plan as lp, server
from tools import build_provenance
import setup


class InstallCheck(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-install-check-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.exe = self.root / "strata-fixture"
        self.build = {"schema_version": 1, "version": "fixture", "backend": "cuda", "commit": "synthetic",
                      "dirty": True, "built_at": "2026-01-01T00:00:00Z", "cuda": "13.4"}
        self.exe.write_bytes(b"fixture\n" + build_provenance.MARKER + build_provenance.json_bytes(self.build) + b"\n")
        self.receipt = self.root / "BUILD.json"
        self.receipt.write_bytes(build_provenance.json_bytes(build_provenance.metadata_for(self.exe.read_bytes(), self.exe.name)))
        self.cfg = {"exe": self.exe.name, "cwd": str(self.root), "backend": "cuda",
                    "api_key": "synthetic-secret", "api_key_file": str(self.root / "do-not-read.key")}
        self.calls = []

    def query(self, argv, **kw):
        self.calls.append((argv, kw))
        self.assertLessEqual(kw["timeout"], 15)
        self.assertLessEqual(kw["limit"] * 2, 256 << 10)
        self.assertFalse(any("API_KEY" in key.upper() for key in kw["env"]))
        if argv[-1] == "--build-info":
            self.assertEqual(argv, [str(self.exe), "--build-info"])
            return lp.canonical(self.build)
        self.assertEqual(argv[:3], [sys.executable, "-B", "-c"])
        if argv[3] == ic.IMPORT_PROBE:
            return lp.canonical({name: [True, True, "import_succeeded"] for name in ic.IMPORTS})
        self.assertEqual(argv[3], ic.RUNTIME_PROBE)
        return lp.canonical({"driver": [True, True, "driver_device_query_succeeded"],
                             "libraries": [True, True, "runtime_device_query_and_blas_load_succeeded"]})

    def run_main(self, cfg=None, extra=(), query=None):
        path = self.root / "config.json"
        path.write_text(json.dumps(self.cfg if cfg is None else cfg), encoding="utf-8")
        argv = ["server", "--check-install", "--config", str(path), *extra]
        output, errors = io.StringIO(), io.StringIO()
        before = {p.name: p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        with patch.object(sys, "argv", argv), patch.object(lp, "bounded_query", query or self.query), \
                patch.object(server, "resolve_credentials", side_effect=AssertionError("credentials read")), \
                patch.object(server, "ByteTokenizer", side_effect=AssertionError("tokenizer constructed")), \
                patch.object(server, "StrataEngine", side_effect=AssertionError("engine constructed")), \
                patch.object(server, "Service", side_effect=AssertionError("service constructed")), \
                patch.object(server, "Server", side_effect=AssertionError("server bound")), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            try:
                code = server.main()
            except SystemExit as error:
                code = error.code
        self.assertEqual(errors.getvalue(), "")
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir() if p.is_file()}, before)
        self.assertNotIn("synthetic-secret", output.getvalue())
        self.assertNotIn("do-not-read.key", output.getvalue())
        return code, json.loads(output.getvalue())

    def test_cli_usable_requires_all_probes_and_no_start_or_writes(self):
        original = copy.deepcopy(self.cfg)
        with patch.dict(os.environ, {"STRATA_API_KEY": "synthetic-secret", "OTHER_API_KEY": "synthetic-secret"}):
            code, result = self.run_main()
        self.assertEqual(code, 0)
        self.assertIs(result["usable"], True)
        self.assertEqual(result["backend"], "cuda")
        self.assertEqual(result["checks"]["executable"], {"present": True, "usable": True, "state": "usable",
                                                        "reason": "build_info_exited_without_model"})
        self.assertEqual(result["checks"]["build"]["reason"], "receipt_embedded_and_runtime_identity_match")
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(original, self.cfg)

    def test_missing_executable_does_not_run_native_or_runtime_probe(self):
        self.exe.unlink()
        code, result = self.run_main()
        self.assertEqual(code, 1)
        self.assertEqual(result["checks"]["executable"], {"present": False, "usable": False, "state": "unavailable",
                                                        "reason": "executable_missing"})
        self.assertEqual(len(self.calls), 1)

    def test_missing_or_mismatched_receipt_never_passes_build(self):
        for content in (None, b"{}", b'{"api_key":"synthetic-secret"}'):
            with self.subTest(content=content):
                if content is None:
                    self.receipt.unlink()
                else:
                    self.receipt.write_bytes(content)
                code, result = self.run_main()
                self.assertEqual(code, 1)
                self.assertIs(result["checks"]["executable"]["present"], True)
                self.assertIsNone(result["checks"]["executable"]["usable"])
                self.assertIs(result["checks"]["build"]["usable"], False)

    def test_native_output_and_failures_are_not_serialized(self):
        for failure in (b"not JSON synthetic-secret", lp.canonical({**self.build, "commit": "wrong"}),
                        OSError("synthetic-secret"), lp.PlanError("synthetic-secret")):
            def probe(argv, **kw):
                if argv[-1] != "--build-info":
                    return self.query(argv, **kw)
                if isinstance(failure, Exception):
                    raise failure
                return failure
            with self.subTest(failure=type(failure).__name__):
                code, result = self.run_main(query=probe)
                self.assertEqual(code, 1)
                self.assertIs(result["checks"]["build"]["usable"], False)

    def test_missing_import_driver_runtime_and_unknown_never_count_as_success(self):
        for target, row in (("jinja2", [False, False, "module_missing"]),
                            ("driver", [True, False, "driver_device_query_failed"]),
                            ("libraries", [False, False, "runtime_or_blas_unavailable"]),
                            ("libraries", [True, False, "runtime_device_query_failed"])):
            def probe(argv, **kw):
                raw = json.loads(self.query(argv, **kw))
                if target in raw:
                    raw[target] = row
                return lp.canonical(raw)
            with self.subTest(target=target, row=row):
                code, result = self.run_main(query=probe)
                self.assertEqual(code, 1)
                item = result["checks"]["imports"].get(target) or result["checks"].get(target)
                self.assertIs(item["usable"], False)
                self.assertEqual(item["present"], row[0])
        def malformed(argv, **kw):
            return b'{}' if argv[3:4] == [ic.RUNTIME_PROBE] else self.query(argv, **kw)
        code, result = self.run_main(query=malformed)
        self.assertEqual(code, 1)
        self.assertIsNone(result["checks"]["libraries"]["usable"])
        self.assertEqual(result["checks"]["libraries"]["state"], "unverified")

    def test_conflicting_explicit_modes_and_overrides_refuse_before_probes(self):
        for extra in (("--print-launch-plan",), ("--launch-plan", "synthetic-secret"),
                      ("--estimate-memory",), ("--port", "synthetic-secret"), ("--api-key", "synthetic-secret"),
                      ("--before-load", "synthetic-secret"), ("--engine", "mock"), ("--help",)):
            with self.subTest(extra=extra):
                code, result = self.run_main(extra=extra)
                self.assertEqual(code, 2)
                self.assertEqual(result["error"]["code"], "installation_error")
                self.assertEqual(self.calls, [])

    def test_malformed_configs_refuse_safely(self):
        for cfg in ([], {"exe": 42}, {"env": []}, {"exe": "synthetic-secret\n"}, {"lib_dirs": [42]}):
            with self.subTest(cfg=cfg):
                code, result = self.run_main(cfg)
                self.assertEqual(code, 2)
                self.assertEqual(result["error"]["code"], "installation_error")

    def test_unsupported_backend_and_vision_are_explicitly_unverified(self):
        for backend in ("hip", "sycl"):
            self.build["backend"] = backend
            self.exe.write_bytes(b"fixture\n" + build_provenance.MARKER + build_provenance.json_bytes(self.build) + b"\n")
            self.receipt.write_bytes(build_provenance.json_bytes(build_provenance.metadata_for(self.exe.read_bytes(), self.exe.name)))
            cfg = {**self.cfg, "backend": backend}
            code, result = self.run_main(cfg)
            self.assertEqual(code, 0 if backend == "hip" else 1)
            self.assertEqual(result["backend"], backend)
            if backend == "sycl":
                self.assertEqual(result["checks"]["libraries"]["reason"], "no_supported_no_model_runtime_probe")
        code, result = self.run_main({**self.cfg, "vision": {"exe": "uninspected"}})
        self.assertEqual(code, 1)
        self.assertEqual(result["checks"]["features"]["reason"], "vision_not_checked")

    def test_public_cli_without_config_or_parent_site_packages_is_json_only(self):
        env = {k: v for k, v in os.environ.items() if k.upper() in ic.ENVIRONMENT}
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        result = subprocess.run([sys.executable, "-B", "-S", "-m", "serve.server", "--check-install"],
                                cwd=self.root, env=env, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, b"")
        self.assertNotIn(b"synthetic-secret", result.stdout)
        data = json.loads(result.stdout)
        self.assertIs(data["usable"], False)
        self.assertEqual(data["checks"]["executable"]["reason"], "no_configured_executable")
        self.assertFalse((self.root / "__pycache__").exists())

    def test_real_isolated_import_failure_is_safe_and_present_is_not_usable(self):
        (self.root / "jinja2.py").write_text('print("synthetic-secret")\nraise ImportError("synthetic-secret")\n', encoding="utf-8")
        data = lp.parse_json(lp.bounded_query([sys.executable, "-B", "-c", ic.IMPORT_PROBE,
                                              str(self.root), '["jinja2"]'], cwd=str(self.root),
                                             env={}, timeout=5, limit=ic.LIMIT))
        self.assertEqual(data, {"jinja2": [True, False, "import_failed"]})
        self.assertFalse((self.root / "__pycache__").exists())

    def test_hip_environment_composition_does_not_create_temp_files_or_query_speed(self):
        cfg = {**self.cfg, "backend": "hip", "gpu": [0, 1], "hip_ordinal": 0}
        with patch.object(server, "hip_env_guard", side_effect=AssertionError("temporary-file write")), \
                patch.object(server, "hip_speed_scores", side_effect=AssertionError("unbounded query")), \
                patch.object(server, "gpu_speed_scores", side_effect=AssertionError("unbounded query")):
            code, result = self.run_main(cfg)
        self.assertEqual(code, 1)  # The configured backend disagrees with the executable.
        self.assertIs(result["checks"]["build"]["usable"], False)

    def test_runtime_does_not_reintroduce_rejected_relative_library_directory(self):
        absolute_extra = str(self.root / "extras")
        class Function:
            def __init__(self, value=0):
                self.value = value
            def __call__(self, arg):
                if hasattr(arg, "_obj"):
                    arg._obj.value = self.value
                return 0
        driver = SimpleNamespace(cuInit=Function(), cuDeviceGetCount=Function(1))
        runtime = SimpleNamespace(cudaRuntimeGetVersion=Function(13040), cudaGetDeviceCount=Function(1))
        for directories in ([], [absolute_extra], ["extras"]):
            with self.subTest(directories=directories):
                environments, loaded = [], []
                def load(name):
                    virtual = {os.path.join(d, "nvcuda.dll"): driver for d in ("extras", absolute_extra)}
                    virtual.update({os.path.join(str(self.root), "cudart64_13.dll"): runtime,
                                    os.path.join(str(self.root), "cublas64_13.dll"): SimpleNamespace()})
                    if name not in virtual:
                        raise OSError("synthetic library unavailable")
                    loaded.append(name)
                    return virtual[name]
                def query(argv, **kwargs):
                    if argv[-1] == "--build-info":
                        environments.append(kwargs["env"])
                        return lp.canonical(self.build)
                    if argv[3] == ic.IMPORT_PROBE:
                        return lp.canonical({name: [True, True, "import_succeeded"] for name in ic.IMPORTS})
                    out = io.StringIO()
                    with patch.object(sys, "argv", ["-c", str(Path.cwd()), argv[5]]), \
                            patch.dict(os.environ, kwargs["env"], clear=True), \
                            patch.object(os.path, "isdir", return_value=True), \
                            patch.object(os, "add_dll_directory", return_value=SimpleNamespace(close=lambda: None)), \
                            patch.object(setup, "cuda_lib_dirs", return_value=[]), \
                            patch.object(ctypes, "CDLL", side_effect=load), contextlib.redirect_stdout(out):
                        exec(ic.RUNTIME_PROBE, {})
                    return out.getvalue().encode()
                with patch.object(Path, "is_dir", lambda path: str(path) == absolute_extra):
                    result = ic.check_install({**self.cfg, "lib_dirs": directories}, query=query,
                                              inherited={"PATH": "C:/synthetic/system"})
                self.assertEqual(result["usable"], directories == [absolute_extra])
                if directories == [absolute_extra]:
                    self.assertIn(absolute_extra, environments[0]["PATH"])
                else:
                    self.assertEqual(environments[0]["PATH"], "C:/synthetic/system")
                    self.assertNotIn(os.path.join("extras", "nvcuda.dll"), loaded)

    def test_shared_probe_cleanup_failure_has_safe_planning_and_replay_json(self):
        config = self.root / "probe-config.json"
        config.write_text("{}", encoding="utf-8")
        for option, target in (("--print-launch-plan", "serve.launch_plan.produce_plan"),
                               ("--launch-plan", "serve.launch_runtime.replay_cli")):
            with self.subTest(option=option):
                argv = (["server", option, "--config", str(config), "--engine", "strata"] if option == "--print-launch-plan" else
                        ["server", option, str(config)])
                output = io.StringIO()
                with patch.object(sys, "argv", argv), \
                        patch(target, side_effect=lp.ProbeCleanupError(object(), [])), \
                        patch.object(server, "StrataEngine", side_effect=AssertionError("engine started")), \
                        patch.object(server, "Server", side_effect=AssertionError("server bound")), \
                        contextlib.redirect_stdout(output):
                    code = server.main()
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["error"], {
                    "code": "invalid_launch_plan", "message": "Native query cleanup could not be confirmed.",
                    "cleanup": "unconfirmed"})

    def test_reader_setup_failures_return_safe_json_through_public_handlers(self):
        from serve import launch_runtime
        config = self.root / "reader-config.json"
        config.write_text("{}", encoding="utf-8")
        real_thread = lp.threading.Thread
        class Process:
            def __init__(self):
                self.stdout = io.BytesIO(lp.canonical({n: [True, True, "import_succeeded"] for n in ic.IMPORTS}))
                self.stderr, self.code = io.BytesIO(), None
            def wait(self, timeout):
                self.code = 0 if self.code is None else self.code
                return self.code
            def kill(self):
                self.code = -9
            def poll(self):
                return self.code
        for mode in ("install", "plan", "replay"):
            for phase in ("create", "start"):
                for ordinal in (1, 2):
                    with self.subTest(mode=mode, phase=phase, ordinal=ordinal):
                        proc, made = Process(), []
                        def thread(*args, **kwargs):
                            made.append(None)
                            if phase == "create" and len(made) == ordinal:
                                raise RuntimeError("synthetic-secret reader creation")
                            reader = real_thread(*args, **kwargs)
                            if phase == "start" and len(made) == ordinal:
                                def refuse():
                                    raise RuntimeError("synthetic-secret reader start")
                                reader.start = refuse
                            return reader
                        def query(*args, **kwargs):
                            return lp.bounded_query(["synthetic"], cwd=str(self.root), env={}, timeout=.01)
                        argv = {"install": ["server", "--check-install"],
                                "plan": ["server", "--print-launch-plan", "--engine", "strata", "--config", str(config)],
                                "replay": ["server", "--launch-plan", str(config)]}[mode]
                        out, errors = io.StringIO(), io.StringIO()
                        with patch.object(sys, "argv", argv), patch.object(server, "popen", return_value=proc), \
                                patch("serve.winjob.contain", return_value=False), \
                                patch.object(lp.threading, "Thread", side_effect=thread), \
                                patch.object(lp, "_produce_plan", side_effect=query), \
                                patch.object(launch_runtime, "replay_cli", side_effect=query), \
                                patch.object(server, "StrataEngine", side_effect=AssertionError("engine constructed")), \
                                patch.object(server, "Server", side_effect=AssertionError("server bound")), \
                                contextlib.redirect_stdout(out), contextlib.redirect_stderr(errors):
                            code = server.main()
                        self.assertEqual(code, 1 if mode == "install" else 2)
                        self.assertEqual(errors.getvalue(), "")
                        self.assertNotIn("synthetic-secret", out.getvalue())
                        result = json.loads(out.getvalue())
                        if mode == "install":
                            self.assertFalse(result["usable"])
                            self.assertEqual(result["checks"]["imports"]["jinja2"]["reason"], "bounded_import_probe_failed")
                        else:
                            self.assertEqual(result["error"]["code"], "invalid_launch_plan")
                        self.assertIsNotNone(proc.poll())
                        self.assertTrue(proc.stdout.closed)
                        self.assertTrue(proc.stderr.closed)


class BoundedProbes(unittest.TestCase):
    def test_public_unconfirmed_probe_exit_is_fatal_not_missing_prerequisite(self):
        class Process:
            def __init__(self):
                self.stdout = io.BytesIO(lp.canonical({n: [True, True, "import_succeeded"] for n in ic.IMPORTS}))
                self.stderr = io.BytesIO()
            def wait(self, timeout):
                raise subprocess.TimeoutExpired("synthetic-leaf", timeout)
            def kill(self):
                pass
            def poll(self):
                return None
        for setup_fails in (False, True):
            with self.subTest(setup_fails=setup_fails):
                proc, output = Process(), io.StringIO()
                setup_fault = (patch.object(lp.threading, "Thread", side_effect=RuntimeError("synthetic-secret setup"))
                               if setup_fails else contextlib.nullcontext())
                with patch.object(server, "popen", return_value=proc), patch("serve.winjob.contain", return_value=False), \
                        setup_fault, contextlib.redirect_stdout(output):
                    code = ic.check_main(SimpleNamespace(config=None, engine="strata"), {"check_install"})
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())["error"], {
                    "code": "installation_error", "message": "Installation probe cleanup could not be confirmed.",
                    "cleanup": "unconfirmed"})
                self.assertIsNone(proc.poll())
                self.assertTrue(proc.stdout.closed)
                self.assertTrue(proc.stderr.closed)

    def test_unconfirmed_exit_with_blocked_readers_is_bounded_and_retained(self):
        original, owned = server.popen, []
        started = time.monotonic()
        with contextlib.ExitStack() as faults:
            def spawn(*args, **kwargs):
                proc = original(*args, **kwargs)
                owned.append(proc)
                faults.enter_context(patch.object(proc, "kill"))
                faults.enter_context(patch.object(proc, "wait", side_effect=subprocess.TimeoutExpired("synthetic-leaf", .05)))
                return proc
            faults.enter_context(patch.object(server, "popen", side_effect=spawn))
            try:
                with self.assertRaises(lp.ProbeCleanupError) as failure:
                    lp.bounded_query([sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                                     cwd=tempfile.gettempdir(), env=dict(os.environ), timeout=.05, limit=1024)
                self.assertLess(time.monotonic() - started, 15)
                self.assertIs(failure.exception.process, owned[0])
                self.assertIsNone(owned[0].poll())
                self.assertTrue(any(reader.is_alive() for reader in failure.exception.readers))
                self.assertTrue(all(reader.daemon for reader in failure.exception.readers))
            finally:
                # Restore process control before mandatory test-owned cleanup.
                faults.close()
                for proc in owned:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait(timeout=5)
                    proc.stdout.close()
                    proc.stderr.close()
            for reader in failure.exception.readers:
                reader.join(timeout=5)
                self.assertFalse(reader.is_alive())
            self.assertTrue(owned[0].stdout.closed)
            self.assertTrue(owned[0].stderr.closed)

    def test_query_cleanup_covers_containment_failure(self):
        class Process:
            def __init__(self):
                self.stdout, self.stderr = io.BytesIO(), io.BytesIO()
                self.returncode = None
            def wait(self, timeout):
                return self.returncode
            def kill(self):
                self.returncode = -9
            def poll(self):
                return self.returncode
        real_thread = lp.threading.Thread
        for seam, ordinal in (("contain", 0), ("thread-create", 1), ("thread-start", 1),
                               ("thread-create", 2), ("thread-start", 2)):
            with self.subTest(seam=seam, ordinal=ordinal):
                proc = Process()
                creations = []
                def thread(*args, **kwargs):
                    creations.append(None)
                    if seam == "thread-create" and len(creations) == ordinal:
                        raise RuntimeError("synthetic startup failure")
                    reader = real_thread(*args, **kwargs)
                    if seam == "thread-start" and len(creations) == ordinal:
                        def fail_start():
                            raise RuntimeError("synthetic startup failure")
                        reader.start = fail_start
                    return reader
                with patch.object(server, "popen", return_value=proc), \
                        patch("serve.winjob.contain", side_effect=RuntimeError("synthetic startup failure") if seam == "contain" else None), \
                        patch.object(lp.threading, "Thread", side_effect=thread), \
                        self.assertRaisesRegex(lp.PlanError, "native query setup refused"):
                    lp.bounded_query(["synthetic"], cwd=tempfile.gettempdir(), env={})
                self.assertIsNotNone(proc.poll())
                self.assertTrue(proc.stdout.closed)
                self.assertTrue(proc.stderr.closed)

    def test_runtime_probe_queries_usability_without_allocation_calls(self):
        for backend, failure, expected in (("cuda", None, True), ("cuda", "cuInit", False),
                                           ("cuda", "cudaGetDeviceCount", False),
                                           ("cuda", "runtime_missing", False), ("cuda", "wrong_major", False), ("hip", None, True),
                                           ("hip", "hipInit", False)):
            with self.subTest(backend=backend, failure=failure):
                calls = []
                class Function:
                    def __init__(self, name):
                        self.name = name
                    def __call__(self, arg):
                        calls.append(self.name)
                        if hasattr(arg, "_obj"):
                            arg._obj.value = (12000 if failure == "wrong_major" else 13040) if self.name == "cudaRuntimeGetVersion" else 1
                        return 1 if failure == self.name else 0
                driver = SimpleNamespace(cuInit=Function("cuInit"), cuDeviceGetCount=Function("cuDeviceGetCount"))
                runtime = SimpleNamespace(cudaRuntimeGetVersion=Function("cudaRuntimeGetVersion"),
                                          cudaGetDeviceCount=Function("cudaGetDeviceCount"),
                                          hipInit=Function("hipInit"), hipGetDeviceCount=Function("hipGetDeviceCount"),
                                          hipRuntimeGetVersion=Function("hipRuntimeGetVersion"))
                def load(name):
                    if "nvcuda" in name or "libcuda.so" in name:
                        return driver
                    if "cudart" in name or "amdhip" in name:
                        if failure == "runtime_missing":
                            raise OSError("synthetic-secret")
                        return runtime
                    if "blas" in name:
                        return SimpleNamespace()
                    raise AssertionError(name)
                out = io.StringIO()
                argv = ["-c", str(Path(__file__).resolve().parents[1]), json.dumps([backend, 13, tempfile.gettempdir()])]
                with patch.object(sys, "argv", argv), patch.object(ctypes, "CDLL", load), \
                        patch.object(setup, "cuda_lib_dirs", return_value=[]), \
                        patch.object(setup, "hip_lib_dirs", return_value=[]), contextlib.redirect_stdout(out):
                    exec(ic.RUNTIME_PROBE, {})
                result = json.loads(out.getvalue())
                self.assertEqual(all(row[1] for row in result.values()), expected)
                self.assertNotIn("synthetic-secret", out.getvalue())
                self.assertIn("cuInit" if backend == "cuda" else "hipInit", calls)
                self.assertTrue(set(calls) <= {"cuInit", "cuDeviceGetCount", "cudaRuntimeGetVersion", "cudaGetDeviceCount",
                                               "hipInit", "hipGetDeviceCount", "hipRuntimeGetVersion"})

    def test_timeout_overflow_malformed_and_exit_leave_exact_child_closed(self):
        env = {k: v for k, v in os.environ.items() if k.upper() in ic.ENVIRONMENT}
        cases = (("import time; time.sleep(30)", .15, 1024),
                 ("import os; os.write(1, b'x'*65536)", 3, 1024),
                 ("import os; os.write(2, b'x'*65536)", 3, 1024),
                 ("raise SystemExit(7)", 3, 1024),
                 ("print('not JSON')", 3, 1024))
        for script, timeout, limit in cases:
            with self.subTest(script=script):
                owned = []
                original = server.popen
                def capture(*args, **kw):
                    proc = original(*args, **kw)
                    owned.append(proc)
                    return proc
                started = time.monotonic()
                with patch.object(server, "popen", capture), self.assertRaises(ValueError):
                    lp.parse_json(lp.bounded_query([sys.executable, "-B", "-c", script], cwd=tempfile.gettempdir(),
                                                   env=env, timeout=timeout, limit=limit))
                self.assertLess(time.monotonic() - started, 15)
                self.assertEqual(len(owned), 1)
                self.assertIsNotNone(owned[0].poll())
                self.assertTrue(owned[0].stdout.closed)
                self.assertTrue(owned[0].stderr.closed)


if __name__ == "__main__":
    unittest.main()
