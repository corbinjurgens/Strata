"""serve/test_runconfig.py - #564: the web page's Settings view of the run config (GET / POST /config), against the
mock engine: only the listed keys change, every other key of the file stays, a bad value changes nothing, and only
Strata's own page may write it (JSON, no foreign Origin, the API key when one is set).

    python -m unittest serve.test_runconfig
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from serve import runconfig
from serve import server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
CFG = {"exe": "engine/strata.exe", "args": ["--pack", "data/packs/q2_0", "--kv", "int8", "--spec-min-p", "0.5"],
       "model_name": "qwen3.8-flash-next-q2_0", "port": 8080, "host": "0.0.0.0", "api_key": "",
       "sampling": {"temperature": 0.7, "presence_penalty": 1.5},
       "mcp_servers": {"files": {"command": "npx", "args": ["-y", "server-filesystem", "."]}},
       "cors_origins": ["http://localhost:3000"]}


class Apply(unittest.TestCase):
    def test_only_the_named_keys_change(self):
        new, changed = runconfig.apply(CFG, {"sampling.top_p": 0.95, "reasoning_budget_tokens": 4096,
                                             "vram_reserve_mib": 2048, "aliases": "qwen, local", "lazy_load": True})
        self.assertEqual(changed, ["sampling.top_p", "reasoning_budget_tokens", "vram_reserve_mib", "aliases",
                                   "lazy_load"])
        self.assertEqual(new["sampling"], {"temperature": 0.7, "presence_penalty": 1.5, "top_p": 0.95})
        self.assertEqual(new["args"][-2:], ["--vram-reserve-mib", "2048"])
        self.assertEqual(new["aliases"], ["qwen", "local"])
        for k in ("exe", "model_name", "port", "host", "api_key", "mcp_servers", "cors_origins"):
            self.assertEqual(new[k], CFG[k], k)
        self.assertEqual(CFG["sampling"], {"temperature": 0.7, "presence_penalty": 1.5})   # the input is left alone

    def test_null_is_the_default_again(self):
        cfg, _ = runconfig.apply(CFG, {"vram_reserve_mib": 2048, "fit_max_tokens": True})
        new, changed = runconfig.apply(cfg, {"vram_reserve_mib": None, "fit_max_tokens": None,
                                             "sampling.temperature": None, "idle_unload_s": None})
        self.assertEqual(changed, ["vram_reserve_mib", "fit_max_tokens", "sampling.temperature"])
        self.assertEqual(new["args"], CFG["args"])
        self.assertNotIn("fit_max_tokens", new)
        self.assertEqual(new["sampling"], {"presence_penalty": 1.5})
        new, _ = runconfig.apply({"sampling": {"top_k": 20}}, {"sampling.top_k": None})
        self.assertNotIn("sampling", new)                                 # an empty block goes

    def test_refused(self):
        for changes in ({"api_key": "x"}, {"host": "0.0.0.0"}, {"mcp_servers": {}}, {"before_load": "calc.exe"},
                        {"args": []}, {"exe": "evil.exe"}, {"trusted_origins": ["*"]}, {"allowed_hosts": ["*"]},
                        {"sampling.temperature": -1}, {"sampling.top_p": 0}, {"sampling.top_k": 65},
                        {"sampling.min_p": 2}, {"reasoning_budget_tokens": 1.5}, {"fit_max_tokens": "yes"},
                        {"effort_position": "middle"}, {"vram_reserve_mib": -5}, {"idle_unload_s": True},
                        {"aliases": [1]}, {}, None, ["x"]):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runconfig.apply(CFG, changes)
        with self.assertRaises(ValueError):                              # one bad value: nothing is applied
            runconfig.apply(CFG, {"fit_max_tokens": True, "sampling.top_k": 0})
        with self.assertRaises(ValueError):
            runconfig.apply({**CFG, "vision": {"gpu": True}}, {"lazy_load": True})

    def test_view(self):
        v = runconfig.view({**CFG, "args": CFG["args"] + ["--vram-reserve-mib", "1500"]}, "x/strata-q2_0.json")
        self.assertEqual(v["file"], "strata-q2_0.json")
        got = {k["key"]: k for k in v["keys"]}
        self.assertEqual(got["sampling.temperature"]["value"], 0.7)
        self.assertEqual(got["vram_reserve_mib"]["value"], 1500)
        self.assertIsNone(got["fit_max_tokens"]["value"])
        self.assertEqual(got["anthropic_thinking"]["choices"], ["model", "on_request"])
        self.assertNotIn("api_key", got)
        self.assertNotIn("mcp_servers", got)


class Startup(unittest.TestCase):
    """Exercise public startup without opening a socket or launching an engine."""

    def run_main(self, path, extra=(), before_engine=None):
        engine = MockEngine(ByteTokenizer(), "ok", max_context=131072)
        engine.info, engine.spawn = {}, (None, [])
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config",
                                                           str(path), *extra]))
            stack.enter_context(patch.object(server, "Server"))
            stack.enter_context(patch.dict(sys.modules, {"strata_tokenizer": SimpleNamespace(
                Tokenizer=lambda *args: ByteTokenizer())}))
            stack.enter_context(patch.object(server, "effort_end_args", return_value=None))
            stack.enter_context(patch.object(server, "warn_budget_over_ram"))
            stack.enter_context(patch.object(server, "warn_tight_ram"))
            stack.enter_context(patch.object(server, "desktop_vram_note", return_value=None))
            stack.enter_context(patch.object(server, "linux_desktop", return_value=False))
            stack.enter_context(patch.object(server, "hub_from_config", return_value=None))
            stack.enter_context(patch.object(server, "lan_addresses", return_value=[]))
            stack.enter_context(patch.object(server.signal, "signal"))
            stack.enter_context(patch.object(server.time, "sleep", side_effect=KeyboardInterrupt))
            stack.enter_context(patch.object(server.Service, "start_idle_unload"))
            def create_engine(*args, **kwargs):
                if before_engine:
                    before_engine()
                return engine
            create = stack.enter_context(patch.object(server, "StrataEngine", side_effect=create_engine))
            http = stack.enter_context(patch.object(server, "serve"))
            browser = stack.enter_context(patch("webbrowser.open"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(server.main(), 0)
        return create, http, browser

    def fixture(self, root, prep):
        tokenizer = root / "tokenizer"
        tokenizer.mkdir()
        for name, content in (("vocab.json", "{}"), ("merges.txt", ""), ("token_type.json", "[]")):
            (tokenizer / name).write_text(content)
        cfg = prep.config_for(root / "source", root / "pack", 131072, None)
        cfg.update(tokenizer=str(tokenizer), gpu=0, host="0.0.0.0", port=8182, open_browser=True)
        cfg["env"]["STRATA_DECODE_TIMING"] = "explicit"
        path = root / "profile.json"
        path.write_bytes(prep.json_bytes(cfg))
        return path, cfg

    def test_direct_and_helper_sample_once_before_engine_with_effective_gpu(self):
        from tools import prepare_unsloth_q2 as prep
        for helper in (False, True):
            for ram, available, vram, budget in ((30.88, 27, 15.99, "12"), (63.7, 58, 6, "30")):
                with self.subTest(helper=helper, vram=vram), tempfile.TemporaryDirectory() as directory, \
                        patch.object(prep, "detected_memory", return_value={"ram_gib": ram,
                            "available_gib": available, "commit_available_gib": 70, "windows_commit": True}) as sample, \
                        patch.object(prep, "detected_gpu", side_effect=lambda pick=None: {
                            "vram_gb": vram, "index": 0 if pick is None else pick}) as gpu, \
                        patch.object(prep.setup, "find_nvcc", return_value=(None, None)), \
                        patch.object(prep, "check_ready") as ready, \
                        patch.dict(os.environ, {"STRATA_API_KEY": "inherited", "STRATA_FETCH_PROFILE": "1"}, clear=True):
                    path, cfg = self.fixture(Path(directory), prep)
                    original = path.read_bytes()
                    sample.reset_mock()
                    gpu.reset_mock()
                    observed = {}
                    def engine_ready():
                        self.assertEqual(sample.call_count, 1)
                        ready.assert_called_once()
                        gpu.assert_called_once_with(1 if not helper else 0)
                    def invoke(command, **kwargs):
                        with patch.dict(os.environ, kwargs["env"], clear=True):
                            observed["main"] = self.run_main(path, command[2:], engine_ready)
                        return SimpleNamespace(returncode=0)
                    if helper:
                        with patch.object(prep.subprocess, "run", side_effect=invoke):
                            self.assertEqual(prep.launch(path), 0)
                        create, http, browser = observed["main"]
                    else:
                        create, http, browser = self.run_main(path, ["--gpu", "1", "--port", "8182"], engine_ready)
                    sample.assert_called_once()
                    ready.assert_called_once()
                    gpu.assert_called_once_with(0 if helper else 1)
                    create.assert_called_once()
                    args = create.call_args.args[1]
                    self.assertEqual(args[args.index("--resident-budget-gib") + 1], budget)
                    self.assertEqual(args[args.index("--max-context") + 1], "131072")
                    env = create.call_args.kwargs["env"]
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0" if helper else "1")
                    self.assertEqual(env["STRATA_DECODE_TIMING"], "explicit")
                    if helper:
                        self.assertNotIn("STRATA_FETCH_PROFILE", env)
                    svc = http.call_args.args[0]
                    self.assertEqual(svc.config_path, str(path))
                    self.assertEqual(svc.shared_path, str(path.with_suffix("")) + ".shared-settings.json")
                    self.assertEqual(svc.api_key, cfg["api_key"] if helper else "inherited")
                    self.assertEqual(http.call_args.kwargs, {"host": "0.0.0.0", "port": 8182})
                    self.assertEqual(browser.call_count, int(helper))
                    saved = json.loads(path.read_text())
                    self.assertEqual(saved["gpu"], 0)
                    self.assertEqual((saved["host"], saved["port"], saved["api_key"]),
                                     ("0.0.0.0", 8182, cfg["api_key"]))
                    self.assertEqual(path.read_bytes(), original)

    def test_policy_headroom_removals_reach_public_engine_environment(self):
        from tools import prepare_unsloth_q2 as prep
        key = "STRATA_RESIDENT_HEADROOM_GIB"
        for removal in ("null", "delete"):
            for inherited in ("8", "2", "0"):
                with self.subTest(removal=removal, inherited=inherited), tempfile.TemporaryDirectory() as directory, \
                        patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                            "available_gib": 27, "commit_available_gib": 60, "windows_commit": True}) as sample, \
                        patch.object(prep, "detected_gpu", return_value={"vram_gb": 16, "index": 0}), \
                        patch.object(prep.setup, "find_nvcc", return_value=(None, None)), \
                        patch.object(prep, "check_ready"), \
                        patch.object(prep, "admit_automatic_launch", wraps=prep.admit_automatic_launch) as admit, \
                        patch.dict(os.environ, {key: inherited, "UNRELATED_CONTROL": "keep"}, clear=True):
                    path, cfg = self.fixture(Path(directory), prep)
                    if removal == "null":
                        cfg["env"][key] = None
                    else:
                        cfg["env"].pop(key)
                    original = prep.json_bytes(cfg)
                    path.write_bytes(original)
                    refreshed = cfg
                    for _ in range(2):
                        refreshed = prep.refreshed_config(refreshed)
                        self.assertIsNone(refreshed["memory_policy"]["overrides"]["env"][key])
                        self.assertNotIn(key, refreshed["env"])
                    sample.reset_mock()
                    create, _, _ = self.run_main(path)
                    sample.assert_called_once()
                    self.assertEqual(admit.call_args.args[0]["headroom_gib"], 4)
                    self.assertTrue(admit.call_args.args[0]["automatic_host_cache"])
                    env = create.call_args.kwargs["env"]
                    self.assertNotIn(key, env)
                    self.assertEqual(env["UNRELATED_CONTROL"], "keep")
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
                    self.assertEqual(path.read_bytes(), original)

    @unittest.skipUnless(os.name == "nt", "Windows policy spelling")
    def test_windows_metadata_only_managed_overrides_reach_refresh_startup_and_evidence(self):
        from tools import prepare_unsloth_q2 as prep
        from tools.benchmark_compact_cache import safe_engine_environment
        for key, raw in (("STRATA_RESIDENT_HEADROOM_GIB", "8"), ("STRATA_RESIDENT_HEADROOM_GIB", 8),
                         ("STRATA_RESIDENT_HEADROOM_GIB", 8.0), ("STRATA_RESIDENT_HEADROOM_GIB", 0),
                         ("STRATA_FILE_CACHE_MIB", "1024"), ("STRATA_FILE_CACHE_MIB", 1024)):
            for spelling in (key, key.lower(), key.title()):
                with self.subTest(key=key, raw=raw, spelling=spelling), tempfile.TemporaryDirectory() as directory, \
                        patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                            "available_gib": 27, "commit_available_gib": 60, "windows_commit": True}), \
                        patch.object(prep, "detected_gpu", return_value={"vram_gb": 16, "index": 0}), \
                        patch.object(prep.setup, "find_nvcc", return_value=(None, None)), \
                        patch.object(prep, "check_ready"), patch.object(prep, "report_memory") as report, \
                        patch.dict(os.environ, {key: "99"}, clear=True):
                    path, cfg = self.fixture(Path(directory), prep)
                    cfg["memory_policy"]["overrides"] = {"env": {spelling: raw}}
                    original = prep.json_bytes(cfg)
                    path.write_bytes(original)
                    for _ in range(2):
                        cfg = prep.refreshed_config(cfg)
                        self.assertEqual(cfg["env"][key], raw)
                        self.assertIs(type(cfg["env"][key]), type(raw))
                        self.assertEqual(cfg["memory_policy"]["sample"]["env"][key], raw)
                        self.assertIs(type(cfg["memory_policy"]["overrides"]["env"][key]), type(raw))
                        self.assertEqual(safe_engine_environment(cfg, {key.lower(): "99"})[key], str(raw))
                    create, _, _ = self.run_main(path)
                    runtime = report.call_args.args[1]
                    plan = runtime["memory_policy"]["sample"]
                    self.assertEqual(plan["headroom_gib"] if "HEADROOM" in key else plan["recent_mib"], float(raw))
                    self.assertEqual(runtime["env"][key], raw)
                    self.assertIs(type(runtime["env"][key]), type(raw))
                    self.assertEqual(create.call_args.kwargs["env"][key], str(raw))
                    self.assertEqual(safe_engine_environment(runtime, {key.lower(): "99"})[key], str(raw))
                    self.assertEqual(path.read_bytes(), original)

    @unittest.skipUnless(os.name == "nt", "Windows policy spelling")
    def test_windows_metadata_override_yields_to_new_saved_edits_and_removals(self):
        from tools import prepare_unsloth_q2 as prep
        from tools.benchmark_compact_cache import safe_engine_environment
        for key, metadata, edited in (("STRATA_RESIDENT_HEADROOM_GIB", "8", 6.0),
                                      ("STRATA_FILE_CACHE_MIB", "1024", 2048)):
            for spelling in (key, key.lower(), key.title()):
                for edit in ("number", "null", "delete"):
                    with self.subTest(key=key, spelling=spelling, edit=edit), tempfile.TemporaryDirectory() as directory, \
                            patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                                "available_gib": 27, "commit_available_gib": 60, "windows_commit": True}), \
                            patch.object(prep, "detected_gpu", return_value={"vram_gb": 16, "index": 0}), \
                            patch.object(prep.setup, "find_nvcc", return_value=(None, None)), \
                            patch.object(prep, "check_ready"), patch.object(prep, "report_memory") as report, \
                            patch.dict(os.environ, {key: "99"}, clear=True):
                        path, cfg = self.fixture(Path(directory), prep)
                        cfg["memory_policy"]["overrides"] = {"env": {spelling: metadata}}
                        cfg["env"].pop(key)
                        if edit != "delete":
                            cfg["env"][key.lower()] = edited if edit == "number" else None
                        original = prep.json_bytes(cfg)
                        path.write_bytes(original)
                        for _ in range(2):
                            cfg = prep.refreshed_config(cfg)
                            self.assertEqual(cfg["memory_policy"]["overrides"]["env"][key],
                                             edited if edit == "number" else None)
                            if edit == "number":
                                self.assertEqual(cfg["env"][key], edited)
                                self.assertIs(type(cfg["env"][key]), type(edited))
                                self.assertEqual(cfg["memory_policy"]["sample"]["env"][key], edited)
                            else:
                                self.assertNotIn(key, cfg["env"])
                                self.assertNotIn(key, safe_engine_environment(cfg, {key.lower(): "99"}))
                        create, _, _ = self.run_main(path)
                        runtime = report.call_args.args[1]
                        child, evidence = create.call_args.kwargs["env"], safe_engine_environment(runtime, {key: "99"})
                        if edit == "number":
                            self.assertEqual(child[key], str(edited))
                            self.assertEqual(evidence[key], str(edited))
                        else:
                            self.assertNotIn(key, child)
                            self.assertNotIn(key, evidence)
                        self.assertEqual(path.read_bytes(), original)

    def test_posix_metadata_only_managed_overrides_reach_refresh_startup_and_evidence(self):
        from tools import prepare_unsloth_q2 as prep
        from tools import benchmark_compact_cache as benchmark
        for key, raw in (("STRATA_RESIDENT_HEADROOM_GIB", "8"), ("STRATA_RESIDENT_HEADROOM_GIB", 8),
                         ("STRATA_RESIDENT_HEADROOM_GIB", 8.0), ("STRATA_RESIDENT_HEADROOM_GIB", 0),
                         ("STRATA_FILE_CACHE_MIB", "1024"), ("STRATA_FILE_CACHE_MIB", 1024),
                         ("STRATA_FILE_CACHE_COMPACT", 0), ("STRATA_OWNED_PRICE", "exact")):
            inherited = {key: "99", key.lower(): "inherited-lower", key.title(): "inherited-mixed",
                         "Unknown_Case": "inherited"}
            posix = SimpleNamespace(**{**vars(os), "name": "posix", "environ": inherited})
            with self.subTest(key=key, raw=raw), tempfile.TemporaryDirectory() as directory, \
                    patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                        "available_gib": 27, "commit_available_gib": None, "windows_commit": False}), \
                    patch.object(prep, "detected_gpu", return_value={"vram_gb": 16, "index": 0}), \
                    patch.object(prep.setup, "find_nvcc", return_value=(None, None)), \
                    patch.object(prep, "check_ready"), patch.object(prep, "report_memory") as report, \
                    patch.object(prep, "os", posix), patch.object(server, "os", posix), \
                    patch.object(benchmark, "os", posix):
                path, cfg = self.fixture(Path(directory), prep)
                cfg["memory_policy"]["overrides"] = {"env": {key: raw}}
                cfg["env"].update({key.lower(): "saved-lower", key.title(): "saved-mixed", "Unknown_Case": "saved"})
                original = prep.json_bytes(cfg)
                path.write_bytes(original)
                for _ in range(2):
                    before = prep.json_bytes(cfg)
                    runtime = prep.refreshed_config(cfg)
                    self.assertEqual(prep.json_bytes(cfg), before)
                    cfg = runtime
                    for values in (cfg["env"], cfg["memory_policy"]["sample"]["env"],
                                   cfg["memory_policy"]["overrides"]["env"]):
                        self.assertEqual(values[key], raw)
                        self.assertIs(type(values[key]), type(raw))
                    self.assertEqual(cfg["env"][key.lower()], "saved-lower")
                    self.assertEqual(cfg["env"][key.title()], "saved-mixed")
                create, _, _ = self.run_main(path)
                runtime = report.call_args.args[1]
                plan = runtime["memory_policy"]["sample"]
                if key == "STRATA_RESIDENT_HEADROOM_GIB":
                    self.assertEqual(plan["headroom_gib"], float(raw))
                elif key == "STRATA_FILE_CACHE_MIB":
                    self.assertEqual(plan["recent_mib"], int(raw))
                self.assertEqual(runtime["env"][key], raw)
                self.assertIs(type(runtime["env"][key]), type(raw))
                self.assertEqual(plan["env"][key], raw)
                self.assertIs(type(plan["env"][key]), type(raw))
                child = create.call_args.kwargs["env"]
                self.assertEqual(child[key], str(raw))
                self.assertEqual(child, server.child_env(runtime))
                self.assertEqual(child[key.lower()], "saved-lower")
                self.assertEqual(child[key.title()], "saved-mixed")
                self.assertEqual(child["Unknown_Case"], "saved")
                evidence = benchmark.safe_engine_environment(runtime, inherited)
                self.assertEqual(evidence[key], child[key])
                self.assertNotIn(key.lower(), evidence)
                self.assertNotIn(key.title(), evidence)
                self.assertEqual(path.read_bytes(), original)

    def test_posix_metadata_override_yields_to_saved_edits_and_removals(self):
        from tools import prepare_unsloth_q2 as prep
        from tools import benchmark_compact_cache as benchmark
        for key, metadata, edited in (("STRATA_RESIDENT_HEADROOM_GIB", "8", 6.0),
                                      ("STRATA_RESIDENT_HEADROOM_GIB", 8, 8.0),
                                      ("STRATA_RESIDENT_HEADROOM_GIB", 8.0, 8),
                                      ("STRATA_FILE_CACHE_MIB", "1024", 2048),
                                      ("STRATA_FILE_CACHE_COMPACT", "1", 0),
                                      ("STRATA_OWNED_PRICE", "exact", "custom")):
            for edit in ("value", "null", "delete", "metadata-null"):
                inherited = {key: "99", key.lower(): "lower", key.title(): "mixed", "Unknown_Case": "keep"}
                posix = SimpleNamespace(**{**vars(os), "name": "posix", "environ": inherited})
                with self.subTest(key=key, metadata=metadata, edited=edited, edit=edit), \
                        tempfile.TemporaryDirectory() as directory, \
                        patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                            "available_gib": 27, "commit_available_gib": None, "windows_commit": False}), \
                        patch.object(prep, "detected_gpu", return_value={"vram_gb": 16, "index": 0}), \
                        patch.object(prep.setup, "find_nvcc", return_value=(None, None)), \
                        patch.object(prep, "check_ready"), patch.object(prep, "report_memory") as report, \
                        patch.object(prep, "os", posix), patch.object(server, "os", posix), \
                        patch.object(benchmark, "os", posix):
                    path, cfg = self.fixture(Path(directory), prep)
                    # Include the normally absent large-GPU control in the saved generated layer.
                    cfg["memory_policy"]["generated_env"].setdefault(key, metadata)
                    if isinstance(metadata, (int, float)):
                        cfg["memory_policy"]["generated_env"][key] = metadata
                    cfg["memory_policy"]["overrides"] = {"env": {key: metadata}}
                    if edit == "metadata-null":
                        cfg["env"][key] = cfg["memory_policy"]["generated_env"][key]
                        cfg["memory_policy"]["overrides"]["env"][key] = None
                    else:
                        cfg["env"].pop(key, None)
                        if edit != "delete":
                            cfg["env"][key] = edited if edit == "value" else None
                    original = prep.json_bytes(cfg)
                    path.write_bytes(original)
                    for _ in range(2):
                        cfg = prep.refreshed_config(cfg)
                        self.assertEqual(cfg["memory_policy"]["overrides"]["env"][key],
                                         edited if edit == "value" else None)
                        if edit == "value":
                            self.assertEqual(cfg["env"][key], edited)
                            self.assertIs(type(cfg["env"][key]), type(edited))
                            self.assertEqual(cfg["memory_policy"]["sample"]["env"][key], edited)
                        else:
                            self.assertNotIn(key, cfg["env"])
                            self.assertNotIn(key, cfg["memory_policy"]["sample"]["env"])
                    create, _, _ = self.run_main(path)
                    runtime = report.call_args.args[1]
                    child = create.call_args.kwargs["env"]
                    evidence = benchmark.safe_engine_environment(runtime, inherited)
                    if edit == "value":
                        for env in (runtime["env"], runtime["memory_policy"]["sample"]["env"],
                                    runtime["memory_policy"]["overrides"]["env"]):
                            self.assertEqual(env[key], edited)
                            self.assertIs(type(env[key]), type(edited))
                        self.assertEqual(child[key], str(edited))
                        self.assertEqual(evidence[key], str(edited))
                    else:
                        self.assertNotIn(key, child)
                        self.assertNotIn(key, evidence)
                        self.assertNotIn("None", child.values())
                    self.assertEqual(child[key.lower()], "lower")
                    self.assertEqual(child[key.title()], "mixed")
                    self.assertEqual(child["Unknown_Case"], "keep")
                    self.assertEqual(path.read_bytes(), original)

    def test_invalid_posix_metadata_stops_public_main_before_sampling_or_startup(self):
        from tools import prepare_unsloth_q2 as prep
        cases = (("STRATA_RESIDENT_HEADROOM_GIB", "garbage"), ("STRATA_RESIDENT_HEADROOM_GIB", True),
                 ("STRATA_RESIDENT_HEADROOM_GIB", "NaN"), ("STRATA_FILE_CACHE_MIB", "1.5"),
                 ("STRATA_FILE_CACHE_MIB", 1.0), ("STRATA_FILE_CACHE_MIB", -1))
        posix = SimpleNamespace(**{**vars(os), "name": "posix", "environ": {}})
        for key, raw in cases:
            with self.subTest(key=key, raw=raw), tempfile.TemporaryDirectory() as directory, \
                    patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                        "available_gib": 27, "commit_available_gib": None, "windows_commit": False}) as memory, \
                    patch.object(prep, "detected_gpu", return_value={"vram_gb": 16, "index": 0}) as gpu, \
                    patch.object(prep.setup, "find_nvcc", return_value=(None, None)), \
                    patch.object(prep, "os", posix), patch.object(server, "os", posix):
                path, cfg = self.fixture(Path(directory), prep)
                cfg["memory_policy"]["overrides"] = {"env": {key: raw}}
                original = prep.json_bytes(cfg)
                path.write_bytes(original)
                memory.reset_mock()
                gpu.reset_mock()
                with patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config", str(path)]), \
                        patch.object(server, "StrataEngine") as engine, patch.object(server, "Server") as listener, \
                        patch.object(server, "serve") as http, patch.object(prep, "check_ready") as ready, \
                        patch.object(prep.subprocess, "run") as process, patch("webbrowser.open") as browser, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    server.main()
                self.assertEqual(error.exception.code, 2)
                for seam in (memory, gpu, engine, listener, http, ready, process, browser):
                    seam.assert_not_called()
                self.assertEqual(path.read_bytes(), original)

    @unittest.skipUnless(os.name == "nt", "Windows policy spelling")
    def test_windows_managed_aliases_match_public_startup_plan_child_and_evidence(self):
        from tools import prepare_unsloth_q2 as prep
        from tools.benchmark_compact_cache import safe_engine_environment
        cases = (("STRATA_RESIDENT_HEADROOM_GIB", "8"), ("STRATA_RESIDENT_HEADROOM_GIB", 8),
                 ("STRATA_FILE_CACHE_MIB", "1024"), ("STRATA_FILE_CACHE_MIB", 1024))
        for key, raw in cases:
            for spelling in (key.lower(), key.title()):
                with self.subTest(key=key, raw=raw, spelling=spelling), tempfile.TemporaryDirectory() as directory, \
                        patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                            "available_gib": 27, "commit_available_gib": 60, "windows_commit": True}) as sample, \
                        patch.object(prep, "detected_gpu", return_value={"vram_gb": 16, "index": 0}), \
                        patch.object(prep.setup, "find_nvcc", return_value=(None, None)), \
                        patch.object(prep, "check_ready"), patch.object(prep, "report_memory") as report, \
                        patch.dict(os.environ, {key: "2", "Unknown_Case": "keep"}, clear=True):
                    path, cfg = self.fixture(Path(directory), prep)
                    cfg["env"].pop(key)
                    cfg["env"][spelling] = raw
                    # Equal aliases in each separate layer collapse, while config overrides inherited values.
                    cfg["env"][key] = raw
                    generated = cfg["memory_policy"]["generated_env"]
                    generated[spelling] = generated.pop(key)
                    cfg["memory_policy"]["overrides"] = {"env": {spelling: raw, key: raw}}
                    for _ in range(2):
                        refreshed = prep.refreshed_config(cfg)
                        self.assertEqual(refreshed["memory_policy"]["sample"]["env"][key], raw)
                        self.assertIs(type(refreshed["env"][key]), type(raw))
                        cfg = refreshed
                    # Exercise public startup from the original spelling, with saved bytes unchanged.
                    cfg["env"][spelling] = cfg["env"].pop(key)
                    original = prep.json_bytes(cfg)
                    path.write_bytes(original)
                    sample.reset_mock()
                    create, _, _ = self.run_main(path)
                    sample.assert_called_once()
                    runtime = report.call_args.args[1]
                    plan = runtime["memory_policy"]["sample"]
                    self.assertEqual(plan["headroom_gib"] if "HEADROOM" in key else plan["recent_mib"],
                                     8 if "HEADROOM" in key else 1024)
                    self.assertEqual(plan["env"][key], raw)
                    self.assertIs(type(plan["env"][key]), type(raw))
                    child = create.call_args.kwargs["env"]
                    self.assertEqual(child[key], str(raw))
                    self.assertNotIn(spelling, child)
                    inherited = {key.lower(): "2", "Unknown_Case": "keep"}
                    effective = server.config_environment(runtime, inherited)
                    self.assertEqual(effective[key], child[key])
                    self.assertEqual(effective["Unknown_Case"], "keep")
                    self.assertEqual(safe_engine_environment(runtime, inherited)[key], child[key])
                    self.assertEqual(path.read_bytes(), original)

    @unittest.skipUnless(os.name == "nt", "Windows policy spelling")
    def test_invalid_windows_managed_aliases_stop_public_main_before_all_startup_seams(self):
        from tools import prepare_unsloth_q2 as prep
        key = "STRATA_RESIDENT_HEADROOM_GIB"
        cases = ((key.lower(), "garbage", False), (key.title(), True, False),
                 ("strata_file_cache_mib", "1.5", False), ("Strata_File_Cache_Mib", 1.0, False),
                 (key.lower(), "8", True), (key.lower(), 4, True))
        for spelling, raw, conflict in cases:
            with self.subTest(spelling=spelling, raw=raw, conflict=conflict), tempfile.TemporaryDirectory() as directory, \
                    patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                        "available_gib": 27, "commit_available_gib": 60, "windows_commit": True}) as memory, \
                    patch.object(prep, "detected_gpu", return_value={"vram_gb": 16, "index": 0}) as gpu, \
                    patch.object(prep.setup, "find_nvcc", return_value=(None, None)):
                path, cfg = self.fixture(Path(directory), prep)
                if not conflict:
                    cfg["env"].pop(spelling.upper())
                elif isinstance(raw, int):
                    # Numeric equality does not erase a raw type conflict.
                    cfg["env"][key] = 4.0
                cfg["env"][spelling] = raw
                original = prep.json_bytes(cfg)
                path.write_bytes(original)
                memory.reset_mock()
                gpu.reset_mock()
                with patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config", str(path)]), \
                        patch.object(server, "StrataEngine") as engine, patch.object(server, "Server") as listener, \
                        patch.object(server, "serve") as http, patch.object(prep, "check_ready") as ready, \
                        patch.object(prep.subprocess, "run") as process, patch("webbrowser.open") as browser, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    server.main()
                self.assertEqual(error.exception.code, 2)
                for seam in (memory, gpu, engine, listener, http, ready, process, browser):
                    seam.assert_not_called()
                self.assertEqual(path.read_bytes(), original)

    @unittest.skipUnless(os.name == "nt", "Windows policy spelling")
    def test_windows_aliased_removals_reach_public_startup_and_evidence(self):
        from tools import prepare_unsloth_q2 as prep
        from tools.benchmark_compact_cache import safe_engine_environment
        for key in prep.MEMORY_ENV:
            for removal in ("null", "delete", "override"):
                with self.subTest(key=key, removal=removal), tempfile.TemporaryDirectory() as directory, \
                        patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                            "available_gib": 27, "commit_available_gib": 60, "windows_commit": True}), \
                        patch.object(prep, "detected_gpu", return_value={"vram_gb": 16, "index": 0}), \
                        patch.object(prep.setup, "find_nvcc", return_value=(None, None)), \
                        patch.object(prep, "check_ready"), patch.object(prep, "report_memory") as report, \
                        patch.dict(os.environ, {key.lower(): "8"}, clear=True):
                    path, cfg = self.fixture(Path(directory), prep)
                    generated = cfg["memory_policy"]["generated_env"]
                    generated[key.title()] = generated.pop(key, "old")
                    cfg["env"].pop(key, None)
                    if removal == "null":
                        cfg["env"][key.lower()] = None
                    elif removal == "override":
                        generated.pop(key.title())
                        cfg["memory_policy"]["overrides"] = {"env": {key.lower(): None}}
                    original = prep.json_bytes(cfg)
                    path.write_bytes(original)
                    for _ in range(2):
                        cfg = prep.refreshed_config(cfg)
                        self.assertIsNone(cfg["memory_policy"]["overrides"]["env"][key])
                    create, _, _ = self.run_main(path)
                    runtime = report.call_args.args[1]
                    child = create.call_args.kwargs["env"]
                    self.assertNotIn(key, child)
                    self.assertNotIn(key.lower(), child)
                    self.assertNotIn(key, runtime["memory_policy"]["sample"]["env"])
                    if "HEADROOM" in key:
                        self.assertEqual(runtime["memory_policy"]["sample"]["headroom_gib"], 4)
                    self.assertNotIn(key, safe_engine_environment(runtime, {key.lower(): "8", key: "8"}))
                    self.assertEqual(path.read_bytes(), original)

    def test_managed_environment_canonicalization_preserves_static_unknown_and_posix_rules(self):
        from tools import prepare_unsloth_q2 as prep
        key = "STRATA_RESIDENT_HEADROOM_GIB"
        inherited = {key: "2", "Unknown_Case": "inherited"}
        configured = {key.lower(): "8", "Unknown_Case": "config"}
        policy = {"version": 1, "generated_env": {key: "4"}, "overrides": {"env": {}}}
        cfg = {"env": configured, "memory_policy": policy}
        original = json.dumps(cfg, sort_keys=True)
        with patch.object(server.os, "name", "nt"):
            self.assertEqual(server.config_environment(cfg, inherited), {key: "8", "Unknown_Case": "config"})
            static = server.config_environment({"env": configured}, inherited)
            self.assertEqual(static, {key: "2", key.lower(): "8", "Unknown_Case": "config"})
            self.assertEqual(prep.refreshed_config({"env": configured}), {"env": configured})
            for layer in ("inherited", "config"):
                aliases = {key: "8", key.lower(): 8}
                with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "Conflicting managed"):
                    server.config_environment({**cfg, "env": aliases} if layer == "config" else cfg,
                                              aliases if layer == "inherited" else inherited)
        with patch.object(server.os, "name", "posix"):
            # Uppercase deletion stays distinct from a lowercase unknown variable on POSIX.
            self.assertEqual(server.config_environment(cfg, inherited),
                             {key.lower(): "8", "Unknown_Case": "config"})
        self.assertEqual(json.dumps(cfg, sort_keys=True), original)

    def test_removed_headroom_admission_preserves_physical_and_commit_limits(self):
        from tools import prepare_unsloth_q2 as prep
        key = "STRATA_RESIDENT_HEADROOM_GIB"
        for available, commit, admitted in ((10, 60, True), (27, 25, True), (8, 60, False), (27, 23, False)):
            with self.subTest(available=available, commit=commit), tempfile.TemporaryDirectory() as directory, \
                    patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                        "available_gib": available, "commit_available_gib": commit, "windows_commit": True}), \
                    patch.object(prep, "detected_gpu", return_value={"vram_gb": 16, "index": 0}), \
                    patch.object(prep.setup, "find_nvcc", return_value=(None, None)), \
                    patch.object(prep, "check_ready") as ready, \
                    patch.object(prep, "admit_automatic_launch", wraps=prep.admit_automatic_launch) as admit, \
                    patch.dict(os.environ, {key: "8"}, clear=True):
                path, cfg = self.fixture(Path(directory), prep)
                cfg["env"].pop(key)
                original = prep.json_bytes(cfg)
                path.write_bytes(original)
                if admitted:
                    create, _, _ = self.run_main(path)
                    self.assertNotIn(key, create.call_args.kwargs["env"])
                else:
                    with patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config", str(path)]), \
                            patch.object(server, "StrataEngine") as engine, patch.object(server, "Server") as http, \
                            contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        server.main()
                    self.assertEqual(error.exception.code, 2)
                    engine.assert_not_called()
                    http.assert_not_called()
                    ready.assert_not_called()
                self.assertEqual(admit.call_args.args[0]["headroom_gib"], 4)
                self.assertEqual(path.read_bytes(), original)

    def test_generated_numeric_and_static_headroom_keep_their_environment_precedence(self):
        from tools import prepare_unsloth_q2 as prep
        key = "STRATA_RESIDENT_HEADROOM_GIB"
        for mode, expected in (("generated", "4"), ("numeric", "2"), ("static", "8")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, \
                    patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                        "available_gib": 27, "commit_available_gib": 60, "windows_commit": True}), \
                    patch.object(prep, "detected_gpu", return_value={"vram_gb": 16, "index": 0}), \
                    patch.object(prep.setup, "find_nvcc", return_value=(None, None)), \
                    patch.object(prep, "check_ready"), patch.dict(os.environ, {key: "8"}, clear=True):
                path, cfg = self.fixture(Path(directory), prep)
                if mode == "numeric":
                    cfg["env"][key] = "2"
                elif mode == "static":
                    cfg.pop("memory_policy")
                    cfg["env"].pop(key)
                original = prep.json_bytes(cfg)
                path.write_bytes(original)
                create, _, _ = self.run_main(path)
                self.assertEqual(create.call_args.kwargs["env"][key], expected)
                self.assertEqual(path.read_bytes(), original)

    def test_environment_removal_requires_known_managed_names_and_policy_version(self):
        key = "STRATA_RESIDENT_HEADROOM_GIB"
        inherited = {key: "8", "UNRELATED_CONTROL": "keep"}
        policy = {"version": 1, "generated_env": {},
                  "overrides": {"env": {key: None, "UNRELATED_CONTROL": None}}}
        self.assertEqual(server.config_environment({"memory_policy": policy, "env": {}}, inherited),
                         {"UNRELATED_CONTROL": "keep"})
        for invalid in ({}, {**policy, "version": 999}, {**policy, "generated_env": []}, "unknown"):
            with self.subTest(policy=invalid):
                self.assertEqual(server.config_environment({"memory_policy": invalid, "env": {}}, inherited), inherited)
        for platform, expected in (("nt", {}), ("posix", {key.lower(): "8"})):
            with self.subTest(platform=platform), patch.object(server.os, "name", platform):
                self.assertEqual(server.config_environment({"memory_policy": policy, "env": {}},
                                                           {key.lower(): "8"}), expected)

    def test_static_main_keeps_bytes_and_skips_policy_preparation(self):
        from tools import prepare_unsloth_q2 as prep
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                    "available_gib": 27, "commit_available_gib": 45, "windows_commit": True}), \
                patch.object(prep, "detected_gpu", return_value={"vram_gb": 15.99, "index": 0}), \
                patch.object(prep.setup, "find_nvcc", return_value=(None, None)), patch.dict(os.environ, {}, clear=True):
            path, cfg = self.fixture(Path(directory), prep)
            cfg.pop("memory_policy")
            path.write_bytes(json.dumps(cfg, indent=3).encode())
            original = path.read_bytes()
            with patch.object(prep, "prepare_launch", side_effect=AssertionError("static policy preparation")):
                for gpu in ("2", "0,2"):
                    create, _, _ = self.run_main(path, ["--gpu", gpu])
                    self.assertEqual(create.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], gpu)
            self.assertEqual(path.read_bytes(), original)

    def test_automatic_gpu_environment_matches_the_sampled_physical_device(self):
        from tools import prepare_unsloth_q2 as prep
        lower_mask = "cuda_visible_devices" if os.name == "nt" else "CUDA_VISIBLE_DEVICES"
        for selected, explicit_env, expected in (
                (None, {}, 2), (None, {lower_mask: "1"}, 1),
                (1, {lower_mask: "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, 1)):
            with self.subTest(selected=selected, env=explicit_env), tempfile.TemporaryDirectory() as directory, \
                    patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                        "available_gib": 17, "commit_available_gib": 45, "windows_commit": True}) as memory, \
                    patch.object(prep, "detected_gpu", side_effect=lambda pick=None: {
                        "vram_gb": 15.99, "index": 2 if pick is None else pick}) as gpu, \
                    patch.object(prep.setup, "find_nvcc", return_value=(None, None)), \
                    patch.object(prep, "check_ready"), patch.object(prep, "atomic_files") as publish, \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "9", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, clear=True):
                path, cfg = self.fixture(Path(directory), prep)
                if selected is None:
                    cfg.pop("gpu")
                else:
                    cfg["gpu"] = selected
                cfg["env"].update(explicit_env)
                path.write_bytes(json.dumps(cfg, indent=3).encode())
                original = path.read_bytes()
                gpu.reset_mock()
                memory.reset_mock()
                def before_engine():
                    gpu.assert_called_once_with(expected if explicit_env or selected is not None else None)
                    memory.assert_called_once()
                create, _, _ = self.run_main(path, before_engine=before_engine)
                child = create.call_args.kwargs["env"]
                self.assertEqual(child["CUDA_VISIBLE_DEVICES"], str(expected))
                self.assertEqual(child["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
                publish.assert_not_called()
                self.assertEqual(path.read_bytes(), original)

    def test_automatic_invalid_gpu_environment_and_context_stop_before_sampling_or_engine(self):
        from tools import prepare_unsloth_q2 as prep
        lower_order = "cuda_device_order" if os.name == "nt" else "CUDA_DEVICE_ORDER"
        cases = [({"CUDA_VISIBLE_DEVICES": value}, "131072") for value in ("1", "0,1", "", "GPU-uuid", None)]
        cases += [({lower_order: "FASTEST_FIRST"}, "131072"),
                  ({"CUDA_VISIBLE_DEVICES": "0", "cuda_visible_devices": "1"}, "131072")] if os.name == "nt" else [
                      ({lower_order: "FASTEST_FIRST"}, "131072")]
        cases += [({}, context) for context in (None, "broken", "0", "-1", "1.5", 123)]
        for explicit_env, context in cases:
            with self.subTest(env=explicit_env, context=context), tempfile.TemporaryDirectory() as directory, \
                    patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                        "available_gib": 27, "commit_available_gib": 45, "windows_commit": True}) as memory, \
                    patch.object(prep, "detected_gpu", return_value={"vram_gb": 15.99, "index": 0}) as gpu, \
                    patch.object(prep.setup, "find_nvcc", return_value=(None, None)):
                path, cfg = self.fixture(Path(directory), prep)
                cfg["env"].update(explicit_env)
                prep.set_flag(cfg["args"], "--max-context", context)
                if isinstance(context, int):
                    cfg["args"][cfg["args"].index("--max-context") + 1] = context
                path.write_bytes(prep.json_bytes(cfg))
                original = path.read_bytes()
                memory.reset_mock()
                gpu.reset_mock()
                stderr = io.StringIO()
                with patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config", str(path)]), \
                        patch.object(server, "StrataEngine") as engine, patch.object(server, "Server") as http, \
                        patch.object(prep, "atomic_files") as publish, contextlib.redirect_stderr(stderr), \
                        self.assertRaises(SystemExit) as error:
                    server.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn("--max-context" if context != "131072" else "CUDA", stderr.getvalue())
                memory.assert_not_called()
                gpu.assert_not_called()
                engine.assert_not_called()
                http.assert_not_called()
                publish.assert_not_called()
                self.assertEqual(path.read_bytes(), original)

    def test_automatic_admission_and_readiness_failures_prevent_engine_start(self):
        from tools import prepare_unsloth_q2 as prep
        for available, commit, readiness in ((None, 45, None), (27, None, None),
                                              (0, 45, None), (27, 45, "stale engine receipt")):
            with self.subTest(available=available, commit=commit, readiness=readiness), \
                    tempfile.TemporaryDirectory() as directory, \
                    patch.object(prep, "detected_memory", return_value={"ram_gib": 30.88,
                        "available_gib": 27, "commit_available_gib": 45, "windows_commit": True}) as memory, \
                    patch.object(prep, "detected_gpu", return_value={"vram_gb": 15.99, "index": 0}), \
                    patch.object(prep.setup, "find_nvcc", return_value=(None, None)):
                path, _ = self.fixture(Path(directory), prep)
                original = path.read_bytes()
                memory.return_value.update(available_gib=available, commit_available_gib=commit)
                with patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config", str(path)]), \
                        patch.object(prep, "check_ready", side_effect=ValueError(readiness) if readiness else None), \
                        patch.object(server, "StrataEngine") as engine, patch.object(server, "Server") as http, \
                        patch.object(prep, "atomic_files") as publish, contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit) as error:
                    server.main()
                self.assertEqual(error.exception.code, 2)
                engine.assert_not_called()
                http.assert_not_called()
                publish.assert_not_called()
                self.assertEqual(path.read_bytes(), original)

    def test_failed_automatic_preparation_stops_public_main_before_engine_or_http(self):
        from tools import prepare_unsloth_q2 as prep
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text('{"memory_policy": {"version": 1}}')
            with patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config", str(path)]), \
                    patch.object(prep, "prepare_launch", side_effect=ValueError("not ready")) as prepare, \
                    patch.object(server, "StrataEngine") as engine, patch.object(server, "Server") as http, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                server.main()
            self.assertEqual(error.exception.code, 2)
            prepare.assert_called_once_with(path, gpu=None)
            engine.assert_not_called()
            http.assert_not_called()


class Http(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "strata-q2_0.json"
        self.path.write_text(json.dumps(CFG, indent=1), encoding="utf-8")
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.config_path = str(self.path)
        self.httpd = serve(self.svc, port=0)
        self.host = f"127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def call(self, method, body=None, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(f"http://{self.host}/config", method=method, headers=h,
                                     data=None if body is None else (body if isinstance(body, bytes)
                                                                     else json.dumps(body).encode()))
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def saved(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_read_and_write_from_the_page(self):
        status, v = self.call("GET")
        self.assertEqual(status, 200, v)
        self.assertEqual({k["key"]: k["value"] for k in v["keys"]}["sampling.temperature"], 0.7)
        with contextlib.redirect_stdout(io.StringIO()):
            status, v = self.call("POST", {"set": {"sampling.temperature": 1.0, "open_browser": False}},
                                  {"Origin": f"http://{self.host}"})
        self.assertEqual(status, 200, v)
        self.assertEqual(v["changed"], ["sampling.temperature", "open_browser"])
        cfg = self.saved()
        self.assertEqual(cfg["sampling"]["temperature"], 1.0)
        self.assertIs(cfg["open_browser"], False)
        for k in ("mcp_servers", "cors_origins", "host", "args", "exe"):
            self.assertEqual(cfg[k], CFG[k], k)
        self.assertEqual(json.loads((self.path.parent / "strata-q2_0.json.bak").read_text(encoding="utf-8")), CFG)
        status, v = self.call("POST", {"set": {"open_browser": False}})   # no Origin (a script on this PC): allowed
        self.assertEqual((status, v["changed"]), (200, []))

    def test_only_from_strata_s_own_page(self):
        before = self.path.read_bytes()
        status, _ = self.call("POST", {"set": {"fit_max_tokens": True}}, {"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.call("POST", b'{"set": {"fit_max_tokens": true}}', {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        for body in ({"set": {"api_key": ""}}, {"set": {"mcp_servers": {"x": {"command": "calc"}}}},
                     {"set": {"sampling.top_k": 0}}, {"nothing": 1}, b"not json"):
            status, b = self.call("POST", body, {"Origin": f"http://{self.host}"})
            self.assertEqual(status, 400, (body, b))
        self.assertEqual(self.path.read_bytes(), before)                 # nothing was written
        self.assertFalse((self.path.parent / "strata-q2_0.json.bak").exists())

    def test_the_api_key(self):
        self.svc.api_key = "secret"
        self.assertEqual(self.call("GET")[0], 401)
        self.assertEqual(self.call("POST", {"set": {"fit_max_tokens": True}})[0], 401)
        with contextlib.redirect_stdout(io.StringIO()):
            status, _ = self.call("POST", {"set": {"fit_max_tokens": True}}, {"Authorization": "Bearer secret"})
        self.assertEqual(status, 200)
        self.assertIs(self.saved()["fit_max_tokens"], True)

    def test_without_a_run_config(self):
        self.svc.config_path = None
        self.assertEqual(self.call("GET")[0], 404)
        self.assertEqual(self.call("POST", {"set": {"fit_max_tokens": True}})[0], 404)

    def test_the_page_has_the_view(self):
        web = ROOT / "serve" / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cfg-card", "cfg-form", "cfg-save", "cfg-msg"):
            self.assertIn(f'id="{el}"', html)
        self.assertIn('fetch("config", {method: "POST", headers: headers(true)', js)


if __name__ == "__main__":
    unittest.main()
