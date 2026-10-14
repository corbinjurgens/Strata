"""Offline checks for distinct hardware budgets and read-only shared model verification."""
import ast
import hashlib
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, Mock

from tools import prepare_unsloth_q2 as prep


def value(args, flag):
    return args[args.index(flag) + 1]


class MemoryProfiles(unittest.TestCase):
    def test_desktop_uses_its_ram_budget_and_normal_prefill(self):
        args, env = prep.memory_settings(30.88, 15.99, 8192, available_gib=27, commit_available_gib=45)
        self.assertEqual(value(args, "--resident-budget-gib"), "12")
        self.assertEqual(env["STRATA_FILE_CACHE_MIB"], "8192")
        self.assertEqual(env["STRATA_FILE_CACHE_COMPACT"], "1")
        self.assertEqual(value(args, "--prefill"), "auto")
        self.assertEqual(value(args, "--kv"), "int8")
        self.assertNotIn("--no-prefill-borrow", args)
        self.assertNotIn("STRATA_OWNED_PRICE", env)
        self.assertNotIn("--mtp", args)

    def test_laptop_keeps_small_prompt_buffers_and_desktop_ram_headroom(self):
        args, env = prep.memory_settings(63.7, 6, 8192, available_gib=58, commit_available_gib=70)
        self.assertEqual(value(args, "--resident-budget-gib"), "30")
        self.assertEqual(value(args, "--prefill"), "64")
        self.assertIn("--no-prefill-borrow", args)
        self.assertEqual(env["STRATA_OWNED_PRICE"], "exact")
        self.assertEqual(env["STRATA_RESIDENT_HEADROOM_GIB"], "8")

    def test_64k_streaming_is_different_on_the_two_gpus(self):
        laptop, _ = prep.memory_settings(63.7, 6, 65536, available_gib=58, commit_available_gib=70)
        desktop, _ = prep.memory_settings(30.88, 15.99, 65536, available_gib=27, commit_available_gib=45)
        self.assertEqual(value(laptop, "--kv"), "q4_0")
        self.assertEqual(value(laptop, "--kv-resident"), "20480")
        self.assertEqual(value(laptop, "--prefill"), "256")
        self.assertEqual(value(desktop, "--kv"), "int8")
        self.assertEqual(value(desktop, "--kv-resident"), "32768")
        self.assertEqual(value(desktop, "--resident-budget-gib"), "12")

    def test_128k_preserves_the_small_gpu_buffer_limit_and_desktop_ram_limit(self):
        laptop, _ = prep.memory_settings(63.7, 6, 131072, available_gib=58, commit_available_gib=70)
        desktop, desktop_env = prep.memory_settings(30.88, 15.99, 131072, available_gib=27, commit_available_gib=45)
        self.assertEqual(value(laptop, "--max-context"), "131072")
        self.assertEqual(value(laptop, "--prefill"), "64")
        self.assertEqual(value(laptop, "--kv"), "q4_0")
        self.assertEqual(value(laptop, "--resident-budget-gib"), "30")
        self.assertIn("--no-prefill-borrow", laptop)
        self.assertEqual(value(desktop, "--max-context"), "131072")
        self.assertEqual(value(desktop, "--kv"), "int8")
        self.assertEqual(value(desktop, "--kv-resident"), "32768")
        self.assertEqual(value(desktop, "--resident-budget-gib"), "12")
        self.assertEqual(desktop_env["STRATA_FILE_CACHE_MIB"], "6476")

    def test_gpu_commit_is_not_subtracted_from_physical_ram(self):
        plan = prep.memory_plan(30.88, 15.99, 131072, available_gib=27, commit_available_gib=60)
        self.assertAlmostEqual(plan["expert_pool_gib"], 18.32421875)
        self.assertAlmostEqual(plan["kv_ram_gib"], 1.67578125)
        self.assertEqual(plan["later_startup_gib"], 1)

    def test_small_gpu_reserve_matches_context_and_commit_budget(self):
        for context, reserve in ((8192, 448), (32768, 448), (65536, 448), (131072, 400)):
            with self.subTest(context=context):
                plan = prep.memory_plan(63.7, 6, context, available_gib=58, commit_available_gib=40)
                self.assertEqual(value(plan["args"], "--vram-reserve-mib"), str(reserve))
                self.assertEqual(plan["gpu_commit_gib"], 6 - reserve / 1024)
                self.assertAlmostEqual(plan["expert_pool_gib"],
                                       40 - 8 - 0.5 - 2 - plan["kv_ram_gib"] - (6 - reserve / 1024))

    def test_tight_commit_reduces_both_caches_without_changing_context(self):
        plan = prep.memory_plan(30.88, 15.99, 131072, available_gib=27, commit_available_gib=28)
        self.assertEqual(plan["pinned_gib"], 3)
        self.assertEqual(plan["recent_mib"], 1366)
        self.assertEqual(value(plan["args"], "--max-context"), "131072")

    def test_tight_physical_ram_reduces_cache_even_with_abundant_commit(self):
        plan = prep.memory_plan(30.88, 15.99, 131072, available_gib=12, commit_available_gib=60)
        self.assertEqual(plan["pinned_gib"], 2)
        self.assertEqual(plan["recent_mib"], 1356)

    def test_low_memory_has_no_implicit_full_residency_or_context_reduction(self):
        plan = prep.memory_plan(30.88, 15.99, 131072, available_gib=3, commit_available_gib=4)
        self.assertEqual(value(plan["args"], "--resident-budget-gib"), "0.25")
        self.assertEqual(value(plan["args"], "--max-context"), "131072")
        self.assertEqual(plan["recent_mib"], 0)
        self.assertTrue(plan["low_memory"])

    def test_missing_available_telemetry_has_a_reported_conservative_fallback(self):
        plan = prep.memory_plan(30.88, 15.99, 131072)
        self.assertAlmostEqual(plan["available_fallback_gib"], 20.072)
        self.assertEqual(plan["pinned_gib"], 7)

    def test_explicit_pinned_request_reduces_automatic_recent_request(self):
        plan = prep.memory_plan(30.88, 15.99, 131072, available_gib=27, commit_available_gib=45,
                                overrides={"args": {"--resident-budget-gib": "17"}})
        self.assertEqual(plan["pinned_gib"], 17)
        self.assertEqual(plan["recent_mib"], 1356)

    def test_explicit_recent_request_reduces_automatic_pinned_request(self):
        plan = prep.memory_plan(30.88, 15.99, 131072, available_gib=27, commit_available_gib=45,
                                overrides={"env": {"STRATA_FILE_CACHE_MIB": "12288"}})
        self.assertEqual(plan["pinned_gib"], 6)
        self.assertEqual(plan["recent_mib"], 12288)

    def test_explicit_large_request_is_preserved_and_reported(self):
        plan = prep.memory_plan(30.88, 15.99, 131072, available_gib=12, commit_available_gib=28,
                                overrides={"args": {"--resident-budget-gib": "20"}})
        self.assertEqual(plan["pinned_gib"], 20)
        self.assertEqual(plan["recent_mib"], 0)
        self.assertTrue(plan["over_budget"])

    def test_finite_oversized_profile_request_does_not_overflow_recent_accounting(self):
        plan = prep.memory_plan(32, 16, 131072, available_gib=27, commit_available_gib=45,
                                overrides={"args": {"--resident-budget-gib": "1e308"}})
        self.assertEqual(value(plan["args"], "--resident-budget-gib"), "1e308")
        self.assertEqual(plan["pinned_gib"], 1e308)
        self.assertEqual(plan["recent_mib"], 0)
        self.assertTrue(plan["over_budget"])

    def test_explicit_kv_format_changes_the_ram_allowance(self):
        plan = prep.memory_plan(63.7, 6, 131072, available_gib=58, commit_available_gib=70,
                                overrides={"args": {"--kv": "int8"}})
        self.assertAlmostEqual(plan["kv_ram_gib"], 1.67578125)
        self.assertEqual(value(plan["args"], "--kv"), "int8")

    def test_fp16_and_q4_alias_use_their_source_defined_cell_sizes(self):
        for kv, gib in (("fp16", 3.25), ("q4", 0.9140625), ("q4_0", 0.9140625), ("k8v4", 1.294921875)):
            with self.subTest(kv=kv):
                plan = prep.memory_plan(30.88, 15.99, 131072, available_gib=27, commit_available_gib=45,
                                        overrides={"args": {"--kv": kv}})
                self.assertEqual(value(plan["args"], "--kv"), kv)
                self.assertEqual(plan["kv_ram_gib"], gib)

    def test_explicit_vram_reserve_changes_commit_exposure(self):
        plan = prep.memory_plan(30.88, 16, 131072, available_gib=27, commit_available_gib=28,
                                overrides={"args": {"--vram-reserve-mib": "0"}})
        self.assertEqual(plan["gpu_commit_gib"], 16)
        self.assertEqual(plan["expert_pool_gib"], 3.32421875)
        self.assertEqual(plan["pinned_gib"], 2)

    def test_explicit_environment_removals_are_omitted_from_plans(self):
        for key in prep.MEMORY_ENV:
            with self.subTest(key=key):
                plan = prep.memory_plan(64, 6, 131072, available_gib=50, commit_available_gib=60,
                                        overrides={"env": {key: None}})
                self.assertNotIn(key, plan["env"])
                self.assertEqual(plan["automatic_host_cache"], key != "STRATA_FILE_CACHE_MIB")
                if key == "STRATA_RESIDENT_HEADROOM_GIB":
                    self.assertEqual(plan["headroom_gib"], 4)

    def test_shared_planning_rejects_invalid_numeric_policy_inputs(self):
        for group, name, invalid in (
                ("args", "--resident-budget-gib", ("nan", "inf", "-inf", "0", "-1", "", False)),
                ("args", "--vram-reserve-mib", ("nan", "inf", "-inf", "-1", "", False)),
                ("env", "STRATA_RESIDENT_HEADROOM_GIB", ("nan", "inf", "-inf", "-1", "", False)),
                ("env", "STRATA_FILE_CACHE_MIB", ("nan", "inf", "-1", "1.5", 1.5, "", False))):
            for invalid_value in invalid:
                with self.subTest(name=name, value=invalid_value), self.assertRaisesRegex(ValueError, name):
                    prep.memory_plan(32, 16, 131072, overrides={group: {name: invalid_value}})

    def test_shared_planning_rejects_python_only_numeric_spellings(self):
        for group, name, invalid in (
                ("args", "--resident-budget-gib", ("1_0", "١٠", "10\u00a0", "0x10", "10GiB")),
                ("args", "--vram-reserve-mib", ("1_024", "١٠٢٤", "1024\u00a0", "0x400")),
                ("env", "STRATA_RESIDENT_HEADROOM_GIB", ("1_0", "١٠", "10\u00a0", "0x10")),
                ("env", "STRATA_FILE_CACHE_MIB", ("1_024", "١٠٢٤", "1024 ", "1024\n", "\u00a01024"))):
            for invalid_value in invalid:
                with self.subTest(name=name, value=invalid_value), self.assertRaisesRegex(ValueError, name):
                    prep.memory_plan(32, 16, 131072, overrides={group: {name: invalid_value}})

    def test_c_compatible_decimal_and_integer_whitespace_is_preserved(self):
        plan = prep.memory_plan(32, 16, 131072, available_gib=27, commit_available_gib=45,
                                overrides={"args": {"--resident-budget-gib": " \t+1.2e1 \r\n"},
                                           "env": {"STRATA_RESIDENT_HEADROOM_GIB": " \t4.e0 \r\n",
                                                   "STRATA_FILE_CACHE_MIB": " \t+0001024"}})
        self.assertEqual(plan["pinned_gib"], 12)
        self.assertEqual(plan["headroom_gib"], 4)
        self.assertEqual(plan["recent_mib"], 1024)
        self.assertEqual(value(plan["args"], "--resident-budget-gib"), " \t+1.2e1 \r\n")
        self.assertEqual(plan["env"]["STRATA_FILE_CACHE_MIB"], " \t+0001024")

    def test_recent_mib_engine_range_accepts_the_maximum_and_rejects_overflow(self):
        for recent in (17592186044415, "17592186044415"):
            with self.subTest(recent=recent):
                plan = prep.memory_plan(32, 16, 131072, available_gib=27, commit_available_gib=45,
                                        overrides={"env": {"STRATA_FILE_CACHE_MIB": recent}})
                self.assertEqual(plan["recent_mib"], 17592186044415)
                self.assertEqual(plan["env"]["STRATA_FILE_CACHE_MIB"], recent)
                self.assertEqual(plan["pinned_gib"], 0.25)
                self.assertTrue(plan["over_budget"])
        for recent in (17592186044416, "17592186044416", 10**312, "1" + "0" * 312):
            with self.subTest(recent=recent), self.assertRaisesRegex(ValueError, "STRATA_FILE_CACHE_MIB.*17592186044415"):
                prep.memory_plan(32, 16, 131072, overrides={"env": {"STRATA_FILE_CACHE_MIB": recent}})

    def test_effective_vram_reserve_rejects_signed_int_prefix_overflow(self):
        for reserve in ("2147483647", "+2147483647.9"):
            with self.subTest(reserve=reserve):
                plan = prep.memory_plan(32, 16, 131072, overrides={"args": {"--vram-reserve-mib": reserve}})
                self.assertEqual(plan["vram_reserve_mib"], 2147483647)
                self.assertEqual(value(plan["args"], "--vram-reserve-mib"), reserve)
        for reserve in ("2147483648", "2147483648e-3"):
            with self.subTest(reserve=reserve), self.assertRaisesRegex(ValueError, "effective integer prefix"):
                prep.memory_plan(32, 16, 131072, overrides={"args": {"--vram-reserve-mib": reserve}})

    def test_zero_recent_headroom_and_reserve_are_valid(self):
        plan = prep.memory_plan(32, 16, 131072, available_gib=27, commit_available_gib=45,
                                overrides={"args": {"--resident-budget-gib": "100", "--vram-reserve-mib": "0"},
                                           "env": {"STRATA_FILE_CACHE_MIB": 0, "STRATA_RESIDENT_HEADROOM_GIB": "0"}})
        self.assertEqual(plan["pinned_gib"], 100)
        self.assertEqual(plan["recent_mib"], 0)
        self.assertEqual(plan["headroom_gib"], 0)
        self.assertEqual(value(plan["args"], "--vram-reserve-mib"), "0")
        self.assertTrue(plan["over_budget"])
        self.assertFalse(plan["automatic_host_cache"])

    def test_windows_commit_requirement_is_explicit_even_with_the_same_fallback_pool(self):
        windows = prep.memory_plan(32, 16, 131072, available_gib=27, windows_commit=True)
        linux = prep.memory_plan(32, 16, 131072, available_gib=27, windows_commit=False)
        self.assertEqual(windows["expert_pool_gib"], linux["expert_pool_gib"])
        self.assertTrue(windows["windows_commit"])
        self.assertFalse(linux["windows_commit"])

    def test_flag_queries_reject_duplicate_managed_values_and_switches(self):
        for flag in ("--expert-cache", "--max-context", "--kv", "--kv-resident", "--spec",
                     "--suffix-draft", "--lookup-chain", "--resident-budget-gib", "--prefill",
                     "--vram-reserve-mib", "--no-prefill-borrow"):
            for first, second in (("first", "last"), ("same", "same")):
                args = [flag, flag] if flag == "--no-prefill-borrow" else [flag, first, flag, second]
                with self.subTest(flag=flag, values=(first, second)), self.assertRaisesRegex(ValueError, f"Duplicate managed flag {flag}"):
                    prep.flag_value(args, flag)


class PortableProfiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "strata-test.json"
        for target, replacement in (("ROOT", self.root), ("detected_memory", Mock(return_value={
                "ram_gib": 30.88, "available_gib": 27, "commit_available_gib": 45, "windows_commit": True}))):
            patcher = patch.object(prep, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target, replacement in (("gpu_info", Mock(return_value={"vram_gb": 15.99, "index": 0, "arch": "89"})),
                                    ("find_nvcc", Mock(return_value=(None, None)))):
            patcher = patch.object(prep.setup, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_config(self, overrides=None):
        return prep.config_for(self.root / "source", self.root / "pack", 131072, None, overrides=overrides)

    def assert_launch_rejected(self, cfg, message):
        original = json.dumps(cfg, indent=4).encode()
        self.config.write_bytes(original)
        with patch.object(prep, "check_ready"), patch.object(prep, "atomic_files") as publish, \
                patch.object(prep.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, message):
                prep.launch(self.config)
        publish.assert_not_called()
        run.assert_not_called()
        self.assertEqual(self.config.read_bytes(), original)

    def test_new_profile_uses_distinct_artifacts_auth_fit_and_no_browser(self):
        cfg = self.make_config()
        self.assertEqual(Path(cfg["exe"]).name, "strata-unsloth-portable.exe")
        self.assertEqual(Path(cfg["build_metadata"]).name, "BUILD-unsloth-portable.json")
        self.assertGreaterEqual(len(cfg["api_key"]), 32)
        self.assertTrue(cfg["fit_max_tokens"])
        self.assertFalse(cfg["open_browser"])
        self.assertEqual(cfg["host"], "127.0.0.1")

    def test_existing_machine_key_host_and_port_are_preserved(self):
        key_config = self.root / "credentials.json"
        key_config.write_text(json.dumps({"api_key": "local-test-key", "host": "0.0.0.0", "port": 8182}))
        cfg = prep.config_for(self.root, self.root / "pack", 131072, key_config)
        self.assertEqual((cfg["api_key"], cfg["host"], cfg["port"]), ("local-test-key", "0.0.0.0", 8182))

    def test_existing_config_is_preserved_byte_for_byte(self):
        cfg = self.make_config()
        cfg.update(fit_max_tokens=False, open_browser=True, port=8182)
        cfg["args"] += ["--pcie-frac", "0.42"]
        original = json.dumps(cfg, indent=4).encode()
        self.config.write_bytes(original)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prep.main(["--write-config", "--config", str(self.config), "--context", "8192"]), 0)
        self.assertEqual(self.config.read_bytes(), original)
        self.assertTrue((self.root / ("run-test.bat" if prep.os.name == "nt" else "run-test.sh")).is_file())

    def test_launch_refresh_reduces_budgets_and_preserves_manual_values(self):
        cfg = self.make_config()
        original = json.dumps(cfg, sort_keys=True)
        prep.detected_memory.return_value = {"ram_gib": 30.88, "available_gib": 17,
                                            "commit_available_gib": 45, "windows_commit": True}
        refreshed = prep.refreshed_config(cfg)
        self.assertEqual(value(refreshed["args"], "--resident-budget-gib"), "5")
        self.assertEqual(refreshed["env"]["STRATA_FILE_CACHE_MIB"], "3404")
        self.assertEqual(json.dumps(cfg, sort_keys=True), original)
        prep.set_flag(cfg["args"], "--resident-budget-gib", "9")
        refreshed = prep.refreshed_config(cfg)
        self.assertEqual(value(refreshed["args"], "--resident-budget-gib"), "9")
        self.assertEqual(refreshed["env"]["STRATA_FILE_CACHE_MIB"], "0")
        self.assertEqual(refreshed["api_key"], cfg["api_key"])

    def test_explicit_override_equal_to_old_auto_value_stays_explicit(self):
        cfg = self.make_config({"args": {"--resident-budget-gib": "12"},
                                "env": {"STRATA_FILE_CACHE_MIB": "6476"}})
        prep.detected_memory.return_value["available_gib"] = 12
        refreshed = prep.refreshed_config(cfg)
        self.assertEqual(value(refreshed["args"], "--resident-budget-gib"), "12")
        self.assertEqual(refreshed["env"]["STRATA_FILE_CACHE_MIB"], "6476")

    def test_launch_refresh_uses_current_gpu_and_ram_for_laptop(self):
        cfg = self.make_config()
        prep.detected_memory.return_value = {"ram_gib": 63.7, "available_gib": 58,
                                            "commit_available_gib": 70, "windows_commit": True}
        prep.setup.gpu_info.return_value = {"vram_gb": 6, "index": 0, "arch": "89"}
        refreshed = prep.refreshed_config(cfg)
        self.assertEqual(value(refreshed["args"], "--resident-budget-gib"), "30")
        self.assertEqual(value(refreshed["args"], "--kv"), "q4_0")
        self.assertEqual(value(refreshed["args"], "--max-context"), "131072")
        self.assertEqual(value(refreshed["args"], "--prefill"), "64")
        self.assertEqual(value(refreshed["args"], "--vram-reserve-mib"), "400")
        self.assertIn("--no-prefill-borrow", refreshed["args"])

    def test_laptop_refresh_preserves_explicit_vram_reserve(self):
        cfg = self.make_config({"args": {"--vram-reserve-mib": "448"}})
        prep.detected_memory.return_value = {"ram_gib": 63.7, "available_gib": 58,
                                            "commit_available_gib": 40, "windows_commit": True}
        prep.setup.gpu_info.return_value = {"vram_gb": 6, "index": 0, "arch": "89"}
        refreshed = prep.refreshed_config(cfg)
        self.assertEqual(value(refreshed["args"], "--vram-reserve-mib"), "448")
        self.assertEqual(refreshed["memory_policy"]["sample"]["gpu_commit_gib"], 6 - 448 / 1024)
        self.assertEqual(value(prep.refreshed_config(refreshed)["args"], "--vram-reserve-mib"), "448")

    def test_removed_automatic_settings_remain_removed(self):
        cfg = self.make_config()
        cfg["env"].pop("STRATA_FILE_CACHE_MIB")
        cfg["env"].pop("STRATA_RESIDENT_HEADROOM_GIB")
        prep.set_flag(cfg["args"], "--resident-budget-gib", None)
        prep.set_flag(cfg["args"], "--vram-reserve-mib", None)
        for _ in range(2):
            cfg = prep.refreshed_config(cfg)
            self.assertNotIn("STRATA_FILE_CACHE_MIB", cfg["env"])
            self.assertNotIn("STRATA_RESIDENT_HEADROOM_GIB", cfg["env"])
            self.assertNotIn("--resident-budget-gib", cfg["args"])
            self.assertNotIn("--vram-reserve-mib", cfg["args"])

    def test_null_environment_removals_survive_refresh_and_launch_without_exporting_none(self):
        server = Path(__file__).resolve().parents[1] / "serve/server.py"
        node = next(n for n in ast.parse(server.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "child_env")
        namespace = {"os": prep.os, "Path": Path, "hip_visible": lambda cfg: [], "gpu_list": lambda cfg: []}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(server), "exec"), namespace)
        parent_env = {k: v for k, v in prep.os.environ.items() if k not in prep.MEMORY_ENV}
        for key in prep.MEMORY_ENV:
            for source in ("override", "saved", "old-generated-null"):
                with self.subTest(key=key, source=source):
                    cfg = self.make_config({"env": {key: None}} if source == "override" else None)
                    if source != "override":
                        cfg["env"][key] = None
                    if source == "old-generated-null":
                        cfg["memory_policy"]["generated_env"][key] = None
                    for _ in range(2):
                        cfg = prep.refreshed_config(cfg)
                        self.assertNotIn(key, cfg["env"])
                        self.assertNotIn(key, cfg["memory_policy"]["generated_env"])
                        self.assertNotIn(key, cfg["memory_policy"]["sample"]["env"])
                        self.assertIn(key, cfg["memory_policy"]["overrides"]["env"])
                        self.assertIsNone(cfg["memory_policy"]["overrides"]["env"][key])
                    self.config.write_bytes(prep.json_bytes(cfg))
                    with patch.object(prep, "check_ready"), patch.object(prep.subprocess, "run") as run, \
                            contextlib.redirect_stdout(io.StringIO()):
                        run.return_value.returncode = 0
                        self.assertEqual(prep.launch(self.config), 0)
                    run.assert_called_once()
                    saved = json.loads(self.config.read_text())
                    self.assertNotIn(key, saved["env"])
                    self.assertIsNone(saved["memory_policy"]["overrides"]["env"][key])
                    with patch.dict(prep.os.environ, parent_env, clear=True):
                        self.assertNotIn(key, namespace["child_env"](saved))

    def test_failed_launcher_publish_rolls_back_new_config(self):
        real_replace = prep.os.replace
        def fail_launcher(source, target):
            if Path(target).suffix in (".bat", ".sh"):
                raise OSError("launcher publish failed")
            return real_replace(source, target)
        with patch.object(prep.os, "replace", side_effect=fail_launcher):
            with self.assertRaisesRegex(OSError, "launcher publish failed"):
                prep.write_profile(self.config, self.make_config(), create_config=True)
        self.assertFalse(self.config.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_failed_metadata_publish_restores_previous_executable(self):
        engine = self.root / "engine.exe"
        metadata = self.root / "BUILD.json"
        engine.write_bytes(b"working engine")
        metadata.write_bytes(b"original metadata")
        real_replace = prep.os.replace
        def fail_metadata(source, target):
            if target == metadata:
                raise OSError("metadata publish failed")
            return real_replace(source, target)
        with patch.object(prep.os, "replace", side_effect=fail_metadata):
            with self.assertRaisesRegex(OSError, "metadata publish failed"):
                prep.atomic_files({engine: b"new engine", metadata: b"new metadata"})
        self.assertEqual(engine.read_bytes(), b"working engine")
        self.assertEqual(metadata.read_bytes(), b"original metadata")
        self.assertEqual(set(self.root.iterdir()), {engine, metadata})

    def test_launch_uses_persistent_config_and_passes_auth_fit_no_browser(self):
        cfg = self.make_config()
        self.config.write_bytes(prep.json_bytes(cfg))
        observed = {}
        def run(command, **kwargs):
            runtime = Path(command[command.index("--config") + 1])
            observed.update(path=runtime, cfg=json.loads(runtime.read_text()), command=command, env=kwargs["env"])
            return SimpleNamespace(returncode=7)
        with patch.object(prep, "check_ready"), patch.object(prep.subprocess, "run", side_effect=run), \
                patch.dict(prep.os.environ, {"STRATA_FETCH_PROFILE": "1"}):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prep.launch(self.config), 7)
        self.assertEqual(observed["cfg"]["api_key"], cfg["api_key"])
        self.assertTrue(observed["cfg"]["fit_max_tokens"])
        self.assertNotIn("--open", observed["command"])
        self.assertEqual(observed["path"], self.config)
        self.assertNotIn("STRATA_FETCH_PROFILE", observed["env"])
        self.assertEqual(json.loads(self.config.read_text())["api_key"], cfg["api_key"])

    def test_repeated_refresh_does_not_turn_automatic_budget_into_override(self):
        cfg = self.make_config()
        prep.detected_memory.return_value["available_gib"] = 17
        first = prep.refreshed_config(cfg)
        self.assertEqual(value(first["args"], "--resident-budget-gib"), "5")
        prep.detected_memory.return_value["available_gib"] = 27
        second = prep.refreshed_config(first)
        self.assertEqual(value(second["args"], "--resident-budget-gib"), "12")
        self.assertNotIn("--resident-budget-gib", second["memory_policy"]["overrides"]["args"])

    def test_saved_key_takes_precedence_over_inherited_key_at_launch(self):
        for inherited in ("unrelated-key", ""):
            with self.subTest(inherited=inherited):
                cfg = self.make_config()
                self.config.write_bytes(prep.json_bytes(cfg))
                with patch.object(prep, "check_ready"), patch.object(prep.subprocess, "run") as run, \
                        patch.dict(prep.os.environ, {"STRATA_API_KEY": inherited}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    run.return_value.returncode = 0
                    self.assertEqual(prep.launch(self.config), 0)
                self.assertNotIn("STRATA_API_KEY", run.call_args.kwargs["env"])
                self.assertNotIn(cfg["api_key"], run.call_args.args[0])
                self.assertEqual(json.loads(self.config.read_text())["api_key"], cfg["api_key"])

    def test_keyless_static_profile_retains_explicit_environment_authentication(self):
        cfg = self.make_config()
        cfg.pop("memory_policy")
        cfg.pop("api_key")
        self.config.write_bytes(prep.json_bytes(cfg))
        with patch.object(prep, "check_ready"), patch.object(prep.subprocess, "run") as run, \
                patch.dict(prep.os.environ, {"STRATA_API_KEY": "explicit-env-key"}):
            run.return_value.returncode = 0
            prep.launch(self.config)
        self.assertEqual(run.call_args.kwargs["env"]["STRATA_API_KEY"], "explicit-env-key")

    def test_legacy_profile_without_policy_remains_static_without_memory_telemetry(self):
        cfg = self.make_config()
        cfg.pop("memory_policy")
        cfg["args"] += ["--vram-reserve-mib", "1e3"]
        original = prep.json_bytes(cfg)
        self.config.write_bytes(original)
        prep.detected_memory.reset_mock()
        prep.detected_memory.side_effect = ValueError("telemetry unavailable")
        with patch.object(prep, "check_ready"), patch.object(prep, "atomic_files") as publish, \
                patch.object(prep.subprocess, "run") as run:
            run.return_value.returncode = 0
            self.assertEqual(prep.launch(self.config), 0)
        prep.detected_memory.assert_not_called()
        publish.assert_not_called()
        run.assert_called_once()
        self.assertEqual(self.config.read_bytes(), original)

    def test_launch_readiness_failure_neither_saves_nor_starts_server(self):
        self.config.write_bytes(prep.json_bytes(self.make_config()))
        original = self.config.read_bytes()
        with patch.object(prep, "check_ready", side_effect=ValueError("not ready")), \
                patch.object(prep.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "not ready"):
                prep.launch(self.config)
        run.assert_not_called()
        self.assertEqual(self.config.read_bytes(), original)

    def test_launch_rejects_zero_and_subminimum_automatic_pools(self):
        cfg = self.make_config()
        for available, commit in ((0, 45), (27, 0), (8.9248046875, 45)):
            with self.subTest(available=available, commit=commit):
                prep.detected_memory.return_value.update(available_gib=available, commit_available_gib=commit)
                self.assert_launch_rejected(cfg, "at least 0.25 GiB")

    def test_launch_admits_exact_minimum_and_small_positive_pools(self):
        cfg = self.make_config()
        cfg.update(host="0.0.0.0", port=8182)
        for available, pool, recent in ((8.92578125, 0.25, "0"), (9.26171875, 0.5859375, "344")):
            with self.subTest(available=available):
                prep.detected_memory.return_value["available_gib"] = available
                self.config.write_bytes(prep.json_bytes(cfg))
                with patch.object(prep, "check_ready"), patch.object(prep.subprocess, "run") as run, \
                        contextlib.redirect_stdout(io.StringIO()):
                    run.return_value.returncode = 0
                    self.assertEqual(prep.launch(self.config), 0)
                run.assert_called_once()
                saved = json.loads(self.config.read_text())
                self.assertEqual(saved["memory_policy"]["sample"]["expert_pool_gib"], pool)
                self.assertTrue(saved["memory_policy"]["sample"]["low_memory"])
                self.assertEqual(value(saved["args"], "--resident-budget-gib"), "0.25")
                self.assertEqual(saved["env"]["STRATA_FILE_CACHE_MIB"], recent)
                self.assertEqual(value(saved["args"], "--max-context"), "131072")
                self.assertEqual((saved["api_key"], saved["host"], saved["port"]), (cfg["api_key"], "0.0.0.0", 8182))

    def test_launch_rejects_missing_required_memory_telemetry(self):
        cfg = self.make_config()
        for available, commit, windows, message in (
                (None, 45, True, "physical-memory telemetry"),
                (27, None, True, "Windows commit telemetry"),
                (None, None, False, "physical-memory telemetry")):
            with self.subTest(windows=windows, available=available, commit=commit):
                prep.detected_memory.return_value.update(available_gib=available, commit_available_gib=commit,
                                                        windows_commit=windows)
                self.assert_launch_rejected(cfg, message)

    def test_linux_launch_does_not_require_windows_commit(self):
        cfg = self.make_config()
        prep.detected_memory.return_value.update(commit_available_gib=None, windows_commit=False)
        self.config.write_bytes(prep.json_bytes(cfg))
        with patch.object(prep, "check_ready"), patch.object(prep.subprocess, "run") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            run.return_value.returncode = 0
            self.assertEqual(prep.launch(self.config), 0)
        run.assert_called_once()
        sample = json.loads(self.config.read_text())["memory_policy"]["sample"]
        self.assertFalse(sample["windows_commit"])
        self.assertIsNone(sample["commit_available_gib"])
        self.assertEqual(sample["expert_pool_gib"], 18.32421875)

    def test_invalid_saved_values_fail_before_publication_or_server(self):
        for group, name, invalid in (
                ("args", "--resident-budget-gib", ("0", "-1", "nan", "inf")),
                ("args", "--vram-reserve-mib", ("-1", "nan", "inf")),
                ("env", "STRATA_RESIDENT_HEADROOM_GIB", ("-1", "nan", "inf")),
                ("env", "STRATA_FILE_CACHE_MIB", ("-1", "1.5", 1.5))):
            for invalid_value in invalid:
                with self.subTest(name=name, value=invalid_value):
                    cfg = self.make_config()
                    if group == "args":
                        prep.set_flag(cfg["args"], name, invalid_value)
                    else:
                        cfg["env"][name] = invalid_value
                    self.assert_launch_rejected(cfg, name)

    def test_incompatible_saved_numeric_spellings_and_ranges_never_publish_or_start_server(self):
        for group, name, invalid in (
                ("args", "--resident-budget-gib", ("1_0", "١٠", "10\u00a0")),
                ("args", "--vram-reserve-mib", ("1_024", "١٠٢٤", "2147483648", "2147483648e-3")),
                ("env", "STRATA_RESIDENT_HEADROOM_GIB", ("1_0", "١٠", "10\u00a0")),
                ("env", "STRATA_FILE_CACHE_MIB", ("1_024", "١٠٢٤", "1024 ", "\u00a01024", 2**44, 10**312))):
            for invalid_value in invalid:
                with self.subTest(name=name, value=invalid_value):
                    cfg = self.make_config()
                    if group == "args":
                        prep.set_flag(cfg["args"], name, invalid_value)
                    else:
                        cfg["env"][name] = invalid_value
                    self.assert_launch_rejected(cfg, name)

    def test_scientific_and_fractional_vram_reserve_preserve_refresh_identity_and_report_effective_value(self):
        for raw, requested, effective in (("1e3", 1000, 1), ("1.75", 1.75, 1),
                                          (".5", 0.5, 0), (" \t+1024.75 \r", 1024.75, 1024)):
            with self.subTest(raw=raw):
                cfg = self.make_config({"args": {"--vram-reserve-mib": raw}})
                for available in (17, 27):
                    prep.detected_memory.return_value["available_gib"] = available
                    cfg = prep.refreshed_config(cfg)
                    self.assertEqual(value(cfg["args"], "--vram-reserve-mib"), raw)
                    self.assertEqual(cfg["memory_policy"]["overrides"]["args"]["--vram-reserve-mib"], raw)
                    sample = cfg["memory_policy"]["sample"]
                    self.assertEqual(sample["vram_reserve_requested_mib"], requested)
                    self.assertEqual(sample["vram_reserve_mib"], effective)
                    self.assertEqual(sample["gpu_commit_gib"], 15.99 - effective / 1024)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    prep.report_memory(sample, cfg)
                self.assertIn(f"VRAM reserve request {raw!r} MiB", output.getvalue())
                self.assertIn(f"engine effective {effective} MiB", output.getvalue())

    def test_automatic_admission_uses_the_engine_integer_prefix_for_scientific_and_fractional_reserve(self):
        prep.setup.gpu_info.return_value["vram_gb"] = 16
        prep.detected_memory.return_value.update(ram_gib=32, available_gib=27, commit_available_gib=24.75)
        for raw in ("1e3", "1.75"):
            with self.subTest(raw=raw):
                cfg = self.make_config({"args": {"--vram-reserve-mib": raw}})
                sample = cfg["memory_policy"]["sample"]
                self.assertTrue(sample["automatic_host_cache"])
                self.assertEqual(sample["vram_reserve_mib"], 1)
                self.assertEqual(sample["expert_pool_gib"], 0.0751953125)
                self.assert_launch_rejected(cfg, "at least 0.25 GiB")

    def test_duplicate_saved_managed_flags_never_publish_or_start_server(self):
        prep.setup.gpu_info.return_value["vram_gb"] = 16
        prep.detected_memory.return_value.update(ram_gib=32, available_gib=27, commit_available_gib=24.75)
        for flag, duplicate in (("--expert-cache", "1"), ("--max-context", "8192"), ("--kv", "fp16"),
                                ("--kv-resident", "1024"), ("--spec", "3"), ("--suffix-draft", "1"),
                                ("--lookup-chain", "1"), ("--resident-budget-gib", "20"), ("--prefill", "64"),
                                ("--vram-reserve-mib", "1e3"), ("--no-prefill-borrow", None)):
            with self.subTest(flag=flag):
                cfg = self.make_config({"args": {"--vram-reserve-mib": "1024"}})
                self.assertEqual(cfg["memory_policy"]["sample"]["expert_pool_gib"], 1.07421875)
                if duplicate is None:
                    cfg["args"] += [flag, flag]
                else:
                    cfg["args"] += [flag, duplicate]
                self.assert_launch_rejected(cfg, f"Duplicate managed flag {flag}")

    def test_duplicate_generated_policy_flags_never_publish_or_start_server(self):
        cfg = self.make_config({"args": {"--vram-reserve-mib": "1024"}})
        cfg["memory_policy"]["generated_args"] += ["--vram-reserve-mib", "1e3"]
        self.assert_launch_rejected(cfg, "Duplicate managed flag --vram-reserve-mib")

    def test_manual_and_removed_cache_overrides_survive_repeated_refresh_and_launch(self):
        for group, name, override in (("args", "--resident-budget-gib", "100"),
                                      ("env", "STRATA_FILE_CACHE_MIB", "20000"),
                                      ("args", "--resident-budget-gib", None),
                                      ("env", "STRATA_FILE_CACHE_MIB", None)):
            with self.subTest(name=name, override=override):
                prep.detected_memory.return_value.update(available_gib=27, commit_available_gib=45)
                cfg = self.make_config()
                if group == "args":
                    prep.set_flag(cfg["args"], name, override)
                elif override is None:
                    cfg["env"].pop(name)
                else:
                    cfg["env"][name] = override
                for available in (0, None):
                    prep.detected_memory.return_value.update(available_gib=available, commit_available_gib=None)
                    cfg = prep.refreshed_config(cfg)
                    self.assertIn(name, cfg["memory_policy"]["overrides"][group])
                    self.assertEqual(cfg["memory_policy"]["overrides"][group][name], override)
                self.config.write_bytes(prep.json_bytes(cfg))
                with patch.object(prep, "check_ready"), patch.object(prep.subprocess, "run") as run, \
                        contextlib.redirect_stdout(io.StringIO()):
                    run.return_value.returncode = 0
                    self.assertEqual(prep.launch(self.config), 0)
                run.assert_called_once()
                saved = json.loads(self.config.read_text())
                actual = prep.flag_value(saved["args"], name) if group == "args" else saved["env"].get(name)
                self.assertEqual(actual, override)
                self.assertEqual(saved["memory_policy"]["overrides"][group][name], override)
                self.assertFalse(saved["memory_policy"]["sample"]["automatic_host_cache"])

    def test_noncache_overrides_do_not_bypass_automatic_admission(self):
        cfg = self.make_config({"args": {"--vram-reserve-mib": "0"},
                                "env": {"STRATA_RESIDENT_HEADROOM_GIB": "0"}})
        prep.detected_memory.return_value["available_gib"] = 0
        self.assert_launch_rejected(cfg, "at least 0.25 GiB")

    def test_preparation_and_read_only_plan_allow_unverified_telemetry(self):
        prep.detected_memory.return_value.update(available_gib=None, commit_available_gib=None)
        pack = self.root / "packs" / prep.TAG
        for name in ("native_experts.txt", "tokenizer/vocab.json"):
            path = pack / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        output = io.StringIO()
        with patch.object(prep, "verify_sources"), patch.object(prep, "check_ready"), \
                patch.object(prep.subprocess, "run") as run, contextlib.redirect_stdout(output):
            self.assertEqual(prep.main(["--prepare", "--config", str(self.config), "--gguf-dir", str(self.root)]), 0)
            original = self.config.read_bytes()
            self.assertEqual(prep.main(["--plan", "--config", str(self.config)]), 0)
        run.assert_not_called()
        self.assertEqual(self.config.read_bytes(), original)
        self.assertIn("Unverified plan", output.getvalue())
        self.assertIn("Windows commit telemetry unavailable", output.getvalue())

    def test_launch_does_not_overwrite_a_concurrent_user_edit(self):
        self.config.write_bytes(prep.json_bytes(self.make_config()))
        def edit(_):
            self.config.write_bytes(b"concurrent edit")
        with patch.object(prep, "check_ready", side_effect=edit), patch.object(prep.subprocess, "run") as run:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Configuration changed"):
                prep.launch(self.config)
        run.assert_not_called()
        self.assertEqual(self.config.read_bytes(), b"concurrent edit")

    def test_plan_does_not_print_api_key(self):
        cfg = self.make_config()
        self.config.write_bytes(prep.json_bytes(cfg))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(prep.main(["--plan", "--config", str(self.config)]), 0)
        self.assertNotIn(cfg["api_key"], output.getvalue())

    def test_install_refuses_binary_with_stale_receipt_before_touching_destination(self):
        built = self.root / "build"
        built.mkdir()
        (built / "strata.exe").write_bytes(b"new binary")
        (built / "BUILD-strata.json").write_text(json.dumps({"src": "same-source", "sha256": "wrong"}))
        target = self.root / "target.exe"
        target.write_bytes(b"working binary")
        with patch.object(prep, "check_source_identity"):
            with self.assertRaisesRegex(ValueError, "does not match the compiled executable"):
                prep.install_engine(built, target)
        self.assertEqual(target.read_bytes(), b"working binary")

    def test_cli_rejects_nonfinite_and_negative_overrides(self):
        for flags in (("--resident-budget-gib", "nan"), ("--file-cache-mib", "-1"),
                      ("--resident-headroom-gib", "inf")):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(prep.main(["--write-config", "--config", str(self.config), *flags]), 1)
        self.assertFalse(self.config.exists())

    def test_unauthenticated_lan_profile_is_rejected(self):
        cfg = self.make_config()
        cfg.update(host="0.0.0.0", api_key="")
        with self.assertRaisesRegex(ValueError, "LAN access requires"):
            prep.check_ready(cfg)

    def test_ready_check_rejects_changed_engine_even_when_other_files_exist(self):
        cfg = self.make_config()
        for path in (Path(cfg["exe"]), self.root / "pack/native_experts.txt",
                     self.root / "pack/tokenizer/vocab.json", self.root / "pack/tokenizer/chat_template.jinja",
                     self.root / "pack/source-verification.json"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        Path(cfg["build_metadata"]).write_text(json.dumps({"sha256": "wrong"}))
        with patch.object(prep, "missing_shards", return_value=[]):
            with self.assertRaisesRegex(ValueError, "Engine SHA-256 differs"):
                prep.check_ready(cfg)

    def test_build_uses_requested_directory_and_installs_matching_identity(self):
        build_dir = self.root / "existing-build"
        target = self.root / "engine/strata-unsloth-portable.exe"
        if prep.os.name != "nt":
            with self.assertRaisesRegex(ValueError, "supports Windows"):
                prep.build_engine(2, build_dir, target)
            return
        cmake_file = self.root / "third_party/llama.cpp/ggml/CMakeLists.txt"
        cmake_file.parent.mkdir(parents=True)
        cmake_file.write_text("fixture")
        def run(command, **kwargs):
            self.assertEqual(command[0], "cmake.exe")
            if "--build" in command:
                (build_dir / "strata.exe").write_bytes(b"compiled fixture")
            else:
                (build_dir / "CMakeCache.txt").write_text(
                    f"STRATA_GGML_DIR:PATH={cmake_file.parents[1]}\nCMAKE_CUDA_ARCHITECTURES:STRING=89\n"
                    "STRATA_ASYNC:BOOL=OFF\nGGML_CUDA:BOOL=ON\nCMAKE_CXX_FLAGS:STRING=/O2\n")
        with patch.object(prep.setup, "find_nvcc", return_value=("nvcc.exe", None)), \
                patch.object(prep.setup, "find_vcvars", return_value="vcvars64.bat"), \
                patch.object(prep.setup, "find_tool", side_effect=lambda name: name + ".exe"), \
                patch.object(prep.setup, "source_hash", return_value="current-source"), \
                patch.object(prep.setup, "source_version", return_value="test-version"), \
                patch.object(prep.subprocess, "check_output", return_value="test-commit\n"), \
                patch.object(prep, "msvc_environment", return_value={"PATH": "fixture"}), \
                patch.object(prep.subprocess, "run", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
            prep.build_engine(2, build_dir, target)
        metadata = json.loads(prep.engine_metadata(target).read_text())
        self.assertEqual(target.read_bytes(), b"compiled fixture")
        self.assertEqual(metadata["sha256"], hashlib.sha256(b"compiled fixture").hexdigest())
        self.assertEqual(metadata["src"], "current-source")
        self.assertEqual(metadata["commit"], "test-commit")
        self.assertEqual(metadata["ggml"]["files"], 1)
        self.assertEqual(metadata["cmake_settings"]["CMAKE_CXX_FLAGS"], "/O2")
        self.assertEqual(metadata["cmake_settings"]["GGML_CUDA"], "ON")
        self.assertEqual(metadata["commands"][1], ["cmake.exe", "--build", str(build_dir), "--target", "strata", "-j", "2"])

    def test_build_failure_cannot_install_or_write_success_receipt(self):
        build_dir = self.root / "failed-build"
        target = self.root / "target.exe"
        if prep.os.name != "nt":
            with self.assertRaisesRegex(ValueError, "supports Windows"):
                prep.build_engine(2, build_dir, target)
            return
        cmake_file = self.root / "third_party/llama.cpp/ggml/CMakeLists.txt"
        cmake_file.parent.mkdir(parents=True)
        cmake_file.write_text("fixture")
        with patch.object(prep.setup, "find_nvcc", return_value=("nvcc.exe", None)), \
                patch.object(prep.setup, "find_vcvars", return_value="vcvars64.bat"), \
                patch.object(prep.setup, "find_tool", return_value="tool.exe"), \
                patch.object(prep.setup, "source_hash", return_value="current-source"), \
                patch.object(prep, "msvc_environment", return_value={"PATH": "fixture"}), \
                patch.object(prep.subprocess, "run", side_effect=prep.subprocess.CalledProcessError(1, "build")):
            with self.assertRaises(prep.subprocess.CalledProcessError):
                prep.build_engine(2, build_dir, target)
        self.assertFalse(target.exists())
        self.assertFalse((build_dir / "BUILD-strata.json").exists())

    def test_ggml_change_during_build_prevents_install_and_success_receipt(self):
        build_dir = self.root / "changing-build"
        target = self.root / "target.exe"
        if prep.os.name != "nt":
            with self.assertRaisesRegex(ValueError, "supports Windows"):
                prep.build_engine(2, build_dir, target)
            return
        cmake_file = self.root / "third_party/llama.cpp/ggml/CMakeLists.txt"
        cmake_file.parent.mkdir(parents=True)
        cmake_file.write_text("before")
        def run(command, **kwargs):
            if "--build" in command:
                (build_dir / "strata.exe").write_bytes(b"compiled fixture")
                cmake_file.write_text("changed during build")
            else:
                (build_dir / "CMakeCache.txt").write_text(f"STRATA_GGML_DIR:PATH={cmake_file.parents[1]}\n")
        with patch.object(prep.setup, "find_nvcc", return_value=("nvcc.exe", None)), \
                patch.object(prep.setup, "find_vcvars", return_value="vcvars64.bat"), \
                patch.object(prep.setup, "find_tool", return_value="tool.exe"), \
                patch.object(prep.setup, "source_hash", return_value="same-engine-source"), \
                patch.object(prep, "msvc_environment", return_value={"PATH": "fixture"}), \
                patch.object(prep.subprocess, "run", side_effect=run):
            with self.assertRaisesRegex(ValueError, "Selected GGML inputs changed during the build"):
                prep.build_engine(2, build_dir, target)
        self.assertFalse(target.exists())
        self.assertFalse((build_dir / "BUILD-strata.json").exists())

    def test_explicit_fp16_and_q4_alias_survive_repeated_refresh(self):
        for kv, gib in (("fp16", 3.25), ("q4", 0.9140625)):
            with self.subTest(kv=kv):
                cfg = self.make_config()
                prep.set_flag(cfg["args"], "--kv", kv)
                first = prep.refreshed_config(cfg)
                second = prep.refreshed_config(first)
                self.assertEqual(value(second["args"], "--kv"), kv)
                self.assertEqual(second["memory_policy"]["sample"]["kv_ram_gib"], gib)

    def test_source_receipt_detects_modified_or_untracked_selected_ggml(self):
        llama = self.root / "third_party/llama.cpp"
        ggml = llama / "ggml"
        ggml.mkdir(parents=True)
        (ggml / "CMakeLists.txt").write_text("fixture")
        (ggml / "ggml.c").write_text("original")
        original = prep.ggml_identity(llama)
        metadata = {"src": "same-source", "ggml": original}
        with patch.object(prep.setup, "source_hash", return_value="same-source"):
            prep.check_source_identity(metadata)
            (ggml / "ggml.c").write_text("modified")
            with self.assertRaisesRegex(ValueError, "selected GGML sources"):
                prep.check_source_identity(metadata)
            (ggml / "ggml.c").write_text("original")
            (ggml / "untracked.c").write_text("new input")
            with self.assertRaisesRegex(ValueError, "selected GGML sources"):
                prep.check_source_identity(metadata)

    def test_ggml_revision_is_recorded_when_selected_checkout_has_git_metadata(self):
        llama = self.root / "selected-llama"
        (llama / "ggml").mkdir(parents=True)
        (llama / "ggml/CMakeLists.txt").write_text("fixture")
        (llama / ".git").write_text("gitdir: fixture")
        with patch.object(prep.subprocess, "check_output", return_value="actual-ggml-revision\n") as git:
            identity = prep.ggml_identity(llama)
        self.assertEqual(identity["revision"], "actual-ggml-revision")
        git.assert_called_once_with(["git", "-C", str(llama.resolve()), "rev-parse", "HEAD"], text=True)

    def test_new_policy_ready_checks_sources_but_static_config_keeps_old_binary(self):
        cfg = self.make_config()
        for path in (Path(cfg["exe"]), self.root / "pack/native_experts.txt",
                     self.root / "pack/tokenizer/vocab.json", self.root / "pack/tokenizer/chat_template.jinja",
                     self.root / "pack/source-verification.json"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        Path(cfg["build_metadata"]).write_text(json.dumps({"sha256": hashlib.sha256(b"fixture").hexdigest()}))
        with patch.object(prep, "missing_shards", return_value=[]), \
                patch.object(prep, "check_source_identity", side_effect=ValueError("stale sources")) as identity:
            with self.assertRaisesRegex(ValueError, "stale sources"):
                prep.check_ready(cfg)
            cfg.pop("memory_policy")
            prep.check_ready(cfg)
        identity.assert_called_once()

    def test_batch_arguments_are_always_quoted_and_percent_escaped(self):
        self.assertEqual(prep.batch_quote(r"C:\Models&Cache\model.json"), '"C:\\Models&Cache\\model.json"')
        self.assertEqual(prep.batch_quote("name%USER%^!().json"), '"name%%USER%%^!().json"')
        for invalid in ('bad"quote', 'bad\nline'):
            with self.assertRaises(ValueError):
                prep.batch_quote(invalid)

    def test_windows_launcher_runs_literal_metacharacter_paths(self):
        if prep.os.name != "nt":
            self.assertIn("%", prep.batch_quote("literal%"))
            return
        folder = self.root / "Models&Cache%STRATA_BATCH_VALUE%^!()"
        folder.mkdir()
        helper = folder / "helper.py"
        helper.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
        config = folder / "strata-safe&name%STRATA_BATCH_VALUE%^!().json"
        with patch.object(prep, "ROOT", folder), patch.object(prep, "__file__", str(helper)):
            script, content = prep.launcher_contents(config)
        script.write_bytes(content)
        env = dict(prep.os.environ, STRATA_TEST_SCRIPT=str(script), STRATA_BATCH_VALUE="UNWANTED_EXPANSION")
        # Expand the entrypoint last so the caller's own delayed expansion cannot eat its literal '!'.
        # The generated launcher must then disable the inherited mode for its literal arguments.
        command = 'cmd.exe /d /s /v:on /c ""!STRATA_TEST_SCRIPT!""'
        result = prep.subprocess.run(command, env=env, capture_output=True, text=True, timeout=10,
                                     stdin=prep.subprocess.DEVNULL)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.strip()), ["--launch", "--config", str(config)])

    def test_windows_msvc_environment_uses_literal_metacharacter_paths(self):
        if prep.os.name != "nt":
            self.assertIn("%%", prep.batch_quote("%PATH%"))
            return
        folder = self.root / "MSVC&Tools%STRATA_BATCH_VALUE%^!()"
        folder.mkdir()
        vcvars = folder / "vcvars64.bat"
        vcvars.write_text('@set "STRATA_TEST_COMPILER_ENV=verified"\n')
        # setup.find_vcvars returns Path in production; exercise that boundary with a real cmd child.
        for supplied in (vcvars, str(vcvars)):
            with self.subTest(path_type=type(supplied).__name__), \
                    patch.dict(prep.os.environ, {"STRATA_BATCH_VALUE": "UNWANTED_EXPANSION"}):
                env = prep.msvc_environment(supplied)
                self.assertEqual(env["STRATA_TEST_COMPILER_ENV"], "verified")


class SharedFiles(unittest.TestCase):
    def test_incomplete_download_is_rejected_before_hashing_or_packing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "shard.gguf"
            source.write_bytes(b"partial")
            pack = root / "pack"
            with patch.dict(prep.setup.UNSLOTH_Q2_K_XL_SHARDS,
                            {source.name: (100, "0" * 64)}, clear=True):
                with self.assertRaisesRegex(ValueError, "Download is not complete"):
                    prep.verify_sources(root, pack)
            self.assertEqual(source.read_bytes(), b"partial")
            self.assertFalse(pack.exists())

    def test_failed_hash_preserves_shared_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "shard.gguf"
            source.write_bytes(b"original")
            with patch.dict(prep.setup.UNSLOTH_Q2_K_XL_SHARDS,
                            {source.name: (8, "0" * 64)}, clear=True):
                with self.assertRaisesRegex(ValueError, "Verification failed"):
                    prep.verify_sources(root, root / "pack")
            self.assertEqual(source.read_bytes(), b"original")
            self.assertFalse((root / "pack").exists())

    def test_success_writes_receipt_only_in_the_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "shard.gguf"
            source.write_bytes(b"original")
            pack = root / "pack"
            with patch.dict(prep.setup.UNSLOTH_Q2_K_XL_SHARDS,
                            {source.name: (8, hashlib.sha256(b"original").hexdigest())}, clear=True):
                prep.verify_sources(root, pack)
                with patch.object(prep.hashlib, "sha256", side_effect=AssertionError("rehashed")):
                    prep.verify_sources(root, pack)
            self.assertTrue((pack / "source-verification.json").exists())
            self.assertFalse((root / "shard.gguf.done").exists())
            self.assertEqual(source.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
