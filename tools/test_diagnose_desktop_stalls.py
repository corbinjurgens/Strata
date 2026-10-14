"""Offline diagnostic orchestration checks; never starts a model or hardware query."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_desktop_stalls as diag


class Diagnostics(unittest.TestCase):
    def test_unavailable_sensors_are_explicit_and_do_not_prevent_ram_sample(self):
        with patch.object(diag.psutil, "virtual_memory", return_value=types.SimpleNamespace(available=123)), \
                patch.object(diag, "windows_commit", side_effect=OSError()), \
                patch.object(diag.psutil, "disk_io_counters", return_value={}), \
                patch.object(diag.shutil, "which", return_value=None):
            result = diag.collect_sample()
        self.assertEqual(result["available_ram_bytes"], 123)
        for key in ("commit", "disks", "gpu"):
            self.assertIsNone(result[key])
            self.assertIn(key, result["errors"])
        self.assertEqual(result["errors"]["disks"], "OSError")

    def test_plan_does_not_start_monitor_or_benchmark(self):
        with patch.object(diag, "check_profiles"), patch.object(diag, "Telemetry") as telemetry, \
                patch.object(diag.benchmark, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(diag.main(["--plan"]), 0)
            telemetry.assert_not_called()
            run.assert_not_called()

    def test_both_arms_use_timing_idle_gap_and_separate_outputs(self):
        with tempfile.TemporaryDirectory(dir=diag.ROOT / "build-unsloth") as folder:
            monitor = Mock()
            monitor.__enter__ = Mock(return_value=types.SimpleNamespace(error=None))
            monitor.__exit__ = Mock(return_value=False)
            result = {"status": "complete", "summary": {"tokens_per_second": 20}}
            with patch.object(diag, "ROOT", Path(folder)), patch.object(diag, "check_profiles"), \
                    patch.object(diag, "Telemetry", return_value=monitor), \
                    patch.object(diag.benchmark, "assert_idle"), \
                    patch.object(diag.benchmark, "run", return_value=result) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(diag.main([]), 0)
            self.assertEqual(run.call_count, 2)
            arms = [call.args[0] for call in run.call_args_list]
            self.assertEqual({arm.output.name for arm in arms}, {"desktop", "async"})
            self.assertTrue(all(arm.timing and arm.idle_seconds == 240 for arm in arms))
            self.assertEqual(monitor.__exit__.call_count, 2)
            report = next(Path(folder).glob("bench/results/*/README.md")).read_text()
            self.assertIn("async: complete", report)

    def test_failure_stops_comparison_and_closes_monitor(self):
        with tempfile.TemporaryDirectory(dir=diag.ROOT / "build-unsloth") as folder:
            monitor = Mock()
            monitor.__enter__ = Mock(return_value=types.SimpleNamespace(error=None))
            monitor.__exit__ = Mock(return_value=False)
            with patch.object(diag, "ROOT", Path(folder)), patch.object(diag, "check_profiles"), \
                    patch.object(diag, "Telemetry", return_value=monitor), \
                    patch.object(diag.benchmark, "assert_idle"), \
                    patch.object(diag.benchmark, "run", side_effect=RuntimeError("failed")) as run, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(diag.main([]), 1)
            self.assertEqual(run.call_count, 1)
            monitor.__exit__.assert_called_once()

    def test_telemetry_writes_parseable_sample_and_stops(self):
        with tempfile.TemporaryDirectory(dir=diag.ROOT / "build-unsloth") as folder:
            telemetry = diag.Telemetry(Path(folder) / "telemetry.jsonl")
            def sample():
                telemetry.done.set()
                return {"unix": 123, "available_ram_bytes": 456}
            with patch.object(diag, "collect_sample", side_effect=sample):
                telemetry.monitor()
            self.assertIsNone(telemetry.error)
            self.assertEqual(json.loads(telemetry.path.read_text())["available_ram_bytes"], 456)


if __name__ == "__main__":
    unittest.main()
