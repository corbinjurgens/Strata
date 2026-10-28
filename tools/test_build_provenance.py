"""Offline provenance tests, including real Git repositories and C++ links.

All Git writes are confined to newly created disposable fixture repositories.
Set STRATA_PROVENANCE_TEST_TEMP to an absolute external scratch directory and
provide CMake/Ninja/C++ on PATH to run the required native fixtures.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import setup
from tools import build_provenance as bp

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("STRATA_PROVENANCE_TEST_TEMP", tempfile.gettempdir())).resolve()


def fixture_payload(record=None):
    record = record or {"schema_version": 1, "commit": None, "dirty": None, "git_reason": "git_unavailable",
                        "version": "1.2.3", "backend": "cuda", "cuda": "13.4.1", "cuda_reason": None,
                        "built_at": "2026-01-01T00:00:00Z"}
    return b"fixture\0" + bp.MARKER + bp.json_bytes(record) + b"\n\0"


class PublicationTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="publication-", dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.built = self.root / "build/strata.exe"
        self.built.parent.mkdir()
        self.built.write_bytes(fixture_payload())
        self.receipt = self.built.parent / "BUILD.json"
        self.receipt.write_bytes(bp.json_bytes(bp.metadata_for(self.built.read_bytes(), self.built.name)))
        self.target = self.root / "installed/strata-custom.exe"
        self.sidecar = self.target.with_name("BUILD-custom.json")

    def test_copy_binds_actual_bytes_preserves_package_fields_and_release_manifest(self):
        self.target.parent.mkdir()
        release = self.target.parent / "BUILD.json"
        release.write_bytes(b"release manifest")
        extra = {"archs": [89], "cuda_dirs": ["relative/lib"], "lib_dirs": ["relative/hip"], "src": "legacy"}
        result = bp.copy_engine(self.built, self.target, self.sidecar, extra)
        self.assertEqual(self.target.read_bytes(), fixture_payload())
        self.assertEqual(bp.validate_metadata(self.target.read_bytes(), result, "strata-custom.exe"),
                         bp.embedded_record(fixture_payload()))
        self.assertEqual(result["archs"], [89])
        self.assertEqual(result["cuda_dirs"], ["relative/lib"])
        self.assertEqual(result["lib_dirs"], ["relative/hip"])
        self.assertEqual(result["src"], "legacy")
        self.assertEqual(result["version"], "1.2.3")
        self.assertEqual(release.read_bytes(), b"release manifest")

    def test_binary_mismatch_copied_name_and_forged_record_are_rejected(self):
        payload = self.built.read_bytes()
        metadata = json.loads(self.receipt.read_bytes())
        for data, receipt, name in ((payload + b"changed", metadata, self.built.name),
                                    (payload, metadata, "renamed.exe"),
                                    (payload, {**metadata, "build": {}}, self.built.name),
                                    (payload, {**metadata, "commit": "invented"}, self.built.name)):
            with self.subTest(name=name, receipt=receipt), self.assertRaises(ValueError):
                bp.validate_metadata(data, receipt, name)

    def test_copy_failure_restores_complete_previous_pair(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b"old binary")
        self.sidecar.write_bytes(bp.json_bytes({"old": "receipt"}))
        before = {p.name: (p.read_bytes(), p.stat().st_mode, p.stat().st_mtime_ns) for p in self.target.parent.iterdir()}
        replace = os.replace
        failed = False
        def fail_once(source, target):
            nonlocal failed
            if target == self.sidecar and not failed:
                failed = True
                raise OSError("injected paired-copy failure")
            return replace(source, target)
        with patch.object(bp.os, "replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "paired-copy failure"):
                bp.copy_engine(self.built, self.target, self.sidecar)
        self.assertTrue(failed)
        self.assertEqual({p.name: (p.read_bytes(), p.stat().st_mode, p.stat().st_mtime_ns)
                          for p in self.target.parent.iterdir()}, before)

    def test_mismatched_sidecar_cannot_touch_existing_destination(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b"previous")
        self.receipt.write_bytes(bp.json_bytes(bp.metadata_for(fixture_payload({**bp.embedded_record(fixture_payload()),
                                                                              "commit": "other"}), "strata.exe")))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            bp.copy_engine(self.built, self.target, self.sidecar)
        self.assertEqual(self.target.read_bytes(), b"previous")
        self.assertFalse(self.sidecar.exists())

    def test_custom_copy_cannot_replace_unbound_release_manifest(self):
        self.target.parent.mkdir()
        release = self.target.parent / "BUILD.json"
        release.write_bytes(b'{"version":"release"}')
        with self.assertRaisesRegex(ValueError, "another executable"):
            bp.copy_engine(self.built, self.target, release)
        self.assertFalse(self.target.exists())
        self.assertEqual(release.read_bytes(), b'{"version":"release"}')

    def test_missing_git_is_unknown_never_clean(self):
        with patch.object(bp.subprocess, "run", side_effect=FileNotFoundError):
            self.assertEqual(bp.git_state(self.root), {"commit": None, "dirty": None, "git_reason": "git_unavailable"})

    def test_version_scanner_keeps_literal_terminal_engine_token(self):
        executable = self.root / "standalone.exe"
        executable.write_bytes(fixture_payload() + b"INFO build_schema=1 engine=1.2.3\n\0")
        self.assertEqual(setup.engine_version(executable), (1, 2, 3))
        unrelated = bp.metadata_for(fixture_payload({**bp.embedded_record(fixture_payload()), "version": "9.9.9"}),
                                    "other.exe")
        (self.root / "BUILD.json").write_bytes(bp.json_bytes(unrelated))
        self.assertEqual(setup.engine_version(executable), (1, 2, 3))

    def test_failed_generated_header_publication_restores_record_and_header(self):
        template = self.root / "include/strata/build_info.hpp.in"
        template.parent.mkdir(parents=True)
        template.write_text('@STRATA_BUILD_JSON@\n')
        inputs = self.root / "inputs.json"
        inputs.write_bytes(bp.json_bytes({"version": "1.2.3", "backend": "cpu", "configuration": "Release",
                                         "architectures": "", "cxx": ""}))
        stage = self.root / "stage"
        bp.prepare(self.root, inputs, stage)
        paths = (stage / "record.json", stage / "strata/build_info.hpp")
        before = {p: p.read_bytes() for p in paths}
        template.write_text('changed @STRATA_BUILD_JSON@\n')
        replace = os.replace
        failed = False
        def fail_once(source, target):
            nonlocal failed
            if target == paths[1] and not failed:
                failed = True
                raise OSError("header stage failure")
            return replace(source, target)
        with patch.object(bp.os, "replace", side_effect=fail_once), self.assertRaisesRegex(OSError, "header stage failure"):
            bp.prepare(self.root, inputs, stage)
        self.assertTrue(failed)
        self.assertEqual({p: p.read_bytes() for p in paths}, before)

    def test_legacy_source_hash_retains_normalization_and_seed_contract(self):
        (self.root / "a.cpp").write_bytes(b"one\r\ntwo\r\n")
        # Independently calculated SHA-256(seed + relative name + NUL + normalized bytes).
        import hashlib
        expected = hashlib.sha256(b"seeda.cpp\0one\ntwo\n").hexdigest()[:16]
        self.assertEqual(bp.legacy_source_hash(self.root, ("a.cpp",), "seed"), expected)

    def test_setup_cuda_and_hip_copy_writers_retain_one_native_record(self):
        for backend in ("cuda", "hip"):
            with self.subTest(backend=backend):
                root = self.root / backend
                root.mkdir()
                eng = root / "engine"
                toolkit = root / "toolkit"
                (toolkit / "bin").mkdir(parents=True)
                (toolkit / "lib").mkdir()
                native_record = {**bp.embedded_record(fixture_payload()), "backend": backend,
                                 "cuda": "13.4.1" if backend == "cuda" else None}
                payload = fixture_payload(native_record)
                def build(src, bdir, *args):
                    bdir.mkdir(parents=True, exist_ok=True)
                    exe = bdir / setup.EXE
                    exe.write_bytes(payload)
                    (bdir / "BUILD.json").write_bytes(bp.json_bytes(bp.metadata_for(payload, exe.name)))
                with patch.object(setup, "ROOT", root), patch.object(setup, "engine_dir", return_value=eng), \
                        patch.object(setup, "source_hash", return_value="legacy-source"), \
                        patch.object(setup, "source_version", return_value="1.2.3"), \
                        patch.object(setup, "cpu_info", return_value=("fixture", True)), \
                        patch.object(setup, "cpu_floor", return_value=""), \
                        patch.object(setup, "install_build_tools", return_value=(toolkit / "bin/nvcc.exe", None)), \
                        patch.object(setup, "rocm_root", return_value=(toolkit, [str(toolkit / "lib")])), \
                        patch.object(setup.shutil, "which", return_value="fixture-compiler"), \
                        patch.object(setup, "cmake_build", side_effect=build), \
                        patch.dict(os.environ):
                    if backend == "cuda":
                        result = setup.build_engine({"arch": 89}, "none", False, root / "llama", toolkit=13)
                    else:
                        result = setup.build_engine_hip({"arch": "gfx1100"}, root / "llama")
                self.assertEqual(result, eng)
                metadata = json.loads((eng / "BUILD.json").read_bytes())
                self.assertEqual(bp.validate_metadata((eng / setup.EXE).read_bytes(), metadata, setup.EXE), native_record)
                self.assertEqual(metadata["src"], "legacy-source")
                self.assertEqual(metadata["vision"], "none")
                self.assertEqual(metadata["archs"], [89] if backend == "cuda" else ["gfx1100"])
                self.assertEqual(metadata["cuda_dirs"] if backend == "cuda" else metadata["lib_dirs"],
                                 [str(toolkit / "bin")] if backend == "cuda" else [str(toolkit / "lib")])


class NativeBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cmake = shutil.which("cmake") or setup.find_tool("cmake")
        cls.ninja = shutil.which("ninja") or setup.find_tool("ninja")
        if not cls.cmake or not cls.ninja or not shutil.which("git"):
            raise RuntimeError("Real provenance fixture requires installed CMake, Ninja, Git and C++ tools.")

    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="native-provenance-", dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.temp_root = Path(self.temp.name)
        self.root = self.temp_root / "source"
        self.root.mkdir()
        self.build = self.temp_root / "build"
        for name in ("cmake/build_provenance.cmake", "tools/build_provenance.py", "include/strata/build_info.hpp.in"):
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        (self.root / "src").mkdir()
        self.main = self.root / "src/main.cpp"
        self.main.write_text('#include "strata/build_info.hpp"\n#include <cstdio>\n'
                             'int main(){ std::fputs(strata::build::json, stdout); return 0; }\n')
        (self.root / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.24)\nproject(fixture VERSION 1.2.3 LANGUAGES CXX)\n'
            'set(CMAKE_CXX_STANDARD 17)\nadd_executable(strata src/main.cpp)\n'
            'set_target_properties(strata PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>")\n'
            'include(cmake/build_provenance.cmake)\nstrata_build_provenance(strata "${CMAKE_CURRENT_SOURCE_DIR}" cpu)\n')
        self.git("init", "--quiet")
        self.git("-c", "user.name=Synthetic Fixture", "-c", "user.email=fixture@example.invalid", "add", ".")
        self.commit("initial fixture")

    def command(self, args, *, success=True):
        result = subprocess.run([str(a) for a in args], cwd=self.root, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=180)
        if success and result.returncode:
            self.fail(f"Command {args!r} exit {result.returncode}:\n{result.stdout[-16000:]}\n{result.stderr[-4000:]}")
        return result

    def git(self, *args):
        # The sole Git-write seam: new temporary source repository only.
        self.assertTrue(self.root.is_relative_to(SCRATCH))
        return self.command(["git", "-C", self.root, *args])

    def commit(self, message, *args):
        self.git("-c", "user.name=Synthetic Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "--quiet", "-m", message, *args)

    def configure(self, *args, multi=False):
        return self.command([self.cmake, "-S", self.root, "-B", self.build, "-G",
                             "Ninja Multi-Config" if multi else "Ninja", f"-DCMAKE_MAKE_PROGRAM={self.ninja}",
                             f"-DPython3_EXECUTABLE={sys.executable}", "-DCMAKE_BUILD_TYPE=Release", *args])

    def compile(self, config="Release", *, success=True):
        return self.command([self.cmake, "--build", self.build, "--config", config, "--target", "strata"], success=success)

    def state(self, config="Release"):
        executable = self.build / config / ("strata.exe" if os.name == "nt" else "strata")
        receipt = executable.parent / "BUILD.json"
        header = self.build / "provenance" / config / "strata/build_info.hpp"
        record = self.build / "provenance" / config / "record.json"
        metadata = json.loads(receipt.read_bytes())
        probe = json.loads(self.command([executable]).stdout)
        self.assertEqual(probe, metadata["build"])
        self.assertEqual(probe, bp.validate_metadata(executable.read_bytes(), metadata, executable.name))
        files = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (executable, receipt, header, record)}
        return probe, files

    def test_real_git_head_dirty_to_dirty_clean_untracked_and_noop(self):
        self.configure()
        self.compile()
        clean, initial = self.state()
        self.assertFalse(clean["dirty"])
        self.assertIsNone(clean["cuda"])
        self.assertEqual(clean["cuda_reason"], "not_cuda")
        self.assertEqual(clean["commit"], self.git("rev-parse", "HEAD").stdout.strip())
        self.compile()
        self.assertEqual(self.state(), (clean, initial))
        (self.build / "Release/BUILD.json").unlink()
        self.compile()
        repaired, repaired_files = self.state()
        self.assertEqual(repaired, clean)
        self.assertEqual(repaired_files["BUILD.json"][0], initial["BUILD.json"][0])
        for name in initial.keys() - {"BUILD.json"}:
            self.assertEqual(repaired_files[name], initial[name])
        self.commit("HEAD-only change", "--allow-empty")
        self.compile()
        head, head_files = self.state()
        self.assertNotEqual(head["commit"], clean["commit"])
        self.assertNotEqual(head["built_at"], clean["built_at"])
        self.assertNotEqual(head_files["strata.exe" if os.name == "nt" else "strata"][0],
                            initial["strata.exe" if os.name == "nt" else "strata"][0])
        self.main.write_text(self.main.read_text() + "// dirty one\n")
        self.compile()
        dirty, _ = self.state()
        self.assertTrue(dirty["dirty"])
        self.main.write_text(self.main.read_text() + "// dirty two\n")
        self.compile()
        dirty2, _ = self.state()
        self.assertTrue(dirty2["dirty"])
        self.assertNotEqual(dirty2["source_sha256"], dirty["source_sha256"])
        self.assertNotEqual(dirty2["built_at"], dirty["built_at"])
        self.git("add", "src/main.cpp")
        self.commit("clean source transition")
        self.compile()
        clean2, _ = self.state()
        self.assertFalse(clean2["dirty"])
        self.assertNotEqual(clean2["commit"], dirty2["commit"])
        (self.root / "src/untracked.hpp").write_text("// new input\n")
        self.compile()
        untracked, _ = self.state()
        self.assertTrue(untracked["dirty"])
        self.assertNotEqual(untracked["source_sha256"], clean2["source_sha256"])
        (self.root / "src/untracked.hpp").unlink()
        (self.root / "private-profile.json").write_text('{"never_read":"private"}')
        self.compile()
        final, files = self.state()
        self.assertFalse(final["dirty"])
        self.assertEqual(final["source_sha256"], clean2["source_sha256"])
        self.assertNotIn("private", json.dumps(final))
        self.compile()
        self.assertEqual(self.state(), (final, files))

    def test_resolved_dependency_changes_are_bound_without_explicit_option(self):
        dependency = self.temp_root / "selected-dependency"
        source = dependency / "ggml/src/fixture.cpp"
        source.parent.mkdir(parents=True)
        source.write_text("// selected dependency one\n")
        self.configure("-DSTRATA_GGML_SOURCE_DIR=" + str(dependency))
        self.compile()
        before, _ = self.state()
        inputs = json.loads((self.build / "provenance/Release/inputs.json").read_bytes())
        self.assertEqual(Path(inputs["ggml"]), dependency)
        source.write_text("// selected dependency two\n")
        self.compile()
        after, files = self.state()
        self.assertNotEqual(before["source_sha256"], after["source_sha256"])
        self.assertNotEqual(before["built_at"], after["built_at"])
        self.compile()
        self.assertEqual(self.state(), (after, files))

    def test_named_target_repairs_mismatched_sidecar_and_refuses_unrelated_manifest(self):
        self.configure()
        self.compile()
        record, before = self.state()
        receipt = self.build / "Release/BUILD.json"
        bad = {**json.loads(receipt.read_bytes()), "sha256": "0" * 64}
        receipt.write_bytes(bp.json_bytes(bad))
        self.compile()
        repaired, files = self.state()
        self.assertEqual(repaired, record)
        self.assertEqual(files["BUILD.json"][0], before["BUILD.json"][0])
        for name in before.keys() - {"BUILD.json"}:
            self.assertEqual(files[name], before[name])
        unrelated = b'{"version":"unrelated-release","executable":"other.exe"}'
        receipt.write_bytes(unrelated)
        failed = self.compile(success=False)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("unrelated BUILD.json", failed.stdout + failed.stderr)
        self.assertEqual(receipt.read_bytes(), unrelated)


    def test_configuration_toolchain_and_multiconfig_isolation(self):
        # Compiler-wrapper bytes change while its compiler version remains fixed.
        compiler = shutil.which("cl" if os.name == "nt" else "c++")
        self.assertIsNotNone(compiler)
        wrapper = self.temp_root / ("compiler.cmd" if os.name == "nt" else "compiler")
        wrapper.write_text(('@echo off\n"' + compiler + '" %*\n') if os.name == "nt" else
                           ('#!/bin/sh\nexec "' + compiler + '" "$@"\n'))
        wrapper.chmod(0o755)
        self.configure(f"-DCMAKE_CXX_COMPILER={wrapper}", multi=True)
        self.compile("Release")
        release, release_files = self.state("Release")
        self.compile("Debug")
        debug, _ = self.state("Debug")
        self.assertEqual(self.state("Release"), (release, release_files))
        self.assertEqual(debug["configuration"], "Debug")
        self.assertNotEqual(debug["config_sha256"], release["config_sha256"])
        self.configure(f"-DCMAKE_CXX_COMPILER={wrapper}", "-DCMAKE_CXX_FLAGS=-DFIXTURE_CONFIGURATION=1", multi=True)
        self.compile()
        changed, _ = self.state()
        self.assertNotEqual(changed["config_sha256"], release["config_sha256"])
        wrapper.write_text(wrapper.read_text() + ("rem toolchain update\n" if os.name == "nt" else "# update\n"))
        self.compile()
        toolchain, files = self.state()
        self.assertNotEqual(toolchain["toolchain_sha256"], changed["toolchain_sha256"])
        self.assertNotEqual(toolchain["built_at"], changed["built_at"])
        self.compile()
        self.assertEqual(self.state(), (toolchain, files))

    def test_failed_compile_link_and_drift_refuse_new_receipt(self):
        self.configure()
        self.compile()
        _, before = self.state()
        original = self.main.read_text()
        self.main.write_text(original + "\n#error injected compile failure\n")
        failed = self.compile(success=False)
        self.assertNotEqual(failed.returncode, 0)
        receipt = self.build / "Release/BUILD.json"
        executable = receipt.parent / ("strata.exe" if os.name == "nt" else "strata")
        self.assertEqual(executable.read_bytes(), before[executable.name][0])
        self.assertEqual(receipt.read_bytes(), before["BUILD.json"][0])
        self.main.write_text(original.replace("return 0;", "extern int missing_symbol(); return missing_symbol();"))
        failed_link = self.compile(success=False)
        self.assertNotEqual(failed_link.returncode, 0)
        self.assertEqual(receipt.read_bytes(), before["BUILD.json"][0])
        if executable.exists():
            # A linker may retain the old binary or leave a partial output. Only
            # the original matching pair can still be admitted.
            try:
                bp.validate_metadata(executable.read_bytes(), json.loads(receipt.read_bytes()), executable.name)
            except ValueError:
                pass
            else:
                self.assertEqual(executable.read_bytes(), before[executable.name][0])
        self.main.write_text(original)
        self.compile()
        _, admitted = self.state()
        stage = self.build / "provenance/Release"
        executable = self.build / "Release" / ("strata.exe" if os.name == "nt" else "strata")
        self.main.write_text(original + "// drift after link\n")
        with self.assertRaisesRegex(ValueError, "changed during compilation"):
            bp.publish(self.root, stage / "inputs.json", stage, executable)
        self.assertEqual(receipt.read_bytes(), admitted["BUILD.json"][0])
        bp.prepare(self.root, stage / "inputs.json", stage)
        with self.assertRaisesRegex(ValueError, "does not match prepared"):
            bp.publish(self.root, stage / "inputs.json", stage, executable)
        self.assertEqual(receipt.read_bytes(), admitted["BUILD.json"][0])

    def test_real_prelink_input_drift_fails_build_and_cannot_publish_success(self):
        self.configure()
        self.compile()
        _, before = self.state()
        cmake_file = self.root / "CMakeLists.txt"
        drift = self.temp_root / "drift.cmake"
        drift.write_text('file(APPEND "' + self.main.as_posix() + '" "// input changed immediately before link\\n")\n')
        cmake_file.write_text(cmake_file.read_text() + '\nadd_custom_command(TARGET strata PRE_LINK COMMAND '
                              '"${CMAKE_COMMAND}" -P "' + drift.as_posix() + '")\n')
        self.configure()
        result = self.compile(success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Build inputs changed during compilation", result.stdout + result.stderr)
        receipt = self.build / "Release/BUILD.json"
        self.assertEqual(receipt.read_bytes(), before["BUILD.json"][0])
        executable = receipt.parent / ("strata.exe" if os.name == "nt" else "strata")
        if executable.exists():
            with self.assertRaises(ValueError):
                bp.validate_metadata(executable.read_bytes(), json.loads(receipt.read_bytes()), executable.name)

    def test_post_read_drift_and_unrelated_manifest_refuse_publication(self):
        self.configure()
        self.compile()
        _, before = self.state()
        stage = self.build / "provenance/Release"
        executable = self.build / "Release" / ("strata.exe" if os.name == "nt" else "strata")
        receipt = executable.parent / "BUILD.json"
        original_snapshot = bp.snapshot
        count = 0
        def drift(*args):
            nonlocal count
            count += 1
            if count == 2:
                self.main.write_text(self.main.read_text() + "// mid-publication drift\n")
            return original_snapshot(*args)
        with patch.object(bp, "snapshot", side_effect=drift), self.assertRaisesRegex(ValueError, "before publication"):
            bp.publish(self.root, stage / "inputs.json", stage, executable)
        self.assertEqual(receipt.read_bytes(), before["BUILD.json"][0])
        self.compile()
        receipt.write_bytes(b'{"version":"release","other_executable":"keep"}')
        with self.assertRaisesRegex(ValueError, "unrelated BUILD.json"):
            bp.publish(self.root, stage / "inputs.json", stage, executable)
        self.assertEqual(receipt.read_bytes(), b'{"version":"release","other_executable":"keep"}')


class HipPackagingTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="hip-package-", dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build, self.rocm, self.out = (self.root / n for n in ("build", "rocm", "out"))
        self.build.mkdir()
        self.record = {**bp.embedded_record(fixture_payload()), "backend": "hip", "cuda": None,
                       "cuda_reason": "not_cuda"}
        self.payload = fixture_payload(self.record)
        (self.build / "strata.exe").write_bytes(self.payload)
        (self.build / "strata-device.exe").write_bytes(b"fixture device helper")
        (self.build / "BUILD.json").write_bytes(bp.json_bytes(bp.metadata_for(self.payload, "strata.exe")))
        for name in ("bin/rocblas/library", "lib/llvm/bin", "include/hipblaslt", ".kpack", "crt"):
            (self.rocm / name).mkdir(parents=True)
        (self.rocm / "bin/amd_comgr.dll").write_bytes(b"fixture library")
        (self.rocm / ".kpack/blas_lib_gfx1100.kpack").write_bytes(b"fixture kernels")
        (self.rocm / "include/hipblaslt/hipblaslt-version.h").write_text(
            "#define HIPBLASLT_VERSION_MAJOR 1\n#define HIPBLASLT_VERSION_MINOR 2\n#define HIPBLASLT_VERSION_PATCH 3\n")

    def package(self):
        from tools.hip import package_windows as package
        from types import SimpleNamespace
        argv = ["package_windows.py", "--build", str(self.build), "--rocm", str(self.rocm),
                "--archs", "gfx1100", "--rocm-version", "fixture", "--out", str(self.out)]
        with patch.object(sys, "argv", argv), patch.object(package, "imports", return_value=[]), \
                patch.object(package, "crt_dir", return_value=self.rocm / "crt"), \
                patch.dict(sys.modules, {"dll_closure": SimpleNamespace(dll_closure=lambda *a: [])}):
            return package.main()

    def test_package_retains_actual_native_identity_and_package_fields(self):
        self.assertEqual(self.package(), 0)
        archive = self.out / "strata-windows-x64-hip.zip"
        with zipfile.ZipFile(archive) as package:
            payload = package.read("strata.exe")
            meta = json.loads(package.read("BUILD.json"))
        self.assertEqual(payload, self.payload)
        self.assertEqual(bp.validate_metadata(payload, meta, "strata.exe"), self.record)
        self.assertEqual(meta["version"], self.record["version"])
        self.assertEqual(meta["archs"], ["gfx1100"])
        self.assertEqual(meta["rocm"], "fixture")
        self.assertEqual(meta["hipblaslt_version"], 100203)
        self.assertEqual(meta["lib_dirs"], ["rocm/bin"])
        self.assertEqual(meta["platform"], "windows-x64")
        self.assertEqual(meta["source"], "prebuilt")

    def test_bad_native_receipt_refuses_before_touching_previous_stage(self):
        stage = self.out / "strata-windows-x64-hip"
        stage.mkdir(parents=True)
        marker = stage / "previous"
        marker.write_bytes(b"preserve previous stage")
        bad = {**json.loads((self.build / "BUILD.json").read_bytes()), "sha256": "0" * 64}
        (self.build / "BUILD.json").write_bytes(bp.json_bytes(bad))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.package()
        self.assertEqual(marker.read_bytes(), b"preserve previous stage")
        self.assertFalse((self.out / "strata-windows-x64-hip.zip").exists())


if __name__ == "__main__":
    unittest.main()
