"""Bounded, read-only checks for the base text server, without model inputs.

Success covers imports, executable identity/loading and driver/runtime queries.
It does not establish model fit, kernel execution or arbitrary dynamic-library
closure. Unknown prerequisites never count as a usable installation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from serve import launch_plan as lp
from tools import build_provenance

LIMIT = 128 << 10  # The shared query owner bounds each of stdout and stderr.
TIMEOUT = 5
IMPORTS = ("jinja2", "regex", "tools.strata_tokenizer", "serve.frontend")
ENVIRONMENT = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE",
               "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING",
               "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}


def prerequisite(present=None, usable=None, reason="not_checked"):
    state = "usable" if usable is True else "unavailable" if usable is False else "present" if present else "unverified"
    return {"present": present, "usable": usable, "state": state, "reason": reason}


IMPORT_PROBE = r'''
import contextlib, importlib, importlib.util, json, os, sys
sys.path.insert(0, sys.argv[1])
result = {}
with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
    for name in json.loads(sys.argv[2]):
        present, usable, reason = None, False, "import_failed"
        try:
            present = importlib.util.find_spec(name) is not None
            if present:
                importlib.import_module(name)
                usable, reason = True, "import_succeeded"
            else:
                reason = "module_missing"
        except Exception:
            pass
        result[name] = [present, usable, reason]
print(json.dumps(result))
'''

RUNTIME_PROBE = r'''
import contextlib, ctypes, json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
backend, major, folder = json.loads(sys.argv[2])
result = {}
handles = []
def load(names):
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    return None
def query(lib, name, pointer=False, required_major=None):
    function = getattr(lib, name)
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.POINTER(ctypes.c_int)] if pointer else [ctypes.c_uint]
    value = ctypes.c_int()
    code = function(ctypes.byref(value) if pointer else 0)
    return code == 0 and (not pointer or value.value > 0) and (required_major is None or value.value // 1000 == required_major)
with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
    try:
        search = os.environ.get("PATH" if os.name == "nt" else "LD_LIBRARY_PATH", "").split(os.pathsep)
        # Use the effective native environment, never rejected raw config paths.
        directories = ([folder] if os.name == "nt" else []) + [d for d in search if d]
        if os.name == "nt":
            for directory in directories:
                if os.path.isdir(directory):
                    handles.append(os.add_dll_directory(os.path.abspath(directory)))
        def library(names):
            return load([os.path.join(d, n) for d in directories for n in names] + names)
        if backend == "cuda":
            driver = library(["nvcuda.dll"] if os.name == "nt" else ["libcuda.so.1"])
            result["driver"] = [driver is not None, False, "driver_missing"]
            if driver:
                ok = query(driver, "cuInit") and query(driver, "cuDeviceGetCount", True)
                result["driver"] = [True, ok, "driver_device_query_succeeded" if ok else "driver_device_query_failed"]
            runtime = library(["cudart64_%d.dll" % major] if os.name == "nt" else ["libcudart.so.%d" % major])
            blas = library(["cublas64_%d.dll" % major] if os.name == "nt" else ["libcublas.so.%d" % major])
            result["libraries"] = [runtime is not None and blas is not None, False, "runtime_or_blas_unavailable"]
            if runtime and blas:
                ok = query(runtime, "cudaRuntimeGetVersion", True, major) and query(runtime, "cudaGetDeviceCount", True)
                result["libraries"] = [True, ok, "runtime_device_query_and_blas_load_succeeded" if ok else "runtime_device_query_failed"]
        else:
            runtime = library(["amdhip64_7.dll", "amdhip64_6.dll"] if os.name == "nt" else ["libamdhip64.so"])
            result["driver"] = [runtime is not None, False, "hip_runtime_unavailable"]
            result["libraries"] = [runtime is not None, False, "hip_runtime_unavailable"]
            if runtime:
                ok = query(runtime, "hipInit") and query(runtime, "hipGetDeviceCount", True)
                result["driver"] = [True, ok, "driver_device_query_succeeded" if ok else "driver_device_query_failed"]
                ok = query(runtime, "hipRuntimeGetVersion", True)
                blas = library(["hipblas.dll"] if os.name == "nt" else ["libhipblas.so"])
                result["libraries"] = [blas is not None, ok and blas is not None, "runtime_query_and_blas_load_succeeded" if ok and blas else "runtime_or_blas_unavailable"]
    except Exception:
        result = {}
    finally:
        for handle in handles:
            handle.close()
print(json.dumps(result))
'''

IMPORT_REASONS = {"import_succeeded", "import_failed", "module_missing"}
RUNTIME_REASONS = {"driver_missing", "driver_device_query_succeeded", "driver_device_query_failed",
                   "runtime_or_blas_unavailable", "runtime_device_query_and_blas_load_succeeded",
                   "runtime_device_query_failed", "hip_runtime_unavailable", "runtime_query_and_blas_load_succeeded"}


def _probe(argv, *, cwd, env, query):
    return lp.parse_json(query(argv, cwd=str(cwd), env=env, timeout=TIMEOUT, limit=LIMIT))


def _results(raw, names, reasons):
    if not isinstance(raw, dict) or set(raw) != set(names):
        raise ValueError()
    result = {}
    for name in names:
        row = raw[name]
        if (not isinstance(row, list) or len(row) != 3 or
                (row[0] is not None and type(row[0]) is not bool) or type(row[1]) is not bool or
                not isinstance(row[2], str) or row[2] not in reasons or
                (row[1] and row[0] is not True)):
            raise ValueError()
        result[name] = prerequisite(*row)
    return result


PROBE_ERRORS = (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError,
                subprocess.SubprocessError)


def check_install(cfg, *, query=None, inherited=None):
    """Return only allowlisted facts. No credential resolution or model inspection."""
    from serve.server import child_env
    query = lp.bounded_query if query is None else query
    if not isinstance(cfg, dict):
        raise ValueError()
    inherited = os.environ if inherited is None else inherited
    safe = {k: v for k, v in inherited.items() if k.upper() in ENVIRONMENT}
    # Reuse ordinary launch composition, without passing unrelated config or secrets.
    native = {k: cfg[k] for k in ("gpu", "backend", "lib_dirs", "gpus_asked", "hip_ordinal") if k in cfg}
    # Ordering is irrelevant to availability. Do not invoke the ordinary
    # multi-card speed query (or its unbounded/secret-bearing environment).
    native["gpu_order"] = "as_given"
    configured = cfg.get("env", {})
    if not isinstance(configured, dict) or any(not isinstance(k, str) for k in configured):
        raise ValueError()
    native["env"] = {k: v for k, v in configured.items() if k.upper() in ENVIRONMENT}
    dirs = native.get("lib_dirs") or []
    if not isinstance(dirs, list) or any(not isinstance(d, str) for d in dirs):
        raise ValueError()
    env = {k: v for k, v in child_env(native, inherited=safe, read_only=True).items() if k.upper() in ENVIRONMENT}
    root = Path(__file__).resolve().parents[1]
    checks = {}
    try:
        raw = _probe([sys.executable, "-B", "-c", IMPORT_PROBE, str(root), json.dumps(IMPORTS)],
                     cwd=root, env=env, query=query)
        checks["imports"] = _results(raw, IMPORTS, IMPORT_REASONS)
    except PROBE_ERRORS:
        checks["imports"] = {name: prerequisite(reason="bounded_import_probe_failed") for name in IMPORTS}
    backend = "unknown"
    checks["executable"] = prerequisite(reason="no_configured_executable")
    checks["build"] = prerequisite(reason="executable_unverified")
    checks["driver"] = prerequisite(reason="backend_unverified")
    checks["libraries"] = prerequisite(reason="backend_unverified")
    if cfg.get("exe") is not None:
        cwd = lp._path(cfg.get("cwd") or str(Path.cwd()), Path.cwd())
        value = cfg["exe"]
        if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
            raise ValueError()
        try:
            exe = lp._path(value, cwd)
        except (OSError, lp.PlanError):
            exe = None
        if exe is None or not exe.is_file():
            checks["executable"] = prerequisite(False, False, "executable_missing")
        else:
            checks["executable"] = prerequisite(True, reason="build_not_verified")
            try:
                binary = lp._executable_bytes(exe)
                receipt = lp._path(cfg.get("build_metadata") or str(exe.parent / "BUILD.json"), cwd)
                build = build_provenance.validate_metadata(binary, lp.parse_json(lp._read(receipt)), exe.name)
                raw = _probe([str(exe), "--build-info"], cwd=cwd, env=env, query=query)
                if raw != build or build["backend"] not in ("cuda", "hip", "sycl"):
                    raise ValueError()
                backend = build["backend"]
                if cfg.get("backend", "cuda") != backend:
                    raise ValueError()
                checks["executable"] = prerequisite(True, True, "build_info_exited_without_model")
                checks["build"] = prerequisite(True, True, "receipt_embedded_and_runtime_identity_match")
            except PROBE_ERRORS:
                checks["build"] = prerequisite(usable=False, reason="receipt_or_build_info_incompatible")
            if checks["build"]["usable"]:
                major = str(build.get("cuda", "")).split(".", 1)[0]
                if (sys.platform not in ("win32", "linux") or backend == "sycl" or
                        (backend == "cuda" and major not in ("12", "13"))):
                    checks["libraries"] = prerequisite(reason="no_supported_no_model_runtime_probe")
                else:
                    try:
                        data = [backend, int(major) if backend == "cuda" else 0, str(exe.parent)]
                        raw = _probe([sys.executable, "-B", "-c", RUNTIME_PROBE, str(root), json.dumps(data)],
                                     cwd=cwd, env=env, query=query)
                        checks.update(_results(raw, ("driver", "libraries"), RUNTIME_REASONS))
                    except PROBE_ERRORS:
                        for name in ("driver", "libraries"):
                            checks[name] = prerequisite(reason="bounded_runtime_probe_failed")
    checks["features"] = (prerequisite(reason="vision_not_checked") if cfg.get("vision") else
                          prerequisite(True, True, "base_text_server_only"))
    required = [*checks["imports"].values(), *(v for k, v in checks.items() if k != "imports")]
    usable = all(item["usable"] is True for item in required)
    return {"schema": 1, "kind": "installation_check", "backend": backend,
            "platform": "windows" if os.name == "nt" else "linux" if sys.platform == "linux" else "unsupported",
            "usable": usable, "checks": checks,
            "scope": "no_model_imports_executable_load_and_driver_runtime_queries",
            "limitations": ["model_fit_and_kernel_execution_not_checked", "arbitrary_dynamic_library_closure_not_checked"]}


def check_main(args, explicit):
    try:
        if explicit - {"check_install", "config", "engine"} or ("engine" in explicit and args.engine != "strata"):
            raise ValueError()
        cfg = lp.parse_json(lp._read(Path(args.config))) if args.config else {}
        result = check_install(cfg)
        print(lp.canonical(result).decode("ascii"))
        return 0 if result["usable"] else 1
    except lp.ProbeCleanupError:
        print('{"error":{"code":"installation_error","message":"Installation probe cleanup could not be confirmed.","cleanup":"unconfirmed"}}')
        return 2
    except PROBE_ERRORS:
        print('{"error":{"code":"installation_error","message":"Invalid or conflicting installation check inputs."}}')
        return 2
