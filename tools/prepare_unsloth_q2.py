"""Build, prepare or explicitly launch a hardware-aware UD-Q2_K_XL profile using existing GGUFs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup

MODEL = "UD-Q2_K_XL"
TAG = "unsloth-ud-q2_k_xl"
PROFILE = "unsloth-portable-128k"
POLICY_VERSION = 1
MIN_PINNED_GIB = 0.25
MAX_RECENT_MIB = (2**64 - 1) >> 20
C_SPACE = r"[ \t\n\r\f\v]*"
C_NUMBER = re.compile(C_SPACE + r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?" + C_SPACE)
C_INTEGER_PREFIX = re.compile(C_SPACE + r"([+-]?[0-9]+)")
C_RECENT_MIB = re.compile(C_SPACE + r"\+?[0-9]+")
MEMORY_FLAGS = ("--expert-cache", "--max-context", "--kv", "--kv-resident", "--spec",
                "--suffix-draft", "--lookup-chain", "--resident-budget-gib", "--prefill",
                "--vram-reserve-mib", "--no-prefill-borrow")
MEMORY_ENV = ("STRATA_RESIDENT_HEADROOM_GIB", "STRATA_FILE_CACHE_MIB",
              "STRATA_FILE_CACHE_COMPACT", "STRATA_OWNED_PRICE")
# Per token/layer: two KV heads, 256 values each, K and V. See kv_q8.hpp,
# kv_q4.hpp and layer.hpp; reserve the 12 main layers and the draft layer as setup does.
KV_CELL_BYTES = {"fp16": 2048, "int8": 1056, "q4_0": 576, "q4": 576, "k8v4": 816}


def memory_snapshot() -> dict:
    """Available commit is a separate limit; GPU commitment does not consume host RAM twice."""
    if os.name == "nt":
        m = setup._memory_status()
        return {"ram_gib": m.ullTotalPhys / 2**30, "available_gib": m.ullAvailPhys / 2**30,
                "commit_available_gib": m.ullAvailPageFile / 2**30, "windows_commit": True}
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        values[key] = int(rest.split()[0]) / 2**20
    # Linux overcommit accounting is not the Windows allocation limit. The engine also uses RAM here.
    return {"ram_gib": values["MemTotal"], "available_gib": values.get("MemAvailable"),
            "commit_available_gib": None, "windows_commit": False}


def detected_memory() -> dict:
    try:
        return memory_snapshot()
    except (OSError, ValueError, AttributeError):
        ram = setup.ram_gb()
        if not ram:
            raise ValueError("Could not detect system RAM; no configuration was written.")
        return {"ram_gib": ram, "available_gib": None, "commit_available_gib": None,
                "windows_commit": os.name == "nt"}


def detected_gpu(pick=None) -> dict:
    gpu = setup.gpu_info(pick)
    if not gpu:
        raise ValueError("No NVIDIA GPU detected; this preparation helper needs a CUDA-capable NVIDIA GPU.")
    return gpu


def memory_number(value, name: str, positive: bool = False) -> float:
    message = f"{name} must be finite and {'positive' if positive else 'nonnegative'}, using decimal/scientific syntax."
    if isinstance(value, bool) or not isinstance(value, (str, int, float)) or not C_NUMBER.fullmatch(str(value)):
        raise ValueError(message)
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ValueError(message) from None
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(message)
    return number


def effective_vram_reserve_mib(value) -> int:
    """Match the engine's atoi integer prefix without its undefined signed-int overflow."""
    memory_number(value, "--vram-reserve-mib")
    prefix = C_INTEGER_PREFIX.match(str(value))
    number = int(prefix[1]) if prefix else 0
    if not 0 <= number <= 2**31 - 1:
        raise ValueError("--vram-reserve-mib effective integer prefix must be between 0 and 2147483647 MiB.")
    return number


def memory_environment(env: dict) -> dict:
    """Canonical Windows spellings for policy-owned names, retaining raw values and types."""
    result = {}
    for key, value in env.items():
        name = key.upper() if os.name == "nt" and isinstance(key, str) else key
        name = name if name in MEMORY_ENV else key
        if name in MEMORY_ENV and name in result:
            if type(result[name]) is not type(value) or result[name] != value:
                raise ValueError(f"Conflicting managed environment aliases for {name}; use one value and type.")
        result[name] = value
    return result


def same_environment_value(first, second) -> bool:
    """Keep raw edits distinct even when numeric values compare equal."""
    return first == second and type(first) is type(second)


def validate_memory_overrides(overrides: dict) -> None:
    env = memory_environment(overrides.get("env", {}))
    for values, name, positive in (
            (overrides.get("args", {}), "--resident-budget-gib", True),
            (env, "STRATA_RESIDENT_HEADROOM_GIB", False)):
        value = values.get(name)
        if value is not None:  # None records an explicit removal.
            memory_number(value, name, positive)
    reserve = overrides.get("args", {}).get("--vram-reserve-mib")
    if reserve is not None:
        effective_vram_reserve_mib(reserve)
    recent = env.get("STRATA_FILE_CACHE_MIB")
    if recent is not None:
        message = f"STRATA_FILE_CACHE_MIB must be a nonnegative integer between 0 and {MAX_RECENT_MIB} MiB."
        if (isinstance(recent, bool) or not isinstance(recent, (str, int)) or
                (isinstance(recent, str) and not C_RECENT_MIB.fullmatch(recent))):
            raise ValueError(message)
        try:
            number = int(recent)
        except ValueError:
            raise ValueError(message) from None
        if not 0 <= number <= MAX_RECENT_MIB:
            raise ValueError(message)


def memory_plan(ram: float, vram: float, context: int, available_gib: float | None = None,
                commit_available_gib: float | None = None, windows_commit: bool = True,
                overrides: dict | None = None) -> dict:
    """Conservative startup allowances, not a prediction of fit or throughput."""
    if not (math.isfinite(ram) and ram > 0 and math.isfinite(vram) and vram > 0):
        raise ValueError("RAM and VRAM must be positive finite GiB values.")
    if context not in (8192, 32768, 65536, 131072):
        raise ValueError("Unsupported context; the helper never reduces the requested context.")
    small_gpu = vram < 8
    overrides = dict(overrides or {})
    if "env" in overrides:
        overrides["env"] = memory_environment(overrides["env"])
    validate_memory_overrides(overrides)
    arg_overrides, env_overrides = overrides.get("args", {}), overrides.get("env", {})
    kv = arg_overrides.get("--kv", "q4_0" if small_gpu and context >= 65536 else "int8") or "fp16"
    if kv not in KV_CELL_BYTES:
        raise ValueError(f"Unsupported KV format: {kv}")
    resident_cells = 20480 if small_gpu else 32768
    streamed = context >= 65536
    resident_value = arg_overrides.get("--kv-resident", resident_cells if streamed else 0)
    prefix = C_INTEGER_PREFIX.match(str(resident_value))
    effective_resident = int(prefix[1]) if prefix else 0  # The engine parses this flag with atoll.
    if not 0 <= effective_resident <= 2**63 - 1:
        raise ValueError("--kv-resident effective integer prefix must be between 0 and 9223372036854775807 cells.")
    # A positive setting can enable the MTP's host-backed ring even when the main
    # context fits in VRAM. Keep the nominal 12-main-plus-draft allowance; MTP uses
    # plain INT8 for a K8V4 draft (mtp.cpp), rather than the main layers' hybrid format.
    kv_cell_bytes = 12 * KV_CELL_BYTES[kv] + KV_CELL_BYTES["int8" if kv == "k8v4" else kv]
    kv_ram = context * kv_cell_bytes / 2**30 if effective_resident > 0 else 0
    headroom_value = env_overrides.get("STRATA_RESIDENT_HEADROOM_GIB", 8 if ram >= 48 else 4)
    headroom = float(4 if headroom_value is None else headroom_value)
    later = 0.5 if small_gpu else 1.0
    fixed = 2.0
    # At 128K on a small GPU, leave reserve selection to the engine's automatic path.
    # Its reserve can adapt, so charge full VRAM to Windows commit when no flag is given.
    default_reserve_mib = (None if context > 65536 else 448) if small_gpu else 1024
    reserve_value = arg_overrides.get("--vram-reserve-mib", default_reserve_mib)
    reserve_mib = effective_vram_reserve_mib(reserve_value) if reserve_value is not None else 0
    gpu_commit = max(0, vram - reserve_mib / 1024) if windows_commit else 0
    fallback = available_gib is None
    available = min(ram, available_gib if available_gib is not None else ram * 0.65)
    reserve = headroom + later + fixed + kv_ram
    physical_pool = max(0, available - reserve)
    commit_pool = (max(0, commit_available_gib - reserve - gpu_commit)
                   if commit_available_gib is not None else physical_pool)
    pool = min(physical_pool, commit_pool, 42.9)
    budget = min(30 if ram >= 48 else 12, math.floor(pool * 0.70))
    if "STRATA_FILE_CACHE_MIB" in env_overrides and "--resident-budget-gib" not in arg_overrides:
        budget = min(budget, math.floor(pool - int(env_overrides["STRATA_FILE_CACHE_MIB"] or 0) / 1024))
    # A positive budget selects the partial resident/file path. With no safe whole GiB left,
    # request a small clamped budget instead of accidentally selecting full residency.
    budget = max(MIN_PINNED_GIB, budget)
    if "--resident-budget-gib" in arg_overrides:
        budget = float(arg_overrides["--resident-budget-gib"] or 0)
    recent_mib = min(16384, math.floor(max(0, pool - budget) * 1024))
    if "STRATA_FILE_CACHE_MIB" in env_overrides:
        recent_mib = int(env_overrides["STRATA_FILE_CACHE_MIB"] or 0)
    args = ["--expert-cache", "auto", "--max-context", str(context), "--kv", kv,
            "--spec", "2", "--suffix-draft", "0", "--lookup-chain", "0",
            "--resident-budget-gib", f"{budget:g}"]
    if reserve_value is not None:
        args += ["--vram-reserve-mib", f"{reserve_mib:g}"]
    env = {"STRATA_RESIDENT_HEADROOM_GIB": f"{headroom:g}", "STRATA_FILE_CACHE_MIB": str(recent_mib),
           "STRATA_FILE_CACHE_COMPACT": "1"}
    if small_gpu:
        chunk = 64 if context < 65536 else 256
        if context > 65536:
            chunk = 64
        args += ["--prefill", str(chunk), "--no-prefill-borrow"]
        env["STRATA_OWNED_PRICE"] = "exact"
    else:
        args += ["--prefill", "auto"]
    if streamed:
        args += ["--kv-resident", str(resident_cells)]
    apply_overrides({"args": args, "env": env}, overrides)
    return {"args": args, "env": env, "ram_gib": ram, "vram_gib": vram,
            "available_gib": available_gib, "commit_available_gib": commit_available_gib,
            "windows_commit": windows_commit,
            "automatic_host_cache": "--resident-budget-gib" not in arg_overrides and
                                    "STRATA_FILE_CACHE_MIB" not in env_overrides,
            "available_fallback_gib": available if fallback else None, "kv_ram_gib": kv_ram,
            "cpu_server_gib": fixed, "later_startup_gib": later, "headroom_gib": headroom,
            "vram_reserve_requested_mib": float(reserve_value) if reserve_value is not None else 0,
            "vram_reserve_mib": reserve_mib,
            "gpu_commit_gib": gpu_commit, "expert_pool_gib": pool, "pinned_gib": budget,
            "recent_mib": recent_mib, "low_memory": pool < 2,
            "over_budget": budget + recent_mib / 1024 > pool}


def memory_settings(ram: float, vram: float, context: int, **kwargs) -> tuple[list[str], dict]:
    plan = memory_plan(ram, vram, context, **kwargs)
    return plan["args"], plan["env"]


def flag_value(args: list[str], flag: str):
    if flag in MEMORY_FLAGS and args.count(flag) > 1:
        raise ValueError(f"Duplicate managed flag {flag}; the saved memory policy is ambiguous.")
    if flag not in args:
        return None
    if flag == "--no-prefill-borrow":
        return True
    i = args.index(flag)
    if i + 1 >= len(args):
        raise ValueError(f"Missing value for {flag}.")
    if not isinstance(args[i + 1], str):
        raise ValueError(f"Invalid value for {flag}; expected a string.")
    if args[i + 1].startswith("--"):
        raise ValueError(f"Missing value for {flag}.")
    return args[i + 1]


def set_flag(args: list[str], flag: str, value) -> None:
    if flag in args:
        i = args.index(flag)
        del args[i:i + (1 if flag == "--no-prefill-borrow" else 2)]
    if value is not None and value is not False:
        args.extend([flag] if value is True else [flag, str(value)])


def report_memory(plan: dict, cfg: dict, sampled_now: bool = True) -> None:
    def amount(value):
        return "unavailable" if value is None else f"{value:.2f} GiB"
    print(f"{'Current' if sampled_now else 'Saved'} memory sample: RAM {plan['ram_gib']:.2f} GiB, available {amount(plan['available_gib'])}, "
          f"commit available {amount(plan['commit_available_gib'])}; GPU {plan['vram_gib']:.2f} GiB.")
    print(f"Policy reserves: host headroom {plan['headroom_gib']:g}, CPU/server {plan['cpu_server_gib']:g}, "
          f"later buffers {plan['later_startup_gib']:g}, streamed KV {plan['kv_ram_gib']:.2f} GiB; "
          f"GPU commitment {plan['gpu_commit_gib']:.2f} GiB only from commit.")
    print(f"Requests: pinned {flag_value(cfg['args'], '--resident-budget-gib')} GiB; compact recent "
          f"{cfg['env'].get('STRATA_FILE_CACHE_MIB', '0')} MiB; "
          f"context {flag_value(cfg['args'], '--max-context')}; KV {flag_value(cfg['args'], '--kv')}.")
    if plan.get("vram_reserve_requested_mib") != plan.get("vram_reserve_mib"):
        print(f"VRAM reserve request {flag_value(cfg['args'], '--vram-reserve-mib')!r} MiB; "
              f"engine effective {plan['vram_reserve_mib']} MiB (integer-prefix parsing).")
    print("Profile requests are startup ceilings subject to allocation checks. Recent retention is a target; "
          "protected entries and replacement storage may exceed it. The launcher resamples memory; explicit settings "
          "are preserved. Neither PC has passed full-model acceptance of the combined automatic policy.")
    if plan["available_fallback_gib"] is not None:
        print("Unverified plan: available-memory telemetry unavailable; using 65% of physical RAM as an estimate.")
    if plan.get("windows_commit", os.name == "nt") and plan["commit_available_gib"] is None:
        print("Unverified plan: Windows commit telemetry unavailable; the physical-memory fallback cannot establish fit.")
    if plan["low_memory"]:
        print("Little memory remains for experts; recent cache is reduced. Close this model and other apps before launch.")
    if plan["over_budget"]:
        print("Explicit or minimum pinned requests exceed the policy estimate; retained as requests, subject to engine clamps.")


def source_folder(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    hub = Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")))
    snapshots = hub / "models--unsloth--Qwen3.8-Flash-Next-GGUF" / "snapshots"
    folders = list(snapshots.glob(f"*/{MODEL}"))
    if not folders:
        raise ValueError("No existing Unsloth snapshot found. Supply --gguf-dir with the shard folder.")
    def rank(folder):
        complete = all((folder / name).is_file() and (folder / name).stat().st_size == size
                       for name, (size, _) in setup.UNSLOTH_Q2_K_XL_SHARDS.items())
        return complete, folder.stat().st_mtime
    return max(folders, key=rank).resolve()


def missing_shards(folder: Path) -> list[str]:
    return [f"{name}: need {size:,} bytes" for name, (size, _) in setup.UNSLOTH_Q2_K_XL_SHARDS.items()
            if not (folder / name).is_file() or (folder / name).stat().st_size != size]


def verify_sources(folder: Path, pack: Path) -> None:
    """Read shared HF files without deleting them or creating .done files beside them."""
    missing = missing_shards(folder)
    if missing:
        raise ValueError("Download is not complete; no packing was attempted.\n" + "\n".join(missing))
    receipt = pack / "source-verification.json"
    old = json.loads(receipt.read_text()) if receipt.is_file() else {}
    verified = {}
    for name, (size, expected) in setup.UNSLOTH_Q2_K_XL_SHARDS.items():
        path = folder / name
        before = path.stat()
        identity = {"path": str(path), "size": size, "mtime_ns": before.st_mtime_ns, "sha256": expected}
        if old.get(name) != identity:
            print(f"Checking SHA-256: {name}", flush=True)
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    h.update(block)
            after = path.stat()
            if h.hexdigest() != expected or (after.st_size, after.st_mtime_ns) != (size, before.st_mtime_ns):
                raise ValueError(f"Verification failed: {name}. Source files were left unchanged.")
        verified[name] = identity
    pack.mkdir(parents=True, exist_ok=True)
    setup.write_config(receipt, verified)


def atomic_files(files: dict[Path, bytes]) -> None:
    """Stage every write, then roll back already replaced files if publishing a pair fails."""
    staged, before, published = {}, {}, []
    try:
        for path, data in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            before[path] = path.read_bytes() if path.exists() else None
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
                staged[path] = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        for path, temporary in staged.items():
            os.replace(temporary, path)
            published.append(path)
    except BaseException:
        for path in reversed(published):
            if before[path] is None:
                path.unlink(missing_ok=True)
            else:
                # Restoring through a new file also avoids leaving a partially written executable.
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.rollback.",
                                                 delete=False) as handle:
                    handle.write(before[path])
                    restore = Path(handle.name)
                try:
                    os.replace(restore, path)
                finally:
                    restore.unlink(missing_ok=True)
        raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def engine_metadata(engine: Path) -> Path:
    return engine.with_name("BUILD-" + engine.stem.removeprefix("strata-") + ".json")


def batch_quote(value: str) -> str:
    """Quote one literal argument in a batch file with delayed expansion disabled."""
    if any(c in value for c in ('"', '\r', '\n', '\0')):
        raise ValueError("Batch arguments cannot contain quotes, newlines or NUL bytes.")
    value = value.replace("%", "%%")
    # Every argument is quoted, including paths without spaces but containing &, ^ or parentheses.
    # A trailing backslash must be doubled for the child process's argv parser.
    trailing = len(value) - len(value.rstrip("\\"))
    return '"' + value + "\\" * trailing + '"'


def msvc_environment(vcvars: str | Path) -> dict:
    """Run the vendor batch once, avoiding CALL's second expansion of literal path characters."""
    env = dict(os.environ, STRATA_VCVARS=os.fspath(vcvars))
    command = 'cmd.exe /d /u /s /v:off /c ""%STRATA_VCVARS%" >nul && set"'
    result = subprocess.run(command, env=env, cwd=ROOT, check=True, capture_output=True,
                            text=True, encoding="utf-16-le", errors="replace")
    prepared = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            prepared[key] = value
    if not any(key.upper() == "PATH" for key in prepared):
        raise ValueError("MSVC environment did not include PATH; no build was attempted.")
    return prepared


def ggml_identity(llama: Path) -> dict:
    """Hash the actual selected ggml inputs, including uncommitted and untracked changes."""
    llama = llama.resolve()
    ggml = llama / "ggml"
    if not (ggml / "CMakeLists.txt").is_file():
        raise ValueError("Selected GGML source is unavailable; rebuild with an existing llama.cpp checkout.")
    digest = hashlib.sha256()
    files = sorted(path for path in ggml.rglob("*") if path.is_file())
    for path in files:
        name = path.relative_to(ggml).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    revision = None
    if (llama / ".git").exists():
        revision = subprocess.check_output(["git", "-C", str(llama), "rev-parse", "HEAD"],
                                            text=True).strip()
    return {"root": str(llama), "revision": revision, "sha256": digest.hexdigest(), "files": len(files)}


def cmake_settings(bdir: Path) -> dict:
    selected = {}
    for line in (bdir / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        typed, value = line.split("=", 1)
        key = typed.split(":", 1)[0]
        if key.startswith(("STRATA_", "GGML_", "CMAKE_CUDA_", "CMAKE_CXX_", "CMAKE_C_COMPILER",
                           "CMAKE_C_FLAGS", "CMAKE_EXE_LINKER_FLAGS")) or key in (
                               "CMAKE_BUILD_TYPE", "CMAKE_GENERATOR", "CMAKE_MAKE_PROGRAM", "CMAKE_TOOLCHAIN_FILE"):
            selected[key] = value
    if not selected.get("STRATA_GGML_DIR"):
        raise ValueError("CMake did not record the selected GGML directory; no engine was installed.")
    return selected


def check_source_identity(metadata: dict) -> None:
    if metadata.get("src") != setup.source_hash(setup.ENGINE_SOURCES):
        raise ValueError("Build receipt does not match current engine sources; rebuild before installing or launching.")
    recorded = metadata.get("ggml")
    if (not isinstance(recorded, dict) or not recorded.get("root") or
            recorded != ggml_identity(Path(recorded["root"]))):
        raise ValueError("Build receipt does not match selected GGML sources; rebuild before installing or launching.")


def build_engine(jobs: int, bdir: Path, target: Path) -> None:
    """Use installed tools; build beside the original release, never download a release engine."""
    if os.name != "nt":
        raise ValueError("This build wrapper currently supports Windows. Use setup.sh --build on Linux.")
    nvcc, _ = setup.find_nvcc()
    vcvars, cmake, ninja = setup.find_vcvars(), setup.find_tool("cmake"), setup.find_tool("ninja")
    llama = ROOT / "third_party/llama.cpp"
    if not all((nvcc, vcvars, cmake, ninja)) or not (llama / "ggml/CMakeLists.txt").is_file():
        raise ValueError("Existing CUDA, MSVC, CMake, Ninja and llama.cpp build dependencies are required.")
    gpu = detected_gpu()
    bdir.mkdir(parents=True, exist_ok=True)
    source = setup.source_hash(setup.ENGINE_SOURCES)
    ggml = ggml_identity(llama)
    commands = [
        [cmake, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-S", str(ROOT), "-B", str(bdir),
         "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=ON",
         f"-DCMAKE_CUDA_ARCHITECTURES={gpu['arch']}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
         f"-DSTRATA_GGML_DIR={llama}"],
        [cmake, "--build", str(bdir), "--target", "strata", "-j", str(jobs)],
    ]
    env = msvc_environment(vcvars)
    for command in commands:
        subprocess.run(command, check=True, cwd=ROOT, env=env)
    settings = cmake_settings(bdir)
    selected_llama = Path(settings["STRATA_GGML_DIR"]).resolve()
    if selected_llama != llama.resolve() or ggml != ggml_identity(selected_llama):
        raise ValueError("Selected GGML inputs changed during the build; no executable was installed. Build again.")
    if source != setup.source_hash(setup.ENGINE_SOURCES):
        raise ValueError("Engine sources changed during the build; no executable was installed. Build again.")
    built = bdir / "strata.exe"
    receipt = {"source": "local", "version": setup.source_version(),
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "src": source, "ggml": ggml, "cmake_settings": settings,
               "archs": [int(gpu["arch"])], "commands": commands,
               "sha256": hashlib.sha256(built.read_bytes()).hexdigest()}
    atomic_files({bdir / "BUILD-strata.json": json_bytes(receipt)})
    install_engine(bdir, target)


def install_engine(bdir: Path, target: Path) -> None:
    built, receipt = bdir / "strata.exe", bdir / "BUILD-strata.json"
    if not built.is_file() or not receipt.is_file():
        raise ValueError("No custom build found; first use --build-engine.")
    metadata = json.loads(receipt.read_text(encoding="utf-8-sig"))
    payload = built.read_bytes()
    check_source_identity(metadata)
    if metadata.get("sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError("Build receipt does not match the compiled executable; rebuild before installing.")
    metadata["exe"] = str(target)
    atomic_files({target: payload, engine_metadata(target): json_bytes(metadata)})
    print(f"Custom engine installed as {target.name}; SHA-256 {metadata['sha256']}.")


def config_for(folder: Path, pack: Path, context: int, key_config: Path | None,
               engine: Path | None = None, overrides: dict | None = None) -> dict:
    overrides = dict(overrides or {})
    if "env" in overrides:
        overrides["env"] = memory_environment(overrides["env"])
    validate_memory_overrides(overrides)
    gpu, memory = detected_gpu(), dict(detected_memory())
    plan = memory_plan(vram=gpu["vram_gb"], context=context,
                       ram=memory.pop("ram_gib"), overrides=overrides, **memory)
    args, env = plan["args"], plan["env"]
    first = folder / next(iter(setup.UNSLOTH_Q2_K_XL_SHARDS))
    nvcc, _ = setup.find_nvcc()
    bindir = Path(nvcc).parent if nvcc else None
    lib_dirs = [str(d) for d in (bindir, bindir / "x64") if d and d.is_dir()] if bindir else []
    engine = engine or ROOT / "engine/strata-unsloth-portable.exe"
    cfg = {"exe": str(engine), "build_metadata": str(engine_metadata(engine)), "cwd": str(ROOT),
           "args": ["--pack", str(pack), "--native", str(first), "--expert-profile",
                    str(ROOT / "data/expert-profile.bin"), *args],
           "tokenizer": str(pack / "tokenizer"), "model_name": "qwen3.8-flash-next-unsloth-ud-q2_k_xl",
           "log": str(ROOT / f"strata-{PROFILE}.log"), "lib_dirs": lib_dirs, "env": env,
           "gpu": gpu["index"], "gpus_asked": True, "parallel": 1, "port": 8080, "host": "127.0.0.1",
           "fit_max_tokens": True, "open_browser": False, "api_key": secrets.token_urlsafe(32),
           "memory_policy": {"version": POLICY_VERSION, "generated_args": list(args),
                             "generated_env": dict(env), "overrides": overrides or {}, "sample": plan}}
    if key_config and key_config.is_file():
        previous = json.loads(key_config.read_text(encoding="utf-8-sig"))
        key = previous.get("api_key")
        if isinstance(key, str) and key.strip():
            cfg.update(api_key=key, host=previous.get("host", "127.0.0.1"), port=previous.get("port", 8080))
    apply_overrides(cfg, overrides or {})
    return cfg


def apply_overrides(cfg: dict, overrides: dict) -> None:
    for flag, value in overrides.get("args", {}).items():
        set_flag(cfg["args"], flag, value)
    for key, value in overrides.get("env", {}).items():
        if value is None:
            cfg["env"].pop(key, None)
        else:
            cfg["env"][key] = value


def memory_environment_removals(cfg: dict) -> set[str]:
    """Recognize managed removals in a saved template or its prepared runtime copy."""
    policy = cfg.get("memory_policy")
    if not isinstance(policy, dict) or policy.get("version") != POLICY_VERSION:
        return set()
    generated, overrides, env = policy.get("generated_env"), policy.get("overrides", {}), cfg.get("env", {})
    if not all(isinstance(values, dict) for values in (generated, overrides, env)):
        return set()
    explicit = overrides.get("env", {})
    if not isinstance(explicit, dict):
        return set()
    generated, explicit, env = (memory_environment(values) for values in (generated, explicit, env))
    removed = set()
    for key in MEMORY_ENV:
        changed = not same_environment_value(env.get(key), generated.get(key)) or (key in env and env[key] is None)
        if (changed and env.get(key) is None) or (not changed and key in explicit and explicit[key] is None):
            removed.add(key)
    return removed


def refreshed_config(cfg: dict) -> dict:
    """Refresh only policy-owned values that the user has not changed in the saved configuration."""
    cfg = copy.deepcopy(cfg)
    policy = cfg.get("memory_policy")
    if not policy:
        return cfg
    if policy.get("version") != POLICY_VERSION:
        raise ValueError("Unknown memory policy version; update the preparation helper before launching.")
    cfg["env"] = memory_environment(cfg.get("env", {}))
    policy["generated_env"] = memory_environment(policy["generated_env"])
    explicit = copy.deepcopy(policy.get("overrides", {}))
    explicit.setdefault("args", {})
    explicit["env"] = memory_environment(explicit.get("env", {}))
    memory_environment(os.environ)  # Refuse ambiguous inherited aliases before any resource sampler.
    context = flag_value(cfg["args"], "--max-context")
    if context is None or not re.fullmatch(r"[0-9]+", str(context).strip()) or int(context) <= 0:
        raise ValueError("Automatic memory policy requires an explicit positive --max-context capacity; "
                         "restore the flag or use a static profile.")
    selected = cfg.get("gpu")
    if isinstance(selected, str) and selected.strip().isdigit():
        selected = int(selected)
    if selected is not None and (isinstance(selected, bool) or not isinstance(selected, int) or selected < 0):
        raise ValueError("Automatic memory policy requires one GPU index; use a static profile for multiple GPUs.")
    env = cfg.setdefault("env", {})
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        names = [key for key in env if (key.upper() if os.name == "nt" else key) == name]
        values = [env[key] for key in names]
        if name == "CUDA_DEVICE_ORDER":
            if any(value != "PCI_BUS_ID" for value in values):
                raise ValueError("Automatic memory policy requires CUDA_DEVICE_ORDER=PCI_BUS_ID.")
        else:
            for value in values:
                if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value.strip()):
                    raise ValueError("Automatic memory policy requires CUDA_VISIBLE_DEVICES to name one GPU index.")
                index = int(value)
                if selected is not None and selected != index:
                    raise ValueError("Automatic memory policy GPU conflicts with CUDA_VISIBLE_DEVICES; "
                                     "make the selected GPU and explicit mask agree.")
                selected = index
        if names:
            # Windows environment names ignore case; use one spelling in the runtime copy.
            for key in names:
                del env[key]
            env[name] = values[0]
    cfg["gpu"] = selected
    validate_memory_overrides({"args": {flag: flag_value(cfg["args"], flag)
                                        for flag in ("--resident-budget-gib", "--vram-reserve-mib")},
                               "env": cfg.get("env", {})})
    for flag in MEMORY_FLAGS:
        if flag_value(cfg["args"], flag) != flag_value(policy["generated_args"], flag):
            explicit["args"][flag] = flag_value(cfg["args"], flag)
    removals = memory_environment_removals(cfg)
    for key in MEMORY_ENV:
        if key in removals:
            explicit["env"][key] = None
        elif not same_environment_value(env.get(key), policy["generated_env"].get(key)):
            explicit["env"][key] = cfg.get("env", {}).get(key)
    validate_memory_overrides(explicit)
    gpu, memory = detected_gpu(cfg.get("gpu")), dict(detected_memory())
    cfg["gpu"] = gpu["index"]
    plan = memory_plan(ram=memory.pop("ram_gib"), vram=gpu["vram_gb"],
                       context=int(context), overrides=explicit, **memory)
    for flag in MEMORY_FLAGS:
        if flag in explicit.get("args", {}):
            continue
        if flag_value(cfg["args"], flag) == flag_value(policy["generated_args"], flag):
            set_flag(cfg["args"], flag, flag_value(plan["args"], flag))
    env = cfg.setdefault("env", {})
    for key in MEMORY_ENV:
        if key in explicit.get("env", {}):
            if explicit["env"][key] is None:
                env.pop(key, None)
            else:
                env[key] = explicit["env"][key]
            continue
        if same_environment_value(env.get(key), policy["generated_env"].get(key)):
            if key in plan["env"]:
                env[key] = plan["env"][key]
            else:
                env.pop(key, None)
    policy["sample"] = plan
    policy["generated_args"] = list(plan["args"])
    policy["generated_env"] = dict(plan["env"])
    policy["overrides"] = explicit
    return cfg


def check_ready(cfg: dict) -> None:
    args = cfg["args"]
    pack = Path(args[args.index("--pack") + 1])
    source = Path(args[args.index("--native") + 1]).parent
    key = cfg.get("api_key")
    if cfg.get("host", "127.0.0.1") not in ("127.0.0.1", "localhost", "::1") and not (isinstance(key, str) and key.strip()):
        raise ValueError("LAN access requires a saved API key.")
    missing = missing_shards(source)
    required = [Path(cfg["exe"]), pack / "native_experts.txt", pack / "tokenizer/vocab.json",
                pack / "tokenizer/chat_template.jinja", pack / "source-verification.json"]
    missing += [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Not ready yet. Finish the download, then run prepare-unsloth-q2.bat.\n" + "\n".join(missing))
    if cfg.get("build_metadata"):
        metadata_path = Path(cfg["build_metadata"])
        if not metadata_path.is_file():
            raise ValueError("Missing engine build metadata; run this helper with --build-engine.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if metadata.get("sha256") != hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest():
            raise ValueError("Engine SHA-256 differs from its build metadata; rebuild before launching.")
        if cfg.get("memory_policy"):
            check_source_identity(metadata)


def launcher_contents(config: Path) -> tuple[Path, bytes]:
    tag = config.stem.removeprefix("strata-")
    command = [sys.executable, str(Path(__file__).resolve()), "--launch", "--config", str(config)]
    if os.name != "nt":
        return ROOT / f"run-{tag}.sh", ("#!/bin/sh\nexec " + shlex.join(command) + "\n").encode("utf-8")
    script = ROOT / f"run-{tag}.bat"
    text = ("@echo off\nsetlocal DisableDelayedExpansion\ntitle " + batch_quote("Strata " + tag) +
            '\ncd /d "%~dp0"\n' + " ".join(batch_quote(arg) for arg in command) +
            "\nif errorlevel 1 (\n  pause\n  exit /b 1\n)\n")
    return script, text.encode("utf-8")


def write_profile(config: Path, cfg: dict, create_config: bool) -> Path:
    script, content = launcher_contents(config)
    files = {config: json_bytes(cfg)} if create_config else {}
    files[script] = content
    atomic_files(files)
    if os.name != "nt":
        script.chmod(script.stat().st_mode | 0o111)
    return script


def admit_automatic_launch(plan: dict) -> None:
    if not plan["automatic_host_cache"]:
        return
    if plan["available_gib"] is None:
        raise ValueError("Automatic launch requires current available physical-memory telemetry; plan is unverified.")
    if plan["windows_commit"] and plan["commit_available_gib"] is None:
        raise ValueError("Automatic launch requires current Windows commit telemetry; plan is unverified.")
    if plan["expert_pool_gib"] < MIN_PINNED_GIB:
        raise ValueError(f"Automatic launch needs at least {MIN_PINNED_GIB:g} GiB in the expert pool; "
                         f"only {plan['expert_pool_gib']:g} GiB remains after policy allowances.")


def prepare_launch(config: Path, *, gpu=None) -> dict:
    """Prepare an automatic profile once, before the server creates its engine.

    A CLI GPU selection applies to this launch's sample and engine only; the
    persistent profile is never written during startup.
    """
    original = config.read_bytes()
    cfg = json.loads(original.decode("utf-8-sig"))
    if gpu is not None:
        cfg["gpu"] = gpu
    if not cfg.get("memory_policy"):
        return cfg
    cfg = refreshed_config(cfg)
    admit_automatic_launch(cfg["memory_policy"]["sample"])
    check_ready(cfg)
    report_memory(cfg["memory_policy"]["sample"], cfg)
    if config.read_bytes() != original:
        raise ValueError("Configuration changed while checking readiness; launch again to preserve those edits.")
    return cfg


def launch(config: Path) -> int:
    cfg = json.loads(config.read_text(encoding="utf-8-sig"))
    # Automatic preparation belongs to server startup, including direct invocations.
    # Static helper profiles retain their existing readiness check.
    if not cfg.get("memory_policy"):
        check_ready(cfg)
    # The server derives persistent web/chat settings from this path; retain the real local config.
    command = [sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata",
               "--config", str(config), "--port", str(cfg.get("port", 8080))]
    if cfg.get("open_browser") is True:
        command.append("--open")
    env = dict(os.environ)
    for name in ("STRATA_VERIFY_PROFILE", "STRATA_DECODE_TIMING", "STRATA_FETCH_PROFILE"):
        env.pop(name, None)  # Explicit config env values are still applied by the server.
    if isinstance(cfg.get("api_key"), str) and cfg["api_key"].strip():
        # server.py gives the inherited key precedence over config; keep this profile's saved credential effective.
        env = {key: value for key, value in env.items() if key.upper() != "STRATA_API_KEY"}
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    for name in ("plan", "write-config", "prepare", "check-ready", "build-engine", "install-engine", "launch"):
        mode.add_argument("--" + name, action="store_true")
    ap.add_argument("--gguf-dir", help="Existing shard folder; default: discover the Unsloth app's HF snapshot")
    ap.add_argument("--context", type=int, default=131072, choices=(8192, 32768, 65536, 131072),
                    help="New profile capacity; never reduced automatically (default: 131072)")
    ap.add_argument("--config", type=Path, default=ROOT / f"strata-{PROFILE}.json")
    ap.add_argument("--key-config", type=Path, default=ROOT / "strata-q2_0.json")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--build-dir", type=Path, default=ROOT / "build-unsloth-portable")
    ap.add_argument("--engine", type=Path, default=ROOT / "engine/strata-unsloth-portable.exe")
    ap.add_argument("--resident-budget-gib", type=float, help="Explicit new-profile pinned request; preserved at launch")
    ap.add_argument("--file-cache-mib", type=int, help="Explicit new-profile recent-cache request; preserved at launch")
    ap.add_argument("--resident-headroom-gib", type=float, help="Explicit new-profile engine headroom")
    a = ap.parse_args(argv)
    try:
        a.config, a.engine, a.build_dir = a.config.resolve(), a.engine.resolve(), a.build_dir.resolve()
        overrides = {"args": {}, "env": {}}
        if a.resident_budget_gib is not None:
            overrides["args"]["--resident-budget-gib"] = f"{a.resident_budget_gib:g}"
        if a.file_cache_mib is not None:
            overrides["env"]["STRATA_FILE_CACHE_MIB"] = str(a.file_cache_mib)
        if a.resident_headroom_gib is not None:
            overrides["env"]["STRATA_RESIDENT_HEADROOM_GIB"] = f"{a.resident_headroom_gib:g}"
        validate_memory_overrides(overrides)
        if a.build_engine:
            if a.jobs < 1:
                raise ValueError("--jobs must be positive.")
            build_engine(a.jobs, a.build_dir, a.engine)
            return 0
        if a.install_engine:
            install_engine(a.build_dir, a.engine)
            return 0
        if a.launch:
            return launch(a.config)
        if a.check_ready:
            check_ready(json.loads(a.config.read_text(encoding="utf-8-sig")))
            print("Ready for model tests. No model was started.")
            return 0
        existing = a.config.is_file()
        if existing:
            cfg = json.loads(a.config.read_text(encoding="utf-8-sig"))
            pack = Path(flag_value(cfg["args"], "--pack"))
            folder = Path(flag_value(cfg["args"], "--native")).parent
        else:
            folder = source_folder(a.gguf_dir)
            pack = ROOT / "packs" / TAG
            cfg = config_for(folder, pack, a.context, a.key_config, a.engine, overrides)
        if cfg.get("memory_policy"):
            preview = refreshed_config(cfg) if existing and a.plan else cfg
            report_memory(preview["memory_policy"]["sample"], preview, sampled_now=not existing or a.plan)
        if a.plan:
            safe = {"config": str(a.config), "existing_config_preserved": existing, "exe": cfg["exe"],
                    "context": flag_value(cfg["args"], "--max-context"), "api_key_enabled": bool(cfg.get("api_key")),
                    "host": cfg.get("host", "127.0.0.1"), "port": cfg.get("port", 8080),
                    "fit_max_tokens": cfg.get("fit_max_tokens"), "open_browser": cfg.get("open_browser"),
                    "pending_shards": missing_shards(folder)}
            print(json.dumps(safe, indent=2))
            return 0
        if a.prepare:
            verify_sources(folder, pack)
            if not (pack / "native_experts.txt").is_file() or not (pack / "tokenizer/vocab.json").is_file():
                env = dict(os.environ, STRATA_GGUF_PY=str(ROOT / "third_party/llama.cpp/gguf-py"))
                subprocess.run([sys.executable, str(ROOT / "tools/iq_pack.py"), "--gguf",
                                str(folder / next(iter(setup.UNSLOTH_Q2_K_XL_SHARDS))), "--out", str(pack),
                                "--compat-bf16"], env=env, cwd=ROOT, check=True)
        script = write_profile(a.config, cfg, create_config=not existing)
        if not existing:
            print(f"Wrote {a.config.name}; saved API key enabled: {bool(cfg.get('api_key'))}.")
        else:
            print("Existing config preserved. Configuration options apply only when creating a new config.")
        print(f"Launcher: {script.name}")
        if a.prepare:
            check_ready(cfg)
            print("Model files prepared. No model was started.")
        else:
            print("Configuration staged. Run this helper with --prepare --config PATH to verify the existing files.")
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
