"""Read-only public launch plans. Native metadata owns all allocation arithmetic.

The digest identifies inputs/settings, not PAIR approval. Large model payloads and
libraries still require PAIR's external content approval. This module never reads
credentials and never starts a serving engine. Strict replay belongs to its caller.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from tools import build_provenance
from tools.gguf_reader import GGUFFile
from tools import prepare_unsloth_q2 as policy

SCHEMA = 1
MAX_JSON = 8 << 20
MAX_PAYLOAD = 1 << 20
MAX_METADATA = 64 << 20
MAX_EXECUTABLE = 512 << 20
PREFIXES = ("STRATA_", "CUDA_", "GGML_", "CUBLAS_", "OMP_")
SEARCH_ENV = ("PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "SYSTEMROOT")
NATIVE_ENV = {
    "STRATA_RESIDENT_HEADROOM_GIB", "STRATA_FILE_CACHE_MIB", "STRATA_FILE_CACHE_COMPACT",
    "STRATA_OWNED_PRICE", "STRATA_PREFILL_RING", "STRATA_PREFILL_MMQ",
    "STRATA_PREFILL_RESERVE_BO", "STRATA_STAGER_THREADS", "STRATA_STAGER_RING",
    "STRATA_STAGER_BATCH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING",
}
CREDENTIAL_ENV = ("STRATA_API_KEY", "STRATA_API_KEY_FILE")
AUTO_PREFILL = {"auto", "auto:8192", "auto:16384", "auto:32768"}
SCALARS = (
    "context prefill spec kv kv_resident batch batch_groups parallel auto_cache per_layer "
    "no_prefill_borrow resident_budget resident_bytes resident_experts resident_pin file_cache "
    "host_headroom vram_reserve shared_stage ring workspace weight_scratch common region "
    "prefill_mmq prefill_stage_own prefill_reserve_bo prefill_ring_enabled experts top_k vocab "
    "max_blob cache_request environment file_cache_effective file_cache_compact vram_required "
    "ram_required commit_required stager_threads stager_ring stager_batch"
).split()
VECTORS = ("layer_blobs", "kv_stage_runs", "resident_order", "slots")
CAPABILITIES = ("schema", "native_schema", "kind", "strict_consumer", "ownership_telemetry",
                "ownership_protocol", "ownership_frame_limit", "ownership_shard_limit", "effective_protocol")
PLAN_KEYS = {"schema", "kind", "plan_digest", "launch", "server", "inputs", "hardware", "requested",
             "resolved", "native", "estimates", "admission", "approval"}
SERVER_KEYS = {
    "model_name", "tokenizer", "host", "port", "aliases", "sampling", "fit_max_tokens", "lazy_load",
    "idle_unload_s", "engine_silence_s", "api_monitor", "open_browser", "min_free_vram_mib",
    "allowed_hosts", "cors_origins", "trusted_origins", "anthropic_thinking", "effort_position",
    "tool_call_recovery", "reasoning_close_retry", "codex_thread_titles", "codex_compaction_cache",
    "launch_memory_metrics", "slot_save_path", "max_queued_requests",
}
CONFIG_KEYS = SERVER_KEYS | {"exe", "args", "cwd", "env", "lib_dirs", "backend", "gpu", "gpus_asked",
                             "parallel", "memory_policy", "build_metadata", "log", "api_key", "api_key_file"}
VALUE_FLAGS = {"--pack", "--native", "--expert-profile", "--kv", "--expert-cache", "--max-context",
               "--prefill", "--spec", "--kv-resident", "--resident-budget-gib", "--vram-reserve-mib",
               "--suffix-draft", "--lookup-chain"}
SWITCHES = {"--no-prefill-borrow", "--expert-cache-per-layer"}


class PlanError(ValueError):
    """Bounded diagnostics must not contain untrusted values or native stderr."""


class ProbeCleanupError(RuntimeError):
    """A failed query whose exact process/readers still require owned cleanup."""

    def __init__(self, process, readers):
        super().__init__("Native query cleanup could not be confirmed.")
        self.process, self.readers = process, tuple(readers)


def canonical(value) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError):
        raise PlanError("invalid_launch_plan: invalid JSON data") from None


def digest(plan: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in plan.items() if k != "plan_digest"})).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PlanError("invalid_launch_plan: duplicate JSON field")
        result[key] = value
    return result


def parse_json(data: bytes):
    if len(data) > MAX_JSON:
        raise PlanError("invalid_launch_plan: JSON exceeds limit")
    try:
        result = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_pairs,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        canonical(result)
        return result
    except (ValueError, UnicodeError, RecursionError):
        raise PlanError("invalid_launch_plan: malformed JSON") from None


def _read(path: Path, limit=MAX_JSON) -> bytes:
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise PlanError("invalid_launch_plan: input exceeds limit")
    return data


def _executable_bytes(path):
    if Path(path).stat().st_size > MAX_EXECUTABLE:
        raise PlanError("invalid_launch_plan: executable exceeds inspection bound")
    return _read(Path(path), MAX_EXECUTABLE)


def _uint(value, *, positive=False):
    if type(value) is not int or not (int(positive) <= value <= 2**64 - 1):
        raise PlanError("invalid_launch_plan: invalid numeric field")
    return value


def _keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise PlanError("invalid_launch_plan: unknown or missing fields")


def server_settings(cfg, cwd):
    """Freeze effective server fields through their ordinary pure validation owners."""
    from serve.server import (allowed_hosts_of, engine_silence_s, model_aliases, normalized_slot_save_dir,
                              origins_of, sampling_defaults_from_config, max_queued_requests_of)
    out = {k: copy.deepcopy(v) for k, v in cfg.items() if k in SERVER_KEYS}
    try:
        if "max_queued_requests" in out:
            out["max_queued_requests"] = max_queued_requests_of(out)
        if "engine_silence_s" in out:
            out["engine_silence_s"] = engine_silence_s(out)
        if "sampling" in out:
            out["sampling"] = sampling_defaults_from_config(out, warn=False)
        if "aliases" in out:
            out["aliases"] = model_aliases(out["aliases"], out.get("model_name", "qwen3.8-flash-next"))
        if "allowed_hosts" in out:
            out["allowed_hosts"] = allowed_hosts_of(out["allowed_hosts"])
        for key, wildcard in (("cors_origins", True), ("trusted_origins", False)):
            if key in out:
                out[key] = origins_of(out[key], key, wildcard)
        for key in ("fit_max_tokens", "lazy_load", "api_monitor", "open_browser", "tool_call_recovery",
                    "reasoning_close_retry", "codex_thread_titles", "codex_compaction_cache", "launch_memory_metrics"):
            if key in out and type(out[key]) is not bool:
                raise ValueError()
        for key in ("model_name", "host", "tokenizer"):
            if key in out and (not isinstance(out[key], str) or not out[key] or any(ord(c) < 32 for c in out[key])):
                raise ValueError()
        if "port" in out and (type(out["port"]) is not int or not 0 < out["port"] < 65536):
            raise ValueError()
        if "idle_unload_s" in out:
            value = out["idle_unload_s"]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError()
        if "min_free_vram_mib" in out and (type(out["min_free_vram_mib"]) is not int or out["min_free_vram_mib"] < 0):
            raise ValueError()
        if "anthropic_thinking" in out:
            out["anthropic_thinking"] = str(out["anthropic_thinking"] or "model")
            if out["anthropic_thinking"] not in ("model", "on_request"):
                raise ValueError()
        if out.get("effort_position", "start") != "start":
            raise ValueError()
        if out.get("slot_save_path") is not None and out["slot_save_path"] != "":
            out["slot_save_path"] = normalized_slot_save_dir(out["slot_save_path"], cwd)
        if "tokenizer" in out:
            # Both CLI and config tokenizer paths are relative to the server cwd.
            out["tokenizer"] = os.path.abspath(out["tokenizer"])
    except (ValueError, TypeError, AttributeError, OverflowError, SystemExit):
        raise PlanError("invalid_launch_plan: invalid server settings") from None
    return out


def _path(value, cwd: Path) -> Path:
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise PlanError("invalid_launch_plan: invalid path")
    path = Path(value)
    # Preserve symlink spellings: native split-family discovery uses the filename.
    path = Path(os.path.abspath(path if path.is_absolute() else cwd / path))
    if not path.exists():
        raise PlanError("invalid_launch_plan: required input path unavailable")
    return path


def bounded_query(argv, *, cwd, env, timeout=20, limit=MAX_JSON) -> bytes:
    """Drain bounded pipes, terminate and confirm exit on timeout or excess output.

    Metadata helpers are synchronous leaf queries, never --serve. The existing
    containment owner also ties them to our lifetime on Windows.
    """
    from serve.server import popen
    from serve.winjob import contain
    cleanup_deadline = time.monotonic() + timeout + 5
    output = bytearray()
    errors = bytearray()
    failed = threading.Event()
    readers = []

    def drain(pipe, retain):
        seen = 0
        try:
            while True:
                block = pipe.read(4096)
                if not block:
                    break
                seen += len(block)
                if seen > limit:
                    failed.set()
                    proc.kill()
                    break
                if retain:
                    output.extend(block)
                elif len(errors) < 4096:
                    errors.extend(block[:4096 - len(errors)])
        except (OSError, ValueError):
            failed.set()
        finally:
            try:
                pipe.close()
            except OSError:
                failed.set()

    proc = popen("native metadata query", argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        contain(proc)
        for pipe, retain in ((proc.stdout, True), (proc.stderr, False)):
            reader = threading.Thread(target=drain, args=(pipe, retain), daemon=True)
            readers.append(reader)
            reader.start()
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            failed.set()
            raise PlanError("invalid_launch_plan: bounded metadata query timed out") from None
        for reader in readers:
            reader.join(timeout=min(5, max(0, cleanup_deadline - time.monotonic())))
        if any(reader.is_alive() for reader in readers):
            raise PlanError("invalid_launch_plan: metadata pipe did not close")
        if code and not failed.is_set() and (errors.startswith(b"invalid_launch_plan: insufficient ") or
                errors.startswith(b"invalid_launch_plan: explicit expert cache exceeds resource admission")):
            raise PlanError("insufficient_resources: native metadata admission refused")
        if failed.is_set() or code:
            raise PlanError("invalid_launch_plan: bounded metadata query refused or exceeded limit")
        if b"READY" in output:
            raise PlanError("invalid_launch_plan: metadata query entered startup")
        return bytes(output)
    except RuntimeError:
        raise PlanError("invalid_launch_plan: native query setup refused") from None
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=max(0, cleanup_deadline - time.monotonic()))
        except (OSError, subprocess.TimeoutExpired):
            pass
        complete = True
        for index, pipe in enumerate((proc.stdout, proc.stderr)):
            reader = readers[index] if index < len(readers) else None
            if reader is not None and reader.ident is not None:
                reader.join(timeout=max(0, cleanup_deadline - time.monotonic()))
            if reader is not None and reader.is_alive():
                # The reader closes its own pipe when it unblocks. Closing a
                # buffered pipe here could wait indefinitely for its read lock.
                complete = False
            else:
                try:
                    pipe.close()
                except OSError:
                    complete = False
        try:
            complete = complete and proc.poll() is not None
        except OSError:
            complete = False
        if not complete:
            raise ProbeCleanupError(proc, readers) from None


def observe_hardware(gpu=None) -> dict:
    """One selected GPU query plus one host snapshot, reused by both resolvers."""
    data = bounded_query(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free",
                          "--format=csv,noheader,nounits"], cwd=str(Path.cwd()),
                         env={k: v for k, v in os.environ.items() if k.upper() not in CREDENTIAL_ENV}, limit=65536)
    rows = []
    for line in data.decode("utf-8").splitlines():
        index, uuid, name, total, free = [part.strip() for part in line.split(",")]
        rows.append({"gpu_index": int(index), "gpu_identity": uuid, "gpu_name": name,
                     "vram_total_bytes": int(total) << 20, "vram_free_bytes": int(free) << 20})
    selected = next((r for r in rows if gpu is not None and r["gpu_index"] == gpu), None)
    if gpu is None and rows:
        selected = max(rows, key=lambda r: (r["vram_total_bytes"], -r["gpu_index"]))
    if selected is None:
        raise PlanError("invalid_launch_plan: selected CUDA GPU unavailable")
    memory = policy.memory_snapshot()
    if memory["available_gib"] is None or (memory["windows_commit"] and memory["commit_available_gib"] is None):
        raise PlanError("invalid_launch_plan: resource telemetry unavailable")
    return {**selected, "ram_total_bytes": int(memory["ram_gib"] * 2**30),
            "ram_available_bytes": int(memory["available_gib"] * 2**30),
            "commit_available_bytes": int((memory["commit_available_gib"] if memory["windows_commit"]
                                            else memory["available_gib"]) * 2**30),
            "windows_commit": memory["windows_commit"], "devices": 1,
            "commit_basis": "available_commit" if memory["windows_commit"] else "available_ram"}


def validate_hardware(h):
    _keys(h, {"gpu_index", "gpu_identity", "gpu_name", "vram_total_bytes", "vram_free_bytes",
              "ram_total_bytes", "ram_available_bytes", "commit_available_bytes", "windows_commit",
              "devices", "commit_basis"})
    for key in ("gpu_index", "vram_total_bytes", "vram_free_bytes", "ram_total_bytes",
                "ram_available_bytes", "commit_available_bytes", "devices"):
        _uint(h[key], positive=key in ("vram_total_bytes", "ram_total_bytes", "devices"))
    if (h["devices"] != 1 or type(h["windows_commit"]) is not bool or
            h["vram_free_bytes"] > h["vram_total_bytes"] or h["ram_available_bytes"] > h["ram_total_bytes"] or
            h["commit_basis"] != ("available_commit" if h["windows_commit"] else "available_ram") or
            any(not isinstance(h[k], str) or not h[k] for k in ("gpu_identity", "gpu_name"))):
        raise PlanError("invalid_launch_plan: inconsistent resource snapshot")


def _relevant(name):
    return name.startswith(PREFIXES) or name in SEARCH_ENV


def _library_search_name(name):
    return name in SEARCH_ENV or re.fullmatch(r"CUDA_PATH(?:_V[0-9]+_[0-9]+)?", name) is not None


def _environment_names(values):
    result = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise PlanError("invalid_launch_plan: invalid environment name")
        name = key.upper() if os.name == "nt" else key
        if name in result and (type(result[name]) is not type(value) or result[name] != value):
            raise PlanError("invalid_launch_plan: conflicting environment aliases")
        result[name] = value
    return result


def freeze_environment(cfg, inherited):
    from serve.server import child_env
    # Relevant unknown controls refuse instead of exposing arbitrary secret values.
    inherited = _environment_names(inherited)
    configured = cfg.get("env", {})
    if not isinstance(configured, dict):
        raise PlanError("invalid_launch_plan: invalid native environment")
    configured = _environment_names(configured)
    if any(k not in NATIVE_ENV and k not in CREDENTIAL_ENV and not _library_search_name(k) for k in configured):
        raise PlanError("invalid_launch_plan: unsupported configured native environment")
    for key in inherited:
        if _relevant(key) and key not in NATIVE_ENV | set(CREDENTIAL_ENV) and not _library_search_name(key):
            raise PlanError("invalid_launch_plan: unsupported inherited native environment")
    safe = {k: v for k, v in inherited.items() if k not in CREDENTIAL_ENV}
    runtime = copy.deepcopy(cfg)
    runtime["env"] = {k: v for k, v in configured.items() if k not in CREDENTIAL_ENV}
    env = child_env(runtime, inherited=safe)
    env["CUDA_MODULE_LOADING"] = env.get("CUDA_MODULE_LOADING", "EAGER")
    frozen = {k: v for k, v in env.items() if _relevant(k)}
    for key, value in frozen.items():
        if not isinstance(value, str) or len(value) > 32768 or any(ord(c) < 32 for c in value):
            raise PlanError("invalid_launch_plan: invalid native environment value")
        if not _library_search_name(key):
            if key in ("CUDA_MODULE_LOADING", "CUDA_DEVICE_ORDER", "STRATA_OWNED_PRICE"):
                choices = {"CUDA_MODULE_LOADING": {"EAGER", "LAZY"}, "CUDA_DEVICE_ORDER": {"PCI_BUS_ID"},
                           "STRATA_OWNED_PRICE": {"exact"}}
                if value not in choices[key]:
                    raise PlanError("invalid_launch_plan: unsupported native environment value")
            elif key == "STRATA_RESIDENT_HEADROOM_GIB":
                _gib_bytes(value)
            elif not re.fullmatch(r"0|[1-9][0-9]*", value):
                raise PlanError("invalid_launch_plan: unsupported numeric native environment")
    contract = {"set": frozen, "remove": sorted(set(CREDENTIAL_ENV) | (NATIVE_ENV - frozen.keys()) |
                                               (set(SEARCH_ENV) - frozen.keys())),
                "remove_prefixes": list(PREFIXES)}
    return env, contract


def replay_environment(contract, inherited=None):
    """PartB starts from a filtered parent and restores only approved native inputs."""
    _keys(contract, {"set", "remove", "remove_prefixes"})
    if contract["remove_prefixes"] != list(PREFIXES):
        raise PlanError("invalid_launch_plan: environment prefix contract mismatch")
    if (not isinstance(contract["set"], dict) or not isinstance(contract["remove"], list) or
            any(not isinstance(k, str) for k in contract["remove"]) or
            contract["remove"] != sorted(set(contract["remove"])) or
            set(CREDENTIAL_ENV) - set(contract["remove"]) or
            set(contract["set"]) & set(contract["remove"])):
        raise PlanError("invalid_launch_plan: invalid environment removals")
    for key, value in contract["set"].items():
        if ((key not in NATIVE_ENV and not _library_search_name(key)) or not isinstance(value, str) or
                len(value) > 32768 or any(ord(c) < 32 for c in value)):
            raise PlanError("invalid_launch_plan: invalid environment addition")
        if key == "STRATA_RESIDENT_HEADROOM_GIB":
            _gib_bytes(value)
    env = {k: v for k, v in (os.environ if inherited is None else inherited).items()
           if not _relevant(k.upper() if os.name == "nt" else k) and
           (k.upper() if os.name == "nt" else k) not in contract["remove"]}
    env.update(contract["set"])
    return env


def _environment_fingerprint(contract):
    value = 14695981039346656037
    for key, item in sorted(contract["set"].items()):
        # Native uses UTF-8 environment bytes. Lengths are byte lengths too.
        name, data = key.encode("utf-8"), item.encode("utf-8")
        entry = str(len(name)).encode() + b":" + name + str(len(data)).encode() + b":" + data
        for byte in entry:
            value = ((value ^ byte) * 1099511628211) & (2**64 - 1)
    return value


def config_digest(cfg):
    """Credential-independent identity of the requested config, before policy refresh."""
    cfg = copy.deepcopy(cfg)
    for name in ("api_key", "api_key_file", "log"):
        cfg.pop(name, None)
    if isinstance(cfg.get("env"), dict):
        cfg["env"] = {k: v for k, v in cfg["env"].items() if k.upper() not in CREDENTIAL_ENV}
    if isinstance(cfg.get("memory_policy"), dict):
        cfg["memory_policy"].pop("sample", None)
        for owner in (cfg["memory_policy"], cfg["memory_policy"].get("overrides", {})):
            for field in ("env", "generated_env"):
                if isinstance(owner.get(field), dict):
                    owner[field] = {k: v for k, v in owner[field].items() if k.upper() not in CREDENTIAL_ENV}
    return hashlib.sha256(canonical(cfg)).hexdigest()


class _HeaderReader:
    def __init__(self, handle):
        self.handle, self.used = handle, 0
        self.sha = hashlib.sha256()

    def read(self, size):
        if size < 0 or size > MAX_METADATA - self.used:
            raise PlanError("invalid_launch_plan: GGUF metadata exceeds bound")
        data = self.handle.read(size)
        if len(data) != size:
            raise PlanError("invalid_launch_plan: truncated GGUF metadata")
        self.used += len(data)
        self.sha.update(data)
        return data

    def tell(self):
        return self.handle.tell()


class _MetadataGGUF(GGUFFile):
    def _parse(self, handle):
        reader = _HeaderReader(handle)
        super()._parse(reader)
        self.header_bytes, self.header_sha256 = reader.used, reader.sha.hexdigest()


def file_identity(path, *, metadata=False, gguf=False):
    path = Path(path)
    before = path.stat()
    result = {"path": str(path), "size_bytes": before.st_size,
              "file_id": [before.st_dev, before.st_ino], "content_sha256": None,
              "metadata_bytes": None, "metadata_sha256": None}
    if gguf:
        parsed = _MetadataGGUF(path)
        result.update(metadata_bytes=parsed.header_bytes, metadata_sha256=parsed.header_sha256)
    elif metadata:
        result["content_sha256"] = hashlib.sha256(_read(path, MAX_METADATA)).hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise PlanError("invalid_launch_plan: input changed while reading")
    return result


def _gib_bytes(value, *, positive=False):
    """Validate decimal transport against the native resident owner's double-to-byte semantics."""
    if not isinstance(value, str) or not re.fullmatch(r"\+?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value):
        raise PlanError("invalid_launch_plan: invalid GiB setting")
    gib = float(value)
    if not math.isfinite(gib) or gib < 0 or (positive and gib <= 0) or gib > 1024:
        raise PlanError("invalid_launch_plan: invalid or unsupported GiB setting")
    return max(1 if positive else 0, int(gib * 2**30))


def _flags(args):
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise PlanError("invalid_launch_plan: invalid native arguments")
    flags, i = {}, 0
    while i < len(args):
        key = args[i]
        if key in flags or key not in VALUE_FLAGS | SWITCHES:
            raise PlanError("invalid_launch_plan: duplicate or unsupported native option")
        if key in SWITCHES:
            flags[key], i = True, i + 1
        else:
            if i + 1 >= len(args):
                raise PlanError("invalid_launch_plan: missing native option value")
            flags[key], i = args[i + 1], i + 2
    if not {"--pack", "--native", "--expert-profile", "--max-context", "--prefill",
            "--resident-budget-gib", "--expert-cache", "--spec", "--no-prefill-borrow"} <= flags.keys():
        raise PlanError("invalid_launch_plan: incomplete generated native profile")
    for key, value in flags.items():
        if key in SWITCHES or key in ("--pack", "--native", "--expert-profile", "--kv"):
            continue
        if key == "--expert-cache" and value == "auto":
            continue
        if key == "--prefill" and value in AUTO_PREFILL:
            continue
        if key == "--resident-budget-gib":
            _gib_bytes(value, positive=True)
            continue
        if not re.fullmatch(r"0|[1-9][0-9]*", value):
            raise PlanError("invalid_launch_plan: native codec requires canonical integer settings")
    if flags["--spec"] != "2" or flags.get("--suffix-draft", "0") != "0" or flags.get("--lookup-chain", "0") != "0":
        raise PlanError("invalid_launch_plan: unsupported native speculation")
    return flags


def main_shard_paths(native):
    native = Path(native)
    match = re.fullmatch(r"(.*)-[0-9]{5}-of-([0-9]{5})\.gguf", native.name)
    shards = [native]
    if match:
        count = int(match[2])
        if not 1 <= count <= 1024:
            raise PlanError("invalid_launch_plan: unsupported shard family")
        shards = [native.with_name(f"{match[1]}-{i:05}-of-{count:05}.gguf") for i in range(1, count + 1)]
    return shards


def allocation_inputs(flags):
    pack, native = Path(flags["--pack"]), Path(flags["--native"])
    paths = [(Path(flags["--expert-profile"]), True, False)]
    for name in ("index.txt", "native_experts.txt"):
        paths.append((pack / name, True, False))
    for name in ("dense.bin", "embd.bin", "experts.bin", "extra.bin"):
        if (pack / name).exists():
            paths.append((pack / name, False, False))
    shards = main_shard_paths(native)
    if not (pack / "experts.bin").exists():
        # The native layout owner's final column names per-layer or per-role
        # sources. Include these even when they are outside the main split family.
        for line in _read(pack / "native_experts.txt", MAX_METADATA).decode("utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            columns = line.split()
            if len(columns) >= 9:
                for name in columns[8].split(","):
                    source = Path(os.path.abspath(native.parent / name)) if name else native
                    if source not in shards:
                        shards.append(source)
    paths += [(p, False, True) for p in shards]
    return [file_identity(p, metadata=m, gguf=g) for p, m, g in paths]


def _codec_fields(payload):
    if not payload or len(payload) > MAX_PAYLOAD or not payload.endswith(b"\n"):
        raise PlanError("invalid_launch_plan: invalid native payload length")
    try:
        if any(byte != 10 and not 32 <= byte <= 126 for byte in payload):
            raise ValueError()
        lines = payload.decode("ascii").splitlines()
        pairs = [line.split("=", 1) for line in lines]
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError()
        fields = _pairs(pairs)
        if any(not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*|", v) for v in fields.values()):
            raise ValueError()
        return fields
    except (ValueError, UnicodeError):
        raise PlanError("invalid_launch_plan: malformed native codec") from None


def validate_native(native, settings):
    _keys(native, {"capabilities", "build", "native_payload_base64", "native_payload_sha256", "effective_payload"})
    caps = native["capabilities"]
    _keys(caps, CAPABILITIES)
    if (type(caps["schema"]) is not int or type(caps["native_schema"]) is not int or
            caps["schema"] != 1 or caps["native_schema"] != 2 or caps["kind"] != "native_metadata" or
            caps["strict_consumer"] is not True or caps["ownership_telemetry"] is not True or
            caps["effective_protocol"] != "STRATA-EFFECTIVE/1" or caps["ownership_protocol"] != "STRATA-OWNERSHIP/1"):
        raise PlanError("invalid_launch_plan: required native capability unavailable")
    for key in ("ownership_frame_limit", "ownership_shard_limit"):
        _uint(caps[key], positive=True)
    build = native["build"]
    if (not isinstance(build, dict) or build.get("schema_version") != 1 or build.get("backend") != "cuda" or
            not {"version", "commit", "dirty", "cuda"} <= build.keys() or "built_at" in build):
        raise PlanError("invalid_launch_plan: invalid native build identity")
    _keys(settings, set(SCALARS) | set(VECTORS))
    for key in SCALARS:
        _uint(settings[key])
    for key in VECTORS:
        if not isinstance(settings[key], list):
            raise PlanError("invalid_launch_plan: invalid native layout")
        values = settings[key]
        if key == "slots":
            if not values or any(not isinstance(row, list) or len(row) != 4 for row in values):
                raise PlanError("invalid_launch_plan: invalid ordered slot layout")
            values = [v for row in values for v in row]
        for value in values:
            _uint(value)
    try:
        payload = base64.b64decode(native["native_payload_base64"], validate=True)
    except (ValueError, TypeError):
        raise PlanError("invalid_launch_plan: malformed payload encoding") from None
    if hashlib.sha256(payload).hexdigest() != native["native_payload_sha256"]:
        raise PlanError("invalid_launch_plan: native payload digest mismatch")
    fields = _codec_fields(payload)
    _keys(fields, {"schema"} | set(SCALARS) | set(VECTORS))
    if fields.pop("schema") != "2":
        raise PlanError("invalid_launch_plan: unsupported native codec schema")
    for key, value in settings.items():
        flat = [n for row in value for n in row] if key == "slots" else value
        expected = ",".join(map(str, flat)) if isinstance(flat, list) else str(flat)
        if fields[key] != expected:
            raise PlanError("invalid_launch_plan: payload disagrees with resolved settings")
    effective_bytes = native["effective_payload"].encode("ascii")
    if len(effective_bytes) > 16384:
        raise PlanError("invalid_launch_plan: effective payload exceeds bound")
    effective = _codec_fields(effective_bytes)
    _keys(effective, {"schema", "verified", "cache_slots", "cache_bytes", "kv_stage_runs",
                      "native_payload_fingerprint"} | set(SCALARS))
    fingerprint = 14695981039346656037
    for byte in payload:
        fingerprint = ((fingerprint ^ byte) * 1099511628211) & (2**64 - 1)
    last = settings["slots"][-1]
    extra = {"schema": "1", "verified": "1", "cache_slots": str(len(settings["slots"])),
             "cache_bytes": str(last[3] + (last[2] + 255) // 256 * 256),
             "kv_stage_runs": fields["kv_stage_runs"], "native_payload_fingerprint": str(fingerprint)}
    if effective != {**{k: fields[k] for k in SCALARS}, **extra}:
        raise PlanError("invalid_launch_plan: effective payload disagreement")
    if (settings["spec"] != 2 or not settings["prefill"] or settings["batch"] != 0 or
            settings["parallel"] != 1 or settings["no_prefill_borrow"] != 1):
        raise PlanError("invalid_launch_plan: unsupported resolved profile")
    s = settings
    if (not 0 < s["context"] <= 1048576 or not s["prefill"] <= min(32768, s["context"]) or
            s["kv"] > 3 or s["batch_groups"] != 1 or s["auto_cache"] > 1 or s["per_layer"] > 1 or
            not s["resident_budget"] or s["resident_pin"] != 1 or s["shared_stage"] != 0 or
            not 0 < s["experts"] <= 512 or s["top_k"] != 10 or len(s["layer_blobs"]) != 48 or
            len(s["slots"]) > 48 * s["experts"] or s["resident_experts"] != len(s["resident_order"]) or
            s["commit_required"] != s["ram_required"] + s["vram_required"] or
            (s["auto_cache"] and s["cache_request"]) or len(s["kv_stage_runs"]) > 4 or
            bool(s["kv_stage_runs"]) != (0 < s["kv_resident"] < s["context"]) or
            any(not s[k] for k in ("vram_required", "ram_required", "commit_required"))):
        raise PlanError("invalid_launch_plan: inconsistent native geometry")
    if any(not 0 < b <= s["max_blob"] for b in s["layer_blobs"]) or max(s["layer_blobs"]) != s["max_blob"]:
        raise PlanError("invalid_launch_plan: invalid native layer geometry")
    seen, offset = set(), 0
    for layer, expert, size, start in s["slots"]:
        if (layer >= 48 or expert >= s["experts"] or size != s["layer_blobs"][layer] or
                start != offset or (layer, expert) in seen):
            raise PlanError("invalid_launch_plan: invalid ordered cache geometry")
        seen.add((layer, expert))
        offset += (size + 255) // 256 * 256
    resident_bytes = 0
    for index in s["resident_order"]:
        layer, expert = divmod(index, s["experts"])
        if layer >= 48 or (layer, expert) in seen:
            raise PlanError("invalid_launch_plan: invalid resident membership")
        seen.add((layer, expert))
        resident_bytes += s["layer_blobs"][layer]
    if resident_bytes != s["resident_bytes"] or resident_bytes > s["resident_budget"]:
        raise PlanError("invalid_launch_plan: invalid resident byte count")
    return payload


def strict_frame(plan):
    """Validated original codec bytes, with the native stdin frame; no re-encoding."""
    validate_plan(plan)
    payload = base64.b64decode(plan["native"]["native_payload_base64"], validate=True)
    return b"STRATA-PLAN/1\n" + str(len(payload)).encode("ascii") + b"\n" + payload


def _validate_plan(plan):
    _keys(plan, PLAN_KEYS)
    if type(plan["schema"]) is not int or plan["schema"] != SCHEMA or plan["kind"] != "strata_launch_plan":
        raise PlanError("invalid_launch_plan: unsupported public schema")
    if plan["plan_digest"] != digest(plan):
        raise PlanError("invalid_launch_plan: public digest mismatch")
    validate_hardware(plan["hardware"])
    validate_native(plan["native"], plan["resolved"])
    _keys(plan["launch"], {"executable", "argv", "cwd", "environment"})
    for key in ("executable", "cwd"):
        if not isinstance(plan["launch"][key], str) or not Path(plan["launch"][key]).is_absolute():
            raise PlanError("invalid_launch_plan: absolute launch paths required")
    argv = plan["launch"]["argv"]
    if (not isinstance(argv, list) or argv[:1] != [plan["launch"]["executable"]] or
            argv[1:3] != ["--strict-launch-stdin", "--serve"]):
        raise PlanError("invalid_launch_plan: invalid strict argv")
    flags = _flags(argv[3:])
    s = plan["resolved"]
    if not set(plan["server"]) <= SERVER_KEYS:
        raise PlanError("invalid_launch_plan: unsupported server settings")
    if server_settings(plan["server"], plan["launch"]["cwd"]) != plan["server"]:
        raise PlanError("invalid_launch_plan: server settings are not resolved")
    for flag, key in (("--max-context", "context"), ("--prefill", "prefill"), ("--spec", "spec"),
                      ("--kv-resident", "kv_resident")):
        if int(flags.get(flag, "0")) != s[key]:
            raise PlanError("invalid_launch_plan: argv contradicts resolved settings")
    if (_gib_bytes(flags["--resident-budget-gib"], positive=True) != s["resident_budget"] or
            (flags["--expert-cache"] == "auto") != bool(s["auto_cache"]) or
            (flags["--expert-cache"] != "auto" and int(flags["--expert-cache"]) != s["cache_request"])):
        raise PlanError("invalid_launch_plan: cache provenance disagreement")
    kv = {"fp16": 0, "int8": 1, "q4_0": 2, "k8v4": 3}
    if (kv.get(flags.get("--kv", "int8")) != s["kv"] or
            int(flags.get("--vram-reserve-mib", "700")) * 2**20 != s["vram_reserve"] or
            bool(flags.get("--expert-cache-per-layer")) != bool(s["per_layer"])):
        raise PlanError("invalid_launch_plan: argv allocation disagreement")
    _keys(plan["admission"], {"vram_free_bytes", "ram_available_bytes", "commit_available_bytes"})
    for resource, key in (("vram_free_bytes", "vram_required"), ("ram_available_bytes", "ram_required"),
                           ("commit_available_bytes", "commit_required")):
        if plan["admission"][resource] != s[key] or s[key] > plan["hardware"][resource]:
            raise PlanError("insufficient_resources: plan admission requirements not met")
    estimates = plan["estimates"]
    _keys(estimates, {"units", "vram", "ram", "vram_required", "ram_required", "commit_required",
                      "unknown_costs", "qualification", "runtime_comparison"})
    if estimates["units"] != "bytes":
        raise PlanError("invalid_launch_plan: invalid estimate units")
    for group, total in (("vram", "vram_required"), ("ram", "ram_required")):
        if not isinstance(estimates[group], dict):
            raise PlanError("invalid_launch_plan: invalid component estimates")
        for value in estimates[group].values():
            _uint(value)
        if sum(estimates[group].values()) != estimates[total] or estimates[total] != s[total]:
            raise PlanError("invalid_launch_plan: component estimate disagreement")
    if estimates["commit_required"] != s["commit_required"]:
        raise PlanError("invalid_launch_plan: commit estimate disagreement")
    replay_environment(plan["launch"]["environment"], {})
    if _gib_bytes(plan["launch"]["environment"]["set"].get("STRATA_RESIDENT_HEADROOM_GIB", "4")) != s["host_headroom"]:
        raise PlanError("invalid_launch_plan: environment headroom disagreement")
    if _environment_fingerprint(plan["launch"]["environment"]) != s["environment"]:
        raise PlanError("invalid_launch_plan: native environment fingerprint mismatch")
    _keys(plan["requested"], {"args", "native_args", "memory_policy"})
    requested_args = plan["requested"]["native_args"]
    requested_flags = _flags(requested_args)
    concrete_args = list(requested_args)
    if requested_flags["--prefill"] in AUTO_PREFILL:
        concrete_args[concrete_args.index("--prefill") + 1] = str(s["prefill"])
    if concrete_args != argv[3:]:
        raise PlanError("invalid_launch_plan: requested native arguments disagree")
    _flags(plan["requested"]["args"])
    _keys(plan["inputs"], {"config_sha256", "config_path", "executable_sha256", "files"})
    for key in ("config_sha256", "executable_sha256"):
        if not isinstance(plan["inputs"][key], str) or not re.fullmatch(r"[0-9a-f]{64}", plan["inputs"][key]):
            raise PlanError("invalid_launch_plan: invalid input digest")
    source = plan["inputs"]["config_path"]
    if source is not None and (not isinstance(source, str) or not Path(source).is_absolute()):
        raise PlanError("invalid_launch_plan: invalid config input path")
    if not isinstance(plan["inputs"]["files"], list) or not plan["inputs"]["files"]:
        raise PlanError("invalid_launch_plan: missing input identities")
    for entry in plan["inputs"]["files"]:
        _keys(entry, {"path", "size_bytes", "file_id", "content_sha256", "metadata_bytes", "metadata_sha256"})
        _uint(entry["size_bytes"])
        if not isinstance(entry["file_id"], list) or len(entry["file_id"]) != 2:
            raise PlanError("invalid_launch_plan: invalid file identity")
        for number in entry["file_id"]:
            _uint(number)
        for key in ("content_sha256", "metadata_sha256"):
            value = entry[key]
            if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
                raise PlanError("invalid_launch_plan: invalid metadata digest")
        if (entry["metadata_bytes"] is None) != (entry["metadata_sha256"] is None):
            raise PlanError("invalid_launch_plan: incomplete metadata identity")
        if entry["metadata_bytes"] is not None:
            _uint(entry["metadata_bytes"], positive=True)
            if entry["metadata_bytes"] > entry["size_bytes"] or entry["content_sha256"] is not None:
                raise PlanError("invalid_launch_plan: inconsistent metadata identity")
        if not isinstance(entry["path"], str) or not Path(entry["path"]).is_absolute():
            raise PlanError("invalid_launch_plan: invalid input path")
    return plan


def validate_plan(plan):
    """Validate schema/integrity/transport consistency; native still owns repricing."""
    try:
        return _validate_plan(plan)
    except PlanError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, UnicodeError, RecursionError, IndexError):
        raise PlanError("invalid_launch_plan: malformed plan contract") from None


def read_plan(path):
    try:
        return validate_plan(parse_json(_read(Path(path))))
    except OSError:
        raise PlanError("invalid_launch_plan: cannot read saved plan") from None


def _validate_inputs(plan):
    """Check actual allocation metadata and executable identity without loading payloads."""
    validate_plan(plan)
    if allocation_inputs(_flags(plan["launch"]["argv"][3:])) != plan["inputs"]["files"]:
        raise PlanError("invalid_launch_plan: allocation input identity changed")
    exe = Path(plan["launch"]["executable"])
    binary = _executable_bytes(exe)
    if hashlib.sha256(binary).hexdigest() != plan["inputs"]["executable_sha256"]:
        raise PlanError("invalid_launch_plan: executable identity changed")
    cfg = {}
    if plan["inputs"]["config_path"] is not None:
        cfg = parse_json(_read(Path(plan["inputs"]["config_path"])))
        if config_digest(cfg) != plan["inputs"]["config_sha256"]:
            raise PlanError("invalid_launch_plan: configuration identity changed")
    receipt_path = _path(cfg.get("build_metadata") or str(exe.parent / "BUILD.json"), Path(plan["launch"]["cwd"]))
    build = build_provenance.validate_metadata(binary, parse_json(_read(receipt_path)), exe.name)
    if {k: v for k, v in build.items() if k != "built_at"} != plan["native"]["build"]:
        raise PlanError("invalid_launch_plan: native build identity changed")
    return plan


def validate_inputs(plan):
    try:
        return _validate_inputs(plan)
    except PlanError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, UnicodeError, RecursionError):
        raise PlanError("invalid_launch_plan: cannot validate saved input identities") from None


def _produce_plan(cfg, *, hardware=None, inherited=None, query=bounded_query, source_config=None, config_path=None):
    """Resolve a config through shared policy/composition and the native codec owner."""
    from serve.server import engine_args
    cfg = copy.deepcopy(cfg)
    if not isinstance(cfg, dict) or set(cfg) - CONFIG_KEYS:
        raise PlanError("invalid_launch_plan: unsupported configuration field")
    _flags(cfg.get("args"))
    config_sha = config_digest(cfg if source_config is None else source_config)
    if (cfg.get("backend", "cuda") != "cuda" or type(cfg.get("parallel", 1)) is not int or
            cfg.get("parallel", 1) != 1):
        raise PlanError("invalid_launch_plan: only single-CUDA serial profiles are supported")
    if cfg.get("effort_position", "start") != "start":
        raise PlanError("invalid_launch_plan: tokenizer-dependent effort settings unsupported")
    selected = cfg.get("gpu")
    if isinstance(selected, str) and selected.isdigit():
        selected = int(selected)
    configured_env = cfg.get("env", {})
    if not isinstance(configured_env, dict):
        raise PlanError("invalid_launch_plan: invalid environment configuration")
    configured_env = _environment_names(configured_env)
    mask = configured_env.get("CUDA_VISIBLE_DEVICES")
    if mask is not None:
        if not isinstance(mask, str) or not mask.isdigit() or (selected is not None and selected != int(mask)):
            raise PlanError("invalid_launch_plan: conflicting GPU selection")
        selected = int(mask)
    if selected is not None and (type(selected) is not int or selected < 0):
        raise PlanError("invalid_launch_plan: single GPU index required")
    inherited = dict(os.environ if inherited is None else inherited)
    hardware = copy.deepcopy(observe_hardware(selected) if hardware is None else hardware)
    validate_hardware(hardware)
    if selected is not None and selected != hardware["gpu_index"]:
        raise PlanError("invalid_launch_plan: sample GPU differs from selected GPU")
    cfg["gpu"] = hardware["gpu_index"]
    requested = {"args": copy.deepcopy(cfg.get("args")), "memory_policy": bool(cfg.get("memory_policy"))}
    if cfg.get("memory_policy"):
        gpu = {"index": hardware["gpu_index"], "vram_gb": hardware["vram_total_bytes"] / 2**30}
        memory = {"ram_gib": hardware["ram_total_bytes"] / 2**30,
                  "available_gib": hardware["ram_available_bytes"] / 2**30,
                  "commit_available_gib": hardware["commit_available_bytes"] / 2**30 if hardware["windows_commit"] else None,
                  "windows_commit": hardware["windows_commit"]}
        cfg = policy.refreshed_config(cfg, sample=(gpu, memory), inherited_env=inherited)
        try:
            policy.admit_automatic_launch(cfg["memory_policy"]["sample"])
        except ValueError:
            raise PlanError("insufficient_resources: automatic memory admission refused") from None
    cwd = _path(cfg.get("cwd") or str(Path.cwd()), Path.cwd())
    server = server_settings(cfg, str(cwd))
    exe = _path(cfg.get("exe"), cwd)
    args = engine_args(cfg)
    flags = _flags(args)
    for flag in ("--pack", "--native", "--expert-profile"):
        args[args.index(flag) + 1] = str(_path(flags[flag], cwd))
    # Strict's parser requires these explicit disabled modes, even when ordinary
    # startup relies on their zero defaults.
    for flag in ("--suffix-draft", "--lookup-chain"):
        if flag not in args:
            args += [flag, "0"]
    flags = _flags(args)
    requested["native_args"] = list(args)
    # Validate original requested markers too; arbitrary unknown argv is never
    # returned in a public record merely because policy would replace it.
    _flags(requested["args"])
    env, contract = freeze_environment(cfg, inherited)
    if env.get("CUDA_VISIBLE_DEVICES") != str(hardware["gpu_index"]):
        raise PlanError("invalid_launch_plan: environment GPU differs from resource sample")
    before = allocation_inputs(flags)
    binary = _executable_bytes(exe)
    receipt_path = _path(cfg.get("build_metadata") or str(exe.parent / "BUILD.json"), cwd)
    receipt = parse_json(_read(receipt_path))
    build = build_provenance.validate_metadata(binary, receipt, exe.name)
    if build["backend"] != "cuda":
        raise PlanError("invalid_launch_plan: CUDA build required")
    argv = [str(exe), "--resolve-launch", *args]
    for name, key in (("vram-free", "vram_free_bytes"), ("ram-available", "ram_available_bytes"),
                      ("commit-available", "commit_available_bytes")):
        argv += ["--plan-" + name + "-bytes", str(hardware[key])]
    result = parse_json(query(argv, cwd=str(cwd), env=env))
    _keys(result, set(CAPABILITIES) | {"native_payload", "effective_payload", "settings", "estimates", "resources"})
    if result["resources"] != {"vram_free": hardware["vram_free_bytes"], "ram_available": hardware["ram_available_bytes"],
                               "commit_available": hardware["commit_available_bytes"], "devices": 1}:
        raise PlanError("invalid_launch_plan: native used a different resource sample")
    payload = result["native_payload"].encode("utf-8")
    native = {"capabilities": {k: result[k] for k in CAPABILITIES},
              "build": {k: v for k, v in build.items() if k != "built_at"},
              "native_payload_base64": base64.b64encode(payload).decode("ascii"),
              "native_payload_sha256": hashlib.sha256(payload).hexdigest(),
              "effective_payload": result["effective_payload"]}
    resolved = result["settings"]
    validate_native(native, resolved)
    # Selection belongs to the native metadata owner. Replay carries its concrete
    # chunk while requested.native_args retains the original automatic marker.
    if flags["--prefill"] in AUTO_PREFILL:
        args[args.index("--prefill") + 1] = str(resolved["prefill"])
    estimates = {**result["estimates"], "units": "bytes",
                 "qualification": "native count-only estimates and admission allowances; not measured fit or physical residency",
                 "runtime_comparison": {"allocated_bytes": None, "physical_residency_bytes": None,
                                        "reason": "no model or engine allocation performed"}}
    plan = {"schema": SCHEMA, "kind": "strata_launch_plan",
            "launch": {"executable": str(exe), "argv": [str(exe), "--strict-launch-stdin", "--serve", *args],
                       "cwd": str(cwd), "environment": contract},
            "server": server,
            "inputs": {"config_sha256": config_sha, "config_path": str(config_path) if config_path else None,
                       "executable_sha256": hashlib.sha256(binary).hexdigest(), "files": before},
            "hardware": hardware, "requested": requested, "resolved": resolved, "native": native,
            "estimates": estimates,
            "admission": {"vram_free_bytes": resolved["vram_required"], "ram_available_bytes": resolved["ram_required"],
                          "commit_available_bytes": resolved["commit_required"]},
            "approval": "integrity identifier only; PAIR must approve complete model/library content and external inputs"}
    plan["plan_digest"] = digest(plan)
    validate_plan(plan)
    if allocation_inputs(flags) != before or hashlib.sha256(_executable_bytes(exe)).hexdigest() != plan["inputs"]["executable_sha256"]:
        raise PlanError("invalid_launch_plan: inputs changed during resolution")
    return plan


def produce_plan(cfg, *, hardware=None, inherited=None, query=bounded_query, source_config=None, config_path=None):
    """Public read-only producer with secret-free diagnostics at shared-owner seams."""
    try:
        return _produce_plan(cfg, hardware=hardware, inherited=inherited, query=query,
                             source_config=source_config, config_path=config_path)
    except PlanError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, UnicodeError,
            RecursionError, IndexError, subprocess.SubprocessError):
        raise PlanError("invalid_launch_plan: configuration, resource or native metadata resolution refused") from None


def planning_main(args):
    """Early public CLI, including failures, writes JSON only and never credentials."""
    try:
        if not args.config or args.engine != "strata":
            raise PlanError("invalid_launch_plan: planning requires --engine strata --config")
        if args.before_load or args.mcp_config or args.script:
            raise PlanError("invalid_launch_plan: unsupported planning override")
        path = Path(args.config)
        original = _read(path)
        cfg = parse_json(original)
        source_config = copy.deepcopy(cfg)
        if args.gpu is not None:
            cfg["gpu"] = args.gpu
        cfg["host"] = args.host or cfg.get("host") or "127.0.0.1"
        cfg["port"] = args.port
        if not cfg.get("tokenizer"):
            cfg["tokenizer"] = args.tokenizer
        # Ordinary startup opens only for --open, and an explicit config false
        # still wins. Freeze the resulting action rather than the source hint.
        if "open_browser" in cfg and type(cfg["open_browser"]) is not bool:
            raise PlanError("invalid_launch_plan: invalid server settings")
        cfg["open_browser"] = bool(args.open and cfg.get("open_browser") is not False)
        for flag, key in (("lazy", "lazy_load"), ("api_monitor", "api_monitor"),
                           ("fit_max_tokens", "fit_max_tokens"), ("launch_memory_metrics", "launch_memory_metrics")):
            if getattr(args, flag):
                cfg[key] = True
        for flag, key in (("idle_unload", "idle_unload_s"), ("min_free_vram_mib", "min_free_vram_mib")):
            if getattr(args, flag) is not None:
                cfg[key] = getattr(args, flag)
        if args.slot_save_path:
            from serve.server import normalized_slot_save_dir
            cfg["slot_save_path"] = normalized_slot_save_dir(args.slot_save_path)
        plan = produce_plan(cfg, source_config=source_config, config_path=path.resolve())
        if _read(path) != original:
            raise PlanError("invalid_launch_plan: config changed during resolution")
        result = plan if args.print_launch_plan else {"schema": SCHEMA, "kind": "strata_memory_estimate",
                  "plan_digest": plan["plan_digest"], "hardware": plan["hardware"],
                  "estimates": plan["estimates"], "admission": plan["admission"]}
        sys.stdout.write(canonical(result).decode("ascii") + "\n")
        return 0
    except ProbeCleanupError:
        sys.stdout.write('{"error":{"code":"invalid_launch_plan","message":"Native query cleanup could not be confirmed.","cleanup":"unconfirmed"}}\n')
        return 2
    except PlanError as error:
        code = "insufficient_resources" if str(error).startswith("insufficient_resources:") else "invalid_launch_plan"
        sys.stdout.write(canonical({"error": {"code": code, "message": str(error)}}).decode("ascii") + "\n")
        return 2
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, UnicodeError, RecursionError):
        # Native diagnostics/config/parser exceptions may include supplied secrets.
        # Never relay their text. Detailed safe seam diagnostics are available to
        # direct API callers as PlanError, without native stderr or argv values.
        sys.stdout.write('{"error":{"code":"invalid_launch_plan","message":"Launch planning refused: unsupported, malformed, changed or insufficient inputs."}}\n')
        return 2
