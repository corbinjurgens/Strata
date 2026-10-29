"""Pure public memory pricing. Native metadata owns all allocation geometry."""
from __future__ import annotations

import copy
from decimal import Decimal
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

from serve import launch_plan as lp
from tools import build_provenance

GROUPS = ("weights", "kv_cache", "compute_scratch", "output_buffers",
          "runtime_driver_overhead", "transient_peak")
KV = {"fp16": 0, "int8": 1, "q4_0": 2, "k8v4": 3}
KV_ALIASES = {"f16": "fp16", "q8_0": "int8"}
MIB = 1 << 20
GIB = 1 << 30
MAX_BYTES = (1 << 64) - 1
NATIVE_UNKNOWNS = "CUDA context/modules, graph capture, CPU TLS/request scratch, driver and OS residency"
NATIVE_ENVELOPE_UNKNOWNS = (NATIVE_UNKNOWNS +
    "; prefill_scratch_ring_stage is a conservative CUDA kernel-capability envelope "
    "(possible MMQ plus FP16 fallback, fused/nonfused carve maxima and native ring bound), not observed runtime allocation; "
    "prefill settings are envelope count inputs, not replayable runtime settings")
# Every allowance is a caller assertion of an upper bound, never a measurement.
ALLOWANCES = {
    "device_runtime_context": "device",
    "device_late_libraries": "device",
    "device_transient_peak": "device",
    "host_ram_runtime": "host_ram",
    "host_ram_transient_peak": "host_ram",
    "host_commit_runtime": "host_commit",
    "host_commit_transient_peak": "host_commit",
}
ENV_CHOICES = {
    "STRATA_FILE_CACHE_COMPACT": {"0", "1"},
    "STRATA_OWNED_PRICE": {"exact"},
    "STRATA_PREFILL_MMQ": {"0", "1"},
    "STRATA_PREFILL_RESERVE_BO": {"0", "1"},
    "CUDA_MODULE_LOADING": {"EAGER", "LAZY"},
    "CUDA_DEVICE_ORDER": {"PCI_BUS_ID"},
}
ENV_NUMBERS = {"STRATA_PREFILL_RING": (8, 512), "STRATA_STAGER_THREADS": (1, 256), "STRATA_STAGER_RING": (2, 256),
               "STRATA_STAGER_BATCH": (1, 256)}
DEVICE_COMPONENTS = {
    "weights_canonical", "weights_native_dense", "weights_native_head", "head_scratch_logits",
    "session_including_resident_kv", "verifier", "prefill_scratch_ring_stage", "prefill_token_ids",
    "prefill_kv_identity", "cache_tables", "decode_hit_scratch", "expert_cache", "overhead_allowance",
}
HOST_COMPONENTS = {"resident_experts", "file_cache_target", "mapped_embedding", "verifier_mapped",
                   "streamed_kv", "host_headroom_allowance", "startup_scratch_allowance", "prefill_host_stager"}


class MemoryPlanError(lp.PlanError):
    """Diagnostics contain fixed text and bounded field names only."""

    def __init__(self, code="invalid_memory_plan", field=None):
        self.code = code
        self.field = field
        super().__init__(code + (": " + field if field else ""))


def _fail(field=None, code="invalid_memory_plan"):
    raise MemoryPlanError(code, field)


def _object(value, allowed, required, field):
    if not isinstance(value, dict):
        _fail(field)
    unknown = set(value) - set(allowed)
    if unknown:
        name = sorted(str(k) for k in unknown)[0]
        safe = name if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,63}", name) else "unknown_field"
        _fail(field + "." + safe, "unsupported_field")
    if set(required) - set(value):
        _fail(field + "." + sorted(set(required) - set(value))[0])
    return value


def _int(value, field, low=0, high=MAX_BYTES):
    if type(value) is not int or not low <= value <= high:
        _fail(field)
    return value


def _sum(values):
    return _int(sum(values), "resource_total")


def _text(value, field):
    if not isinstance(value, str) or not value or len(value) > 4096 or any(ord(c) < 32 for c in value):
        _fail(field)
    return value


def _gib(value):
    # All binary GiB fractions here terminate in decimal. Refuse byte values the
    # native double-to-byte transport cannot represent instead of rounding them.
    result = format(Decimal(value) / Decimal(GIB), "f")
    if lp._gib_bytes(result, positive=value > 0) != value:
        _fail("extensions.strata.byte_transport")
    return result


def _validate(request):
    _object(request, {"schema_version", "model", "config", "budget", "extensions"},
            {"schema_version", "model", "config", "extensions"}, "request")
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        _fail("schema_version")
    model = _object(request["model"], {"id", "content_sha256"}, {"id"}, "model")
    _text(model["id"], "model.id")
    sha = model.get("content_sha256")
    if sha is not None and (not isinstance(sha, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", sha)):
        _fail("model.content_sha256")
    cfg = _object(request["config"], {"context", "kv_type", "batch_size", "microbatch_size",
                  "parallel_slots", "devices", "offload"}, {"context", "devices", "offload"}, "config")
    context = _int(cfg["context"], "config.context", 1, 1048576)
    for name in ("batch_size", "parallel_slots"):
        if _int(cfg.get(name, 1), "config." + name, 1) != 1:
            _fail("config." + name, "unsupported_field")
    micro = cfg.get("microbatch_size", "auto")
    if not (isinstance(micro, str) and micro in lp.AUTO_PREFILL):
        _int(micro, "config.microbatch_size", 1, min(context, 32768))
    kv = cfg.get("kv_type", "int8")
    if not isinstance(kv, str) or KV_ALIASES.get(kv, kv) not in KV:
        _fail("config.kv_type", "unsupported_field")
    kv = KV_ALIASES.get(kv, kv)
    devices = cfg["devices"]
    if not isinstance(devices, list) or len(devices) != 1:
        _fail("config.devices", "unsupported_field")
    dev = _object(devices[0], {"id", "backend"}, {"id", "backend"}, "config.devices")
    _int(dev["id"], "config.devices.id", 0, 65535)
    if dev["backend"] != "cuda":
        _fail("config.devices.backend", "unsupported_field")
    offload = _object(cfg["offload"], {"mode"}, {"mode"}, "config.offload")
    if offload["mode"] != "hybrid":
        _fail("config.offload.mode", "unsupported_field")
    ext = _object(request["extensions"], {"strata"}, {"strata"}, "extensions")
    required = {"executable", "working_directory", "pack", "native_model", "expert_profile",
                "resident_budget_bytes", "file_cache_bytes", "host_headroom_bytes", "vram_reserve_bytes",
                "cache_slots", "kv_resident_cells", "spec", "no_prefill_borrow"}
    s = _object(ext["strata"], required | {"environment", "allowances"}, required, "extensions.strata")
    for key in ("resident_budget_bytes", "file_cache_bytes", "host_headroom_bytes", "vram_reserve_bytes"):
        _int(s[key], "extensions.strata." + key, int(key == "resident_budget_bytes"), 1 << 40)
    for key in ("file_cache_bytes", "vram_reserve_bytes"):
        if s[key] % MIB:
            _fail("extensions.strata." + key)
    _int(s["spec"], "extensions.strata.spec", 2, 2)
    if s["no_prefill_borrow"] is not True:
        _fail("extensions.strata.no_prefill_borrow", "unsupported_field")
    _int(s["kv_resident_cells"], "extensions.strata.kv_resident_cells", 0, context - 1)
    cache = s["cache_slots"]
    if cache != "auto":
        _int(cache, "extensions.strata.cache_slots", 1, 48 * 512)
    allowances = _object(s.get("allowances", {}), ALLOWANCES, set(), "extensions.strata.allowances")
    for key, value in allowances.items():
        _int(value, "extensions.strata.allowances." + key)
    env = _object(s.get("environment", {}), ENV_CHOICES.keys() | ENV_NUMBERS.keys(), set(),
                  "extensions.strata.environment")
    for key, value in env.items():
        if not isinstance(value, str):
            _fail("extensions.strata.environment." + key)
        if key in ENV_CHOICES:
            if value not in ENV_CHOICES[key]:
                _fail("extensions.strata.environment." + key)
        elif (not re.fullmatch(r"[1-9][0-9]{0,2}", value) or
              not ENV_NUMBERS[key][0] <= int(value) <= ENV_NUMBERS[key][1]):
            _fail("extensions.strata.environment." + key)
        if key == "STRATA_PREFILL_RING" and 8 < int(value) < 16:
            _fail("extensions.strata.environment." + key)
    budget = request.get("budget")
    if "budget" in request:
        _object(budget, {"basis", "devices", "host_ram_bytes", "host_commit_bytes"},
                {"basis", "devices", "host_ram_bytes", "host_commit_bytes"}, "budget")
        if budget["basis"] not in ("before_runtime_initialization", "after_runtime_initialization"):
            _fail("budget.basis")
        bdev = budget["devices"]
        if not isinstance(bdev, list) or len(bdev) != 1:
            _fail("budget.devices")
        _object(bdev[0], {"id", "bytes"}, {"id", "bytes"}, "budget.devices")
        if _int(bdev[0]["id"], "budget.devices.id") != dev["id"]:
            _fail("budget.devices.id")
        for value, field in ((bdev[0]["bytes"], "budget.devices.bytes"),
                             (budget["host_ram_bytes"], "budget.host_ram_bytes"),
                             (budget["host_commit_bytes"], "budget.host_commit_bytes")):
            _int(value, field)
    if budget is None and (cache == "auto" or isinstance(micro, str)):
        _fail("budget", "requires_budget")
    return cfg, s, dev, kv, micro, budget


def _transport(s, cfg, dev, kv, micro, budget):
    cwd = lp._path(s["working_directory"], Path.cwd())
    if not cwd.is_dir():
        _fail("extensions.strata.working_directory")
    paths = {k: lp._path(s[k], cwd) for k in ("executable", "pack", "native_model", "expert_profile")}
    match = re.search(r"-([0-9]{5})-of-[0-9]{5}\.gguf$", paths["native_model"].name)
    if match and match[1] != "00001":
        _fail("extensions.strata.native_model")
    argv = [str(paths["executable"]), "--resolve-launch", "--price-only"]
    for flag, key in (("--pack", "pack"), ("--native", "native_model"), ("--expert-profile", "expert_profile")):
        argv.extend((flag, str(paths[key])))
    for flag, value in (("--max-context", cfg["context"]), ("--prefill", micro), ("--kv", kv),
                        ("--spec", 2), ("--kv-resident", s["kv_resident_cells"]),
                        ("--resident-budget-gib", _gib(s["resident_budget_bytes"])),
                        ("--expert-cache", s["cache_slots"]),
                        ("--vram-reserve-mib", s["vram_reserve_bytes"] // MIB)):
        argv.extend((flag, str(value)))
    argv.append("--no-prefill-borrow")
    if budget is not None:
        for flag, value in (("vram-free", budget["devices"][0]["bytes"]),
                            ("ram-available", budget["host_ram_bytes"]),
                            ("commit-available", budget["host_commit_bytes"])):
            argv.extend(("--plan-" + flag + "-bytes", str(value)))
    # Do not inherit allocation controls, API credentials, preload hooks or CUDA
    # discovery settings. Only the OS loader/temporary-directory environment is inherited.
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP") if k in os.environ}
    env.update(s.get("environment", {}))
    env.update(CUDA_VISIBLE_DEVICES=str(dev["id"]),
               STRATA_FILE_CACHE_MIB=str(s["file_cache_bytes"] // MIB),
               STRATA_RESIDENT_HEADROOM_GIB=_gib(s["host_headroom_bytes"]))
    flags = {"--pack": str(paths["pack"]), "--native": str(paths["native_model"]),
             "--expert-profile": str(paths["expert_profile"])}
    return cwd, paths["executable"], argv, env, flags


def _native(result, budget):
    if (not isinstance(result, dict) or result.get("schema") != 1 or result.get("kind") != "native_memory_price" or
            type(result.get("configuration_complete")) is not bool):
        _fail(code="invalid_native_price")
    expected = None if budget is None else {"vram_free": budget["devices"][0]["bytes"],
        "ram_available": budget["host_ram_bytes"], "commit_available": budget["host_commit_bytes"], "devices": 1}
    if result.get("budget") != expected:
        _fail(code="invalid_native_price")
    e = result["estimates"]
    for group, allowed, total in (("vram", DEVICE_COMPONENTS, "vram_required"),
                                  ("ram", HOST_COMPONENTS, "ram_required")):
        if not isinstance(e[group], dict) or set(e[group]) != allowed:
            _fail(code="unsupported_native_components")
        for value in e[group].values():
            _int(value, "native.components")
        if _sum(e[group].values()) != _int(e[total], "native.total"):
            _fail(code="invalid_native_price")
    if _sum((e["ram_required"], e["vram_required"])) != _int(e["commit_required"], "native.total"):
        _fail(code="invalid_native_price")
    policy_minimum = _int(e.get("minimum_policy_vram_free_bytes", 0), "native.minimum_policy_capacity")
    fit = result["fit"]
    if budget is None:
        expected_status = "not_evaluated"
        shortfall = {"vram": 0, "ram": 0, "commit": 0}
    else:
        shortfall = {name: max(0, e[total] - expected[key]) for name, total, key in (
            ("vram", "vram_required", "vram_free"), ("ram", "ram_required", "ram_available"),
            ("commit", "commit_required", "commit_available"))}
        shortfall["vram"] = max(0, max(e["vram_required"], policy_minimum) - expected["vram_free"])
        expected_status = "does_not_fit" if any(shortfall.values()) else "within_priced_budget"
    if fit != {"status": expected_status, "shortfall_bytes": shortfall}:
        _fail(code="invalid_native_price")
    if not result["configuration_complete"]:
        if result.get("settings") is not None or expected_status != "does_not_fit":
            _fail(code="invalid_native_price")
    return e


def _quantity(value, basis, quality="exact", **extra):
    return {"bytes": value, "quality": "unknown" if value is None else quality, "basis": basis, **extra}


def _groups(kind, e):
    groups = {name: _quantity(None, "Not separately priced by the native metadata owner.") for name in GROUPS}
    groups["transient_peak"] = _quantity(None, "Additional peak above priced allocations is unknown; nonadditive.",
                                          additive=False)
    groups["runtime_driver_overhead"] = _quantity(None, "Unmeasured runtime, library, driver and OS costs.")
    if kind == "device":
        groups["weights"] = _quantity(_sum(e[k] for k in (
            "weights_canonical", "weights_native_dense", "weights_native_head", "expert_cache")),
            "Native counted weight allocations; excludes tables and combined workspace.")
        groups["compute_scratch"]["combined_membership"] = ["kv_cache", "compute_scratch", "output_buffers"]
        groups["kv_cache"]["combined_membership"] = ["kv_cache", "compute_scratch", "output_buffers"]
        groups["output_buffers"]["combined_membership"] = ["compute_scratch", "output_buffers"]
    elif kind == "host_ram":
        groups["weights"] = _quantity(_sum(e[k] for k in ("resident_experts", "mapped_embedding")),
            "Pinned resident weight bytes plus mapped embedding span, not measured physical residency.", "estimated")
        groups["kv_cache"] = _quantity(e["streamed_kv"], "Native counted host KV backing allocations.")
        groups["compute_scratch"]["combined_membership"] = ["kv_cache", "compute_scratch", "output_buffers"]
    else:
        for name in ("weights", "kv_cache", "compute_scratch", "output_buffers"):
            groups[name]["basis"] = "Not partitioned: native commit is the combined device and host capacity requirement."
    return groups


def _required_allowances(kind, basis):
    return [k for k, owner in ALLOWANCES.items() if owner == kind and
            not (k == "device_runtime_context" and basis == "after_runtime_initialization")]


def _resource(kind, estimates, budget_bytes, allowances, basis):
    e = estimates["vram" if kind == "device" else "ram"]
    total = estimates[{"device": "vram_required", "host_ram": "ram_required", "host_commit": "commit_required"}[kind]]
    required_unknowns = _required_allowances(kind, basis)
    if estimates.get("unknown_costs") not in (NATIVE_UNKNOWNS, NATIVE_ENVELOPE_UNKNOWNS):
        required_unknowns.append("unclassified_native_costs")
    applicable = {k: allowances[k] for k in required_unknowns if k in allowances}
    # Native reserves/headroom stay protected. Explicit extra caller bounds are
    # charged separately, so the same bytes cannot cover startup and final reserve.
    allowance_total = _sum(applicable.values())
    policy_minimum = estimates.get("minimum_policy_vram_free_bytes", 0) if kind == "device" else 0
    required = _sum((max(total, policy_minimum), allowance_total))
    uncovered = [k for k in required_unknowns if k not in applicable]
    groups = _groups(kind, e)
    assumptions = []
    combined = []
    mapped = 0
    if kind == "device":
        allocated = total - e["overhead_allowance"]
        assumptions = [{"name": "protected_device_reserve", "bytes": e["overhead_allowance"]}]
        for keys, members in ((("session_including_resident_kv", "verifier"), ["kv_cache", "compute_scratch", "output_buffers"]),
                              (("head_scratch_logits",), ["compute_scratch", "output_buffers"]),
                              (("prefill_scratch_ring_stage",), ["compute_scratch", "kv_cache"])):
            combined.append({"bytes": _sum(e[k] for k in keys),
                             "quality": "estimated" if "prefill_scratch_ring_stage" in keys else "exact", "members": members,
                             "basis": "Counted once in priced_total_bytes; prefill may bound kernel variants; partition unavailable."})
    elif kind == "host_ram":
        assumptions = [{"name": name, "bytes": e[key]} for name, key in (
            ("file_cache_target", "file_cache_target"), ("protected_host_headroom", "host_headroom_allowance"),
            ("startup_scratch_allowance", "startup_scratch_allowance"))]
        mapped = _sum(e[k] for k in ("mapped_embedding", "verifier_mapped"))
        allocated = total - mapped - _sum(x["bytes"] for x in assumptions)
        combined = [{"bytes": e["verifier_mapped"], "quality": "estimated",
                     "members": ["kv_cache", "compute_scratch", "output_buffers"],
                     "basis": "Mapped span counted once in priced_total_bytes; physical residency unknown."}]
    else:
        allocated = None
        assumptions = [{"name": "combined_capacity_model", "bytes": total,
                        "basis": "Host RAM requirement plus device requirement, not measured OS commit."}]
    return {"groups": groups, "combined_allocations": combined,
            "allocated_bytes": allocated, "mapped_view_bytes": mapped if kind != "host_commit" else None,
            "assumptions": assumptions, "priced_total_bytes": total,
            "minimum_selection_capacity_bytes": policy_minimum,
            "minimum_selection_capacity_basis": "Native selection-policy requirement, not an allocated component.",
            "priced_total_basis": "Native capacity model includes allocations, mapped spans, targets and protected allowances as applicable; category rows are not additive.",
            "caller_allowances": applicable, "required_with_allowances_bytes": required,
            "budget_bytes": budget_bytes,
            "shortfall_bytes": None if budget_bytes is None else max(0, required - budget_bytes),
            "shortfall_basis": "Lower bound under the capacity model; uncovered runtime costs may increase it.",
            "uncovered_costs": uncovered}


def _memory_plan(request, query):
    # Round trip also bounds size, refuses non-JSON values, and detaches callers' data.
    request = lp.parse_json(lp.canonical(request))
    cfg, s, dev, kv, micro, budget = _validate(request)
    allowances = s.get("allowances", {})
    basis = budget["basis"] if budget else None
    pricing_budget = copy.deepcopy(budget)
    if pricing_budget is not None:
        for kind, owner, key in (("device", pricing_budget["devices"][0], "bytes"),
                                 ("host_ram", pricing_budget, "host_ram_bytes"),
                                 ("host_commit", pricing_budget, "host_commit_bytes")):
            withheld = _sum(allowances.get(k, 0) for k in _required_allowances(kind, basis))
            owner[key] = max(0, owner[key] - withheld)
    cwd, exe, argv, env, flags = _transport(s, cfg, dev, kv, micro, pricing_budget)
    before = lp.allocation_inputs(flags)
    binary = lp._executable_bytes(exe)
    build = build_provenance.embedded_record(binary)
    if build.get("backend") != "cuda":
        _fail("extensions.strata.executable", "unsupported_build")
    sha = hashlib.sha256(binary).hexdigest()
    result = lp.parse_json(query(argv, cwd=str(cwd), env=env))
    e = _native(result, pricing_budget)
    complete = result["configuration_complete"]
    resolved = None
    if complete:
        r = result["settings"]
        for key in lp.SCALARS:
            _int(r[key], "native.settings")
        if (r["context"] != cfg["context"] or r["kv"] != KV[kv] or r["spec"] != 2 or r["batch"] != 0 or
                r["parallel"] != 1 or r["batch_groups"] != 1 or r["no_prefill_borrow"] != 1 or
                not 0 < r["prefill"] <= min(32768, cfg["context"]) or
                (type(micro) is int and r["prefill"] != micro) or
                r["resident_budget"] != s["resident_budget_bytes"] or r["file_cache"] != s["file_cache_bytes"] or
                r["host_headroom"] != s["host_headroom_bytes"] or r["vram_reserve"] != s["vram_reserve_bytes"] or
                r["kv_resident"] != s["kv_resident_cells"] or
                (s["cache_slots"] != "auto" and r["cache_request"] != s["cache_slots"])):
            _fail(code="invalid_native_price")
        resolved = {"context": r["context"], "kv_type": kv, "batch_size": 1,
                    "microbatch_size": r["prefill"], "parallel_slots": 1,
                    "devices": copy.deepcopy(cfg["devices"]), "offload": copy.deepcopy(cfg["offload"])}
    device = _resource("device", e, budget["devices"][0]["bytes"] if budget else None, allowances, basis)
    device.update(id=dev["id"], backend="cuda")
    ram = _resource("host_ram", e, budget["host_ram_bytes"] if budget else None, allowances, basis)
    commit = _resource("host_commit", e, budget["host_commit_bytes"] if budget else None, allowances, basis)
    resources = [device, ram, commit]
    status = ("not_evaluated" if budget is None else "does_not_fit" if any(x["shortfall_bytes"] for x in resources)
              else "indeterminate" if any(x["uncovered_costs"] for x in resources) else "fits_within_allowances")
    if lp.allocation_inputs(flags) != before or hashlib.sha256(lp._executable_bytes(exe)).hexdigest() != sha:
        _fail(code="inputs_changed")
    return {"kind": "engine_memory_plan", "schema_version": 1,
            "engine": {"id": "strata", "build": build, "executable_sha256": sha,
                       "identity_basis": "Embedded build record bound to executable content, no execution."},
            "model": {"id": request["model"]["id"], "content_sha256": None,
                      "asserted_content_sha256": request["model"].get("content_sha256"),
                      "identity_basis": "Bounded metadata/header hashes and file stat identity; caller digest not verified."},
            "requested_config": request["config"], "resolved_config": resolved,
            "configuration_complete": complete, "budget_basis": basis,
            "pricing_scope": "resolved_configuration" if complete else "minimum_supported_layout",
            "resources": {"devices": [device], "host_ram": ram, "host_commit": commit},
            "fit": {"status": status, "conditional": True,
                    "basis": "Caller capacities and explicit upper-bound allowances, not live admission or measured residency."},
            "cacheable": False, "cacheable_reason": "Complete model content identity is unverified; v1 has no persistent cache.",
            "warnings": ["Unpartitioned groups are already represented by combined counted allocations.",
                         "Transient peaks are nonadditive and unknown until explicitly bounded by the caller.",
                         "After-runtime budgets do not cover late libraries, graph capture, OS or request costs automatically.",
                         "Protected reserve and headroom do not also cover unpriced startup costs."],
            "capabilities": {"live_observation": False, "max_context_search": False, "automatic_cache_fill": True,
                             "supported_backends": ["cuda"], "serial_only": True},
            "extensions": {"strata": {"requested": s, "resolved": result["settings"],
                "native_components": e, "native_fit": result["fit"], "pricing_budget": pricing_budget, "inputs": before,
                "estimate_basis": result.get("estimate_basis", "native_count_geometry"),
                "query": {"argv": argv, "environment": {k: v for k, v in env.items()
                          if k in ENV_CHOICES.keys() | ENV_NUMBERS.keys() |
                          {"CUDA_VISIBLE_DEVICES", "STRATA_FILE_CACHE_MIB", "STRATA_RESIDENT_HEADROOM_GIB"}},
                          "working_directory": str(cwd)}}}}


def memory_plan(request, *, query=None):
    """Return a deterministic engine-neutral plan, or raise a safe MemoryPlanError.

    The sole executable operation is the bounded --resolve-launch --price-only
    query. Injectable query(argv, *, cwd, env) returns native JSON bytes.
    """
    try:
        return _memory_plan(request, lp.bounded_query if query is None else query)
    except MemoryPlanError:
        raise
    except lp.ProbeCleanupError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, UnicodeError,
            RecursionError, IndexError, subprocess.SubprocessError):
        raise MemoryPlanError() from None


def planning_main(args, explicit):
    """Route only pure requests, before config/credential/server startup."""
    try:
        if (not args.estimate_memory or explicit - {"engine", "estimate_memory", "memory_plan_request"} or
                ("engine" in explicit and args.engine != "strata")):
            _fail(code="invalid_memory_plan_arguments")
        request = lp.parse_json(lp._read(Path(args.memory_plan_request)))
        result = memory_plan(request)
        sys.stdout.write(lp.canonical(result).decode("ascii") + "\n")
        return 0
    except lp.ProbeCleanupError:
        result = {"error": {"code": "invalid_memory_plan", "message": "Native query cleanup unconfirmed.", "cleanup": "unconfirmed"}}
    except MemoryPlanError as error:
        result = {"error": {"code": error.code, "message": "Memory planning refused.", "field": error.field}}
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        result = {"error": {"code": "invalid_memory_plan", "message": "Memory plan request unavailable or malformed."}}
    sys.stdout.write(lp.canonical(result).decode("ascii") + "\n")
    return 2
