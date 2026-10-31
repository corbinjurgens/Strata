"""Bounded launch admission and stdout records; StrataEngine owns all I/O/lifetime."""
from __future__ import annotations

import copy
import re
from pathlib import Path

from serve import launch_plan as lp
from tools import build_provenance

LIMIT = 16384
U64 = 2**64 - 1
RUNTIME_SETTINGS = frozenset("""context prefill spec kv kv_resident batch batch_groups parallel auto_cache
per_layer no_prefill_borrow resident_budget resident_bytes resident_experts resident_pin file_cache_effective
file_cache_compact host_headroom vram_reserve cache_slots cache_bytes shared_stage ring stager_threads
stager_ring stager_batch prefill_borrowed""".split())
RUNTIME_FLAGS = frozenset("""auto_cache per_layer no_prefill_borrow resident_pin file_cache_compact
shared_stage prefill_borrowed""".split())
COMPONENTS = {
    "vram": frozenset("""weights_canonical weights_native_dense weights_native_head head_scratch_logits
session_including_resident_kv verifier prefill_scratch_ring_stage prefill_token_ids prefill_kv_identity
cache_tables decode_hit_scratch expert_cache overhead_allowance decode_auxiliary embedding_fallback""".split()),
    "ram": frozenset("""resident_experts file_cache_target mapped_embedding verifier_mapped streamed_kv
prefill_host_stager host_headroom_allowance startup_scratch_allowance expert_file_views resident_exchange
session_host_staging prefill_host_buffers""".split())}
MEMORY_FIELDS = ("requested_bytes", "rounding_bytes", "logical_bytes", "mapped_bytes", "pinned_bytes", "allowance_bytes")


def build_record(executable):
    binary = lp._executable_bytes(executable)
    return build_provenance.validate_metadata(
        binary, build_provenance.metadata_for(binary, Path(executable).name), Path(executable).name)


def admit(plan):
    """Recheck identities and resources without invoking policy or native resolution."""
    lp.validate_inputs(plan)
    main_shards(plan)
    try:
        build = build_record(plan["launch"]["executable"])
        if {k: v for k, v in build.items() if k != "built_at"} != plan["native"]["build"]:
            raise ValueError()
        current = lp.observe_hardware(plan["hardware"]["gpu_index"])
        lp.validate_hardware(current)
        for key in ("gpu_index", "gpu_identity", "gpu_name", "vram_total_bytes", "devices", "windows_commit", "commit_basis"):
            if current[key] != plan["hardware"][key]:
                raise ValueError()
    except (OSError, ValueError, KeyError, TypeError):
        raise lp.PlanError("invalid_launch_plan: build or selected device topology changed") from None
    if any(current[k] < required for k, required in plan["admission"].items()):
        raise lp.PlanError("insufficient_resources: saved launch requirements not available")
    return build


def replay_cli(args, explicit):
    """Load local credentials/diagnostic destination separately from approved settings."""
    plan = lp.read_plan(args.launch_plan)
    cfg = {}
    source = plan["inputs"]["config_path"]
    if args.config and (source is None or Path(args.config).resolve() != Path(source).resolve()):
        raise lp.PlanError("invalid_launch_plan: conflicting config override")
    if source:
        cfg = lp.parse_json(lp._read(Path(source)))
    if explicit & {"before_load", "mcp_config", "script"} or ("engine" in explicit and args.engine != "strata"):
        raise lp.PlanError("invalid_launch_plan: conflicting replay override")
    fields = {"host": "host", "port": "port", "tokenizer": "tokenizer", "open": "open_browser",
              "fit_max_tokens": "fit_max_tokens", "lazy": "lazy_load", "api_monitor": "api_monitor",
              "idle_unload": "idle_unload_s", "min_free_vram_mib": "min_free_vram_mib",
              "slot_save_path": "slot_save_path", "launch_memory_metrics": "launch_memory_metrics"}
    from serve.server import DEFAULT_TOKENIZER
    defaults = {"host": "127.0.0.1", "port": 8095, "tokenizer": DEFAULT_TOKENIZER}
    for flag, key in fields.items():
        if flag in explicit:
            value = getattr(args, flag)
            if key in ("slot_save_path", "tokenizer"):
                from serve.server import normalized_slot_save_dir
                value = normalized_slot_save_dir(value) if key == "slot_save_path" else str(Path(value).absolute())
            if value != plan["server"].get(key, defaults.get(key, False if isinstance(value, bool) else 0)):
                raise lp.PlanError("invalid_launch_plan: conflicting server override")
    if "gpu" in explicit and args.gpu != str(plan["hardware"]["gpu_index"]):
        raise lp.PlanError("invalid_launch_plan: conflicting GPU override")
    admit(plan)
    from serve.credentials import resolve_credentials
    key = resolve_credentials(cfg, args.api_key)
    frozen = copy.deepcopy(plan["server"])
    frozen.update(exe=plan["launch"]["executable"], cwd=plan["launch"]["cwd"],
                  args=plan["launch"]["argv"][3:], gpu=plan["hardware"]["gpu_index"], backend="cuda")
    # Logs are local operational destinations, excluded from public plan identity.
    # Retain the source config's destination so replay does not discard native
    # startup diagnostics. Use the existing engine log/cleanup owner unchanged.
    log = cfg.get("log")
    if log not in (None, ""):
        if not isinstance(log, str) or any(ord(c) < 32 for c in log):
            raise lp.PlanError("invalid_launch_plan: invalid local log destination")
        frozen["log"] = str((Path(plan["launch"]["cwd"]) / log).absolute())
    for flag, name in fields.items():
        if flag in ("host", "port", "tokenizer"):
            setattr(args, flag, frozen.get(name, defaults[flag]))
        else:
            setattr(args, flag, frozen.get("open_browser", False) if flag == "open" else
                    (None if flag in ("idle_unload", "min_free_vram_mib", "slot_save_path") else False))
    args.engine, args.config, args.gpu = "strata", None, None
    return plan, frozen, key


def _number(value):
    if not re.fullmatch(r"0|[1-9][0-9]{0,19}", value) or int(value) > U64:
        raise ValueError()
    return int(value)


def main_shards(plan):
    argv = plan["launch"]["argv"]
    path = Path(argv[argv.index("--native") + 1])
    paths = lp.main_shard_paths(path)
    if len(paths) > 64:
        raise lp.PlanError("invalid_launch_plan: ownership main shard limit exceeded")
    return list(map(str, paths))


def ownership(payload, shard_count=None, *, runtime=False):
    rows = [line.split("\t") for line in payload.splitlines()]
    if len(rows) < 4 or [row[0] for row in rows[:3]] != ["device", "state", "physical"]:
        raise ValueError()
    if len(rows[0]) != 4 or rows[1] not in (["state", "known"], ["state", "unknown"]) or rows[2] != ["physical", "unavailable"]:
        raise ValueError()
    def unhex(value):
        if not value or not re.fullmatch(r"(?:[0-9a-f]{2})+", value):
            raise ValueError()
        raw = bytes.fromhex(value)
        decoded = raw.decode("utf-8")
        if len(raw) > 256 or any(ord(c) < 32 or ord(c) == 127 for c in decoded):
            raise ValueError()
        return decoded
    device = {"backend": unhex(rows[0][1]), "ordinal": _number(rows[0][2]),
              "identity": unhex(rows[0][3]), "identity_kind": None}
    identified = bool(re.fullmatch(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", device["identity"]))
    if runtime and (device["backend"] not in ("cuda", "hip", "sycl") or device["ordinal"] > 2**32 - 1 or
                    (not identified and device["identity"] != "unavailable")):
        raise ValueError()
    supported = device["backend"] == "cuda" and device["ordinal"] == 0 and identified
    if identified and device["backend"] == "cuda":
        device["identity_kind"] = "cuda_pci_bus_id"
    if not supported and (not runtime or rows[1][1] != "unknown"):
        raise ValueError()
    out = {"device": device, "state": rows[1][1], "physical_residency_bytes": None,
           "physical_reason": "physical residency unavailable", "totals": None, "shards": None}
    if rows[1][1] == "unknown":
        if rows[3:] != [["shards", "0"]]:
            raise ValueError()
        return out
    if len(rows) < 5 or len(rows[3]) != 8 or rows[3][0] != "totals" or len(rows[4]) != 2 or rows[4][0] != "shards":
        raise ValueError()
    totals = dict(zip(("capacity", "logical", "padding", "scratch", "shared", "unattributed", "mapped_host"),
                      map(_number, rows[3][1:])))
    count = _number(rows[4][1])
    if not 0 < count <= 64 or (shard_count is not None and count != shard_count) or len(rows) != 5 + count:
        raise ValueError()
    shards = []
    for i, row in enumerate(rows[5:]):
        if len(row) != 5 or row[0] != "shard" or _number(row[1]) != i:
            raise ValueError()
        held, unique, duplicate = map(_number, row[2:])
        if unique + duplicate != held:
            raise ValueError()
        shards.append({"index": i, "logical_bytes": held, "unique_source_bytes": unique, "duplicate_copy_bytes": duplicate})
    if (sum(s["logical_bytes"] for s in shards) + totals["unattributed"] != totals["logical"] or
            totals["logical"] + totals["padding"] + totals["scratch"] + totals["shared"] != totals["capacity"]):
        raise ValueError()
    out.update(totals={k + "_bytes": v for k, v in totals.items()}, shards=shards)
    return out


def runtime_snapshot(payload, shard_count=None):
    """Decode the closed version-one row grammar without publishing partial data."""
    if not payload.endswith("\n"):
        raise ValueError()
    lines = payload.splitlines(keepends=True)
    split = next((i for i, row in enumerate(lines) if row.startswith("total_allocated\t")), None)
    if split is None:
        raise ValueError()
    weights = ownership("".join(lines[:split]), shard_count, runtime=True)
    rows = [line.rstrip("\n").split("\t") for line in lines[split:]]
    if rows[:2] != [["total_allocated", "unknown"], ["driver_os", "unknown"]]:
        raise ValueError()
    settings, components = {}, {"vram": {}, "ram": {}}
    phase, previous = 0, ""
    for row in rows[2:]:
        if row[0] == "setting":
            if len(row) != 3 or phase or row[1] not in RUNTIME_SETTINGS or row[1] <= previous:
                raise ValueError()
            previous = row[1]
            settings[row[1]] = _number(row[2])
        elif row[0] == "component":
            if len(row) != 10 or row[1] not in COMPONENTS or row[2] not in COMPONENTS[row[1]]:
                raise ValueError()
            next_phase = 1 if row[1] == "vram" else 2
            if next_phase < phase or (next_phase == phase and row[2] <= previous):
                raise ValueError()
            phase, previous = next_phase, row[2]
            if row[3] == "unknown":
                if row[4:] != ["-"] * len(MEMORY_FIELDS):
                    raise ValueError()
                memory = dict.fromkeys(MEMORY_FIELDS)
                rounded = None
            elif row[3] == "known":
                memory = dict(zip(MEMORY_FIELDS, map(_number, row[4:])))
                requested = memory["requested_bytes"]
                rounded = requested + memory["rounding_bytes"]
                if rounded > U64 or any(memory[k] > requested for k in ("logical_bytes", "mapped_bytes", "pinned_bytes")):
                    raise ValueError()
            else:
                raise ValueError()
            components[row[1]][row[2]] = {"state": row[3], **memory, "rounded_bytes": rounded}
        else:
            raise ValueError()
    if set(settings) != RUNTIME_SETTINGS or any(settings[k] > 1 for k in RUNTIME_FLAGS):
        raise ValueError()
    if (not settings["context"] or settings["kv"] > 3 or
            not settings["parallel"] or not settings["batch_groups"] or
            (settings["batch"] == 0 and settings["parallel"] != 1) or
            (settings["no_prefill_borrow"] and settings["prefill_borrowed"])):
        raise ValueError()
    device = weights["device"]
    supported = (device["backend"] == "cuda" and device["ordinal"] == 0 and device["identity_kind"] is not None and
                 settings["batch"] == 0 and settings["parallel"] == 1 and settings["batch_groups"] == 1)
    if not supported and (weights["state"] != "unknown" or
                          any(m["state"] != "unknown" for space in components.values() for m in space.values())):
        raise ValueError()
    for space in components.values():
        for field in (*MEMORY_FIELDS, "rounded_bytes"):
            if sum(m[field] or 0 for m in space.values()) > U64:
                raise ValueError()
    return {"settings": settings, "weights": weights, "components": components}


def captured_shards(argv, cwd):
    """Bind native split order from captured arguments, without reading tensor payloads."""
    try:
        if argv.count("--native") != 1:
            return None
        path = Path(argv[argv.index("--native") + 1])
        if not path.is_absolute():
            path = Path(cwd or Path.cwd()) / path
        paths = lp.main_shard_paths(path.absolute())
        return list(map(str, paths)) if len(paths) <= 64 else None
    except (ValueError, IndexError, OSError):
        return None


class Records:
    """One process's records. Uncertain boundaries quarantine the remaining stream."""
    def __init__(self, plan=None, build=None, *, argv=None, cwd=None):
        self.plan, self.expected_build = plan, build
        self.build, self.effective, self.holdings = None, None, None
        self.sequence = 0
        self.last_valid = None
        self._published = (0, None, None)
        self.runtime_sequence = 0
        self._runtime_published = None
        self._runtime_device = None
        self.runtime_gap = False
        self.error = None
        self.quarantined = False
        self.pending = False
        self.paths = main_shards(plan) if plan else captured_shards(argv, cwd) if argv is not None else None
        self.shard_binding = argv is None or self.paths is not None

    def consume(self, line, stream):
        if self.quarantined:
            return True
        if not line.startswith(("BUILD ", "STRATA-EFFECTIVE", "STRATA-OWNERSHIP", "STRATA-RUNTIME")):
            return False
        boundary = False
        try:
            if line.startswith("BUILD "):
                record = lp.parse_json(line[6:].encode("ascii"))
                if self.build is not None or not self.expected_build or record != self.expected_build:
                    raise ValueError()
                self.build = record
                return True
            header = line.rstrip("\n")
            if header not in ("STRATA-EFFECTIVE/1", "STRATA-OWNERSHIP/1", "STRATA-RUNTIME/1"):
                raise ValueError()
            self.pending = True
            sequence_line, size_line = stream.readline(32), stream.readline(32)
            if not sequence_line.endswith("\n") or not size_line.endswith("\n"):
                raise ValueError()
            sequence, size = _number(sequence_line[:-1]), _number(size_line[:-1])
            runtime = header == "STRATA-RUNTIME/1"
            if not 0 < size <= (LIMIT - 128 if runtime else LIMIT):
                raise ValueError()
            payload = stream.read(size)
            if len(payload) != size or any(c != "\n" and c != "\t" and not 32 <= ord(c) <= 126 for c in payload):
                raise ValueError()
            if header == "STRATA-EFFECTIVE/1":
                if sequence != 1 or self.effective is not None or not self.plan or payload != self.plan["native"]["effective_payload"]:
                    raise ValueError()
                fields = lp._codec_fields(payload.encode("ascii"))
                self.effective = {k: ([int(v) for v in value.split(",")] if value else [])
                                  if k == "kv_stage_runs" else int(value) for k, value in fields.items()}
            else:
                end = stream.readline(64)
                marker = stream.readline(64)
                protocol = "RUNTIME" if runtime else "OWNERSHIP"
                if (end != "\n" or marker != f"STRATA-{protocol}-END/1 {sequence}\n" or
                        len(line) + len(sequence_line) + len(size_line) + size + len(end) + len(marker) > LIMIT):
                    raise ValueError()
                boundary = True
                if runtime:
                    if sequence <= self.runtime_sequence or (self.runtime_sequence == 0 and sequence != 1):
                        raise ValueError()
                    snapshot = runtime_snapshot(payload, len(self.paths) if self.paths else None)
                    if not self.shard_binding and snapshot["weights"]["state"] == "known":
                        raise ValueError()
                    device = snapshot["weights"]["device"]
                    identified = device["backend"] == "cuda" and device["ordinal"] == 0 and device["identity_kind"] is not None
                    if identified and self._runtime_device is not None and device != self._runtime_device:
                        raise ValueError()
                    self.check_runtime(snapshot["settings"])
                    previous = self._runtime_published
                    last_valid = previous[2] if previous else None
                    if snapshot["weights"]["state"] == "known":
                        last_valid = {"sequence": sequence, **snapshot["weights"]}
                    gap = self.runtime_gap or sequence != self.runtime_sequence + 1
                    # One reference is the publication point for settings, memory
                    # and holdings. Status takes this reference exactly once.
                    self._runtime_published = (sequence, snapshot, last_valid, gap)
                    self.runtime_sequence, self.runtime_gap = sequence, gap
                    if identified:
                        self._runtime_device = device
                    self.pending = False
                    return True
                if sequence <= self.sequence or (self.sequence == 0 and sequence != 1):
                    raise ValueError()
                snapshot = ownership(payload, len(self.paths) if self.paths else None)
                if self.holdings and snapshot["device"] != self.holdings["device"]:
                    raise ValueError()
                if sequence != self.sequence + 1:
                    self.error = "native ownership sequence gap; freshness lost"
                self.sequence, self.holdings = sequence, snapshot
                if snapshot["state"] == "known":
                    self.last_valid = {"sequence": sequence, **snapshot}
                self._published = (sequence, snapshot, self.last_valid)
            self.pending = False
            return True
        except (ValueError, TypeError, KeyError, IndexError, UnicodeError, OSError):
            self.error = "invalid or truncated native status record"
            self.quarantined = not boundary
            self.pending = False
            return True

    def ready(self, context, info):
        if not self.plan:
            return
        s = self.plan["resolved"]
        expected = self.expected_info()
        required = ("context", "kv", "kv_resident", "expert_slots", "spec", "engine")
        if (self.error or self.build is None or self.effective is None or self.holdings is None or self.sequence != 1 or
                context != s["context"] or any(str(info.get(k)) != str(expected[k]) for k in required)):
            raise lp.PlanError("invalid_launch_plan: native startup records missing or disagree")
        if self._runtime_published is not None:
            self.check_runtime(self._runtime_published[1]["settings"])

    def check_runtime(self, settings):
        if not self.plan:
            return
        expected = lp._codec_fields(self.plan["native"]["effective_payload"].encode("ascii"))
        if any(value != int(expected[k]) for k, value in settings.items() if k in expected):
            raise ValueError()

    def expected_info(self):
        s = self.plan["resolved"]
        effective = lp._codec_fields(self.plan["native"]["effective_payload"].encode("ascii"))
        return {"context": s["context"], "parallel": s["parallel"], "prefill": s["prefill"],
                "kv": {0: "fp16", 1: "int8", 2: "q4_0", 3: "k8v4"}[s["kv"]], "kv_resident": s["kv_resident"],
                "spec": s["spec"], "expert_slots": effective["cache_slots"], "expert_slots_primary": effective["cache_slots"],
                "expert_cache_mib": int(effective["cache_bytes"]) >> 20,
                "batch_slots": s["batch"], "batch_groups": s["batch_groups"], "engine": self.plan["native"]["build"]["version"]}

    def check_info(self, fields):
        if not self.plan:
            return
        expected = self.expected_info()
        if any(str(value) != str(expected[k]) for k, value in fields.items() if k in expected):
            self.error = "native INFO contradicts saved launch"

    def info_line(self, line):
        if self.plan and self.build is None:
            self.error = "native INFO precedes BUILD"
        fields = {}
        for item in line.split()[1:]:
            key, sep, value = item.partition("=")
            if not sep or key in fields:
                self.error = "invalid native INFO record"
            fields[key] = value
        self.check_info(fields)

    def publication(self):
        runtime = self._runtime_published
        if runtime is None:
            sequence, holdings, last_valid = self._published
            settings, components, gap = self.effective, None, False
        else:
            sequence, snapshot, last_valid, gap = runtime
            holdings, settings, components = snapshot["weights"], snapshot["settings"], snapshot["components"]
        snapshot = copy.deepcopy(holdings)
        if snapshot and snapshot.get("shards") is not None and self.paths:
            for shard, path in zip(snapshot["shards"], self.paths):
                shard["path"] = path
        unknown = bool(snapshot and snapshot["state"] == "unknown")
        current = {"freshness": "stale" if self.error or self.pending or gap else ("unknown" if unknown else ("current" if snapshot else "unavailable")),
                "error": self.error or ("incomplete native status record" if self.pending else
                                        ("native runtime sequence gap; freshness lost" if gap else
                                        ("native ownership unavailable" if unknown else None))), "sequence": sequence or None,
                "last_valid_snapshot": copy.deepcopy(last_valid) if unknown else None,
                **(snapshot or {"state": "unknown", "totals": None, "shards": None,
                                                "physical_residency_bytes": None, "physical_reason": "physical residency unavailable"})}
        return {"ownership": current, "effective": copy.deepcopy(settings), "components": copy.deepcopy(components),
                "runtime": runtime is not None}

    def public(self):
        return self.publication()["ownership"]


def component_comparison(estimates, components, freshness):
    result = {}
    for space in ("vram", "ram"):
        estimated = (estimates or {}).get(space, {})
        actual = (components or {}).get(space, {})
        rows = {}
        for name in sorted(estimated.keys() | actual.keys()):
            memory, estimate = actual.get(name), estimated.get(name)
            if name.endswith("_allowance") or name == "file_cache_target":
                field, basis = "allowance_bytes", "policy_allowance"
            elif space == "vram":
                field, basis = "rounded_bytes", "allocator_granule_model"
            elif name == "streamed_kv":
                field, basis = "logical_bytes", "logical_payload"
            else:
                field, basis = "requested_bytes", "requested_storage"
            value = memory[field] if memory else None
            state = ("unavailable" if components is None else "absent" if memory is None else
                     "unknown" if memory["state"] == "unknown" else "extra" if estimate is None else "comparable")
            rows[name] = {"state": state, "estimated_bytes": estimate, "actual": copy.deepcopy(memory),
                          "basis": basis, "actual_comparable_bytes": value,
                          "delta_bytes": value - estimate if value is not None and estimate is not None else None,
                          "freshness": freshness,
                          "reason": {"unavailable": "native runtime components unavailable",
                                     "absent": "component absent from native runtime snapshot",
                                     "unknown": "native owner measurement unavailable",
                                     "extra": "component absent from estimate"}.get(state)}
        result[space] = rows
    return result


def status(engine):
    records = getattr(engine, "records", None)
    plan = getattr(engine, "launch_plan", None)
    publication = records.publication() if records else {
        "ownership": {"state": "unknown", "freshness": "unavailable"},
        "effective": None, "components": None, "runtime": False}
    current = publication["ownership"]
    effective, components = publication["effective"], publication["components"]
    freshness = current["freshness"]
    effective_state = ("stale" if freshness == "stale" else
                       "observed" if publication["runtime"] else "verified") if effective else "unavailable"
    spawn = getattr(engine, "spawn", None)
    launch = {"mode": "planned" if plan else "ordinary", "plan_digest": plan["plan_digest"] if plan else None,
              "native_argv": list(plan["launch"]["argv"]) if plan else ([spawn[0], "--serve", *spawn[1]] if spawn else None),
              "effective": effective,
              "effective_state": effective_state,
              "effective_reason": current.get("error") if effective else "native effective settings not verified for a current process",
              "verified_startup": copy.deepcopy(records.effective) if records else None,
              "approved": {k: int(v) for k, v in lp._codec_fields(plan["native"]["effective_payload"].encode("ascii")).items()
                           if k in RUNTIME_SETTINGS} if plan else None,
              "requested": copy.deepcopy(plan["requested"]) if plan else None,
              "estimated": copy.deepcopy(plan["estimates"]) if plan else None,
              "current_ownership": current,
              "runtime_memory": {"protocol": "STRATA-RUNTIME/1" if publication["runtime"] else None,
                                 "sequence": current.get("sequence") if publication["runtime"] else None,
                                 "freshness": freshness if publication["runtime"] else "unavailable",
                                 "error": current.get("error"), "components": components,
                                 "total_allocated_bytes": None, "driver_os_bytes": None,
                                 "physical_residency_bytes": None},
              "runtime_comparison": {"weight_allocations": current.get("totals"), "total_allocated_bytes": None,
                                     "physical_residency_bytes": None,
                                     **component_comparison(plan["estimates"] if plan else None, components, freshness),
                                     "reason": "component storage and policy bases only; overlapping weight ownership is not added; "
                                               "process-wide allocations, driver/OS usage and physical residency unmeasured"}}
    build = records.build if records else None
    public_build = {"status": "current" if build else "unavailable", "reason": None if build else "current native BUILD record unavailable",
                    **{k: build.get(k) if build else None for k in ("commit", "dirty", "built_at", "cuda", "version", "backend")}}
    return launch, public_build
