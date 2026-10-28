"""Bounded launch admission and stdout records; StrataEngine owns all I/O/lifetime."""
from __future__ import annotations

import copy
import re
from pathlib import Path

from serve import launch_plan as lp
from tools import build_provenance

LIMIT = 16384
U64 = 2**64 - 1


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
    """Load the original credential source separately from frozen public settings."""
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


def ownership(payload, shard_count=None):
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
              "identity": unhex(rows[0][3]), "identity_kind": "cuda_pci_bus_id"}
    if (device["backend"] != "cuda" or device["ordinal"] != 0 or
            not re.fullmatch(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", device["identity"])):
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


class Records:
    """One process's records. Uncertain boundaries quarantine the remaining stream."""
    def __init__(self, plan=None, build=None):
        self.plan, self.expected_build = plan, build
        self.build, self.effective, self.holdings = None, None, None
        self.sequence = 0
        self.last_valid = None
        self._published = (0, None, None)
        self.error = None
        self.quarantined = False
        self.pending = False
        self.paths = main_shards(plan) if plan else None

    def consume(self, line, stream):
        if self.quarantined:
            return True
        if not line.startswith(("BUILD ", "STRATA-EFFECTIVE", "STRATA-OWNERSHIP")):
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
            if header not in ("STRATA-EFFECTIVE/1", "STRATA-OWNERSHIP/1"):
                raise ValueError()
            self.pending = True
            sequence_line, size_line = stream.readline(32), stream.readline(32)
            if not sequence_line.endswith("\n") or not size_line.endswith("\n"):
                raise ValueError()
            sequence, size = _number(sequence_line[:-1]), _number(size_line[:-1])
            if not 0 < size <= LIMIT:
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
                if (end != "\n" or marker != f"STRATA-OWNERSHIP-END/1 {sequence}\n" or
                        len(line) + len(sequence_line) + len(size_line) + size + len(end) + len(marker) > LIMIT):
                    raise ValueError()
                boundary = True
                if sequence <= self.sequence or (self.sequence == 0 and sequence != 1):
                    raise ValueError()
                snapshot = ownership(payload, len(self.paths) if self.paths else None)
                if self.holdings and snapshot["device"] != self.holdings["device"]:
                    raise ValueError()
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

    def public(self):
        sequence, holdings, last_valid = self._published
        snapshot = copy.deepcopy(holdings)
        if snapshot and snapshot.get("shards") is not None and self.paths:
            for shard, path in zip(snapshot["shards"], self.paths):
                shard["path"] = path
        unknown = bool(snapshot and snapshot["state"] == "unknown")
        return {"freshness": "stale" if self.error or self.pending else ("unknown" if unknown else ("current" if snapshot else "unavailable")),
                "error": self.error or ("incomplete native status record" if self.pending else
                                        ("native ownership unavailable" if unknown else None)), "sequence": sequence or None,
                "last_valid_snapshot": copy.deepcopy(last_valid) if unknown else None,
                **(snapshot or {"state": "unknown", "totals": None, "shards": None,
                                                "physical_residency_bytes": None, "physical_reason": "physical residency unavailable"})}


def status(engine):
    records = getattr(engine, "records", None)
    plan = getattr(engine, "launch_plan", None)
    current = records.public() if records else {"state": "unknown", "freshness": "unavailable"}
    spawn = getattr(engine, "spawn", None)
    launch = {"mode": "planned" if plan else "ordinary", "plan_digest": plan["plan_digest"] if plan else None,
              "native_argv": list(plan["launch"]["argv"]) if plan else ([spawn[0], "--serve", *spawn[1]] if spawn else None),
              "effective": copy.deepcopy(records.effective) if records else None,
              "effective_state": ("stale" if records.error else "verified") if records and records.effective else "unavailable",
              "effective_reason": None if records and records.effective else "native effective settings not verified for a current process",
              "requested": copy.deepcopy(plan["requested"]) if plan else None,
              "estimated": copy.deepcopy(plan["estimates"]) if plan else None,
              "current_ownership": current,
              "runtime_comparison": {"weight_allocations": current.get("totals"), "total_allocated_bytes": None,
                                     "physical_residency_bytes": None,
                                     "reason": "weight ownership only; total allocations and physical residency unmeasured"}}
    build = records.build if records else None
    public_build = {"status": "current" if build else "unavailable", "reason": None if build else "current native BUILD record unavailable",
                    **{k: build.get(k) if build else None for k in ("commit", "dirty", "built_at", "cuda", "version", "backend")}}
    return launch, public_build
