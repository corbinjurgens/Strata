"""One owner for native build records and executable-bound publication.

STRATA_BUILD/1:<compact JSON>\n is embedded verbatim in the executable.
--build-info prints that JSON and exits before native startup. BUILD.json has
the identical object in `build`, plus executable (basename) and sha256. Hashes
bind bytes, not authorization. Consumers must validate both hash and embedding.
commit/dirty are null with git_reason when Git is unavailable; CUDA is null
with cuda_reason for non-CUDA/unknown toolchains. built_at is UTC link-input
preparation time, retained on a no-op. Source/config/toolchain hashes describe
the prepared inputs; publication refuses if those inputs drift during linking.
Only known source trees/extensions are read, never profiles or environment dumps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

MARKER = b"STRATA_BUILD/1:"
SOURCE_PARTS = ("CMakeLists.txt", "cmake", "src", "include", "sycl", "third_party/ggml",
                "tools/build_provenance.py")
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".hxx",
                   ".in", ".cmake", ".inc", ".s", ".S", ".py"}


def json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def legacy_source_hash(root: Path, parts, seed: str) -> str:
    """Keep setup.source_hash's existing newline-normalized 16-character contract."""
    digest = hashlib.sha256(seed.encode())
    for part in parts:
        base = root / part
        for path in [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file()):
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" +
                          path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:16]


def source_file(path: Path) -> bool:
    return path.name == "CMakeLists.txt" or path.suffix in SOURCE_SUFFIXES


def source_identity(root: Path, ggml: Path | None = None) -> str:
    digest = hashlib.sha256()
    trees = [(part, root / part) for part in SOURCE_PARTS]
    if ggml:
        trees.append(("selected-ggml", ggml / "ggml"))
    for label, base in trees:
        files = [base] if base.is_file() else sorted(
            p for p in base.rglob("*") if p.is_file() and source_file(p) and
            not any(s in {".git", "__pycache__"} for s in p.relative_to(base).parts))
        for path in files:
            name = (label + "/" + (path.relative_to(base).as_posix() if path != base else "")).encode()
            data = path.read_bytes()
            digest.update(len(name).to_bytes(8, "little") + name + len(data).to_bytes(8, "little") + data)
    return digest.hexdigest()


def git_state(root: Path) -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=30,
                              env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    try:
        top = git("rev-parse", "--show-toplevel")
        if top.returncode or Path(os.fsdecode(top.stdout).strip()).resolve() != root.resolve():
            return {"commit": None, "dirty": None, "git_reason": "not_repository_root"}
        head = git("rev-parse", "HEAD")
        commit = head.stdout.decode("ascii").strip()
        if head.returncode or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            return {"commit": None, "dirty": None, "git_reason": "head_unavailable"}
        tracked = git("diff", "--quiet", "HEAD", "--")
        untracked = git("ls-files", "--others", "--exclude-standard", "-z", "--", *SOURCE_PARTS)
        if tracked.returncode not in (0, 1) or untracked.returncode:
            return {"commit": commit, "dirty": None, "git_reason": "status_unavailable"}
        dirty = tracked.returncode == 1 or any(source_file(Path(os.fsdecode(p))) for p in
                                              untracked.stdout.split(b"\0") if p)
        return {"commit": commit, "dirty": dirty, "git_reason": None}
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return {"commit": None, "dirty": None, "git_reason": "git_unavailable"}


def snapshot(root: Path, inputs: Path) -> dict:
    # Inputs are generated privately by CMake. Only their digest leaves the build
    # tree: flags and compiler locations can contain machine-specific strings.
    settings = json.loads(inputs.read_text(encoding="utf-8"))
    configuration = {"inputs": sha256(inputs.read_bytes())}
    if settings.get("target_options"):
        configuration["target_options"] = sha256((inputs.parent / settings["target_options"]).read_bytes())
    compilers = {}
    for key in ("cxx", "c_compiler", "cuda_compiler", "hip_compiler", "toolchain_file"):
        compiler = settings.get(key)
        if compiler:
            compilers[key] = sha256(Path(compiler).read_bytes())
    cuda = None
    if settings["backend"] == "cuda" and settings.get("cuda_compiler"):
        try:
            version = subprocess.run([settings["cuda_compiler"], "--version"], capture_output=True, timeout=15)
            match = re.search(rb"\bV(\d+\.\d+\.\d+)", version.stdout) if version.returncode == 0 else None
            cuda = match.group(1).decode("ascii") if match else None
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {"schema_version": 1, "version": settings["version"], "backend": settings["backend"],
            "configuration": settings["configuration"], "architectures": settings["architectures"],
            **git_state(root), "cuda": cuda,
            "cuda_reason": None if cuda else ("not_cuda" if settings["backend"] != "cuda" else "version_unavailable"),
            "source_sha256": source_identity(root, Path(settings["ggml"]) if settings.get("ggml") else None),
            "config_sha256": sha256(json_bytes(configuration)), "toolchain_sha256": sha256(json_bytes(compilers))}


def atomic_files(files: dict[Path, bytes], executable=()) -> None:
    """Stage all writes, replace in order, restore replaced files on failure.

    Readers must validate the pair, including during replacement. Rollback
    failure raises explicitly; no caller may report successful publication.
    """
    staged, old, modes, times, replaced = {}, {}, {}, {}, []
    try:
        for path, data in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            previous = path.stat() if path.exists() else None
            old[path] = path.read_bytes() if previous else None
            modes[path] = previous.st_mode if previous else None
            times[path] = (previous.st_atime_ns, previous.st_mtime_ns) if previous else None
            if old[path] == data:
                continue
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".strata-build-", delete=False) as stream:
                staged[path] = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if modes[path] is not None:
                os.chmod(staged[path], modes[path])
            if path in executable:
                os.chmod(staged[path], staged[path].stat().st_mode | 0o111)
        for path, stage in staged.items():
            os.replace(stage, path)
            replaced.append(path)
    except BaseException:
        for path in reversed(replaced):
            if old[path] is None:
                path.unlink(missing_ok=True)
            else:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".strata-restore-", delete=False) as stream:
                    restore = Path(stream.name)
                    stream.write(old[path])
                try:
                    os.chmod(restore, modes[path])
                    os.utime(restore, ns=times[path])
                    os.replace(restore, path)
                finally:
                    restore.unlink(missing_ok=True)
        raise
    finally:
        for stage in staged.values():
            stage.unlink(missing_ok=True)


def prepare(root: Path, inputs: Path, stage: Path) -> dict:
    current = snapshot(root, inputs)
    record_file = stage / "record.json"
    previous = json.loads(record_file.read_bytes()) if record_file.exists() else {}
    if {k: v for k, v in previous.items() if k != "built_at"} == current:
        record = previous
    else:
        record = dict(current, built_at=datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"))
    template = (root / "include/strata/build_info.hpp.in").read_text(encoding="utf-8")
    header = template.replace("@STRATA_BUILD_JSON@", json.dumps(json_bytes(record).decode("ascii")))
    atomic_files({record_file: json_bytes(record) + b"\n", stage / "strata/build_info.hpp": header.encode("utf-8")})
    destination = stage / "executable.txt"
    if destination.exists():
        executable = Path(destination.read_text(encoding="utf-8"))
        if executable.is_file():
            # Validate/repair the sidecar on a no-op named-target build only
            # when the linked bytes already contain these exact prepared facts.
            try:
                linked = embedded_record(executable.read_bytes())
            except ValueError:
                linked = None
            if linked == record:
                publish(root, inputs, stage, executable)
    return record


def embedded_record(payload: bytes) -> dict:
    records = []
    for match in re.finditer(re.escape(MARKER) + rb"([^\n\0]{1,16384})\n", payload):
        try:
            record = json.loads(match.group(1))
        except (ValueError, UnicodeError):
            continue
        if isinstance(record, dict) and record.get("schema_version") == 1:
            records.append(record)
    if len(records) != 1:
        raise ValueError("Executable must contain exactly one STRATA_BUILD/1 record.")
    return records[0]


def validate_metadata(payload: bytes, metadata: dict, name: str | None = None) -> dict:
    if metadata.get("sha256") != sha256(payload):
        raise ValueError("Build receipt does not match the compiled executable SHA-256.")
    record = embedded_record(payload)
    if not {"version", "backend", "commit", "dirty", "built_at", "cuda"} <= record.keys():
        raise ValueError("Embedded build record is incomplete.")
    if metadata.get("build") != record or (name is not None and metadata.get("executable") != name):
        raise ValueError("Build receipt does not match the embedded build record or executable name.")
    for key in ("version", "commit", "dirty", "built_at", "cuda", "backend"):
        if key in metadata and metadata[key] != record[key]:
            raise ValueError("Build receipt has conflicting " + key + ".")
    return record


def metadata_for(payload: bytes, name: str, extra: dict | None = None) -> dict:
    record = embedded_record(payload)
    return {**(extra or {}), **{k: record[k] for k in ("version", "commit", "dirty", "built_at", "cuda", "backend")},
            "build": record, "executable": name, "sha256": sha256(payload)}


def publish(root: Path, inputs: Path, stage: Path, executable: Path) -> None:
    record = json.loads((stage / "record.json").read_bytes())
    if {k: v for k, v in record.items() if k != "built_at"} != snapshot(root, inputs):
        raise ValueError("Build inputs changed during compilation; refusing build receipt publication. Build again.")
    payload = executable.read_bytes()
    if embedded_record(payload) != record:
        raise ValueError("Linked executable does not match prepared build inputs; refusing publication.")
    receipt = executable.parent / "BUILD.json"
    old = json.loads(receipt.read_bytes()) if receipt.exists() else {}
    if old and (not isinstance(old.get("build"), dict) or old.get("executable") != executable.name):
        raise ValueError("Refusing to overwrite an unrelated BUILD.json manifest.")
    metadata = metadata_for(payload, executable.name, old)
    # Sample after reading the linked executable too; facts belong to this link.
    if {k: v for k, v in record.items() if k != "built_at"} != snapshot(root, inputs):
        raise ValueError("Build inputs changed before publication; refusing build receipt publication.")
    atomic_files({receipt: json_bytes(metadata) + b"\n"})


def copy_engine(executable: Path, target: Path, receipt: Path, extra: dict | None = None) -> dict:
    """Copy a validated pair without sampling the later working tree."""
    payload = executable.read_bytes()
    metadata = json.loads((executable.parent / "BUILD.json").read_bytes())
    validate_metadata(payload, metadata, executable.name)
    if receipt.exists():
        old = json.loads(receipt.read_bytes())
        if (old.get("executable") not in (None, target.name) or
                (old.get("executable") is None and receipt.name == "BUILD.json" and
                 target.name not in ("strata", "strata.exe"))):
            raise ValueError("Refusing to overwrite another executable's manifest.")
    result = metadata_for(payload, target.name, {**metadata, **(extra or {})})
    atomic_files({target: payload, receipt: json_bytes(result) + b"\n"}, executable=(target,))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.inputs, args.stage)
    else:
        if args.executable is None:
            parser.error("publish requires --executable")
        publish(args.root, args.inputs, args.stage, args.executable)


if __name__ == "__main__":
    main()
