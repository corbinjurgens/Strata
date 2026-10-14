"""Sequential, local-only A/B decode measurements; the existing engine and key stay in place."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

import psutil

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PROMPTS = [
    "Explain how a hash table handles collisions, resizing, and lookup. Give a detailed technical explanation.",
    "Explain how to design reliable retries for a background job queue, including idempotency and exponential backoff.",
    "Explain how database composite indexes affect filtering, sorting, and write performance, with concrete examples.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "strata-unsloth-ud-q2_k_xl-128k.json")
    parser.add_argument("--exe", default="engine/strata-unsloth.exe")
    parser.add_argument("--ram-gib", type=int, default=8)
    parser.add_argument("--cache-mib", type=int, default=0)
    parser.add_argument("--expert-cache", default="auto")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--validate", action="store_true", help="Also run the existing smoke and near-128K retrieval checks")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    cfg.update(exe=str(ROOT / args.exe), host="127.0.0.1", port=8080,
               log=str(ROOT / f"build-unsloth/{args.name}.engine.log"))
    cfg["args"][cfg["args"].index("--resident-budget-gib") + 1] = str(args.ram_gib)
    cfg["args"][cfg["args"].index("--expert-cache") + 1] = args.expert_cache
    cfg["env"]["STRATA_FILE_CACHE_MIB"] = str(args.cache_mib)
    # This is an equal-budget comparison, retaining at least 4 GiB at startup.
    cfg["env"]["STRATA_RESIDENT_HEADROOM_GIB"] = "4"
    cfgpath = ROOT / f"build-unsloth/{args.name}.json"
    cfgpath.write_text(json.dumps(cfg, indent=2))
    base = "http://127.0.0.1:8080"
    headers = {"Authorization": "Bearer " + cfg["api_key"], "Content-Type": "application/json"}

    def get(path):
        with OPENER.open(urllib.request.Request(base + path, headers=headers), timeout=10) as response:
            return json.load(response)

    try:
        get("/health")
    except OSError:
        pass
    else:
        raise SystemExit("A server is already using port 8080; no process was started or stopped.")

    samples, stop = [], threading.Event()
    def monitor():
        while not stop.wait(1):
            vm = psutil.virtual_memory()
            samples.append({"available": vm.available, "used": vm.used,
                            "pagefile_used": psutil.swap_memory().used})

    rows = []
    record = {"name": args.name, "exe": args.exe, "context": 131072, "ram_gib": args.ram_gib,
              "cache_mib": args.cache_mib, "expert_cache": args.expert_cache, "rows": rows}
    stdout = (ROOT / f"build-unsloth/{args.name}.server.log").open("w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata",
                             "--config", str(cfgpath), "--port", "8080"],
                            cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    def save():
        (OUT / f"{args.name}.json").write_text(json.dumps(record, indent=2) + "\n")
    def chat(messages, tokens):
        payload = {"model": cfg["model_name"], "messages": messages, "temperature": 0,
                   "max_tokens": tokens, "chat_template_kwargs": {"enable_thinking": False}}
        start = time.monotonic()
        with OPENER.open(urllib.request.Request(base + "/v1/chat/completions", headers=headers,
                                               data=json.dumps(payload).encode()), timeout=1800) as response:
            data = json.load(response)
        data["wall_s"] = time.monotonic() - start
        return data

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("Server exited during startup; inspect its log.")
            try:
                health = get("/health")
                if health.get("loaded"):
                    break
            except OSError:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("Server did not load within ten minutes.")
        record["health"] = health
        record["engine"] = get("/metrics")["engine"]
        print(args.name + ": loaded", flush=True)
        warm = chat([{"role": "user", "content": "Explain what a database transaction does in a few sentences."}], 64)
        record["warmup"] = warm
        save()
        for round_index in range(args.rounds):
            for index, prompt in enumerate(PROMPTS):
                messages = [{"role": "system", "content": f"Technical explanation case {index}."},
                            {"role": "user", "content": prompt}]
                response = chat(messages, 160)
                metric = get("/metrics")["requests"][0]
                rows.append({"round": round_index, "case": index, "prompt": prompt,
                             "response": response, "metric": metric})
                save()
                print(json.dumps({"name": args.name, "round": round_index, "case": index,
                                  "tokens": response["usage"]["completion_tokens"],
                                  "tok_s": response["timings"]["predicted_per_second"],
                                  "file_mb": metric["file_mb"]}), flush=True)
        record["metrics"] = {k: v for k, v in get("/metrics").items() if k in ("engine", "totals")}
        if args.validate:
            path = OUT / "validation.py"
            spec = importlib.util.spec_from_file_location("strata_validation", path)
            validation = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(validation)
            validation.OUT = OUT / (args.name + "-validation")
            validation.OUT.mkdir(exist_ok=True)
            client = validation.Client(cfg)
            record["smoke_passed"] = validation.smoke(client)
            save()
            record["long_passed"] = validation.long_test(client, 129024)
            save()
            if not record["smoke_passed"] or not record["long_passed"]:
                raise RuntimeError("Candidate failed model validation; inspect the saved results.")
    finally:
        stop.set()
        mon.join(timeout=3)
        if samples:
            record["memory"] = {"min_available_gib": min(x["available"] for x in samples) / 2**30,
                                "max_pagefile_used_gib": max(x["pagefile_used"] for x in samples) / 2**30}
        try:
            parent = psutil.Process(proc.pid)
            children = parent.children(recursive=True)
            # Only descendants of the server launched by this invocation are stopped.
            for child in reversed(children):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            parent.terminate()
            psutil.wait_procs([*children, parent], timeout=15)
        except psutil.NoSuchProcess:
            pass
        stdout.close()
        save()


if __name__ == "__main__":
    main()
