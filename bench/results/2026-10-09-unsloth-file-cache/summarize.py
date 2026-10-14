"""Summarize saved measurements without loading the model or any credentials."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main():
    summary = {}
    records = {}
    for path in sorted(OUT.glob("*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not data.get("rows") or "ram_gib" not in data:
            continue
        rows = data["rows"]
        timings = [r["response"]["timings"] for r in rows]
        summary[path.stem] = {
            "aggregate_tok_s": sum(t["predicted_n"] for t in timings)
            / (sum(t["predicted_ms"] for t in timings) / 1000),
            "total_file_mb": sum(r["metric"]["file_mb"] for r in rows),
            "responses": len(rows),
            "expert_slots": data["engine"]["expert_slots"],
            "memory": data.get("memory"),
        }
        records[path.stem] = data
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    baseline = records.get("fixed-baseline")
    candidate = records.get("fixed-recent-4g")
    if baseline and candidate:
        comparisons = [a["response"]["choices"][0]["message"] == b["response"]["choices"][0]["message"]
                       for a, b in zip(baseline["rows"], candidate["rows"])]
        original, new = summary["fixed-baseline"], summary["fixed-recent-4g"]
        comparison = {
            "baseline": "fixed-baseline", "candidate": "fixed-recent-4g",
            "equal_response_count": len(baseline["rows"]) == len(candidate["rows"]),
            "identical_messages": sum(comparisons),
            "generation_gain_percent": 100 * (new["aggregate_tok_s"] / original["aggregate_tok_s"] - 1),
            "file_byte_reduction_percent": 100 * (1 - new["total_file_mb"] / original["total_file_mb"]),
        }
        (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        larger = records.get("fixed-baseline-12g")
        if larger:
            original = summary["fixed-baseline-12g"]
            matches = [a["response"]["choices"][0]["message"] == b["response"]["choices"][0]["message"]
                       for a, b in zip(larger["rows"], candidate["rows"])]
            comparison["larger_original_cache"] = {
                "baseline": "fixed-baseline-12g",
                "equal_response_count": len(larger["rows"]) == len(candidate["rows"]),
                "identical_messages": sum(matches),
                "generation_gain_percent": 100 * (new["aggregate_tok_s"] / original["aggregate_tok_s"] - 1),
                "file_byte_reduction_percent": 100 * (1 - new["total_file_mb"] / original["total_file_mb"]),
            }
            (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    monitor = OUT / "fixed-recent-4g-memory.jsonl"
    if monitor.exists():
        samples = [json.loads(line) for line in monitor.read_text().splitlines()]
        processes = [p for sample in samples for p in sample["processes"]
                     if p["name"] == "strata-unsloth-file-cache.exe"]
        memory = {
            "samples": len(samples), "sample_interval_seconds": 10,
            "coverage": "Started during candidate benchmark; includes all smoke and long-context validation",
            "minimum_available_gib": min(s["memory"]["available"] for s in samples) / 2**30,
            "peak_system_commit_gib": max(s["commit_bytes"] for s in samples) / 2**30,
            "system_commit_limit_gib": min(s["commit_limit_bytes"] for s in samples) / 2**30,
            "peak_pagefile_used_gib": max(s["pagefile"]["used"] for s in samples) / 2**30,
            "peak_engine_private_commit_gib": max(p["windows_memory"]["private_commit_bytes"] for p in processes) / 2**30,
            "peak_engine_working_set_gib": max(p["windows_memory"]["working_set_bytes"] for p in processes) / 2**30,
        }
        (OUT / "memory-summary.json").write_text(json.dumps(memory, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
