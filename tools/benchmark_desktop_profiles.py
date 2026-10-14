"""Compare local desktop profiles after closing other apps; --plan starts nothing."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import benchmark_compact_cache as benchmark
import prepare_unsloth_q2 as preparation

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("compact-before", "strata-unsloth-compact-128k.json", False),
    ("desktop", "strata-unsloth-desktop-128k.json", True),
    ("desktop-gpu", "strata-unsloth-desktop-gpu-128k.json", True),
    ("compact-after", "strata-unsloth-compact-128k.json", False),
)


def report(records):
    lines = ["# Desktop profile comparison", "",
             "Short prompts at 131,072 context capacity; this does not test a nearly full context.",
             "GPU placement can change output text. A text match is not a model-quality evaluation.", "",
             "| Run | Decode tokens/s | Timed tokens | Exact answers vs first run | Smoke checks |",
             "| --- | ---: | ---: | --- | --- |"]
    baseline = records[0][1]["rows"] if records else []
    for label, result in records:
        rows, summary = result["rows"], result["summary"]
        matches = sum(a["response"]["choices"] == b["response"]["choices"]
                      for a, b in zip(baseline, rows))
        smoke = "passed" if result.get("smoke_passed") else "not requested"
        lines.append(f"| {label} | {summary['tokens_per_second']:.2f} | "
                     f"{summary['completion_tokens']} | {matches}/{len(baseline)} | {smoke} |")
    lines += ["", "Each run has results.json, engine.log and server.log with memory and cache evidence.",
              "The repeated compact run helps detect warm-up or background drift; repeat comparisons before choosing a winner.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Check files and print run order without starting a model")
    args = parser.parse_args(argv)
    output = None
    try:
        for label, filename, smoke in CASES:
            cfg = json.loads((ROOT / filename).read_text(encoding="utf-8-sig"))
            preparation.check_ready(cfg)
            # A saved learned profile would change the baseline between starts.
            if cfg.get("expert_profile_save") or "--expert-profile-save" in cfg["args"]:
                raise ValueError("Disable learned-profile saving before this comparison.")
            print(f"{label}: {filename}" + (" + smoke checks" if smoke else ""), flush=True)
        if args.plan:
            print("Readiness passed. No model was started.")
            return 0
        benchmark.assert_idle()
        output = ROOT / "bench/results" / ("desktop-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        output.mkdir(parents=True, exist_ok=False)
        print(f"Results: {output}", flush=True)
        print("Four sequential runs; close Codex and other apps before using this test.", flush=True)
        records = []
        for label, filename, smoke in CASES:
            result = benchmark.run(argparse.Namespace(config=ROOT / filename, output=output / label,
                                   timing=False, smoke=smoke, smoke_only=False))
            records.append((label, result))
            (output / "README.md").write_text(report(records), encoding="utf-8")
            print(f"{label}: {result['summary']['tokens_per_second']:.2f} tokens/s", flush=True)
        print(f"Finished. Comparison: {output / 'README.md'}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        # The individual runner redacts credentials in its detailed records.
        print(f"Comparison stopped ({type(exc).__name__}).", file=sys.stderr)
        if output:
            print(f"Inspect results in {output}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
