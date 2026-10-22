"""Compare local desktop profiles after closing other apps; --plan starts nothing."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
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


def comparison_exclusion(result, context=None):
    """Apply the shared environment boundary and this report's completion checks."""
    if (type(context) is not tuple or len(context) != 2
            or any(type(value) is not dict for value in context)):
        return "independent launch context unavailable"
    if type(context[0].get("env", {})) is not dict:
        return "independent launch context malformed"
    try:
        classification = benchmark.consume_normal_comparison(result, *context, server_log=True)
    except (AttributeError, KeyError, TypeError, ValueError):
        return "independent launch context malformed"
    if classification["complete"] is not True or classification["eligible"] is not True:
        return "launch environment classification unavailable or excluded"
    if (result.get("status") != "complete" or result.get("timing_diagnostic") is not False
            or any(result.get(key, False) is not False for key in (
                "decode_timing", "fetch_timing", "smoke_only", "validation_only"))
            or any(key in result for key in ("error", "cleanup_error", "artifact_error"))
            or ("smoke_passed" in result and result["smoke_passed"] is not True)):
        return "run incomplete, failed or diagnostic"
    rows, summary = result.get("rows"), result.get("summary")
    if (type(rows) is not list or not rows or type(summary) is not dict
            or not benchmark.valid_number(summary.get("tokens_per_second"))
            or summary["tokens_per_second"] <= 0
            or type(summary.get("completion_tokens")) is not int or summary["completion_tokens"] <= 0
            or any(type(row) is not dict or type(row.get("response")) is not dict
                   or type(row["response"].get("choices")) is not list or not row["response"]["choices"]
                   for row in rows)):
        return "complete timed results unavailable"
    # Ordinary responses carry assistant text and a terminal stop/length reason.
    # Incomplete historical shapes remain raw evidence, not answer baselines.
    for row in rows:
        for choice in row["response"]["choices"]:
            if (type(choice) is not dict or type(choice.get("message")) is not dict
                    or choice["message"].get("role") != "assistant"
                    or type(choice["message"].get("content")) is not str
                    or choice.get("finish_reason") not in ("stop", "length")):
                return "complete assistant responses unavailable"
    return None


def report(records, *, contexts=None):
    """Contexts are private, independently captured (clean config, inherited env) pairs by label."""
    lines = ["# Desktop profile comparison", "",
             "Short prompts at 131,072 context capacity; this does not test a nearly full context.",
             "GPU placement can change output text. A text match is not a model-quality evaluation.", "",
             "Environment eligibility covers only the source-classified launch environment boundary.",
             "It does not establish matching build, model, workload, allocation, cohort or performance, or hardware admission.",
             "Check those independently before drawing performance conclusions.", "",
             "## Normal comparison candidates", "",
             "| Run | Decode tokens/s | Timed tokens | Exact answers vs first eligible run | Smoke checks |",
             "| --- | ---: | ---: | --- | --- |"]
    baseline, excluded = None, []
    for label, result in records:
        context = contexts.get(label) if type(contexts) is dict else None
        reason = comparison_exclusion(result, context)
        if reason:
            excluded.append((label, reason))
            continue
        rows, summary = result["rows"], result["summary"]
        if baseline is None:
            baseline = rows
        matches = sum(a["response"]["choices"] == b["response"]["choices"]
                      for a, b in zip(baseline, rows))
        smoke = "passed" if result.get("smoke_passed") else "not requested"
        lines.append(f"| {label} | {summary['tokens_per_second']:.2f} | "
                     f"{summary['completion_tokens']} | {matches}/{len(baseline)} | {smoke} |")
    if baseline is None:
        lines += ["", "No eligible complete runs are available for normal comparison."]
    if excluded:
        lines += ["", "## Excluded runs", "", "| Run | Exclusion |", "| --- | --- |"]
        lines += [f"| {label} | {reason}; inspect raw results |" for label, reason in excluded]
    lines += ["", "Each run has results.json, engine.log and server.log with memory and cache evidence.",
              "The repeated compact run helps detect warm-up or background drift; repeat comparisons before choosing a winner.", ""]
    lines += ["Historical bench/results/desktop-fetch-20261010/summarize.py is retained evidence and is unsupported for new normal-comparison conclusions.", ""]
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
        records, contexts = [], {}
        for label, filename, smoke in CASES:
            # Keep independent inputs in memory; never recover them from a returned record.
            cfg = json.loads((ROOT / filename).read_text(encoding="utf-8-sig"))
            contexts[label] = benchmark.clean_environment(cfg, False, inherited=dict(os.environ))
            result = benchmark.run(argparse.Namespace(config=ROOT / filename, output=output / label,
                                   timing=False, smoke=smoke, smoke_only=False))
            records.append((label, result))
            (output / "README.md").write_text(report(records, contexts=contexts), encoding="utf-8")
            reason = comparison_exclusion(result, contexts[label])
            if reason:
                print(f"{label}: excluded from normal comparison ({reason}); inspect raw results", flush=True)
            else:
                print(f"{label}: {result['summary']['tokens_per_second']:.2f} tokens/s (environment-eligible candidate)", flush=True)
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
