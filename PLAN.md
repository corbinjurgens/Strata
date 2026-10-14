# Execution plan: profile the Unsloth desktop decode path

Status, 2026-10-10: the user subsequently authorized implementing compact recent
expert buffers. That implementation and its bounded acceptance run are tracked
in `.claude/plan-runs/unsloth-compact-cache-20261009/plan.md`. This original plan
remains the record of the broader profiling matrix; it has not all been executed.

## Goal

Produce reproducible evidence of where the current Unsloth UD-Q2_K_XL candidate
spends time on the RTX 4080 SUPER desktop, preserving 131,072-token capacity.
Explain ordinary generation and the slow tool-result case sufficiently to name
the next experiment, including an explicit unknown when existing timers cannot
answer it. This plan measures the current engine; it does not implement a new
cache policy or promise the laptop's observed 17 tokens/s.

## Context the executor needs

- Repository: `C:/Users/Joes/Tools/Strata`; Windows, PowerShell, Ryzen 7 9800X3D,
  RTX 4080 SUPER 16 GiB, 30.88 GiB usable RAM, Crucial P3 NVMe, 24 GiB pagefile.
- Branch `codex/unsloth-file-cache`, HEAD
  `6794d901f81717ed9c5b5513c83c0eba1187f068`. Existing recent-cache edits in
  `include/strata/core/expert_source.hpp`, `src/core/expert_source.cpp`,
  `src/program/generate.cpp` and `tests/core/file_expert_source_test.cpp`, their
  documentation and results are uncommitted and mostly staged. Preserve the
  index and unrelated Unsloth preparation files. A previous commit was blocked
  by missing Git author identity; do not invent an identity or change Git config.
- Local ignored config `strata-unsloth-fast-128k.json` supplies the API key,
  tokenizer and pack paths. Read credentials only at runtime; never print them,
  pass them in command-line arguments, or put config copies in tracked results.
  Its executable is `engine/strata-unsloth-file-cache.exe`, SHA256
  `204674c9ba60a8b8f4e0e5096c5e3004de06cb1a7dcdbed634bbf440f2d983f2`.
- Preserve 8 GiB resident profile cache, 4096 MiB recent target,
  `--expert-cache 4300` (previously 4,990 actual slots), max context 131072,
  int8 KV, GPU-resident KV 32768, spec 2, no MTP, suffix draft 0, lookup chain 0,
  parallel 1. Verify effective settings from health, metrics and startup logs.
- Model: `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
  `766911a6b7369840a91dbcd95f9f997acaab6cd6`, UD-Q2_K_XL. Cache directory:
  `C:/Users/Joes/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q2_K_XL`.
- Prior harness and reusable workloads:
  `bench/results/2026-10-09-unsloth-file-cache/benchmark.py` (`PROMPTS`),
  `validation.py` (`Client`, `smoke`, `long_test`). Their raw JSON and README
  establish controlled aggregate 11.19 tokens/s for this candidate, six passing
  smoke checks, and a passing 129,023-token retrieval plus cached continuation.
  The full-context test took about 16.5 minutes. The reported 13 desktop and 17
  laptop tokens/s are unmatched observations.
- Verified findings are in
  `.claude/research-prompts/2026-10-09-strata-unsloth-low-ram-speed/verification.md`.
  Native GPU plan publication already precedes CPU file prefetch.
  API `file_mb` is logical payload during the decode reporting interval, not
  physical SSD traffic; an earlier prefill-inclusive description was corrected.
  API `predicted_ms` excludes prompt processing, final commit wait and later
  bookkeeping. Measure client wall time and streaming TTFT separately.
- Existing `STRATA_DECODE_TIMING=1` prints CPU/host stages per request.
  `STRATA_VERIFY_PROFILE=1` supplies GPU stages, but also disables shared-expert
  stream overlap in `src/core/verify.cpp`. GPU-profile runs diagnose that altered
  schedule and cannot serve as normal throughput evidence.

## Decisions (settled — do not revisit)

- Keep the same weights, quantization, context capacity, allocation and executable.
- Preserve the working launcher and any existing server. Never kill a process
  discovered by port number or name; only terminate descendants this run owns.
- Keep diagnostic servers on `127.0.0.1`, authenticated with the existing key.
- Use existing timers first. New engine counters and optimization patches belong
  to later work after this report identifies the missing evidence.
- Keep the historical research response unedited. Use the verification note's
  corrections when interpreting it.

## Decisions made while writing this plan

- Use three modes: `normal` removes both profiling variables; `cpu` enables only
  `STRATA_DECODE_TIMING=1`; `gpu` enables both. Remove inherited values from both
  the parent environment and copied config before applying the mode.
- Use three alternating normal/CPU short-suite pairs, then one GPU-profile short
  suite. Run one normal and one CPU full-context suite separately. GPU profiling
  at full context is outside this first diagnostic pass.
- Use port 8081 and refuse to start if any Strata engine/server is already running.
  This avoids loading two copies into limited RAM. A running user server is a
  scheduling blocker; prepare and test the harness before asking the user to
  close it.
- Store results in the first unused directory matching
  `bench/results/unsloth-decode-profile-YYYYMMDD-HHMMSS[-N]/`. Keep scratch configs
  with credentials under ignored `build-unsloth/` and remove only owned scratch
  files after process shutdown.

## Tasks

### Task 1: Create a repeatable diagnostic runner

**Files:** `tools/profile_unsloth_decode.py`, `tools/test_profile_unsloth_decode.py`, `docs/UNSLOTH_DECODE_PROFILE.md`.

**Depends on:** none

**Complexity:** judgment

**Do:** Implement CLI `--mode normal|cpu|gpu --suite short|long --output DIR`
using `.venv/Scripts/python.exe`. Default config is the ignored fast config
named above. Validate paths and executable hash, reject output collisions,
reject existing Strata processes and an occupied port, then launch a hidden
child server with a temporary config and loopback port 8081. Poll readiness
with a deadline, record sanitized effective settings and fail before the
workload on allocation/context mismatch. Reuse the prior `PROMPTS` and
validation functions through explicit path imports; do not edit their files.
Send the same 64-token warmup, then the three 160-token technical requests
twice for the short suite, followed by the entire smoke sequence in its
original order, including the generated tool call and its result. Preserve
response text and actual rendered tool-call arguments so differences remain
visible. The long suite runs the warmup, then `long_test(client, 129024)` and
its continuation. Override the imported client's URL/output destination;
it must never send to port 8080. Use temperature zero and thinking disabled.
Stream measured requests with usage reporting; retain wall time, first
content/tool delta time, API timings and request-ID-matched metrics. Do not
assume the newest metric is necessarily the matching request. Retain raw
stderr with diagnostic lines and mode/request associations. Sample available
RAM, commit, pagefile, process memory and GPU memory at one-second intervals.
Save partial results on failure and stop only the owned process tree in a
finally block. Document exact runnable commands. Unit tests use a fake
server/process layer to cover refusal, all three mode environments, exact
request association, streaming final usage, redaction, and owned cleanup.

**Verify:** Run `.venv/Scripts/python.exe tools/test_profile_unsloth_decode.py`
and `.venv/Scripts/python.exe tools/profile_unsloth_decode.py --help`.
Both exit zero; tests cover the behaviors above without GPU use. Review all
serialization/logging paths to confirm the secret cannot enter results.

### Task 2: Capture the diagnostic sessions

**Files:** The new result directory selected by the naming rule above, including `manifest.json`, per-run raw responses/metrics/logs and memory samples.

**Depends on:** 1 — uses the runner

**Complexity:** judgment

**Do:** Read current process status. If a user server exists, report that it
must be closed before proceeding; leave it intact and resume after the user
does so. Record other heavy applications and keep their state constant
throughout the short paired runs. Do not close user applications yourself.
Record the candidate executable hash and effective configuration. Run short
suites in order normal-1, cpu-1, normal-2, cpu-2, normal-3, cpu-3, gpu-1,
each in a fresh server with the same warmup. Run long-normal-1 followed by
long-cpu-1. Use the runner's CLI and a distinct subdirectory for every run.
Record exclusions and preserve interrupted results. If available RAM falls
below 2 GiB for five consecutive samples, stop the owned run and report
memory pressure; do not automatically change cache sizes. If a correctness
check fails, stop further runs and investigate the failure before timing
conclusions. Verify keys are absent from logs before retaining them. Save a
sanitized manifest mapping every session to its mode, suite, timestamps,
executable hash, settings, status and artifact paths.

**Verify:** Every planned session has complete saved responses, matching
metrics and expected timing lines for its mode. All smoke and retrieval
checks pass, long local prompt counts are recorded near 129K with 131072
capacity, and no sampled allocation differs between comparable sessions.
Any unmet check marks the capture incomplete with the exact reason rather
than silently omitting the session. No owned server remains after completion.

### Task 3: Produce the decision evidence

**Files:** `tools/summarize_unsloth_decode_profile.py`, `docs/UNSLOTH_DECODE_PROFILE.md`, the task 2 result directory's `summary.json` and `README.md`.

**Depends on:** 2 — requires captures; also follows task 1's documentation edit

**Complexity:** judgment

**Do:** Implement a summary command accepting the result directory. Report
per-request and aggregate decode rates, TTFT, wall time, completion counts,
cache reuse, actual context occupancy and memory minima. Keep technical,
smoke/tool and full-context workloads separate. Compare CPU-timing mode to
normal mode across the three paired sessions and state the observed
variability/perturbation. Parse jobs, CPU, plan, activation, GPU-reach wait,
stage, commit/emit and draft fields as defined in the source; do not add
overlapping CPU/GPU work as elapsed time. GPU-profile measurements describe
an altered overlap schedule. Compare the entire reproduced tool sequence
with the prior outlier and report whether it recurs. Name one next measured
bottleneck and the evidence needed for its experiment. If jobs/file wait
cannot be separated, explicitly request true demand misses, Windows
submitted/completed bytes and fill-wait timing in a later instrumentation
plan; do not infer SSD bandwidth from logical bytes. If CPU compute
dominates instead, state why cache/merge tuning is not yet justified. Keep
compact cache, merge-gap and admission changes as conditional hypotheses.
Link this result from the profile guide. Do not implement the next patch.

**Verify:** Run `.venv/Scripts/python.exe tools/summarize_unsloth_decode_profile.py RESULT_DIR`.
It exits zero and reports all manifest sessions, with explicit incomplete
status for missing data. Independently recalculate one technical aggregate
as total engine-generated tokens / summed decode seconds and compare within
rounding tolerance. Cross-check one CPU timing row against its raw log and
one long-context row against raw usage. The report states profiler effects,
output differences, memory pressure and every unresolved attribution.

## Verification (whole-plan)

The runner tests and summary consistency checks are locally runnable. Performance
captures require an idle GPU and the downloaded model; they were not run while
writing this plan. The final report must contain normal-runtime measurements,
separate altered-schedule GPU diagnostics, correctness results, and a bounded
next question. A speed improvement is not an acceptance criterion for diagnostics.

## Do not

- Do not refactor beyond what the tasks specify.
- Do not edit engine code, replace binaries, reduce context, change quantization,
  increase RAM budgets, download weights, create sidecars, or tune policies here.
- Do not overwrite existing results, stage unrelated files, change Git identity,
  discard the existing index, commit prior edits, or push.
- Do not expose secrets or change LAN authentication. Do not stop a user server.
- Do not claim physical disk throughput from `file_mb`, full-context speed from
  a configured limit, or a desktop/laptop comparison without matched workloads.

## If stuck

For import/API differences, adapt only the new runner to the locally installed
interfaces and re-run its tests. On a changed executable hash or changed settings,
stop and identify the mismatch before measurement. On a runtime error preserve
partial logs and stop only owned processes. An unexplained timing stage is a
valid reported unknown; do not turn this plan into unlimited tracing work.

## Note lifecycle

This file is working state in a repository, not documentation.

- When an item lands, append ` — done <sha>` to its heading, where
  `<sha>` is the commit that landed it. An item consciously dropped gets
  ` — dropped` and a `project-note deferred` entry.
- When every item carries one of those marks, delete this file in the
  commit that lands the last item. Nothing in it is kept: the commits and
  the run archive are the record, and `project-note done` records the
  completion.
- In an execute run the orchestrator does this after each unit lands,
  not the unit. A plain session does it as each item lands.
