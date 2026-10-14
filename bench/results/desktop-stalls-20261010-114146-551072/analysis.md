# Completed desktop stall diagnostic - 2026-10-10

Both profiles completed all six timed requests and their four-minute idle
periods. Keep `run-unsloth-desktop-128k.bat` for normal use. Async adaptation
showed no overall benefit in this run. There is no need to repeat this same
diagnostic immediately.

## Throughput

Rates below are output tokens divided by total decode time, excluding the
64-token warm-up and prompt processing. Each timed response produced 160
tokens; each profile produced 960 timed tokens.

| Profile | Before idle, tokens/s | After idle, tokens/s | Overall, tokens/s |
| --- | ---: | ---: | ---: |
| Desktop | 19.67 | 21.16 | 20.39 |
| Async | 20.15 | 19.60 | 19.87 |

Desktop individual rates: 19.3, 22.8, 17.6; then 22.2, 24.9, 17.6.
Async individual rates: 19.5, 26.9, 16.5; then 21.5, 19.9, 17.7.

Async was 2.5% slower overall in this pair. None of the six corresponding
output choices matched exactly across profiles, despite temperature zero.
Expert routing therefore differed. Auto-sized recent-cache budgets also
differed: 5,878.7 MiB desktop versus 6,102.4 MiB async. This is evidence
against selecting async as a demonstrated improvement, not a precise
same-output measurement of its isolated overhead or an answer-quality test.

Neither profile reproduced the earlier 4.7/8.4 tokens/s responses. The
original severe slowdown remains unexplained; this run does not establish
that it is fixed. A four-minute idle pause did not reliably trigger it.

## Where time went

Mean CPU-side timing across the six equal-length responses, in milliseconds
per one-token window:

| Counter | Desktop | Async |
| --- | ---: | ---: |
| Total wall time | 49.05 | 50.33 |
| Expert fetching and job preparation (`jobs`) | 31.67 | 34.71 |
| CPU expert execution (`CPU`) | 3.10 | 3.38 |
| GPU-reach wait | 9.11 | 9.13 |

The `jobs` counter covers `prefetch`, `blob` retrieval and job assembly in
`src/core/expert_source.cpp:2681-2739`; CPU execution is measured separately
through line 2768. Fetching/preparation accounts for about 65% of desktop
wall time. It includes more than physical disk reads: cache access, copies,
synchronization and bookkeeping can contribute. GPU-reach wait is a CPU
wait measurement, not a complete GPU kernel profile.

System-wide physical-disk counters also show substantial reads during the
active phases: approximately 418/445 MB/s before/after idle for desktop and
542/583 MB/s for async. These use consecutive telemetry samples wholly
inside each phase; they include prompt processing and other system activity,
and do not cover the exact request boundaries. Most reads were on
PhysicalDrive2. Decode-only logical file counters totalled 18,643.6 MB for
desktop and 27,916.7 MB for async; these are not physical SSD bytes.

The next useful engine investigation is to split the expert fetch path into
cache lookup, file read, blob assembly/copy and waiting time, then optimize
the measured dominant component. The current evidence does not justify
expecting a large gain from faster CPU expert arithmetic or another GPU-share
preset. No additional engine change or speedup is claimed here.

## Hardware and limits

- Minimum available RAM was 2.93 GiB desktop and 2.92 GiB async. Neither
  low-memory guard stopped the test. This run has little spare RAM for a
  substantially larger cache.
- Timed-request GPU samples used about 14,959 MiB VRAM. Typical active SM
  clocks were around 2,715 MHz; sampled temperatures stayed at or below
  34 C. No sustained clock or thermal collapse appeared in the samples.
  Some request-boundary samples still reflected idle clocks.
- Telemetry has 79 desktop and 70 async samples with no sensor errors.
  Sampling about every five seconds cannot exclude brief stalls or explain
  an earlier event that was not captured. GPU utilization alone does not
  establish that useful arithmetic saturated the GPU.
- Both used the same Unsloth UD-Q2_K_XL model and compact engine, 128K context
  capacity, int8 KV with 32,768 resident cells, 5,272 GPU expert slots and
  PCIe fraction 0.55. The intentional profile change was `--adapt-async 1`.
- Requests were short technical prompts, with thinking disabled and a
  160-token response cap. This does not reproduce every condition of the
  earlier reasoning-enabled requests, test a nearly full 128K context, or
  establish answer quality. CPU timing and hardware polling add overhead.

Sources: each profile's `results.json`, `engine.log`, and its corresponding
top-level `*-telemetry.jsonl`. Raw results are retained unchanged.
