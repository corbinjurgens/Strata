# Partial diagnostic interpretation

2026-10-10, Japan time. The user reported uncertainty about whether another
Strata launcher was needed and reconnecting through Parsec. The timing of that
interaction is not independently established by these samples.

The desktop baseline loaded correctly, completed warm-up and three timed
responses, and entered the deliberate four-minute idle period. The saved run
is incomplete: `status=failed`, three timed rows, no second round, no async run.
The old error string is empty, so the exact exception cannot be recovered from
this file. The memory guard has no stop reason and did not trigger.

- Warm-up: 14.9 tokens/s, 64 output tokens.
- Timed cases: 19.2, 23.0 and 17.7 tokens/s, each 160 output tokens.
- Idle began at 11:31:09; its planned end was 11:35:09.
- Result/log files were finalized at 11:34:57, about 12 seconds before that end.
- Telemetry contains 60 samples, through 11:34:54.
- During requests, sampled available RAM was 3.10-7.50 GiB. Over the whole
  guarded run the minimum was 2.71 GiB; the last idle samples show a modest
  increase in memory/GPU use, with no evidence identifying which application.
- During requests, GPU samples showed 96-100% utilization, 2,715-2,775 MHz SM
  clocks and 28-33 C. These samples do not show a clock/temperature collapse
  during the captured normal responses. Utilization alone does not establish
  productive kernel occupancy or the bottleneck.

The three timed traces attribute 25.43-38.16 ms per one-token window to the
`jobs` section, versus 2.94-3.22 ms to the CPU execution section. Source
`src/core/expert_source.cpp` measures `jobs` across expert prefetch, blob
retrieval and job construction, before invoking the CPU pool. This makes
fetching/preparing expert data a useful optimization lead. It is not a direct
SSD latency measurement or evidence explaining the earlier 4.7/8.4 tok/s stalls.

Useful baseline data was saved, but neither post-idle behavior nor the async
comparison was captured. Rerun the diagnostic alone and let it finish. It
starts its own authenticated server on localhost port 8081, sends the fixed
requests, intentionally idles, then stops that server and runs the second
profile. The usual port 8080 is not served by this test.

The launcher now explains that sequence and prints FINISHED/STOPPED. Idle
countdown messages appear every 30 seconds. Future failure records include
the exception type even when the exception has no message. Existing results
were preserved. The 28 offline helper/diagnostic tests passed after these
clarifications; no new model test was started.
