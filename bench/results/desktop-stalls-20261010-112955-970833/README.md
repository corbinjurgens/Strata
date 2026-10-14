# Desktop stall diagnostics

Diagnostic timings only; same short workloads at 128K capacity, with a four-minute idle gap.
Each results.json records request/idle Unix timestamps. Match them to telemetry JSONL timestamps.
engine.log has CPU decode timing, cache/file counters, and async stats where enabled.
Commit is system commit, not pagefile usage. Disk counters are cumulative and system-wide.
GPU polling is sampled, not a per-kernel trace. Missing optional sensors are null plus an error type.
Async ms/round is cumulative elapsed time and can include idle gaps; it is not per-request stall time.
No nearly-full-context test, output-quality claim or automatic winning profile selection.
