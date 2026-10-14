# Compact recent-cache desktop experiment

RTX 4080 SUPER 16 GB, Ryzen 7 9800X3D, 32 GB installed RAM (30.88 GiB usable),
Windows. Codex remained open. Same Unsloth UD-Q2_K_XL weights, 131072-token
capacity, 8 GiB pinned profile cache and 4096 MiB requested recent-file cache.
GPU allocation remained 4,990 slots / 8,922 MiB. Normal runs disable both profiling
switches. Temperature zero, thinking disabled, same warmup and three short
technical prompts repeated twice, each capped at 160 output tokens.

## First complete timed comparison

| Run | Decode tokens | Decode seconds | Weighted tokens/sec | Minimum available RAM |
| --- | ---: | ---: | ---: | ---: |
| baseline-normal-4 | 960 | 89.8449 | 10.6851 | 3.4706 GiB |
| compact-normal | 960 | 81.8894 | 11.7231 | 3.0271 GiB |

All six choices match exactly. The compact run is 9.71% faster in this pair.
Logical decode-interval file bytes fell from 42,793.0 MB to 39,606.0 MB (7.45%).
These are API file_mb counters, excluding prior prefill; they are not measured
physical SSD traffic. Actual startup recent targets were 4094.4 MiB for largest
slots and 4096.0 MiB for compact buffers.

The compact timed phase completed before a helper logging error stopped its
subsequent smoke checks: BoundedCapture lacked flush(). The engine responses and
matched timing records were already saved. The helper has been repaired and a
second complete compact run with smoke checks is in progress. Do not treat the
first run's overall failed status as a successful smoke result.

One sequential baseline and two compact runs cannot establish a stable general
speedup. Run order, OS cache warmth, competing applications and workload can
change results. These are short prompts at configured 128K capacity, not a
nearly-full-context performance test. The larger diagnostic matrix in PLAN.md
was not completed. No claim that this desktop reaches the laptop's observed
17 tokens/sec follows from these numbers.

## Reproduce after closing Codex

Close any running Strata server first. From C:\Users\Joes\Tools\Strata, run each
command separately. The output directory must be new:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_compact_cache.py --config strata-unsloth-fast-128k.json --output build-unsloth\closed-app-baseline
.\.venv\Scripts\python.exe tools\benchmark_compact_cache.py --config strata-unsloth-compact-128k.json --output build-unsloth\closed-app-compact --smoke
```

The helper loads one authenticated server on loopback port 8081 and stops only
its own process tree. It aborts if available RAM remains below 2 GiB for five
samples. Normal daily use is run-unsloth-compact-128k.bat (port 8080 and existing
LAN binding/key); rollback is run-unsloth-fast-128k.bat.

## Evidence and excluded attempts

comparison-first-pair.json contains matched rows and exact calculations.
Each run folder retains results.json and bounded engine/server logs. attempts.json
identifies infrastructure failures and interrupted runs. The stale running marker
in baseline-normal-3 was left untouched; its process was gone before the accepted
baseline. The orphaned temporary credential config was removed after verifying
its exact generated path and loopback test settings.

Engine build, mixed-layout tests, independent native review, 12 deterministic
allocation-failure cases, deployment hash checks and helper tests are recorded
under .claude/plan-runs/unsloth-compact-cache-20261009. The original fast launcher
and binary remain byte-identical, and the Git index remains unchanged.
