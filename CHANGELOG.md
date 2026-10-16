# Changelog

Keep measured results tied to their hardware, workload and evidence. Preserve dated entries;
record changes that have not been released under Unreleased.

## Unreleased

Shared development now uses `main`; each PC keeps its own engine build, models and
profiles. No release is claimed. The following desktop results describe the earlier
`codex/unsloth-file-cache` workload; the dated entries record subsequent changes.

- **What changed:** opt-in fetch diagnostics, a default-off Windows read-path experiment,
  hardware-aware UD-Q2_K_XL build/configuration/launch support, and guarded benchmark acceptance.
  A separate trained desktop placement preserves the shipped profile and rollback launcher.
- **What worked:** on the RTX 4080 SUPER / 32 GB Windows desktop, the technical workload measured
  26.84–27.07 tok/s with trained placement versus 18.41/18.34 with matched shipped placement;
  the unchanged daily baseline measured 19.81. A fresh 130,880-token prompt and full-history
  reuse passed at 131072 capacity. Final long/idle minimum available RAM was 4.03/3.96 GiB.
- **What did not or remains open:** no dependable unseen-workload gain; intermittent read stalls
  persist. Mapping retirement did not improve speed and remains off. Buffered loading triggered
  a RAM guard and is rejected. Both placements missed the same interval-union model question.
  The new laptop policy has simulated-hardware coverage but no laptop runtime validation.
- **Verification:** native engine build and file-expert fixture; 120 combined Python tests,
  eight public-probe checks, independent native Codex review, authenticated actual launcher
  probes, six smoke checks, near-limit/reuse and repeated throughput runs.

See the [measurement report and reproduction checklist](bench/results/desktop-fetch-20261010/analysis.md)
and [hardware-aware setup instructions](docs/UNSLOTH_Q2.md#hardware-aware-preparation-for-both-pcs).

## 2026-10-14 — Shared memory safety (unreleased)

Implementation range: `89433d16ab0754939e2bcce16bc60c2caadea168..e4e7567a3ce2305caa727d1f7a26c18e1e1e06d5`.

- **What changed:** the recent expert cache shares the active resident RAM headroom;
  every successful automatic GPU-cache candidate checks VRAM reserves, including the final attempt.
  Invalid saved memory values and infeasible automatic launch plans fail before configuration publication
  or server startup. Explicit environment removals survive refresh; GiB conversion preserves positive
  sub-byte budgets and safely saturates oversized finite requests.
- **What worked:** the focused Python suite and CPU memory/retry fixtures pass. A separate CUDA candidate
  builds and starts its help command with matching source, executable hash and build receipt.
  Previously selected engines and configurations remain unchanged.
- **What did not or remains open:** no full model was downloaded or launched. Combined automatic-policy
  fit, sustained application headroom and performance on either PC remain unverified. These changes do
  not establish the cause of desktop stalls. Cache ratios, batching and context settings are unchanged.
- **Verification:** Windows CUDA architecture 89 candidate build; 90 Python tests and two targeted CTests,
  including 21 GPU-free retry checks; independent whole-diff review after corrective repairs;
  reviewed/committed diff identity and unchanged hashes for six protected engine/configuration files.

## 2026-10-16 — Shared main and automatic startup (unreleased)

Implementation range: `63a01bba5defba75d6e688f9c4d77a0dd2f7520f..9a73411d3a917f333bd327fd07626572fc28d2a6`.

- **What changed:** combined the MSI main branch with the laptop memory safeguards and
  desktop validation branch. Automatic profiles refresh and check admission once at
  server startup for the effective GPU, using current physical RAM and Windows commit.
  Refreshed settings stay in memory; startup leaves the saved profile unchanged.
  Explicit overrides and static profiles retain their existing ownership.
- **What worked:** an isolated CUDA build on the RTX 4080 SUPER Windows desktop passed
  the smoke workload and an automatic-start arithmetic/code/reuse run. The automatic run
  completed four requests with minimum available RAM 4.249 GiB, available commit
  16.004 GiB and sampled CUDA-free VRAM 731 MiB. All protected local runtime files
  matched their original hashes; the working installation was not switched.
- **What did not or remains open:** no new speedup is established. These automated runs
  do not establish watched responsiveness, merged-build full-128K acceptance or MSI
  runtime acceptance. HIP/SYCL remain untested. CUDA memory readings were stage samples,
  not continuous minima. Each PC must build locally and retain its own model paths.
- **Verification:** 181 Python tests, 12 focused C++ checks, CUDA build, independent
  semantic review and final reviewed-source hash comparison passed. GPU-selection
  conflicts and missing automatic context fail before engine creation; runtime-only
  refresh removes the reproduced saved-profile publication race.

## 2026-10-16 — Ready-cache copy reuse experiments (unreleased)

Implementation range: `228cb8ab155fd334da840dcedb1bca46b5a16118..f235a929b6bf33ca5a348832f8e2be0f8e097766`.
This range covers tracked Strata implementation and measurements documentation; local runtime
profiles, engines and diagnostic artifacts are separate evidence, not part of that range.

- **What changed:** default-off ready-cache copy reuse and independent copy/prefill payload
  counters, an opt-in demand-only prefill path with full fallback scratch, and complete fetch
  diagnostic line framing. Working profiles, shared defaults and model quantization stay unchanged.
- **What worked:** on the RTX 4080 SUPER / Ryzen 7 9800X3D / 32 GB Windows desktop, clean
  off/on/on/off runs measured 20.4298/18.9222/18.9218/20.5188 seconds for 8K prefill. Both enabled
  repeats reduced latency by 7.38% versus the faster control. Fixed-budget 130880-token retrieval
  and 130903-token reuse passed; fresh prefill took 253.983 seconds and the follow-up 3.044 seconds.
  A separate unselected automatic candidate resampled to 9 GiB pinned / 4040 MiB recent cache and
  passed arithmetic, code recall and prefix reuse; this short check has different memory budgets.
- **What did not or remains open:** no repeated decode or cached-reply gain. Demand-only reduced
  unused transfer payload but was slower; RAM splits, stager sleep and GPU ring changes did not
  meet the 5% screen. Both reuse-on/off held-out controls missed interval union identically
  (17 instead of 20); quality remains three of four, not a full acceptance pass. Long retrieval
  has no matched speed control. Read-stall cause, watched responsiveness, laptop and HIP/SYCL
  remain unverified; file-level WPR capture was denied. No new retention/pipeline rewrite was justified.
- **Verification:** isolated CUDA build, 131 helper/benchmark Python tests, 11 C++ checks,
  19 local harness probes, independent whole-diff and residual evidence review, raw telemetry,
  protected-file hashes and owned-process cleanup passed. Long-run minimum available RAM was
  4.407 GiB, Windows commit 16.028 GiB and logged CUDA-free VRAM 677 MiB; CUDA readings are stages.

See [conditions, measurements and limits](docs/UNSLOTH_Q2.md#ready-copy-reuse-and-remaining-optimization-routes-2026-10-16).
