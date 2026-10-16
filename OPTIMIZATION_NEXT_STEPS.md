# Remaining optimization experiments and validation

Status: 2026-10-16, after `a61874f3` on `main`. This records work not yet tried or
fully validated, with prerequisites for conditional experiments.
Measurements and reproduction context are in
[the 4080 SUPER report](docs/UNSLOTH_Q2.md#ready-copy-reuse-and-remaining-optimization-routes-2026-10-16).

Ready-cache copy reuse reduced 8K prefill latency by 7.38% in both enabled runs of
one clean off/on/on/off block. It did not establish a decode or cached-reply gain.
Fixed-budget near-128K retrieval/reuse passed once. The separate automatic candidate
passed short startup checks with different memory budgets. Defaults and the selected
working profile remain unchanged; read stalls and the matched held-out quality
failure remain unresolved.

## Next measurements

### Attribute fast and stalled reads

The process/thread/file-level Windows trace is still missing. WPR DiskIO/FileIO
startup returned access denied `0x80070005`; no trace was captured. Use a suitable
capture context before retrying, without silently changing security settings.

Capture from before the fast request through a slowdown. Correlate expert and PLE
reads, actual OS completion, host scheduling, hard faults, competing I/O, cache
allocations and per-worker scratch. Bound trace storage and account for its own I/O.
Separate submission, completion and the host observing usable bytes: completion-wait
timers alone do not identify SSD, filesystem, antivirus, paging or free-space causes.
The user cleared disk space, but its timing relative to earlier trials is unknown.

### Repeat the gain and measure long-context performance

- Repeat the successful 8K off/on/on/off comparison in another independent block
  and under representative application load. Compare both candidates against the
  faster same-day control, keeping prompt order, outputs and realized allocations.
- Run matched, repeated near-128K controls with reuse disabled/enabled. The single
  253.983-second prefill and 3.044-second reused follow-up establish correctness,
  not a long-context speedup.
- Test sustained generation at long context with enough output room. The retrieval
  answers were only 24 and 14 tokens. For example, target an actually encoded prompt
  near 128000 tokens and reserve 2048 output tokens plus framing within 131072.
  Record actual generated length; early termination is not a throughput improvement.
- Measure actual first-content latency separately from engine prompt time. Track
  slow-request incidence and task transitions as well as aggregate tokens/second.

### Validate the automatic candidate beyond short startup

The isolated automatic profile resampled to 9 GiB pinned / 4040 MiB recent, whereas
the repeated performance and long-retrieval runs froze 8 GiB / 4051 MiB. Full short
screens, sustained generation, near-128K retrieval/reuse and idle monitoring remain
to be run for the refreshed automatic budgets. Refresh once immediately before each
comparison, save the effective configuration, then freeze it during measurement.

Watched responsiveness with the new candidate remains unverified. Tell the user
what to observe and wait for a fresh explicit **ready** before starting a watched
test. Do not infer responsiveness from unattended telemetry.

## Conditional investigations

- **Retention value:** collect identity-level first-demand and reuse-distance traces,
  separating compulsory reads, in-flight joins, retained hits, evicted rereads and
  duplication across tiers. Replay a bounded alternative in shadow analysis before
  changing admission/replacement. Aggregate hit counts and the negative RAM-split
  pilots do not justify a new policy by themselves.
- **Useful overlap and bounded issuance:** only revisit a feeder/pipeline if tracing
  exposes work that can safely overlap the remaining wait. Existing readers already
  batch submissions and prefill already overlaps owned buffers. Measure genuinely
  cold unrouted file bytes separately from logical H2D payload. Demand-only already
  removed unused payload but was slower; a redesign would need a different measured
  mechanism. Change PLE issuance only if its interference is demonstrated.
- **Asynchronous ownership:** audit `blob_stable()` consumers and establish owned
  copy-out or completion-bound leases before extending asynchronous source reuse.
  Its direct-stage lifetime contract remains unresolved; corruption was not shown.
  Exercise replacement beyond the protection window, ring wrap, cancellation,
  I/O errors and close/drain behavior. Keep `STRATA_STAGE_PIN` disabled.
- **Later bottlenecks:** revisit CPU/PCIe assignment or KV-prefetch only after file
  waits cease to dominate and the relevant current paths, flags, buffers and costs
  are verified. No new knob or performance benefit is established here.

## Other acceptance gaps

- Rebuild and validate the latest code on the other PC. RTX 4050 laptop default
  eager-loading startup, usual external-monitor conditions and full near-128K
  retrieval/reuse remain separate acceptance work. Preserve each PC's local models,
  paths and automatic budgets.
- HIP/SYCL builds and runtime behavior, and Linux acceptance, remain unverified for
  these changes. CUDA desktop results do not establish them.
- Broader task quality remains unproven. Both reuse-on/off controls passed three
  held-out cases and answered interval union with 17 instead of 20. Preserve this
  failed result separately from successful retrieval and performance tests.

## Already tried: avoid repeating unchanged

The existing [stager-thread comparison](docs/UNSLOTH_Q2.md#returned-research-stager-thread-comparison)
already tested 32 versus 16 threads with host ring 128 and batch 1; it did not meet
the repeated 5% criterion. Demand-only prefill, equal-total pinned/recent splits,
stager sleep and GPU prefill ring 48 also failed the latest performance screen.
Earlier tests covered larger caches, reduced VRAM reserve, window eight without
useful drafting, `--short-read 0`, canonical MTP, PCIe fraction 1.0, historical
placement and mapping retention. Reopen these only with a changed hypothesis or
new evidence. Forced buffered loading previously triggered the RAM guard.

## Test constraints

Keep UD-Q2_K_XL, 131072 capacity, int8 KV and eager CUDA for matched desktop
comparisons. Preserve working engines, profiles, credentials and rollback; do not
switch selection merely because a test passes. Use authenticated loopback and an
owned-process harness with available physical RAM, Windows commit and actual
CUDA-free telemetry. Retain immediate stops below 2 GiB RAM/commit, sustained stops
below 2.5 GiB RAM or 4 GiB commit for five samples, and CUDA-stage stops below
256 MiB. Record effective allocations, partial failures and cleanup. CUDA-stage
samples are not continuous VRAM minima. Never relax guards to make an arm fit.
