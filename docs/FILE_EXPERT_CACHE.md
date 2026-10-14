# Recent file expert cache (experimental)

`STRATA_FILE_CACHE_MIB` lets the engine retain recently read experts in its existing
host assembly buffers. It is intended for native GGUF models whose expert weights
exceed the combined GPU and profile-ranked RAM caches, such as UD-Q2_K_XL on a
32 GB PC. A frequently reused file expert can then survive the next token's layer
traversal without another SSD read.

The default is zero, which keeps the original staging policy. Enable it in a local
server configuration, for example:

```json
"env": {
  "STRATA_RESIDENT_HEADROOM_GIB": "4",
  "STRATA_FILE_CACHE_MIB": "2048"
}
```

This requires an engine built from the source change. It is additional to
`--resident-budget-gib`; it does not change model weights, routing, or arithmetic.
The recent cache uses ordinary host memory, and the experts still execute through
the original file-tier CPU path. The profile-ranked pinned RAM cache continues to
participate in the existing GPU/RAM exchanges.

The desktop experiment uses `4096` MiB with an 8 GiB profile cache. Use a smaller
target when other applications need more RAM, and restart the server after
changing the value. The cache is sized at startup and does not automatically
shrink when another application allocates memory later. The engine's unset default
stays zero on all machines. The separate hardware-aware preparation helper below
explicitly selects a recent-cache budget for its generated local profiles.

At startup, after allocating the pinned RAM cache, the requested size is clamped to
remaining physical RAM and Windows commit capacity with 4 GiB of headroom. The log
reports the effective size. Buffers are allocated lazily and retained in recency
order. With compact mode disabled, the target is rounded down to whole
largest-blob slots. Compact mode uses the byte target and charges each buffer's
actual capacity, as described below. Both modes preserve the original lifetime
rules for buffers still needed by in-flight readers, which can require more space
than the retention target. If an optional allocation fails, an eligible old buffer
large enough for the requested expert is reused when one exists. Closing the model
clears the cache.

The expert-source regression test checks exact bytes, reuse across token-like
layer traversals, eviction and reopening. The same test executable checks the
existing mapped and unbuffered file paths:

```text
cmake --build BUILD_DIR --target strata file_expert_source_test
BUILD_DIR/file_expert_source_test
```

Desktop performance measurements and their limits are recorded in the
[RTX 4080 SUPER experiment](../bench/results/2026-10-09-unsloth-file-cache/README.md).

## Compact buffers (opt-in experiment)

`STRATA_FILE_CACHE_COMPACT=1`, together with a positive `STRATA_FILE_CACHE_MIB`,
allocates new recent-cache buffers at the requested layer's expert size. This
avoids reserving the largest layer's size for every expert in a mixed-quantization
model. The default is disabled; unset the variable or set it to `0` to retain
largest-blob buffers. Other nonempty values are rejected. Restart the server to
change this setting.

The cache accounts for the capacity of every allocated buffer. Replacing an old
entry with a differently sized expert can resize its buffer and release other
eligible old entries to recover the byte target. Busy or too-recent entries keep
their existing lifetime protection and stable addresses. If all remaining
buffers are protected, the pool can exceed its target until a later miss can
reclaim them. This is a retention target, not a hard process-memory ceiling;
allocator overhead, cache metadata and other engine allocations are additional.
The startup physical-memory and commit-headroom clamp still applies.

The mixed-size regression fixture checks exact model bytes, both directions of
cross-size replacement, live-buffer protection, recovery after overflow, retained
capacity accounting and close/reopen behavior. These checks establish storage
behavior; they do not establish a decoding-speed gain.

The earlier local RTX 4080 SUPER package provides a separate launcher:

```text
run-unsloth-compact-128k.bat
```

It uses `strata-unsloth-compact-128k.json` and
`engine/strata-unsloth-compact.exe`; build provenance is saved in
`engine/BUILD-unsloth-compact.json`. Its configuration copies the existing fast
profile, changing only the executable, log path and compact opt-in. It retains
the same UD-Q2_K_XL model, 8 GiB profile cache, 4096 MiB recent target, GPU/KV
settings, saved API key and network binding. The maximum context is configured
as **131072 tokens (128K)**. This paragraph describes the earlier 8 GiB package.
See the [2026-10-10 desktop measurements](UNSLOTH_Q2.md#desktop-fetch-and-placement-measurements-2026-10-10)
for the later 12 GiB profiles and their acceptance limits; a configured maximum
alone does not verify a full-context run.

Close an existing Strata server before clicking the new launcher. It clears
`STRATA_VERIFY_PROFILE` and `STRATA_DECODE_TIMING` in its local environment so an
inherited profiling setting does not affect a normal test. To roll back, close
the compact server and run `run-unsloth-fast-128k.bat`, which still uses the
unchanged baseline executable and configuration.

The [guarded benchmark helper](../tools/benchmark_compact_cache.py) can run the
same short workload against either configuration. Run these separately from the
repository folder, with no Strata server already running; use a new output
directory each time:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_compact_cache.py --config strata-unsloth-fast-128k.json --output build-unsloth\cache-baseline-test
.\.venv\Scripts\python.exe tools\benchmark_compact_cache.py --config strata-unsloth-compact-128k.json --output build-unsloth\cache-compact-test
```

The helper runs on loopback port 8081, preserves authentication, records partial
results, and stops its own run if available RAM stays below 2 GiB for five
consecutive one-second samples, or immediately below 1 GiB. It refuses an existing
server and does not stop one started by the user.
Add `--smoke` for the six existing model smoke checks. Add `--timing` only for a
separate CPU-stage diagnostic run. GPU profiling with `STRATA_VERIFY_PROFILE`
changes CPU/GPU overlap, so its timings must not be treated as normal decoding
speed; the helper strips that variable in every run. No compact-cache speed gain
has been measured in this packaging step.

## Hardware-aware profiles

The [UD-Q2_K_XL preparation guide](UNSLOTH_Q2.md#hardware-aware-preparation-for-both-pcs)
describes the separate `strata-unsloth-portable-128k.json` profile and
`run-unsloth-portable-128k.bat` launcher. Build it from the current source with
`tools/prepare_unsloth_q2.py --build-engine`, then use `--plan` and `--prepare`
with the existing GGUF folder. The default build directory is
`build-unsloth-portable`; `--build-dir build-unsloth` can reuse a compatible local
build tree. The portable executable and build receipt have distinct names, so
the saved desktop/compact rollback launchers remain available.

The helper defaults to 131072 context tokens and enables compact recent buffers.
It samples available physical RAM and Windows commit, accounts for streamed KV,
GPU commitment, later startup allocations and application headroom, then divides
the remaining expert pool between pinned profile experts and recent buffers.
The selected requests and allowances are printed. A 32 GB desktop and 64 GB
laptop therefore receive different budgets; 12 GiB pinned and 16384 MiB recent
are caps in the desktop policy, not unconditional allocations for both PCs.
The engine's later clamp and live-buffer protections still apply.

Every generated portable launch samples memory again and refreshes only its
automatic values. Saved manual overrides, keys, context, output-budget fitting
and browser preference survive refresh. This is startup adaptation; it does not
shrink a running cache when another application needs memory. Policy choices
have simulated desktop/laptop tests, not laptop runtime validation or a claim
of guaranteed best performance. Use the guide's exact build, credential and
override instructions when reproducing the setup on another PC.

## Fetch diagnostics

`STRATA_FETCH_PROFILE=1` enables separate expert-fetch counters in a source-built
engine. Unset or `0` disables them; other nonempty values are rejected. Keep this
out of normal throughput runs. The guarded helper exposes a diagnostic-only arm:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_compact_cache.py --config strata-unsloth-portable-128k.json --output build-unsloth\fetch-diagnostic --fetch-timing --timing
```

Use a new output directory and a configuration naming an engine that includes
the counters. The helper retains authenticated loopback, existing-server refusal
and the low-memory guard. Without `--fetch-timing`/`--timing`, it strips those
profiling variables from benchmark runs. The portable launcher also clears
inherited profiling variables; an explicit config `env` value remains deliberate.

Each `strata fetch profile decode:` line reports a snapshot difference for that
request's decode interval, excluding prefill and time idle between requests.
`decode_wall_ms` is elapsed request decode time. Nanosecond counters sum thread
durations: caller, claim, fill, publish, direct-read and buffered-read totals
include their child regions and must not be added together. Concurrent snapshots
can be approximate. The regions distinguish:

- Cache lookup, victim scans and slots examined, allocation/metadata growth,
  eviction, mutex waits and waits for another buffer fill.
- Direct-read planning, OS submission and completion waits, and assembly copies.
- Mapped copies, whose time includes page faults, plus buffered positional
  fallback reads when mappings have been retired.

`submitted_bytes` and `completed_bytes` are OS request bytes, including alignment
and merged gaps. They are not physical-disk measurements. Useful expert bytes
are tracked separately in direct-copy, mapped-copy and buffered-read counters.
Claim/hit counts include both prefetch reservations and later blob lookups;
they are not the GPU expert-cache hit rate. Cache target, allocated capacity and
buffer count accompany the snapshot. The fixture checks disabled/reset behavior,
aligned versus useful bytes, warm hits, safe reuse and read-phase timing.

## Optional Windows mapping-retirement experiment

`STRATA_FILE_DIRECT_UNMAP=1` is a separate, default-off read-path experiment.
It requires a Windows native-GGUF file source already using unbuffered reads
and `--ple-io direct`. Other modes reject it before retiring mappings. Use a
distinct experimental config and executable; the portable helper does not enable
it automatically. Unset or `0` preserves the normal mapped fallback.

After initial GPU/RAM copies and read-mode selection, before request workers,
the experiment releases only the expert source's redundant GGUF views and section
handles. Validated absolute offsets remain available to direct I/O. Startup logs
the exact mapped-byte total before and after; a successful retirement ends at
zero for this source. Other components/processes and the OS may still cache or
map the same files, so this is not a system-wide assertion about the file cache.

Fallback handles are opened before retirement without reading model payloads.
If direct I/O later fails, buffered positional reads fill the existing destination
in bounded chunks with per-call events and explicit offsets. This preserves
concurrent readers without a shared file position or another expert arena. If
both read paths fail, the fill fails instead of publishing partial bytes. Failure
to open fallback handles leaves mappings intact; an OS unmap failure is reported
and startup rejects the experiment while retaining ownership of remaining views.

The Windows fixture checks exact mixed-size expert bytes across split role files,
batching and `copy_blob`, live staging addresses, safe rejection, fallback-open
failure, direct-read failure, concurrent fallbacks, double-I/O failure recovery,
and close/reopen. Existing age/sequence/busy protections remain in force. These
tests do not establish a speed gain. The 2026-10-10 desktop diagnostic retired all
owned views, but its throughput run still stalled and showed no improvement.
Keep the flag off. The cause of the long read waits remains unresolved; no laptop
or Linux runtime benefit is claimed. See the
[measured experiment](../bench/results/desktop-fetch-20261010/analysis.md#fetch-diagnosis-and-rejected-experiments).
