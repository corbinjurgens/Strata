# RTX 4080 SUPER shared automatic policy, 2026-10-15

The guarded desktop workload passed on implementation commit
`563c136c1df98ae416022e595e606b963c7f2d22`. This validates the measured startup,
short generation, cache reuse, sustained generation, idle and near-128K retrieval sequence.
It does not explain the earlier desktop stalls or establish broad model quality.

Hardware: RTX 4080 SUPER, 16376 MiB VRAM, Ryzen 7 9800X3D (8 cores/16 threads),
30.88 GiB usable RAM, Windows, NVIDIA driver 617.14, CUDA 13.4.92, MSVC 19.44.
The automatically managed pagefile was 28 GiB; Windows commit limit was 58.880 GiB throughout.
The existing UD-Q2_K_XL GGUF shards and BF16 compatibility pack were reused without changing quantization.

The remote branch was initially missing the commit. After the laptop pushed it, the desktop fetched
and checked out the exact commit into a separate detached worktree:
`C:\Users\Joes\Tools\Strata-4080s-headroom`. The original repository stayed on
`codex/unsloth-file-cache` at `411ee5409c75e10ec6e280a16b85649c69285d56`.
No working profile or launcher selection was changed.

## Frozen settings

The isolated build used two compile jobs and eager CUDA module loading. Executable SHA-256:
`f815d5590c92689341e846d963322780d6e70f40cf6d1d951ff41f0165dabd67`.

The policy was refreshed once immediately before the watched test, admitted and saved, then held
unchanged for both model runs. Its SHA-256 was
`4a251e5fef7dcc20f096f55c681003f724339dbe03f3a875af2993023ea6a333`.
The credential-bearing config remains local and is excluded from this report.

- 131072-token capacity, int8 KV, 32768 resident cells.
- Automatic GPU expert cache, 5 GiB pinned request, 3486 MiB compact recent-cache request.
- 4 GiB host headroom, 1024 MiB GPU reserve, automatic prefill (8192 tokens for the large request).
- Verifier window 2, MTP/suffix/lookup off, parallel 1; existing cache ratios and batching unchanged.
- Authenticated `127.0.0.1:8081`, no browser opening, output-budget fitting enabled.
- `STRATA_TRACE=1`; `CUDA_MODULE_LOADING=EAGER`.

These budgets came from this desktop's live readings, not manual performance tuning. Fresh physical
availability was 17.08 GiB and available commit 43.47 GiB when settings were frozen. The helper,
engine receipt and source checks passed. The saved effective nonsecret settings are in `summary.json`.

## Results

The watched phase ran for 389.69 seconds, including a 60-second idle period. Short exact answers,
the first cached follow-up, three 512-token generations and all their cached follow-ups passed.
Sustained decode rates were 7.1, 8.2 and 8.2 tokens/s. Each sustained follow-up reused 560 tokens,
processed 29 new prompt tokens and returned the correct arithmetic answer.
During this phase the user reported **responsive throughout**.

The separate long phase ran for 1385.38 seconds with the same frozen settings. All six smoke checks
passed. A fresh prompt retained all 130880 API prompt tokens, used no prefix cache and returned
the three reference codes in order. Prefill took 1160.4925 seconds at about 112.8 tokens/s;
request wall time was 1166.828 seconds and decode was 4.5 tokens/s. The requested 256-token output
budget exceeded remaining capacity; fitting passed without dropping prompt tokens.
Internal chunk traces counted 130873 tokens; full tokenizer/API accounting was 130880.

The follow-up retained 130944 prompt tokens, reused 130906 tokens and processed 38 new prompt tokens.
It returned the correct reference code and `391`, generating 14 tokens at 5.9 tokens/s.
Prefill took 6.6385 seconds and request wall time was 9.3332 seconds. Every long-context check passed.
The user was away during this phase, so long-phase desktop responsiveness was **unobserved**.

## Memory and cleanup

One-second Windows monitoring used GetPerformanceInfo's CommitLimit minus CommitTotal for available
commit, separately from physical RAM. It did not substitute pagefile usage for commit headroom.

- Watched phase: minimum available physical RAM **4.2493 GiB**, maximum physical RAM used
  **26.6310 GiB**; minimum available commit **15.0368 GiB**, maximum system commit **43.8434 GiB**.
  Peak owned RSS was 11.0768 GiB; peak owned private allocation was 28.3080 GiB.
- Long phase: minimum available physical RAM **8.5835 GiB**, maximum physical RAM used
  **22.2968 GiB**; minimum available commit **18.7306 GiB**, maximum system commit **40.1496 GiB**.
  Peak owned RSS was 11.5229 GiB; peak owned private allocation was 29.1396 GiB.
- The engine's actual CUDA query reported **761 MiB free at serving** in both runs.
  Completed-request samples were 731 MiB in the watched run; the long-run minimum logged sample
  was **677 MiB**. These are stage samples, not continuous VRAM minima. No extra CUDA context was
  created for monitoring. Supplemental nvidia-smi readings are labelled separately.

No guard triggered. The physical guard stopped below 1 GiB immediately or below 2 GiB for five
samples. The additional commit guard stopped below 2 GiB immediately or below 4 GiB for five samples.
Serving/request-stage CUDA samples below 256 MiB, telemetry failures, bounded-log failures or a
5400-second phase limit also stopped only the owned process tree. HTTP requests retained the
benchmark's 1800-second timeout. Synthetic guard tests passed before model launch.

Both owned servers exited and port 8081 was released. All 33 hashed working configs, launchers,
engines and placement files remained unchanged; the original checkout HEAD was unchanged.
After cleanup, physical availability was 22.62 GiB and available commit about 48.97 GiB.
nvidia-smi returned to 10 MiB used / 16036 MiB free; this is not an engine CUDA-free query.
No Strata service was left running and no profile was promoted to the working selection.

The 127 helper/benchmark unit tests and the `expert_cache_retry_test` and `file_expert_source_test`
CTest fixtures passed. Model preparation and readiness checks passed without starting the model.

## Evidence and reproduction

`summary.json` contains sanitized effective settings, per-request timings and aggregate memory.
`short-01/` and `long-01/` contain credential-redacted request results and one-second telemetry.
`build-receipt.json`, `baseline.json`, `post-snapshot.json` and `fixtures.txt` retain build,
preservation, cleanup and fixture evidence. Full bounded engine/server logs remain local under
`.claude/4080s-validation/{short-01,long-01}/`.

The measured wrapper and helper snapshots are in `harness/`. They expect to be installed under
`.claude/4080s-validation/` in a checkout of the implementation commit; keep that relative location.
The local candidate uses the existing `.venv` and llama.cpp checkout through directory junctions.
Build and stage a separate engine/config with `tools/prepare_unsloth_q2.py`; preserve working files,
reuse verified model files and choose unused paths. The local `freeze_profile.py` samples, admits,
checks readiness and freezes this isolated profile. It must run only once immediately before a
comparison. Keep credentials local and do not restore a measurement budget as a universal default.

After telling the observer what to watch and receiving an explicit `ready`, the measured commands were:

```powershell
.\.venv\Scripts\python.exe -B .claude\4080s-validation\freeze_profile.py
.\.venv\Scripts\python.exe -B .claude\4080s-validation\watched_benchmark.py --config strata-unsloth-4080s-headroom-128k.json --output .claude\4080s-validation\short-01 --watched-short
.\.venv\Scripts\python.exe -B .claude\4080s-validation\watched_benchmark.py --config strata-unsloth-4080s-headroom-128k.json --output .claude\4080s-validation\long-01 --smoke --long-context --validation-only
```

These output directories already exist; reproduction requires fresh names. The original benchmark
starts its own server from saved settings and does not refresh automatic settings itself.

## Remaining work

At the time this report was first saved, the handoff's deferred entry was unavailable locally.
It became available during the same day's performance follow-up. The completed desktop portion was
recorded and the original entry archived through the project-note helper; laptop eager startup and
external-display acceptance remain open. Full laptop 128K and Linux acceptance remain unverified.
The historical desktop stall cause remains unresolved.
