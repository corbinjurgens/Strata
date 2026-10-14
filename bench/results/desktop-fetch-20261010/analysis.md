# Desktop fetch and expert placement, 2026-10-10

## Scope and identity

Windows desktop: RTX 4080 SUPER 16 GB, Ryzen 7 9800X3D, 30.88 GiB usable physical RAM,
Crucial CT2000P3SSD8 NVMe model storage. Codex CLI remained open; no desktop app or unrelated
process was killed. The model is Unsloth Qwen3.8-Flash-Next UD-Q2_K_XL, native weights unchanged,
with **131072-token capacity**. The verified existing three GGUF shards and pack were reused.
No model download or conversion was performed.

Git HEAD remains `6794d901f81717ed9c5b5513c83c0eba1187f068` on `codex/unsloth-file-cache`.
The working tree contains both pre-existing changes and this run's changes; HEAD alone cannot
reproduce either current executable. The mixed index was preserved; nothing was staged, committed or pushed.

| Executable | SHA-256 | Use |
| --- | --- | --- |
| `engine/strata-unsloth-compact.exe` | `46f22bebde5517a880c7e3605095dc5df72f1c10219f53ea5197943ac3404082` | Unchanged daily rollback and early placement tests |
| `engine/strata-fetch-profile.exe` | `d215ac687b55bff03896463eaa27f32f7ff94acf64b7dbc9576535e0700828dd` | Fetch timing diagnostic |
| `engine/strata-unsloth-portable.exe` / `strata-read-path.exe` | `fc0ae8c0e97ad501e6c1a994ee51f571d486e1b9e4fae73da9e84a82f7f17b4b` | Current source build; experimental read flag off in final profiles |

The current build receipt is `engine/BUILD-unsloth-portable.json`: engine source fingerprint
`51760fcdf51c450e`, Release, CUDA/native experts enabled, architecture 89. It records actual CMake
settings and selected GGML source hash `a538e37d3a812a0f2a3f3178c4d14f121400a1a455eef9afe4a141d5ffcc6f2d`
over 1376 files. That local extracted tree has no Git metadata; setup's pinned revision is
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Native compilation, the file-expert fixture, and the
public build/installation path ran here. The public build reused the already compiled current tree.

## Method

Each throughput run starts a fresh authenticated server on `127.0.0.1:8081`, uses one 64-token
warm-up, then two rounds of three fixed technical prompts capped at 160 output tokens. Thinking
is disabled. Aggregate speed is total output tokens divided by total reported decode time;
prefill is reported separately. See each directory's `results.json`, `engine.log` and `server.log`,
plus the adjacent `*-telemetry.jsonl` where recorded for five-second GPU/physical-disk/commit samples.
The first `baseline-1` predates that telemetry wrapper; it has the one-second RAM guard and initial
ambient GPU observation (424 MiB, 26 C, P8), but no adjacent five-second telemetry file.
These disk counters are system-wide samples, not exact per-request disk traffic.

The first telemetry samples in matched baseline/trained/baseline order recorded available RAM
25.502 / 25.580 / 25.555 GiB, system commit 8.406 / 8.444 / 8.490 GiB against a 55.880 GiB
limit, and GPU memory 545 / 545 / 516 MiB. Available RAM and commitment are separate constraints.

The helper refuses an existing Strata server and only stops processes it owns and identifies.
RAM is sampled once per second. The original guard required five consecutive samples below 2 GiB;
after the failed buffered experiment it also stops immediately below 1 GiB. Raw failed results
were retained. Profiling runs are excluded from throughput comparisons. No async/PCIe-share
retuning was repeated; the previous diagnostic already found no overall async gain.

All normal runs use int8 KV, 32768 resident cells, prefill auto, 1024 MiB VRAM reserve, verifier 2,
no MTP/suffix/lookup, and PCIe fraction 0.55. API authentication, `fit_max_tokens=true` and
`open_browser=false` are preserved. These tests do not validate broad answer quality.

## Completed throughput comparisons

| Run | Main tokens / decode seconds | Main tok/s | Main prefill seconds | Actual recent MiB | Minimum available GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Unchanged daily baseline](baseline-1/results.json) | 960 / 48.4624 | 19.809 | 20.8673 | 5420.5 | 2.901 |
| [Baseline with answer-only held-outs](baseline-held-2/results.json) | 960 / 83.3931 | 11.512 | 32.0020 | 5413.0 | 2.620 |
| [Trained placement, first restart](trained-held-1/results.json) | 960 / 34.5536 | 27.783 | 7.0576 | 5455.6 | 2.588 |
| [Trained placement, second restart](trained-transfer-2/results.json) | 960 / 34.6883 | 27.675 | 7.1882 | 4983.5 | 2.692 |
| [Daily baseline recheck](baseline-transfer-3/results.json) | 960 / 60.2881 | 15.924 | 25.829 | 5133.4 | 2.835 |
| [Matched 4 GiB, shipped placement](fixed-4g-base-1/results.json) | 960 / 52.1444 | 18.410 | 23.5774 | 4096 | 4.360 |
| [Matched 4 GiB, trained placement](fixed-4g-trained-1/results.json) | 960 / 35.7739 | 26.835 | 7.4205 | 4096 | 4.393 |
| [Matched 4 GiB baseline recheck](fixed-4g-base-2/results.json) | 960 / 52.3579 | 18.335 | 23.5820 | 4096 | 4.625 |

The matched runs use the same current executable, 12 GiB pinned experts and 4096 MiB recent cache.
Only expert placement changes. Main-workload logical decode file bytes fell from 21853 MB to
10988 MB. Those are engine-requested expert bytes, not measured physical disk reads. The original
daily baseline and first trained restart similarly fell from 19330.8 MB to 10099.1 MB.

The saved placement was trained on the technical workload, so its large gain on those prompts
is a **workload-specific result**. It changes which experts execute on the CPU/GPU and the
floating-point rounding path: all six main output strings differ between shipped and trained
placement. Expert source bytes and the quantization are unchanged. This is not an exact-output
speed comparison. Repeated outputs within each placement were stable in the matched runs.

### Unseen prompts and correctness

Four separate 160-token-cap transfer prompts cover run-length Python, list-mutation debugging,
inventory arithmetic and a constrained project summary. They were not used for training.

| Pair | Shipped placement | Trained placement | Separate prefill totals |
| --- | --- | --- | --- |
| Original cache ceilings | 494 / 58.8238 s = 8.398 tok/s | 529 / 57.3124 s = 9.230 tok/s | 57.8249 / 46.4587 s |
| Matched 4096 MiB recent | 494 / 104.1546 s = 4.743 tok/s | 529 / 105.1832 s = 5.029 tok/s | 57.3045 / 51.9480 s |

Different output lengths and variable read stalls limit attribution of these small aggregate
gains. Both matched runs stalled despite more than 4.3 GiB available RAM. Placement does not
establish a stall fix or a general 40% speedup.

Manual review plus AST-restricted execution of the inspected pure Python functions passed empty,
repeated, nonconsecutive and `None` run-length examples, and negative-filtering examples without
input mutation. See [transfer-code-checks.json](transfer-code-checks.json). Baseline debugging
prose incorrectly says the shortened list has length 2 instead of 3; the trained answer describes
that trace correctly. Both inventory answers reach the correct 174 but truncate the requested
final reconciliation at the cap. The project summary is identical and preserves the supplied facts.
The four matched transfer answers are exactly the corresponding earlier answer from their own placement.

Four concise exact-answer held-outs scored **3/4 for both placements**: Python aliasing, generator
mutation and dependency scheduling pass; inclusive interval union returns 17 instead of 20.
The oracle was not weakened. Their aggregate rates were 9.902 vs 9.823 tok/s: no gain. The initial
held-out protocol asked for explanations and truncated all four before their answer markers; that
failed run remains under `baseline-held-1`, and the revised answer-only prompts were rerun on both.

## Fetch diagnosis and rejected experiments

The opt-in counters separate lookup/victim scanning/allocation/eviction, lock and fill waiting,
read planning/submission/waiting, and blob copies. They only measure request decode deltas, not
inter-request idle. Inclusive counters and sums over threads must not be added as disjoint wall time.

In [fetch-profile-1](fetch-profile-1/results.json), two responses reproduced the severe slowdown
while all six output strings exactly matched the unchanged baseline. Over 960 tokens, read wait
averaged 59.5033 ms/token; direct copies 0.9856, read submission 0.551, victim scan 0.0241,
allocation 0.0151, eviction 0.0711 and stage-lock wait 0.0065 ms/token. Waiting for another fill
was zero. Submitted/useful read bytes were 1.00823. Existing reads already batch usefully;
rewriting cache bookkeeping or adding another batching layer is not supported by these measurements.
Read waiting is an OS I/O boundary, not proof of a specific SSD hardware cause.

`STRATA_FILE_DIRECT_UNMAP=1` retires owned whole-file views after startup copies and uses an
independent positional buffered fallback if a direct read fails. Its native fixture passes
exact bytes, split roles, live buffers, fallback concurrency, failed-fill invalidation and reset.
The diagnostic retired all 78,858,182,240 mapped bytes and used no fallback reads. Nevertheless,
the diagnostic scored 19.483 tok/s and the normal run only 12.000 tok/s with stalls. The flag stays
**off**; source correctness does not imply a speed benefit. OS unmap failure, >4 GiB fallback
offsets and >8 MiB fallback chunks are not forced by the fixture; Linux runtime is untested.

Forcing `STRATA_UNBUFFERED_LOAD=0` failed during startup: available RAM reached 0.06731 GiB before
the original five-sample guard stopped the owned processes. No throughput result exists. GPU
expert fill took 53.9 seconds instead of a typical 11.8; pressure rose while building the pinned
tier. The mapped/buffered path is rejected on this desktop. Memory recovered, no Windows setting
was changed, and the stronger emergency guard was added and tested. No retry near exhaustion ran.

The repeated slowdowns also occur on the original unprofiled executable. Sampled GPU clocks and
temperatures do not establish a sustained GPU collapse; five-second telemetry cannot exclude
brief events. The root cause remains unresolved. The async milliseconds-per-round counter is
not used as evidence because it can include idle time.

## Hardware-aware build and launch

The public `tools/prepare_unsloth_q2.py` path built and installed the current executable, reused
the verified model pack, generated a distinct local profile, and actually launched it twice under
the owned-process RAM guard. In [public-launch-2](public-launch-2/results.json), a deliberately
wrong inherited API key did not override the saved credential. Missing/wrong requests returned
401; authenticated arithmetic returned 391; capacity was 131072, output fitting stayed enabled,
browser opening stayed disabled, and cleanup completed. Launch refreshed the automatic policy
from 10 GiB pinned / 5851 MiB recent to 11 GiB / 6013 MiB. The short probe retained at least
6.288 GiB available RAM; it is not a throughput or full-context result for that automatic budget.

The policy considers available physical RAM and Windows commit separately, GPU commitment,
all streamed-KV storage, CPU/server use, later startup buffers and application headroom. It
preserves explicit overrides and machine-local keys; generated launchers resample automatic
values at each start. Budgets do not resize while the server is running. The RTX 4050 policy
uses q4_0 KV, 20480 resident cells, 64-token separate prefill buffers, 448 MiB GPU reserve and
exact owned-buffer pricing, with RAM budgets calculated from its own 64 GB system. The model
weights and capacity stay unchanged. These laptop choices are simulation-tested, **not runtime
validated on the laptop**; its reported 15–17 tok/s is not a controlled comparison.

The selected desktop placement is saved separately as
`data/expert-profile-unsloth-desktop-20261010.bin`, 196632 bytes, SHA-256
`3c6b35a8f063bf439f5864582e1eb7c07a9c019930a7bb365ac2bea94e217027`.
The shipped `data/expert-profile.bin` remains intact. Desktop-only fixed cache overrides must not
be copied to the laptop. The generic launcher remains `run-unsloth-portable-128k.bat`.

### Reproduce elsewhere

The source branch must contain the existing native/compact-cache work **and** this run's changes.
In particular, preserve the working versions of `setup.py`, `tools/test_setup_config.py`,
`include/strata/core/expert_source.hpp`, `src/core/expert_source.cpp`, `src/program/generate.cpp`,
`tests/core/file_expert_source_test.cpp`, `tools/prepare_unsloth_q2.py` and its test, and
`tools/benchmark_compact_cache.py` and its test. Benchmark reproduction also needs
`tools/diagnose_desktop_stalls.py` and its test, and the existing
`bench/results/2026-10-09-unsloth-file-cache/{benchmark.py,validation.py}` workload modules.
Keep the updated docs and this evidence report. These files are not all contained in HEAD yet.
The [final artifact inventory](final-artifacts.json) records their hashes and the separate locally
generated files. `CHANGELOG.md` records this work as uncommitted under Unreleased.

On each PC, use that source state and its local Python/build prerequisites; obtain the selected
llama.cpp/ggml dependency tree, build for the detected GPU, and prepare from that PC's existing
GGUF folder. The [hardware-aware recipe](../../../docs/UNSLOTH_Q2.md#hardware-aware-preparation-for-both-pcs)
generates paths and credentials locally, verifies/reuses the model pack, and writes the separate
portable launcher. No account-specific path is hardcoded in the helper. Do not commit ignored
JSON credentials, model weights, model packs, build trees or executables. Build receipts and
launchers are generated locally. Rebuilding with a different compiler may change the executable
hash; retain the resulting receipt rather than claiming byte identity with this Windows build.

To reproduce this desktop's placement exactly, retain the separate 196632-byte profile above
as a nonsecret artifact (it may be committed separately from credentials). It contains expert
placement rankings derived from routing observations, not model weights or prompts. After generating a **new desktop-only** config with the
following command, set its existing `--expert-profile` argument to that file's local absolute path,
retain the key privately and preserve the keyed source config's host and port. On this desktop
that is 0.0.0.0:8080 for LAN access. The generic portable profile needs no placement override.

```bat
.venv\Scripts\python.exe tools\prepare_unsloth_q2.py --prepare --config strata-unsloth-desktop-trained-128k.json --key-config strata-unsloth-desktop-128k.json --resident-budget-gib 12 --file-cache-mib 4096
```

Then, from the checkout in PowerShell, select the separately saved placement without printing
or replacing the credential. This command uses local paths and publishes the generated config
through the helper's normal atomic write:

```powershell
@'
from pathlib import Path
import json
from tools import prepare_unsloth_q2 as prep
p = Path('strata-unsloth-desktop-trained-128k.json')
placement = Path('data/expert-profile-unsloth-desktop-20261010.bin').resolve(strict=True)
c = json.loads(p.read_text(encoding='utf-8-sig'))
prep.set_flag(c['args'], '--expert-profile', str(placement))
c.update(fit_max_tokens=True, open_browser=False,
         log=str(Path('strata-unsloth-desktop-trained-128k.log').resolve()))
prep.check_ready(c)
prep.atomic_files({p: prep.json_bytes(c)})
print('Separate desktop placement selected; saved credentials retained.')
'@ | .\.venv\Scripts\python.exe -
```

Use a machine-local keyed config for `--key-config`, or omit that option to generate a local
key when no usable default key exists. Existing configurations are preserved; the preparation
command does not replace their settings. The public helper records the explicit desktop budgets
so later refresh does not silently remove them. On the laptop, create the ordinary portable
profile without these two fixed overrides.

The placement can instead be regenerated from a distinct training config using the existing
top-level `expert_profile_save` and `expert_profile_save_every` settings. This run set the save
path to `bench/results/desktop-fetch-20261010/trained-profile.bin` and the period to 0.001 minutes,
then ran the normal six-case benchmark once (`train-1`). Load the resulting file with
`--expert-profile` in a separate test config. Do not overwrite the shipped profile; remove save
settings from an evaluation config, restart, and evaluate unseen prompts. A new workload or
machine can select different useful placements, so the desktop result is not a laptop default.

Offline verification ran 120 combined Python tests, 8 public-launch probe self-tests and the
native `file_expert_source_test`. Independent native Codex review ran 24 targeted tests plus
boundary and identity probes. Native fixtures cover expert bytes, compact eviction/accounting,
allocation failure, live/in-flight buffers and the experimental read fallback. Raw gate logs and
source review are under `.claude/plan-runs/desktop-fetch-20261010/`; all earlier staged, unstaged
and untracked work is retained. No changes were pushed.

## Final acceptance and selected launcher

For the measured technical workload, use **`run-unsloth-desktop-trained-128k.bat`** on this desktop. It uses the exact same
UD-Q2_K_XL weights, capacity 131072, int8 KV/ 32768 resident, pinned 12 GiB and recent 4096 MiB,
with the current source-built portable executable and the separately trained placement above.
It is authenticated loopback on 8080, with output fitting enabled and browser opening disabled.
The original `run-unsloth-desktop-128k.bat` and its executable/config remain the rollback.

The [final idle repeat](trained-idle-2/results.json) completed 960 tokens in
35.4641 decode seconds = **27.070 tok/s**, with
7.4922 seconds total prefill. Before/after the four-minute idle
gap it measured 26.255/27.936 tok/s. The earlier matched trained run
was 26.835 tok/s; matched shipped-placement runs were 18.410 and 18.335. Compared with the initial
unchanged daily baseline 19.809, this final main-workload result is 36.7% higher.
This percentage applies to those training-like prompts; unseen performance and output differences
remain the limits described above. Idle time is excluded from decode-time aggregation.
The final unseen repeat produced 529 tokens in 72.8976 decode seconds
= 7.257 tok/s, with 85.0356 seconds
prefill. Across trained runs, unseen rates ranged 5.029–9.230 tok/s. There is no dependable broad-workload
win; this placement stays a separate optional desktop profile. Generic portable defaults keep the
shipped placement, and the old daily profile remains untouched. Final main and unseen output strings
exactly match the earlier matched-cache trained run; the inspected Python examples still pass.

All six smoke checks passed. The [long-context run](trained-long-1/results.json) passed fresh
retrieval with the entire locally counted prompt, an output request larger than remaining capacity,
and a follow-up that retained the complete history and reused its long prefix:

| Check | Prompt tokens | Reused | Prefill seconds | Decode seconds | Answer |
| --- | ---: | ---: | ---: | ---: | --- |
| Fresh near-limit | 130880 | 0 | 844.7818 | 4.9289 | field-cedar-284 marble-finch-693 cove-lantern-517 |
| Full-history follow-up | 130944 | 130906 | 3.2078 | 1.1512 | cove-lantern-517 391 |


These successful requests establish observed full-prompt retention, fitting acceptance and bounded
completion; early EOS alone does not prove every internal output-cap boundary. The response codes
were placed near 5%, 50%, and 95% of the reference. Available RAM never fell below
4.025 GiB in the one-second long-run guard; no guard cancellation.
The final idle repeat minimum was 3.959 GiB.
This validates the measured retrieval/reuse sequence, not arbitrary reasoning over 128K tokens.
The short long-context answers decoded at 5.5 tok/s (27 tokens) and 12.2 tok/s (14 tokens), so the
technical short-prompt throughput must not be presented as full-context generation speed.

The [selected public launcher probe](selected-public-launch/results.json) then passed an actual
helper launch with a conflicting inherited key, saved-key authentication, missing/wrong 401,
131072 capacity, fitting, disabled browser opening, 391 arithmetic and owned-process cleanup.
Its explicit 12 GiB/4096 MiB budgets and saved placement survived policy refresh. All benchmark
servers were stopped by their owner; no test server remains running.

This is a defensible plateau in the tested environment: workload placement is the demonstrated
improvement; bookkeeping cost is too small to justify a rewrite, mapping retirement did not help,
and buffered loading was rejected. The disk-wait stalls remain unresolved, so no claim of global
optimality or a universal speedup is made. Laptop runtime remains the next machine-specific
validation, not an unperformed desktop test handed back to the user.

## LAN launcher correction, 2026-10-10

The initial trained profile was bound to loopback, which prevented LAN clients reaching it.
At the user's request its binding now matches the original desktop profile: `0.0.0.0:8080`.
The API key already matched and was retained. The installation recipe now preserves the generated
host and port instead of forcing loopback. The original desktop rollback config is unchanged.

The actual `run-unsloth-desktop-trained-128k.bat` passed a guarded launch test: an owned listener
on all IPv4 interfaces; saved-key requests returned 200 and missing/wrong keys returned 401
on both localhost and `192.168.11.49`. An authenticated generation through the LAN IP returned
391 for 17 times 23. Context stayed 131072, output fitting stayed enabled, browser opening stayed
disabled, and an injected conflicting inherited token did not replace the saved key.
Minimum available RAM was 4.95 GiB; the guard did not stop the run.
Only probe-owned processes were stopped afterward. These requests originated on the desktop;
a second device's firewall/network path was not tested. No Windows or firewall settings changed.
Open `http://192.168.11.49:8080` in a browser, or use `http://192.168.11.49:8080/v1`
for OpenAI-compatible clients with the existing token.

[LAN probe results](lan-followup/results.json) and [bounded server log](lan-followup/server.log)
record this correction separately from the unchanged benchmark measurements.
