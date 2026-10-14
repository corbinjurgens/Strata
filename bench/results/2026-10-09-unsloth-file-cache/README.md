# Recent expert caching on the RTX 4080 SUPER

This experiment adds an opt-in recent-file cache on branch `codex/unsloth-file-cache`,
based on `6794d901f81717ed9c5b5513c83c0eba1187f068`. It uses the same verified Unsloth
UD-Q2_K_XL files identified by revision, sizes and hashes in [metadata.json](metadata.json).
The engine retains file experts in the existing assembly buffers across token
traversals. It keeps the stored weights and CPU arithmetic unchanged.

The baseline executable is `engine/strata-unsloth.exe`; the candidate is
`engine/strata-unsloth-file-cache.exe`. [source.patch](source.patch) records the exact
engine/test changes, and [metadata.json](metadata.json) identifies both builds and
the hardware. [Feature documentation](../../../docs/FILE_EXPERT_CACHE.md) describes
the switch, allocation guard, and buffer lifetime constraints.
The patch uses zero context; applying it to the recorded base requires
`git apply --unidiff-zero source.patch`.

## Method

Windows, RTX 4080 SUPER 16 GB, Ryzen 7 9800X3D, 30.88 GiB usable RAM, a 24 GiB pagefile,
and a Crucial CT2000P3SSD8 NVMe model drive. All arms configure 131,072 context tokens,
int8 KV with 32,768 resident cells, automatic prompt chunks, a 1,024 MiB GPU reserve,
one request at a time, and no MTP. `STRATA_RESIDENT_HEADROOM_GIB=4` is the same across
these arms. This is distinct from the earlier conservative 8 GiB headroom trial.

Each arm starts a fresh server, sends a 64-token warmup, then the same three technical
explanation prompts with temperature zero and thinking disabled. Each measured
answer reaches its 160-token cap. The reported generation rate is total output
tokens divided by summed API decode time, excluding prompt processing and startup.
Memory is sampled once per second. Tests run sequentially; compilation does not
overlap measured generation. The harness stops only its own server process tree.

## Initial comparison with automatic GPU sizing

| Engine and host expert storage | Aggregate generation | File-tier reads during measured requests | Minimum available RAM |
| --- | ---: | ---: | ---: |
| Original, 8 GiB profile cache | 7.33 tokens/s | 35,795 MB | 7.42 GiB |
| Original, 10 GiB profile cache | 8.29 tokens/s | 30,410 MB | 5.33 GiB |
| Candidate, 8 GiB profile + 2 GiB recent cache | 9.71 tokens/s | 25,285 MB | 6.21 GiB |

The candidate is 33% faster than the 8 GiB original and 17% faster than adding the
same nominal 2 GiB to the existing profile cache in this sequence. Request file-byte
counts are logical expert payload filled during the decode reporting interval;
they exclude the preceding prompt pass. They do not measure physical SSD traffic
(sector padding, merged gaps, retries and fallbacks are not fully represented).
This interpretation was corrected on 2026-10-09 by tracing `file_bytes0` through
the engine's `DONE` record to the server's `file_mb` field. The recent-cache target
includes the assembly buffers already used by the original engine; it does not
necessarily allocate the entire additional amount shown in the table.

These first arms use automatic GPU sizing, which varies slightly with available
VRAM. Two of the candidate's three answers match the original 8 GiB arm exactly;
one produces a different coherent explanation. These are freely generated
responses, so timing comparisons are workload measurements rather than exact
token-by-token replay. The source regression checks compare stored bytes exactly.

## Fixed GPU allocation and validation

Both engines use `--expert-cache 4300`, which produced the same 4,990 native GPU
expert slots (8,922 MiB), and an 8 GiB profile-ranked RAM cache. The candidate also
requests a 4,096 MiB recent cache; its rounded target is 4,094.4 MiB. Each engine
runs two rounds of the same three prompts, for 960 measured output tokens.

| Engine | Aggregate generation | Individual responses | File-tier reads |
| --- | ---: | ---: | ---: |
| Original | 6.71 tokens/s | 6.1-7.4 tokens/s | 80,169 MB |
| Original, increased to 12 GiB profile cache | 9.15 tokens/s | 8.1-10.0 tokens/s | 47,901 MB |
| Candidate, 4 GiB recent cache | 11.19 tokens/s | 10.1-12.3 tokens/s | 42,793 MB |

All six candidate responses match the corresponding baseline text exactly. The
candidate is **66.8% faster**, with **46.6% fewer file-tier bytes** in this workload.
These arms have equal GPU and profile-cache allocations; the candidate uses extra
host RAM for its recent cache. This is not an equal-total-memory comparison.

Giving the original engine a 12 GiB profile cache instead produces 9.15 tokens/s.
The new 8 + 4 GiB arrangement is **22.2% faster** with **10.7% fewer file-tier
bytes** than that larger original cache. Both runs have the same GPU allocation,
prompts, 160-token output caps and two rounds. The 12 GiB original produces
different response text, so this is a workload comparison, not exact-token
replay. Its once-per-second RAM minimum is 3.69 GiB. The nominal expert-cache
budgets both total 12 GiB, but actual total memory is not identical: the recent
target includes existing assembly buffers, whereas the profile cache is separate
from them. The 12 GiB original arm runs last, after the candidate's validation.

All six model smoke checks pass: streamed echo, arithmetic, conversation history,
a generated Python function, tool calling, and use of a tool result. These short
replies are more variable: 1.4-4.6 generation tokens/s in this run. In particular,
the 56-token tool-result answer takes 65.8 seconds including prompt processing.
The longer-response gain does not establish an improvement for every prompt or
an OpenCode session end to end.

With the same 131,072-token context limit, a fresh **129,023-token prompt** returns
all three records placed near 5%, 50% and 95% of the text. Total time is **990.279 s**
(16m 30s): prompt processing is 131.1 tokens/s, and its 27-token answer is 4.9
tokens/s. The follow-up correctly recalls the last record and calculates 17 x 23;
it reuses 129,049 prompt tokens and finishes in **7.457 s**, generating 14 tokens
at 7.0 tokens/s. The result filename uses the requested 129,024-token target; the
actual token count is one below it. See the [raw validation results](fixed-recent-4g-validation).

The once-per-second monitor across the complete candidate run records at least
**3.64 GiB available physical RAM**. The separate 10-second Windows monitor starts
during the benchmark and covers all smoke and long-context checks: peak system
commit is **46.39 GiB / 54.88 GiB**, peak pagefile use 0.12 GiB, and peak engine
working set 14.69 GiB. Its sampled RAM minimum is 3.86 GiB; the finer monitor catches
a lower transient. The machine retains its existing 24 GiB pagefile. ChatGPT is
open during testing. No allocation failure or output loop occurs in these checks.

These are desktop measurements, not a validation of the laptop's memory settings.
The long prompt is a correctness and capacity test, not a controlled prefill-speed
comparison with earlier source snapshots.

## Local launcher

`run-unsloth-fast-128k.bat` reads the ignored `strata-unsloth-fast-128k.json` and uses
the separate candidate executable. It retains the existing API key, LAN binding
and port 8080, with an 8 GiB profile cache, 4,096 MiB recent-cache target and the
fixed GPU setting above. Stop any running Strata server before using it.

The original `run-unsloth-ud-q2_k_xl-128k.bat` and `engine/strata-unsloth.exe` remain
available for comparison. The benchmark servers are temporary and stop after
their measurements. Model files and quantization are unchanged.

## Reproduce

From the checkout, after compiling the separate candidate executable:

The measured build uses Release, CUDA architecture 89, CUDA 13.4 and the existing
`third_party/llama.cpp` dependency. With the Windows compiler environment loaded,
the CMake configuration has `STRATA_ENABLE_CUDA=ON`, `STRATA_BUILD_TESTS=ON`,
`CMAKE_CUDA_ARCHITECTURES=89`, and `STRATA_GGML_DIR=third_party/llama.cpp` (absolute
path). Build targets `strata` and `file_expert_source_test`; copy the candidate to
`engine/strata-unsloth-file-cache.exe` while preserving the baseline executable.
The `file_expert_source_test` run passes, including exact-byte cache reuse,
eviction and reopen checks. Both binaries' hashes are in the build metadata.

Then run the measurements:

```bat
.venv\Scripts\python.exe bench\results\2026-10-09-unsloth-file-cache\benchmark.py --name baseline-8g
.venv\Scripts\python.exe bench\results\2026-10-09-unsloth-file-cache\benchmark.py --name baseline-10g --ram-gib 10
.venv\Scripts\python.exe bench\results\2026-10-09-unsloth-file-cache\benchmark.py --name recent-2g --exe engine/strata-unsloth-file-cache.exe --cache-mib 2048
.venv\Scripts\python.exe bench\results\2026-10-09-unsloth-file-cache\benchmark.py --name fixed-baseline --expert-cache 4300 --rounds 2
.venv\Scripts\python.exe bench\results\2026-10-09-unsloth-file-cache\benchmark.py --name fixed-recent-4g --exe engine/strata-unsloth-file-cache.exe --cache-mib 4096 --expert-cache 4300 --rounds 2 --validate
.venv\Scripts\python.exe bench\results\2026-10-09-unsloth-file-cache\benchmark.py --name fixed-baseline-12g --ram-gib 12 --expert-cache 4300 --rounds 2
```

The harness reads paths and the API key from the ignored 128K configuration, writes
its temporary configurations/logs under `build-unsloth`, and binds its server to
localhost. Raw result JSONs contain responses and timings, without credentials.
Choose a new `--name` when retaining existing results.
Use `--config PATH` to supply an equivalent local 128K configuration on another PC.
The optional model checks are kept in [validation.py](validation.py); the large
prompt uses public source text from the checkout via `tools/needle_bench.py`.
Run `python bench/results/2026-10-09-unsloth-file-cache/summarize.py` after the
servers and memory monitor exit to regenerate the aggregate JSON summaries.
