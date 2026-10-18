# Unsloth UD-Q2_K_XL (experimental)

Setup offers `UD-Q2_K_XL` as an experimental choice. Full-model smoke tests pass on an RTX 4050 Laptop 6 GB
with the manual memory settings below; setup's default configuration does not fit that card. These are limited
runtime checks, not a quality benchmark or a guarantee on other PCs. Setup requires engine 0.1.40 or newer
for this experiment. Images are disabled; AMD use is unvalidated.

## Shared source and local profiles

Use `origin/main` from `https://github.com/corbinjurgens/Strata.git` on both the RTX 4050 laptop and
RTX 4080 SUPER desktop. The shared source combines the laptop allocation and memory-safety work with
the desktop cache work. Hardware differences belong in each PC's generated profile, rather than in
separate source branches. The dated results below describe their original engines and conditions.
The 2026-10-16 desktop experiments use the merged engine; earlier measurements do not establish
runtime acceptance of that engine, and the laptop still needs its own merged-engine acceptance.

Build and prepare locally using the [hardware-aware preparation steps](#hardware-aware-preparation-for-both-pcs).
Each PC needs its own executable and build receipt, model paths and native pack, runtime libraries,
credentials and generated profile. Do not copy the desktop JSON or its fixed cache budgets onto the
laptop. Preserve existing working profiles, launchers and their selection while preparing a separate
candidate. The automatic policy keeps 131072-token capacity and eager CUDA loading by default; memory
admission and runtime checks still apply. The dedicated-use desktop settings below remain opt-in.
New profiles default to authenticated loopback at `127.0.0.1:8080`; retain authentication if deliberately
enabling access from another machine.

## Files and formats

The source is [Unsloth's UD-Q2_K_XL at revision 38bb39e](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q2_K_XL).
The files total 78,869,128,864 bytes (78.9 GB). Sizes and SHA-256 come from that revision's Hugging Face tree API;
setup checks both after download or when using files supplied with `--gguf-dir`.

| Shard | Bytes | SHA-256 |
| --- | --- | --- |
| 1 of 3 | 10,946,624 | `a4f3b21e77353999829f2f767e9ac21ce9c71d29a74f2cc9eda48c9bf23c8b86` |
| 2 of 3 | 49,979,779,296 | `2e3bf1ee7d2a04e261e9f342a2d968f696cce5941d082b0e434deb9b1edc12c6` |
| 3 of 3 | 28,878,402,944 | `ec8c106759fdf4f463039c34c0707718d7d8908d53d892bd4f002e71620803f9` |

All shards use the filename `Qwen3.8-Flash-Next-UD-Q2_K_XL-0000N-of-00003.gguf`, with `N` replaced by the shard
number. Keep their original filenames together in one folder.

Headers inspected on 2026-10-07 with `tools/gguf_reader.py` contain no Q2_K tensors despite the filename:

- Routed experts: IQ2_XS gate/up in 47 layers, IQ3_XXS gate/up in one, IQ4_NL down throughout; 46,084,915,200 bytes.
- Attention and shared-expert projections: Q5_K, Q6_K and Q8_0. Token embedding: Q5_K. Output head: Q4_K.
- PLE table: IQ4_NL; PLE key: Q8_0. Small projections and norms also use Q8_0, F32 and BF16.

These use existing CPU and GPU paths. Setup packs with `--compat-bf16`, as for the other Unsloth choices:
small projections required in BF16 are converted with rounding recorded in `conversions.json`; the experts
and native quantized projections retain their stored formats. No `experts.bin` copy is made.

## Compatibility checks

[Upstream PR #911](https://github.com/Niko1221/Strata/pull/911) identified the same expert formats and called out
the Q4_K output head and Q5_K token embedding for runtime checks. Its header inspector was
[ported into v0.1.40.2](https://github.com/Niko1221/Strata/pull/911#issuecomment-6037868027); it classifies this
model as untested. [PR #844](https://github.com/Niko1221/Strata/pull/844) explores a setup-only integration of
UD-Q3_K_XL, another mixed-format file. Its results concern that larger model and different hardware.

Local checks on 2026-10-07, using engine source 0.1.40, CUDA 13.4.92 and an RTX 4050 Laptop GPU (6 GB):

- Offline synthetic split-GGUF packing preserves the head, embedding, dense and expert formats, retains the
  correct per-shard expert offsets, and converts the required small projections to BF16. Source files stay
  unchanged, and the normal pack does not copy the experts.
- `native_expert_parity` passes IQ2_XS/IQ4_NL, IQ3_XXS/IQ4_NL and Q5_K/Q8_0 against its GGML-dequantized reference.
  CPU relative errors are 0.0137-0.0146 and GPU errors 0.0115-0.0130, below its 0.03 limit. Dequantized values and
  embedding rows match the reference exactly.
- `mmvq_multi_parity` passes its Q4_K, Q5_K and Q6_K cases, among the other formats tested. Its exact mode has
  no differing or non-finite outputs across 148,480 values. This checks the multi-column arithmetic against the
  single-column kernels; it is not an independent reference for a complete model.
- `native_head_embed_test` loads a Q4_K head and Q5_K embedding from split GGUFs with a metadata-only first shard
  and nonzero tensor offsets. The head's relative L1 errors against a GGML-dequantized CPU matrix product are
  0.003612, 0.008655 and 0.004889 (limit 0.02). Both single-token and batched embedding gathers match the CPU
  reference exactly. This exercises the actual loaders and GPU entry points with small synthetic weights.

The offline packer checks need the normal Python dependencies but no GPU or model download:

```bat
.venv\Scripts\python.exe -m unittest tools.test_unsloth_q2_pack tools.test_iq_pack tools.test_setup_unsloth
```

For an existing CUDA build, build `native_expert_parity`, `mmvq_multi_parity` and `native_head_embed_test`, then
run the executables from its output directory:

```text
native_expert_parity --synthetic iq2_xs/iq4_nl iq3_xxs/iq4_nl q5_K/q8_0
mmvq_multi_parity
native_head_embed_test
```

These small fixtures establish format compatibility for the paths tested. The full-model checks below use
the actual downloaded files.

## Full-model checks on a 6 GB GPU

On 2026-10-08, the pinned files passed SHA-256 verification through Hugging Face snapshot symlinks at revision
`766911a6b7369840a91dbcd95f9f997acaab6cd6`. Their hashes match the older pinned revision above. The packer retained
the GGUF files in place, including the experts split across shards in layer 19. It produced a 1.38 GiB weight arena
and the normal tokenizer and native-expert index.

Hardware: RTX 4050 Laptop 6 GB, i7-13700H, 64 GB RAM, Windows, NVIDIA driver 616.92, release engine 0.1.40.
Windows had no page file. The requested 40 GiB expert RAM budget was clamped to 36.64 GiB by the engine's commit
headroom check; 6.3 GiB of experts remained in the file tier, which could use the OS file cache.

At 8K, the default draft model and FP16 KV configuration was 897 MiB short of the minimum expert cache.
Removing `--mtp` and using int8 KV reduced the shortage to 307 MiB. Using `--prefill 64 --no-prefill-borrow`
then allowed startup: a one-slot GPU expert cache and 206 MiB of VRAM free after loading. This uses the existing
engine; no new kernels or weight conversions were needed beyond setup's normal `--compat-bf16` pack.

The 8K smoke tests passed exact text reproduction, recall from conversation history, arithmetic (`17 * 23 = 391`),
and a Python function preserving order while removing duplicates. Responses generated at 13.1-15.2 tokens/s;
the 150-token coding answer ran at 13.7 tokens/s. These short requests do not establish broader answer quality.

For this 8K trial, the generated config's `args` used the following memory settings, with the `--mtp PATH` pair
removed (keep `--spec 4`; serving native packs requires at least 2):

```text
--max-context 8192 --kv int8 --prefill 64 --no-prefill-borrow --expert-cache auto --resident-budget-gib 40
```

At context 8192, setup does not write a `--kv` argument even if requested: add it directly to the config for this
trial. The separate prompt buffers reduce the minimum cache requirement but process prompts in small chunks.
Normal starts use the saved config; running setup again can restore its larger defaults.

For 64K, int8 KV with 20,480 resident cells was still 65 MiB short at cache sizing. Switching to `q4_0` allowed
startup and a correct short answer, with only 16 MiB reported free after loading. The draft model remained off:

```text
--max-context 65536 --kv q4_0 --kv-resident 20480 --prefill 64 --no-prefill-borrow --expert-cache auto --resident-budget-gib 39
```

The engine kept 36.30 GiB of experts in RAM after its commit headroom check. The KV resident limit is a GPU cache
size, not a limit on the context the model can access. Q4 KV uses less memory and changes precision; see the
[KV measurements](DETAILS.md#speed-measured). This card has very little VRAM margin, so these settings remain an
experiment rather than a general recommendation for 6 GB GPUs.

A retrieval test with `tools/needle_bench.py --lengths 40k --depths 50` passed: the model returned exactly
`meadow-copper-504` from a 40,285-token document, with zero reused prompt tokens. Fresh prefill took 999.903 seconds
(40.3 tokens/s); the nine-token answer took 0.684 seconds (13.2 tokens/s). End-to-end time was 1,000.7 seconds,
about 16 minutes 41 seconds. The engine reported 113 MiB of KV reads from RAM. This checks one prompt at one
needle depth; it does not establish recall quality across arbitrary long documents or a full 64K prompt.

At 128K, the same Q4 KV, 20,480 resident cells and small-prompt-buffer settings failed startup: the engine reported
65 MiB short at minimum cache sizing. The tested configuration was restored to 64K. More context needs further
memory savings or a larger GPU. No explicit reserve override was used.

### Retry after restarting Windows

A restart on the same day did not remove the 128K cache-sizing failure: CUDA still reported 0.43 GiB free at
that stage and a 65 MiB shortfall, despite `nvidia-smi` showing more idle VRAM before launch.

96K started after reducing prompt chunks and the verifier window. MTP remained off; suffix drafting was also
disabled so it would not expand `--spec 2` to a larger window:

```text
--max-context 98304 --kv q4_0 --kv-resident 20480 --prefill 16 --no-prefill-borrow --spec 2 --suffix-draft 0 --expert-cache auto --resident-budget-gib 39
```

The engine kept 36.97 GiB of experts in RAM and used a two-token verifier with 44.6 MiB of device buffers.
It reported **0 MiB of VRAM free** after loading. A 94-token conversation-history/arithmetic request returned
the correct codename and product, generating 44 tokens at 14.3 tokens/s; prefill was 17.4 tokens/s.
At the end of that trial, the saved local configuration was left at 96K. This verifies startup and a short
response only: a full 96K prompt has not been tested, and the lack of VRAM headroom makes this a fragile
experimental setting.

### Local allocation changes on the same GPU

The local engine now packs native projection weights and their shared activation scratch into one aligned GPU
allocation. It also sizes prompt selection/attention scratch to the configured chunk, packs common prompt buffers
together, and shares weight scratch between dense projections and routed experts whose work does not overlap.
The copy-stream upload ring and cuBLAS workspace remain separate. The dense GEMM workspace keeps its original
capacity; these changes preserve the stored weight bytes and existing arithmetic kernels.

Automatic expert-cache retries now retain the owned-prompt, draft-binding and pipeline allocations reserved by
the initial sizing. Previously, the Windows post-touch check omitted the owned-prompt and draft-binding parts.
`STRATA_OWNED_PRICE=exact` still needs an explicit reserve and runtime measurements to preserve useful headroom;
its allocation estimate does not include every first-use library allocation.

On 2026-10-08, two local CUDA 13.4.92 / MSVC 19.43 builds were compared with identical 64K/Q4 settings, a fixed
one-slot expert cache, prompt chunk 64, two-token verifier and MTP/suffix/lookup drafting disabled. The comparison
build predates the allocation changes; both builds contain the reserve-accounting correction, which does not
affect the fixed-cache comparison.

| Startup checkpoint | Before allocation changes | After allocation changes |
| --- | --- | --- |
| Weights and session loaded | 552 MiB free | 868 MiB free |
| Prompt path initialized | 84 MiB free | 470 MiB free |
| Verifier initialized | 38 MiB free | 424 MiB free |

The observed final recovery was 386 MiB. A 78-token history/arithmetic/code request produced the same 83-token
answer byte for byte. Prompt rates were 25.5 and 25.3 tokens/s; generation rates were 12.1 and 12.6 tokens/s.
These single short samples do not establish a throughput improvement. Windows retained 34.91 versus 35.09 GiB
of RAM experts, so host/file-tier conditions were not identical. No output-head offload or HC requantization was
needed for this memory reduction.

The focused allocation tests cover mixed-format native weights, optional packed Q8 copies, failed-load rollback,
owned prompt allocation/reset, borrowed relayout, exact-sized selection/attention buffers and forced MMQ fallback.
They passed on this GPU, as did the existing native expert, head/embedding and multi-column parity checks. CUDA
Compute Sanitizer could not attach to the fixture process on this Windows setup; sanitizer coverage remains
unverified. Multi-GPU/HIP behavior was reviewed in source but was not tested on hardware.

### Validated 64K configuration after the allocation changes

The same RTX 4050 Laptop 6 GB / i7-13700H / 64 GB Windows PC, without a pagefile, passed a fresh 58,119-token
retrieval test and a conversation-reuse check with these settings. The GPU was freed of other model workloads
before starting. Set `STRATA_OWNED_PRICE=exact` in the server config's `env` object as well as these engine args:

```text
--max-context 65536 --kv q4_0 --kv-resident 20480 --prefill 256 --no-prefill-borrow --spec 2 --suffix-draft 0 --lookup-chain 0 --expert-cache auto --vram-reserve-mib 448 --resident-budget-gib 39 --pool-workers 9
```

MTP and images remain disabled. The engine selected 39 GPU expert slots, kept 34.95 GiB of experts in RAM after
the commit-headroom check, and reported 294 MiB free after initialization. That free-memory figure is a startup
checkpoint, not a continuous measurement of headroom during requests.

| Check | Measured result |
| --- | --- |
| Fresh 58,119-token prompt, codes near 5%, 50% and 95% of the text | All three codes correct; no reused tokens |
| Fresh prompt processing | 595.249 s, 97.6 tokens/s |
| 27-token answer | 1.759 s, 15.3 tokens/s; 597.251 s end to end |
| Follow-up recalling the final code and calculating 17 x 23 | Both correct; 58,145 cached tokens, 34 newly read; 3.029 s end to end |

This exercises a prompt near the 64K limit and reuse beyond the 20,480-cell GPU KV cache. It is a retrieval and
stability check, not a broad model-quality evaluation. For the same 2,137-token needle prompt, chunk 128 processed
at 63.9 tokens/s and chunk 256 at 96.2 tokens/s; both retrieved the code. These are single-run measurements.
The earlier 40.3 tokens/s long-prompt result used a different build and prompt length and is not a controlled
throughput comparison.

The local `run-unsloth-ud-q2_k_xl.bat` now uses this 64K configuration through `engine/strata-low-vram.exe`, built
with CUDA 13.4.92 for this card. The original executable remains available, and the previous JSON is saved as
`strata-unsloth-ud-q2_k_xl.before-memory.json`. Setup may regenerate the normal JSON; these are local settings,
not new setup defaults.

### 128K startup check after the allocation changes

With the new build, 128K started with 244 MiB free, 16 GPU expert slots and 34.75 GiB of RAM experts. It used
the same settings above except `--max-context 131072 --prefill 64 --vram-reserve-mib 400`. A fresh 2,137-token
needle test passed: prefill took 50.934 s (42.0 tokens/s), and nine output tokens took 0.691 s (13.0 tokens/s).
The engine still reports low headroom at this level. A full 128K prompt has not been tested, so this does not
by itself establish long-prompt stability at 128K. The longer conversation below was observed later.

The optional local `run-unsloth-ud-q2_k_xl-128k.bat` uses `strata-unsloth-ud-q2_k_xl-128k.json`. Close the 64K
server before starting it; both use port 8080. The normal launcher stays at the faster, long-prompt-tested 64K.

### Longer 128K conversation and desktop memory pressure

On 2026-10-09, the user reported sustained use of the 128K configuration for about 1 hour 50 minutes, including
client-side compaction. The engine log corroborates substantial context use: a completed request had 115,860
prompt tokens and generated 3,583 more at 16.1 tokens/s. The next request had 18,606 prompt tokens with no reused
prefix, consistent with that compaction, and subsequent requests continued. This was a growing conversation with
cached prefixes; it was not a fresh 115,860-token prefill or a test filling every one of the 131,072 context slots.

The user also reported Chrome and Codex crashes. During inspection on the same laptop, system commit was about
60.3 of 63.71 GiB (94%), only 3.2 GiB of physical RAM was free, and no pagefile was configured. Strata held about
44.35 GiB of private committed memory, including 34.26 GiB of pinned experts. Windows resource-exhaustion event
2004 preceded a Chrome crash on October 8 and a Codex crash on October 9 by roughly two seconds in each case;
both events identified Strata as the largest memory consumer. This strongly supports system commit exhaustion
as a contributor to those crashes. GPU usage alone does not establish their cause.

The requested 39 GiB expert budget had already been clamped to 34.26 GiB at startup, leaving the default 4 GiB
headroom plus a 0.25 GiB margin against available commit. That was insufficient for concurrent desktop apps in
this run. Running the model successfully therefore does not establish comfortable desktop multitasking.
One unmeasured next trial is `--resident-budget-gib 30` with `STRATA_RESIDENT_HEADROOM_GIB=8` in the config's
`env` object, retaining the working context/GPU settings. This would reserve more RAM for applications at the
cost of more expert reads from the SSD. These proposed settings have not been applied or benchmarked.

## Combined branch and upstream fixes, 2026-10-14

At this stage, `codex/unsloth-low-vram` incorporated the desktop's `codex/unsloth-file-cache` work (`411ee540`)
alongside the laptop's allocation changes. This history is now included in `origin/main`. The hardware-aware
helper selects a local profile; the desktop's separately trained placement remains an explicit experiment
rather than a hardware default.

The branch selectively adapts upstream's [batched prompt reads](https://github.com/Niko1221/Strata/commit/0b030b96fc6a13a3f815ec25f4d6491f47745b1c)
and Windows commit checks from [its memory fix](https://github.com/Niko1221/Strata/commit/5728828a8498a4581f51a61f5a64805174648c0f).
`STRATA_STAGER_BATCH=8` lets the prompt stager group consecutive jobs through the existing direct reader.
The branch keeps the default at one job because larger batches retain more per-worker RAM and have not yet been
measured with the combined policy. Prompt copies do not populate the desktop's recent decode-expert cache.

Upstream also closes unused GGUF mappings automatically, but this branch retains the desktop's tested opt-in
`STRATA_FILE_DIRECT_UNMAP` behavior and buffered positional fallback. Its earlier desktop trial did not improve
performance. The combined reader uses one mapping lifetime and file-offset representation for both paths.

Conversation parking and session save/restore now consider the smaller of physical RAM and available Windows
commit. A startup diagnostic reports commit headroom and warns when it cannot cover the conversation cache,
a 2 GiB checkpoint/staging allowance and 4 GiB headroom. This warning is not a global memory cap and does not
reserve that space for other applications. The expert RAM budget still needs to leave room for the desktop.

Upstream `v0.1.44.1` (`296b59d2`) still excludes UD-Q2_K_XL from setup and does not contain this branch's native
weight arena or compact prompt buffers. A full merge also brings inference-default and kernel changes, so those
are left for separate validation. In particular, no route-tail skipping or HC requantization is enabled here.

The combined source built on the RTX 4050 laptop with CUDA 13.4.92 / MSVC 19.43. On 2026-10-14, 270 offline
Python tests and eight focused CTests passed, covering setup, hardware budgets, explicit overrides, cache I/O,
native weights/head/embedding and prompt allocations. The cache rotation fixture also passed exact-byte checks
for pinned, rotated and pageable storage. Split-GGUF batch tests cover direct reads, retired mappings, buffered
fallback, injected read failures and reopening. A preparation dry run detected the laptop and preserved its
authenticated LAN settings. No full model was loaded; combined-policy performance, live prompt batching and
desktop runtime validation remain outstanding. Existing laptop launchers continue to select their working engine.

## Original manual setup

The historical manual trials used upstream `v0.1.40.1` (`82f46a8`) with the selective backports
described above. The [hardware-aware helper below](#hardware-aware-preparation-for-both-pcs) prepares a separate
automatic profile from the current shared source. The earlier manual method remains available. For a separate
installation, clone shared `main` into a new folder:

```text
git clone --branch main --single-branch https://github.com/corbinjurgens/Strata.git
cd Strata
```

For files already downloaded, run setup from that checkout in a Windows command prompt:

```bat
START-HERE.bat --setup --build --family unsloth --model UD-Q2_K_XL --gguf-dir "D:\Models\UD-Q2_K_XL" --context 8192 --draft-vocab en --vision no --no-start
```

Replace the folder with the one containing the shards. Omit `--gguf-dir` to let setup download them. `--no-start`
prepares the files and config without loading the model. **Keep `--build`:** it compiles this checkout's source
for the detected GPU. Without it, setup can download the upstream release engine, which lacks these allocation
changes. Building requires the normal CUDA/MSVC tools on Windows; setup guides their installation.

On a 6 GB GPU, apply the manual settings above to the generated `strata-unsloth-ud-q2_k_xl.json` before starting
`run-unsloth-ud-q2_k_xl.bat`: replace the corresponding flags in `args`, remove the `--mtp PATH` pair, and add
`"STRATA_OWNED_PRICE": "exact"` to the config's `env` object. Preserve setup's paths to the engine, model and
tokenizer. A source build writes `engine/strata.exe`; the separate `strata-low-vram.exe` name and optional 128K
launcher described in the measurements are local trial artifacts and are not included in a clone. To try 128K,
change the generated config's context, prompt chunk and reserve as described above. Linux uses `./setup.sh`
with the same flags and a `.sh` start script.

The 39 GiB RAM request and nine pool workers in the measured commands belong to the 64 GB/i7 laptop. On another
PC, retain setup's hardware-specific RAM budget and worker count as the starting point; in particular, do not
copy a 39 GiB RAM request onto a 32 GB machine. The desktop validation below measures this fork's loading and
128K operation. The memory savings of the individual allocation changes compared with upstream are unmeasured there.

Setup applies the existing Unsloth RAM budget: system RAM less 24 GiB, capped by the experts' size, with a streamed
context cache deducted. On a 64 GB PC this gives 40 GiB at 8K context and 38 GiB at 128K with int8 KV streaming.
The remainder of the experts is read from the model files. This RAM calculation does not establish GPU fit:
the fixed weights, draft layer, context, prompt buffers and GPU expert cache must also fit.

Start validation at 8K to separate loading failures from context memory pressure. After coherent short answers and
a multi-turn conversation work, try larger contexts separately and check both startup and a long prompt. A configured
context limit alone does not establish that a prompt of that length fits or runs at a usable speed.

## Hardware-aware preparation for both PCs

`tools/prepare_unsloth_q2.py` creates a separate local profile for the same UD-Q2_K_XL model, with
**131072-token context capacity by default**. It detects RAM, available physical memory, Windows available
commit and GPU VRAM, then reports its allocation requests and reserved allowances. It enables the compact
recent-file cache from this branch. This is an explained startup policy, not a guarantee of optimal speed or
of fit for every workload. The guarded RTX 4080 SUPER workload below passed on 2026-10-15. Full laptop
128K and Linux acceptance remain unverified.
The historical laptop and desktop measurements used their stated manual settings.

The helper defaults to `engine/strata-unsloth-portable.exe`, `engine/BUILD-unsloth-portable.json`,
`strata-unsloth-portable-128k.json` and `run-unsloth-portable-128k.bat`. That executable is also the historically
selected desktop engine below, so its name does not establish isolation. For this candidate, use
`engine/strata-shared-memory-safety.exe`, its matching `engine/BUILD-shared-memory-safety.json` receipt and
`strata-unsloth-shared-memory-safety-128k.json`. Check the executable paths in locally selected configs before
building; if this candidate path is selected, choose an unused numeric suffix before the extension. Generate
the configuration on each PC; copying a desktop JSON would carry its paths and budgets to the laptop.

### Reproduce from source and existing model files

Use a checkout of shared `origin/main` containing the current engine, setup and helper changes, including
`tools/prepare_unsloth_q2.py`. Update an existing checkout only after preserving its local changes and working
profile selection, or use the separate clone above. Ignored executables, generated configs, build
receipts and launchers must be rebuilt or generated locally. Keep API keys out of Git.

The Windows build wrapper needs the normal Python environment and installed CUDA, MSVC, CMake, Ninja and
`third_party/llama.cpp` dependencies. It reports missing prerequisites; it does not install them or download
a release engine. See [installation prerequisites](INSTALL.md). A fresh checkout must also obtain the
`LLAMA_CPP_COMMIT` tree named in `setup.py` under `third_party/llama.cpp` (including `ggml` and `gguf-py`);
the tested checkout uses `3cf03257f219afbe7334045ff7c6a06ac68c627d`. An existing copy can be reused. The
receipt hashes the actual selected files, rather than treating that version label as proof of identity.
With those prerequisites ready, run from the checkout:

```bat
.venv\Scripts\python.exe tools\prepare_unsloth_q2.py --build-engine --engine engine/strata-shared-memory-safety.exe
.venv\Scripts\python.exe tools\prepare_unsloth_q2.py --plan --engine engine/strata-shared-memory-safety.exe --config strata-unsloth-shared-memory-safety-128k.json --gguf-dir "D:\Models\UD-Q2_K_XL"
.venv\Scripts\python.exe tools\prepare_unsloth_q2.py --prepare --engine engine/strata-shared-memory-safety.exe --config strata-unsloth-shared-memory-safety-128k.json --gguf-dir "D:\Models\UD-Q2_K_XL"
.venv\Scripts\python.exe tools\prepare_unsloth_q2.py --check-ready --config strata-unsloth-shared-memory-safety-128k.json
```

Replace the model folder with the existing three-shard folder. Omit `--gguf-dir` to discover a complete
Unsloth snapshot under `HF_HUB_CACHE` or the current user's Hugging Face cache. The helper never downloads
model shards. It verifies their pinned sizes and SHA-256, reusing an unchanged verification receipt in
`packs/unsloth-ud-q2_k_xl/source-verification.json`. If the native expert index and tokenizer already exist,
it reuses the pack. Otherwise it runs the normal `--compat-bf16` packer once. It does not create an
`experts.bin` copy or change the native expert quantization. Failed verification leaves source files intact.

The build defaults to a separate `build-unsloth-portable` directory. To reuse an existing compatible local
build tree, use `--build-engine --build-dir build-unsloth --jobs 4`. `--engine PATH` chooses another separate
output executable. The build receipt records the binary hash, engine source fingerprint, selected GGML
source fingerprint, GPU architecture, commands and effective CMake settings. Changed source during build,
a stale source receipt or a mismatched executable prevents installation or policy-profile launch. Build on
each target PC so its detected GPU architecture is used.

To inherit an existing machine-local API key and authenticated host/port, add
`--key-config strata-unsloth-desktop-128k.json` to the preparation command on this desktop, or name the
appropriate local credential config on the other PC. Without a usable saved key, the helper generates a
new random key and defaults to `127.0.0.1:8080`. Keys are saved only in the ignored local JSON and are not
printed. Existing keyed host/port settings are preserved; readiness refuses an unkeyed non-loopback host.

`--plan`, `--prepare`, `--write-config` and `--check-ready` do not start a server. The generated launcher
for these candidate commands is `run-unsloth-shared-memory-safety-128k.bat`; it uses
`--launch --config strata-unsloth-shared-memory-safety-128k.json`. Running it is the separate full-model launch
step, after offline preparation. Close an existing Strata server before starting another profile.
For another experiment, pass a distinct `--config`
filename consistently. `--context` affects a new profile only, accepts 8192/ 32768/65536/131072, and is never
reduced automatically. Existing configurations are preserved during preparation.

### Budget policy and overrides

At 131072 context, the default GPU choices are:

| Setting | RTX 4050 Laptop 6 GB | RTX 4080 SUPER 16 GB |
| --- | --- | --- |
| KV storage / GPU resident cells | q4_0 / 20480 | int8 / 32768 |
| Prefill chunks | 64, separate buffers (`--no-prefill-borrow`) | auto, normal borrowing |
| Explicit VRAM reserve | 400 MiB at 128K; 448 MiB at 64K or below | 1024 MiB |
| Owned-buffer pricing | `STRATA_OWNED_PRICE=exact` | normal pricing |
| MTP, suffix and lookup drafting | off; verifier window 2 | off; verifier window 2 |

These small-GPU controls apply below 8 GiB detected VRAM. Q4 KV changes the precision of cached attention
keys/values; it does **not** requantize the UD-Q2_K_XL model weights or lower the context capacity. The
resident-cell limit is a GPU cache size, not the amount of context the model can access.

The expert pool is the smaller remaining physical/commit budget, capped at 42.9 GiB. Before assigning it,
the helper allows 2 GiB for CPU/server use, 0.5 GiB for later startup buffers on a small GPU or 1 GiB on the
desktop, streamed KV storage for 13 layers at the selected context/precision, and application headroom of
8 GiB on systems with at least 48 GiB RAM or 4 GiB below that. The Windows commit calculation also reserves
GPU memory less the configured VRAM reserve; this GPU allowance is not subtracted again from physical RAM.
If available-memory telemetry is missing, the helper discloses its fallback estimate of 65% of total RAM.
Missing Windows commit telemetry falls back to the physical pool for planning. These fallbacks are labelled
unverified: read-only planning and preparation remain available, but cannot establish launch feasibility.
These are policy allowances, not measured worst-case memory requirements. The laptop's 8 GiB headroom applies
to system RAM; its 400 MiB GPU reserve is separate and has not been validated for normal external-monitor use.
Everyday laptop acceptance keeps the external monitor connected and normal applications open, and samples
actual free VRAM rather than assuming all 6 GB is available.

The pinned profile-cache request is 70% of that pool, rounded down to whole GiB and capped at 30 GiB on the
larger-RAM machine or 12 GiB below 48 GiB RAM. The remainder requests compact recent-cache storage, capped at
16384 MiB. Under severe pressure the helper requests a small positive 0.25 GiB pinned budget and can reduce
the recent cache to zero; it reports the low-memory condition. The engine still clamps requests at startup.
Profile requests are ceilings subject to allocation checks. The engine finalizes the recent target after
required startup buffers and checks current physical RAM and Windows commit on retention misses. Optional
growth always leaves the existing 256 MiB clamp margin above the requested headroom, including when the
configured target itself fits. Under pressure it reuses fitting safe buffers and releases only eligible cold
entries; existing capacities are already charged to the memory sample. Recent retention is a lazy target: protected live entries and mandatory
staging may exceed it. The pinned expert budget remains fixed after launch. These checks cannot reserve
memory against later allocations or other applications.

On Windows/NVIDIA, automatic expert-cache startup with a requested reserve of at least 256 MiB also checks
the final CUDA allocation budget before serving. It refuses startup below the existing 256 MiB stall-warning
floor, or when that memory query fails. An explicit cache size or a deliberately smaller reserve retains
its existing warning behavior. This final check includes runtime/library allocations absent from the early
cache forecast; increasing the reserve cannot create room when fixed allocations already fill the GPU.

After resampling, a launch with both host-cache requests still automatic requires current available physical
memory, available Windows commit on Windows, and at least 0.25 GiB in the computed expert pool. A measured zero
is insufficient capacity, rather than unknown telemetry. Linux does not require Windows commit. Rejection
preserves the saved config and prevents engine creation and HTTP serving. The helper may already have started
the Python server process to perform those checks. The below-2-GiB warning alone does not reject a plan.

The saved `memory_policy` opts a profile into automatic preparation. Both the helper's `--launch` path
and direct server startup from that config use the same preparation before loading the engine. Hardware
and memory are sampled once per server start. The saved profile is an automatic template: preparation
checks admission and readiness, then returns refreshed managed values for that server's lifetime without
rewriting the saved JSON. This avoids overwriting concurrent user edits. Edits observed during preparation
cause refusal, but an edit made after those checks is not guaranteed to block startup. An idle reload or
recovery after an engine crash does not promise another policy refresh; restart the server to plan again.
The preparation commands still create and save new profiles. Configurations without `memory_policy`
remain static/manual.
Automatic profiles select one GPU index. A temporary server `--gpu` selection is used for that launch's
policy sample and engine without replacing the saved GPU selection. An explicit `CUDA_VISIBLE_DEVICES`
that matches the selected GPU is accepted; conflicting or invalid visibility selections or ordering are
rejected before engine creation and HTTP serving. Static/manual profiles retain their multi-GPU path.

The policy records which values are automatic. User edits saved in its managed arguments/environment
are treated as overrides on later starts; other config values, credentials and server settings are
preserved. Supported cache and KV argument removals and managed environment removals remain explicit
overrides. An automatic profile requires an explicit, supported `--max-context` capacity: if it is missing
or invalid, correct the saved argument before starting. A static profile can use the engine's defaults.
A manual pinned budget reduces an automatic recent-cache request, and an explicit recent request limits
an automatic pinned request. To mark
deliberate new-profile overrides even when they equal the current automatic choice, use
`--resident-budget-gib`, `--file-cache-mib` or `--resident-headroom-gib` during creation. Oversized explicit
requests remain explicit and are reported; this does not bypass engine safety clamps. Remove `memory_policy`
only when a completely static configuration is intended.

An explicit override or removal of either `--resident-budget-gib` or `STRATA_FILE_CACHE_MIB` retains the manual
launch path and its warnings, including across repeated refreshes. Profile requests, when present, must be
finite and positive; recent MiB must be a nonnegative integer; host headroom and VRAM reserve must be finite
and nonnegative. Zero recent retention and zero headroom remain valid. The same validation applies to new
CLI overrides and edited policy configs before profile creation or engine launch.
JSON null in a managed environment setting means explicit removal: that key is omitted from the runtime
environment across refreshes. It is never passed to the engine as the string `None`.
The engine saturates oversized finite GiB-to-byte requests before allocation checks. A positive profile
request below one byte keeps a one-byte budget, so it cannot select the zero/full-complement sentinel;
such a budget can still fail allocation because no expert fits it.

Each managed memory-policy argument must appear at most once, including context, KV and boolean switches.
Duplicate managed arguments are rejected before profile creation or engine startup; preparation preserves
the saved bytes so the ambiguous argument list can be corrected explicitly.

Numeric strings use ASCII decimal or scientific syntax; separators such as underscores and non-ASCII
digits are rejected. Recent MiB must be a complete integer with no trailing whitespace and cannot exceed
17592186044415 MiB, the engine's byte-counter limit. Larger ordinary requests within that range remain
explicit and subject to allocation checks.

The engine reads `--vram-reserve-mib` with an integer-prefix parser: `1e3` means 1 MiB and `1024.5` means
1024 MiB. The helper preserves those manual argument spellings, uses the effective integer in its commit
allowance, and reports any requested/effective difference. An effective prefix above 2147483647 MiB is
rejected before launch to avoid signed-integer overflow. Removing the argument retains the conservative
zero-reserve allowance used by the planner.

`tools/benchmark_compact_cache.py` consumes saved settings without refreshing the portable policy. A saved
automatic template does not contain the latest server's refreshed runtime budgets. For comparisons, use
a separate frozen static config containing the intended effective settings, and record those settings
alongside actual residency. `--plan` previews a refresh without saving it. Historical benchmark results
do not validate a later automatic policy.

New profiles set `fit_max_tokens=true` and `open_browser=false`. Output-budget fitting shortens a client's
requested output allowance to the remaining context; it does not truncate the prompt or increase capacity.
The launcher uses a nonblank saved API key even if its parent environment contains a different or blank
`STRATA_API_KEY`. It omits browser opening unless explicitly enabled in the config. It clears inherited
`STRATA_VERIFY_PROFILE`, `STRATA_DECODE_TIMING` and `STRATA_FETCH_PROFILE`; values deliberately placed in the
config's `env` remain explicit diagnostic opt-ins.

The engine source includes Windows and Linux paths, and the helper can generate a shell launcher, but its
`--build-engine` wrapper currently supports Windows/NVIDIA only. On Linux, use the repository's normal
`setup.sh --build`/CMake path and local paths/configuration; the Windows build commands above are not a
validated Linux recipe. The laptop startup/cache-warmup checks below are partial runtime measurements;
Linux and the read-path experiment remain unmeasured here.

### Laptop startup and cache warmup, 2026-10-14 to 2026-10-15

An isolated candidate was tested on the RTX 4050 Laptop 6 GB with 63.71 GiB usable RAM, Windows, no pagefile,
CUDA 13.4 and driver 617.42. Normal applications stayed open; only the internal display was active. Context
capacity remained 131072, with q4_0 KV, 20480 resident cells, 64-token prefill chunks, and the unchanged automatic
host-cache split. The original engines and saved configurations were retained.

Before the runtime admission correction, available Windows commit reached 7.20 GiB during cache warmup,
despite the 8 GiB startup headroom request. The corrected watched run kept available commit above 8 GiB;
the user reported responsive typing, window switching and scrolling throughout.

With the experimental lazy-loading profile, a subsequent fresh 32767-token prompt returned all three codes
distributed through the input; the follow-up reused 32792 tokens and returned the final code plus the correct
integer. Fresh prefill took 1586.4 seconds (20.7 tokens/s). This approximately 30-minute run preceded the final
margin refinement: available commit briefly reached 7.88 GiB, with 14 one-second samples below 8 GiB.
That result prompted applying the existing 256 MiB margin consistently to optional cache growth.

The final candidate then completed three 512-token generations, correct cached follow-ups after each, and
121 seconds of idle monitoring. Across 282 seconds of telemetry, available physical RAM stayed above
17.58 GiB and available commit above 8.39 GiB, with no sample below 8 GiB. The sustained generations decoded
at 13.0 to 13.5 tokens/s. CUDA reported 424 MiB free at startup and 392 MiB after every completed request.
This final run validates the margin on that workload; the 32K retrieval run was on the preceding candidate,
and a near-full 128K laptop prompt remains untested.

GPU tracing separated the host-complement mapping from prompt initialization. With the engine's default
eager CUDA module loading, cuBLAS creation cost 220 MiB in an isolated library probe, versus 12 MiB with lazy
loading. The full model's startup CUDA-free memory was 214 MiB under eager loading and 424 MiB in a separate
profile with `CUDA_MODULE_LOADING=LAZY`. The user also reported responsiveness during that comparison.
NVIDIA documents that lazy loading postpones kernel loading; the startup saving alone does not verify all
first-use kernels or near-full-context stability. The engine's eager default remains unchanged, and the lazy
setting is an explicit experimental profile override. See [CUDA module loading](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-module-loading).
On the final candidate, direct startup probes confirmed that the eager profile was refused before `READY`,
while the lazy profile reached `READY` and exited cleanly on `QUIT`.

Local raw evidence is under `.claude/laptop-headroom-correction-20261014/`: `watched-startup`,
`lazy-comparison`, `long-retrieval`, `margin-stress`, `startup-checks-final`, and `eager-cublas-probe.json`.
Responsiveness was observed by the user during the two watched short runs; the longer automated runs were
not watched for responsiveness. These measurements do not establish external-monitor fit, RTX 4080 SUPER
acceptance, Linux runtime behavior, or desktop-stall causation.

Offline checks for policy choices, physical/commit limits, credentials, explicit overrides, repeated refresh,
publication failures, command quoting and build receipts are in
[`tools/test_prepare_unsloth_q2.py`](../tools/test_prepare_unsloth_q2.py):

```bat
.venv\Scripts\python.exe -m unittest tools.test_prepare_unsloth_q2 tools.test_setup_config
```

Historical preparation on 2026-10-09 used the RTX 4080 SUPER 16 GB, 30.88 GiB usable RAM, Windows,
driver 617.14, CUDA 13.4.92 and MSVC 19.44.35229. Its sm_89 engine built and ran its help command; all three
model hashes passed, and `--compat-bf16` packing produced the 1.38 GiB arena. Those checks establish that
preparation step, not the performance of the newer automatic policy.

## RTX 4050 comparison after the desktop changes, 2026-10-16

The pre-pull source `63a01bba` and desktop source `9c16f6f1` were rebuilt with the same
CUDA 13.4 / MSVC 19.43 toolchain, sm_89 flags and identical local GGML inputs. The machine
was the RTX 4050 Laptop 6 GB / i7-13700H, with 63.71 GiB usable RAM, Windows 11,
driver 617.42, no pagefile, an MSI M480 PRO NVMe SSD, the internal display and normal
applications including Codex open. These were automated tests; UI responsiveness was
not observed, and an external monitor was not connected.

The existing static 39 GiB pinned profile was stopped during warmup on the old source:
available commit reached 2.006 GiB and stayed below 4 GiB for five samples, despite
the engine clamping the pinned allocation to 29.26 GiB. This is separate from an
engine performance comparison. An eager-loading current-source control started at
266 MiB CUDA-free but reached 234 MiB after warmup; the guard stopped it too.
A separate baseline diagnostic, with only the current request-end `mem_mark` added,
reproduced the same 234 MiB reading under identical eager controls. This confirms
that shortfall predates the pull. Its source and original executable were restored;
the diagnostic binary was not used for performance measurements.

The matched comparison therefore used explicit `CUDA_MODULE_LOADING=LAZY`, 16 GiB
pinned experts, 4096 MiB compact recent retention and 8 GiB host headroom. Capacity
was 65536, q4_0 KV with 20480 resident cells, 256-token separate prefill buffers,
verifier window 2 and no MTP, suffix or lookup drafting. These frozen controls
used the existing GGUF-in-place pack; they are not refreshed automatic budgets.

| Source / ready-copy setting | Decode tokens/s, two runs | Fresh 2048-token prefill, seconds |
| --- | --- | --- |
| Pre-pull `63a01bba`, off | 10.30 / 10.51 | 67.642 / 64.127 |
| Current `9c16f6f1`, off | 10.16 / 10.41 | 64.124 / 66.549 |
| Current `9c16f6f1`, on | 10.32 / 10.21 | 52.452 / 53.450 |

Order was old / current-off / current-on / current-on / current-off / old. Each
run used the same warmup, two rounds of the same three 160-token generation cases,
fresh retrieval and a cached follow-up. All timed output sequences matched exactly,
all retrieval/follow-up checks passed, and each run realized 59 GPU expert slots
and 16 GiB pinned experts. Pooled decode was 10.40 before and 10.28 after; these
overlapping short-run ranges do not establish a material regression or gain.

Ready-copy reuse reduced prefill time by 18.20% and 16.65% against the faster
current-source off control. It did not establish a decode gain. This is a 2K
workload at 64K capacity, not the desktop's 8K test or a full-context result.
Across this block, available commit stayed above 11.28 GiB. Current-source stage
samples stayed at or above 420 MiB CUDA-free; the old source lacked the request-end
trace, so its 472 MiB startup sample is not a continuous or comparable runtime minimum.

The hardware-policy comparison produced identical argument/environment choices at
both revisions across 48 healthy RAM/VRAM/context/memory combinations. The Python
preparation, server configuration and benchmark suites passed 146 tests; the focused
C++ cache-retry, file-source and prefill-allocation checks also passed. Local raw
results, protected-file receipts, build provenance and harnesses are under
`build-unsloth/msi4050-20261016/`. Existing engines and saved launchers were preserved.

### Contiguous-layout pilot and refreshed automatic profile

A separate contiguous pack was built with bounded buffers and independently compared
against every original GGUF expert slice. All 46084915200 expert bytes matched; its
SHA-256 was `5a71d41249331ac874f97f998e0ec807c13318ca0e0b8a334a9c0ca4e600cc07`,
the same as the desktop artifact. Dense weights, index, layout and tokenizer hashes
also matched the installed pack. With only the pack path changed from the frozen
current-source off controls, one pilot measured 12.20 decode tokens/s and 22.807 seconds
for fresh 2K prefill. That is 17.22% faster decode and 64.43% less prefill time than
the faster current-source controls. All timed outputs and retrieval checks matched.
Preparation and verification warmed storage; this was not a cold-cache test or a
repeated storage-layout acceptance block. The additional 42.92 GiB expert file and
copied pack files were removed after measurement; receipts and reproduction code remain.

The old local pack initially lacked `source-verification.json`, so automatic startup
was refused before engine creation. All original GGUF SHA-256 checks then passed and
the verification receipt was added without rewriting the weights. The prepared eager
128K profile was subsequently refused at 216 MiB CUDA-free, below the 256 MiB floor.

A separate refreshed automatic profile using the original GGUF-in-place pack,
explicit lazy CUDA and ready-copy reuse passed arithmetic, cached answers, three
512-token generations, their cached follow-ups, 30 seconds of idle monitoring, and
fresh 2048-token retrieval plus 2073-token cached reuse. Capacity stayed 131072,
q4_0 KV and 20480 resident cells; prefill was 64. The runtime requested 17 GiB pinned
and 7562 MiB recent, with the recent target finalized at 7053 MiB. Sustained generations
measured 11.5 / 11.2 / 12.0 tokens/s. These different automatic allocations and workloads
are not a matched speed comparison against the frozen 64K controls.

Minimum physical RAM available was 19.52 GiB, commit available 7.549 GiB and logged
CUDA-free memory 378 MiB. Five memory samples during fresh retrieval were below the
requested 8 GiB commit allowance; the final memory sample recovered to 8.67 GiB.
The run passed the 2 GiB immediate / 4 GiB sustained commit guards, but does not
establish that the requested 8 GiB headroom is always retained. Stage readings are
not continuous VRAM minima. Full near-128K inputs, external displays, observed UI
responsiveness, HIP/SYCL and Linux remain unverified.

An unselected local candidate is available at
`build-unsloth/msi4050-20261016/run-automatic-128k.bat`, with an authenticated loopback
profile on port 8081. It keeps automatic memory sizing, explicit lazy loading and
ready-copy reuse; restart resamples memory. This is an experimental local candidate,
not a default change or full-context acceptance.

## Opt-in shared prefill stage on the RTX 4050, 2026-10-18

Commit `1357a2d7` adds a prototype switch that is off by default. Setting
`STRATA_PREFILL_SHARED_STAGE=1` lets the Windows CUDA prompt path with an automatic expert cache
choose a shared prefill stage instead of the separate one. Nothing changes without the variable,
with an explicit `--expert-cache`, or with a helper, peer, split, pipeline, batch or borrowed
buffer, and HIP/SYCL never select it. The automatic cache price is unchanged, so any saving appears
as free VRAM rather than extra slots. The measurement below is a local prototype result, not
adoption, and no product default changed.

Measured on the same RTX 4050 Laptop 6 GB / i7-13700H / 63.71 GiB RAM / Windows 11 / no pagefile
machine, CUDA 13.4 / MSVC 19.43, EAGER q4_0, with the short-validation sequence: warmup, exact arithmetic, cached
follow-up, three 512-token sustained generations with cached follow-ups, a fresh 2048-token
retrieval and its cached reuse.

| Capacity, reserve | Default (env unset) | `STRATA_PREFILL_SHARED_STAGE=1` |
| --- | --- | --- |
| 128K, automatic | Refused before any request: 238 MiB free after all buffers, below the 256 MiB floor | Ready at 340 MiB, all requests passed, minimum 270 MiB |
| 64K, saved 448 MiB | 234 MiB after the first request (pre-prototype run) | All requests passed, minimum 262 MiB, 32 (37) slots |
| 64K, automatic | Separate by default; no matched automatic run | Selected separate, all requests passed, minimum 258 MiB |

The frozen 128K reserve from the 2026-10-16 block was a different operating point: at an explicit
400 MiB reserve the prototype selected shared and still fell to 248 MiB after the fresh 2K prompt,
eight below the floor, and the guard stopped that run before its retrieval checks were saved. The
216 MiB refusal in that earlier block was on this same frozen reserve. The automatic path instead
adapted 700 MiB to a 423 MiB reserve with 1 slot and passed at 270 MiB. At 64K the automatic path
selects separate even with the variable, because the selection threshold is 256 MiB plus a 64 MiB
allowance (`src/prefill/buffer_layout.hpp`) and free minus owned was 360 MiB; the variable changes
nothing there.

Output parity was checked at 64K on a private build that forces the choice and differs from the
committed heuristic by that predicate alone (never committed): all 11 requests produced identical
outputs on both layouts, and shared left 66 MiB more free (396 vs 330 MiB at ready, 326 vs 260 MiB
after the sequence). The committed shared selection was measured directly at 128K automatic, where
outputs matched the 64K separate run at equal slot count; the 13-slot 128K run differed on the three
sustained 512-token texts, consistent with a slot-count rounding effect that was not isolated. The
8K fresh prompt was tested only at 64K on the separate layout (265.2 s, 256 MiB minimum, 4 MiB more
than 2K). No speed claim follows: the 131.2 s 2K prefill was measured at 13 slots in the frozen-400
run that ended at 248 MiB, while the automatic 1-slot run measured 130.8 s.

Several phase-1 arms sampled minimum commit between 7.21 and 7.88 GiB, below the configured 8 GiB
laptop headroom, although the emergency stops (commit below 2 GiB immediately, below 4 GiB over five
samples) never fired. This does not establish whole-plan acceptance. The RTX 4080 SUPER healthy-card
check now has the bounded 2026-10-18 result below; the gfx1100 HIP engine and its fixture compile and
link in a private ROCm 10.2 / Clang 24 SDK with no AMD runtime. The full SYCL engine also compiles
and links in a private WSL2 / Ubuntu 24.04 build with oneAPI 2026.2 after correcting stale
`NativeDense` private-member references in the SYCL implementation; its allocation and cleanup
sequence is unchanged. No AMD or Intel GPU runtime was exercised. Near-128K inputs, an external
display, observed responsiveness and host commit headroom remain open. Raw records (per-arm profiles,
protection receipts, request-arm records and console logs) are under
`build-unsloth/cross-pc-memory-20261016-opencode/phase1/`.

## Shared-stage preservation and selector check on the RTX 4080 SUPER, 2026-10-18

Commit `1357a2d7` was then checked against its parent `e6dab366` on the same desktop used on
2026-10-15: RTX 4080 SUPER 16 GB, Ryzen 7 9800X3D, 30.88 GiB usable RAM and Windows with a 28 GiB
pagefile. Both builds used the same CUDA 13.4.92 / MSVC 19.44.35229 / sm89 toolchain, and each passed
the same four focused CTests (expert-cache retry, prefill buffer layout, prefill allocation and
prefill fallback allocation); the candidate's test time was 3.32 s and the parent's 4.16 s, which is
not a model-speed result.

Twelve arms ran one build each: the parent, the candidate with the variable unset and the candidate
with `STRATA_PREFILL_SHARED_STAGE=1`, at 64K and 128K, in two groups. Every arm ran the same
11-request sequence (warmup, exact arithmetic, cached follow-up, three 512-token generations with
cached follow-ups, a fresh 2K retrieval and its cached reuse) on UD-Q2_K_XL with int8 KV, 32768
resident cells, verifier window 2, automatic expert cache, a 12 GiB resident budget, a 1024 MiB GPU
reserve, 4 GiB host headroom, a compact recent cache, eager CUDA loading and ready-copy reuse off.
Both capacities are the configured limits; no arm sent a full-context input.

Group A kept the preserved desktop profile, whose prompt path borrows 1963 cache slots (3.43 GiB) at
128K and 1926 slots (3.37 GiB) at 64K. That path emits no selection line and never exercises the
shared-stage choice. Group B was a controlled diagnostic that adds only `--no-prefill-borrow`; its
prompt path allocates its own buffers with an effective prefill chunk of 2048. This diagnostic
does not replace the user's saved profile or change a default.

Every candidate arm reproduced its parent: all 11 requests returned identical text and identical
prompt and completion usage counts, in all eight parent/candidate comparisons. Slot counts matched
within each group (A 5270 at 128K and 5334 at 64K, B 4422 at 128K and 4486 at 64K), and the minimum
enforced free VRAM was 683, 683, 1011 and 1077 MiB respectively, the same across the three arms
of each group. Both group-B arms that set the variable selected the separate stage, so no arm on this
desktop ran the shared stage.

The group-B arms with the variable set prepared the GEMM handle earlier, before the pressure
query, so its free-after-host-staging reading is 218 MiB lower (2545 to 2327 MiB at 128K, 2543 to
2325 MiB at 64K) while the later stage and free VRAM match. That is a startup ordering difference,
not a saving, and not a speed proof. The parent's group-A 64K arm reached a physical minimum of
3.9724617 GiB with 6 of 250 samples below 4 GiB, missing its configured host floor even though the
emergency guard never fired; the other eleven arms sampled physical minima from 4.0253 to 4.2447 GiB
and commit minima from 14.0364 to 14.7118 GiB. Machine-wide samples give no overall clean acceptance
and no basis for blaming the parent's code. Recent-cache host targets varied per arm automatically,
so bit-identical host allocations are not claimed. The original CPU guard self-check command and
pass output were later recovered from the desktop session transcript, including the pinned harness
hashes; it completed before the first model arm. The same transcript records a final 55-file
comparison with no changes. Per-arm receipts retain the 55-file count and empty change lists,
alongside 37-file hash maps; full 55-file per-arm post-maps were not saved.

Raw arms, layouts, protection receipts and telemetry are under
`build-unsloth/cross-pc-memory-20261018-desktop-evidence-r1/`.

## Shared automatic policy on the RTX 4080 SUPER, 2026-10-15

Commit `563c136c` passed a separate, guarded desktop run on an RTX 4080 SUPER 16 GB, Ryzen 7 9800X3D,
30.88 GiB usable RAM, Windows, driver 617.14 and CUDA 13.4.92. Windows had a 28 GiB pagefile and a
58.88 GiB commit limit. The isolated engine used eager CUDA loading and the existing UD-Q2_K_XL model.
The working engine, profiles and selected launcher were preserved.

The policy was refreshed once immediately before testing and then frozen. It requested 5 GiB pinned
experts, 3486 MiB compact recent cache, 4 GiB host headroom and a 1024 MiB GPU reserve. Capacity remained
131072 tokens, with int8 KV, 32768 resident cells, automatic prefill, verifier window 2 and parallel 1.
These are the desktop's sampled automatic settings, not recommended fixed budgets for other starts.

Short exact answers, cached follow-ups, three 512-token generations and a 60-second idle check passed.
The sustained generations decoded at 7.1, 8.2 and 8.2 tokens/s. During this watched phase the user reported
"responsive throughout". Its 390 seconds of one-second monitoring found at least 4.249 GiB available
physical RAM and 15.037 GiB available commit, with at most 26.631 GiB physical RAM used and 43.843 GiB
system commit. CUDA reported 761 MiB free at serving and 731 MiB after completed requests.

All six smoke checks then passed on a separate owned server using the same frozen profile. A fresh
130880-token prompt returned all three reference codes in order and passed output-budget fitting.
Prefill took 1160.49 seconds; total request wall time was 1166.83 seconds. The follow-up retained 130944
prompt tokens, reused 130906 tokens, processed 38 new prompt tokens and returned the correct code and
arithmetic answer in 9.33 seconds. Over this 1385-second phase, available RAM stayed above 8.583 GiB,
available commit above 18.730 GiB, and logged CUDA-free samples at or above 677 MiB.

CUDA figures are engine queries at startup and request stages, not continuous VRAM minima. No guard
triggered. Both servers exited, port 8081 was released, and hashes of all 33 preserved working files
were unchanged. The user was away during the long phase, so its desktop responsiveness was unobserved.
These checks do not explain the earlier desktop stalls or establish broad model quality. Laptop external
display, full laptop 128K and Linux acceptance remain separate.

See the [desktop report and sanitized results](../bench/results/2026-10-15-4080s-shared-policy/README.md).

## Desktop cache and placement comparisons, 2026-10-15

The same desktop and `563c136c` executable were then tested with more available RAM. The comparison
profiles were frozen with 22.18 GiB physical RAM and 48.75 GiB commit available, versus 17.08 and
43.47 GiB at the earlier acceptance freeze. Idle nvidia-smi free memory was still 16036 MiB; the GPU
already had similar availability. Capacity, int8 KV, 32768 resident cells, eager CUDA loading,
automatic prefill, parallel 1 and the 1024 MiB GPU reserve stayed unchanged.

Each comparison used the same three technical prompts twice with 160-token output limits, then
the fixed broader coding, arithmetic and note-writing prompts. The technical answers generated
960 tokens per run. Rates below are completion tokens divided by engine decode time; startup,
prefill and idle time are excluded. They are not comparable to the 512-token tutorial rates alone.
All comparison profiles requested 4 GiB host headroom. Pinned/recent requests and measured decode
rates, listed as technical / broader, were:

- Shipped placement, 5 GiB / 3486 MiB: 7.67 / 4.80 tokens/s; repeated after the other profiles,
  7.43 / 4.16 tokens/s.
- Refreshed automatic budgets, shipped placement, 9 GiB / 4609 MiB: 9.18 / 6.56 tokens/s.
- The same budgets with the existing desktop-trained placement: 19.35 / 4.87 tokens/s.
- Trained placement, 12 GiB / 1537 MiB: 24.24 / 5.66 tokens/s.
- Shipped placement, 12 GiB / 1537 MiB: 14.01 / 5.56 tokens/s.

The 9/4609 and 12/1537 pairs requested the same total host-cache bytes. Actual allocation still
depends on admission and protected buffers. Starting placements were copied and hashed, with profile
saving disabled. These results support a technical-workload placement benefit and a budget-split
tradeoff; the large technical gain did not carry over consistently to the broader workload. The
original/repeated control and per-request records retain the variability. One refreshed-profile
technical response fell to 3.4 tokens/s, and the initial control's inventory prompt took 69.59 seconds
in prefill.

For the first technical request, engine-reported logical expert-file payload fell from 8934.6 MB in
the control to 1630.7 MB with the trained 12 GiB profile. These counters are not physical SSD traffic.
All these runs completed without an emergency guard trigger, with logged CUDA-free samples at or
above 673 MiB. The trained 9 GiB profile briefly reached 3.797 GiB available RAM, below its requested
4 GiB headroom. Its sharpest dip occurred with nearly unchanged owned RSS/private memory, consistent
with global memory variation but insufficient to identify its owner. Headroom is an allocation
policy, not a guaranteed system floor. These automated comparisons had no new responsiveness observation.

The broader answers retained quality limitations: the code cases passed review, but inventory
reconciliation was incomplete at the 160-token budget and the status paragraph omitted the explicit
derived count of 593 scans. They are performance workloads, not a claim that all output requirements passed.

A separate candidate therefore requested 6 GiB headroom. With shipped placement and an equal
11.5 GiB total cache request, 11 GiB pinned / 512 MiB recent decoded three 512-token tutorials at
9.1, 10.2 and 9.3 tokens/s, but cached follow-ups took 13.77-15.27 seconds. Moving 2 GiB from pinned
to recent storage, for 9 GiB / 2560 MiB, gave 10.8, 12.5 and 12.8 tokens/s and 6.34-6.52 second
follow-ups. All reused their 560-token prefix and processed 29 new prompt tokens. The balanced
candidate also passed idle monitoring and a complete 256-token-budget JSON inventory check,
including the 320-unit reconciliation. Minimum available RAM was 6.017 GiB, available commit
16.069 GiB, and logged CUDA-free memory 731 MiB over this 332-second run.

Without refreshing that balanced profile, a separate 1259-second run passed the API smoke checks,
fresh near-limit retrieval, context fitting and prefix reuse. The fresh request retained all
130880 prompt tokens and returned the three reference codes in order. Prefill took 1079.271 seconds
and request wall time was 1099.079 seconds, versus 1160.493 and 1166.828 seconds in the earlier
acceptance. Its short 27-token answer decoded at 1.4 tokens/s versus 4.5 earlier, so the faster
prefill does not establish a large-context decode improvement. The follow-up retained its full
130944-token history, reused 130906 tokens, processed 38 new tokens and correctly returned the code
and arithmetic result in 8.204 seconds, versus 9.333 earlier.

Minimum available RAM in that long run was 5.856 GiB, available commit 15.574 GiB, and logged
CUDA-free memory 677 MiB. No guard triggered. The requested 6 GiB headroom was again a soft target;
the sampled RAM minimum was slightly below it. CUDA figures are engine stage samples, not continuous
VRAM monitoring. These single-run comparisons combine changed cache budgets and current machine
conditions, and do not establish desktop responsiveness while unattended. After cleanup, the
working files and original checkout still matched their baseline, frozen configurations and engine
hashes were unchanged, no Strata process remained and loopback port 8081 was free.

These are separate experimental configurations; the shared defaults and working profile selection
were not changed. Local per-request results, frozen nonsecret settings and telemetry are under
`.claude/4080s-performance-20261015/`; the new artifacts are not included in the repository.

### Dedicated-use follow-up

The user then confirmed the desktop was in its intended Strata-only state, with normal applications
closed. At the comparison freeze, available physical RAM was 22.741 GiB and available commit was
48.651 GiB. The executable, shipped expert placement, 131072-token capacity, int8 KV, 32768 resident
cells, eager loading, prefill and speculation settings stayed unchanged. These were automated runs;
desktop responsiveness was not observed. Each profile generated the same three 512-token tutorials,
their cached arithmetic follow-ups, a complete inventory answer and idle checks.

The rates below count only those 1536 tutorial completion tokens divided by their summed engine
decode time. They exclude prefill and startup, and differ from the earlier 160-token technical set:

- Control, 9 GiB pinned / 2560 MiB recent / 6 GiB headroom: 11.93 tokens/s, with individual rates
  10.7, 12.5 and 12.8. Cached follow-ups took 6.28-6.53 seconds.
- More recent storage, 9 GiB / 4608 MiB / 4 GiB headroom: 11.01 tokens/s, with rates 9.5, 11.9 and
  12.0. Follow-ups took 6.80-7.37 seconds. The complete inventory answer took 26.29 seconds versus
  the initial control's 82.45 seconds; a later control repeat took 31.21 seconds. That variation
  prevents attributing the inventory improvement to recent-cache capacity alone.
- More pinned storage, 11 GiB / 2560 MiB / 4 GiB headroom: 13.34 tokens/s, with rates 10.5, 15.5 and
  15.2. Follow-ups took 26.49, 5.28 and 5.65 seconds; the first was a substantial latency outlier.
- That same host allocation with GPU reserve reduced from 1024 to 768 MiB: 8.73 tokens/s, with
  rates 6.4, 10.2 and 11.1. Actual GPU expert storage grew from 5272 slots / 9.21 GiB to 5415 slots /
  9.46 GiB, and logged CUDA-free memory fell from 731 to 475 MiB. Spending that extra VRAM did not
  improve this run; its follow-ups took 7.33, 6.11 and 6.95 seconds. Remaining trials retained the
  1024 MiB reserve.
- After those runs, 12 GiB / 2560 MiB / 3 GiB headroom: 15.85 tokens/s, with rates 15.4, 16.1 and
  16.1. Follow-ups took 4.78, 4.67 and 24.37 seconds. This increased requested host expert storage
  from the control's 11.5 GiB to 14.5 GiB and confirmed 12.00 GiB actually pinned. It improved
  tutorial throughput but did not eliminate latency outliers. Minimum available RAM was 3.587 GiB,
  available commit 14.114 GiB and logged CUDA-free memory 731 MiB over 250.887 seconds.

Repeating the control after those candidates measured 11.91 tokens/s overall, again with individual
rates of 10.7, 12.5 and 12.8 and follow-ups at 6.26-6.62 seconds. The 12 GiB-pinned candidate's
aggregate tutorial decode rate was about 33% higher than the controls, while its 24.37-second
follow-up prevents a claim of uniformly lower interactive latency. These measurements do not
identify the cause of the latency outliers.

All these short runs passed the exact, reuse, inventory-reconciliation and idle checks without a
guard trigger. Before the 3 GiB-headroom trial, fresh planning admitted its 14.5 GiB request with
22.882 GiB RAM available. The local guard was tightened for that trial: stop immediately below
2 GiB available RAM, or below 2.5 GiB for five consecutive samples, while retaining commit and
stage-CUDA checks. Injected checks exercised the actual guard's immediate stop, sustained stop,
boundary reset and opt-in behavior. Admission and headroom still do not guarantee a runtime floor.
Per-request records and frozen settings are local under `.claude/4080s-performance-20261015/dedicated/`.

The frozen 12 GiB-pinned candidate then passed smoke, fresh near-limit retrieval, context fitting
and reuse in a separate 1033.891-second run. A fresh admission check found 22.924 GiB available RAM;
the profile itself was not refreshed. The fresh request retained all 130880 prompt tokens and
returned the three reference codes in order. Prefill took 849.223 seconds and request wall time
was 853.728 seconds, versus the balanced candidate's 1079.271 and 1099.079 seconds. That is about
21% less prefill time in this measured pair. The follow-up retained its full 130944-token history,
reused 130906 tokens, processed 38 new tokens and returned the correct code and arithmetic result
in 5.325 seconds, versus 8.204 seconds. The near-limit answers decoded at 6.5 and 9.4 tokens/s,
but contained only 27 and 14 tokens; they do not establish sustained near-128K decode throughput.

Minimum available RAM was 3.196 GiB, available commit 12.999 GiB, and logged CUDA-free memory
677 MiB. No guard triggered. After cleanup, the baseline working files, original checkout,
frozen configurations, shipped placement and engine hashes were unchanged; no Strata process
remained and port 8081 was free. The local `summary.json` and `post-snapshot.json` retain the
measurements and preservation checks. The shared defaults and selected working profile remain
unchanged. This separate candidate uses the dedicated desktop conditions and has no new watched
responsiveness acceptance; its short-request latency outlier remains unresolved.

## Merged-engine desktop performance experiments, 2026-10-16

These measurements use shared `main` source `c197d915`, engine 0.1.41 built for sm_89, on the
RTX 4080 SUPER 16 GB / Ryzen 7 9800X3D / 30.88 GiB usable-RAM desktop. The isolated executable's
SHA-256 is `68e4ea7aa7c814d12474ac761a7e5ad3656383e97bd82a77ec8f171bb41a5de8`.
The starting automatic sample had about 20.5 GiB available RAM and 47.5 GiB available Windows
commit. Model files are on the Crucial CT2000P3SSD8 NVMe drive. Keep the earlier dedicated-use
numbers separate: those runs started with more available RAM and used a different workload/build.

All screens retain UD-Q2_K_XL, 131072-token capacity, int8 KV with 32768 resident cells, eager CUDA,
the shipped placement unless explicitly varied, and authenticated loopback. A frozen copy of the
effective automatic profile supplies the control; each trial starts its own server. Existing engines,
profiles, launchers and selected working profile are preserved. These are automated runs without a
new watched responsiveness assessment.

The first screen generates 384 tokens for each of three prompts: a database tutorial, a web-request
walkthrough and Python LRU-cache code. The table aggregates decode tokens divided by decode time,
excluding prefill. Each generation has an exact arithmetic cached follow-up; separate arithmetic,
history and complete inventory-reconciliation checks also run. Token-budget completion is not a
code-quality test. Fresh 8192-token distributed-code retrieval also passed in these full screens.

| Trial | Change from the frozen automatic control | Aggregate decode tokens/s |
| --- | --- | ---: |
| 01 | 8 GiB pinned, 3909 MiB recent, 4 GiB requested host headroom | 13.07 |
| 02 | 11 GiB pinned, 2048 MiB recent, 2.5 GiB requested headroom | 14.05 |
| 03 | Trial 02 plus the existing canonical MTP pack, window 2 | 10.86 |
| 04 | Trial 02 with `--short-read 0` | 7.15 |
| 05 | Trial 02 with verifier window 8, without a drafter | 6.73 |

These single comparisons do not establish a reliable gain. Trial 02's database decode was
13.0 tokens/s; an unchanged database-only repeat later measured 7.1, then another measured 12.3.
The 13.0 and 7.1 runs returned byte-identical database output and had identical GPU hits, PCIe share,
RAM/file blob demand and adaptive exchange counts. Logical file bytes differed by only 2.5%.
The first 8192-token prefill pair was 121.55 versus 65.15 seconds, but the changing controls also
prevent treating that as a repeatable RAM-setting speedup.

The following direction tests use only the same database prompt and its cached arithmetic follow-up,
with Trial 02's RAM settings. Their decode rates are not comparable to the three-prompt aggregate.

| Trial | Setting | Database decode tokens/s | Cached follow-up wall seconds |
| --- | --- | ---: | ---: |
| 06 | Unchanged control | 7.14 | 5.18 |
| 08 | VRAM reserve 2048 MiB instead of 1024 | 7.78 | 20.00 |
| 09 | Existing desktop-trained placement | 13.94 | 11.66 |
| 10 | Explicit `--pcie-frac 1.0` | 11.65 | 11.96 |
| 11 | Unchanged control again | 12.31 | 5.17 |

The larger VRAM reserve and higher PCIe share do not show a dependable overall improvement here.
The placement result remains workload-specific and its follow-up is slower. None of these settings
has been promoted into the shared defaults or selected working profile.

The aggressive screens used about 15.1 GiB according to NVML and generally left 2.6-3.0 GiB
physical RAM available. Trial 04 briefly sampled 2.455 GiB; it did not remain below 2.5 GiB for five
samples. The guard stops immediately below 2 GiB physical RAM or available commit, after five samples
below 2.5 GiB physical RAM or 4 GiB available commit, or when an enforced CUDA-stage sample is below
256 MiB. Available commit stayed above 13.5 GiB in these screens.
CUDA-free readings are engine-stage samples, not continuous telemetry; the separate NVML counters
do not replace them. No completed screen triggered the guard, and owned-process cleanup passed.

### Read-completion diagnosis

A separate instrumented database run (07) measured 81.697 seconds decoding 384 tokens.
Expert-job preparation was about 71.7 seconds (186.72 ms/window), versus about 1.32 seconds of
CPU expert arithmetic (3.44 ms/window). The fetch report recorded 70.254 seconds of read-completion
service across 7786 stage misses, or about 9.02 ms per miss, and 23227 read requests: 2.983 requests
per missed expert. Submitted bytes were 14.789 GB versus 14.636 GB copied into expert buffers.
Cache victim scanning took only 11.6 ms. Counters include nested and summed thread regions; do not
add them together or interpret their ratios as independent critical-path percentages. Instrumented
throughput is not an unprofiled tuning comparison.

The supported reader already submits a layer's batch before waiting. In-place GGUF storage normally
needs separate gate/up/down reads. A contiguous `experts.bin` instead stores those same slices together.
This changes both layout and the reader path, and costs an additional 42.92 GiB of expert storage.
The local experiment prepares a separate pack while the model is stopped, verifies all expert bytes
against the GGUF and compares the dense/index/layout hashes before any inference test. Preparation
and full verification alter storage/cache state; the first following request is not a cold-cache test.

The Windows GPU-budget snapshot stayed below its reported local and nonlocal budgets, and an NVMe
SMART sample reported 42.85 degrees C composite temperature with no temperature warning. Neither
snapshot establishes the cause of the changing read delays. Current evidence localizes the measured
delay to read completion; it does not establish a WDDM eviction or SSD-throttling explanation.

Per-request outputs, frozen local configurations, telemetry, guard results and engine logs are under
`.claude/4080s-perf-20261016/`. Local configurations contain credentials and are not publication artifacts.
The existing selected launcher and shared automatic policy remain unchanged.

### Contiguous expert storage

The isolated contiguous pack subsequently completed full verification: all 46084915200 expert bytes
match their original GGUF slices in canonical per-expert order, across all layers. Its SHA-256 is
`5a71d41249331ac874f97f998e0ec807c13318ca0e0b8a334a9c0ca4e600cc07`.
The dense weights, index, expert layout, tokenizer vocabulary, merges and chat template also match
the original pack. The first preparation attempt stopped at its conservative 6 GiB RAM floor;
the scratch writer resumed only after validating the completed prefix, and released its own mapped
file pages between layers. A separate process then verified the entire expert output. Neither source
GGUFs nor the installed pack changed.

The same three-prompt screen and 8192-token retrieval sequence then ran with only the pack path
changed, retaining 11 GiB pinned, 2048 MiB recent and 2.5 GiB requested headroom:

| Trial | Expert storage | Decode tokens/s | Three cached follow-ups, seconds | 8192-token prefill, seconds |
| --- | --- | ---: | ---: | ---: |
| 02 | In-place GGUF, earlier faster control | 14.05 | 14.58 | 65.15 |
| 12 | Contiguous | 21.77 | 8.76 | 20.73 |
| 13 | In-place GGUF, fresh control | 8.52 | 15.92 | 64.98 |
| 14 | Contiguous repeat | 21.63 | 16.82 | 38.09 |

Sustained decode repeats within about 1% and is 54-55% faster than the faster original-layout
control. The fresh control also contains a slow code-generation case, so comparing only against
its 8.52 tokens/s would overstate the dependable gain. The two 8192-token prefill times are 41-68%
lower than the original-layout controls. Short-response latency still varies: the contiguous repeat
includes a 10.96-second cached code follow-up. This is a repeatable sustained-generation benefit,
not a uniform conversation-latency improvement.

The timed tutorial/code responses and retrieval answers are byte-identical across the layout
comparisons. The repeated inventory response uses different JSON formatting but reconciles to the
same expected values. Correctness, reuse and cleanup checks pass, with no guard trigger. Trials
12 and 14 logged at least 681 MiB CUDA-free and 2.719 GiB available physical RAM; GPU cache
capacity and RAM requests are unchanged. The speedup therefore does not require larger requested
host-cache budgets or GPU expert-cache capacity.

Keeping the contiguous expert file mapped (`STRATA_KEEP_MAPPING=1`, Trial 15) measured 20.49
tokens/s on the database-only case, versus 20.2/20.9 without it. This does not establish closing
that mapping as the main cause. The experiment changes on-disk layout and the reader path together;
the in-place role reader and the contiguous blob reader issue different request patterns.
Trial 16 added the existing MTP pack, verifier window 4 and `--spec-min-p 0.5`; aggregate decode was
20.86 tokens/s, below both contiguous controls. MTP remains off.

The matching contiguous diagnostic (17) recorded 7535 reads for 7552 stage misses, about one
request per miss, versus about three in diagnostic 07. Read-completion service was 11.353 seconds,
or 1.50 ms per miss, and expert-job preparation was about 12.44 seconds; CPU expert arithmetic
remained about 1.23 seconds. Submission (0.319 seconds), direct copy (0.645 seconds), completion
and cache claiming (0.101 seconds) account for almost all that preparation phase without adding
the nested totals again. Useful bytes differed by about 3%; the actual recent-cache allocation
was 2.147 GB versus 1.922 GB in 07, despite identical 2048 MiB requests. This is evidence of a
read-path improvement, not a standalone SSD latency benchmark. The unprofiled layout repeats
above establish the practical speed gain.

### Automatic sizing with the contiguous pack

A fresh automatic sample selected 8 GiB pinned, 4051 MiB recent and 4 GiB requested host headroom.
The policy was then frozen for a matched comparison, changing only the pack path. Both runs retained
the same GPU expert-cache capacity and logged at least 683 MiB CUDA-free memory.

| Trial | Expert storage | Decode tokens/s | Three cached follow-ups, seconds | 8192-token prefill, seconds | Minimum available RAM, GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 18 | Contiguous | 22.27 | 9.03 | 22.15 | 4.102 |
| 19 | In-place GGUF | 13.28 | 16.95 | 70.24 | 4.202 |

In this pair the contiguous run decoded about 68% faster and reduced fresh prefill time by about
68%. It matched the aggressive contiguous screens' roughly 22 tokens/s while retaining more physical
RAM headroom. Minimum available commit was 15.317 GiB versus 15.419 GiB. All arithmetic, history,
inventory and retrieval checks passed, as did frozen-profile and original-file preservation checks.
However, an unchanged later repeat (23) fell to 12.58 tokens/s overall: database and systems decode
were 18.3 and 24.8 tokens/s, but code decode fell to 7.0. Its code prefill took 17.40 seconds and
the three cached follow-ups totaled 20.25 seconds, including an 11.98-second systems follow-up.
Fresh 8192-token prefill remained fast at 22.88 seconds. Memory admission, correctness and cleanup
still passed, with at least 4.164 GiB RAM and 683 MiB CUDA-free. Thus the 68% decode gain is a measured
pair, not a dependable automatic-profile guarantee. The candidate retains automatic sizing for its
memory headroom; the layout does not eliminate the remaining latency variability.

The separate local candidate is `run-unsloth-contiguous-auto-20261016.bat`, with
`strata-unsloth-contiguous-auto-20261016.json`. Its automatic policy has no fixed memory overrides;
each launch samples that PC again. It uses the existing isolated merged engine and the verified pack
at `packs/unsloth-ud-q2_k_xl-contiguous-20261016`, with authenticated loopback on port 8081.
The existing selected working launcher is unchanged. The pack adds about 44 GiB including its other
files, leaving about 20 GiB free on this desktop's model drive after preparation. This layout is an
opt-in local artifact, not a change to setup's storage-saving default or a bundle of machine-specific
paths and credentials to copy onto the laptop.

The same frozen automatic candidate then passed the functional smoke checks, context fitting,
fresh near-limit retrieval and history reuse in a separate 522.652-second run (20). The fresh
request retained all 130880 input tokens and returned the three reference codes in order. Prefill
took 437.669 seconds and the request 440.390 seconds. The follow-up retained its full 130941-token
history, reused 130903 tokens, processed 38 new tokens and returned the correct code and arithmetic
result in 3.759 seconds. Its prefill took 2.554 seconds. The answers contained only 24 and 14 tokens;
their 9.2 and 12.6 tokens/s do not establish sustained near-128K decode performance.

Minimum available physical RAM was 3.659 GiB, available Windows commit 15.196 GiB, and logged
CUDA-free memory 681 MiB. The requested 4 GiB headroom is not a runtime floor. NVML peaked at
15110 MiB used; its counters are distinct from actual CUDA-free queries. No guard triggered,
the frozen profile and protected working files were unchanged, and owned-process cleanup passed.
The new engine's long-context result does not replace the still-unverified laptop, external-display
and HIP/SYCL conditions, or establish the cause of historical desktop stalls.

Two final combinations kept the same automatic budgets and contiguous storage. The existing
desktop-trained placement (21) measured 22.10 tokens/s overall, with 8.46 seconds across the three
cached follow-ups and 22.22 seconds for 8192-token prefill. Disabling adaptive swaps with
`--adapt-swaps 0` under the shipped placement (22) measured 22.37 tokens/s, 7.61 seconds and
22.70 seconds respectively. Both passed the full screen and preservation checks. Neither establishes
an additional sustained-decode gain over 22.27 tokens/s, so the candidate keeps shipped placement
and adaptation enabled. The existing packer/Unsloth packing unit suites also passed (26 tests).

A final instrumented replay (24) captured another slowdown. Code decode took 69.248 seconds,
with 61.011 seconds of summed direct-read completion service over 5661 batches. Expert-job
preparation was about 62.24 seconds, versus 1.16 seconds of CPU arithmetic. GPU-reach wait stayed
near 9.3-9.4 ms/window across database, systems and code. Submitted code bytes were 13.71 GiB,
less than the database case's 17.97 GiB, while completion waiting rose from 15.416 to 61.011 seconds.
The counters overlap other regions and are not independent additive wall-time components.

One-second physical-drive sampling covered systems and code, with median read latency increasing
from 4.66 to 12.98 ms and median read throughput falling from 352 to 214 MB/s. These drive counters
include other processes. WDDM nonlocal usage stayed at 10.906 GiB against a 14.690 GiB budget;
local usage stayed at 14.219 GiB. The telemetry does not measure individual eviction events.
Coverage began after database, so its fast phase has no matching external disk samples. The latest
NVMe SMART snapshot reported 40.85 degrees C and no temperature warning, without proving the
thermal state during every request.

This diagnostic used the same requested settings, but the engine's post-verifier memory check
reclamped the recent cache from 4051 to 2713.1 MiB; actual buffers varied below that target.
Instrumentation and different realized cache capacity prevent a clean speed comparison with 18/23.
Subsequent source verification also found that this diagnostic bundle enables
`STRATA_VERIFY_PROFILE`: its presence sets `prof_on_`, which disables shared-expert overlap in
the verifier. Setting it to `0` still enables it. Its effect is more than timer overhead;
future phase diagnostics should leave that variable absent unless specifically measuring this path.
The supported conclusion is a reproduced read-service slowdown, accompanied by worsening physical
disk latency and stable GPU-reach timing and budget accounting. It does not prove that the exact
same event caused 23 or earlier desktop stalls. Further OS/storage attribution remains open;
changing CPU pool workers would not directly change this fetch concurrency, because blobs are
fetched before CPU expert arithmetic starts.

The final idle check found no Strata server, port 8081 free, and the original protected files
unchanged. The automatic candidate still matched its frozen measurement copy after removing only
the policy metadata, with empty memory overrides and an unchanged generated launcher. NVIDIA
reported 513 MiB used and 15533 MiB free after cleanup. The model drive then had about 17 GiB free;
the additional pack remains local, and no existing model or working configuration was removed.

### Returned research: stager-thread comparison

Source checks confirmed that `copy_blob()` and `copy_blobs()` bypass ready entries in the recent
cache, but the current 27/38-token text suffixes use verifier windows at the default short-read
threshold of 64. Ready-copy reuse therefore needs an opportunity measurement on batched prefill;
it is not a demonstrated fix for those cached-response stalls. The stream-all prefill path schedules
non-GPU experts before routing, while its fallback filters actual demand. The amount of unnecessary
cold file traffic is unmeasured; selecting the fallback also changes buffer and kernel conditions.

The existing stager controls were then compared in A-B-B-A order, with only
`STRATA_STAGER_THREADS` changing between 32 and 16. Both arms explicitly used host ring 128,
batch 1, the same frozen 8 GiB pinned/4051 MiB recent/4 GiB headroom requests, and no verifier
profiling. Capacity remained 131072, KV int8, eager CUDA and shipped placement. These are
unattended screens, not watched responsiveness acceptance or new 128K tests.

| Trial | Stager threads | 8192-token prefill, seconds | Full 8K request, seconds | Three-prompt decode tokens/s | Three cached follow-ups, seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25 | 32 | 59.55 | 62.68 | 12.83 | 23.03 |
| 26 | 16 | 23.36 | 26.49 | 16.80 | 13.16 |
| 27 | 16 | 22.84 | 26.29 | 17.43 | 13.38 |
| 28 | 32 | 23.62 | 31.15 | 17.48 | 12.95 |

The final control returned close to the candidates' prefill times. Neither candidate's prefill time
beat that control by the predeclared 5% screening margin; the apparent mean gain is driven by the slow
first control. These results do not establish a dependable stager-thread speedup. The decode
cases precede the batched 8K prompt and are not evidence that fewer stager threads accelerated
direct decode reads. The candidate and shared defaults remain unchanged.

Effective settings matched apart from thread count, all engine metadata matched, GPU expert slots
remained 5270, and both cache-sizing reports retained 4051 MiB in every arm. All correctness,
retrieval, frozen-profile, protected-file and owned-process cleanup checks passed, with no guard
trigger. Across these screens minimum available RAM was 4.253 GiB, available Windows commit
15.803 GiB and logged CUDA-free memory 679 MiB. Results and final preservation checks are local
in `.claude/4080s-perf-20261016/stager-comparison.json`; the earlier sweep's summary is preserved.

At cleanup the model volume had about 172.5 GiB free, compared with 16.8 GiB in the earlier
snapshot. The user confirmed clearing space manually, without establishing its timing relative to
the arms. This comparison did not record free space at each arm's start, so the timing and effect
are not established. The change must not be used to explain a particular stall. Further read-service
attribution and ready-copy opportunity measurement remain open.

### Ready-copy reuse and remaining optimization routes, 2026-10-16

The next pass kept UD-Q2_K_XL, the verified contiguous pack, 131072 capacity, int8 KV,
eager CUDA and shipped placement. On the same RTX 4080 SUPER / Ryzen 7 9800X3D / 32 GB
Windows desktop, the frozen automatic baseline requested 8 GiB pinned experts, 4051 MiB
recent cache and 4 GiB host headroom. Measurements used host stager threads 32, ring 128,
batch 1, GPU prefill ring 96, stream threshold 1024 and prefill CPU share disabled.
These controls were held fixed unless named below. The working installation was not switched.

The source now offers independent startup experiments, all disabled by default:

- `STRATA_FILE_COPY_READY=1` synchronously copies an already-ready recent-cache entry under
  its mutex before falling back to file copying. It does not admit misses, promote entries
  or lend recyclable pointers to asynchronous DMA. Resident/exchange bytes retain precedence.
  Only the exact value `1` enables this flag.
- `STRATA_FILE_COPY_STATS=1` observes logical copy opportunities independently of reuse.
  Cumulative snapshots appear at `file_cache_stats()` and close; request-end snapshots require
  fetch profiling. Only the exact value `1` enables it. These counts are not OS submissions
  or physical SSD traffic; concurrent snapshots are approximate.
- `STRATA_PREFILL_DEMAND_ONLY=1` selects routed-only prefill while retaining the original
  threshold and allocated ring. It needs full fallback scratch and can select different kernels;
  this is a whole-configuration experiment, not a pure scheduling comparison.
- `STRATA_PREFILL_DEMAND_STATS=1` reports scheduled, routed, unused, unclassified and submitted
  H2D payload without introducing GPU readbacks. Fused GPU-only routing remains unclassified.
  The two prefill flags use the existing leading-`1` convention.

An instrumented 8K prompt exposed 1972 ready blobs / 3.45 GiB of useful copy-out payload.
Reuse served those bytes from ready buffers. Stream-all prefill scheduled 37.15 GiB of payload,
including 3.15 GiB not selected by routing; unclassified payload was zero in this measured chunk.
The engine chunk was 8185 tokens within the 8192-token API prompt. Demand-only eliminated the
unused payload but took 21.60 seconds versus 20.53 seconds for the instrumented control.
The clean demand-only pilot took 21.52 seconds. Moving fewer bytes was not sufficient to win.

Clean ready-reuse confirmation used off/on/on/off order with identical requested workloads,
engine and effective settings except the explicit reuse flag:

| Arm | Reuse | 8K prefill, seconds | Full 8K request, seconds | Decode tokens/s | Cached follow-ups, seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| 13 | Off | 20.4298 | 23.204 | 23.47 | 8.396 |
| 14 | On | 18.9222 | 21.729 | 13.05 | 16.465 |
| 15 | On | 18.9218 | 21.697 | 23.61 | 8.349 |
| 16 | Off | 20.5188 | 23.333 | 23.51 | 8.364 |

Both enabled repeats reduced prefill latency by 7.38% against the faster disabled control,
passing the predeclared 5% criterion. Screen text hashes matched. Decode and cached replies
did not show a repeated gain: arm 14 stalled, while arm 15 returned to baseline speed.
This is a measured prefill improvement, not a general decode or stall fix.

Other clean pilots compared against the faster same-day baseline rather than a slow outlier:

| Arm | Changed control | 8K prefill, seconds | Decode tokens/s | Cached follow-ups, seconds |
| --- | --- | ---: | ---: | ---: |
| 08 | Pinned 7 GiB / recent 5075 MiB | 87.55 | 21.93 | 8.439 |
| 09 | Pinned 9 GiB / recent 3027 MiB | 19.97 | 20.99 | 7.969 |
| 10 | Stager sleep enabled | 20.48 | 23.38 | 8.415 |
| 11 | Unchanged fresh baseline | 20.46 | 23.57 | 8.367 |
| 12 | GPU prefill ring 48 | 20.33 | 13.91 | 18.255 |

None passed the 5% screen. Equal requested host-cache totals do not imply identical physical
placement or arithmetic paths. The RAM splits realized 7167/9214 MiB pinned arenas and the
requested recent targets; GPU expert slots remained 5270. No combination of these negative
pilots was promoted. Aggregate opportunities do not establish reuse distance or avoidable
eviction, so a new retention policy was not justified.

Existing reads already submit batches before waiting; prefill already overlaps bounded owned
host buffers, an issuer and event-controlled GPU slot reuse. No additional exposed interval
with safe source ownership was demonstrated for a pipeline rewrite. Request-boundary timing
without verifier profiling still localizes substantial decode host time to read completion
waiting. Summed concurrent/nested counters are not additive wall-time components. WPR DiskIO
and FileIO capture failed with access denied `0x80070005`; no trace started. Historical global
disk counters and WDDM allocation budgets cannot identify a process/file-level storage cause,
prove paging, or attribute a stall to free space, antivirus or the SSD.

Arms 01–16 passed correctness, retrieval, protected-file and owned-process cleanup
checks without a guard stop. Across these arms, minimum available RAM was 4.439 GiB, available
Windows commit 16.154 GiB and logged CUDA-free VRAM 683 MiB. CUDA observations are stage
samples, not continuous minima. These were unattended runs and do not establish responsiveness.
The implementation has CUDA build and focused C++/Python coverage; HIP/SYCL and laptop runtime
behavior remain unverified. Shared defaults and the selected working profile remain unchanged.
Local evidence is under `.claude/plan-runs/4080s-remaining-optimization-20261016/`, including
`reuse-comparison.json` and the per-arm manifests, settings, logs and telemetry.

The retained reuse candidate was rebuilt separately after a diagnostic-only line-order correction.
The r2 binary SHA-256 is `869c44ee0510bc72e61482664c288aee5f601066ede8ae7daad0a23cf48586b0`;
the measured clean path is unchanged by that correction. CUDA build, 131 helper/benchmark Python
tests, 11 C++ checks and the real request-end diagnostic framing probe passed.

Held-out task transitions passed Python aliasing, generator evaluation and dependency scheduling,
but interval union returned 17 instead of 20. A matched reuse-disabled r2 control produced exactly
the same answers and pass/fail flags. Both attempts correctly stopped before long retrieval and
remain failed quality runs. This does not establish broad model quality or a full acceptance pass.

Retrieval was then tested separately, explicitly preserving the quality failure. A fresh
130880-token prompt returned all three distributed codes; prefill took 253.983 seconds and the
full request 256.507 seconds. The follow-up reused 130903 tokens and returned the final code plus
391 in 3.044 seconds. This is a single retrieval/reuse validation, not a matched long-context speed
comparison. Minimum available RAM was 4.407 GiB, available Windows commit 16.028 GiB and logged
CUDA-free VRAM 677 MiB, without a guard stop. Frozen settings, working files and owned-process
cleanup checks passed. Evidence: `heldout-equivalence.json`, `retrieval-invocation.json` and
`experiment/arms/20-reuse-retrieval-only/` in the local run directory.

A separate local automatic candidate is available through
`run-unsloth-ready-reuse-auto-20261016.bat`, using the r2 engine, reuse enabled and the controlled
staging settings above. It retains automatic memory policy with empty memory overrides;
its server resamples at startup. In the actual automatic-start check it requested 9 GiB pinned
and 4040 MiB recent, with the same 5270 GPU slots. Arithmetic, code recall and cached reuse passed.
Minimum available RAM was 4.251 GiB, available commit 16.337 GiB and logged CUDA-free memory
731 MiB. The profile and launcher stayed unchanged, protected originals matched and cleanup
passed. These different automatic budgets have a short startup check, not a matched speed or
near128K result. Evidence is `automatic-startup/` and `automatic-candidate-receipt.json` locally.
The candidate remains unselected and authenticated on loopback. Its machine paths and saved
credential remain local; another PC should rebuild locally and generate its own automatic profile.

## Earlier RTX 4080 SUPER desktop validation at 128K

Measured on 2026-10-09: RTX 4080 SUPER 16 GB, Ryzen 7 9800X3D, 30.88 GiB usable RAM, a 24 GiB Windows pagefile,
and model files on a Crucial CT2000P3SSD8 NVMe drive. The source-built engine is commit `6794d90` for sm_89.
The configuration uses 131,072 context tokens, int8 KV with 32,768 resident cells, 8 GiB of experts in RAM,
automatic prompt chunks (8,192 selected), and a 1,024 MiB explicit GPU reserve. MTP is off, verifier window 2,
suffix/lookup drafting off, and parallel requests set to one. Startup left 836 MiB free on the GPU.

All six short API checks passed: streamed exact output, arithmetic, conversation history, an executable Python
function, a tool call and an answer from its result. Generation ranged from 1.7 to 4.1 tokens/s across these checks.
The long-context sequence also passed:

| Check | Prompt tokens | Reused tokens | Wall time | Prompt tokens/s | Generated tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three records spread through a fresh reference | 129,024 | 0 | 1,045.004 s | 124.5 | 3.2 |
| Recall and arithmetic follow-up | 129,088 | 129,050 | 11.759 s | 4.8 | 3.9 |

The fresh response returned all three codes in order. The follow-up returned the final code and the correct
integer. Both completed normally. Ten-second Windows sampling found at least 6.65 GiB of available physical RAM
and at most 42.44 GiB system commit against a 54.88 GiB limit. The pagefile is part of this tested setup.

128K fits this measured sequence, but a fresh near-limit prompt takes about 17 minutes on this RAM budget and
file tier. These checks do not establish OpenCode agent reliability or broad model quality. The separate launcher
is `run-unsloth-ud-q2_k_xl-128k.bat`; the original 8K configuration remains available.
See the [full report, reproduction commands and raw results](../bench/results/2026-10-09-4080-super-unsloth-q2-128k/README.md).

## Desktop fetch and placement measurements, 2026-10-10

The local `run-unsloth-desktop-128k.bat` profile uses the built compact-cache engine with the same Unsloth
UD-Q2_K_XL files and 131,072-token capacity. It requests 12 GiB of pinned profile experts, automatic GPU
expert-cache sizing with a 1,024 MiB reserve, and a 16,384 MiB compact recent-cache ceiling. The engine clamps
the recent-cache target to the RAM and commit capacity left at startup, minus its reserve; it does not
allocate another 16 GiB unconditionally. Close applications **before** starting this launcher so its sizing
sees the memory they released. The fixed ceiling in the earlier compact profile was 4,096 MiB.

The unchanged launcher measured 19.81 tok/s over six 160-token technical responses with Codex CLI open.
Later baseline runs slowed, including severe read waits; available RAM alone does not explain them.
The reserve is a startup estimate: later prefill allocations and protected live entries still matter.
The original launcher, config and executable remain the rollback. Its executable SHA-256 is
`46f22bebde5517a880c7e3605095dc5df72f1c10219f53ea5197943ac3404082`.

A separately trained expert placement reduced logical expert-file traffic on the training-like workload.
With the same new executable, pinned 12 GiB and recent 4096 MiB, shipped placement measured 18.41 and
18.34 tok/s before/after the candidate; trained placement measured 26.84 tok/s. Earlier restarted trained
runs measured 27.78 and 27.68 tok/s with auto-clamped recent budgets. Output strings differ between
placements because CPU/GPU arithmetic follows different rounding paths; the expert bytes are unchanged.
The unseen four-prompt workload improved only from 4.74 to 5.03 tok/s in the matched-cache pair, and both
stalled. Both placements passed three of four concise reasoning checks and returned the same wrong
interval-union answer (17 instead of 20). This is a workload-specific gain, not a broad speed guarantee.

The trained placement is separate from `data/expert-profile.bin`; its desktop profile uses
`data/expert-profile-unsloth-desktop-20261010.bin`, 12 GiB pinned and 4096 MiB recent. The hardware-aware
portable profile continues to use the shipped placement and calculates each machine's own budgets.
Do not copy the desktop's fixed overrides onto the 64 GB laptop.

The fetch diagnostic found read waiting dominant, with victim scanning about 0.024 ms/token versus
59.5 ms/token waiting for reads during a run containing two stalls. Existing reads already batch.
Retiring file mappings did not improve speed, and forcing buffered loading exhausted available RAM
during startup and triggered the benchmark guard. Keep `STRATA_FILE_DIRECT_UNMAP` off and leave the
normal direct-read policy in place. Higher PCIe share and async adaptation remain unproven experiments.

The [complete experiment ledger and raw results](../bench/results/desktop-fetch-20261010/analysis.md)
separate throughput, prefill, correctness and diagnostic measurements. The benchmark uses authenticated
loopback, refuses an existing server, and stops only its own processes if RAM stays below 2 GiB for five
samples or immediately falls below 1 GiB. It preserves partial failure evidence.


### Verified desktop launcher

Use `run-unsloth-desktop-trained-128k.bat` for the measured technical-workload desktop placement, keeping
`run-unsloth-desktop-128k.bat` as the unchanged rollback. The final restarted repeat measured
27.07 tok/s on the six technical responses (26.26/27.94 before/after
a four-minute idle gap). This is a workload-specific gain over the initial 19.81 tok/s baseline.
The final unseen repeat measured 7.26 tok/s; there is no dependable
broad-workload win. This remains an optional desktop profile, while generic portable defaults use
the shipped placement. Read stalls remain unresolved.

The selected config is `strata-unsloth-desktop-trained-128k.json`, using
`engine/strata-unsloth-portable.exe` (SHA-256
`fc0ae8c0e97ad501e6c1a994ee51f571d486e1b9e4fae73da9e84a82f7f17b4b`), pinned 12 GiB and
recent 4096 MiB. It retains UD-Q2_K_XL, 131072 capacity, int8 KV/ 32768 resident, saved authentication,
output fitting and no browser opening. Its binding is 0.0.0.0:8080 for LAN access, matching the
original desktop profile and its saved API key. Connect using the desktop's LAN IP, not 0.0.0.0.

All six smoke checks passed. A fresh 130,880-token prompt
returned all three distributed codes in order; a 130,944-token
follow-up retained the full history, reused 130,906 tokens and
returned the final code plus 391. Fresh prefill took 844.8 seconds.
The long-run RAM guard minimum was 4.03 GiB, with no cancellation.
The short long-context answers decoded at 5.5 and 12.2 tok/s (27 and 14 tokens), respectively.
These are measured retrieval/reuse checks, not broad 128K reasoning validation. See the report's
[reproduction steps](../bench/results/desktop-fetch-20261010/analysis.md#reproduce-elsewhere),
including the separate nonsecret placement artifact and files still needing a source commit.
