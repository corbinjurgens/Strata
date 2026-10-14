# Unsloth UD-Q2_K_XL (experimental)

Setup offers `UD-Q2_K_XL` as an experimental choice. Full-model smoke tests pass on an RTX 4050 Laptop 6 GB
with the manual memory settings below; setup's default configuration does not fit that card. These are limited
runtime checks, not a quality benchmark or a guarantee on other PCs. Setup requires engine 0.1.40 or newer
for this experiment. Images are disabled; AMD use is unvalidated.

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

`codex/unsloth-low-vram` incorporates the desktop's `codex/unsloth-file-cache` work (`411ee540`) alongside the
laptop's allocation changes. Use this combined branch on either PC. Its hardware-aware helper selects a local
profile; the desktop's separately trained placement remains an explicit experiment rather than a hardware default.

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

This custom branch uses the tested upstream `v0.1.40.1` source baseline (`82f46a8`) with the selective backports
described above. The [hardware-aware helper below](#hardware-aware-preparation-for-both-pcs) prepares a separate
automatic profile. The earlier manual method remains available. For a separate installation, clone the branch:

```text
git clone --branch codex/unsloth-low-vram --single-branch https://github.com/corbinjurgens/Strata.git
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
of fit for every workload. The new laptop policy has offline simulated-hardware coverage; it has not been
runtime tested on the RTX 4050. The historical laptop measurements above used their stated manual settings.

The helper builds `engine/strata-unsloth-portable.exe` and writes `engine/BUILD-unsloth-portable.json`,
`strata-unsloth-portable-128k.json` and `run-unsloth-portable-128k.bat`. These names are separate from the
working desktop launcher and rollback engines described below. Generate the configuration on each PC;
copying a desktop JSON would carry that machine's paths and budgets to the laptop.

### Reproduce from source and existing model files

Use a checkout containing the current engine, setup and helper changes, including
`tools/prepare_unsloth_q2.py`. An older checkout does not include this branch's engine and helper changes. Check out
this branch before reproducing them elsewhere; ignored executables, generated configs, build
receipts and launchers must be rebuilt or generated locally. Keep API keys out of Git.

The Windows build wrapper needs the normal Python environment and installed CUDA, MSVC, CMake, Ninja and
`third_party/llama.cpp` dependencies. It reports missing prerequisites; it does not install them or download
a release engine. See [installation prerequisites](INSTALL.md). A fresh checkout must also obtain the
`LLAMA_CPP_COMMIT` tree named in `setup.py` under `third_party/llama.cpp` (including `ggml` and `gguf-py`);
the tested checkout uses `3cf03257f219afbe7334045ff7c6a06ac68c627d`. An existing copy can be reused. The
receipt hashes the actual selected files, rather than treating that version label as proof of identity.
With those prerequisites ready, run from the checkout:

```bat
.venv\Scripts\python.exe tools\prepare_unsloth_q2.py --build-engine
.venv\Scripts\python.exe tools\prepare_unsloth_q2.py --plan --gguf-dir "D:\Models\UD-Q2_K_XL"
.venv\Scripts\python.exe tools\prepare_unsloth_q2.py --prepare --gguf-dir "D:\Models\UD-Q2_K_XL"
.venv\Scripts\python.exe tools\prepare_unsloth_q2.py --check-ready
run-unsloth-portable-128k.bat
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
uses `--launch --config strata-unsloth-portable-128k.json`; that is the explicit launch step. Close an
existing Strata server before starting another profile. For another experiment, pass a distinct `--config`
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
These are policy allowances, not measured worst-case memory requirements.

The pinned profile-cache request is 70% of that pool, rounded down to whole GiB and capped at 30 GiB on the
larger-RAM machine or 12 GiB below 48 GiB RAM. The remainder requests compact recent-cache storage, capped at
16384 MiB. Under severe pressure the helper requests a small positive 0.25 GiB pinned budget and can reduce
the recent cache to zero; it reports the low-memory condition. The engine still clamps requests at startup.
Both caches allocate according to their own runtime rules; protected live entries may exceed recent
retention, and budgets do not continuously resize after launch.

The launcher samples hardware and memory again before each start. The saved `memory_policy` records which
values are automatic. Manual changes to its managed arguments/environment become persistent overrides;
other config values, credentials and server settings are preserved. A manual pinned budget reduces an
automatic recent-cache request, and an explicit recent request limits an automatic pinned request. To mark
deliberate new-profile overrides even when they equal the current automatic choice, use
`--resident-budget-gib`, `--file-cache-mib` or `--resident-headroom-gib` during creation. Oversized explicit
requests remain explicit and are reported; this does not bypass engine safety clamps. Remove `memory_policy`
only when a completely static configuration is intended.

New profiles set `fit_max_tokens=true` and `open_browser=false`. Output-budget fitting shortens a client's
requested output allowance to the remaining context; it does not truncate the prompt or increase capacity.
The launcher uses a nonblank saved API key even if its parent environment contains a different or blank
`STRATA_API_KEY`. It omits browser opening unless explicitly enabled in the config. It clears inherited
`STRATA_VERIFY_PROFILE`, `STRATA_DECODE_TIMING` and `STRATA_FETCH_PROFILE`; values deliberately placed in the
config's `env` remain explicit diagnostic opt-ins.

The engine source includes Windows and Linux paths, and the helper can generate a shell launcher, but its
`--build-engine` wrapper currently supports Windows/NVIDIA only. On Linux, use the repository's normal
`setup.sh --build`/CMake path and local paths/configuration; the Windows build commands above are not a
validated Linux recipe. Neither this new helper policy nor the read-path experiment below has laptop or
Linux runtime measurements in this work.

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

## RTX 4080 SUPER desktop validation at 128K

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
