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
The saved local configuration was left at 96K. This verifies startup and a short response only: a full 96K
prompt has not been tested, and the lack of VRAM headroom makes this a fragile experimental setting.

## Try it

For files already downloaded, in a Windows command prompt:

```bat
START-HERE.bat --setup --family unsloth --model UD-Q2_K_XL --gguf-dir "D:\Models\UD-Q2_K_XL" --context 8192 --draft-vocab en --no-start
```

Replace the folder with the one containing the shards. Omit `--gguf-dir` to let setup download them. `--no-start`
prepares the files and config without loading the model. On a 6 GB GPU, apply the manual settings above to
`strata-unsloth-ud-q2_k_xl.json` before starting `run-unsloth-ud-q2_k_xl.bat`. Linux uses `./setup.sh` with the same
flags and a `.sh` start script.

Setup applies the existing Unsloth RAM budget: system RAM less 24 GiB, capped by the experts' size, with a streamed
context cache deducted. On a 64 GB PC this gives 40 GiB at 8K context and 38 GiB at 128K with int8 KV streaming.
The remainder of the experts is read from the model files. This RAM calculation does not establish GPU fit:
the fixed weights, draft layer, context, prompt buffers and GPU expert cache must also fit.

Start validation at 8K to separate loading failures from context memory pressure. After coherent short answers and
a multi-turn conversation work, try larger contexts separately and check both startup and a long prompt. A configured
context limit alone does not establish that a prompt of that length fits or runs at a usable speed.
