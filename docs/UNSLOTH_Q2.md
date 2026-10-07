# Unsloth UD-Q2_K_XL (experimental)

Setup offers `UD-Q2_K_XL` as an experimental choice. The tensor headers have been inspected and the formats have
existing engine paths, but full-model startup, generated output, speed and quality have not been validated.
In particular, this is not yet a verified way to run Unsloth on a 6 GB GPU. Setup requires engine 0.1.40 or newer
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

These small fixtures establish format compatibility for the paths tested. Full-model generation, prefill,
memory use and performance still require the actual files.

## Try it

For files already downloaded, in a Windows command prompt:

```bat
START-HERE.bat --setup --family unsloth --model UD-Q2_K_XL --gguf-dir "D:\Models\UD-Q2_K_XL" --context 8192 --draft-vocab en
```

Replace the folder with the one containing the shards. Omit `--gguf-dir` to let setup download them, or add
`--no-start` to prepare the files and config without loading the model. Linux uses `./setup.sh` with the same flags.

Setup applies the existing Unsloth RAM budget: system RAM less 24 GiB, capped by the experts' size, with a streamed
context cache deducted. On a 64 GB PC this gives 40 GiB at 8K context and 38 GiB at 128K with int8 KV streaming.
The remainder of the experts is read from the model files. This RAM calculation does not establish GPU fit:
the fixed weights, draft layer, context, prompt buffers and GPU expert cache must also fit.

Until a full-model run is verified, there are no speed, quality or supported-context measurements for this choice.
Start the validation at 8K to separate loading failures from context memory pressure. After coherent short answers
and a multi-turn conversation work, try 64K and 128K separately and check both startup and a long prompt. A configured
context limit alone does not establish that a prompt of that length fits or runs at a usable speed.
