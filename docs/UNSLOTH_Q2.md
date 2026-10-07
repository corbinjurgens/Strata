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
