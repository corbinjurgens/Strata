# Unsloth UD-Q2_K_XL at 128K on the RTX 4080 SUPER

Measured on 2026-10-09 on Windows with an RTX 4080 SUPER (16,376 MiB), Ryzen 7 9800X3D
(8 cores / 16 threads), 30.88 GiB usable RAM, and a 24 GiB Windows pagefile. The model
files reside on a Crucial CT2000P3SSD8 NVMe drive. The pagefile contributes to Windows'
54.88 GiB commit limit; it is not additional physical RAM.

## Source and model

The engine is compiled from `corbinjurgens/Strata`, branch `codex/unsloth-low-vram`,
commit `6794d901f81717ed9c5b5513c83c0eba1187f068`, using CUDA 13.4.92 and MSVC
19.44.35229 for sm_89. The local executable is `engine/strata-unsloth.exe`, separate
from the existing release executable. [metadata.json](metadata.json) records its hash,
hardware, settings, and the three verified model shard hashes.

The model is `unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q2_K_XL`, snapshot
`766911a6b7369840a91dbcd95f9f997acaab6cd6`. All three existing shards passed size and
SHA-256 verification. Preparation reads these Hugging Face files in place; the
`--compat-bf16` pack adds a 1.38 GiB dense arena and tokenizer/native-expert metadata.

## Configuration

```text
--expert-cache auto --max-context 131072 --kv int8
--spec 2 --suffix-draft 0 --lookup-chain 0
--resident-budget-gib 8 --prefill auto
--vram-reserve-mib 1024 --kv-resident 32768
STRATA_RESIDENT_HEADROOM_GIB=8
```

One request at a time, images and MTP disabled. Startup chose 8,192-token prompt
chunks, 5,083 GPU experts (8.88 GiB), and 4,574 RAM experts (8.00 GiB). Prompt buffers
temporarily borrow 1,964 GPU expert slots (3.43 GiB). There were 836 MiB free on the
GPU after initialization. The rest of the experts remain in the file tier. The
server uses port 8080 and the existing saved LAN API key. Through this PC's LAN
address, both missing and incorrect keys returned HTTP 401; the saved key returned
HTTP 200. [api-checks.json](api-checks.json) records these checks. This does not test
the firewall path from a second device. No credential is included in these results.

## Validation

The existing offline setup/packer selection ran 148 tests, with one skipped. Seven
preparation-helper tests also passed. The live API smoke checks all passed:

| Check | Wall time | Generated tokens/s |
| --- | ---: | ---: |
| Streaming exact echo | 15.261 s | 1.8 |
| Arithmetic | 14.277 s | 1.7 |
| Conversation history | 11.634 s | 2.2 |
| Python function, executed on two inputs | 39.643 s | 1.8 |
| Tool call with expected argument | 43.280 s | 2.6 |
| Answer from a tool result | 24.946 s | 4.1 |

[smoke.json](smoke.json) contains the responses, token counts and API timings. The
streaming echo's first content token arrived after 10.42 seconds. These small prompts
do not measure the throughput of the larger batched prompt chunks.

Both near-limit checks passed:

| Check | Input tokens | Reused tokens | New tokens read | Wall time | Prompt tokens/s | Generated tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Three-record retrieval | 129,024 | 0 | 129,024 | 1,045.004 s | 124.5 | 3.2 |
| Recall and arithmetic follow-up | 129,088 | 129,050 | 38 | 11.759 s | 4.8 | 3.9 |

The fresh answer returned all three codes, in order, with 27 generated tokens. The
follow-up returned the final code and `391`, with 14 generated tokens. Both finished
normally, without a cancellation, CUDA allocation failure or observed output loop.
[long-129024.json](long-129024.json) contains the complete responses and timings;
[final-metrics.json](final-metrics.json) contains the server's cache and request data.

The first response took **17 minutes 25 seconds**, including 17 minutes 16 seconds
to process its fresh input. The follow-up reused the context and took **11.8 seconds**.
The 131,072-token configuration fits this measured sequence on this PC, but fresh
large prompts have substantial latency with the 8 GiB RAM expert budget and file tier.
No additional decoding, MTP, or RAM-budget tuning was measured in this run.

Windows memory sampling every ten seconds covered the smoke and long-context checks:

| Measurement | Result |
| --- | ---: |
| Minimum available physical RAM | 6.65 GiB |
| Maximum sampled system commit | 42.44 GiB / 54.88 GiB limit |
| Maximum engine private commit | 28.63 GiB |
| Maximum engine working set | 11.45 GiB |
| Maximum reported pagefile space in use | 0.11 GiB |
| Engine reads during monitoring | 585.71 GiB |

Private commit is not resident physical RAM. The existing 24 GiB pagefile provides
commit headroom and is part of the tested setup, even though little pagefile space
was reported in use. The engine read count includes repeated reads of model data,
not new downloads. [memory-summary.json](memory-summary.json) summarizes the samples
in [windows-memory.jsonl](windows-memory.jsonl). Sampling can miss shorter peaks.

## Reproduce

From the checkout, after building and preparing the model:

```bat
.venv\Scripts\python.exe tools\prepare_unsloth_q2.py --write-config --context 131072 --config strata-unsloth-ud-q2_k_xl-128k.json
run-unsloth-ud-q2_k_xl-128k.bat
```

The helper preserves an existing config. For this experiment, the 128K config has a
separate log path. The generated launcher explicitly supplies port 8080 and checks
the pack and API-key requirement before starting. The temporary background test
server is stopped after validation so the launcher can use the port.

Once the server reports loaded, in a second terminal:

```bat
.venv\Scripts\python.exe bench\results\2026-10-09-4080-super-unsloth-q2-128k\run_tests.py --stage smoke
.venv\Scripts\python.exe bench\results\2026-10-09-4080-super-unsloth-q2-128k\run_tests.py --stage long --target 129024
```

The runner reads the key from the ignored local config. The long reference uses
public repository text and the same tokenizer/template encoder as the server. It
inserts three synthetic records near 5%, 50%, and 95% of the text by character
position, leaving room for answers below the 131,072-token limit. The second turn
tests reuse of the long context. These checks cover loading, basic output and API
behavior; they are not an OpenCode end-to-end test or a broad model-quality study.
