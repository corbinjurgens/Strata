# Research question: Reliable long-context Strata on a 6 GB RTX 4050

You are answering from this message alone. You cannot see the repository, its
history, its data, or any file not pasted below. Where an answer depends on
something not shown, say what you would need and answer conditionally rather
than guessing.

## Question

Which changes to Strata would most effectively make the already downloaded Unsloth Qwen3.8-Flash-Next UD-Q2_K_XL usable at 64K-128K context on a Windows laptop with a 6 GB RTX 4050 and 64 GB system RAM, while retaining useful prompt-processing and generation speed? Evaluate the candidate memory savings below critically, identify better opportunities in the attached code, and recommend a small, staged set of experiments. Distinguish merely passing startup from processing a long prompt reliably and efficiently. A larger configured context with no memory margin is not the desired outcome.

## Why it is being asked

This answer will inform the choice of an implementation approach, followed by local code review and measurements. The user wants long conversations and coding-agent workloads; 8K context is too restrictive for that use. Existing tuning achieved one successful 40K-token retrieval prompt in a 64K session, but took nearly 17 minutes to prefill. A 96K session starts only with effectively no reported VRAM headroom, and 128K fails admission. We need to decide which modest engine changes are worth implementing first, how to spend the recovered memory, and which larger changes merit a separate experiment. Do not treat this request as approval to implement your suggestions, and do not assume the largest theoretical saving is the best choice.

## Background

The repository is Strata: a C++/CUDA/HIP engine, Python API server, web client and installer. The engine distributes a mixture-of-experts model across GPU, pinned/system RAM and GGUF files on SSD. Relevant mechanisms include a GPU expert cache, RAM-resident expert copies and CPU workers, quantized native dense projections, prompt prefill, speculative verification with CUDA graphs, and streamed KV with only a subset resident on the GPU. Prompt processing has separate staging/scratch requirements from decode. `--no-prefill-borrow` gives prefill its own buffers instead of borrowing the expert arena. MTP is the optional model-based drafter; it is disabled in the successful local configurations. A verifier is still used without MTP.

Environment and evidence date: 2026-10-08. Windows, NVIDIA GeForce RTX 4050 Laptop GPU, compute capability 8.9, 6 GB VRAM; Intel i7-13700H; 64 GB RAM; no Windows pagefile. Driver 616.92. Full-model tests used the release engine 0.1.40 (CUDA 13.0 build); small compatibility fixtures were built with CUDA 13.4.92 and MSVC 19.43. Attached source is the local 0.1.40 tree with experimental Q2 setup/test integration, at commit `b3a788673faa280f33edcb4004202493b437f93e`. Proposed memory optimizations have not been implemented. No model is running now.

The model is already available locally: three GGUF shards, 78,869,128,864 bytes total. Despite its UD-Q2_K_XL name, it contains no Q2_K tensors. Expert gate/up tensors are IQ2_XS in 47 layers and IQ3_XXS in one; expert down tensors are IQ4_NL. Native dense projections are Q5_K/Q6_K/Q8_0; the token embedding is Q5_K; the output head is Q4_K. Setup's normal `--compat-bf16` pack converts required small projections to BF16 and records the conversions. It does not copy the expert tensors into a second experts.bin. `docs/UNSLOTH_Q2.md` provides format checks and the actual runtime observations, including upstream discussion references. Those discussions establish interest and format compatibility, not success of the proposed optimizations on this laptop.

Attached files cover prefill allocation and GEMM/MMQ dispatch, layer and session allocation, native weight loaders, the weight arena, and streamed-KV addressing. They are an intentional subset, not a complete engine. In particular, the large `src/program/generate.cpp`, `src/core/verify.cpp`, `src/core/mtp.cpp`, most CUDA arithmetic kernels and conversation save/restore implementations are omitted to keep the paste manageable. Important previously traced behavior from those files is stated below. Any recommendation depending on their unseen details must identify the missing check rather than inventing an implementation.

## Already known or ruled out

### Measured runtime results

These are observations, not predictions. More detail and the validation limits are in `docs/UNSLOTH_Q2.md`.

| Context / change | Result on the 6 GB card |
| --- | --- |
| 8K, normal MTP and FP16 KV | Failed minimum expert-cache admission, 897 MiB short. |
| 8K, remove MTP, int8 KV | Still 307 MiB short. |
| 8K, also `--prefill 64 --no-prefill-borrow` | Started with one GPU expert-cache slot and 206 MiB reported free after loading. Short text/history/arithmetic/code checks passed; generation 13.1-15.2 tokens/s. |
| 64K, int8 KV, 20,480 resident cells, prefill chunk 64, owned buffers | Failed admission, 65 MiB short. |
| 64K, change KV to `q4_0` | Started, 16 MiB reported free. A fresh 40,285-token needle-at-50%-depth prompt, zero reused tokens, returned the exact answer. Prefill 999.903 s / 40.3 tokens/s; nine-token answer 13.2 tokens/s. This is one retrieval sample, not a full 64K quality or stability test. |
| 128K, same Q4 KV / 20,480 residency / prefill 64 settings | Failed admission, 65 MiB short. Restarting Windows did not fix it: CUDA reported about 0.43 GiB available at the cache-sizing point despite a much larger idle NVML figure before launch. |
| 96K, prefill 16, verifier 2, suffix drafting off | Started with 0 MiB reported free. A 94-token history/arithmetic prompt produced 44 correct response tokens at 14.3 tokens/s; prefill was 17.4 tokens/s. No long 96K prompt has been tested. |

Latest 96K memory-related configuration, MTP absent:

```text
--max-context 98304 --kv q4_0 --kv-resident 20480
--prefill 16 --no-prefill-borrow --spec 2 --suffix-draft 0
--expert-cache auto --resident-budget-gib 39 --pool-workers 9
```

The lookup-chain option remains at its zero default. Suffix and lookup expansion matter because a nominal `--spec 2` alone does not necessarily bound the actual verifier window. The two-token verifier reported 44.6 MiB of device buffers. Images are disabled. The RAM expert budget is clamped by Windows commit headroom: approximately 36-37 GiB was actually retained, with the remaining experts in the file tier. These numbers do not show how much WDDM has paged, the peak allocation of a long request, or the best attainable PCIe bandwidth. The short-prompt and long-prompt prefill rates are different workloads and must not be used as a controlled comparison of chunk sizes.

### Weight and buffer ledger from the downloaded model and source

MiB/GiB below are binary units. Weight sizes are calculated from actual tensor metadata or recorded loader allocations. This is a partial ledger, not a complete peak-memory sum.

| Component | Size / placement |
| --- | --- |
| Routed experts | 46,084,915,200 bytes, about 42.92 GiB, distributed across host/file/GPU tiers. |
| Native dense projections | 301 matrices, 2,043.93 MiB of GPU weights. |
| BF16 packed weight arena | 1,485,688,320 bytes, about 1,416.8 MiB on GPU. Its hyper-connection projections dominate; do not count them again on top of the arena. |
| Hyper-connection down/up matrices, including final mixer | 194 matrices / 635,699,200 elements: BF16 1,271,398,400 bytes = 1,212.5 MiB; original Q8_0 675,430,400 bytes = 644.140625 MiB. |
| Native output head | Q4_K, 2560 x 248320, 357,580,800 bytes = 341.015625 MiB, GPU allocation. |
| Token embedding | Approximately 417 MiB, mapped pinned host memory already; moving it off GPU again saves nothing in these runs. |

The source investigation was independently checked, but its candidate savings are arithmetic, not measured implementations. Please challenge these findings if the attached code contradicts them.

### Candidate A: right-size owned prompt scratch

At maximum context 131072 and prompt chunk T=16, `src/prefill/prefill.cpp` allocates selection scratch for 256 rows and attention scratch for 32 queries. Tying those capacities to the actual required batch could remove about 30.001831 MiB plus 12.475586 MiB, or 42.48 MiB of logical payload. The attention region still dominates the other scratch alternatives after that change. Allocation carving and size estimation must use the same helper, including relayout; examine roughly lines 570-572, 719, 960-1040, 1518, 2360 and 2471. Kernel launch bounds must be audited, not inferred merely from a smaller allocation.

The FP16 expert dequantization ring holds two gate/up-plus-down workspaces, totaling 18.75 MiB. Its users appear to be the fallback path. Conditional allocation might save that only when the actual MMQ plan has no fallback layers. Format support alone does not prove this: `moe_mmq.cu` also checks device capabilities/shared memory, including visible devices. The installed binary's exact all-layer eligibility was not probed.

The dense GEMM workspace is 64 MiB. A 16 MiB variant would save 48 MiB, and `Gemm::native` already slices output rows when the matrix exceeds its workspace. That preserves the mathematical K dimension but changes launch/GEMM shapes; numeric rounding and prefill speed must be measured. Relevant attached `gemm.cu` code is around lines 416, 650, 772 and 811. The three ideas total roughly 109.23 MiB only if all assumptions hold; they are not three measured savings. In borrowed mode, a smaller workspace reduces the loan from an already allocated arena rather than necessarily freeing VRAM.

### Candidate B: mapped pinned output-head weights

`NativeHead::load` currently allocates 341.02 MiB of Q4_K head weights on the GPU. The embedding loader already demonstrates a pinned mapped host allocation and device alias. A single-GPU, MTP-off experiment could preserve the original Q4_K bytes and GPU compute kernels while sourcing this weight pointer from mapped host memory. This removes the device weight allocation and adds host commit/pinned memory. Auto expert-cache sizing could consume the recovered device space, so the same amount need not appear as final free VRAM.

Consumers are not limited to `NativeHead::run`: the omitted verifier calls `native_mmvq` directly on `head_->weights()` near `verify.cpp:1409` and captures that pointer in CUDA graphs. MTP has subset-gather and full-head consumers; multi-GPU setup has another head on the final stage. Initial scope could reject these combinations. Ownership and teardown must keep the mapped allocation alive until all queued graph/kernel uses complete and free it with `cudaFreeHost`, not `cudaFree`. The Q6 head's optional packing path would need its own audit if this were generalized beyond Q4_K.

Every head evaluation needs the weight data, and multi-column kernels may reuse it. Fourteen single-token evaluations per second imply about 5.0 GB/s of unique head-weight bytes alone, before actual PCIe transactions, other model traffic or caching. That is an illustrative bandwidth calculation, not a measured laptop link capability or a throughput promise. Head reads compete with experts and streamed KV; a CPU-computed head is a separate, more invasive design because the existing verifier expects GPU logits. Evaluate mapped access versus explicit staged transfers versus CPU calculation rather than assuming all host placement performs alike.

### Candidate C: compact hyper-connection storage with BF16 reconstruction

The original HC matrices are Q8_0, but the normal compatibility pack stores BF16 rounded values. The current `STRATA_HC_Q8=1` path adds a Q8 copy while retaining the BF16 arena; it does not by itself save 568 MiB. HC matrices are not in `NativeDense::served_names()`' normal skip set. Actual savings require a consistent HC skip set in both `WeightTable::pool_bytes` and `WeightTable::load`, and safe handling of every consumer. Nulling a pointer cannot release space inside a monolithic arena.

One candidate is to keep only Q8 HC source weights in VRAM and reconstruct one BF16 down/up pair in a 12.5 MiB reusable workspace immediately before use. Gross storage reduction is 568.359375 MiB; after that workspace, roughly 555.859375 MiB. This is a proposed schedule, not a proven implementation. It covers all 97 pairs, including the final output mixer. Additional lanes/workspaces, alignment and transient allocations reduce the benefit. The attached `dequant_bf16.cu` Q8 path uses round-to-nearest-even, but byte parity with the packer's BF16 output has not been tested; the omitted `tools/iq_pack.py` is the other side of that required audit. Consider a partial HC experiment too: eight block layers would save an estimated 81.25 MiB net, or sixteen 175 MiB net, each assuming one 12.5 MiB staging pair. This may trade less reconstruction work for smaller savings.

Prompt, decode and verifier paths still consume BF16 pointers. In the omitted `fused_gr.cu`, `fused_gr_read_multi` checks that BF16 down/up pointers are non-null before it can select Q8 execution (around lines 1631-1648). Its direct-Q8 arithmetic uses unrounded weights and a different reduction order, so it is an output-changing alternative. For reconstructed BF16 execution, dequantization must precede each consuming operation on the proper stream and be part of graph capture/replay. Avoid rewriting shared immutable `WeightRef` pointers as layers execute; concurrent lanes need safe scratch ownership. Evaluate the cost of repeated reconstruction, possible fusion and whether exact compatibility is worth that cost. MTP, multi-GPU and concurrent requests would require additional coverage.

### Candidate D: reduce streamed-KV residency with a proven query bound

`layer.cpp` enforces a minimum of 20,480 resident cells, sized for the union of up to eight concurrent verifier selections. With a centrally enforced limit of two query selections per resolve/attention batch, effective residency of 8192 cells could save about 81 MiB of Q4 KV payload plus 0.703125 MiB of mapping arrays across 12 layers. A conservative two-selection bound is `2 * (floor(2051 / 4) + 2) = 1028` blocks, below `8192 / 4 = 2048` resident blocks; the resolver also has a minimum-slot check of 1024. This is only a proposed bound and schedule.

Changing a floor constant is insufficient: allocation uses the maximum of the requested residency and the floor, so the requested/effective value must also fall. The resolver miss list writes precede its overflow flag; allowing an oversized union is unsafe. Resolve and attend must complete in order before another batch can evict its blocks. The omitted verifier's multi-query resolve/attention paths around `verify.cpp:1049-1062`, plus suffix/lookup window expansion, graphs and conversation-state restore, must be checked. Ordinary one-query decode is not the only consumer. Lower residency may increase PCIe misses; it does not eliminate the separate full-context prompt KV staging allocation.

### Allocation order and the admission gate

The omitted `src/program/generate.cpp` places weight-arena sizing/load around lines 2505-2605, output-head loading near 3605, automatic expert-cache sizing near 3799, owned prefill allocation near 5845, and verifier allocation near 6140. Thus owned prefill and verifier allocations happen after admission. Fixed weight or session savings and late scratch savings affect the gate differently.

This abbreviated pseudocode transcribes the relevant cache-sizing logic near lines 3814-3865. It is not a replacement for the omitted full function:

```cpp
owned_prefill_mib = 0; // unless prefill_chunk > 0 && !pf_borrow
// Default when prompt buffers are owned:
owned_prefill_mib = 160 + prefill_chunk * 680 / 1024;
// Opt-in STRATA_OWNED_PRICE=exact instead uses:
owned_prefill_mib = ceil_MiB(Prefill::bytes_needed_owned(g, ss, prefill_chunk)) + 64;
// bytes_needed_owned models 2 MiB allocation pages.
free_b = device_free_bytes();
reserve = MiB(vram_reserve_mib + owned_prefill_mib) + mtp_bind + pipe_first;
slots = max(0, (free_b - reserve) / expert_blob_bytes);
min_slots = (prefill_chunk > 0 && pf_borrow)
    ? ceil_div(Prefill::bytes_needed(g, ss, 256), expert_blob_bytes) + 128
    : 1;
// Automatic default-reserve adaptation must retain min_slots and at least
// kSmallReserveMib = 300; an explicit user reserve is treated separately.
```

Reducing late allocations without changing their accounting does not necessarily make the old admission test pass. Conversely, pricing allocations more precisely does not itself free memory or prove safe runtime headroom. `STRATA_OWNED_PRICE=exact` can reserve more than the old heuristic for tiny chunks at long context. Do not claim it is an automatic fix, or propose simply lowering safety reserves until loading succeeds. Recommend measurements that distinguish requested bytes, allocator/page-rounded bytes, peak live allocations, WDDM residency and minimum expert-cache requirements.

### Shortcuts already examined

- Removing MTP and reducing the verifier window have already been used. There is no further MTP weight saving in the current run.
- The Q4 full-context prompt KV staging area is about 72 MiB at 128K and uses absolute positions with an identity page table. Shrinking it requires an addressing/scheduling change, not just a smaller allocation.
- FP32 pooled indexer keys are about 192 MiB at 128K across 12 layers and influence token selection. Reducing their precision needs quality/selection checks. The shared FP32 RoPE table is about 32 MiB at 128K and has active consumers.
- `--no-capture` is incompatible with the CPU expert pool, and disabling a token graph does not remove all verifier graphs. Do not recommend disabling graphs as a free universal saving.
- Optional packing/requantization paths can retain original weights; count actual allocation ownership and lifetime before claiming savings. The main canonical head is already skipped unless explicitly retained.
- Lowering the host expert RAM budget does not remove the fixed device allocations before cache admission. An extra pagefile may change host commit behavior, but is not a VRAM optimization and does not explain away this measured gate failure.

## Constraints

- Target the existing Windows laptop, 6 GB GPU and downloaded UD-Q2_K_XL. New hardware, switching OS, a different model or another large download can be fallback comparisons, not the primary answer.
- Favor reliable 64K first and 128K if practical. Discuss useful measured headroom, for example an initial 150-256 MiB target, without pretending a fixed margin guarantees all workloads fit. Context capacity alone is insufficient.
- Preserve current quantized weight bytes and model behavior where feasible. Separate mathematically equivalent storage/scheduling changes from numerical-rounding changes and quality-changing approximations. Do not silently prune experts, selected attention tokens or vocabulary.
- Start with single GPU, one request at a time, images off and MTP off. Clearly name combinations an experimental implementation must reject or leave on the old path. Keep existing supported behavior for other cards/configurations.
- System RAM and pinned memory are also constrained. The no-pagefile commit limit already reduces the expert budget. Include CPU, SSD and PCIe contention in any proposed device-to-host trade.
- Prefer small opt-in changes that can be independently reverted and measured. Larger redesigns are welcome when the benefit justifies them, but separate them from the first experiments.
- Do not claim unmeasured throughput, full-context quality, numerical parity, safe graph replay or successful 128K operation. Source-derived byte savings are estimates until exercised with this engine and hardware.
- Propose tests for later execution; the user stopped the model runs. The pack contains no model weights and does not enable you to run or inspect the machine. Any unseen behavior must remain conditional.

## What to return

1. A direct answer, in a few sentences.
2. Findings, numbered, each with its reasoning and a confidence of high, medium
   or low.
3. Recommendations, in order of preference, each naming its trade-off and the
   alternative it beats.
4. Assumptions: every point where the answer depends on code, data or
   behaviour not shown here.
5. Checks: concrete things to verify in the repository before acting, each
   stated so someone with the code can run or read it.
6. Open questions whose answers would change the recommendation.

Cite attached files by path. Show only the lines that change, never a whole
rewritten file.

Add a ranked comparison table with: candidate; specific files/functions and allocation lifetimes; estimated steady and peak VRAM savings with arithmetic; extra host/PCIe/compute cost; effect on admission versus post-admission allocations; numerical/quality risk; implementation scope; and the first experiment that could falsify the benefit. Do not add overlapping savings twice or present a fabricated complete VRAM ledger.

Recommend how to spend recovered VRAM among safety headroom, 128K capacity, int8 rather than Q4 KV, larger prompt chunks and a larger GPU expert cache. Address the 40.3 tokens/s long-prompt result explicitly: which savings may improve time to first token, which are likely to hurt it, and what measurements would tell us? Look beyond the four candidates when the supplied source supports an opportunity, including duplicate allocations, scratch lifetimes, stream ordering, host/GPU work balance and allocation-page overhead. Do not let the candidate list anchor the entire answer.

Give a minimal sequence of experiments with go/no-go criteria. Include tensor/dequant parity, short and long prompt correctness, KV eviction and conversation reuse, graph capture/replay, startup and runtime peak VRAM/host commit, fresh and reused prompt throughput, decode throughput, and several retrieval positions near the target context. Start with cheap fixtures and a baseline; avoid an indiscriminate test matrix. State which omitted files must be inspected before implementing each recommendation. Patch sketches are optional and must be local, conditional on unseen code, and limited to changed lines. Finish with the first change you would try and the evidence that would make you choose a different one.
