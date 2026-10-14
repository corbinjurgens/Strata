# Changelog

Keep measured results tied to their hardware, workload and evidence. Preserve dated entries;
record changes that have not been released under Unreleased.

## Unreleased

Work on `codex/unsloth-file-cache`, including the engine changes, preparation tools,
tests, and measured desktop results. No release is claimed.

- **What changed:** opt-in fetch diagnostics, a default-off Windows read-path experiment,
  hardware-aware UD-Q2_K_XL build/configuration/launch support, and guarded benchmark acceptance.
  A separate trained desktop placement preserves the shipped profile and rollback launcher.
- **What worked:** on the RTX 4080 SUPER / 32 GB Windows desktop, the technical workload measured
  26.84–27.07 tok/s with trained placement versus 18.41/18.34 with matched shipped placement;
  the unchanged daily baseline measured 19.81. A fresh 130,880-token prompt and full-history
  reuse passed at 131072 capacity. Final long/idle minimum available RAM was 4.03/3.96 GiB.
- **What did not or remains open:** no dependable unseen-workload gain; intermittent read stalls
  persist. Mapping retirement did not improve speed and remains off. Buffered loading triggered
  a RAM guard and is rejected. Both placements missed the same interval-union model question.
  The new laptop policy has simulated-hardware coverage but no laptop runtime validation.
- **Verification:** native engine build and file-expert fixture; 120 combined Python tests,
  eight public-probe checks, independent native Codex review, authenticated actual launcher
  probes, six smoke checks, near-limit/reuse and repeated throughput runs.

See the [measurement report and reproduction checklist](bench/results/desktop-fetch-20261010/analysis.md)
and [hardware-aware setup instructions](docs/UNSLOTH_Q2.md#hardware-aware-preparation-for-both-pcs).
