# Pure memory plans

`serve.memory_plan.memory_plan(request, query=...)` returns an engine-neutral
`engine_memory_plan` object, schema version 1. The optional query seam accepts
`query(argv, *, cwd, env)` and returns native JSON bytes. Without injection, the
only executed operation is the existing bounded native metadata query with
`--resolve-launch --price-only`. There is no device discovery, model loading,
tokenizer construction, credential resolution or listener. The executable must
support that transport. An older executable produces a safe input error.

The public command is:

```console
python -m serve.server --estimate-memory --memory-plan-request request.json
```

`--engine strata` is optional. Other CLI overrides, including `--config`, are
rejected in this mode. Existing `--estimate-memory --engine strata --config ...`
continues to use the saved-launch planner and its existing hardware observation.
Launch-plan production, strict replay and ordinary server calls retain their
existing parsing. Abbreviated options remain supported; parsing conflicts and
invalid values produce sanitized JSON errors.

Valid plans, including `does_not_fit`, exit 0. Invalid requests exit 2 with an
`error` object. Direct callers receive `MemoryPlanError` with a safe `code` and
optional bounded `field`; native stderr, arbitrary path values and argv are never
copied into an error. Unconfirmed bounded-query cleanup remains a distinct fatal
error. A pricing result is not an approved or replayable launch plan.

## Request

All byte values are nonnegative JSON integers, bounded by unsigned 64-bit range.
Booleans and floating-point numbers are not integers for this contract. Unknown
fields and unsupported settings are rejected, with a bounded field name.
No field silently enables another backend, batching or discovery.

```json
{
  "schema_version": 1,
  "model": {"id": "local-model"},
  "config": {
    "context": 8192,
    "kv_type": "int8",
    "batch_size": 1,
    "microbatch_size": 1024,
    "parallel_slots": 1,
    "devices": [{"id": 0, "backend": "cuda"}],
    "offload": {"mode": "hybrid"}
  },
  "budget": {
    "basis": "before_runtime_initialization",
    "devices": [{"id": 0, "bytes": 6442450944}],
    "host_ram_bytes": 51539607552,
    "host_commit_bytes": 68719476736
  },
  "extensions": {
    "strata": {
      "executable": "build/strata.exe",
      "working_directory": ".",
      "pack": "pack/local",
      "native_model": "models/model-00001-of-00002.gguf",
      "expert_profile": "profiles/local.bin",
      "resident_budget_bytes": 8589934592,
      "file_cache_bytes": 0,
      "host_headroom_bytes": 4294967296,
      "vram_reserve_bytes": 734003200,
      "cache_slots": "auto",
      "kv_resident_cells": 0,
      "spec": 2,
      "no_prefill_borrow": true
    }
  }
}
```

The capacities in this example are caller inputs, not measurements or a claim
that the example model fits. Paths are local and resolved against
`working_directory`; that directory is resolved against the caller's directory.
`native_model` must identify the first shard of an explicitly named split family.
Only that family and explicitly referenced allocation inputs are inspected.

Accepted core fields:

| Object | Fields and supported values |
|---|---|
| Root | Required `schema_version`, `model`, `config`, `extensions`; optional `budget` |
| `model` | Required nonempty `id`; optional `content_sha256` as a hexadecimal SHA-256 string or null, treated as an unverified assertion |
| `config` | Required `context`, `devices`, `offload`; optional `kv_type`, `batch_size`, `microbatch_size`, `parallel_slots` |
| `context` | Positive integer up to 1048576 |
| `kv_type` | `fp16`, `int8`, `q4_0`, `k8v4`; default `int8`; aliases `f16` and `q8_0` resolve to `fp16` and `int8` |
| `batch_size`, `parallel_slots` | Only 1; default 1. Serial native batch mode is used |
| `microbatch_size` | Integer from 1 through the smaller of context and 32768; or `auto`, `auto:8192`, `auto:16384`, `auto:32768`; default `auto`. Native prefill ownership resolves automatic markers |
| `devices` | One object with required integer `id` (0 through 65535) and `backend` equal to `cuda`; explicit selection, no current-device lookup |
| `offload` | Required `mode` equal to `hybrid`; no layer-count or alternate placement controls in v1 |
| `budget` | If present, all fields are required: `basis`, `devices`, `host_ram_bytes`, `host_commit_bytes`. Device list has one object with matching `id` and integer `bytes`; explicit zero is valid |
| `budget.basis` | `before_runtime_initialization` or `after_runtime_initialization` |
| `extensions` | Only the `strata` object |

All `extensions.strata` fields in the example are required. Only `environment`
and `allowances` are additional optional fields. Resident budget is positive;
resident budget, file cache, headroom and reserve are at most 1 TiB. File cache and
VRAM reserve must be whole MiB because the native transport uses MiB. Resident
budget and headroom must round trip exactly through native GiB transport.
`cache_slots` is `auto` or a positive integer up to 24576; selection and backing
rounding belong to native metadata. `kv_resident_cells` is zero or a native
supported streamed-KV setting below context. Native geometry validates minimum
residency and page alignment. `spec` must be 2 and `no_prefill_borrow` must be
true. Neither can silently select an unpriced mode.

Omitting the budget is supported for explicit cache and microbatch settings,
returning `not_evaluated`. Either automatic setting without a budget returns
`requires_budget`. No host RAM value substitutes for commit and no explicit zero
causes a live resource fallback. Automatic cache filling uses the existing native
selection; largest-fitting-context search is deferred.

`environment` accepts only these explicitly named native allocation controls:

| Names | Values |
|---|---|
| `STRATA_FILE_CACHE_COMPACT`, `STRATA_PREFILL_MMQ`, `STRATA_PREFILL_RESERVE_BO` | String `0` or `1` |
| `STRATA_OWNED_PRICE` | String `exact` |
| `STRATA_PREFILL_RING` | Canonical integer string `8` or from `16` through `512`, subject to native profile validation |
| `STRATA_STAGER_THREADS`, `STRATA_STAGER_BATCH` | Canonical integer strings from 1 through 256, subject to native resolution |
| `STRATA_STAGER_RING` | Canonical integer string from 2 through 256 |
| `CUDA_MODULE_LOADING` | `EAGER` or `LAZY` |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` |

File cache and headroom environment values are composed from their byte fields;
the device mask is composed from `devices.id`. Other inherited allocation
controls, credentials and preload hooks are excluded. Only `PATH`, `SYSTEMROOT`,
`TEMP` and `TMP` are inherited for the OS loader and temporary paths. Custom
library search configuration is not supported by this v1 request.

## Response and accounting

The core contains `engine`, `model`, `requested_config`, `resolved_config`,
`configuration_complete`, `resources`, `fit`, `warnings` and `capabilities`.
Requested values retain omitted defaults and aliases; resolved values use
canonical values. Native settings, metadata identities, component maps, native
fit and exact query inputs remain under `extensions.strata`. If the minimum
automatic layout cannot fit, resolved configuration is null and incomplete;
`pricing_scope` is `minimum_supported_layout`, and the resource prices describe
that layout. Complete resolutions use `pricing_scope` of `resolved_configuration`.

Each device, host RAM and host commit resource includes `groups` named `weights`,
`kv_cache`, `compute_scratch`, `output_buffers`, `runtime_driver_overhead` and
`transient_peak`. Every group has integer or null `bytes`, `quality` of `exact`,
`estimated` or `unknown`, and a textual `basis`. Exact means native allocation
geometry, not observed physical residency. Host mapped spans are nominal views,
and host weight residency is estimated.

Session and verifier geometry combines KV, workspaces and output buffers. The
adapter does not invent partitions. Unavailable groups remain null/unknown;
`combined_allocations` identifies their membership and counted byte amounts.
`priced_total_bytes` counts native components once, including the native
reserves, headroom, mapped spans and targets. Group rows and combined rows are
explanatory views of that total, not another list to sum. The namespaced native
map is the authoritative component attribution.

An automatic cache can require more free capacity to satisfy its page-rounded
uniform grant policy than its minimum allocated arena consumes.
`minimum_selection_capacity_bytes` reports this nonadditive selection threshold.
Device shortfall uses the greater of this threshold and the priced allocations,
plus applicable caller allowances. It is never added to allocated bytes or
category sums. A policy-only shortfall can therefore occur while the counted
minimum allocations fit.

`allocated_bytes` excludes protected allowances, targets and mapped views;
`mapped_view_bytes` and `assumptions` disclose those separately. Native host
commit is the combined host and device capacity requirement; it is separate from
physical host RAM and is not a measured OS commit charge. It is not partitioned
into fabricated group prices. `transient_peak` is an unknown additional peak
above priced allocations and is nonadditive; it must never be added as another
resident category.

The adapter reads bounded metadata/header hashes and stat identities using the
existing input readers. It does not hash full model tensor payloads.
`model.content_sha256` is null; a supplied digest appears separately as
`asserted_content_sha256` and is never claimed verified. `cacheable` stays false
with a reason, even if a caller supplies a digest. Executable content can be
hashed separately, with its embedded BUILD record read without `--build-info`,
`--check-install` or other native execution. No persistent cache is implemented.

## Fit and explicit allowances

`within_priced_budget` from native is only a capacity-model result. Public fit is:

| Status | Meaning |
|---|---|
| `does_not_fit` | A priced requirement, including applicable explicit allowances, exceeds a caller capacity; resource `shortfall_bytes` gives a lower bound under this model |
| `not_evaluated` | Explicit configuration, no budget |
| `indeterminate` | Capacities cover prices but required unknown costs lack explicit applicable upper bounds |
| `fits_within_allowances` | All required unknowns have caller-asserted named upper bounds and capacities cover prices plus these bounds; conditional, not measured admission |

`extensions.strata.allowances` accepts the following integer byte upper bounds.
Explicit zero asserts that the caller has established a zero additional cost;
absence leaves the cost unknown. These are assertions, not empirical estimates.

| Allowance | Resource and coverage |
|---|---|
| `device_runtime_context` | Device context initialization and initial driver residency; required only before runtime initialization |
| `device_late_libraries` | Additional device library/module, driver and graph-capture costs, including late cuBLAS initialization |
| `device_transient_peak` | Additional device transient peak above priced allocations |
| `host_ram_runtime` | Additional physical host runtime, library, TLS/request and OS residency |
| `host_ram_transient_peak` | Additional physical host transient peak |
| `host_commit_runtime` | Additional host commit for runtime, libraries, TLS/request and OS/driver accounting |
| `host_commit_transient_peak` | Additional host transient commit peak |

Each applicable allowance is charged once in
`required_with_allowances_bytes`, separately from `priced_total_bytes`. Runtime
allowances are withheld from the native selection budget before resolving an
automatic cache, so cache selection cannot consume the same capacity. The
original caller capacities remain in resource results; `extensions.strata.pricing_budget`
records the reduced selection capacities. An allowance larger than its budget
reduces that selection capacity to zero and produces a normal shortfall. Runtime
allowances and peak allowances bound additional, disjoint costs. Native reserves
and headroom remain protected and cannot cover these costs simultaneously.
After-runtime basis excludes the already-paid device context, but does not
automatically cover late libraries, graph capture, OS or request costs. A new,
unrecognized native unknown-cost description remains uncovered and prevents a
fit guarantee. Allowances cannot fill in unavailable category partitions.

The result establishes no actual memory use, throughput, runtime startup success,
physical residency, content authorization or replay approval.

Native pure prefill pricing uses a conservative CUDA kernel-capability envelope
through the same count owners as runtime allocation. It includes possible MMQ
and FP16 fallback buffers, fused/nonfused carve maxima and the native ring bound,
without querying a device. Prefill allocation quantities are estimates and may
exceed the later device-selected layout. Namespaced `estimate_basis` and native
unknown-cost text preserve this qualification. These count inputs are not an
approved replay configuration.
