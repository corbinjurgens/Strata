# PAIR managed launch contract

This fork supports an opt-in approved launch on a single NVIDIA CUDA GPU, with
the generated native profile, serial requests, `--spec 2` and
`--no-prefill-borrow`. Ordinary startup remains available. PAIR owns approval;
Strata resolves settings, checks saved inputs and refuses incompatible replay.
These contracts do not establish real model fit, speed, GPU preemption or
physical WDDM residency.

## First use

Use the Python environment containing Strata's dependencies, from the checkout.
Prepare model files and a CUDA build with its adjacent `BUILD.json` separately.
This interface does not install or update them. Substitute local absolute paths
for the uppercase placeholders in this complete config example. It illustrates
the generated 64 GiB RAM / 6 GiB GPU / 8K profile's controls, not a sizing
recommendation for another machine. Prefer the preparation helper's generated
config for the actual machine; its `memory_policy` can be resolved by planning.

```json
{
  "exe": "ABSOLUTE_ENGINE_PATH",
  "cwd": "ABSOLUTE_CHECKOUT_PATH",
  "backend": "cuda",
  "gpu": 0,
  "parallel": 1,
  "max_queued_requests": 32,
  "model_name": "pair-model",
  "tokenizer": "ABSOLUTE_TOKENIZER_DIRECTORY",
  "lazy_load": true,
  "host": "127.0.0.1",
  "port": 8095,
  "api_key_file": "ABSOLUTE_PRIVATE_KEY_FILE",
  "args": [
    "--pack", "ABSOLUTE_PACK_DIRECTORY",
    "--native", "ABSOLUTE_MAIN_GGUF_PATH",
    "--expert-profile", "ABSOLUTE_EXPERT_PROFILE_PATH",
    "--max-context", "8192", "--prefill", "64", "--spec", "2",
    "--kv", "int8", "--expert-cache", "auto",
    "--resident-budget-gib", "22", "--no-prefill-borrow"
  ],
  "env": {
    "STRATA_RESIDENT_HEADROOM_GIB": "4",
    "STRATA_FILE_CACHE_MIB": "10176",
    "STRATA_FILE_CACHE_COMPACT": "1"
  }
}
```

`api_key_file` may be omitted when PAIR supplies `STRATA_API_KEY` to the server.
Do not add an empty credential. Supply CUDA DLL directories with `lib_dirs` if
needed, before producing the plan; they become part of the frozen search inputs.
No inherited environment dump is returned, but accepted native/search values
and absolute artifact paths are public plan data: inspect those values before
sharing a plan. Keep secrets out of `args`, `env` and search paths.

These PowerShell commands use placeholders, fixed loopback port 8095, and
UTF-8 files without PowerShell redirection's encoding ambiguity. Planning and
estimation start no listener or model. They can make a bounded native
`--resolve-launch` metadata query, which does not enter `--serve` or allocate a model.

```powershell
$config = 'ABSOLUTE_CONFIG_PATH'
$plan = 'ABSOLUTE_SAVED_PLAN_PATH'
python -B -m serve.server --help
python -B -m serve.server --engine strata --config "$config" --check-install
if ($LASTEXITCODE -ne 0) { throw 'Installation check refused; inspect its JSON' }
$json = python -B -m serve.server --engine strata --config "$config" --port 8095 --print-launch-plan
if ($LASTEXITCODE -ne 0) { throw 'Planning refused; inspect its JSON' }
[IO.File]::WriteAllText($plan, ($json -join "`n") + "`n", [Text.UTF8Encoding]::new($false))
python -B -m serve.server --engine strata --config "$config" --port 8095 --estimate-memory
if ($LASTEXITCODE -ne 0) { throw 'Estimate refused' }
# PAIR now verifies complete artifacts and approves this saved plan.
python -B -m serve.server --launch-plan "$plan"
```

Planning's `--port` defaults to 8095 and is frozen in the plan; pass it
explicitly even when a config has a port. An unchanged observed snapshot and
identical inputs yield byte-identical plan JSON. Free memory is an input, so
two real observations can legitimately yield different plans. Do not require
byte equality across different observations. Replay can observe current
resources for admission, but does not run policy or resize approved settings.
Conflicting replay CLI overrides refuse; supply credentials separately.

Saved replay also retains `log` from the original local config, outside the
public plan and its digest. Relative log paths resolve against the saved launch
working directory. PAIR should set a private local diagnostic destination before
startup so native refusal details survive; HTTP errors remain sanitized. The
existing engine owner opens and closes that log. Changing only its destination
does not change approved native settings or the plan digest. Do not put secrets
in native arguments or share raw logs without checking their contents.

In another PowerShell terminal, these are executable authenticated HTTP
requests. `PAIR_API_KEY` below is a client-side placeholder environment variable,
not a Strata credential source. It must contain the same key as the server.
The chat call uses a model when run against a real engine; run the fixture below
for a no-model demonstration instead.

```powershell
$base = 'http://127.0.0.1:8095'
$headers = @{ Authorization = 'Bearer ' + $env:PAIR_API_KEY }
Invoke-RestMethod "$base/v1/models" -Headers $headers
Invoke-RestMethod "$base/v1/load" -Method Post -Headers $headers -ContentType 'application/json' -Body '{"model":"pair-model"}' | ConvertTo-Json -Depth 64
Invoke-RestMethod "$base/v1/status" -Headers $headers | ConvertTo-Json -Depth 64
Invoke-RestMethod "$base/v1/chat/completions" -Method Post -Headers $headers -ContentType 'application/json' -Body '{"model":"pair-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"reasoning_effort":"none"}'
Invoke-RestMethod "$base/v1/unload" -Method Post -Headers $headers -ContentType 'application/json' -Body '{}' | ConvertTo-Json -Depth 64
```

The repository-owned first-use fixture executes the actual `server.main` CLI
dispatch, a real metadata subprocess, strict fake-engine stdin/stdout and real
loopback HTTP. Only hardware, artifacts, tokenizer and native computation are
synthetic. It checks repeatable production, an independently computed digest,
the matching estimate, saved replay without policy recomputation, models,
load/status/build/chat/unload, insufficient-resource refusal before bind/spawn,
closed child pipes/readers and a released listener. It reuses
`serve.test_launch_runtime.Runtime` and `serve.test_launch_plan` fixtures.

```powershell
$fixtureTemp = Join-Path ([IO.Path]::GetTempPath()) 'strata-pair-first-use'
New-Item -ItemType Directory -Force -Path $fixtureTemp | Out-Null
$env:TEMP = $fixtureTemp
$env:TMP = $fixtureTemp
python -B -m unittest -q serve.test_launch_runtime.Runtime.test_documented_first_use_cli_http_and_cleanup
```

## Plan schema and PAIR approval

`--print-launch-plan` emits a complete JSON object, public schema `1`, kind
`strata_launch_plan`. Save that emitted artifact, rather than constructing a
plan from a hand-written example. Unknown/missing fields, duplicate JSON keys,
nonfinite numbers, unresolved applied settings, changed inputs, unsupported
topologies and inconsistent payloads refuse. The public JSON limit is 8 MiB;
the native payload limit is 1 MiB.

| Field | Contract |
| --- | --- |
| `schema`, `kind`, `plan_digest` | Version, record type, SHA256 of canonical content excluding only the top-level digest |
| `launch` | Absolute `executable`, exact `argv` including executable, absolute `cwd`, native `environment` |
| `server` | Sanitized effective server settings, without credentials |
| `inputs` | Sanitized config identity/path, executable SHA256, file identity/metadata records |
| `hardware` | Selected index/name/identity, device count, inventory total VRAM, native CUDA free VRAM, total/available RAM, commit availability and basis |
| `requested` | Original args, policy-presence flag and composed native request; automatic markers may remain here |
| `resolved` | Concrete native numbers, ordered cache slot layout, resident order, layer blob sizes and KV stage runs |
| `native` | Capabilities, build identity, exact `native_payload_base64`, its SHA256, expected `effective_payload` |
| `estimates` | Categorized native count-only VRAM/RAM estimates, allowances, unknown costs and unavailable runtime comparison |
| `admission` | Required `vram_free_bytes`, `ram_available_bytes`, `commit_available_bytes` |
| `approval` | Explicit statement that integrity is not PAIR authorization |

Live planning samples free VRAM in the native resolver under the frozen CUDA
environment. NVIDIA inventory supplies device identity and total VRAM; its
pre-context free counter is not the native allocation budget. The two APIs can
report slightly different totals for the same card, so replay compares inventory
identity with inventory identity. Explicit injected test budgets remain exact.
This does not change the discovery-free pure memory-plan API or guarantee that
unpriced runtime costs fit. Host policy and strict startup can still refuse.

`launch.environment` has `set`, `remove` and `remove_prefixes`.
Native controls/search inputs are frozen; unexpected
controls refuse at production. Replay removes managed ambient inputs before
applying saved values and keeps unrelated variables. API credential variables
are removed from the native child. This is not global OS hermeticity.

Canonicalization is JSON with sorted keys, separators `,` and `:`, ASCII escapes,
no NaN/Infinity, encoded as ASCII; no trailing newline is hashed. This independent
recipe verifies the settings digest and exact native payload hash:

```powershell
@'
import base64, hashlib, json, sys
with open(sys.argv[1], encoding="utf-8-sig") as stream:
    p = json.load(stream)
body = {k: v for k, v in p.items() if k != "plan_digest"}
raw = json.dumps(body, sort_keys=True, separators=(",", ":"),
                 ensure_ascii=True, allow_nan=False).encode("ascii")
assert p["schema"] == 1 and p["kind"] == "strata_launch_plan"
assert hashlib.sha256(raw).hexdigest() == p["plan_digest"]
payload = base64.b64decode(p["native"]["native_payload_base64"], validate=True)
assert hashlib.sha256(payload).hexdigest() == p["native"]["native_payload_sha256"]
print(p["plan_digest"])
'@ | python -B - "$plan"
if ($LASTEXITCODE -ne 0) { throw 'Digest verification failed' }
```

This recipe assumes JSON from the trusted producer. For untrusted input, use
`serve.launch_plan.read_plan` for bounded parsing, duplicate-key rejection and
full schema validation. The digest is neither a signature nor approval. Anyone
who changes the plan can recompute it.

PAIR must approve the complete saved plan and separately hash the original
config, executable, adjacent build record, tokenizer/template files, libraries
actually selected, all GGUF shards, expert packs and every referenced external
payload/profile. Bind hashes to paths and roles unambiguously. Strata hashes
the executable and allocation metadata; GGUF records hash headers/metadata,
and large payload records may have `content_sha256: null` with size/file identity.
The sanitized config hash excludes credentials. File identities, sizes and
metadata hashes cannot replace PAIR's full content hashes or library resolution
approval. PAIR must prevent artifact replacement between approval and use.

Replay sends exactly `STRATA-PLAN/1\n`, decimal UTF-8 payload length, `\n`, then
the decoded native codec bytes (native schema `2`), before normal stdin commands.
The codec freezes layout, not just slot count. Native strict startup checks
resources, applies these values without fallback/shrinking and reports verified
effective settings before READY. Python rejects disagreement with the approved
payload/build/context and cleans up the owned child. Lazy load and restart use
the same plan. Requested/estimated settings alone never prove effective startup.

PAIR should compare `launch.plan_digest` and `launch.native_argv` from status
with its approved artifact and require a non-null `launch.verified_startup`.
That object is the exact verified native startup map, including its payload
fingerprint. `launch.approved` contains saved scalar expectations; it does not
represent external PAIR authorization. Compare those values with the approved
resolution. With current runtime telemetry, require
`launch.effective_state == "observed"` and
`launch.runtime_memory.freshness == "current"`, then compare the observed context,
prefill, KV residency, resident/cache bytes, batch and parallel settings.
Older planned engines retain `effective_state == "verified"` and the verified
startup map but cannot supply the new current component observations. Compare the
reported build fields with the approved embedded/adjacent record. Wait for
`lifecycle == "loaded"`; lazy HTTP readiness alone is insufficient.

Planned mode refuses non-CUDA backends, multiple selected GPUs, parallel/batched
profiles, vision, unsupported config/native flags, tokenizer-dependent effort
settings, unsupported speculation/MTP and borrowed-prefill modes. The generated
8K and 128K small-GPU profiles are supported, including spec 2 and the 128K
`q4_0` / 20480 streamed-KV form. These are supported contract shapes, not claims
that every device can admit them. Keep ordinary launch for other supported
Strata modes; never silently fall back from planned to ordinary startup.

## Credentials

Precedence is explicit `--api-key`, then `STRATA_API_KEY`, then config
`api_key_file`, then legacy inline `api_key`. Absence disables authentication;
an explicitly empty selected source refuses before engine construction. Clients
send `Authorization: Bearer KEY` or `x-api-key: KEY` on `/v1/*`. Health's
`api_key` field is a boolean indicating whether authentication is enabled.

`api_key_file` must be absolute, a regular file owned by the current user, with
one nonempty literal UTF-8 key (optional terminal newline). Control characters,
multiple lines, oversized files, links/reparse points and unsafe permissions
refuse. POSIX group/other permission bits must be zero (normally mode 0600).
Windows creation uses a protected DACL granting only the current user access;
validation rejects other allowed principals. Do not repair permissions silently.
The preparation helper creates/reuses this private file and writes its path to
config. It migrates only a legacy source resolving to one key; multiple keys
refuse before publication and existing files remain intact.

Legacy strings keep comma-separated semantics; legacy lists keep literal
elements. A file always holds one literal key, including a comma-containing key.
No multi-key-file config is supported. Credentials resolve separately on replay;
credential-only config changes do not change the sanitized plan config identity.
Never put keys in saved plans, diagnostics or engine environment. Public
health/status/errors do not return key values or the key-file path.

## Status and units

`GET /v1/status` preserves existing fields and adds `lifecycle`, `launch` and
`engine_build`. Lifecycle is `unloaded`, `loading`, `loaded`, `unloading` or
`failed`. `loaded` means lifecycle loaded; `auto_load` indicates the reload
capability. `concurrency.requested` is configured slots; `serving` is effective
slots when loaded, zero otherwise.

`launch` contains `mode` (`planned`/`ordinary`), digest, exact native argv,
`requested`, `estimated`, `approved`, `verified_startup`, `effective`,
`effective_state`, `effective_reason`, `current_ownership`, `runtime_memory`
and `runtime_comparison`. Unavailable effective values are
null, not guessed from requested values. On unload, failed load or restart,
current-process effective/holdings/build data are cleared. Ordinary engines
with `STRATA-RUNTIME/1` report actual applied scalar settings and captured native
argv without a saved plan. Their digest, requested, estimated, approved and
verified-startup fields are null. Engines without the new protocol report
unavailable current runtime values.

Resolved/effective allocation fields are integer bytes: `resident_budget`,
`resident_bytes`, `file_cache`, `file_cache_effective`, `host_headroom`,
`vram_reserve`, `workspace`, `weight_scratch`, `common`, `region`, `max_blob`,
`vram_required`, `ram_required`, `commit_required`, `cache_bytes`, layer blob
sizes, KV stage runs and slot sizes/offsets. Context, prefill and `kv_resident`
are token counts; cache slots, experts, batch groups, parallel, ring and thread
settings are counts; flag values are 0/1. `kv` is a codec enum: 0 fp16, 1 int8,
2 q4_0, 3 k8v4. `environment` and `native_payload_fingerprint` are native
integrity fingerprints, not byte costs or PAIR hashes. They belong to the verified
startup map, not the observed scalar map. Observed settings include actual
cache slots/bytes, resident bytes, prefill borrowing and host-stager geometry;
ordinary prefill 0 means disabled. Status omits the large ordered `slots` and
`resident_order` vectors. Other legacy status
fields explicitly use MiB (2^20), GiB (2^30), seconds or milliseconds in their
names. Do not compare legacy MiB/GiB values directly with plan bytes.

Here is a complete synthetic `current_ownership` object (a subobject of
`launch`, not a complete `/v1/status` response):

```json
{
  "freshness": "current", "error": null, "sequence": 1,
  "last_valid_snapshot": null,
  "device": {"backend": "cuda", "ordinal": 0, "identity": "0000:01:00.0", "identity_kind": "cuda_pci_bus_id"},
  "state": "known", "physical_residency_bytes": null,
  "physical_reason": "physical residency unavailable",
  "totals": {"capacity_bytes": 1024, "logical_bytes": 288, "padding_bytes": 544,
             "scratch_bytes": 64, "shared_bytes": 128, "unattributed_bytes": 32, "mapped_host_bytes": 4096},
  "shards": [{"index": 0, "path": "ABSOLUTE_MAIN_GGUF_PATH", "logical_bytes": 256,
              "unique_source_bytes": 240, "duplicate_copy_bytes": 16}]
}
```

Shard records describe current logical weight holdings from successful device
allocations, including duplicate packed copies. Capacity also contains padding,
scratch and shared space; mapped host memory is separate. Unproved shard lineage
is unattributed, not apportioned to invented shards. These are neither cumulative
upload bytes nor physical OS page residency. Device ordinal 0 is inside the
selected CUDA mask; retain its PCI identity when correlating it with hardware.
Only coherent complete snapshots from the captured process publish. Malformed
or incomplete frames retain the last complete snapshot as `stale` with an error.
Explicit unknown reports null totals/shards and `last_valid_snapshot` when
available. Sequence numbers are process-scoped, not monotonic across restarts.
Runtime and legacy ownership sequences are independent and start at 1. A forward
gap permanently marks freshness lost for that process. Consecutive complete
snapshots may legitimately have identical byte totals after cache reassignment.
The Python public consumer bounds aggregates to 64 main shards and 16 KiB records.

Estimates separate weights/expert cache/KV/prefill workspace/scratch/allowances
where the native inventory supplies them, with `units: "bytes"`, `unknown_costs`
and qualification text. `--estimate-memory` returns schema 1, kind
`strata_memory_estimate`, digest, hardware, estimates and admission. Its
`runtime_comparison.allocated_bytes` and `physical_residency_bytes` are null.
Runtime `launch.runtime_memory` contains protocol, sequence, freshness, error
and `components.vram`/`components.ram`. Each component reports `state` and byte
fields `requested_bytes`, `rounding_bytes`, `rounded_bytes`, `logical_bytes`,
`mapped_bytes`, `pinned_bytes`, `allowance_bytes`. Unknown components have null
numbers. Requested storage is successfully owned API-request bytes, or the
logical address span for mapped file views. Rounding is an allocator-granule
model, not observed driver allocation. Mapped and pinned bytes are subsets of
requested storage. Allowances are policy budgets; do not add them or the subsets
to storage. Do not add overlapping weight-ownership totals to components.

`launch.runtime_comparison.vram` and `.ram` use the union of estimated and
observed component names. Each row has `state`, `estimated_bytes`, `actual`,
`basis`, `actual_comparable_bytes`, `delta_bytes`, `freshness` and `reason`.
States distinguish comparable, extra, absent, unknown and unavailable data.
Delta is actual comparable bytes minus estimated bytes only when both exist.

| Component | Comparison basis |
| --- | --- |
| VRAM storage | `allocator_granule_model`, rounded request bytes |
| Names ending in `_allowance`, and RAM `file_cache_target` | `policy_allowance`, policy budget |
| RAM `streamed_kv` | `logical_payload`, useful KV bytes |
| Other RAM | `requested_storage`, owned request bytes |

For example, this complete component object describes synthetic cache storage:

```json
{"state":"known","requested_bytes":512,"rounding_bytes":2096640,"rounded_bytes":2097152,"logical_bytes":256,"mapped_bytes":0,"pinned_bytes":0,"allowance_bytes":0}
```

Components cover canonical/native weights, expert cache, session/KV, verifier,
prefill/scratch/staging, resident experts and file-cache target where supported.
Missing components are not zero. A stale comparison is diagnostic even when its
numbers remain present. Process-wide allocated bytes, driver/OS usage and physical
residency stay null. These data do not establish a measured gain or complete
process memory accounting.

## Build identity

Every supported native build writes adjacent `BUILD.json`, containing `build`,
`executable` (basename) and executable `sha256`. The same `build` object is
embedded as `STRATA_BUILD/1` and available through the no-model native
`--build-info` command and runtime structured `BUILD` record. Existing INFO's
terminal `engine=VERSION` scanner remains compatible.

The build object includes schema/version/backend/configuration/architectures,
`commit`, `dirty`, `built_at`, `cuda`, `git_reason`, `cuda_reason`, and
`source_sha256`, `config_sha256`, `toolchain_sha256`. Missing Git/CUDA identity
is null with a reason, never a fabricated clean revision. `built_at` is UTC
link-input preparation time, retained for no-op builds, not exact link completion.
Source/config/toolchain hashes bind selected inputs, including the selected
GGML source; they do not claim universal reproducible or hermetic builds.
Setup/helper packaging retains the executable-bound sidecar and its other
backend/search/version fields.

A complete synthetic public `engine_build` subobject is:

```json
{"status":"current","reason":null,"commit":"SYNTHETIC_SOURCE_REVISION","dirty":true,"built_at":"2026-01-01T00:00:00Z","cuda":"SYNTHETIC_CUDA_VERSION","version":"test","backend":"cuda"}
```

Public status exposes these fields, not the full source/config/toolchain record.
Validate the full approved `BUILD.json` against the executable/embedded record
outside HTTP. A runtime record is current only for the current captured engine;
unload yields `status: "unavailable"`, a reason and null identity fields. Dirty
artifacts can correctly name a precommit revision: do not relabel them with the
later implementation commit.

## API lifecycle, completion and failures

`/v1/models` always lists the configured model and aliases, independent of idle
unload. `status.value` is `unloaded`, `loading` or `loaded`; detailed unloading
and failure states are in `/v1/status.lifecycle`. Discovery does not load it.
For example, a complete unloaded model list without aliases is:

```json
{"object":"list","data":[{"id":"pair-model","object":"model","status":{"value":"unloaded"},"meta":{"n_ctx":8192},"architecture":{"input_modalities":["text"],"output_modalities":["text"]}}]}
```

`POST /v1/load` and `/v1/unload` require JSON and the configured key, accepting
`{}` or `{"model":"pair-model"}` (an alias is also accepted). No foreign browser
Origin is allowed. Routes are unchanged; `/load` and `/unload` remain available.

| Result | HTTP | Body/completion |
| --- | --- | --- |
| Load, including already loaded | 200 | `status: "loaded"` merged with the complete current `/v1/status` object; READY and planned effective checks completed |
| Unload | 200 | `status: "unloaded"` merged with status; owned child exit confirmed and resources/handles released |
| Already unloaded | 200 | `status: "not loaded"` merged with status |
| Engine lacks unload support | 200 | `status: "unsupported"` merged with status |
| Active/queued request or control | 409 | `error.code: "busy"` |
| Unknown control model | 404 | `{"error":{"message":"model not found"}}` |
| Typed startup/native failure or uncertain cleanup | 503 | Structured runtime error; no false completed load/unload |

There is no asynchronous 202 completion contract. Inspect `status`, `loaded`
and lifecycle as well as HTTP 200, especially with a non-native engine.
Repeated unload is idempotent in effect, with the different status above.

Complete runtime error examples (message text is diagnostic, code is stable):

```json
{"error":{"type":"model_busy","code":"busy","message":"a request is running or queued"}}
```

```json
{"error":{"type":"server_error","code":"out_of_memory","message":"out_of_memory: native engine refused the operation"}}
```

`--lazy`/`lazy_load` starts the HTTP API unloaded. First chat waits for loading;
concurrent first chats share that load and its failure. There is no guaranteed
immediate 503/Retry-After response while loading. An unavailable/restarting
engine can return 503 `model_not_loaded`; no Retry-After header is promised.

| Stable code | Meaning and normal transport |
| --- | --- |
| `out_of_memory` | Typed native allocation failure, HTTP 503 before streaming |
| `model_not_loaded` | Engine starting/unavailable for preparation, HTTP 503 |
| `busy` | Generation queue full, HTTP 429 before streaming; active/queued work conflicts with an explicit control, HTTP 409 |
| `engine_died` | Generic exit, protocol/cancellation failure or unconfirmed termination, HTTP 503 |
| `invalid_launch_plan` | Bad schema/digest/input/effective settings or unsupported planned mode, CLI exit 2 or load HTTP 503 |
| `insufficient_resources` | Current admission/preload headroom refused, CLI exit 2 or HTTP 503 |
| `installation_error` | Invalid installation-check input or unconfirmed probe cleanup, CLI exit 2; a completed unusable check instead returns `checks`/`usable: false`, exit 1 |
| `port_in_use` | Saved launch cannot bind the chosen port, safe JSON and CLI exit 2 |
| `listen_failed` | Other saved-launch bind failure, safe JSON and CLI exit 2 |

Already-started SSE keeps HTTP 200, emits a structured error with the same
cause code, then its protocol terminator (`[DONE]` for Chat Completions).
Do not infer OOM from generic process death. Validation/auth/other API errors
can retain older message-only or numeric-code shapes; the table is not a new
uniform schema for every endpoint. Public failures must not echo credentials.
`max_queued_requests` is an optional integer from 0 through 65536. Absent/null
preserves legacy unbounded waiting; generated PAIR configs explicitly use 32.
The value is frozen in saved plans. The existing FIFO/slot scheduler still owns
ordering; one shared reservation bounds admitted generation requests before
loading/preparation. Capacity is effective active slots plus the waiting limit.
Slots still draining cancellation count until acknowledgement or confirmed exit.
When capacity is full, generation returns HTTP 429 with `error.code: "busy"`
before SSE headers. Status `admission` exposes the limit, active slots, reserved,
stopping slots and queued counts. Explicit load/unload conflicts remain HTTP 409.
The listener backlog is separate from this application admission limit.

On disconnect the owner sends STOP/BSTOP and keeps the slot until DONE/BDONE
acknowledgement or confirmed owned-engine exit. The cooperative grace is five
seconds; fallback uses bounded owned close (one-second phases). Unconfirmed
termination retains ownership, reports failed/`engine_died` and blocks replacement
and slot reuse. OS scheduling/termination can defeat wall-clock targets.
Existing subprocess/HTTP cancellation fixtures assert STOP within 1.5 seconds,
cooperative settlement within 2.5 seconds, and confirmed nonresponsive-child
exit after at least five and before eight seconds. Those are fake-engine test
bounds, not native GPU preemption guarantees. Actual native/device cancellation
and real OS-denial timing remain unverified.

The requested host/port is checked before model startup and bound again at
listener creation. Occupation, permission failure and the final-bind race fail
clearly and clean up the exact owned engine. There is no alternate-port search.
The library test fixture uses ephemeral loopback ports for isolation; production
CLI replay keeps the approved fixed port.

## Installation checks and remaining validation

`--check-install` exits without loading a model or starting HTTP. It checks
required Python imports, executable/embedded/adjacent build compatibility and a
bounded native driver/runtime/BLAS usability probe using the selected library
search inputs. JSON distinguishes present from usable and unknown. Exit 0 means
the checked installation is usable, 1 is a completed negative check, and 2 is
invalid input/probe cleanup refusal. Unconfirmed cleanup is a fatal refusal,
with `cleanup: "unconfirmed"`, retaining process ownership.

It does not prove arbitrary dynamic-library closure, model fit, kernel execution,
vision support or other backends. CUDA compilation and no-model probes must be
reported separately from fake runtime and CPU checks. HIP/SYCL build gates,
loaded-device behavior, physical residency and performance require their own
evidence. No real model workload is part of the first-use fixture or this guide's
no-model verification.

The optional test `Runtime.test_runtime_native_cpp_emitter_decoder_and_public_status`
in `serve.test_launch_runtime` consumes the real CPU C++ producer selected by the
test-only `PAIR_TEST_RUNTIME_CODEC` executable path. It calls `--runtime-fixture`
with device visibility disabled, then checks decoding and fake-engine HTTP status.
Without that binary the test is skipped and native transport remains unverified;
a skipped test is not evidence that the native producer passed. Its synthetic
numbers test transport, not model fit.

An October 29, 2026 Windows check on an RTX 4050 Laptop GPU (6 GiB) loaded
UD-Q2_K_XL with an explicit small-cache 8K profile. Saved and ordinary launches
published current settings and memory components; the saved launch completed a
two-token chat, settled a streaming disconnect and unloaded with confirmed child
exit. This is bounded correctness evidence, not a performance benchmark or
acceptance of every generated profile. The automatic profile was refused because
the planning NVIDIA free-memory reading exceeded the later CUDA-context reading.
That admission discrepancy remains open; reducing reserve floors is not a remedy
established by this check. Physical residency and HIP/SYCL remain unverified.
