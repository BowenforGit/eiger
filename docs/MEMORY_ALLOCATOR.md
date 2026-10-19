<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# Device memory allocator

Status: implemented
Applies to: CUDA builds using RMM 26.06
Primary implementation:

- `src/cuda/memory/mem_manager.hpp`
- `src/cuda/memory/mem_manager.cu`
- `src/cuda/memory/device_buffer.hpp`

This document begins with the operational contract and then explains the
architecture, invariants, and rationale behind it.

## Operational quick reference

GaMax uses one generation-aware RMM resource per process and CUDA device. A
process may bind to exactly one device; ALPS therefore requires one local MPI
rank per visible GPU. Every host worker thread must call `cudaSetDevice` for
that rank's device before allocating or accessing the resource.

### Environment

Configuration is read before the first allocation or explicit prewarm:

- `GAMAX_RMM_RESOURCE=arena|cuda` selects the thread-safe arena (default) or
  direct `cudaMalloc`/`cudaFree` diagnostic mode.
- `GAMAX_MEM_RESERVE_BYTES=N` gives a positive decimal byte reserve. The arena
  is `available-N`, aligned down to 256 bytes. Unset uses 75% of available
  memory. Empty, signed, zero, malformed, overflowing, and values greater than
  or equal to available memory are errors.
- `GAMAX_RMM_TRACKING=off|on|stacks` enables diagnostic allocation tracking.
  Arena mode defaults off; direct mode defaults on. Stack capture is explicit.
- `GAMAX_RMM_STATISTICS=off|on` enables serialized scoped counters. It is
  instrumentation and should not be used for authoritative allocator timing.

Call `gamax::prewarm_device_memory_resource()` after selecting the CUDA device
and before benchmark timing or worker launch.

### Allocation checklist

- Use `gamax::device_buffer`, project columns, RMM execution-policy handles,
  RMM-backed CUB scratch, and explicit RMM allocators for cuCollections.
- Pass an explicit allocation stream. `buffer_init::zero` enqueues a memset on
  that stream; it does not synchronize.
- Allocation on stream A is ready on A. First use on stream B needs an
  event/wait or host synchronization.
- Destruction is stream ordered, but the retirement stream must be ordered
  after every consumer and remain alive. `set_stream()` creates no ordering.
- `column_view` and `table_view` are non-owning. Their owners must survive all
  asynchronous GPU uses.
- Fabric/IPC VMM, pinned host memory, and unavoidable library allocations are
  outside RMM and must fit in the configured reserve.

### OOM recovery checklist

Only `rmm::out_of_memory` is eligible for allocator recovery. At a quiescent
boundary:

1. drain submitted GPU work while failed-operation owners remain alive;
2. unwind every buffer, container, policy, cache entry, and generation lease;
3. drain stream-ordered deallocations again; and
4. call `reset_device_memory_resource(device)`.

Reset rejects live leases or tracked allocations, removes the old RMM map
entry, proves that the generation expired, and installs a new generation. A
fatal synchronization error, illegal access, invalid argument, or poisoned CUDA
context is not recoverable allocator OOM. Call
`shutdown_device_memory_resource(device)` before process teardown when an
executable owns long-lived GPU caches.

### Validation commands

The focused Compute Sanitizer target compiles the allocator implementation
directly so the tool can attach before the full multi-fatbin library is loaded:

```bash
compute-sanitizer --tool memcheck --error-exitcode=99 \
  ./build-rmm3/src/cuda/tests/test_allocator_sanitizer
```

The latency guard compares the generation-aware wrapper with a bare RMM arena,
both with tracking and statistics disabled:

```bash
./build-rmm3/src/cuda/benchmarks/benchmark_allocator_latency 100000 \
  docs/allocator_latency_gh200_2026-07-12.csv
```

On the GH200 validation host on 2026-07-12, median nanoseconds per 4 KiB
allocation were 298.956 versus 277.428 for one thread (1.0776x), 640.938 versus
779.162 for two threads (0.822599x), and 538.223 versus 622.317 for four
threads (0.864869x). The single-thread result satisfied the 10% guard and the
multi-thread cases beat the bare arena baseline.

## 1. Overview

GaMax uses one process-wide, generation-aware RMM resource for the CUDA device
assigned to the process or MPI rank. The default resource is a persistent,
thread-safe RMM arena. Project buffers and allocator-aware library objects keep
an explicit lease on the resource generation from which they were created.

The design provides:

- safe allocation from multiple host threads and CUDA streams;
- stream-ordered allocation and retirement through RAII;
- deterministic rejection of wrong-device access;
- optional allocation tracking, stack capture, and scoped statistics;
- validated recovery from allocator exhaustion by replacing an entire resource
  generation; and
- explicit allocator routing for Thrust, CUB, and cuCollections.

At a high level:

```text
one process / MPI rank
│
├── process_resource_context
│   ├── state machine + selected device
│   ├── current resource_generation
│   └── access epoch
│
├── resource_generation N
│   ├── immutable device/options snapshot
│   ├── RMM resource stack
│   ├── lease slots
│   └── optional diagnostics
│
├── per-thread weak cache ───────┐
│                                │
└── owning project objects       │
    ├── resource_handle lease ◄──┘
    └── RMM buffer/policy/container
```

The process context is the only owner allowed to install or replace a
generation. A project object does not own the generation through a
`shared_ptr`; instead, it holds a counted `resource_handle`. Reset first proves
that the external lease count is zero, then removes the generation from the RMM
per-device map and verifies that the generation expires.

## 2. Motivation

The previous allocator consisted of a process-global custom slab allocator and
an alternate direct CUDA allocator. It could not safely support the execution
model GaMax needs:

- shared free lists, allocation maps, and counters were mutated without host
  synchronization;
- slab blocks were recycled immediately even if asynchronous kernels still
  used them on another stream;
- `device_buffer` forgot its allocation stream and retired memory on the
  default stream;
- pointer-only deallocation discarded the original size, alignment, resource,
  device, and retirement stream;
- several stream-taking operators still allocated outputs on stream zero;
- global slab reset did not prove that GPU work or allocation owners were
  drained; and
- the singleton was device-agnostic and unsafe to initialize or replace
  concurrently.

The redesign replaces allocator internals rather than adding locks around the
old API. RMM supplies the allocator behavior, while GaMax adds device binding,
generation ownership, lifecycle control, and project-wide routing.

## 3. Goals and non-goals

### Goals

- Make normal project allocation safe for multiple host threads and streams.
- Preserve CUDA stream ordering from allocation through final retirement.
- Route ordinary project-controlled device allocation through one resource
  generation.
- Detect use of a resource on the wrong current CUDA device.
- Support lazy initialization and explicit prewarming.
- Make hard reset reject live owners instead of invalidating them.
- Provide low-overhead production mode and richer diagnostic modes.
- Make OOM recovery explicit, narrow, and testable.

### Non-goals

- Supporting multiple CUDA devices in one process. Multi-GPU execution uses one
  MPI rank per GPU.
- Making hard reset concurrent with allocation or kernel submission. Reset is a
  quiescent operation.
- Automatically tracking dependencies between arbitrary CUDA streams.
- Making borrowed views own their source columns.
- Repairing a poisoned CUDA context after an illegal access or fatal launch
  failure.
- Routing special fabric, IPC, VMM, pinned-host, or unavoidable third-party
  allocations through RMM when their APIs require different memory semantics.
- Preserving the retired ROCm/hipify allocator path.

## 4. Device and process model

Each process binds its allocator to exactly one CUDA device. In ALPS this means
one local MPI rank per visible GPU; local ranks are never wrapped with modulo
device count.

CUDA's current device is host-thread-local. Consequently:

1. Rank startup selects the assigned device.
2. Every host worker calls `cudaSetDevice(rank_device)` before using GaMax GPU
   allocation.
3. Initialization, public resource access, allocation, reset, and shutdown
   validate the current device.
4. A resource cannot be used to allocate on another current device.
5. Buffers do not migrate between devices.

The nonempty `device_buffer` fast path calls `cuCtxGetDevice` at every
allocation boundary. The thread-local generation cache may avoid a mutex, but
it is never used as evidence of the current device. Buffer destruction also
debug-asserts that its recorded device is current. GaMax does not silently
switch devices on behalf of a caller.

## 5. Core components

### 5.1 Process resource context

`process_resource_context` is process-lifetime storage containing:

- the selected device;
- the allocator state;
- the current generation;
- the active option snapshot;
- the next monotonic generation ID;
- a saved initialization/reset failure;
- a mutex and condition variable for state transitions; and
- an atomic access epoch for the allocation fast path.

The context object is intentionally never destroyed. Explicit reset or shutdown
releases CUDA resources; static destruction must not invoke CUDA after runtime
teardown.

Only initialization, reset, and shutdown mutate the current generation.
High-level operations acquire a handle to the installed resource rather than
receiving arbitrary resource parameters.

### 5.2 Resource generation

A `resource_generation` is an immutable configuration and ownership boundary.
It contains:

- a monotonic generation ID;
- the CUDA device ID;
- the complete options snapshot;
- arena-sizing information when applicable;
- the upstream and adaptor resources;
- the final type-erased outer resource; and
- per-thread lease slots.

The resource members are declared in dependency order so C++ destroys them in
the reverse order: outer resource, statistics, tracking, arena, then CUDA
upstream.

Options never change inside a live generation. A mode, tracking, statistics,
or explicit-size change requires generation replacement through hard reset.

### 5.3 Resource stack

Resources are composed from upstream to outermost:

```text
rmm::mr::cuda_memory_resource
    └── optional rmm::mr::arena_memory_resource
        └── optional rmm::mr::tracking_resource_adaptor
            └── optional rmm::mr::statistics_resource_adaptor
```

The arena is present in the default production mode and absent in direct CUDA
mode. Tracking and statistics wrap the resource seen by project allocations so
they observe project requests rather than only arena backing allocations.

The final outer resource is installed in RMM's per-device resource map. That is
a safety net for unavoidable default-RMM consumers; project code must still
pass the generation's resource explicitly to every owning object.

### 5.4 Resource handles and leases

`resource_handle` couples a type-erased RMM resource with a logical lease on
its generation. Any object that captures an owning RMM resource must retain a
handle for at least as long as the RMM object.

Examples include:

- `gamax::device_buffer`;
- RMM-backed columns and metadata buffers;
- Thrust execution policies;
- CUB temporary storage;
- cuCollections allocators and owning containers; and
- process caches that own GPU columns.

The generation owns one cache-aligned lease slot per participating host thread.
The owner thread updates its atomic count with relaxed load/store operations.
Handles copied or destroyed on another thread update a separate signed atomic
adjustment. Diagnostic traversal locks the slot collection and sums both
values. This keeps the common same-thread path inexpensive while making live
diagnostics data-race-free.

Member declaration order is part of the ownership contract: the
`resource_handle` is declared before the RMM object, causing the RMM object to
be destroyed first and its lease to be released afterward.

Leases track resource ownership, not merely allocated bytes. An empty RMM
container can still own a generation and must block reset. In contrast, a
GaMax zero-sized `device_buffer` deliberately creates no RMM object and holds no
generation lease.

### 5.5 Thread-local generation cache

Resource acquisition uses a non-owning thread-local cache containing:

- the last ready epoch;
- a raw generation pointer; and
- that thread's lease slot.

The process context publishes epochs as follows:

| Epoch value | Meaning |
|---:|---|
| `0` | No generation is available |
| even, nonzero | Ready generation; value is `generation_id << 1` |
| odd | Generation replacement is in progress |

The normal public handle path loads the epoch before and after consulting the
cache. A stable matching even epoch permits handle acquisition without locking
the process context. A cache miss, stale epoch, initialization, or reset enters
the mutex-protected state-machine path.

`device_buffer` has a specialized private installation path to avoid an extra
temporary handle assignment. It accepts only an empty destination handle and
still performs the current-device query on every nonempty allocation.

The cache is weak: it does not extend generation lifetime. The quiescent-reset
contract and epoch publication prevent it from being used as an owning
reference.

### 5.6 `gamax::device_buffer`

`device_buffer` is the project allocation boundary and wraps
`rmm::device_buffer`. A nonempty buffer records:

- the RMM buffer;
- its generation lease;
- size and alignment;
- allocation device; and
- current retirement stream.

Construction requires an explicit CUDA stream. `buffer_init::zero` enqueues a
checked `cudaMemsetAsync` on that stream; it does not synchronize.

`set_stream()` changes only the stream used by future deallocation. It does not
record an event, wait for earlier consumers, migrate memory, or establish a
cross-stream dependency. Streams are non-owning, so their lifetime must extend
through buffer destruction.

The raw-pointer adopting constructor and unsized raw deallocation API were
removed because they cannot preserve sufficient ownership metadata.

### 5.7 Library integration

Allocator-aware wrappers make the same generation lifetime rules apply beyond
`device_buffer`:

- `rmm_exec_policy_handle` provides a nonsynchronizing RMM Thrust policy for
  device-only results.
- `rmm_sync_exec_policy_handle` is reserved for operations that genuinely need
  host completion or a host result.
- CUB temporary storage uses RMM-backed RAII buffers.
- `rmm_cuco_allocator` carries both the type-erased resource and its generation
  handle; allocator equality is generation identity.

Installing a resource globally does not excuse omitted explicit allocator
plumbing. A Thrust, CUB, or cuCollections path using a default allocator is a
migration defect because its ownership cannot be proved at reset.

## 6. Initialization state machine

The allocator has five states:

| State | Meaning |
|---|---|
| `uninitialized` | No generation is installed |
| `initializing` | One thread is constructing and installing a generation |
| `ready` | Allocation and resource acquisition are available |
| `resetting` | Access is closed while the current generation is replaced or removed |
| `failed` | Initialization or replacement failed; the saved exception is authoritative |

Normal transitions are:

```text
uninitialized ──initialize──▶ initializing ──success──▶ ready
                                  │
                                  └──failure──▶ failed

ready ──reset──▶ resetting ──replacement success──▶ ready
                    │
                    ├──shutdown success──▶ uninitialized
                    └──failure──▶ failed
```

Initialization validates the current device before taking the state mutex.
Threads arriving during `initializing` or `resetting` wait on the condition
variable. A ready generation is idempotently reused only when all requested
options match; changing options in place is rejected.

Generation construction occurs outside the state mutex because CUDA and arena
creation can be expensive. The state and epoch are published only after the
fully composed resource has been installed in RMM's per-device map.

Lazy initialization occurs on the first resource request. Benchmarks and ALPS
call `prewarm_device_memory_resource()` after selecting the device and before
timing or launching workers, avoiding cold-context and arena creation costs in
measurements.

Before querying free memory, initialization performs a checked one-byte
allocation/free. This reliably creates a cold CUDA primary context before
`cudaMemGetInfo`.

## 7. Arena sizing and configuration

The production default is an RMM arena sized to 75% of currently available
device memory, aligned down to 256 bytes. The remainder is reserved for CUDA
context state, communication libraries, fabric/VMM regions, benchmark
infrastructure, and other allocation sources outside the RMM coverage boundary.

`GAMAX_MEM_RESERVE_BYTES` overrides the percentage policy:

```text
arena_bytes = align_down(available_bytes - reserve_bytes, 256)
```

Parsing is deliberately strict. Empty, signed, zero, malformed, overflowing,
or too-large values are rejected rather than silently falling back.

Configuration is read before first initialization and snapshotted into the
generation. The supported environment variables and exact defaults are listed
in the operational quick reference above.

## 8. Allocation path

For a warmed, nonempty `device_buffer`, allocation proceeds as follows:

1. Load the process access epoch.
2. Verify that it is a stable ready epoch matching the TLS cache.
3. Query the active context's device with `cuCtxGetDevice`.
4. Reject a device mismatch before invoking RMM.
5. Increment the thread's generation lease count.
6. Copy the generation's outer resource value.
7. Construct `rmm::device_buffer` with the requested size, alignment, and
   stream.
8. Optionally enqueue zero initialization on the same stream.

The cold path first enters the state machine, lazily initializes if necessary,
creates the calling thread's lease slot, seeds TLS, and then returns a handle.

The default arena and wrapper path is designed for host concurrency. There is
no process-context mutex in the warmed allocation path. The extra host work
relative to bare RMM is the device check, epoch/cache validation, and generation
lease accounting.

## 9. Stream and lifetime model

Allocation ordering and deallocation ordering are separate obligations.

### Same-stream use

Memory allocated on stream A is ready for operations subsequently enqueued on
A. Destroying the owner after the last operation has been enqueued on A is safe
when A remains the retirement stream: stream ordering places deallocation after
the operation.

```text
stream A: allocate ── kernel/use ── deallocate
```

### First use on another stream

Allocation on A does not make memory immediately ready on unrelated stream B.
The caller must use an event/wait handoff, synchronize A on the host, or allocate
directly on B.

```text
stream A: allocate ── record event
                           │
stream B:             wait event ── first use
```

### Multiple consumers

If streams A and B both use a buffer, retirement must happen on a stream ordered
after both consumers. This can be implemented by joining events onto a
retirement stream or by synchronizing both streams before destruction. Merely
calling `set_stream(B)` does not order A before B.

### Exceptions after submission

An operation that can throw after submitting GPU work must establish its joins
before an owner can unwind. If a later allocation can throw first, the operation
must catch internally while owners remain alive, synchronize or join every
submitted stream, and only then rethrow.

### Borrowed views

`column_view` and `table_view` are allocation-free, non-owning descriptions.
Creating a view does not enqueue CUDA work or extend owner lifetime. Every owner
must outlive all asynchronous work using its views. Device metadata contains
only trivially copyable non-owning views; owning resource objects are never
copied into device memory.

## 10. Hard reset and OOM recovery

Normal successful work reuses the same arena. Hard reset is reserved for an
explicit lifecycle boundary or confirmed `rmm::out_of_memory` recovery.

Reset is legal only when no other thread is allocating, acquiring resources, or
submitting GPU work. The mutex serializes state transitions, but it does not
turn concurrent reset into a supported operation.

The recovery sequence is:

1. Catch `rmm::out_of_memory` inside the failed operation while its submitted
   work and owners are still represented.
2. Synchronize the device, or perform a verified complete stream join.
3. Unwind all cell-owned buffers, policies, containers, and leases.
4. Enter `reset_device_memory_resource(device)` at a quiescent boundary.
5. Validate the current device and synchronize again to drain stream-ordered
   deallocations.
6. Clear process caches such as `StatsStorage` that may own GPU columns.
7. Synchronize a second time to drain cleanup caused by cache destruction.
8. Publish an odd access epoch and count all external generation leases.
9. If tracking is enabled, also require zero tracked allocations. Reject reset
   with pointer diagnostics when available, or lease diagnostics otherwise.
10. Remove the old resource from RMM's per-device map.
11. Drop the context's old generation owner and verify through a `weak_ptr` that
    the generation expired.
12. Construct and install the replacement generation, then publish its even
    ready epoch.

Reset never silently changes allocator mode after OOM. A replacement option
snapshot may be supplied explicitly; otherwise the previous generation's
options are retained.

Only `rmm::out_of_memory` is recoverable through this path. Invalid arguments,
generic allocation failures, CUDA API errors, kernel failures, and poisoned
contexts are not reclassified as allocator exhaustion.

`shutdown_device_memory_resource(device)` follows the same validation and
teardown sequence but leaves the context uninitialized instead of installing a
replacement. Executables with long-lived GPU caches should shut down explicitly
before CUDA runtime teardown.

## 11. Diagnostics and statistics

The design separates low-overhead ownership diagnostics from optional detailed
instrumentation.

### Lease diagnostics

Lease counts are always available and remain race-free during ordinary live
allocation. They answer whether a high-level object still owns the current
generation, including objects with no outstanding device bytes.

### Allocation tracking

Tracking records live allocation requests and can produce pointer-level
diagnostics. `stacks` mode additionally captures allocation stacks. Tracking is
off by default in arena/performance mode because it is a diagnostic adaptor,
not a production timing configuration.

Direct CUDA mode defaults tracking on and is intended for tools such as Compute
Sanitizer. It uses RMM's direct CUDA resource instead of an arena so individual
allocations remain visible to memory-checking tools.

### Scoped statistics

Statistics collect current, peak, and total byte/allocation counters.
RMM statistics counters form a stack, so `scoped_memory_statistics` serializes
probe scopes across host threads. Statistics must be enabled when the generation
is constructed; adaptors are never inserted around live allocations.

Instrumented runs are not authoritative allocator performance measurements.

## 12. Allocation coverage

Every ordinary project-controlled device allocation should use the current RMM
generation:

- columns and tables;
- expression programs and metadata;
- operator outputs and temporary buffers;
- Thrust policies and storage;
- CUB scratch space; and
- allocator-aware cuCollections containers.

The following remain outside the RMM generation when required by their APIs:

- CUDA context, module, and driver state;
- NCCL, MPI, NVBench, and other library internals without allocator hooks;
- fabric, IPC, and exportable VMM allocations;
- pinned host memory; and
- third-party workspaces without a supported allocator or user-workspace hook.

These allocations are invisible to RMM lease/allocation checks and must fit in
the configured reserve. If a third-party component offers a compatible
allocator hook, using the project RMM resource is preferred.

## 13. Cache and allocation identity

A device address is not a stable allocation identity because an arena may reuse
it. Host caches must use a logical owner/allocation identity or unregister
entries on destruction; pointer equality alone can return stale metadata for a
new allocation.

Caches that own GPU data are also generation owners. They must retain leases,
be cleared before reset, and not defer CUDA-touching destruction to process
static teardown.

## 14. Concurrency guarantees

Supported:

- concurrent lazy resource access;
- concurrent allocation and deallocation by multiple correctly initialized
  host threads;
- many-to-many host-thread and CUDA-stream usage;
- live lease diagnostics during ordinary allocation; and
- cross-thread movement/copying of generation handles.

Not supported:

- resetting while another thread allocates, acquires a resource, or submits
  work;
- allocating without selecting the rank device on the current host thread;
- using a buffer after its owner has retired it;
- destroying a buffer after its retirement stream has been destroyed; or
- assuming `set_stream()` creates cross-stream ordering.

The design intentionally makes these boundaries explicit rather than hiding
them behind device-wide synchronization or silent device switches.

## 15. Failure model

- CUDA calls that establish allocator correctness are checked and surfaced as
  exceptions where the API permits.
- Initialization failure is stored in the process context and rethrown to later
  accessors.
- A reset rejected for live leases restores the ready state and original ready
  epoch; it does not invalidate current owners.
- A failure after teardown/replacement begins moves the context to `failed` and
  closes the fast path.
- Destructors remain `noexcept`; debug assertions expose wrong-device cleanup,
  while unrecoverable diagnostic-stack failures terminate.
- There is no automatic fallback from arena mode to direct mode.

## 16. Performance design

The persistent arena amortizes upstream CUDA allocation and provides RMM's
thread/stream-aware arena behavior. The production stack omits tracking and
statistics.

The warmed GaMax allocation path avoids the context mutex through TLS and an
atomic epoch. Its intentional overhead over bare RMM consists of:

- a current-device query;
- epoch and TLS validation;
- generation lease increment/decrement; and
- the project RAII wrapper.

The allocator benchmark compares a 4 KiB GaMax buffer lifetime against a bare
RMM arena with identical tracking/statistics settings. The merge guard requires
median one-thread overhead within 10% and no material two-/four-thread scaling
regression. Commands and the reference result are in the operational quick
reference above.

## 17. Validation strategy

Validation is split by the hardware or instrumentation it requires:

- ordinary CUDA CI: sizing, configuration, generation ownership, stream
  ordering, multithread stress, diagnostics, OOM recovery, library routing, and
  full regression;
- Compute Sanitizer: focused direct/tracked allocation and stream-lifetime
  tests;
- green-context hardware: allocation/readiness and operator correctness on
  partitioned streams;
- multi-GPU hardware: deterministic wrong-device rejection and one-rank-per-GPU
  ALPS tests; and
- performance workers: wrapper-versus-bare latency and scaling.

Unavailable hardware must produce a named skip, not an apparent pass. The
hardware-specific tier must still pass on its designated worker before release
sign-off.

## 18. Rejected alternatives

### Add locks to the custom allocator

This would address only host data races. It would not repair lost stream,
alignment, device, and ownership information or provide a standard allocator
interface to other CUDA libraries.

### One allocator registry for every device in a process

The execution architecture already assigns one process/rank per GPU. A
multi-device registry would add lifecycle and routing complexity without a
supported consumer.

### Use tracker allocation count as the reset proof

An object can own a resource generation without a live allocation. Generation
leases are therefore authoritative for ownership; tracking adds pointer detail
but cannot replace lease validation.

### Keep implicit default-stream allocation

Default arguments hide missed stream propagation and can introduce unwanted
serialization. Internal allocation APIs require an explicit stream; intentional
stream-zero use is visible at the call site.

### Let buffers silently switch CUDA devices

Silent switching masks caller bugs and mutates thread-local CUDA state. Wrong
device use is rejected instead.

### Automatically track all stream consumers

Per-buffer event graphs or shared ownership would impose pervasive runtime and
API cost. Operators already know their dependency structure and must establish
the necessary event joins explicitly.

### Reset after every benchmark cell

Recreating a large arena is expensive and hides lifecycle defects. Successful
cells reuse a generation; reset is reserved for explicit teardown or confirmed
OOM recovery.

## 19. Extension rules

New allocator-aware code should follow these rules:

1. Require an explicit CUDA stream at allocation boundaries.
2. Use `gamax::device_buffer` or the provided RMM-aware wrapper instead of raw
   `cudaMalloc`/`cudaFree` for ordinary device memory.
3. Keep a `resource_handle` alive beside every RMM owning object, including
   empty containers.
4. Declare the handle before the owning RMM member so the RMM object is
   destroyed first.
5. Forward the resource explicitly to Thrust, CUB, and cuCollections.
6. Establish allocation-readiness and retirement ordering when streams differ.
7. Catch after-submit exceptions while owners are still alive and drain/join
   submitted work before unwinding.
8. Treat views as borrowed and keep owners alive through asynchronous use.
9. Give caches stable allocation identities and explicit teardown.
10. Add reset-ownership, stream-lifetime, and allocator-routing tests for each
    new owning abstraction.
11. Do not add a default allocator fallback or silently use stream zero.
12. Keep special VMM/fabric/pinned allocations explicit and account for them in
    the arena reserve.

## 20. Source and dependency map

| Concern | Primary location |
|---|---|
| Process context, generations, reset, diagnostics | `src/cuda/memory/mem_manager.cu` |
| Public handles, policies, allocators, statistics API | `src/cuda/memory/mem_manager.hpp` |
| Project buffer semantics | `src/cuda/memory/device_buffer.hpp` |
| Operational configuration, architecture, and commands | `docs/MEMORY_ALLOCATOR.md` |
| Allocator unit and stream tests | `src/cuda/tests/memory/mem_manager_test.cu` |
| Focused sanitizer tests | `src/cuda/tests/memory/allocator_sanitizer_test.cu` |
| Latency/scaling benchmark | `src/cuda/benchmarks/memory/allocator_latency_bench.cu` |

The allocator uses the RAPIDS 26.06 dependency cohort: RMM 26.06, its pinned
CCCL revision, and its pinned cuCollections revision. `rmm::rmm` is a public
dependency because project headers contain RMM types.
