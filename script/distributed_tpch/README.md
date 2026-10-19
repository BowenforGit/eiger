<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# Distributed TPC-H reproduction runner

For a fresh-machine walkthrough covering dependencies, compilation, dataset
preparation, local and multi-node runs, profiling, and result handling, see
[`docs/DISTRIBUTED_TPCH_BENCHMARK_GUIDE.md`](../../docs/DISTRIBUTED_TPCH_BENCHMARK_GUIDE.md).

`run.py` runs a sequential distributed TPC-H experiment from an already-running
container. It supports a local multi-GPU node or an explicitly selected subset
of directly reachable multi-GPU hosts. It does not run a feasibility gate, and
Slurm support is intentionally postponed until it can be tested.

## Campaign lifecycle

A campaign is created by exactly one `new` invocation. The campaign directory
must not exist, even if it would be empty:

```bash
python3 script/distributed_tpch/run.py new /results/tpch-sf300 \
  --build-dir /path/to/eiger/build-b200 \
  --tpch-data /path/to/tpch \
  --scale 300 \
  --arms legacy,legacy_aligned_both \
  --stages timing,breakdown,memory,profiles \
  --queries 1-22
```

Scale factor, queries, arms, stages, topology, binaries, data, runtime policy,
and synthesis policy are frozen in the campaign manifest. Resume consequently
accepts only the campaign directory:

```bash
python3 script/distributed_tpch/run.py resume /results/tpch-sf300
```

Resume reruns the frozen matrix. A query cell is skipped when a prior attempt
passed and all of its expected artifacts are still nonempty. Failed or
interrupted evaluation cells get a new immutable attempt directory. Completed
synthesis queries are skipped, while incomplete synthesis queries reuse their
checkpoints. Each invocation is recorded under `invocations/` without changing
the immutable campaign fingerprint.

Campaign status can be inspected without running or modifying the campaign:

```bash
python3 script/distributed_tpch/run.py status /results/tpch-sf300
```

The command compares the frozen manifest matrix with the latest `status.csv`
row for every arm, stage, and query. It reports completion, the overall result,
failed and dependency-blocked cases, and up to twenty missing cases. A failed
cell that later passes on resume is considered successful. A synthesized
evaluation cell is recorded as `blocked` when its query plan is missing, and
the MPI or profiler command is not launched. Exit status is 0 for complete
success, 1 for a complete campaign with failures or blocked cells, 2 for an
incomplete campaign, and 64 for an invalid campaign or status journal. Because
campaigns do not have a live-process marker, incomplete means the campaign is
either still running or was interrupted.

Selected unsuccessful cells can be retried without replaying the full frozen
matrix:

```bash
python3 script/distributed_tpch/run.py retry /results/tpch-sf1000 \
  --queries 21 --allow-code-drift
```

`--queries`, `--arms`, and `--stages` restrict the selection. Within that
selection, retry runs only the latest `fail`, `blocked`, or missing cells.
Passed cells remain untouched, except that timing and breakdown are promoted
together because the campaign executes them as one combined measurement. When
a selected synthesized evaluation has no plan, its synthesis cell is added as
a dependency and runs first. Use `--dry-run` to show the expanded cell set and
resolved commands without creating an invocation, attempt, or status row.

Retry normally enforces the same integrity checks as resume. The explicit
`--allow-code-drift` exception permits only saved workflow and executable
hashes to differ, which is needed to apply a runner or binary bug fix to an old
campaign. Manifest integrity, dataset contents, topology, runtime environment,
NCCL configuration, imported plans, and agreement between current local and
remote executables remain strict. The accepted old/new hashes and expanded
cell set are recorded in the retry invocation; the immutable campaign manifest
is not rewritten.

The current manifest schema is version 4. Older campaigns are rejected rather
than silently migrated because CTA ownership changes the meaning of an
unsuffixed baseline arm.

The result tree is organized as arms, then stages, then queries:

```text
<campaign>/
├── manifest/{campaign.json,preflight.json}
├── invocations/*.json
├── status.csv
└── arms/
    ├── legacy/timing/q01/attempt-0001/...
    └── synthesized/
        ├── synthesis/q01/{plan.json,checkpoint.json,candidates.csv,...}
        └── timing/q01/attempt-0001/...
```

`new --dry-run` resolves and prints the commands but does not create the
campaign directory. An already existing path is still rejected.

## Host-local generated data

Use `--generate-local-data` to avoid reading table data from shared storage.
The shared `--tpch-data` scale directory is still used for verification
goldens, while each selected host generates the files needed by its MPI ranks
under `/tmp/alps-tpch`:

```bash
python3 script/distributed_tpch/run.py new /shared/results/tpch-sf3000 \
  --build-dir /shared/eiger/build-b200 \
  --tpch-data /storage/tpch --scale 3000 \
  --hosts gb200-001:4,gb200-002:4 --gpus-per-node 4 \
  --generate-local-data --generated-data-partitions 64 \
  --arms legacy --stages timing --queries 1-22
```

The global partition count is fixed in the campaign. Use one partition for
single-file, row-group sharding, or at least as many partitions as MPI ranks
for round-robin file sharding. Partitioned host-local generation rejects a
conflicting synthesis `--file-shard-strategy` override. Each invocation
prepares every selected host before MPI starts. Existing files with the same
scale factor, global partition count, table, and partition number are reused;
files missing after a host restart are regenerated automatically by `resume`
and `retry`.

Preparation runs concurrently across hosts and divides the host's available
CPUs among concurrent `tpchgen-cli` processes. The launcher publishes a global
manifest under the campaign and installs it in each partial local cache, so the
existing manifest reader remains in use. The normal shared-data path remains
the default when `--generate-local-data` is omitted. Override the defaults with
`--local-data-cache-root` or `--tpchgen` if needed. Only the coordinator needs
the configured generator installed: the launcher stages an executable copy in
the shared campaign directory for all selected hosts.

### Collecting measurements after a verification mismatch

By default, a golden-result mismatch stops the current benchmark before its
measurement CSVs are written. Add `--collect-on-verify-failure` when creating
the campaign to finish the current timing, breakdown, or memory collection and
retain its artifacts before returning a failed verdict:

```bash
python3 script/distributed_tpch/run.py new /results/tpch-diagnostic \
  ... \
  --collect-on-verify-failure \
  --keep-going
```

The attempt still has `status: fail`; it is never promoted as a successful
measurement. Its `result.json` records `failure_kind: verification`, the
verification status, and whether the expected measurements are complete. The
full verifier diagnostic is stored in `verification.json`. `--keep-going` is
orthogonal: use it as above to continue other cells after the completed failed
attempt; without it, the campaign stops after preserving the current cell.

### Per-query GPU memory reserve

`--memory-reserve-bytes` remains the campaign default. Repeat
`--query-memory-reserve-bytes QUERY=BYTES` to override it for individual
selected queries without splitting them into separate campaigns. Query IDs may
be written with or without the `Q` prefix:

```bash
python3 script/distributed_tpch/run.py new \
  /results/tpch-sf3000-16gpu-ce-pull \
  --build-dir /storage/alps-tpch/build \
  --tpch-data /storage/tpch \
  --scale 3000 \
  --arms all_co_ce_pull \
  --stages timing,breakdown,memory,profiles \
  --queries 1-22 \
  --memory-reserve-bytes 40000000000 \
  --query-memory-reserve-bytes Q1=12000000000 \
  --query-memory-reserve-bytes Q8=48000000000 \
  --query-memory-reserve-bytes Q9=58000000000 \
  --query-memory-reserve-bytes Q21=20000000000
```

The default and normalized override map are frozen in `manifest/campaign.json`.
Synthesis and every evaluation stage for a query receive the same resolved
`GAMAX_MEM_RESERVE_BYTES`; resume and retry replay that frozen value. Each
evaluation attempt records the effective value in both `command.json` and
`result.json`, and dry runs print it with the other controlled environment
settings. Schema-version 4 manifests created before this option are interpreted
as having no query overrides; the existing workflow-hash integrity check still
applies to resume and retry.

An override is query-scoped and therefore applies to every arm in the
campaign. The SF3000 values above were measured for `all_co_ce_pull`; they
should not be reused blindly in a campaign containing different arms whose RMM
and VMM balance may differ.

### Parquet decode threads

`--parquet-decode-threads N` controls Arrow CPU decode threads per MPI rank and
defaults to 8. The value must be positive. It is frozen in
`manifest/campaign.json`, forwarded to both plan synthesis and every evaluation
stage, and restored unchanged by resume and retry. Campaigns created before
this option was exposed are interpreted as using the historical default of 8;
the normal workflow-hash validation still applies when replaying them.

For example, a 24-rank campaign using four decode threads per rank includes:

```bash
  --hosts gb200-005:4,gb200-015:4,gb200-016:4,gb200-017:4,gb200-004:4,gb200-008:4 \
  --gpus-per-node 4 \
  --parquet-decode-threads 4
```

This controls CPU decoding, not the number of files assigned to a rank. Each
rank still visits its assigned Parquet files sequentially, while Arrow may
decode projected columns from the current file in parallel.

## Inputs and build selection

`--tpch-data` names the root containing `sf<N>` directories, not the scale
directory itself. For example, use `/storage/tpch` for
`/storage/tpch/sf300`. The selected scale must contain all eight tables and a
`golden/qNN.parquet` for every selected query. `TPCH_DATA_PATH` is the fallback.

`--build-dir` names a CMake build tree containing the required executables under
`src/cuda/tpch/`. `EIGER_BUILD_DIR` is the fallback. Without either, the runner
accepts a repository `build*` tree only if exactly one contains all executables
required by the selected stages. Paths and content hashes are recorded and
checked again on resume.

The base arms are `legacy`, `legacy_ce_pull`, `legacy_aligned_both`,
`all_co_nccl`, `all_co_ce_pull`, `group_co_nccl`, `group_co_ce_pull`,
`semi_co_nccl`, `semi_co_ce_pull`, and `synthesized`.
`legacy_ce_pull` selects CE-pull globally, so Legacy shuffle, replication,
allgather, and gather-to-root paths all use bounded CE-pull transfers.
`all_co_ce_pull` does the same while ColumnOverlap operators inherit the global
backend. The isolated `group_co_ce_pull`/`semi_co_ce_pull` arms keep global
communication on NCCL and use `--column-exchange-backend ce-pull` only for the
selected ColumnOverlap exchange.

The established `legacy`, `legacy_aligned_both`, `all_co_nccl`, and
`synthesized` NCCL bases also have fixed CTA variants with suffixes `_cta8`,
`_cta16`, `_cta32`, and `_cta64`. For example, `all_co_nccl_cta32` uses the
same benchmark arguments as `all_co_nccl` while setting both `NCCL_MIN_CTAS`
and `NCCL_MAX_CTAS` to 32. CE-pull and operator-isolated arms intentionally do
not add CTA variants.
The stages are `synthesis`, `timing`, `breakdown`, `memory`, `messages`,
`profiles`, and `ncu_profiles`.
Arms and stages are required on `new` to avoid accidentally starting an
expensive default matrix.

### Exact NCCL API payload sizes

The `messages` stage is a Legacy-only diagnostic pass. After the normal
adaptive warmup, it executes one verified query iteration and writes one row
per nonzero NCCL API invocation. It is deliberately independent of timing and
breakdown so tracing cannot alter their samples:

```bash
python3 script/distributed_tpch/run.py new /results/tpch-sf1000-messages \
  --build-dir build-b200 --tpch-data /storage/tpch --scale 1000 \
  --arms legacy --stages messages --queries 1-22 \
  --memory-reserve-bytes 40000000000
```

Artifacts use the ordinary immutable-attempt layout at
`arms/legacy/messages/qNN/attempt-NNNN/messages.csv`; `status`, `resume`, and
`retry --stages messages` work like every other stage. Existing campaigns
cannot gain a new stage because their manifests are immutable, so collect
message traces in companion campaigns with the same topology and runtime
settings.

`messages.csv` reports `element_count`, `element_size`, and their exact product
`payload_bytes`. Fixed columns produce one NCCL value operation per peer.
Strings produce separate `string_length` (`INT64`, eight bytes per row) and
`string_chars` (`INT8`, actual character bytes) operations; Arrow offsets are
rebuilt on the receiver and are not transmitted. P2P send and receive calls
are both retained, and self-peer NCCL calls are marked with `is_self=1`.
These are application payload sizes, not NCCL protocol or physical wire bytes.

### Rank-0 Nsight Compute profiles

`ncu_profiles` runs the ordinary warmed `--profile` workload, but only MPI rank
0 is launched under Nsight Compute. Collection is limited to the existing
`gamax.tpch@tpch.profile/` NVTX push/pop range. Only kernels whose demangled
names contain one of the literal substrings in
[`ncu_kernels.txt`](../../ncu_kernels.txt) are profiled; kernels containing
`nccl` (case insensitively) remain excluded. The range is interpreted at
process scope so launches from any rank-0 host thread are eligible. Peer ranks
run normally and synchronize with rank 0 as the profiled local kernels are
replayed.

The complete metric selection lives in one native Nsight Compute section,
[`ncu_metrics.section`](../../ncu_metrics.section). It contains 2,134 exact
GB200 counter, ratio, and throughput metrics plus all 18 `launch__*`
attributes from NCU's `LaunchStats` section. Thirty-six base selectors from the
source inventory have no collectable suffix on GB200 with NCU 2025.3.1; they
are retained as comments in the section rather than producing `n/a` rows or
aborting the capture. The same file is passed directly to both the standalone
preflight and the rank-0 capture with
`ncu --section-folder ... --section AlpsTpchMetrics`; there is no CSV parser or
generated `--metrics` command string:

```bash
python3 script/distributed_tpch/run.py new /results/tpch-sf1000-ncu \
  --build-dir build --tpch-data /storage/tpch --scale 1000 \
  --hosts gb200-005:4,gb200-015:4 --gpus-per-node 4 --ssh-port 2222 \
  --arms legacy_cta64 --stages ncu_profiles --queries 1 \
  --ncu-metrics-file /storage/alps-tpch/ncu_metrics.section \
  --ncu-kernels-file /storage/alps-tpch/ncu_kernels.txt \
  --ncu-capture-timeout 7200
```

The section selectors, kernel substrings, and file digests are frozen into the
campaign manifest. The inputs are copied to `manifest/ncu_metrics.section` and
`manifest/ncu_kernels.txt`. Before
launching TPC-H, the runner profiles a standalone CUDA kernel on rank 0's host.
This verifies that NCU can load the section, access the counters, collect every
selector and launch attribute, and honor the NCCL-excluding kernel regex.
Counter permission is needed only on rank 0's node.

Each successful attempt contains one `qNN_<host>_rank0.ncu-rep`, a detailed
`qNN_ncu_metrics.csv` export, and `qNN_ncu_validation.json`. Validation rejects
empty reports, missing requested or launch metrics, captured kernels outside
the selected subset, and any captured NCCL kernel. A timeout terminates the MPI launch rather than allowing a
replay-unsafe kernel to hang indefinitely.

Rank-0 replay is contention-free. Launch geometry, registers, shared memory,
occupancy, and counter totals are the intended evidence. Kernel duration,
throughput percentages, bandwidth, latency, and stall results are isolated and
must not be presented as contention-realistic distributed measurements.
Non-NCCL does not itself guarantee replay safety: custom kernels with an
in-kernel peer dependency can still deadlock, while cross-GPU writes can make
replayed movement counters indicative rather than exact.

Summarize one or more completed campaigns with:

```bash
python3 tools/visualize/summarize_distributed_tpch_messages.py \
  /results/tpch-sf1000-8gpu-messages \
  /results/tpch-sf1000-16gpu-messages \
  --output /results/tpch-message-summary.csv
```

Receive rows are excluded by default to avoid double-counting point-to-point
payload. Add `--include-receives` for API-call completeness or `--by-rank` for
rank-level distributions.

For an operator-isolated timing campaign without Nsight collection:

```bash
python3 script/distributed_tpch/run.py new /results/tpch-relational-ce \
  --build-dir build-b200 --tpch-data /storage/tpch --scale 300 \
  --arms group_co_nccl,group_co_ce_pull,semi_co_nccl,semi_co_ce_pull \
  --stages timing --queries 1,4,9,16,20,21
```

CE-pull timing and breakdown cells include `ce_pull.csv`. The runner checks
this artifact during resume for both the all-ColumnOverlap and isolated CE-pull
arms.

Unsuffixed arms use NCCL's automatic CTA policy. The runner removes inherited
CTA controls, including deprecated channel/ring aliases, before every rank
launch. It warns about and records inherited values rather than allowing them
to contaminate the matrix. Campaign creation inspects `/etc/nccl.conf`, the
selected `NCCL_CONF_FILE` or `~/.nccl.conf`, and NCCL environment plugins on
every selected host; a source that can inject CTA controls is rejected. The
inspected sources and hashes are frozen in the campaign manifest and rechecked
on resume.

Environment-plugin discovery is best-effort. The preflight searches the
ambient loader configuration and the library directories that the launchers
add for MPI ranks, but it does not load or introspect a plugin. A plugin exposed
later through a custom loader namespace, changed mount, or other dynamic
mechanism can therefore evade the static check. For explicit remote-host runs,
only participating hosts are blocking; coordinator-only NCCL configuration is
recorded as inherited process context when applicable but does not reject the
campaign.

CTA variants are ordinary explicit arms and can be mixed with their automatic
baselines:

```bash
python3 script/distributed_tpch/run.py new /results/tpch-cta \
  --build-dir build-b200 --tpch-data /storage/tpch --scale 300 \
  --arms legacy,legacy_cta8,legacy_cta16,legacy_cta32,legacy_cta64 \
  --stages timing,breakdown --queries 1-22
```

All `synthesized_cta<N>` arms share the single plan stored under
`arms/synthesized/synthesis`; synthesis itself uses automatic NCCL CTA policy.

When both `timing` and `breakdown` are selected, the runner collects them in one
C++ benchmark invocation per arm/query: one data load and adaptive warmup, then
ten clean end-to-end samples followed by five instrumented step/phase samples.
Their artifacts and resume records remain separated under the two stage
directories. `memory` remains a clean, independent benchmark invocation, and
`profiles` remains an independent Nsight capture.

## Synthesized plans

To synthesize plans and then evaluate them:

```bash
python3 script/distributed_tpch/run.py new /results/tpch-plan-comparison \
  --build-dir build-b200 --tpch-data /storage/tpch --scale 300 \
  --arms legacy,synthesized \
  --stages synthesis,timing,breakdown,memory,profiles \
  --queries 1-22
```

The synthesis stage runs before evaluation. Its timeout, restart count, seed,
and extra binary arguments are configurable through the `--synthesis-*`
options.

If the initial seed plan exceeds the partition GroupBy slot-table capacity,
the synthesizer catches that specific failure, replaces every fixed partition
GroupBy seed override with HASH, records the replacements in the plan's
`seed_fallbacks` array, rebuilds the seed plan, and verifies it before search.
Other seed failures are terminal. The launcher also stops immediately for a
terminal synthesis exit and only restarts a crashed or timed-out process when
the checkpoint digest changed during that attempt. This prevents identical
attempts from being launched repeatedly when no resumable progress was made.

Existing guarded plans may instead be imported on campaign creation:

```bash
python3 script/distributed_tpch/run.py new /results/tpch-plan-evaluation \
  --build-dir build-b200 --tpch-data /storage/tpch --scale 300 \
  --physical-plan-dir /results/old-plans \
  --arms legacy,synthesized --stages timing,breakdown --queries 1,6,9-12
```

The source directory must contain `qNN.json`. Selected plans are copied into
`arms/synthesized/synthesis/qNN/plan.json` and hashed, so subsequent resume does
not require the external plan directory.

## Explicit multi-node execution

The runner uses only the hosts supplied to `--hosts`; it never claims all nodes
of an MNNVL machine:

```bash
python3 script/distributed_tpch/run.py new /shared/results/tpch-2node \
  --build-dir /shared/eiger/build-b200 --tpch-data /shared/tpch --scale 300 \
  --hosts gb200-001:4,gb200-002:4 --gpus-per-node 4 --ssh-port 2222 \
  --arms legacy --stages timing --queries 1-22
```

Selected hosts must expose the specified uniform GPU count and see the build,
data, campaign, and profiler-launcher paths at the same absolute locations.
Preflight checks those paths, table and golden data, GPU idleness, executable
hashes, and Nsight availability when requested. Coordinator GPUs are not
selected for a remote-only run. Local profiling produces one report; multi-node
profiling requires one report per rank.

Before any profile query starts, the runner tests the selected GPUs one at a
time with a bounded Nsight Systems hardware-metrics capture. A candidate passes
only when Nsight exits before `--profile-preflight-timeout`, its report exports
successfully to SQLite, and both the `GPU_METRICS` sample table and metric
definition table contain rows. The first passing physical GPU is recorded in
`manifest/profile_gpu.json`. Subsequent profile queries collect hardware
metrics only for that GPU; CUDA/NVTX tracing still covers the full local MPI
tree or all selected multi-node ranks. Resume revalidates the recorded GPU and
does not silently switch the campaign to another device.

The defaults are a two-second validation capture, a twenty-second timeout per
candidate, and the same 10 kHz sampling frequency used by profile queries.
They can be changed on `new` with `--profile-preflight-seconds`,
`--profile-preflight-timeout`, and `--profile-gpu-metrics-frequency`.
