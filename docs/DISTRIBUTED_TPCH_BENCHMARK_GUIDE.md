<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# Running the distributed TPC-H benchmarks

This guide starts with a fresh checkout and ends with reproducible local or
multi-node benchmark campaigns driven by
`script/distributed_tpch/run.py`. Run all commands from the repository root
unless noted otherwise.

The runner launches one MPI rank per GPU. It supports timing, operator/phase
breakdowns, peak-memory measurement, NCCL message tracing, Nsight Systems and
Nsight Compute profiles, and physical-plan synthesis. Campaign inputs are
frozen at creation so interrupted work can be resumed without silently
changing the experiment.

## 1. Get the source

```bash
git clone --recursive <repository-url> alps-tpch
cd alps-tpch

# Needed when the initial clone did not use --recursive.
git submodule update --init --recursive
```

CMake downloads the pinned RAPIDS 26.06/RMM, CCCL, cuCollections, and test
dependencies during configuration. The configuring machine therefore needs
network access or a pre-populated CMake/CPM cache.

## 2. Install dependencies

### Required hardware and system software

- One or more NVIDIA GPUs, with a compatible NVIDIA driver and CUDA Toolkit.
- Git, including submodule support.
- CMake 3.30.4 or newer and Ninja.
- A host compiler supported by the installed CUDA Toolkit, with C++20 support.
- OpenMP development files.
- MPI with C++ support. Use the same MPI installation to compile and run.
- NCCL headers and shared library.
- Apache Arrow and Parquet C++ libraries.
- OpenSSL and RapidJSON development files.
- Python 3.10 or newer.
- `nvidia-smi`, which the runner uses for inventory and idle-GPU checks.

Multi-node execution additionally requires direct SSH access between the
coordinator and every selected host. The current runner operates inside an
already-running environment or container; it does not allocate Slurm nodes.

The exact system packages are platform-specific. A conda environment is a
convenient way to provide the host-side build and analysis dependencies while
CUDA, MPI, and NCCL come from the machine or cluster installation:

```bash
conda create -n alps-tpch -c conda-forge \
  'python>=3.10' 'cmake>=3.30.4' ninja \
  libarrow libparquet openssl rapidjson \
  duckdb numpy pandas matplotlib pyarrow
conda activate alps-tpch

# Version 3.0.0 matches the CLI used by the repository's generation helpers.
python -m pip install 'tpchgen-cli==3.0.0'
```

`duckdb` is needed only to generate golden query answers. NumPy, pandas,
Matplotlib, and PyArrow are analysis/plotting dependencies; the campaign
runner itself uses the Python standard library.

Verify that the external toolchain is the one you intend to use:

```bash
nvcc --version
nvidia-smi
cmake --version
ninja --version
mpicxx --version
mpirun --version
```

If NCCL, MPI, or Arrow are installed in nonstandard prefixes, expose them
before configuring. Adjust these examples for the local installation:

```bash
export NCCL_ROOT=/path/to/nccl
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

The CMake NCCL finder honors `NCCL_ROOT` and `NCCL_HOME`. CMake's MPI finder
uses the selected `mpicxx`; an explicit compiler path is shown below to avoid
accidentally compiling against one MPI and launching with another.

### Optional profiling tools

- Install Nsight Systems and put `nsys` on `PATH` for the `profiles` stage.
- Install Nsight Compute and put `ncu` on `PATH` for the `ncu_profiles` stage.
- NCU performance counters must be enabled for the user on rank 0's host.

The repository's default `ncu_metrics.section` was resolved for GB200 with
Nsight Compute 2025.3.1. Use a platform-specific native NCU section on other
architectures rather than assuming every selector is portable.

## 3. Configure and compile

Choose the CUDA architecture for the target GPU. Common values are 70 for
V100, 75 for T4, 80 for A100, 90 for H100, and 100 for B200/GB200.

```bash
export BUILD_DIR="$PWD/build"
export CUDA_ARCH=100                    # change for the target GPU
export MPI_CXX_COMPILER="$(command -v mpicxx)"

cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DMPI_CXX_COMPILER="$MPI_CXX_COMPILER" \
  -DBUILD_ALPS=ON \
  -DBUILD_TPCH=ON \
  -DBUILD_TEST=ON \
  -DBUILD_BENCHMARK=OFF
```

If Arrow or Parquet is not found through `CMAKE_PREFIX_PATH`, pass their CMake
package directories explicitly:

```bash
cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DMPI_CXX_COMPILER="$MPI_CXX_COMPILER" \
  -DArrow_DIR="$CONDA_PREFIX/lib/cmake/Arrow" \
  -DParquet_DIR="$CONDA_PREFIX/lib/cmake/Parquet" \
  -DBUILD_ALPS=ON -DBUILD_TPCH=ON -DBUILD_TEST=ON \
  -DBUILD_BENCHMARK=OFF
```

Build all executables used by the campaign runner:

```bash
cmake --build "$BUILD_DIR" --target \
  benchmark_tpch_ir_mpi \
  synthesize_tpch_ir_mpi \
  tpch_parquet_metadata \
  -j"$(nproc)"
```

`benchmark_tpch_ir_mpi` also builds the CUDA device-count and NCU preflight
helpers. The important outputs are:

```text
build/src/cuda/tpch/benchmark_tpch_ir_mpi
build/src/cuda/tpch/synthesize_tpch_ir_mpi
build/src/cuda/tpch/tpch_parquet_metadata
build/src/cuda/tpch/cuda_visible_device_count
build/src/cuda/tpch/ncu_counter_probe
```

Run the lightweight runner tests and executable checks before using GPU time:

```bash
python3 -m unittest discover -s script/distributed_tpch/tests -p 'test_*.py'

ctest --test-dir "$BUILD_DIR" --output-on-failure \
  -R 'distributed_tpch_(benchmark_help|benchmark_query_help|visible_device_count)'
```

For a multi-node run, the repository, build tree, dataset, campaign directory,
and profiling launchers must be visible at the same absolute paths on every
selected host. Build once on shared storage or reproduce the exact build at
the same path everywhere.

## 4. Prepare TPC-H data and goldens

`--tpch-data` names a root containing scale-factor directories, not an
individual scale directory. Every selected query requires a golden result:

```text
<tpch-root>/
  sf1/
    customer.parquet
    lineitem.parquet
    nation.parquet
    orders.parquet
    part.parquet
    partsupp.parquet
    region.parquet
    supplier.parquet
    golden/q01.parquet ... q22.parquet
```

### Small, single-file dataset

Install a [`tpchgen-cli`](https://pypi.org/project/tpchgen-cli/) capable of
Parquet output, then generate a small scale for the first smoke test:

```bash
export TPCHGEN=/path/to/tpchgen-cli
export TPCH_DATA_ROOT=/path/to/tpch

mkdir -p "$TPCH_DATA_ROOT/sf1"
"$TPCHGEN" parquet \
  --scale-factor 1 \
  --output-dir "$TPCH_DATA_ROOT/sf1" \
  --num-threads 16 \
  --compression SNAPPY
```

Generate reference results with DuckDB:

```bash
python3 script/gen_tpch_golden.py \
  --sf 1 \
  --out "$TPCH_DATA_ROOT" \
  --queries 1-22
```

DuckDB downloads its TPC-H extension on first use, so that first invocation
needs network access or a pre-installed extension.

### Large, partitioned dataset

For large scales, use the restartable generator and publish only after the
complete dataset has been validated. The publish destination must not exist:

```bash
export SCALE=1000
export PARTS=144
export STAGING="$TPCH_DATA_ROOT/sf${SCALE}.staging.$USER"

python3 script/generate_tpch_multifile.py \
  --scale "$SCALE" \
  --parts "$PARTS" \
  --staging-dir "$STAGING" \
  --published-dir "$TPCH_DATA_ROOT/sf${SCALE}" \
  --generator "$TPCHGEN" \
  --threads 16

python3 script/finalize_tpch_manifest.py \
  --staging-dir "$STAGING" \
  --metadata-tool "$BUILD_DIR/src/cuda/tpch/tpch_parquet_metadata" \
  --publish-dir "$TPCH_DATA_ROOT/sf${SCALE}"

python3 script/gen_tpch_golden.py \
  --sf "$SCALE" \
  --out "$TPCH_DATA_ROOT" \
  --queries 1-22
```

Re-running the identical generation command resumes completed parts. See
`docs/TPCH_MULTIFILE_DATASET.md` before changing the part count,
publication procedure, or manifest layout.

### Optional host-local generation

For multi-node experiments on shared storage, `run.py new` can generate each
host's assigned table parts under local temporary storage. Add:

```text
--generate-local-data
--generated-data-partitions <at-least-the-total-rank-count>
--tpchgen /path/to/tpchgen-cli
```

The shared `sf<N>` directory and its goldens are still required. Only the
coordinator needs the generator initially; the runner copies it into the
campaign and stages the required files for the selected hosts.

## 5. Prepare the runtime environment

Ensure the runtime linker can find Arrow, Parquet, CUDA, MPI, NCCL, and the
libraries in the build tree. The launchers add the build-tree directories and
preserve the caller's `LD_LIBRARY_PATH`, so export nonstandard external paths
before creating the campaign:

```bash
export PATH="/path/to/mpi/bin:/path/to/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/path/to/mpi/lib:/path/to/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TPCH_DATA_PATH="$TPCH_DATA_ROOT"
export EIGER_BUILD_DIR="$BUILD_DIR"
```

On an ordinary PCIe/NVLink system, disabling MNNVL discovery is a conservative
starting point. Enable it only when the platform and NCCL installation support
MNNVL:

```bash
export NCCL_MNNVL_ENABLE=0             # use 1 on a configured MNNVL system
```

For a local run, select the exact GPUs before invoking the runner:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

Do not share those GPUs with another workload. Local preflight rejects active
GPU compute processes. The campaign also records GPU inventory, topology, MPI
version, binary hashes, dataset metadata, and controlled NCCL settings.

For multi-node execution, verify before launching:

- the same MPI and NCCL versions are available on every host;
- passwordless SSH works on the configured port;
- the selected network interface exists on every host;
- all selected GPUs are idle;
- build, data, campaign, and script paths resolve identically everywhere; and
- no `/etc/nccl.conf`, `NCCL_CONF_FILE`, or NCCL environment plugin injects
  conflicting CTA/channel settings.

## 6. Run a local smoke campaign

Start with SF1, one query, the baseline arm, and timing only. A new campaign
directory must not already exist.

```bash
export CAMPAIGN="$PWD/results/distributed_tpch/sf1-local-smoke"
export MPIRUN="$(command -v mpirun)"

python3 script/distributed_tpch/run.py new "$CAMPAIGN" \
  --build-dir "$BUILD_DIR" \
  --tpch-data "$TPCH_DATA_ROOT" \
  --scale 1 \
  --gpus-per-node 2 \
  --mpirun "$MPIRUN" \
  --memory-reserve-bytes 4294967296 \
  --arms legacy \
  --stages timing \
  --queries 6 \
  --dry-run
```

Inspect the resolved topology and command. If they are correct, rerun without
`--dry-run`:

```bash
python3 script/distributed_tpch/run.py new "$CAMPAIGN" \
  --build-dir "$BUILD_DIR" \
  --tpch-data "$TPCH_DATA_ROOT" \
  --scale 1 \
  --gpus-per-node 2 \
  --mpirun "$MPIRUN" \
  --memory-reserve-bytes 4294967296 \
  --arms legacy \
  --stages timing \
  --queries 6
```

Change `--gpus-per-node` to match the number of devices in
`CUDA_VISIBLE_DEVICES`. The 4 GiB reserve is a portable SF1 starting point;
larger runs need a reserve chosen for the GPU capacity, NCCL workspace, arm,
and query.

Check the result without modifying it:

```bash
python3 script/distributed_tpch/run.py status "$CAMPAIGN"
```

## 7. Run on explicit remote hosts

The host list is exact; the runner never claims additional nodes. Each
`host:slots` value is the number of MPI ranks/GPUs on that host:

```bash
export CAMPAIGN=/shared/results/distributed_tpch/sf100-8gpu-baseline

python3 script/distributed_tpch/run.py new "$CAMPAIGN" \
  --build-dir /shared/alps-tpch/build \
  --tpch-data /shared/tpch \
  --scale 100 \
  --hosts gpu01:4,gpu02:4 \
  --gpus-per-node 4 \
  --ssh-port 22 \
  --mpirun /path/to/mpirun \
  --nccl-socket-ifname eth0 \
  --memory-reserve-bytes 12000000000 \
  --arms legacy \
  --stages timing,breakdown \
  --queries 1-22 \
  --keep-going
```

Use a shared campaign path: rank launchers and remote preflight access it by
the same absolute name. For host-local table reads, add the generation options
from the previous section.

## 8. Choose arms and stages

Print the authoritative lists and current descriptions with:

```bash
python3 script/distributed_tpch/run.py --help
python3 script/distributed_tpch/run.py new --help
```

Useful starting arms are:

- `legacy`: stock physical operators and NCCL communication.
- `legacy_ce_pull`: stock operators with CE-pull bulk communication.
- `all_co_nccl`: ColumnOverlap operators with NCCL exchange.
- `all_co_ce_pull`: ColumnOverlap operators with CE-pull exchange.
- `legacy_cta8`, `legacy_cta16`, `legacy_cta32`, `legacy_cta64`: fixed NCCL
  CTA policies.
- `synthesized`: guarded per-query plans produced by the synthesis stage or
  imported from `--physical-plan-dir`.

Available stages have different cost and dependencies:

| Stage | Output | Notes |
| --- | --- | --- |
| `timing` | Ten clean end-to-end samples | Recommended first measurement. |
| `breakdown` | Five step and phase samples | Shares load/warmup with `timing` when both are selected. |
| `memory` | One RMM peak-memory sample | Independent benchmark invocation. |
| `messages` | Exact NCCL API payload rows | Currently requires the `legacy` arm. |
| `profiles` | Nsight Systems reports | Requires `nsys` on participating hosts. |
| `ncu_profiles` | Rank-0 NCU report and metric CSV | Requires `ncu`, counter access, and replay-safe selected kernels. |
| `synthesis` | Guarded physical plans | Build `synthesize_tpch_ir_mpi`; potentially long-running. |

A normal comparison campaign can group timing, breakdown, and memory:

```bash
python3 script/distributed_tpch/run.py new /shared/results/tpch-comparison \
  --build-dir "$BUILD_DIR" \
  --tpch-data "$TPCH_DATA_ROOT" \
  --scale 100 \
  --arms legacy,all_co_nccl,all_co_ce_pull \
  --stages timing,breakdown,memory \
  --queries 1-22 \
  --memory-reserve-bytes 12000000000 \
  --keep-going
```

Create profiling and message-tracing campaigns separately so instrumentation
does not contaminate timing samples and so each campaign has a clear purpose.

To synthesize and evaluate plans in one campaign:

```bash
python3 script/distributed_tpch/run.py new /shared/results/tpch-synthesis \
  --build-dir "$BUILD_DIR" \
  --tpch-data "$TPCH_DATA_ROOT" \
  --scale 100 \
  --arms legacy,synthesized \
  --stages synthesis,timing,breakdown,memory \
  --queries 1-22 \
  --keep-going
```

## 9. Resume, inspect, and retry

Campaign configuration is immutable. Do not rerun `new` on the same path and
do not edit the manifest. Resume an interrupted campaign with only its path:

```bash
python3 script/distributed_tpch/run.py resume "$CAMPAIGN"
```

Inspect completion and failures:

```bash
python3 script/distributed_tpch/run.py status "$CAMPAIGN"
```

Preview and retry selected unsuccessful cells:

```bash
python3 script/distributed_tpch/run.py retry "$CAMPAIGN" \
  --queries 6,9-12 \
  --stages timing,breakdown \
  --dry-run

python3 script/distributed_tpch/run.py retry "$CAMPAIGN" \
  --queries 6,9-12 \
  --stages timing,breakdown
```

Use `--allow-code-drift` only when intentionally retrying an old campaign with
a changed runner or executable. The exception is recorded and does not relax
dataset, topology, environment, plan, or manifest checks.

The result tree is immutable by attempt:

```text
<campaign>/
  manifest/campaign.json
  manifest/preflight.json
  invocations/*.json
  status.csv
  arms/<arm>/<stage>/qNN/attempt-NNNN/
```

Read `result.json`, `command.json`, `stdout.log`, and the stage CSVs inside an
attempt when diagnosing a failure. A later successful retry supersedes an
earlier failure for campaign status without deleting the failed attempt.

## 10. Profiling examples

Create a separate Nsight Systems campaign:

```bash
python3 script/distributed_tpch/run.py new /shared/results/tpch-nsys \
  --build-dir "$BUILD_DIR" \
  --tpch-data "$TPCH_DATA_ROOT" \
  --scale 100 \
  --arms legacy \
  --stages profiles \
  --queries 6 \
  --memory-reserve-bytes 12000000000
```

For the repository's GB200 NCU selection:

```bash
python3 script/distributed_tpch/run.py new /shared/results/tpch-ncu \
  --build-dir "$BUILD_DIR" \
  --tpch-data "$TPCH_DATA_ROOT" \
  --scale 100 \
  --arms legacy \
  --stages ncu_profiles \
  --queries 6 \
  --memory-reserve-bytes 12000000000 \
  --ncu-metrics-file "$PWD/ncu_metrics.section" \
  --ncu-kernels-file "$PWD/ncu_kernels.txt" \
  --ncu-capture-timeout 7200
```

Only rank 0 is placed under NCU; peer ranks continue normally. NCU replay can
deadlock kernels with in-kernel peer dependencies even when NCCL kernels are
excluded. Start with one query and keep the capture timeout enabled.

## 11. Summarize results

The campaign directory is the source of truth. Some useful post-processing
entry points are:

```bash
# NCCL application payload sizes.
python3 tools/visualize/summarize_distributed_tpch_messages.py \
  /path/to/messages-campaign \
  --output /path/to/messages-campaign/messages-summary.csv

# Nsight Systems summaries for a profile campaign.
python3 tools/visualize/summarize_distributed_tpch_nsys.py \
  /path/to/profile-campaign \
  --output-dir /path/to/profile-campaign/nsys-summary

# Show available report/plot tools.
find tools/visualize -maxdepth 1 -name '*distributed_tpch*.py' -print | sort
```

Most plotting tools provide `--help` and either accept campaign directories
directly or expect the standard `results/distributed_tpch` campaign layout.

## 12. Common failures

- **CMake cannot find Arrow/Parquet:** activate the environment and set
  `CMAKE_PREFIX_PATH`, or pass `Arrow_DIR` and `Parquet_DIR` explicitly.
- **CMake finds the wrong MPI:** remove the build directory or use a fresh one,
  then configure with the intended `MPI_CXX_COMPILER`.
- **NCCL is not found:** set `NCCL_ROOT`/`NCCL_HOME` to a prefix containing
  `include/nccl.h` and `lib*/libnccl.so`.
- **A rank cannot load a shared library:** make the relevant library path
  visible on every host and export it in `LD_LIBRARY_PATH` before campaign
  creation. Runtime environment values are frozen in the manifest.
- **The campaign path already exists:** choose a new path for `new`; use
  `resume` or `retry` for an existing campaign.
- **Dataset preflight fails:** verify all eight tables, selected goldens,
  manifest hash, `_SUCCESS`, and file permissions from every host.
- **GPU preflight reports an active process:** wait for the GPU to become idle
  or select another device/host; do not bypass the check for benchmark runs.
- **RMM allocation fails:** decreasing `--memory-reserve-bytes` gives the RMM
  arena more memory. If NCCL or CE workspace allocation fails, increasing the
  reserve leaves more memory outside the arena. Tune per query with repeated
  `--query-memory-reserve-bytes QN=BYTES` options.
- **NCCL CTA configuration is rejected:** remove inherited CTA/channel/ring
  settings from NCCL config files or plugins. Each experiment arm owns that
  policy.
- **Nsight preflight fails:** check profiler installation, permissions, target
  GPU support, and performance-counter access before starting a full profile
  campaign.

For the complete campaign semantics and advanced options, see
`script/distributed_tpch/README.md` and `run.py new --help`.
