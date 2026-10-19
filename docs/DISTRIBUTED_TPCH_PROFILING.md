<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# Distributed TPC-H profiling

This document defines how to collect and interpret Nsight Systems hardware and
overlap profiles for distributed TPC-H. The final sections retain the dated
four-GPU GB200 collection as a reproducible historical result.

For fresh-machine setup and the current campaign runner, start with
[`DISTRIBUTED_TPCH_BENCHMARK_GUIDE.md`](DISTRIBUTED_TPCH_BENCHMARK_GUIDE.md).

## Current workflow

Create an isolated campaign with the `profiles` stage as described in
`docs/DISTRIBUTED_TPCH_BENCHMARK_GUIDE.md`. The campaign runner freezes the
binary, environment, topology, dataset metadata, profiler settings, and command
before capture. Keep hardware profiles separate from timing campaigns so
instrumentation cannot contaminate the clean samples.

```bash
python3 script/distributed_tpch/run.py new /shared/results/tpch-nsys \
  --build-dir "$BUILD_DIR" \
  --tpch-data "$TPCH_DATA_ROOT" \
  --scale 100 \
  --arms legacy \
  --stages profiles \
  --queries 1-22 \
  --memory-reserve-bytes 12000000000
```

Use `tools/visualize/summarize_distributed_tpch_nsys.py` on the completed
campaign and retain the reports, exported summaries, campaign manifest, and
status records together. Metric availability and names depend on GPU and
Nsight Systems versions; validate the report structure before launching a full
suite.

## Historical collection method (2026-07-21)

Each query runs in a fresh four-rank MPI invocation. NSYS wraps the local MPI
process tree, rather than one MPI rank, and writes one `qNN.nsys-rep` containing
all four rank traces and all four GPU metric streams. This distinction is
required: selecting all GPU metric devices from an NSYS instance attached only
to rank 0 did not produce valid detailed SM and DRAM counters for the other
rank processes.

The benchmark loads projected source columns and performs adaptive warmup
before entering the `tpch.profile` NVTX range. The captured run is therefore
one post-load query execution. Verification against
`/storage/tpch/sf*/golden/qNN.parquet` runs after capture and must pass before a
report is accepted.

Collection uses the NSYS `gb10x` general GPU metric set selected automatically
for GB200 and samples at 10 kHz:

```bash
TPCH_DATA_PATH=/storage/tpch \
  ./script/collect_distributed_tpch_nsys.sh \
    --scales 100,300 --queries 1-22 --ranks 4 \
    --gpu-metrics-frequency 10000 \
    --output-dir \
      /storage/eiger/exp_results/distributed_tpch_nsys_metrics_20260721
```

The collector is resumable: a nonempty existing `qNN.nsys-rep` is skipped.
`system_metadata.txt` records the host, driver, GPU inventory, NSYS version,
topology, and per-link NVLink rates. `status.csv` records every attempted query
and its report and log paths.

## Utilization Definitions

The SQLite summarizer restricts samples to the distributed query envelope from
the earliest rank's `tpch.profile` start through the latest rank's completion.
This removes profiler startup samples while retaining real rank-start delay,
collective wait, and straggler time. Start and end skew are preserved as
quality fields. A strict intersection would incorrectly discard waiting time
and inflate utilization when one rank enters later than its peers.

- **SM utilization** is NSYS `SMs Active [Throughput %]`.
- **DRAM utilization** is the per-sample sum of NSYS DRAM read and write
  throughput percentages. The summary also preserves read and write
  separately.
- **NVLink RX utilization** is the per-sample sum of request/response protocol
  and user-data RX throughput percentages. TX is computed analogously.
- **NVLink full-duplex utilization** is `(RX + TX) / 2`. RX and TX each use the
  one-direction endpoint bandwidth as their denominator, so division by two
  expresses utilization against combined bidirectional capacity.
- **PCIe utilization** is preserved by direction and as the sum of RX and TX.

All original NSYS metric streams, including GR active, SM issue, tensor active,
clocks, and individual NVLink traffic classes, remain in the report and in the
long-form CSV. Derived totals are not clipped at 100%; this avoids hiding
counter or normalization behavior.

On this machine each GB200 has 18 active links at 53.125 GB/s, or 956.25 GB/s
per direction. The summarizer can therefore convert endpoint NVLink percentages
to GB/s. `TARGET_INFO_GPU.memoryBandwidth` supplies the corresponding DRAM peak
(8,064 GB/s per GPU).

```bash
python3 tools/visualize/summarize_distributed_tpch_nsys.py \
  /storage/eiger/exp_results/distributed_tpch_nsys_metrics_20260721 \
  --nvlink-unidirectional-gbps 956.25
```

Outputs under `summary/` are:

- `hardware_metrics_long.csv`: mean, median, p95, and maximum of every raw and
  derived metric for each query/GPU.
- `hardware_utilization_per_gpu.csv`: a wide table of the principal compute,
  memory, NVLink, and PCIe metrics for each query/GPU.
- `hardware_utilization_per_query.csv`: the same principal metrics pooled over
  all four GPU sample streams, plus aggregate endpoint bandwidth estimates.

The `.nsys-rep` files retain the complete sampled time series and can be opened
in Nsight Systems for interval-level inspection.

## Known Limitations

- Hardware metrics are sampled and quantized by NSYS. They are not exact byte
  counters and a short query may contain only a few hundred samples.
- The query-level percentiles pool GPU endpoint samples. Use the per-GPU table
  when rank imbalance matters.
- Summed endpoint NVLink GB/s counts transmit and receive endpoints separately;
  it is not unique payload bytes crossing the fabric.
- NSYS NIC metric collection is unavailable in the current container because
  `libibmad.so.5` is missing. This does not affect GPU-resident NVLink metrics.
- The one-report process-tree mode is local-node only. Multi-node collection
  requires one report per node or a post-capture merge strategy.

## Historical collected artifacts

Collection completed on 2026-07-21 under:

`/storage/eiger/exp_results/distributed_tpch_nsys_metrics_20260721`

- 22 SF100 and 22 SF300 `.nsys-rep` files are present, one per query.
- All 44 profiled executions passed golden-result verification.
- Export validation found four completed rank ranges and four GPU metric
  streams in every report.
- The summary contains 44 query rows, 176 query/GPU rows, and 4,576 long-form
  query/GPU/metric rows covering 21 raw plus 5 derived metrics.
- Per-GPU sample counts range from 87 to 6,309 at 10 kHz.
- `sha256sums.txt` contains one checksum for each of the 44 reports.

The highest query-level mean values at each scale are:

| Scale | Highest SM active | Highest DRAM total | Highest NVLink full-duplex |
|---:|---:|---:|---:|
| 100 | Q6, 95.67% | Q21, 21.46% | Q9, 7.76% |
| 300 | Q6, 98.58% | Q21, 22.69% | Q9, 8.07% |

SF300 Q19 is the only material rank-entry outlier: one rank entered the profile
range 27.18 ms before the other ranks, while the maximum end skew was 0.63 ms
over the full suite. Its summary deliberately includes that initial waiting
time in the distributed query envelope. All other start skews were below 1 ms.

The complete collection occupies approximately 71 MiB, including reports,
logs, summaries, metadata, and checksums.

## Column-Overlap Supplemental Trace (2026-07-22)

The column-overlapped equality-join milestone added a focused kernel-interval
analyzer, `script/analyze_column_overlap_nsys.py`. It identifies the named
communication, preparation, and match/materialization CUDA streams in an
exported NSYS SQLite database, builds the kernel-busy interval union for each
stream, and measures the intersection between communication and
match/materialization. This is distinct from sampled query-level NVLink
utilization: it directly answers whether useful kernels overlap in time.

The final trace uses the accepted high-priority communication stream and
`Interleave` schedule on the full SF300-derived Q5 join shape:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  --output=/storage/eiger/exp_results/alps_column_overlap_20260722/\
q5_sf300shape_interleave_high \
  /usr/local/mpi/bin/mpiexec -n 4 \
  build/src/cuda/alps/alps_column_overlap_join_bench \
  --shape q5 --variant interleave --divisor 1 \
  --comm-priority high --warmups 1 --iterations 1 --stage-timing

nsys export --type sqlite --force-overwrite true \
  --output /storage/eiger/exp_results/alps_column_overlap_20260722/\
q5_sf300shape_interleave_high.sqlite \
  /storage/eiger/exp_results/alps_column_overlap_20260722/\
q5_sf300shape_interleave_high.nsys-rep

python3 script/analyze_column_overlap_nsys.py \
  /storage/eiger/exp_results/alps_column_overlap_20260722/\
q5_sf300shape_interleave_high.sqlite
```

Per-rank NCCL/match-materialization overlap was 51.515, 50.994, 51.764, and
51.072 ms. The mean was 51.336 ms, or 74.5% of NCCL kernel-busy time. The
report and SQLite database are retained at the paths above.

The Arrow `STRING` payload extension added a production Q10 SF300 capture at
`/tmp/column_overlap_string_q10_sf300.{nsys-rep,sqlite}`. Q10 executes two
shuffle joins per measured run. In the string-bearing customer join, mean
communication/match-materialization kernel overlap was 2.260 ms across four
ranks, or 20.0% of communication-stream busy time. The fixed-width shuffle join
overlapped about 9.19 ms, or 60.5%. The lower string percentage reflects its
additional length exchange, offset reconstruction, host count coordination,
and character exchange; it still demonstrates real compute/communication
concurrency rather than merely reordered host submission.

Production phase CSVs complement this trace. Their `start_ms` and `end_ms`
fields share a per-step CUDA-event origin across streams, and elapsed
attribution uses the union of child intervals. Individual overlapping phase
durations remain independently visible but must not be summed as if they were
serial elapsed time.
