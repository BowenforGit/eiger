<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# Multi-file TPC-H dataset and reader design

Status: implementation contract
Date: 2026-07-24 UTC

## Scope

This design covers generation, publication, discovery, validation, and GPU
assignment for manifest-controlled multi-file TPC-H Parquet datasets. It
deliberately retains eager materialization of each query's projected local
shard. Bounded-memory streaming and spilling are separate execution-engine
work. The published SF1000 dataset is the reference deployment.

SF1000 query execution targets 36 or 72 GPUs on an NVL72 system. Four-GPU
testing is used for smaller-scale correctness validation, not as a supported
SF1000 execution configuration.

## Published layout

The permanent dataset path is `/storage/tpch/sf1000`. Generation must use a
unique sibling staging directory and publish only after complete validation.
An existing published directory must never be overwritten.

```text
sf1000/
  manifest.json
  _SUCCESS
  region.parquet
  nation.parquet
  supplier/supplier.1.parquet ... supplier.144.parquet
  customer/customer.1.parquet ... customer.144.parquet
  part/part.1.parquet ... part.144.parquet
  partsupp/partsupp.1.parquet ... partsupp.144.parquet
  orders/orders.1.parquet ... orders.144.parquet
  lineitem/lineitem.1.parquet ... lineitem.144.parquet
```

`region` and `nation` are single-file replicated tables. Every other base
table has exactly 144 parts. Part order is the numeric order recorded by the
manifest; consumers must never derive order from lexical path sorting.

Generation settings:

```text
tpchgen-cli parquet
  --scale-factor 1000
  --parts 144
  --compression SNAPPY
  --row-group-bytes 67108864
```

The generation driver invokes exactly one `--part` at a time. It must not run
concurrent generator processes. This intentionally limits pressure on the
shared NFS server and gives restart granularity of one output file.

Build the metadata finalizer and start or resume generation with:

```bash
cmake --build build --target tpch_parquet_metadata

python3 script/generate_tpch_multifile.py \
  --scale 1000 \
  --parts 144 \
  --staging-dir /storage/tpch/sf1000.staging.<unique-id> \
  --threads 16

python3 script/finalize_tpch_manifest.py \
  --staging-dir /storage/tpch/sf1000.staging.<unique-id> \
  --publish-dir /storage/tpch/sf1000
```

The driver holds an exclusive staging lock, refuses untracked files, and
atomically records each completed output. Re-running the identical command
validates and skips completed files. `--stop-after N` provides a controlled
rehearsal or pause point. Finalization remains a separate explicit step and
refuses an existing publish path.

## Manifest and completion contract

`manifest.json` is versioned and records:

- scale factor, generator identity/version, compression, and row-group target;
- each table's `single` or `partitioned` layout and replicated/sharded policy;
- canonical Arrow schema and schema fingerprint;
- an ordered part list containing numeric ID, relative path, rows, bytes,
  row-group count, and SHA-256;
- table row and byte totals.

The manifest is written atomically after all files pass validation. `_SUCCESS`
is written atomically last and contains:

```text
manifest_sha256=<lowercase SHA-256 of manifest.json>
```

A manifest-controlled reader rejects a missing or malformed completion marker,
a manifest hash mismatch, undeclared/missing parts, duplicate or non-contiguous
part IDs, unsafe paths, mismatched recorded sizes, or schema mismatch.

Existing single-file SF1-SF300 datasets remain supported without a manifest.
Unmanifested multi-file directories are rejected by default; an explicit
legacy opt-in exists only for migration and focused compatibility tests.

## GPU file assignment

Assignment is a runtime reader option and is not baked into the dataset.

### Round-robin (default)

For one-based part ID `p` and GPU count `G`:

```text
rank(p) = (p - 1) mod G
```

### Contiguous range

For zero-based rank `r`, part count `P`, and GPU count `G`:

```text
start = floor(r * P / G)
end   = floor((r + 1) * P / G)
```

The rank reads zero-based part indices `[start, end)`.

With 144 parts, each table contributes four parts per GPU at 36 GPUs and two
parts per GPU at 72 GPUs. Both strategies remain planner distribution
`LocalShard`; contiguous file placement must not be advertised as
`RangePartitioned` without proven key boundaries and key metadata.

## Reader execution

Before query loading, rank 0 performs a dataset preflight over all manifest
entries and broadcasts success or failure. Each rank then:

1. validates the manifest and completion marker;
2. calculates its assigned part IDs;
3. opens only assigned files (all files for replicated tables);
4. validates opened schemas against the canonical schema;
5. reads only requested columns;
6. reads assigned files sequentially;
7. uses bounded Arrow intra-file decoding threads;
8. concatenates the projected local parts and copies complete local columns to
   pinned host memory and GPU memory.

The initial decoder-thread cap is eight per rank and must be benchmarked at
two, four, and eight threads. Parallel file reads are disabled initially.

The single-file compatibility path retains row-group sharding for sharded
tables. File-partitioned tables are assigned at file granularity and do not
also use row-group sharding.

## Published SF1000 result

SF1000 was generated and published on 2026-07-25 UTC at
`/storage/tpch/sf1000`. Generation ran from approximately 03:27:33 through
04:16:57 UTC (about 49 minutes 24 seconds), and validation plus atomic
publication completed at 04:17:17 UTC.

The published dataset contains exactly 866 Parquet files and
371,215,331,409 bytes (346 GiB as reported by `du`). Its manifest SHA-256 is:

```text
7401129c28d371774dac4799a26c3d992b28945600a80561bed62dc34acfe3c1
```

Measured table totals:

| Table | Files | Rows | Bytes | Max/min part-byte ratio |
| --- | ---: | ---: | ---: | ---: |
| region | 1 | 5 | 1,227 | 1.000000 |
| nation | 1 | 25 | 2,670 | 1.000000 |
| supplier | 144 | 10,000,000 | 872,976,921 | 1.004591 |
| customer | 144 | 150,000,000 | 12,740,432,130 | 1.001185 |
| part | 144 | 200,000,000 | 6,383,620,947 | 1.000549 |
| partsupp | 144 | 800,000,000 | 43,870,854,935 | 1.001259 |
| orders | 144 | 1,500,000,000 | 68,698,365,104 | 1.000184 |
| lineitem | 144 | 5,999,989,709 | 238,649,077,475 | 1.000766 |

Publication validation proved that all 866 generator checkpoints correspond
to non-empty files; every Parquet footer and schema is readable; part row and
byte sums match table totals; the completion marker matches the manifest;
and the published Parquet path set exactly equals the manifest path set. The
strict reader resolver successfully resolved all tables. For both
`round_robin` and `contiguous_range`, a 36-rank coverage audit assigned every
sharded file exactly once and assigned four files from each sharded table to
each rank. The native reader-policy test suite passed all 10 tests.

## Correctness and rollout gates

1. Prove one-part-at-a-time deterministic generation at SF1.
2. Exercise missing-part restart without overwriting completed parts.
3. Validate manifest rejection and both assignment strategies with focused
   tests, including non-divisible part/rank counts.
4. Run Q1-Q22 at SF1 with both strategies on four GPUs.
5. Generate a staged multi-file SF100 and compare Q1-Q22 with existing SF100
   goldens.
6. Inspect SF100 part skew, generation throughput, decode-thread sensitivity,
   host/pinned/GPU memory, and NFS traffic.
7. Only then start SF1000, one partition at a time.
8. Validate SF1000 metadata and row totals before publication.
9. Run Q1-Q22 at 36 GPUs and representative scaling checks at 72 GPUs.

The golden generator must resolve the same manifest-declared ordered paths,
must not use broad globs, and must write each golden through a temporary path
before atomic publication.
