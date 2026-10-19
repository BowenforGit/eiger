#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Generate canonical TPC-H golden answers by running DuckDB's tpch
extension against the existing Parquet tables at $TPCH_DATA_PATH/sfN.

For each of the 22 queries, we:
  * Register the existing sfN/*.parquet files as DuckDB views — no data
    regeneration, no extra copy on disk.
  * Run the spec-compliant SQL served by `tpch_queries()`.
  * Coerce the DuckDB result types so they match what gamax produces:
      - DATE → INT32 packed as YYYYMMDD (gamax's tpch_reader converts
        Arrow DATE32 to int32 YYYYMMDD at load time).
      - DECIMAL(p,s) → DOUBLE (gamax flattens all decimals to float64).
  * Write the coerced result to sfN/golden/qNN.parquet.

Usage:
    python script/gen_tpch_golden.py --sf 1 --out /scratch/wubo
    python script/gen_tpch_golden.py --sf 10 --out /scratch/wubo --queries 1,6,16

Requires `duckdb` (conda base env has it) and the TPC-H extension (DuckDB
auto-downloads on first INSTALL). Read-only against the input Parquet.
"""
import argparse
import os
import sys
import time

import duckdb

from tpch_dataset_layout import resolve_all_tables


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def register_views(
    con: duckdb.DuckDBPyConnection,
    sf_dir: str,
    sf: int,
    *,
    allow_legacy_multifile: bool = False,
) -> None:
    layouts = resolve_all_tables(
        sf_dir, sf, allow_legacy_multifile=allow_legacy_multifile)
    for table, paths in layouts.items():
        path_list = ", ".join(_sql_string(path) for path in paths)
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS "
            f"SELECT * FROM read_parquet([{path_list}])")


def get_query_text(con: duckdb.DuckDBPyConnection, qid: int) -> str:
    row = con.execute(
        "SELECT query FROM tpch_queries() WHERE query_nr = ?",
        [qid]).fetchone()
    if not row:
        raise RuntimeError(f"tpch_queries() has no query #{qid}")
    return row[0]


def coerced_projection(
    con: duckdb.DuckDBPyConnection, temp_table: str,
) -> str:
    """Build a `SELECT ... FROM <temp_table>` that coerces DATE→int32
    YYYYMMDD and DECIMAL→DOUBLE, passing everything else through.
    """
    cols = con.execute(f"DESCRIBE {temp_table}").fetchall()
    exprs = []
    for col_name, col_type, *_ in cols:
        # Output column names sometimes contain parentheses (e.g. Q18's
        # `sum(l_quantity)` — DuckDB doesn't auto-alias the anonymous
        # aggregate). Quote every identifier defensively.
        q = f'"{col_name}"'
        upper = col_type.upper()
        if upper == "DATE":
            exprs.append(
                f"CAST(EXTRACT(YEAR FROM {q}) * 10000 + "
                f"EXTRACT(MONTH FROM {q}) * 100 + "
                f"EXTRACT(DAY FROM {q}) AS INTEGER) "
                f"AS {q}")
        elif upper.startswith("DECIMAL"):
            exprs.append(f"CAST({q} AS DOUBLE) AS {q}")
        else:
            exprs.append(f"{q} AS {q}")
    return f"SELECT {', '.join(exprs)} FROM {temp_table}"


def run_one_query(
    con: duckdb.DuckDBPyConnection,
    qid: int,
    golden_dir: str,
) -> tuple[int, int]:
    query_sql = get_query_text(con, qid).rstrip().rstrip(";")
    tmp = f"q{qid:02d}_raw"
    con.execute(f"CREATE OR REPLACE TEMP TABLE {tmp} AS {query_sql}")
    (n_rows,) = con.execute(f"SELECT COUNT(*) FROM {tmp}").fetchone()
    n_cols = len(con.execute(f"DESCRIBE {tmp}").fetchall())

    out_path = os.path.join(golden_dir, f"q{qid:02d}.parquet")
    temp_path = f"{out_path}.tmp.{os.getpid()}"
    if os.path.exists(temp_path):
        os.unlink(temp_path)
    proj = coerced_projection(con, tmp)
    con.execute(
        f"COPY ({proj}) TO {_sql_string(temp_path)} (FORMAT PARQUET)")
    os.replace(temp_path, out_path)
    con.execute(f"DROP TABLE {tmp}")
    return n_rows, n_cols


def parse_queries(arg: str | None) -> list[int]:
    if not arg:
        return list(range(1, 23))
    out: list[int] = []
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo, hi = piece.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(piece))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sf", type=int, required=True,
                    help="scale factor (looks for <out>/sf<SF>/*.parquet)")
    ap.add_argument("--out", required=True,
                    help="TPC-H data root (contains sfN/*.parquet)")
    ap.add_argument("--queries", default=None,
                    help="comma-separated query ids or ranges "
                         "(default: 1-22)")
    ap.add_argument(
        "--allow-legacy-multifile", action="store_true",
        help="allow directory tables without a manifest (migration only)")
    args = ap.parse_args()

    sf_dir = os.path.join(args.out, f"sf{args.sf}")
    if not os.path.isdir(sf_dir):
        print(f"error: {sf_dir} does not exist", file=sys.stderr)
        return 1
    golden_dir = os.path.join(sf_dir, "golden")
    os.makedirs(golden_dir, exist_ok=True)

    qids = parse_queries(args.queries)
    print(f"[gen_tpch_golden] SF={args.sf} out={sf_dir} goldens→{golden_dir}")
    print(f"[gen_tpch_golden] queries: {qids}")

    con = duckdb.connect()
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    register_views(
        con, sf_dir, args.sf,
        allow_legacy_multifile=args.allow_legacy_multifile)

    failures: list[tuple[int, str]] = []
    for q in qids:
        t0 = time.time()
        try:
            n_rows, n_cols = run_one_query(con, q, golden_dir)
            dt = time.time() - t0
            print(f"  q{q:02d}: {n_rows} rows × {n_cols} cols "
                  f"({dt*1000:.0f} ms)")
        except Exception as e:
            failures.append((q, str(e)))
            print(f"  q{q:02d}: FAILED — {e}")

    if failures:
        print(f"[gen_tpch_golden] {len(failures)} FAILED: "
              f"{[q for q, _ in failures]}", file=sys.stderr)
        return 2
    print(f"[gen_tpch_golden] OK — wrote {len(qids)} goldens to "
          f"{golden_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
