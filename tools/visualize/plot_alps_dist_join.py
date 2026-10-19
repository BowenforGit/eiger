#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot ALPS distributed-join benchmark results.

Reads the CSV produced by `alps_dist_join_bench` / `script/run_alps_dist_join.sh`
and emits:
  1) throughput (10^9 tuples/s) vs total table size, one line per local-join
     config, one subplot per GPU count;
  2) a stacked partition / comm / join time breakdown at the largest size,
     one group per config, one subplot per GPU count.

Usage:
    python plot_alps_dist_join.py [results.csv]
If no path is given, uses tools/results/data/alps/dist_join.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_util import create_plot, adjust_ax, save_fig

HERE = os.path.dirname(__file__)
DEFAULT_CSV = os.path.join(HERE, os.pardir, "results", "data", "alps", "dist_join.csv")


def save_both(fig, name, subdir="alps"):
    """Save a PDF (for the repo) and a PNG (for quick previewing)."""
    save_fig(fig, name, subdir=subdir, suffix="pdf")
    save_fig(fig, name, subdir=subdir, suffix="png", dpi=110)


def load(csv_path):
    df = pd.read_csv(csv_path)
    # Drop GFTR variants for now: materialization is ~equivalent to GFUR in the
    # current data, so keep one curve per join algorithm.
    df = df[df["mat_algo"] == "gfur"].reset_index(drop=True)
    df["cfg"] = df["join_algo"] + "-" + df["mat_algo"]
    df = df.sort_values(["gpus", "cfg", "total_tuples"]).reset_index(drop=True)
    return df


def plot_throughput(df, params_str):
    gpus_list = sorted(df["gpus"].unique())
    fig, axes = create_plot((6 * len(gpus_list), 5), dim=(1, len(gpus_list)))
    if len(gpus_list) == 1:
        axes = [axes]

    for ax, g in zip(axes, gpus_list):
        sub = df[df["gpus"] == g]
        for cfg in sorted(sub["cfg"].unique()):
            c = sub[sub["cfg"] == cfg]
            ax.errorbar(c["total_tuples"], c["throughput_gtuples_s_mean"],
                        yerr=c["throughput_gtuples_s_std"], marker="o", capsize=3, label=cfg)
        ax.legend(title="join-mat", fontsize=11)
        adjust_ax(ax, fontsize=15, legend_fontsize=11,
                  title=f"{g} GPU(s)",
                  xlabel={"label": "Total tuples (pk + fk)"},
                  xscale={"scale": "log", "base": 2},
                  ylabel={"label": "Throughput (10^9 tuples/s)"})
        ax.grid(True, which="both", ls=":", alpha=0.4)
    save_both(fig, f"alps_dist_join_throughput_{params_str}")


def plot_breakdown(df, params_str):
    # Drop the 1-GPU case (no shuffle). Select the size by PER-RANK N, not
    # total_tuples: total_tuples = gpus*2N, so its global max only exists at the
    # largest gpus -> picking by it would leave every other facet empty.
    gpus_list = [g for g in sorted(df["gpus"].unique()) if g > 1]
    fig, axes = create_plot((6 * len(gpus_list), 5), dim=(1, len(gpus_list)))
    if len(gpus_list) == 1:
        axes = [axes]

    for ax, g in zip(axes, gpus_list):
        sub_g = df[df["gpus"] == g]
        # Largest per-rank N at which all configs ran (phj is skipped at the very
        # largest N for g>=2).
        cnt = sub_g.groupby("pk_rows_per_rank")["cfg"].nunique()
        full = cnt[cnt == cnt.max()].index
        N = max(full) if len(full) else sub_g["pk_rows_per_rank"].max()
        sub = sub_g[sub_g["pk_rows_per_rank"] == N].sort_values("cfg")
        cfgs = sub["cfg"].tolist()
        x = np.arange(len(cfgs))
        part = sub["partition_ms_mean"].to_numpy()
        comm = sub["comm_ms_mean"].to_numpy()
        join = sub["join_ms_mean"].to_numpy()
        ax.bar(x, part, label="partition")
        ax.bar(x, comm, bottom=part, label="comm (all-to-all)")
        ax.bar(x, join, bottom=part + comm, label="local join")
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs, rotation=30, ha="right")
        ax.legend(fontsize=11)
        adjust_ax(ax, fontsize=15, legend_fontsize=11,
                  title=f"{g} GPU(s), N=2^{int(np.log2(N))}/rank",
                  ylabel={"label": "Time (ms)"})
        ax.grid(True, axis="y", ls=":", alpha=0.4)
    save_both(fig, f"alps_dist_join_breakdown_{params_str}")


def plot_comm_vs_compute(df, params_str):
    """Per algorithm, grouped bars of partition / comm / local-join time, so the
    comm (all-to-all, identical across algorithms) sits next to each algorithm's
    local GPU compute -> shows which algorithms are comm-bound vs compute-bound.
    One subplot per GPU count, at the largest size where every config ran."""
    cfg_order = ["hash-gfur", "smj-gfur", "smj-gftr", "phj-gfur", "phj-gftr"]
    gpus_list = sorted(df["gpus"].unique())
    fig, axes = create_plot((5 * len(gpus_list), 5), dim=(1, len(gpus_list)))
    if len(gpus_list) == 1:
        axes = [axes]

    for ax, g in zip(axes, gpus_list):
        sub = df[df["gpus"] == g]
        # Largest per-rank size at which all available configs are present
        # (phj is skipped at the very largest N for >=2 GPUs).
        cnt = sub.groupby("pk_rows_per_rank")["cfg"].nunique()
        full = cnt[cnt == cnt.max()].index
        N = max(full) if len(full) else sub["pk_rows_per_rank"].max()
        d = sub[sub["pk_rows_per_rank"] == N].set_index("cfg")
        cfgs = [c for c in cfg_order if c in d.index]
        x = np.arange(len(cfgs))
        w = 0.27
        part = [d.loc[c, "partition_ms_mean"] for c in cfgs]
        comm = [d.loc[c, "comm_ms_mean"] for c in cfgs]
        join = [d.loc[c, "join_ms_mean"] for c in cfgs]
        ax.bar(x - w, part, w, label="partition (GPU)", color="#8c9eb2")
        ax.bar(x,     comm, w, label="comm (all-to-all)", color="#e08214")
        ax.bar(x + w, join, w, label="local join (GPU)", color="#2c7fb8")
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs, rotation=20, ha="right")
        ax.legend(fontsize=10)
        adjust_ax(ax, fontsize=14, legend_fontsize=10,
                  title=f"{g} GPU(s), N=2^{int(np.log2(N))}/rank",
                  ylabel={"label": "Time (ms)"})
        ax.grid(True, axis="y", ls=":", alpha=0.4)
    save_both(fig, f"alps_dist_join_comm_vs_compute_{params_str}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    print(f"Loading {csv_path}")
    df = load(csv_path)
    key_types = "_".join(sorted(df["key_type"].unique()))
    params_str = f"key_{key_types}"
    plot_throughput(df, params_str)
    plot_breakdown(df, params_str)
    plot_comm_vs_compute(df, params_str)


if __name__ == "__main__":
    main()
