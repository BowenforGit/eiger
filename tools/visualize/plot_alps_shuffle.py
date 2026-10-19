#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot ALPS shuffle-only (all-to-all) bandwidth microbenchmark results.

Reads the CSV produced by `alps_shuffle_bench` / `script/run_alps_shuffle.sh`
and emits one figure with three panels:
  1) primary throughput   (g*N*sizeof(T)/time) vs per-peer message size;
  2) cross-GPU bandwidth  ((g-1)*N*sizeof(T)/time) vs per-peer message size;
  3) per-GPU link bandwidth = throughput * (g-1)/g^2  (= cross_gpu/g), the rate
     each GPU pushes to the fabric; on a non-blocking fabric this converges to a
     constant across g.
All as GB/s, one line per GPU count, with std bands.

Usage:
    python plot_alps_shuffle.py [results.csv]
If no path is given, uses tools/results/data/alps/shuffle.csv.
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

from plot_util import create_plot, adjust_ax, save_fig

HERE = os.path.dirname(__file__)
DEFAULT_CSV = os.path.join(HERE, os.pardir, "results", "data", "alps", "shuffle.csv")


def save_both(fig, name, subdir="alps"):
    """Save a PDF (for the repo) and a PNG (for quick previewing)."""
    save_fig(fig, name, subdir=subdir, suffix="pdf")
    save_fig(fig, name, subdir=subdir, suffix="png", dpi=110)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    print(f"Loading {csv_path}")
    df = pd.read_csv(csv_path).sort_values(["g", "per_peer_bytes"]).reset_index(drop=True)
    # Drop the g=1 loopback series: it has no network (cross-GPU/link = 0) and
    # just distracts from the multi-GPU shuffle curves.
    df = df[df["g"] > 1].reset_index(drop=True)

    # Derived per-GPU link bandwidth = throughput * (g-1)/g^2  (= cross_gpu/g).
    factor = (df["g"] - 1) / (df["g"] ** 2)
    df["link_gbps_mean"] = df["throughput_gbps_mean"] * factor
    df["link_gbps_std"]  = df["throughput_gbps_std"] * factor

    # If the CSV has both aligned (odd_count=0) and misaligned (odd_count=1)
    # rows, overlay them: solid = aligned, dashed = misaligned.
    has_odd = "odd_count" in df.columns and df["odd_count"].nunique() > 1
    gs = sorted(df["g"].unique())
    cmap = plt.get_cmap("tab10")
    gcolor = {g: cmap(i % 10) for i, g in enumerate(gs)}

    panels = [
        ("throughput_gbps", "Throughput g*N (GB/s)"),
        ("cross_gpu_gbps",  "Cross-GPU (g-1)*N (GB/s)"),
        ("link_gbps",       "Per-GPU link bw = thpt*(g-1)/g^2 (GB/s)"),
    ]
    fig, axes = create_plot((21, 5), dim=(1, 3))
    for ax, (col, ylabel) in zip(axes, panels):
        for g in gs:
            for oc in ([0, 1] if has_odd else [0]):
                c = df[(df["g"] == g) & (df.get("odd_count", 0) == oc)]
                if c.empty:
                    continue
                style = "--" if oc == 1 else "-"
                tag = " misaligned" if oc == 1 else (" aligned" if has_odd else "")
                ax.errorbar(c["per_peer_bytes"], c[f"{col}_mean"], yerr=c[f"{col}_std"],
                            marker=("x" if oc == 1 else "o"), ls=style, color=gcolor[g],
                            capsize=3, label=f"{g} GPU(s){tag}")
        ax.legend(fontsize=9, ncol=2 if has_odd else 1)
        adjust_ax(ax, fontsize=15, legend_fontsize=9,
                  title="",
                  xlabel={"label": "Per-peer message bytes"},
                  xscale={"scale": "log", "base": 2},
                  ylabel={"label": ylabel})
        ax.grid(True, which="both", ls=":", alpha=0.4)
    save_both(fig, "alps_shuffle_alignment" if has_odd else "alps_shuffle_bandwidth")


if __name__ == "__main__":
    main()
