#!/usr/bin/env python3
# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

"""Plot the ALPS shuffle-vs-broadcast distributed-join scaling sweep.

Reads the CSV produced by `alps_dist_join_scaling_bench` /
`script/run_alps_dist_join_scaling.sh` and emits, for each scaling mode:

  1) total time vs GPU count, shuffle vs broadcast, one subplot per FK:PK ratio
     (weak: shuffle ~flat, broadcast rises as its replicated side grows with P;
      strong: shuffle scales down, broadcast hits a fixed-replication floor);
  2) broadcast speedup (shuffle_total / broadcast_total) vs GPU count, one line
     per ratio, with the y=1 crossover marked (broadcast wins above it);
  3) partition / comm / join stacked breakdown, shuffle vs broadcast, across GPU
     counts -- shows WHY (broadcast = comm+join, no partition).

Sizes are held fixed for the line/breakdown plots: weak keys on FK rows/rank,
strong on the nominal total-FK exponent (round(log2(total_fk)), since per-rank
rounding makes the realized total differ slightly across GPU counts).

Usage:
    python plot_alps_dist_join_scaling.py <dist_join_scaling.csv>
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_util import create_plot, adjust_ax, save_fig

HERE = os.path.dirname(__file__)
DEFAULT_CSV = os.path.join(HERE, os.pardir, "results", "data", "gb200", "alps",
                           "dist_join_scaling.csv")

STRAT = {
    "shuffle":   dict(color="#2c7fb8", marker="o", label="shuffle"),
    "broadcast": dict(color="#e08214", marker="s", label="broadcast"),
}
STAGE = {"partition": "#8c9eb2", "comm": "#7fbf7b", "join": "#2c7fb8"}


def save_both(fig, name, subdir="alps"):
    save_fig(fig, name, subdir=subdir, suffix="pdf")
    save_fig(fig, name, subdir=subdir, suffix="png", dpi=110)


ALGO_ORDER = ["hash", "smj", "phj"]
ALGO_STYLE = {"hash": "#2c7fb8", "smj": "#d95f0e", "phj": "#31a354"}
ALGO_HATCH = [None, "//", "xx"]  # by algo index, to distinguish grouped bars


def pkfk_label(r):
    """CSV `ratio` is FK:PK (>=1); show it as the user's PK:FK fraction."""
    return "1" if int(r) == 1 else f"1/{int(r)}"


def load(csv_path):
    df = pd.read_csv(csv_path)
    if "join_algo" not in df.columns:   # strategy-comparison CSV predates the algo column
        df["join_algo"] = "hash"
    # Size key + label per scaling mode.
    df["skey"] = np.where(df["scaling"] == "weak",
                          df["fk_per_rank"],
                          np.round(np.log2(df["total_fk"])).astype(int))
    return df


def size_label(scaling, skey):
    if scaling == "weak":
        return f"FK/rank=2^{int(round(np.log2(skey)))}"
    return f"total FK=2^{int(skey)}"


def pick_skey(sub):
    """Largest size key present for both strategies at every GPU count."""
    gpus = sorted(sub["gpus"].unique())
    ok = []
    for sk in sorted(sub["skey"].unique()):
        s = sub[sub["skey"] == sk]
        if all(set(s[s["gpus"] == g]["strategy"]) >= {"shuffle", "broadcast"} for g in gpus):
            ok.append(sk)
    return max(ok) if ok else sub["skey"].max()


def plot_scaling(df, scaling):
    sub = df[df["scaling"] == scaling]
    ratios = sorted(sub["ratio"].unique())
    skey = pick_skey(sub)
    fig, axes = create_plot((5 * len(ratios), 4.5), dim=(1, len(ratios)))
    if len(ratios) == 1:
        axes = [axes]
    for ax, r in zip(axes, ratios):
        d = sub[(sub["ratio"] == r) & (sub["skey"] == skey)]
        for strat, st in STRAT.items():
            c = d[d["strategy"] == strat].sort_values("gpus")
            ax.plot(c["gpus"], c["total_ms_mean"], marker=st["marker"],
                    color=st["color"], label=st["label"], lw=2, ms=8)
        ax.set_xticks(sorted(d["gpus"].unique()))
        ax.legend(fontsize=12)
        adjust_ax(ax, fontsize=14, legend_fontsize=12,
                  title=f"FK:PK = {r}:1   ({size_label(scaling, skey)})",
                  xlabel={"label": "GPUs (4=1 node, 8=2, 12=3)"},
                  ylabel={"label": "Join time (ms)"})
        ax.set_ylim(bottom=0)
        ax.grid(True, ls=":", alpha=0.4)
    fig.suptitle(f"{scaling} scaling: shuffle vs broadcast", fontsize=15)
    save_both(fig, f"alps_dist_join_scaling_{scaling}")


def plot_speedup(df):
    modes = [m for m in ("weak", "strong") if m in set(df["scaling"])]
    if not modes:
        return
    fig, axes = create_plot((6.5 * len(modes), 5), dim=(1, len(modes)))
    if len(modes) == 1:
        axes = [axes]
    for ax, scaling in zip(axes, modes):
        sub = df[df["scaling"] == scaling]
        skey = pick_skey(sub)
        for r in sorted(sub["ratio"].unique()):
            d = sub[(sub["ratio"] == r) & (sub["skey"] == skey)]
            piv = d.pivot_table(index="gpus", columns="strategy", values="total_ms_mean")
            piv = piv.dropna().sort_index()
            if piv.empty:
                continue
            sp = piv["shuffle"] / piv["broadcast"]
            ax.plot(piv.index, sp, marker="o", lw=2, ms=7, label=f"{r}:1")
        ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.7)
        ax.set_xticks(sorted(sub["gpus"].unique()))
        ax.legend(title="FK:PK", fontsize=11)
        adjust_ax(ax, fontsize=14, legend_fontsize=11,
                  title=f"{scaling} scaling   ({size_label(scaling, skey)})",
                  xlabel={"label": "GPUs"},
                  ylabel={"label": "broadcast speedup (shuffle / broadcast)"})
        ax.grid(True, ls=":", alpha=0.4)
        # annotate the crossover band
        ax.text(0.02, 0.97, "broadcast wins\n^ above 1 ^", transform=ax.transAxes,
                fontsize=9, va="top", color="#555")
    fig.suptitle("Broadcast speedup over shuffle vs node count", fontsize=15)
    save_both(fig, "alps_dist_join_scaling_speedup")


def plot_breakdown(df, scaling):
    sub = df[df["scaling"] == scaling]
    ratios = sorted(sub["ratio"].unique())
    skey = pick_skey(sub)
    fig, axes = create_plot((5 * len(ratios), 4.5), dim=(1, len(ratios)))
    if len(ratios) == 1:
        axes = [axes]
    w = 0.38
    for ax, r in zip(axes, ratios):
        d = sub[(sub["ratio"] == r) & (sub["skey"] == skey)]
        gpus = sorted(d["gpus"].unique())
        x = np.arange(len(gpus))
        for off, strat, hatch in [(-w / 2, "shuffle", None), (w / 2, "broadcast", "//")]:
            rows = d[d["strategy"] == strat].set_index("gpus")
            part = np.array([rows.loc[g, "partition_ms_mean"] if g in rows.index else 0 for g in gpus])
            comm = np.array([rows.loc[g, "comm_ms_mean"]      if g in rows.index else 0 for g in gpus])
            join = np.array([rows.loc[g, "join_ms_mean"]      if g in rows.index else 0 for g in gpus])
            ax.bar(x + off, part, w, color=STAGE["partition"], hatch=hatch, edgecolor="white")
            ax.bar(x + off, comm, w, bottom=part, color=STAGE["comm"], hatch=hatch, edgecolor="white")
            ax.bar(x + off, join, w, bottom=part + comm, color=STAGE["join"], hatch=hatch, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\nGPU" for g in gpus])
        # legends: stage colors + strategy hatch
        from matplotlib.patches import Patch
        stage_handles = [Patch(facecolor=c, label=s) for s, c in STAGE.items()]
        strat_handles = [Patch(facecolor="white", edgecolor="k", label="shuffle"),
                         Patch(facecolor="white", edgecolor="k", hatch="//", label="broadcast")]
        leg1 = ax.legend(handles=stage_handles, fontsize=10, loc="upper left", title="stage")
        ax.add_artist(leg1)
        ax.legend(handles=strat_handles, fontsize=10, loc="upper right", title="bar")
        adjust_ax(ax, fontsize=14,
                  title=f"FK:PK = {r}:1   ({size_label(scaling, skey)})",
                  ylabel={"label": "Time (ms)"})
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.suptitle(f"{scaling} scaling stage breakdown (left=shuffle, right=broadcast)", fontsize=15)
    save_both(fig, f"alps_dist_join_scaling_breakdown_{scaling}")


def plot_algo_breakdown(df, scaling="strong"):
    """Local-join algorithm comparison: partition/comm/join stacked bars, one bar
    per algo, one subplot per GPU count, at the largest total size. partition+comm
    are algo-independent (same shuffle); only the join stage varies."""
    sub = df[df["scaling"] == scaling]
    if "shuffle" in set(sub["strategy"]):
        sub = sub[sub["strategy"] == "shuffle"]
    skey = int(sub["skey"].max())
    d0 = sub[sub["skey"] == skey]
    gpus = sorted(d0["gpus"].unique())
    fig, axes = create_plot((4.5 * len(gpus), 4.5), dim=(1, len(gpus)))
    if len(gpus) == 1:
        axes = [axes]
    for ax, g in zip(axes, gpus):
        d = d0[d0["gpus"] == g].set_index("join_algo")
        algos = [a for a in ALGO_ORDER if a in d.index]
        x = np.arange(len(algos))
        part = np.array([d.loc[a, "partition_ms_mean"] for a in algos])
        comm = np.array([d.loc[a, "comm_ms_mean"] for a in algos])
        join = np.array([d.loc[a, "join_ms_mean"] for a in algos])
        ax.bar(x, part, color=STAGE["partition"], label="partition")
        ax.bar(x, comm, bottom=part, color=STAGE["comm"], label="comm (all-to-all)")
        ax.bar(x, join, bottom=part + comm, color=STAGE["join"], label="local join")
        ax.set_xticks(x); ax.set_xticklabels(algos)
        ax.legend(fontsize=10)
        per_rank = int(d.iloc[0]["fk_per_rank"])
        adjust_ax(ax, fontsize=14, legend_fontsize=10,
                  title=f"{g} GPUs   (FK/rank≈2^{int(round(np.log2(per_rank)))})",
                  ylabel={"label": "Time (ms)"})
        ax.set_ylim(bottom=0); ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.suptitle(f"Local-join algorithm breakdown ({scaling} scaling, total FK=2^{skey})", fontsize=15)
    save_both(fig, f"alps_dist_join_algo_breakdown_{scaling}")


def plot_algo_join_vs_size(df, scaling="strong"):
    """The local-join stage alone vs per-rank input size, one line per algorithm —
    the algo crossover decoupled from the distributed stages (join depends on the
    per-rank shard size, pooled across GPU counts)."""
    sub = df[df["scaling"] == scaling]
    if "shuffle" in set(sub["strategy"]):
        sub = sub[sub["strategy"] == "shuffle"]
    fig, ax = create_plot((7, 5))
    for a in ALGO_ORDER:
        c = sub[sub["join_algo"] == a].sort_values("fk_per_rank")
        if c.empty:
            continue
        ax.plot(c["fk_per_rank"], c["join_ms_mean"], marker="o", ms=7, lw=2,
                color=ALGO_STYLE[a], label=a)
    ax.legend(title="local join", fontsize=12)
    adjust_ax(ax, fontsize=14, legend_fontsize=12,
              title=f"Local-join time vs per-rank size ({scaling} scaling)",
              xlabel={"label": "FK rows / rank (post-shuffle shard)"},
              xscale={"scale": "log", "base": 2},
              ylabel={"label": "local join time (ms)"})
    ax.set_ylim(bottom=0); ax.grid(True, which="both", ls=":", alpha=0.4)
    save_both(fig, f"alps_dist_join_algo_join_vs_size_{scaling}")


def plot_algo_ratio_breakdown(df, scaling="strong"):
    """Local-join algorithm comparison across the PK:FK ratio sweep, one subplot
    per GPU count: grouped bars per algo (hatched), each stacked partition/comm/
    join. partition+comm are algo-independent (same shuffle); only the join stage
    differs, so the bars share a base and split at the top."""
    from matplotlib.patches import Patch
    sub = df[df["scaling"] == scaling]
    if "shuffle" in set(sub["strategy"]):
        sub = sub[sub["strategy"] == "shuffle"]
    algos = [a for a in ALGO_ORDER if a in set(sub["join_algo"])]
    gpus = sorted(sub["gpus"].unique())
    ratios = sorted(sub["ratio"].unique(), reverse=True)  # FK:PK 96..1 -> PK:FK 1/96..1
    fig, axes = create_plot((1 + 3.6 * len(gpus), 5), dim=(1, len(gpus)))
    if len(gpus) == 1:
        axes = [axes]
    w = 0.8 / max(1, len(algos))
    for ax, g in zip(axes, gpus):
        d = sub[sub["gpus"] == g]
        x = np.arange(len(ratios))
        for ai, a in enumerate(algos):
            da = d[d["join_algo"] == a].set_index("ratio")
            off = (ai - (len(algos) - 1) / 2) * w
            col = lambda m: np.array([da.loc[r, m] if r in da.index else 0.0 for r in ratios])
            part, comm, join = col("partition_ms_mean"), col("comm_ms_mean"), col("join_ms_mean")
            h = ALGO_HATCH[ai % len(ALGO_HATCH)]
            ax.bar(x + off, part, w, color=STAGE["partition"], hatch=h, edgecolor="white", linewidth=0.3)
            ax.bar(x + off, comm, w, bottom=part, color=STAGE["comm"], hatch=h, edgecolor="white", linewidth=0.3)
            ax.bar(x + off, join, w, bottom=part + comm, color=STAGE["join"], hatch=h, edgecolor="white", linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels([pkfk_label(r) for r in ratios], rotation=45, ha="right")
        stage_handles = [Patch(facecolor=c, label=s) for s, c in STAGE.items()]
        algo_handles = [Patch(facecolor="white", edgecolor="k",
                              hatch=ALGO_HATCH[ai % len(ALGO_HATCH)], label=a) for ai, a in enumerate(algos)]
        leg1 = ax.legend(handles=stage_handles, fontsize=9, loc="upper left", title="stage")
        ax.add_artist(leg1)
        ax.legend(handles=algo_handles, fontsize=9, loc="upper right", title="algo")
        adjust_ax(ax, fontsize=13, title=f"{g} GPUs", xlabel={"label": "PK:FK"},
                  ylabel={"label": "Time (ms)"})
        ax.set_ylim(bottom=0); ax.grid(True, axis="y", ls=":", alpha=0.4)
    tot = int(sub["total_fk"].max())
    fig.suptitle(f"Local-join algo breakdown vs PK:FK ({scaling}, total FK≈{tot/2**28:.0f}·2^28)", fontsize=15)
    save_both(fig, f"alps_dist_join_algo_ratio_breakdown_{scaling}")


def plot_algo_join_vs_ratio(df, scaling="strong"):
    """Local-join time alone vs PK:FK, one line per algo, one subplot per GPU count
    — the algorithm crossover as the build (PK) side shrinks."""
    sub = df[df["scaling"] == scaling]
    if "shuffle" in set(sub["strategy"]):
        sub = sub[sub["strategy"] == "shuffle"]
    gpus = sorted(sub["gpus"].unique())
    ratios = sorted(sub["ratio"].unique(), reverse=True)
    fig, axes = create_plot((3.6 * len(gpus), 4.5), dim=(1, len(gpus)))
    if len(gpus) == 1:
        axes = [axes]
    for ax, g in zip(axes, gpus):
        d = sub[sub["gpus"] == g]
        for a in [x for x in ALGO_ORDER if x in set(d["join_algo"])]:
            da = d[d["join_algo"] == a].set_index("ratio")
            y = [da.loc[r, "join_ms_mean"] if r in da.index else np.nan for r in ratios]
            ax.plot(range(len(ratios)), y, marker="o", ms=6, lw=2, color=ALGO_STYLE[a], label=a)
        ax.set_xticks(range(len(ratios))); ax.set_xticklabels([pkfk_label(r) for r in ratios], rotation=45, ha="right")
        ax.legend(title="local join", fontsize=10)
        adjust_ax(ax, fontsize=13, title=f"{g} GPUs", xlabel={"label": "PK:FK"},
                  ylabel={"label": "local join time (ms)"})
        ax.set_ylim(bottom=0); ax.grid(True, ls=":", alpha=0.4)
    fig.suptitle(f"Local-join time vs PK:FK ({scaling} scaling)", fontsize=15)
    save_both(fig, f"alps_dist_join_algo_join_vs_ratio_{scaling}")


# ---------------------------------------------------------------------------
# Strong-scaling views: GPU count on the x-axis (fixed total problem size).
# The algo-breakdown plots above facet BY GPU count; these instead show how a
# fixed-size join scales AS GPUs are added -- the heart of a strong-scaling study.
# ---------------------------------------------------------------------------

def _gpu_axis(ax, gpus):
    """Linear x-axis with every GPU count as an integer tick. Linear (not log2)
    so dense integer counts (1..8) are evenly spaced and a step change at a
    particular count -- e.g. the NVLink-island wall at 4->5 on a 2x4 node -- is
    visually obvious rather than bunched at the right edge."""
    ax.set_xticks(list(gpus))
    ax.set_xticklabels([str(int(g)) for g in gpus])
    ax.set_xlim(min(gpus) - 0.3, max(gpus) + 0.3)
    ax.minorticks_off()


def _ratio_colors(ratios):
    """One color per PK:FK ratio (dark = small PK, i.e. large FK:PK)."""
    cols = plt.cm.viridis(np.linspace(0.0, 0.92, len(ratios)))
    return {r: cols[i] for i, r in enumerate(ratios)}


def plot_strong_throughput(df, strategy="shuffle"):
    """Throughput (Gtuples/s) vs #GPUs, one subplot per local-join algo, one line
    per PK:FK ratio. The headline strong-scaling result for a fixed total FK."""
    sub = df[(df["scaling"] == "strong") & (df["strategy"] == strategy)]
    if sub.empty:
        return
    algos = [a for a in ALGO_ORDER if a in set(sub["join_algo"])]
    ratios = sorted(sub["ratio"].unique(), reverse=True)  # FK:PK 128..1 -> PK:FK 1/128..1
    gpus = sorted(sub["gpus"].unique())
    rc = _ratio_colors(ratios)
    fig, axes = create_plot((4.8 * len(algos), 4.6), dim=(1, len(algos)))
    if len(algos) == 1:
        axes = [axes]
    for ax, a in zip(axes, algos):
        d = sub[sub["join_algo"] == a]
        for r in ratios:
            c = d[d["ratio"] == r].sort_values("gpus")
            if c.empty:
                continue
            ax.plot(c["gpus"], c["throughput_gtuples_s_mean"], marker="o", ms=6, lw=2,
                    color=rc[r], label=pkfk_label(r))
        _gpu_axis(ax, gpus)
        ax.legend(title="PK:FK", fontsize=8, ncol=2)
        adjust_ax(ax, fontsize=14, legend_fontsize=8, title=a,
                  xlabel={"label": "GPUs"},
                  ylabel={"label": "throughput (Gtuples/s)"})
        ax.set_ylim(bottom=0)
        ax.grid(True, which="both", ls=":", alpha=0.4)
    tot = int(sub["total_fk"].max())
    fig.suptitle(f"Strong scaling: throughput vs #GPUs ({strategy}, total FK={tot/2**28:.0f}·2^28)",
                 fontsize=15)
    save_both(fig, f"alps_dist_join_strong_throughput_{strategy}")


def plot_strong_speedup(df, strategy="shuffle"):
    """Speedup over the 1-GPU baseline vs #GPUs, one subplot per algo, one line per
    PK:FK ratio, with the ideal-linear reference. Lines lacking a 1-GPU point
    (e.g. phj that overflowed at np=1 for a large PK) are dropped."""
    sub = df[(df["scaling"] == "strong") & (df["strategy"] == strategy)]
    if sub.empty:
        return
    algos = [a for a in ALGO_ORDER if a in set(sub["join_algo"])]
    ratios = sorted(sub["ratio"].unique(), reverse=True)
    gpus = sorted(sub["gpus"].unique())
    rc = _ratio_colors(ratios)
    fig, axes = create_plot((4.8 * len(algos), 4.6), dim=(1, len(algos)))
    if len(algos) == 1:
        axes = [axes]
    for ax, a in zip(axes, algos):
        d = sub[sub["join_algo"] == a]
        for r in ratios:
            piv = d[d["ratio"] == r].set_index("gpus")["total_ms_mean"].sort_index()
            base = piv.index.min()
            if base != 1 or piv.empty:        # need the 1-GPU anchor for a true speedup
                continue
            ax.plot(piv.index, piv.loc[1] / piv.values, marker="o", ms=6, lw=2,
                    color=rc[r], label=pkfk_label(r))
        ax.plot(gpus, gpus, "k--", lw=1.2, alpha=0.7, label="ideal")
        _gpu_axis(ax, gpus)
        ax.set_ylim(bottom=0)
        ax.legend(title="PK:FK", fontsize=8, ncol=2)
        adjust_ax(ax, fontsize=14, legend_fontsize=8, title=a,
                  xlabel={"label": "GPUs"},
                  ylabel={"label": "speedup vs 1 GPU"})
        ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.suptitle(f"Strong scaling: speedup vs #GPUs ({strategy})", fontsize=15)
    save_both(fig, f"alps_dist_join_strong_speedup_{strategy}")


def plot_bcast_speedup_vs_gpus(df):
    """Broadcast speedup over shuffle (shuffle_total / broadcast_total) vs #GPUs,
    one subplot per algo, one line per PK:FK ratio. >1 means broadcast wins. The
    bench gates broadcast to P <= ratio, so the lines truncate where broadcasting
    stops being sensible."""
    sub = df[df["scaling"] == "strong"]
    if not ({"shuffle", "broadcast"} <= set(sub["strategy"])):
        return
    algos = [a for a in ALGO_ORDER if a in set(sub["join_algo"])]
    ratios = sorted(sub["ratio"].unique(), reverse=True)
    gpus = sorted(sub["gpus"].unique())
    rc = _ratio_colors(ratios)
    fig, axes = create_plot((4.8 * len(algos), 4.6), dim=(1, len(algos)))
    if len(algos) == 1:
        axes = [axes]
    for ax, a in zip(axes, algos):
        d = sub[sub["join_algo"] == a]
        for r in ratios:
            piv = d[d["ratio"] == r].pivot_table(index="gpus", columns="strategy",
                                                 values="total_ms_mean").sort_index()
            if not {"shuffle", "broadcast"} <= set(piv.columns):
                continue
            piv = piv.dropna(subset=["shuffle", "broadcast"])
            if piv.empty:
                continue
            ax.plot(piv.index, piv["shuffle"] / piv["broadcast"], marker="o", ms=6, lw=2,
                    color=rc[r], label=pkfk_label(r))
        ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.7)
        _gpu_axis(ax, gpus)
        ax.legend(title="PK:FK", fontsize=8, ncol=2)
        adjust_ax(ax, fontsize=14, legend_fontsize=8, title=a,
                  xlabel={"label": "GPUs"},
                  ylabel={"label": "broadcast speedup (shuffle / broadcast)"})
        ax.set_ylim(bottom=0)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.text(0.02, 0.97, "broadcast wins\n^ above 1 ^", transform=ax.transAxes,
                fontsize=9, va="top", color="#555")
    fig.suptitle("Strong scaling: broadcast vs shuffle vs #GPUs", fontsize=15)
    save_both(fig, "alps_dist_join_strong_bcast_speedup")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    print(f"Loading {csv_path}")
    df = load(csv_path)

    # Strong-scaling-vs-#GPUs views (GPU count on the x-axis). Emitted whenever the
    # CSV spans multiple GPU counts in strong mode -- the focus of this study.
    if df["gpus"].nunique() > 1 and "strong" in set(df["scaling"]):
        for strat in ("shuffle", "broadcast"):
            if strat in set(df["strategy"]):
                plot_strong_throughput(df, strat)
                plot_strong_speedup(df, strat)
        plot_bcast_speedup_vs_gpus(df)

    if df["join_algo"].nunique() > 1:
        # Local-join algorithm study (e.g. --algos hash,smj,phj).
        ratio_sweep = df["ratio"].nunique() > 1
        for scaling in sorted(df["scaling"].unique()):
            if ratio_sweep:
                plot_algo_ratio_breakdown(df, scaling)
                plot_algo_join_vs_ratio(df, scaling)
            else:
                plot_algo_breakdown(df, scaling)
                plot_algo_join_vs_size(df, scaling)
    else:
        # Shuffle-vs-broadcast strategy study.
        for scaling in sorted(df["scaling"].unique()):
            plot_scaling(df, scaling)
            plot_breakdown(df, scaling)
        plot_speedup(df)


if __name__ == "__main__":
    main()
