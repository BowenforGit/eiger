import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_util import *
import pathlib
import numpy as np


def plot_throughput_heatmap(csv_path, machine, **params):
    df = pd.read_csv(csv_path)
    df = df[df['Skipped'] != 'Yes']
    df = df[df['GPU Time (sec)'].notna()]
    df = filter_df(df, **params, strict=True)

    df['throughput_GBps'] = df['Reads (bytes)'] / df['GPU Time (sec)'] / 1e9

    pivot = df.pivot_table(index='vt', columns='grid_dim', values='throughput_GBps')
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot[sorted(pivot.columns)]

    max_val = pivot.max().max()
    pivot_norm = pivot / max_val

    fig, ax = create_plot((12, 6))
    sns.heatmap(
        pivot_norm,
        annot=pivot_norm.applymap(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"),
        fmt='',
        cmap='YlOrRd',
        vmin=0, vmax=1,
        linewidths=0.5,
        ax=ax,
        mask=pivot_norm.isna(),
        cbar_kws={'label': 'Normalized Throughput'}
    )

    # overlay N/A text on missing cells
    for i, vt in enumerate(pivot_norm.index):
        for j, gd in enumerate(pivot_norm.columns):
            if pd.isna(pivot_norm.loc[vt, gd]):
                ax.text(j + 0.5, i + 0.5, 'N/A', ha='center', va='center',
                        fontsize=12, color='gray')

    adjust_ax(
        ax,
        fontsize=16,
        title=f"Normalized Throughput (block_dim={params.get('block_dim')}, "
              f"num_items={params.get('num_items (pow2)')}, "
              f"ValueType={params.get('ValueType')})",
        xlabel={"label": "grid_dim", "padding": 5},
        ylabel={"label": "vt"},
    )

    plt.tight_layout()
    save_fig(fig, f"l2bw_throughput_{dict_to_str(params)}", subdir=f"archim/{machine}", suffix="pdf")


def plot_num_waves_heatmap(csv_path, machine, **params):
    df = pd.read_csv(csv_path)
    df = df[df['Skipped'] != 'Yes']
    df = df[df['GPU Time (sec)'].notna()]
    df = filter_df(df, **params, strict=True)

    pivot = df.pivot_table(index='vt', columns='grid_dim', values='num_waves')
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = create_plot((12, 6))
    sns.heatmap(
        pivot,
        annot=pivot.applymap(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"),
        fmt='',
        cmap='YlGnBu',
        linewidths=0.5,
        ax=ax,
        mask=pivot.isna(),
        cbar_kws={'label': 'num_waves'}
    )

    for i, vt in enumerate(pivot.index):
        for j, gd in enumerate(pivot.columns):
            if pd.isna(pivot.loc[vt, gd]):
                ax.text(j + 0.5, i + 0.5, 'N/A', ha='center', va='center',
                        fontsize=12, color='gray')

    adjust_ax(
        ax,
        fontsize=16,
        title=f"num_waves (block_dim={params.get('block_dim')}, "
              f"num_items={params.get('num_items (pow2)')}, "
              f"ValueType={params.get('ValueType')})",
        xlabel={"label": "grid_dim", "padding": 5},
        ylabel={"label": "vt"},
    )

    plt.tight_layout()
    save_fig(fig, f"l2bw_num_waves_{dict_to_str(params)}", subdir=f"archim/{machine}", suffix="pdf")


if __name__ == "__main__":
    machines = ['rtx3090']
    resdir = pathlib.Path(__file__).parent.absolute() / ".." / ".." / "exp_results"

    for machine in machines:
        csv_path = latest_result_file(resdir / machine / "archim" / "l2_req" / f"l2_req_{machine}")
        for block_dim in [256]:
            for num_items_pow2 in ['2^20']:
                for value_type in ['I32']:
                    plot_throughput_heatmap(
                        csv_path, machine,
                        block_dim=block_dim,
                        **{'num_items (pow2)': num_items_pow2},
                        ValueType=value_type,
                    )
                    plot_num_waves_heatmap(
                        csv_path, machine,
                        block_dim=block_dim,
                        **{'num_items (pow2)': num_items_pow2},
                        ValueType=value_type,
                    )
