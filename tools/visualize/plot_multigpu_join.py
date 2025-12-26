import pandas as pd
from plot_util import *
import seaborn as sns
import os
import pathlib
import json

def reduce_json(dict):
    for time in ["run_times_ms", "compute_times_ms", "transfer_times_ms"]:
        dict["max_"+time] = max(dict[time])
        del dict[time]

    del dict["stage_compute_times_ms"], dict["stage_transfer_times_ms"]
    return dict

def get_multigpu_results(path_to_json, **params):
    df = json_to_df(path_to_json, reduce_json)
    df = filter_df(df, join_alltoall_impl="portable", join_allgather_impl="portable", **params)
    df = df[['operation', 'variant', 'benchmark_type','num_gpus', 'gamax_data_type',
       'total_rows', 'num_unique_keys', 'num_value_cols', 'total_rows_b',
       'num_unique_keys_b', 'num_value_cols_b', 'match_ratio', 'zipf_factor',
       'randomize_key_location', 'key_distribution', 'input_table_bytes',
       'input_table_gb', 'join_alltoall_impl',
       'join_allgather_impl', 'buf_alignment_opt', 
       'max_run_times_ms', 'max_compute_times_ms', 'max_transfer_times_ms']]
    return df

def combine_multigpu_results(paths_to_json, **params):
    data = []
    for p, d in paths_to_json.items():
        df = get_multigpu_results(p, **d)
        data.append(df)
    
    df = filter_df(pd.concat(data), **params)
    return df

def plot_thpt_vs_table_size(df, **params):
    df = filter_df(df, benchmark_type="table_size", **params).copy()
    df['total_rows'] += df['total_rows_b']
    df['thpt'] = df['total_rows'] / (df['max_run_times_ms'] / 1000) / 1e9 # B rows/s

    fig, ax = create_plot((8, 6))
    sns.lineplot(data=df, x="total_rows", y="thpt", hue="num_gpus", style="num_gpus", palette="Set2", markers=True, markersize=8, ax=ax)
    ax.legend(title="# GPUs", loc="upper left", ncols=2)
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=22, 
        title="", 
        xlabel = {"label": "Total rows"},
        xscale = {"scale": "log", "base": 2},
        # xticklabels = {'ticks': [f"{float(i.get_text())*100} %" for i in ax.get_xticklabels()]},
        ylabel = {"label": "Thpt (10^9 Tuples/s)"},
        yscale = {"scale": "log", "base": 2},
        ylim = [0, df['thpt'].max() * 1.25])
    
    save_fig(fig, f"multigpu_join_thpt_vs_table_size_{dict_to_str(params)}", subdir="multigpu_join", suffix="pdf")

def plot_time_breakdown(df, **params):
    df = filter_df(df, benchmark_type="table_size", **params).copy()
    df.sort_values("num_gpus", inplace=True)
    # df['total_rows'] += df['total_rows_b']
    # df['thpt'] = df['total_rows'] / (df['max_run_times_ms'] / 1000) / 1e9 # B rows/s

    fig, ax = create_plot((8, 6))
    # df = df.groupby("num_gpus")[["max_compute_times_ms", "max_transfer_times_ms"]].mean().reset_index()
    # df = pd.melt(df, id_vars=["num_gpus"], value_vars=["max_compute_times_ms", "max_transfer_times_ms"], 
    #              var_name="time_type", value_name="time_ms")
    # print(df)

    df.rename(columns={"max_compute_times_ms": "Local Compute", "max_transfer_times_ms": "Data Transfer"}, inplace=True)

    # sns.barplot(data=df, x="num_gpus", y="time_ms", hue="time_type", palette="Set2", ax=ax)
    df.plot(kind='bar', x='num_gpus', y=['Local Compute', 'Data Transfer'], stacked=True, ax=ax)
    ax.legend(title="", loc="upper right", ncols=1)
    # sns.lineplot(data=df, x="total_rows", y="thpt", hue="num_gpus", style="num_gpus", palette="Set2", markers=True, markersize=8, ax=ax)
    # ax.legend(title="# GPUs", loc="upper left", ncols=2)
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=22, 
        title="", 
        xlabel = {"label": "# GPUs"},
        # xscale = {"scale": "log", "base": 2},
        xticklabels = {'rotation': 0},
        ylabel = {"label": "Time (ms)"})
        # yscale = {"scale": "log", "base": 2}
    
    save_fig(fig, f"time_breakdown{dict_to_str(params)}", subdir="multigpu_join", suffix="pdf")

def plot_low_match_ratio(df, **params):
    df = filter_df(df, benchmark_type="low_match_opt", **params).copy()
    df.sort_values("num_gpus", inplace=True)
    df['total_rows'] += df['total_rows_b']
    df['thpt'] = df['total_rows'] / (df['max_run_times_ms'] / 1000) / 1e9 # B rows/s
    print(df)

    fig, ax = create_plot((8, 6))

    # df.rename(columns={"max_compute_times_ms": "Local Compute", "max_transfer_times_ms": "Data Transfer"}, inplace=True)

    # sns.barplot(data=df, x="num_gpus", y="time_ms", hue="time_type", palette="Set2", ax=ax)
    # df.plot(kind='bar', x='num_gpus', y=['Local Compute', 'Data Transfer'], stacked=True, ax=ax)
    sns.lineplot(data=df, x="match_ratio", y="thpt", hue="variant", style="variant", palette="Set2", markers=True, markersize=8, ax=ax)
    ax.legend(title="", loc="upper right", ncols=1)
    # sns.lineplot(data=df, x="total_rows", y="thpt", hue="num_gpus", style="num_gpus", palette="Set2", markers=True, markersize=8, ax=ax)
    ax.legend(title="", loc="upper right", ncols=1)
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=22, 
        title="", 
        xlabel = {"label": "Match ratio"},
        # xscale = {"scale": "log", "base": 2},
        # xticklabels = {'ticks': [f"{float(i.get_text())*100} %" for i in ax.get_xticklabels()]},
        ylabel = {"label": "Thpt (10^9 Tuples/s)"},
        yscale = {"scale": "log", "base": 2},
        ylim = [0, df['thpt'].max() * 1.25])
    
    save_fig(fig, f"low_match_ratio{dict_to_str(params)}", subdir="multigpu_join", suffix="pdf")

if __name__ == "__main__":
    df = combine_multigpu_results({
        "data/measurements/2025-03-26/results_join_2025-03-26_05-19-38.json": {},
        "data/measurements/2025-03-25/results_join_2025-03-25_19-17-28.json": {}, # 32
        "data/measurements/2025-03-25/results_join_2025-03-25_12-52-18.json": {},
        "data/measurements/2025-03-24/results_join_2025-03-24_19-11-05.json": {},
        "data/measurements/2025-03-24/results_join_2025-03-24_20-30-28.json": {},
        "data/measurements/2025-03-24/results_join_2025-03-24_20-33-07.json": {},
        "data/measurements/2025-03-25/results_join_2025-03-25_12-39-23.json": {},
    })

    # print(df)
    # plot_thpt_vs_table_size(df)
    # plot_time_breakdown(df, total_rows=2**31)
    plot_low_match_ratio(df, num_gpus=4, num_value_cols=4, num_value_cols_b=4)