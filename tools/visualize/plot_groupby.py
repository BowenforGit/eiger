import pandas as pd
from plot_util import *
import seaborn as sns
import os
import pathlib
import numpy as np

def read_our_results(path):
    df = pd.read_csv(path)
    df = df[df['Elem/s (elem/sec)'].notna()]
    df['algo'] = df['algo'].map({
        "sort" : "SORT/TR",
        "sort_opt" : "SORT-OPT",
        "sort_gfur" : "SORT/UR",
        "partition": "PARTITION/TR",
        "partition_gfur": "PARTITION/UR",
        "hash" : "Hash",
        "auto" : "Auto"
    })
    df['thpt'] = df['Elem/s (elem/sec)'] / 1e9
    df['num_groups_log2'] = np.log2(df['num_groups'])
    df['zipf_factor'] = np.round(df['zipf_factor'], 1)
    return df

def read_cudf_results(path):
    df = pd.read_csv(path)
    type_map = {'int32': 'I32', 'int64': 'I64'}
    if 'key_type' in df.columns:
        df['Key'] = df['key_type'].map(type_map)
    df['algo'] = 'cuDF'
    df['thpt'] = df['num_items'] / df[f'{which_cudf_stats}_time'] / 1e9
    df['num_groups_log2'] = np.log2(df['num_groups'])
    df['zipf_factor'] = 0.0
    return df

def plot_groupby_thpt_vs_ngroups(df_in, machine, **params):
    df = df_in.copy()
    df = filter_df(
        df, 
        **params, 
        strict=True, 
        must_have_attr=["Key", "num_items", "num_aggs", "zipf_factor"])
    
    if df.empty:
        print(f"Warning: no data for params {params}")
        return

    # print(df.pivot_table(index=['num_groups'], values='thpt', columns = ['algo'], aggfunc='mean'))
    
    fig, ax = create_plot((8, 6))
    ax = sns.lineplot(data=df, x='num_groups_log2', y='thpt', hue="algo",  style = "algo", markers=True, markersize=8, ax=ax)
    
    ax.legend(title="", loc='best')
    
    ticks = sorted(df['num_groups_log2'].unique())
    if len(ticks) > 10:
        ticks = ticks[::2]

    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$2^{{{int(tick)}}}$" for tick in ticks])
    ax.grid(True, which='both', axis='both', linestyle=':', alpha=0.75, zorder=0)
    
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=20, 
        title="", 
        xlabel = {"label": "Number of groups"},
        # xscale = {"scale" : "log", "base": 2},
        ylabel = {"label": "Thpt (Billion Tuples/s)"},
        # yscale = {"scale" : "log", "base": 2},
        ylim = [0, df['thpt'].max() * 1.25])

    save_fig(fig, f"groupby_thpt_vs_ngroups_{dict_to_str(params)}", subdir=f"groupby/{machine}", suffix="pdf")

def plot_groupby_thpt_vs_zipf(df_in, machine, **params):
    df = df_in.copy()
    df = filter_df(
        df, 
        **params, 
        strict=True, 
        must_have_attr=["Key", "num_items", "num_aggs", "num_groups"])
    
    if df.empty:
        print(f"Warning: no data for params {params}")
        return

    # print(df.pivot_table(index=['zipf_factor'], values='thpt', columns = ['algo'], aggfunc='mean'))
    
    fig, ax = create_plot((12, 5))
    ax = sns.lineplot(data=df, x='zipf_factor', y='thpt', hue="algo",  style = "algo", markers=True, markersize=8, ax=ax)
    
    ax.legend(title="", loc='best')
    
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=20, 
        title="", 
        xlabel = {"label": "Zipf Factor"},
        ylabel = {"label": "Thpt (Billion Tuples/s)"},
        ylim = [0, df['thpt'].max() * 1.25])

    save_fig(fig, f"groupby_thpt_vs_zipf_{dict_to_str(params)}", subdir=f"groupby/{machine}", suffix="pdf")

if __name__ == "__main__":
    workload = "groupby"
    machines = ["a100", "cscs"]
    resdir = pathlib.Path(__file__).parent.absolute() / ".."  / ".." / "exp_results"
    
    # vary number of groups
    for machine in machines:
        try:
            respath = latest_result_file(resdir / machine / workload / f"groupby_max_{machine}")
            ours = read_our_results(respath)

            try:
                cudf_path = latest_result_file(resdir / machine / workload / "cudf_groupby_max")
                cudf = read_cudf_results(cudf_path)
                df = pd.concat([ours, cudf], ignore_index=True)
            except (FileNotFoundError, ValueError) as e:
                print(f"Skipping cuDF for {machine}: {e}")
                df = ours

            df_base = df[df['zipf_factor'] == 0.0]
            
            for Key in ["I32", "I64"]:
                for num_items in sorted(ours['num_items'].unique()):
                    for num_aggs in sorted(ours['num_aggs'].unique()):
                        plot_groupby_thpt_vs_ngroups(
                            df_base,
                            machine,
                            Key=Key, 
                            num_items=num_items, 
                            num_aggs=num_aggs, 
                            zipf_factor=0.0)
        except Exception as e:
            print(f"Error filtering groupby vs ngroups for {machine}: {e}")

    # vary zipf factor (Ours only)      
    for machine in machines:
        try:
            respath = latest_result_file(resdir / machine / workload / f"groupby_max_{machine}")
            ours = read_our_results(respath)
            
            df = ours[ours['zipf_factor'] != 0.0]
            for Key in sorted(df['Key'].unique()):
                for num_items in sorted(df['num_items'].unique()):
                    for num_groups in [int(2**i) for i in [8, 14, 20, 26]]:
                        for num_aggs in sorted(df['num_aggs'].unique()):
                            plot_groupby_thpt_vs_zipf(
                                df,
                                machine,
                                Key=Key, 
                                num_items=num_items, 
                                num_groups=num_groups,
                                num_aggs=num_aggs)
        except Exception as e:
            print(f"Error filtering groupby vs zipf for {machine}: {e}")