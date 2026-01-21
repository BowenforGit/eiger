import pandas as pd
from plot_util import *
import seaborn as sns
import os
import pathlib
import numpy as np

def preprocess(df):
    df = df[df['Elem/s (elem/sec)'].notna()]
    df['algo'] = df['algo'].str.upper()
    df['thpt'] = df['Elem/s (elem/sec)'] / 1e9
    df['num_groups_log2'] = np.log2(df['num_groups'])
    df['zipf_factor'] = np.round(df['zipf_factor'], 1)
    return df

def plot_groupby_thpt_vs_ngroups(path_to_csv, machine, **params):
    df = preprocess(pd.read_csv(path_to_csv))
    df = filter_df(
        df, 
        **params, 
        strict=True, 
        must_have_attr=["Key", "num_items", "num_aggs", "zipf_factor"])
    
    if df.empty:
        print(f"Warning: no data for params {params}")
        return

    print(df.pivot_table(index=['num_groups'], values='thpt', columns = ['algo'], aggfunc='mean'))
    
    fig, ax = create_plot((12, 5))
    ax = sns.lineplot(data=df, x='num_groups_log2', y='thpt', hue="algo",  style = "algo", markers=True, markersize=8, ax=ax)
    
    ax.legend(title="", loc='best')
    ax.set_xticks(sorted(df['num_groups_log2'].unique()[::2]))
    ax.set_xticklabels([f"$2^{{{int(tick)}}}$" for tick in sorted(df['num_groups_log2'].unique()[::2])])
    
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

def plot_groupby_thpt_vs_zipf(path_to_csv, machine, **params):
    df = preprocess(pd.read_csv(path_to_csv))
    df = filter_df(
        df, 
        **params, 
        strict=True, 
        must_have_attr=["Key", "num_items", "num_aggs", "num_groups"])
    
    if df.empty:
        print(f"Warning: no data for params {params}")
        return

    print(df.pivot_table(index=['zipf_factor'], values='thpt', columns = ['algo'], aggfunc='mean'))
    
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
    machines = ["cscs"]
    resdir = pathlib.Path(__file__).parent.absolute() / ".."  / ".." / "exp_results"
    
    # vary number of groups
    for machine in machines:
        respath = latest_result_file(resdir / machine / workload / f"groupby_max_{machine}")
        df = pd.read_csv(respath)
        df = df[df['zipf_factor'] == 0.0]
        for Key in ["I32", "I64"]:
            for num_items in sorted(df['num_items'].unique()):
                for num_aggs in sorted(df['num_aggs'].unique()):
                    plot_groupby_thpt_vs_ngroups(
                        respath,
                        machine,
                        Key=Key, 
                        num_items=num_items, 
                        num_aggs=num_aggs, 
                        zipf_factor=0.0)

    # vary zipf factor                 
    for machine in machines:
        respath = latest_result_file(resdir / machine / workload / f"groupby_max_{machine}")
        df = preprocess(pd.read_csv(respath))
        df = df[df['zipf_factor'] != 0.0]
        for Key in sorted(df['Key'].unique()):
            for num_items in sorted(df['num_items'].unique()):
                for num_groups in [int(2**i) for i in [8, 14, 20, 26]]:
                    for num_aggs in sorted(df['num_aggs'].unique()):
                        plot_groupby_thpt_vs_zipf(
                            respath,
                            machine,
                            Key=Key, 
                            num_items=num_items, 
                            num_groups=num_groups,
                            num_aggs=num_aggs)