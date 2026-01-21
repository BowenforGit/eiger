import pandas as pd
from plot_util import *
import seaborn as sns
import os
import pathlib

def read_eiger_results(path):
    df = pd.read_csv(path)
    df = df[df['Elem/s (elem/sec)'].notna()]
    df['time (ms)'] = df['GPU Time (sec)'] * 1000
    df['thpt'] = df['Elem/s (elem/sec)'] / 1e9
    df['algo'] = "sort"
    df.loc[df['use_prefix_sort'] == 0, 'algo'] = "Naive sort"
    df.loc[df['use_prefix_sort'] == 1, 'algo'] = "Opt. sort"

    return df

def plot_string_sort_time_vs_strlen_barplot(df_in, **params):
    df = df_in.copy()
    df = filter_df(
        df, 
        **params, 
        must_have_attr=["op", "num_strings"])
    
    if df.empty:
        print(f"Warning: no data for params {params}")
        return
    
    df['string_length'] = df['string_length'].astype(str)
    
    # Create plot
    fig, ax = create_plot((6, 5))
    ax = sns.barplot(data=df, x='string_length', y='time (ms)', hue="algo",  ax=ax)
    ax.legend(title="", loc='best')

    adjust_ax(
        ax,
        fontsize=24, 
        legend_fontsize=18,
        title="", 
        xlabel = {"label": "String Length (bytes)"},
        ylabel = {"label": "Time (ms)", "rotation": 0, "x":0.02, "y":1.02})
    ax.grid(axis='y', alpha=0.75, linestyle=':', zorder=0)
    ax.set_axisbelow(True)
    
    save_fig(fig, f"string_sort_time_vs_strlen_barplot_{dict_to_str(params)}", subdir=f"string/{params.get('machine', '')}", suffix="pdf")

if __name__ == "__main__":
    workload = "string"
    machines = ["cscs"]
    resdir = pathlib.Path(__file__).parent.absolute() / ".."  / ".." / "exp_results"

    for machine in machines:
        df = read_eiger_results(
            latest_result_file(resdir / machine / workload / f"string_sort_{machine}")
        )
        
        for num_strings in sorted(df['num_strings'].unique()):
            plot_string_sort_time_vs_strlen_barplot(
                df,
                machine=machine,
                op="argsort",
                num_strings=num_strings)