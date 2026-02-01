import pandas as pd
from plot_util import *
import seaborn as sns
import os
import pathlib

def preprocess(df):
    df = df[df['GPU Time (sec)'].notna()]
    df['time (ms)'] = df['GPU Time (sec)'] * 1e3
    df['workload'] = df['num_key_cols'].astype(str) + "x" + df['KeyType'].astype(str)
    df['algo'] = "w/ SKF"
    df.loc[df['smart_keys_fusion'] == 0, 'algo'] = "w/o SKF"
    return df

def plot_time_vs_workload(df, machine, **params):
    df = preprocess(df)
    df = filter_df(df, **params, must_have_attr=["num_rows"])

    if df.empty:
        print(f"No data for params: {params}")
        return
    
    df = df.sort_values(by=['workload'])
    pivoted_df = df.pivot_table(
        index=['workload'],
        columns='algo', 
        values='time (ms)', 
        aggfunc='mean').reset_index()
    pivoted_df['w/o SKF vs w/ SKF'] = pivoted_df['w/o SKF'] / pivoted_df['w/ SKF']
    print(pivoted_df)

    fig, ax = create_plot((6, 5))
    
    ax = sns.barplot(
        data=df, 
        x='workload', 
        y='time (ms)', 
        hue="algo", 
        ax=ax
    )
    
    ax.legend(title="", loc='upper left')
    
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=22, 
        title="", 
        xlabel = {"label": "#Keys and key type"},
        ylabel = {"label": "Time (ms)", "rotation": 0, "x":0.1, "y":1.02},
        xticklabels = {"rotation": 0}
    )
    ax.grid(True, which='both', axis='y', linestyle=':', alpha=0.75, zorder=0)
    
    save_fig(
        fig, 
        f"sort_fused_time_vs_workload_{dict_to_str(params)}", 
        subdir=f"sort/{machine}", 
        suffix="pdf"
    )

if __name__ == "__main__":
    workload = "sort"
    machines = ['a100', 'cscs'] # Add others if needed: 'a100', 'h100' based on directory structure
    resdir = pathlib.Path(__file__).parent.absolute() / ".."  / ".." / "exp_results"

    for machine in machines:
        try:
            ours_path = latest_result_file(resdir / machine / workload / f"sort_table_fused_key_{machine}")
            ours = pd.read_csv(ours_path)
            plot_time_vs_workload(ours, machine, num_rows=ours['num_rows'].max())
        except FileNotFoundError as e:
            print(f"Skipping machine {machine} due to missing file: {e}")
            