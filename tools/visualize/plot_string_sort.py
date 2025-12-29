import pandas as pd
from plot_util import *
import seaborn as sns
import os
import pathlib

machine = "gpu06"

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
        strict=True, 
        must_have_attr=["op", "num_strings"])
    
    if df.empty:
        print(f"Warning: no data for params {params}")
        return
    
    print(df)
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
    
    save_fig(fig, f"string_sort_time_vs_strlen_barplot_{dict_to_str(params)}", subdir=f"string/{machine}", suffix="pdf")

if __name__ == "__main__":
    resdir = pathlib.Path(__file__).parent.absolute() / ".."  / ".." / "exp_results" / "string"
    df = read_eiger_results(
        resdir / f"eiger_string_sort_{machine}_20251229.csv"
    )
    
    for num_strings in [2**26]:
        plot_string_sort_time_vs_strlen_barplot(
            df,
            op="argsort",
            num_strings=num_strings)