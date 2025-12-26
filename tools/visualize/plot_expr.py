import pandas as pd
from plot_util import *
import seaborn as sns
import os
import pathlib

expr_label_map = {
    "one_col_mult" : "E1",
    "two_col_arith": "E2",
    "three_col_arith": "E3",
    "four_col_arith": "E4",
    "simple_gt": "E5",
    "two_col_and": "E6",
    "two_col_and_or": "E7"
}

def read_eiger_results(path):
    df = pd.read_csv(path)
    df = df[df['Elem/s (elem/sec)'].notna()]
    df['time (ms)'] = df['GPU Time (sec)'] * 1e3
    df['expr_label'] = df['expr_type'].map(expr_label_map)
    df['algo'] = df['impl'].map({"fuse": "PTI", "nofuse": "BB"})
    df = df.sort_values(by=['expr_label', 'algo'])
    return df

def read_cudf_results(path, **constants):
    df = pd.read_csv(path)
    for k, v in constants.items():
        df[k] = v
    df['time (ms)'] = df['avg_time_s'] * 1e3
    df['algo'] = "cuDF"
    df['expr_label'] = df['expr_type'].map(expr_label_map)
    return df

def plot_time_vs_expr(df_in, **params):
    df = df_in.copy()
    df = filter_df(
        df, 
        **params, 
        strict=True, 
        must_have_attr=["num_items", "KeyType"])
    
    fig, ax = create_plot((5,5))
    ax = sns.barplot(data=df, x='expr_label', y='time (ms)', hue="algo",  ax=ax)
    ax.legend(title="", loc='best')
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=20, 
        title="", 
        xlabel = {"label": ""},
        ylabel = {"label": "Time (ms)", "rotation": 0, "x":0.1, "y":1.02})

    save_fig(fig, f"expr_time_vs_expr_{dict_to_str(params)}", subdir="expr", suffix="pdf")

if __name__ == "__main__":
    resdir = pathlib.Path(__file__).parent.absolute() / ".."  / ".." / "exp_results" / "expr"
    ours = read_eiger_results(resdir / "eiger_expr_gpu05_20251222.csv")
    cudf = read_cudf_results(resdir / "cudf_expr_sgs-gpu05_20251222.csv", impl="cudf")
    df = pd.concat([ours, cudf], ignore_index=True)
    
    for num_items in [2**29]:
        for KeyType in ['I32', 'I64']:
            plot_time_vs_expr(
                df, 
                num_items=num_items,
                KeyType=KeyType)