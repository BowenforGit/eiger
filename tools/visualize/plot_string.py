import pandas as pd
from plot_util import *
import seaborn as sns
import os
import pathlib

def read_eiger_results(path):
    df = pd.read_csv(path)
    df = df[df['Elem/s (elem/sec)'].notna()]
    df['thpt'] = df['Elem/s (elem/sec)'] / 1e9
    df['algo'] = df['impl'].astype(str)
    df.loc[df['op'] != 'contains', 'algo'] = df.loc[df['op'] != 'contains', 'algo'].astype(str) + " (CG = " + df.loc[df['op'] != 'contains', 'cg_size'].astype(str) + ")"

    df = df[(df['op'] != 'contains') | (df['cg_size'] == 1)]
    df.loc[df['op'] == 'contains', 'algo'] = (
        df.loc[df['op'] == 'contains', 'algo'].astype(str) + 
        " (KMP = " + df.loc[df['op'] == 'contains', 'use_kmp'].astype(str) + ")"
    )
    return df

def read_cudf_results(path, **constants):
    df = pd.read_csv(path)
    for k, v in constants.items():
        df[k] = v
    df['thpt'] = df['thpt'] / 1e9
    return df
    
def plot_thpt_vs_strlen(df_in, **params):
    df = df_in.copy()
    df = filter_df(
        df, 
        **params, 
        strict=True, 
        must_have_attr=["op", "num_strings", "selectivity", "pattern_length"])
    
    fig, ax = create_plot((5, 5))
    ax = sns.lineplot(data=df, x='fixed_length', y='thpt', hue="algo",  style = "algo", markers=True, markersize=8, ax=ax)
    ax.legend(title="", loc='best')
    # Set x-axis ticks to the unique values from df['fixed_length']
    ax.set_xticks(sorted(df['fixed_length'].unique()))
    ax.set_xticklabels(sorted(df['fixed_length'].unique()))
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=16, 
        title="", 
        xlabel = {"label": "String Length (bytes)"},
        ylabel = {"label": "Thpt ($10^9$ strings/s)", "rotation": 0, "x":0.25, "y":1.02},
        ylim = [0, df['thpt'].max() * 1.25])

    save_fig(fig, f"string_thpt_vs_strlen_{dict_to_str(params)}", subdir="string", suffix="pdf")

if __name__ == "__main__":
    resdir = pathlib.Path(__file__).parent.absolute() / ".."  / ".." / "exp_results" / "string"
    ours = read_eiger_results(resdir / "eiger_stringops_gpu05_20251219.csv")
    cudf = read_cudf_results(resdir / "cudf_stringops_gpu05_20251220.csv", algo="cudf", cg_size=1, selectivity=1)
    df = pd.concat([ours, cudf], ignore_index=True)
    
    for op in ["startswith", "endswith", "contains"]:
        for num_strings in [2**28]:
            for selectivity in [1]:
                for pattern_length in df['pattern_length'].unique():
                    plot_thpt_vs_strlen(
                        df,
                        op=op,
                        impl=["arrow","cudf"],
                        num_strings=num_strings,
                        selectivity=selectivity,
                        pattern_length=pattern_length,
                        cg_size = [1,2])