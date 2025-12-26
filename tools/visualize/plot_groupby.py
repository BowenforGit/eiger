import pandas as pd
from plot_util import *
import seaborn as sns
import os
import pathlib

def plot_groupby_thpt_vs_ngroups(path_to_csv, **params):
    df = pd.read_csv(path_to_csv)
    df = filter_df(
        df, 
        **params, 
        strict=True, 
        must_have_attr=["Key", "num_items", "num_aggs", "zipf_factor"])

    df['algo'] = mark_ours(df['algo'], ours=["sort", "sort_opt"])
    df['algo'] = df['algo'].str.upper()
    df['thpt'] = df['Elem/s (elem/sec)'] / 1e9

    print(df.pivot_table(index=['num_groups'], values='thpt', columns = ['algo'], aggfunc='mean'))
    
    fig, ax = create_plot((8, 6))
    ax = sns.lineplot(data=df, x='num_groups', y='thpt', hue="algo",  style = "algo", markers=True, markersize=8, ax=ax)
    ax.legend(title="Algo", loc='best')
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=20, 
        title="", 
        xlabel = {"label": "Number of groups"},
        xscale = {"scale" : "log", "base": 2},
        ylabel = {"label": "Thpt (10^9 Tuples/s)"},
        yscale = {"scale" : "log", "base": 2},
        ylim = [0, df['thpt'].max() * 1.25])

    save_fig(fig, f"ieq_groupby_thpt_vs_ngroups_{dict_to_str(params)}", subdir="groupby", suffix="pdf")

if __name__ == "__main__":
    for Key in ["I32", "I64"]:
        plot_groupby_thpt_vs_ngroups(
            os.path.join(pathlib.Path(__file__).parent.absolute(), "data", "groupby_gh200_daint.csv"),
            num_items=2**29, 
            num_aggs=3, 
            zipf_factor=0.0, 
            Key=Key)