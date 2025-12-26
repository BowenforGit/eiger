import pandas as pd
from plot_util import *
import seaborn as sns
import os
import pathlib

def plot_ieq_join_thpt_vs_algo(path_to_csv, **params):
    df = pd.read_csv(path_to_csv)
    df = filter_df(
        df, 
        **params, 
        strict=True, 
        must_have_attr=["Key", "num_items", "ratio", "num_payloads_each", "match_ratio", "zipf_factor"])

    df['algo'] = df['join_algo'].str.upper() + "+" + df['mat_algo'].str.upper()
    df['thpt'] = df['Elem/s (elem/sec)'] / 1e9
    df['join_algo'] = df['join_algo'].str.upper()
    df['mat_algo'] = df['mat_algo'].str.upper()
    df['mat_algo'] = df['mat_algo'].replace("GFTR", "GFTR (Ours)")
    
    fig, ax = create_plot((8, 6))
    ax = sns.barplot(data=df, x='join_algo', y='thpt', hue="mat_algo",  ax=ax)
    ax.legend(title="Mat. Algo", loc='upper left')
    for i, c in enumerate(ax.containers):
        ax.bar_label(c, label_type='edge', fmt='%.2f', fontsize=22, rotation=0)
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=22, 
        title="", 
        xlabel = {"label": "Join Algorithms"},
        ylabel = {"label": "Thpt (10^9 Tuples/s)"},
        ylim = [0, 14])

    save_fig(fig, f"ieq_join_thpt_vs_algo_{dict_to_str(params)}", subdir="ieq_join", suffix="pdf")

def plot_ieq_join_match_ratio(path_to_csv, **params):
    df = pd.read_csv(path_to_csv)
    df = filter_df(
        df, 
        **params, 
        strict=True, 
        must_have_attr=["Key", "num_items", "ratio", "num_payloads_each", "zipf_factor"])

    df['algo'] = df['join_algo'].str.upper() + "+" + df['mat_algo'].str.upper()
    df['thpt'] = df['Elem/s (elem/sec)'] / 1e9
    df.sort_values(by='algo', ascending=False, inplace=True)
    
    fig, ax = create_plot((8, 6))
    ax = sns.barplot(data=df, x='match_ratio', y='thpt', hue="algo",  ax=ax)
    ax.legend(title="", loc='upper right', ncols=2)
    for i, c in enumerate(ax.containers):
        ax.bar_label(c, label_type='edge', fmt='%.2f', fontsize=18, rotation=90)
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=22, 
        title="", 
        xlabel = {"label": "Match ratio"},
        xticklabels = {'ticks': [f"{float(i.get_text())*100} %" for i in ax.get_xticklabels()]},
        ylabel = {"label": "Thpt (10^9 Tuples/s)"},
        ylim = [0, df['thpt'].max() * 1.25])

    save_fig(fig, f"ieq_join_match_ratio_{dict_to_str(params)}", subdir="ieq_join", suffix="pdf")

def plot_ieq_join_zipf_factor(path_to_csv, **params):
    df = pd.read_csv(path_to_csv)
    df = filter_df(
        df, 
        **params, 
        strict=True, 
        must_have_attr=["Key", "num_items", "ratio", "num_payloads_each", "match_ratio"])

    df['algo'] = df['join_algo'].str.upper() + "+" + df['mat_algo'].str.upper()
    df['thpt'] = df['Elem/s (elem/sec)'] / 1e9
    df.sort_values(by='algo', ascending=False, inplace=True)
    
    fig, ax = create_plot((8, 6))
    ax = sns.barplot(data=df, x='zipf_factor', y='thpt', hue="algo",  ax=ax)
    ax.legend(title="", loc='upper right', ncols=2)
    for i, c in enumerate(ax.containers):
        ax.bar_label(c, label_type='edge', fmt='%.2f', fontsize=18, rotation=90)
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=22, 
        title="", 
        xlabel = {"label": "Zipf factor"},
        # xticklabels = {'ticks': [f"{float(i.get_text())*100} %" for i in ax.get_xticklabels()]},
        ylabel = {"label": "Thpt (10^9 Tuples/s)"},
        ylim = [6, df['thpt'].max() * 1.5])

    save_fig(fig, f"ieq_join_zipf_factor_{dict_to_str(params)}", subdir="ieq_join", suffix="pdf")

if __name__ == "__main__":
    for Key in ["I32", "I64"]:
        # plot_ieq_join_thpt_vs_algo(
        #     os.path.join(pathlib.Path(__file__).parent.absolute(), "data", "ieq_join_h100_sgs-gpu06.ethz.ch.csv"),
        #     num_items=2**28, 
        #     ratio=2, 
        #     num_payloads_each=3, 
        #     match_ratio=1, 
        #     zipf_factor=0.0, 
        #     Key=Key)
        pass
    
    # plot_ieq_join_match_ratio(
    #     os.path.join(pathlib.Path(__file__).parent.absolute(), "data", "ieq_join_h100_sgs-gpu06.ethz.ch.csv"),
    #     num_items=2**28, 
    #     ratio=2, 
    #     num_payloads_each=2, 
    #     zipf_factor=0.0, 
    #     Key="I32")

    plot_ieq_join_zipf_factor(
        os.path.join(pathlib.Path(__file__).parent.absolute(), "data", "ieq_join_h100_sgs-gpu06.ethz.ch.csv"),
        num_items=2**28, 
        ratio=2, 
        num_payloads_each=2, 
        match_ratio=1, 
        Key="I32")