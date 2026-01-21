import pandas as pd
from plot_util import *
import seaborn as sns
import os
import pathlib
import numpy as np

def preprocess(df):
    df = df[df['Elem/s (elem/sec)'].notna()]
    df['algo'] = df['join_algo'].str.upper() + "+" + df['mat_algo'].str.upper()
    df['thpt'] = df['Elem/s (elem/sec)'] / 1e9
    df['left_num_items_log2'] = np.log2(df['left_num_items'])
    if 'ratio' in df.columns:
        df['right_num_items'] = df['left_num_items'] * df['ratio']
    df['right_num_items_log2'] = np.log2(df['right_num_items'])
    df['match_ratio'] = np.round(df['match_ratio'], 1)
    df['zipf_factor'] = np.round(df['zipf_factor'], 1)
    return df

def plot_vary_table_size(path_to_csv, **params):
    df = preprocess(pd.read_csv(path_to_csv))
    if (df['zipf_factor'] != 0.0).any():
        print("Warning: zipf_factor contains values other than 0.0, but it shouldn't."
              " Filtering to only include zipf_factor == 0.0")
        df = df[df['zipf_factor'] == 0.0]
    
    if (df['match_ratio'] != 1.0).any():
        print("Warning: match_ratio contains values other than 1.0, but it shouldn't."
              " Filtering to only include match_ratio == 1.0")
        df = df[df['match_ratio'] == 1.0]
    
    df = filter_df(
        df, 
        **params, 
        strict=False, 
        must_have_attr=["Key", "left_ncols", "right_ncols"])
    
    if df.empty:
        print(f"Warning: no data for params {params}")
        return
    
    fig, ax = create_plot((12, 5))

    if "left_num_items" in params:
        side = "right"
        table = "S"
    elif "right_num_items" in params:
        side = "left"
        table = "R"
    else:
        raise ValueError("Either left_num_items or right_num_items must be specified in params")    
    x_col = f"{side}_num_items_log2"
    
    
    ax = sns.lineplot(data=df, x=x_col, y='thpt', hue="algo", style="algo", markers=True, markersize=8, ax=ax)
    ax.legend(title="", loc='best', ncols=2)

    ax.set_xticks(sorted(df[x_col].unique()[::2]))
    ax.set_xticklabels([f"$2^{{{int(tick)}}}$" for tick in sorted(df[x_col].unique()[::2])])
    
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=22, 
        title="", 
        xlabel = {"label": f"Size of Table {table} (Number of Tuples)"},
        ylabel = {"label": "Thpt (Billion Tuples/s)"})

    save_fig(fig, f"ieq_join_vary_{side}_table_size_{dict_to_str(params)}", subdir=f"join/{params.get('machine', '')}", suffix="pdf")

def plot_vary_match_ratio(path_to_csv, **params):
    df = preprocess(pd.read_csv(path_to_csv))
    df = filter_df(
        df, 
        **params, 
        strict=False, 
        must_have_attr=["Key", "left_num_items", "left_ncols", "right_ncols"])
    
    
    if (df['zipf_factor'] != 0.0).any():
        print("Warning: zipf_factor contains values other than 0.0, but it shouldn't."
              " Filtering to only include zipf_factor == 0.0")
        df = df[df['zipf_factor'] == 0.0]

    if df.empty:
        print(f"Warning: no data for params {params}")
        return
    
    fig, ax = create_plot((8, 5))
    ax = sns.lineplot(data=df, x='match_ratio', y='thpt', hue="algo", style="algo", markers=True, markersize=8, ax=ax)
    ax.legend(title="", loc='best', ncols=2)

    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=22, 
        title="", 
        xlabel = {"label": "Match ratio"},
        xticklabels = {'ticks': [f"{float(i.get_text())*100} %" for i in ax.get_xticklabels()]},
        ylabel = {"label": "Thpt (10^9 Tuples/s)"},
        ylim = [0, df['thpt'].max() * 1.25])

    save_fig(fig, f"ieq_join_match_ratio_{dict_to_str(params)}", subdir=f"join/{params.get('machine', '')}", suffix="pdf")

def plot_vary_zipf_factor(path_to_csv, **params):
    df = preprocess(pd.read_csv(path_to_csv))
    df = filter_df(
        df, 
        **params, 
        strict=False, 
        must_have_attr=["Key", "left_num_items", "left_ncols", "right_ncols"])
    
    
    if (df['match_ratio'] != 0.0).any():
        print("Warning: match_ratio contains values other than 1.0, but it shouldn't."
              " Filtering to only include match_ratio == 1.0")
        df = df[df['match_ratio'] == 1.0]

    if df.empty:
        print(f"Warning: no data for params {params}")
        return
    
    fig, ax = create_plot((8, 5))
    ax = sns.lineplot(data=df, x='zipf_factor', y='thpt', hue="algo", style="algo", markers=True, markersize=8, ax=ax)
    ax.legend(title="", loc='best', ncols=2)

    ax.set_xticks(sorted(df['zipf_factor'].unique()))
    # ax.set_xticklabels([str(tick) for tick in sorted(df['zipf_factor'].unique())])

    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=22, 
        title="", 
        xlabel = {"label": "Zipf factor"},
        ylabel = {"label": "Thpt (10^9 Tuples/s)"},
        ylim = [0, df['thpt'].max() * 1.25])

    save_fig(fig, f"ieq_join_zipf_factor_{dict_to_str(params)}", subdir=f"join/{params.get('machine', '')}", suffix="pdf")

if __name__ == "__main__":
    machines = ["cscs"]
    workload = "join"
    resdir = pathlib.Path(__file__).parent.absolute() / ".."  / ".." / "exp_results"
    
    # vary left table size
    for machine in machines:
        try:
            respath = latest_result_file(resdir / machine / workload / f"vary_left_table_size_{machine}")
        except FileNotFoundError:
            print(f"Skipping missing results for {machine}")
            continue
        
        for Key in ["I32", "I64"]:
            for left_ncols in [2, 4]:
                for right_ncols in [2, 4]:
                    plot_vary_table_size(
                        respath,
                        machine = machine,
                        Key=Key,
                        right_num_items = int(2**28),
                        left_ncols=left_ncols,
                        right_ncols=right_ncols)

    # vary right table size
    for machine in machines:
        try:
            respath = latest_result_file(resdir / machine / workload / f"vary_right_table_size_{machine}")
        except FileNotFoundError:
            print(f"Skipping missing results for {machine}")
            continue
        
        for Key in ["I32", "I64"]:
            for left_num_items in [int(2**i) for i in range(12, 26, 4)]:
                for left_ncols in [2, 4]:
                    for right_ncols in [2, 4]:
                        plot_vary_table_size(
                            respath,
                            machine = machine,
                            Key=Key,
                            left_num_items = left_num_items,
                            left_ncols=left_ncols,
                            right_ncols=right_ncols)
                    
    # # vary match_ratio
    for machine in machines:
        try:
            respath = latest_result_file(resdir / machine / workload / f"vary_match_ratio_{machine}")
        except FileNotFoundError:
            print(f"Skipping missing results for {machine}")
            continue
        
        for Key in ["I32", "I64"]:
            for left_num_items in [int(2**i) for i in range(12, 29, 4)]:
                for left_ncols in [4]:
                    for right_ncols in [4]:
                        try:
                            plot_vary_match_ratio(
                                respath,
                                machine = machine,
                                Key=Key,
                                left_num_items = left_num_items,
                                left_ncols=left_ncols,
                                right_ncols=right_ncols)
                        except Exception as e:
                            print(f"Error plotting for params: machine={machine}, Key={Key}, left_num_items={left_num_items}, left_ncols={left_ncols}, right_ncols={right_ncols}")
                            print(e)
                            continue

    # vary zipf_factor
    for machine in machines:
        try:
            respath = latest_result_file(resdir / machine / workload / f"vary_zipfian_factor_{machine}")
        except FileNotFoundError:
            print(f"Skipping missing results for {machine}")
            continue
        
        for Key in ["I32", "I64"]:
            for left_num_items in [int(2**i) for i in range(12, 29, 4)]:
                for left_ncols in [4]:
                    for right_ncols in [4]:
                        try:
                            plot_vary_zipf_factor(
                                respath,
                                machine = machine,
                                Key=Key,
                                left_num_items = left_num_items,
                                left_ncols=left_ncols,
                                right_ncols=right_ncols)
                        except Exception as e:
                            print(f"Error plotting for params: machine={machine}, Key={Key}, left_num_items={left_num_items}, left_ncols={left_ncols}, right_ncols={right_ncols}")
                            print(e)
                            continue
