import pandas as pd
from plot_util import *
import seaborn as sns
import os
import pathlib

machine = "gpu06"

def read_eiger_results(path):
    df = pd.read_csv(path)
    df = df[df['Elem/s (elem/sec)'].notna()]
    df['impl'] = "Ours"
    df['thpt'] = df['Elem/s (elem/sec)'] / 1e9
    df['algo'] = df['impl'].astype(str)
    df['cg_aw'] = df['cg_size'].astype(str) + "," + df['access_width'].astype(str)
    # df.loc[df['op'] != 'contains', 'algo'] = df.loc[df['op'] != 'contains', 'algo'].astype(str) + " (CG = " + df.loc[df['op'] != 'contains', 'cg_size'].astype(str) + ")"
    
    df['algo'] = df['algo'].astype(str) + " [" + df['cg_aw'].astype(str) + "]"
    df = df[(df['op'] != 'contains') | (df['cg_size'] == 1)]

    if 'use_kmp' in df.columns:
        df.loc[(df['op'] == 'contains') & (df['use_kmp'] == 1), 'algo'] = (
            df.loc[(df['op'] == 'contains') & (df['use_kmp'] == 1), 'algo'].astype(str) + 
            " (KMP)"
        )
    print(df)
    return df

def read_cudf_results(path, **constants):
    df = pd.read_csv(path)
    for k, v in constants.items():
        df[k] = v
    df['cg_aw'] = df['cg_size'].astype(str) + "," + df['access_width'].astype(str)
    df['thpt'] = df['thpt'] / 1e9
    return df
    
def plot_thpt_vs_strlen(df_in, **params):
    df = df_in.copy()
    df = filter_df(
        df, 
        **params, 
        strict=True, 
        must_have_attr=["op", "num_strings", "selectivity"])
    
    if df.empty:
        print(f"Warning: no data for params {params}")
        return
    
     # Create plot
    fig, ax = create_plot((6, 5))
    ax = sns.lineplot(data=df, x='fixed_length', y='thpt', hue="algo",  style = "algo", markers=True, markersize=8, ax=ax)
    ax.legend(title="", loc='best')
    # Set x-axis ticks to the unique values from df['fixed_length']
    ax.set_xticks(sorted(df['fixed_length'].unique()))
    ax.set_xticklabels(sorted(df['fixed_length'].unique()))
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=20, 
        title="", 
        xlabel = {"label": "String Length (bytes)", "padding": 5},
        ylabel = {"label": "Thpt ($10^9$ strings/s)", "rotation": 0, "x":0.25, "y":1.02},
        ylim = [0, df['thpt'].max() * 1.25])
    ax.grid(axis='y', alpha=0.75, linestyle=':', zorder=0)
    ax.set_axisbelow(True)
    save_fig(fig, f"string_thpt_vs_strlen_{dict_to_str(params)}", subdir=f"string/{machine}", suffix="pdf")

def plot_thpt_vs_strlen_arrow_best(df_in, **params):
    df = df_in.copy()
    df = filter_df(
        df, 
        **params, 
        strict=True, 
        must_have_attr=["op", "num_strings", "selectivity"])
    
    if df.empty:
        print(f"Warning: no data for params {params}")
        return

    df = df.groupby(["op", "num_strings", "selectivity",'fixed_length', "pattern_length", 'impl'], as_index=False).apply(
        lambda x: x.loc[x['thpt'].idxmax()], include_groups=False
    )
    
    # Create plot
    fig, ax = create_plot((6, 5))
    ax = sns.lineplot(data=df, x='fixed_length', y='thpt', hue="impl", style = "impl", markers=True, markersize=8, ax=ax)
    ax.legend(title="", loc='best')
    ax.set_xticks(sorted(df['fixed_length'].unique()))
    ax.set_xticklabels(sorted(df['fixed_length'].unique()))
    # Find the cg_aw value that maximizes thpt for each fixed_length in arrow implementation
    arrow_df = df[df['impl'] == 'Ours']
    max_thpt_per_length = arrow_df.loc[arrow_df.groupby('fixed_length')['thpt'].idxmax()]

    # Annotate data points with cg_aw values, alternating above/below
    for idx, (_, row) in enumerate(max_thpt_per_length.iterrows()):
        va = 'bottom' if idx % 2 == 0 else 'top'
        ax.annotate('[' + row['cg_aw'] + ']', 
                    xy=(row['fixed_length'], row['thpt']),
                    xytext=(0, 10 if va == 'bottom' else -10),
                    textcoords='offset points',
                    ha='center',
                    va=va,
                    fontsize=12)
    adjust_ax(
        ax, 
        fontsize=24, 
        legend_fontsize=20, 
        title="", 
        xlabel = {"label": "String Length (bytes)", "padding": 5},
        ylabel = {"label": "Thpt ($10^9$ strings/s)", "rotation": 0, "x":0.25, "y":1.02},
        ylim = [0, df['thpt'].max() * 1.25])
    ax.grid(axis='y', alpha=0.75, linestyle=':', zorder=0)
    ax.set_axisbelow(True)
    save_fig(fig, f"string_thpt_vs_strlen_{dict_to_str(params)}", subdir=f"string/{machine}", suffix="pdf")

def plot_equal_cg_size_vs_access_width_heatmap(df_in, **params):
    df = df_in.copy()
    df = filter_df(
        df, 
        **params, 
        strict=True, 
        must_have_attr=["op", "num_strings", "selectivity", "pattern_length"])
    
    if df.empty:
        print(f"Warning: no data for params {params}")
        return
    
    # Normalize throughput relative to cg_size == 1 and access_width == 1
    baseline = df[(df['cg_size'] == 1) & (df['access_width'] == 1)]['thpt'].values
    if len(baseline) > 0:
        df['thpt'] = df['thpt'] / baseline[0]
    
    # Pivot the DataFrame to create a matrix for heatmap
    heatmap_data = df.pivot(index="cg_size", columns="access_width", values="thpt")
    
    # Create plot
    fig, ax = create_plot((6, 6))
    # sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, 
                annot_kws={"size": 22}, cbar_kws={"shrink": 0.8})
    # Increase colorbar tick label size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)
    adjust_ax(
        ax, 
        fontsize=24, 
        title="", 
        xlabel = {"label": "Access Width (bytes)", "padding": 8},
        ylabel = {"label": "CG Size", "rotation": 0, "x":0.1, "y":1.02},
        yticklabels = {"rotation":0})

    save_fig(fig, f"string_equal_cg_size_vs_access_width_heatmap_{dict_to_str(params)}", subdir=f"string/{machine}", suffix="pdf")

if __name__ == "__main__":
    resdir = pathlib.Path(__file__).parent.absolute() / ".."  / ".." / "exp_results" / "string"
    ours_partial = read_eiger_results(resdir / f"eiger_stringops_partial_match_{machine}_20251225.csv")
    ours_full = read_eiger_results(resdir / f"eiger_stringops_full_match_{machine}_20251225.csv")
    ours_full['pattern_length'] = ours_full['fixed_length']
    cudf = read_cudf_results(resdir / f"cudf_stringops_sgs-{machine}_20251225.csv", algo="cuDF", impl="cuDF", cg_size=1, selectivity=1, access_width=1)
    df = pd.concat([ours_partial, ours_full, cudf], ignore_index=True)
    
    for op in ["startswith", "endswith"]:
        for num_strings in [2**28]:
            for selectivity in [1]:
                for pattern_length in df['pattern_length'].unique():
                    plot_thpt_vs_strlen_arrow_best(
                        df,
                        op=op,
                        impl=["Ours","cuDF"],
                        num_strings=num_strings,
                        selectivity=selectivity,
                        pattern_length=pattern_length)
    
    for op in ["contains"]:
        for num_strings in [2**28]:
            for selectivity in [1]:
                for pattern_length in df['pattern_length'].unique():
                    plot_thpt_vs_strlen(
                        df,
                        op=op,
                        impl=["Ours","cuDF"],
                        num_strings=num_strings,
                        selectivity=selectivity,
                        pattern_length=pattern_length,
                        cg_aw = ["1,1"])
    
    for num_strings in [2**28]:
        for selectivity in [1]:
            plot_thpt_vs_strlen_arrow_best(
                df,
                op="equal",
                impl=["Ours","cuDF"],
                num_strings=num_strings,
                selectivity=selectivity)
    
    for pattern_length in df['pattern_length'].unique():
        plot_equal_cg_size_vs_access_width_heatmap(
            df,
            op="equal",
            impl=["Ours"],
            num_strings=2**28,
            selectivity=1,
            pattern_length=pattern_length)